"""Load generator: N simultaneous users on the mixed workload, against a running server.

    python serving/loadgen.py --url http://127.0.0.1:8080 --policy plain \
        --kind-dir data/first_hour --users 1 4 8 16 --out serving/results/part7/<run>

It does not start a server: point it at llama-server or at the metrics gateway in front of it.
One call measures one policy (the server's drafting setting, named by --policy); run it once
per policy, each against a server started for that policy.

Users are closed-loop, as in bench/load_check.py: each user sends its next request as soon as
the previous one returns, so "N users" means N requests in flight at all times. Requests are
sent the same way too (template rendered by the server, then a raw completion; greedy; prompt
cache off), so rows from here can be set beside the load check of 7 October.

Workload, one of:
  --workload FILE     a JSON-lines file in its frozen order (rows: id, kind, messages, and
                      optionally tools and thinking)
  --kind-dir DIR      one <kind>.jsonl per kind; --count rows per kind from row --start, then
                      one shuffle with --seed. The defaults give the day-one five-kind mix of the
                      7 October load check (200 requests), in the same order.

Per run: throughput (tokens written by all users per second of wall time, prompt reading
included) and request latency (send to full reply) at p50, p95 and p99. A percentile is the
nearest-rank value, so with 200 requests p99 is the third-slowest request.

With --quiet-check the machine is checked before every user count (serving/quiet.py) and the
readings are stored in the results file; a run does not start on a busy machine.

Results: <out>/<policy>_u<users>.json, one file per user count; finished files are skipped.
Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import queue
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

KINDS = ["code", "tool_calls", "chat", "chinese", "reasoning"]
KIND_HEADER = "X-SpecEdge-Kind"


def kind_dir_workload(folder: Path, kinds: list[str], start: int, count: int, seed: int) -> list[dict]:
    """--count rows of every kind, then one seeded shuffle. Same construction as bench/load_check.py."""
    requests = []
    for kind in kinds:
        lines = (folder / f"{kind}.jsonl").read_text(encoding="utf-8").splitlines()
        chosen = lines[start:start + count]
        if len(chosen) < count:
            raise ValueError(f"{kind}.jsonl has {len(lines)} rows; rows {start} to {start + count} were asked for")
        for line in chosen:
            requests.append({**json.loads(line), "kind": kind})
    random.Random(seed).shuffle(requests)
    return requests


def file_workload(path: Path) -> list[dict]:
    requests = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, request in enumerate(requests):
        if "messages" not in request or "kind" not in request:
            raise ValueError(f"{path} row {index + 1}: needs 'messages' and 'kind'")
        request.setdefault("id", f"row{index + 1}")
    return requests


def workload_digest(requests: list[dict]) -> str:
    """SHA-256 of the requests in the order they are sent, so two runs can be shown to match."""
    text = "\n".join(json.dumps(request, sort_keys=True, ensure_ascii=False) for request in requests)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile; the same rule as bench/load_check.py."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * len(ordered) + 0.5) - 1))]


class Client:
    def __init__(self, url: str, timeout_s: float = 1200.0) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def post(self, path: str, body: dict, kind: str | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if kind:
            headers[KIND_HEADER] = kind
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read())

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=30) as response:
            return json.loads(response.read())

    def generate(self, request: dict, max_tokens: int) -> dict:
        """One request: render the chat template on the server, then a raw greedy completion."""
        template = {"messages": request["messages"],
                    "chat_template_kwargs": {"enable_thinking": bool(request.get("thinking"))}}
        if request.get("tools"):
            template["tools"] = request["tools"]
        prompt = self.post("/apply-template", template)["prompt"]
        return self.post("/completion", {
            "prompt": prompt, "n_predict": max_tokens, "temperature": 0.0, "top_k": 1, "seed": 0,
            "cache_prompt": False,
        }, kind=request["kind"])


def one_pass(client: Client, requests: list[dict], users: int, max_tokens: int) -> dict:
    pending: queue.Queue = queue.Queue()
    for request in requests:
        pending.put(request)
    done: list[dict] = []
    lock = threading.Lock()

    def user() -> None:
        while True:
            try:
                request = pending.get_nowait()
            except queue.Empty:
                return
            row = {"id": request["id"], "kind": request["kind"], "error": None, "tokens": 0,
                   "sent_unix": time.time()}
            begin = time.perf_counter()
            try:
                reply = client.generate(request, max_tokens * (2 if request.get("thinking") else 1))
                timings = reply["timings"]
                row.update(tokens=timings["predicted_n"], predicted_ms=timings["predicted_ms"],
                           prompt_n=timings.get("prompt_n"), prompt_ms=timings["prompt_ms"],
                           draft_n=timings.get("draft_n", 0), draft_n_accepted=timings.get("draft_n_accepted", 0),
                           stop_type=reply.get("stop_type"))
            except urllib.error.HTTPError as error:   # a failed request under load is a result
                row["error"] = f"HTTP {error.code}: {error.read()[:160].decode('utf-8', 'replace')}"
            except Exception as error:  # noqa: BLE001
                row["error"] = f"{type(error).__name__}: {str(error)[:160]}"
            row["latency_s"] = time.perf_counter() - begin
            with lock:
                done.append(row)

    begin = time.perf_counter()
    threads = [threading.Thread(target=user) for _ in range(users)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return summarize(done, time.perf_counter() - begin)


def summarize(rows: list[dict], wall_s: float) -> dict:
    good = [row for row in rows if not row["error"]]
    latencies = [row["latency_s"] for row in good]
    tokens = sum(row["tokens"] for row in good)
    drafted = sum(row.get("draft_n") or 0 for row in good)
    by_kind = {}
    for kind in sorted({row["kind"] for row in good}):
        speeds = [row["tokens"] / (row["predicted_ms"] / 1000) for row in good
                  if row["kind"] == kind and row["predicted_ms"]]
        by_kind[kind] = {"requests": sum(1 for row in good if row["kind"] == kind),
                         "server_tok_s_median": statistics.median(speeds) if speeds else None}
    return {
        "wall_s": wall_s, "requests": len(rows), "failed": len(rows) - len(good), "tokens": tokens,
        "throughput_tok_s": tokens / wall_s if wall_s else None,
        "latency_p50_s": percentile(latencies, 0.50) if latencies else None,
        "latency_p95_s": percentile(latencies, 0.95) if latencies else None,
        "latency_p99_s": percentile(latencies, 0.99) if latencies else None,
        "latency_mean_s": statistics.mean(latencies) if latencies else None,
        "draft_tokens": drafted,
        "accepted_share": sum(row.get("draft_n_accepted") or 0 for row in good) / drafted if drafted else None,
        "by_kind": by_kind, "rows": rows,
    }


def client_hardware() -> dict:
    quota = Path("/sys/fs/cgroup/cpu.max")
    return {"machine": platform.machine(), "system": platform.platform(), "processor": platform.processor(),
            "cpus_visible": os.cpu_count(), "cpu_quota": quota.read_text().strip() if quota.exists() else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="llama-server or the metrics gateway")
    parser.add_argument("--policy", required=True, help="name of the server's drafting setting; goes in the file name")
    parser.add_argument("--server-args", default="", help="the server's arguments for this policy, recorded as given")
    parser.add_argument("--out", required=True, help="results folder")
    parser.add_argument("--users", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=192, help="doubled for requests with thinking on")
    parser.add_argument("--workload", help="a JSON-lines workload file, sent in its own order")
    parser.add_argument("--kind-dir", help="folder with one <kind>.jsonl per kind")
    parser.add_argument("--kinds", nargs="+", default=KINDS)
    parser.add_argument("--start", type=int, default=30, help="with --kind-dir: first row taken from each kind")
    parser.add_argument("--count", type=int, default=40, help="with --kind-dir: rows per kind")
    parser.add_argument("--seed", type=int, default=11, help="with --kind-dir: order seed")
    parser.add_argument("--limit", type=int, help="send only the first N requests of the workload (small test runs)")
    parser.add_argument("--hardware", default="", help="the server's machine, in words, when it is not this one")
    parser.add_argument("--llama-commit", default="", help="llama.cpp commit of the server")
    parser.add_argument("--quiet-check", action="store_true",
                        help="before each user count, check the machine is quiet (serving/quiet.py); stop if it is not")
    parser.add_argument("--server-port", type=int, help="with --quiet-check: llama-server's port, so it is not a stranger")
    parser.add_argument("--also-allow-port", type=int, nargs="*", default=[],
                        help="with --quiet-check: ports of further llama-servers that belong to this run")
    parser.add_argument("--purpose", default="measurement",
                        help="'measurement', or 'function-check' for a run whose timings must not be quoted")
    args = parser.parse_args()
    if bool(args.workload) == bool(args.kind_dir):
        parser.error("give exactly one of --workload and --kind-dir")

    if args.workload:
        requests = file_workload(Path(args.workload))
        source = {"file": args.workload}
    else:
        requests = kind_dir_workload(Path(args.kind_dir), args.kinds, args.start, args.count, args.seed)
        source = {"kind_dir": args.kind_dir, "kinds": args.kinds, "start": args.start,
                  "per_kind": args.count, "order_seed": args.seed}
    if args.limit:
        requests = requests[:args.limit]
    source.update(requests=len(requests), sha256_in_order=workload_digest(requests),
                  per_kind_sent={kind: sum(1 for r in requests if r["kind"] == kind)
                                 for kind in sorted({r["kind"] for r in requests})})

    client = Client(args.url)
    props = client.get("/props")
    server = {"url": args.url, "model_path": props.get("model_path"), "total_slots": props.get("total_slots"),
              "build_info": props.get("build_info"),
              "context_per_slot": (props.get("default_generation_settings") or {}).get("n_ctx"),
              "args": args.server_args, "llama_cpp_commit": args.llama_commit, "hardware": args.hardware}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client.generate({"id": "warmup", "kind": "warmup", "messages": [{"role": "user", "content": "Hello"}]}, 16)

    for users in args.users:
        path = out / f"{args.policy}_u{users}.json"
        if path.exists():
            print(f"{args.policy} users={users}: already done, skipped", flush=True)
            continue
        quiet_record = None
        if args.quiet_check:
            import quiet
            quiet_record = quiet.wait_until_quiet(set().union(*(quiet.server_pids(port) for port in
                                                                      [args.server_port, *args.also_allow_port] if port)))
            if not quiet_record["quiet"]:
                print(f"{args.policy} users={users}: machine not quiet, run not started: {quiet_record['problems']}", flush=True)
                raise SystemExit(4)
        started = time.time()
        passes = [one_pass(client, requests, users, args.max_tokens) for _ in range(args.repeats)]
        record = {"part": 7, "purpose": args.purpose, "policy": args.policy, "users": users, "repeats": args.repeats,
                  "max_tokens": args.max_tokens, "decoding": "greedy, prompt cache off", "workload": source,
                  "server": server, "client_hardware": client_hardware(), "quiet_check": quiet_record,
                  "started_unix": started,
                  "wall_s": time.time() - started, "passes": passes}
        path.write_text(json.dumps(record), encoding="utf-8")
        throughput = [p["throughput_tok_s"] for p in passes]
        p50 = [p["latency_p50_s"] for p in passes if p["latency_p50_s"] is not None]
        print(f"{args.policy} users={users}: throughput {statistics.median(throughput):.1f} tok/s "
              f"({min(throughput):.1f}-{max(throughput):.1f}), "
              f"p50 {statistics.median(p50):.2f} s, " if p50 else f"{args.policy} users={users}: no request succeeded, ",
              f"failed {sum(p['failed'] for p in passes)} of {sum(p['requests'] for p in passes)}", flush=True)
    print("loadgen DONE", flush=True)


if __name__ == "__main__":
    main()
