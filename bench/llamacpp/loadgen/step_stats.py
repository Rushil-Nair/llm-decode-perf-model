"""Server-side time per token for each user count, from a llama-server log of a load run.

    python serving/step_stats.py <policy>_slots<N>.server.log --users 1 4 8 9 16 --repeats 3

serving/loadgen.py sends one warm-up request and then, for each user count in order, the
workload --repeats times. This splits the server's own per-request timing lines into those
phases. With plain decoding one decode step writes one token for every request in flight, so
(users / time per token) is the throughput the server's own timer implies, and
(time per token) is the time of one decode step holding that many tokens. Same method as
bench/load_log_stats.py. With drafting on, a step can write several tokens, so only the
per-token time is meaningful.

The last column is the share of decode steps that reused the previous step's compute graph:
the server's own "graphs reused" counter, against (tokens written / users) as the number of
steps. Plain decoding only.
"""

from __future__ import annotations

import argparse
import re
import statistics

EVAL = re.compile(r"print_timing: id\s+\d+ \| task \d+ \|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
REUSED = re.compile(r"graphs reused =\s+(\d+)")


def phases(log_text: str, users: list[int], repeats: int, requests: int) -> list[dict]:
    evals, reused = [], []
    for line in log_text.splitlines():
        match = EVAL.search(line)
        if match and "prompt eval" not in line:
            evals.append((float(match[1]), int(match[2])))
        match = REUSED.search(line)
        if match:
            reused.append(int(match[1]))
    evals, reused = evals[1:], reused[1:]          # the warm-up request
    size = requests * repeats
    rows = []
    for index, count in enumerate(users):
        phase = [(ms, n) for ms, n in evals[index * size:(index + 1) * size] if n]
        if not phase:
            continue
        per_token = statistics.median(ms / n for ms, n in phase)
        tokens = sum(n for _, n in phase)
        share = None
        if len(reused) >= (index + 1) * size:
            steps_reused = reused[(index + 1) * size - 1] - (reused[index * size - 1] if index else 0)
            share = steps_reused * count / tokens
        rows.append({"users": count, "requests": len(phase), "tokens": tokens, "ms_per_token_median": per_token,
                     "implied_tok_s": 1000 * count / per_token, "graph_reuse_share": share})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("log")
    parser.add_argument("--users", nargs="+", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--requests", type=int, default=200, help="requests in one pass of the workload")
    args = parser.parse_args()
    text = open(args.log, encoding="utf-8", errors="replace").read()
    print("| Users | Requests | Tokens written | Server time per token, median | Tokens/s that implies for all users | Decode steps that reused the graph |")
    print("|---|---|---|---|---|---|")
    for row in phases(text, args.users, args.repeats, args.requests):
        reuse = "" if row["graph_reuse_share"] is None else f"{row['graph_reuse_share']:.0%}"
        print(f"| {row['users']} | {row['requests']} | {row['tokens']} | {row['ms_per_token_median']:.2f} ms | "
              f"{row['implied_tok_s']:.0f} | {reuse} |")


if __name__ == "__main__":
    main()
