"""Is the machine quiet enough for a timed run?

    python serving/quiet.py [--allow-pid PID ...]      # exit 0 quiet, 1 not quiet

Checked, on the machine the server runs on:
- the GPU is idle: utilisation at most 5% in each of three readings one second apart;
- no process uses the GPU other than the ones allowed (the server being measured);
- the CPUs are idle: less than half a core busy over two seconds (from /proc/stat);
- no other llama-server, training or benchmark process is running.

The load generator calls this before every timed run and stores the readings in the results
file. When the machine is not quiet it waits and looks again, for up to --wait seconds, then
gives up: a timed run does not start on a busy machine. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BUSY_NAMES = ("llama-server", "llama-bench", "llama-cli", "train.py", "specedge-agent", "loadgen.py",
              "fallback_test.py", "speed_check.py", "quantize", "convert_hf_to_gguf")


def run(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout


def cpu_busy_cores(seconds: float = 2.0) -> float:
    def snapshot() -> tuple[int, int]:
        fields = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        idle = fields[3] + fields[4]
        return sum(fields) - idle, sum(fields)
    busy0, total0 = snapshot()
    time.sleep(seconds)
    busy1, total1 = snapshot()
    share = (busy1 - busy0) / max(1, total1 - total0)
    return share * (os.cpu_count() or 1)


def reading(allowed: set[int]) -> dict:
    gpu = []
    for _ in range(3):
        line = run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"]).strip()
        gpu.append([float(x) for x in line.split(",")] if line else None)
        time.sleep(1)
    apps = [line.split(", ") for line in run(["nvidia-smi", "--query-compute-apps=pid,name,used_memory",
                                               "--format=csv,noheader"]).strip().splitlines() if line.strip()]
    own = {os.getpid(), os.getppid()}
    others = []
    for line in run(["ps", "-eo", "pid,args"]).splitlines()[1:]:
        pid, _, args = line.strip().partition(" ")
        if pid.isdigit() and int(pid) not in allowed | own and any(name in args for name in BUSY_NAMES) \
                and "quiet.py" not in args and "tmux" not in args and "bash" not in args.split()[0]:
            others.append({"pid": int(pid), "args": args[:120]})
    busy = cpu_busy_cores()
    problems = []
    if any(sample is None or sample[0] > 5 for sample in gpu):
        problems.append(f"GPU busy: utilisation readings {[s[0] if s else None for s in gpu]}")
    foreign = [app for app in apps if app and app[0].isdigit() and int(app[0]) not in allowed]
    if foreign:
        problems.append(f"other GPU processes: {foreign}")
    if busy >= 0.5:
        problems.append(f"CPU busy: {busy:.2f} cores over 2 s")
    if others:
        problems.append(f"other benchmark-like processes: {others}")
    return {"unix": time.time(), "gpu_util_mem_power": gpu, "gpu_processes": apps, "allowed_pids": sorted(allowed),
            "cpu_busy_cores": round(busy, 3), "load_average": list(os.getloadavg()),
            "other_processes": others, "quiet": not problems, "problems": problems}


def wait_until_quiet(allowed: set[int], wait_s: float = 180.0) -> dict:
    """The last reading, with how many looks it took. 'quiet' is False if it never became quiet."""
    deadline = time.time() + wait_s
    looks = 0
    while True:
        looks += 1
        result = reading(allowed)
        result["looks"] = looks
        if result["quiet"] or time.time() > deadline:
            return result
        time.sleep(5)


def server_pids(port: int) -> set[int]:
    """Pids of the llama-server listening on the port, so it is not counted as a stranger."""
    pids = set()
    for line in run(["ps", "-eo", "pid,args"]).splitlines()[1:]:
        pid, _, args = line.strip().partition(" ")
        if pid.isdigit() and "llama-server" in args and f"--port {port}" in args:
            pids.add(int(pid))
    return pids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--allow-pid", type=int, nargs="*", default=[])
    parser.add_argument("--wait", type=float, default=0.0)
    args = parser.parse_args()
    result = wait_until_quiet(set(args.allow_pid), args.wait)
    print(json.dumps(result, indent=1))
    return 0 if result["quiet"] else 1


if __name__ == "__main__":
    sys.exit(main())
