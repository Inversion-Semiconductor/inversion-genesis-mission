"""Find where the GPU sits idle in a profiled window and what the host was doing.

Usage: python -m bench.idle_gaps RUN_OR_BATCH [RUN_OR_BATCH...] [--min-us N] [--top N]

Reads <run>/profile/rank0.sqlite (the nsys export written by bench.kernels) and
result.json for the captured step count. Kernels, memory copies, and memsets
form the GPU busy timeline; every interval between them longer than --min-us
is an idle gap. The operation that ends a gap was issued by a CUDA API call
(linked by correlation ID):

- If the host issued it after the GPU went idle, the host was late. The host
  time inside the gap is split by the CUDA calls that thread made (for example
  cudaFree, which synchronizes the device); the rest is host work without a
  CUDA call (Python, NumPy, I/O). The gap gets the label of the largest part.
- If the host issued it before the GPU went idle, the GPU waited on something
  else (a dependency, launch latency, or a copy engine).

Times are per profiled step. nsys tracing adds a small cost to every API call.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from bench.families import classify, short_name

# Calls that can block the host until earlier GPU work finishes.
BLOCKING = re.compile(
    r"Synchronize|cudaMemcpy(?!.*Async)|cudaMemcpyAsync|cuMemcpyDtoH|cudaFree"
    r"|cuMemFree|cudaMalloc|cuMemAlloc|cudaHostAlloc|cudaEventQuery|cuStreamQuery"
)
COPY_KINDS = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}


def api_name(name: str) -> str:
    """Drop the version suffix of runtime API names, e.g. ``_v3020``."""
    return re.sub(r"_v\d+$", "", name)


def load(run: Path):
    database = run / "profile" / "rank0.sqlite"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        strings = dict(connection.execute("SELECT id, value FROM StringIds"))
        ops = []
        for start, end, corr, name in connection.execute(
            "SELECT start, end, correlationId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ):
            label = short_name(strings.get(name, "?"))
            ops.append((start, end, corr, classify(label), label))
        for start, end, corr, kind, size in connection.execute(
            "SELECT start, end, correlationId, copyKind, bytes "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        ):
            label = f"memcpy {COPY_KINDS.get(kind, kind)}"
            ops.append((start, end, corr, label, f"{label} {size} B"))
        for start, end, corr in connection.execute(
            "SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_MEMSET"
        ):
            ops.append((start, end, corr, "memset", "memset"))
        calls = connection.execute(
            "SELECT start, end, globalTid, correlationId, nameId "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY globalTid, start"
        ).fetchall()
        copies = connection.execute(
            "SELECT copyKind, count(*), sum(bytes), sum(end - start) "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"
        ).fetchall()
    ops.sort()
    calls = [
        (s, e, tid, corr, api_name(strings.get(n, "?"))) for s, e, tid, corr, n in calls
    ]
    return ops, calls, copies


def host_cause(calls, starts, thread_first, index, gap_start, totals) -> str:
    """Split the host time from the gap start to the issuing call by CUDA call."""
    issue_start, _, tid, _, _ = calls[index]
    window = issue_start - gap_start
    parts = defaultdict(float)
    # Calls of this thread that overlap [gap_start, issue_start).
    first = max(
        thread_first[tid],
        bisect.bisect_left(starts, gap_start, thread_first[tid], index) - 1,
    )
    for start, end, call_tid, _, name in calls[first:index]:
        if call_tid != tid:
            continue
        overlap = min(end, issue_start) - max(start, gap_start)
        if overlap > 0:
            parts[name] += overlap
    parts["host work (no CUDA call)"] = max(0.0, window - sum(parts.values()))
    for name, ns in parts.items():
        totals[name] += ns
    name, ns = max(parts.items(), key=lambda item: item[1])
    return f"host late: {name}"


def analyze(run: Path, min_us: float) -> dict:
    result = json.loads((run / "result.json").read_text())
    steps = (result.get("profiled") or {}).get("steps")
    if not steps:
        raise SystemExit(f"{run}: result.json has no profiled steps")
    ops, calls, copies = load(run)
    if not ops:
        raise SystemExit(f"{run}: no GPU activity in the capture")

    # The call that issued each GPU operation, and the calls of each host thread.
    position = {corr: index for index, (_, _, _, corr, _) in enumerate(calls)}
    thread_first = {}
    for index, (_, _, tid, _, _) in enumerate(calls):
        thread_first.setdefault(tid, index)
    starts = [call[0] for call in calls]
    host_in_gaps = defaultdict(
        float
    )  # CUDA call or "host work" -> ns inside host-late gaps

    busy_ns = 0
    idle = defaultdict(float)  # cause -> ns
    sites = defaultdict(
        lambda: [0.0, 0]
    )  # (previous op, next op, cause) -> [ns, count]
    window_start, busy_end = ops[0][0], ops[0][0]
    previous_label = "start"
    gaps = 0
    for start, end, corr, family, label in ops:
        gap = start - busy_end
        if gap > min_us * 1e3:
            gaps += 1
            cause = "unattributed"
            index = position.get(corr)
            if index is not None:
                issue_start = calls[index][0]
                if issue_start >= busy_end:
                    cause = host_cause(
                        calls, starts, thread_first, index, busy_end, host_in_gaps
                    )
                else:
                    cause = "issued before idle (GPU-side wait)"
            idle[cause] += gap
            key = (previous_label, family if family != "other" else label, cause)
            sites[key][0] += gap
            sites[key][1] += 1
        if end > busy_end:
            busy_ns += end - max(start, busy_end)
            busy_end = end
        previous_label = family if family != "other" else label
    wall_ns = busy_end - window_start
    blocking_calls = defaultdict(lambda: [0, 0.0])
    for start, end, _, _, name in calls:
        if BLOCKING.search(name):
            blocking_calls[name][0] += 1
            blocking_calls[name][1] += end - start
    return {
        "run": f"{run.parent.name}/{run.name}",
        "variant": result.get("variant", run.name),
        "steps": steps,
        "wall_ms": wall_ns / 1e6 / steps,
        "busy_ms": busy_ns / 1e6 / steps,
        "idle_ms": (wall_ns - busy_ns) / 1e6 / steps,
        "gaps_per_step": gaps / steps,
        "idle_by_cause": {k: v / 1e6 / steps for k, v in idle.items()},
        "host_in_gaps": {k: v / 1e6 / steps for k, v in host_in_gaps.items()},
        "sites": [
            (prev, nxt, cause, ns / 1e6 / steps, count / steps)
            for (prev, nxt, cause), (ns, count) in sites.items()
        ],
        "blocking_calls": {
            name: (count / steps, ns / 1e6 / steps)
            for name, (count, ns) in blocking_calls.items()
        },
        "copies": [
            (
                COPY_KINDS.get(kind, str(kind)),
                count / steps,
                size / steps / 1e6,
                ns / 1e6 / steps,
            )
            for kind, count, size, ns in copies
        ],
    }


def report(entry: dict, top: int) -> None:
    print(f"## {entry['variant']} ({entry['run']}, {entry['steps']} steps)")
    print()
    print(
        f"wall {entry['wall_ms']:.1f} ms/step, GPU busy {entry['busy_ms']:.1f}, "
        f"GPU idle {entry['idle_ms']:.1f} in {entry['gaps_per_step']:.0f} gaps per step"
    )
    print()
    print("cause of idle time | ms/step | share of idle")
    print("---|---|---")
    for cause, ms in sorted(entry["idle_by_cause"].items(), key=lambda i: -i[1]):
        print(f"{cause} | {ms:.2f} | {100 * ms / max(entry['idle_ms'], 1e-9):.0f}%")
    print()
    print("host time inside host-late gaps | ms/step")
    print("---|---")
    for name, ms in sorted(entry["host_in_gaps"].items(), key=lambda i: -i[1]):
        if ms >= 0.01:
            print(f"{name} | {ms:.2f}")
    print()
    print(f"largest idle sites (top {top}) | ms/step | gaps/step")
    print("---|---|---")
    for prev, nxt, cause, ms, count in sorted(entry["sites"], key=lambda s: -s[3])[
        :top
    ]:
        print(f"{prev} -> {nxt} ({cause}) | {ms:.2f} | {count:.3f}")
    print()
    print(
        "blocking host calls | calls/step | host ms/step (includes waiting for the GPU)"
    )
    print("---|---|---")
    for name, (count, ms) in sorted(
        entry["blocking_calls"].items(), key=lambda i: -i[1][1]
    ):
        print(f"{name} | {count:.1f} | {ms:.2f}")
    print()
    print("memory copies | copies/step | MB/step | GPU ms/step")
    print("---|---|---|---")
    for kind, count, size, ms in entry["copies"]:
        print(f"{kind} | {count:.1f} | {size:.2f} | {ms:.3f}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="run or batch directories")
    parser.add_argument(
        "--min-us", type=float, default=2.0, help="smallest gap counted"
    )
    parser.add_argument("--top", type=int, default=15, help="idle sites to list")
    args = parser.parse_args()
    runs = []
    for path in args.paths:
        if (path / "profile" / "rank0.sqlite").exists():
            runs.append(path)
        else:
            runs.extend(
                run
                for run in sorted(path.glob("*_r*"))
                if (run / "profile" / "rank0.sqlite").exists()
            )
    if not runs:
        raise SystemExit("no runs with profile/rank0.sqlite")
    for run in runs:
        report(analyze(run, args.min_us), args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
