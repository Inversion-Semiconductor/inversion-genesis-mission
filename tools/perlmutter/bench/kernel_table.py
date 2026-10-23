"""Print launch settings and times of matching kernels for each profiled run.

Usage: python -m bench.kernel_table BATCH_DIR [BATCH_DIR...] [--match REGEX] [--csv FILE]

Reads <batch>/<variant>_r<k>/profile/rank0.sqlite (written by nsys stats) and
result.json for the captured step count. For each kernel name that matches
--match (default: deposition and gather kernels), it prints the registers per
thread, local (spill) memory per thread, threads per block, launches per step,
mean microseconds per launch, and milliseconds per step. Runs without a profile
are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from bench.families import short_name
from bench.variants import VARIANTS

DEFAULT_MATCH = r"deposit|fused_linear|gather"

QUERY = """
SELECT s.value, k.registersPerThread, k.localMemoryPerThread, k.blockX,
       count(*), sum(k.end - k.start)
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS s ON s.id = k.shortName
GROUP BY s.value, k.registersPerThread, k.localMemoryPerThread, k.blockX
"""


def kernel_rows(run: Path, pattern: re.Pattern) -> list[dict]:
    """Return one row per matching kernel and launch configuration of a run."""
    result = json.loads((run / "result.json").read_text())
    steps = (result.get("profiled") or {}).get("steps")
    database = run / "profile" / "rank0.sqlite"
    if not steps or not database.exists():
        return []
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        records = connection.execute(QUERY).fetchall()
    rows = []
    for name, registers, local_bytes, block, launches, total_ns in records:
        if not pattern.search(name):
            continue
        rows.append(
            {
                "batch": run.parent.name,
                "run": run.name,
                "variant": result.get("variant", run.name),
                "kernel": short_name(name),
                "registers": registers,
                "local_bytes": local_bytes,
                "tpb": block,
                "launches_per_step": launches / steps,
                "us_per_launch": total_ns / 1e3 / launches,
                "ms_per_step": total_ns / 1e6 / steps,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batches", nargs="+", type=Path)
    parser.add_argument("--match", default=DEFAULT_MATCH, help="kernel name regex")
    parser.add_argument("--csv", type=Path, help="also write the rows to this file")
    args = parser.parse_args()
    pattern = re.compile(args.match)
    order = {name: index for index, name in enumerate(VARIANTS)}
    rows = []
    for batch in args.batches:
        for run in sorted(batch.glob("*_r*")):
            if (run / "result.json").exists():
                rows.extend(kernel_rows(run, pattern))
    if not rows:
        raise SystemExit("no profiled runs with matching kernels")
    rows.sort(
        key=lambda r: (
            order.get(r["variant"], len(order)),
            r["batch"],
            r["run"],
            r["kernel"],
        )
    )
    header = (
        "variant | batch/run | kernel | regs | spill B | tpb | launches/step"
        " | us/launch | ms/step"
    )
    print(header)
    print("|".join("---" for _ in header.split("|")))
    for r in rows:
        print(
            f"{r['variant']} | {r['batch']}/{r['run']} | {r['kernel']} | "
            f"{r['registers']} | {r['local_bytes']} | {r['tpb']} | "
            f"{r['launches_per_step']:.2f} | {r['us_per_launch']:.1f} | "
            f"{r['ms_per_step']:.3f}"
        )
    # Sum the matching kernels of each run, then average the runs of a variant.
    per_run: dict[tuple[str, str, str], float] = defaultdict(float)
    for r in rows:
        per_run[(r["variant"], r["batch"], r["run"])] += r["ms_per_step"]
    per_variant: dict[str, list[float]] = defaultdict(list)
    for (variant, _, _), total in per_run.items():
        per_variant[variant].append(total)
    print()
    print("variant | runs | matching kernels, mean ms/step | min | max")
    print("---|---|---|---|---")
    for variant, totals in sorted(
        per_variant.items(), key=lambda item: order.get(item[0], len(order))
    ):
        print(
            f"{variant} | {len(totals)} | {sum(totals) / len(totals):.3f} | "
            f"{min(totals):.3f} | {max(totals):.3f}"
        )
    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
