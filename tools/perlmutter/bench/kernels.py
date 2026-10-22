"""Summarize the nsys capture of one run into per-kernel and per-family tables.

Usage: python -m bench.kernels RUN_DIR [--nsys PATH]

Writes profile/kernels.csv, profile/families.csv, and profile/kernels.json.
Times are normalized per PIC step with the captured window in result.json.
Run it on a login node if the job ended before the summary step.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from bench.families import classify, short_name

REPORTS = ("cuda_gpu_kern_sum", "cuda_gpu_mem_time_sum")


def nsys_stats(nsys: str, report: Path, out_prefix: Path) -> dict[str, list[dict]]:
    """Export the nsys reports as CSV and return their rows."""
    subprocess.run(
        [
            nsys,
            "stats",
            "--report",
            ",".join(REPORTS),
            "--format",
            "csv",
            "--output",
            str(out_prefix),
            "--force-export=true",
            "--force-overwrite=true",
            str(report),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    tables = {}
    for name in REPORTS:
        path = Path(f"{out_prefix}_{name}.csv")
        tables[name] = list(csv.DictReader(path.open())) if path.exists() else []
    return tables


def summarize(run: Path, nsys: str) -> dict:
    result = json.loads((run / "result.json").read_text())
    profiled = result.get("profiled") or {}
    steps = profiled.get("steps")
    if not steps:
        raise SystemExit(f"{run}: result.json has no completed profile window")
    reports = sorted((run / "profile").glob("rank*.nsys-rep"))
    if not reports:
        raise SystemExit(f"{run}: no profile/rank*.nsys-rep")
    tables = nsys_stats(nsys, reports[0], run / "profile" / "stats")

    kernels = []
    for row in tables["cuda_gpu_kern_sum"]:
        total_ns = float(row["Total Time (ns)"])
        kernels.append(
            {
                "family": classify(row["Name"]),
                "kernel": short_name(row["Name"]),
                "instances": int(row["Instances"]),
                "total_ms": total_ns / 1e6,
                "ms_per_step": total_ns / 1e6 / steps,
                "calls_per_step": int(row["Instances"]) / steps,
                "name": row["Name"],
            }
        )
    for row in tables["cuda_gpu_mem_time_sum"]:
        total_ns = float(row["Total Time (ns)"])
        kernels.append(
            {
                "family": "memory_ops",
                "kernel": row["Operation"],
                "instances": int(row["Count"]),
                "total_ms": total_ns / 1e6,
                "ms_per_step": total_ns / 1e6 / steps,
                "calls_per_step": int(row["Count"]) / steps,
                "name": row["Operation"],
            }
        )
    kernels.sort(key=lambda k: -k["total_ms"])

    families: dict[str, dict] = defaultdict(lambda: {"ms_per_step": 0.0, "kernels": 0})
    for kernel in kernels:
        families[kernel["family"]]["ms_per_step"] += kernel["ms_per_step"]
        families[kernel["family"]]["kernels"] += 1
    gpu_ms_per_step = sum(f["ms_per_step"] for f in families.values())
    family_rows = [
        {
            "family": name,
            "ms_per_step": value["ms_per_step"],
            "share": value["ms_per_step"] / gpu_ms_per_step if gpu_ms_per_step else 0.0,
            "kernels": value["kernels"],
        }
        for name, value in sorted(
            families.items(), key=lambda kv: -kv[1]["ms_per_step"]
        )
    ]

    profile = run / "profile"
    with (profile / "kernels.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(kernels[0]))
        writer.writeheader()
        writer.writerows(kernels)
    with (profile / "families.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(family_rows[0]))
        writer.writeheader()
        writer.writerows(family_rows)
    summary = {
        "variant": result.get("variant"),
        "profiled_steps": steps,
        "window_wall_ms_per_step": 1e3 * profiled["seconds"] / steps,
        "gpu_busy_ms_per_step": gpu_ms_per_step,
        "other_share": next(
            (r["share"] for r in family_rows if r["family"] == "other"), 0.0
        ),
        "families": family_rows,
        "report": str(reports[0]),
    }
    (profile / "kernels.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--nsys", default=os.environ.get("NSYS") or shutil.which("nsys")
    )
    args = parser.parse_args()
    if not args.nsys:
        parser.error("nsys not found; pass --nsys or load the cudatoolkit module")
    for run in args.runs:
        summary = summarize(run.resolve(), args.nsys)
        print(
            f"{run}: {summary['variant']} gpu {summary['gpu_busy_ms_per_step']:.3f} ms/step over {summary['profiled_steps']} steps"
        )
        for row in summary["families"]:
            print(
                f"  {row['family']:22s} {row['ms_per_step']:9.4f} ms/step  {100 * row['share']:5.1f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
