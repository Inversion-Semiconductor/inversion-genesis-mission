"""Summarize an Nsight Compute capture: how close each kernel runs to its limits.

Usage: python -m bench.ncu_table RUN_OR_BATCH [RUN_OR_BATCH...] [--ncu PATH] [--csv FILE]

Reads <run>/profile/rank0.ncu-rep (written by ncu_wrap.sh in mode ncu) through
``ncu --import --page raw --csv`` and result.json for the captured step count.
Launches are grouped by kernel name. Times are summed; percentages of peak are
weighted by launch time. Each kernel gets a verdict from its highest unit
utilization: DRAM, L2, L1, FP64 pipe, or instruction issue. A kernel with every
unit below 40 % is limited by latency (too few warps or dependent loads), not by
a hardware throughput.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from bench.families import classify, short_name

# A100 SXM4 40 GB peaks, for the achieved-rate columns only. The percentages come
# from ncu, which measures them against the locked profiling clock.
PEAK_DRAM_GBS = 1555.0
PEAK_FP64_GFLOPS = 9700.0

TIME = "gpu__time_duration.sum"
UNITS = {
    "DRAM": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "L2": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "L1": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "FP64": "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active",
    "issue": "smsp__issue_active.avg.pct_of_peak_sustained_active",
}
WEIGHTED = {
    **UNITS,
    "occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "occupancy_theory": "sm__maximum_warps_per_active_cycle_pct",
    "l1_hit": "l1tex__t_sector_hit_rate.pct",
    "l2_hit": "lts__t_sector_hit_rate.pct",
    "ld_bytes_per_sector": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    "stall_long_scoreboard": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "stall_lg_throttle": "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "stall_math_pipe": "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
}
SUMMED = {
    "dram_read": "dram__bytes_read.sum",
    "dram_write": "dram__bytes_write.sum",
    "dadd": "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
    "dmul": "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
    "dfma": "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
    "fadd": "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "fmul": "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "ffma": "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
}
LAUNCH = {
    "registers": "launch__registers_per_thread",
    "tpb": "launch__block_size",
    "occupancy_limit_registers": "launch__occupancy_limit_registers",
}


def number(text: str) -> float | None:
    """Parse an ncu CSV value; ncu groups thousands with commas."""
    text = text.strip().replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def read_report(report: Path, ncu: str) -> list[dict[str, str]]:
    """Return one dict per profiled launch, keyed by metric name."""
    output = subprocess.run(
        # --csv alone scales units per value (us, Mbyte, Kbyte); force base units.
        [
            ncu,
            "--import",
            str(report),
            "--page",
            "raw",
            "--csv",
            "--print-units",
            "base",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = list(csv.reader(io.StringIO(output)))
    if len(rows) < 3:
        return []
    header, units = rows[0], dict(zip(rows[0], rows[1]))
    # The second row holds the units of each column; the sums below assume these.
    expected = {TIME: "ns", SUMMED["dram_read"]: "byte", SUMMED["dram_write"]: "byte"}
    for metric, unit in expected.items():
        if metric in units and units[metric] != unit:
            raise SystemExit(
                f"{report}: {metric} is in {units[metric]}, expected {unit}"
            )
    return [dict(zip(header, row)) for row in rows[2:] if len(row) == len(header)]


def summarize(run: Path, ncu: str) -> list[dict]:
    """Group the launches of one run by kernel and compute the limits."""
    result = json.loads((run / "result.json").read_text())
    steps = (result.get("profiled") or {}).get("steps") or 1
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for launch in read_report(run / "profile" / "rank0.ncu-rep", ncu):
        groups[short_name(launch.get("Kernel Name", "?"))].append(launch)
    kernels = []
    for name, launches in groups.items():
        times = [number(launch.get(TIME, "")) or 0.0 for launch in launches]
        total_ns = sum(times)
        if total_ns <= 0:
            continue
        entry: dict = {
            "variant": result.get("variant", run.name),
            "kernel": name,
            "family": classify(name),
            "launches_per_step": len(launches) / steps,
            "ms_per_step": total_ns / 1e6 / steps,
            "us_per_launch": total_ns / 1e3 / len(launches),
        }
        for key, metric in WEIGHTED.items():
            pairs = [
                (value, t)
                for launch, t in zip(launches, times)
                if (value := number(launch.get(metric, ""))) is not None
            ]
            weight = sum(t for _, t in pairs)
            entry[key] = sum(v * t for v, t in pairs) / weight if weight else None
        for key, metric in SUMMED.items():
            entry[key] = sum(
                number(launch.get(metric, "")) or 0.0 for launch in launches
            )
        for key, metric in LAUNCH.items():
            values = {launch.get(metric, "") for launch in launches}
            entry[key] = "/".join(sorted(v for v in values if v)) or None
        seconds = total_ns / 1e9
        dram_bytes = entry["dram_read"] + entry["dram_write"]
        fp64_flops = entry["dadd"] + entry["dmul"] + 2 * entry["dfma"]
        entry["dram_gbs"] = dram_bytes / seconds / 1e9
        entry["fp64_gflops"] = fp64_flops / seconds / 1e9
        entry["fp64_per_dram_byte"] = fp64_flops / dram_bytes if dram_bytes else None
        entry["verdict"] = verdict(entry)
        kernels.append(entry)
    kernels.sort(key=lambda e: -e["ms_per_step"])
    return kernels


def verdict(entry: dict) -> str:
    """Name the unit closest to its peak and how close it is."""
    usage = {unit: entry[unit] for unit in UNITS if entry.get(unit) is not None}
    if not usage:
        return "no data"
    unit, value = max(usage.items(), key=lambda item: item[1])
    if value >= 60:
        return f"{unit}-bound ({value:.0f}%)"
    if value >= 40:
        return f"{unit} {value:.0f}%, headroom"
    return f"latency-bound (max {unit} {value:.0f}%)"


def fmt(value, spec: str = ".0f") -> str:
    return "-" if value is None else format(value, spec)


def print_tables(kernels: list[dict]) -> None:
    print(
        "variant | kernel | family | ms/step | launches/step | DRAM % | L2 % | L1 % "
        "| FP64 % | issue % | occupancy % (theory) | DRAM GB/s | FP64 GFLOP/s "
        "| FLOP/B | ld sector use % | regs | verdict"
    )
    print("|".join(["---"] * 17))
    for e in kernels:
        print(
            f"{e['variant']} | {e['kernel']} | {e['family']} | {e['ms_per_step']:.3f} | "
            f"{e['launches_per_step']:.2f} | {fmt(e['DRAM'])} | {fmt(e['L2'])} | "
            f"{fmt(e['L1'])} | {fmt(e['FP64'])} | {fmt(e['issue'])} | "
            f"{fmt(e['occupancy'])} ({fmt(e['occupancy_theory'])}) | "
            f"{e['dram_gbs']:.0f} | {e['fp64_gflops']:.0f} | "
            f"{fmt(e['fp64_per_dram_byte'], '.2f')} | "
            f"{fmt(e['ld_bytes_per_sector'])} | {e['registers'] or '-'} | "
            f"{e['verdict']}"
        )
    print()
    print(
        f"Peaks: DRAM {PEAK_DRAM_GBS:.0f} GB/s, FP64 {PEAK_FP64_GFLOPS:.0f} GFLOP/s "
        "(non-tensor). Ridge point: "
        f"{PEAK_FP64_GFLOPS / PEAK_DRAM_GBS:.1f} FP64 FLOP per DRAM byte."
    )
    print(
        "ncu locks the SM clock at base (about 1095 MHz on A100) while profiling, so"
        " ms/step and GFLOP/s are below boost-clock values; the percentages are"
        " relative to the locked clock."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="run or batch directories")
    parser.add_argument("--ncu", default=os.environ.get("NCU") or shutil.which("ncu"))
    parser.add_argument("--csv", type=Path, help="also write the rows to this file")
    args = parser.parse_args()
    if not args.ncu:
        raise SystemExit("ncu not found; pass --ncu or load cudatoolkit")
    runs = []
    for path in args.paths:
        if (path / "profile" / "rank0.ncu-rep").exists():
            runs.append(path)
        else:
            runs.extend(
                run
                for run in sorted(path.glob("*_r*"))
                if (run / "profile" / "rank0.ncu-rep").exists()
            )
    if not runs:
        raise SystemExit("no runs with profile/rank0.ncu-rep")
    kernels = [entry for run in runs for entry in summarize(run, args.ncu)]
    if not kernels:
        raise SystemExit("the reports contain no kernel launches")
    print_tables(kernels)
    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(kernels[0]))
            writer.writeheader()
            writer.writerows(kernels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
