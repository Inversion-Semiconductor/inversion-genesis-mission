"""Compare benchmark runs against the numba baseline and write a report.

Usage: python -m bench.compare BATCH_DIR [BATCH_DIR...] [--out DIR]

Reads every <batch>/<variant>_r<k>/result.json (and profile/kernels.json when
present). Writes report.md, summary.csv, runs.csv, segments.csv, and
families.csv to --out (default: <first batch>/report).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from bench.variants import BASELINE, VARIANTS


def load_runs(batches: list[Path]) -> list[dict]:
    runs = []
    for batch in batches:
        for path in sorted(batch.glob("*_r*/result.json")):
            result = json.loads(path.read_text())
            result["_run"] = path.parent
            result["_batch"] = batch.name
            kernels = path.parent / "profile" / "kernels.json"
            if kernels.exists():
                result["_kernels"] = json.loads(kernels.read_text())
            runs.append(result)
    return runs


def case_of(run: dict) -> str:
    return (run.get("case") or {}).get("name", "reduced")


def peak_gpu_memory_gib(run_dir: Path) -> float | None:
    """Return the peak nvidia-smi memory.used of the run's GPU, in GiB."""
    path = run_dir / "resources.jsonl"
    if not path.exists():
        return None
    peak = None
    for line in path.read_text().splitlines():
        values = json.loads(line).get("values")
        try:
            used = float(values[4]) / 1024
        except (TypeError, IndexError, ValueError):
            continue
        peak = used if peak is None else max(peak, used)
    return peak


CASE_KEYS = (
    ("nz", "nz"),
    ("nr", "nr"),
    ("Nm", "n_azimuthal_modes"),
    ("gamma_boost", "gamma_boost"),
    ("zmin (m)", "zmin"),
    ("rmax (m)", "rmax"),
    ("p_rmax (m)", "p_rmax"),
    ("ppc shared", ("p_nz", "p_nr", "p_nt")),
    ("ppc He", "helium_ppc"),
    ("ppc N", "nitrogen_ppc"),
    ("lab dumps", "number_dumps_lab"),
)
PHYSICS_KEYS = (
    ("peak density (cm^-3)", "peak_neutral_density_cm3"),
    ("N dopant fraction", "nitrogen_dopant_fraction"),
    ("density center (m)", "density_center_location_m"),
    ("laser focus (m)", "laser_focal_position_m"),
    ("laser energy (J)", "laser_energy_J"),
)


def case_table(runs: list[dict]) -> tuple[list[str], list[list]]:
    """Describe each case once, so reduced and production differences are explicit."""
    first: dict[str, dict] = {}
    for run in runs:
        if run.get("status") == "success":
            first.setdefault(case_of(run), run)
    names = sorted(first)
    cases = {name: first[name]["case"] for name in names}
    reference = next(
        (
            r.get("production_reference")
            for r in first.values()
            if r.get("production_reference", {}).get("hyperparameters")
        ),
        None,
    )
    if reference and "production" not in cases:
        names.append("production (reference, not run)")
        cases[names[-1]] = reference
    rows = []
    for label, key in CASE_KEYS:
        row = [label]
        for name in names:
            hyper = cases[name]["hyperparameters"]
            if isinstance(key, tuple):
                value = "/".join(str(hyper.get(k)) for k in key)
            else:
                value = hyper.get(key)
            row.append("-" if value is None else str(value))
        rows.append(row)
    for label, key in PHYSICS_KEYS:
        rows.append(
            [label] + [str(cases[n]["physical_parameters"].get(key)) for n in names]
        )
    rows.append(
        ["full steps"]
        + [first[n].get("full_steps") if n in first else "-" for n in names]
    )
    rows.append(
        ["dt (s)"]
        + [f"{first[n].get('dt_s', 0):.3e}" if n in first else "-" for n in names]
    )
    peak = {}
    particles = {}
    for run in runs:
        if run.get("status") != "success":
            continue
        name = case_of(run)
        mem = peak_gpu_memory_gib(run["_run"])
        if mem is not None:
            peak[name] = max(peak.get(name, 0.0), mem)
        total = sum(p["count"] for p in run.get("final_particles", []))
        key = (name, run.get("steps"))
        particles[key] = max(particles.get(key, 0), total)
    rows.append(["peak GPU memory (GiB)"] + [peak.get(n) for n in names])
    rows.append(
        ["macroparticles at end (steps)"]
        + [
            "; ".join(
                f"{particles[k]:,} ({k[1]})"
                for k in sorted(particles, key=lambda k: k[1] or 0)
                if k[0] == n
            )
            for n in names
        ]
    )
    rows.append(["source"] + [cases[n].get("source", "-") for n in names])
    return ["parameter"] + names, rows


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def ratio(baseline: float | None, value: float | None) -> float | None:
    if not baseline or not value:
        return None
    return baseline / value


def summarize(runs: list[dict]) -> dict[tuple, dict]:
    """Aggregate successful runs per (case, steps, variant)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs:
        if run.get("status") == "success":
            groups[(case_of(run), run.get("steps"), run["variant"])].append(run)
    summary = {}
    for key, members in groups.items():
        ms = [
            r["ms_per_step_after_warmup"]
            for r in members
            if r.get("ms_per_step_after_warmup")
        ]
        timing = defaultdict(list)
        for r in members:
            for name, value in r["timing"].items():
                timing[name].append(value)
        summary[key] = {
            "runs": len(members),
            "ms_per_step_mean": mean(ms),
            "ms_per_step_min": min(ms) if ms else None,
            "ms_per_step_stdev": statistics.stdev(ms) if len(ms) > 1 else None,
            **{f"{name}_mean": mean(values) for name, values in timing.items()},
            "gpu": members[0].get("gpu", {}).get("name"),
            "full_run": members[0].get("full_run"),
        }
    return summary


def correctness(runs: list[dict]) -> list[list]:
    """Compare final species counts and ionization levels with the baseline."""
    baseline = {}
    for run in runs:
        if run.get("status") == "success" and run["variant"] == BASELINE:
            baseline.setdefault((case_of(run), run.get("steps")), run)
    rows = []
    for run in runs:
        if run.get("status") != "success" or run["variant"] == BASELINE:
            continue
        reference = baseline.get((case_of(run), run.get("steps")))
        if reference is None:
            continue
        mine_ions = {e["index"]: e for e in run.get("final_ion_states", [])}
        ref_ions = {e["index"]: e for e in reference.get("final_ion_states", [])}
        for mine, ref in zip(run["final_particles"], reference["final_particles"]):
            count_diff = (
                (mine["count"] - ref["count"]) / ref["count"] if ref["count"] else None
            )
            hist_diff = None
            if "level_histogram" in ref and "level_histogram" in mine:
                a, b = mine["level_histogram"], ref["level_histogram"]
                size = max(len(a), len(b))
                a, b = a + [0] * (size - len(a)), b + [0] * (size - len(b))
                total = sum(b)
                hist_diff = (
                    sum(abs(x - y) for x, y in zip(a, b)) / total if total else None
                )
            rows.append(
                [
                    run["_run"].name,
                    case_of(run),
                    run.get("steps"),
                    mine["index"],
                    f"{mine['charge_e']:+.0f}e/{mine['mass_me']:.0f}me",
                    ref["count"],
                    mine["count"],
                    count_diff,
                    hist_diff,
                    _same(mine_ions, ref_ions, mine["index"], "ids_sha256"),
                    _same(mine_ions, ref_ions, mine["index"], "levels_sha256"),
                ]
            )
    return rows


def _same(mine: dict, ref: dict, index: int, key: str) -> str:
    """Report whether a per-species hash matches the baseline bit for bit."""
    if index not in mine or index not in ref:
        return "-"
    a, b = mine[index].get(key), ref[index].get(key)
    if a is None or b is None:
        return "-"
    return "identical" if a == b else "differs"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def kernel_section(case: str, runs: list[dict], out: Path) -> list[str]:
    """Build the per-family kernel table of one case from nsys summaries."""
    profiled: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        if r.get("_kernels"):
            profiled[r["variant"]].append(r["_kernels"])
    per_variant: dict[str, dict[str, float]] = {}
    for name, items in profiled.items():
        totals: dict[str, list[float]] = defaultdict(list)
        for item in items:
            for row in item["families"]:
                totals[row["family"]].append(row["ms_per_step"])
        per_variant[name] = {family: mean(values) for family, values in totals.items()}
    names = [n for n in VARIANTS if n in per_variant]
    families = sorted(
        {f for values in per_variant.values() for f in values},
        key=lambda f: -per_variant.get(BASELINE, {}).get(f, 0.0),
    )
    rows, family_rows = [], []
    for family in families:
        base = per_variant.get(BASELINE, {}).get(family)
        row = [family]
        for name in names:
            value = per_variant[name].get(family)
            row.append(value)
            family_rows.append(
                {"case": case, "family": family, "variant": name, "ms_per_step": value}
            )
        row += [ratio(base, per_variant[n].get(family)) for n in names if n != BASELINE]
        rows.append(row)
    totals = ["GPU total"] + [sum(per_variant[n].values()) for n in names]
    base_total = totals[1 + names.index(BASELINE)] if BASELINE in names else None
    totals += [
        ratio(base_total, totals[1 + i]) for i, n in enumerate(names) if n != BASELINE
    ]
    rows.append(totals)
    headers = (
        ["family"]
        + [f"{n} ms/step" for n in names]
        + [f"{n} speedup" for n in names if n != BASELINE]
    )
    windows = sorted(
        {
            f"{item['profiled_steps']} steps"
            for items in profiled.values()
            for item in items
        }
    )
    write_csv(out / f"families_{case}.csv", family_rows)
    return [
        f"## {case} case: GPU time per step by kernel family (nsys)",
        "",
        "Family time is the summed kernel and memory-operation time in the capture window,"
        f" divided by its steps (window: {', '.join(windows)}).",
        "",
        table(headers, rows),
        "",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    batches = [b.resolve() for b in args.batches]
    out = (args.out or batches[0] / "report").resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(batches)
    if not runs:
        raise SystemExit("no result.json files found")
    order = {name: i for i, name in enumerate(VARIANTS)}
    runs.sort(
        key=lambda r: (
            case_of(r),
            r.get("steps") or 0,
            order.get(r["variant"], 99),
            str(r["_run"]),
        )
    )

    run_rows = [
        {
            "batch": r["_batch"],
            "run": r["_run"].name,
            "case": case_of(r),
            "variant": r["variant"],
            "rng": r.get("rng"),
            "deposit_tpb": r.get("deposit_tpb_override"),
            "status": r.get("status"),
            "steps": r.get("steps"),
            "full_run": r.get("full_run"),
            "ms_per_step_after_warmup": r.get("ms_per_step_after_warmup"),
            **{k: v for k, v in r.get("timing", {}).items()},
            "gpu": r.get("gpu", {}).get("name"),
            "source_sha": r.get("task", {}).get("SOURCE_SHA"),
            "error": r.get("error"),
        }
        for r in runs
    ]
    write_csv(out / "runs.csv", run_rows)
    write_csv(
        out / "segments.csv",
        [
            {
                "run": r["_run"].name,
                "case": case_of(r),
                "variant": r["variant"],
                **segment,
            }
            for r in runs
            for segment in r.get("segments", [])
        ],
    )

    summary = summarize(runs)
    summary_rows = []
    sections = ["# FBPIC numba vs Warp benchmark", ""]
    sections.append("Batches: " + ", ".join(f"`{b.name}`" for b in batches))
    sections.append("")
    headers, rows = case_table(runs)
    sections += [
        "## Cases",
        "",
        "Runs use the cases in the run columns. A production column marked 'reference, not run' lists"
        " the production campaign inputs, to show how the measured case differs from production.",
        "",
        table(headers, rows),
        "",
    ]
    for case, steps in sorted(
        {k[:2] for k in summary}, key=lambda k: (k[0], k[1] or 0)
    ):
        base = summary.get((case, steps, BASELINE), {})
        rows = []
        for name in VARIANTS:
            entry = summary.get((case, steps, name))
            if entry is None:
                continue
            summary_rows.append(
                {"case": case, "steps": steps, "variant": name, **entry}
            )
            rows.append(
                [
                    name,
                    entry["runs"],
                    entry["ms_per_step_mean"],
                    entry["ms_per_step_stdev"],
                    ratio(base.get("ms_per_step_mean"), entry["ms_per_step_mean"]),
                    entry.get("steps_s_mean"),
                    ratio(base.get("steps_s_mean"), entry.get("steps_s_mean")),
                    entry.get("setup_s_mean"),
                    entry.get("end_to_end_s_mean"),
                    ratio(
                        base.get("end_to_end_s_mean"), entry.get("end_to_end_s_mean")
                    ),
                ]
            )
        kind = (
            "full run"
            if any(
                summary[(case, steps, v)]["full_run"]
                for v in VARIANTS
                if (case, steps, v) in summary
            )
            else "fixed window"
        )
        sections += [
            f"## {case} case, {steps} steps ({kind})",
            "",
            "ms/step excludes the first 200 steps (JIT and module load).",
            "Speedup is numba time divided by variant time.",
            "",
            table(
                [
                    "variant",
                    "runs",
                    "ms/step",
                    "stdev",
                    "speedup",
                    "steps s",
                    "speedup",
                    "setup s",
                    "end-to-end s",
                    "speedup",
                ],
                rows,
            ),
            "",
        ]
    write_csv(out / "summary.csv", summary_rows)

    for case in sorted({case_of(r) for r in runs if r.get("_kernels")}):
        sections += kernel_section(case, [r for r in runs if case_of(r) == case], out)

    check = correctness(runs)
    if check:
        sections += [
            "## Correctness against numba (same step count)",
            "",
            "Rounding and atomics make counts differ slightly between runs. FP32 Hankel adds round-off.",
            "",
            table(
                [
                    "run",
                    "case",
                    "steps",
                    "species",
                    "q/m",
                    "numba count",
                    "count",
                    "rel. count diff",
                    "level hist. L1 diff",
                    "ion IDs",
                    "ion levels",
                ],
                check,
            ),
            "",
        ]

    failed = [r for r in runs if r.get("status") != "success"]
    if failed:
        sections += ["## Failed runs", ""]
        sections += [f"- `{r['_run']}`: {r.get('error')}" for r in failed]
        sections.append("")

    (out / "report.md").write_text("\n".join(sections))
    print((out / "report.md").read_text())
    print(f"\nreport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
