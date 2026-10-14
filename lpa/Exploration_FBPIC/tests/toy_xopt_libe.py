"""GPU resource-set validation with Xopt exploration of ``f = A sin(B)``."""

from __future__ import annotations

import argparse
from contextlib import chdir
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

def sine_response(amplitude: float, phase_rad: float) -> float:
    """Evaluate the deterministic response returned by one toy task."""
    return float(amplitude * np.sin(phase_rad))


def initial_design(config: dict[str, object], variables: dict[str, list[float]]) -> list[dict[str, float]]:
    """Return configured seed points or nominal plus low/high axis anchors."""
    configured_points = config.get("initial_points")
    if configured_points is not None:
        if not isinstance(configured_points, list):
            raise ValueError("initial_points must be a list of parameter mappings")
        return [{name: float(point[name]) for name in variables} for point in configured_points]
    nominal = {name: (bounds[0] + bounds[1]) / 2 for name, bounds in variables.items()}
    points = [nominal]
    for name, bounds in variables.items():
        for value in bounds:
            point = nominal.copy()
            point[name] = value
            points.append(point)
    return points


def write_timing(path: Path, event: str, elapsed_s: float, **details: object) -> None:
    """Append one manager-side Xopt timing measurement as JSON."""
    record = {"event": event, "elapsed_s": elapsed_s, **details}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def save_surrogate_snapshot(generator: Any, variables: dict[str, list[float]], path: Path) -> None:
    """Persist the trained model's posterior mean and standard deviation on a 2D grid."""
    import torch

    phase = np.linspace(*variables["phase_rad"], 100)
    amplitude = np.linspace(*variables["amplitude"], 80)
    phase_grid, amplitude_grid = np.meshgrid(phase, amplitude)
    grid = pd.DataFrame(
        {"amplitude": amplitude_grid.ravel(), "phase_rad": phase_grid.ravel()}
    )
    model = generator.train_model()
    inputs = generator.get_input_data(grid)
    with torch.no_grad():
        posterior = model.posterior(inputs)
    np.savez_compressed(
        path,
        phase_rad=phase,
        amplitude=amplitude,
        mean=posterior.mean.detach().cpu().numpy().reshape(amplitude_grid.shape),
        stddev=posterior.variance.sqrt().detach().cpu().numpy().reshape(amplitude_grid.shape),
    )


def persistent_sine_xopt(
    _: np.ndarray,
    persis_info: dict[str, Any],
    gen_specs: dict[str, Any],
    libE_info: dict[str, Any],
):
    """Run initial samples, then feed toy results into Xopt asynchronously."""
    from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
    from libensemble.tools.persistent_support import PersistentSupport
    from xopt import Evaluator, VOCS, Xopt
    from xopt.generators.bayesian import BayesianExplorationGenerator

    user = gen_specs["user"]
    variables: dict[str, list[float]] = user["variables"]
    feature = user["feature"]
    vocs = VOCS(variables=variables, objectives={feature: "EXPLORE"})
    xopt = Xopt(
        vocs=vocs,
        generator=BayesianExplorationGenerator(vocs=vocs),
        evaluator=Evaluator(function=lambda _: {}),
    )
    pending = list(user["initial_points"])
    submitted = 0
    sim_max = int(user["sim_max"])
    batch_size = int(user["batch_size"])
    timing_path = Path(user["timing_path"])
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_directory = Path(user.get("snapshot_directory", timing_path.parent / "surrogate_snapshots"))
    snapshot_directory.mkdir(parents=True, exist_ok=True)
    support = PersistentSupport(libE_info, EVAL_GEN_TAG)
    tag = None
    batch = 0

    while tag not in {STOP_TAG, PERSIS_STOP}:
        if pending:
            points = pending[:batch_size]
            pending = pending[len(points) :]
        else:
            count = min(batch_size, sim_max - submitted)
            if count <= 0:
                break
            started = time.perf_counter()
            points = xopt.generator.generate(count)
            write_timing(timing_path, "generate", time.perf_counter() - started, candidates=count)
        generated = np.zeros(len(points), dtype=gen_specs["out"])
        for index, point in enumerate(points):
            generated["amplitude"][index] = point["amplitude"]
            generated["phase_rad"][index] = point["phase_rad"]
            generated["batch"][index] = batch
        submitted += len(generated)
        tag, _, completed = support.send_recv(generated)
        if completed is not None and len(completed):
            data = pd.DataFrame({name: completed[name] for name in ("amplitude", "phase_rad", feature)})
            successful = completed["status_code"] == 0
            if successful.any():
                started = time.perf_counter()
                xopt.add_data(data.loc[successful].reset_index(drop=True))
                write_timing(
                    timing_path,
                    "add_data",
                    time.perf_counter() - started,
                    observations=int(successful.sum()),
                    total_observations=len(xopt.data),
                )
                save_surrogate_snapshot(xopt.generator, variables, snapshot_directory / f"batch_{batch:02d}.npz")
        batch += 1

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


def run_sine_task() -> None:
    """Run under MPI, recording every rank's visible GPU assignment."""
    from mpi4py import MPI

    communicator = MPI.COMM_WORLD
    rank = communicator.Get_rank()
    size = communicator.Get_size()
    payload = json.loads(Path("toy_input.json").read_text(encoding="utf-8"))
    assignment = {
        "rank": rank,
        "mpi_size": size,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    Path(f"gpu_assignment_rank_{rank}.json").write_text(json.dumps(assignment) + "\n", encoding="utf-8")
    if rank == 0:
        result = {
            "f": sine_response(float(payload["amplitude"]), float(payload["phase_rad"])),
            "amplitude": payload["amplitude"],
            "phase_rad": payload["phase_rad"],
        }
        Path("toy_result.json").write_text(json.dumps(result) + "\n", encoding="utf-8")
    communicator.Barrier()


def run_toy_simulation(H: np.ndarray, persis_info: dict[str, Any], sim_specs: dict[str, Any], libE_info: dict[str, Any]):
    """Launch one two-rank, two-GPU toy task through libEnsemble's MPIExecutor."""
    from libensemble.executors.executor import TimeoutExpired

    user = sim_specs["user"]
    run_root = Path(user["run_root"])
    output = np.zeros(len(H), dtype=sim_specs["out"])
    for index, row in enumerate(H):
        run_directory = run_root / f"sim_{int(row['sim_id']):06d}"
        run_directory.mkdir(parents=True, exist_ok=False)
        payload = {
            "amplitude": float(row["amplitude"]),
            "phase_rad": float(row["phase_rad"]),
            "batch": int(row["batch"]),
        }
        (run_directory / "toy_input.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
        started = time.perf_counter()
        with chdir(run_directory):
            task = libE_info["executor"].submit(
                app_name="toy_sine",
                app_args="--task",
                auto_assign_gpus=True,
                match_procs_to_gpus=True,
                extra_args="--cpu-bind=none",
                stdout="stdout.log",
                stderr="stderr.log",
            )
            try:
                task.wait(timeout=float(user["timeout_s"]))
            except TimeoutExpired:
                task.kill()
                output["status_code"][index] = 1
                continue
        if task.state != "FINISHED":
            output["status_code"][index] = 1
            continue
        result = json.loads((run_directory / "toy_result.json").read_text(encoding="utf-8"))
        output["f"][index] = result["f"]
        output["wall_s"][index] = time.perf_counter() - started
    return output, persis_info


def make_ensemble(config_path: str | Path):
    """Build the Perlmutter GPU resource-set toy workflow."""
    from libensemble import Ensemble
    from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
    from libensemble.executors import MPIExecutor
    from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    variables = {name: [float(value) for value in bounds] for name, bounds in config["variables"].items()}
    initial_points = initial_design(config, variables)
    settings = config["resources"]
    executor = MPIExecutor()
    executor.register_app(full_path=str(Path(__file__).resolve()), app_name="toy_sine")
    ensemble = Ensemble(parse_args=True, executor=executor)
    ensemble.libE_specs = LibeSpecs(
        comms="mpi",
        gen_on_manager=True,
        num_resource_sets=int(settings["concurrent_simulations"]),
        gpus_per_group=int(settings["gpus_per_simulation"]),
        platform="perlmutter_g",
        sim_dirs_make=False,
    )
    ensemble.sim_specs = SimSpecs(
        sim_f=run_toy_simulation,
        inputs=["sim_id", "amplitude", "phase_rad", "batch"],
        outputs=[("status_code", int), ("f", float), ("wall_s", float)],
        user={
            "run_root": str((config_path.parent / config["run_root"]).resolve()),
            "timeout_s": float(config["timeout_s"]),
        },
    )
    ensemble.gen_specs = GenSpecs(
        gen_f=persistent_sine_xopt,
        persis_in=["sim_id", "status_code", "amplitude", "phase_rad", "f"],
        out=[("amplitude", float), ("phase_rad", float), ("batch", int)],
        user={
            "variables": variables,
            "feature": "f",
            "initial_points": initial_points,
            "sim_max": int(config["max_evaluations"]),
            "batch_size": int(settings["concurrent_simulations"]),
            "timing_path": str((config_path.parent / config["run_root"] / "xopt_timing.jsonl").resolve()),
            "snapshot_directory": str((config_path.parent / config["run_root"] / "surrogate_snapshots").resolve()),
        },
    )
    ensemble.alloc_specs = AllocSpecs(
        alloc_f=only_persistent_gens,
        user={"async_return": True, "init_sample_size": len(initial_points)},
    )
    ensemble.exit_criteria = ExitCriteria(sim_max=int(config["max_evaluations"]))
    ensemble.add_random_streams()
    return ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the libEnsemble GPU sine-surrogate test")
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--task", action="store_true")
    args = parser.parse_args()
    if args.task:
        run_sine_task()
        return
    if args.config is None:
        parser.error("config is required unless --task is used")
    ensemble = make_ensemble(args.config)
    if ensemble.nworkers < ensemble.gen_specs.user["batch_size"]:
        raise RuntimeError(f"need {ensemble.gen_specs.user['batch_size']} simulation workers")
    ensemble.run()


if __name__ == "__main__":
    main()