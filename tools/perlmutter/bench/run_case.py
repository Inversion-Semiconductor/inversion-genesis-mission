"""Run the producer case once under the benchmark clock and write result.json.

The run executes the unchanged ``IonizationInjectionSimulation.run()`` from the
run's source snapshot (``runs/initial_sample/ionization_injection_runscript_00``).
Two methods of ``inversion_fbpic.lib.simulation.Simulation`` are wrapped:

- ``setup_simulation`` adds the variant's FBPIC options, disables hash skipping,
  and appends a ``StepClock`` diagnostic.
- ``run_simulation`` caps the step count and times stepping and the post-run
  analysis separately. It keeps one ``sim.step(N)`` call, so particles stay on
  the GPU for the whole timed window.

Timings use ``cudaDeviceSynchronize`` plus ``time.perf_counter``.
"""

from __future__ import annotations

import time

PROCESS_START = time.perf_counter()

import argparse  # noqa: E402 - the clock starts before any import
import hashlib  # noqa: E402
import importlib.metadata  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shlex  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

from bench.variants import RNG_SEED, RNG_STREAMS, VARIANTS  # noqa: E402

# Steps excluded from ms_per_step: numba JIT and Warp module loads happen here.
WARMUP_STEPS = 200
RUNSCRIPT = "ionization_injection_runscript_00"
# Production case: nominal parameters of this campaign with its resolution profile.
PRODUCTION_CAMPAIGN = "runs/production_study/campaign.yaml"
PACKAGES = (
    "numpy",
    "scipy",
    "numba",
    "cupy",
    "warp-lang",
    "mpi4py",
    "h5py",
    "lasy",
    "openPMD-viewer",
)


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)


def read_task_env(path: Path) -> dict[str, str]:
    """Parse the KEY=%q lines that launch.sh writes."""
    values = {}
    for line in path.read_text().splitlines():
        key, _, raw = line.partition("=")
        parts = shlex.split(raw)
        values[key] = parts[0] if parts else ""
    return values


def is_under(path: str | Path, root: str | Path) -> bool:
    return Path(root).resolve() in Path(path).resolve().parents


class StepClock:
    """FBPIC diagnostic that records synchronized times at fixed iterations.

    FBPIC calls ``write(iteration)`` once per PIC cycle, right after gathering.
    Two calls ``k`` apart therefore bracket exactly ``k`` complete cycles.
    The clock also starts and stops the CUDA profiler for nsys capture ranges.

    Args:
        every: Mark interval in iterations.
        window: Optional (start, stop) iterations for the profiler capture.
    """

    def __init__(self, every: int, window: tuple[int, int] | None):
        import cupy

        self._cupy = cupy
        self.every = every
        self.window = window
        self.marks: list[tuple[int, float]] = []
        self.profiled: dict | None = None
        self._profiling = False

    def write(self, iteration: int) -> None:
        mark = iteration % self.every == 0
        edge = self.window is not None and iteration in self.window
        if not (mark or edge):
            return
        self._cupy.cuda.runtime.deviceSynchronize()
        now = time.perf_counter()
        if mark:
            self.marks.append((iteration, now))
        if edge and iteration == self.window[0] and not self._profiling:
            self._start_profiler(iteration, now)
        elif edge and iteration == self.window[1] and self._profiling:
            self._stop_profiler(iteration, now)

    def flush_to_disk(self) -> None:
        """FBPIC diagnostic interface; the clock writes nothing."""

    def close(self, iteration: int) -> None:
        """Stop a capture that is still open when stepping ends."""
        if self._profiling:
            self._cupy.cuda.runtime.deviceSynchronize()
            self._stop_profiler(iteration, time.perf_counter())

    def _start_profiler(self, iteration: int, now: float) -> None:
        self._cupy.cuda.runtime.profilerStart()
        self._nvtx("push")
        self._profiling = True
        self.profiled = {"start_iteration": iteration, "start_time": now}

    def _stop_profiler(self, iteration: int, now: float) -> None:
        self._nvtx("pop")
        self._cupy.cuda.runtime.profilerStop()
        self._profiling = False
        self.profiled["stop_iteration"] = iteration
        self.profiled["steps"] = iteration - self.profiled["start_iteration"]
        self.profiled["seconds"] = now - self.profiled.pop("start_time")

    def _nvtx(self, action: str) -> None:
        try:
            if action == "push":
                self._cupy.cuda.nvtx.RangePush("profile_window")
            else:
                self._cupy.cuda.nvtx.RangePop()
        except (AttributeError, RuntimeError):
            pass

    def summary(self) -> dict:
        """Return per-segment and post-warm-up step times."""
        segments = [
            {
                "start_iteration": a,
                "stop_iteration": b,
                "ms_per_step": 1e3 * (tb - ta) / (b - a),
            }
            for (a, ta), (b, tb) in zip(self.marks, self.marks[1:])
        ]
        steady = [(i, t) for i, t in self.marks if i >= WARMUP_STEPS]
        ms_per_step = None
        if len(steady) >= 2:
            (a, ta), (b, tb) = steady[0], steady[-1]
            ms_per_step = 1e3 * (tb - ta) / (b - a)
        return {
            "clock_every": self.every,
            "warmup_steps": WARMUP_STEPS,
            "ms_per_step_after_warmup": ms_per_step,
            "steady_window": [steady[0][0], steady[-1][0]] if steady else None,
            "segments": segments,
            "profiled": self.profiled,
        }


class GpuMonitor:
    """Sample clocks, power, and memory of the job's GPU with nvidia-smi."""

    def __init__(self, path: Path, uuid: str | None, period: float = 5.0):
        self.path = path
        self.uuid = uuid
        self.period = period
        self.phase = "imports"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if self.uuid:
            self._thread.start()

    def _run(self) -> None:
        query = "clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used,utilization.gpu,clocks_throttle_reasons.active"
        with self.path.open("w", buffering=1) as stream:
            while not self._stop.is_set():
                sample = {"t": time.perf_counter() - PROCESS_START, "phase": self.phase}
                try:
                    out = subprocess.run(
                        [
                            "nvidia-smi",
                            f"--query-gpu={query}",
                            "--format=csv,noheader,nounits",
                            "-i",
                            self.uuid,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=20,
                        check=True,
                    ).stdout
                    sample["values"] = [v.strip() for v in out.split(",")]
                except (OSError, subprocess.SubprocessError) as error:
                    sample["error"] = str(error)
                stream.write(json.dumps(sample) + "\n")
                self._stop.wait(self.period)

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=30)


def to_numpy(array):
    import numpy as np

    if hasattr(array, "get"):
        array = array.get()
    return np.asarray(array)


def particle_summary(fbpic_sim) -> list[dict]:
    """Count particles and ionization levels per species, in species order."""
    import numpy as np
    from scipy.constants import e, m_e

    summary = []
    for index, species in enumerate(fbpic_sim.ptcl):
        entry = {
            "index": index,
            "charge_e": float(species.q / e),
            "mass_me": float(species.m / m_e),
            "count": int(species.Ntot),
        }
        ionizer = getattr(species, "ionizer", None)
        if ionizer is not None:
            levels = to_numpy(ionizer.ionization_level).astype(np.int64)
            entry["level_histogram"] = np.bincount(levels).tolist()
        summary.append(entry)
    return summary


def execution_settings(fbpic_sim) -> dict:
    """Read back the solver options that the variant requested."""
    hankel = []
    for transform in fbpic_sim.fld.trans:
        for channel in ("dht0", "dhtp", "dhtm"):
            buffer = getattr(getattr(transform, channel, None), "d_in", None)
            hankel.append(None if buffer is None else str(buffer.dtype))
    return {
        "use_cuda_graphs": getattr(fbpic_sim, "use_cuda_graphs", None),
        "fuse_particle_kernels": getattr(fbpic_sim, "fuse_particle_kernels", None),
        "species_fusion": [
            {
                "deposition": getattr(species, "fuse_deposition", None),
                "gather": getattr(species, "fuse_gather", None),
                "deposit_tpb": getattr(species, "deposit_tpb", None),
            }
            for species in fbpic_sim.ptcl
        ],
        "hankel_dtypes": sorted({d for d in hankel if d}),
        "ionization_rng": [
            {
                "species": index,
                "seed": (
                    None
                    if getattr(s.ionizer, "rng_seed", None) is None
                    else int(s.ionizer.rng_seed)
                ),
                "stream": (
                    None
                    if getattr(s.ionizer, "rng_stream", None) is None
                    else int(s.ionizer.rng_stream)
                ),
            }
            for index, s in enumerate(fbpic_sim.ptcl)
            if getattr(s, "ionizer", None) is not None
        ],
    }


def gpu_identity() -> dict:
    import cupy

    properties = cupy.cuda.runtime.getDeviceProperties(0)
    uuid = properties.get("uuid")
    info = {
        "name": properties["name"].decode(),
        "uuid": None,
        "driver": cupy.cuda.runtime.driverGetVersion(),
        "runtime": cupy.cuda.runtime.runtimeGetVersion(),
        "memory_bytes": int(properties["totalGlobalMem"]),
    }
    if isinstance(uuid, bytes) and len(uuid) == 16:
        h = uuid.hex()
        info["uuid"] = f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
    if info["uuid"]:
        try:
            info["nvidia_smi"] = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit,ecc.mode.current",
                    "--format=csv,noheader",
                    "-i",
                    info["uuid"],
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            info["nvidia_smi_error"] = str(error)
    return info


def build_case(name: str, task: dict[str, str], out: Path) -> dict:
    """Return the template inputs of the reduced or the production case.

    The reduced case is runs/initial_sample/ionization_injection_runscript_00.py.
    The production case is the nominal point of the production campaign, with the
    resolution profile that the campaign selects, as exploration_fbpic builds it.
    """
    if name == "reduced":
        runscript = __import__(RUNSCRIPT)
        hyperparameters = dict(runscript.HYPERPARAMETERS)
        physical = dict(runscript.PHYSICAL_PARAMETERS)
        source = f"{RUNSCRIPT}.py"
    else:
        from exploration_fbpic.campaign import CampaignConfig

        campaign = Path(task["EXPLORATION_DIR"]) / PRODUCTION_CAMPAIGN
        config = CampaignConfig.from_file(campaign)
        if not is_under(config.template_script, task["PRODUCER_DIR"]):
            raise RuntimeError(
                f"campaign template {config.template_script} is outside the snapshot"
            )
        hyperparameters = dict(config.execution["hyperparameters"])
        physical = config.complete_parameters(next(iter(config.stages)))
        source = f"{PRODUCTION_CAMPAIGN} (profile {config.execution.get('resolution_profile')})"
    hyperparameters["lasy_file"] = out / "laser" / "experimental_laser"
    hyperparameters["lab_diagnostic_directory"] = out / "lab_diags"
    hyperparameters["results_directory"] = out / "results"
    (out / "laser").mkdir(parents=True, exist_ok=True)
    return {
        "name": name,
        "source": source,
        "hyperparameters": hyperparameters,
        "physical_parameters": physical,
    }


def production_reference(task: dict[str, str]) -> dict | None:
    """Return the production case inputs for the report, without running them."""
    try:
        from exploration_fbpic.campaign import CampaignConfig

        config = CampaignConfig.from_file(
            Path(task["EXPLORATION_DIR"]) / PRODUCTION_CAMPAIGN
        )
        return {
            "name": "production",
            "source": f"{PRODUCTION_CAMPAIGN} (profile {config.execution.get('resolution_profile')})",
            "hyperparameters": dict(config.execution["hyperparameters"]),
            "physical_parameters": config.complete_parameters(
                next(iter(config.stages))
            ),
        }
    except Exception as error:  # noqa: BLE001 - the reference is informational
        return {"error": repr(error)}


def ion_states(fbpic_sim) -> list[dict]:
    """Hash ion IDs and ionization levels in ID order, for bit-for-bit checks."""
    import numpy as np

    states = []
    for index, species in enumerate(fbpic_sim.ptcl):
        ionizer = getattr(species, "ionizer", None)
        if ionizer is None:
            continue
        levels = to_numpy(ionizer.ionization_level).astype(np.uint8)
        entry = {"index": index, "count": int(species.Ntot)}
        tracker = getattr(species, "tracker", None)
        if tracker is not None:
            ids = to_numpy(tracker.id).astype(np.uint64)
            order = np.argsort(ids, kind="stable")
            entry["ids_sha256"] = hashlib.sha256(ids[order].tobytes()).hexdigest()
            entry["levels_sha256"] = hashlib.sha256(levels[order].tobytes()).hexdigest()
        states.append(entry)
    return states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument(
        "--steps",
        default="0",
        help="N steps, 0 for the full interaction, or 1/K for a fraction of it",
    )
    parser.add_argument("--case", choices=("reduced", "production"), default="reduced")
    parser.add_argument(
        "--rng",
        choices=("shared", "original"),
        default="shared",
        help="shared: ID-based Philox draws (identical in both solvers)",
    )
    parser.add_argument("--clock-every", type=int, default=100)
    parser.add_argument(
        "--profile-window",
        help="A:B steps for nsys capture; B may be 'end'; negative A counts from the end",
    )
    args = parser.parse_args()
    if args.steps.startswith("1/"):
        args.fraction, args.steps = int(args.steps[2:]), 0
    else:
        args.fraction, args.steps = None, int(args.steps)
    if args.steps < 0 or (args.fraction is not None and args.fraction < 1):
        parser.error("--steps must be N >= 0 or 1/K with K >= 1")
    if args.profile_window:
        start, _, stop = args.profile_window.partition(":")
        args.profile_window = (int(start), sys.maxsize if stop == "end" else int(stop))
        if not args.profile_window[0] < args.profile_window[1]:
            parser.error("--profile-window needs A < B")
    return args


def main() -> int:
    args = parse_args()
    run = args.run_dir.resolve()
    variant = VARIANTS[args.variant]
    task = read_task_env(run / "meta" / "task.env")
    out = run / "out"
    result: dict = {
        "status": "running",
        "variant": variant.name,
        "engine": variant.engine,
        "simulation_kwargs": variant.simulation_kwargs,
        "deposit_tpb_override": variant.deposit_tpb,
        "rng": args.rng,
        "requested_steps": (
            f"1/{args.fraction}" if args.fraction else args.steps or None
        ),
        "task": task,
        "command": sys.argv,
        "python": sys.executable,
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "timing": {},
    }
    write_json(run / "result.json", result)
    timing = result["timing"]
    monitor = None
    clock = None
    state: dict = {}
    try:
        import cupy

        import fbpic

        result["fbpic_version"] = fbpic.__version__
        result["fbpic_file"] = str(Path(fbpic.__file__).resolve())
        if not is_under(fbpic.__file__, task["SOLVER_DIR"]):
            raise RuntimeError(
                f"fbpic imported from {fbpic.__file__}, not {task['SOLVER_DIR']}"
            )
        if ("+warp" in fbpic.__version__) != (variant.engine == "warp"):
            raise RuntimeError(
                f"fbpic {fbpic.__version__} does not match engine {variant.engine}"
            )

        from inversion_fbpic.lib import simulation as sm

        if not is_under(sm.__file__, task["PRODUCER_DIR"]):
            raise RuntimeError(f"inversion_fbpic imported from {sm.__file__}")
        from fbpic.particles import Particles
        from ionization_injection_template import IonizationInjectionSimulation

        timing["imports_s"] = time.perf_counter() - PROCESS_START
        gpu = gpu_identity()
        result["gpu"] = gpu
        monitor = GpuMonitor(run / "resources.jsonl", gpu["uuid"])
        monitor.start()

        case = build_case(args.case, task, out)
        result["case"] = case
        if args.case != "production":
            result["production_reference"] = production_reference(task)
        hyperparameters = case["hyperparameters"]
        physical = case["physical_parameters"]

        sync = cupy.cuda.runtime.deviceSynchronize
        original_setup = sm.Simulation.setup_simulation
        original_run = sm.Simulation.run_simulation
        original_fbpic = sm.FBPICSimulation
        original_ionizable = Particles.make_ionizable
        if (
            args.rng == "shared"
            and "rng_seed" not in inspect.signature(original_ionizable).parameters
        ):
            raise RuntimeError(
                f"{fbpic.__file__} lacks ID-based ionization draws (rng_seed); "
                "use the patched upstream commit or --rng original"
            )

        def fbpic_simulation(*args_, **kwargs):
            # The template builds the FBPIC Simulation here; inject the variant options.
            kwargs.update(variant.simulation_kwargs)
            state["fbpic"] = original_fbpic(*args_, **kwargs)
            return state["fbpic"]

        def make_ionizable(species, element, *args_, **kwargs):
            if args.rng == "shared":
                if element not in RNG_STREAMS:
                    raise RuntimeError(f"no RNG stream for element {element!r}")
                if species.tracker is None:
                    species.track(state["fbpic"].comm)
                kwargs.update(rng_seed=RNG_SEED, rng_stream=RNG_STREAMS[element])
            return original_ionizable(species, element, *args_, **kwargs)

        def setup_simulation(
            self, working_directory=None, skip_if_hashed=True, **kwargs
        ):
            nonlocal clock
            sync()
            start = time.perf_counter()
            original_setup(self, working_directory, skip_if_hashed=False, **kwargs)
            sync()
            timing["setup_s"] = time.perf_counter() - start
            clock = StepClock(args.clock_every, args.profile_window)
            self.simulation.diags.append(clock)
            state["sim"] = self
            if variant.deposit_tpb is not None:
                for species in self.simulation.ptcl:
                    if species.particle_shape == "linear":
                        species.deposit_tpb = variant.deposit_tpb
            settings = execution_settings(self.simulation)
            result["execution"] = settings
            expected = {"seed": RNG_SEED if args.rng == "shared" else None}
            for entry in settings["ionization_rng"]:
                if entry["seed"] != expected["seed"]:
                    raise RuntimeError(
                        f"ionization RNG is {entry}, expected {args.rng}"
                    )
            if variant.engine == "warp":
                if (
                    settings["use_cuda_graphs"]
                    is not variant.simulation_kwargs["use_cuda_graphs"]
                ):
                    raise RuntimeError(
                        f"use_cuda_graphs is {settings['use_cuda_graphs']}"
                    )
                if settings["hankel_dtypes"] != [variant.hankel_dtype]:
                    raise RuntimeError(
                        f"Hankel buffers are {settings['hankel_dtypes']}"
                    )
            result["status"] = "stepping"
            write_json(run / "result.json", result)

        def run_simulation(
            self,
            show_progress=True,
            logger_friendly_progress=False,
            record_hash=True,
            skip_if_hashed=True,
        ):
            fbpic_sim = self.simulation
            full_steps = self.num_steps
            if args.fraction:
                steps = full_steps // args.fraction
            else:
                steps = full_steps if not args.steps else min(args.steps, full_steps)
            first = fbpic_sim.iteration
            if clock.window is not None:
                start, stop = clock.window
                if start < 0:
                    start = max(0, steps + start)
                stop = min(stop, steps - 1)
                if start >= stop:
                    raise RuntimeError(
                        f"profile window {start}:{stop} is empty for {steps} steps"
                    )
                clock.window = (first + start, first + stop)
            result.update(
                full_steps=full_steps,
                steps=steps,
                full_run=steps == full_steps,
                dt_s=fbpic_sim.dt,
                initial_particles=particle_summary(fbpic_sim),
            )
            monitor and setattr(monitor, "phase", "steps")
            sync()
            start = time.perf_counter()
            fbpic_sim.step(steps, show_progress=False)
            sync()
            timing["steps_s"] = time.perf_counter() - start
            clock.close(fbpic_sim.iteration)
            if fbpic_sim.iteration - first != steps:
                raise RuntimeError(
                    f"ran {fbpic_sim.iteration - first} of {steps} steps"
                )
            if record_hash and self.do_write_to_disk:
                self._save_hash()
            monitor and setattr(monitor, "phase", "analysis")
            start = time.perf_counter()
            self._analyze_diagnostics(before_simulation=False)
            timing["analysis_s"] = time.perf_counter() - start

        sm.Simulation.setup_simulation = setup_simulation
        sm.Simulation.run_simulation = run_simulation
        sm.FBPICSimulation = fbpic_simulation
        Particles.make_ionizable = make_ionizable
        try:
            monitor.phase = "setup"
            start = time.perf_counter()
            IonizationInjectionSimulation(physical, hyperparameters).run()
            sync()
            timing["producer_run_s"] = time.perf_counter() - start
        finally:
            sm.Simulation.setup_simulation = original_setup
            sm.Simulation.run_simulation = original_run
            sm.FBPICSimulation = original_fbpic
            Particles.make_ionizable = original_ionizable
        timing["end_to_end_s"] = time.perf_counter() - PROCESS_START

        result.update(clock.summary())
        fbpic_sim = state["sim"].simulation
        result["final_iteration"] = fbpic_sim.iteration
        result["final_particles"] = particle_summary(fbpic_sim)
        result["final_ion_states"] = ion_states(fbpic_sim)
        result["status"] = "success"
    except BaseException as error:  # noqa: BLE001 - keep failure evidence
        result.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        traceback.print_exc()
    finally:
        if monitor is not None:
            monitor.stop()
        if clock is not None and "segments" not in result:
            result.update(clock.summary())
        result["packages"] = {}
        for name in PACKAGES:
            try:
                result["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result["packages"][name] = None
        result["wall_s"] = time.perf_counter() - PROCESS_START
        write_json(run / "result.json", result)

    brief = {
        k: result.get(k)
        for k in (
            "status",
            "variant",
            "steps",
            "full_steps",
            "ms_per_step_after_warmup",
            "timing",
            "error",
        )
    }
    print(json.dumps(brief, indent=2), flush=True)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
