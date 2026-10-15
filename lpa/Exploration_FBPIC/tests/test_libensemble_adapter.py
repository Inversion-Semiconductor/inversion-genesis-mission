from pathlib import Path

import numpy as np

from exploration_fbpic.executor import FBPICRunExecutor
from exploration_fbpic.libensemble_adapter import run_fbpic_simulation


class FinishedTask:
    state = "FINISHED"
    errcode = 0

    def wait(self, timeout: float) -> None:
        return None


class RecordingExecutor:
    def __init__(self) -> None:
        self.submit_kwargs: dict[str, object] = {}

    def submit(self, **kwargs: object) -> FinishedTask:
        self.submit_kwargs = kwargs
        return FinishedTask()


def adapter_inputs(campaign_executor, store, mpi_executor):
    history = np.array([(0, 24e-6)], dtype=[("sim_id", int), ("laser_spot_size_m", float)])
    sim_specs = {
        "out": [
            ("status_code", int),
            ("total_beam_charge_pc", float),
            ("mean_kinetic_energy_MeV", float),
        ],
        "user": {
            "executor": campaign_executor,
            "store": store,
            "stage": "smoke",
            "active_parameter_names": ("laser_spot_size_m",),
                "log_scaled_names": (),
                "linear_scales": {"laser_spot_size_m": 1.0},
        },
    }
    return history, sim_specs, {"executor": mpi_executor}


def test_adapter_uses_resource_set_gpu_matching_without_explicit_rank_count(
    tmp_path: Path, monkeypatch
) -> None:
    run_directory = tmp_path / "sim_000000"
    run_directory.mkdir()
    manifest = object()
    campaign_executor = object.__new__(FBPICRunExecutor)
    campaign_executor.prepare_run = lambda *_: (run_directory, manifest)  # type: ignore[attr-defined]
    campaign_executor.config = type("Config", (), {"execution": {"timeout_s": 1}})()
    campaign_executor.extract_features = lambda _: {"total_beam_charge_pc": 1.0}  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "exploration_fbpic.libensemble_adapter.FBPICRunExecutor.finalize",
        lambda *args: None,
    )
    store = type("Store", (), {"append": lambda *_: None})()
    mpi_executor = RecordingExecutor()
    history, sim_specs, libe_info = adapter_inputs(campaign_executor, store, mpi_executor)

    out, _ = run_fbpic_simulation(history, {}, sim_specs, libe_info)

    assert mpi_executor.submit_kwargs["auto_assign_gpus"] is True
    assert mpi_executor.submit_kwargs["match_procs_to_gpus"] is True
    assert mpi_executor.submit_kwargs["extra_args"] == "--cpu-bind=none"
    assert "num_procs" not in mpi_executor.submit_kwargs
    assert "num_gpus" not in mpi_executor.submit_kwargs
    assert out["status_code"][0] == 0
    assert out["total_beam_charge_pc"][0] == 1.0
    assert np.isnan(out["mean_kinetic_energy_MeV"][0])


def test_adapter_records_unexpected_extraction_error(tmp_path: Path, monkeypatch) -> None:
    run_directory = tmp_path / "sim_000000"
    run_directory.mkdir()
    manifest = object()
    campaign_executor = object.__new__(FBPICRunExecutor)
    campaign_executor.prepare_run = lambda *_: (run_directory, manifest)  # type: ignore[attr-defined]
    campaign_executor.config = type("Config", (), {"execution": {"timeout_s": 1}})()
    campaign_executor.extract_features = lambda _: (_ for _ in ()).throw(KeyError("nitrogen_electrons"))  # type: ignore[attr-defined]
    written = []
    monkeypatch.setattr(
        "exploration_fbpic.libensemble_adapter.FBPICRunExecutor.finalize",
        lambda *args: written.append(args[-1]),
    )
    stored = []
    store = type("Store", (), {"append": lambda _, result: stored.append(result)})()
    mpi_executor = RecordingExecutor()
    history, sim_specs, libe_info = adapter_inputs(campaign_executor, store, mpi_executor)

    out, _ = run_fbpic_simulation(history, {}, sim_specs, libe_info)

    assert out["status_code"][0] == 1
    assert np.isnan(out["total_beam_charge_pc"][0])
    assert written[0].status == "error"
    assert "KeyError" in written[0].message
    assert stored[0] is written[0]