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
        "exploration_fbpic.libensemble_adapter.FBPICRunExecutor._write_result",
        lambda *args: None,
    )
    store = type("Store", (), {"append": lambda *_: None})()
    mpi_executor = RecordingExecutor()
    history = np.array([(0, 24e-6)], dtype=[("sim_id", int), ("laser_spot_size_m", float)])

    run_fbpic_simulation(
        history,
        {},
        {
            "out": [("status_code", int), ("total_beam_charge_pc", float)],
            "user": {
                "executor": campaign_executor,
                "store": store,
                "stage": "smoke",
                "active_parameter_names": ("laser_spot_size_m",),
            },
        },
        {"executor": mpi_executor},
    )

    assert mpi_executor.submit_kwargs["auto_assign_gpus"] is True
    assert mpi_executor.submit_kwargs["match_procs_to_gpus"] is True
    assert mpi_executor.submit_kwargs["extra_args"] == "--cpu-bind=none"
    assert "num_procs" not in mpi_executor.submit_kwargs
    assert "num_gpus" not in mpi_executor.submit_kwargs