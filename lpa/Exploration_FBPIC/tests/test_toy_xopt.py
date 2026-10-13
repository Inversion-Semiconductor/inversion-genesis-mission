from pathlib import Path
import importlib.util

import numpy as np
import pytest
from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG, STOP_TAG

LIBE_CONFIG = Path(__file__).parent / "smoke_campaigns" / "nersc_xopt_sine_libe" / "campaign.yaml"
TOY_MODULE_PATH = Path(__file__).parent / "toy_xopt_libe.py"


def load_toy_module():
    spec = importlib.util.spec_from_file_location("toy_xopt_libe", TOY_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sine_response() -> None:
    assert load_toy_module().sine_response(2.0, 1.5707963267948966) == pytest.approx(2.0)


def test_libe_toy_assigns_eight_two_gpu_resource_sets() -> None:
    ensemble = load_toy_module().make_ensemble(LIBE_CONFIG)

    assert ensemble.libE_specs.num_resource_sets == 8
    assert ensemble.libE_specs.gpus_per_group == 2
    assert ensemble.libE_specs.platform == "perlmutter_g"
    assert ensemble.sim_specs.inputs == ["sim_id", "amplitude", "phase_rad", "batch"]
    assert ensemble.exit_criteria.sim_max == 53
    assert ensemble.gen_specs.user["timing_path"].endswith("xopt_timing.jsonl")


def test_persistent_libe_toy_accepts_empty_final_result(monkeypatch, tmp_path: Path) -> None:
    class StopAfterSend:
        def __init__(self, *_: object) -> None:
            pass

        def send_recv(self, _: object) -> tuple[int, None, None]:
            return STOP_TAG, None, None

    monkeypatch.setattr("libensemble.tools.persistent_support.PersistentSupport", StopAfterSend)
    config = {
        "variables": {"amplitude": [0.5, 2.0], "phase_rad": [-3.141592653589793, 3.141592653589793]},
        "feature": "f",
        "initial_points": [{"amplitude": 1.25, "phase_rad": 0.0}],
        "sim_max": 1,
        "batch_size": 1,
        "timing_path": str(tmp_path / "timing.jsonl"),
        "snapshot_directory": str(tmp_path / "surrogate_snapshots"),
    }
    out = [("amplitude", float), ("phase_rad", float), ("batch", int)]

    _, _, tag = load_toy_module().persistent_sine_xopt(
        np.empty(0),
        {},
        {"user": config, "out": out},
        {"comm": object()},
    )

    assert tag == FINISHED_PERSISTENT_GEN_TAG