from pathlib import Path
import json

import numpy as np
import pytest
from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG, STOP_TAG

from exploration_fbpic.campaign import Campaign
from exploration_fbpic.exploration import XoptFBPICExplorer
from exploration_fbpic.libensemble_driver import fallback_design, make_ensemble, next_run_id_offset, persistent_xopt
from exploration_fbpic.store import CampaignStore
from exploration_fbpic.executor import EvaluationResult


CONFIG = Path(__file__).parent / "smoke_campaigns" / "nersc_smoke" / "campaign.yaml"
SURROGATE_CONFIG = Path(__file__).parent / "smoke_campaigns" / "nersc_surrogate_smoke" / "campaign.yaml"


def test_smoke_driver_assigns_eight_two_gpu_resource_sets() -> None:
    ensemble = make_ensemble(CONFIG, "smoke")

    assert ensemble.libE_specs.num_resource_sets == 8
    assert ensemble.libE_specs.gpus_per_group == 2
    assert ensemble.libE_specs.platform == "perlmutter_g"
    assert ensemble.sim_specs.inputs == [
        "sim_id",
        "laser_spot_size_m",
        "laser_focal_position_m",
    ]
    assert {field[0] for field in ensemble.sim_specs.outputs} == {
        "status_code",
        "total_beam_charge_pc",
        "mean_kinetic_energy_MeV",
    }


def test_make_ensemble_validates_training_data_before_allocation(monkeypatch) -> None:
    def fail_preflight(_: XoptFBPICExplorer) -> object:
        raise ValueError("invalid reverse-stage history")

    monkeypatch.setattr(XoptFBPICExplorer, "training_data", fail_preflight)

    with pytest.raises(ValueError, match="invalid reverse-stage history"):
        make_ensemble(CONFIG, "smoke")


def test_next_run_id_offset_skips_existing_simulation_directories(tmp_path: Path) -> None:
    (tmp_path / "sim_000003").mkdir()
    (tmp_path / "sim_000011").mkdir()
    (tmp_path / "smoke_000").mkdir()

    assert next_run_id_offset(tmp_path) == 12


def test_surrogate_smoke_bootstraps_with_six_initial_points() -> None:
    ensemble = make_ensemble(SURROGATE_CONFIG, "surrogate_smoke")

    assert ensemble.exit_criteria.sim_max == 18
    assert len(ensemble.gen_specs.user["initial_points"]) == 6
    assert ensemble.alloc_specs.user["init_sample_size"] == 6
    assert ensemble.alloc_specs.user["async_return"] is False
    assert "xopt" not in ensemble.gen_specs.user
    assert ensemble.gen_specs.user["config_path"] == str(SURROGATE_CONFIG.resolve())
    assert ensemble.gen_specs.user["stage"] == "surrogate_smoke"


def test_surrogate_smoke_uses_full_physics_seed_validation() -> None:
    campaign = make_ensemble(SURROGATE_CONFIG, "surrogate_smoke").sim_specs.user["executor"].campaign
    parameters = campaign.config.parameters
    hyperparameters = campaign.config.execution["hyperparameters"]

    assert parameters["density_alpha_m"].nominal == 1.1e-3
    assert parameters["density_center_location_m"].nominal == 2.3e-3
    assert parameters["laser_focal_position_m"].nominal == 3.5e-3
    assert parameters["laser_focal_position_m"].lower == 2.5e-3
    assert parameters["laser_focal_position_m"].upper == 4.5e-3
    assert hyperparameters["gamma_boost"] == 3.0
    assert hyperparameters["nz"] == 3000
    assert hyperparameters["number_dumps_lab"] == 3
    assert hyperparameters["zmin"] == -150e-6


def test_persistent_driver_accepts_empty_final_result(monkeypatch, tmp_path: Path) -> None:
    class StopAfterSend:
        def __init__(self, *_: object) -> None:
            pass

        def send_recv(self, _: object) -> tuple[int, None, None]:
            return STOP_TAG, None, None

    monkeypatch.setattr("libensemble.tools.persistent_support.PersistentSupport", StopAfterSend)
    user = {
        "config_path": str(SURROGATE_CONFIG),
        "stage": "surrogate_smoke",
        "variable_names": ("laser_spot_size_m", "laser_focal_position_m"),
        "feature_names": ("total_beam_charge_pc", "mean_kinetic_energy_MeV"),
        "initial_points": [{"laser_spot_size_m": 24e-6, "laser_focal_position_m": 3.5e-3}],
        "sim_max": 1,
        "batch_size": 1,
    }

    _, _, tag = persistent_xopt(
        np.empty(0),
        {},
        {"user": user, "out": [("laser_spot_size_m", float), ("laser_focal_position_m", float)]},
        {"comm": object()},
    )

    assert tag == FINISHED_PERSISTENT_GEN_TAG


def test_fallback_design_advances_past_initial_anchors() -> None:
    explorer = XoptFBPICExplorer(Campaign.from_file(SURROGATE_CONFIG), "surrogate_smoke")

    points = fallback_design(explorer, offset=6, count=6)

    assert len(points) == 6
    assert all(point not in explorer.initial_design(6) for point in points)


def test_continuation_uses_only_compatible_stored_evaluations(tmp_path: Path) -> None:
    source = SURROGATE_CONFIG.read_text(encoding="utf-8")
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        source.replace(
            "template_script: ../../../../Simulation_FBPIC/runs/initial_sample/ionization_injection_template.py",
            "template_script: " + str((SURROGATE_CONFIG.parents[4] / "Simulation_FBPIC/runs/initial_sample/ionization_injection_template.py").resolve()),
        )
        .replace("run_root: /pscratch/sd/c/cedoss/surrogate_smoke_test", f"run_root: {tmp_path / 'runs'}")
        .replace("continuation: false", "continuation: true"),
        encoding="utf-8",
    )
    campaign = Campaign.from_file(config_path)
    run_root = campaign.config.run_root
    store = CampaignStore(run_root)
    for run_id, stage, fingerprint in [
        ("sim_000000", "surrogate_smoke", campaign.config.fingerprint()),
        ("sim_000001", "other_stage", campaign.config.fingerprint()),
        ("sim_000002", "surrogate_smoke", "different-fingerprint"),
    ]:
        directory = run_root / run_id
        directory.mkdir(parents=True)
        manifest = campaign.make_manifest(run_id, "surrogate_smoke")
        manifest.write(directory)
        manifest_path = directory / "run_manifest.json"
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        record["stage"] = stage
        record["campaign_fingerprint"] = fingerprint
        manifest_path.write_text(json.dumps(record), encoding="utf-8")
        store.append(EvaluationResult(run_id, "ok", {"total_beam_charge_pc": 1.0, "mean_kinetic_energy_MeV": 1.0}, directory))

    ensemble = make_ensemble(config_path, "surrogate_smoke")

    assert ensemble.exit_criteria.sim_max == 17
    assert ensemble.gen_specs.user["initial_points"] == []