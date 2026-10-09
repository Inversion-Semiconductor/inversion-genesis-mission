from pathlib import Path

from exploration_fbpic.libensemble_driver import make_ensemble, next_run_id_offset


CONFIG = Path(__file__).parents[1] / "runs" / "nersc_smoke" / "campaign.yaml"


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


def test_next_run_id_offset_skips_existing_simulation_directories(tmp_path: Path) -> None:
    (tmp_path / "sim_000003").mkdir()
    (tmp_path / "sim_000011").mkdir()
    (tmp_path / "smoke_000").mkdir()

    assert next_run_id_offset(tmp_path) == 12