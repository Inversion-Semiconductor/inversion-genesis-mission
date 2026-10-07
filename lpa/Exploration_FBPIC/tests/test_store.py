from pathlib import Path

from exploration_fbpic.campaign import Campaign
from exploration_fbpic.executor import EvaluationResult
from exploration_fbpic.store import CampaignStore


CONFIG = Path(__file__).parents[1] / "runs" / "initial_study" / "campaign.yaml"


def test_store_persists_full_parameter_vector(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    run_directory = tmp_path / "sim_000001"
    run_directory.mkdir()
    campaign.make_manifest("sim_000001", "stage_1").write(run_directory)
    result = EvaluationResult("sim_000001", "ok", {"total_beam_charge_pc": 5.0}, run_directory)

    store = CampaignStore(tmp_path)
    store.append(result)

    row = store.load().iloc[0]
    assert row["p:laser_energy_J"] == 2.5
    assert row["p:zernike_secondary_trefoil_x"] == 0.0
    assert row["f:total_beam_charge_pc"] == 5.0
