import importlib.util
from pathlib import Path

WORKFLOW_PATH = (
    Path(__file__).parents[1] / "runs" / "initial_study" / "one_offs" / "run_one_off.py"
)
CANDIDATE = WORKFLOW_PATH.parent / "candidate_mock_stage2_016.json"


def _workflow_module():
    specification = importlib.util.spec_from_file_location(
        "one_off_workflow", WORKFLOW_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_one_off_candidate_prepares_normal_stage_two_run(tmp_path: Path) -> None:
    workflow = _workflow_module()

    run_directory = workflow.prepare(CANDIDATE, tmp_path, "mock_stage2_016")

    manifest = (run_directory / "run_manifest.json").read_text(encoding="utf-8")
    assert run_directory == tmp_path / "mock_stage2_016"
    assert '"stage": "stage_2"' in manifest
    assert '"laser_energy_J": 2.344387555025423' in manifest
    assert (run_directory / "run_fbpic.py").is_file()
