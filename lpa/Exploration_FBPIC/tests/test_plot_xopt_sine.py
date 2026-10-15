import importlib.util
import json
from pathlib import Path


PLOT_SCRIPT = (
    Path(__file__).parent / "smoke_campaigns" / "nersc_xopt_sine_libe" / "plot_parameter_space.py"
)


def load_plot_module():
    spec = importlib.util.spec_from_file_location("plot_xopt_sine", PLOT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_results_derives_batches_and_gpu_assignments(tmp_path: Path) -> None:
    for sim_id, amplitude, phase_rad, value in [(0, 1.25, 0.0, 0.0), (5, 2.0, 1.57, 2.0)]:
        run_directory = tmp_path / f"sim_{sim_id:06d}"
        run_directory.mkdir()
        (run_directory / "toy_input.json").write_text(
            json.dumps({"amplitude": amplitude, "phase_rad": phase_rad}), encoding="utf-8"
        )
        (run_directory / "toy_result.json").write_text(json.dumps({"f": value}), encoding="utf-8")
        for rank, gpu in enumerate(("0", "1")):
            (run_directory / f"gpu_assignment_rank_{rank}.json").write_text(
                json.dumps({"rank": rank, "cuda_visible_devices": gpu}), encoding="utf-8"
            )

    results = load_plot_module().load_results(tmp_path, initial_design_size=5)

    assert list(results["sim_id"]) == [0, 5]
    assert list(results["batch"]) == [0, 1]
    assert list(results["source"]) == ["initial", "xopt"]
    assert list(results["gpu_assignments"]) == ["0, 1", "0, 1"]