from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import yaml

from fludat_fit.cli_common import FitWindow
from fludat_fit.dataset import NozzleDataset
from fludat_fit.explore_fits import FitExplorer, main
from fludat_fit.families import DEFAULT_FAMILY_NAMES, get_families
from fludat_fit.fitting import MultiStartLocalFit
from fludat_fit.lineout import LineoutConditions

FAMILIES = ["conical:supergaussian", "asymmetric_sine"]


def test_list_families(capsys):
    assert main(["--list-families"]) == 0
    assert capsys.readouterr().out.split() == list(DEFAULT_FAMILY_NAMES)


def test_explorer_sliders_fit_and_save(small_cube_path, tmp_path, capsys):
    dataset = NozzleDataset.load(small_cube_path)
    scheme = MultiStartLocalFit(n_starts=1)
    explorer = FitExplorer(
        dataset,
        get_families(FAMILIES),
        scheme,
        FitWindow(max_points=200),
        conditions=LineoutConditions(1.0, 10.0, 0.0),
        angle_range=(-30.0, 30.0),
        output_dir=tmp_path / "fits",
        species="H",
    )
    assert len(explorer.sliders) == 3
    assert explorer.angle_range == (-30.0, 30.0)
    assert explorer.conditions == LineoutConditions(1.0, 10.0, 0.0)
    assert explorer.ax_res.get_xlabel() == "z [mm]"
    assert explorer.lineout is not None and explorer.lineout.n_points <= 200
    assert explorer.comparison is None
    assert "press Fit" in explorer.status.get_text()

    comparison = explorer.fit()
    assert comparison is not None
    assert comparison.best.family_name == "conical:supergaussian"
    assert explorer.comparison is comparison
    assert set(scheme.warm_starts) == set(FAMILIES)
    assert len(explorer.model_lines[0].get_xdata()) == 1000
    assert len(explorer.model_lines[2].get_xdata()) == 0  # only two families fitted
    assert "conical:supergaussian" in explorer.status.get_text()
    assert "# best: conical:supergaussian" in capsys.readouterr().out

    # Moving the angle slider re-extracts an oblique lineout and clears the stale fit.
    axial_peak = explorer.lineout.peak
    explorer.angle_slider.set_val(20.0)
    assert explorer.conditions.angle_deg == 20.0
    assert explorer.comparison is None
    assert len(explorer.model_lines[0].get_xdata()) == 0
    assert explorer.ax_res.get_xlabel().startswith("t along")
    assert explorer.lineout.peak != axial_peak  # density grows with x along the tilt

    # Save fits first when nothing is fitted yet.
    paths = explorer.save()
    assert [path.suffix for path in paths] == [".json", ".yaml"]
    assert all(path.parent == tmp_path / "fits" for path in paths)
    payload = json.loads(paths[0].read_text())
    assert payload["conditions"] == {
        "x_mm": 1.0,
        "pressure_bar": 10.0,
        "angle_deg": 20.0,
    }
    assert payload["interp_from_h5"]["lineout_axis"]["x_mm"]["origin"] == 1.0
    assert payload["interp_from_h5"]["interpolation_points"] == {"pressure_bar": 10.0}
    config = yaml.safe_load(paths[1].read_text())
    assert config["subclass"] == "generic_conical_target"
    assert config["parameters"]["species"] == "H"
    assert "a20deg" in paths[1].name


def test_live_mode_refits_on_slider_change(small_cube_path):
    dataset = NozzleDataset.load(small_cube_path)
    explorer = FitExplorer(
        dataset,
        get_families(["asymmetric_sine"]),
        MultiStartLocalFit(n_starts=1),
        FitWindow(max_points=100),
        live=True,
    )
    assert explorer.conditions.angle_deg == 0.0
    explorer.pressure_slider.set_val(7.5)
    assert explorer.comparison is not None
    assert explorer.comparison.lineout.conditions.pressure_bar == 7.5


def test_explorer_handles_empty_lineouts(small_cube_path, monkeypatch):
    dataset = NozzleDataset.load(small_cube_path)
    explorer = FitExplorer(
        dataset, get_families(["asymmetric_sine"]), MultiStartLocalFit(n_starts=1)
    )
    real = dataset.lineout

    def vacuum(conditions, t_values=None):
        lineout = real(conditions, t_values)
        return replace(lineout, density=np.zeros_like(lineout.density))

    monkeypatch.setattr(dataset, "lineout", vacuum)
    explorer.set_conditions(LineoutConditions(2.0, 10.0))
    assert explorer.lineout.peak == 0.0
    assert "no density" in explorer.status.get_text()
    assert explorer.fit() is None
    assert explorer.save() == []


def test_static_output(small_cube_path, tmp_path, capsys):
    output = tmp_path / "plots" / "explorer.png"
    code = main(
        [
            str(small_cube_path),
            "--x",
            "1",
            "--pressure",
            "10",
            "--angle",
            "5",
            "--families",
            *FAMILIES,
            "--starts",
            "1",
            "--max-points",
            "150",
            "-o",
            str(output),
        ]
    )
    assert code == 0
    assert output.is_file()
    assert "angle=5 deg" in capsys.readouterr().out


def test_requires_a_cube_and_a_sane_angle_range(small_cube_path):
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(ValueError, match="angle_range"):
        FitExplorer(
            NozzleDataset.load(small_cube_path),
            get_families(["asymmetric_sine"]),
            MultiStartLocalFit(n_starts=1),
            angle_range=(95.0, 100.0),
        )
