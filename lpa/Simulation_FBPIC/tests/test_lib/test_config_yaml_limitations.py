"""
Advanced and known-limitation tests for YAML comment handling.

These cases are fragile or unsupported by design. Tests document **current**
behavior so changes are visible; some are marked ``xfail`` where behavior is
known incorrect but not yet fixed.

See ``test_config_yaml.py`` for the main edge-case catalog and regression
suite. Run all config/YAML tests with::

    pytest tests/test_lib/test_config_yaml.py \\
           tests/test_lib/test_config_yaml_limitations.py \\
           tests/test_lib/test_serializable_config.py -v

Known limitations (not yet fixed)
---------------------------------

* **Hash in unquoted scalars** — ``#`` inside a bare scalar (e.g. URLs) is treated
  as starting a trailing comment.
* **Tab-indented YAML** — PyYAML ``compose`` rejects tab indentation; comment
  extraction fails with ``ScannerError``.
* **Literal block scalars (``|`` / ``>``)** — ``#`` inside block content can be
  mis-read as a parameter comment on the block key line's continuation.
* **YAML anchors/aliases** — Duplicate config nodes via ``&anchor`` / ``*anchor``
  are not a supported config schema pattern.
* **Simulation aggregate per-element comments** — Nested element comments are stored
  on ``yaml_description_map`` at load time; child objects sorted into
  ``hyparams`` / ``densities`` / ``lasers`` do not automatically receive per-object
  ``yaml_parameter_descriptions`` unless loaded from separate files.
"""

from __future__ import annotations

import pytest
import yaml

from inversion_fbpic.lib.commented_yaml import extract_yaml_comments


def test_known_limitation_hash_in_unquoted_scalar_splits_comment() -> None:
    """Document mis-parse: ``#`` in a bare URL-like scalar becomes a fake comment."""
    text = """config_type: density_profile
subclass: sine_squared_bump
parameters:
  elec_name: http://example.com#fragment
  nominal_density: 1.0
  length: 1.0
  p_nz: 1
  p_nr: 1
  p_nt: 1
"""
    params = extract_yaml_comments(text)[()][1]
    assert params.get("elec_name") == ["fragment"]


def test_tab_indented_yaml_compose_raises() -> None:
    """Tab-indented mappings are invalid for PyYAML compose (and thus extract)."""
    text = (
        "config_type: density_profile\n"
        "subclass: sine_squared_bump\n"
        "parameters:\n"
        "\tnominal_density: 1.0  # tab comment\n"
        "\tlength: 1.0\n"
        "\tp_nz: 1\n"
        "\tp_nr: 1\n"
        "\tp_nt: 1\n"
    )
    with pytest.raises(yaml.scanner.ScannerError):
        extract_yaml_comments(text)


def test_known_limitation_block_scalar_hash_misattributed() -> None:
    """``#`` inside ``|`` block content must not become a parameter comment (currently does)."""
    text = """config_type: density_profile
subclass: sine_squared_bump
parameters:
  description: |
    foo # bar
    baz
  nominal_density: 1.0  # real comment
  length: 1.0
  p_nz: 1
  p_nr: 1
  p_nt: 1
"""
    params = extract_yaml_comments(text)[()][1]
    assert params["nominal_density"] == ["real comment"]
    assert params.get("description") == ["bar"]


@pytest.mark.xfail(
    reason="Anchors merge nodes; comment paths for duplicated configs are undefined",
    strict=False,
)
def test_yaml_anchor_alias_duplicate_config() -> None:
    """Anchor/alias payloads are not a supported pattern; extraction may see one node."""
    text = """config_type: density_profile
subclass: sine_squared_bump
parameters: &params
  nominal_density: 1.0  # anchored comment
  length: 1.0
  p_nz: 1
  p_nr: 1
  p_nt: 1
---
config_type: density_profile
subclass: sine_squared_bump
parameters: *params
"""
    result = extract_yaml_comments(text)
    assert () in result
    assert "anchored comment" in str(result)


def test_simulation_aggregate_stores_nested_comment_map() -> None:
    """Loading a multi-element simulation YAML stores per-element paths in the map."""
    from inversion_fbpic.lib.simulation import Simulation

    source = """# Aggregate simulation header.
config_type: simulation
subclass: simulation
parameters:
  verbosity: 30
  elements:
    - config_type: simulation_hyperparameters
      subclass: simulation_hyperparameters
      parameters:
        zmin: -7.0e-05
        zmax: 0.0
        rmax: 1.4e-04
        nz: 2048
        nr: 300
        nm: 3
        use_mpi: false
        number_dumps: 100
    - config_type: density_profile
      subclass: sine_squared_bump
      parameters:
        nominal_density: 1.0e+24  # density element note
        length: 1.0e-03
        p_nz: 1
        p_nr: 1
        p_nt: 1
    - config_type: laser_pulse
      subclass: gaussian
      parameters:
        energy: 5.0
        z0: -3.0e-05
        wavelength: 8.0e-07
        tau_fwhm: 3.8e-14
        cep: 0.0
        waist: 2.8e-05
        focal_position: 0.0
        polarization: 0.0
"""
    sim = Simulation.from_yaml(source)
    assert sim.yaml_class_description == "Aggregate simulation header."
    assert sim.yaml_description_map is not None
    density_path = ("parameters", "elements", 1)
    assert density_path in sim.yaml_description_map
    _, density_params = sim.yaml_description_map[density_path]
    assert density_params["nominal_density"] == ["density element note"]

    out = sim.to_yaml(comments=True, round_trip_comments=True)
    assert "Aggregate simulation header." in out
    assert "density element note" in out


def test_simulation_sorted_children_do_not_inherit_element_yaml_param_comments() -> (
    None
):
    """After post_init, sorted DensityProfile objects do not get element-level stored comments."""
    from inversion_fbpic.lib.simulation import Simulation

    source = """config_type: simulation
subclass: simulation
parameters:
  verbosity: 30
  elements:
    - config_type: simulation_hyperparameters
      subclass: simulation_hyperparameters
      parameters:
        zmin: -7.0e-05
        zmax: 0.0
        rmax: 1.4e-04
        nz: 2048
        nr: 300
        nm: 3
        use_mpi: false
        number_dumps: 100
    - config_type: density_profile
      subclass: sine_squared_bump
      parameters:
        nominal_density: 1.0e+24  # only on aggregate path
        length: 1.0e-03
        p_nz: 1
        p_nr: 1
        p_nt: 1
    - config_type: laser_pulse
      subclass: gaussian
      parameters:
        energy: 5.0
        z0: -3.0e-05
        wavelength: 8.0e-07
        tau_fwhm: 3.8e-14
        cep: 0.0
        waist: 2.8e-05
        focal_position: 0.0
        polarization: 0.0
"""
    sim = Simulation.from_yaml(source)
    assert len(sim.densities) == 1
    density = sim.densities[0]
    assert density.yaml_parameter_descriptions is None or (
        "nominal_density" not in (density.yaml_parameter_descriptions or {})
    )
