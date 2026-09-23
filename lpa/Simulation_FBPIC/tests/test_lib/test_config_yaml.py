"""
Edge-case tests for SerializableConfig and commented_yaml.

Covers comment extraction/injection, round-trip behavior, deserialization
fallbacks, and docstring parsing — including cases that are easy to regress.

For advanced / known-broken YAML comment scenarios (tabs, anchors, block
scalars, unquoted ``#``, simulation aggregate propagation), see
``test_config_yaml_limitations.py``.

Run from Simulation_FBPIC::

    pytest tests/test_lib/test_config_yaml.py \\
           tests/test_lib/test_config_yaml_limitations.py \\
           tests/test_lib/test_serializable_config.py -v

Test catalog
------------

Comment extraction (``commented_yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------------+----------------------------------------------------------+
| Edge case                     | Test                                                     |
+===============================+==========================================================+
| Empty / comment-free YAML     | ``test_extract_empty_*``,                                |
|                               | ``test_extract_yaml_without_comments_*``                 |
+-------------------------------+----------------------------------------------------------+
| ``#`` inside double- or       | ``test_extract_does_not_treat_hash_*``,                  |
| single-quoted values          | ``test_extract_hash_inside_single_quoted_string``        |
+-------------------------------+----------------------------------------------------------+
| Multi-line class header       | ``test_extract_multiline_class_header_comments``         |
| comments                      |                                                          |
+-------------------------------+----------------------------------------------------------+
| Continuation-only comments    | ``test_extract_continuation_only_parameter_comments``    |
| (no inline on key line)       |                                                          |
+-------------------------------+----------------------------------------------------------+
| Nested configs in lists       | ``test_extract_nested_config_paths_in_list``             |
| (path map)                    |                                                          |
+-------------------------------+----------------------------------------------------------+
| Comments on nested dict keys  | ``test_extract_dict_value_inline_comment_on_nested_key`` |
| (``elec_select.uz``)          |                                                          |
+-------------------------------+----------------------------------------------------------+

Comment injection / dump
~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------------+----------------------------------------------------------+
| Edge case                     | Test                                                     |
+===============================+==========================================================+
| ``comments=False`` → plain    | ``test_dump_yaml_with_comments_false_*``                 |
| YAML                          |                                                          |
+-------------------------------+----------------------------------------------------------+
| Empty description map →       | ``test_inject_empty_descriptions_*``                     |
| unchanged                     |                                                          |
+-------------------------------+----------------------------------------------------------+
| Unknown parameter in          | ``test_inject_unknown_parameter_key_*``                  |
| description map               |                                                          |
+-------------------------------+----------------------------------------------------------+
| Inject ↔ extract round-trip   | ``test_inject_and_extract_round_trip_scalar_comments``   |
+-------------------------------+----------------------------------------------------------+
| More comment lines than dict  | ``test_inject_overflow_comments_beyond_dict_anchors``    |
| anchors (overflow)            |                                                          |
+-------------------------------+----------------------------------------------------------+

``SerializableConfig`` I/O
~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------------+----------------------------------------------------------+
| Edge case                     | Test                                                     |
+===============================+==========================================================+
| ``from_dict`` / ``from_json`` | ``test_from_dict_does_not_*``,                           |
| do not store YAML comments    | ``test_from_json_*``                                     |
+-------------------------------+----------------------------------------------------------+
| ``round_trip_comments=False`` | ``test_round_trip_comments_false_*``                     |
| → docstrings after YAML load  |                                                          |
+-------------------------------+----------------------------------------------------------+
| ``comments=False`` on         | ``test_to_yaml_comments_false_*``                         |
| ``to_yaml``                   |                                                          |
+-------------------------------+----------------------------------------------------------+
| Invalid YAML root, bad        | ``test_from_yaml_invalid_*``,                            |
| ``parameters``, unknown       | ``test_from_dict_missing_*``,                            |
| ``config_type``               | ``test_from_dict_unknown_*``                             |
+-------------------------------+----------------------------------------------------------+
| Wrong domain when calling     | ``test_from_dict_wrong_domain_class_raises``              |
| subclass ``from_dict``        |                                                          |
+-------------------------------+----------------------------------------------------------+
| Missing optional keys + type  | ``test_from_dict_optional_union_*``,                     |
| fallbacks / attrs defaults    | ``test_example_dict_uses_non_none_*``                    |
+-------------------------------+----------------------------------------------------------+
| ``include_nones=False`` drops | ``test_to_dict_include_nones_false_*``                   |
| null fields                   |                                                          |
+-------------------------------+----------------------------------------------------------+
| Plain YAML reload → no stored | ``test_plain_yaml_reload_*``                             |
| comments                      |                                                          |
+-------------------------------+----------------------------------------------------------+
| Dict-valued param comment     | ``test_yaml_load_dict_value_round_trip_*``               |
| round-trip                    |                                                          |
+-------------------------------+----------------------------------------------------------+
| Complex ``polarization`` →    | ``test_to_dict_serializes_complex_and_path``             |
| ``[re, im]`` in JSON          |                                                          |
+-------------------------------+----------------------------------------------------------+

Docstring ``Args:`` parsing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------------+----------------------------------------------------------+
| Edge case                     | Test                                                     |
+===============================+==========================================================+
| Stops at ``Returns:``         | ``test_parse_args_stops_at_returns_section``             |
+-------------------------------+----------------------------------------------------------+
| Preserves nested example      | ``test_parse_args_preserves_continuation_indentation``   |
| indentation                   |                                                          |
+-------------------------------+----------------------------------------------------------+
| Subclass overrides parent     | ``test_subclass_parameter_description_overrides_parent`` |
| param docs (MRO)              |                                                          |
+-------------------------------+----------------------------------------------------------+
| Class summary excludes        | ``test_class_description_excludes_args_block``           |
| ``Args:`` block               |                                                          |
+-------------------------------+----------------------------------------------------------+
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from inversion_fbpic.lib.commented_yaml import (
    YAML_INLINE_COMMENT_COLUMN,
    _comment_padding,
    dump_yaml_with_comments,
    extract_yaml_comments,
    inject_yaml_comments,
)
from inversion_fbpic.lib.density_core import _DensityProfile
from inversion_fbpic.lib.density_profiles import ExampleDensityProfile
from inversion_fbpic.lib.serializable_config import (
    CONFIG_TYPE_STR,
    PARAMETERS_STR,
    SUBCLASS_STR,
    SerializableConfig,
    _parameter_descriptions_from_doc,
)


# ---------------------------------------------------------------------------
# commented_yaml: extraction
# ---------------------------------------------------------------------------


def test_extract_empty_yaml_returns_empty_map() -> None:
    assert extract_yaml_comments("") == {}


def test_extract_yaml_without_comments_returns_empty_map() -> None:
    text = """config_type: density_profile
subclass: sine_squared_bump
parameters:
  nominal_density: 1.0
  length: 1.0
  p_nz: 1
  p_nr: 1
  p_nt: 1
"""
    assert extract_yaml_comments(text) == {}


def test_extract_does_not_treat_hash_inside_quoted_strings_as_comments() -> None:
    text = """config_type: density_profile
subclass: sine_squared_bump
parameters:
  elec_name: "electrons # not a yaml comment"
  nominal_density: 1.0  # real trailing comment
  length: 1.0
  p_nz: 1
  p_nr: 1
  p_nt: 1
"""
    params = extract_yaml_comments(text)[()][1]
    assert params["nominal_density"] == ["real trailing comment"]
    assert "elec_name" not in params


def test_extract_multiline_class_header_comments() -> None:
    text = """# Line one.
# Line two.
config_type: density_profile
subclass: sine_squared_bump
parameters:
  nominal_density: 1.0
  length: 1.0
  p_nz: 1
  p_nr: 1
  p_nt: 1
"""
    class_desc, _ = extract_yaml_comments(text)[()]
    assert class_desc == "Line one.\nLine two."


def test_extract_continuation_only_parameter_comments() -> None:
    """Key line has no inline comment; continuations are still captured."""
    text = """config_type: density_profile
subclass: sine_squared_bump
parameters:
  p_nt: 1
                                       # orphan continuation only
  length: 1.0
  nominal_density: 1.0
  p_nz: 1
  p_nr: 1
"""
    params = extract_yaml_comments(text)[()][1]
    assert params["p_nt"] == ["orphan continuation only"]


def test_extract_nested_config_paths_in_list() -> None:
    text_commented = """# density note
    - config_type: density_profile
      subclass: sine_squared_bump
      parameters:
        length: 1.0  # len note
        nominal_density: 1.0
        p_nz: 1
        p_nr: 1
        p_nt: 1
"""
    wrapped = f"""config_type: simulation
subclass: simulation
parameters:
  elements:
{text_commented}
"""
    nested = extract_yaml_comments(wrapped)
    assert ("parameters", "elements", 0) in nested
    assert nested[("parameters", "elements", 0)][0] == "density note"
    assert nested[("parameters", "elements", 0)][1]["length"] == ["len note"]


def test_extract_dict_value_inline_comment_on_nested_key() -> None:
    text = """config_type: density_profile
subclass: sine_squared_bump
parameters:
  nominal_density: 1.0
  length: 1.0
  p_nz: 1
  p_nr: 1
  p_nt: 1
  elec_select:
    uz: [1.0, null]  # uz filter note
    z: [-1.0e-06, 1.0e-06]
"""
    params = extract_yaml_comments(text)[()][1]
    assert "elec_select" in params
    assert "uz filter note" in " ".join(params["elec_select"])


# ---------------------------------------------------------------------------
# commented_yaml: injection / dump
# ---------------------------------------------------------------------------


def test_dump_yaml_with_comments_false_is_plain_yaml() -> None:
    data = {"a": 1, "b": [2, 3]}
    out = dump_yaml_with_comments(data, ("", {}), comments=False)
    assert "#" not in out
    assert yaml.safe_load(out) == data


def test_inject_empty_descriptions_returns_unchanged() -> None:
    raw = "x: 1\n"
    assert inject_yaml_comments(raw, ("", {})) == raw


def test_inject_unknown_parameter_key_is_ignored() -> None:
    raw = """config_type: density_profile
subclass: sine_squared_bump
parameters:
  length: 1.0
  nominal_density: 1.0
  p_nz: 1
  p_nr: 1
  p_nt: 1
"""
    out = inject_yaml_comments(
        raw,
        ("", {"nonexistent_field": ["should not appear"]}),
    )
    assert "should not appear" not in out


def test_comment_padding_never_zero_before_hash() -> None:
    """Inline comments must be separated from scalar values (YAML ``#`` rules)."""
    assert _comment_padding(39) >= 1
    assert _comment_padding(YAML_INLINE_COMMENT_COLUMN - 1) >= 1


def test_inject_inline_comment_preserves_numeric_scalar_type() -> None:
    """Regression: 39-char parameter lines must not glue ``#`` to the value."""
    raw = """config_type: laser_pulse
subclass: gaussian
parameters:
  focal_position: 0.0027653024196624756
"""
    out = inject_yaml_comments(raw, ("", {"focal_position": ["focal note"]}))
    focal_line = next(line for line in out.splitlines() if "focal_position" in line)
    assert "#" in focal_line
    assert "4756 #" in focal_line or "4756  #" in focal_line
    parsed = yaml.safe_load(out)["parameters"]["focal_position"]
    assert isinstance(parsed, float)
    assert parsed == pytest.approx(0.0027653024196624756)


def test_inject_and_extract_round_trip_scalar_comments() -> None:
    raw = """config_type: density_profile
subclass: sine_squared_bump
parameters:
  length: 1.0
  nominal_density: 1.0
  p_nz: 1
  p_nr: 1
  p_nt: 1
"""
    descriptions = ("Header text", {"length": ["length parameter note"]})
    injected = inject_yaml_comments(raw, descriptions)
    extracted = extract_yaml_comments(injected)[()]
    assert extracted[0] == "Header text"
    assert extracted[1]["length"] == ["length parameter note"]


# ---------------------------------------------------------------------------
# serializable_config: loading / comment storage
# ---------------------------------------------------------------------------


def test_from_dict_does_not_attach_yaml_comments() -> None:
    payload = ExampleDensityProfile(
        nominal_density=1.0e24,
        length=1.0e-5,
        p_nz=1,
        p_nr=1,
        p_nt=1,
    ).to_dict()
    obj = SerializableConfig.from_dict(payload)
    assert obj.yaml_class_description is None
    assert obj.yaml_parameter_descriptions is None
    assert obj.yaml_description_map is None


def test_from_json_does_not_attach_yaml_comments() -> None:
    payload = ExampleDensityProfile(
        nominal_density=1.0e24,
        length=1.0e-5,
        p_nz=1,
        p_nr=1,
        p_nt=1,
    ).to_json()
    obj = SerializableConfig.from_json(payload)
    assert not obj._has_round_trip_yaml_comments()


def test_round_trip_comments_false_uses_docstrings_after_yaml_load() -> None:
    source = """# Only in source file.
config_type: density_profile
subclass: sine_squared_bump
parameters:
  nominal_density: 1.0e+24
  length: 1.0e-05
  p_nz: 1
  p_nr: 1
  p_nt: 1
"""
    profile = ExampleDensityProfile.from_yaml(source)
    out = profile.to_yaml(comments=True, round_trip_comments=False)
    assert "Only in source file" not in out
    assert "Finite sine-squared" in out


def test_to_yaml_comments_false_omits_all_hash_marks() -> None:
    profile = ExampleDensityProfile(
        nominal_density=1.0e24,
        length=1.0e-5,
        p_nz=1,
        p_nr=1,
        p_nt=1,
    )
    text = profile.to_yaml(comments=False)
    assert "#" not in text


def test_from_yaml_invalid_root_type_raises() -> None:
    with pytest.raises(ValueError, match="YAML object"):
        SerializableConfig.from_yaml("- not a mapping\n")


def test_from_dict_missing_parameters_raises() -> None:
    with pytest.raises(ValueError, match=PARAMETERS_STR):
        SerializableConfig.from_dict(
            {
                CONFIG_TYPE_STR: "density_profile",
                SUBCLASS_STR: "sine_squared_bump",
                "parameters": [],
            }
        )


def test_from_dict_unknown_config_type_raises() -> None:
    with pytest.raises(ValueError, match=CONFIG_TYPE_STR):
        SerializableConfig.from_dict(
            {
                CONFIG_TYPE_STR: "not_a_real_domain",
                SUBCLASS_STR: "x",
                PARAMETERS_STR: {},
            }
        )


def test_from_dict_wrong_domain_class_raises() -> None:
    payload = {
        CONFIG_TYPE_STR: "laser_pulse",
        SUBCLASS_STR: "gaussian",
        PARAMETERS_STR: {
            "energy": 1.0,
            "z0": 0.0,
            "wavelength": 800e-9,
            "tau_fwhm": 38e-15,
            "cep": 0.0,
            "waist": 28e-6,
            "focal_position": 0.0,
            "polarization": 0.0,
        },
    }
    with pytest.raises(ValueError, match="not a subclass"):
        ExampleDensityProfile.from_dict(payload)


def test_from_dict_optional_union_defaults_applied() -> None:
    obj = ExampleDensityProfile.from_dict(
        {
            CONFIG_TYPE_STR: "density_profile",
            SUBCLASS_STR: "sine_squared_bump",
            PARAMETERS_STR: {
                "nominal_density": 1.0e24,
                "length": 1.0e-5,
                "p_nz": 1,
                "p_nr": 1,
                "p_nt": 1,
            },
        }
    )
    assert obj.species == "H"
    assert obj.elec_name is None
    assert obj.elec_select is None


def _minimal_density_payload(*, nominal_density: float = 1.0e24) -> dict:
    return {
        CONFIG_TYPE_STR: "density_profile",
        SUBCLASS_STR: "sine_squared_bump",
        PARAMETERS_STR: {
            "nominal_density": nominal_density,
            "length": 1.0e-5,
            "p_nz": 1,
            "p_nr": 1,
            "p_nt": 1,
        },
    }


def test_from_dict_overrides_replace_scalar_parameter() -> None:
    payload = _minimal_density_payload(nominal_density=1.0e24)
    obj = ExampleDensityProfile.from_dict(
        payload,
        overrides={PARAMETERS_STR: {"nominal_density": 2.5e24}},
    )
    assert obj.nominal_density == 2.5e24
    assert payload[PARAMETERS_STR]["nominal_density"] == 1.0e24


def test_from_dict_overrides_merge_nested_parameter() -> None:
    payload = _minimal_density_payload()
    payload[PARAMETERS_STR]["elec_select"] = {
        "uz": [1.0, None],
        "z": [-1.0e-6, 1.0e-6],
    }
    obj = ExampleDensityProfile.from_dict(
        payload,
        overrides={PARAMETERS_STR: {"elec_select": {"uz": [20.0, None]}}},
    )
    assert obj.elec_select is not None
    assert obj.elec_select["uz"] == [20.0, None]
    assert obj.elec_select["z"] == [-1.0e-6, 1.0e-6]


def test_from_json_overrides_parameter() -> None:
    from inversion_fbpic.lib.laser import GaussianLaserPulse

    payload = GaussianLaserPulse(
        energy=5.0,
        z0=-3.0e-5,
        wavelength=8.0e-7,
        tau_fwhm=3.8e-14,
        cep=0.0,
        waist=2.8e-5,
        focal_position=3.0e-3,
        polarization=0.0,
    ).to_json()
    obj = GaussianLaserPulse.from_json(
        payload,
        overrides={PARAMETERS_STR: {"energy": 9.0}},
    )
    assert obj.energy == 9.0


def test_from_yaml_overrides_parameter() -> None:
    source = """config_type: density_profile
subclass: sine_squared_bump
parameters:
  nominal_density: 1.0e+24
  length: 1.0e-05
  p_nz: 1
  p_nr: 1
  p_nt: 1
"""
    obj = ExampleDensityProfile.from_yaml(
        source,
        overrides={PARAMETERS_STR: {"length": 2.0e-5}},
    )
    assert obj.length == 2.0e-5


def test_from_file_overrides_parameter(tmp_path: Path) -> None:
    yaml_path = tmp_path / "density.yaml"
    yaml_path.write_text(
        """config_type: density_profile
subclass: sine_squared_bump
parameters:
  nominal_density: 1.0e+24
  length: 1.0e-05
  p_nz: 1
  p_nr: 1
  p_nt: 1
""",
        encoding="utf-8",
    )
    obj = SerializableConfig.from_file(
        yaml_path,
        overrides={PARAMETERS_STR: {"species": "N"}},
    )
    assert isinstance(obj, ExampleDensityProfile)
    assert obj.species == "N"


def test_from_file_json_overrides_parameter(tmp_path: Path) -> None:
    from inversion_fbpic.lib.laser import GaussianLaserPulse

    json_path = tmp_path / "laser.json"
    json_path.write_text(
        GaussianLaserPulse(
            energy=5.0,
            z0=-3.0e-5,
            wavelength=8.0e-7,
            tau_fwhm=3.8e-14,
            cep=0.0,
            waist=2.8e-5,
            focal_position=3.0e-3,
            polarization=0.0,
        ).to_json(),
        encoding="utf-8",
    )
    obj = SerializableConfig.from_file(
        json_path,
        overrides={PARAMETERS_STR: {"energy": 7.5}},
    )
    assert isinstance(obj, GaussianLaserPulse)
    assert obj.energy == 7.5


def test_example_dict_uses_non_none_optional_defaults() -> None:
    example = ExampleDensityProfile.example_dict()
    params = example[PARAMETERS_STR]
    assert params["species"] == "H"
    assert params["ionization"] == 0
    assert params["start_position"] == 0.0


def test_to_dict_include_nones_false_drops_null_optional_fields() -> None:
    profile = ExampleDensityProfile(
        nominal_density=1.0e24,
        length=1.0e-5,
        p_nz=1,
        p_nr=1,
        p_nt=1,
    )
    params = profile.to_dict(include_nones=False)[PARAMETERS_STR]
    assert "elec_name" not in params
    assert "p_rmax" not in params


def test_plain_yaml_reload_has_no_stored_comments() -> None:
    profile = ExampleDensityProfile(
        nominal_density=1.0e24,
        length=1.0e-5,
        p_nz=1,
        p_nr=1,
        p_nt=1,
    )
    plain = profile.to_yaml(comments=False)
    reloaded = ExampleDensityProfile.from_yaml(plain)
    assert not reloaded._has_round_trip_yaml_comments()


def test_yaml_load_dict_value_round_trip_preserves_param_comments() -> None:
    source = """config_type: density_profile
subclass: sine_squared_bump
parameters:
  nominal_density: 1.0e+24
  length: 1.0e-03
  p_nz: 1
  p_nr: 1
  p_nt: 1
  elec_select:
    uz: [10.0, null]  # keep this uz note
    z: [-1.0e-06, 1.0e-06]
"""
    profile = ExampleDensityProfile.from_yaml(source)
    assert "elec_select" in profile.yaml_parameter_descriptions
    joined = " ".join(profile.yaml_parameter_descriptions["elec_select"])
    assert "keep this uz note" in joined

    out = profile.to_yaml(comments=True, round_trip_comments=True)
    assert "keep this uz note" in out


# ---------------------------------------------------------------------------
# Docstring Args parsing (serializable_config helpers)
# ---------------------------------------------------------------------------


def test_parse_args_stops_at_returns_section() -> None:
    doc = """Summary line.

    Args:
        alpha: First parameter.
        beta: Second parameter.

    Returns:
        Nothing.
    """
    desc = _parameter_descriptions_from_doc(doc)
    assert set(desc) == {"alpha", "beta"}
    assert "Nothing" not in desc.get("alpha", [""])[0]


def test_parse_args_preserves_continuation_indentation() -> None:
    doc = """Args:
        elec_select: (dict) Optional filter.
            Example YAML:
                elec_select:
                  uz: 1
    """
    desc = _parameter_descriptions_from_doc(doc)
    lines = desc["elec_select"]
    assert lines[1] == "    Example YAML:"
    assert lines[2] == "        elec_select:"


def test_subclass_parameter_description_overrides_parent() -> None:
    class _Parent(_DensityProfile):
        """Parent.

        Args:
            length: Parent length docs.
        """

    class _Child(_Parent):
        """Child.

        Args:
            length: Child length docs.
        """

    desc = _Child._parameter_descriptions()
    assert desc["length"][0].startswith("Child length docs")


def test_class_description_excludes_args_block() -> None:
    summary = ExampleDensityProfile._class_description()
    assert "Args:" not in summary
    assert "Finite sine-squared" in summary


def test_extract_hash_inside_single_quoted_string() -> None:
    text = """config_type: density_profile
subclass: sine_squared_bump
parameters:
  elec_name: 'foo # bar'
  nominal_density: 1.0
  length: 1.0
  p_nz: 1
  p_nr: 1
  p_nt: 1
"""
    assert extract_yaml_comments(text) == {}


def test_to_dict_serializes_complex_and_path(tmp_path: Path) -> None:
    from inversion_fbpic.lib.laser import GaussianLaserPulse

    pulse = GaussianLaserPulse(
        a0=1.0,
        z0=0.0,
        wavelength=800e-9,
        tau_fwhm=38e-15,
        cep=0.0,
        waist=28e-6,
        focal_position=0.0,
        polarization=1j,
    )
    params = pulse.to_dict()[PARAMETERS_STR]
    assert params["polarization"] == [0.0, 1.0]

    pulse.to_json_file(tmp_path / "laser.json")
    reloaded = SerializableConfig.from_file(tmp_path / "laser.json")
    assert reloaded.to_dict()[PARAMETERS_STR]["polarization"] == [0.0, 1.0]


def test_inject_overflow_comments_beyond_dict_anchors() -> None:
    raw = """config_type: density_profile
subclass: sine_squared_bump
parameters:
  elec_select:
    uz: [1.0, null]
  length: 1.0
  nominal_density: 1.0
  p_nz: 1
  p_nr: 1
  p_nt: 1
"""
    descriptions = (
        "",
        {
            "elec_select": [
                "line zero",
                "line one",
                "line two",
                "line three",
                "line four overflow",
            ]
        },
    )
    out = inject_yaml_comments(raw, descriptions)
    assert "line four overflow" in out
