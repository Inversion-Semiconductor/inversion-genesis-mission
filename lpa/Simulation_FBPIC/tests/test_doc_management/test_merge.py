"""Tests for docstring parsing and merging (merge.py)."""

from __future__ import annotations

from inversion_fbpic.lib._doc_management import merge
from inversion_fbpic.lib._doc_management.index import LibIndex, default_lib_dir

LIB_DIR = default_lib_dir()


def _index() -> LibIndex:
    return LibIndex.from_lib_dir(LIB_DIR)


def test_merge_parses_args_without_blank_line_before_header() -> None:
    doc = """Summary line.
    Args:
        peak_z0: Position of the peak.
    """
    descriptions = merge.parameter_descriptions_from_doc(doc)
    assert descriptions["peak_z0"] == ["Position of the peak."]


def test_class_description_normalizes_multiline_source_indentation() -> None:
    doc = """
    Summary line.
    Continuation line.
    Args:
        value: An input parameter.
    """

    assert merge.class_description(doc) == "Summary line.\nContinuation line."


def test_merge_ignores_nested_name_lines_in_yaml_examples() -> None:
    doc = """Summary.

    Args:
        elec_select: Filters for electrons.
            Example YAML:
                elec_select:
                  uz:
                  - 10.0
                  - null
                  z:
                  - -1.0e-06
                  - 1.0e-06
            Defaults to None.
    """
    descriptions = merge.parameter_descriptions_from_doc(doc)
    assert set(descriptions) == {"elec_select"}
    joined = " ".join(descriptions["elec_select"])
    assert "Example YAML" in joined
    assert "uz:" in joined


def test_merge_formats_nested_examples_without_absorbing_next_parameter() -> None:
    doc = """Summary.

    Args:
        elec_select: Filters for electrons.
            Example YAML:
                elec_select:
                  uz:
                  - 10.0
            Defaults to None.
        ion_name: Name of the ion species.
    """

    descriptions = merge.parameter_descriptions_from_doc(doc)
    rendered = merge.format_args_block(descriptions)

    assert "    elec_select: Filters for electrons." in rendered
    assert "        Example YAML:" in rendered
    assert "`{elec_select: {uz: [10.0]}}`" in rendered
    assert "    ion_name: Name of the ion species." in rendered


def test_merge_stops_args_at_returns_section() -> None:
    doc = """Summary.

    Args:
        x: An input parameter.

    Returns:
        y: Must not be treated as an init parameter.
    """
    descriptions = merge.parameter_descriptions_from_doc(doc)
    assert set(descriptions) == {"x"}
    assert "Returns" not in " ".join(descriptions["x"])
    assert "y" not in descriptions


def test_merge_subclass_param_description_overrides_parent() -> None:
    parent = "Parent summary.\n\nArgs:\n    foo: Parent description."
    child = "Child summary.\n\nArgs:\n    foo: Child override."
    merged = merge.build_merged_docstring(child, [parent, child], ["foo"])
    assert "Child override." in merged
    assert "Parent description." not in merged


def test_merge_summary_only_class_inherits_parent_args() -> None:
    index = _index()
    merged = index.merged_docstring(index.classes[("laser", "GaussianLaserPulse")])
    assert "Temporal-gaussian laser pulse with a Gaussian transverse profile." in merged
    assert "energy:" in merged
    assert "wavelength:" in merged
    assert "peak_z0:" not in merged


def test_merge_args_only_docstring_has_no_summary() -> None:
    doc = "Args:\n    only_param: Required."
    assert merge.class_description(doc) == ""
    merged = merge.build_merged_docstring(doc, [doc], ["only_param"])
    assert merged.startswith("Args:")
    assert "only_param:" in merged


def test_merge_empty_docstring_returns_empty_string() -> None:
    assert merge.build_merged_docstring(None, [], []) == ""
