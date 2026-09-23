"""Stdlib-only helpers for parsing and merging Google-style config docstrings.

This module operates on raw docstring text with no third-party dependencies,
making it safe to use in pre-commit hooks under any Python environment.
"""

from __future__ import annotations

import inspect
import re
from typing import Iterable

_ARGS_HEADER_RE = re.compile(r"^([ \t]*)Args:\s*$", re.MULTILINE)
_PARAM_LINE_RE = re.compile(r"^(\w+):\s*(.*)$")
_ARGS_SECTION_END_RE = re.compile(
    r"^(?:Returns|Raises|Note|Yields|Warns|See Also|Examples):\s*$",
    re.IGNORECASE,
)


def _continuation_line_text(param_indent: str, line: str) -> str:
    """Strip *param_indent* from a continuation line, preserving deeper indent."""
    stripped_start = line.lstrip(" ")
    indent = line[: len(line) - len(stripped_start)]
    if len(indent) <= len(param_indent):
        return stripped_start
    return indent[len(param_indent) :] + stripped_start


def _args_header_indent(header_match: re.Match[str]) -> str:
    """Horizontal indent of the ``Args:`` line (group 1 is ``[ \\t]*``)."""
    return header_match.group(1)


def _parse_args_parameter_descriptions(
    args_text: str, *, header_indent: str
) -> dict[str, list[str]]:
    """Parse the body of a Google-style ``Args:`` section.

    Args:
        args_text: Text immediately following the ``Args:`` header line.
        header_indent: Whitespace indent of the ``Args:`` header, used to
            detect the end of the section when a sibling header (e.g.
            ``Returns:``) appears at the same or lesser indent.

    Returns:
        Mapping of parameter names to their description lines (with leading
        indent normalised relative to the parameter name).
    """
    descriptions: dict[str, list[str]] = {}
    param_indent: str | None = None
    current_name: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_name, current_lines
        if current_name is not None and param_indent is not None:
            descriptions[current_name] = [
                _continuation_line_text(param_indent, line)
                for line in current_lines
                if line.strip()
            ]
        current_name = None
        current_lines = []

    for line in args_text.splitlines():
        if not line.strip():
            continue

        stripped_start = line.lstrip(" ")
        indent = line[: len(line) - len(stripped_start)]

        if len(indent) <= len(header_indent) and _ARGS_SECTION_END_RE.match(
            stripped_start
        ):
            break

        param_match = _PARAM_LINE_RE.match(stripped_start)
        if param_match is None:
            if current_name is not None:
                current_lines.append(line)
            continue

        if param_indent is None:
            param_indent = indent

        if indent == param_indent:
            flush()
            current_name = param_match.group(1)
            rest = param_match.group(2).strip()
            if rest:
                current_lines.append(rest)
        elif current_name is not None:
            current_lines.append(line)

    flush()
    return descriptions


def parameter_descriptions_from_doc(doc: str) -> dict[str, list[str]]:
    """Extract parameter descriptions from a single docstring."""
    header = _ARGS_HEADER_RE.search(doc)
    if header is None:
        return {}
    args_text = doc[header.end() :].lstrip("\n")
    return _parse_args_parameter_descriptions(
        args_text, header_indent=_args_header_indent(header)
    )


def class_description(doc: str | None) -> str:
    """Return the summary portion of a docstring, excluding ``Args:`` blocks."""
    if not doc:
        return ""
    clean_doc = inspect.cleandoc(doc)
    block = _ARGS_HEADER_RE.search(clean_doc)
    if block:
        return clean_doc[: block.start()].strip()
    return clean_doc


def _yaml_flow_value(lines: list[tuple[int, str]], position: int) -> tuple[str, int]:
    """Render one indented YAML block as a flow-style mapping or sequence."""
    indent, _line = lines[position]
    is_sequence = lines[position][1].startswith("- ")
    values: list[str] = []

    while position < len(lines) and lines[position][0] == indent:
        _current_indent, content = lines[position]
        if is_sequence:
            if not content.startswith("- "):
                break
            values.append(content[2:])
            position += 1
            continue

        if ":" not in content:
            break
        key, value = content.split(":", maxsplit=1)
        if value.strip():
            values.append(f"{key}: {value.strip()}")
            position += 1
            continue
        if position + 1 >= len(lines):
            values.append(f"{key}: null")
            position += 1
            continue
        next_indent, next_content = lines[position + 1]
        if next_indent < indent or (
            next_indent == indent and not next_content.startswith("- ")
        ):
            values.append(f"{key}: null")
            position += 1
            continue
        nested_value, position = _yaml_flow_value(lines, position + 1)
        values.append(f"{key}: {nested_value}")

    wrapper = ("[", "]") if is_sequence else ("{", "}")
    return f"{wrapper[0]}{', '.join(values)}{wrapper[1]}", position


def _yaml_flow_style(yaml_lines: list[str]) -> str:
    """Convert a block-style YAML mapping or sequence into YAML flow style."""
    parsed_lines = [
        (len(line) - len(line.lstrip()), line.strip())
        for line in yaml_lines
        if line.strip()
    ]
    if not parsed_lines:
        return ""
    return _yaml_flow_value(parsed_lines, 0)[0]


def _inline_yaml_examples(desc_lines: list[str]) -> list[str]:
    """Convert indented YAML examples to one flow-style continuation for Pylance."""
    rendered: list[str] = []
    position = 0
    while position < len(desc_lines):
        line = desc_lines[position]
        if line.strip() != "Example YAML:":
            rendered.append(line)
            position += 1
            continue

        marker_indent = len(line) - len(line.lstrip())
        yaml_lines: list[str] = []
        position += 1
        while position < len(desc_lines):
            candidate = desc_lines[position]
            candidate_indent = len(candidate) - len(candidate.lstrip())
            if candidate_indent <= marker_indent:
                break
            yaml_lines.append(candidate)
            position += 1
        yaml_example = _yaml_flow_style(yaml_lines)
        rendered.append(f"{' ' * marker_indent}Example YAML: `{yaml_example}`")
    return rendered


def format_args_block(descriptions: dict[str, list[str]]) -> str:
    """Render a Google-style ``Args:`` section from parameter descriptions."""
    if not descriptions:
        return ""
    lines = ["Args:"]
    for name, desc_lines in descriptions.items():
        desc_lines = _inline_yaml_examples(desc_lines)
        if not desc_lines:
            lines.append(f"    {name}:")
            continue
        lines.append(f"    {name}: {desc_lines[0]}")
        for continuation in desc_lines[1:]:
            lines.append(f"    {continuation}")
    return "\n".join(lines)


def merge_parameter_descriptions(docs: Iterable[str | None]) -> dict[str, list[str]]:
    """Merge ``Args:`` blocks; later docs override earlier ones."""
    descriptions: dict[str, list[str]] = {}
    for doc in docs:
        if doc:
            descriptions.update(parameter_descriptions_from_doc(doc))
    return descriptions


def _order_param_descriptions(
    param_descs: dict[str, list[str]], field_names: Iterable[str]
) -> dict[str, list[str]]:
    """Order *param_descs* so init-field names come first, then any extras."""
    ordered: dict[str, list[str]] = {}
    for name in field_names:
        if name in param_descs:
            ordered[name] = param_descs[name]
    for name, desc_lines in param_descs.items():
        if name not in ordered:
            ordered[name] = desc_lines
    return ordered


def build_merged_docstring(
    summary_doc: str | None,
    mro_docs_subclass_last: Iterable[str | None],
    field_names: Iterable[str],
) -> str:
    """Build a class docstring with merged ``Args:`` descriptions."""
    summary = class_description(summary_doc)
    param_descs = merge_parameter_descriptions(mro_docs_subclass_last)
    args_block = format_args_block(_order_param_descriptions(param_descs, field_names))
    if not summary:
        return args_block
    if not args_block:
        return summary
    return f"{summary}\n\n{args_block}"
