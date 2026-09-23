"""YAML comment injection and extraction utilities.

Functions for inserting docstring-derived comments into dumped YAML strings and
for reading comments back out of YAML sources for round-trip serialization.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping
from typing import Any

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

_PARAM_KEY_LINE_RE = re.compile(r"^([A-Za-z_][\w]*):\s*(.*)$")

# 1-based column where ``#`` starts for aligned inline comments under the block key.
YAML_INLINE_COMMENT_COLUMN = 40

YamlDescriptions = tuple[str, dict[str, list[str]]]
YamlPathPart = str | int
YamlPath = tuple[YamlPathPart, ...]
YamlDescriptionMap = Mapping[YamlPath, YamlDescriptions]

_OnMapping = Callable[[MappingNode, YamlPath], None]


class _FlowStyleDict(dict[str, Any]):
    """Mapping marker rendered using YAML flow style."""


class _FlowStyleList(list[Any]):
    """Sequence marker rendered using YAML flow style."""


class _FlowStyleDumper(yaml.Dumper):
    """PyYAML dumper with explicit flow-style container markers."""


def _represent_flow_style_dict(
    dumper: yaml.Dumper, value: _FlowStyleDict
) -> MappingNode:
    return dumper.represent_mapping("tag:yaml.org,2002:map", value, flow_style=True)


def _represent_flow_style_list(
    dumper: yaml.Dumper, value: _FlowStyleList
) -> SequenceNode:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True)


_FlowStyleDumper.add_representer(_FlowStyleDict, _represent_flow_style_dict)
_FlowStyleDumper.add_representer(_FlowStyleList, _represent_flow_style_list)


def _split_yaml_line_comment(line: str) -> tuple[str, str | None]:
    """Split a YAML source line into body text and trailing ``#`` comment text."""
    stripped = line.strip()
    if stripped.startswith("#"):
        comment = line[line.index("#") + 1 :]
        if comment.startswith(" "):
            comment = comment[1:]
        return "", comment if comment else None

    in_single = in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            comment = line[i + 1 :]
            if comment:
                comment = _normalize_trailing_comment_text(comment)
            return line[:i].rstrip(), comment if comment else None
    return line.rstrip(), None


def _normalize_trailing_comment_text(comment: str) -> str:
    """Preserve indented comment blocks; strip only the YAML ``# `` separator."""
    if comment.startswith("  "):
        return comment
    if comment.startswith(" "):
        return comment[1:]
    return comment


def _build_stripped_yaml(yaml_str: str) -> tuple[str, list[int]]:
    """Return YAML without comment-only lines and a stripped->original line map."""
    stripped_lines: list[str] = []
    orig_indices: list[int] = []
    for orig_idx, line in enumerate(yaml_str.splitlines()):
        body, comment = _split_yaml_line_comment(line)
        if not body.strip() and comment is not None:
            continue
        stripped_lines.append(body if body.strip() else "")
        orig_indices.append(orig_idx)

    text = "\n".join(stripped_lines)
    if yaml_str.endswith("\n"):
        text += "\n"
    return text, orig_indices


def _extract_preceding_comment_lines(lines: list[str], before_idx: int) -> list[str]:
    """Collect full-line comments immediately above *before_idx*."""
    collected: list[str] = []
    for idx in range(before_idx - 1, -1, -1):
        body, comment = _split_yaml_line_comment(lines[idx])
        if body.strip():
            break
        if comment is not None:
            collected.append(comment)
    return list(reversed(collected))


def _extract_parameter_comments(
    lines: list[str],
    key_orig_idx: int,
    param_indent: int,
) -> list[str]:
    """Collect inline and continuation comments for one parameter key line."""
    comments: list[str] = []
    _, inline = _split_yaml_line_comment(lines[key_orig_idx])
    if inline is not None:
        comments.append(inline)

    for idx in range(key_orig_idx + 1, len(lines)):
        line = lines[idx]
        if not line.strip():
            continue

        body, comment = _split_yaml_line_comment(line)
        body_stripped = body.strip()
        line_indent = _line_indent(line)

        if body_stripped:
            if line_indent == param_indent and _PARAM_KEY_LINE_RE.match(body_stripped):
                break
            if comment is not None:
                comments.append(comment)
            continue

        if comment is not None:
            comments.append(comment)

    return comments


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _mapping_entry(node: MappingNode, key: str) -> tuple[ScalarNode, Node] | None:
    for key_node, value_node in node.value:
        if isinstance(key_node, ScalarNode) and key_node.value == key:
            return key_node, value_node
    return None


def _traverse_yaml_tree(node: Node, path: YamlPath, on_mapping: _OnMapping) -> None:
    """Walk a composed YAML tree, calling *on_mapping* at each mapping node."""
    if isinstance(node, MappingNode):
        on_mapping(node, path)
        for key_node, value_node in node.value:
            if isinstance(key_node, ScalarNode):
                _traverse_yaml_tree(value_node, (*path, key_node.value), on_mapping)
        return

    if isinstance(node, SequenceNode):
        for idx, item_node in enumerate(node.value):
            _traverse_yaml_tree(item_node, (*path, idx), on_mapping)


def _comment_anchor_lines(node: Node) -> list[int]:
    """Return line numbers where multiline parameter comments may attach."""
    if isinstance(node, MappingNode):
        lines: list[int] = []
        for key_node, value_node in node.value:
            if isinstance(key_node, ScalarNode):
                lines.append(key_node.start_mark.line)
            lines.extend(_comment_anchor_lines(value_node))
        return lines

    if isinstance(node, SequenceNode):
        lines: list[int] = []
        for item_node in node.value:
            if isinstance(item_node, ScalarNode):
                lines.append(item_node.start_mark.line)
            else:
                lines.extend(_comment_anchor_lines(item_node))
        return lines

    if isinstance(node, ScalarNode):
        return [node.start_mark.line]

    return []


def extract_yaml_comments(
    yaml_str: str,
    *,
    header_key: str = "config_type",
    block_key: str = "parameters",
) -> dict[YamlPath, YamlDescriptions]:
    """Extract class and parameter comments from a YAML source string."""
    lines = yaml_str.splitlines()
    if not lines:
        return {}

    stripped_text, stripped_to_orig = _build_stripped_yaml(yaml_str)
    root = yaml.compose(stripped_text)
    if root is None:
        return {}

    def orig_line(stripped_idx: int) -> int:
        return stripped_to_orig[stripped_idx]

    results: dict[YamlPath, YamlDescriptions] = {}

    def on_mapping(node: MappingNode, path: YamlPath) -> None:
        if _mapping_entry(node, header_key) is None:
            return

        header_node, _ = _mapping_entry(node, header_key)
        header_orig = orig_line(header_node.start_mark.line)
        class_desc = "\n".join(_extract_preceding_comment_lines(lines, header_orig))

        param_descs: dict[str, list[str]] = {}
        block_entry = _mapping_entry(node, block_key)
        if block_entry is not None:
            _, block_node = block_entry
            if isinstance(block_node, MappingNode):
                for key_node, _value_node in block_node.value:
                    if not isinstance(key_node, ScalarNode):
                        continue
                    key_orig = orig_line(key_node.start_mark.line)
                    comments = _extract_parameter_comments(
                        lines, key_orig, _line_indent(lines[key_orig])
                    )
                    if comments:
                        param_descs[key_node.value] = comments

        if class_desc or param_descs:
            results[path] = (class_desc, param_descs)

    _traverse_yaml_tree(root, (), on_mapping)
    return results


def _comment_padding(body_len: int) -> int:
    col = YAML_INLINE_COMMENT_COLUMN
    if body_len < col:
        # YAML treats ``#`` as starting a comment only when preceded by whitespace.
        return max(col - body_len - 1, 1)
    return 2


def _normalize_description_map(
    descriptions: YamlDescriptions | YamlDescriptionMap,
) -> dict[YamlPath, YamlDescriptions]:
    if isinstance(descriptions, tuple):
        return {(): descriptions}
    return {
        path: description
        for path, description in descriptions.items()
        if description[0] or description[1]
    }


def _apply_flow_style_paths(
    value: Any,
    *,
    path: YamlPath = (),
    flow_style_paths: Collection[YamlPath],
) -> Any:
    """Copy *value*, marking selected mapping and sequence paths as flow style."""
    if isinstance(value, dict):
        converted = {
            key: _apply_flow_style_paths(
                nested,
                path=(*path, str(key)),
                flow_style_paths=flow_style_paths,
            )
            for key, nested in value.items()
        }
        return _FlowStyleDict(converted) if path in flow_style_paths else converted
    if isinstance(value, list):
        converted = [
            _apply_flow_style_paths(
                nested,
                path=(*path, idx),
                flow_style_paths=flow_style_paths,
            )
            for idx, nested in enumerate(value)
        ]
        return _FlowStyleList(converted) if path in flow_style_paths else converted
    return value


def _clean_comment_lines(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if line.strip()]


def _render_commented_yaml(
    lines: list[str],
    *,
    before_comments: dict[int, list[tuple[int, str]]],
    inline_comments: dict[int, list[str]],
    after_comments: dict[int, list[tuple[int, str]]],
) -> str:
    """Merge comment annotations into a line-oriented YAML dump."""
    result: list[str] = []
    for idx, line in enumerate(lines):
        for comment_indent, comment in before_comments.get(idx, []):
            result.append(f"{' ' * comment_indent}# {comment}\n")

        line_out = line
        inline = inline_comments.get(idx, [])
        if inline:
            body = line.rstrip("\r\n")
            pad = _comment_padding(len(body))
            first, *rest = inline
            line_out = f"{body}{' ' * pad}# {first}\n"
            if rest:
                after_comments.setdefault(idx, []).extend(
                    (len(body) + pad, comment) for comment in rest
                )
        result.append(line_out)

        for comment_col, comment in after_comments.get(idx, []):
            result.append(f"{' ' * comment_col}# {comment}\n")

    return "".join(result)


def inject_yaml_comments(
    yaml_str: str,
    descriptions: YamlDescriptions | YamlDescriptionMap,
    *,
    indent: int = 2,
    header_key: str = "config_type",
    block_key: str = "parameters",
) -> str:
    """Insert class and parameter comments into a YAML string."""
    del indent

    description_map = _normalize_description_map(descriptions)
    if not description_map:
        return yaml_str

    root = yaml.compose(yaml_str)
    if root is None:
        return yaml_str

    lines = yaml_str.splitlines(keepends=True)
    before_comments: dict[int, list[tuple[int, str]]] = {}
    inline_comments: dict[int, list[str]] = {}
    after_comments: dict[int, list[tuple[int, str]]] = {}

    def add_before(line_idx: int, comment_lines: list[str]) -> None:
        if not comment_lines:
            return
        indent_spaces = _line_indent(lines[line_idx])
        before_comments.setdefault(line_idx, []).extend(
            (indent_spaces, comment) for comment in comment_lines
        )

    def add_inline(line_idx: int, comment: str) -> int:
        inline_comments.setdefault(line_idx, []).append(comment)
        body = lines[line_idx].rstrip("\r\n")
        return len(body) + _comment_padding(len(body))

    def add_after(line_idx: int, column: int, comment_lines: list[str]) -> None:
        if comment_lines:
            after_comments.setdefault(line_idx, []).extend(
                (column, comment) for comment in comment_lines
            )

    def annotate_parameter(
        key_node: ScalarNode, value_node: Node, comment_lines: list[str]
    ) -> None:
        if not comment_lines:
            return

        first, *rest = comment_lines
        hash_col = add_inline(key_node.start_mark.line, first)
        if not rest:
            return

        anchor_lines = _comment_anchor_lines(value_node)
        key_line = key_node.start_mark.line
        if len(anchor_lines) == 1 and anchor_lines[0] == key_line:
            add_after(key_line, hash_col, rest)
            return

        if anchor_lines:
            for comment, line_idx in zip(rest, anchor_lines, strict=False):
                add_inline(line_idx, comment)
            overflow = rest[len(anchor_lines) :]
            if overflow:
                add_after(anchor_lines[-1], hash_col, overflow)
            return

        add_after(key_line, hash_col, rest)

    def on_mapping(node: MappingNode, path: YamlPath) -> None:
        description = description_map.get(path)
        if description is None:
            return

        class_description, parameter_descriptions = description
        header_entry = _mapping_entry(node, header_key)
        if header_entry is not None:
            header_node, _ = header_entry
            add_before(
                header_node.start_mark.line,
                _clean_comment_lines(class_description.splitlines()),
            )

        block_entry = _mapping_entry(node, block_key)
        if block_entry is None:
            return

        _, block_node = block_entry
        if not isinstance(block_node, MappingNode):
            return

        for key_node, value_node in block_node.value:
            if isinstance(key_node, ScalarNode):
                annotate_parameter(
                    key_node,
                    value_node,
                    parameter_descriptions.get(key_node.value, []),
                )

    _traverse_yaml_tree(root, (), on_mapping)
    return _render_commented_yaml(
        lines,
        before_comments=before_comments,
        inline_comments=inline_comments,
        after_comments=after_comments,
    )


def dump_yaml_with_comments(
    data: dict[str, Any],
    descriptions: YamlDescriptions | YamlDescriptionMap,
    *,
    indent: int = 2,
    comments: bool = True,
    header_key: str = "config_type",
    block_key: str = "parameters",
    flow_style_paths: Collection[YamlPath] = (),
) -> str:
    """Dump *data* to YAML and optionally inject comments from *descriptions*."""
    prepared_data = _apply_flow_style_paths(data, flow_style_paths=flow_style_paths)
    raw = yaml.dump(
        prepared_data, Dumper=_FlowStyleDumper, indent=indent, sort_keys=False
    )
    if not comments or not _normalize_description_map(descriptions):
        return raw
    return inject_yaml_comments(
        raw, descriptions, indent=indent, header_key=header_key, block_key=block_key
    )
