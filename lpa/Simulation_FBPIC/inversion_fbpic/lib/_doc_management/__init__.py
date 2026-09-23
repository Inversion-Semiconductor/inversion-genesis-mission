"""Docstring parsing, merging, and config stub generation."""

from .index import LibIndex
from .merge import (
    build_merged_docstring,
    class_description,
    format_args_block,
    merge_parameter_descriptions,
    parameter_descriptions_from_doc,
)
from .render import generate_module_pyi
from .stub_sync import main, sync_config_stubs

__all__ = [
    "LibIndex",
    "build_merged_docstring",
    "class_description",
    "format_args_block",
    "generate_module_pyi",
    "main",
    "merge_parameter_descriptions",
    "parameter_descriptions_from_doc",
    "sync_config_stubs",
]
