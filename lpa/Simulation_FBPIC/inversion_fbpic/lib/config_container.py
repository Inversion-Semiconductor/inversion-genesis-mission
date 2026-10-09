"""An ordered serializable container for heterogeneous configuration objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import attrs

from inversion_fbpic.lib.serializable_config import SerializableConfig


@attrs.define(kw_only=True, slots=False)
class ConfigContainer(SerializableConfig):
    """Manage an ordered collection of configuration references.

    Construction resolves entries using ``SerializableConfig.from_any()``.
    Existing objects are retained by identity; payloads, serialized strings,
    and file references are loaded into objects. Relative file references use
    the containing config file or the active path-resolution context.
    Without either anchor, paths resolve against the construction-time working
    directory. Passing the container to a Simulation does not re-resolve them;
    use ``resolving_paths_relative_to()`` around construction to choose a base.

    Serialization embeds the resolved configurations inline, not as external
    references. Order and repeated entries are retained, but shared identity
    does not survive saving and reloading. Nested containers are supported
    without flattening; directory scanning and execution are not performed.

    Args:
        configs: (list[SerializableConfig | str | Path | dict[str, Any]])
            |OPTIONAL| Config objects, tagged payloads, YAML/JSON strings, or
            individual config file references. Defaults to a new empty list.
            After construction, add only live configuration objects to this list.
    """

    CONFIG_TYPE: ClassVar[str] = "config_container"
    _CONCRETE_REGISTRY: ClassVar[dict[str, type["ConfigContainer"]]] = {}
    SUBCLASS: ClassVar[str] = "config_container"

    configs: list[SerializableConfig | str | Path | dict[str, Any]] = attrs.field(
        factory=list, validator=attrs.validators.instance_of(list)
    )

    def __attrs_post_init__(self) -> None:
        resolved: list[SerializableConfig | str | Path | dict[str, Any]] = []
        for index, source in enumerate(self.configs):
            try:
                resolved.append(SerializableConfig.from_any(source))
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"Invalid config at configs[{index}]: {exc}") from exc
        self.configs = resolved
