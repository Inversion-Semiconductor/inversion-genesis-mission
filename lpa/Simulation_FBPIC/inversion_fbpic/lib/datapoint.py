"""Serializable dictionaries for curated inputs and post-analysis results."""

from __future__ import annotations

from typing import Any, ClassVar

import attrs

from inversion_fbpic.lib.serializable_config import SerializableConfig


@attrs.define(kw_only=True, slots=False)
class _Datapoint(SerializableConfig):
    """Shared configuration domain for dictionary-backed datapoints.

    Use :class:`Parameters` for curated inputs or implement a concrete
    :class:`~inversion_fbpic.lib.diagnostics._Diagnostic` subclass for analysis
    results. This domain base has
    no concrete serialization tag of its own.

    Args:
        data: (dict[str, Any]) |OPTIONAL| Stored values. Defaults to a new empty
            dictionary. Values use the standard configuration serialization
            conversions; arbitrary dictionaries are not interpreted as configs.
    """

    CONFIG_TYPE: ClassVar[str] = "datapoint"
    _CONCRETE_REGISTRY: ClassVar[dict[str, type["_Datapoint"]]] = {}

    data: dict[str, Any] = attrs.field(
        factory=dict, validator=attrs.validators.instance_of(dict)
    )


@attrs.define(kw_only=True, slots=False)
class Parameters(_Datapoint):
    """Store a curated set of user-provided input parameters.

    The supplied dictionary is retained by identity. Mutations through
    ``data`` or the original dictionary are visible to both callers. Access
    values through ``data`` rather than through configuration attributes.

    Args:
        data: (dict[str, Any]) The user-provided dictionary, retained without
            copying. Required for Python construction; omitted serialized
            dictionaries use the standard empty-dictionary loading fallback.
    """

    SUBCLASS: ClassVar[str] = "parameters"

    data: dict[str, Any] = attrs.field(validator=attrs.validators.instance_of(dict))
