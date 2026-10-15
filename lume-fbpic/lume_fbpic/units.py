"""The units of lume_fbpic's variables, used as the default `unit` of an action.

An action on a config field takes the unit tagged in brackets in that field's documentation
(`config_unit()`); a statistic of `stats` its entry in `STAT_UNITS`; a moment-descriptor feature
its entry in `lume_fbpic.descriptor.features()`. Every unit is a symbol `beamphysics` knows
(`is_unit()`).
"""

from __future__ import annotations

import re
import typing

from beamphysics.units import pmd_unit

from lume_fbpic.utils import STATISTICS

# Units of the statistics a simulator keeps in `stats`, by name.
STAT_UNITS = {name: statistic.unit for name, statistic in STATISTICS.items()}

# A tag in brackets in a config field's documentation: a unit (`[m]`), or a type or the allowed
# values (`[str]`, `["left", "right"]`), which `config_unit()` skips.
UNIT_TAG = re.compile(r"\[([^\]]+)\]")


def config_unit(config: typing.Any, field_name: str) -> str | None:
    """The unit tagged in brackets in the documentation of field `field_name` of `config`.

    The documentation of a config class lists each field as `name: (type) [unit] description`;
    the tags that are not units (`[str]`, `[int]`) are skipped. The text is that of
    `SerializableConfig._parameter_descriptions()`, which also holds the fields a class
    inherits. None if the field has no unit tag, or the class has no such documentation.
    """
    descriptions = getattr(type(config), "_parameter_descriptions", None)
    if descriptions is None:
        return None
    text = " ".join(descriptions().get(field_name, []))
    for tag in UNIT_TAG.findall(text):
        if is_unit(tag):
            return tag
    return None


def is_unit(text: str) -> bool:
    """Whether `text` is a unit symbol `beamphysics` knows."""
    try:
        pmd_unit(text)
    except ValueError:
        return False
    return True
