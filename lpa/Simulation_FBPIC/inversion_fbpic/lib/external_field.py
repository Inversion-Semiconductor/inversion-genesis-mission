"""Serializable external fields applied to FBPIC particle species."""

from __future__ import annotations

import ast
from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import attrs

from fbpic.lpa_utils.external_fields import ExternalField as FBPICExternalField
from inversion_fbpic.lib.math_expression import MathExpression
from inversion_fbpic.lib.serializable_config import SerializableConfig

if TYPE_CHECKING:
    from fbpic.particles.particles import Particles

FieldType = Literal["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
_FIELD_TYPES = frozenset({"Ex", "Ey", "Ez", "Bx", "By", "Bz"})
_EXTERNAL_EXPRESSION_VARIABLES = frozenset({"x", "y", "z", "t"})
_RESERVED_CALLBACK_PARAMETERS = frozenset(
    {"gathered_field", "x", "y", "z", "t", "amplitude", "length_scale"}
)


def _field_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = dict(value)
    if not fields:
        raise ValueError("field_functions must contain at least one field type.")
    invalid = sorted(set(fields) - _FIELD_TYPES)
    if invalid:
        raise ValueError(
            f"field_functions contain unsupported field type(s): {invalid!r}."
        )
    return fields


def _uniform_field_mapping(value: Mapping[str, float]) -> dict[str, float]:
    return {name: float(strength) for name, strength in _field_mapping(value).items()}


def _expression_field_mapping(
    value: Mapping[str, MathExpression | Mapping[str, Any]],
) -> dict[str, MathExpression]:
    fields = {
        name: MathExpression.from_any(expression)
        for name, expression in _field_mapping(value).items()
    }
    for fieldtype, expression in fields.items():
        invalid = set(expression.variables) - _EXTERNAL_EXPRESSION_VARIABLES
        if invalid:
            raise ValueError(
                f"{fieldtype} expression variables must be drawn from "
                f"{tuple(sorted(_EXTERNAL_EXPRESSION_VARIABLES))!r}; got {sorted(invalid)!r}."
            )
        reserved = set(expression.parameters) & _RESERVED_CALLBACK_PARAMETERS
        if reserved:
            raise ValueError(
                f"{fieldtype} expression parameters cannot use FBPIC callback "
                f"name(s): {sorted(reserved)!r}."
            )
        unsupported = {
            node.id
            for node in ast.walk(ast.parse(expression.expression, mode="eval"))
            if isinstance(node, ast.Name)
            and node.id in {"sp", "scipy", "i"}  # Not numba compatible
        }
        if unsupported:
            raise ValueError(
                f"{fieldtype} expression cannot use Numba-incompatible "
                f"name(s): {sorted(unsupported)!r}."
            )
    return fields


@attrs.define(kw_only=True, slots=False, frozen=True)
class _ExternalField(SerializableConfig):
    """Base class for serializable FBPIC external fields.

    Field values and the ``x``, ``y``, ``z``, and ``t`` coordinates supplied
    to field functions are in the lab frame. For boosted simulations, FBPIC
    transforms the fields and coordinates to the boosted frame as needed.

    Args:
        species: (str|None) |OPTIONAL| Name configured for a particle diagnostic (`elec_name`, `ion_name`, or `particle_name`) to which the field applies. If None, applies to all particles. To target a species without retaining diagnostic particles, configure its corresponding selection to an impossible range, for example ``{"x": [1.0e12, None]}``. Defaults to None.
        override: (bool) |OPTIONAL| Whether to replace the field gathered from the simulation grid rather than add to it. Defaults to False.
    """

    CONFIG_TYPE: ClassVar[str] = "external_field"
    _CONCRETE_REGISTRY: ClassVar[dict[str, type["_ExternalField"]]] = {}

    species: str | None = attrs.field(default=None)
    override: bool = attrs.field(default=False, converter=bool)

    def build_external_fields(
        self,
        *,
        species: "Particles | None",
        gamma_boost: float | None,
    ) -> list[FBPICExternalField]:
        """Build one FBPIC field object per configured field type."""
        return [
            FBPICExternalField(
                self._build_field_function(
                    lab_expression, namespace, override=self.override
                ),
                fieldtype,
                1.0,
                1.0,
                species=species,
                gamma_boost=gamma_boost,
            )
            for fieldtype, (
                lab_expression,
                namespace,
            ) in self.build_lab_field_expressions().items()
        ]

    @staticmethod
    def _build_field_function(
        lab_expression: str,
        namespace: Mapping[str, Any],
        *,
        override: bool,
    ):
        """Apply FBPIC's amplitude around a lab-frame expression."""
        field_value = f"amplitude * ({lab_expression})"
        result = field_value if override else f"gathered_field + {field_value}"
        source = (
            "def field_function(gathered_field, x, y, z, t, amplitude, length_scale):\n"
            f"    return {result}\n"
        )
        globals_namespace = dict(namespace)
        exec(compile(source, "<external-field>", "exec"), globals_namespace)
        return globals_namespace["field_function"]

    @abstractmethod
    def build_lab_field_expressions(
        self,
    ) -> Mapping[FieldType, tuple[str, Mapping[str, Any]]]:
        """Return user-provided lab-frame expressions keyed by field type."""


@attrs.define(kw_only=True, slots=False, frozen=True)
class UniformExternalField(_ExternalField):
    """An external field with constant components.

    Args:
        field_functions: (dict[FieldType, float]) Field types mapped to their constant strengths.
    """

    SUBCLASS: ClassVar[str] = "uniform"

    field_functions: dict[str, float] = attrs.field(converter=_uniform_field_mapping)

    def build_lab_field_expressions(
        self,
    ) -> Mapping[FieldType, tuple[str, Mapping[str, Any]]]:
        return {
            fieldtype: (repr(strength), {})
            for fieldtype, strength in self.field_functions.items()
        }


@attrs.define(kw_only=True, slots=False, frozen=True)
class ExpressionExternalField(_ExternalField):
    """An external field with :class:`MathExpression` components.

    Expressions may use the lab-frame coordinates ``x``, ``y``, ``z``, and
    ``t`` and return lab-frame field contributions added to the field gathered
    from the simulation grid. FBPIC handles any required boosted-frame
    transformation. FBPIC compiles each expression with Numba for
    per-particle evaluation; use only scalar arithmetic, comparisons, and
    Numba-compatible ``np``/``numpy`` or ``math``/``mt`` functions. SciPy
    aliases (``sp`` and ``scipy``), the complex constant ``i``, list or tuple
    literals, and array-oriented routines such as ``np.linalg.norm`` are
    unsupported.

    Args:
        field_functions: (dict[FieldType, MathExpression]) Field types mapped to mathematical expressions. Expression parameters cannot be named ``amplitude``, ``length_scale``, ``x``, ``y``, ``z``, or ``t`` because FBPIC reserves those names for callback arguments.
    """

    SUBCLASS: ClassVar[str] = "expression"

    field_functions: dict[str, MathExpression] = attrs.field(
        converter=_expression_field_mapping
    )

    def build_lab_field_expressions(
        self,
    ) -> Mapping[FieldType, tuple[str, Mapping[str, Any]]]:
        return {
            fieldtype: (expression.expression, expression.compilation_namespace())
            for fieldtype, expression in self.field_functions.items()
        }
