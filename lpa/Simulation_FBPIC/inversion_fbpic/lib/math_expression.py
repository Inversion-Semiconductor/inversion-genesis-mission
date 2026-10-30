"""Safe, declarative, serializable numerical functions.

Functions are stored as validated expressions rather than pickled callables.  This
makes configuration files portable and prevents deserialization from executing
arbitrary Python code.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence
from types import CodeType
from typing import Any, ClassVar

import attrs
import numpy as np
import numpy.typing as npt
import scipy

from inversion_fbpic.lib.serializable_config import SerializableConfig

# Only these module roots are visible to evaluated configuration expressions.
# Their public submodules and re-exported public attributes remain available.
_ALLOWED_MODULES = {
    "np": np,
    "numpy": np,
    "sp": scipy,
    "scipy": scipy,
    "mt": math,
    "math": math,
}
_CONSTANTS = {"e": math.e, "pi": math.pi, "i": 1.0j}
_ALLOWED_BINARY_OPERATORS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
)
_ALLOWED_UNARY_OPERATORS = (ast.UAdd, ast.USub, ast.Invert)
_ALLOWED_COMPARISON_OPERATORS = (
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def _variables(value: Sequence[str]) -> tuple[str, ...]:
    variables = tuple(value)
    if not variables:
        raise ValueError("variables must contain at least one name.")
    if len(set(variables)) != len(variables):
        raise ValueError("variables must be unique.")
    invalid = [
        name
        for name in variables
        if not isinstance(name, str) or not name.isidentifier()
    ]
    if invalid:
        raise ValueError(f"variables must be Python identifiers: {invalid!r}.")
    reserved = set(_ALLOWED_MODULES) | set(_CONSTANTS)
    if set(variables) & reserved:
        raise ValueError("variables cannot shadow supported modules or constants.")
    return variables


def _parameters(value: Mapping[str, float]) -> dict[str, float]:
    parameters = {str(name): float(number) for name, number in value.items()}
    invalid = [
        name
        for name, number in parameters.items()
        if not name.isidentifier() or not math.isfinite(number)
    ]
    if invalid:
        raise ValueError("parameters must have identifier names and finite values.")
    reserved = set(_ALLOWED_MODULES) | set(_CONSTANTS)
    if set(parameters) & reserved:
        raise ValueError("parameters cannot shadow supported modules or constants.")
    return parameters


def _is_public_module_attribute(node: ast.AST) -> bool:
    """Return whether *node* is a public attribute chain below an allowed root."""
    while isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            return False
        node = node.value
    return isinstance(node, ast.Name) and node.id in _ALLOWED_MODULES


def _validate_expression(expression: str, names: set[str]) -> CodeType:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("expression must be a valid Python expression.") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in names and node.id not in _ALLOWED_MODULES:
                raise ValueError(f"Unsupported name in expression: {node.id!r}.")
        elif isinstance(node, ast.Attribute):
            if not _is_public_module_attribute(node):
                raise ValueError("Only public NumPy or SciPy attributes are allowed.")
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("expression constants must be real numbers.")
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINARY_OPERATORS):
                raise ValueError("Unsupported binary operator in expression.")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARY_OPERATORS):
                raise ValueError("Unsupported unary operator in expression.")
        elif isinstance(node, ast.Compare):
            if not all(
                isinstance(op, _ALLOWED_COMPARISON_OPERATORS) for op in node.ops
            ):
                raise ValueError("Unsupported comparison operator in expression.")
        elif isinstance(node, ast.Call):
            if not _is_public_module_attribute(node.func):
                raise ValueError("Only public NumPy or SciPy functions may be called.")
        elif isinstance(node, ast.keyword):
            if node.arg is None or not node.arg.isidentifier():
                raise ValueError("Expression calls cannot unpack keyword arguments.")
        elif isinstance(
            node,
            (
                ast.Expression,
                ast.Load,
                ast.List,
                ast.Tuple,
                ast.Subscript,
                ast.Slice,
                *_ALLOWED_BINARY_OPERATORS,
                *_ALLOWED_UNARY_OPERATORS,
                *_ALLOWED_COMPARISON_OPERATORS,
            ),
        ):
            continue
        else:
            raise ValueError(f"Unsupported expression syntax: {type(node).__name__}.")
    return compile(tree, "<serializable-function>", "eval")


@attrs.define(kw_only=True, slots=False, frozen=True)
class _MathExpression(SerializableConfig):
    """Base class for serializable mathematical expressions."""

    CONFIG_TYPE: ClassVar[str] = "math_expression"
    _CONCRETE_REGISTRY: ClassVar[dict[str, type["_MathExpression"]]] = {}


@attrs.define(kw_only=True, slots=False, frozen=True)
class MathExpression(_MathExpression):
    """A validated multivariate numerical expression.

    Examples:
        A one-dimensional sine wave:

            MathExpression(expression="np.sin(z)", variables=("z",))

        A scaled Gaussian with serialized constants:

            MathExpression(
                expression="amplitude * np.exp(-((z - center) / width) ** 2)",
                variables=("z",),
                parameters={"amplitude": 1.0, "center": 0.0, "width": 2.0},
            )

        A two-dimensional profile with a radial cutoff:

            MathExpression(
                expression="np.where(r <= radius, np.exp(-(z / length) ** 2), 0)",
                variables=("z", "r"),
                parameters={"length": 1.0e-3, "radius": 50.0e-6},
            )

        A SciPy special function combined with a NumPy linear-algebra operation:

            MathExpression(
                expression="scipy.special.jv(order, z) * np.linalg.norm([z, r])",
                variables=("z", "r"),
                parameters={"order": 0.0},
            )

    Args:
        expression: (str) Numerical expression using variables, finite scalar parameters, ``pi``, ``e``, ``i``, and public NumPy/SciPy/math attributes under ``np``, ``numpy``/ ``sp``, ``scipy``/ ``mt``, ``math`` (for example ``np.exp(z)`` or ``sp.special.jv(0, z)``). Python attributes outside those module roots, private attributes, strings, and arbitrary Python callables are rejected.
        variables: (tuple[str, ...]) Ordered argument names. The function can be one-dimensional or multivariate.
        parameters: (dict[str, float]) Named finite scalar constants used by the expression. Defaults to an empty mapping.
    """

    SUBCLASS: ClassVar[str] = "math_expression"

    expression: str = attrs.field()
    variables: tuple[str, ...] = attrs.field(converter=_variables)
    parameters: dict[str, float] = attrs.field(factory=dict, converter=_parameters)
    _code: CodeType = attrs.field(init=False, repr=False, eq=False)

    def __attrs_post_init__(self) -> None:
        names = set(self.variables) | set(self.parameters) | set(_CONSTANTS)
        object.__setattr__(self, "_code", _validate_expression(self.expression, names))

    def __call__(self, *arguments: npt.ArrayLike) -> npt.ArrayLike:
        """Evaluate the expression for arguments in ``variables`` order."""
        if len(arguments) != len(self.variables):
            raise TypeError(
                f"Expected {len(self.variables)} arguments for {self.variables!r}; "
                f"received {len(arguments)}."
            )
        namespace: dict[str, Any] = {
            **_ALLOWED_MODULES,
            **_CONSTANTS,
            **self.parameters,
            **dict(zip(self.variables, arguments, strict=True)),
        }
        return eval(self._code, {"__builtins__": {}}, namespace)

    @classmethod
    def from_any(cls, value: "MathExpression | Mapping[str, Any]") -> "MathExpression":
        """Construct from an instance or its serialized configuration mapping."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            result = cls.from_dict(dict(value))
            if isinstance(result, cls):
                return result
        raise TypeError("function must be a MathExpression or serialized mapping.")
