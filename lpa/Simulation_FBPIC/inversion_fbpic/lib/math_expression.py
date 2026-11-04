"""Safe, declarative, serializable numerical functions."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence
from types import CodeType, SimpleNamespace
from typing import Any, ClassVar

import attrs
import numpy as np
import numpy.typing as npt
import scipy.special

from inversion_fbpic.lib.serializable_config import SerializableConfig

# Evaluation exposes exact numerical symbols, never the real third-party modules.
# NumPy/SciPy contain public I/O, pickle, FFI, and mutable-state APIs; exposing a
# module object would make every one of those APIs reachable from a config.
_NUMPY_FUNCTION_NAMES = (
    "all",
    "allclose",
    "any",
    "append",
    "arange",
    "argmax",
    "argmin",
    "argsort",
    "array",
    "array_equal",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "average",
    "broadcast_arrays",
    "column_stack",
    "concatenate",
    "copy",
    "corrcoef",
    "cov",
    "cross",
    "cumprod",
    "cumsum",
    "diag",
    "diagonal",
    "diff",
    "digitize",
    "dot",
    "einsum",
    "expand_dims",
    "flip",
    "gradient",
    "histogram",
    "hsplit",
    "hstack",
    "inner",
    "interp",
    "linspace",
    "logspace",
    "mean",
    "median",
    "meshgrid",
    "moveaxis",
    "nanmean",
    "nanmedian",
    "nanprod",
    "nanstd",
    "nansum",
    "nanvar",
    "outer",
    "percentile",
    "prod",
    "ptp",
    "quantile",
    "ravel",
    "repeat",
    "reshape",
    "roll",
    "rot90",
    "searchsorted",
    "select",
    "shape",
    "sort",
    "split",
    "squeeze",
    "stack",
    "std",
    "sum",
    "take",
    "tensordot",
    "tile",
    "trace",
    "transpose",
    "trapezoid",
    "unique",
    "var",
    "vstack",
    "where",
)
_NUMPY_LINALG_FUNCTION_NAMES = (
    "cholesky",
    "cond",
    "det",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "inv",
    "lstsq",
    "matrix_power",
    "matrix_rank",
    "multi_dot",
    "norm",
    "pinv",
    "qr",
    "slogdet",
    "solve",
    "svd",
    "tensorinv",
    "tensorsolve",
)
_SCIPY_SPECIAL_FUNCTION_NAMES = ("log_softmax", "logsumexp", "softmax")


def _public_ufuncs(namespace: Mapping[str, Any]) -> dict[str, np.ufunc]:
    """Return numerical ufuncs only; ufuncs cannot load files or execute Python."""
    return {
        name: value
        for name, value in namespace.items()
        if not name.startswith("_") and isinstance(value, np.ufunc)
    }


def _named_callables(namespace: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """Return explicitly reviewed callables that exist in the installed version."""
    return {
        name: value
        for name in names
        if callable(value := getattr(namespace, name, None))
    }


def _symbol_aliases(
    prefixes: tuple[str, ...], symbols: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply every supported import-style alias to a symbol mapping."""
    return {
        f"{prefix}.{name}": value
        for prefix in prefixes
        for name, value in symbols.items()
    }


_NUMPY_SYMBOLS = {
    **_public_ufuncs(vars(np)),
    **_named_callables(np, _NUMPY_FUNCTION_NAMES),
}
_NUMPY_LINALG_SYMBOLS = _named_callables(np.linalg, _NUMPY_LINALG_FUNCTION_NAMES)
_SCIPY_SPECIAL_SYMBOLS = {
    **_public_ufuncs(vars(scipy.special)),
    **_named_callables(scipy.special, _SCIPY_SPECIAL_FUNCTION_NAMES),
}
_MATH_SYMBOLS = {
    name: value
    for name in dir(math)
    if not name.startswith("_") and callable(value := getattr(math, name))
}
_ALLOWED_SYMBOLS: dict[str, Any] = {
    **_symbol_aliases(("np", "numpy"), _NUMPY_SYMBOLS),
    **_symbol_aliases(("np.linalg", "numpy.linalg"), _NUMPY_LINALG_SYMBOLS),
    **_symbol_aliases(("sp.special", "scipy.special"), _SCIPY_SPECIAL_SYMBOLS),
    **_symbol_aliases(("mt", "math"), _MATH_SYMBOLS),
    "e": np.e,
    "pi": np.pi,
    "i": 1.0j,
    "nan": np.nan,
    "inf": np.inf,
}
_ALLOWED_CALLABLES = {
    path: value for path, value in _ALLOWED_SYMBOLS.items() if callable(value)
}
_ALLOWED_MODULES = {"np", "numpy", "sp", "scipy", "mt", "math"}
_ALLOWED_CONSTANTS = {
    path: value for path, value in _ALLOWED_SYMBOLS.items() if "." not in path
}


def _build_namespace(symbols: Mapping[str, Any]) -> dict[str, SimpleNamespace]:
    """Build read-only-by-validation namespace proxies for dotted safe symbols."""
    roots: dict[str, SimpleNamespace] = {}
    for path, value in symbols.items():
        root, *parts = path.split(".")
        if not parts:
            continue
        current = roots.setdefault(root, SimpleNamespace())
        for part in parts[:-1]:
            child = getattr(current, part, None)
            if child is None:
                child = SimpleNamespace()
                setattr(current, part, child)
            current = child
        setattr(current, parts[-1], value)
    return roots


_ALLOWED_NAMESPACES = _build_namespace(_ALLOWED_SYMBOLS)
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
    reserved = set(_ALLOWED_MODULES) | set(_ALLOWED_CONSTANTS)
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
    reserved = set(_ALLOWED_MODULES) | set(_ALLOWED_CONSTANTS)
    if set(parameters) & reserved:
        raise ValueError("parameters cannot shadow supported modules or constants.")
    return parameters


def _attribute_path(node: ast.AST) -> str | None:
    """Return a dotted attribute path rooted at an expression namespace."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join((node.id, *reversed(parts)))


def _is_allowed_symbol(path: str) -> bool:
    """Return whether *path* is a symbol or namespace prefix in the allowlist."""
    return path in _ALLOWED_SYMBOLS or any(
        symbol.startswith(f"{path}.") for symbol in _ALLOWED_SYMBOLS
    )


def _validate_expression(expression: str, names: set[str]) -> CodeType:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("expression must be a valid Python expression.") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if (
                node.id not in names
                and node.id not in _ALLOWED_MODULES
                and node.id not in _ALLOWED_CONSTANTS
            ):
                raise ValueError(f"Unsupported name in expression: {node.id!r}.")
        elif isinstance(node, ast.Attribute):
            path = _attribute_path(node)
            if path is None or not _is_allowed_symbol(path):
                raise ValueError(f"Unsupported numerical symbol: {path!r}.")
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
            path = _attribute_path(node.func)
            if path not in _ALLOWED_CALLABLES:
                raise ValueError(
                    f"Only allowlisted numerical functions may be called: {path!r}."
                )
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
        expression: (str) Numerical expression using variables, finite scalar parameters, ``pi``, ``e``, ``i``, and allowlisted NumPy, SciPy-special, or math symbols under ``np``, ``numpy`` / ``sp``, ``scipy`` / ``mt``, ``math`` (for example ``np.exp(z)``, ``sp.special.jv(0, z)``, or ``math.log(z)``). File I/O, pickle loading, FFI, random-state, and unregistered callables are rejected.
        variables: (tuple[str, ...]) Ordered argument names. The function can be one-dimensional or multivariate.
        parameters: (dict[str, float]) Named finite scalar constants used by the expression. Defaults to an empty mapping.
    """

    SUBCLASS: ClassVar[str] = "math_expression"

    expression: str = attrs.field()
    variables: tuple[str, ...] = attrs.field(converter=_variables)
    parameters: dict[str, float] = attrs.field(factory=dict, converter=_parameters)
    _code: CodeType = attrs.field(init=False, repr=False, eq=False)

    def __attrs_post_init__(self) -> None:
        names = set(self.variables) | set(self.parameters) | set(_ALLOWED_CONSTANTS)
        object.__setattr__(self, "_code", _validate_expression(self.expression, names))

    def __call__(self, *arguments: npt.ArrayLike) -> npt.ArrayLike:
        """Evaluate the expression for arguments in ``variables`` order."""
        if len(arguments) != len(self.variables):
            raise TypeError(
                f"Expected {len(self.variables)} arguments for {self.variables!r}; "
                f"received {len(arguments)}."
            )
        namespace: dict[str, Any] = {
            **_ALLOWED_NAMESPACES,
            **_ALLOWED_CONSTANTS,
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
