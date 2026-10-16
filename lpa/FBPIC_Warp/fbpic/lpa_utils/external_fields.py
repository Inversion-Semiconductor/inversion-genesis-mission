# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
import ast
import inspect
import textwrap
import types

from numba import vectorize, float64, njit
from scipy.constants import c
import numpy as np

# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed

inv_c = 1.0 / c

if cuda_installed:
    from fbpic.utils.cuda import cuda_tpb_bpg_1d
    from fbpic.utils.warp import warp_kernel
    import warp as wp
    from . import external_field_math as _field_math


class _WarpFieldSource(ast.NodeTransformer):
    """Translate scalar callback syntax while preserving double precision."""

    def __init__(self, namespace):
        self.namespace = namespace
        self.integer_names = set()
        self.tables = {}
        self.table_values = []
        self.local_constants = {}
        self.variable_types = {}
        self.local_names = set()

    @staticmethod
    def _call(name, *args):
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="wp", ctx=ast.Load()), attr=name, ctx=ast.Load()
            ),
            args=list(args),
            keywords=[],
        )

    def _double(self, node):
        return self._call("float64", node)

    def _resolve(self, node):
        if isinstance(node, ast.Name):
            return self.local_constants.get(node.id, self.namespace.get(node.id))
        if isinstance(node, ast.Attribute):
            value = self._resolve(node.value)
            return getattr(value, node.attr, None)
        return None

    def _table_and_indices(self, node):
        indices = []
        base = node
        while isinstance(base, ast.Subscript):
            current = (
                list(base.slice.elts)
                if isinstance(base.slice, ast.Tuple)
                else [base.slice]
            )
            indices = current + indices
            base = base.value
        value = self._resolve(base)
        if value is None and isinstance(base, (ast.Tuple, ast.List)):
            try:
                value = ast.literal_eval(base)
            except (ValueError, TypeError):
                pass
        if not isinstance(value, (tuple, list, np.ndarray)):
            raise ValueError(
                "ExternalField lookup data must be a constant tuple or NumPy array."
            )
        return base, np.asarray(value), indices

    @staticmethod
    def _promote(types, arithmetic=False):
        types = [dtype for dtype in types if dtype is not None]
        if not types:
            return None
        if "float64" in types:
            return "float64"
        if "float32" in types:
            return (
                "float32" if all(dtype == "float32" for dtype in types) else "float64"
            )
        if arithmetic:
            return (
                "uint64"
                if all(dtype.startswith("uint") for dtype in types)
                else "int64"
            )
        return np.result_type(*[np.dtype(dtype) for dtype in types]).name

    def _type(self, node):
        if isinstance(node, ast.Constant):
            return "float64" if isinstance(node.value, float) else "int64"
        if isinstance(node, ast.Name):
            if node.id in self.variable_types:
                return self.variable_types[node.id]
            if node.id in self.local_names:
                return None
            value = self._resolve(node)
            return "int64" if isinstance(value, (int, np.integer)) else "float64"
        if isinstance(node, ast.Subscript):
            _, table, _ = self._table_and_indices(node)
            return table.dtype.name
        if isinstance(node, ast.UnaryOp):
            return self._type(node.operand)
        if isinstance(node, ast.IfExp):
            return self._promote([self._type(node.body), self._type(node.orelse)])
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                try:
                    value = ast.literal_eval(node.left) ** ast.literal_eval(node.right)
                except (ValueError, TypeError, OverflowError, ZeroDivisionError):
                    value = None
                if isinstance(value, float):
                    return "float64"
            if isinstance(node.op, ast.Div):
                return "float64"
            return self._promote(
                [self._type(node.left), self._type(node.right)], arithmetic=True
            )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "int":
                return "int64"
            value = self._resolve(node.func)
            if getattr(value, "__name__", None) in {"floor", "ceil", "trunc"}:
                return "int64"
            if isinstance(node.func, ast.Name) and node.func.id in {
                "abs",
                "min",
                "max",
            }:
                return self._promote([self._type(arg) for arg in node.args])
        return "float64"

    def visit_Constant(self, node):
        if isinstance(node.value, (float, int)) and not isinstance(node.value, bool):
            return self._call(self._type(node), node)
        return node

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            return node
        value = self.namespace.get(node.id)
        if isinstance(value, (float, int, np.floating, np.integer)):
            literal = (
                int(value) if isinstance(value, (int, np.integer)) else float(value)
            )
            return self._call(self._type(node), ast.Constant(value=literal))
        if node.id in self.integer_names:
            return self._call("int64", node)
        return node

    def visit_Attribute(self, node):
        value = self._resolve(node)
        if isinstance(value, (float, int, np.floating, np.integer)):
            return self._call(
                "int64" if isinstance(value, (int, np.integer)) else "float64",
                ast.Constant(
                    value=(
                        int(value)
                        if isinstance(value, (int, np.integer))
                        else float(value)
                    )
                ),
            )
        return self.generic_visit(node)

    @staticmethod
    def _math_call(name, *args):
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="_field_math", ctx=ast.Load()),
                attr=name,
                ctx=ast.Load(),
            ),
            args=list(args),
            keywords=[],
        )

    def visit_BinOp(self, node):
        dtype = self._type(node)
        if isinstance(node.op, ast.Pow):
            try:
                value = ast.literal_eval(node.left) ** ast.literal_eval(node.right)
            except (ValueError, TypeError, OverflowError, ZeroDivisionError):
                value = None
            if isinstance(value, (int, float)):
                return self.visit_Constant(ast.Constant(value=value))
        node.left = self._call(dtype, self.visit(node.left))
        node.right = self._call(dtype, self.visit(node.right))
        if isinstance(node.op, ast.Pow):
            if dtype in {"int64", "uint64"}:
                return self._math_call("integer_power_" + dtype, node.left, node.right)
            return self._call(
                dtype,
                self._call("pow", self._double(node.left), self._double(node.right)),
            )
        if isinstance(node.op, (ast.FloorDiv, ast.Mod)) and dtype in {
            "float64",
            "int64",
        }:
            operation = (
                "floor_divide_" if isinstance(node.op, ast.FloorDiv) else "modulo_"
            )
            return self._math_call(operation + dtype, node.left, node.right)
        return node

    def visit_IfExp(self, node):
        dtype = self._type(node)
        node.test = self.visit(node.test)
        node.body = self._call(dtype, self.visit(node.body))
        node.orelse = self._call(dtype, self.visit(node.orelse))
        return node

    def visit_Compare(self, node):
        dtype = self._promote(
            [self._type(value) for value in [node.left, *node.comparators]]
        )
        node.left = self._call(dtype, self.visit(node.left))
        node.comparators = [
            self._call(dtype, self.visit(value)) for value in node.comparators
        ]
        return node

    def visit_Return(self, node):
        node.value = self._double(self.visit(node.value))
        return node

    def visit_Call(self, node):
        value = self._resolve(node.func)
        if getattr(value, "__module__", None) == "math":
            name = value.__name__
            if name == "fabs":
                name = "abs"
            if hasattr(_field_math, name):
                args = [self._double(self.visit(arg)) for arg in node.args]
                if name == "ldexp":
                    args[1] = self._call("int32", args[1])
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="_field_math", ctx=ast.Load()),
                        attr=name,
                        ctx=ast.Load(),
                    ),
                    args=args,
                    keywords=[],
                )
            if not hasattr(wp, name):
                raise ValueError(
                    f"ExternalField callback uses math.{value.__name__}, "
                    "which Warp 1.17 cannot compile."
                )
            args = [
                (
                    self.visit(arg)
                    if self._type(arg).startswith("float")
                    else self._double(self.visit(arg))
                )
                for arg in node.args
            ]
            result = self._call(name, *args)
            if name in {"floor", "ceil", "trunc"}:
                return self._call("int64", result)
            return result
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "range":
                return ast.Call(
                    func=node.func,
                    args=[self._call("int32", self.visit(arg)) for arg in node.args],
                    keywords=[],
                )
            if name in {"int", "float", "bool"}:
                target = {"int": "int64", "float": "float64", "bool": "bool"}[name]
                return self._call(target, *[self.visit(arg) for arg in node.args])
            if name in {"abs", "min", "max"}:
                dtype = self._type(node)
                return ast.Call(
                    func=node.func,
                    args=[self._call(dtype, self.visit(arg)) for arg in node.args],
                    keywords=[],
                )
        if isinstance(value, wp.Function):
            return self.generic_visit(node)
        raise ValueError(
            "ExternalField callback contains an unsupported call: "
            f"{ast.unparse(node.func)}. Use scalar math functions."
        )

    def visit_For(self, node):
        if not isinstance(node.target, ast.Name):
            raise ValueError("ExternalField callback loops need a scalar index.")
        self.integer_names.add(node.target.id)
        self.variable_types[node.target.id] = "int64"
        return self.generic_visit(node)

    def visit_Assign(self, node):
        # Constant local tuples use the same read-only table representation.
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                value = None
            if isinstance(value, (tuple, list)):
                self.local_constants[node.targets[0].id] = value
                return ast.Pass()
        if isinstance(node.value, ast.Call):
            value = self._resolve(node.value.func)
            if getattr(value, "__name__", None) == "frexp":
                if isinstance(node.targets[0], ast.Tuple):
                    self.integer_names.add(node.targets[0].elts[1].id)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            dtype = self.variable_types[node.targets[0].id]
            node.value = self._call(dtype, self.visit(node.value))
            return node
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        dtype = self.variable_types[node.target.id]
        node.value = self._call(dtype, self.visit(node.value))
        return node

    def visit_Subscript(self, node):
        base, table, indices = self._table_and_indices(node)
        if table.dtype.kind not in "biuf" or table.ndim != len(indices):
            raise ValueError(
                "ExternalField lookup indices must select one real numeric value."
            )
        key = (ast.unparse(base), table.dtype.str, table.shape, table.tobytes())
        if key not in self.tables:
            self.tables[key] = len(self.table_values)
            flat = np.ascontiguousarray(table).ravel()
            if table.dtype.kind == "f":
                bits = flat.view(np.uint64 if table.dtype.itemsize == 8 else np.uint32)
            else:
                bits = flat.astype(np.uint64)
            self.table_values.extend(bits.astype(np.uint64).tolist())
        offset = ast.Constant(value=self.tables[key])
        for axis, index in enumerate(indices):
            if isinstance(index, ast.Slice):
                raise ValueError(
                    "ExternalField lookup expressions require scalar indices."
                )
            selected = self._call("int32", self.visit(index))
            wrapped = ast.IfExp(
                test=ast.Compare(
                    left=selected, ops=[ast.Lt()], comparators=[ast.Constant(value=0)]
                ),
                body=ast.BinOp(
                    left=selected,
                    op=ast.Add(),
                    right=ast.Constant(value=table.shape[axis]),
                ),
                orelse=selected,
            )
            stride = int(np.prod(table.shape[axis + 1 :]))
            offset = ast.BinOp(
                left=offset,
                op=ast.Add(),
                right=ast.BinOp(
                    left=wrapped, op=ast.Mult(), right=ast.Constant(value=stride)
                ),
            )
        value = ast.Subscript(
            value=ast.Name(id="_fbpic_lookup", ctx=ast.Load()),
            slice=offset,
            ctx=ast.Load(),
        )
        if table.dtype.kind == "f":
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_field_math", ctx=ast.Load()),
                    attr=table.dtype.name + "_from_bits",
                    ctx=ast.Load(),
                ),
                args=[value],
                keywords=[],
            )
        return self._call(
            "bool" if table.dtype.kind == "b" else table.dtype.name, value
        )

    def _infer_variable_types(self, function):
        self.local_names = {
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        self.variable_types.update({arg.arg: "float64" for arg in function.args.args})
        for statement in ast.walk(function):
            if isinstance(statement, ast.For) and isinstance(
                statement.target, ast.Name
            ):
                self.variable_types[statement.target.id] = "int64"
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name):
                    try:
                        value = ast.literal_eval(statement.value)
                    except (ValueError, TypeError):
                        value = None
                    if isinstance(value, (tuple, list)):
                        self.local_constants[target.id] = value
        # The final local type must include loop and branch assignments.
        # This lets Warp initialize a promoted accumulator with the correct type.
        for _ in range(len(self.local_names) + 1):
            previous = dict(self.variable_types)
            for statement in ast.walk(function):
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                    target = statement.targets[0]
                    if (
                        isinstance(target, ast.Name)
                        and target.id not in self.local_constants
                    ):
                        dtype = self._type(statement.value)
                        self.variable_types[target.id] = self._promote(
                            [self.variable_types.get(target.id), dtype]
                        )
                elif isinstance(statement, ast.AugAssign) and isinstance(
                    statement.target, ast.Name
                ):
                    name = statement.target.id
                    dtype = (
                        "float64"
                        if isinstance(statement.op, ast.Div)
                        else self._promote(
                            [
                                self.variable_types.get(name),
                                self._type(statement.value),
                            ],
                            arithmetic=True,
                        )
                    )
                    self.variable_types[name] = self._promote(
                        [self.variable_types.get(name), dtype]
                    )
            if previous == self.variable_types:
                break
        self.local_constants.clear()

    def visit_FunctionDef(self, node):
        if any(
            isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            for stmt in node.body
        ):
            raise ValueError(
                "ExternalField GPU callbacks cannot define nested functions."
            )
        self._infer_variable_types(node)
        node.decorator_list = []
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        node.body = [self.visit(stmt) for stmt in node.body]
        return node


def _compile_warp_field(field_func, gamma_boost):
    """Compile a scalar Python callback and its particle kernel for Warp."""
    source = getattr(field_func, "__fbpic_source__", None)
    if source is None:
        try:
            source = inspect.getsource(field_func)
        except (OSError, TypeError) as exc:
            raise ValueError(
                "ExternalField requires inspectable Python source for GPU callbacks. "
                "Define the callback in a Python file, or provide __fbpic_source__."
            ) from exc
    tree = ast.parse(textwrap.dedent(source))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if field_func.__name__ == "<lambda>":
        lambdas = [node for node in ast.walk(tree) if isinstance(node, ast.Lambda)]
        if len(lambdas) != 1:
            raise ValueError(
                "Place each ExternalField lambda on a separate source line."
            )
        function = ast.FunctionDef(
            name="external_field_lambda",
            args=lambdas[0].args,
            body=[ast.Return(value=lambdas[0].body)],
            decorator_list=[],
            returns=None,
        )
        tree = ast.Module(body=[function], type_ignores=[])
    elif len(functions) == 1:
        function = functions[0]
    else:
        raise ValueError(
            "ExternalField GPU callbacks require one Python function definition."
        )
    if (
        len(function.args.args) != 7
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or function.args.kwonlyargs
    ):
        raise ValueError("ExternalField callbacks require seven scalar arguments.")
    closure = inspect.getclosurevars(field_func)
    namespace = dict(field_func.__globals__)
    namespace.update(closure.globals)
    namespace.update(closure.nonlocals)
    # Formal arguments and local variables shadow globals with the same name.
    for arg in function.args.args:
        namespace.pop(arg.arg, None)
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            namespace.pop(node.id, None)
    transformer = _WarpFieldSource(namespace)
    tree = transformer.visit(tree)
    function.args.args.append(ast.arg(arg="_fbpic_lookup"))
    source = ast.unparse(ast.fix_missing_locations(tree))
    globals_copy = dict(field_func.__globals__)
    globals_copy.update(closure.nonlocals)
    globals_copy["wp"] = wp
    globals_copy["_field_math"] = _field_math
    typed_func = types.FunctionType(
        field_func.__code__,
        globals_copy,
        field_func.__name__,
        field_func.__defaults__,
        field_func.__closure__,
    )
    inline_func = wp.Function(
        func=typed_func,
        key="external_field_callback",
        namespace="",
        module=wp.Module("external_field_callback", None),
        source=source,
        overloaded_annotations={
            **{arg.arg: wp.float64 for arg in function.args.args[:-1]},
            "_fbpic_lookup": wp.array(dtype=wp.uint64),
        },
    )
    gamma = wp.constant(wp.float64(1.0 if gamma_boost is None else gamma_boost))
    beta = wp.constant(
        wp.float64(0.0 if gamma_boost is None else np.sqrt(1.0 - 1.0 / gamma_boost**2))
    )
    light_speed = wp.constant(wp.float64(c))
    inverse_light_speed = wp.constant(wp.float64(inv_c))

    def external_field_kernel(
        F: wp.array(dtype=wp.float64),
        x: wp.array(dtype=wp.float64),
        y: wp.array(dtype=wp.float64),
        z: wp.array(dtype=wp.float64),
        t: wp.float64,
        amplitude: wp.float64,
        length_scale: wp.float64,
        lookup: wp.array(dtype=wp.uint64),
    ):
        i = wp.tid()
        if i < F.shape[0]:
            zlab = gamma * (z[i] + beta * light_speed * t)
            tlab = gamma * (t + beta * inverse_light_speed * z[i])
            F[i] = inline_func(
                F[i], x[i], y[i], zlab, tlab, amplitude, length_scale, lookup
            )

    return _GpuExternalField(
        warp_kernel(external_field_kernel, module="unique"), transformer.table_values
    )


class _GpuExternalField:
    """Keep constant lookup data alive beside the compiled callback kernel."""

    def __init__(self, kernel, table_values):
        self._kernel = kernel
        self.kernel = kernel.kernel
        self.table_values = np.asarray(table_values, dtype=np.uint64)
        self.table_data = None

    def __getitem__(self, launch_shape):
        def launch(*args):
            if self.table_data is None:
                import cupy

                self.table_data = cupy.asarray(self.table_values)
            self._kernel[launch_shape](*args, self.table_data)

        return launch


def _snapshot_callback(function):
    """Keep the baseline JIT timing for globals and closure constants."""

    def snapshot(value):
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, tuple):
            return tuple(snapshot(item) for item in value)
        if isinstance(value, list):
            return [snapshot(item) for item in value]
        return value

    closure = inspect.getclosurevars(function)
    namespace = dict(function.__globals__)
    namespace.update({name: snapshot(value) for name, value in closure.globals.items()})
    cells = None
    if function.__closure__ is not None:
        cells = tuple(
            types.CellType(snapshot(cell.cell_contents))
            for cell in function.__closure__
        )
    result = types.FunctionType(
        function.__code__, namespace, function.__name__, function.__defaults__, cells
    )
    if hasattr(function, "__fbpic_source__"):
        result.__fbpic_source__ = function.__fbpic_source__
    return result


class ExternalField(object):

    def __init__(
        self,
        field_func,
        fieldtype,
        amplitude,
        length_scale,
        species=None,
        gamma_boost=None,
    ):
        """
        Initialize an ExternalField object, so that the function
        `field_func` is called at each time step on the field `fieldtype`

        This object should be added to the list `external_fields`,
        which is an attribute of the Simulation object, so that the
        fields are applied at each timestep. (See the example below)

        The function `field_func` is automatically converted to a GPU
        function if needed, by using Warp. The CPU path uses Numba.

        Parameters
        ----------
        field_func: callable
            Function of the form `field_func( F, x, y, z, t, amplitude,
            length_scale )` and which returns the modified field F'
            (in the lab frame)

            This function will be called at each timestep, with:

            - F: 1d array of shape (N_ptcl,), containing the field
              designated by fieldtype, gathered on the particles
            - x, y, z: 1d arrays of shape (N_ptcl), containing the
              positions of the particles (in the lab frame)
            - t: float, the time in the simulation (in the lab frame)
            - amplitude and length_scale: floats that can be used within
              the function expression

            .. warning::
                In the PIC loop, this function is called after
                the field gathering. Thus this function can potentially
                overwrite the fields that were gathered on the grid. To avoid
                this, use "return(F + external_field) " inside the definition
                of `field_func` instead of "return(external_field)"

            .. warning::
                Inside the definition of `field_func` please use
                the `math` module for mathematical functions, instead of numpy.
                This will allow the function to be compiled for GPU.

        fieldtype: string
            Specifies on which field `field_func` will be applied.
            Either 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'

        species: a Particles object, optionals
            The species on which the external field has to be applied.
            If no species is specified, the external field is applied
            to all particles.

        gamma_boost: float, optional
            When running the simulation in a boosted frame, set the
            value of `gamma_boost` to the corresponding Lorentz factor.
            The external fields will be automatically converted to the
            boosted frame.

        Example
        -------
        In order to define a magnetic undulator, polarized along y, with
        a field of 1 Tesla and a period of 1 cm :

        ::

            def field_func( F, x, y, z, t , amplitude, length_scale ):
                return( F + amplitude * math.cos( 2*np.pi*z/length_scale ) )

            sim.external_fields = [ ExternalField( field_func, 'By', 1., 1.e-2 ) ]

        .. warning::

            Note that, in principle, ``field_func`` does not necessarily need
            to use the arguments ``amplitude`` and ``length_scale``.
            For instance, in the above example, we could have used

            ::

                def field_func( F, x, y, z, t , amplitude, length_scale ):
                    return( F + 1. * math.cos( 2*np.pi*z/1.e-2 ) )

            However, **when running the simulation in a boosted frame**
            (i.e. when setting the above argument ``gamma_boost``),
            the expression of the external fields **needs to be proportional
            to** ``amplitude``. This is because, internally, the automatic
            conversion of the external fields to the boosted frame relies
            on this variable. (There is no similar constraint for
            ``length_scale``, however.)
        """
        # Register the arguments
        self.length_scale = length_scale
        self.species = species
        # Check that fieldtype is a correct field
        if (fieldtype in ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]) is False:
            raise ValueError("`fieldtype` must be one of Ex, Ey, Ez, Bx, By, Bz")

        # Note: when `gamma_boost` is passed, the fields are evaluated in
        # the boosted frame, even though the user-provided function corresponds
        # to the lab frame. This is done (conceptually) in 2 steps:
        # - the field is computed in the lab frame, by transforming
        #   the current position/time of particles to the lab frame
        # - the field is converted back to the boosted frame, by using
        #   Lorentz transform formulas for E and B

        gpu_field_func = field_func

        # Modify user-input function, so as to evaluate field in the lab frame
        if (gamma_boost is not None) and (gamma_boost != 1.0):
            beta_boost = np.sqrt(1.0 - 1.0 / gamma_boost**2)
            field_func = njit(field_func)

            def func(F, x, y, z, t, amplitude, length_scale):
                zlab = gamma_boost * (z + beta_boost * c * t)
                tlab = gamma_boost * (t + beta_boost * inv_c * z)
                return field_func(F, x, y, zlab, tlab, amplitude, length_scale)

        else:
            func = field_func

        # Compile the CPU callback now. Compile the GPU callback on first use.
        signature = [
            float64(float64, float64, float64, float64, float64, float64, float64)
        ]
        cpu_compiler = vectorize(signature, target="cpu", nopython=True)
        self.cpu_func = cpu_compiler(func)
        self.gpu_func = None
        self._gpu_field_func = _snapshot_callback(gpu_field_func)
        self._gamma_boost = gamma_boost

        # Convert the field back to the boosted frame
        if (gamma_boost is not None) and (gamma_boost != 1.0):
            g = gamma_boost
            gb = gamma_boost * beta_boost
            if fieldtype == "Ex":
                self.fieldtypes_and_amplitudes = (
                    ("Ex", g * amplitude),
                    ("By", -gb * inv_c * amplitude),
                )
            elif fieldtype == "Ey":
                self.fieldtypes_and_amplitudes = (
                    ("Ey", g * amplitude),
                    ("Bx", gb * inv_c * amplitude),
                )
            elif fieldtype == "Bx":
                self.fieldtypes_and_amplitudes = (
                    ("Bx", g * amplitude),
                    ("Ey", gb * c * amplitude),
                )
            elif fieldtype == "By":
                self.fieldtypes_and_amplitudes = (
                    ("By", g * amplitude),
                    ("Ex", -gb * c * amplitude),
                )
            elif (fieldtype == "Ez") or (fieldtype == "Bz"):
                self.fieldtypes_and_amplitudes = ((fieldtype, amplitude),)
        else:
            self.fieldtypes_and_amplitudes = ((fieldtype, amplitude),)

    def apply_expression(self, ptcl, t):
        """
        Apply the external field function to the particles

        This function is called at each timestep, after field gathering
        in the step function.

        Parameters
        ----------
        ptcl: a list a Particles objects
            The particles on which the external fields will be applied

        t: float (seconds)
            The time in the simulation
        """
        for species in ptcl:

            # If any species was specified at initialization,
            # apply the field only on this species
            if (self.species is None) or (species is self.species):

                # Only apply the field if there are macroparticles
                # in this species
                if species.Ntot <= 0:
                    continue

                # Loop over the different fields involved
                for fieldtype, amplitude in self.fieldtypes_and_amplitudes:

                    field = getattr(species, fieldtype)

                    if type(field) is np.ndarray:
                        # Call the CPU function
                        self.cpu_func(
                            field,
                            species.x,
                            species.y,
                            species.z,
                            t,
                            amplitude,
                            self.length_scale,
                            out=field,
                        )
                    else:
                        if self.gpu_func is None:
                            self.gpu_func = _compile_warp_field(
                                self._gpu_field_func, self._gamma_boost
                            )
                        # Get the threads per block and the blocks per grid
                        dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d(species.Ntot)
                        # Call the GPU kernel
                        self.gpu_func[dim_grid_1d, dim_block_1d](
                            field,
                            species.x,
                            species.y,
                            species.z,
                            t,
                            amplitude,
                            self.length_scale,
                        )
