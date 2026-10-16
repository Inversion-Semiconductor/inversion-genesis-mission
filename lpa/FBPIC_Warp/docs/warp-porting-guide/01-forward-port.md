# Build and validate the forward solver

This chapter gives repeatable steps for a forward GPU port of an existing baseline solver.
The numbered steps and the **Common errors** paragraphs apply to any baseline code.
Paragraphs marked **FBPIC example** show how this repository applied the steps.
FBPIC examples are not requirements for other ports.

Establish forward correctness on one GPU first.
Validate automatic differentiation and execution across multiple GPUs as separate milestones.

## Phase 1: Establish the reference and ownership

1. Record the baseline commit, working-tree state, package versions, CUDA libraries, and device identity.
2. Run the baseline tests on the target system. Record existing failures and acceptance tolerances.
3. Inventory numerical algorithms, optional features, interfaces, diagnostics, and existing tests.
4. Decide which filenames, public symbols, array conventions, and operation order the port preserves.
5. Assign one owner for the shared runtime before other contributors port separate modules.

An unchanged baseline provides executable reference behavior.
The shared-runtime owner controls allocation, stream selection, launch conversion, graph execution, and final validation.
Other contributors can port separate modules after the shared interface is defined.
Preserved names and operation order make each ported operation easy to compare with its baseline.

**Common errors:**

- If the baseline checkout changes during the port, earlier comparisons lose their reference.
- A commit identifier does not identify uncommitted changes. If compared code has uncommitted changes, record source hashes.
- A test that already fails on the baseline can look like a port regression.
- Two contributors who edit one file create conflicting changes. Assign each file to one contributor.
- Concurrent GPU tests on one device compete for memory and change timings. Run them serially.

**FBPIC example:** The port uses baseline commit `dc1cc32ca36ab1ee813d82ea5e9f6b180be0e2a9`.
The [baseline change record](../baseline-changes/README.md) records the changes in each module.
The milestone plan delegated disjoint kernel modules to worker agents and serialized GPU tests on the selected GPU.
The port retains Python orchestration, CPU kernels, and the `fbpic` package name.
The benchmark launcher `benchmarks/run_matrix.py` on the profiling branch records source hashes for each compared implementation.

## Phase 2: Define memory and stream contracts

1. Define each array's owner, shape, byte strides, dtype, device, and mutation rules.
2. If libraries can share a device representation, borrow the existing storage instead of copying it.
3. Keep each allocation owner alive while a borrowed view or captured graph uses it.
4. Run dependent kernels, copies, and library calls on one explicit CUDA stream.
5. Check the normalization, layout, and index order of each retained library call against the baseline.

**Common errors:**

- Sorting, resizing, and buffer replacement can make a cached view refer to obsolete storage. After these operations, construct a new view from the current array.
- Input and output arrays can alias. State the aliasing rule for each kernel.
- Work on different streams has no guaranteed order. A library call on another stream can read data before a kernel writes it.
- Warp has no native complex type. Define the element representation and the complex arithmetic helpers explicitly.
- A changed element type can add an array dimension or change strides. Check shape and strides after each conversion.

**FBPIC example:** `as_warp_array` in the [shared runtime](../../fbpic/utils/warp.py) borrows CuPy storage without copying.
The view retains the original shape, byte strides, device, and CuPy owner.
One `complex128` element becomes one `wp.vec2d` containing its real and imaginary components.
The array does not gain another dimension.
The `cmul` and `mul_i` helpers implement the required complex operations.

`CudaRuntime` lets CuPy own the stream while Warp borrows its CUDA handle.
The [Fourier transform](../../fbpic/fields/spectral_transform/fourier.py) retains cuFFT plans and inverse normalization.
The [Hankel transform](../../fbpic/fields/spectral_transform/hankel.py) retains cuBLAS matrix products and real/complex layout conversion.
Retaining these libraries reduced the first milestone's implementation scope.

## Phase 3: Port algorithms without changing behavior

1. Port one complete numerical operation with its inline helpers and scratch arrays.
2. Specify scalar types, integer widths, constants, and complex arithmetic explicitly.
3. Compare the translated operation with its baseline definition, its caller, and its outputs.
4. Connect validated operations in the baseline execution order.
5. Port boundary conditions, sources, optional physics, and diagnostics before you claim application coverage.

Typed launch adapters can keep recognizable call sites while the device implementation changes.
The adapter must preserve launch bounds and array types.
Structural similarity supports review; numerical checks establish correctness.

**Common errors:**

- Small source changes can alter numerical promotion or generated arithmetic.
- A floating-exponent power function can replace an integer power. Write small integer powers as explicit multiplication.
- `wp.tid()` returns a 32-bit integer. Compute offsets into large arrays with 64-bit integers.
- Python and C conventions differ for negative integer division and modulo. Check each translated operator.
- A precision change during the port mixes two sources of error. Keep the baseline precision until the forward port passes.

**FBPIC example:** `WarpKernel` retains `[blocks, threads]` calls in the [shared runtime](../../fbpic/utils/warp.py).
GPU implementations retain names such as `cuda_methods.py` and `push_p_gpu`.
The forward port retains `float64` particle state and `complex128` field state.
On `snidhan/dev-profiling`, `benchmarks/kernel_mapping.json` connects measured operations to definitions and callers in both implementations.
The [baseline change record](../baseline-changes/README.md) identifies structural and behavioral adaptations.

The first FBPIC port used floating-exponent `wp.pow` calls where baseline kernels used integer squares.
The [performance chapter](02-performance.md) records the cost of that translation.
Selective precision changes are covered in the [autodiff and precision chapter](03-autodiff-and-precision.md).

### User-defined functions

1. If an interface generates functions at run time, keep the source text of each function.
2. Preserve closure capture, numeric promotion, lookup types, and signed arithmetic.
3. Compile a GPU function the first time GPU execution requires it.
4. Supply and test each math function that the target library does not provide.

Warp translates Python source into device code.
A function created at run time can have no source that inspection can find.
Compilation at construction adds cost to runs that never use the GPU path.

**FBPIC example:** [ExternalField](../../fbpic/lpa_utils/external_fields.py) translates supported scalar callback syntax and retains the Numba CPU path.
Its lookup representation preserves numeric types and bit patterns.
The [native math helpers](../../fbpic/lpa_utils/external_field_math.py) supply missing double-precision operations, including `expm1`, `log1p`, and `hypot`.
[PICMI integration](../../fbpic/picmi/simulation.py) attaches `__fbpic_source__` to generated callbacks.
PICMI means Particle-In-Cell Modeling Interface.

### Random numbers

1. Define the owner and update rules of each random number generator state.
2. Validate stochastic outputs with the existing statistical criteria.
3. Prevent compilation and graph preparation from consuming physical random samples.

A different generator produces different individual events.
Compare distributions and physical results, not individual events.

**FBPIC example:** The [random helpers](../../fbpic/particles/elementary_process/warp_random.py) update explicit integer states and construct double-precision samples.
The optional physics checks use distribution and physical assertions.

### Parallel interfaces

Apply these steps if the baseline supports distributed execution.

1. Retain communicator objects and decomposition metadata.
2. Preserve guard-cell widths, exchange order, buffer layouts, and integer identifiers.
3. If communication requires host participation, keep it outside GPU graph capture.
4. Report retained interfaces separately from tested execution coverage.

Single-GPU tests do not exercise the exchange paths.
A retained interface is not evidence of correct parallel execution.

**FBPIC example:** [MPI utilities](../../fbpic/utils/mpi.py) and [field exchange buffers](../../fbpic/boundaries/field_buffer_handling.py) remain in the port.
MPI means Message Passing Interface.
The port has no validation with multiple MPI ranks.
Preserving these interfaces supports later parallelization without requiring a second solver architecture.

## Phase 4: Add graphs after eager execution works

1. Identify reusable sections whose launches preserve the solver's required ordering.
2. Prepare kernels and library plans without advancing physical state.
3. Capture supported sections with `wp.ScopedCapture`.
4. When arguments change, invalidate the affected graph.
5. Compare eager execution with capture and repeated replay.

Graph keys must describe the launched work.
Relevant values include kernel identity, dimensions, array addresses, shapes, byte strides, dtypes, devices, and scalar arguments.
Retained owners keep captured addresses valid.
Keep an eager execution switch for comparison.

**Common errors:**

- A captured graph fixes array addresses and scalar values. A replay after reallocation or a scalar change uses obsolete values.
- Host synchronization and host decisions based on device values cannot occur inside a capture.
- Some library calls fail during capture. Keep those calls outside the graph on the same stream.
- If arguments change at most steps, new captures can cancel the replay savings. Include capture cost in solver timing.

**FBPIC example:** `CudaGraphSections` prepares commands through `wp.launch(..., record_cmd=True)` in the [shared runtime](../../fbpic/utils/warp.py).
This preparation compiles kernels without executing the physical update.
Capture records the prepared commands, and `wp.capture_launch` executes them.
[Simulation.step](../../fbpic/main.py) retains host orchestration and clears cached graphs around GPU data transfers.
Diagnostics, sorting, allocation, callbacks, and communication remain at their original execution positions outside these sections.

The recorded setup check captured cuFFT successfully, but CuPy rejected its cuBLAS calls during capture.
Production graphs therefore capture selected Warp sections while library calls retain ordering on the shared stream.
That restriction describes the tested integration, rather than every cuBLAS configuration.
Moving-window geometry, scalar arguments, and particle-buffer changes can invalidate cached graphs.
`use_cuda_graphs=False` provides the eager comparison path.

## Phase 5: Use algorithm checks as acceptance gates

1. Verify the imported source path and actual GPU execution before you accept results.
2. Run existing analytical and physical checks for each ported operation.
3. Check boundary indices, irregular dimensions, partial blocks, identifiers, and array mutation.
4. Run coupled cases with eager execution and graph replay.
5. Record exact test selection, package versions, exclusions, failures, and remaining coverage limits.

Existing tolerances remain the starting point for unchanged precision and algorithms.
A failed assertion requires investigation before any tolerance change.
Integer indices and identifiers require exact checks where the algorithm defines exact results.
Stochastic tests require their established statistical criteria.

**Common errors:**

- If the baseline and the port use the same package name, one process can import the wrong checkout. Run each implementation in a separate process.
- A test can pass on a CPU path while the GPU path is inactive.
- Near-zero components inflate relative errors. Normalize each error by the physical quantity under comparison.

**FBPIC example:** [Transform tests](../../tests/test_cuda_transform.py) check irregular shapes, both directions, and scalar/vector fields against CPU results.
Their bound is `1e-12` times each reference output's maximum magnitude.
The bound has a floor at the smallest positive normal `float64` value.
[Plasma-wave tests](../../tests/test_periodic_plasma_wave.py) retain analytical checks and assert graph capture and reuse.
[Deposition tests](../../tests/test_cpu_gpu_deposition.py) compare CPU and GPU charge and current components.
[Callback tests](../../tests/test_external_fields.py) cover arithmetic, lookups, promotion, and lazy compilation.

The baseline and the port both use the `fbpic` package name.
The benchmark launcher `benchmarks/run_matrix.py` on the profiling branch therefore runs each implementation in a separate process.
The benchmark driver `benchmarks/run_lwfa.py` on the profiling branch checks the imported package path.

### Report coverage precisely

1. State the test selection, package versions, and exclusions for each reported result.
2. Distinguish a complete suite run from targeted reruns.
3. State what each result does not establish.

A successful short simulation is a setup check.
Application correctness requires the relevant algorithm and interface tests.

**FBPIC example:** The [milestone validation report](../baseline-changes/milestone-1-validation.md) records 65 passing consolidated tests under Warp 1.17.
Ten tests requiring MPI execution were excluded.
Later callback fixes passed 24 targeted external-field tests.
Those targeted results do not constitute another consolidated run of the expanded suite.

The profiling branch report `benchmark-lwfa-warp-1.18.md` records eight compatibility checks and six subsequent preflight tests.
The full suite was not rerun for that benchmark.
The later benchmark provides workload-specific numerical evidence and performance measurements.
It does not establish gradient correctness or parallel execution correctness.
