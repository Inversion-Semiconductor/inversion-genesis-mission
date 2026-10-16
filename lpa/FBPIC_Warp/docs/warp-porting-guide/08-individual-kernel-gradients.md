# Individual kernel gradients

This chapter records the first validated FP64 kernel differentiation adapters after forward integration and performance validation.
The adapters use eager execution, explicit state versions, and continuous array gradients.
They do not make the production `Simulation.step` automatically differentiable.
The production wrapper retains `enable_backward=False`; opt-in modules enable or register their own backward operations.
The branch `snidhan/dev-autodiff` holds this work, based on the cleaned baseline `snidhan/dev-main`.
The [autodiff chapter](03-autodiff-and-precision.md) describes the later requirements for complete trajectories and external transforms.

## Define the derivative contract

1. Select continuous input arrays and explicit output buffers for each operator.
2. Hold timesteps, grid coefficients, indices, permutations, and event decisions fixed.
3. Use the real inner product on paired real and imaginary components.
4. Preserve every primal value required by the reverse operation.
5. Verify gradients before composing operators or introducing reverse CUDA graphs.

For an arbitrary output cotangent `v`, verify the input gradient of `dot(v, output)`.
Array gradients use the unweighted real storage inner product unless the objective explicitly supplies physical weights.
A volume-weighted observable includes those weights in its output cotangent.
This distinction prevents silently inserting physical norm weights into an operator's storage adjoint.
Complex128 fields use the existing `wp.vec2d` real/imaginary representation.

The implemented initial milestone differentiates continuous floating arrays.
It excludes scalar-parameter gradients, transform-matrix construction, discrete event derivatives, and variable trajectory length.
Gather and deposition derivatives are piecewise smooth away from stencil changes and axis singularities.
Boundary tests must identify the selected branch instead of crossing an undefined derivative with central differences.

## Preserve buffers and ownership

CuPy owns production storage, while Warp views share pointers on the existing runtime stream.
Pointer sharing alone does not preserve primal values or register external library derivatives.
Keep gradient-enabled views alive throughout forward and backward execution.
Use explicit output buffers to expose mutation and accumulation semantics.
Retain the existing production behavior when differentiation is not requested.

In-place particle and field kernels overwrite values needed by later reverse operations.
An overwritten input requires a saved pre-operation value or a separately versioned output.
Atomic accumulation and dynamically bounded particle loops require direct gradient checks.
The verified particle and field adapters use generated adjoints in unique Warp modules.
Fused deposition preserves the production forward kernel and registers an explicit callback for its particle-wise vector-Jacobian product, or VJP.

Reset gradient buffers before each independent verification.
Preserve intended gradient accumulation within a composed reverse pass.
Reallocation, sorting-buffer swaps, and restart restoration require recreating views that refer to replaced storage.
Eager verification must pass before graph capture introduces buffer-lifetime and replay constraints.

## Use the opt-in interfaces

Import operator modules from `fbpic.autodiff`; production solver callers do not select these modules automatically.
Callers provide gradient-enabled continuous inputs, fixed integer and coefficient inputs, and separate output storage.
Keep inputs and intermediate outputs alive until the tape completes backward execution.
Do not alias an operator's input and output storage or supply partially overlapping state versions.

| Interface | Implemented behavior |
| --- | --- |
| `arrays.DifferentiableArrays` | Retain stable borrowed CuPy views and Warp-owned gradients. Exact aliases share views and gradients; gradient policy remains explicit. |
| `particles.push_positions` | Write new positions from fixed timestep, positions, momenta, and inverse gamma. |
| `particles.push_momenta` and `push_ion_momenta` | Write Vay momenta and inverse gamma through the production scalar helper. Ionization levels remain fixed. |
| `particles.get_gather_kernel()` | Return a backward-enabled clone of the production five-mode linear gather in an independent module. |
| `deposition.deposit_five_modes(...)` | Return five charge grids or fifteen current grids and register a fixed-stencil VJP on the active tape. |
| `fields` kernels | Write separate outputs for copies, component conversions, fixed permutations, field updates, corrections, filtering, damping, and shifts. |
| `verification.check_vjp(...)` | Verify arbitrary output-cotangent gradients through three scaled input directions and repeated reverse execution. |

The [array registry](../../fbpic/autodiff/arrays.py) borrows float64 and complex128 CuPy storage without a host transfer.
Its view cache includes device, pointer, shape, strides, and dtype.
It retains CuPy primal owners and persistent gradient storage.
Integer inputs require `requires_grad=False`; differentiated FP32 inputs are rejected.
Nonexact views with overlapping storage ranges are rejected when either view requires gradients.
Call `zero_gradients()` between independent computations.

The [deposition adapter](../../fbpic/autodiff/deposition.py) clones initial grids and executes the exact production fused forward kernels.
It derives fixed particle-cell assignments from prefix sums on the device.
The backward callback evaluates an equivalent per-particle expression with static mode and stencil loops.
It passes output-gradient arrays through the production buffer structs and preserves identity gradients for initial grids.
The adapter performs no host transfer, but its extra output storage and metadata work remain unbenchmarked.
Prefix values must be monotonic, end at the particle count, and match the sorted particle-cell assignments.
Those value constraints are caller preconditions, avoiding host reads in the operator path.
Ruyten coefficients remain fixed, and the adapter rejects coefficient arrays with gradients enabled.

The [verification helper](../../fbpic/autodiff/verification.py) deliberately synchronizes and transfers arrays to the host for numerical checks.
Those transfers belong to validation and are outside the differentiable operator path.

## Validate the operators in stages

| Stage | Required operations | Important checks |
| --- | --- | --- |
| Storage and conversions | Copies, real/imaginary packing, and field-component conversions. | Layout, strides, ownership, accumulation, and complex pairing. |
| Particle updates | Position and momentum pushes. | Saved primal state, momentum-dependent inverse gamma, and physical scaling. |
| Particle-field coupling | Fused five-mode gather and deposition. | Mode factors, stencil weights, effective ion weights, and repeated cell contributions. |
| Field operators | Comoving current correction, comoving field advance, filters, volume normalization, and damping. | Real/imaginary derivatives, fixed coefficients, and overwritten spectral values. |
| Fixed data motion | Spectral window shifts and floating payload copies through fixed permutations. | Reverse shift or inverse permutation and boundary treatment. |

Start from the deployed Inversion configuration and compare each operator's primal output with its production path.
Test multiple particles in one cell and uneven cell populations.
Test effective ion weights while holding integer ionization levels fixed.
Exercise empty populations and nonmultiple launch sizes where the operator supports them.
Keep cell-index construction, sorting decisions, particle creation, and stochastic thresholds outside the continuous derivative contract.

## Check adjoint identities and directional derivatives

For a linear operator `A`, verify the storage adjoint identity with independent input and output directions.

```text
dot(A * input_direction, output_cotangent)
    = dot(input_direction, adjoint(A) * output_cotangent)
```

For a smooth nonlinear operator `F`, compare the reverse gradient with central directional differences.

```text
phi(x) = dot(output_cotangent, F(x))

directional_reverse = dot(input_gradient, input_direction)

directional_central = (phi(x + h * input_direction)
                    - phi(x - h * input_direction)) / (2 * h)
```

1. Scale directions to the units and magnitudes of each input block.
2. Evaluate several perturbation sizes and require a stable agreement interval.
3. Repeat with three independent directions and arbitrary output cotangents.
4. Test real and imaginary components separately and together.
5. Verify gradient resets and a composition of two operators.

Require normalized agreement within `1e-10` for linear checks and `1e-6` for smooth nonlinear directional checks.
Record the normalization formula, perturbation sizes, scalar objectives, absolute residuals, and relative residuals with each result.
Near-zero contractions require an absolute or scaled normalization rather than division by a vanishing scalar.
Finite differences must use the same fixed branches and integer arrays as the reverse operation.
One successful perturbation or scalar comparison does not establish derivative correctness.

The implemented linear finite-difference ladder is `1`, `1e-1`, `1e-2`, `1e-3`, `1e-4`, and `1e-5`.
The nonlinear ladder is `1e-2`, `1e-3`, `1e-4`, `1e-5`, `1e-6`, and `1e-7`.
Each of three directions requires two adjacent perturbations within tolerance.
The VJP residual denominator is the largest absolute analytical contraction, absolute numerical contraction, sum of absolute input-block contractions, or `1e-12`.
Each independently checked linear adjoint identity uses a norm-product denominator with a floating-point minimum floor.
The helper verifies unchanged primals, finite gradients, and repeatability after `tape.zero()`.
Set `FBPIC_AUTODIFF_REPORT_DIR` to save per-case JSON records instead of retaining only pytest summaries.
Reports include perturbation residuals, fixture hashes, input shapes and dtypes, operator identity, device, and selected continuous inputs.

## Diagnose backward failures independently of forward parity

The initial spectral-shift adapter had correct primal outputs but incorrect field gradients.
Its dynamic power loop updated a local accumulator that backward execution did not preserve correctly.
Moving the same power calculation into `_fixed_shift_power`, a `@wp.func`, restored backward replay.
The forward formula remained unchanged.
Tests include zero displacement, positive displacements of three and four cells, and negative displacements of two and five cells.

Small finite-difference steps also caused cancellation in weak linear filter outputs.
Larger linear perturbations restored stable agreement without changing the `1e-10` tolerance.
The nonlinear ladder retains smaller perturbations for truncation-error checks.
This distinction separates a finite-difference measurement problem from a true backward implementation failure.

## Record evidence and explicit limits

Each operator result needs its source hash, launch configuration, shape, dtype, fixed inputs, differentiated arrays, and verification command.
Report generated and explicit adjoints separately.
Retain failing cases and their diagnosed cause before accepting a replacement backward operation.
Use physical GPU 0 and leave GPU 1 available for other work.
Run GPU verification serially while independent source audits proceed concurrently.

Final GPU verification passed 122 tests on physical GPU 0.
The total includes 17 deposition tests, 17 particle tests, 84 field tests, and four borrowed-array tests.
The 84 field tests cover 28 configurations, each with primal parity, directional VJP checks, and independent linear adjoint identities.
CPU verification passed 118 tests and skipped the four GPU-only borrowed-array tests.
Borrowed-CuPy tests require a GPU and do not establish CPU array borrowing.
Each device retained 52 VJP reports with 156 directions: 117 linear and 39 nonlinear.
Every reported direction satisfied the required two-adjacent-perturbation agreement interval.

| Group | Verified coverage |
| --- | --- |
| Particles | Electron and fixed-level ion pushes, neutral-ion identity gradients, position updates, fused gather, and momentum-to-position composition. |
| Deposition | Fused forward parity, equivalent particle-expression parity, position/weight/current gradients, initial-grid identity, and empty populations. |
| Fields | Copies, real/imaginary packing, basis conversions, fixed permutations, clearing, volume normalization, filters, charge shifts, and standard/comoving updates. |
| Boundaries and window | Left/right damping and fixed spectral displacement, including zero and larger signed shifts. |
| Borrowed arrays | Stable alias ownership, explicit gradient policy, complex strided views, gradient reset, and two-filter composition. |

The field-advance checks include both true-charge and divergence-derived-charge branches.
Current correction includes standard and comoving curl-free operators.
Deposition checks include multiple particles per cell and independent position, weight, momentum, inverse-gamma, and initial-grid gradient groups.
Callers can supply effective weights containing fixed ion levels; these tests do not differentiate integer levels or ionization decisions.
Additional checks compose charge or current deposition with filtering and reject differentiated fixed Ruyten coefficients.
Smooth-region tests keep stencil and event decisions fixed.
These fixtures do not establish derivatives at the axis, stencil crossings, or event thresholds.

Run the verified tests from the package directory in the `fbpic-warp` environment.

```bash
CUDA_VISIBLE_DEVICES=0 FBPIC_AUTODIFF_DEVICE=cuda:0 \
  NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  FBPIC_AUTODIFF_REPORT_DIR=.cache/autodiff_validation/gpu0 \
  python -m pytest -q tests/test_autodiff_arrays.py tests/test_autodiff_particles.py \
  tests/test_autodiff_fields.py tests/test_autodiff_deposition.py \
  --junitxml=.cache/autodiff_validation/gpu0/pytest.xml

CUDA_VISIBLE_DEVICES='' FBPIC_AUTODIFF_DEVICE=cpu \
  NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  FBPIC_AUTODIFF_REPORT_DIR=.cache/autodiff_validation/cpu \
  python -m pytest -q tests/test_autodiff_arrays.py tests/test_autodiff_particles.py \
  tests/test_autodiff_fields.py tests/test_autodiff_deposition.py \
  --junitxml=.cache/autodiff_validation/cpu/pytest.xml
```

The CPU command hides CUDA devices and skips GPU-only borrowed-CuPy tests.
JUnit summaries and per-case JSON records remain in the selected `.cache/autodiff_validation` device directory.
These generated verification artifacts remain excluded from Git.
Verification allocates separate outputs and saves primal arrays, so reverse execution has higher memory cost than the production forward path.
No backward timing, memory-budget conclusion, complete-timestep gradient, or long-horizon gradient result is established.
The forward accuracy and performance evidence in [chapter 07](07-inversion-full-workload.md) remains separate from these derivative checks.

FFT and GEMM adjoints follow after local kernel verification.
Long trajectories require state checkpointing or deterministic recomputation.
Stochastic ionization requires an explicit pathwise or expectation-gradient contract.
Reverse CUDA graphs, scalar-parameter gradients, and MPI adjoints remain later milestones.
These exclusions prevent presenting differentiable kernels as a differentiable full simulation.
