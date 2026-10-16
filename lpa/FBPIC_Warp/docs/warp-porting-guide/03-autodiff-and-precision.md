# Autodiff and selective mixed precision

## Current status

The current port implements forward execution. Its kernel wrapper explicitly sets `enable_backward=False` in [the Warp runtime](../../fbpic/utils/warp.py).
The port has no validated simulation gradients. Forward tests and CUDA graph replay do not establish gradient correctness.

The original port plan, summarized in the [baseline change record](../baseline-changes/README.md), deferred gradient implementation, gradient validation, and execution with multiple GPU ranks.
Message Passing Interface (MPI) infrastructure remains available for later parallel execution. MPI adjoints remain deferred.

The approved next milestone validates individual FP64 kernel derivatives in eager execution.
The [kernel-gradient chapter](08-individual-kernel-gradients.md) defines its continuous inputs, fixed discrete decisions, and acceptance checks.
The broader simulation-gradient sequence below remains future work.

The earlier proposal for whole-solver FP32/FP64 selection was deferred.
The current precision work examines individual FP32 operations while simulation state and other operations retain FP64.
The [FP32 Hankel experiment](05-hankel-fp32-and-ionization-rng.md) now implements this approach on the FP32 Hankel experiment branch.
The [complete Inversion study](07-inversion-full-workload.md) extends validation to 17,720 steps with fusion enabled.
Other precision changes, additional configurations, and differentiated mixed-precision execution remain research work.

## Define the gradient before implementing it

A useful gradient requires a precise objective, parameter set, and execution horizon.
An initial particle coordinate, a laser amplitude, and a transform matrix require different derivative paths.
Geometry parameters can affect initialization, transform coefficients, particle allocation, and event decisions.

1. Specify the scalar objective, differentiated parameters, units, and observation time.
2. Identify fixed inputs, parameter-dependent initialization, and excluded derivatives.
3. Define the inner product for complex fields and any physical cell weights.
4. Define the treatment of particle creation, removal, sorting, and random events.
5. Select a short deterministic case before extending the differentiation horizon.

The existing [simulation loop](../../fbpic/main.py) provides the execution order.
Its gathering, particle pushes, deposition, current correction, and field advance provide useful operator boundaries.
Retaining baseline structure makes these boundaries easier to compare and test.

## Build operator adjoints and preserve state

1. Isolate an operator with explicit inputs and outputs.
2. Verify its local adjoint before composing a simulation step.
3. Register adjoints for retained external library calls.
4. Preserve forward values required by reverse execution.
5. Define checkpoint reconstruction before extending to longer trajectories.

The [zero-copy array adapter](../../fbpic/utils/warp.py) represents `complex128` values as `wp.vec2d` elements.
An adjoint must use the same real and imaginary component order, array layout, strides, and ownership rules.
Sharing a pointer does not register derivatives for operations executed through CuPy.

The [Fourier transform](../../fbpic/fields/spectral_transform/fourier.py) uses an unnormalized forward transform and divides the inverse result by `Nz`.
For the standard complex Euclidean inner product, the adjoints therefore satisfy:

```text
forward_adjoint = Nz * inverse_transform
inverse_adjoint = forward_transform / Nz
```

These relations include the implemented normalization.
The real inner product on paired real and imaginary components gives the same result for a real scalar objective.
Copy kernels and layout conversions also require corresponding reverse operations.

The [Hankel transform](../../fbpic/fields/spectral_transform/hankel.py) multiplies real buffers by `M` or `invM` through cuBLAS.
Reverse accumulation requires the transpose of the matrix used by that operation.
The inverse matrix and the transpose serve different purposes.
Column-major library arguments and the code's stored matrix orientation determine the actual transpose flags.

If a parameter changes a transform matrix, differentiating only the input field omits part of its derivative.
Parameter-dependent matrix construction requires a separate derivative contract.

The simulation overwrites particle coordinates, momenta, gathered fields, and spectral buffers.
Sorting also swaps particle arrays and reuses storage in [the particle implementation](../../fbpic/particles/particles.py).
A reverse pass requires the correct earlier values, even when the current pointer still exists.
Possible solutions include separate state versions, saved intermediates, and deterministic checkpoint reconstruction.
Each solution has memory, replay-cost, and numerical consequences.

## Treat discrete events explicitly

For a fixed permutation, the adjoint of particle reordering uses the inverse permutation.
The [cell-index and sorting code](../../fbpic/particles/utilities/cuda_sorting.py) also makes discrete decisions from particle positions.
Those decisions need a documented treatment at cell boundaries.

Gathering and deposition require derivatives of interpolation weights and accumulated values.
Their mode factors, normalization, and axis corrections require direct checks against the implemented formulas.
A similar scatter pattern alone does not establish an adjoint relation.

Particle injection, removal, and ionization change the state size or event history.
A derivative through one fixed event history has a different meaning from a derivative of an expected physical observable.
That distinction belongs in the objective contract.

The [random-number implementation](../../fbpic/particles/elementary_process/warp_random.py) advances mutable states.
Fixed random inputs can support conditional pathwise checks where the computation permits them.
Threshold decisions and rejection sampling require separate treatment.
Restarting with the same initial seed does not guarantee the same subsequent event history after parameters change.

## Verify gradients before optimizing reverse execution

1. Check local adjoint identities with independently chosen input and output perturbations.
2. Check directional finite differences across several perturbation sizes.
3. Compare composed gradients over short deterministic trajectories.
4. Verify graph replay after the corresponding eager forward and reverse paths pass.
5. Extend to MPI only after the single-device derivative contract is established.

For an operator Jacobian `J`, the local identity is:

```text
<J * input_direction, output_direction>
    = <input_direction, adjoint(J) * output_direction>
```

Finite-difference checks need a useful interval between truncation error and floating-point cancellation.
One perturbation size or one passing scalar comparison provides weak evidence.
Checks should include complex components, transform normalization, and parameters that affect initialization.

The current [graph cache](../../fbpic/utils/warp.py) depends on pointers, shapes, scalar arguments, and dtypes.
Future reverse graphs also require valid intermediate states, gradient buffers, and accumulation resets.
Particle reallocation and checkpoint restoration can invalidate these assumptions.

MPI adjoints require reverse communication in the correct order.
Halo sums, particle migration, ownership changes, and forward exchanges determine the corresponding reverse operations.
Preserving MPI plumbing helps retain these boundaries, but does not implement their adjoints.

## Explore mixed precision one operation at a time

FP64 simulation state and the existing CPU implementation remain the reference.
A candidate GPU operation can use explicit FP32 scratch arrays while surrounding state retains FP64.
The experiment must include conversion boundaries, scratch lifetime, and accumulation precision.

1. Select one measured operation and define its contribution to simulation error.
2. Add explicit input conversion, FP32 execution, and output conversion for that experiment.
3. Measure the complete operation, including conversions and required synchronization.
4. Compare operator outputs and physical observables against the FP64 reference.
5. Retain the change only after representative longer simulations satisfy their accuracy requirements.

Persistent scratch buffers can avoid repeated allocation, but their contents require correct refresh and invalidation rules.
Conversion and extra memory traffic can remove a faster kernel's benefit.
Arithmetic precision, accumulation precision, and library math settings must be recorded separately.

The current Fourier path creates a `CUFFT_Z2Z` plan with `complex128` buffers.
A candidate FP32 Fourier operation requires an explicit `CUFFT_C2C` plan and matching buffers.
The default Hankel path calls `cublas.dgemm` with FP64 real buffers.
The FP32 Hankel experiment branch adds `cublas.sgemm` with FP32 matrices, operands, and scalars.
It uses separate reusable scratch and a dedicated cuBLAS handle with pedantic arithmetic on the shared runtime stream.
Fields remain complex128, particles remain float64, and FFTs retain double precision.
The [experiment record](05-hankel-fp32-and-ionization-rng.md) covers operator tests, simulation comparisons, the ionization RNG fix, and the production rerun.
The recorded 2048-step result does not validate the complete interaction or other FP32 operations.
The later [complete comparison](07-inversion-full-workload.md) measured maximum field relative L2 error `0.2701%` with FP32 Hankel.
That run also changed 53 final ionization levels and final electron counts by `+5` helium and `-10` nitrogen.
These changes must accompany the speedup claim. They do not establish an application-specific precision acceptance threshold.

## Precision hazards found in this port

| Area | Evidence and implication |
| --- | --- |
| Ionization coefficients | [ADK coefficients](../../fbpic/particles/elementary_process/ionization/ionizer.py) can exceed FP32 range before multiplication produces a finite probability. |
| Field-solver coefficients | [PSATD coefficients](../../fbpic/fields/psatd_coefs.py) contain differences such as `1 - cos(w*dt)` and `1 - sin(w*dt)/(w*dt)`. |
| Transform setup | [DHT construction](../../fbpic/fields/spectral_transform/hankel.py) uses Bessel functions, matrix inversion, and a pseudoinverse for some modes. |
| Small physical quantities | SI-valued intermediate products can underflow even when the final result is representable. |
| Random sampling | [Normal sampling](../../fbpic/particles/elementary_process/warp_random.py) evaluates `log(1 - uniform_sample)`. FP32 conversion can round a sample to one. |

ADK means Ammosov-Delone-Krainov ionization. PSATD means pseudo-spectral analytical time-domain. DHT means discrete Hankel transform.

A CPU evaluation of the existing chlorine test coefficients gave a maximum ADK prefactor of approximately `7.58e53`.
The timestep was `2e-6 / (10*c)`, as in [the spin-ionization test](../../tests/test_spin_ionization.py).
Nine chlorine coefficient entries exceeded FP32 range.
This example concerns coefficients, rather than the applied field amplitude.
It does not establish an end-to-end FP32 result.

For PSATD, coefficient construction and coefficient storage require separate decisions.
FP64 construction can reduce cancellation before a later cast, but cannot preserve a stored coefficient outside FP32 range.
DHT matrix construction and its runtime multiplication also require separate error checks.
Neither setup precision nor runtime precision should follow an automatic global substitution.

Discrete data retains its existing integer representation.
In particular, [particle identifiers](../../fbpic/particles/tracking/tracking.py) use `uint64` and must preserve exact identity during sorting and output.

The [restart reader](../../fbpic/openpmd_diag/checkpoint_restart.py) asserts FP64 particle arrays and recreates FP64 buffers.
Keeping persistent state in FP64 preserves that contract, but experimental scratch buffers still require recreation after restart.
The [external-field adapter](../../fbpic/lpa_utils/external_fields.py) compiles FP64 signatures and retains one compiled GPU callback per object.
Any precision experiment involving callbacks needs explicit specialization and cache ownership.

## Use physical error budgets

The [forward validation report](../baseline-changes/milestone-1-validation.md) records the current evidence and its limits.
Its FP64 transform and cancellation checks cannot serve as universal mixed-precision tolerances.
Existing physical and statistical acceptance criteria remain the starting point.

Operator comparisons can separate input conversion error from execution error using FP64 evaluation of the converted inputs.
Simulation comparisons must also account for phase error, conservation residuals, particle distributions, and accumulated error over the intended duration.
Diagnostic reductions can remain FP64 to reduce measurement error.

Exceptions require evidence from numerical range analysis, focused tests, and representative longer runs.
One successful kernel benchmark does not establish simulation accuracy.
A blanket tolerance increase would hide which operation introduced the error.

Future gradient checks must use the actual mixed-precision execution path when that path is part of the intended simulation.
Forward accuracy and derivative accuracy require separate evidence.
