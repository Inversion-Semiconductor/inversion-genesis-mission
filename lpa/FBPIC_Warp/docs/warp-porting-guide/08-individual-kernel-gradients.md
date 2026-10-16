# Individual kernel gradients

This chapter defines the approved next milestone after forward integration and performance validation.
The contract starts with individual FP64 kernels in eager execution.
It does not claim a differentiable complete simulation or validated kernel gradients until corresponding verification results exist.
The production wrapper currently disables generated backward execution by default.
The separate `kernel-autodiff` worktree uses branch `snidhan/dev-autodiff`, based on approved production integration `edc52c4`.
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

The initial milestone differentiates continuous floating arrays.
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
Use generated adjoints only where those checks pass.
Use explicit backward operations when mutation or dynamic accumulation prevents a correct generated adjoint.

Reset gradient buffers before each independent verification.
Preserve intended gradient accumulation within a composed reverse pass.
Reallocation, sorting-buffer swaps, and restart restoration require recreating views that refer to replaced storage.
Eager verification must pass before graph capture introduces buffer-lifetime and replay constraints.

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

Require normalized agreement within `1e-10` for linear adjoint identities and `1e-6` for smooth nonlinear directional checks.
Record the normalization formula, perturbation sizes, scalar objectives, absolute residuals, and relative residuals with each result.
Near-zero contractions require an absolute or scaled normalization rather than division by a vanishing scalar.
Finite differences must use the same fixed branches and integer arrays as the reverse operation.
One successful perturbation or scalar comparison does not establish derivative correctness.

## Record evidence and explicit limits

Each operator result needs its source hash, launch configuration, shape, dtype, fixed inputs, differentiated arrays, and verification command.
Report generated and explicit adjoints separately.
Retain failing cases and their diagnosed cause before accepting a replacement backward operation.
Use physical GPU 0 and leave GPU 1 available for other work.
Run GPU verification serially while independent source audits proceed concurrently.

No gradient result is recorded in this chapter yet.
Add measured results only after the individual-kernel verification completes.
Forward parity and performance measurements in [chapter 07](07-inversion-full-workload.md) do not establish this milestone's derivative correctness.

FFT and GEMM adjoints follow after local kernel verification.
Long trajectories require state checkpointing or deterministic recomputation.
Stochastic ionization requires an explicit pathwise or expectation-gradient contract.
Reverse CUDA graphs, scalar-parameter gradients, and MPI adjoints remain later milestones.
These exclusions prevent presenting differentiable kernels as a differentiable full simulation.
