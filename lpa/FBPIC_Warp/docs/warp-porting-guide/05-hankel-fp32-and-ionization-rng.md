# FP32 Hankel GEMM and reproducible ionization

This chapter records the development sequence on October 7 and 8, 2026, in Pacific time.
The work progressed from isolated FP32 GEMM measurements to simulation comparisons, ionization draw diagnosis, and a production-input rerun.
The final production comparison retained float32 random draws and FP64 simulation state.
Warp FP64 agreed closely with Numba FP64. Warp FP32 Hankel reduced compute time with small measured field differences.
These historical measurements cover 2048 steps and precede the final fusion changes.
The [complete Inversion chapter](07-inversion-full-workload.md) records the later producer revision, 17,720-step timings, and accumulated errors.

## Branches and source identity

| Role | Workspace path | Branch | Measured parent revision |
| --- | --- | --- | --- |
| Original Numba reference | `fbpic/` | `dev` | `dc1cc32ca36ab1ee813d82ea5e9f6b180be0e2a9` |
| Numba RNG fix | `fbpic-worktrees/ionization-rng/` | Numba ionization-RNG branch | `dc1cc32ca36ab1ee813d82ea5e9f6b180be0e2a9` |
| Warp precision and RNG changes | `fbpic-warp-worktrees/hankel-fp32/` | FP32 Hankel experiment branch | `0a717d062994751388f08b88b34051778005285b` |

These paths are relative to the shared FBPIC workspace.
The measured solver changes were uncommitted. Parent revisions identify ancestry, rather than the complete measured source.
Each manifest records the branch, dirty state, and SHA256 hashes of solver Python files.
The final runs also retain the measured launcher and harness.

General profiling tooling remains on `snidhan/dev-profiling`.
This chapter and the related simulation harnesses accompany the requested experiment on the FP32 Hankel experiment branch.

## 1. Measure FP32 Hankel GEMM in isolation

The isolated exercise `roofline/verify_hankel_precision.py` on the profiling branch compared FP32 and FP64 transforms.
It used synthetic inputs, sensitive matrix directions, transverse cancellation cases, and saved laser-wakefield acceleration fields.
The measurements included both GEMM alone and packing, GEMM, and unpacking together.

| Operator observation | Retained result |
| --- | ---: |
| Largest relative Frobenius error for saved fields | `2.730e-7` |
| Largest relative Frobenius error for sensitive directions | `1.240e-3` |
| Largest geometric volume error after 100 FP32 transform pairs | `1.565e-2` |
| GEMM speedup at `Nr=300`, forward and inverse | `21.500x` |
| Conversion-inclusive speedup at `Nr=300`, forward | `10.051x` |

The forward conversion-inclusive medians were `566.048` microseconds for FP64 and `56.320` microseconds for FP32.
Saved fields covered `Nr=100` and modes zero and one. Other sizes and modes used synthetic inputs.
Repeated transform pairs were operator stress tests. These results did not establish simulation accuracy or a physical acceptance tolerance.

The implementation exposes `Simulation(..., hankel_precision="float32")`.
The default remains `"float64"`.
The [Hankel implementation](../../fbpic/fields/spectral_transform/hankel.py) constructs matrices in FP64, then converts completed matrices and runtime operands into FP32 buffers.
SGEMM uses a separate cuBLAS handle with pedantic math mode `2` on the shared runtime stream.
Real and imaginary components occupy separate halves of reusable real buffers with shape `(2*Nz, Nr)`.
The transform copies results back into complex128 storage.

Particle storage, field storage, FFTs, solver coefficients, and ADK calculations retain double precision.
The observed CuPy conversions can flush FP32 subnormal values to zero.
Pedantic GEMM settings do not change that conversion behavior.
The [precision tests](../../tests/test_hankel_precision.py) cover transforms, aliasing, foreign streams, precision validation, and eager versus graph execution.

Existing arithmetic replacements also remain on this branch.
Selected squares and cubes use explicit products. Variable ADK powers remain unchanged.
On the measured Ada GPU, linear deposition uses 32 threads per block and linear gathering uses 128.
The production configuration uses linear particle shapes. It does not exercise cubic-shape paths.

## 2. Compare complete simulations and check the inputs

The first 2048-step comparison used a reconstruction prepared before the producer scripts were available.
It compared Numba FP64, Warp FP64, and Warp FP32 Hankel.
The final Warp FP64 net-charge relative L2 error was `0.9885396`, approximately `98.85%`.
The FP32 comparison also had a large charge difference.
A mismatch between the two FP64 backends showed that reduced GEMM precision could not explain the complete discrepancy.

The later producer scripts defined a different particle-loading and ionization configuration.
The [producer revision](https://github.com/Inversion-Semiconductor/inversion-genesis-mission/tree/2223c2bf447930befccf62ab2492197932bb964b/lpa/Simulation_FBPIC/runs/initial_sample) is `2223c2bf447930befccf62ab2492197932bb964b`.

| Setting | Initial reconstruction | Producer configuration |
| --- | --- | --- |
| Initial species | He+, N5+, and background electrons | Neutral He and N; electron species empty |
| Particles per cell | `(1, 2, 20)` | `(2, 2, 15)` |
| Loading radius | `200` micrometers | `90` micrometers |
| Longitudinal solver order | `-1` | `32` |
| LASY input | Archived reconstruction asset | Producer-generated shared asset |

An initial producer FP64 comparison retained the original index-based draws.
Its final net-charge relative L2 error was already `9.19335e-11`.
Those runs overlapped on two physical GPUs, so their timings do not provide an isolated speed comparison.
This result separates the original reconstruction mismatch from the production configuration.
Neither comparison establishes grid convergence.

## 3. Trace random draws through particle sorting

The original GPU ionizer called `cupy.random.rand(Ntot, dtype=cupy.float32)` at each ionization step.
The draw array associated entry `i` with the ion currently stored at array index `i`.
Both backends could produce identical draw arrays while assigning those draws to different physical ions.

The retained step-128 snapshots contained `3,720,000` ions.
Matching their unique coordinates after rounding to 10 picometers produced the following observations.

| Step-128 observation | Result |
| --- | ---: |
| Random arrays compared by array index | Bitwise identical |
| Largest longitudinal position difference for matched ions | `1.86347e-20 m` |
| Different cell keys for matched ions | `13` |
| Matched ions assigned different array indices and draws | `159,718` |

Tiny position differences crossed cell-index boundaries in this loading configuration.
The resulting sorting permutations changed draw assignment and later ionization histories.
Changing only the seed or generator name would not remove this dependence on array order.
The diagnosis concerns stochastic coupling between runs. It does not demonstrate inadequate spatial resolution.

At step 150, captured ionization inputs still had identical random arrays by index.
The stored ionization levels already differed at 23 array entries.
Array-index comparisons after sorting cannot identify the corresponding physical ions without an additional identity mapping.

## 4. Assign float32 draws by persistent ion ID

The fix adds optional `rng_seed` and `rng_stream` arguments to `Particles.make_ionizable`.
It requires ion tracking and passes the global simulation iteration into ionization.
The implementation uses Philox4x32-10 with fixed unsigned integer arithmetic in both backends.

| Generator input | Encoding |
| --- | --- |
| Persistent `uint64` ion ID | First two 32-bit counter words |
| Global iteration | Third counter word; below `2**32` |
| Species stream and ionization level | Fourth word: `(stream << 8) | level` |
| Unsigned 64-bit seed | Two 32-bit key words |
| Float32 output | Top 24 bits of the first result word, multiplied by `2**-24` |

The output lies in `[0, 1)` and remains float32.
Sorting or launch geometry does not change a draw for the same ID, iteration, stream, level, and seed.
The production comparison uses seed `0`, helium stream `0`, and nitrogen stream `1`.

```python
atoms_he.track(sim.comm)
atoms_n.track(sim.comm)
atoms_he.make_ionizable(
    "He", target_species=elec_he, level_start=0, rng_seed=0, rng_stream=0,
)
atoms_n.make_ionizable(
    "N", target_species=elec_n, level_start=0, rng_seed=0, rng_stream=1,
)
```

Without `rng_seed`, the original sampling path remains active.
ID-based sampling changes the realized random sequence. The ADK model and probability calculation remain unchanged.
Small probability differences can still change discrete decisions with a shared draw.
Statistical physics agreement across realizations requires separate validation.

The fix also preserves tracking counters and integer-safe ID allocation during restart.
Restarting ID-based ionization requires checkpoint IDs and the same seed and stream configuration.
The [CPU implementation](../../fbpic/particles/elementary_process/ionization/random.py), [Warp implementation](../../fbpic/particles/elementary_process/ionization/warp_random.py), and [tests](../../tests/test_ionization_rng.py) define the contract.
The implementation stage recorded 29 targeted passing tests in each backend.
Coverage included known-answer integers, float32 bit patterns, permutations, launch geometry, restart, injection, ADK, and spin regressions.

The intermediate 512-step reconstruction rerun reduced the final charge error from `55.55%` to `0.8088%`.
Its largest electromagnetic-component error was `5.976e-11`.
Final ion IDs and levels matched, but the re-deposited charge residual remained unexplained.
The attempted CPU charge cross-check omitted the forward transform and was invalid. Its result is excluded from the evidence.

## 5. Rerun the producer configuration on one GPU

The final launcher ran from 2026-10-08 23:27 UTC through 2026-10-09 00:06 UTC.
It ran Numba FP64, Warp FP64, and Warp FP32 Hankel sequentially in separate processes on physical GPU 1.
The device was an NVIDIA RTX 6000 Ada Generation GPU.
The Warp variants used identical solver source hashes and differed only in Hankel precision.
All variants used the new float32 ID-based RNG, identical producer settings, and the same LASY file.

The grid contained `1500 x 300` cells and five azimuthal modes.
The lab-frame input domain was `[-70, 0]` micrometers, with a `200`-micrometer radial extent and boost factor `1.5`.
`Simulation` converts the longitudinal domain and input timestep into the boosted frame.
The boosted timestep was `4.075316647058382e-16 s`.
The run advanced 2048 steps to boosted time `8.34624849317576e-13 s`.
The producer's complete interaction requires 17,720 steps.

The laser used `800 nm`, `2.5 J`, `30 fs`, a `24`-micrometer spot size, and super-Gaussian order `3`.
Peak plasma density was `4e18 cm^-3`, with nitrogen dopant fraction `0.03`.
The source manifests retain all physical parameters, template defaults, and numerical settings.
The shared LASY SHA256 was `66f4cb67d18e42df85203a7c60302cac6d0575527dc9e3a76a7db1572b62ab8a`.

### Runtime and memory

| Variant | Compute, seconds | Speedup versus Numba | Last 256 steps, median milliseconds | Peak GPU memory, GiB |
| --- | ---: | ---: | ---: | ---: |
| Numba FP64 | `1038.568` | `1.000x` | `666.589` | `30.048` |
| Warp FP64 | `590.665` | `1.758x` | `364.271` | `27.382` |
| Warp FP32 Hankel | `530.984` | `1.956x` | `334.554` | `30.331` |

Compute timing uses synchronized intervals around the original iteration body.
It includes within-step transfers and first-use kernel initialization. It excludes diagnostics, checkpoints, and outer setup and finalization transfers.
FP32 Hankel reduced compute time by `10.10%` relative to Warp FP64.
These complete simulation timings include the existing branch optimizations. They do not isolate each arithmetic or launch-geometry change.

Numba diagnostics took `49.625` seconds, or `4.54%` of its loop wall time.
Numba's previous producer compute time was `1038.490` seconds. The observed difference was negligible in these single runs.
That comparison does not isolate RNG overhead from changes in the realized particle histories.
GPU memory values are whole-device peaks sampled every two seconds.
FP32 Hankel did not reduce the measured whole-device memory peak. These measurements do not identify its cause.

### Final field errors

Errors compare the same boosted grid, phase, coordinates, and time without alignment.
The norm uses cylindrical cell volumes and packed modal weights `(1, 0.5, 0.5, ...)`.

\[
\epsilon_f(t)=\frac{\|f_{\mathrm{candidate}}(t)-f_{\mathrm{reference}}(t)\|_{L^2(dV)}}{\|f_{\mathrm{reference}}(t)\|_{L^2(dV)}}.
\]

| Component | Warp FP64 versus Numba FP64 | Warp FP32 Hankel versus Numba FP64 |
| --- | ---: | ---: |
| `Er` | `4.213e-13` | `1.267e-6` |
| `Et` | `3.503e-13` | `9.462e-7` |
| `Ez` | `6.040e-12` | `2.652e-5` |
| `Br` | `3.418e-13` | `8.510e-7` |
| `Bt` | `4.357e-13` | `1.314e-6` |
| `Bz` | `3.647e-12` | `1.945e-5` |
| Re-deposited `rho` | `3.517e-11` | `1.262e-3` |
| Solver `rho` | `3.518e-11` | `1.269e-3` |

The final re-deposited FP32 charge error was `0.1262%`. The largest final electromagnetic-component error was `0.002652%`.
The sampled FP32 charge error reached `0.2478%` at step 1536 during the developed pulse.
The species-charge norms summed to approximately 93 times the net-charge norm.
Ion-electron cancellation therefore increases the sensitivity of net-charge relative errors.

Final helium and nitrogen ion IDs and levels matched exactly between the FP64 backends.
FP32 had two different helium ion levels among `13,757,850` ions. All nitrogen ion levels matched.
The final helium-electron counts matched. FP32 retained two more nitrogen electrons than the FP64 runs.
Matching final ion levels does not establish matching birth times or electron trajectories.

### Initial relative-error spikes

At step 128, the Numba charge norm was `1.04480e-12` and the FP64 difference norm was `5.12500e-13`.
Their ratio was `0.4905`, although the absolute difference was tiny compared with the developed charge distribution.
The Numba field energy then was `3.54787e-25 J`, compared with `0.829269 J` at the final step.
These early spikes reflect near-zero reference norms. They do not establish large errors in developed fields.

The field trajectories also report differences divided by each component's final reference norm.
This normalization provides a fixed scale while preserving the original instantaneous relative errors.
Zero-reference ratios are undefined. Exact zeros are omitted on logarithmic axes.

## Retained evidence and reproduction

Generated results remain excluded from Git. The tables above preserve their principal findings in the branch documentation.
Paths below identify artifacts in the historical `hankel-fp32` worktree unless marked as profiling paths.
These paths identify local artifacts, not files in the repository.

| Stage | Retained local evidence |
| --- | --- |
| Isolated FP32 operator | Profiling path `roofline/results/hankel_fp32_gpu1_20261007/report.md` and `summary.json` |
| Original reconstructed comparison | Comparison JSON `benchmarks/results/hankel_simulation_2048_gpu1_20261007/plots/comparison.json` |
| Sorting and draw assignment | `benchmarks/results/ionization_order_{numba,warp}_129/ionization_step_128.npz` |
| Intermediate RNG rerun | 512-step summary `benchmarks/results/ionization_rng_fp64_512_gpu1_20261008/comparison/summary.json` |
| Producer comparisons | Original FP64 summary `benchmarks/results/producer_fp64_2048_gpu1_2223c2bf/comparison/summary.json` and final three-variant summary `benchmarks/results/producer_rng_2048_gpu1_2223c2bf_20261008/comparison/summary.json` |

The benchmark instructions in `hankel-fp32/benchmarks/README.md#producer-rng-comparison` (local artifact, not in the repository) provide launch and plotting commands.
The final comparison directory retains the following analysis files.

1. Read `field_errors_final.csv` for all final field components and both final charge definitions.
2. Read `field_error_trajectory_normalized.csv` for 17 solver snapshots and both error normalizations.
3. Read `ionization_state_errors.csv` and `species_charge_errors.csv` for identity and species checks.
4. Read `performance.csv` and each run's `step_times.csv` for direct compute and diagnostic timings.
5. Read each run's `manifest.json` and `resources.jsonl` for source hashes, inputs, precision, and device telemetry.

The relative-error trajectories in `hankel-fp32/benchmarks/results/producer_rng_2048_gpu1_2223c2bf_20261008/comparison/field_relative_error_trajectories.png` (local artifact, not in the repository) show all seven field components and all three pairwise comparisons.
The fixed-reference trajectories in `hankel-fp32/benchmarks/results/producer_rng_2048_gpu1_2223c2bf_20261008/comparison/field_error_final_reference.png` show the initial differences against developed-field scales.
The final trajectory point uses the saved solver charge. The separate final comparison also records freshly re-deposited charge.

The evidence supports backend consistency for the recorded configurations and horizons.
It does not establish full-interaction beam quality, grid convergence, stochastic ensemble agreement, timing variability, MPI parity, or derivative correctness.
The intermediate reconstruction charge residual remains unresolved.
