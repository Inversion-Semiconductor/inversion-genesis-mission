# Threads, shared memory, and fusion

This chapter records the October 9 and 10, 2026 experiments after the arithmetic, Hankel, and RNG work.
The final [Inversion study](07-inversion-full-workload.md) later measured the selected changes over the complete interaction.
Historical experiments used separate worktrees and process-local adapters, preserving existing solver files.

## Apply the optimization sequence

1. Identify repeated loads, geometry calculations, launches, and atomic updates in measured operations.
2. Test launch configurations while retaining precision and identical captured inputs.
3. Compare one fusion or reduction strategy at a time in separate worktrees.
4. Verify outputs before measuring representative solver intervals.
5. Retain changes only when complete-workload results support their benefit.

Record unsuccessful experiments, input hashes, source hashes, and rejection reasons.
Fewer atomic operations can require more scratch traffic, synchronization, launches, or register storage.
Increasing block size can change occupancy and work distribution without improving the measured workload.
Neither change guarantees a speedup.

## Arithmetic and thread configuration

The user replaced floating-exponent powers with explicit multiplication before these experiments.
The [performance chapter](02-performance.md) preserves matched Warp 1.17 kernel comparisons and the later Warp 1.18 solver rerun.
The multiplication changes are already present in Warp main at commit `9be836a`.
They retain FP64 arithmetic and do not establish a mixed-precision result.

Linear deposition initially used eight threads per block on the Ada device.
The selected configuration uses 32 threads, retaining the V100 default of 16 and the cubic default of 32.
The final matched ablation measured charge speedup `3.029x` and current speedup `2.872x` for eight-to-32-thread deposition.
Its momentum-push arithmetic ablation measured `7.751x` against the old Warp power expressions.
These isolated ratios have separate denominators and cannot predict end-to-end speedup through multiplication.

Inspect PTX or SASS when source and timing evidence leave the dominant cost unresolved.
Record compiler settings, architecture, symbols, register counts, local memory, and relevant instructions with the measured source.
Register counts alone do not prove occupancy limits or spills.
The original profiler report identifies floating `wp.pow` as a hypothesis; its resource table does not prove the instruction-level cause.
No instruction-level analysis record is retained.

## Shared-memory deposition: 256 threads on 16 by 16 cells

The user selected 256 threads and a 16-by-16 source-cell patch for this experiment.
Each thread retained the sorted accumulation for one source cell.
Shared-memory gathers combined overlapping linear stencils before global atomic writes.
Empty and padded lanes still participated in block collectives.
Particle values remained float64, and deposited fields remained complex128.

The artificial forward workload contained `1500 x 300` cells, five modes, and `1,619,106` particles.
Each configuration used three processes, five warmup cycles, and 20 measured cycles.
Timing included sorting, transforms, dispatch, and complete periodic solver cycles.
Hankel multiplication used production DGEMM.

| Configuration | Threads | Median ms/step | Speedup against 32-thread reference |
| --- | ---: | ---: | ---: |
| Existing deposition | 32 | 55.535 | 1.000x |
| Combined complex updates | 256 | 55.648 | 0.998x |
| Existing deposition | 256 | 55.817 | 0.995x |
| Shared-memory patch | 256 | 56.259 | 0.987x |

All successful runs exited with status zero and finite fields and particles.
Numerical verification passed 356 checks against existing kernels and an independent CPU calculation.
The largest scaled maximum error was `4.930e-15`.
Checks included axis particles, boundary folds, uneven cell populations, modes zero through four, and saved LWFA inputs.
The selected production path excludes this shared-memory prototype because this workload showed no speedup.

The retained report `profiling/roofline/results/forward_smem_16x16_256_20261009b/report.md` (local artifact, not in the repository) identifies the actual measurement scope.
Its `samples.csv` also records configurations that exceeded launch resource limits.
This artificial workload does not reproduce the nonlinear producer simulation.

## Five-mode deposition fusion and two-pass deposition

The `deposition-fusion` and `deposition-two-pass` worktrees started from profiling commit `2653b790`.
Each experiment used the first 2048 steps of the pinned producer configuration.
The FP64 source was `hankel-fp32`, explicitly configured for FP64 Hankel multiplication and shared ID-based ionization draws.
Each physical GPU ran its own serial process queue.
Diagnostics remained outside compute timing, while first-use compilation remained inside it.

Deposition fusion shared particle loads and geometry across modes.
For five modes, charge passes changed from five to one, and current passes changed from five to three.
The current path grouped two modes per pass, with a final unpaired mode.
The selected cell kernels used 32 threads and retained global complex128 atomic updates.

Two-pass deposition accumulated cell scratch before a separate node gather.
Its accumulation used 32 threads; its node gather used 256 threads.
Scratch required `136.10 MiB` on the producer's allocated `1852 x 300` local grid.
The smaller periodic verification case required `110.23 MiB`.

| Experiment | Reference compute s | Candidate compute s | Speedup |
| --- | ---: | ---: | ---: |
| FP64 fusion | 608.012 | 523.062 | 1.16241x |
| FP64 fusion, candidate first | 592.539 | 511.219 | 1.15907x |
| FP64 two-pass | 594.154 | 608.244 | 0.97684x |
| FP32 Hankel with fusion | 542.073 | 458.665 | 1.18185x |
| FP32 Hankel with fusion, candidate first | 532.797 | 450.551 | 1.18254x |

Every ratio uses a reference from the same physical GPU and the same Hankel precision.
The candidate-first confirmation changed execution order and reproduced the fusion gain.
Fusion improved all four populated FP64 intervals on GPU 0 by approximately `1.183x` to `1.187x`.
Two-pass lost in all eight recorded intervals, so the selected production path excludes it.
The final FP64 runs contained `56,672,758` particles.

FP64 fusion and two-pass preserved final counts, ion IDs, and ionization levels.
Their largest final relative volume L2 field errors were `2.72e-11` and `2.05e-11`, respectively.
The probes passed 336 fusion checks and 280 two-pass checks.
The deposition report `profiling/roofline/results/deposition_workload_summary_20261010/report.md` (local artifact, not in the repository) preserves interval timings and numerical details.

At this stage, fusion plus FP32 Hankel took `458.665 s`, versus original Numba's `1064.037 s`, giving `2.31986x` compute speedup.
That ratio covered 2048 steps and combined backend, threads, RNG, precision, and fusion differences.
It is historical evidence, separate from the final `2.561x` complete-run result.
FP32 trajectory differences also changed ionization decisions despite fixed random draws.
One repeated FP32 baseline changed an ionization level, so one repeat did not define a statistical uncertainty envelope.

## Additional fusion candidates

The next FP64 matrix retained deposition fusion as its Warp reference.
Matched Numba controls used the same ID-based draws, seed zero, helium stream zero, and nitrogen stream one.
Particles and Hankel multiplication remained float64; fields remained complex128.
Warp retained CUDA graphs, and Numba retained its original launch path.

| Candidate | Reference compute s | Candidate compute s | Speedup over deposition-fused Warp | Selection |
| --- | ---: | ---: | ---: | --- |
| Gather reset plus five modes | 523.334 | 489.200 | 1.070x | Include. |
| Momentum and position push | 523.334 | 520.011 | 1.006x | Exclude; benefit was small in one process. |
| Field filter | 510.366 | 509.882 | 1.001x | Exclude; benefit was below 0.3%. |
| Field basis conversions | 510.366 | 508.962 | 1.003x | Exclude; benefit was below 0.3%. |
| Inverse-FFT normalization | 510.366 | 509.497 | 1.002x | Exclude; benefit was below 0.3%. |

Gather fusion combines field reset and five mode gathers in one 128-thread launch.
It preserves mode accumulation order and FP64 arithmetic.
The combined experiment included other exploratory fusions and measured `1.082x` over its fresh Warp reference.
That aggregate result does not justify including each marginal candidate.
The fusion matrix report `profiling/roofline/results/fusion_fp64_summary_20261010/report.md` (local artifact, not in the repository) records all variants, field errors, particle checks, and telemetry.

## Selected production behavior

Integration commit `edc52c4` includes 32-thread linear deposition, five-mode deposition fusion, and five-mode gather fusion.
Local `main` was fast-forwarded to this approved commit.
Fusion applies to linear particles with exactly five modes; other cases retain existing paths.
Set `species.fuse_deposition = False` or `species.fuse_gather = False` for independent ablations.
Both switches default to `True`.
FP64 remains the precision default, and the optional FP32 Hankel path is described in the [precision chapter](05-hankel-fp32-and-ionization-rng.md).

The selected implementation excludes shared-memory deposition, two-pass deposition, and the marginal push, filter, basis, and normalization fusions.
No rejected worktree or result archive was deleted.
The [complete workload chapter](07-inversion-full-workload.md) measures this selected combination against original and matched-RNG Numba.

Integration verification passed 101 GPU tests and 39 CPU tests, with 14 skipped CPU cases.
The GPU total includes 53 precision/RNG tests, 17 existing coupled-solver tests, and 31 new fusion tests.
The current openPMD reader required `OPENPMD_VERIFY_HOMOGENEOUS_EXTENTS=0` for constant-record extents.
No dependencies changed.
The measured numerical modules were preserved; an extra blank line was removed from the fused deposition file.

## Reproduce focused historical checks

Run these commands from the named historical worktree in the existing `fbpic-warp` environment.
Use fresh output directories and physical GPU 0 for new measurements.

```bash
# profiling worktree
python roofline/verify_deposition_smem.py --gpu 0 --small --verify-only
python roofline/run_forward_smem.py --gpu 0 --blocks 256

# deposition-fusion worktree
python roofline/verify_fused_deposition.py --gpu 0

# deposition-two-pass worktree
python roofline/verify_deposition_two_pass.py --gpu 0 --output roofline/results/new_two_pass_check
```

Saved reports and run manifests preserve the historical source and commands.
Running an edited worktree produces a new measurement, even when its branch name is unchanged.
