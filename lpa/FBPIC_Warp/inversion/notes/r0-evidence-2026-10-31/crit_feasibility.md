# R0 critique: feasibility, cost, code claims, and test design

Critic key: `crit_feasibility`. Date: 2026-10-31. Scope: sections `sec_vm.md`, `sec_gradf.md`, `sec_rv.md`, `sec_multi.md`, `sec_tests.md`.

Labels: **[code]** read in the worktree at `path:line` (paths relative to `lpa/FBPIC_Warp/`); **[plan]** `DIFFERENTIABLE_INVERSION_PLAN.md`; **[measured]** measured by this critic, CPU only, scripts in `inversion/notes/r0-evidence-2026-10-31/work_crit_feasibility/`; **[derived]**; **[estimated]**; **[section]** a claim or number of a section under review.

The measured items read the cached P7 runs in `inversion/runs/p7_fe7188bf9100/` and `inversion/results/p7_jacobian.json` (read only). No GPU was used. Nothing in the worktree was changed.

## 0. Verdict

**No-go for R3 as a cost play. Conditional and narrow go for a Twin-S-native quality test.**

1. Pure particle-pdf (the Track R method of plan Section 7) cannot represent the Twin-S background. All three derivation sections agree, with independent evidence. The only viable variant is a hybrid that keeps Lagrangian tangents (R-v or R-v-s).
2. A variant that keeps Lagrangian tangents has a marginal cost per parameter of at least about 1 primal run (P5, no EV: 0.96) and 2.47 as built (P7). One-sided FD costs 1. No forward Track R variant that is accurate on cold particles beats FD on cost at any $K$ [derived from measured inputs].
3. On Twin-S itself, the residual map at $N_t = 2000$ has no scale-free derivative better than about 30 to 50%. Local secants at $h \le 0.01$ vary by 41 to 54% of the column norm across neighbouring centers, and the R2 regularity rule fails for 7 of 11 residuals when applied to Twin-S [measured]. A regular method therefore computes a coarse-grained derivative at a tunable scale, like FD with $h$.
4. FD at $h = 0.005$ to $0.02$, central or one-sided, already passes the guarded gate at the T0 solution [measured], and FD recovery already passed the Stage A gate [plan 12.5]. The Stage B blocker is global search, which exact gradients do not fix [plan 12.5, 2.6].

The design note should state these four points before it asks for R1 to R3 effort.

## 1. Code claims checked

| Claim | Location | Result |
|---|---|---|
| Tangent state already carries a weight tangent `w` | `fbpic/autodiff/state.py:105` (`TANGENT_PARTICLE_NAMES` ends with `"w"`), `:126-140` | True [code] |
| Deposit tangent uses `dw` | `fbpic/autodiff/tangent.py:994` (J), `:1014` (rho) | True [code] |
| Loss contraction uses `t["w"]` | `fbpic/autodiff/loss.py:273` | True [code] |
| Newborn gets `dw = dW'` and the parent `xi` | `fbpic/autodiff/ionization.py:762-764` | True [code] |
| Vay tangent applies the field-tangent kick `econst * dE` | `fbpic/autodiff/tangent.py:582-599` | True [code] |
| Production `deposit_fused(particles, grids, fieldtype, weight)` | `fbpic/particles/deposition/cuda_methods_fused.py:471`, lane option `:327` | True [code]. One call per weight array; no $K$ batching. |
| Twin-S uses curl-free correction; species load cold | `inversion/twin_config.py:236`, `:272-275` (no thermal momentum) | True [code] |
| P7 growth diagnostics are infinity norms, maximum over all species | `fbpic/autodiff/driver.py:118-136` | True [code]. No per-species or median data exist. |
| Deposit tangent keeps a per-thread matrix of $4 \cdot$ components $\cdot K \times 2$ doubles | `fbpic/autodiff/tangent.py:888` | True [code]: 48$K$ doubles for J with $N_m = 2$. Register spill above small $K$. |
| `k = 0` of the curl-free correction is set to zero | `fbpic/fields/spectral_grid.py:140-144` | True [code] |

Additional fact that the sections missed [code]. `driver.jvp` takes `callback(state, tangents, decisions)` after every step (`fbpic/autodiff/driver.py:150-182`), and `value_and_grad` forwards `**jvp_kwargs` (`driver.py:249`). An R-v reset, and the per-species growth diagnostics, can therefore run as a callback in an `inversion/` script. No verified kernel and no file in `fbpic/autodiff/` changes. This keeps the P7 step 5 cache valid (its key is the hash of `fbpic/autodiff/*.py`, plan 12.10 "To close Phase P7", item 2) and respects D17.

Claims that do not hold:

1. `sec_vm` Section 4.3 item 2: "Hybrid R-v ... about 2.5 run-equivalents if the weight markers use the cheap deposit." [derived] For a reset period $P > 1$, the Vay tangent adds a kick to every converted marker on the first step after a reset (`tangent.py:582-599`), so $\xi \neq 0$ again. All markers then need `deposit_tangent` until the next reset. The cheap `deposit_fused` path applies only to $P = 1$. Hybrid R-v costs the present 3.47 run-equivalents plus the reset cost.
2. `sec_vm` Section 4.3 item 1 and `sec_multi` Section 2.3: "each further parameter adds about 1.2 to 1.8" or "$c_1 \approx 0.4$ to 0.8". These numbers hold only for markers that are in pure weight form every step, which is the variant that fails on cold particles. For any design that keeps Lagrangian tangents, the measured marginal cost is 2.47 [plan 12.10 step 3], and about 0.96 without the dense-cell serialization [plan 12.7].
3. `sec_rv` Section 5.2 item 2 and 5.3: Gram assembly "with atomics ... 2 to 5 deposit costs (50 to 120 ms)". [derived] With degree 4 in 5 variables there are 126 monomials per slab, and a particle touches 2 slabs, 4 grid nodes $\times$ 3 mode rows, and 13 loss rows: about 277 rows. The outer product is about $3.8\cdot10^4$ products per particle, and all particles of one slab add into the same $126^2$ block. Atomic assembly is not viable. A per-slab GEMM costs about $3 \cdot 126^2 \cdot 2 \approx 9.5\cdot10^4$ FP64 flops per particle, or about $1.3\cdot10^6$ at degree 6. On the RTX 6000 Ada (FP64 at 1/64 of FP32, about 1.4 TFLOP/s) that is about 0.13 s (degree 4) to 1.8 s (degree 6) per reset for 2 M converted particles, and 5 times more for all 10 M `elec_from_N` [estimated]. At 20 to 40 resets the overhead is 1% to 25% (2 M) or 5% to 120% (10 M) of the 298 s gradient, not 1% to 10%. Raw monomials of $u_z \approx 12 \pm 4$ to degree 6 also need per-slab centering and an orthogonal basis.

## 2. Cost and memory, recomputed

Inputs [plan 8, 12.10 step 3, 12.7]: primal run 86 s; $K = 1$ Lagrangian gradient 3.47 run-equivalents ($c_1 = 2.47$); without EV $c_1 = 0.96$; profile at step 1200: 228 ms per tangent step, deposit tangents 81 + 40 ms, primal deposits 24 + 14 ms; peak 3.78 GB used. FD recovery used 5 to 8 Jacobians per 1-D solve [plan 12.5].

Per-Jacobian cost, including the residual run at the iterate (`work_crit_feasibility/cost_compare.py`, output `cost_compare.out`) [estimated from measured inputs]:

| Option | K = 1 | K = 2 | K = 5 | K = 10 | K = 25 |
|---|---|---|---|---|---|
| Central FD, $h = 0.02$ | 3.0 | 5.0 | 11.0 | 21.0 | 51.0 |
| One-sided FD | 2.0 | 3.0 | 6.0 | 11.0 | 26.0 |
| R-v as built ($c_1 = 2.47$, reset 0.05 to 0.9) | 3.5 to 4.4 | 6.0 to 6.8 | 13.4 to 14.3 | 25.8 to 26.6 | 62.8 to 63.7 |
| R-v after a deposit-tangent rewrite ($c_1 = 1.0$ to 1.3) | 2.0 to 3.2 | 3.0 to 4.5 | 6.0 to 8.4 | 11.1 to 14.9 | 26.1 to 34.4 |
| Pure weights ($c_0 = 2$ to 8, $c_1 = 0.7$ to 0.8), not accurate on cold particles | 3.7 to 9.8 | 4.4 to 10.6 | 6.5 to 13.0 | 10.0 to 17.0 | 20.5 to 29.0 |
| Dipole pairs (`sec_gradf` model) | 4 to 6 | 5 to 8 | 8 to 14 | 13 to 24 | 28 to 54 |

Reading [derived]:

1. Break-even with one-sided FD needs $c_0 + c_1 K < K$, so $c_1 < 1$. Every variant that keeps Lagrangian tangents has $c_1 \ge 0.96$. It cannot beat one-sided FD at any $K$.
2. R-v as built fails the cost bar of `sec_tests` Section 6.4 (marginal cost at most 2 primal runs). The sections recommend R-v and also set this bar. The design note must say that a GO for R-v requires a rewrite of the verified P4 deposit tangent first (dense-cell split, as in the production lane kernel), with re-verification.
3. Best case for multi-parameter work: pure weights at $K = 25$ save at most about 20% against one-sided FD (20.5 against 26 runs). That variant is the one that the cold-limit result rules out.
4. Wall time at $K = 25$ with one-sided FD: 3.1 to 5.0 h per solve on one GPU, 1.6 to 2.5 h on two. FD runs are independent and split perfectly over the two GPUs. A tangent pass splits only over $K$, and each GPU repeats the primal and $c_0$.

Memory: the Twin-S field tangent is about 0.013 GB per parameter, and particle tangents are 0.9 GB per parameter (plan 8). Memory does not limit any Twin-S option at $K \le 25$. The `sec_gradf` pair estimate (6 to 7 GB for $10^8$ companions) omits the double buffer of the sort; about 12 to 15 GB is more likely [estimated]. It still fits in 48 GB.

## 3. Is there a regular derivative to compute? Twin-S evidence

The sections rest the continuum-stability assumption on "FD at $h = 10^{-2}$ and $10^{-3}$ agree" (plan 12.10 step 4). That pair differs by 17% of the column norm at the nominal point, and by a factor 2.4 for $\langle u_x\rangle$ [derived from the plan table]. I used the 41-point T0 solution slice (spacing 0.0025 in $\hat E_L$) to test the assumption directly.

Script `work_crit_feasibility/slice_secant_spread.py`, output `slice_secant_spread.out` [measured]. Central secants at all 17 centers within $\pm 0.02$ of 0.588:

| $h$ | mean $\lVert D_c - \bar D\rVert / \lVert J_{sm}\rVert$ | min cos$(D_c, J_{sm})$ | median cos |
|---|---|---|---|
| 0.0025 | 0.54 | 0.25 | 0.92 |
| 0.005 | 0.49 | 0.33 | 0.93 |
| 0.01 | 0.41 | 0.48 | 0.96 |
| 0.02 | 0.25 | 0.44 | 0.97 |

The spread falls by only 2.2 times while $h$ grows 8 times. White roughness would give 8 times. The map therefore has correlated structure on the scale 0.005 to 0.02 in $\hat E_L$ (0.4% to 1.6% of $E_L$).

Script `work_crit_feasibility/r2b_on_twins.py`, output `r2b_on_twins.out` [measured]. I applied the R2 test (b) rule of `sec_tests` Section 5.3 (secants at $h = 0.01$, 0.02, 0.04 agree within 10% plus 2 combined SE) to Twin-S, with the across-center spread as the SE proxy. Only 4 of 11 residuals pass. $D_{0.01}$ and $D_{0.04}$ differ by 105% of the column norm. $\langle\zeta\rangle$, $\log\sigma_x$, $\log\sigma_{u_x}$, $\log\sigma_y$, $\log\sigma_{u_y}$ and the log spread change sign between $h = 0.01$ and $0.04$. Part of this is curvature (the slice $J$ has a bump at 0.555 to 0.575), and part is roughness. The scale window between roughness and curvature is narrow or empty at the 10% level.

Consequences [derived]:

1. On the plan's own data, Twin-S would get "NO-GO, physics" from R2 test (b). The rule confuses curvature with irregularity. It must be replaced (Section 5).
2. A perfect "regular" derivative at a small effective scale is not unique to better than about 40 to 50% between nearby $\theta$. The target of Track R is a coarse-grained quantity, and its scale ($\sigma_u$, $T$, basis degree) plays the role of $h$. This agrees with `sec_gradf` G.3.3 and the 1D tests.
3. The same holds in every 1D trapping toy of R0. `sec_multi` 5.3 item 1 says "the macroscopic response is regular". Its own output (`work_multi/adjoint_pdf_1d1v_N5e4_Nt2000.out`) shows CRN FD at $h = 0.02$, 0.01, 0.005 of 14.0, 6.2, 0.14 at $t = 85$, and 23.0, 10.2, $-2.4$ at $t = 135$, with seed SD of 1.6 to 3.6. The reference is a quadratic fit over $\theta \in [0.05, 0.11]$ at $\theta_0 = 0.08$, a secant on the scale of 40% to 75% of $\theta$. The "+24 bias" of particle-pdf is measured against a scale-dependent reference. Kill criterion 0 of `sec_multi` is ill-posed as written. In `sec_gradf` G.4 ($T = 30$) the cold-beam reference moves from $-0.98$ to $+0.44$ with $h$. In the R2 pilot at $t = 75$, single-loading FD at $h = 10^{-2}$, $5\cdot10^{-3}$, $10^{-3}$ gives $-23.3$, $-2.05$, $-46.5$ for $\langle u\rangle$.
4. No R0 prototype produced a converged regular sensitivity after trapping or mixing: `sec_rv` E2 (stable but wrong, 0.21 to 0.39 against 0.74), `sec_multi` 5.2 (biased, mesh dependent), `sec_gradf` G.4 (scale dependent). The paper has no trapping case. The only positive evidence is the Debye case and warm-beam pairs, which do not have the Twin-S geometry.

## 4. Can R1 and R2 falsify the method?

### 4.1 R1

R1 reproduces the paper's warm, non-trapping Debye case. The pilot already matches it (`sec_tests` 2.3: 11.79 and 12.86 against 12.4). R1 tests the implementation, not the hypothesis. It cannot fail for the reasons that threaten Twin-S (cold sheets, trapping, filamentation). The one 1D test with falsifying power, the warm driven-trapping test of `sec_multi` Section 5, is not in the `sec_tests` R1 plan. `sec_multi` kill criterion 0 refers to an "R1 driven-trapping test" that the R1 design does not contain. Add it to R1, with a matched-scale reference (Section 5).

### 4.2 R2: could R2 pass while FBPIC fails?

Yes. R2a differs from Twin-S in the properties that decide the outcome [derived]:

1. No transverse momentum. The laser quiver makes the cold Twin-S sheet slope about 1.6 in $u_\perp$ per $z$ cell (`sec_gradf` G.2.4). That is the hardest case for any $\nabla_u f$ estimate, and only the optional R2b tests it. R2b must be mandatory if R2 is kept.
2. The driver is prescribed and does not evolve; $a_0$ enters every step as an explicit source. In Twin-S, $E_L$ enters only through the initial laser fields, and the laser tangent evolves self-consistently (plan 4.8). The Twin-S field tangent grows by about $3\cdot10^3$ from step 1200 to 2000 (plan 12.10 step 4). R2a has 1D Gauss's law with no field memory.
3. Codimension. A cold sheet in 1D-1V has codimension 1 in $u$. The Twin-S background has codimension 3, and the EV bunch codimension 2 (`sec_gradf` G.2). Noise grows as $1/\sigma_u$ per kicked direction, and mesh cost as $l^{2d}$.
4. Rate. R2a has $\lambda \approx 0.23\,\omega_p$; Twin-S about $0.72\,\omega_p$ at peak (`sec_tests` 8 item 4). The type of chaos (N-body or mean-field) is not known for Twin-S, and R2a cannot settle it for Twin-S.
5. Whitening. The R2 gate uses seed SDs; Twin-S uses model-error $\Delta$. At $t = 58$ the R2 exact column passed the guards only because the dominant whitened components were regular (`sec_tests` 2.4). The gate outcome depends on an arbitrary whitening.

Could R2 fail while FBPIC works? Also yes. 1D has no transverse phase mixing and no birth-phase spread in $u_\perp$. `sec_gradf` reports no $\sigma_u$-plateau for a 1D cold beam, and notes that the Twin-S birth-phase spread may smooth the folds. R2 is neither necessary nor sufficient for Twin-S.

### 4.3 Does R2 reproduce the P7 chaos?

Partly. The pilot shows exponential tangent growth and a wrong-signed exact column at $t = 75$ (one loading). The growth is fastest in the cold plasma electrons (0.23) and slower in the EV electrons (0.11 to 0.20). The mechanism is not identified (`sec_tests` 8 item 3). The single-loading FD at $t = 75$ is itself not regular (Section 3, item 3), so the "wrong sign" is not yet established against a regular reference. It is the phenomenon, not proven to be the same mechanism.

## 5. Go/no-go thresholds and the guarded metric

1. $\kappa = \lVert I - J_A^+ J_{ref}\rVert$ and $\cos \ge 0.9$ are sound as Gauss-Newton contraction checks, and the scale invariance of $\delta$ is correctly derived (`sec_tests` 6.1). I confirm the P7 values: $J_{AD}$ has $\cos = 0.247$; central FD at $h = 0.02$ has $\kappa = 0.087$ [measured, `slice_scale.out`].
2. One-sided FD passes the gate at the T0 solution. Forward and backward differences at $h = 0.005$, 0.01, 0.02 give $\cos = 0.92$ to 0.99, $\kappa = 0.003$ to 0.39, $\delta \le 2.6\cdot10^{-3}$ [measured, `onesided_gate.py`, `onesided_gate.out`]. So the cheapest FD Jacobian is already "usable" by the gate that Track R must pass.
3. The gate cannot show that Track R is better than FD. Its reference is a cubic fit over $\pm 0.05$, a large-scale secant, and the thresholds were calibrated so that FD passes. Track R can at best tie FD under it.
4. Per-point $\cos \ge 0.9$ can fail a perfect method. Local secants at matched small scale have minimum cosines of 0.25 to 0.48 to $J_{sm}$ (Section 3). Use a reference at matched scale, or gate the median over a slice and report the minimum.
5. R2 test (b) must separate curvature from roughness. Fit $D_h = D_0 + c\,h^2 + \text{noise}$ over several $h$, or use local cubic fits over two window sizes. Test that the extrapolated $D_0$ is stable within its uncertainty, and report the roughness amplitude separately.
6. The off-solution reference slices are incomplete (76 of 123 P7 step 5 runs missing, about 6.3 GPU-h). Any edit of `fbpic/autodiff/*.py` invalidates the cache key (plan 12.10). Finish P7 step 5 before R3 edits, or keep R-v in a callback.
7. `sec_vm` kill criterion 5 and `sec_tests` cost criterion are consistent with each other, but both are failed now by R-v as built ($c_1 = 2.47$). State that explicitly.

## 6. Design gaps that affect feasibility

1. `sec_vm` Section 6 first-moment projection drops the intra-bin $\mathrm{Cov}(\xi, \mathbf z)$ at every reset. For the selected beam, $\delta\sigma_k^2 = 2\langle(\xi_k - \bar\xi_k)\,\delta\xi_k\rangle$, so the lost term carries the regular sensitivity of the spread residuals. Six of the 11 residuals are spreads (plan 2.4 items 3 and 4). With the projection $\Delta W_p = W_p(\mathbf z_p - \bar{\mathbf z})^T C^{-1}\bar\xi$, the new weights change a bin's second moment only through its third moments, which vanish for symmetric bins [derived]. The `sec_rv` Galerkin form with the 13 loss rows keeps them at the reset instant. Use that form, and add the spread residuals to the projection-fidelity kill criterion.
2. Cold nodes make the Galerkin system rank deficient. If $\delta J$ rows are added, a cold node gives $J$ rows equal to $v_0$ times the $\rho$ rows, so $\delta\rho = 0$ and $\delta J = q n \delta v$ cannot both hold. This is the `sec_vm` cold-limit result in matrix form [derived]. Only R-v-s (bunch only) is consistent. If the background carries the chaos (`sec_tests` pilot: plasma tangents grow fastest), R-v-s fails.
3. $K$ batching of the deposit tangent spills registers: 48$K$ doubles per thread for J (`tangent.py:888`). The cost model must tile over $K$ (`sec_multi` notes this; `sec_gradf` does not).

## 7. Multi-parameter path: effort

[estimated from section numbers]

| Item | Days |
|---|---|
| R1 + R2a + R2b (mandatory, Section 4.2) | 9 to 10 |
| R-v on Twin-S (callback prototype, then R3 port) | 4 to 7, then 8 to 15 |
| Deposit-tangent rewrite and re-verification (needed for any cost case) | 3 to 5 |
| $K$ batching with tiling | 2 to 4 |
| R4 and P7 step 5 completion | 3 to 5 plus 6.3 GPU-h |
| Forward multi-parameter total | about 30 to 45 |
| Adjoint particle-pdf (only pure-weight form; needs deterministic deposits, which production lacks, plan 6.3 rule 7) | plus 15 to 25 |

Stage B needs a global-search decision first (plan 2.6 Stage B item 2, plan 12.5). Track R does not change it.

## 8. Alternatives

Ranked for the user's multi-parameter goal [estimated unless marked].

1. **One-sided or central FD at $h = 0.01$ to 0.02.** Cost $1 + K$ or $1 + 2K$ per Jacobian, perfectly parallel over 2 GPUs. Passes the guarded gate at the T0 solution [measured]. Already recovered $E_L$ and $z_f$ in 5 to 8 Jacobians [plan 12.5]. Zero new code. Risk: curvature at $h \ge 0.03$ (cos 0.33 at $h = 0.05$) and parameter-specific $h$.
2. **Model-based derivative-free least squares** (DFO-LS or POUNDERS type: interpolation Jacobians of the residual vector, trust region). After $K + 1$ initial runs, each iteration typically adds 1 to 2 runs and reuses old points; the sample radius acts as a large $h$. This is the cheapest per iteration at large $K$. Iteration counts at $K = 25$ on this problem are not known; the literature claim is from memory, not checked here. A Stage A test costs a few GPU-hours.
3. **Surrogate (GP or quadratic) on the global design.** Stage B and later need a global design anyway (plan 2.6). A surrogate fit gives smooth Jacobians at the design scale and addresses the global-search blocker. For $K = 25$, a few hundred runs (6 to 12 GPU-h on 2 GPUs).
4. **Ensemble-smoothed FD** (average of secants over 2 or 3 $h$, or over loadings). Needed only if single-$h$ FD fails the gate somewhere. Costs 2 to 3 times FD.
5. **Short-horizon AD.** Exact only to about step 500 (strict) or 1000 (reported). The observable is at step 2000. Not usable for this loss.
6. **Track R (R-v-s).** Cost at least FD; quality at best equal to FD under the agreed gate; 30 to 45 days to a multi-parameter forward capability. Its one possible advantage: a linearized derivative has no curvature bias, so it may help a parameter for which no single FD $h$ lies between roughness and curvature. That is a hypothesis, not evidence.

## 9. Recommended order, if the user keeps D34

1. Twin-S-native diagnostics first, through `driver.jvp` callbacks, with no `fbpic/autodiff` edits (about 1 day plus about 3 GPU-h): per-species and median tangent growth; particle-count doubling (`p_nt` 8 to 16) for the chaos type (`sec_rv` kill 1); slice roughness at `p_nt` 16 against 8 (roughness that falls as $N^{-1/2}$ supports an N-body type and a meaningful continuum target).
2. Finish P7 step 5 (6.3 GPU-h) before any `fbpic/autodiff` edit.
3. R-v-s as a CuPy callback prototype on Twin-S with a $T$ sweep, judged against matched-scale references (4 to 7 days). This is the most direct falsifier and replaces most of R2a.
4. Keep R1 short (1 day), and add the warm driven-trapping test.
5. Run DFO-LS or one-sided FD on Stage A in parallel as the cost baseline.
6. Go to R3 only if item 3 passes per seed and the deposit-tangent rewrite gives $c_1 \le 1.3$ by measurement.

## 10. Strongest honest no-go case

Track R in its paper form cannot carry the cold Twin-S background: a weight change in a cold cell carries no current response, and three sections show it analytically and in 1D. The remaining hybrid keeps the existing Lagrangian tangents, so each parameter costs at least one primal run and 2.47 as built; one-sided FD costs one run and already passes the agreed usability gate at the T0 solution. Twin-S has no scale-free derivative better than about 40 to 50% at $h \le 0.01$, and every R0 trapping prototype gave biased or scale-dependent values after trapping. So Track R can at best reproduce what FD already gives, at higher cost and with 30 to 45 days of work. Its multi-parameter cost benefit exists only in an adjoint of the pure-weight form, which is the form that fails on cold particles. The Stage B blocker is global search, which no gradient method fixes.

## 11. Files

1. `work_crit_feasibility/slice_scale.py`, `slice_scale.out`: secants against $h$ at 0.588, roughness, gate metrics.
2. `work_crit_feasibility/slice_secant_spread.py`, `slice_secant_spread.out`: secant spread over neighbouring centers.
3. `work_crit_feasibility/r2b_on_twins.py`, `r2b_on_twins.out`: R2 test (b) rule applied to Twin-S.
4. `work_crit_feasibility/onesided_gate.py`, `onesided_gate.out`: gate metrics of one-sided FD.
5. `work_crit_feasibility/cost_compare.py`, `cost_compare.out`: cost table and the 11-D cosine null probability ($P(\cos \ge 0.9) = 3.3\cdot10^{-5}$).

Run each with `python3 -I <script> inversion/runs/p7_fe7188bf9100 inversion/results/p7_jacobian.json` from `lpa/FBPIC_Warp` (`r2b_on_twins.py` takes only the cache directory; `cost_compare.py` takes no arguments).
