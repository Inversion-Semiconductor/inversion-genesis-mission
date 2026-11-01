# Finding 3 (2026-11-01): Track R feasibility study (Phase R0) and the pending closure recommendation

Status: current as a study result. The recommendation is **pending**. It is not a user decision.
Source: [design note](../../../notes/r0-design-note-2026-10-31.md) (read completely), five derivation sections, two critiques, saved outputs.
Verification level: reports read; selected saved outputs inspected; no check rerun by this checkpoint. The design note states that its author re-ran three critic scripts with identical output (`work_note/rerun_*.out`).

## What Track R is

Track R follows Chung, Bond, Cyr, and Freund (JCP 400, 108969, 2020). The method differentiates the Vlasov equation first. It then carries \(\delta f\) on particle weights. The goal is to avoid the exponential growth of Lagrangian tangents. See the plan Section 7 and the design note Section 3 for the FBPIC mapping.

## Mathematics and the step that changes the interpretation

Linearized Vlasov source, with \(\mathbf u = \mathbf p/(mc)\), \(\delta\mathbf a = \frac{q}{mc}(\delta\mathbf E + \mathbf v\times\delta\mathbf B)\):

\[
\partial_t \delta f + \mathbf v\cdot\nabla_x \delta f + \mathbf a\cdot\nabla_u \delta f = -\nabla_u\cdot(f\,\delta\mathbf a) + \delta G .
\]

Identity (design note A.1): because \(\mathbf v = c\nabla_u\gamma\), \(\nabla_u\cdot\delta\mathbf a = 0\), so \(-\delta\mathbf a\cdot\nabla_u f = -\nabla_u\cdot(f\,\delta\mathbf a)\). Both critics re-derived it. It holds for the relativistic magnetic term.

Consequence: the weight form is a score-function (likelihood-ratio) estimator, \(\Delta\hat W_p = -W_p\,\Delta t\,\delta\mathbf a_p\cdot\nabla_u\ln f\). The Lagrangian tangent is the pathwise estimator of the same quantity. The step that changes the method: the score needs \(\nabla_u \ln f\). For a cold sheet \(f \propto \delta^3(\mathbf u - \mathbf U)\) this does not exist. Any scheme must smooth \(f\) over a width \(\sigma_u\). The smoothing bias is \(O(\sigma_u^2/v_T^2)\) for a warm \(f\) and \(O(\sigma_u^2/\ell_\psi^2)\) for a cold \(f\) (design note A.3, equations A2 to A4). Here \(\ell_\psi\) is the momentum scale of the pulled-back observable.

Assumptions: smooth kernel of variance \(\sigma_u^2\); volume-preserving flow. The estimate is bounded by \(\mathrm{osc}(\psi)/\sigma_u\) per unit source time. That bound explains why such estimates look regular. The critic states: boundedness is not evidence of accuracy.

Other derived items: adjoint is the exact transpose of the forward weight form (A.6), so it is regular exactly when the forward scheme is. Reverse-mode AD of the PIC step carries \(\nabla\lambda\) and grows like the forward tangent. The cold-limit theorem: a charge-neutral weight change in a cold cell gives zero current change, but the exact response is \(q n \delta\mathbf v \ne 0\) (A.2).

## Algorithm sources

| Item | Source |
|---|---|
| Particle-pdf (collocated) | Chung et al. 2020, Eq. (36), (41), (42), (44), pp. 15 to 18. Paper not opened by this checkpoint. Page numbers are as reported by the design note. |
| Variant R-v (Galerkin conversion at resets) | Design note 4.3 and A.4; [sec_rv.md](../../../notes/r0-evidence-2026-10-31/sec_rv.md). Collocated particle-pdf equals R-v with reset interval \(T = 1\) (derived). |
| Adjoint | [sec_multi.md](../../../notes/r0-evidence-2026-10-31/sec_multi.md); Bartsch et al., arXiv:2304.02083 (summary only in the source). |
| Cost model | [sec_multi.md](../../../notes/r0-evidence-2026-10-31/sec_multi.md) Section 2.3; [cost_compare.out](../../../notes/r0-evidence-2026-10-31/work_crit_feasibility/cost_compare.out). |
| DFO-LS as alternative | Cartis et al., arXiv:1804.00154. Cited from memory in the source. Not checked. |

## What worked

1. The algebra holds. Two critics found no sign or factor error (`crit_physics` Section 1).
2. Warm Debye case: particle-pdf gave \(11.79 \pm 0.35\) (non-collocated) and \(12.86 \pm 0.48\) (collocated) against a continuum slope of \(12.4 \pm 0.2\) (`sec_tests` 2.3). The seed SD is 3.5 times the paper's, unexplained.
3. Galerkin conversion (E1, warm 1D case, 8 seeds, \(N = 10^5\)): the mean agreed with the reference within the seed standard deviation for all \(\sigma \ge 0.01\) and all observables, with one stated exception. At \(\sigma = 0.5\) the observable \(\cos(x + 20u)\) has an absolute offset of 0.010: saved mean 0.002804 against reference 0.012921, seed standard deviation 0.003928 (`sec_rv.md:281`; `work_rv/e1_results_N100000.json:13` and `:122`). The source explains it by an observable that oscillates faster in \(u\) than the width of \(f\). Agreement within seed scatter at finite sample size does not establish unbiasedness. Moment preservation is exact only for moments inside the chosen basis space; the source states that \(\langle u\rangle\) and \(\langle u^2\rangle\) are in that space (`sec_rv.md:282`). In the two-stream case (E1b) the `sel_u` error depends on basis degree \(p\): at \(\sigma = 0.3\) it is \(-5.6\%\), \(-1.2\%\), \(-0.6\%\) for \(p = 2, 4, 6\). The \(\cos(x + 5u)\) error is \(-21\%\), \(-9\%\), \(-10\%\) with seed standard deviation about 10%. Degree 4 to 6 is needed when one node holds two populations (`sec_rv.md:290` to `:296`). Exactly cold sheets (\(\sigma = 0\)) fail silently (`sec_rv.md:282`).
4. The 1D adjoint equals the forward estimate to \(9.7\cdot10^{-16}\) with bounded adjoint values (RMS 1.39 to 1.59) while \(\xi_v\) grew to \(2.9\cdot10^{11}\). Confirmed in [adjoint_pdf_1d1v_N5e4_Nt2000.out](../../../notes/r0-evidence-2026-10-31/work_multi/adjoint_pdf_1d1v_N5e4_Nt2000.out) (relative difference \(9.71\cdot10^{-16}\), \(\lambda\) rms 1.59 to 1.39).

## What failed or did not transfer

| Test | Result | Reading |
|---|---|---|
| Cold beam, prescribed waves, \(T = 10\), bias | 10% bias needs \(\sigma_u \approx 10^{-3}\); the standard error with \(8\cdot10^4\) base particles (8 seeds of \(10^4\)) is still 17% for \(J_1\) (`sec_gradf.md:175` to `:184`). This row reports bias and its standard error at \(T = 10\) only. It does not test the noise law below. | Saved: [small_sigma_T10.log](../../../notes/r0-evidence-2026-10-31/work_gradf/small_sigma_T10.log) holds three \(\sigma\) rows. I did not map columns to the quoted figures. |
| Cold beam, prescribed waves, decorrelated \(T = 30\), \(N = 4000\) base particles, dipole-pair estimator: noise scaling | The pair standard error times \(\sigma\) is 0.0064, 0.0068, 0.0077, 0.0076 for \(\sigma = 0.01, 0.03, 0.1, 0.3\). The standard error is therefore proportional to \(1/\sigma\) and the variance to \(1/\sigma^2\) in this tested regime (`sec_gradf.md:210`, `:224`, `:228`; saved log `cold_beam_bias.log`, not opened here). `crit_physics.md:28` states the variance form. The design note wording "Variance ... grows as \(1/\sigma_u\)" (`r0-design-note-2026-10-31.md:162`) and `sec_gradf.md:228` "variance constant \(\propto 1/\sigma\)" describe the standard error. | The source tests this law for one estimator, one regime, and one sample count. It does not establish the law for other weight-form estimators, other horizons, or other sample counts. The \(T = 10\) bias row above uses different sample counts. |
| Driven trapping 1D-1V | Particle-pdf biased by about +24 after trapping; CRN FD at \(t = 85\) falls 14.0, 6.2, 0.14 for \(h = 0.02, 0.01, 0.005\) | The reference itself depends on scale. Confirmed in the saved .out file (FD columns at \(t = 85\)). The pdf column stays between 30 and 63 from \(t = 35\) onward, while the exact column and its seed spread grow by orders of magnitude. |
| Standard map reset (filamenting \(f\)) | 0.39, 0.21, \(-0.17\) against 0.74 | Stable but wrong. Source: `sec_rv` E2. Not opened as raw JSON. |
| Pendulum (integrable) | Relative error at \(T = 40\) (5.8 rotations): 0.16 to 0.56 for the first quantity across six resolutions, and 0.08 to 0.56 over all three quantities | [pendulum_summary.log](../../../notes/r0-evidence-2026-10-31/work_crit_physics/pendulum_summary.log). The "15% to 56%" in the note picks the paper-relative resolutions. I could not verify which rows. |
| R2 pilot, EV trapping, \(t = 75\) | Exact column cosine \(-0.28\) with the FD slice fit; single-loading FD itself irregular | `sec_tests` 2.4. Report read. |

The method as built for Twin-S also needs: current rows in the Galerkin basis (critic F7, accepted), degree \(\ge 2\) basis (F6, accepted), and a per-node cold residual detector.

## Cost (estimates, not measurements)

Per Jacobian in primal-run equivalents: one-sided FD \(1 + K\); central FD \(1 + 2K\). A credible Track R forward variant has \(c_1 \approx 1.0\) to 1.3 after a deposit-tangent rewrite (estimate). As built, \(c_1 = 2.47\) (measured in P7 step 3 from 3.47 minus 1). At \(K = 25\), the best pure-weight case saves at most about 20% (20.5 against 26 runs; [cost_compare.out](../../../notes/r0-evidence-2026-10-31/work_crit_feasibility/cost_compare.out)). The pure-weight form fails on cold particles. The adjoint crossover \(K^\ast \approx 13\) to 18 compares adjoint particle-pdf with forward particle-pdf (not with FD), for the pure-weight form only. It assumes the full residual Jacobian: 11 adjoint sets (the rank of the \(11\times13\) host matrix \(C = L^{-1}\partial v/\partial S\), which maps the 13 particle sums \(S_a\) to the 11 whitened residuals; `sec_multi.md:131`, `:198`), with \(c_1^{adj}/c_1 = 1\) to 1.3 and \(R/c_1 = 1.5\) to 3 (`sec_multi.md:193` to `:211`). A scalar-objective gradient needs one adjoint set (`sec_multi.md:199`). The value is a cost estimate, not a measured crossover against FD. Effort for Track R: about 30 to 50 days (estimate). These are projections from stated inputs, not runs.

## Birth models differ by test

The tests do not share one injection model. The sections read describe no ionization births for the warm Debye and Galerkin tests. The R2 pilot (`sec_tests.md:236`) uses simplified EV emission: birth at rest, \(\Delta W = W_{ion} r (1 - e^{-\Gamma\Delta t})\), \(\Gamma = 10 e^{-8/a}\), threshold \(\epsilon = 10^{-4}\), neutral pair. The pendulum and standard-map tests have none. Twin-S uses nitrogen K-shell EV ionization in quasi-cylindrical geometry with PSATD. Compare results only within a test.

## Corrections made by the study

| Earlier statement | Replacement | Evidence |
|---|---|---|
| The growth is "the Lyapunov instability of the N-body dynamics" (plan Track R Reason; plan 12.10 conclusion 2) | The mechanism is unidentified. | Fastest growth (0.80 \(\omega_{p0}\), steps 1500 to 1800) follows the laser exit at step 1520 while the bunch crosses the downramp ([timeline_check.out](../../../notes/r0-evidence-2026-10-31/work_note/timeline_check.out)). P7 logs only the maximum over species. |
| "FD secants at \(10^{-2}\) and \(10^{-3}\) agree" (plan open problem 4) | They differ by 17.4% of the column norm; \(\langle u_x\rangle\) by a factor 2.43 | [holder_check.out](../../../notes/r0-evidence-2026-10-31/work_note/holder_check.out) |
| Critic `crit_feasibility`: Lagrangian variants have \(c_1 \ge 0.96\) | A Lagrangian background alone costs about 0.2 EV-run equivalents | Design note Section 6.1, from P5 timings. Derived, not measured with EV. |
| Critic: roughness 41% to 54% | 31% to 44% after removing curvature | Design note C-2; [detrended_spread.out](../../../notes/r0-evidence-2026-10-31/work_note/detrended_spread.out) |
| "FD recovered \(E_L\) and \(z_f\) in Phase P3" (design note Section 1 item 3) | Each parameter was recovered separately on T0'. The two-parameter gate failed on global search | Plan 12.5; plan Section 2.6 |

The last row is my correction of a source wording. It does not change the source file.

## The recommendation and what would change it

Design note Section 1, authored by the R0 synthesis agent, with both critics aligned: close Track R at R0 and use FD-type Jacobians. Proposed D35. The user decision is open.

Reopening conditions in the note: (a) the FD roughness falls as \(N^{-1/2}\) with particle count (Q1: \(A_4/A_1 \le 0.6\) at every \(h \le 10^{-3}\)), and (b) a parameter or slice point exists where every FD step fails the guarded gate. Kill if \(A_4/A_1 \ge 0.85\). Q1 costs about 22 runs, 2 to 3 GPU-h (estimate). It has not run.

The critic `crit_feasibility` Section 8 item 6 gives one possible Track R advantage: a linearized derivative has no curvature bias. The critic labels that a hypothesis without evidence.

## Alternatives ranked by the study (proposals)

1. Central FD at \(h = 0.02\) or one-sided FD at \(h = 0.01\) to 0.02. 2. DFO-LS type model-based least squares (iteration counts unknown at \(K = 25\)). 3. Surrogate on a global design. 4. Track R R-v-s. 5. Short-horizon AD. 6. Reverse-mode AD (inherits the growth). Stage B first needs the global-search decision (plan 12.5, 2.6).

## Open decision

Accept, reject, or modify D35. Until the user chooses, the plan keeps D33 and D34 as written.
