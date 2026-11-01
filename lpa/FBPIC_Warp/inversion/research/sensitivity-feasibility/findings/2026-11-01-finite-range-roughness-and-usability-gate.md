# Finding 2 (2026-11-01): finite-range FD roughness and a guarded usability gate

Status: current, qualified on 2026-11-01. All values here were measured at `p_nr` = 2. [Finding 4](2026-11-01-particle-count-roughness-and-tangent-growth.md) shows that the roughness at \(h = 10^{-3}\) to \(3\cdot10^{-3}\) falls 4 to 9 times at 4x particles, and [finding 5](2026-11-01-radial-particle-convergence.md) shows that the Jacobian is not converged in radial particles. The historical conclusion below is unchanged. Source: design note Sections 2 and 7.3, Appendix B; saved CPU outputs in `r0-evidence-2026-10-31/work_note/` and `work_crit_feasibility/`. Verification level: saved output inspected. No rerun.

## Conclusion

Over the tested range, FD secants of the Twin-S residuals at \(N_t = 2000\) change with the step \(h\). The change does not fall as \(h\) falls. The source summarizes this as "no scale-free derivative".

This is a source interpretation of finite-range data. It is not a proof that the map is nondifferentiable. It does not identify an instability mechanism. It does not show what happens as the particle count \(N \to \infty\). The decisive test for that question (Q1) is proposed and has not run.

## Evidence

| Observation | Value | Conditions | Source |
|---|---|---|---|
| Roughness amplitude \(A(h)\), median over 11 residuals | \(5.60\cdot10^{-3}\), \(1.88\cdot10^{-3}\), \(4.11\cdot10^{-4}\), \(6.14\cdot10^{-5}\), \(7.06\cdot10^{-6}\) at \(h = 10^{-3}\) to \(10^{-7}\) | Nominal point, one center, central FD, one loading. \(A(h) = \mathrm{median}\; h\lvert D(h) - D(10^{-2})\rvert\), as coded in `holder_check.py:16`. **Lineage: the input FD table was typed by hand from the plan step 4 table (`holder_check.py:1` to `:15`). The output is a computed transformation of copied values. It does not independently corroborate the GPU measurements.** The plan table is the only measurement source. | [holder_check.out](../../../notes/r0-evidence-2026-10-31/work_note/holder_check.out), [holder_check.py](../../../notes/r0-evidence-2026-10-31/work_note/holder_check.py) |
| Global fit | \(A \propto h^{0.73}\) over \(10^{-3}\) to \(10^{-7}\); \(h^{0.81}\) over \(10^{-4}\) to \(10^{-7}\) | Same | Same |
| Column change between \(h = 10^{-2}\) and \(10^{-3}\) | 17.4% of the column norm; \(\langle u_x\rangle\) ratio 2.43 | Same | Same |
| Slice roughness after removing a cubic or quintic fit, \(h = 0.0025\) / 0.005 / 0.01 / 0.02 | 31% to 44% / 24% to 38% / 15% to 28% / 3% to 9% of \(\lVert J_{sm}\rVert\) | 41-point slice, spacing 0.0025, 17 centers within \(\pm0.02\) of 0.588 | [detrended_spread.out](../../../notes/r0-evidence-2026-10-31/work_note/detrended_spread.out) |
| Raw secant spread, no detrending | 0.541 at \(h = 0.0025\), 0.246 at \(h = 0.02\) | Same slice | [slice_secant_spread.out](../../../notes/r0-evidence-2026-10-31/work_crit_feasibility/slice_secant_spread.out) |

My arithmetic on the saved \(A(h)\) values gives local decade exponents of about 0.47, 0.66, 0.83, 0.94 from \(10^{-3}\) down to \(10^{-7}\). The single exponent 0.73 therefore summarizes a trend, not a clean power law. The note also states that below \(h = 10^{-4}\) the runs change removal counts, so the small-\(h\) exponent mixes mechanisms (design note Section 9, P-F3). I label this reading as mine, derived from the saved output.

The headline "15% to 44%" covers \(h = 0.0025\) to 0.01. At \(h = 0.02\) the detrended roughness is only 3% to 9%. That is consistent with FD at \(h = 0.02\) meeting the solution-slice thresholds below. It also means the roughness is a function of scale, as the source states.

## The guarded usability gate

Definition (design note 7.3, derived in A.7): for a candidate Jacobian \(J_A\) and a reference \(J_{ref}\),

\[
\kappa = \lVert I - J_A^{+} J_{ref}\rVert_2, \qquad
\delta = \lVert (J_A^T J_{ref})^{-1} (J_A - J_{ref})^T r_w^\ast \rVert_\infty .
\]

For \(K = 1\), \(\kappa = \lvert 1 - \cos/\rho\rvert\) with \(\rho = \lVert J_A\rVert / \lVert J_{ref}\rVert\). Assumption: linearization near the least-squares solution, with \(J^T r^\ast = 0\).

Proposed gate (design note `r0-design-note-2026-10-31.md:273` to `:279`): (1) median \(\kappa \le 0.5\), p90 \(\kappa \le 0.8\), \(\kappa < 1\) everywhere; (2) median per-column cosine \(\ge 0.85\); (3) \(\delta \le 0.005\) at the solution; (4) checks on the solution, false-minimum, and far-field slices; (5) for a stochastic column, ensemble and per-seed checks.

The table below is a **partial calibration**. It tests items 1 and 2 on the 17 solution-slice centers only. Item 3 was met by AD only formally (see below). Item 4 is incomplete: the false-minimum and far-field slices have 5 of 41 and 1 of 41 points (`r0-design-note-2026-10-31.md:293`). Item 5 was not done for Twin-S. No column has met the full gate.

| Column | \(\kappa\) median / p90 / max | cosine median / min | Solution-slice thresholds (items 1, 2) |
|---|---|---|---|
| Central FD \(h = 0.0025\) | 0.206 / 0.415 / 0.526 | 0.920 / 0.788 | met |
| Central FD \(h = 0.01\) | 0.116 / 0.249 / 0.264 | 0.963 / 0.856 | met |
| Central FD \(h = 0.02\) | 0.106 / 0.165 / 0.181 | 0.994 / 0.940 | met |
| Forward FD \(h = 0.02\) | 0.183 / 0.596 / 0.697 | 0.853 / 0.621 | met, marginal cosine |
| AD, \(N_t = 2000\) | 1.000 / 1.000 / 1.001 | 0.210 / \(-0.681\) | not met |

Source: [gate_calibration.out](../../../notes/r0-evidence-2026-10-31/work_note/gate_calibration.out), 17 centers.

## Corrected interpretation: the \(\delta\) bar alone is not a gate

The D32 bar \(\delta \le 0.005\) was met by the AD column (\(\delta = 3.5\cdot10^{-3}\)) although its cosine to the smoothed column is 0.25 or below. At a least-squares solution \(\delta\) is invariant to the scale of \(J_A\). It is small when \(\lVert r_w\rVert / \lVert J\rVert\) is small. On P7 that ratio is 0.013. The design note reports that 19.5% of random 11-vectors pass \(\delta \le 0.005\) (from `sec_tests` 6.1 and 6.2). I did not re-run the null test. The saved `gate_calibration.out` ends with a separate null (\(P(\kappa \le 0.5\ \text{and}\ \cos \ge 0.9) = 0.0\)).

Superseded conclusion: "AD usable by the \(\delta\) bar". Replacement: \(\delta\) says where the iteration stops, not whether it gets there. Practical consequence: any usability claim needs \(\kappa\) and the cosine.

## Limits that must travel with these numbers

1. The reference is a cubic fit of the same 41-point slice that supplies the FD secants. FD meets the solution-slice thresholds largely by construction. The calibration shows parity with FD, not superiority. The design note states this and accepts the critic's point (C-5).
2. Reference dependence is large. Against the fixed 0.588 column, the critic's [onesided_gate.out](../../../notes/r0-evidence-2026-10-31/work_crit_feasibility/onesided_gate.out) gives \(\kappa = 0.003\) for forward FD at \(h = 0.02\). Against the local reference it gives 0.183. Values from the two references are not comparable.
3. The false-minimum and far-field slices are incomplete (5 of 41 and 1 of 41). Gate item 4 is not met for any column, so no column has satisfied the full proposed gate.
4. All numbers are for one parameter (\(E_L\)) with \(z_f\) fixed, 11 residuals, and one injection seed. The ensemble and per-seed check of gate item 5 was not done for Twin-S.
5. Particle counts: Twin-S has about 2 M loaded macroparticles at `p_nt = 8` (plan Section 3.3) and 12.7 M EV births. Roughness versus particle count is untested.
6. Dimensionless perturbation: \(\hat E_L\) is normalized to \([0, 1]\) by the plan's bounds. \(h = 0.02\) in \(\hat E_L\) is about 2% of the bound range. The note converts \(h = 0.005\) to 0.02 into 0.4% to 1.6% of \(E_L\) (critic `crit_feasibility` Section 3). The conversion factor is not stated in the saved files I read.
