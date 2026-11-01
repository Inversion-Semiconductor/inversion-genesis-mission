# Finding 1 (2026-10-31): the exact AD tangent is verified to step 500 and unusable at step 2000

Status: current, qualified on 2026-11-01. The P7 runs used `p_nr` = 2, where the Jacobian is not converged ([finding 5](2026-11-01-radial-particle-convergence.md)). The tangent growth itself persists at 4x particles ([finding 4](2026-11-01-particle-count-roughness-and-tangent-growth.md)). Source: plan Section 12.10 (Phase P7), checkpointed 2026-10-31. Verification level: report read, and saved output spot-checked.

## Conclusion and tested conditions

Forward-mode AD through the expected-value (EV) ionization model matches frozen-decision FD where FD can check it. The error range \(1.8\cdot10^{-6}\) to \(3.1\cdot10^{-5}\) applies only to the horizons \(N_t = 460\) and 500, at the step \(h\) with sampled decision-flip count \(K(h) = 0\) (plan lines 1788 and 1789). It is not a range for all horizons up to 500.

At \(N_t = 50\) the zero-flip check is not available. The plan rejects \(h \le 10^{-5}\) because noise dominates and uses the remaining \(h\), which have \(K > 0\) and agree to \(1.4\cdot10^{-5}\) (plan lines 1787 and 1794). From \(N_t = 650\), \(K(h) > 0\) at every \(h\).

After step 1200 the particle tangents grow about tenfold per 120 steps. At step 2000, \(\lvert\partial u_x/\partial E_L\rvert = 1.7\cdot10^{11}\) per J. The AD residual column at \(N_t = 2000\) is 10 to 1000 times the smooth FD secant. It has the opposite sign for 6 of 11 residuals.

Conditions by measurement group. All groups: Twin-S with EV, \(K = 1\) tangent for \(E_L\), \(\alpha_D = 0\), code on top of commit `4c5ff56`, physical GPU 1 (RTX 6000 Ada), FP64.

| Group | Point | Plan location |
|---|---|---|
| Horizon sweep, tangent growth, full-horizon loss column | "nominal \(\theta\)" with the T0 target. The plan does not print the numeric value at these lines. | `DIFFERENTIABLE_INVERSION_PLAN.md:1781`, `:1806` |
| Solution-slice and gate statistics (41 points) | Slice centered near \(\hat E_L = 0.588\), with \(\hat z_f = 0.4\) fixed | Plan step 5; [finding 2](2026-11-01-finite-range-roughness-and-usability-gate.md) |
| Gradient cost 3.47 run-equivalents | Plan step 3. Its parameter point and step count were not re-checked for this record. | `DIFFERENTIABLE_INVERSION_PLAN.md:1778` |

The earlier version of this record applied \(\hat\theta = (0.588, 0.4)\) to every group. The sources do not support that.

## Decisive numbers

| Quantity | Value | Source |
|---|---|---|
| Gradient cost | 3.47 primal-run equivalents (298 s against 86 s). The step count of this cost run is not re-checked here. | Plan 12.10 step 3 |
| Tangent growth | \(\times 10\) per 120 steps after step 1200 | Plan 12.10 step 4 table |
| Solution slice, 41 points, column error median / p90 | 2010 / 7103 | Plan 12.10 step 5; confirmed in [p7_jacobian.json](../../../results/p7_jacobian.json) (values 2010.41 and 7103.41) |
| Cosine of AD column and smoothed column, median / min | \(-0.14\) / \(-0.68\) | Same; median \(-0.144\) confirmed in the JSON |
| Fixed-point shift \(\delta\) at the solution | \(3.5\cdot10^{-3}\) | Plan 12.10 step 5 (reported) |
| Frozen versus free FD | agree to 3 digits at all three centers | Plan 12.10 step 5 item 1 |
| AD versus frozen FD | differ by 99.6% to 100% at every \(h\) down to \(10^{-5}\) | Plan 12.10 step 5 item 2 |
| \(\epsilon \to 0\) test (reduced Twin-S) | AD columns do not converge; AD at \(\epsilon = 0\) differs from its own FD by 99.98% | Plan 12.10 step 6 |
| Particle spread of the error | 50% of \(\sum\lvert c_{ji}\rvert\) from \(10^4\) to \(7\cdot10^4\) particles | Plan 12.10 step 5 item 3 |

## Interpretation and limits

1. The plan reads this as chaotic sensitivity of the trajectories, not a code defect and not the emission threshold. The evidence for "not a code defect" is agreement with FD to step 500 and the frozen-versus-free agreement.
2. The plan called it "the Lyapunov instability of the N-body particle dynamics". The R0 critique rejected that label as unproven. See [finding 3](2026-11-01-track-r-feasibility-study.md). The P7 log stores only the maximum tangent norm over all species. The species that carries the growth is unknown.
3. FD at \(N_t \ge 1000\) is not a frozen-decision reference, because removal counts at the window back change from iteration 854 to 1148 (plan 12.10 step 4).
4. Step 5 is partial. The solution slice (41 points) and three centers are complete. The false-minimum slice has 5 of 41 points. The far-field slice has 1 of 41. Cached runs are in `inversion/runs/p7_fe7188bf9100/`. The remaining 76 runs (about 6.3 GPU-h) are paused by D34.
5. The \(\delta\) bar of D32 is met only formally. See [finding 2](2026-11-01-finite-range-roughness-and-usability-gate.md) for the correction.

## Consequence

The plan's M3 recommendation was FD fallback (D32 branch 3), with AD reported only. The user did not choose a branch. D34 deferred it.
