# Finding 7 (2026-11-01): at the converged resolution a macroscopic Jacobian exists, and exact AD still misses it by about 10^4

Status: current. Source: GPU runs of `inversion/q1_tangent_growth.py` (label 8, suffix `_ad`; label 1, suffix `_ad`) and `inversion/q1_roughness.py` (label 8, \(h = 3\cdot10^{-3}, 10^{-3}\)), 2026-11-01, this session, started by the agent under the user's "keep crunching" (2026-11-01).
Verification level: saved outputs inspected; analysis computed in this session.

## Conclusion and tested conditions

1. At `p_nr` = 8 (converged within about 10%, [finding 5](2026-11-01-radial-particle-convergence.md)), the FD Jacobian has a stable direction from \(h = 10^{-2}\) down to \(10^{-3}\). Cosines with the \(h = 10^{-2}\) secant are 0.96 to 0.99 at steps 1520 and 2000. The norm differs by 5% to 14%; the relative vector difference \(\lVert D(h) - D(10^{-2})\rVert / \lVert D(10^{-2})\rVert\) is 18% to 29%. At `p_nr` = 2 the same comparison gave 23% to 66% deviation ([finding 4](2026-11-01-particle-count-roughness-and-tangent-growth.md)).
2. The columns that change sign between step sizes are the transverse centroids \(\langle x\rangle\) and \(\langle u_x\rangle\), whose values are small (about 4 to 8). The dominant columns (\(\langle\zeta\rangle\), \(\langle u_z\rangle\), the spreads) agree within about 10% to 40%.
3. Exact forward-mode AD on the converged model is still unusable. At step 2000 the AD column norm is \(6.2\cdot10^5\) against an FD norm of 63 to 73: about 8500 to 9800 times larger, cosine 0.26 to 0.41. Already at step 1200 the AD norm is 908 against 73 to 97 (about 9 to 12 times), cosine 0.60 to 0.85. From step 1640 the AD column grows by orders of magnitude.
4. Tangent growth at `p_nr` = 8 is similar to `p_nr` 2 and 4: bunch median \(\lvert\partial u/\partial\hat E_L\rvert\) 341 at step 2000, bunch 90th percentile \(4.1\cdot10^4\), background median 22 at step 1800.

Conditions: Twin-S, EV with free thresholds, \(\hat E_L = 0.588\), \(\hat z_f = 0.4\), `p_nr` 8, `p_nt` 16 (FD) and 16 (AD, label 8), default injection seed.

## Decisive numbers

| Step | AD norm | FD norm, \(h = 10^{-2}\) / \(3\cdot10^{-3}\) / \(10^{-3}\) | Cosine AD with FD at \(h = 10^{-2}\) |
|---|---|---|---|
| 1200 | 908 | 97 / 90 / 73 | 0.85 |
| 1520 | 1230 | 61 / 54 / 58 | 0.11 |
| 1760 | \(1.9\cdot10^6\) | 70 / 60 / 66 | 0.09 |
| 2000 | \(6.2\cdot10^5\) | 73 / 63 / 66 | 0.26 |

Sources: [q1_tangent_growth_m8_ad.json](../../../results/q1_tangent_growth_m8_ad.json), key `ad_columns.<step>.jac_hat`; `inversion/runs/q1/m8_default_{0.578,0.585,0.587,0.588,0.589,0.591,0.598}000.json`, key `snaps.<step>.r_w`. AD run wall time 2438 s on GPU 0.

## Seed spread of the converged reference

Three plasma-injection seeds (default, 1, 2) at `p_nr` = 8, central secant at \(h = 10^{-2}\): relative spread 4.6% at step 1520 and 4.0% at step 2000; every seed column has cosine 1.000 with the mean. At step 2000 the mean column is \(\log Q\) 3.1 ± 0.1, \(\langle x\rangle\) 5.6 ± 1.0, \(\langle u_x\rangle\) 3.6 ± 0.7, \(\langle\zeta\rangle\) −52.7 ± 1.3, \(\langle u_z\rangle\) 16.6 ± 0.2, spreads 12.7 to 23.7 with standard deviations 0.4 to 1.4. Sources: `inversion/runs/q1/m8_{default,s1,s2}_{0.578000,0.598000}.json`. Run time 655 to 682 s per point. This column is the candidate Twin-S reference for a macroscopic-gradient method at \(\hat E_L = 0.588\).

## Reproducibility check of the AD column

The new 1x AD column at step 2000 (`q1_tangent_growth_m1_ad.json`) differs from the cached P7 column at the same point (`inversion/runs/p7_fe7188bf9100/center_0.588000.json`, key `jac_hat`) by 63% of its norm, cosine 0.78. Both use the same contraction (`ResidualValue.contract`, as in `p7_jacobian.py:157`). Agent interpretation: the full-horizon AD column is not reproducible between runs, because nondeterministic GPU atomics (about \(10^{-12}\) relative per run) are amplified by the tangent growth. This interpretation was not tested directly.

## Is the AD error carried by a few particles?

No, at `p_nr` = 2. The P7 contribution diagnostic at \(\hat E_L = 0.588\) (`inversion/runs/p7_fe7188bf9100/center_0.588000.json`, key `contributions`, 1.69 M selected particles) shows that half of \(\sum_i\lvert c_{ji}\rvert\) needs 10,000 to 76,000 particles for 9 of the 11 residuals; the top 1000 particles hold only 4% to 45%. \(\log Q\) is concentrated (half in 168 particles), and \(\log\sigma_\zeta\) partly so (half in 1665 particles). So trimming or robust aggregation of particle contributions would not recover the FD column. This was not repeated at `p_nr` = 8.

## Interpretation and limits

- Agent interpretation: on the converged model the macroscopic response is smooth enough that a Jacobian at scales \(10^{-3}\) to \(10^{-2}\) is well defined in direction. This removes the R0 premise that Twin-S has "no scale-free derivative" at these scales; that premise came from `p_nr` = 2 data. A macroscopic-gradient method therefore has a meaningful target on the converged model.
- Exact AD diverges from that target from about step 1200 at `p_nr` = 8, earlier than the step-1450 onset seen in the per-species medians.
- One point, one parameter. The `p_nr` = 8 seed spread covers \(h = 10^{-2}\) only, with 3 seeds (default, 1, 2). Smaller \(h\) than \(10^{-3}\) was not tested at `p_nr` = 8.
