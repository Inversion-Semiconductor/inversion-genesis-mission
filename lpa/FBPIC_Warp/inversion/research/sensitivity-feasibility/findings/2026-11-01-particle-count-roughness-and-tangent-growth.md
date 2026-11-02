# Finding 4 (2026-11-01): the h = 0.01 secant is seed-stable, roughness forms inside the plasma, and more particles do not stop tangent growth

Status: current. Qualified by [finding 5](2026-11-01-radial-particle-convergence.md): the 1x and 4x runs differ in radial sampling, and the Jacobian is not converged in radial particles.
Source: GPU runs of `inversion/q1_roughness.py` and `inversion/q1_tangent_growth.py`, 2026-11-01, this session. Authorization: the user approved the particle-count test with a plasma-exit observable ("let's do that no?", 2026-11-01).
Verification level: saved outputs inspected; analysis computed in this session from the run JSON files.

## Conclusion and tested conditions

1. At the current resolution (1x: `p_nz` 1, `p_nr` 2, `p_nt` 8), the central secant at \(h = 10^{-2}\) is a reproducible quantity. Over 4 plasma-injection seeds, its relative spread is 3.5% to 4.1% at every snapshot from step 1400 to 2000, and every seed column has cosine at least 0.999 with the seed mean.
2. Below that step the secants are rough. At 1x and at steps 1520 to 2000, the secant at \(h\) differs from the \(h = 10^{-2}\) secant by 23% to 66% of its norm for \(h = 3\cdot10^{-3}\) to \(10^{-4}\).
3. The roughness is already present at step 1520 (laser exit from the plasma) with about the same size as at step 2000. It forms inside the plasma, not at the density cliff or in the vacuum drift.
4. At 4x particles (`p_nr` 4, `p_nt` 16) and steps 1520 to 2000, the secants stay within 4% to 9% of the \(h = 10^{-2}\) secant down to \(h = 10^{-3}\). The roughness amplitude ratio \(A_4/A_1\) is 0.11 to 0.23 at \(h = 10^{-3}\) and \(3\cdot10^{-3}\) for steps 1520 to 2000. At \(h = 10^{-4}\) the ratio is 0.77 to 1.34, so the smallest scale stays rough.
5. The exact forward tangent grows collectively. The median momentum tangent of the background electrons and of the injected bunch both grow from about step 1450. At 4x particles the growth rates fall by only 10% to 30%, and the step-2000 tangents are as large or larger (table below).

All runs: Twin-S, expected-value (EV) ionization with free emission thresholds, \(\hat E_L = 0.588 \pm h\), \(\hat z_f = \hat z_f^\ast = 0.4\), \(N_t = 2000\), snapshots at steps 1200, 1400, 1520, 1640, 1760, 1880, 2000. Residuals are the 11 whitened T0 residuals; at snapshots before step 2000 only their differences in \(\theta\) are meaningful.

## Decisive numbers

Roughness, relative deviation \(\lVert D(h) - D(10^{-2})\rVert / \lVert D(10^{-2})\rVert\) of the central secant \(D(h)\) (computed from `inversion/runs/q1/m{1,4}_default_*.json`):

| Step | Multiplier | \(h = 3\cdot10^{-3}\) | \(10^{-3}\) | \(3\cdot10^{-4}\) | \(10^{-4}\) |
|---|---|---|---|---|---|
| 1520 | 1x | 0.23 | 0.36 | 0.35 | 0.59 |
| 1520 | 4x | 0.04 | 0.09 | 0.18 | 0.61 |
| 2000 | 1x | 0.25 | 0.43 | 0.51 | 0.66 |
| 2000 | 4x | 0.06 | 0.09 | 0.15 | 0.38 |

Amplitude ratios \(A_4/A_1\) with \(A_m(h) = \mathrm{median}_j\, h\,\lvert D_{m,j}(h) - D_{m,j}(10^{-2})\rvert\): [q1_roughness.json](../../../results/q1_roughness.json), key `"<step>"."A4_over_A1"`. Seed spread at 1x: same file, key `"<step>".m1.seed_spread_rel`.

Tangent growth, \(\lvert\partial u/\partial\hat E_L\rvert\) per species ([m1](../../../results/q1_tangent_growth_m1.json), [m4](../../../results/q1_tangent_growth_m4.json), key `rows[].<species>.du`):

| Quantity | 1x | 4x |
|---|---|---|
| Background `electrons` median, growth rate steps 1400 to 1650 | 0.0160 per step | 0.0127 per step |
| Bunch `elec_from_N` median, rate steps 1450 to 1750 | 0.0054 | 0.0048 |
| Bunch 90th percentile, rate steps 1450 to 1750 | 0.0105 | 0.0072 |
| Bunch median at step 2000 | 296 | 328 |
| Bunch 90th percentile at step 2000 | \(7.1\cdot10^3\) | \(4.95\cdot10^4\) |
| Bunch maximum at step 2000 | \(1.2\cdot10^{12}\) | \(4.3\cdot10^{12}\) |

The P7 rate of "tenfold per 120 steps" (0.019 per step) came from the maximum over all species ([finding 1](2026-10-31-ad-tangent-at-full-horizon.md)). The per-species medians here grow from about step 1450, not 1200.

## Interpretation and limits

- Agent interpretation: growth in the median of both species points to a collective (mean-field) mechanism in the density downramp, not to N-body noise alone. The mechanism is not identified. Wave breaking in the downramp is a hypothesis.
- The 1x seed check covers only \(h = 10^{-2}\). Only 1 of the 9 planned 4x seed runs finished; the other 8 were stopped to save time, so no 4x seed spread exists.
- The proposed kill rule of the R0 note (\(A_4/A_1 \ge 0.85\) at every \(h \le 10^{-3}\)) is not met, and the continue rule (\(\le 0.6\) at every \(h \le 10^{-3}\)) fails at \(h = 10^{-4}\). The R0 rules assumed a converged model; finding 5 shows that assumption fails.
- One parameter point, one parameter direction (\(\hat E_L\)).

## Consequence

More particles alone do not make exact forward-mode AD usable at \(N_t = 2000\). The roughness that R0 reported is largely a resolution effect at the scales \(10^{-3}\) to \(10^{-2}\).
