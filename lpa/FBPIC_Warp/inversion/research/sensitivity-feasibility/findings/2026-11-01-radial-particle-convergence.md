# Finding 5 (2026-11-01): the Twin-S Jacobian is not converged in radial particles per cell

Status: current, with one pending run (`p_nr` = 16). This finding qualifies findings 1, 2, and 4: their Jacobians were measured at `p_nr` = 2.
Source: GPU runs of `inversion/q1_roughness.py` (stage `points` and direct `_child` calls), 2026-11-01, this session. These runs were started by the agent as a follow-up of the authorized particle-count test, after the 1x and 4x Jacobians disagreed in sign. The user was told and did not object; no separate user decision exists.
Verification level: saved outputs inspected; analysis computed in this session.

## Conclusion and tested conditions

1. Only the radial count `p_nr` changes the Jacobian. Doubling the angular count (`p_nt` 8 to 16) changes nothing: cosine 1.00 with the 1x column.
2. `p_nr` 2 to 4 reverses the Jacobian direction. Cosine between the step-2000 central secants (\(h = 10^{-2}\)) is \(-0.67\) to \(-0.69\). Most spread residuals change sign.
3. `p_nr` 4 to 8 keeps the direction (cosine 0.93 at step 2000, 0.90 at step 1520) but the norm falls by 2.7 times (199 to 73). The transverse centroid columns fall from about 39 to 6, the spread columns by 3 to 4 times.
4. The observables themselves change little. The center residual vector moves by 0.69 whitened units from `p_nr` 4 to 8 and by 1.6 to 2.2 units from 2 to 8.

Conditions: Twin-S, EV with free thresholds, \(\hat E_L = 0.588 \pm 10^{-2}\), \(\hat z_f = 0.4\), central secants of the 11 whitened residuals, default injection seed.

## Decisive numbers (step 2000)

| Run label | `p_nr` × `p_nt` | Secant norm | Cosine with `p_nr` 8 |
|---|---|---|---|
| 1 | 2 × 8 | 172 | negative |
| 2t | 2 × 16 | 172 | negative (cosine 1.00 with label 1) |
| 2r | 4 × 8 | 194 | 0.93 |
| 4 | 4 × 16 | 199 | 0.93 |
| 8 | 8 × 16 | 73 | 1 |
| 16r | 16 × 8 | pending | pending |

Selected residual columns (\(\partial r_j/\partial\hat E_L\), \(h = 10^{-2}\), step 2000):

| Residual | `p_nr` 2 | `p_nr` 4 | `p_nr` 8 |
|---|---|---|---|
| \(\langle x\rangle\) | −115.5 | 39.3 | 6.4 |
| \(\langle\zeta\rangle\) | 26.0 | −107.5 | −54.2 |
| \(\log\sigma_x\) | −44.6 | 70.9 | 17.7 |
| \(\log(\sigma_{u_z}/\langle u_z\rangle)\) | −21.0 | 62.2 | 15.0 |

Source files: `inversion/runs/q1/m{1,2r,2t,4,8}_default_{0.578000,0.588000,0.598000}.json`, key `snaps."2000".r_w`. The runs directory is git-ignored. Run time per point: about 90 s (`p_nr` 2), 180 to 200 s (`p_nr` 4), 1000 to 1500 s (`p_nr` 8, while other jobs shared the host).

## Interpretation and limits

- Agent interpretation: the radial sampling at `p_nr` = 2 creates a large spurious sensitivity, most visibly in the transverse centroids, which should be small by symmetry. The physical response to \(E_L\) at this point is weaker than the 1x twin reports. The cause (axis loading, radial noise, or EV sampling of the thin injection region) is not isolated.
- Convergence is not shown: the norm still changes by 2.7 times from `p_nr` 4 to 8. The `p_nr` 16 run decides whether `p_nr` 8 is close.
- The grid (`Nr` 50) and time step were not varied.
- The twin experiment stays internally consistent: target and model share `p_nr` 2, so the Stage A FD recovery remains a valid twin test of the optimizer. Its Jacobians and landscape are not those of a converged model.

## Consequence

Every Twin-S gradient reference so far (P7 AD, the R0 roughness and gate, finding 4) belongs to an under-resolved model. A converged reference resolution must be chosen before any macroscopic-gradient method is validated on Twin-S.
