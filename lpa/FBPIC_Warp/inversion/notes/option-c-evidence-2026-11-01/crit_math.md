# Critique of option C: mathematics and physics (key `crit_math`)

Date: 2026-11-01. Lens: is the Girsanov / Feynman-Kac representation right for the self-consistent system; is the mean-field versus joint split right; can collective modes amplify the weights; is the noise physically defensible for LWFA; is the novelty claim honest.

Work directory: `inversion/notes/option-c-evidence-2026-11-01/work_crit_math/` (written `wc/` below). All runs are CPU NumPy with `python -I`.

Labels: **[derived]**, **[measured: script]**, **[literature: citation]**, **[estimated]**, **[checked]** (a claim of the reviewed sections that I re-derived or re-ran and that holds).

## 0. Verdict

The representation is correct. The physics case and two central claims of the theory are not.

1. **Correct [checked].** The pure and mixed Girsanov identities, the discrete Vay-step score, the birth-weight source, and the mean-field closure are unbiased up to $O(1/N)$. The toy and wake ensembles confirm this.
2. **Wrong: "collective growth of the weights is physical sensitivity that every exact method sees" (theory 4.4a).** In the two-stream toy the true gradient decays in time while the weights grow by $10^3$. The growth is estimator variance in the linearized operator, not sensitivity [measured, existing data]. Option C relocates the exponential growth from particles to the field tangent. It does not remove it by construction.
3. **Untested before this review: the charge-neutral increments (theory 4.5).** The theory calls them "the most important design element", but `sec_toy` and `sec_wake` ran only the plain estimator. I added them to the wake harness. After trapping they cut the growth rate by 3 to 10 times, and at $\sigma = 0.03$ the rate falls with $N$ toward zero (0.27, 0.083, 0.034 at $N = 1, 4, 16$). The "N-independent floor near the chaotic rate" reported in `sec_wake` is not present with centering [measured]. At $N = 16$ the centered estimator agrees with FD, but its single-run noise is still 1 to 500 times the gradient (Section 3).
4. **Wrong: "the feedback control (2.6) removes the laser quiver term" (theory 4.3, 4.5 item 3).** In a resolved relativistic laser pulse, (2.6) gives no reduction at $a_0 = 1$ and a $3\cdot10^{10}$ times larger weight variance at $a_0 = 4$ (time-step converged) [measured]. The Duffing check used a spatially uniform fast force and cannot show this.
5. **Not physically defensible as stated.** The proposed noise is 8 to 11 times the collisional value of the cold background (60 to 130 times in diffusivity), and more than $10^6$ times the collisional diffusivity for electrons moving in the wake. Pure diffusion is unbounded heating, not a temperature. Independent per-particle kicks are not shot-to-shot jitter. The noise changes trapped charge at first order (wake: $-12$ percent at $\sigma = 0.01$), which smoothing in $\theta$ never does [estimated, measured by `sec_wake`].
6. **Novelty claim is honest in tone but misses the closest probabilistic prior art:** particle approximations of measure derivatives in sequential Monte Carlo (Poyiadjis-Doucet-Singh 2011; Del Moral-Doucet-Singh 2015) and a 2026 virtual tangent interacting particle system for McKean-Vlasov likelihood gradients (Sharrock-Kantas-Pavliotis). The SMC results also name the cure for the variance growth: marginalize the path-accumulated score [literature].

Consequence for the concrete goal: goals (2) cold-safe and (3) unbiased against the noisy model hold. Goal (1) no exponential growth is not established. In 1D, centering removes most of the growth after trapping, but a rate of about 0.13 remains during injection, and no argument bounds the field-level rate in general. The toy and wake "fails" verdicts must be read as verdicts on the plain estimator. Goal (4) gets worse: the laser needs $\mathbf h = 0$ plus a handover, not the feedback assumed in the cost table.

## 1. Findings

| # | Severity | Section reviewed | Claim | Problem |
|---|---|---|---|---|
| 1 | blocking | theory 4.4(a); toy 0 item 3, 3.4; lit 3 | Exponential weight growth beyond the shot-noise loop is physical collective amplification, seen by every exact method | Falsified by existing toy data: the FD gradient decays while the weights grow. The growth is variance (Section 2) |
| 2 | major | theory 0 item 4, 4.4(b), 7 item 1; toy; wake | Charge-neutral increments are the key fix | Never tested in toy or wake. Tested here: large gain, still insufficient at the $N$ tested (Section 3) |
| 3 | major | wake 3.1 item 3, 4 item 2 | An $N$-independent rate floor of about 0.23 exists and equals the chaotic rate, because $\delta f/f$ resolves the same fine structure | Mechanism is mathematically wrong (Itô isometry). With centering, no floor is seen after trapping; a smaller floor (about 0.13) remains during injection (Section 3) |
| 4 | major | theory 2.3, 4.3, 4.5 item 3, 6 | Feedback (2.6) keeps the quiver tangent pathwise and suppresses its weight variance by $(\omega_c/\omega_0)^4$ | Measured: no suppression at $a_0 = 1$; catastrophic at $a_0 = 4$ (Section 4) |
| 5 | major | theory 3, eq. (3.1); lit Identity 2 | Joint LR: at fixed paths only the vacuum laser tangent depends on $\theta$; it needs no field tangent | Omits the $\theta$-dependence of EV birth weights at fixed paths. In FBPIC the vacuum tangent needs a Maxwell solve (Section 5) |
| 6 | major | theory 1.3, 1.3 item 4; task brief | The noise is a physical smoothing scale like temperature, collisions or jitter; $h_{eq} \approx u_n/\lvert\partial u/\partial\theta\rvert$ | Not supported (Section 6) |
| 7 | major | lit 1-2, 6 item 5 | No combination of per-particle weights with a mean-field closure was published | Misses SMC tangent-filter and McKean-Vlasov virtual-tangent work (Section 7) |
| 8 | minor | theory 2.4 | "Given the primal paths" the noise expectation solves (2.1)-(2.2) exactly | The conditioning is ill-posed; the correct statement is $O(1/N)$ with a time-growing constant (Section 5) |
| 9 | minor | theory 1.3 item 3 | Selection bias $O(u_n^2/(4\delta u)^2)$: 4 percent at $u_n = 0.03$ | Wrong scaling for an observation smoothing (Section 6) |
| 10 | minor | theory 1.3 item 1 | Thermal wave-breaking limit is far above the wake amplitude for $u_n \le 0.1$ | At $u_n = 0.1$ the limit is about $2.4E_0$, comparable to a nonlinear wake |
| 11 | minor | toy 5 item 3 | Small weights require a noise spread of tens of $mc$ | Overstated; the relevant ratio is $s_g/(u_n\sqrt{N_{eff}})$, theory (4.3) |

## 2. Collective amplification is variance, not sensitivity (finding 1)

**Claim reviewed.** Theory 4.4(a): "If the linearized VFP has a growing mode, $\delta\mathbf E$ grows, and so does the true sensitivity of $\mathbb E[J]$. Every exact method sees it. It is not a defect of option C." The toy attributes the excess of the two-stream weight growth over the loop rate to "collective amplification".

**Evidence against [measured: existing `work_toy/out/es_long_summary_ref0.1.txt`].** Two-stream, $\sigma = 0.03$, $N = 524288$, 32 seeds. The FD reference gradient (CRN, $h = 0.1$) and the weight rms:

| $t$ | FD $\partial K$ / $\partial m_1$ / $\partial s_1$ | $\omega_{rms}$ (median) | SD of mf estimate for $K$ |
|---|---|---|---|
| 20 | 0.119 / -0.576 / 0.583 | 24 | 0.034 |
| 40 | 0.035 / -0.128 / 0.088 | 168 | 0.33 |
| 60 | 0.001 / -0.033 / 0.052 | 725 | 1.6 |
| 100 | -0.013 / -0.010 / -0.016 | $1.35\cdot10^4$ | 98 |

The true sensitivity falls by 10 to 50 while the weights grow by 560. The excess rate (0.073 measured against 0.035 from the loop) is therefore not in $\mathbb E[\delta\mathbf E]$. The wake shows the same: FD gradients stay $O(1$ to $20)$ at $t = 75$ while the plain weights reach $10^{8}$ to $10^{30}$.

**Mechanism [derived].** The estimator field tangent is $\delta\mathbf E^N = \delta\mathbf E + \eta$. Its mean $\delta\mathbf E$ solves the linearized VFP with the coherent $\theta$ source. Its fluctuation $\eta$ solves the same linear system driven by white sampling noise of the weighted deposit. A linear operator around a nonlinearly saturated state (post-saturation two-stream, a trapped bunch, multi-stream wake) can have growing directions that the coherent source excites weakly, because its response phase-mixes. White noise excites every direction. So $\mathrm{Var}\,\omega$ grows at the instability rate while the true gradient stays bounded. The error after time $T$ is then about $e^{\gamma T}/\sqrt N$. This is the same form as the error of the averaged Lagrangian tangent, $e^{\lambda T}/\sqrt N$, with a field-level rate $\gamma$ instead of the particle rate $\lambda$. Option C is exponential-free only if $\gamma \le 0$ for the linearized operator of the noisy primal, and nothing in the theory guarantees that.

**What survives.** For a fixed, bounded $\delta\mathbf a$ the Itô isometry makes $\mathrm{Var}\,\omega$ linear in time. Single-particle Lyapunov stretching does not enter [checked: `sec_toy` maps; wake `NOFB` runs, $\sigma\omega_{rms} = 0.5$ to $0.8$ at all $\sigma$, $N$].

## 3. Charge-neutral increments in the trapping regime (findings 2, 3)

**Gap.** `work_toy/es1d_lr.py` and `work_wake/wake_lr.py` contain no centering. Both "fails" verdicts are for the plain estimator.

**Test [measured: `wc/wake_lr_c.py`, `wc/an_grow.py`, outputs `wc/out/grow/`].** I copied the wake harness and added (4.5) as a leave-one-out CIC-weighted local mean of the weight increments (env `CENTER=2`), which keeps the estimator unbiased to $O(1/N)$. Seed 1, $a_0 = 2.5$, rate of $\ln\omega_{rms}$ (plasma) plain / centered:

| $\sigma$ | $N$ | rate $t \in [10, 30]$ | rate $t \in [30, 75]$ (after trapping) | $\omega_{rms}(75)$ |
|---|---|---|---|---|
| 0.01 | 1 | 2.94 / 0.66 | 3.22 / 0.94 | $2\cdot10^{93}$ / $3\cdot10^{25}$ |
| 0.01 | 4 | 1.44 / 0.26 | 1.59 / 0.31 | $9\cdot10^{46}$ / $1\cdot10^{10}$ |
| 0.03 | 1 | 0.94 / 0.26 | 1.10 / 0.27 | $2\cdot10^{32}$ / $1.2\cdot10^{9}$ |
| 0.03 | 4 | 0.41 / 0.17 | 0.46 / 0.083 | $7\cdot10^{13}$ / $2.2\cdot10^{4}$ |
| 0.03 | 16 | 0.30 / 0.15 | 0.22 / 0.034 | $5\cdot10^{8}$ / $1.5\cdot10^{3}$ |
| 0.1 | 1 | 0.27 / 0.14 | 0.29 / 0.066 | $1.3\cdot10^{9}$ / $1.6\cdot10^{3}$ |
| 0.1 | 4 | 0.094 / 0.096 | 0.15 / 0.016 | $6\cdot10^{4}$ / 71 |
| 0.1 | 16 | 0.11 / 0.10 | 0.034 / 0.009 | 430 / 58 |
| 0.01 | 16 | 0.78 / 0.21 | 0.66 / 0.15 | $7\cdot10^{21}$ / $2\cdot10^{6}$ |
| 0.03 | 64 ($t \le 45$) | 0.19 / 0.13 | not run | |

Readings:

1. Centering cuts the post-trapping rate by 3 to 10 times. The "multi-stream spread restores the loop" risk (theory 4.4) does not appear at $\sigma \ge 0.03$, $N \ge 4$ [measured].
2. At $\sigma = 0.03$ the centered post-trapping rate fits $0.018 + 0.26/N$ from $N = 4, 16$ and predicts $0.28$ at $N = 1$ (measured 0.27) [estimated fit, one seed]. There is no floor near 0.23. The plain rates 1.10, 0.46, 0.22 also do not fit a clean $a + b/N$, so the `sec_wake` floor of 0.23 from $t \in [10, 30]$ mixes the trapping transient with the loop.
3. During injection ($t \in [10, 30]$) the centered rate levels off: 0.26, 0.17, 0.15, 0.13 at $N = 1, 4, 16, 64$ ($\sigma = 0.03$), and about 0.10 at $\sigma = 0.1$ for $N = 4, 16$. This $N$-independent part grows $\omega_{rms}$ by about $e^{2.6}$ over the injection window. It is either growth of the true signal during bunch formation or the field-level instability of Section 2. This review did not resolve which.
4. At $\sigma = 0.01$ the centered post-trapping rate is still strongly $N$-dependent (0.94, 0.31, 0.15 at $N = 1, 4, 16$). The centered rate is not "nearly independent of $\sigma$" after trapping, unlike the laminar toy of theory 4.4.

**Gradient quality [measured: `wc/an_ens.py`, `wc/out/ens/`; FD from `work_wake/out/fd/`].** Centered estimator against CRN central FD of the noisy model at $h = 0.04$ (FD at $N = 1$ from `work_wake/out/fd/`, FD at $N = 4$, 10 seeds, from `wc/out/fd4/`; the $N = 16$ runs are compared with FD at $N = 4$):

| $\sigma$, $N$ | $t$ | single-run SD / $\lvert g\rvert$, six observables | $\lvert z\rvert > 2$ |
|---|---|---|---|
| 0.1, 1 | 30 | 3.0, 3.2, 3.7, 6.9, 21, 82 | 0 of 6 |
| 0.1, 1 | 75 | 27, 66, 220, 26, 250, 2500 | 0 of 6 |
| 0.1, 4 | 30 | 1.3, 1.3, 0.51, 2.2, 6.9, 15 | 2 of 6 ($z = 2.8$, $-2.2$) |
| 0.1, 4 | 75 | 2.1, 5.0, 8.5, 3.5, 8.2, 37 | 1 of 6 ($z = -2.6$) |
| 0.03, 1 | 75 | $10^7$ to $10^8$ | diverged |
| 0.01, 1 | 75 | $10^{21}$ to $10^{23}$ | diverged |
| 0.03, 4 (30 seeds, $t = 30$; 10 seeds, $t = 75$) | 30 | 1.3 to 225 | 5 of 6 at 30 seeds ($z = -2.4, 2.2, 2.4, -2.1, -2.2$) |
| 0.03, 4 | 75 | 280 to $1.8\cdot10^4$ | 0 of 6 |
| 0.03, 16 (8 seeds) | 30 | 0.77, 1.4, 3.1, 8.1, 11, 51 | 0 of 6 |
| 0.03, 16 | 75 | 9.2, 14, 23, 25, 34, 490 | 0 of 6 |

At $\sigma = 0.1$, $N = 4$, $t = 75$ centering improves the single-run noise by about 30 to 700 times over the plain values of `sec_wake` (140 to 25000), but it stays 2 to 37 times the gradient. Three of 12 comparisons at $N = 4$ have $\lvert z\rvert$ between 2.2 and 2.8. With 11 heavy-tailed seeds this may be chance; a small bias of the centering is not excluded.

Bias check of the centering itself [measured: `wc/an_paired.py`]. At $\sigma = 0.3$, $N = 4$, the centered and plain estimators run on the same 11 seeds and noise. The paired difference is zero within 1 SE at $t = 30$ (largest $\lvert z\rvert = 0.85$) and within 1.7 SE at $t = 75$. Centering cuts the SE at $t = 75$ by up to 5 times (for example $\langle\xi\rangle$: 6.2 to 1.1). So the control variate is unbiased where it can be tested.

The $\sigma = 0.03$, $N = 4$, $t = 30$ deviations (5 of 6 observables at about 2.2 SE, same sign at 6 and 30 seeds, medians also off) vanish at $N = 16$ (all $\lvert z\rvert < 0.8$). This fits an $O(1/N)$ bias, either of the mean-field closure against the finite-$N$ FD (finding 8) or of the centering, that is visible only once the variance is small enough. It is not resolved.

Summary of Section 3. With centering and $N = 16$ the estimator is stable after trapping at $\sigma = 0.03$ and agrees with FD. Its single-run noise is still 0.8 to 50 times the gradient at $t = 30$ and 9 to 490 times at $t = 75$, against 3 percent for one CRN FD pair of $\log Q$ at $N = 4$. The $t = 30$ noise falls roughly as $N^{-1/2}$ from $N = 4$ to 16, so 15 percent on $\log Q$ at $t = 30$ needs about $(3.1/0.15)^2 \approx 430$ times more markers than $N = 16$ [estimated]. The concrete goal is not reached in 1D, but the reason has changed: it is now variance at a fixed rate, not the divergence reported by `sec_toy` and `sec_wake`.

## 4. The feedback control does not remove the laser quiver (finding 4)

**Claim reviewed.** Theory 4.3: with (2.6) and $\omega_c \approx 1$ to $3$, "the quiver contribution to $\mathbf h$ falls by about $(\omega_c/\omega_0)^2$, and to the variance by its square". Evidence offered: a Duffing oscillator with a fast force $B\cos(\Omega t)$ that does not depend on position or momentum.

**Why the scaling fails [derived].** (2.6) puts $(\partial\mathbf b/\partial\mathbf x)\xi_x + (\partial\mathbf b/\partial\mathbf u)\xi_u$ into $\mathbf h$. In a laser, $\partial\mathbf b/\partial\mathbf u$ contains the magnetic rotation $\lvert\mathbf B\rvert/\gamma \sim a_0\omega_0/\gamma$, and $\partial\mathbf b/\partial z \sim k_0\lvert\mathbf F\rvert$. With $\xi_u \sim \lvert\delta\mathbf a\rvert/\omega_0$, the term $(\partial_u\mathbf b)\xi_u \sim (a_0/\gamma)\lvert\delta\mathbf a\rvert$ is of the same size as $\delta\mathbf a$. The $\omega_c^2M^{-1}$ term also carries $M^{-1} \propto \gamma^3$ along the motion. The theory's own formula $\lvert\mathbf h\rvert \approx \lvert\delta\mathbf a\rvert(\omega_c^2 + \lvert\partial_x\mathbf b\,M\rvert)/\Omega^2$ already shows the second term, but the estimate dropped it.

**Test [measured: `wc/laser_feedback.py`, outputs `wc/out/lf_*.txt`].** One electron at rest, $x_0 = 1$, hit by a linearly polarized pulse, $\omega_0 = 14.8$, $\tau = 2.55$ (Twin-S values from theory 1.3), $w_0 = 4$, $\theta = \ln E_L$ so $\delta\mathbf a = \mathbf F/2$. RK4 with 80 steps per period; the $a_0 = 4$ feedback value is the same at 160 and 320 steps per period. Entries are $\int\lvert\mathbf h\rvert^2dt$, the weight variance times $\sigma^2$:

| $a_0$ | $\omega_c$ | pure Girsanov | feedback (2.6) | damping only ($\mathbf h = \omega_c^2M^{-1}\xi_x + 2\omega_c\xi_u$) |
|---|---|---|---|---|
| 1 | 0.3 | 80 | 82 | 24 |
| 1 | 1 | 80 | 75 | 222 |
| 1 | 3 | 80 | 17 | 521 |
| 4 | 1 | 824 | $2.4\cdot10^{13}$ | $3.2\cdot10^{6}$ |
| 4 | 3 | 824 | $1.7\cdot10^{8}$ | $2.1\cdot10^{7}$ |

The pathwise tangent ($\mathbf h = 0$) stays bounded through the pulse: $\lvert\xi_u\rvert = 0.21$ at $a_0 = 1$ and $3.4$ at $a_0 = 4$ after the pulse. The pure-Girsanov value at $a_0 = 4$ (824) confirms the order of the theory estimate ($1.4\cdot10^3$ in the lab frame; the co-moving electron sees $(1 - v_z)$ times the force) [checked].

Consequence: in the laser the only cheap choice is $\mathbf h = 0$, that is no noise there (a spatial mask) and the exact pathwise tangent, which is bounded in a single pass. The handover of that tangent to weights behind the laser is then unavoidable and costs $\gamma^2$ to $\gamma^6$ (theory 2.3). The Twin-S hybrid cost estimates in theory 6 assume a feedback that does not work.

## 5. Representation, closure, births, Vay step, joint versus mean-field (findings 5, 8)

**Checked and correct.**

1. Pure Girsanov (2.3) and the mixed identity (2.5) [checked, derived]. A shorter proof of (2.5): under the measure $Q^\epsilon$ with Girsanov density for the drift $\epsilon\mathbf h$, the process with drift $\mathbf b + \epsilon(\delta\mathbf a - \mathbf h)$ has drift $\mathbf b + \epsilon\delta\mathbf a$. Differentiating at $\epsilon = 0$ gives $\mathbb E[\omega g + \xi\cdot\nabla g]$. It needs $\mathbf h \in \mathrm{range}(\Sigma)$ and square integrability; feedback with a dependence on $\xi$ is allowed because only $\mathbf h$ at $\epsilon = 0$ enters.
2. Discrete step [checked]. The transition $\mathbf u^{n+1/2} \sim N(\mathbf m^n, \Sigma\Sigma^T\Delta t)$ has score $(\Sigma^{-1}\mathbf h)\cdot\boldsymbol\zeta/\sqrt{\Delta t}$ in direction $\mathbf h$. The implicit Vay map enters only through $\mathbf m^n$, so the derivative of the discrete noisy chain is exact [literature: Chen and Glasserman, Stoch. Proc. Appl. 117 (2007), from memory].
3. EV births [checked]. A deterministic birth weight $W_b(\theta)$ is a source term in the Feynman-Kac representation. Its tangent enters as a weight $\dot W_b$, with no score and no $\nabla f$, so a cold newborn is safe. The wake harness implements this and passes FD checks at $\sigma = 0.3, 1$.
4. Mean-field closure [checked]. The weight increments have zero conditional mean even when $\delta\mathbf E^N$ is noisy, because $\delta\mathbf E^N$ is adapted. The map from weights to field tangent is linear. The cross terms $\mathbb E[\eta(Z_i)\nabla_u\varphi(Z_i)]$ vanish up to $O(1/N)$ correlations and the self term.

**Finding 8 (minor).** Theory 2.4 says the noise expectation solves (2.1)-(2.2) "exactly, given the primal paths". Given the paths, the weights are deterministic, because they are built from the same increments. The correct statement: the estimator is unbiased for the mean-field derivative up to the propagation-of-chaos error. That error is $O(1/N)$ with a Gronwall constant $e^{LT}$, where $L$ includes the field-level rate of Section 2. So the bias constant can also grow exponentially after trapping and must be measured.

**Finding 5 (major).** Theory (3.1) and lit Identity 2 take the explicit $\theta$-dependence at fixed paths as the vacuum laser tangent only. For the inversion this is incomplete in two ways [derived]:

1. EV birth weights depend on $\theta$ at fixed paths through $\lvert\mathbf E\rvert$ at the ion. Their change alters the deposits and hence the fields at fixed paths. So the joint score must use the field tangent sourced by $\delta W_b$ (and the remaining ion fraction), and $\mathbb E[J]$ gets an extra pathwise term $\partial_{W_b}J\,\delta W_b$. The wake harness does this correctly (`dEW`, `dobs_W` in `work_wake/wake_lr.py`); the theory formula does not.
2. In FBPIC $\theta$ enters through the initial field. The vacuum tangent at later times is a Maxwell solve per parameter, sourced also by $\delta W_b$. "Needs no field tangent" is false; the joint form needs a source-limited field tangent. This does not change the conclusion that the joint form is too noisy ($\mathrm{Var} \propto N\,I_1$).

The mean-field versus joint distinction itself is right: the joint form targets $d\mathbb E_N/d\theta$ exactly with $O(1)$ single-run error; the mean-field form targets the McKean-Vlasov derivative with $O(1/N)$ variance per run when the loop is controlled [checked against `sec_toy` 3.3-3.5 and `sec_wake` 3.5].

## 6. Is the noise physically defensible for LWFA? (findings 6, 9, 10)

**Collisions [estimated: `wc/phys_noise.py`, NRL formulary rates].** For $n_e = 8\cdot10^{18}\,\mathrm{cm^{-3}}$ and $T_e = 5$ to $50$ eV after field ionization, $\nu_{ei}/\omega_p = 0.05$ to $0.0025$, and the momentum diffusion gives $\sigma_{phys} \approx (0.7$ to $1.0)\cdot10^{-3}$ per $\omega_p^{-1/2}$. The proposed $u_n = 0.03$ over $T_n = 14.6$ needs $\sigma = 7.9\cdot10^{-3}$: 8 to 11 times $\sigma_{phys}$, 60 to 130 times in diffusivity. For electrons moving at $\lvert u\rvert \sim 1$ in the wake, the collision rate drops by $(u_{th}/u)^3 \approx 2.4\cdot10^{-7}$, and for the bunch at $\gamma \gg 1$ it is smaller still. So "collisions" does not justify the level [estimated].

**Temperature.** A real initial temperature is a static spread $u_{th} \approx 0.003$ to $0.01$. Pure diffusion is unbounded heating: the wake harness at $\sigma = 0.01$ reaches a spread of 0.084 (3.6 keV equivalent) after 70 periods. A warm initial plasma would be more physical, but it does not help option C: weights need noise at the times where $\delta\mathbf a$ acts.

**Shot-to-shot jitter.** Jitter is a coherent fluctuation of $\theta$ (or of a few laser and plasma parameters) shared by all particles. Smoothing $\mathbb E[J]$ over it is a convolution in $\theta$ with $O(h^2)$ bias. Independent per-particle kicks change the model itself. Measured consequence (`sec_wake` 3.3): $\sigma = 0.01$ removes 12 percent of the trapped charge at $t = 75$. Diffusion across the trapping separatrix changes the injected population at first order in $D$. No $\theta$-smoothing does that. So the equivalence $h_{eq} \approx u_n/\lvert\partial u/\partial\theta\rvert$ (theory 1.3 item 4) has no derivation and is contradicted at the level of $\mathbb E[J]$ [derived; measured by `sec_wake`].

**Effect on the trapped bunch.** Three channels [derived, estimated]: (i) injection, via separatrix diffusion of the newborns and of the background, which is the dominant and first-order effect; (ii) the wake, via background heating and phase mixing, which at $\sigma \ge 0.3$ removes acceleration in the 1D wake; (iii) the accelerated bunch, where longitudinal kicks are negligible against $\gamma$ and the transverse emittance growth $\Delta\varepsilon_n = u_n^2/\sqrt{2\gamma}\,k_p^{-1}$ is small [checked: theory 1.3 item 2 formula re-derived]. Channel (i) is why the bias does not vanish as $\sigma^2$ in the trapping regime.

**Finding 9 (minor).** Theory 1.3 item 3 sets the selection-charge bias at $O(u_n^2/(4\delta u)^2)$. For a smoothing of the observation alone, $\mathbb E[S(u + \eta)] - \mathbb E[S(u)] = \tfrac12u_n^2\int S''f\,du = \tfrac12u_n^2\int S f''\,du$ to leading order. The bias depends on the curvature of $f$ at $u_c$, not on the step width. The 4 percent and 43 percent figures are unsupported [derived].

**Finding 10 (minor).** For $\gamma_p \gg 1$, $E_{max}/E_0 \approx 0.76\,\beta_{th}^{-1/2}$ [literature: Katsouleas and Mori, PRL 61, 90 (1988), from memory]. At $u_n = 0.1$ this is about 2.4, comparable to a nonlinear wake at $a_0 = 4$, not "far above".

## 7. Novelty and literature (finding 7)

The `sec_lit` verdict (algorithmic and applied novelty, mathematics not new) is honest. It misses the line of work closest to "per-particle signed weights closed through a mean-field quantity, with a stability analysis":

1. Poyiadjis, Doucet, Singh, "Particle approximations of the score and observed information matrix in state space models with application to parameter estimation", Biometrika 98 (2011) [literature, abstract]. The path-space score estimator degenerates in time; an $O(N^2)$ marginalized estimator stays stable.
2. Del Moral, Doucet, Singh, "Uniform stability of a particle approximation of the optimal filter derivative", SIAM J. Control Optim. 53(3):1278 (2015), [arXiv:1106.2525](https://arxiv.org/abs/1106.2525) [literature, abstract]. $L^p$ bounds and asymptotic variance uniform in time under mixing. Tadić and Doucet, [arXiv:1806.09590](https://arxiv.org/abs/1806.09590), bound the bias uniformly in time.
3. Sharrock, Kantas, Pavliotis, "Recursive maximum likelihood estimation for interacting particle systems using virtual particles", [arXiv:2605.00786](https://arxiv.org/abs/2605.00786) (2026) [literature, abstract]. A virtual tangent interacting particle system estimates gradients of the McKean-Vlasov likelihood. It is the closest McKean-Vlasov algorithm found; it is pathwise and stationary, not LR and transient.

Two consequences. First, the paper must cite these and state the difference (Vlasov-Maxwell field closure, cold start, forward mode, transient). Second, the SMC result that path-accumulated weights degenerate and marginalized weights do not is the theoretical support for the Rao-Blackwell fallback in `sec_lit` 6 item 4. That fallback needs a phase-space mesh, which the brief disfavors.

A web search (2026-11-01) for Girsanov or likelihood-ratio weights in PIC or Vlasov-Fokker-Planck sensitivities found nothing beyond Chung et al. (JCP 400, 2020). This supports the "combination not found" claim but is not exhaustive.

## 8. What a decision needs next

1. Decide whether the injection-phase floor (centered rate about 0.13 at $N = 16$ to 64) is true signal growth or a field-level instability. Test: compare $\lVert\mathbb E[\delta\mathbf E^N]\rVert$ over seeds with the seed-to-seed spread of $\delta\mathbf E^N$ during $t \in [10, 30]$ (Section 2).
2. Replace (2.6) in the laser by $\mathbf h = 0$ with a noise mask, and price the handover. Re-run the cost table of theory 6 with that design.
3. Fix the smoothing scale by physics, not by variance. The only defensible floor in this critique is $\sigma \lesssim 10^{-3}$ (collisional), which every measured variant fails by orders of magnitude.

## Files

- `wc/wake_lr_c.py`: copy of `work_wake/wake_lr.py` with charge-neutral increments (`CENTER=1` self-inclusive, `CENTER=2` leave-one-out).
- `wc/an_grow.py`: growth rates, plain and centered, seed 1; outputs in `wc/out/grow/`.
- `wc/an_ens.py`, `wc/an_ens30.py`, `wc/an_paired.py`: centered ensembles against CRN FD, and the paired centered-minus-plain test; outputs in `wc/out/ens/` and `wc/out/ens30/`, FD at $N = 4$, $\sigma = 0.03$ in `wc/out/fd4/` (`wc/fd_batch.py`).
- `wc/laser_feedback.py`: single electron in a resolved laser pulse, controls `path`, `lr`, `fb`, `damp`; outputs `wc/out/lf_*.txt`.
- `wc/phys_noise.py`: collisional momentum diffusion at Twin-S density.
