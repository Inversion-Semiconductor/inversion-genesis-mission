# Option C: noise-regularized likelihood-ratio weights for macroscopic gradients

| Item | Value |
|---|---|
| Date | 2026-11-01 |
| Status | Option C study complete (derivation, literature, two 1D evaluations, two critics, synthesis with one new 1D test). Recommendation: no-go for an `fbpic/autodiff` implementation. One cheap Twin-S check is offered. |
| Goal under test | Forward-mode $d\,\mathbb E[J]/d\theta$ of the 11 whitened beam-moment residuals from one run plus one weight set per parameter, which (1) does not grow exponentially under chaotic trajectories, (2) works on a cold plasma and a thin injected bunch, (3) matches a trusted reference within statistical error, and (4) costs well below one run per extra parameter. |
| Base directory | `inversion/notes/option-c-evidence-2026-11-01/`, written `lr/` below |
| Inputs | `lr/sec_theory.md`, `lr/sec_lit.md`, `lr/sec_toy.md`, `lr/sec_wake.md`, `lr/crit_math.md`, `lr/crit_evidence.md`; R0 note `inversion/notes/r0-design-note-2026-10-31.md`; findings 4 to 6 in `inversion/research/sensitivity-feasibility/findings/` |

Labels. **[derived]** derived here or in an input section and checked here (Appendix A). **[measured: script]** CPU NumPy run with `python -I`; paths relative to `lr/`. **[literature: citation]**. **[estimated]** an estimate with stated inputs. A result from an input section carries its key, for example [measured, `crit_evidence`: `work_crit_evidence/wake_lr_c.py`]. New runs of this note are in `lr/work_note/`.

## 1. Decision for the user

**Decision.** Stop option C at the 1D stage, without an implementation in `fbpic/autodiff`?

**Recommendation: yes.** Option C computes the right quantity, but the 1D evidence shows no noise level at which it is both accurate and cheap.

1. It is correct. The weights give an unbiased gradient of the noisy model, also on a cold plasma, wherever their variance is finite [derived; measured]. Goals (2) and (3) hold against the noisy model.
2. It has no usable noise window on the 1D cold wake. The trapped charge stays within 5% only for $u_n \lesssim 0.07$. Already at $u_n = 0.2$ the single-run error is 56 to 150 times the gradient at $N = 4$, and it grows as $u_n$ falls. Noise that bounds the weights ($u_n \ge 0.67$) changes the trapped charge by $-27\%$ to $-55\%$ and still gives a single-run error 2 to 39 times the gradient [measured, Section 4]. This holds for every variant tried, including the best design found here: noise only after injection, a mixed estimator, and leave-one-out (LOO) charge-neutral increments.
3. Its exponential growth is no longer chaotic, but it remains. With LOO centering, the weight growth rate falls roughly as $1/N$ and is not tied to the trajectory Lyapunov rate [measured]. At the noise level that keeps the physics ($\sigma = 0.01$, noise from $t = 0$), the rate is still 0.31 and 0.15 $\omega_p$ at $N = 4$ and 16. Goal (1) is not met there.
4. It costs more than FD. One weight set costs 2.1 to 2.9 primal runs in NumPy and about the existing tangent cost in FBPIC, because the live trajectory tangent is still needed. The variance needs 16 to 600 times more runs than one CRN FD seed for the same standard error (SE) at $u_n \ge 0.67$, and more than $10^4$ times at $u_n = 0.2$ [measured; estimated]. Goal (4) fails by one to four orders of magnitude.

**Consequences if the user accepts.**

1. No option C code in `fbpic/autodiff`. The P7 step-5 cache stays valid.
2. No forward-mode method in hand meets all four goal criteria. The open choice of finding 6 remains: the target derivative (noise-regularized or pointwise) and the Twin-S reference resolution (`p_nr` 8, finding 5).
3. The negative result is reportable. Option C turns trajectory chaos into a field-level sampling problem with an information floor, $\mathrm{SE}/\lvert\delta m\rvert \gtrsim s_g/(u_n\sqrt{N_{eff}})$ (Section 3.4). This floor is a general limit for score-type estimators on cold, thin bunches.

**One cheap Twin-S check, if the user wants the closure to be Twin-S-native.** Gate B of Section 6 needs one tangent run at `p_nr` 8 to step 1520 with a logging callback, about 1 GPU-h and half a day of analysis. It needs no kernel and no `fbpic/autodiff` edit. It kills option C on Twin-S if a perfect estimator would need $u_n > 0.07$ for 30% error on the charge or energy residual. The 1D evidence predicts that it kills [estimated].

## 2. The method in plain terms

1. Add a small random momentum kick to every electron push in a time window: $\mathbf u \leftarrow \mathbf u + \sigma\sqrt{\Delta t}\,\boldsymbol\zeta$, $\boldsymbol\zeta \sim N(0, I)$. The expectation $\mathbb E[J]$ of this noisy model is smooth in $\theta$. The noise level $u_n = \sigma\sqrt{T_n}$ (momentum spread after a window $T_n$) is the smoothing scale.
2. Do not differentiate the trajectories. Instead, give each particle a scalar weight $\omega_i$. Each step, add the projection of the force tangent on the kick that the particle actually received: $\Delta\omega_i = \delta\mathbf a_i\cdot\boldsymbol\zeta_i\sqrt{\Delta t}/\sigma$.
3. Deposit the weights as a charge and current tangent, $\delta\rho = \sum_iq w_i\omega_iS$ and $\delta\mathbf J = \sum_iq w_i\omega_i\mathbf v_iS$. The existing linear field-tangent solver closes the loop and returns $\delta\mathbf E, \delta\mathbf B$ for the next $\delta\mathbf a$.
4. Read the gradient of a moment as a weighted covariance: $\delta m = \sum_iw_i\omega_i(g_i - m)/\sum_iw_i$.
5. For $K$ parameters, carry $K$ weight sets on the same particles and the same kicks.

**Small example [derived].** One cold electron, one step, force $a(\theta)$: $u' = u_0 + a\Delta t + \sigma\sqrt{\Delta t}\,\zeta$. Take $g(u) = u$. The weight is $\omega = a'\zeta\sqrt{\Delta t}/\sigma$, and
$$
\mathbb E[g(u')\,\omega] = \mathbb E\big[(u_0 + a\Delta t + \sigma\sqrt{\Delta t}\,\zeta)\,\zeta\big]\,\frac{a'\sqrt{\Delta t}}{\sigma} = a'\Delta t = \frac{d\,\mathbb E[u']}{d\theta}.
$$
The derivative moved from the path onto the Gaussian density of the kick. The weight needs no $\nabla_u f$, so a cold sheet is no problem. The current tangent is not zero, because $\omega$ is correlated with the kick that moved $\mathbf v$. The price is visible too: $\mathrm{Var}\,\omega = a'^2\Delta t/\sigma^2$, so a small kick gives a large weight.

## 3. Theory

### 3.1 What is estimated

Noisy model per electron [derived, `sec_theory` 1.1]:
$$
d\mathbf x = \mathbf v\,dt,\qquad d\mathbf u = \mathbf a(\mathbf x, \mathbf u, t;\theta)\,dt + \Sigma(t, \mathbf x)\,d\mathbf W,
$$
with $\Sigma$ nonzero only in a time window and, optionally, outside a spatial mask. The phase-space density solves a Vlasov-Fokker-Planck (VFP) equation. Its $\theta$-derivative $\delta f$ solves the linearized VFP with source $-\nabla_u\cdot(f\,\delta\mathbf a)$, coupled to linear Maxwell equations for $\delta\mathbf E, \delta\mathbf B$.

**Mixed identity [derived, Appendix A.2].** Let $\mathbf h$ be any adapted per-particle process in the range of $\Sigma$. Split the one-step tangent: the pathwise tangent $\xi$ carries $\delta\mathbf a - \mathbf h$, the weight carries $d\omega = (\Sigma^{-1}\mathbf h)\cdot d\mathbf W$. Then
$$
\int g\,\delta f(T) = \mathbb E\big[\omega_T\,g(Z_T) + \xi_T\cdot\nabla g(Z_T)\big].
$$
$\mathbf h = 0$ is the existing exact tangent. $\mathbf h = \delta\mathbf a$ is pure Girsanov. Any feedback $\mathbf h = \delta\mathbf a + (\text{terms in }\xi)$ that damps $\xi$ is a handover from paths to weights. In the time-discrete FBPIC step the identity is exact for the discrete noisy chain, because the transition density of $\mathbf u^{n+1/2}$ is $N(\mathbf m^n, \Sigma\Sigma^T\Delta t)$ and the Vay map enters only through $\mathbf m^n$ [derived; checked by `crit_math` 5].

**Mean-field versus joint [derived, Appendix A.5].**

1. *Mean-field estimator* (this method). Weights close through the field tangent. Its target is the derivative of the mean-field limit. Its bias against the finite-$N$ noisy PIC is the propagation-of-chaos error, $O(1/N)$ with a Gronwall constant that can grow in time. The statement in `sec_theory` 2.4 that it is exact "given the primal paths" is withdrawn: given the paths, the weights are fixed.
2. *Joint estimator* (one scalar score for the whole $N$-particle chain). It is exact at finite $N$ and has no closure loop. Its score uses only the explicit $\theta$-dependence at fixed paths. That is the vacuum laser tangent **plus** the field of the birth-weight change $\delta W_b$ at fixed paths; the estimator also needs the pathwise term $\sum_b\partial_{W_b}J\,\delta W_b$. In FBPIC both need a source-limited Maxwell tangent solve. Its single-run variance does not fall with $N$ [derived; measured, `sec_wake` 3.5: SD/$\lvert g\rvert$ 10 to 15 at $N = 1$ and 4].
3. Use the joint form only for few particles with many independent replicas and a weak, short explicit $\theta$-source. For $\theta = E_L$ the explicit source is the whole laser acting on every background electron, so the joint form is not a candidate.

### 3.2 The cold case

The identity needs noise, not a density of $f$. A cold sheet at the window start acquires a Gaussian width $u_n$ in the noisy directions. The weight increments are then correlated with the velocity changes, $\mathbb E[\Delta\omega\,\Delta\mathbf u] = \delta\mathbf a\,\Delta t$, so a charge-neutral weight change in a cold cell still carries current. The R0 cold-limit theorem does not apply [derived, `sec_theory` 2.2; checked by `crit_math`]. EV birth weights are a deterministic Feynman-Kac source: a newborn starts with $\omega_b = \dot W_b/W_b$ and needs no score [derived; measured: the wake harness passes FD checks with births, `sec_wake` 3.2].

### 3.3 Bias: the noise changes the model

1. The method returns the gradient of the noisy model, not of the deterministic one. For a smooth deterministic response the difference is $O(\sigma^2)$ [measured, `sec_toy` 2.5 and 3.5 before saturation].
2. Trapping is not smooth. Diffusion across the trapping separatrix changes the trapped charge directly. On the 1D wake, noise from $t = 0$ at $\sigma = 0.01$ ($u_n \approx 0.074$) removes $13\%$ of the trapped charge [measured, `crit_evidence` `bias.py`]. Noise only after injection ($t \ge 30$) removes $3.5 \pm 0.4\%$ at $u_n = 0.067$ and $17\%$ at $u_n = 0.2$ [measured: `work_note/an_win.py`].
3. The noise is not a physical scale. It is 8 to 11 times the collisional momentum diffusion of the cold background (60 to 130 times in diffusivity), and many orders larger for fast electrons [estimated, `crit_math` `phys_noise.py`, NRL formulary]. Pure diffusion is unbounded heating, not a temperature. Shot-to-shot jitter is a coherent fluctuation of $\theta$, which smooths $\mathbb E[J]$ at $O(h^2)$ and does not change the trapped charge. Independent per-particle kicks change the model. The equivalence $h_{eq} \approx u_n/\lvert\partial u/\partial\theta\rvert$ of `sec_theory` 1.3 has no derivation and is withdrawn.
4. Consequence: with option C, the inversion fits a different model. The size of that model change must be measured on Twin-S before any use (Gate A).

### 3.4 Variance: what grows and what does not

**Itô isometry [derived, A.3].** $\mathbb E[\omega_T^2] = \mathbb E\int_0^T\lvert\Sigma^{-1}\mathbf h\rvert^2dt$. For a bounded force tangent the weight variance grows linearly in time. Single-particle Lyapunov stretching does not enter the weights. This is the core promise of option C, and it holds [measured: maps in `sec_toy` 2.3; the wake without field feedback has $\sigma\omega_{rms} = 0.5$ to 0.8 at every $\sigma$ and $N$, `sec_wake` 3.1].

**Exponential growth needs a growing gathered field tangent [derived, A.4].** The estimator's field tangent is $\delta\mathbf E^N = \delta\mathbf E + \boldsymbol\eta$. The mean $\delta\mathbf E$ is the true response. The fluctuation $\boldsymbol\eta$ solves the same linearized system, driven by the sampling noise of the weighted deposit. Two mechanisms grow $\boldsymbol\eta$:

1. *Sampling loop.* Weight noise deposits random charge, which feeds new increments. The rate scales as $c/(n_m\sigma^2)$ for the plain estimator ($n_m$ = markers per resolved volume) [measured, `sec_theory` 4.4, `sec_toy` 3.2].
2. *Field-level amplification.* Around a saturated or multi-stream state, the linearized operator can amplify directions that the coherent $\theta$-source barely excites. White sampling noise excites all of them. Then $\mathrm{Var}\,\omega$ grows while the true gradient stays bounded. In the two-stream toy the FD gradient falls 10 to 50 times from $t = 20$ to 100 while $\omega_{rms}$ grows 560 times [measured, `crit_math` 2, from `work_toy/out/es_long_summary_ref0.1.txt`].

The claim of `sec_theory` 4.4(a), that excess weight growth is true sensitivity that every exact method sees, is wrong and withdrawn. Option C moves the exponential growth from particle tangents to field-tangent noise. It removes the growth only if the field-level rate of the noisy primal is small, and no general argument bounds that rate.

**Charge-neutral (LOO-centered) increments [derived, A.6; measured].** Subtracting from each increment the shape-weighted local mean of the other particles' increments keeps the expectation (to $O(1/N)$) and removes most of the loop. The plain local mean (self included) is biased by 20 to 40% at 5 markers per cell, so the LOO form is required [measured, `crit_evidence` 4]. LOO centering is mandatory in any implementation.

**Information floor [derived, A.3].** A weight built from noise of spread $u_n$ that must represent a momentum shift $\xi_u$ has $\mathrm{Var}\,\omega \ge \lvert\xi_u\rvert^2/u_n^2$ (Cauchy-Schwarz on the Itô isometry). For a moment with per-particle spread $s_g$, if $\omega^2$ and $(g - m)^2$ are nearly uncorrelated:
$$
\frac{\mathrm{SE}(\widehat{\delta m})}{\lvert\delta m\rvert} \gtrsim \sqrt R\;\frac{s_g\,\lvert\xi_u\rvert}{u_n\sqrt{N_{eff}}\,\lvert\delta m\rvert},\qquad R \ge 1 .
$$
$R$ is the inefficiency of the actual estimator over this floor. On the 1D wake $R \approx 60$ [estimated, `crit_evidence` 6]. More particles do not change the ratio $s_g/u_n$. Only a larger $u_n$ does, and a larger $u_n$ changes the physics. This, not the loop, is the binding limit after centering.

### 3.5 Fast forces: the laser

1. Pure weights in a resolved laser cost $\int\lvert\delta\mathbf a\rvert^2dt/\sigma^2$ with the full quiver force, while the useful signal is ponderomotive, $(\omega_p/\omega_0)^2$ smaller. Relative variance grows as $\omega_0^4$ [derived; measured, `sec_toy` 4].
2. The feedback control (2.6) of `sec_theory` does not fix this. In a laser, $\partial\mathbf b/\partial\mathbf u$ (magnetic rotation) and $\partial\mathbf b/\partial z$ are fast and large, so the feedback terms are as large as $\delta\mathbf a$. In a resolved pulse with Twin-S scales, (2.6) gives no gain at $a_0 = 1$ and a $3\cdot10^{10}$ times larger weight variance at $a_0 = 4$ (time-step converged) [measured, `crit_math` `laser_feedback.py`]. The Duffing test of `sec_theory` 4.3 used a force that does not depend on the state and cannot show this.
3. The usable design is $\mathbf h = 0$ in the laser: no noise there, exact pathwise tangent, which stays bounded in one pass ($\lvert\xi_u\rvert = 3.4$ after the pulse at $a_0 = 4$) [measured, `crit_math`]. On Twin-S the laser leaves the plasma at step 1520 [R0, Section 2]. A noise window from step 1520 therefore avoids the laser, and probably most of the injection [estimated; Gate A checks the charge].
4. Handover of the tangent at the window start costs, for free streaming over $\tau$, at least $\lvert\xi_u\rvert^2/(\sigma^2\tau) + 12\lvert\xi_x\rvert^2/(M^2\sigma^2\tau^3)$, with $M = \partial v/\partial u$ [derived, A.7]. For the longitudinal position of a relativistic bunch, $M = \gamma^{-3}$ makes the second term scale as $\gamma^6$. So keep $\xi_x$ pathwise and damp only $\xi_u$ (the variant tested in Section 4.3).

### 3.6 What option C can and cannot do

| Can | Cannot |
|---|---|
| Give an unbiased gradient of the noisy model, for any $\sigma > 0$, on cold particles and EV newborns | Give the gradient of the deterministic model; the bias depends on $u_n$ and is large wherever trapping decides the observable |
| Avoid single-particle Lyapunov growth (linear-in-time weight variance for bounded force tangents) | Bound field-level growth; the estimator noise can grow under collective amplification while the true gradient decays |
| Share the noise and paths across $K$ parameters | Cost less than one run per parameter while the live pathwise tangent is needed (laser, handover) |
| Remove most of the sampling loop with LOO centering | Beat the information floor $s_g/(u_n\sqrt{N_{eff}})$ without a phase-space basis (Rao-Blackwellization) |

## 4. 1D evidence against the four goal criteria

Harnesses: the 1D relativistic electrostatic wake with ponderomotive driver, cold plasma, EV injection, $\theta = a_0 = 2.5$, $t_f = 75$ (`work_wake/wake_lr.py`, with LOO centering in `work_crit_evidence/wake_lr_c.py`, with a late window and the mixed estimator in `work_note/wake_win.py`); the two-stream and map toys of `sec_toy`. "$N$" multiplies the particle count: plasma $5N$ per cell. References are CRN central FD of the same noisy model (cubic fit over $\lvert h\rvert \le 0.04$), and deterministic CRN FD for bias. "SD/$\lvert g\rvert$" is the single-run standard deviation over the reference gradient.

### 4.1 Growth of the weights (criterion 1)

Late growth rate of $\ln\omega_{rms}$ (plasma), plain against LOO centered, noise from $t = 0$, seed 1, $t \in [30, 75]$ [measured, `crit_math` `an_grow.py`, `crit_evidence` `show_grow.py`]:

| $\sigma$ | $N = 1$ | $N = 4$ | $N = 16$ |
|---|---|---|---|
| 0.01 | 3.22 / 0.94 | 1.59 / 0.31 | 0.66 / 0.15 |
| 0.03 | 1.10 / 0.27 | 0.46 / 0.083 | 0.22 / 0.034 |
| 0.1 | 0.29 / 0.066 | 0.15 / 0.016 | 0.034 / 0.009 |

New: noise only for $t \ge 30$ (after injection), mixed estimator with $\xi_u$ damped at rate $\kappa = 1$, LOO centered, median over 16 seeds, rate over $t \in [55, 75]$, bunch / plasma [measured: `work_note/an_win.py`, output `work_note/out/an_win.txt`]:

| $\sigma$ ($u_n = \sigma\sqrt{45}$) | $N = 1$ | $N = 4$ |
|---|---|---|
| 0.03 (0.20) | 0.27 / 0.28 | 0.097 / 0.086 |
| 0.1 (0.67) | 0.061 / 0.072 | 0.018 / 0.032 |
| 0.3 (2.0) | 0.014 / 0.016 | 0.008 / 0.008 |

Reading. The deterministic trapped-tangent rate is about $0.23\,\omega_p$ [R0 pilot]. With centering, the rates fall roughly as $1/N$ and rise as $\sigma$ falls. The "$N$-independent floor near the chaotic rate" reported in `sec_wake` 3.1 is residual sampling feedback; centering removes it [measured, both critics]. During injection a centered rate of about 0.13 remains at $N = 16$ to 64 ($\sigma = 0.03$); whether it is signal or field-level noise is not resolved [measured, `crit_math` 3]. In the two-stream toy (a collective instability), the centered rate is 0.07 at $\sigma = 0.03$, below the tangent rate of 0.17, and 0.33 at $\sigma = 0.01$, above it [measured, `crit_evidence` 3].

**Criterion 1: partly met.** The growth is no longer tied to trajectory chaos. At noise levels that keep the physics ($u_n \lesssim 0.07$) it is still exponential at the particle counts tested.

### 4.2 Bias against the deterministic model (criterion 3, second half)

Relative gradient bias $\pm$ SE at $t = 75$, $N = 1$, CRN FD at $h = 0.04$:

| Noise | $\sigma$ | $u_n$ | $\log Q$ | $\langle\xi\rangle$ | $\log\sigma_u$ | trapped charge |
|---|---|---|---|---|---|---|
| from $t = 0$ [`crit_evidence` `bias.py`, 32 to 38 seeds] | 0.01 | 0.074 | $0.05 \pm 0.06$ | $0.05 \pm 0.17$ | $0.43 \pm 0.12$ | $-13\%$ |
| | 0.03 | 0.22 | $0.21 \pm 0.05$ | $0.25 \pm 0.17$ | $0.65 \pm 0.18$ | $-28\%$ |
| | 0.1 | 0.74 | $0.27 \pm 0.04$ | $0.76 \pm 0.11$ | $0.29 \pm 0.05$ | $-55\%$ |
| from $t = 30$ [new, `work_note/an_win.py`, 24 seeds paired with the deterministic loading] | 0.01 | 0.067 | $-0.03 \pm 0.04$ | $0.01 \pm 0.11$ | $0.07 \pm 0.05$ | $-3.5 \pm 0.4\%$ |
| | 0.03 | 0.20 | $0.03 \pm 0.05$ | $-0.42 \pm 0.14$ | $0.76 \pm 0.10$ | $-16.8 \pm 0.8\%$ |
| | 0.1 | 0.67 | $-0.20 \pm 0.05$ | $0.35 \pm 0.14$ | $-0.14 \pm 0.05$ | $-27.1 \pm 0.7\%$ |

The deterministic reference itself moves by $0.14 \pm 0.06$ ($\log Q$) between $h = 0.02$ and 0.04 at $t = 75$, so $\langle u\rangle$ and the spread columns cannot anchor a tight bias target [measured, `crit_evidence` 6]. The late window cuts the charge bias 2 to 4 times. The bias is small only for $u_n \lesssim 0.07$.

### 4.3 Agreement and single-run noise (criteria 3 and 4)

Late window ($t \ge 30$), mixed estimator, LOO centered, 16 LR seeds against 12 to 24 FD seeds of the same noisy model [measured: `work_note/an_win.py`]:

| $\sigma$, $N$ | $t$ | max $\lvert z\rvert$ over 6 observables (powered comparisons) | LR SD/$\lvert g\rvert$: $\log Q$, $\log\sigma_u$ | FD one seed SD/$\lvert g\rvert$: same |
|---|---|---|---|---|
| 0.03, 4 | 75 | 1.8 (none) | 56, 72 | 0.20, 0.27 |
| 0.1, 4 | 45 | 3.1 (none) | 2.7, 5.4 | 0.038, 0.35 |
| 0.1, 4 | 75 | 1.1 (none) | 2.7, 2.0 | 0.11, 0.11 |
| 0.3, 4 | 45 | 3.1 (5) | 0.49, 0.33 | 0.074, 0.083 |
| 0.3, 4 | 75 | 1.0 (5) | 0.50, 0.56 | 0.13, 0.075 |
| 0.3, 1 | 75 | 2.7 (none) | 2.3, 1.4 | 0.36, 0.25 |

Earlier results with noise from $t = 0$: at $\sigma = 0.1$, $N = 4$, LOO centered, LR SD/$\lvert g\rvert$ for $\log Q$ is 1.32 ($t = 30$) and 2.08 ($t = 75$), against 0.05 and 0.14 for one FD seed; 32 seeds, max $\lvert z\rvert$ 2.3 [measured, `crit_evidence` 5]. At $\sigma = 0.03$, $N = 16$ the estimator agrees with FD (8 seeds, all $\lvert z\rvert < 0.9$) with SD/$\lvert g\rvert$ 9 to 490 at $t = 75$ [measured, `crit_math` 3]. The plain estimator diverges at $\sigma \le 0.03$ ($10^{24}$ to $10^{30}$) [measured, `sec_wake` 3.2].

Damping at $\kappa = 0.3$ instead of 1 changes nothing within SE ($\sigma = 0.1$, $N = 4$). The $t = 45$ deviations at $\sigma = 0.3$, $N = 4$ (four observables at $\lvert z\rvert$ 2.4 to 3.1; $\sigma = 0.1$ shows the same pattern without power) were checked as follows. The FD reference is stable over $h = 0.005$ to 0.04 [measured: `work_note/an_smallh.py`]. The exact pathwise tangent on the same 16 noise paths deviates in the same direction ($\lvert z\rvert$ 1.3 to 1.7), and the paired difference LR minus pathwise is zero within 1.3 SE for all six observables [measured: `work_note/an_pair.py`]. At $N = 16$ (16 seeds) the LR offsets at $t = 45$ shrink to $\lvert z\rvert \le 2.3$, and the LR values match the $N = 4$ pathwise values within 0.1 SE to 0.5 SE; at $t = 75$ all six agree with FD ($\lvert z\rvert \le 1.3$) [measured: `work_note/an_diag.py`, output `work_note/out/an_diag.txt`]. So option C agrees with the exact derivative on its own paths. The residual offset of all pointwise estimators from FD at $t = 45$ is unresolved; it is not specific to option C, and a small $O(1/N)$ effect is not excluded.

**Criterion 3: met against the noisy model** where the comparison has power ($\sigma \ge 0.1$, mostly $\sigma = 0.3$). Against the deterministic model it is met only for $u_n \lesssim 0.07$, where the variance is not usable.

### 4.4 Cost (criterion 4)

1. Work per run, NumPy, $N = 1$, CPU time relative to the primal: exact pathwise tangent 2.36, mean-field LR 3.08, LR with LOO centering 3.92 [measured, `crit_evidence` `timing.py`]. One weight set therefore costs 2.1 to 2.9 primal runs.
2. FBPIC [estimated, `sec_theory` 6, corrected by `crit_math` 4]. Pure weights would cost 0.4 to 0.8 primal runs per parameter, but they fail in the laser. The usable design keeps the live tangent before the window and through the handover, so it costs at least the as-built tangent, $c_1 = 2.47$ primal runs, plus about 0.3 for the weight update and the centering deposit and gather. After the planned dense-cell deposit-tangent rewrite this falls to about 1.3 to 1.6.
3. Repeat factor. For the same SE, the LR needs $(s_{LR}/s_{FD})^2$ times more runs than one CRN central-FD seed: about 16 to 56 at $\sigma = 0.3$ (physics destroyed), 350 to 600 at $\sigma = 0.1$, and $> 10^4$ at $\sigma = 0.03$ (late window, $N = 4$, $t = 75$, Section 4.3 table). LR beats central FD for $K$ parameters only if $\rho^2(1 + c_1K) < 2K$ with $\rho^2$ the repeat factor. For $\rho^2 \ge 16$ this needs $c_1 < 0.12$; for $\rho^2 = 220$ it needs $c_1 < 0.009$ [derived, A.9].

**Criterion 4: not met**, by a factor of 10 to $10^4$.

### 4.5 Summary against the goal

| Criterion | Status | Evidence |
|---|---|---|
| (1) no exponential growth under chaos | Partly. Not tied to trajectory chaos after LOO centering; still exponential at physics-preserving $u_n$; field-level amplification possible | 4.1 |
| (2) cold plasma and thin bunch | Met | 3.2; FD agreement with births |
| (3) matches a trusted reference | Met against the noisy model where powered ($\sigma \ge 0.1$); against the deterministic model only at $u_n \lesssim 0.07$ | 4.2, 4.3 |
| (4) cost per parameter well below one run | Not met (per-run 1.3 to 2.9 runs; repeat factor 16 to $10^4$) | 4.4 |

Limits of the evidence. All of it is 1D or toy, CPU only. The 1D wake has a cycle-averaged driver, a bunch spread $s_u \approx 27$ that is large against Twin-S, and a different injection geometry. Twin-S could have a smaller $s_g/u_n$; Gate B measures that.

## 5. Novelty and prior art

1. **Every ingredient is published [literature].** Per-particle Malliavin or Girsanov weights in interacting particle simulations (Warren and Allen, PRL 109, 250601, 2012). Bismut-type integration by parts for McKean-Vlasov laws (Baños, Ann. IHP 54, 2018; Crisan and McMurray, PTRF 171, 2018). Noise as a regularizer of chaotic linear response, and the path-kernel blend of tangents and LR (Ni, J. Nonlinear Sci. 35, 90, 2025; arXiv:2503.00718). $\delta f$ PIC with weight spreading under Langevin collisions (Brunner, Valeo, Krommes, Phys. Plasmas 6, 4504, 1999). The PIC weight route that needs $\nabla_u f$ (Chung, Bond, Cyr, Freund, JCP 400, 108969, 2020).
2. **Closest probabilistic prior art, missed by `sec_lit` [literature].** Particle approximations of the score and of filter derivatives in sequential Monte Carlo use signed weights in an interacting particle system. The path-space score estimator degenerates in time; a marginalized $O(N^2)$ estimator stays stable (Poyiadjis, Doucet, Singh, Biometrika 98, 2011, abstract checked by `crit_math`). Del Moral, Doucet, Singh prove $L^p$ bounds and an asymptotic variance uniform in time under mixing (arXiv:1106.2525; abstract checked here 2026-11-01). Sharrock, Kantas, Pavliotis (arXiv:2605.00786, 2026; abstract checked here) estimate gradients of the stationary mean-field likelihood with a virtual tangent interacting particle system. That method is pathwise and stationary, not LR and transient.
3. **What was not found.** No paper found builds a Monte Carlo solver for the linearized Vlasov-Fokker-Planck-Maxwell equation from per-particle Girsanov weights closed through a self-consistent field tangent, or uses it against trajectory chaos in PIC. Two searches (`sec_lit`, `crit_math`, 2026-11-01) support this; they were not exhaustive.
4. **Honest claim.** The novelty would be algorithmic and applied: a forward-mode, cold-safe, mean-field Girsanov-weight method for PIC sensitivities with LOO charge-neutral increments. With the present evidence, the publishable content is a characterization with a negative outcome: the method trades trajectory chaos for an information floor and a field-level variance problem. The SMC result (path-accumulated weights degenerate, marginalized weights do not) points to the only known cure, Rao-Blackwellization, which needs a phase-space basis.

## 6. The decisive Twin-S forward-mode experiment

Run only if the user wants a Twin-S-native confirmation. Run in order and stop at the first kill.

**Common settings.** Twin-S, EV with free thresholds, $\hat E_L = 0.588$, $\hat z_f = 0.4$, $\theta = \ln E_L$, $K = 1$. Resolution `p_nr` 8, `p_nt` 8 (the smallest radial count with a converged Jacobian; `p_nt` does not change it, finding 5). Noise on electron species (background and `elec_from_N`) only from step 1520 (laser exit) to step 2000: $T_n = 480 \times 0.0266 = 12.8\,\omega_p^{-1}$, $\sigma = u_n/\sqrt{T_n}$. No spatial mask is needed after laser exit. Reference: the deterministic central secant at $h = 10^{-2}$ from finding 5 (`p_nr` 8), and CRN FD of the noisy model from Gate A.

### 6.1 Gate B: information floor (no kernel, no `fbpic/autodiff` edit)

1. Run the existing tangent to step 1520 with a `driver.jvp` callback (`fbpic/autodiff/driver.py:182`). Log per-particle $\xi_u$ of `elec_from_N` and `electrons` with tracker IDs. Continue the primal to step 2000 and log the selected bunch.
2. For each residual $k$, compute
$$
F_k(u_n) = \frac{1}{\sqrt{N_{eff}}\,\lvert\delta m_k\rvert}\Big[\frac{\sum_iw_i\,\lvert\xi_{u,i}(1520)\rvert^2\,(g_{k,i} - m_k)^2}{u_n^2\sum_iw_i}\Big]^{1/2},
$$
with $g$, $m$, $w$ at step 2000 for the selected markers, Kish $N_{eff}$, and $\delta m_k$ from the `p_nr` 8 secant. Then $u_n^\ast = u_nF_k(u_n)/0.3$ is the smallest noise at which a perfect estimator ($R = 1$) reaches 30% single-run error.
3. **Kill** if $u_n^\ast > 0.07$ for the charge or the mean-energy residual (1D bias limit, Section 4.2). Otherwise continue to Gate A at $u_n^\ast$.
4. Also report the background handover variance $\mathrm{median}\,\lvert\xi_u\rvert^2/u_n^{\ast2}$; it predicts the field-tangent noise.
5. Cost: one tangent run to step 1520 and one primal tail, about 1 GPU-h at `p_nr` 8 [estimated from finding 5 run times]; half a day for the callback and analysis.

### 6.2 Gate A: bias of the noisy model (noise kernel only)

1. Runs: $u_n \in \{u_n^\ast, u_n^\ast/3\}$, 16 seeds, CRN central FD at $h = 10^{-2}$: 64 noisy runs. With restart from step-1520 checkpoints at the three $\theta$ values, each run covers 24% of the steps.
2. **Kill** if, at $u_n^\ast$, the charge residual changes by more than 5% against the deterministic model, or the whitened gradient differs by more than 20% in norm at $z > 2$.
3. Cost: 4 to 5 GPU-h with restarts, 15 to 25 GPU-h without [estimated]. Code: the noise kernel (6.4 item 1).

### 6.3 Main run: the estimator (noise and weight kernels)

Estimator: exact tangent to step 1520; from step 1520, the mixed estimator with $\xi_u$ damped at $\kappa = 1\,\omega_p$ and $\xi_x$ kept pathwise (Section 3.5 item 4); LOO-centered increments; EV birth weights as monopole sources (existing). 16 seeds at `p_nr` 8 and 16 seeds at `p_nr` 16, at $u_n^\ast$.

| Quantity | Pass | Kill |
|---|---|---|
| Weight growth $\omega_{rms}(2000)/\omega_{rms}(1600)$ | $\le 4.5$ | $> 20$ |
| Loop diagnosis: late rate at `p_nr` 16 over `p_nr` 8 | $\le 0.6$ | $> 0.85$ (field-level growth) |
| Agreement of the 11 whitened gradients with Gate A FD | all within 2 SE, power $\ge 3$ on $\ge 4$ residuals including charge | 2 or more beyond 3 SE |
| Single-run error, median over seeds of $\lVert\hat g - \bar g\rVert_W/\lVert\bar g\rVert_W$ | $\le 0.3$ | $> 1$ |
| Inefficiency $R$ = measured variance over the Gate B floor | $\le 10$ | $> 50$ (1D: about 60) |
| Cost per parameter at $K = 4$ times runs for 0.3 error | $< 1$ primal run | $\ge 2$ |

Cost: about 3 primal-run equivalents per LR run; 16 runs at `p_nr` 8 and 16 at `p_nr` 16 come to about 30 GPU-h [estimated from finding 5 run times]. Code: 3 to 5 days.

### 6.4 Code changes (production `fbpic` unchanged)

The weight tangent already exists and is already deposited and contracted: the `"w"` tangent (`fbpic/autodiff/state.py:105`), the `dw` terms of `deposit_tangent` (`fbpic/autodiff/tangent.py:994`, `:1014`), and the observable contraction (`fbpic/autodiff/loss.py:273`). The field tangents, gathers, and EV birth tangents are reused unchanged.

1. **Noise kernel** (new Warp kernel, new module, for example `fbpic/autodiff/girsanov.py`). For each marker of a noisy species, when the step is in the window: draw $\boldsymbol\zeta$ (3 normals) from a counter-based generator keyed by (seed, tracker ID, step), so sorting and exchange do not change the stream; add $\Sigma\sqrt{\Delta t}\,\boldsymbol\zeta$ to $(u_x, u_y, u_z)$; recompute `inv_gamma`. It must run after the Vay push and its tangent (`fbpic/autodiff/stepper.py:362-365`) and before `_push_positions` (`:366`), in runs with and without tangents. This needs one new hook call between `:365` and `:366`; the existing `tangent_vay:after` hook fires only with tangents.
2. **Weight kernel** (same module). Before the Vay tangent (hook `tangent_vay:before`), keep $\xi_u^{n-1/2}$. After it, with $\delta\mathbf m$ in the momentum tangents: $\mathbf H = \delta\mathbf m - (1 - \kappa\Delta t)\,\xi_u^{n-1/2}$; regenerate the same $\boldsymbol\zeta$; add $w\,(\mathbf H\cdot\Sigma^{-1}\boldsymbol\zeta)/\sqrt{\Delta t}$ to the `"w"` tangent; set the momentum tangent to $(1 - \kappa\Delta t)\,\xi_u^{n-1/2}$; recompute the `inv_gamma` tangent at the noisy $\mathbf u$. For $K$ parameters, loop over the tangent sets with the same $\boldsymbol\zeta$.
3. **LOO centering**. Deposit two scalars, $w\,\Delta\omega$ and $w\,\mathbb 1_{on}$, with the existing $\rho$ deposition kernel; gather them back with the same shape and remove each particle's own contribution; subtract the local mean. This needs one new scalar-gather kernel. In the quasi-cylindrical grid, centering on mode 0 removes the axisymmetric charge jump only; measure the mode-1 residual.
4. **Diagnostics** through the `driver.jvp` callback in `inversion/`: $\omega_{rms}$ per species, field-tangent norm per step.
5. Side effect: any change to `fbpic/autodiff/*.py` changes the P7 step-5 cache key (plan 12.10).

## 7. Risks

1. **Field-level amplification.** The two-stream data show estimator noise growing while the true gradient decays. Twin-S has its fastest tangent growth (0.8 $\omega_p$, steps 1500 to 1800) in the density downramp after laser exit [R0]. If that growth is collective, the weights inherit it. The loop-diagnosis row of 6.3 detects it.
2. **Information floor.** With $R \approx 60$ as in 1D, even $F = 0.1$ at $R = 1$ gives a single-run error near 0.8.
3. **Bias through trapping and dephasing.** In 1D, noise after injection still changed the trapped charge by 17% at $u_n = 0.2$ by dephasing and detrapping near the selection threshold [measured]. Twin-S injection may not be complete at step 1520 [estimated]; Gate A measures this.
4. **Handover of position tangents.** Damping $\xi_z$ of a relativistic bunch costs $\gamma^6$; keeping it pathwise moves the cost into $\int\lvert\partial_z\mathbf a\,\xi_z\rvert^2/\sigma^2$, which is unmeasured on Twin-S.
5. **Reference.** The Twin-S Jacobian converges only at `p_nr` $\ge 8$ (finding 5), so every Gate costs 4 to 8 times more per run than at `p_nr` 2. A small $O(1/N)$ closure bias seen in 1D at $N = 4$ is unresolved.

## 8. Critic findings and resolutions

Blocking and major findings first; minor findings last.

| # | Source | Severity | Finding | Resolution |
|---|---|---|---|---|
| M1 | `crit_math` | blocking | Weight growth beyond the loop is not physical sensitivity; the true gradient decays while weights grow | **Accepted.** Section 3.4 rewritten: growth needs a growing field-tangent fluctuation $\boldsymbol\eta$; option C relocates exponential growth to the field level. `sec_theory` 4.4(a) and `sec_lit` 3 "every exact method sees it" withdrawn. Evidence: two-stream FD 0.119 to $-0.013$ while $\omega_{rms}$ 24 to $1.35\cdot10^4$ |
| E1 | `crit_evidence` | blocking | The $N$-independent floor of about 0.23 is residual sampling feedback, not physics | **Accepted.** Checked against both critics' runs and the new late-window runs: centered rates fall about as $1/N$ (Section 4.1). `sec_wake` 3.1 item 3 and conclusion 2 withdrawn |
| E6 | `crit_evidence` | blocking | No $\sigma$ window on the cold wake | **Accepted and extended.** The new late-window design (noise only after injection, mixed estimator, LOO) cuts the charge bias 2 to 4 times but does not open a window: bias small only for $u_n \lesssim 0.07$; at $u_n = 0.2$ single-run error 56 to 150 times the gradient (Sections 4.2, 4.3) |
| M2, E2 | both | major | Toy and wake verdicts omitted LOO centering | **Accepted.** All verdicts in this note use the LOO-centered estimator. The verdict holds for a different reason: variance at physics-preserving $\sigma$ and the information floor, not divergence |
| M3 | `crit_math` | major | The mechanism "$\delta f/f$ resolves fine structure, so weights grow at the chaotic rate" is wrong | **Accepted.** $\chi^2(\delta f\mid f) \le \mathrm{Var}\,\omega = \mathbb E\int\lvert\Sigma^{-1}\mathbf h\rvert^2$; with a bounded $\mathbf h$ this is linear in time (A.3, A.4). The 0.23 fit used 3 points over $t \in [10, 30]$, which mixes injection and loop |
| M4 | `crit_math` | major | Feedback (2.6) fails in a resolved laser | **Accepted.** Section 3.5: $\mathbf h = 0$ in the laser; on Twin-S the window starts at laser exit (step 1520), which removes the quiver term. Cost tables use the live tangent (Section 4.4) |
| M5 | `crit_math` | major | Joint LR omits the birth-weight field term; in FBPIC it needs a Maxwell solve | **Accepted.** Corrected formula in Section 3.1 item 2 and A.5. Conclusion (joint form not a candidate) unchanged |
| M6 | `crit_math` | major | The noise is not a physical scale; $h_{eq}$ has no derivation | **Accepted.** Section 3.3: the noise is a model change that the inversion would fit; its size must be measured (Gate A). $h_{eq}$ withdrawn |
| M7 | `crit_math` | major | Novelty review missed SMC filter-derivative work and Sharrock et al. 2026 | **Accepted.** Section 5 item 2; two abstracts re-checked here |
| E3 | `crit_evidence` | major | Wake bias figures lack SE | **Accepted.** Section 4.2 uses the `crit_evidence` pooled values with SE and new paired values with SE |
| E4 | `crit_evidence` | major | "23 of 24 within 2 SE" has power only at $\sigma \ge 0.3$ | **Accepted.** Section 4.3 reports power; correctness is claimed only where powered |
| E5 | `crit_evidence` | major | Plain local mean biased at 5 per cell; centered rate depends on $\sigma$ after trapping | **Accepted.** LOO form mandatory (Section 3.4). $\sigma$ dependence confirmed by the late-window runs |
| E7 | `crit_evidence` | major | Cost ignores the repeat factor | **Accepted.** Section 4.4 item 3 and A.9 |
| E8 | `crit_evidence` | major | The one-step $\lambda$ test cannot decide; use a bias gate and a variance floor | **Accepted with changes.** Gate B now comes first and needs no kernel; the window starts at step 1520 (laser exit), not 1450; the resolution is `p_nr` 8 and 16 (finding 5), not 1x and 4x |
| m8 | `crit_math` | minor | "Exact given the primal paths" is ill-posed | Accepted: $O(1/N)$ with a time-growing constant (Section 3.1) |
| m9 | `crit_math` | minor | Selection-bias scaling $O(u_n^2/(4\delta u)^2)$ is wrong | Accepted: $\tfrac12u_n^2\int Sf''du$ to leading order (A.8) |
| m10 | `crit_math` | minor | Thermal wave breaking is not far above the wake at $u_n = 0.1$ | Accepted: about $0.76\,\beta_{th}^{-1/2} \approx 2.4\,E_0$ (Katsouleas and Mori, PRL 61, 90, 1988, from memory); irrelevant at $u_n \le 0.07$ |
| m11 | `crit_math` | minor | "Tens of $mc$ of noise" is overstated | Accepted: the relevant ratio is $s_g/(u_n\sqrt{N_{eff}})$ (Section 3.4) |
| m12 | `crit_evidence` | minor | Map LR rms $z = 1.29$ over 30 comparisons ($p \approx 0.01$ if independent) | Accepted as a flag. Seeds are shared across $\sigma$, so the comparisons are correlated; no conclusion depends on the map results |

## Appendix A. Derivations

### A.1 One-step Girsanov weight

The transition $u' = m(\theta) + s\zeta$, $s = \sigma\sqrt{\Delta t}$, has density $p(u';\theta) = \varphi((u' - m)/s)/s$ and score $\partial_\theta\ln p = m'(u' - m)/s^2 = m'\zeta/s$. Hence $\partial_\theta\mathbb E[g(u')] = \mathbb E[g(u')\,m'\zeta/s]$ (Stein's lemma). Over $n$ steps, the tower property sums the scores, so $\omega_n = \sum_k m_k'\zeta_k/s$. With $m' = \delta a\,\Delta t$: $\Delta\omega = \delta a\,\zeta\sqrt{\Delta t}/\sigma$.

### A.2 Mixed identity

Let $\mathbb Q^\epsilon$ have the Girsanov density for the extra drift $\epsilon\mathbf h$. Under $\mathbb Q^\epsilon$, the process driven by $\mathbf b + \epsilon(\delta\mathbf a - \mathbf h)$ has drift $\mathbf b + \epsilon\,\delta\mathbf a$, which is the perturbed model. Differentiate $\mathbb E^{\mathbb Q^\epsilon}[g(Z^\epsilon_T)]$ at $\epsilon = 0$: the density gives $\mathbb E[\omega_Tg]$ with $d\omega = (\Sigma^{-1}\mathbf h)\cdot d\mathbf W$, and the path gives $\mathbb E[\xi_T\cdot\nabla g]$ with $\xi$ driven by $\delta\mathbf a - \mathbf h$. Only $\mathbf h$ at $\epsilon = 0$ enters, so $\mathbf h$ may depend on $\xi$ and on the field tangents. The discrete version replaces the density by A.1 per step. In the 1D harness, $\mathbf H = \delta\mathbf m - (1 - \kappa\Delta t)\xi_u$ per step, $\xi_u \leftarrow (1 - \kappa\Delta t)\xi_u$, $\Delta\omega = \mathbf H\zeta/(\sigma\sqrt{\Delta t})$.

### A.3 Itô isometry and the information floor

$\omega_T = \int_0^T(\Sigma^{-1}\mathbf h)\cdot d\mathbf W$ is a martingale, so $\mathbb E\omega_T^2 = \mathbb E\int_0^T\lvert\Sigma^{-1}\mathbf h\rvert^2dt$. Since $\mathbb E[\omega\mid Z] = \delta f/f$, Jensen gives $\chi^2(\delta f\mid f) = \int(\delta f)^2/f \le \mathbb E\omega^2$. Floor: if the weights must represent a net momentum shift $\xi_u = \int_0^{T_n}\mathbf h\,dt$ of a particle, Cauchy-Schwarz gives $\lvert\xi_u\rvert^2 \le T_n\int\lvert\mathbf h\rvert^2dt$, so $\mathbb E\omega^2 \ge \lvert\xi_u\rvert^2/(\sigma^2T_n) = \lvert\xi_u\rvert^2/u_n^2$. For $\widehat{\delta m} = \sum w\omega(g - m)/\sum w$, $\mathrm{Var} \approx \mathbb E[\omega^2(g - m)^2]/N_{eff}$, which gives the Section 3.4 bound when $\omega^2$ and $(g - m)^2$ are nearly uncorrelated.

### A.4 Why estimator variance can grow while the gradient decays

Write $\delta\mathbf E^N = \delta\mathbf E + \boldsymbol\eta$. Both parts solve the same linear field-tangent system $\mathcal L$. The mean has the coherent source $\delta\rho[\delta f]$. The fluctuation has the sampling source $\delta\rho^N - \mathbb E\,\delta\rho^N$, a white field of variance $\propto\mathrm{Var}\,\omega/n_m$. By A.3, $d\,\mathbb E\omega^2/dt = \mathbb E\lvert\Sigma^{-1}(\delta\mathbf a + q\boldsymbol\eta_\perp)\rvert^2$. If $\mathcal L$ around the saturated primal has a growing direction $e^{\gamma t}$ that the coherent source excites weakly (its projection phase-mixes), $\delta\mathbf E$ stays bounded while $\boldsymbol\eta$ and $\mathbb E\omega^2$ grow as $e^{\gamma t}$. The single-run error then scales as $e^{\gamma T}/\sqrt N$, the same form as the averaged Lagrangian tangent with a field-level rate.

### A.5 Joint likelihood ratio with births

The $N$-particle path density factors into Gaussian transitions given the full state. At fixed paths, $\theta$ enters the drift through the laser field tangent and through the fields of the birth weights $W_b(\theta)$, whose deposits depend on $\theta$ at fixed positions. So
$$
\Omega = \sum_i\sum_n(\Sigma^{-1}\delta\mathbf a_i^{expl,n})\cdot\boldsymbol\zeta_i^n\sqrt{\Delta t},\qquad
\frac{d\,\mathbb E_N[J]}{d\theta} = \mathbb E\big[(J - \bar J)\,\Omega\big] + \mathbb E\Big[\sum_b\partial_{W_b}J\,\delta W_b\Big],
$$
where $\delta\mathbf a^{expl}$ uses the field tangent sourced only by the initial laser tangent and by $\delta W_b$ at fixed paths, and $\bar J$ is a leave-one-out replica mean. $\mathrm{Var}\,\Omega \propto N$ and $\mathrm{Var}\,J \propto 1/N$, so the single-run variance is $O(1)$ in $N$.

### A.6 LOO centering keeps the expectation

The centered increment is $\Delta\tilde\omega_i = \Delta\omega_i - \sum_{j\ne i}c_{ij}\Delta\omega_j$, with $c_{ij}$ from the shapes and weights at step $n$ (adapted). Each $\Delta\omega_j$ has zero conditional mean and is independent of particle $i$'s own kick. Its correlation with $g(Z_i(T))$ arises only through the fields, which is $O(1/N)$ per neighbor. With $\sum_jc_{ij} = 1$, the total bias is $O(1/N)$. With the self term included, $c_{ii} \approx 1/N_{cell}$ and the increment correlates with its own kick, so the estimate shrinks by about $1/N_{cell}$; measured 20 to 40% at 5 per cell [`crit_evidence` 4].

### A.7 Handover cost

For free streaming in one component, $\dot\xi_x = M\xi_u$, $\dot\xi_u = -h$, driving $(\xi_x, \xi_u)$ to zero over $\tau$ with minimum $\int h^2dt$ uses the controllability Gramian $W_c(\tau)$. The minimum is $\lvert\xi_u\rvert^2/\tau$ for a pure momentum tangent and $12\lvert\xi_x\rvert^2/(M^2\tau^3)$ for a pure position tangent; divide by $\sigma^2$ for $\mathrm{Var}\,\omega$ [derived, `sec_theory` 2.3].

### A.8 Selection bias of an observation smoothing

For $\eta \sim N(0, u_n^2)$, $\mathbb E[S(u + \eta)] - \mathbb E[S(u)] = \tfrac12u_n^2\int S''f\,du + O(u_n^4) = \tfrac12u_n^2\int Sf''\,du + O(u_n^4)$ after two integrations by parts. It depends on the curvature of $f$ at the threshold, not on the width of $S$.

### A.9 Cost crossover with the repeat factor

Let $s_{LR}$ and $s_{FD}$ be the single-run SDs of the LR estimate and of one CRN central-FD seed, $\rho^2 = (s_{LR}/s_{FD})^2$. For equal SE on $K$ parameters, LR costs $M\rho^2(1 + c_1K)$ primal runs and central FD costs $2MK$. LR is cheaper iff $\rho^2(1 + c_1K) < 2K$, which requires $c_1 < 2/\rho^2$ and $K > \rho^2/(2 - \rho^2c_1)$.

## Appendix B. New runs of this note

All in `lr/work_note/`, CPU NumPy, `python -I`.

| File | Content |
|---|---|
| `wake_win.py` | copy of `work_crit_evidence/wake_lr_c.py` with `T_ON` (noise only for $t \ge T_{ON}$) and `MIXK` (mixed estimator, $\xi_u$ damped at rate $\kappa$) |
| `fd_win.py` | CRN FD driver for `wake_win.py` |
| `jobs1.txt` to `jobs4.txt` | run lists: FD at $\sigma \in \{0, 0.01, 0.03, 0.1, 0.3\}$, $N \in \{1, 4\}$; LR at $\sigma \in \{0.03, 0.1, 0.3\}$, $N \in \{1, 4\}$, $\kappa = 1$; LR at $\kappa = 0.3$ ($\sigma = 0.1$, $N = 4$); LR at $N = 16$ and pathwise and plain controls ($\sigma = 0.3$); FD at $h = 0.005, 0.01$ ($\sigma = 0.3$, $N = 4$) |
| `an_win.py` -> `out/an_win.txt` | bias, agreement, single-run noise, growth rates |
| `an_smallh.py` -> `out/an_smallh.txt` | FD secants at $h = 0.005$ to 0.04 for the $t = 45$ check |
| `an_pair.py` -> `out/an_pair.txt` | paired LR minus pathwise on shared seeds |
| `an_diag.py` | pathwise, plain and centered LR against FD at $\sigma = 0.3$ |
| `out/fd/`, `out/lr/` | raw JSON outputs |
