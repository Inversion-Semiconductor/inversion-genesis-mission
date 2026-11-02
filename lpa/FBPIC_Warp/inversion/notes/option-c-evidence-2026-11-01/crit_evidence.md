# Critic: evidence and feasibility of option C

Lens: evidence and feasibility. Work directory: `inversion/notes/option-c-evidence-2026-11-01/work_crit_evidence/` (written `wc/` below). All runs are CPU NumPy, `python -I`, single thread, on a shared node at load 30 to 50.

Labels: [measured: script], [derived], [estimated], [checked] (I read the code or recomputed from the authors' own outputs).

## 0. Verdict

The negative verdict on option C stands, but two of its stated reasons are wrong. Neither the wake nor the toy evaluator tested the theory section's main variance control, the charge-neutral (centered) weight increment. I added it to a copy of the wake harness and of the two-stream toy.

1. **Centering changes the mechanism, not the outcome.** With leave-one-out (LOO) centering, the "N-independent floor" of about 0.23 in the wake disappears: the late growth rate at $\sigma = 0.03$, $N = 16$ falls from 0.223 to 0.034 [measured: `wc/wake_lr_c.py`]. So the claim that the weights inherit the chaotic rate as $\sigma \to 0$ is not supported. The remaining problem is the size of the weights, not their exponential growth.
2. **There is still no usable $\sigma$ window on the cold wake.** Gradient bias of the leading observables is small only for $\sigma \le 0.01$ at $t = 75$ (and $\sigma \le 0.03$ at $t = 30$). Bounded centered weights need $\sigma \ge 0.1$ at $N = 4$. Even there, one run gives SD/$|g|$ = 1.3 to 2.1 for $\log Q$, against 0.05 to 0.14 for one CRN FD seed [measured]. That is 200 to 700 times more runs for the same standard error.
3. **The bias claims of the wake section are overstated.** At $\sigma = 0.01$, $t = 75$ the gradient bias of $\log Q$, $\langle u\rangle$, $\langle\xi\rangle$ is $5 \pm 6$, $20 \pm 43$, $5 \pm 17$ percent with 32 to 38 seeds, not significant [measured: `wc/bias.py`]. The 12 to 92 percent figures in `sec_wake.md` have no SE and are mostly seed noise. Trapped charge does fall by 13 percent at $\sigma = 0.01$ (significant).
4. **The "23 of 24 within 2 SE" claim is weaker than it reads.** Only 14 of the 24 comparisons can detect a 100 percent error ($|g|/\mathrm{SE}_{\rm comb} \ge 3$). All 14 are at $\sigma \ge 0.3$, where the physics is destroyed. Correctness is therefore shown only where it does not matter. My LOO-centered runs extend it to $\sigma = 0.1$ (32 seeds, all $|z| \le 2.3$).
5. **Cost (goal 4) is not met, by two independent margins.** The per-parameter work is 2.1 (plain) to 2.9 (centered) primal runs in NumPy [measured], and of order one run in FBPIC once the feedback control keeps $\xi$ [estimated, theory Section 6]. The variance penalty above multiplies this by about 200.

**Decision experiment (Section 7).** One Twin-S run set decides the method: a late-window, LOO-centered mixed estimator for $\theta = \ln E_L$ at $u_n = 0.03$, 16 seeds at 1x and 4x particles, against CRN FD of the same noisy model. Two zero-implementation gates come first and are likely to kill it: a bias gate (noisy primal plus FD only) and a Cramér-Rao-type variance floor computed from one primal run.

## 1. What I checked

| Claim | Check | Result |
|---|---|---|
| CRN is exact in the wake harness | Read `work_wake/wake_lr.py`: noise is `default_rng([seed, n])` over a $\theta$-independent slot range; births follow a $\theta$-independent schedule | [checked] correct |
| Plain growth rates (wake 3.1) | Reran seed 1 with the same code | [measured] identical: 2.945, 1.444, 0.944, 0.414 for $(\sigma, N) = (0.01, 1), (0.01, 4), (0.03, 1), (0.03, 4)$ |
| Reference is the noisy-model gradient | Cubic fit and secants at $h = 0.02, 0.04$ agree within 1 SE for $\sigma \ge 0.1$ at $t = 30$ | [measured] yes for $\sigma \ge 0.1$; at $\sigma = 0$, $t = 75$ the $h = 0.02$ and $0.04$ secants differ by 14 percent ($\log Q$) to 104 percent ($\langle u\rangle$) |
| LR and FD seeds independent | LR seeds 1001+, FD seeds 1 to 17 | [checked] yes |
| Bias table of `sec_wake.md` 3.3 | Compared with `work_wake/out/tables_t75.0.md` | [checked] text and regenerated table differ (for example $\log\sigma_\xi$ at $\sigma = 0.01$: 0.69 in text, 0.41 in table); neither has SE |
| Map LR agreement (toy 2.2) | Recomputed from the authors' $z$ values | [checked] rms $z = 1.29$ over 30 comparisons gives $\chi^2 = 50$ on 30 degrees of freedom ($p \approx 0.01$ if independent). Seed sharing may explain it. Mild flag only |
| Shot-noise $1/N$ law (toy 3.2) | Ratios of the authors' $\lambda$ | [checked] 3.5 and 4.2 per factor 4 in $N$; consistent |
| Weight scaling without feedback | Wake table 3.1: $\sigma\,\omega_{\rm rms} = 0.5$ to $0.8$ over two decades | [checked] consistent with $\omega \propto \sigma^{-1}$ |

Nothing in the authors' scripts is tuned to the answer. The one reference choice made after seeing data is the toy's FD step ($h = 0.3$ to $h = 0.1$ for the long run), and it is justified by a 13 SE curvature effect [checked].

## 2. Power of the wake agreement test [checked]

From `sec_wake.md` Section 3.2, a comparison can detect an error of 100 percent of $|g|$ only if $|g|/\mathrm{SE}_{\rm comb} \ge 3$.

| Case | Powered comparisons | Their $z$ |
|---|---|---|
| $\sigma = 0.3$, $N = 4$, $t = 30$ | $\log Q$, $\langle u\rangle$, $\log\sigma_u$ | 0.1, 1.2, 0.9 |
| $\sigma = 1$, $N = 4$, $t = 30$ | all 6 | 0.2 to 0.3 |
| $\sigma = 0.3$, $N = 4$, $t = 75$ | $\log\sigma_u$, $\log\sigma_u/\langle u\rangle$ | 0.2, 0.8 |
| $\sigma = 1$, $N = 4$, $t = 75$ | $\log Q$, $\log\sigma_u$ | 0.4, 0.5 |

So 14 of 24 comparisons have power, all pass, and all are at $\sigma \ge 0.3$. The $\sigma = 1$, $t = 75$ rows compare $N = 4$ LR with $N = 1$ FD, so the $O(1/N)$ closure bias is not tested there.

## 3. Centered increments in the cold wake [measured: `wc/wake_lr_c.py`, `wc/show_grow.py`]

I added theory eq. (4.5) to a copy of the wake harness: the weight increment of each noisy electron is reduced by the CIC-weighted local mean of the increments. `CENTER=1` uses the plain local mean. `CENTER=2` uses a leave-one-out mean (the particle's own contribution is removed from each node before the gather). Everything else is unchanged.

Plasma $\omega_{\rm rms}$, seed 1, and fitted growth rate of $\ln\omega_{\rm rms}$ over $t \in [30, 75]$:

| $\sigma$ | $N$ | plain: $\omega(30)$, $\omega(75)$, rate | LOO centered: $\omega(30)$, $\omega(75)$, rate |
|---|---|---|---|
| 0.01 | 1 | $2\times10^{33}$, $2\times10^{93}$, 3.22 | $2.4\times10^{7}$, $3\times10^{25}$, 0.94 |
| 0.01 | 4 | $1.9\times10^{15}$, $9\times10^{46}$, 1.59 | $1.5\times10^{4}$, $1.1\times10^{10}$, 0.31 |
| 0.03 | 1 | $1.3\times10^{10}$, $1.6\times10^{32}$, 1.10 | $3.7\times10^{3}$, $1.2\times10^{9}$, 0.27 |
| 0.03 | 4 | $6.7\times10^{4}$, $6.7\times10^{13}$, 0.46 | 507, $2.2\times10^{4}$, 0.083 |
| 0.03 | 16 | $7.6\times10^{3}$, $5.3\times10^{8}$, 0.22 | 298, $1.5\times10^{3}$, 0.034 |
| 0.1 | 1 | $1.9\times10^{3}$, $1.3\times10^{9}$, 0.29 | 82, $1.6\times10^{3}$, 0.066 |
| 0.1 | 4 | 41, $6.5\times10^{4}$, 0.15 | 34, 71, 0.016 |
| 0.3 | 1 | 9.3, 105, 0.057 | 6.7, 11.6, 0.012 |
| 0.3 | 4 | 5.1, 18.7, 0.032 | 4.7, 6.2, 0.006 |

Findings:

1. Centering cuts $\omega(75)$ by a factor of 3 (at $\sigma = 0.3$) up to $10^{68}$ (at $\sigma = 0.01$). At $\sigma = 0.1$, $N = 4$ the weights are now bounded (71 against $6.5\times10^4$).
2. **The plain "floor" is not physics.** `sec_wake.md` 3.1 reads the $N$-independent part of the plain rate (0.23 at $\sigma = 0.03$, 0.1 at $\sigma = 0.1$) as growth of the true linear response that approaches the chaotic rate. Both estimators have the same expectation, so a true growing response would grow both. The centered rate at $\sigma = 0.03$ falls as $0.27, 0.083, 0.034$ for $N = 1, 4, 16$, roughly $1/N$. The floor is residual sampling feedback.
3. **The theory's claim that centering makes the rate "nearly independent of $\sigma$" fails after trapping.** At $N = 4$ the centered late rate is 0.006, 0.016, 0.083, 0.31 for $\sigma = 0.3, 0.1, 0.03, 0.01$. The theory section flagged this risk (multi-stream $v - \bar v$).
4. The early phase ($t = 10$ to 30, injection and trapping) grows at 0.15 to 0.17 for both $N = 4$ and $N = 16$ at $\sigma = 0.03$. This part is not reduced by $N$, and it sets $\omega(30) \approx 300$ to 500. It matches the size expected from the response itself (Section 6).

Two-stream toy, single seed, $N = 131072$, $t = 60$ [measured: `wc/es1d_lr_c.py`, `wc/out/es/diag_*.log`]: plain $\omega_{\rm rms} = 3.7\times10^4$ (late rate 0.19), centered 367 (late rate 0.07), exact tangent $1.7\times10^3$ (0.17). At $\sigma = 0.01$ the centered rate stays at 0.33, above the tangent. The toy section's mean-field verdict was therefore also reached without the best variant, but the outcome at small $\sigma$ is the same.

## 4. Centering needs the leave-one-out form [measured]

At 5 plasma markers per cell, the plain local mean (theory eq. 4.5 as written) is biased. With 32 seeds at $\sigma = 0.3$, $N = 1$, against CRN FD (cubic fit):

| Variant | $\log Q$ at $t = 30$ | $z$ | $\log Q$ at $t = 75$ | $z$ |
|---|---|---|---|---|
| FD reference | $1.64 \pm 0.03$ | | $1.27 \pm 0.04$ | |
| `CENTER=1` (plain mean) | $0.96 \pm 0.17$ | $-4.0$ | $1.02 \pm 0.10$ | $-2.4$ |
| `CENTER=2` (LOO mean) | $1.53 \pm 0.19$ | $-0.6$ | $1.16 \pm 0.11$ | $-1.0$ |

The plain mean removes 20 to 40 percent of the signal, more than the $1/N_{\rm cell}$ the theory estimates. On Twin-S (16 per cell) use the LOO form.

## 5. Agreement and single-run noise with LOO centering [measured: `wc/an.py`]

32 LR seeds per case; FD = CRN FD of the same noisy model, seeds pooled with the authors' runs and de-duplicated (13 to 16 seeds).

| Case | $t$ | max $\lvert z\rvert$ over 6 observables | powered comparisons | LR SD/$\lvert g\rvert$: $\log Q$, $\langle u\rangle$, $\langle\xi\rangle$ | FD per seed SD/$\lvert g\rvert$: same |
|---|---|---|---|---|---|
| $\sigma = 0.3$, $N = 1$ | 30 | 0.8 | 4 | 0.64, 0.63, 0.77 | 0.07, 0.18, 0.24 |
| $\sigma = 0.3$, $N = 1$ | 75 | 1.4 | 4 | 0.49, 1.75, 2.09 | 0.13, 0.19, 0.54 |
| $\sigma = 0.1$, $N = 4$ | 30 | 2.3 ($\log\sigma_u$, power 1.9) | 3 | 1.32, 1.22, 0.92 | 0.05, 0.07, 0.09 |
| $\sigma = 0.1$, $N = 4$ | 75 | 0.9 | 0 | 2.08, 3.84, 3.95 | 0.14, 0.42, 0.92 |
| $\sigma = 0.03$, $N = 4$ | 75 | diverged ($\omega_{\rm pl} = 2.6\times10^4$) | | | |

The estimator is unbiased wherever it is stable, now including $\sigma = 0.1$. Its single-run noise is 15 to 27 times that of one CRN FD seed for $\log Q$. For the same standard error that is about $(2.08/0.14)^2 \approx 220$ times more runs at $t = 75$. With a per-parameter cost $c$ per run, LR beats central FD for $K$ parameters only if $220(1 + cK) < 2K$, which no $K$ satisfies for $c > 0.01$ [derived from measured inputs].

## 6. Is there a usable $\sigma$ window on the cold wake? [measured: `wc/bias.py`; derived]

Gradient bias against the deterministic model, CRN FD at $h = 0.04$, $N = 1$, 32 to 38 seeds pooled (relative difference $\pm$ SE, and $z$):

| $\sigma$ | $t$ | $\log Q$ | $\langle u\rangle$ | $\langle\xi\rangle$ | $\log\sigma_u$ | trapped charge |
|---|---|---|---|---|---|---|
| 0.01 | 30 | $0.01 \pm 0.01$ | $0.01 \pm 0.02$ | $0.00 \pm 0.01$ | $0.05 \pm 0.15$ | $-0.7$ % |
| 0.03 | 30 | $0.00 \pm 0.01$ | $0.03 \pm 0.02$ | $0.00 \pm 0.01$ | $0.32 \pm 0.16$ | $-8$ % |
| 0.1 | 30 | $0.23 \pm 0.02$ ($z$ 13) | $0.15 \pm 0.02$ | $0.07 \pm 0.04$ | $0.73 \pm 0.12$ | $-40$ % |
| 0.01 | 75 | $0.05 \pm 0.06$ | $0.20 \pm 0.43$ | $0.05 \pm 0.17$ | $0.43 \pm 0.12$ ($z$ 3.5) | $-13$ % |
| 0.03 | 75 | $0.21 \pm 0.05$ ($z$ 3.9) | $1.2 \pm 0.45$ | $0.25 \pm 0.17$ | $0.65 \pm 0.18$ | $-28$ % |
| 0.1 | 75 | $0.27 \pm 0.04$ | $2.6 \pm 0.3$ | $0.76 \pm 0.11$ | $0.29 \pm 0.05$ | $-55$ % |

The deterministic reference itself moves by $0.14 \pm 0.06$ ($\log Q$) and $1.0 \pm 0.4$ ($\langle u\rangle$) between $h = 0.02$ and $0.04$ at $t = 75$, so $\langle u\rangle$ and the spreads at $t = 75$ cannot anchor a bias target.

Window. Bias of the leading gradients ($\log Q$, $\langle\xi\rangle$) is at most about 10 percent for $\sigma \le 0.01$ at $t = 75$ and $\sigma \le 0.03$ at $t = 30$. Bounded centered weights need $\sigma \ge 0.1$ at $N = 4$ and give 130 to 210 percent single-run noise there. The gap is a factor of 3 to 10 in $\sigma$, at a single-run noise 15 to 27 times that of one FD seed. **No window exists at $N \le 16$** [measured].

Why more particles do not close it [derived, estimated inputs]. Even with no feedback at all, an LR weight that represents a momentum shift $\xi_u$ under noise spread $u_n$ has $\mathrm{Var}\,\omega \ge \xi_u^2/u_n^2$ (theory 4.1). For a centroid this gives $\mathrm{SE}/|\delta m| \ge s_g/(u_n\sqrt{N_{\rm eff}})$. At $\sigma = 0.01$, $t = 75$: $s_u \approx 27$, $u_n \approx 0.01\sqrt{55} = 0.074$, so 15 percent noise needs $N_{\rm eff} \ge (27/(0.074 \cdot 0.15))^2 \approx 6\times10^6$ trapped markers, against about $8\times10^3$ at $N = 4$. Measured weights are well above the floor: at $\sigma = 0.1$, $N = 4$ the trapped $\omega_{\rm rms} = 183$ against a floor of about $17.5/0.74 \approx 24$, so the conversion inefficiency is $R \approx 60$ [estimated].

## 7. Cost [measured: `wc/timing.py`, `wc/out/timing.log`; estimated]

CPU time to $t = 75$, $N = 1$, $\sigma = 0.3$, 3 repeats on the loaded node, joint score off: primal 1.00, exact pathwise tangent 2.36, mean-field LR 3.08, LR with LOO centering 3.92. So in NumPy one LR parameter costs 2.1 to 2.9 primal runs, more than the exact tangent (1.4). The authors' range (1.4 to 2.4) is consistent.

In FBPIC, pure weights would need one scalar deposit of $w\omega$ for $\rho$ and $\mathbf J$, one field-tangent solve, one gather, and one centering deposit and gather: about 0.4 to 0.8 primal [estimated, theory 6]. But the laser quiver forces the feedback control (2.6) with a live $\xi$, so the credible variant costs 1.0 to 2.5 primal per parameter [estimated, theory 6]. The repeat factor of 200 to 700 (Section 5) dominates either number. **Goal (4) is not met.**

## 8. The single Twin-S forward-mode experiment that decides option C

Setup (one experiment, run in order; stop at the first kill).

1. **Gate A, bias (primal only).** Twin-S, 4x particles, momentum noise on all electrons in the window from step 1450 to 2000 ($T_n = 14.6\,\omega_p^{-1}$), $u_n \in \{0.01, 0.03\}$, counter-based RNG keyed by particle id and step. Compute CRN central FD of the 11 whitened residuals at $h = 10^{-2}$ over 16 seeds, and deterministic FD at $h = 10^{-2}$ and $3\times10^{-3}$.
   - Kill if, at both $u_n$, the whitened gradient differs from the deterministic one by more than 20 percent in norm with $z > 2$, or the charge residual changes by more than 5 percent.
2. **Gate B, variance floor (one primal run, no new code).** For each residual $k$, compute $F_k = s_{g_k}\,|\xi_u^{\rm net}|/(u_n\sqrt{N_{\rm eff}}\,|\delta m_k|)$. Take $s_{g_k}$ and the Kish $N_{\rm eff}$ from the selected bunch at step 2000. Take $|\xi_u^{\rm net}|$ from the exact tangent at step 1450 and $\delta m_k$ from Gate A.
   - Kill if $F_k > 0.3$ for the charge or the mean-energy residual. Then even a perfect estimator ($R = 1$) gives more than 30 percent single-run error.
3. **Main run.** $\theta = \ln E_L$, $K = 1$. Exact tangent to step 1450. In the window, the mixed estimator (theory 2.4) with feedback control (2.6, $\omega_c = 1\,\omega_p$) on all electrons, LOO-centered increments, and EV birth weights as monopole sources. 16 seeds at 1x and 16 at 4x. Reference: the Gate A CRN FD of the same noisy model at the same $u_n$.

Numeric criteria for the main run.

| Quantity | Pass | Kill |
|---|---|---|
| Weight growth over the window, $\omega_{\rm rms}(2000)/\omega_{\rm rms}(1500)$ | $\le 4.5$ ($e^{1.5}$) | $> 20$ ($\Gamma T_n > 3$, theory's own kill) |
| Loop diagnosis, late rate at 4x over rate at 1x | $\le 0.4$ (feedback, fixable by $N$) | $> 0.7$ (physical or collective growth) |
| Agreement, 11 whitened residual gradients | all within 2 SE$_{\rm comb}$, and power $\ge 3$ on at least 4 residuals including charge | 2 or more beyond 3 SE |
| Single-run error, median over seeds of $\lVert\hat g - \bar g\rVert_W/\lVert\bar g\rVert_W$ | $\le 0.3$ | $> 1$ |
| Inefficiency $R$ = measured $\mathrm{Var}\,\omega$ over the Gate B floor | $\le 10$ | $> 50$ (the 1D wake value is about 60) |
| Cost per parameter at $K = 4$, times runs needed for 0.3 error | $< 1$ primal | $\ge 2$ (loses to central CRN FD) |

Prediction from the 1D evidence [estimated]: the method fails at the single-run error row, or earlier at Gate B, because $s_g/u_n$ is large for any $u_n$ that passes Gate A. Gates A and B cost about 50 primal runs and no new derivative code, so they should run before anyone implements the feedback control.

## 9. Corrections to the section files

1. `sec_wake.md` 3.1 finding 3 and conclusion 2: the $N$-independent floor is residual sampling feedback, not the physical response. LOO centering removes it (Section 3).
2. `sec_wake.md` 3.3: the bias figures need SE. At $\sigma = 0.01$ only $\log\sigma_u$ and $\log\sigma_\xi$ biases are significant at $t = 75$. The text table does not match `work_wake/out/tables_t75.0.md`.
3. `sec_wake.md` and `sec_toy.md` verdicts: both omit centered increments. The verdicts survive, for the reasons in Sections 5 and 6, not for the stated mechanism.
4. `sec_theory.md` 4.4: the plain local mean (4.5) is biased by 20 to 40 percent at 5 markers per cell; use the LOO form. The "nearly $\sigma$-independent" centered rate does not hold after trapping or orbit crossing.
5. `sec_wake.md` 3.2: report the power of each comparison. Correctness is established only where $|g|/\mathrm{SE}_{\rm comb} \ge 3$.

## 10. Files

- `wc/wake_lr_c.py`: copy of `work_wake/wake_lr.py` with `CENTER` = 0, 1, 2 (plain, local mean, LOO local mean).
- `wc/fd_batch_c.py`: CRN FD driver for the copy.
- `wc/es1d_lr_c.py`: copy of `work_toy/es1d_lr.py` with `ES_CENTER`.
- `wc/show_grow.py`, `wc/an.py`, `wc/bias.py`, `wc/timing.py`: analysis and timing.
- `wc/out/grow/`, `wc/out/grow16/`, `wc/out/lr/`, `wc/out/fd/`, `wc/out/es/`, `wc/out/timing.log`: raw outputs.
