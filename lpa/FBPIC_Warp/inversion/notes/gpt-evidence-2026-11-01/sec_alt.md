# sec_alt: other ways to differentiate a chaotic cold particle system macroscopically (forward mode)

Task key: `alt`. Date: 2026-11-01. Scope: forward mode only, cold plasma, many parameters.

Labels: **[derived]** derived here; **[measured]** measured on CPU with the named script (all scripts in `mt/work_alt/`, full path `inversion/notes/gpt-evidence-2026-11-01/work_alt/`); **[R0]** the R0 design note `inversion/notes/r0-design-note-2026-10-31.md` and its evidence; **[lit]** literature, with the citation and whether I checked it; **[estimated]** an estimate with stated inputs.

## 0. Result in brief

1. Three alternatives were worked out and two were tested in 1D:
   - **A. NLR, noise-injected likelihood ratio.** Add a small known Gaussian momentum noise to the bunch. The score of that noise is exact, so the momentum kicks become weight tangents without $\nabla_u f$. Works on cold plasma by construction, costs one scalar per particle per parameter.
   - **B. LIBP, label-space integration by parts on the cold sheet.** Split $\xi$ into a part along the sheet and a part normal to it. Integrate the along-sheet part by parts in label space. It is an exact identity, it is native to cold plasma, and it needs second-order tangents.
   - **C. Regression, Koopman, and moment-closure tangents.** Rejected by derivation: each one is the Galerkin weight conversion of R0 (A.4) at every step, so it inherits the cold-limit failure.
2. Measured on a cold beam in prescribed chaotic waves (T1), against a lattice-converged truth:
   - NLR is unbiased for the noisy system (it matches common-random-number FD of the noisy system within 2.3 combined SE in all 12 $(T, \sigma)$ cases). Its bias against the noiseless system is large and has no plateau: at $T = 10$, $\sigma = 0.01$ gives $-20\%$ to $-31\%$, $\sigma = 0.03$ gives $-42\%$ to $-64\%$.
   - LIBP is exact (lattice sums agree with the Lagrangian sum to $10^{-6}$). In its raw form its per-particle summand is heavy tailed at compressed, sharply curved points of the sheet. A hybrid that applies LIBP only where the sheet is stretched ($\lvert t\rvert > 30$ to 100) cuts the per-particle SD by 7 to 8 times at $T = 15$ and $T = 20$ (variance by 50 to 60 times) and is unbiased at $T = 10$ and $T = 15$.
3. Main limit, shared by every method here and by GPT: on an unmixed cold sheet there is no exact, cheap, bounded macroscopic tangent. Exact methods (LIBP) give the pointwise continuum derivative. That derivative is useful only where the continuum response is smooth at the optimizer's step. In T1 it is smooth at $T = 15$ only for $h \le 0.01$ and is not useful at $T = 20$. Methods that give a regular large-scale slope (NLR, finite-basis GPT, FD) are coarse-grained at a scale that sets the answer.
4. Comparison with GPT [derived]: on a cold sheet the GPT gauge is small. In 1D-1V on a closed sheet it is one-dimensional (a uniform label shift). The exploding along-sheet part of $\xi$ is mostly not gauge, so exact GPT does not bound $\xi$; finite-basis GPT regularizes only by averaging across folds (coarse-graining). LIBP is the exact cold-sheet counterpart of what GPT tries to do; GPT is the right tool for the warm or phase-mixed part.
5. Recommendation: none of the three beats FD for Twin-S multi-parameter work today. LIBP-hybrid is the only exact candidate and the most novel; build it further only if R0 Q1 shows that the Twin-S roughness falls as $N^{-1/2}$. NLR is the cheapest cold-proof regularizer, but in T1 its bias is uncontrolled and its variance is larger than FD of the same noisy model at small $\sigma$.

## 1. Frame: what a macroscopic tangent must represent

For $A(\theta) = \int\phi\,d\mu_T$ with $\mu_T = \Phi_{\#}\mu_0$ and the per-particle tangent $\xi = \partial_\theta z$:

$$
\delta A = \int \nabla\phi\cdot\xi\,d\mu_T, \qquad \delta\mu_T = -\nabla_z\cdot(\mu_T\,\xi) \quad\text{(weak form)}.
$$

Only $\bar\xi(z) = E_{\mu}[\xi\mid z]$ enters $\delta\mu$; any $\eta$ with $\nabla\cdot(\mu\eta) = 0$ is a gauge [derived].

**Warm against cold** [derived].

1. Warm $f$ (smooth density): many labels pass through each phase-space point. $\xi - \bar\xi$ is gauge. If the true $\delta f$ is bounded, $\bar\xi$ has a bounded gradient part. The Lagrangian explosion is then mostly in $\xi - \bar\xi$ and in the divergence-free part of $\bar\xi$. GPT's premise holds here.
2. Cold sheet (the Twin-S background and the EV bunch before mixing): one label per phase-space point, so $\bar\xi = \xi$; nothing averages exactly. Let the sheet be $z(a, t)$ with labels $a \in \mathbb R^d$, label measure $w(a)\,da$, and $T = \partial z/\partial a$ ($2n\times d$). Testing $\nabla\cdot(\mu\eta) = 0$ with all smooth $\psi$ gives: $\eta$ is tangent to the sheet, $\eta = T\tau$, and $\nabla_a\cdot(w\tau) = 0$. So the gauge is the set of label-space divergence-free relabelings. In 1D-1V ($d = 1$) on a closed sheet it is one-dimensional: $\tau = c/w$.
3. The exploding part on a cold sheet is the along-sheet displacement $T\tau$ with $\lvert T\rvert \sim e^{\lambda t}$ and $\tau = O(1)$ in label units (Section 3.2). Its label divergence is the physical along-sheet density change. It is not gauge.

**Consequence for GPT** [derived]. The $L^2(\mu)$ projection onto gradient fields removes only the gauge. On a 1D-1V cold sheet with an exact (complete) basis it changes $\xi$ by a uniform label shift, that is, almost nothing. In Twin-S ($d = 3$ for the background, $d = 4$ for the EV manifold) it removes the label-divergence-free part of $\tau$; no argument shows that this part carries the explosion. A finite smooth basis cannot resolve the folded sheet. It averages adjacent folds, which is a coarse-graining with a scale set by the basis. The doubling-map check in the brief is a warm case ($\mu$ uniform on the circle, $\xi$ exactly gauge) and does not test this.

**Which representation for which direction** [derived].

| Direction on a cold sheet | Displacement $\xi$ | Weight (score) |
|---|---|---|
| Normal to the sheet | bounded (symplectic contraction, Section 3.2) | impossible: needs $\nabla_u f$ across a delta (R0 A.2) |
| Along the sheet | explodes as $\lvert T\rvert$ | exact by label IBP: the density along the sheet is the smooth label measure |

Method B uses this table directly. Method A changes the problem so that weights are legal in every direction. Method C uses weights everywhere and fails in the normal direction.

## 2. Method A: NLR, noise-injected likelihood ratio

### 2.1 Equations [derived]

Add a known Gaussian kick to the momentum update of the noisy species (the EV bunch; optionally all electrons), with counter-based noise keyed on particle id and step:

$$
\mathbf u_i^{n+1} = \boldsymbol\mu_i^n(z^n;\theta) + \sigma\sqrt{\Delta t}\,\boldsymbol\varepsilon_i^n,\qquad \boldsymbol\varepsilon_i^n\sim N(0, I).
$$

The transition density is Gaussian with a known score, so

$$
\partial_\theta\log p(\mathbf u_i^{n+1}\mid z^n) = \frac{\boldsymbol\varepsilon_i^n\cdot\delta\boldsymbol\mu_i^n}{\sigma\sqrt{\Delta t}}.
$$

Forward-mode update per noisy particle and parameter:

$$
\dot W_i \mathrel{+}= W_i\,\frac{\boldsymbol\varepsilon_i^n\cdot\delta\boldsymbol\mu_i^n}{\sigma\sqrt{\Delta t}},\qquad \delta\mathbf u_i^{n+1} = 0,\qquad
\delta\mathbf x_i^{n+1} = \delta\mathbf x_i^n + \Delta t\,\partial_\theta\mathbf v\big|_{\mathbf u^{n+1}}.
$$

$\delta\boldsymbol\mu$ contains the field-tangent kick $\frac{q}{mc}(\delta\mathbf E + \mathbf v\times\delta\mathbf B)\Delta t$ at the particle and the effect of any remaining $\delta\mathbf x$. In FBPIC the position update has no explicit $\theta$, so $\delta\mathbf x = 0$ for born-noisy markers; EV newborns keep only the small parent-ion displacement. The background and the ions stay exact Lagrangian. Field tangents come from the deposit of $(\dot W, \delta\mathbf x)$ as now. The estimator is

$$
\delta A = \sum_i \dot W_i\,(\phi_i - \bar\phi) + \sum_i W_i\nabla\phi_i\cdot\xi_i + \text{(birth monopoles)},
$$

where the centering $\bar\phi$ is exact because $E[\dot W^{LR}] = 0$.

### 2.2 Why it does not explode [derived]

No displacement of a noisy marker is propagated, so no factor $D\Phi$ appears. The weight variance grows linearly: $\mathrm{Var}(\dot W_i/W_i) = \sigma^{-2}\int\lvert\delta\mathbf a_i\rvert^2dt$. The cold-limit theorem (R0 A.2) does not apply: after one step the noisy markers in a cell have a velocity spread $\sigma\sqrt{t}$, and $E[\dot W\,\mathbf v] = \delta\langle\mathbf v\rangle$. The method is the R0 particle-pdf score form with the unknown score $\nabla_u\ln\tilde f$ replaced by the known score of an injected kernel. It is the Girsanov or likelihood-ratio derivative of an SDE [lit: Glynn, "Likelihood ratio gradient estimation for stochastic systems", CACM 33 (1990); Fournié et al., "Applications of Malliavin calculus to Monte Carlo methods in finance", Finance Stoch. 3 (1999); from memory, not re-read].

### 2.3 Bias, variance, cost

1. Bias [derived]: NLR is exact for the noisy model. Its bias against the noiseless model is $\partial_\theta E_\sigma[A] - \partial_\theta A$, which depends on $\sigma$ like the FD step $h$ or the R0 smoothing width $\sigma_u$. The noise also changes the primal, so the inversion fits the noisy model.
2. Variance [derived]: with $N_b$ noisy markers, the SE is about $\mathrm{SD}(\phi)\,\sigma^{-1}\big(\int\lvert\delta\mathbf a\rvert^2dt / N_b\big)^{1/2}$. It is independent of $\lambda$ but grows as $1/\sigma$.
3. Cost [estimated]: per parameter, one FP64 weight tangent per noisy marker, the existing field tangent and gather, and a weight-only deposit of the bunch. The noise is shared by all parameters. This is the R0 variant "Lagrangian background plus weight-form bunch", $c_1 \approx 0.8$ to $1.3$ (R0 Section 6.1), minus the $\nabla_u f$ estimate. GPU detail: a counter-based Philox draw keyed on (particle id, step, seed) makes the noise reproducible across $\theta$ and across ranks without storage.

### 2.4 Test T1: cold beam in prescribed chaotic waves [measured, `t1_sheet.py`, log `t1_sheet.log`, data `t1_sheet.json`]

Model: $x' = x + \Delta t\,v$, $v' = v + \Delta t\,F(x', t') + \sigma\sqrt{\Delta t}\,\varepsilon$, $F = -A\sin x - 0.2\sin(x - 2.2t)$, $\theta = A = 1$, cold beam $v_0 = 1$, $x_0$ uniform (the R0 `cold_beam_bias.py` model with a symplectic Euler step). Observables $J_1 = \langle s(v)\rangle$, $J_2 = \langle s\,v\rangle$, $J_3 = \langle v^2\rangle$ with a smooth window $s$ at $v_c = 2$. Truth: lattice (midpoint) label quadrature of the Lagrangian sum, $N$ up to $10^7$; it converges once the lattice resolves the stretched sheet. NLR: $N = 10^5$ random labels, 32 seeds. Noisy-system reference: CRN central FD at $h = 0.02$, same noise.

Truth at $T = 10$ (lattice $10^7$; secant at $h = 0.003$ agrees to $0.05\%$): $\partial J/\partial A = (0.5567, 1.5223, 5.3903)$.

| $T$ | $\sigma$ | NLR $\partial J_1$, $\partial J_3$ (SE) | Noisy CRN FD $\partial J_1$, $\partial J_3$ (SE) | Noiseless reference $\partial J_1$, $\partial J_3$ |
|---|---|---|---|---|
| 10 | 0.01 | 0.460 (0.034), 4.45 (0.22) | 0.429 (0.002), 4.30 (0.012) | 0.557, 5.39 |
| 10 | 0.03 | 0.245 (0.014), 3.21 (0.089) | 0.236 (0.002), 3.14 (0.010) | 0.557, 5.39 |
| 10 | 0.1 | 0.059 (0.004), 2.27 (0.022) | 0.057 (0.003), 2.24 (0.014) | 0.557, 5.39 |
| 10 | 0.3 | 0.037 (0.001), 0.97 (0.012) | 0.042 (0.003), 0.99 (0.016) | 0.557, 5.39 |
| 20 | 0.01 | 0.427 (0.038), 5.25 (0.31) | 0.388 (0.005), 4.79 (0.027) | no scale-free value (secants 0.15 to 0.75) |
| 20 | 0.03 | 0.235 (0.017), 2.42 (0.12) | 0.214 (0.004), 2.29 (0.023) | same |
| 20 | 0.1 | 0.108 (0.005), 0.93 (0.040) | 0.106 (0.005), 0.94 (0.033) | same |
| 30 | 0.01 | $-0.044$ (0.075), $-1.54$ (0.50) | $-0.011$ (0.006), $-1.10$ (0.039) | same |

Findings [measured]:

1. NLR estimates the noisy-model derivative without bias: in all 12 $(T, \sigma)$ pairs and 3 observables, NLR and CRN FD of the noisy model agree within 2.3 combined SE (most within 1 SE).
2. No $\sigma$ plateau: the noisy-model derivative changes by a factor 2 per factor 3 in $\sigma$, also in the chaotic case ($T = 20$: 0.43, 0.24, 0.11). At $T = 10$ the smallest noise ($\sigma\sqrt T = 0.03$ in $v$) already changes the derivative by $-20\%$ to $-31\%$ (noisy CRN FD against the truth).
3. At $\sigma \le 0.03$ the NLR SE is 4 to 14 times the SE of CRN FD of the same noisy model, at equal particle count. At $\sigma = 0.1$ they are equal; at $\sigma = 0.3$ NLR is 2 times better, where the bias is $-90\%$.
4. NLR stays regular at every $T$, as the Lagrangian sum does not (Lagrangian SE at $N = 10^5$: 0.012 at $T = 10$, 0.50 at $T = 20$, 45 at $T = 30$).

Break-even with FD [derived from row 3]: to match the SE of CRN FD of the same model, NLR needs $r^2$ times more samples, with $r = 4$ to 14 at $\sigma \le 0.03$. Its cost is then $r^2(1 + c_1K)$ against $1 + 2K$ for central FD; with $c_1 \approx 1$, NLR loses for every $K$ once $r^2 > 2$. It wins only at $\sigma \ge 0.1$, where the bias at $T = 10$ is $-58\%$ to $-92\%$.

### 2.5 Test T2: self-consistent cold wake with EV injection [measured, `t2_nlr_r2.py`, jobs `t2_jobs.txt`, data `t2_results.jsonl`]

Model: the R0 R2 pilot (1D relativistic electrostatic wake, prescribed ponderomotive driver, EV ionization injection, cold plasma, about $2\cdot10^5$ markers, $t_f = 75$, $\theta = a_0 = 2.5$), with an exact forward tangent added (field tangent from $\delta\rho$, Lagrangian particle tangents, EV birth-weight tangent). The tangent matches central FD at $h = 10^{-7}$ to 5 digits at $t_f = 20$ (`chk_t2b.out`). At $t_f = 75$ the exact column is $(0.24, 68.9, 113, -0.50, -8.7, -2.2)$, close to the R0 twin value $(0.29, 67.1, 111.5, -0.48, -8.6, -2.15)$, so the pathology is reproduced. The six observables are $\log Q$, $\langle u\rangle$, $\langle\xi\rangle$, $\log\sigma_u$, $\log\sigma_\xi$, $\log(\sigma_u/\langle u\rangle)$.

NLR with $\sigma = 0.3$ on the EV bunch only (`noisy = b`), background exact Lagrangian, 8 noise seeds; noisy-model reference: CRN central FD at $h = 0.01$, 6 seeds.

| Column | $\log Q$ | $\langle u\rangle$ | $\langle\xi\rangle$ | $\log\sigma_u$ | $\log\sigma_\xi$ | $\log(\sigma_u/\langle u\rangle)$ |
|---|---|---|---|---|---|---|
| Noiseless cubic-fit slope (R0) | 2.30 | $-14.0$ | $-21.8$ | 0.77 | 0.23 | 1.12 |
| Noiseless exact tangent | 0.24 | 68.9 | 113 | $-0.50$ | $-8.7$ | $-2.2$ |
| Noisy model, CRN FD (SE) | 1.30 (0.01) | 14.4 (0.5) | $-4.9$ (0.5) | 0.49 (0.01) | 0.04 (0.04) | 0.14 (0.02) |
| NLR-b mean (SE), 8 seeds | $-1.9$ (2.9) | 140 (109) | 152 (149) | $-2.3$ (2.2) | $-21$ (21) | $-5.7$ (4.9) |

Findings [measured]:

1. NLR on the bunch alone fails. Single-seed columns range from $-267$ to $+611$ for $\partial\langle u\rangle/\partial a_0$, against 14.4 for the noisy model.
2. The cause is the background. With the bunch displacements frozen (median $\lvert\delta x_b\rvert = 0.085$), the plasma-electron tangents still grow at the noiseless rate: 90th percentile $\lvert\delta x_e\rvert$ from 10 at $t = 50$ to 748 at $t = 75$ ($\lambda \approx 0.17\,\omega_p$; noiseless: 11 to 963). The field tangent grows to $\max\lvert\delta E\rvert = 1.9\cdot10^5$, and the bunch weights then absorb these kicks (90th percentile $\lvert\dot W/W\rvert = 1340$). This is the R0 Q2 risk in the 1D pilot: the background carries the growth, so any bunch-only scheme (NLR-b, R-v-s, bunch-only GPT or LIBP) fails here.
3. The noise changes the model strongly: $\sigma = 0.3$ raises the selected charge by a factor 2.2 ($\log Q$: $-2.77$ against $-3.56$) and flips the sign of $\partial\langle u\rangle/\partial a_0$. Noise during injection moves electrons across the trapping threshold. A usable NLR would need zero noise until trapping and a much smaller $\sigma$, which raises the variance.

T2_EB_PLACEHOLDER

### 2.6 Verdict on NLR

Conditional, leaning to fails for Twin-S. It is the simplest cold-proof regularizer and scales to many parameters at about one weight per particle per parameter. But it is a coarse-grained derivative at scale $\sigma$ with no plateau, it changes the primal model, and at small $\sigma$ its variance is worse than FD of the same model. It is worth keeping only as a control: a $\sigma$-sweep shows how strongly a Twin-S observable depends on unresolved fine structure.

## 3. Method B: LIBP, label-space integration by parts on the cold sheet

### 3.1 Equations [derived]

On a cold sheet $z(a, t)$, $a\in\mathbb R^d$, weights $w(a)$, $T = \partial_a z$ ($2n\times d$), split the tangent:

$$
\xi = T\tau + \nu,\qquad \tau = (T^{\mathsf T}T)^{-1}T^{\mathsf T}\xi,\qquad \nu = \big(I - T(T^{\mathsf T}T)^{-1}T^{\mathsf T}\big)\xi .
$$

The along-sheet term is a label derivative of $\phi\circ z$, and labels carry a smooth measure, so

$$
\int w\,\nabla\phi\cdot T\tau\,da = \int w\,\tau\cdot\nabla_a(\phi\circ z)\,da = -\int \phi\circ z\;\nabla_a\cdot(w\tau)\,da .
$$

Hence, with a smooth partition $\chi$ (a function of the local stretch $s = \log\lvert T\rvert$),

$$
\delta A = \int\Big[(1-\chi)\,w\,\nabla\phi\cdot T\tau \;-\; \phi\,\nabla_a\cdot(w\chi\tau) \;+\; w\,\nabla\phi\cdot\nu\Big]da .
$$

$\chi = 0$ is the Lagrangian sum; $\chi = 1$ is full LIBP. The identity holds for any smooth $\chi$ (periodic or compactly supported labels). Discretely, each marker carries a weight-like scalar $\omega_i = -\nabla_a\cdot(w\chi\tau)/w$ and a displacement $\nu_i + (1-\chi_i)T_i\tau_i$. The same identity with $\phi \to S_g$, $q\mathbf v S_g$ gives $\delta\rho$ and $\delta\mathbf J$, so the field tangents are consistent.

$\nabla_a\cdot\tau$ needs the label derivatives of $\xi$ and of $T$:

$$
\partial_{a_k}\tau = (T^{\mathsf T}T)^{-1}\Big[(\partial_{a_k}T)^{\mathsf T}\xi + T^{\mathsf T}\partial_{a_k}\xi - \big((\partial_{a_k}T)^{\mathsf T}T + T^{\mathsf T}\partial_{a_k}T\big)\tau\Big].
$$

$T$ and $\partial_aT$ are homogeneous first- and second-order tangents of the test-particle flow (shared by all parameters). $\partial_a\xi$ is a mixed second-order tangent (one set per parameter): $\dot{\xi_a} = D\!F\,\xi_a + D^2\!F[T, \xi] + D(\delta F)\,T$.

### 3.2 Why it does not explode [derived]

1. Normal part. For a Lagrangian sheet ($d = n$) in a Hamiltonian flow, $D\Phi$ maps $TL$ to $TL$ and acts on the quotient $T_zM/TL \cong T^*L$ as $(D\Phi|_{TL})^{-\mathsf T}$. Where the sheet stretches, the normal part contracts. In 1D-1V: $t\times\xi$ obeys $\frac{d}{dt}(t\times\xi) = t\times b$, so $\nu = (t\times\xi)/\lvert t\rvert = \int t(s)\times b(s)\,ds/\lvert t(t)\rvert$, a weighted average of past forcing with weights $\lvert t(s)\rvert/\lvert t(t)\rvert \le 1$ under monotone stretching.
2. Along-sheet part. A forcing $b$ smooth in $z$ at scale $\ell$ gives $\tau \sim \lvert b\rvert/\lvert t\rvert$ and $\partial_a\tau \sim \lvert\nabla b\rvert + \kappa\lvert b\rvert$, independent of $e^{\lambda t}$, except where the curvature $\kappa$ of the sheet is large.
3. Doubling map: $\xi = 2^n = 2^n\cdot 1$, so $\tau = 1$, $\partial_a\tau = 0$, $\nu = 0$: LIBP gives 0 exactly.
4. Literature: this is the space-split idea of Chandramoorthy and Wang (S3: unstable part by integration by parts along unstable manifolds, stable part by tangents) [lit: N. Chandramoorthy, Q. Wang, "Efficient computation of linear response of chaotic attractors with one-dimensional unstable manifolds", SIAM J. Appl. Dyn. Syst. 21 (2022); from memory, not re-read]. The difference here: the sheet is known from the labels, the density along it is the smooth label measure, and no ergodic or SRB assumption is needed. I do not know of a PIC application.

### 3.3 Test T1b: LIBP on the cold beam [measured, `t1_sheet.py`, `t1_diag.py`, `t1_hybrid.py`, `t1_sec15.py`; outputs `t1_diag.out`, `t1_hybrid.json`, `t1_sec15.json`]

The tangents $t$, $t_a$, $\xi$, $\xi_a$ match central FD to 7 or more digits (`chk_t1.py`).

Identity check (lattice $N = 4\cdot10^6$, $T = 10$): Lagrangian and every LIBP variant give $\partial J = (0.556715, 1.522299, 5.390293)$ to 6 digits.

Raw LIBP ($\chi = 1$) fails as a Monte Carlo estimator. Per-particle SD of the $J_1$ summand, $N = 2\cdot10^5$ random labels (`t1_diag.out`):

| $T$ | median $\lvert t\rvert$ | Lagrangian SD | raw LIBP SD | top 0.1% of $\lvert$LIBP$\rvert$: median rank of $\kappa$, of $\lvert t\rvert$ |
|---|---|---|---|---|
| 5 | 2.0 | 0.014 | 0.012 | 0.32, 0.98 |
| 10 | 4.1 | 20.7 | $8.3\cdot10^4$ | 0.997, 0.000 |
| 15 | 10.6 | 165 | $1.1\cdot10^3$ | 0.997, 0.10 |

The spikes sit where the sheet is compressed along itself ($\lvert t\rvert$ small) and turns sharply ($\kappa$ in the top 0.3%). There $\tau = \xi\cdot t/\lvert t\rvert^2$ is ill-conditioned. With lattice sampling these spikes are resolved and the sum converges; with random sampling they dominate the variance.

Hybrid LIBP (smoothstep $\chi$ in $\log\lvert t\rvert$ between the cut values), $N = 10^5$ random labels, 16 seeds; values for $J_1$:

| $T$ | truth | Lagrangian mean (SE), particle SD | hybrid 10 to 30 | hybrid 30 to 100: mean (SE), particle SD |
|---|---|---|---|---|
| 10 | 0.5567 | 0.521 (0.018), 23.7 | 0.541 (0.008), 9.5 | 0.551 (0.010), 8.9 |
| 15 | $-0.595$ | $-0.44$ (0.38), 337 | $-0.610$ (0.057), 87 | $-0.599$ (0.054), 44 |
| 20 | about 4 (lattice: 3.7 to 5.1) | 5.2 (1.65), 1330 | 6.7 (3.0), 1659 | 3.91 (0.13), 186 |

Truth at $T = 15$: lattice $4\cdot10^6$ gives $-0.597$ (Lagrangian) and $-0.593$ to $-0.595$ (LIBP variants). Lattice secants: $-0.5925$ ($h = 0.001$), $-0.588$ ($0.003$), $-0.552$ ($0.01$), $+0.206$ ($0.03$), $+0.232$ ($0.1$) (`t1_sec15.log`).

Findings [measured]:

1. Hybrid LIBP with cuts at $\lvert t\rvert = 30$ to 100 is unbiased within 1 SE at $T = 10$ and 15 and reduces the per-particle SD by 2.7, 7.6 and 7.2 times at $T = 10, 15, 20$ (variance by 7 to 58 times). The SE gain at $T = 20$ is 13 times.
2. Low cuts (3 to 10) are worse than the Lagrangian sum at $T = 15$ and 20: converting moderately stretched, curved parts imports the spikes.
3. The exact derivative is useful only where the continuum response is smooth. At $T = 15$ it agrees with secants for $h \le 0.01$, and the response bends at $h = 0.03$. At $T = 20$ the pointwise derivative is about 4, while lattice secants at $h = 0.003$ to 0.1 are 0.15 to 0.75 (`t1_sheet.log`): a correct pointwise derivative and a poor descent direction.

### 3.4 Bias, cost, and risks for Twin-S

1. Bias [derived]: none in the continuum; the estimator targets the exact mean-field derivative. It does not target the derivative of the discrete $N$-particle system.
2. Variance [measured, 1D]: set by fold curvature and compression, not by $e^{\lambda t}$; still grows with $T$ (particle SD 8.9, 44, 186 at $T = 10, 15, 20$, against 23.7, 337, 1330 for Lagrangian).
3. Cost [estimated]: shared per marker, $T$ and $\partial_aT$: $6d + 6d(d+1)/2$ doubles (54 for $d = 3$, 84 for $d = 4$). Per parameter: $\xi$ and $\partial_a\xi$, $6 + 6d$ doubles (24 to 30, against 6 now). The tangent kernels need second derivatives of the gathered fields, $\nabla\nabla\mathbf E$ and $\nabla\nabla\mathbf B$. With the current $c_1 = 2.47$ (R0), $c_1$ would be about 7 to 12 [estimated, 3 to 5 times the per-parameter tangent work]. Restricting LIBP to the EV bunch keeps the background Lagrangian.
4. Shape functions [derived]: second-order tangents of the discrete map need $C^2$ gather shapes. Linear shapes give $\nabla\nabla E = 0$ almost everywhere plus missed jumps. Twin-S would need cubic shapes for the bunch gather.
5. EV manifold [derived, untested]: $d = 4$ in 6-D is not Lagrangian. If it aligns with the top four Oseledets directions ($\lambda_1, \lambda_2, \lambda_3, -\lambda_3$), one along-sheet direction contracts, and $\tau$ along it grows as $e^{\lambda_3t}$. A per-direction $\chi$ would be needed.
6. Precision [derived]: $\partial_a\tau$ is an $O(1)$ result of $O(e^{\lambda t})$ terms; FP64 loses about $\log_{10}e^{\lambda t}$ digits (11 at the P7 maximum of $1.7\cdot10^{11}$). FP32 is not possible.
7. $N$-body chaos: $T$ is a test-particle tangent in the actual noisy fields. If much of the P7 growth is grid-scale $N$-body scrambling rather than mean-field stretching, the sheet is rough at the label scale and the spikes multiply. R0 Q1 and Q2 decide this.

### 3.5 Verdict on LIBP

Promising as a research direction, conditional for Twin-S. It is exact, it is native to cold plasma, and it is the cold-sheet counterpart of the gauge idea. The 1D hybrid gives a 7 to 58 times variance reduction without bias. Against it: a high implementation cost (second-order tangent kernels, $C^2$ shapes), $c_1$ about 7 to 12, untested on a self-consistent field and on a $d = 4$ manifold, and it only helps when the continuum response is smooth at the optimizer's scale.

## 4. Method C (rejected): regression, Koopman, and moment-closure tangents

### 4.1 Equivalence to the Galerkin weight conversion [derived]

Regress the final observable on smooth features $g(z) \in \mathbb R^m$ of the state at each source time $s$: $\hat\psi_s = c_s^{\mathsf T}g$, with $c_s = G_s^{-1}\sum_iw_ig(z_i(s))\phi(z_i(T))$ and $G_s = \sum_iw_ig\,g^{\mathsf T}$. Put $\hat\psi_s$ into the weak form $\delta A = \int ds\sum_iw_i\nabla\psi_s\cdot b_i(s)$:

$$
\delta A \approx \int ds\; s_s^{\mathsf T}c_s = \sum_i w_i\,\phi(z_i(T))\,\omega_i,\qquad \omega_i = \int ds\; g(z_i(s))^{\mathsf T}G_s^{-1}s_s,\qquad s_s = \sum_jw_j\nabla g(z_j(s))\cdot b_j(s).
$$

$\omega_i$ is exactly the per-step Galerkin weight conversion of R0 A.4 (R-v with $T = 1$), that is, collocated particle-pdf in a Galerkin basis. EDMD or Koopman regression ($K = G^{-1}C$) and Gaussian or moment closures (degree 1 or 2 polynomial features per slice) are special cases. They therefore inherit the cold-limit theorem (R0 A.2): on a cold node the feature rows are collinear on the sheet, and no weight on the sheet reproduces a normal displacement, so the current response is lost.

### 4.2 What would rescue it, and why that fails [derived]

Gradient-enhanced (Sobolev) regression would add the normal derivative from $D\Phi$ data. Over one step this is bounded. Composed over many steps it is the Lagrangian tangent again. With resets it becomes GPT or R-v. No cheap test is proposed, because R0 already measured the cold failure of this family (one-step $\delta J/\delta J_{exact} \le 0.006$, R0 Appendix B).

## 5. Comparison with GPT

| Aspect | GPT (lead) | A. NLR | B. LIBP hybrid | C. Regression |
|---|---|---|---|---|
| What it targets | coarse-grained Wasserstein tangent at the basis scale | exact derivative of a noisy model | exact continuum derivative | Galerkin projection of the score |
| Exact on a cold sheet? | exact GPT removes only label-divergence-free relabelings (1-D gauge in 1D-1V); regularizes only by basis truncation [derived] | yes for the noisy model | yes [derived; measured identity] | no (cold-limit theorem) |
| Needs $\nabla_u f$ or density | no | no | no | no (but loses normal response) |
| Removes $e^{\lambda t}$ from | gauge part only; the rest regrows as $e^{\lambda T}$ between projections | all noisy markers | along-sheet part (stretched regions) | all |
| Bias | basis truncation across folds; uncontrolled before phase mixing | $\sigma$-dependent, no plateau in T1 [measured] | none in the continuum | cold-limit, $O(1)$ |
| Variance | from the regrown part and the projection noise; not measured here | grows as $1/\sigma$; 4 to 14 times FD at small $\sigma$ [measured] | spikes at compressed, curved points; 7 to 58 times below Lagrangian with the hybrid [measured] | low |
| Field-tangent consistency | needs $\delta\rho$, $\delta\mathbf J$ deposited from projected $\xi$; projection changes them unless charge and current rows are in the basis | automatic (weights plus Lagrangian background) | automatic (same identity with $S_g$) | needs charge and current rows |
| Cost per parameter | projection solve per reset plus current tangents | about 1 weight per marker; $c_1 \approx 0.8$ to 1.3 [estimated] | 4 to 5 times the current tangent state; $c_1 \approx 7$ to 12 [estimated] | low |
| Warm or mixed $f$ | right tool (gauge is large) | works | fails (labels = initial $z$, pull-back explodes along stable directions) | works (it is the score form) |
| Novelty for PIC | high | moderate (known SDE technique, new use) | high (S3-type split with the label sheet; no SRB density) | low |

Summary of the comparison [derived]:

1. GPT and LIBP address the same object, the along-sheet displacement. GPT keeps it as a displacement and removes only the gauge; on a cold sheet that gauge is small. LIBP converts the along-sheet part into a weight by an exact identity.
2. They are complementary. LIBP fits the cold, unmixed sheet (background, early bunch). GPT fits the warm or coarse-grained phase-mixed part, where many labels share a point and $\xi - \bar\xi$ is gauge.
3. Any GPT test in 1D-1V on a cold background is a worst case for GPT, because there the exact gauge is one-dimensional. Its result then measures the basis truncation only.

## 6. Recommendation and cheap next tests

1. Do not replace FD for Twin-S multi-parameter work with any of A, B, C now. Same reasons as R0: at best parity in cost (A) or a large cost (B), and the targeted derivative is useful only if the continuum response is smooth at the optimizer's step.
2. If the lead wants one novel method, LIBP-hybrid is the candidate. Gate it on R0 Q1 (roughness falls as $N^{-1/2}$) and Q2 (the background stays laminar).
3. Cheap next tests (CPU, 1 to 3 days):
   - T3: LIBP-hybrid on a self-consistent 1D cold plasma (R2 pilot with quadratic shapes, background sheet only, $\theta = a_0$). Pass: unbiased against lattice-converged truth and particle SD at least 5 times below Lagrangian at $\lambda t \ge 8$.
   - T4: LIBP on a 2-label manifold in 1D-1V with one contracting label direction (EV-like births over time). Measures item 5 of Section 3.4.
   - T5: GPT on the T1 cold sheet with Fourier times polynomial bases of increasing size. Prediction [derived]: the exact-basis limit equals the Lagrangian sum; any variance reduction comes with a basis-dependent bias.

## Files

All under `inversion/notes/gpt-evidence-2026-11-01/work_alt/`:

1. `t1_sheet.py`, `t1_sheet.log`, `t1_sheet.json`: T1 model, lattice truth, secants, Lagrangian, raw LIBP, NLR, noisy CRN FD.
2. `chk_t1.py`: FD check of the first- and second-order tangents.
3. `t1_diag.py`, `t1_diag.out`: location of the raw-LIBP spikes.
4. `t1_hybrid.py`, `t1_hybrid.log`, `t1_hybrid.json`: hybrid LIBP.
5. `t1_sec15.py`, `t1_sec15.log`, `t1_sec15.json`: lattice secants at $T = 15$.
6. `t2_nlr_r2.py`, `t2_jobs.txt`, `t2_results.jsonl`, `t2_errors.log`; `r2_pilot_copy.py` (copy of the R0 pilot); `chk_t2a.out`, `chk_t2b.out` (tangent check against FD at $h = 10^{-7}$).
