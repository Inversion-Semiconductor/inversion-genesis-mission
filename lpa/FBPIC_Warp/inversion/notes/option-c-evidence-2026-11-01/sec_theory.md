# Option C theory: Girsanov weights on a noise-regularized PIC model

| Item | Value |
|---|---|
| Key | `theory` |
| Date | 2026-11-01 |
| Scope | Derivation of option C (likelihood-ratio weights on a Langevin-regularized FBPIC model) for $d\,\mathbb E[J]/d\theta$ of beam-moment residuals, forward mode only |
| Work directory | `inversion/notes/option-c-evidence-2026-11-01/work_theory/` (written `wt/` below) |

Labels: **[derived]** derived here; **[measured]** measured on CPU with the named script in `wt/`; **[literature]** with citation; **[estimated]** an estimate with stated inputs; **[R0]** the R0 design note `inversion/notes/r0-design-note-2026-10-31.md`.

## 0. Result in brief

1. **The method is exact for the noisy model.** For any noise level $\sigma > 0$ in all momentum directions where the force tangent acts, the per-particle weight $d\omega_i = \sigma^{-1}\delta\mathbf a_i\cdot d\mathbf W_i$ with the self-consistent field tangent represents $\delta f$ of the linearized Vlasov-Fokker-Planck (VFP) equation without bias in the mean-field limit. It needs no trajectory tangent and no $\nabla_u f$, so it is cold-safe [derived, Sections 2.2 and 2.4; measured, Section 4.6].
2. **It is one end of a family.** For any adapted control $\mathbf h$, splitting the drift tangent into a pathwise part $\delta\mathbf a - \mathbf h$ (carried by $\xi$) and a likelihood-ratio part $\mathbf h$ (carried by $\omega$) gives an exact estimator. A feedback control $\mathbf h = K\xi$ keeps $\xi$ bounded and moves only the slow and unstable part of the tangent into weights [derived, Section 2.3; measured, Section 4.5].
3. **Pure Girsanov from $t = 0$ is not viable on Twin-S.** The laser quiver force tangent enters the weight variance as $\int|\delta\mathbf a|^2dt/\sigma^2$ without cancellation. At $a_0 = 4.0$ this gives $\mathrm{Var}\,\omega \approx 7.6\cdot10^4/u_n^2$ for noise acting over the whole run, with $u_n$ the noise-induced momentum spread [estimated, Section 4.3]. The 1D harness `r2_pilot.py` uses a cycle-averaged ponderomotive driver and cannot show this failure.
4. **The plain mean-field estimator has an exponential variance loop.** Weight noise deposits random $\delta\rho$, which gives a random $\delta\mathbf E$, which feeds new weight increments. The measured rate is $\Gamma \approx 0.25/(n_m\sigma^2)$ in a cold 1D plasma ($n_m$ = markers per $k_p^{-1}$): $\Gamma = 2.9\,\omega_p$ at $\sigma = 0.01$, $N = 6400$, so $\mathrm{Var}\,\omega$ reached $10^{37}$ [measured, Section 4.4]. Charge-neutral (locally centered) weight increments cut the rate to about $0.09$ to $0.16\,\omega_p$, remove the $\sigma^{-2}$ dependence, and recover the FD reference for all three observables at $\sigma = 0.01$ within $1\,$SE [measured]. The toy has no orbit crossing; the centering gain after trapping is untested. This is the most important design element.
5. **Cost per extra parameter is not well below one simulation for a credible variant.** The weight-only form costs about $0.4$ to $0.8$ primal runs per parameter, but it needs pure Girsanov in the laser (item 3). Credible hybrids that keep the Lagrangian tangent in the laser and before the chaotic window cost about $0.8$ to $2.5$ primal runs per parameter, plus repeats if one run is too noisy [estimated, Section 6]. Goal (4) is therefore not met by the variants derived here. Goals (1) to (3) are reachable in principle.

## 1. The noisy model in FBPIC variables

### 1.1 Continuous model

Notation as in [R0, 3.1]: $\mathbf u = \mathbf p/(mc)$, $\gamma = \sqrt{1 + u^2}$, $\mathbf v = c\mathbf u/\gamma$, $\mathbf a_s = \frac{q_s}{m_sc}(\mathbf E + \mathbf v\times\mathbf B)$. For a noisy species $s$:

$$
d\mathbf x = \mathbf v(\mathbf u)\,dt,\qquad
d\mathbf u = \big[\mathbf a_s(\mathbf x, \mathbf u, t) - \nu_s(\mathbf u - \mathbf U_s)\big]dt + \Sigma_s(t)\,d\mathbf W, \tag{1.1}
$$

with $\mathbf W$ a standard 3D Brownian motion per marker, $\Sigma_s = \mathrm{diag}(\sigma_\perp, \sigma_\perp, \sigma_\parallel)\,\chi_s(t)$, and $\chi_s(t) \in \{0, 1\}$ a time window. $D = \sigma^2/2$ is the momentum diffusivity (units $u^2$ per time). The phase-space density obeys

$$
\partial_tf_s + \mathbf v\cdot\nabla_xf_s + \nabla_u\cdot\big[(\mathbf a_s - \nu_s(\mathbf u - \mathbf U_s))f_s\big] = \tfrac12\nabla_u\cdot(\Sigma_s\Sigma_s^T\nabla_uf_s). \tag{1.2}
$$

Itô and Stratonovich forms coincide because $\Sigma_s$ does not depend on the state [derived]. A position-dependent mask $\sigma(\mathbf x)$ is allowed, but then the pathwise tangent gets the extra term $(\partial_x\sigma\,\xi_x)\,d\mathbf W$ (Section 5.1). A time-only window avoids that term.

### 1.2 Choice of form

| Form | Effect on the physics | Verdict |
|---|---|---|
| Pure diffusion ($\nu = 0$) in a window $[t_s, T]$ | heating $\langle\Delta u_k^2\rangle = \sigma_k^2(t - t_s)$; no momentum or wake damping | **Recommended** [derived] |
| OU drag to zero, $-\nu\mathbf u$ | collisional damping of the wake at rate $\nu/2$ and a drag on the bunch; bias $O(\nu T)$ on the wake amplitude | Rejected [derived] |
| Dougherty drag to the local mean $\mathbf U(\mathbf x, t)$ (Lenard-Bernstein form) | bounded temperature $D/\nu$; conserves local momentum in the nonrelativistic limit; after orbit crossing it couples the trapped bunch to the background mean velocity, which is wrong physics | Rejected for now. It also adds a deposit of $\mathbf U$ and a $\nu\,\delta\mathbf U$ term to $\delta\mathbf a$ [derived] |
| Noise on the bunch only (`elec_from_N`) | no background heating | Valid only if the background tangent growth is driven by the bunch and not intrinsic. Unknown [R0, 2 "not supported" item 3] |
| Transverse-only noise on the bunch ($\sigma_\parallel = 0$) | no energy-spread bias | Valid if the bunch $\xi$ growth is transverse; then $\xi_z, \xi_{u_z}$ stay pathwise. Unknown |

Rule [derived, Section 2.3]: a weight can absorb a force tangent only in a direction where noise acts, $\mathbf h \in \mathrm{range}(\Sigma)$. Every other component stays in the pathwise tangent $\xi$.

### 1.3 Effect on the physics as a function of $\sigma$

Define the noise-induced momentum spread at the end of a window of length $T_n$: $u_n = \sigma\sqrt{T_n}$. Twin-S scales, from `inversion/configs/twin_s.json` [estimated, computed]: $n_{e0} = 8\cdot10^{24}\,\mathrm m^{-3}$, $\omega_p = 1.60\cdot10^{14}\,\mathrm s^{-1}$, $k_p^{-1} = 1.88\,\mu\mathrm m$, $\omega_p\Delta t = 0.0266$, full run $T = 53.2\,\omega_p^{-1}$, window from step 1450 $T_n = 14.6\,\omega_p^{-1}$, $\omega_0/\omega_p = 14.8$, $a_0 = 4.0$ at $E_L = 0.27\,$J.

1. **Background temperature** (nonrelativistic): $k_BT = m_ec^2u_n^2$. $u_n = 0.01, 0.03, 0.1$ give 51 eV, 0.46 keV, 5.1 keV [derived]. The thermal wave-breaking limit stays far above the wake amplitude for $u_n \le 0.1$ (Katsouleas and Mori 1988 scale $E_{max} \propto \beta_{th}^{-1/2}$) [literature, estimated].
2. **Bunch transverse emittance.** A matched beam in an ion channel ($\beta_\perp = \sqrt{2\gamma}\,k_p^{-1}$) with uncorrelated kicks has $d\varepsilon_n/dt = \beta_\perp D/\gamma$ (multiple-scattering formula $\Delta\varepsilon = \tfrac12\beta\Delta\langle x'^2\rangle$), so $\Delta\varepsilon_n = u_n^2/\sqrt{2\gamma}\;k_p^{-1}$ [derived]. At $\gamma = 6$, $u_n = 0.1$: $\Delta\varepsilon_n = 5\,$nm, small next to the residual quiver spread of ionization-injected electrons ($\sigma_{u_x} = O(1)$) [estimated].
3. **Bunch energy spread and selection.** Longitudinal noise adds $u_n^2$ to $\sigma_{u_z}^2$. The Twin-S selection is a smooth step of width $4\delta u = 0.154$ at $u_c = 6.25$. Smoothing it at $u_n$ changes the selected charge by a relative $O(u_n^2/(4\delta u)^2)$: 4% at $u_n = 0.03$, 43% at $u_n = 0.1$ [estimated]. This bounds $\sigma_\parallel$ on the bunch: $u_{n,\parallel} \lesssim 0.03$, or $\sigma_\parallel = 0$.
4. **Regularization scale.** The noisy model is smooth in $\theta$ at the parameter scale $h_{eq} \approx u_n/|\partial u/\partial\theta|$. For $|\partial u_z/\partial\ln E_L| \approx 3$ to 6 (bunch at $u \approx 6$), the FD step $h = 10^{-2}$ that is smooth at 1x particles corresponds to $u_n \approx 0.03$ to 0.06 [estimated]. The noise level is a stated physical smoothing scale, like temperature or shot-to-shot jitter. The inversion then fits the noisy model, or accepts a bias of the same kind as an FD step.

## 2. Linearized VFP and the weight representation

### 2.1 Sensitivity equations

With $\delta(\cdot) = \partial_\theta(\cdot)$ and $L_s$ the generator of (1.1) at the unperturbed fields,

$$
\partial_t\delta f_s = L_s^*\delta f_s - \nabla_u\cdot(f_s\,\delta\mathbf a_s) + \delta G_s,\qquad
\delta\mathbf a_s = \frac{q_s}{m_sc}(\delta\mathbf E + \mathbf v\times\delta\mathbf B) + \partial_\theta\mathbf a_s\big|_{expl}, \tag{2.1}
$$

$$
\partial_t\delta\mathbf B = -\nabla\times\delta\mathbf E,\quad \partial_t\delta\mathbf E = c^2\nabla\times\delta\mathbf B - \delta\mathbf J/\varepsilon_0,\quad
\delta\rho = \sum_sq_s\!\int\!\delta f_s,\quad \delta\mathbf J = \sum_sq_s\!\int\!\mathbf v\,\delta f_s, \tag{2.2}
$$

$L_s^*g = -\mathbf v\cdot\nabla_xg - \nabla_u\cdot[(\mathbf a_s - \nu_s(\mathbf u - \mathbf U_s))g] + \tfrac12\nabla_u\cdot(\Sigma\Sigma^T\nabla_ug)$. The diffusion term is $\theta$-independent and produces no source [derived]. With the Dougherty drag, $\delta\mathbf a_s$ gets $+\nu_s\delta\mathbf U_s$. For $E_L$: $\delta f(0) = 0$ and $(\delta\mathbf E, \delta\mathbf B)(0) = F_0/(2E_L)$ [R0, 3.1]. In `r2_pilot.py` the driver is an explicit force, so $\partial_\theta\mathbf a|_{expl} = -\partial_{a_0}\nabla A/(2\gamma)$ with $\partial_{a_0}A = 2A/a_0$ and the $\gamma$ dependence on $A$ included.

### 2.2 Pure Girsanov representation [derived]

Fix the unperturbed fields (mean-field limit: deterministic). Let $Z_t = (\mathbf x_t, \mathbf u_t)$ follow (1.1). Let $\omega$ obey

$$
d\omega_t = \big(\Sigma^{-1}\delta\mathbf a(Z_t, t)\big)\cdot d\mathbf W_t,\qquad \omega_0 = 0. \tag{2.3}
$$

**Claim.** For every bounded test function $g$ and every $T$: $\int g\,\delta f(T) = \mathbb E[\omega_Tg(Z_T)]$, with $f(0)$ the initial law and no births.

**Proof.** Let $\varphi$ solve the backward Kolmogorov equation $\partial_t\varphi + L\varphi = 0$, $\varphi(T) = g$. With $\Sigma > 0$ on the range of $\delta\mathbf a$, $\varphi(t)$ is smooth in $\mathbf u$ for $t < T$ (Gaussian convolution in the noisy directions). Itô's formula gives $d\varphi(t, Z_t) = \nabla_u\varphi\cdot\Sigma\,d\mathbf W$. The product rule gives $d(\omega\varphi) = \omega\,d\varphi + \varphi\,d\omega + \delta\mathbf a\cdot\nabla_u\varphi\,dt$. The first two terms are martingales, so $\mathbb E[\omega_Tg(Z_T)] = \int_0^T\!\int f\,\delta\mathbf a\cdot\nabla_u\varphi\,dz\,dt$. From (2.1), $\frac{d}{dt}\int\varphi\,\delta f = \int\varphi\,(-\nabla_u\cdot(f\delta\mathbf a)) = \int f\,\delta\mathbf a\cdot\nabla_u\varphi$. Both sides agree and $\delta f(0) = 0$. $\square$

This is Girsanov's theorem applied to a drift perturbation (Girsanov 1960; for sensitivities: Glasserman 2004, Sec. 7.3; Malliavin weight sampling: Warren and Allen, PRL 109, 250601, 2012) [literature]. The bias of the pathwise-free estimator is zero at any $\sigma > 0$. The price is variance (Section 4).

Cold safety. The claim needs no density of $f$, only noise. A cold sheet at $t_s$ acquires a Gaussian width $\sigma\sqrt{t - t_s}$ in $\mathbf u$. The weight current $\sum_iq w_i\omega_i\mathbf v_i$ has the right expectation because $\omega_i$ correlates with the noise that moved $\mathbf v_i$: $\mathbb E[\Delta\omega\,\Delta\mathbf u] = \delta\mathbf a\,\Delta t$ per step. The R0 cold-limit theorem [R0, A.2] does not apply, because the velocities in a cell are no longer equal and the weight increments are not independent of them [derived].

### 2.3 Mixed pathwise and Girsanov representation [derived]

Let $\mathbf h_t$ be any adapted process with values in $\mathrm{range}(\Sigma)$ (per particle; it can depend on the state, on $\xi$, and on the field tangents). Let

$$
d\xi_x = \frac{\partial\mathbf v}{\partial\mathbf u}\xi_u\,dt,\qquad
d\xi_u = \Big[\frac{\partial\mathbf b}{\partial\mathbf x}\xi_x + \frac{\partial\mathbf b}{\partial\mathbf u}\xi_u + \delta\mathbf a - \mathbf h\Big]dt,\qquad
d\omega = (\Sigma^{-1}\mathbf h)\cdot d\mathbf W, \tag{2.4}
$$

with $\mathbf b = \mathbf a - \nu(\mathbf u - \mathbf U)$ and $\xi(0) = 0$. Then

$$
\int g\,\delta f(T) = \mathbb E\big[\omega_Tg(Z_T) + \xi_T\cdot\nabla g(Z_T)\big]. \tag{2.5}
$$

**Proof.** With additive noise, $\xi$ is the derivative of the path under the drift perturbation $\delta\mathbf a - \mathbf h$ at fixed noise. Hence $\mathbb E[\xi_T\cdot\nabla g(Z_T)] = \int_0^T\mathbb E[(\delta\mathbf a - \mathbf h)\cdot\nabla_u\varphi(t, Z_t)]\,dt$ (first variation along the flow, then the tower property). By the proof of 2.2 with $\delta\mathbf a$ replaced by $\mathbf h$, $\mathbb E[\omega_Tg(Z_T)] = \int_0^T\mathbb E[\mathbf h\cdot\nabla_u\varphi]\,dt$. The sum is $\int_0^T\mathbb E[\delta\mathbf a\cdot\nabla_u\varphi]$. $\square$

A one-step version is Gaussian integration by parts (Stein's lemma): $\mathbb E[\nabla G(\mathbf m + \sigma\sqrt{\Delta t}\boldsymbol\zeta)\cdot\mathbf h] = \mathbb E[G\,\boldsymbol\zeta\cdot\mathbf h]/(\sigma\sqrt{\Delta t})$. Any part of the derivative can be moved from $G$ onto the Gaussian density at any step (Fournié et al., Finance Stoch. 3, 391, 1999) [literature].

Special cases:

| Control | Meaning | Variance growth | Bias |
|---|---|---|---|
| $\mathbf h = 0$ | exact Lagrangian tangent (the current AD) | $\lvert\xi\rvert \sim e^{\lambda t}$ | none |
| $\mathbf h = \delta\mathbf a$ | pure Girsanov; $\xi \equiv 0$ if $\xi(0) = 0$ | $\int\lvert\delta\mathbf a\rvert^2/\sigma^2$, linear in $t$ for bounded $\delta\mathbf a$ | none |
| $\mathbf h = \mathbf h_{fb}(\xi)$, feedback (2.6) | pathwise for fast forces, weights for slow and unstable parts | $\int\lvert\mathbf h_{fb}\rvert^2/\sigma^2$ with $\xi$ bounded | none |
| $\mathbf h$ over $[t_s, t_s + \tau]$ that steers $\xi$ to 0 | handover of an existing $\xi$ into weights (Bismut-Elworthy-Li type) | $\ge\xi^T\Phi^TW_c^{-1}\Phi\xi/\sigma^2$ | none |

Feedback control. Choose per component a damped target oscillator with corner frequency $\omega_c$:

$$
\mathbf h_{fb} = \Big(\frac{\partial\mathbf b}{\partial\mathbf x} + \omega_c^2M^{-1}\Big)\xi_x + \Big(\frac{\partial\mathbf b}{\partial\mathbf u} + 2\omega_c\Big)\xi_u,\qquad M = \frac{\partial\mathbf v}{\partial\mathbf u}, \tag{2.6}
$$

so that $\ddot\xi_x + 2\omega_c\dot\xi_x + \omega_c^2\xi_x = M\delta\mathbf a$ (for slowly varying $M$). A source at frequency $\Omega \gg \omega_c$ leaves $\lvert\xi_x\rvert \approx \lvert M\delta\mathbf a\rvert/\Omega^2$ and $\lvert\mathbf h\rvert \approx \lvert\delta\mathbf a\rvert(\omega_c^2 + \lvert\partial_x\mathbf b\,M\rvert)/\Omega^2$, so the fast part stays pathwise and costs almost no variance. A slow source passes to $\mathbf h \approx \delta\mathbf a$. An unstable direction ($\partial_x\mathbf b\,M > 0$) is cancelled, so $\xi$ cannot grow exponentially [derived]. The terms $(\partial\mathbf b/\partial\mathbf x)\xi_x$ and $(\partial\mathbf b/\partial\mathbf u)\xi_u$ are what `gather_tangent` and `vay_tangent` already compute.

Handover cost [derived]. For free streaming in one component, the controllability Gramian over $\tau$ gives the minimum weight variance $\lvert\xi_u\rvert^2/(\sigma^2\tau)$ for a momentum tangent and $12\lvert\xi_x\rvert^2/(M^2\sigma^2\tau^3)$ for a position tangent. With $M = c/\gamma$ (transverse) or $c/\gamma^3$ (longitudinal), the position cost scales as $\gamma^2$ or $\gamma^6$. Consequence: convert a relativistic particle's position tangent early (at low $\gamma$, for example soon after the EV birth), or keep $\xi_z$ pathwise when it does not grow ($\dot\xi_z \propto \gamma^{-3}\xi_{u_z}$).

### 2.4 Self-consistent field coupling [derived]

The particle estimator of $\delta f_s$ is the signed measure

$$
\delta f_s^N = \sum_iw_i\big[\omega_i\,\delta_{Z_i} - \nabla_z\cdot(\xi_i\,\delta_{Z_i})\big]. \tag{2.7}
$$

Its moments are the existing deposit tangent with $\dot w_i = w_i\omega_i$ [R0, 3.4: `deposit_tangent` `dw` terms]:

$$
\delta\rho = \sum_iq w_i\big[\omega_iS(\mathbf x - \mathbf x_i) - \xi_{x,i}\cdot\nabla S\big],\qquad
\delta\mathbf J = \sum_iq w_i\big[\omega_i\mathbf v_iS + M_i\xi_{u,i}S - \mathbf v_i\,\xi_{x,i}\cdot\nabla S\big]. \tag{2.8}
$$

The field tangent solve is linear and unchanged. Unbiasedness with the closure: the increments $(\Sigma^{-1}\mathbf h)\cdot d\mathbf W$ have zero conditional mean also when $\mathbf h$ depends on the noisy $\delta\mathbf E$, because $\delta\mathbf E$ at $t$ is adapted. The map $\delta f^N \mapsto (\delta\mathbf E, \delta\mathbf B)$ is linear. So the noise expectation of $(\delta f^N, \delta\mathbf E)$ solves (2.1) and (2.2) exactly, given the primal paths [derived]. The only bias at finite $N$ comes from the randomness of the primal fields themselves (Section 3).

### 2.5 Sources, births, and boundaries [derived]

1. **Laser.** The parameter enters through the initial field tangent. At a particle in the laser $\delta\mathbf a$ contains the quiver force $\mathbf a_L/2$ (for $\theta = \ln E_L$). This is the dominant variance term for pure Girsanov (Section 4.3). With (2.6), particles in the laser keep it pathwise.
2. **EV births.** A birth at step $n$ has deterministic weight $\Delta W = W_{ion}r(1 - e^{-\Gamma(\lvert\mathbf E\rvert)\Delta t})$. Its tangent is a monopole weight source: the newborn starts with $\omega_b = \Delta\dot W/\Delta W$ (the existing birth tangent $\dot W'$ divided by the weight) and $\xi_b = \xi_{ion}$ (the existing copy, `ionization.py:762-764` [R0]). Ions stay noise-free and Lagrangian: they are heavy and laminar. The remaining ion fraction $r$ keeps its existing tangent.
3. **Newborns in the laser.** Ionization injection happens at the laser peak. The feedback control (2.6) keeps the newborn's quiver tangent pathwise. The newborn's position tangent should be handed over while its $\gamma$ is small (Section 2.3).
4. **Moving window.** The shift is $\theta$-independent. Injected plasma is cold and $\theta$-independent: $\omega = 0$, $\xi = 0$ on inflow. Particles leaving at the back take their $\omega$ and $\xi$ with them; the selection weight is 0 there [R0, 3.6]. A time-only noise window (Section 1.2) needs no mask term.
5. **Noise mask in space.** If $\sigma$ depends on $\mathbf x$, the pathwise tangent gets $d\xi_u \mathrel{+}= (\partial_x\Sigma\,\xi_x)\,d\mathbf W$, and $\mathbf h$ must vanish where $\Sigma = 0$. A particle that enters a noisy region carries its $\xi$ in; (2.6) then hands it over.

## 3. Joint-measure versus mean-field likelihood ratio

The finite-$N$ noisy PIC is one Markov chain in $(Z_1, \dots, Z_N)$. Its fields are deterministic functions of the past paths and $\theta$.

**Joint (exact finite-$N$) LR** [derived]. Differentiate the joint path density at fixed paths. The deposits of fixed paths do not depend on $\theta$, so the explicit drift tangent is the field tangent computed with $\delta\rho = \delta\mathbf J = 0$: the vacuum laser tangent $\delta\mathbf a^{vac}$. Then

$$
\frac{d}{d\theta}\mathbb E_N[G] = \mathbb E\big[(G - \bar G)\,\Omega\big],\qquad \Omega = \sum_i\int_0^T(\Sigma^{-1}\delta\mathbf a_i^{vac})\cdot d\mathbf W_i. \tag{3.1}
$$

1. Target: the exact derivative of the finite-$N$ noisy PIC expectation, including all finite-$N$ correlations. No closure, so no variance loop.
2. Variance: $\mathrm{Var}\,\Omega = N\,\overline{\int\lvert\delta\mathbf a^{vac}\rvert^2/\sigma^2}$, and $\mathrm{Var}(G) \approx s_g^2/N_{eff}$. Per run $\mathrm{Var}[(G - \bar G)\Omega] \approx s_g^2\,(N/N_{eff})\,\overline{\int\lvert\delta\mathbf a^{vac}\rvert^2}/\sigma^2$: independent of $N$, so only independent replicas $M$ reduce it, as $M^{-1/2}$. Without the control variate $\bar G$ it grows as $N$.
3. $\bar G$ must come from other replicas (leave-one-out) to keep (3.1) unbiased.
4. A mixed joint form exists: carry the exact $N$-particle tangent $\Xi$ (the current AD) with per-particle feedback $\mathbf h_i = K\xi_i$, and $\Omega = \sum_i\int\Sigma^{-1}\mathbf h_i\cdot d\mathbf W_i$. It is exact at finite $N$ and has no loop, but its per-run variance again does not fall with $N$.

**Mean-field LR** (Section 2) [derived].

1. Target: the derivative of the propagation-of-chaos limit $\mathbb E^\infty[g]$, that is, the linearized VFP (2.1). Its bias relative to $d\mathbb E_N[G]/d\theta$ is $O(1/N)$ under propagation of chaos; the constant can grow in time (Gronwall), so in trapping and chaos it should be measured, not assumed.
2. Variance per run falls as $1/N_{eff}$ if the loop of Section 4.4 is controlled. It has a cross-particle term from shared field-tangent noise that does not fall with $N$ (Section 4.2).

**Measured on the 1D cold-plasma toy** (`wt/verify_joint_lr.py`, `wt/verify_mf_girsanov.py`; periodic $L = 2\pi$, 64 cells, driver $\theta\sin x\,e^{-(t-5)^2/4}$, $\theta = 0.3$, $T = 30$; observable 1 = $\langle u^2\rangle$; reference = CRN central FD over 24 seeds):

| $N$ | $\sigma$ | Reference | Mean-field, plain | Mean-field, centered | Joint LR (300 runs) |
|---|---|---|---|---|---|
| 6400 | 0.1 | $0.373 \pm 0.004$ | $0.397 \pm 0.016$ | $0.401 \pm 0.015$ | $0.46 \pm 0.34$ |
| 25600 | 0.1 | $0.381 \pm 0.002$ | $0.379 \pm 0.009$ | not run | |
| 6400 | 0.03 | $0.279 \pm 0.009$ | $-1.9 \pm 1.3$ | $0.252 \pm 0.045$ | $0.54 \pm 0.24$ |
| 25600 | 0.03 | $0.287 \pm 0.006$ | $0.21 \pm 0.07$ | not run | not run |
| 6400 | 0.01 | $0.485 \pm 0.0003$ | $-7\cdot10^{17} \pm 2\cdot10^{18}$ | $0.492 \pm 0.045$ | |
| 25600 | 0.01 | $0.485 \pm 0.0002$ | $75 \pm 110$ | $0.477 \pm 0.020$ (12 seeds) | |

The mean-field values are means over 24 seeds (12 where marked) with their SE. At $N = 25600$, $\sigma = 0.01$ the centered estimator also matches the other two observables: $-0.586 \pm 0.024$ against $-0.599$, and $0.106 \pm 0.028$ against $0.105$ [measured]. "Not run": the runs were stopped by the host and not repeated. At $N = 6400$, $\sigma = 0.1$ the joint SE per run is about $0.34\sqrt{300} \approx 5.9$, against $0.016\sqrt{24} \approx 0.08$ for the mean-field estimator, a factor of about 75 [measured].

**Which to use** [derived]. Use the mean-field form, with charge-neutral increments (Section 4.5), whenever the loop rate satisfies $\Gamma T_n \lesssim 1$. The joint form is the fallback only for small $N$ with many cheap replicas; on Twin-S its quiver term $N\int\lvert\delta\mathbf a^{vac}\rvert^2/\sigma^2$ makes it useless (Section 4.3).

## 4. Variance

### 4.1 Weight variance

By the Itô isometry [derived]:

$$
\mathbb E[\omega_T^2] = \mathbb E\int_0^T\lvert\Sigma^{-1}\mathbf h_t\rvert^2dt. \tag{4.1}
$$

For pure Girsanov this is $\int\lvert\delta\mathbf a\rvert^2/\sigma^2$, linear in time for bounded $\delta\mathbf a$. A lower bound for any scheme: to represent a shift $\xi_u$ of a distribution whose noise width is $u_n$, $\mathrm{Var}\,\omega \ge \lvert\xi_u\rvert^2/u_n^2$ (Fisher information of a Gaussian location family; $\chi^2$ divergence of shifted Gaussians) [derived]. Define the conversion inefficiency $R = \mathrm{Var}\,\omega\cdot u_n^2/\lvert\xi_u^{net}\rvert^2 \ge 1$, with $\xi_u^{net}$ the net pathwise momentum response before chaotic stretching.

### 4.2 Variance of a moment gradient

For a normalized moment $m = \sum_iw_ig_i/\sum_iw_i$ the estimator is $\widehat{\delta m} = \sum_iw_i\big[\omega_i(g_i - m) + \xi_i\cdot\nabla g_i\big]/\sum_iw_i$. The subtraction of $m$ is the quotient rule and is also the control variate. For ratio moments (centroids and spreads with the selection weight), the same quotient rule applies to numerator and denominator [derived]. Then

$$
\mathrm{Var}\,\widehat{\delta m} \approx \frac{\mathbb E[\omega^2(g - m)^2]}{N_{eff}} + \frac{1}{W^2}\sum_{i\ne j}w_iw_j\,\mathrm{Cov}\big(\omega_i(g_i - m), \omega_j(g_j - m)\big). \tag{4.2}
$$

The first term gives the relative standard error

$$
\frac{\mathrm{SE}}{\lvert\delta m\rvert} \approx \sqrt R\;\frac{\lvert\xi_u^{net}\rvert}{u_n}\;\frac{s_g}{\lvert\delta m\rvert\sqrt{N_{eff}}}, \tag{4.3}
$$

with $s_g$ the spread of $g$ over the selected particles. For a centroid, $\lvert\delta m\rvert \approx \lvert\xi_u^{net}\rvert$, so $\mathrm{SE}/\lvert\delta m\rvert \approx \sqrt R\,s_g/(u_n\sqrt{N_{eff}})$ [derived]. The second term is the collective part: field-tangent noise is shared by nearby particles and does not average out with $N$.

Twin-S example [estimated]: bunch $s_{u_z} = 0.3$ to 1, $u_n = 0.03$, $N_{eff} = 10^4$ to $10^5$ selected markers, $R = 1$ to 4: $\mathrm{SE}/\lvert\delta m\rvert \approx 0.03$ to 0.7. The inputs $s_{u_z}$ and $N_{eff}$ are not measured.

### 4.3 The laser quiver term on Twin-S [estimated]

For $\theta = \ln E_L$, a particle in the laser has $\delta\mathbf a \approx \mathbf a_L/2$ with amplitude $a_0\omega_0/2 = 29.6\,\omega_p$ (in $u$ per $\omega_p^{-1}$). The intensity envelope gives $\int\mathrm{env}^2dt = \tau\sqrt{\pi/2} = 3.2\,\omega_p^{-1}$. So $\int\lvert\delta\mathbf a\rvert^2dt \approx \tfrac12(29.6)^2\cdot3.2 \approx 1.4\cdot10^3$, and with noise over the full run ($\sigma^2 = u_n^2/53$): $\mathrm{Var}\,\omega \approx 7.6\cdot10^4/u_n^2$. At $u_n = 0.1$, $\mathrm{std}\,\omega \approx 2.8\cdot10^3$. The net momentum response after the pulse is of order the ponderomotive response, about $1\,\omega_p$ times $3\,\omega_p^{-1}$, so $R \sim 10^3$ [estimated]. Pure Girsanov in the laser is not viable. The joint form has the same term times $N$.

The 1D harness `r2_pilot.py` has a cycle-averaged ponderomotive driver; it cannot show this term. A test of option C on that harness alone would be optimistic.

With the feedback control (2.6) and $\omega_c \approx 1$ to $3\,\omega_p$, the quiver contribution to $\mathbf h$ falls by about $(\omega_c/\omega_0)^2 \approx 5\cdot10^{-3}$ to $4\cdot10^{-2}$, and to the variance by its square [derived scaling]. Measured on a chaotic noisy Duffing oscillator with a fast force ($B = 3$, $\Omega = 15$, $\omega_c = 0.5$, $\sigma = 0.1$, $10^5$ paths; `wt/verify_mixed_feedback.py`): pathwise $-119 \pm 133$; pure Girsanov $-0.06 \pm 0.22$; feedback $-0.278 \pm 0.037$; reference CRN FD $-0.225$ and independent FD $-0.240 \pm 0.032$. The feedback cuts the variance by a factor of 35 and agrees with the two FD values within 0.8 and 1.4 combined SE. Without the fast force ($B = 0$), pure Girsanov is better ($\pm 0.022$ against $\pm 0.036$) [measured]. So the feedback is the right tool only where fast forces are present.

### 4.4 Collective amplification

Two mechanisms can make $\mathrm{Var}\,\omega$ grow exponentially.

**(a) Physical collective instability** [derived]. If the linearized VFP (2.1)-(2.2) has a growing mode (Raman, hosing, beam-plasma), then $\delta\mathbf E$ grows, and so does the true sensitivity of $\mathbb E[J]$. Every exact method sees it. It is not a defect of option C. Single-particle Lyapunov stretching does not enter: the weights are scalars carried along paths, and $\int\omega^2f$ changes only through (4.1).

**(b) Noise feedback loop of the mean-field estimator** [derived scaling, measured constant]. Weight noise deposits a random charge $\delta\rho_{noise}$. Its field enters $\delta\mathbf a$ at all particles, and (4.1) adds $\mathrm{Var}(\delta\mathbf a_{noise})/\sigma^2$ per unit time. With $\mathrm{Var}(\delta E_{noise}) \approx c\,\mathrm{Var}\,\omega/n_m$:

$$
\frac{d\,\mathrm{Var}\,\omega}{dt} \approx \frac{\lvert\delta\mathbf a_{mean}\rvert^2}{\sigma^2} + \Gamma\,\mathrm{Var}\,\omega,\qquad \Gamma \approx \frac{c}{n_m\sigma^2}. \tag{4.4}
$$

Measured in the cold 1D toy (`wt/verify_mf_girsanov.py`, late-time growth from $t = 15$ to 30):

| $N$ | $\sigma$ | $\Gamma$ plain [$\omega_p$] | $\Gamma n_m\sigma^2$ | $\Gamma$ centered [$\omega_p$] | $\mathrm{Var}\,\omega(30)$ plain / centered |
|---|---|---|---|---|---|
| 6400 | 0.1 | $\le 0.03$ (linear) | | $\le 0.03$ | 120 / 109 |
| 6400 | 0.03 | 0.25 | 0.23 | 0.107 | $8.1\cdot10^5$ / $2.1\cdot10^4$ |
| 25600 | 0.03 | 0.11 | 0.40 (linear part not removed) | not run | $2.9\cdot10^4$ / not run |
| 6400 | 0.01 | 2.9 | 0.30 | 0.156 | $1.7\cdot10^{37}$ / $5.2\cdot10^5$ |
| 25600 | 0.01 | 0.62 | 0.25 | 0.090 | $3.0\cdot10^{10}$ / $1.6\cdot10^5$ |

So $c \approx 0.25$ in this geometry, and the plain rate scales as $1/(n_m\sigma^2)$ [measured]. In 3D the field noise of white charge noise is dominated by the smallest resolved scale ($\int r^{-4}r^2dr$), so the effective $n_m$ is the marker count in a volume of about $k_p^{-2}\Delta$, not $k_p^{-3}$ [derived]. Twin-S has about 170 to 4200 markers per $k_p^{-3}$ per species (16 per cell, from $r = 5\,\mu$m to $0.2\,\mu$m) [computed]. With the window $T_n = 14.6$ and $\sigma^2 = u_n^2/T_n$, $\Gamma T_n = cT_n^2/(n_mu_n^2) \approx 53/(n_mu_n^2)$. At $u_n = 0.03$ and $n_m = 10^2$ to $10^3$ this is 60 to 600 e-folds [estimated]. The plain mean-field estimator therefore fails on Twin-S at any $u_n$ that keeps the physics bias small.

**Charge-neutral increments remove most of the loop** [derived, measured]. The source $-\nabla_u\cdot(f\delta\mathbf a)$ has zero charge in every cell. The plain increments have zero charge only in expectation. Subtracting the local shape-weighted mean of the increments,

$$
\Delta\omega_i \leftarrow \Delta\omega_i - \frac{\sum_jw_jS_{ij}\Delta\omega_j}{\sum_jw_jS_{ij}}\quad\text{(gathered to particle } i\text{)}, \tag{4.5}
$$

enforces this per cell. It also removes the current noise $\bar{\mathbf v}\sum w\Delta\omega$ carried by the mean flow. The remaining noise current is $\sum w\,\omega(\mathbf v - \bar{\mathbf v})$, which is small in a cold flow: $\mathrm{std}\,\omega\cdot\sigma_v \sim \lvert\delta\mathbf a\rvert t^{3/2}\cdot t^{1/2}$ is independent of $\sigma$ [derived]. The measured centered rate is 0.09 to 0.16, depends only weakly on $\sigma$, and falls with $N$ (0.156 to 0.090 for $4\times$ markers at $\sigma = 0.01$); the late growth is closer to a power law than to an exponential; at $\sigma = 0.01$ the centered estimator gives $0.492 \pm 0.045$, $-0.591 \pm 0.054$, $0.042 \pm 0.078$ against FD $0.485$, $-0.600$, $0.107$ [measured]. Bias of (4.5): the self term $j = i$ shrinks the estimate by about $1/N_{cell}$; a leave-one-out mean removes it at the cost of an $O(1/N_{cell})$ residual charge [derived]. After orbit crossing $\mathbf v - \bar{\mathbf v}$ is large (multi-stream), so (4.5) helps less there. The toy at $\theta = 0.3$ has no orbit crossing. This is the first thing the evaluators must test in the trapping regime.

### 4.5 Variance-control options, ranked [derived unless marked]

1. **Charge-neutral increments (4.5)**: removes the loop in the cold, laminar regime [measured]. Cost: one extra scalar deposit and gather per step (same kernels as $\rho$). Optionally also match the current moment per cell, $\sum_iw_iS\Delta\omega_i\mathbf v_i = \sum_iw_iSM_i\mathbf h_i\Delta t$, which is the exact one-step source current.
2. **Control variate $g - m$**: built into the quotient rule (4.2). Mandatory. Without it the variance carries $m^2\,\mathrm{Var}\,\omega$.
3. **Feedback control (2.6)**: removes the fast-force variance [measured, 35x]. Use only where fast forces act (laser region, newborns); elsewhere $\mathbf h = \delta\mathbf a$ is better [measured].
4. **Noise window**: $\Gamma T_n \propto T_n^2/u_n^2$, so a short window (the chaotic phase, about step 1450 to 2000) is much cheaper than noise from $t = 0$. Before the window the existing exact tangent is laminar and verified [R0].
5. **Antithetic pairs**: two markers at the same initial state with $\boldsymbol\zeta' = -\boldsymbol\zeta$. While their paths stay close, $\omega' \approx -\omega$, and the pair is an exact local charge-neutral increment pair. The benefit lasts until the noise-induced separation is stretched to the observable scale, about $\lambda^{-1}\ln(\ell_g/u_n)$; in the chaotic window that is short [estimated]. It doubles the marker count.
6. **Low-pass filter on the weight deposit only**: raises the effective $n_m$ in (4.4); bias $O((k\ell_f)^2)$ on the field tangent at scale $\ell_f$; keep $\ell_f \ll k_p^{-1}$.
7. **Rao-Blackwell regression** $\omega_i \to \mathbb E[\omega\mid Z_i]$ on a slab polynomial basis: removes the conditional variance, but it is the R-v Galerkin reset with its basis-truncation bias [R0, A.4]. With noise, $\delta f/f$ is smooth at the scale $u_n$, so the basis must resolve $u_n$. Not recommended first.
8. **Truncated-memory weights** (Warren and Allen 2012) apply to stationary problems only. The wake problem is transient, so truncation is biased. Rejected.

## 5. Discrete-time version

### 5.1 One FBPIC step [derived]

Step order as in [R0, 3.3]. Add the noise after the Vay push and before the first half position push:

1. Gather $(\mathbf E^n, \mathbf B^n)$ and the field tangents at $\mathbf x^n$.
2. Primal: $\mathbf m^n = V(\mathbf u^{n-1/2}, \mathbf E^n, \mathbf B^n)$ (Vay). Draw $\boldsymbol\zeta^n \sim N(0, I_3)$ (counter-based RNG keyed by particle id and step, so that sorting does not change the stream). Set $\mathbf u^{n+1/2} = \mathbf m^n + \Sigma\sqrt{\Delta t}\,\boldsymbol\zeta^n$.
3. Tangent: $\delta\mathbf m^n = J_u\xi_u^{n-1/2} + J_x\xi_x^n + J_F(\delta\mathbf E^n, \delta\mathbf B^n)$. This is the existing `vay_tangent` output. Choose $\mathbf h^n$ (momentum displacement per step) from $\{0, \delta\mathbf m^n, \text{feedback}\}$.
4. $\xi_u^{n+1/2} = \delta\mathbf m^n - \mathbf h^n$ (plus $(\partial_x\Sigma\,\xi_x^n)\sqrt{\Delta t}\boldsymbol\zeta^n$ for a spatial mask). $\Delta\omega^n = (\Sigma^{-1}\mathbf h^n)\cdot\boldsymbol\zeta^n/\sqrt{\Delta t}$, then (4.5) if enabled; $\omega^{n+1/2} = \omega^{n-1/2} + \Delta\omega^n$.
5. Position half pushes, EV with births (Section 2.5), J deposit at $t^{n+1/2}$, $\rho$ deposit at $t^{n+1}$, field and field-tangent update: unchanged.

Exactness: the transition density of $\mathbf u^{n+1/2}$ given the past is $N(\mathbf m^n, \Sigma\Sigma^T\Delta t)$. Its score in the direction $\mathbf h^n$ is $\mathbf h^n\cdot(\Sigma\Sigma^T\Delta t)^{-1}(\mathbf u^{n+1/2} - \mathbf m^n) = (\Sigma^{-1}\mathbf h^n)\cdot\boldsymbol\zeta^n/\sqrt{\Delta t}$. By Stein's lemma per step and the tower property, $\mathbb E[\omega g + \xi\cdot\nabla g]$ is the exact derivative of the discrete noisy chain, with no time-step bias in the derivative (the bias is only that of the discrete model) [derived]. Units: with $\mathbf h^n = \delta\mathbf a\,\Delta t$, $\Delta\omega = \delta\mathbf a\cdot\boldsymbol\zeta\sqrt{\Delta t}/\sigma$, as in the continuum.

$K$ parameters: the paths and $\boldsymbol\zeta^n$ are shared; each parameter has its own $(\xi^{(k)}, \omega^{(k)})$ and field tangent. The weight update is one FMA per parameter per particle.

### 5.2 Charge and current consistency [derived]

1. The weight-transport part is consistent: $\delta\rho^{n+1}$ uses $(\omega^{n+1/2}, \mathbf x^{n+1})$, $\delta\mathbf J^{n+1/2}$ uses $(\omega^{n+1/2}, \mathbf x^{n+1/2}, \mathbf v^{n+1/2})$, so the curl-free correction sees the same weights on both sides [R0, 3.5].
2. The increment $\sum_iqw_iS(\mathbf x^n - \mathbf x_i)\Delta\omega_i^n$ is a charge jump with no current. It has zero conditional mean but not zero value. Option (i): keep $\delta\rho_{prev}$ as deposited with $\omega^{n-1/2}$; the curl-free correction then turns the jump into a longitudinal $\delta\mathbf J$, and Gauss's law holds for the noisy $\delta\rho$. This is the plain estimator, unbiased, with the loop of Section 4.4. Option (ii): re-deposit $\delta\rho_{prev}$ with $\omega^{n+1/2}$; then the field never sees the jump, but the jump still enters later through $\bar{\mathbf v}\sum w\Delta\omega$ in $\delta\mathbf J$. Option (iii): the charge-neutral increments (4.5) make the jump zero per node up to the shape mismatch; with the exact node projection (solve the charge Gram system per slab, tridiagonal per direction for linear shapes) it is zero to rounding. **Use (iii)**; it is the discrete form of the zero-charge property of the source.
3. The $k = 0$ mode of the curl-free correction is zeroed in FBPIC [R0, 3.5]. With (iii) the total increment charge is zero, so this does not matter.

## 6. Cost per extra parameter on Twin-S [estimated]

Inputs: primal step 43 ms with EV, 9.2 ms without EV; as-built tangent $c_1 = 2.47$ primal runs per parameter; P7 deposits 24 + 14 ms primal against 81 + 40 ms tangent [R0, 3.4, 6.1].

| Variant | Per-parameter work per step | $c_1$ (primal runs per parameter) | Accurate? |
|---|---|---|---|
| Pure Girsanov, all electrons, $t = 0$ to $T$ | field-tangent gather without $\xi$, weight FMA, one $K$-channel deposit of $w\omega$ for $\rho$ and $\mathbf J$ (fused primal path), field tangent solve, EV birth weights | 0.4 to 0.8 | No: quiver variance (4.3) and loop (4.4) |
| Hybrid as built: exact tangent to $t_s \approx 1450$, then mixed (2.4) with (2.6) and (4.5) | existing tangent plus RNG, feedback, weight update, one extra scalar deposit for (4.5) | about 2.5 | Unknown: needs Sections 7 items 1 to 4 |
| Hybrid after the dense-cell deposit-tangent rewrite [R0, 6.1] | as above | 1.0 to 1.3 | as above |
| Bunch-only weights in the window, background Lagrangian | background tangent about 0.2 EV-run-eq [R0, 6.1] plus bunch weights and deposit | 0.8 to 1.3 | Only if background growth is bunch-driven |

Added primal cost: three normals per marker per step from a counter-based generator, about the cost of one push kernel, below 5% of 43 ms. Statistical repeats multiply all of the above by $M$ if one run gives too large an SE (4.3). $K$-batching shares the shape factors in the weight deposit; the existing deposit tangent already spills registers beyond small $K$ (48K doubles per thread for J), so tile over $K$ in groups of 2 to 4 [R0, 6.1].

Conclusion: the only variant with $c_1$ clearly below 1 is pure Girsanov, and it fails on variance. Credible variants tie one-sided FD ($c_1 = 1$) at best. This repeats the R0 cost finding for a different reason: R0's cheap variants failed on cold particles; here the cheap variant fails on the fast laser force and the variance loop.

## 7. What the evaluators must test (falsifiable predictions)

1. **Loop and centering in the trapping regime.** On `r2_pilot.py` with noise in a late window, measure $\mathrm{Var}\,\omega(t)$ for plain and centered increments at two $N$ and two $\sigma$. Prediction: plain $\Gamma \propto 1/(n_m\sigma^2)$; centered $\Gamma$ nearly independent of $\sigma$. Kill if centered $\Gamma T_n > 3$ at $u_n \le 0.1$ after trapping.
2. **Agreement with a trusted reference.** The reference is the derivative of the noisy model's $\mathbb E[J]$: CRN central FD over many seeds at an $h$ above the roughness scale. Pass if the weight estimate is within $2\,$SE for all moments.
3. **Smoothing scale.** Report every option-C column against $u_n$, as FD columns are reported against $h$. Check the equivalence $h_{eq} \approx u_n/\lvert\partial u/\partial\theta\rvert$ (Section 1.3, item 4).
4. **Where the growth lives.** Log per-species and per-component median $\xi$ growth on Twin-S. If the background growth is bunch-driven, bunch-only noise suffices. If the bunch growth is transverse, $\sigma_\parallel = 0$ removes the energy-spread bias.
5. **Quiver term.** Do not accept a 1D pilot result as evidence for Twin-S unless the harness has a resolved laser oscillation or the feedback (2.6) is shown to remove a fast force of the Twin-S ratio $\omega_0/\omega_p \approx 15$.

## References

1. Chung, Bond, Cyr, Freund, "A particle-pdf method for sensitivity analysis of plasma kinetics", J. Comput. Phys. 400, 108969 (2020), OSTI 1570259.
2. I. V. Girsanov, Theory Probab. Appl. 5, 285 (1960).
3. P. Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer (2004), Ch. 7 (pathwise and likelihood-ratio estimators).
4. E. Fournié, J.-M. Lasry, J. Lebuchoux, P.-L. Lions, N. Touzi, "Applications of Malliavin calculus to Monte Carlo methods in finance", Finance Stoch. 3, 391 (1999).
5. P. B. Warren, R. J. Allen, "Malliavin weight sampling for computing sensitivity coefficients in Brownian dynamics simulations", Phys. Rev. Lett. 109, 250601 (2012).
6. K. D. Elworthy, X.-M. Li, "Formulae for the derivatives of heat semigroups", J. Funct. Anal. 125, 252 (1994).
7. T. Katsouleas, W. B. Mori, "Wave-breaking amplitude of relativistic oceanic plasma waves", Phys. Rev. Lett. 61, 90 (1988).

## Appendix. Scripts and outputs (all in `wt/`)

| File | Content |
|---|---|
| `verify_mf_girsanov.py` | 1D periodic cold electrostatic plasma, mean-field Girsanov weights, plain and centered increments, CRN FD reference |
| `verify_joint_lr.py` | joint-measure LR (3.1) on the same toy, leave-one-out control variate |
| `verify_mixed_feedback.py` | chaotic noisy Duffing oscillator with a fast force: pathwise, pure Girsanov, feedback (2.6), FD |
| `scan_lin.jsonl`, `scan_lin2.jsonl`, `scan_lin3.jsonl` | mean-field scan over $N$, $\sigma$, variant (`summ.py` prints it); some $N = 25600$ centered runs were stopped by the host |
| `joint.jsonl` | joint LR results (`joint_signbug.jsonl`: first run with a sign error in the score, superseded) |
| `mixed.jsonl` | feedback results ($B = 3$ and $B = 0$ at $\sigma = 0.1$; the $\sigma = 0.03$ run was stopped by the host) |
