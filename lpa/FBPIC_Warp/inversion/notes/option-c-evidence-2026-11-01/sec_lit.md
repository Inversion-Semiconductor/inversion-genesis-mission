# Option C: literature and novelty check

Task key: `lit`. Date: 2026-11-01. Scope: likelihood-ratio (LR), Girsanov, and Malliavin weights for interacting particle systems; weighted-particle methods for linearized kinetic equations; noise regularization of chaotic sensitivity; differentiable PIC.

Labels: [lit] = literature, with citation. [derived] = short derivation in this note. [estimated] = judgement, not measured. No measurements were made for this section.

Search method: web search and abstract or full-text reads on 2026-11-01. Full text was read for Warren and Allen (2012) and for Bond et al. (SAND2022-12721, SAND2019-11524R). The other entries rest on abstracts, author pages, or citing papers. The search was not exhaustive.

## 1. Verdict

1. Every ingredient of option C is published. Per-particle Girsanov weights in interacting-particle simulations are standard in soft matter [lit: Warren and Allen 2012]. Malliavin integration by parts for McKean-Vlasov laws is proved [lit: Baños 2018; Crisan and McMurray 2018]. Noise as a regularizer for chaotic linear response is old and active [lit: Thuburn 2005; Ni 2025].
2. The combination was not found. No paper found builds a Monte Carlo solver for the linearized Vlasov-Fokker-Planck-Maxwell equation from per-particle Girsanov weights closed through a self-consistent field tangent. No paper found uses such weights to make PIC gradients cold-safe and chaos-robust.
3. The novelty is algorithmic and applied, not mathematical. A probabilist will read the mean-field estimator as a particle discretization of the McKean-Vlasov Bismut formula. A plasma physicist will read it as a $\delta f$ scheme whose weight source uses the noise instead of $\partial_u \ln f_0$. Both readings are correct [derived, Section 3].
4. The literature supports the method in principle and names its main risk. Pure kernel differentiation (the LR method) "can be expensive for small noise" [lit: Ni 2025, JNS, abstract]. Its variance is "inversely proportional to the scale of the noise" [lit: Ni 2025, arXiv:2503.00718]. Option C needs small noise for small bias, so the bias-variance trade-off is the central question.
5. The most likely holder of unpublished prior art is the Sandia group (Bond, Lehoucq, Franke, Cyr). They have Malliavin estimators for particle transport and co-wrote the Chung et al. PIC paper. Their public list (checked 2026-11-01) has no Malliavin-for-PIC paper [lit: Bond publication page].

## 2. Map of the ingredients

| Ingredient of option C | Closest published work | Gap relative to option C |
|---|---|---|
| Per-particle LR weight from the noise the particle received | Warren and Allen 2012; Szamel 2017; Garnier, Ip, Mertz 2024 | One joint weight for all $N$ particles; no field closure; no plasma |
| Mean-field closure: weights deposit $\delta\rho$, $\delta J$, drive field tangents | Baños 2018; Crisan and McMurray 2018; Ren and Wang 2019 (theory) | No algorithm for Vlasov-Maxwell; derivatives mostly in the initial law |
| Weighted particles for a linearized kinetic equation | $\delta f$ PIC (Parker and Lee 1993; Hu and Krommes 1994); collisional $\delta f$ (Brunner, Valeo, Krommes 1999); deviational MC (Baker and Hadjiconstantinou 2005; Péraud and Hadjiconstantinou 2011, 2015); Chung et al. 2020 | Weight source needs $\partial_u f_0$ or a reference Maxwellian; fails for a cold sheet |
| Noise as an explicit regularizer for chaotic sensitivity | Lea, Allen, Haine 2000; Eyink, Haine, Lea 2004; Thuburn 2005; Hairer and Majda 2010; Ni 2025 | Low-dimensional or single-SDE; no mean-field particle system |
| Score terms inside particle Monte Carlo sensitivity | Caflisch, Silantyev, Yang 2021; Yang, Silantyev, Caflisch 2023; Caflisch, Chen, Yang 2026; Bond et al. 2019, 2022 | DSMC or linear transport; reverse mode; no self-consistent field |
| Randomized smoothing against chaos | Nesterov and Spokoiny 2017; Parmas et al. 2018; Suh et al. 2022; Metz et al. 2021 | Smoothing in parameter space; cost per parameter is not below one simulation |

## 3. Two identities that place option C in the literature [derived]

Model. Each particle obeys $\mathrm{d}u_i = F_\theta(x_i,u_i,t)\,\mathrm{d}t - \gamma u_i\,\mathrm{d}t + \sigma\,\mathrm{d}W_i$, with $\sigma=\sqrt{2D}$ and optional drag $\gamma$. The force $F_\theta$ is the self-consistent Lorentz force plus the laser.

**Identity 1: option C is a $\delta f$ method with a stochastic weight source.** Perturb the drift by $\epsilon\,\delta F$. Girsanov gives the path density derivative $\omega_i(t)=\int_0^t \sigma^{-1}\delta F_i\cdot\mathrm{d}W_i$. For any test function $g$, $\mathbb{E}[g(Z_t)\,\omega(t)] = \int g\,\delta f\,\mathrm{d}z$. Hence the conditional mean satisfies

$$\mathbb{E}[\omega(t)\mid Z_t=z]\,f(z,t)=\delta f(z,t).$$

This is the analog of Warren and Allen's Eq. (4), $\langle q_\lambda\rangle_{\{r\}}=\partial\ln P/\partial\lambda$ [lit: Warren and Allen 2012]. The linear $\delta f$ scheme instead integrates $\mathrm{d}w/\mathrm{d}t=-\delta F\cdot\partial_u\ln f_0$ along characteristics [lit: Parker and Lee 1993; Hu and Krommes 1994]. Both give the same $\delta f$ in the limit of many markers. Option C moves the $u$-derivative from $f$ onto the Gaussian noise kernel by integration by parts. This is why it needs no $\nabla_u f$ and stays defined for a cold initial sheet when $\sigma>0$.

The price is variance. $\mathrm{Var}\,\omega_i(t)=\mathbb{E}\int_0^t|\delta F_i|^2/\sigma^2\,\mathrm{d}s$. This grows linearly in $t$ for bounded $\delta F$ and as $\sigma^{-2}$. The $\sigma^{-2}$ scaling matches Ni's statement for kernel differentiation [lit: Ni 2025, arXiv:2503.00718]. The linear growth matches the random-walk behavior of $q_\lambda$ in Warren and Allen (2012, p. 3) [lit] and the "weight spreading" of collisional $\delta f$ [lit: Brunner, Valeo, Krommes 1999].

**Identity 2: the joint LR needs no field tangent; the mean-field LR does.** The $N$-particle path density factors into Gaussian transition kernels conditioned on the full state. Its $\theta$-score is $S=\sum_i\int\sigma^{-1}\partial_\theta F_i^{\mathrm{expl}}\cdot\mathrm{d}W_i$. Only the explicit $\theta$ dependence enters, because the field is a deterministic function of the conditioning state. This estimator is exact for finite $N$. It is the Warren-Allen estimator: their shear example uses one weight $\mathrm{d}q/\mathrm{d}t=(2k_BT)^{-1}\sum_{i=1}^N y_i\eta_{x,i}$ [lit: Warren and Allen 2012, p. 3].

Its variance scales badly with $N$ in the PIC regime. For the laser parameter, $\partial_\theta F_i^{\mathrm{expl}}\neq0$ for most particles, so $\mathrm{Var}\,S\propto N$. The observable is an empirical mean with $\mathrm{Var}\,\hat J\propto N^{-1}$. With a baseline, $\mathrm{Var}[(\hat J-\bar J)S]\approx\mathrm{Var}\,\hat J\,\mathrm{Var}\,S=O(1)$. One run therefore gives an $O(1)$ relative error that does not fall with $N$ [derived, sketch]. The mean-field LR uses per-particle weights $\omega_i$ and the estimator $N^{-1}\sum_i g(Z_i)\omega_i$. Its variance is $O(N^{-1})$ if the $\omega_i$ are weakly correlated, at the cost of a finite-$N$ bias [estimated]. Entropic propagation-of-chaos bounds suggest that bias is $O(N^{-1})$ for smooth interactions [lit: Lacker 2023, k-marginal entropy $O((k/n)^2)$; extrapolation to PIC is estimated]. The full derivation belongs to the derivation subagents.

**Consequence for variance growth.** In the mean-field LR, $\delta F$ comes from the field tangent, which the weights drive. If the linearized Vlasov-Fokker-Planck operator has a growing mode, $\delta F$ grows, and so does $\mathrm{Var}\,\omega$. Linear response of McKean-Vlasov systems breaks down at phase transitions, where the linearized operator loses stability [lit: Lucarini, Pavliotis, Zagli 2020]. Trapping and wave breaking in the wake are candidates for such amplification [estimated]. This is the one route by which option C can still grow exponentially.

## 4. Close works, one by one

### 4.1 Likelihood-ratio and Malliavin weights in particle simulations

**Warren and Allen, "Malliavin weight sampling for computing sensitivity coefficients in Brownian dynamics simulations", PRL 109, 250601 (2012), [arXiv:1207.4691](https://arxiv.org/abs/1207.4691).** [lit, full text read]
- What: an auxiliary weight $q_\lambda$ is updated with the same random numbers as the positions. Then $\partial\langle A\rangle/\partial\lambda=\langle A\,q_\lambda\rangle$ from an unperturbed run. Many parameters cost "marginal additional cost". Interacting overdamped particles in a trap under shear are tested.
- Difference: one joint weight per system, summed over all particles. No mean-field linearization and no field tangent. Overdamped Brownian dynamics with physical temperature, not a cold Vlasov plasma. For steady states they use a windowed correlation $\langle A(t)\,\Delta q(t,t_0)\rangle$ because $q$ random-walks.
- Solves our problem: no. It is the joint-measure variant of option C, applied to small $N$.

**Warren and Allen, "Malliavin weight sampling: a practical guide", Entropy 16, 221 (2014); "Steady-state parameter sensitivity in stochastic modeling via trajectory reweighting", J. Chem. Phys. 136, 104106 (2012).** [lit, abstracts] Practical recipes and steady-state variants. Same gap as above.

**Szamel, "Evaluating linear response in active systems with no perturbing field", EPL 117, 50010 (2017), [arXiv:1607.00326](https://arxiv.org/abs/1607.00326); Garnier, Ip, Mertz, "Sensitivity analysis of colored noise-driven interacting particle systems", [arXiv:2406.11454](https://arxiv.org/abs/2406.11454) (2024).** [lit, abstracts] Extend weight sampling to Ornstein-Uhlenbeck (colored) noise and to many interacting particles with a screened Coulomb potential. Relevant if option C uses Ornstein-Uhlenbeck noise in $u$. Not mean-field, no fields, not chaotic-regularization oriented.

**Bond, Franke, Lehoucq, Smith, McKinley, "Sensitivity analyses for Monte Carlo sampling-based particle simulations", SAND2022-12721 (2022), DOI 10.2172/1889334; Bond, Franke, Lehoucq, Smith, SAND2019-11524R (2019), DOI 10.2172/1568989.** [lit, full text read]
- What: likelihood-ratio or Malliavin estimators reuse the forward trajectories. Tested on linear neutron transport (jump SDE) and on Brownian dynamics of 256 Lennard-Jones particles. They report that the Malliavin estimator has about half the variance of optimal correlated FD, and that its extra runtime is "negligible" when computed with the force.
- Difference: linear transport, or a joint weight for a small interacting system. No self-consistent field closure. No chaos discussion.
- Solves our problem: no. It is supporting evidence that per-particle weights are cheap.

**Fournié, Lasry, Lebuchoux, Lions, Touzi, Finance and Stochastics 3, 391 (1999); Gobet and Munos, SIAM J. Control Optim. 43 (2005); Chen and Glasserman, "Malliavin Greeks without Malliavin calculus", Stoch. Proc. Appl. 117 (2007).** [lit, from memory and search snippets] Foundations of LR and Malliavin weights for SDEs. Chen and Glasserman show that the Euler-scheme LR equals the Malliavin weight. That supports using the discrete Girsanov weight per time step.

### 4.2 McKean-Vlasov theory

**Baños, "The Bismut-Elworthy-Li formula for mean-field stochastic differential equations", Ann. IHP Probab. Stat. 54, 220 (2018), [arXiv:1510.06961](https://arxiv.org/abs/1510.06961); Crisan and McMurray, "Smoothing properties of McKean-Vlasov SDEs", PTRF 171 (2018), [arXiv:1702.01397](https://arxiv.org/abs/1702.01397).** [lit, abstracts]
- What: integration-by-parts formulas on Wiener space for laws of McKean-Vlasov SDEs. They cover derivatives in a real variable and in the measure (Lions derivative), for uniformly elliptic noise.
- Difference: derivatives in the initial state or initial law, not a drift parameter entering through a field. Theory only, with no particle algorithm or variance analysis. Uniform ellipticity is assumed; our noise acts only on $u$ (hypoelliptic).
- Solves our problem: no. It is the mathematical basis of the mean-field LR and should be cited for it.

**Ren and Wang, J. Differential Equations (2019), [arXiv:1809.06068](https://arxiv.org/abs/1809.06068); Bao, Ren, Wang, JDE 282 (2021); Zhao, "Propagation of chaos for derivatives of McKean-Vlasov SDEs", [arXiv:2601.20350](https://arxiv.org/abs/2601.20350) (2026).** [lit, abstracts] Bismut formulas for Lions derivatives with random initial data and memory. Zhao proves that derivative flows and the Bismut formula of the $N$-particle system converge to the mean-field ones, with rates. That result is the closest theory for "mean-field LR approximates the true derivative as $N$ grows". It treats initial-value and noise perturbations, not drift parameters.

**Lacker, "Hierarchies, entropy, and quantitative propagation of chaos for mean field diffusions", Probab. Math. Phys. 4, 377 (2023).** [lit, abstract] Relative entropy between $k$-particle marginals and the product measure is $O((k/n)^2)$. Useful for the joint-versus-mean-field bias argument. It does not address sensitivities.

**Lucarini, Pavliotis, Zagli, "Response theory and phase transitions for the thermodynamic limit of interacting identical systems", Proc. R. Soc. A (2020), [arXiv:2008.10912](https://arxiv.org/abs/2008.10912).** [lit, abstract]
- What: linear response of mean-field Fokker-Planck systems, with a self-consistent correction for the mutual interaction. A breakdown of linear response marks a phase transition of the McKean-Vlasov equation.
- Relevance: this is the collective-mode amplification risk for option C. It also shows the mean-field response needs the self-consistent feedback term, which is the field tangent in option C.
- Solves our problem: no. It is analytic response theory, not a particle estimator.

### 4.3 Weighted particles for linearized kinetic equations

**$\delta f$ PIC: Parker and Lee, Phys. Fluids B 5, 77 (1993); Hu and Krommes, Phys. Plasmas 1, 863 (1994); Aydemir, Phys. Plasmas 1, 822 (1994).** [lit, from memory] Markers carry $w=\delta f/g$, updated by $\mathrm{d}w/\mathrm{d}t\propto-\delta F\cdot\partial_u f_0/g$. The structure matches option C: weighted deposits close the loop through the field. The difference is the weight source, which needs $\partial_u f_0$. For a cold sheet that source is a $\delta'$ in $u$, the same failure as Chung et al.

**Brunner, Valeo, Krommes, "Collisional delta-f scheme with evolving background for transport time scale simulations", Phys. Plasmas 6, 4504 (1999); Chen and Parker, "Coarse-graining phase space in $\delta f$ particle-in-cell simulations", Phys. Plasmas 14, 082301 (2007).** [lit, abstracts]
- What: Langevin collisions in $\delta f$ cause weight spreading that grows in time. Remedies are weight-spread reduction and periodic phase-space coarse-graining onto a grid.
- Relevance: this is the closest plasma analog of the $\omega$ variance growth in option C. Coarse-graining $\omega$ onto phase-space bins estimates $\mathbb{E}[\omega\mid z]$. That is a Rao-Blackwell step and keeps the estimator consistent [derived from Identity 1]. This differs from the rejected coarse-graining of $\xi$, which the doubling map defeats. It needs a phase-space mesh, so it is an optional fallback.
- Solves our problem: no. The noise is physical collisions, and the weight source still uses $\partial_u f_0$.

**Deviational Monte Carlo: Baker and Hadjiconstantinou, Phys. Fluids 17, 051703 (2005); Homolle and Hadjiconstantinou, JCP 226 (2007); Péraud and Hadjiconstantinou, PRB 84, 205331 (2011); Péraud and Hadjiconstantinou, "Adjoint-based deviational Monte Carlo methods for phonon transport calculations", PRB 91, 235321 (2015).** [lit, abstracts]
- What: signed particles represent the deviation from a known equilibrium for the (linearized) Boltzmann equation. The 2015 paper uses the adjoint of the linearized equation and asymptotic solutions as control variates.
- Difference: the deviation is from a Maxwellian reference, not a parameter sensitivity. There is no self-consistent field. Variance reduction comes from a known reference, which a cold beam-driven wake does not have.
- Solves our problem: no. The adjoint duality and control variates are possible variance tools later.

**Chung, Bond, Cyr, Freund, "Regular sensitivity computation avoiding chaotic effects in particle-in-cell plasma methods", JCP 400, 108969 (2020).** [lit; read for the R0 note] Same goal as option C. Particle-pdf weights need $\nabla_u f$ from a phase-space reconstruction, so the method fails for a cold sheet. Option C replaces $\nabla_u f$ by the noise score. It is the main comparison method and must be cited.

### 4.4 Sensitivity of DSMC and Monte Carlo transport

**Caflisch, Silantyev, Yang, "Adjoint DSMC for nonlinear Boltzmann equation constrained optimization", JCP 439, 110404 (2021); Yang, Silantyev, Caflisch, JCP 488, 112247 (2023); Caflisch, Chen, Yang, [arXiv:2603.20946](https://arxiv.org/abs/2603.20946) (2026); Caflisch and Yang, "Adjoint Monte Carlo method", [arXiv:2401.08361](https://arxiv.org/abs/2401.08361) (2024).** [lit, abstracts and seminar slides]
- What: discretize-then-optimize adjoints of DSMC. A general collision law adds a score-function term ($\partial_v\log$ of the acceptance density) to the adjoint. The 2026 paper uses score terms for diffuse-wall events and an ensemble adjoint on a locally linearized collision operator.
- Difference: reverse mode; homogeneous or wall-bounded gas; no long-range self-consistent field; no deliberate noise for chaos.
- Solves our problem: no. It shows score terms combined with pathwise terms inside particle Monte Carlo, which is the same pattern as option C.

**Horsten, Carli, Dekeyser, "Sensitivity calculation for Monte Carlo particle simulations of neutrals in the plasma edge", Contrib. Plasma Phys. 64 (2024).** [lit, abstract] Pathwise AD of EIRENE avoids FD decorrelation. Linear neutral transport, no LR, no chaos.

### 4.5 Noise regularization of chaotic sensitivity

**Lea, Allen, Haine, Tellus 52A, 523 (2000); Eyink, Haine, Lea, "Ruelle's linear response formula, ensemble adjoint schemes and Lévy flights", Nonlinearity 17, 1867 (2004).** [lit, citing-paper summaries] Adjoint gradients on long chaotic windows diverge from the climate sensitivity. Ensemble adjoints over short windows converge slowly, with heavy-tailed estimates. This is the deterministic failure that option C avoids by moving the derivative onto the noise.

**Thuburn, "Climate sensitivities via a Fokker-Planck adjoint approach", QJRMS 131, 73 (2005).** [lit, citing-paper summary] Adds artificial Wiener forcing so the stationary density is smooth, then uses the Fokker-Planck adjoint on a phase-space grid. Same regularization idea as option C. Grid-based, so it is limited to low dimension. Option C replaces the grid by particles and weights.

**Hairer and Majda, "A simple framework to justify linear response theory", Nonlinearity 23, 909 (2010).** [lit, abstract] Noise makes linear response rigorous for dissipative stochastic systems. It supports the claim that $\mathbb{E}[J]$ of the noisy model is differentiable in $\theta$.

**Ni, "Ergodic and foliated kernel-differentiation method for linear responses of random systems", J. Nonlinear Sci. 35, 90 (2025), [arXiv:2410.10138](https://arxiv.org/abs/2410.10138); Ni, "Path-kernel method for differentiating unstable diffusions", [arXiv:2503.00718](https://arxiv.org/abs/2503.00718) (2025); Ni, "No-propagate algorithm...", [arXiv:2308.07841](https://arxiv.org/abs/2308.07841) (2023, superseded).** [lit, abstracts and the 2503.00718 HTML]
- What: kernel differentiation is the Girsanov or LR formula, $\delta\mathbb{E}[\Phi(X_T)]=\mathbb{E}[\Phi(X_T)\int_0^T\sigma^{-1}\delta F\cdot\mathrm{d}B]$. It avoids gradient explosion, the curse of dimension, and non-hyperbolicity. Alone, it approximates the deterministic linear response with an error proportional to the noise (2308.07841 abstract). Its cost rises for small noise. The path-kernel formula damps the tangent at rate $\alpha$ and moves the damped part into the kernel term. It interpolates between pure path perturbation ($\alpha=0$) and pure kernel differentiation. Tested on 40-dimensional noisy Lorenz-96.
- Difference: one SDE in a moderate dimension. No interacting particles, no mean-field closure, no weighted deposits. Ni also adds foliated noise in selected directions to cut the bias.
- Solves our problem: no, but it is the closest methodological precedent. Two transferable ideas follow. First, noise only in $u$ is a foliated choice, which Ni reports has smaller error than isotropic noise. Second, the path-kernel blend is a direct way to cut option C variance. It keeps the exact tangent $\xi$ over a short stable window and switches to weights after step about 1450 [estimated]. The brief rules out trajectory tangents as the main method, so this is only a fallback.

**Shadowing and Ruelle-based methods: Wang, Hu, Blonigan, JCP 267 (2014); Blonigan and Wang, "Probability density adjoint for sensitivity analysis of the mean of chaos", JCP 270 (2014); Ni and Wang, JCP 347 (2017); Chandramoorthy and Wang, SIADS 21 (2022).** [lit, abstracts] Deterministic methods for hyperbolic attractors. They need hyperbolicity or a low-dimensional phase-space grid. They target long-time statistics, not a finite-horizon transient with $10^5$ to $10^7$ particles. Not competitors for our case.

### 4.6 Randomized smoothing and machine learning

**Nesterov and Spokoiny, Found. Comput. Math. 17 (2017); Duchi, Bartlett, Wainwright, SIAM J. Optim. 22 (2012).** [lit, from memory] Gaussian smoothing in parameter space. The gradient estimator $\mathbb{E}[J(\theta+s\epsilon)\epsilon]/s$ needs many simulations per parameter. It is an FD-like method, not option C.

**Parmas, Rasmussen, Peters, Doya, "PIPPS: flexible model-based policy search robust to the curse of chaos", ICML (2018); Metz, Freeman, Schoenholz, Kachman, "Gradients are not all you need", [arXiv:2111.05803](https://arxiv.org/abs/2111.05803) (2021); Suh, Simchowitz, Zhang, Tedrake, "Do differentiable simulators give better policy gradients?", ICML (2022), [arXiv:2202.00817](https://arxiv.org/abs/2202.00817).** [lit, abstracts and citing text]
- What: reparameterized (pathwise) gradient variance grows exponentially with horizon in chaotic simulators, even when the averaged loss is smooth. LR or black-box gradients and inverse-variance blends of LR and pathwise gradients are recommended.
- Difference: noise in actions or policies, not in a mean-field particle system. No field closure.
- Solves our problem: no. It confirms in a different field that LR gradients on a noisy model are the standard response to chaos. The PIPPS and Suh et al. blends are analogs of Ni's path-kernel blend.

### 4.7 Differentiable PIC

**JAX-in-Cell, [arXiv:2512.12160](https://arxiv.org/abs/2512.12160) (2025); Joglekar and Thomas, differentiable Vlasov solvers (2022); Bartsch, Knopf, Scheurer, Weber, [arXiv:2304.02083](https://arxiv.org/abs/2304.02083) (2023); review [arXiv:2603.11231](https://arxiv.org/abs/2603.11231) (2026).** [lit, abstracts] Pathwise AD through PIC or grid Vlasov, or a continuum adjoint. None found uses noise plus LR weights to handle trajectory chaos. The 2026 review and the search found no treatment of exploding PIC gradients from chaotic trajectories.

## 5. Papers to cite or compare against

1. Warren and Allen, PRL 109, 250601 (2012). The per-particle Malliavin weight and the joint-measure estimator. Compare variance against it.
2. Ni, J. Nonlinear Sci. 35, 90 (2025), and arXiv:2503.00718 (2025). Noise-regularized LR for chaotic systems, the $\sigma^{-1}$ cost and noise-proportional bias, and the path-kernel blend.
3. Crisan and McMurray, PTRF 171 (2018), with Baños, Ann. IHP 54 (2018). Mathematical basis for the mean-field LR with the Lions-derivative feedback term.
4. Chung, Bond, Cyr, Freund, JCP 400 (2020), with Brunner, Valeo, Krommes, Phys. Plasmas 6 (1999). The competing PIC weight route that needs $\nabla_u f$, and the plasma precedent for Langevin weight spreading.
5. Lucarini, Pavliotis, Zagli, Proc. R. Soc. A (2020). Breakdown of mean-field linear response at collective instabilities. This is the named failure mode for option C.

Second tier: Thuburn (2005) for noise-regularized Fokker-Planck adjoints; Bond et al. SAND2022-12721 for Malliavin cost in particle codes; Caflisch, Silantyev, Yang (2021, 2023) for score terms in DSMC adjoints; Péraud and Hadjiconstantinou (2015) for adjoint deviational MC.

## 6. Implications for the option C study [estimated unless labelled]

1. Test both estimators. The joint LR (Warren-Allen) needs no field tangent and is exact at finite $N$, but its single-run variance is $O(1)$ [derived, sketch]. The mean-field LR has single-run variance $O(N^{-1})$ and finite-$N$ bias. The harness should measure both against $N$.
2. Measure bias against $\sigma$ before variance. Ni reports bias proportional to noise for kernel differentiation [lit]. For FBPIC, the physical floor is the real plasma temperature and the ionization momentum spread. That floor sets an honest lower $\sigma$.
3. Watch $\delta F$ growth, not only $\omega$ growth. Linear growth of $\mathrm{Var}\,\omega$ is expected [lit: Warren and Allen; Brunner et al.]. Exponential growth signals a collective mode [lit: Lucarini et al.] and would kill the method as an exponential-free estimator.
4. Keep two fallbacks in view. Rao-Blackwell coarse-graining of $\omega$ on a phase-space grid cuts variance and stays consistent [derived]. The Ni path-kernel blend cuts variance with short-window tangents [lit]. Both break a stated preference (no mesh, no tangents), so use them only if pure option C fails the variance gate.
5. Novelty claim to make in a paper: a forward-mode, cold-safe, mean-field Girsanov weight method for PIC sensitivities, closed through field tangents. Do not claim the Girsanov or McKean-Vlasov mathematics as new.

## Sources

- Warren and Allen 2012: https://arxiv.org/abs/1207.4691 ; Entropy guide: https://www.mdpi.com/1099-4300/16/1/221
- Szamel 2017: https://arxiv.org/abs/1607.00326 ; Garnier, Ip, Mertz 2024: https://arxiv.org/abs/2406.11454
- Bond et al. SAND2022-12721: https://www.sandia.gov/app/uploads/sites/205/2022/09/SAND2022-12721.pdf ; SAND2019-11524R: https://www.osti.gov/biblio/1568989 ; Bond publications: https://www.sandia.gov/-sdbond/publications/
- Baños 2018: https://arxiv.org/abs/1510.06961 ; Crisan and McMurray 2018: https://arxiv.org/abs/1702.01397 ; Ren and Wang: https://arxiv.org/abs/1809.06068 ; Zhao 2026: https://arxiv.org/abs/2601.20350
- Lacker 2023: https://msp.org/pmp/2023/4-2/pmp-v4-n2-p04-p.pdf
- Lucarini, Pavliotis, Zagli 2020: https://arxiv.org/abs/2008.10912
- Brunner, Valeo, Krommes 1999: https://www.osti.gov/biblio/8844 ; Chen and Parker 2007: https://pubs.aip.org/aip/pop/article/14/8/082301/105089
- Péraud and Hadjiconstantinou 2015: https://www.osti.gov/pages/biblio/1370983
- Chung et al. 2020: https://www.osti.gov/biblio/1570259
- Caflisch, Silantyev, Yang 2021: https://www.osti.gov/pages/biblio/1849936 ; Yang, Silantyev, Caflisch 2023: https://www.sciencedirect.com/science/article/pii/S002199912300342X ; Caflisch, Chen, Yang 2026: https://arxiv.org/abs/2603.20946 ; Caflisch and Yang 2024: https://arxiv.org/abs/2401.08361
- Horsten, Carli, Dekeyser 2024: https://onlinelibrary.wiley.com/doi/10.1002/ctpp.202300138
- Hairer and Majda 2010: https://www.hairer.org/papers/FDT.pdf
- Ni: https://arxiv.org/abs/2410.10138 ; https://arxiv.org/abs/2503.00718 ; https://arxiv.org/abs/2308.07841 ; publication list https://faculty.sites.uci.edu/niangxiu/publications/
- Blonigan and Wang 2014: https://arxiv.org/abs/1306.3800
- Metz et al. 2021: https://arxiv.org/abs/2111.05803 ; Suh et al. 2022: https://arxiv.org/abs/2202.00817
- JAX-in-Cell: https://arxiv.org/abs/2512.12160 ; differentiable plasma review: https://arxiv.org/abs/2603.11231

Citations marked "from memory" (Parker and Lee 1993, Hu and Krommes 1994, Aydemir 1994, Fournié et al. 1999, Gobet and Munos 2005, Chen and Glasserman 2007, Nesterov and Spokoiny 2017, Duchi et al. 2012, Lea et al. 2000 volume and pages, Thuburn 2005 volume and pages, PIPPS venue) were not re-checked against the original text.
