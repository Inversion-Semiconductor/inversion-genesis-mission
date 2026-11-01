# sec_lit: Literature and novelty check for the gauge-projected tangent (GPT)

Date: 2026-11-01. Task key: `lit`.

Labels: **[lit]** from the cited work; **[lit, summary]** from an abstract or a search summary, not read in full; **[derived]** derived here; **[measured]** measured with the named script; **[estimated]** an estimate with stated inputs; **[memory]** from memory, not re-checked.

Scratch base: `inversion/notes/gpt-evidence-2026-11-01/`, written `mt/` below.

## 1. Verdict

**GPT is partially known.** Each ingredient has prior art. The combination does not: I found no work that applies an $L^2(f)$ projection onto gradient fields inside a forward tangent integration to stop chaotic growth, in PIC or in any N-body or particle simulation.

| Ingredient of GPT | Status | Closest prior art |
|---|---|---|
| Split $\xi = \nabla\phi + \eta$, $\nabla\cdot(f\eta) = 0$, $L^2(f)$-orthogonal | known | Brenier 1991 (linearized polar factorization); Otto 2001; Ambrosio, Gigli, Savaré 2008 |
| Gradient part is the minimum-norm representative of $\delta f$ | known | Ambrosio, Gigli, Savaré 2008; Jankowiak and Obermeyer 2018, Eq. (14) and (15) |
| A Lagrangian displacement has a relabeling gauge; fix it by orthogonality | known (fluids, stars) | Friedman and Schutz 1978 ("trivial displacements", "canonical displacements") |
| Choose the optimal-transport velocity among all pathwise derivatives | known (static sampling) | Jankowiak and Obermeyer 2018 (OMT gradient) |
| Sample least squares of a particle Jacobian onto a finite gradient basis | known (static) | Arbel, Gretton, Li, Montúfar 2020 (KWNG), Prop. 3 and Eq. (17) and (18) |
| Periodic conversion or coarse-graining of a particle sensitivity | known | Chung et al. 2020 (particle-pdf); Chen and Parker 2007 ($\delta f$ PIC); R0 Galerkin reset |
| Periodic Wasserstein projection of a forward tangent to control chaos | **not found** | none |
| Displacement-based reset that keeps cold-sheet currents | **not found** | none |

A defensible novelty claim: "the first use of the Otto (Wasserstein) gauge projection to regularize forward tangents of a chaotic particle simulation, in a form that stays defined on cold phase-space sheets." The decomposition, the estimator, and the gauge idea are not new and must be cited.

**No prior work solves our problem.** Section 4 says why each candidate fails on cold, transient LPA.

**Main risk found through the literature [derived, measured].** The GPT right-hand side $b_k = \sum_iw_i\,\xi_i\cdot\nabla\psi_k(z_i)$ is exactly the particle-exact derivative of the moment $\sum_iw_i\psi_k(z_i)$. GPT keeps it unchanged, and so keeps its sampling noise. On the doubling map the projected tangent is not 0. It is $\lVert P\xi\rVert/\lVert\xi\rVert \approx \sqrt{2K/N}$ ($K$ Fourier modes), so the noise floor does not go away (Section 3.2). GPT reduces the effective chaotic horizon from the full run to one projection interval. It does not remove Chung's non-commutation (paper p4).

## 2. Close prior work

### 2.1 Mathematical basis of the gauge split

1. **Brenier, CPAM 44 (1991).** Every $L^2$ map factors as $\nabla u\circ s$, with $s$ measure preserving. The Helmholtz decomposition is the linearization of this factorization [lit, summary]. With a reference density $f$, a first-order measure-preserving perturbation $s = \mathrm{id} + \epsilon\eta$ needs $\nabla\cdot(f\eta) = 0$. This is the GPT gauge set [derived]. *Difference:* Brenier treats maps, not dynamics or sensitivities. *Solves our problem:* no.
2. **Otto, Comm. PDE 26 (2001); Ambrosio, Gigli, Savaré, *Gradient Flows* (2008), Ch. 8.** The tangent space of $\mathcal P_2$ at $\mu$ is the $L^2(\mu)$ closure of $\{\nabla\phi\}$. Its orthogonal complement is $\{\eta : \nabla\cdot(\mu\eta) = 0\}$. A velocity field is in the tangent space if and only if it has minimal norm among the fields that move $\mu$ the same way [memory: AGS Prop. 8.4.5]. *Difference:* the theory is static geometry, with no chaotic propagation and no particle estimator. *Solves:* no.
3. **Peyre, ESAIM COCV 24 (2018); Peyré and Cuturi (2019), Sec. 8.** $W_2$ is locally equivalent to the weighted $\dot H^{-1}(\mu)$ norm, $\lVert\rho\rVert_{H^{-1}(\mu)}^2 = \min_{\nabla\cdot s = \rho}\int\lvert s\rvert^2d\mu$ [lit, summary]. The comparison constants blow up when the density goes to 0 or to infinity [lit, summary]. *Consequence for GPT:* the claim "Wasserstein tangent norm = $H^{-1}(f)$ norm of $\delta f$" is correct. On a cold sheet ($f$ singular) the norm equivalence with flat $\dot H^{-1}$ has no uniform constants, so flat-space mixing results do not transfer directly [derived from the cited bounds].
4. **Lions derivative (Cardaliaguet notes; Crisan and McMurray, PTRF 2018).** For $U(\mu)$, $\partial_\mu U(\mu)(\cdot) \in L^2(\mu)$ is a gradient, $\nabla_x\,\delta U/\delta m$. On the empirical measure, $\nabla_{x_j}U(\mu^N) = N^{-1}\partial_\mu U(\mu^N)(x_j)$ [lit, summary]. *Consequence:* at a fixed time, any observable sees only the gradient part of $\xi$. The gauge only matters through propagation. This is the GPT premise in standard form. *Solves:* no. The pathwise derivatives of McKean-Vlasov SDEs in this literature rely on noise; our plasma is deterministic and cold.

### 2.2 Gauge fixing of pathwise (Lagrangian) derivatives by optimal transport

5. **Jankowiak and Obermeyer, "Pathwise derivatives beyond the reparameterization trick", ICML 2018, [arXiv:1806.01851](https://arxiv.org/abs/1806.01851).** Read in Sec. 4. They write $\partial_\theta q_\theta + \nabla_z\cdot(q_\theta v_\theta) = 0$ (Eq. 12). They state that this "admits an infinite dimensional space of solutions" (Sec. 4.3). They fix the solution by $\partial_jv_i = \partial_iv_j$ (Eq. 14), which minimizes $\frac12\int q\lvert v\rvert^2$ (Eq. 15). They show that the reparameterization velocity $\partial_\theta T$ (the Lagrangian tangent) is one non-optimal member of the family (Eq. 17 to 19). They warn that the OMT field minimizes variance only for test functions with $\nabla f$ roughly constant (Eq. 16) [lit]. *This is the closest conceptual match to GPT.* *Difference:* the sampling is static (no dynamics, no chaos). The OMT field is derived analytically for Gamma, Beta, Dirichlet and the multivariate normal, not projected from samples. *Solves:* no.
6. **Jankowiak and Karaletsos, AISTATS 2019, [arXiv:1806.01856](https://arxiv.org/abs/1806.01856).** They use the null solutions of the transport equation (particle motions that leave $q$ unchanged, the GPT gauge set) as control variates to reduce gradient variance [lit, summary]. *Difference:* they add gauge components to minimize variance, while GPT removes them to minimize norm. Static. *Solves:* no.
7. **Arbel, Gretton, Li, Montúfar, "Kernelized Wasserstein natural gradient", ICLR 2020, [arXiv:1910.09652](https://arxiv.org/abs/1910.09652).** Read in Sec. 2 and 3. For a push-forward model $\rho_\theta = (h_\theta)_\#\nu$ they use $\nabla\rho_\theta(f) = \int\nabla h_\theta(z)^T\nabla_xf(h_\theta(z))\,d\nu$ (Eq. 7) and $u^TG_Wu = \sup_f\,[\nabla\rho_\theta(f)u - \frac12\int\lvert\nabla_xf\rvert^2d\rho_\theta]$ (Prop. 3, Eq. 8). They restrict $f$ to kernel derivatives at $M$ subsampled points and solve from $N$ samples (Eq. 17, 18). They note $T = CB$, with $B$ the particle Jacobian $\partial_\theta h_\theta(Z_n)$ [lit]. The dual problem is the GPT least squares with a kernel basis [derived]: the optimal $\nabla f$ is the $L^2(\rho_\theta)$ projection of the particle Jacobian onto the basis gradients. *GPT's discrete estimator is therefore known.* *Difference:* it is used for a metric tensor (natural gradient), once per optimizer step, not inside a time integration. *Solves:* no.
8. **Li and Zhao, "Wasserstein information matrix", Inf. Geom. 6 (2023), [arXiv:1910.11248](https://arxiv.org/abs/1910.11248); Chen and Li, [arXiv:1805.08380](https://arxiv.org/abs/1805.08380); Lin, Li, Osher, Montúfar, "Wasserstein proximal of GANs", [arXiv:2102.06862](https://arxiv.org/abs/2102.06862).** The Wasserstein score $\Phi_i$ solves $-\nabla\cdot(\rho\nabla\Phi_i) = \partial_i\rho$, the counterpart of the Fisher score $\partial_i\log\rho$ [lit, summary]. Lin et al. state that the Wasserstein natural gradient is defined "even when the support of the distributions is low dimensional and the Fisher-Rao natural gradient is not well defined" [lit, summary]. *Consequence:* this is the cold-sheet argument of GPT against the weight (Fisher-Rao) form of R0 and Chung, already stated for generators. *Solves:* no.
9. **Stein variational gradient descent (Liu 2017; Duncan, Nüsken, Szpruch, JMLR 24 (2023)).** The particle velocity is a kernel-smoothed Wasserstein gradient (the "Stein geometry") [lit, summary]. *Difference:* this is a projection onto an RKHS of velocities, not onto gradient fields, for sampling, not sensitivity. *Solves:* no.

### 2.3 Relabeling gauge in Lagrangian perturbation theory

10. **Friedman and Schutz, ApJ 221, 937 (1978), "Lagrangian perturbation theory of nonrelativistic fluids".** They define "trivial displacements" $\eta$, for which all Eulerian perturbations vanish. Trivials are carried along by the unperturbed flow. The canonical energy is not invariant under them. They fix the gauge with "canonical displacements" that are orthogonal to all trivials under the symplectic product, and that stay orthogonal in time [lit, summary via abstract and Glampedakis and Andersson 2007]. *This is the earliest statement of the GPT gauge idea for a Lagrangian tangent.* *Difference:* fluid, linear stability of stationary stars, symplectic (not $L^2(\rho)$) orthogonality, no particles, no chaos. *Solves:* no.
11. **Morrison and Pfirsch, Phys. Rev. A 40, 3898 (1989); Phys. Fluids B 2, 1105 (1990).** In Vlasov-Maxwell they restrict perturbations to "dynamically accessible" ones generated by a function $W$: $\delta f = \{W, f\}$ at first order. These preserve all Casimirs [lit, summary]. *Consequence:* the kinetic-plasma tradition fixes the displacement freedom with a Hamiltonian generator, $\xi = \mathbb J\nabla W$. GPT uses a gradient $\nabla\phi$ in a chosen Euclidean metric on $z = (\mathbf x, \mathbf u)$. Both represent the same $\delta f$, with different norms [derived]. *Solves:* no.
12. **Relabeling symmetry: Salmon 1988; Padhye and Morrison 1996; Holland 2011, [arXiv:1105.3764](https://arxiv.org/abs/1105.3764).** Relabeling is the symmetry behind Kelvin circulation and Casimirs. Holland treats the label as a gauge [lit, summary]. *Solves:* no. Nothing in this line uses the gauge to regularize sensitivities.

### 2.4 Chaotic sensitivity methods

13. **Least squares shadowing (Wang 2014, [arXiv:1204.0159](https://arxiv.org/abs/1204.0159)); NILSS (Ni and Wang 2017, [arXiv:1611.00880](https://arxiv.org/abs/1611.00880)); FD-NILSS; multiple shooting shadowing.** LSS picks the minimum-norm tangent within the shadowing class [lit, summary]. *Analogy:* GPT picks the minimum-norm tangent within the gauge class. *Difference:* LSS needs ergodicity and a long-time average. NILSS cost scales with the number of positive Lyapunov exponents, which is large in N-body systems. No application to particle or kinetic systems was found [lit, summary]. *Solves:* no. LPA observables are transient, and the cold beam is not on an attractor.
14. **Linear response: Ruelle; space-split (Śliwiak and Wang, [arXiv:2109.13313](https://arxiv.org/abs/2109.13313)); S3 (Chandramoorthy and Wang, [arXiv:2002.04117](https://arxiv.org/abs/2002.04117)).** They split the perturbation along stable and unstable directions of the dynamics, and integrate the unstable part by parts on the SRB measure [lit, summary]. *Difference:* their split depends on the dynamics, not on $f$. They need hyperbolicity and stationarity. *Solves:* no.
15. **Ni, kernel and path-kernel methods ([arXiv:2410.10138](https://arxiv.org/abs/2410.10138), [arXiv:2503.00718](https://arxiv.org/abs/2503.00718), [arXiv:2509.03992](https://arxiv.org/abs/2509.03992)).** These move the derivative gradually from path perturbation to kernel differentiation, in SDEs with noise [lit, summary]. *Consequence:* this is the stochastic counterpart of the R0 weight conversion. The kernel is the noise transition density. *Solves:* no. A cold deterministic plasma has no kernel.
16. **Chung and Freund, JCP 457, 111077 (2022), multistep penalty.** They split time into segments of about one chaos time, allow jumps $\Delta q$, and penalize them in stages [lit, summary]. *Difference:* this is an optimizer change, not a derivative. *Solves:* partly, as an optimizer idea. It is not a forward tangent.
17. **Metz, Freeman, Schoenholz, Kachman, [arXiv:2111.05803](https://arxiv.org/abs/2111.05803) (2021).** Reparameterized gradients through chaotic simulations (rigid bodies, MD disk packing) have variance that grows exponentially with horizon. Black-box estimators can do better [lit, summary]. *Consequence:* this is the same finding as P7 and R0 (FD beats AD). *Solves:* no.
18. **Wormell and Gottwald, J. Stat. Phys. 172 (2018); Chaos 29, 113127 (2019).** In mean-field-coupled high-dimensional systems, macroscopic observables can have a linear response when the microscopic units do not. They also give a counterexample in which the macroscopic response fails [lit, summary]. *Consequence:* this is the best theoretical support for "a macroscopic derivative exists while particle tangents explode". It is not guaranteed. *Solves:* no algorithm.
19. **Yang, Nurbekyan, Negrini, Martin, Pasha, SIADS 22 (2023), [arXiv:2104.15138](https://arxiv.org/abs/2104.15138).** They identify parameters of chaotic systems by $W_2$ misfit of invariant measures, computed by an Eulerian Perron-Frobenius PDE [lit, summary]. *Difference:* Eulerian, stationary, low dimensional. *Solves:* no.
20. **N-body microchaos: Goodman, Heggie, Hut 1993; Hemsendorf and Merritt 2002, [astro-ph/0205538](https://arxiv.org/abs/astro-ph/0205538); Di Cintio and Casetti 2019; Asano and Portegies Zwart 2026, [arXiv:2604.12053](https://arxiv.org/abs/2604.12053).** Unsoftened N-body growth rates do not fall with $N$. With softening, chaos is suppressed, and the Lyapunov time grows with $N$ [lit, summary]. *Consequence:* PIC grid smoothing acts as softening. If the P7 growth were discreteness chaos, it would weaken with $N$. This links to R0 question Q1 (particle-count scaling of the FD roughness). *Solves:* no.

### 2.5 PIC and kinetic sensitivity work

21. **Chung, Bond, Cyr, Freund, JCP 400, 108969 (2020).** Particle-pdf sensitivity, a weight (Fisher-Rao) representation that needs $\nabla_uf$ [lit]. Page 49 (read on the page image): the method rests on a Lyapunov-stable $N\to\infty$ continuum limit, whose "specific criterion remains unknown". Wang, Miller, Colella (SISC 33, 2011) "show that periodic redistribution of particles can suppress errors coming from diverging particle trajectories, though it is unclear that the redistribution would obviate the non-commutability challenge, especially upon considering the effect of interpolation shown in figure 11 (d)" (pp. 49 to 50). Page 50: the accuracy is judged against a brute-force curve that "remains effectively non-differentiable" [lit]. *Difference:* weights, not displacements; per-step conversion. Fails on a cold plasma (R0 A.2). *Solves:* no.
22. **Papers that cite Chung et al. 2020** (Semantic Scholar API, 2026-11-01): Chung and Freund 2022 (item 16); Medaglia, Tosin, Zanella, JCP 2023, stochastic Galerkin particle methods for kinetic UQ; Medaglia et al. 2023, Landau-Fokker-Planck particles with uncertain data ([arXiv:2306.07701](https://arxiv.org/abs/2306.07701)); Albi et al., JCP 2025, robust feedback control of collisional plasma ([arXiv:2505.19992](https://arxiv.org/abs/2505.19992)); and one unrelated paper on chaotic circuits [lit, summary]. None of them extends the chaos regularization. ScienceDirect lists 5 citations. Google Scholar was not checked.
23. **Adjoint and control of Vlasov-Poisson:** Bartsch, Knopf, Scheurer, Weber 2023/2024 ([arXiv:2304.02083](https://arxiv.org/abs/2304.02083)), PIC Monte Carlo with an $\lambda\,\partial_vf$ adjoint; Einkemmer, Li, Wang, Yang, JCP 498, 112662 (2024), Eulerian adjoint [lit, summary]. *Difference:* warm or Eulerian, no chaos treatment. *Solves:* no.
24. **Differentiable PIC:** JAX-in-Cell, [arXiv:2512.12160](https://arxiv.org/abs/2512.12160) (2025), end-to-end AD of a PIC code (bump-on-tail demo) [lit, summary]. No treatment of tangent growth was found. *Solves:* no.
25. **$\delta f$ PIC weight growth:** Krommes and Hu, Phys. Plasmas 1 (1994), "entropy paradox"; Chen and Parker, Phys. Plasmas 14, 082301 (2007), periodic deposit of $\delta f$ on a 5-D grid and partial weight reset [lit, summary]. "The coarse-graining approach introduces dissipation" [lit, summary of a citing paper]. *Consequence:* phase mixing makes weight representations grow; the cure is grid coarse-graining with a dissipation scale. This is the weight-side twin of GPT's projection. *Solves:* no.
26. **Phase-space remapping and OT Lagrangian schemes:** Wang, Miller, Colella 2011 (remap of primal particles); Gallouët and Mérigot, FoCM 18 (2018), a Lagrangian scheme for Euler by projection onto measure-preserving maps with semi-discrete OT [lit, summary]. *Consequence:* GPT is the linearized, tangent-level form of this projection. *Solves:* no.

### 2.6 Mixing norms (for the $H^{-1}$ heuristic)

27. **Mathew, Mezić, Petzold, Physica D 211 (2005); Lin, Thiffeault, Doering, JFM 675 (2011); Thiffeault, Nonlinearity 25 (2012).** Negative Sobolev norms ($H^{-1/2}$, $H^{-1}$) measure mixing. They decay under chaotic advection of a passive scalar [lit, summary]. *Consequence:* this supports the hope that $\lVert\delta f\rVert_{H^{-1}(f)}$ stays bounded while $\lVert\xi\rVert$ grows. With no field coupling, $\delta f$ is a passive scalar of the volume-preserving flow [derived]. The cited results use a flat measure. Item 3 says that the transfer to a singular $f$ has no uniform constants.

## 3. What the literature implies for GPT

### 3.1 GPT is the Wasserstein twin of the R0 Galerkin reset [derived]

GPT solves $\min_c\sum_iw_i\lvert\xi_i - \sum_kc_k\nabla\psi_k(z_i)\rvert^2$, with normal equations

$$
\sum_k\Big[\sum_iw_i\nabla\psi_j\cdot\nabla\psi_k\Big]c_k = \sum_iw_i\,\xi_i\cdot\nabla\psi_j(z_i) = b_j .
$$

The vector $b$ is the same as in the R0 reset (R0 Appendix A.4, Eq. R2). Both keep every moment $\frac{d}{d\theta}\sum_iw_i\psi_j(z_i)$ for $\psi_j \in V$. They differ in the Gram matrix: $\langle\nabla\psi_j,\nabla\psi_k\rangle_f$ (Wasserstein) against $\langle\psi_j,\psi_k\rangle_f$ (Fisher-Rao). They also differ in the representation: a displacement against a weight. This is the Wasserstein against Fisher natural-gradient contrast of items 7 and 8. Item 8 states the cold-sheet advantage of the Wasserstein form. For GPT, that advantage is real: the displacement keeps $\delta\mathbf J = q\sum_iw_i(\delta\mathbf v_iS + \mathbf v_i\,\xi_{x,i}\cdot\nabla S)$ in a cold cell, where a weight change gives zero (R0 A.2).

### 3.2 Noise floor: GPT keeps the particle-exact moment noise [derived, measured]

On the doubling map with $\xi_i = 2^n$ for all $i$, $b_k = 2^n\sum_iw_i\psi_k'(x_i)$. For random samples this sum is $O(N^{-1/2})$, not 0. So $P\xi \ne 0$.

Measured with $K$ Fourier modes ($2K$ basis functions), $n = 20$ [measured, `mt/lit/doubling_gpt_noise.py`, output `mt/lit/doubling_gpt_noise.out`]:

| $N$ | $K$ | random samples: $\lVert P\xi\rVert/2^n$ | $\sqrt{2K/N}$ | near-lattice samples |
|---|---|---|---|---|
| $10^3$ | 4 | $8.0\cdot10^{-2}$ | $8.9\cdot10^{-2}$ | $2.4\cdot10^{-3}$ |
| $10^4$ | 4 | $2.1\cdot10^{-2}$ | $2.8\cdot10^{-2}$ | $6.5\cdot10^{-4}$ |
| $10^5$ | 16 | $1.7\cdot10^{-2}$ | $1.8\cdot10^{-2}$ | $1.1\cdot10^{-3}$ |

So one projection shrinks a coherent gauge tangent by about $\sqrt{N_b}$, with $N_b$ the number of particles per basis function. Low-discrepancy (quiet-start) samples do much better while the lattice structure survives.

Consequence [estimated]: if the gauge part dominates, the projected tangent obeys $\lVert\xi_{k+1}\rVert \approx e^{\lambda T}N_b^{-1/2}\lVert\xi_k\rVert$ plus the physical part. A bounded recursion needs $e^{\lambda T} < \sqrt{N_b}$. With $\lambda = 0.019$ per step: $N_b = 10$ gives $T < 61$ steps; $N_b = 100$ gives $T < 121$ steps. In a 6-D local polynomial basis, $N_b$ is the particles per cell divided by the polynomial count, which can be small for the EV bunch. This is Chung's non-commutation (paper p4) in a local form. GPT bounds it per interval; it does not remove it. Jankowiak and Obermeyer Eq. (16) (item 5) give the matching static caveat: the minimum-norm field is not the minimum-variance estimator.

### 3.3 Truncation is a coarse-graining with a known error form [derived]

Let $\psi_t$ be the response function of the final observable to a displacement at time $t$ (with no field coupling, $\psi_t = \phi\circ\Phi_{t\to T}$). The error of one projection is

$$
\epsilon = \sum_iw_i\,[(I - P)\xi]_i\cdot\nabla\psi_t(z_i) = \big\langle(I - P)\xi,\ \nabla\psi_t - P\nabla\psi_t\big\rangle_f .
$$

It is zero when $(I - P)\xi$ is exact gauge, because $\nabla\psi_t$ is a gradient. Otherwise $\lvert\epsilon\rvert \le \lVert(I - P)\xi\rVert_f\,\mathrm{dist}_f(\nabla\psi_t, \nabla V)$. In a chaotic region $\nabla\psi_t$ grows like $e^{\lambda(T - t)}$ and filaments below the basis scale, so the distance is not small. This has the same structure as R0 bound R4. For Dirac particles, $\nabla\cdot(f_N\eta) = 0$ forces $\eta_i = 0$ for all $i$. The gauge therefore exists only relative to the test space $V$. GPT is a coarse-grained derivative at the basis scale, like every R0 estimator, Chung's method (item 21), and $\delta f$ coarse-graining (item 25). This answers the R0 main risk only partly: the GPT output also looks regular by construction.

### 3.4 In the continuum, dropping exact gauge is exact [derived]

The linearized Vlasov-Maxwell system evolves $(\delta f, \delta\mathbf E, \delta\mathbf B)$ from its own state. Two tangents with the same $\delta f_t$ give the same future. So removing an exact gauge component at any time changes no future observable. This includes coherent mean-field (physical) instabilities, which live in $\delta f$ and are kept. The approximation error comes only from the finite $N$ (Section 3.2) and the finite basis (Section 3.3).

### 3.5 The metric on $z$ is a free choice [derived]

The gradient and the orthogonality use a metric on $z = (\mathbf x, \mathbf u)$, for example $\lvert\delta\mathbf x\rvert^2/L^2 + \lvert\delta\mathbf u\rvert^2/U^2$. Changing $L/U$ changes $P\xi$ but not the exact-gauge set. So, in the continuum, it changes nothing physical. With a finite basis it changes the truncation error. Friedman and Schutz (item 10) and Morrison and Pfirsch (item 11) use symplectic structure instead, which needs no scale. A symplectic variant (fit $\xi \approx \mathbb J\nabla W$) is an untested alternative with the same $b$.

## 4. Does any published method already solve the P7 problem?

No.

1. Particle-pdf and other weight forms (items 21, 25, R0): fail on cold sheets (no $\nabla_uf$).
2. LSS, NILSS, space-split, S3 (items 13, 14): need ergodic, stationary, hyperbolic dynamics, and their cost scales with the unstable dimension. LPA is transient and cold.
3. Kernel and path-kernel methods (item 15): need noise.
4. The OT-gradient literature (items 5 to 9): has the right static object and the right estimator, with no dynamics.
5. FD (Metz et al., item 17; R0): works at a coarse-graining scale $h$, as GPT does at the basis scale.

## 5. Search coverage and limits

1. About 25 web searches and 8 page reads: Chung et al. citations, Otto and Wasserstein tangents, relabeling and gauge (fluids, stars, Vlasov, SPH, vortex), chaotic sensitivity (LSS, NILSS, Ruelle, Ni), MD and N-body chaos, OT pathwise gradients, Wasserstein natural gradient, differentiable PIC, Vlasov control, $\delta f$ coarse-graining, mixing norms.
2. Read in full sections: Jankowiak and Obermeyer 2018 Sec. 4; Arbel et al. 2020 Sec. 2 and 3; Chung et al. 2020 pp. 49 to 50 (page images) and reference list.
3. Not checked: Google Scholar citation lists; Friedman and Schutz 1978 full text (abstract and secondary sources only); AGS 2008 proposition numbers (memory); vortex-method remeshing and SPH adjoint literature beyond one search (no relevant hit).
4. A negative search result is weak evidence of novelty. A specialist search in OT-for-kinetic-equations (for example, Wasserstein-Hamiltonian flows, deterministic particle methods for Landau or Fokker-Planck) is the most likely place for an unseen close match.

## References

1. Y. Brenier, "Polar factorization and monotone rearrangement of vector-valued functions", CPAM 44 (1991) 375-417. [PDF](https://www.ceremade.dauphine.fr/~carlier/Brenier91.pdf).
2. F. Otto, "The geometry of dissipative evolution equations: the porous medium equation", Comm. PDE 26 (2001). [memory]
3. L. Ambrosio, N. Gigli, G. Savaré, *Gradient Flows in Metric Spaces and in the Space of Probability Measures*, Birkhäuser (2008), Ch. 8. [memory]
4. R. Peyre, "Comparison between $W_2$ distance and $\dot H^{-1}$ norm, and localization of Wasserstein distance", ESAIM COCV 24 (2018) 1489-1501, [arXiv:1104.4631](https://arxiv.org/abs/1104.4631).
5. G. Peyré, M. Cuturi, *Computational Optimal Transport* (2019), Sec. 8, [arXiv:1803.00567](https://arxiv.org/abs/1803.00567).
6. D. Crisan, E. McMurray, "Smoothing properties of McKean-Vlasov SDEs", PTRF (2018), [link](https://link.springer.com/article/10.1007/s00440-017-0774-0).
7. M. Jankowiak, F. Obermeyer, "Pathwise derivatives beyond the reparameterization trick", ICML 2018, [arXiv:1806.01851](https://arxiv.org/abs/1806.01851).
8. M. Jankowiak, T. Karaletsos, "Pathwise derivatives for multivariate distributions", AISTATS 2019, [arXiv:1806.01856](https://arxiv.org/abs/1806.01856).
9. M. Arbel, A. Gretton, W. Li, G. Montúfar, "Kernelized Wasserstein natural gradient", ICLR 2020, [arXiv:1910.09652](https://arxiv.org/abs/1910.09652).
10. W. Li, J. Zhao, "Wasserstein information matrix", Inf. Geom. 6 (2023) 203-255, [arXiv:1910.11248](https://arxiv.org/abs/1910.11248).
11. Y. Chen, W. Li, "Natural gradient in Wasserstein statistical manifold", [arXiv:1805.08380](https://arxiv.org/abs/1805.08380).
12. A. T. Lin, W. Li, S. Osher, G. Montúfar, "Wasserstein proximal of GANs", [arXiv:2102.06862](https://arxiv.org/abs/2102.06862).
13. A. Duncan, N. Nüsken, L. Szpruch, "On the geometry of Stein variational gradient descent", JMLR 24 (2023), [arXiv:1912.00894](https://arxiv.org/abs/1912.00894).
14. J. L. Friedman, B. F. Schutz, "Lagrangian perturbation theory of nonrelativistic fluids", ApJ 221 (1978) 937, [ADS](https://ui.adsabs.harvard.edu/abs/1978ApJ...221..937F/abstract); secondary account in Glampedakis, Andersson, [astro-ph/0702382](https://arxiv.org/abs/astro-ph/0702382).
15. P. J. Morrison, D. Pfirsch, "Free-energy expressions for Vlasov equilibria", Phys. Rev. A 40 (1989) 3898, [PDF](https://web2.ph.utexas.edu/~morrison/89PRA_morrison.pdf); "The free energy of Maxwell-Vlasov equilibria", Phys. Fluids B 2 (1990) 1105, [PDF](https://web2.ph.utexas.edu/~morrison/90PFB1105_morrison.pdf).
16. P. Holland, "Hydrodynamics, particle relabelling and relativity", [arXiv:1105.3764](https://arxiv.org/abs/1105.3764).
17. Q. Wang, R. Hu, P. Blonigan, "Least squares shadowing sensitivity analysis of chaotic limit cycle oscillations", JCP 267 (2014), [arXiv:1204.0159](https://arxiv.org/abs/1204.0159).
18. A. Ni, Q. Wang, "NILSS", JCP (2017), [arXiv:1611.00880](https://arxiv.org/abs/1611.00880).
19. A. A. Śliwiak, Q. Wang, "Space-split algorithm for sensitivity analysis of discrete chaotic systems", [arXiv:2109.13313](https://arxiv.org/abs/2109.13313).
20. N. Chandramoorthy, Q. Wang, "A computable realization of Ruelle's formula", [arXiv:2002.04117](https://arxiv.org/abs/2002.04117).
21. A. Ni, [arXiv:2410.10138](https://arxiv.org/abs/2410.10138); "Differentiating unstable diffusion", [arXiv:2503.00718](https://arxiv.org/abs/2503.00718); "Divergence-kernel method for linear responses of densities and generative models", [arXiv:2509.03992](https://arxiv.org/abs/2509.03992).
22. S. W. Chung, J. B. Freund, "An optimization method for chaotic turbulent flow", JCP 457 (2022) 111077, [OSTI](https://www.osti.gov/pages/biblio/1977280).
23. L. Metz, C. D. Freeman, S. S. Schoenholz, T. Kachman, "Gradients are not all you need", [arXiv:2111.05803](https://arxiv.org/abs/2111.05803).
24. C. L. Wormell, G. A. Gottwald, J. Stat. Phys. 172 (2018), [arXiv:1801.09377](https://arxiv.org/abs/1801.09377); Chaos 29 (2019) 113127, [arXiv:1907.13490](https://arxiv.org/abs/1907.13490).
25. Y. Yang, L. Nurbekyan, E. Negrini, R. Martin, M. Pasha, SIADS 22 (2023), [arXiv:2104.15138](https://arxiv.org/abs/2104.15138).
26. M. Hemsendorf, D. Merritt, [astro-ph/0205538](https://arxiv.org/abs/astro-ph/0205538); P. Di Cintio et al., [arXiv:1907.12774](https://arxiv.org/abs/1907.12774); T. Asano, S. Portegies Zwart, [arXiv:2604.12053](https://arxiv.org/abs/2604.12053).
27. S. W. Chung, S. D. Bond, E. C. Cyr, J. B. Freund, JCP 400 (2020) 108969, [OSTI 1570259](https://www.osti.gov/biblio/1570259); local PDF `/mnt/work/snidhan/projects/applied-research/solvers/fbpic/literature/Chung2020_regular_sensitivity_PIC_OSTI1570259.pdf`.
28. Citing papers: Medaglia et al., [arXiv:2208.00692](https://arxiv.org/abs/2208.00692), [arXiv:2306.07701](https://arxiv.org/abs/2306.07701); Albi et al., [arXiv:2505.19992](https://arxiv.org/abs/2505.19992). Source: Semantic Scholar API for DOI 10.1016/j.jcp.2019.108969.
29. Bartsch, Knopf, Scheurer, Weber, [arXiv:2304.02083](https://arxiv.org/abs/2304.02083); Einkemmer, Li, Wang, Yang, JCP 498 (2024) 112662, [link](https://www.sciencedirect.com/science/article/pii/S002199912300757X).
30. JAX-in-Cell, [arXiv:2512.12160](https://arxiv.org/abs/2512.12160).
31. Y. Chen, S. E. Parker, "Coarse-graining phase space in $\delta f$ particle-in-cell simulations", Phys. Plasmas 14 (2007) 082301, [OSTI](https://www.osti.gov/biblio/21072620); J. A. Krommes, G. Hu, Phys. Plasmas 1 (1994).
32. B. Wang, G. H. Miller, P. Colella, "A particle-in-cell method with adaptive phase-space remapping for kinetic plasmas", SISC 33 (2011) 3509-3537 (via Chung et al. ref. [29]).
33. T. Gallouët, Q. Mérigot, "A Lagrangian scheme à la Brenier for the incompressible Euler equations", FoCM 18 (2018) 835-865.
34. J.-L. Thiffeault, "Using multiscale norms to quantify mixing and transport", Nonlinearity 25 (2012) R1, [arXiv:1105.1101](https://arxiv.org/abs/1105.1101); Z. Lin, J.-L. Thiffeault, C. R. Doering, JFM 675 (2011); G. Mathew, I. Mezić, L. Petzold, Physica D 211 (2005).
