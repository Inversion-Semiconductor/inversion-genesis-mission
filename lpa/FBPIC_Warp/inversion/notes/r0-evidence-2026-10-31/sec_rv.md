# R0 section: variant R-v, periodic projection of Lagrangian tangents into weight form

Author: R0 derivation agent (key `rv`). Date: 2026-10-31. Status: study result, not a decision.

Evidence labels used below: **[paper]** = Chung, Bond, Cyr, Freund, JCP 400, 108969 (2020), checked against the page images; **[derived]** = derived in this note; **[measured]** = measured by this agent on the CPU with the named script; **[plan]** = taken from `DIFFERENTIABLE_INVERSION_PLAN.md`; **[estimated]** = order-of-magnitude estimate.

Scripts and results are in `inversion/notes/r0-evidence-2026-10-31/work_rv/`. In this note, `work_rv/` stands for that directory.

## 0. Result in brief

1. R-v is well defined and cheap to build on the existing code. The tangent state already carries a per-particle weight tangent $\dot w$ (`fbpic/autodiff/state.py:105`). The deposit tangent already deposits it (`fbpic/autodiff/tangent.py:994`, `:1014`). The loss contraction already reads it (`fbpic/autodiff/loss.py:273`). A reset therefore needs one new operation only: compute $\Delta w$ from $\xi$, add it to $\dot w$, and set $\xi = 0$. **[derived from code]**
2. Between resets, R-v is the exact JVP of the discrete PIC map (P4 to P6 kernels, verified). At a reset, R-v changes the representation of $\delta f$ but keeps every moment in the conversion basis $V$ exactly. The error on any other observable is the weighted inner product of two residuals: the part of $\delta f/f$ outside $V$, and the part of the (pulled-back) observable outside $V$. **[derived]**
3. The reset removes the chaotic stretching by one integration by parts in phase space at the reset time. It moves the gradient from the future observable, whose gradient grows as $e^{\lambda(t_f - t_r)}$, onto $f(t_r)$, whose gradient is bounded. **[derived]**
4. R-v with interval $T = 1$ step is the collocated particle-pdf method of the paper, up to the discretization of $\nabla_u f$. Particle-pdf and R-v are one family. $T$ is a free parameter, and a sweep in $T$ is a built-in self-check. **[derived]**
5. The paper-style phase-space mesh is not usable for the LWFA beam. Its bias is about $\Delta u^2/(6\sigma_u^2)$ in the cold direction [measured, E1], and a 6D mesh fine enough for 5% bias holds about one beam particle per cell [estimated]. A Galerkin basis of grid shape functions times polynomials in momentum (degree 4 to 6) is unbiased for warm and two-stream test cases [measured, E1, E1b]. It fails silently for an exactly cold sheet [measured, E1, $\sigma = 0$].
6. The main risk is not the reset interval. The main risk is the type of chaos. If the chaos of P7 is N-body chaos with a smooth $f$ (the case of the paper), R-v and particle-pdf both work. If it is mean-field chaos that filaments $f$ itself, every finite conversion basis loses accuracy as $f$ filaments. On the chaotic standard map (a mean-field toy), R-v and per-step conversion give stable but wrong answers: 0.21 to 0.39, and $-0.17$ on the finest mesh, against a reference of about 0.74 [measured, E2]. R1 or R2 must measure the type of chaos before R3 starts.

Verdict: **conditional**. R-v is the recommended form of Track R if the chaos is N-body type and the beam $f$ is resolvable by a degree-4 to 6 momentum basis at the reset times. Section 8 lists the kill criteria.

## 1. Inputs used

- Paper, Sections 2.1 to 3.2.5 (PDF pages 5 to 20), Section 6 (pages 28 to 35), Section 8.3 to 9 (pages 44 to 47), read from the page images.
- Plan Sections 4.2 to 4.9, 5.1 to 5.3, 7 (Track R), 8, and 12.10. Notes `inversion/notes/m3-state-2026-10-31.md`. Script `inversion/diag_tangent_growth.py`.
- Code: `fbpic/autodiff/{stepper,tangent,state,ionization,loss}.py` (line numbers below), `fbpic/fields/spectral_grid.py:219` (current correction).

P7 numbers used [plan, Section 12.10]: AD matches frozen FD to $N_t = 500$; at $N_t = 650$ and 1000 the error still falls with $h$ ($e = 5.7\cdot10^{-3}$ and $1.2\cdot10^{-2}$ at $h = 10^{-7}$) but cell crossings remain. The $\infty$-norm of $\partial u_x/\partial E_L$ saturates near $10^6$ J$^{-1}$ from step 600 to 1200. It then grows to $1.5\cdot10^8$ (step 1500), $8.7\cdot10^{10}$ (1800), and $1.7\cdot10^{11}$ (2000). The local rates are $0.014$, $0.021$, and $0.003$ per step on these three intervals [derived from the table]. I use $\lambda = 0.02$ per step as the upper value. One laser period is 16 steps, so $1/\lambda \approx 50$ steps $\approx 3$ laser periods [derived].

## 2. Formulation

### 2.1 Representation of the sensitivity

Write the particle measure of one species as $\mu_\theta = \sum_i w_i(\theta)\,\delta_{z_i(\theta)}$, with $z = (\mathbf x, \mathbf u)$. Its $\theta$-derivative, as a distribution in phase space, is **[derived]**

$$
\mu' = \sum_i \dot w_i\,\delta_{z_i} \;-\; \nabla_z\cdot\Big(\sum_i w_i\,\xi_i\,\delta_{z_i}\Big), \qquad \xi_i = \frac{dz_i}{d\theta}.
$$

The first term is the weight form (monopoles). The second term is the Lagrangian form (dipoles with moment $w_i\xi_i$). For any test function $\psi$,

$$
\langle \mu', \psi\rangle = \sum_i \dot w_i\,\psi(z_i) + \sum_i w_i\,\xi_i\cdot\nabla_z\psi(z_i). \tag{R1}
$$

The R-v tangent state is the pair $(\dot w_i, \xi_i)$ per particle. This is exactly the existing `TangentState` (`state.py:109-140`): `x, y, z, ux, uy, uz, inv_gamma` hold $\xi$, and `w` holds $\dot w$. The paper's continuum form is $\partial_\theta f = -\nabla_z\cdot(f\xi)$ for the Lagrangian part. Its particle form is the second term of $\mu'$.

### 2.2 Between resets

Between resets, R-v runs the existing tangent step unchanged. The P4 to P6 kernels compute the exact derivative of the discrete step with the decisions frozen. They push $\xi$ through the gather, Vay, and position tangents. They carry $\dot w$ unchanged along each particle, through sorts and exchanges. The deposit tangent adds both $\dot w\,S$ and $w\,\xi\cdot\nabla S$ to $\delta\rho$ and $\delta J$. EV births set $\dot w$ of a new electron to $\dot W'$ and copy $\xi$ from the parent ion (`ionization.py:762-764`). **[code]**

So between resets R-v is the exact JVP of the discrete map, as verified by L1 to L4 in P4 to P6. No new approximation enters there. **[plan, verified]**

The source term of the paper is already inside this step. The field tangent enters the Vay tangent as a momentum kick. For the relativistic force $\mathbf F = q(\mathbf E + \mathbf v\times\mathbf B)$ with $\mathbf v = \mathbf u/\gamma$,

$$
\nabla_u\cdot(\mathbf v\times\delta\mathbf B) = \epsilon_{jkl}\,\frac{\partial v_k}{\partial u_j}\,\delta B_l = 0, \qquad \frac{\partial v_k}{\partial u_j} = \frac{\delta_{kj} - v_k v_j/c^2}{\gamma}\ \text{(symmetric in } j,k\text{)}.
$$

So $-q(\delta\mathbf E + \mathbf v\times\delta\mathbf B)\cdot\nabla_u f = -\nabla_u\cdot\big(f\,q(\delta\mathbf E + \mathbf v\times\delta\mathbf B)\big)$ **[derived; the hint is correct]**. The source of the linearized Vlasov equation (paper eq. 25, p. 12, in relativistic EM form) is a phase-space dipole density. Over one step it equals the dipole $-\nabla_u\cdot(f\,\Delta\xi_u)$ with $\Delta\xi_u = q(\delta\mathbf E + \mathbf v\times\delta\mathbf B)\Delta t/m$. This is exactly the field-tangent part of the Vay tangent. Particle-pdf converts this dipole into a weight every step (paper eqs. 44 and 52 to 54a, pp. 18 to 20). R-v accumulates it as $\xi$ for $T$ steps, together with the homogeneous tangent dynamics, and converts the sum.

### 2.3 The reset: conversion of dipoles to weights

At a reset time $t_r$, choose a subset $\mathcal C$ of particles (Section 4.3). Replace

$$
(\dot w_i,\ \xi_i) \;\to\; (\dot w_i + \Delta w_i,\ 0), \qquad i \in \mathcal C,
$$

with $\Delta w$ such that $\sum_i \Delta w_i\,\delta_{z_i} \approx -\nabla_z\cdot\big(\sum_i w_i\xi_i\delta_{z_i}\big)$. Three forms of the conversion exist.

**(a) Local formula, not recommended.** If $\xi(z)$ were a smooth field, $-\nabla\cdot(f\xi) = -f\,(\xi\cdot\nabla\ln f + \nabla\cdot\xi)$, and $\Delta w_i = -w_i\,\big(\xi_i\cdot\nabla_z\ln f(z_i) + (\nabla_z\cdot\xi)(z_i)\big)$. The term $\nabla\cdot\xi$ vanishes only for a volume-preserving flow without sources. The continuum relativistic Vlasov flow is volume preserving in $(\mathbf x, \mathbf u)$. The Vay pusher is not exactly volume preserving (Higuera and Cary, 2017; not verified here), and EV births are a source. The per-particle $\xi_i$ also multiplies $\nabla\ln f$ without any averaging, so the chaotic part of $\xi_i$ passes straight into $\Delta w_i$. **[derived]**

**(b) Paper-style mesh conversion.** Deposit the dipole moments on a phase-space B-spline mesh, take a discrete divergence, and redistribute with the particle number density, as in paper eqs. 41 and 42 (p. 16) and 53a, 53b (p. 20):

$$
P_g = \sum_i w_i\,\xi_i\,S_g(z_i), \quad \Delta F_g = -(\nabla_h\cdot P)_g, \quad \hat n_g = \sum_i S_g(z_i), \quad \Delta w_i = \sum_g \frac{\Delta F_g}{\hat n_g}\,S_g(z_i).
$$

$P_g$ is a weight sum, so $\Delta F_g$ is the weight change per mesh cell and $\Delta F_g/\hat n_g$ is the weight change per particle. This form is conservative: $\sum_i\Delta w_i = 0$ on a periodic or closed mesh. It needs $\hat n_g > 0$ where $\Delta F_g \ne 0$ (paper eq. 43, p. 17). **[paper; derived scaling]**

The mesh form is not feasible for the LWFA beam in 6D. E1 shows a relative bias of about $\Delta^2/(6\sigma^2)$ per resolved direction, so a 5% bias needs $\Delta \lesssim 0.55\,\sigma$. A beam that spans $\pm3\sigma$ then needs about 11 cells per direction and $11^6 \approx 1.8\cdot10^6$ cells, for 1.7 M selected macroparticles (plan Section 12.10): about one particle per cell. $\hat n_g$ is then pure noise. **[estimated]**

**(c) Galerkin moment conversion, recommended.** Choose a basis $\{\psi_a\}$ of test functions and set

$$
\Delta w_i = w_i\sum_a c_a\,\psi_a(z_i), \qquad G\,c = b, \qquad G_{ab} = \sum_{i\in\mathcal C} w_i\,\psi_a(z_i)\,\psi_b(z_i), \qquad b_a = \sum_{i\in\mathcal C} w_i\,\xi_i\cdot\nabla_z\psi_a(z_i). \tag{R2}
$$

This is the minimum-$\chi^2$ weight change, $\min\sum_i \Delta w_i^2/w_i$, that reproduces every moment $\psi_a$ exactly: $\sum_i\Delta w_i\,\psi_a(z_i) = b_a$, which is the Lagrangian moment derivative (R1). The vector $b$ is a derivative of moments, so the existing deposit tangent computes it for grid-shape test functions: set $\dot w = 0$ and read the raw kernel output, before the volume division and the spectral transform. **[derived]**

The basis for FBPIC (proposal, per converted species):

1. Field part: $\psi = S_g(z, r)\,e^{-im\varphi}$ for every grid node $g$ touched by $\mathcal C$, and $m \in \{0, 1\}$: three real rows per node ($m = 0$, and the real and imaginary parts of $m = 1$). These are the test functions of the $\rho$ deposit (linear shape, Ruyten radial correction; plan Section 4.7). With them in $V$, the reset leaves the $\rho$ tangent on the grid unchanged to rounding.
2. Beam part: $\psi = S_k(z)\times$ monomials of degree $\le p$ in $(x, y, u_x, u_y, u_z)$, with $S_k$ a coarse linear shape in the longitudinal position $z$ (slabs of about 1 µm). Degree 2 reproduces a correlated Gaussian under affine displacements exactly; degree 4 to 6 covers two-stream nodes (E1b). Add the 13 base sums of the loss ($\sigma$, $\sigma\xi_k$, $\sigma\xi_k^2$; plan Section 12.10, Design) as 13 global rows. Then the loss Jacobian column is continuous across a reset to rounding.

The basis needs no velocity mesh. Linear shear and rotation of a Gaussian (betatron rotation, free streaming) stay inside a degree-2 polynomial span, while a fixed mesh needs ever finer cells (E2c).

### 2.4 Total charge and current tangents

At any time, for every species,

$$
\delta\rho_g = q\sum_i \dot w_i\,S_g(\mathbf x_i) + q\sum_i w_i\,\xi_{x,i}\cdot\nabla S_g(\mathbf x_i),
$$

$$
\delta\mathbf J_g = q\sum_i \dot w_i\,\mathbf v_i\,S_g(\mathbf x_i) + q\sum_i w_i\big[\delta\mathbf v_i\,S_g(\mathbf x_i) + \mathbf v_i\,\xi_{x,i}\cdot\nabla S_g(\mathbf x_i)\big], \qquad \delta\mathbf v_i = c\big(\xi_{u,i}\gamma_i^{-1} + \mathbf u_i\,\dot{(\gamma^{-1})}_i\big),
$$

with the azimuthal phase and rotation terms of plan Section 4.7. The existing `deposit_tangent` (`tangent.py:873-1113`) computes both sums in one pass, because it takes `w` and `dw` together (call at `stepper.py:545`, `DEPOSIT_TANGENT_NAMES` at `stepper.py:439`). After a reset, $\xi = 0$ for the converted particles, and only the weighted deposit remains for them. **[code]**

Charge consistency at the reset. The curl-free correction uses $\rho_\text{prev}$ and $\rho_\text{next}$ (`fields/spectral_grid.py:219-252`). If the reset changed the $\rho$ tangent by $\Delta\rho$, the next step would add a longitudinal current $\propto \Delta\rho/\Delta t$ and move the longitudinal field tangent by the Coulomb field of $\Delta\rho$. Field part 1 of the basis sets $\Delta\rho = 0$ exactly. Without it, the mesh form (b) moves the density moment by up to 14% of the momentum response at $\sigma_u/\Delta u = 2$ (E1, observable `cos_t0`, which must stay 0). **[derived; measured]**

### 2.5 Relation to particle-pdf

With $T = 1$ and the mesh form (b), R-v converts the one-step kick $\Delta\xi_u = q\,\delta\mathbf F\,\Delta t/m$ every step. That is the collocated weight-evolution scheme of the paper (eq. 44, p. 18; eq. 54a, p. 20), with $(\delta\mathbf E\cdot\nabla_v f)_g$ computed as the divergence of a deposited dipole instead of the derivative of the velocity shape function (eq. 36, p. 15). The two differ at $O(\Delta u^2)$. With $T \to \infty$, R-v is the particle-exact method (paper eq. 19, p. 11). **[derived]**

## 3. Consistency

### 3.1 Between resets

R-v propagates $(\dot w, \xi)$ with the exact JVP of the discrete step (Section 2.2). After a reset, the state $(\dot w + \Delta w, 0)$ is the input tangent of a different one-parameter family of discrete initial states: the same positions with weights $w + \epsilon(\dot w + \Delta w)$. The propagation from there is again the exact JVP of the discrete map for that family. So no approximation is made between resets. **[derived]**

### 3.2 At a reset

Let $W = \mathrm{diag}(w_i)$, $\Psi_{ia} = \psi_a(z_i)$, and $V = \mathrm{span}\{\psi_a\}$. For an observable $\psi$, split $\psi = \Psi a + r$ with $\sum_i w_i\,r(z_i)\,\psi_b(z_i) = 0$ for every $b$. With (R2),

$$
\sum_i\Delta w_i\,\psi(z_i) - \sum_i w_i\,\xi_i\cdot\nabla\psi(z_i) = -\sum_i w_i\,\xi_i\cdot\nabla r(z_i). \tag{R3}
$$

Proof: $\sum_i \Delta w_i\,\psi(z_i) = c^T\Psi^T W(\Psi a + r) = c^T G a = b^T a = \sum_i w_i\,\xi_i\cdot\nabla(\Psi a)(z_i)$, because $\Psi^T W r = 0$. Subtract the Lagrangian value of $\psi = \Psi a + r$. **[derived]**

So the reset is exact for every $\psi \in V$. In the continuum limit, (R3) is $\int \nabla\cdot(f\xi)\,r\,dz = -\int (\delta f/f)\,r\,f\,dz$ with $\delta f = -\nabla\cdot(f\xi)$. Because $r = (I-P)\psi$ is $f$-orthogonal to $V$, this equals

$$
\epsilon_\text{reset}(\psi) = -\Big\langle (I - P)\frac{\delta f}{f},\ (I - P)\,\psi\Big\rangle_f, \qquad |\epsilon_\text{reset}| \le \Big\lVert (I-P)\frac{\delta f}{f}\Big\rVert_f\,\lVert (I-P)\psi\rVert_f, \tag{R4}
$$

where $P$ is the $f$-weighted projection onto $V$. The error is the product of two residuals. It vanishes when $\delta f/f \in V$, for example a Gaussian $f$ with an affine displacement and $p \ge 2$. **[derived]**

The relevant $\psi$ is not the final observable $\phi$. It is the linearized response of $\phi(t_f)$ to a weight perturbation at $t_r$: in a frozen field, $\psi = \phi\circ\Phi_{t_f\leftarrow t_r}$; with field feedback, the solution of the adjoint linearized system at $t_r$. **[derived]**

Consistency statement. As $N \to \infty$ and $V$ is refined, the reset error goes to 0 if $f(t_r)$, $\delta f(t_r)$, and $\psi(t_r)$ stay resolvable by $V$. Then R-v converges to the continuum linearized Vlasov-Maxwell sensitivity, as particle-pdf does (paper Section 6, eqs. 80 and 81, p. 29). If they do not stay resolvable, the needed refinement grows with time and the method is not usable at long horizons (Section 3.4).

### 3.3 Where the chaotic stretching is removed

Consider the dipole state $\xi(t_r)$ and an observable at $t_f$. The Lagrangian form contributes $\sum_i w_i\,\xi_i(t_r)\cdot\nabla\psi(z_i(t_r))$, with $\psi = \phi\circ\Phi_{t_f\leftarrow t_r}$. The weight form contributes $\sum_i \Delta w_i\,\psi(z_i(t_r))$. In the continuum both equal $\int f\,\xi\cdot\nabla\psi = -\int \nabla\cdot(f\xi)\,\psi$. The per-particle terms differ **[derived]**:

- Lagrangian: $|\nabla\psi| = |D\Phi^T\nabla\phi| \sim e^{\lambda(t_f - t_r)}$ per particle, with signs that vary from particle to particle. The Monte Carlo variance grows as $e^{2\lambda(t_f - t_r)}/N$.
- Weight: $|\Delta w_i|\,|\psi| \le |\Delta w_i|\,\lVert\phi\rVert_\infty$. Nothing grows after $t_r$, because $\dot w$ is a scalar that the flow carries without any Jacobian factor.

The identity behind this is the push-forward of a dipole measure: for any smooth map $\Phi$, $\Phi_\#\big(-\nabla\cdot P\big) = -\nabla\cdot\Phi_\#(D\Phi\,P)$. It needs no volume preservation. The Lagrangian form carries $D\Phi$; the weight form has already absorbed it. So the reset removes the stretching by one integration by parts at $t_r$. Between resets, $\xi$ restarts from 0 and grows at most by $e^{\lambda T}$.

### 3.4 Where the argument fails: mean-field chaos

R4 needs $\delta f/f$ resolvable at $t_r$. In the paper's Debye case the chaos is an N-body effect, and $f$ stays a smooth Maxwellian (paper Section 6.1, eqs. 77 to 81, p. 29: the spurious part is the derivative of the finite-$N$ error $\epsilon_N$). If the mean-field characteristics are chaotic, $f$ itself develops filaments of width $\propto e^{-\lambda t}$. Then $\nabla\ln f$ is not resolvable at late $t_r$, and every fixed basis loses accuracy. This is a property of the problem, not of R-v: particle-pdf with a fixed mesh has the same failure (E2 with $T = 1$). The same mechanism is known as weight growth in $\delta f$ PIC of turbulent plasmas (my recollection, not verified here).

Evidence from P7 does not decide the type. FD secants at $h = 10^{-2}$ and $10^{-3}$ agree (plan Section 12.10), so the coarse response is regular, which holds for both types. Two observations suggest some mean-field content [estimated]: the e-folding time is about 3 laser periods, which matches laser-driven orbit chaos better than slow grid noise, and the growth starts with trapping near step 1200 (separatrix crossing). The spread of the AD column over $10^4$ to $4\cdot10^5$ particles (plan Section 12.10, step 5) is compatible with both types. The P7 data do not decide the question. It is the most important open question for Track R.

## 4. Reset interval and error budget

### 4.1 When to start resetting

No conversion is needed before the chaos starts. The Lagrangian tangent matches FD to step 500, and the $\infty$-norm does not grow from step 600 to 1200 [plan]. The cold background is laminar there, and the conversion cannot represent an exactly cold sheet (E1). So R-v uses the exact Lagrangian tangent from step 0 to $t_1$, and starts resets at $t_1 \approx 1000$ to 1100, before trapping. Particle-pdf would have to convert at every step from the time the laser reaches the plasma (step $n_c = 450$), including the cold laminar phase where $\nabla_u f$ is singular. There, collocated weights cannot represent the source (E1, $\sigma = 0$); the paper's particle-addition scheme (eqs. 45 to 48, p. 18) can, at the cost of new sensitivity particles off the cold sheet. **[derived]**

### 4.2 Interval

Within one interval, a kick at time $s$ grows by $e^{\lambda(t_r - s)}$ along unstable directions. Relative to the linear accumulation $T\,|\delta F|$, the unstable part is amplified by

$$
A(T) = \frac{e^{\lambda T} - 1}{\lambda T}.
$$

With $\lambda = 0.02$ per step [plan, upper local rate]:

| $T$ (steps) | $\lambda T$ | $e^{\lambda T}$ | $A(T)$ | Resets from 1000 to 2000 |
|---|---|---|---|---|
| 10 | 0.2 | 1.22 | 1.11 | 100 |
| 25 | 0.5 | 1.65 | 1.30 | 40 |
| 50 | 1.0 | 2.72 | 1.72 | 20 |
| 120 | 2.4 | 11.0 | 4.2 | 8 |

**[derived]**

Error model [derived, heuristic]. Assume that the unstable part of the dipoles is incoherent between particles at the basis scale (N-body type). Then the deposit of $b$ averages it down by $N_\text{node}^{-1/2}$, and successive resets add independent errors. The relative error of the total sensitivity then scales as

$$
\varepsilon_\text{total} \approx \kappa\,A(T)\,\sqrt{\frac{T}{(t_\text{end} - t_1)\,N_\text{node}}},
$$

with $\kappa$ the fraction of each kick that projects on unstable directions ($\kappa \le 1$, unknown). This grows as $\sqrt T$ for $\lambda T \ll 1$ and as $e^{\lambda T}/\sqrt T$ for $\lambda T \gg 1$. Smaller $T$ always helps the error in this model. The reset error of Section 3.2 does not depend on $T$ to first order: each reset projects a dipole of size $\propto T$, and there are $\propto 1/T$ resets.

Choice: $T = 25$ steps ($\lambda T = 0.5$) as the default, with a sweep $T \in \{5, 10, 25, 50\}$ in R2 and R4. For $N_\text{node} \approx 10^3$ and $\kappa = 1$, the model gives $\varepsilon_\text{total} \approx 1.3\sqrt{25/(10^3\cdot10^3)} \approx 0.6\%$ [estimated]. Agreement of the columns across $T$ within the statistical error is the self-check; drift with $T$ shows chaos leaking through.

Adaptive trigger [derived, heuristic]. Per basis node $g$, compare the spread of the dipoles with their mean: $R_g = \mathrm{Var}_g(\xi)/|\overline{\xi}_g|^2$ (weights $w_iS_g$). The Lagrangian estimator has noise $\propto \mathrm{std}_g(\xi)$; the weight estimator has noise $\propto |\overline\xi_g|$. Convert node $g$ when $R_g > 1$. This keeps the exact Lagrangian form wherever it is still the lower-variance form, for example in the cold laminar background.

### 4.3 Subset to convert

Convert per species, never across species. Start with `elec_from_N`, the observed species. P7 shows that the field tangents grow too from step 1500 [plan], so the background electrons may need resets as well. That needs a per-species growth measurement (Section 8). Never convert N ions: their $\xi$ is small, their population tangents are already scalar, and their $\dot w_\text{tl}$ feeds the charge directly (`ionization.py:676-688`).

### 4.4 Where R-v is better and where it is worse than per-step particle-pdf

Better [derived]:

1. Steps 0 to about 1000: R-v is exact and needs no mesh. The cold background has a delta-function $f$ in $u$, which no collocated conversion can represent (E1, $\sigma = 0$). Particle-pdf must either add sensitivity particles off the sheet or smooth $f$; smoothing gives a bias of about $\Delta u^2/(6\sigma_u^2)$ when $\sigma_u > 0$ and fails at $\sigma_u = 0$ (E1).
2. Cost: 20 to 40 conversions instead of 2000 source evaluations. The Gram matrix depends only on the primal state, so all $K$ parameters share one factorization per reset. This matters for the 3 to 25 parameter goal.
3. The parameter source $\partial_\theta\mathcal V$ needs no separate treatment. Laser parameters enter only the initial fields (plan Section 4.8), and future density parameters enter as $\dot w$ at injection (plan Section 4.9).
4. A sweep in $T$ gives a self-check that particle-pdf does not have.

Worse [derived]:

1. Larger dipoles per conversion: $|\Delta w/w| \sim |\xi_u|/\sigma_u$ grows with $T$. In E1 this did not raise the variance for smooth $\xi$ (Galerkin with $\sigma = 0.01$ and $|\xi| = 1$ had the same scatter as the Lagrangian estimator). With a noisy $\xi$, the noise enters $b$ in proportion to $A(T)$.
2. If $\lambda$ is locally much larger than 0.02 (the $\infty$-norm rate is already the worst particle, but short bursts are possible), a fixed $T$ lets some particles stretch by more than $e^{\lambda T}$ before conversion.
3. Variance is not lower in principle. The conversion is linear in the dipoles. If the basis and the particles change little within $T$ steps, $T$ per-step conversions sum to one conversion of the accumulated dipole. So the advantage of R-v is cost and the exact early phase, not lower variance.

## 5. Implementation on the existing code

### 5.1 Reused unchanged

| Function | Location | Role in R-v |
|---|---|---|
| `TangentState` with `w` tangent | `fbpic/autodiff/state.py:105`, `:109-140` | $\dot w$ storage already exists; no new particle arrays |
| `gather_tangent` | `fbpic/autodiff/tangent.py:260-540` | $\xi$ propagation; after a reset its $\partial S\cdot\xi$ term is 0 for converted particles |
| `vay_tangent`, `vay_ion_tangent` | `tangent.py:649-713`, `:717-795` | kick $\Delta\xi_u$ from $\delta E, \delta B$ (the paper's source $\mathcal H$ in dipole form) |
| `position_tangent` | `tangent.py:799-855` | $\xi_x$ update |
| `deposit_tangent` | `tangent.py:873-1113` (uses `dw` at `:994`, `:1014`) | $\delta\rho$, $\delta J$ from both forms; also computes the field rows of $b$ with $\dot w = 0$ |
| `ev_tangent`, `ionize_tangent`, `charge_weight_tangents` | `fbpic/autodiff/ionization.py:237`, `:690-764`, `:676` | birth weight source $\dot W'$ (already weight form) and parent $\xi$ copy |
| `sort_all`, `_realign_tangents` | `fbpic/autodiff/stepper.py:137`, `:291-308` | carry $\dot w$ with the particles, drop it at removal, zero at injection |
| field tangents | `fbpic/autodiff/field_tangents.py:45-118` | linear field operators, unchanged |
| `ResidualValue.contract` | `fbpic/autodiff/loss.py:264-282` (`t["w"]` at `:273`) | observable sensitivity from both forms |

### 5.2 New code

1. `project_tangents(state, tangents, species, basis, mask)`: builds $b$ (the deposit tangent with $\dot w = 0$ for the field rows; CuPy reductions for the polynomial and loss rows), assembles $G$, solves (R2) for all $K$ tangents, writes $\dot w \mathrel{+}= \Delta w$, and zeros `x, y, z, ux, uy, uz, inv_gamma` of the converted particles.
2. One Warp kernel that assembles $G$: per particle, the outer product of its nonzero test functions (4 grid nodes x 3 real mode rows for the field part, plus the polynomial rows of its $z$ slabs), added with atomics. It has the structure of the deposit kernel. The deposit-tangent cost issue of P7 (one thread per dense cell) applies; a particle-parallel atomic form avoids it.
3. A sparse solve of about $3N_\text{nodes}$ field unknowns plus $N_\text{slabs}\cdot 126$ polynomial unknowns (degree 4 in 5 variables; degree 6 gives 462 per slab). CuPy sparse CG or a direct solve. The system is shared by all $K$ tangents.
4. A reset schedule in `driver.jvp` and `value_and_grad` (`driver.py:144`, `:225`): after `step`, at the step boundary, call `project_tangents` when the schedule or the trigger fires.
5. Checks: (i) the 13 loss base sums and the $\rho$ tangent before and after a reset agree to rounding; (ii) a reset at step 450 or 500, where the Lagrangian tangent is verified, gives a column within the statistical error of the AD column at $N_t = 500$.

EV births need nothing new. A born electron starts with $\dot w = \dot W'$ (weight form) and with the parent's $\xi$ (small and regular for N ions). After the next reset, its $\xi$ part also becomes weight. The emission threshold stays a frozen decision (plan Section 4.5), as in the current AD.

### 5.3 Memory and time

| Item | Estimate |
|---|---|
| New per-particle state | none ($\dot w$ exists) |
| Gram matrix, field part | at most about $1.4\cdot10^5$ unknowns ($\approx 900\times50$ nodes $\times$ 3 real rows), 9-point stencil of $3\times3$ blocks: below 0.1 GB in CSR, much less if restricted to the beam region |
| Gram matrix, polynomial part | about 100 slabs $\times\ 126^2\times 8$ B $\approx$ 13 MB |
| Per-reset temporaries | $c$ evaluated per particle, $K\times N$: 64 MB per tangent at 8 M particles |
| Time per reset | $b$: one $\rho$ deposit tangent restricted to $\mathcal C$ (at most 40 ms; plan Section 12.10 step 3); $G$: 2 to 5 deposit costs (50 to 120 ms); solve: 0.05 to 0.5 s; $\Delta w$: one gather-like pass (10 ms). Total 0.2 to 0.7 s |
| Gradient overhead | 20 to 40 resets: 4 to 28 s, against 298 s for a $K = 1$ gradient (1% to 10%) |

All rows are **[estimated]** from the P7 profile (deposit tangents 81 + 40 ms per step at 9.8 M `elec_from_N`; primal deposits 24 + 14 ms). The device peak stays near the measured 3.78 GB plus at most 0.5 GB.

For comparison, the paper reports 1.89 to 2.96 PIC-run costs for its three particle-pdf schemes in 1D-1V (Section 7.1, p. 35). That cost is on top of the PIC run. R-v adds 1% to 10% on top of the existing tangent run [estimated]; the tangent run itself is 3.47 run-equivalents [plan].

## 6. CPU experiments

All runs use NumPy 1.26.4 and SciPy 1.16.1, `python3 -I`, on the host CPU. No GPU was used.

### 6.1 E1: conversion of a warm-to-cold beam, 1D-1V

Script: `work_rv/e1_cold_conversion.py`; output `work_rv/e1_results_N100000.json`. $f = (2\pi)^{-1}\,\mathcal N(u;\,0.5\cos x,\,\sigma^2)$ on a periodic $x$, $\xi_u = 1 + 0.5\cos x$, $\xi_x = 0$. $N = 10^5$, 8 seeds, reference = Lagrangian (exact) at $N = 4\cdot10^6$. Converter A is the mesh form (b) with 32 $x$-cells and $\Delta u \in \{0.05, 0.2\}$. Converter B is the Galerkin form (c) with linear $x$-shapes times polynomials in $u$ of degree $p$. Observables: $\langle u\rangle$, $\langle u^2\rangle$, and $\cos(x + ut)$ at $t \in \{0, 5, 20\}$ (free streaming after the conversion; the exact value at $t = 0$ is 0).

$\delta\langle u\rangle$ (exact 1.000) **[measured]**:

| $\sigma$ | A, $\Delta u = 0.05$ | A, $\Delta u = 0.2$ | B, $p = 2$ | Lagrangian |
|---|---|---|---|---|
| 0.5 | 0.993 | 0.973 | 1.0005 | 1.0005 |
| 0.2 | 0.988 | 0.859 | 1.0003 | 1.0003 |
| 0.1 | 0.962 | 0.637 | 1.0004 | 1.0004 |
| 0.05 | 0.884 | 0.379 | 1.0002 | 1.0002 |
| 0.02 | 0.700 | 0.109 | 1.0001 | 1.0001 |
| 0.01 | 0.566 | 0.061 | 0.9997 | 0.9997 |
| 0 | $-0.22$ | $-0.27$ | 0.998 | 0.9997 |

Findings:

1. Mesh form A has a smoothing bias. The fit $\delta\langle u\rangle_A \approx \sigma^2/(\sigma^2 + c\,\Delta u^2)$ gives $c = 0.13$ to $0.18$, close to the variance $\Delta u^2/6$ of a linear-shape deposit followed by a linear-shape interpolation **[measured; derived interpretation]**. A 5% bias needs $\Delta u \lesssim 0.55\,\sigma_u$ in every resolved direction.
2. Mesh form A also creates a spurious density response: $\cos$ at $t = 0$ is 0.0066 ($\sigma = 0.5$), 0.14 ($\sigma = 0.1$), and 0.79 ($\sigma = 0.02$) at $\Delta u = 0.05$, against 0 exactly.
3. Galerkin form B agrees with the reference within the seed standard deviation (8 seeds) for all $\sigma \ge 0.01$ and all observables, with one exception: at $\sigma = 0.5$, $\cos(x + 20u)$ has an absolute offset of 0.010, where the exact value is only 0.013 (the observable oscillates faster in $u$ than the width of $f$, so $(I-P)\psi$ is large). The scatter of B equals that of the Lagrangian estimator. At $\sigma = 0.01$ the weight changes are about 100 times the weights, but the scatter does not grow.
4. For an exactly cold sheet ($\sigma = 0$), both forms fail. B keeps $\langle u\rangle$ and $\langle u^2\rangle$ (they are in $V$), but gives $\cos$ at $t = 0$ equal to 2.00 instead of 0, and $-3.32$ instead of $-2.49$ at $t = 5$. Weights on a sheet can only represent a measure on the sheet; a momentum shift moves mass off the sheet ($\delta'$ normal to it). The failure is silent: nothing in the solve flags it. **[measured; derived reason]**

### 6.2 E1b: two-stream node and nonlinear displacement

Script: `work_rv/e1b_two_stream.py`; output `work_rv/e1b_results_N100000.json`. Two streams at $u = \pm1 + 0.3\cos x$, width $\sigma$, and $\xi_u = \tanh(u/0.3)(1 + 0.5\cos x)$ (opposite shifts). Extra observable `sel_u` $= S_5((u - 0.5)/0.4)\,u$, a smooth selection like the loss.

Relative error of `sel_u` **[measured]**:

| $\sigma$ | A, $\Delta u = 0.05$ | B, $p = 2$ | B, $p = 4$ | B, $p = 6$ |
|---|---|---|---|---|
| 0.3 | $-1.1\%$ | $-5.6\%$ | $-1.2\%$ | $-0.6\%$ |
| 0.1 | $-4.5\%$ | $-0.5\%$ | $0.0\%$ | $-0.1\%$ |
| 0.03 | $-27\%$ | $+0.2\%$ | $0.0\%$ | $-0.1\%$ |

At $\sigma = 0.3$, $\cos(x + 5u)$ has errors of $-21\%$, $-9\%$, and $-10\%$ for $p = 2, 4, 6$ (the seed standard deviation is about 10%). Degree 4 to 6 is needed when one node holds two populations.

### 6.3 E2: periodic resets on a chaotic ensemble (mean-field chaos)

Script: `work_rv/e2_standard_map_reset.py`; output `work_rv/e2_results_K1.5_nf25.json`. Standard map $p' = p + K\sin x$, $x' = x + p'$ on the torus, $\theta = K = 1.5$, Gaussian blob of width 0.3, observable $\phi = \cos x + 0.5\sin p$ at $n_f = 25$. The map preserves area, so weights are constant along orbits, as in the Vlasov flow. There is no self-consistent field. The chaos is in the single-particle (mean-field) dynamics, and $f$ filaments. Median Lyapunov rate 0.145 per step, $e^{\lambda n_f} = 37$.

Reference: central FD with common random numbers, $N = 2\cdot10^7$: $0.728 \pm 0.008$ at $h = 0.01$ and $0.636 \pm 0.003$ at $h = 0.03$. A quadratic extrapolation in $h$ gives about $0.74 \pm 0.02$.

| Estimator ($N = 10^6$, 6 seeds) | Mean | Std |
|---|---|---|
| Lagrangian, no reset | 22 | 316 |
| R-v mesh $M = 32$, $T = 1$ | 0.388 | 0.018 |
| R-v mesh $M = 64$, $T = 1$ | 0.212 | 0.027 |
| R-v mesh $M = 128$, $T = 1$ | $-0.175$ | 0.040 |
| R-v mesh $M = 64$, $T = 3$ | $-0.323$ | 0.028 |
| R-v mesh $M = 64$, $T = 8$ | 0.015 | 0.196 |
| R-v mesh $M = 64$, $T = 12$ | 0.92 | 1.08 |

**[measured]** The resets remove the variance (factor $10^4$), as Section 3.3 predicts. They do not give the right value. The result does not converge with $M$ or $T$.

Calibration (E2b, `work_rv/e2b_calibration.py`, output `work_rv/e2b_results.json`): at short horizons the Lagrangian estimator is exact and has a small variance, so it is the reference. At $n_f = 2$, one conversion with $M = 32, 64, 128$ gives errors of 0.029, 0.0074, 0.0021 against 0.0562: second-order convergence, so the converter is correct. At $n_f = 8$ with $T = 1$, $M = 128$ gives 0.51 against 0.77; the error grows with time as the blob shears and folds.

E2c (`work_rv/e2c_single_conversion.py`, output `work_rv/e2c.log`; rows $t_r = 1$ to 6 were complete when this note was written): one conversion at step $t_r$, then 2 exact steps, paired with the Lagrangian value on the same particles. The polynomial basis uses the unwrapped coordinates. Relative error **[measured]**:

| $t_r$ | Mesh 64 | Mesh 128 | Mesh 256 | Poly 2 | Poly 4 | Poly 6 | Poly 8 |
|---|---|---|---|---|---|---|---|
| 1 | $-2.8\%$ | $-0.7\%$ | $-0.2\%$ | $-0.7\%$ | $-0.1\%$ | $0.0\%$ | $0.0\%$ |
| 2 | $-4.5\%$ | $-1.1\%$ | $-0.1\%$ | $+48\%$ | $+11\%$ | $+4.1\%$ | $+3.6\%$ |
| 3 | $-6.1\%$ | $-1.8\%$ | $-0.4\%$ | $-18\%$ | $+3.4\%$ | $+0.6\%$ | $-4.9\%$ |
| 4 | $-108\%$ | $-38\%$ | $-9.7\%$ | $-57\%$ | $-26\%$ | $-12\%$ | $+3.1\%$ |
| 5 | $-25\%$ | $+41\%$ | $+60\%$ | $-102\%$ | $-232\%$ | $+124\%$ | $+130\%$ |
| 6 | $-21\%$ | $+18\%$ | $+60\%$ | $-137\%$ | $-105\%$ | $+74\%$ | $+66\%$ |

One particle sample ($N = 10^6$) per row; the comparison is paired, so the numbers include the statistical error of the conversion itself. Up to $t_r = 3$ the mesh converges at second order and the polynomial basis converges with degree. At $t_r = 4$ the blob folds over (the sine kick bends it), and only the finest bases stay within 10%. From $t_r = 5$ no basis is within 18%, and refining the mesh makes the error larger, because $\hat n_g$ becomes noisy before the folds are resolved.

E2b at longer horizons with resets every step or every second step (mean of 4 seeds, $M = 128$; reference = Lagrangian at $N = 4\cdot10^6$): $n_f = 4$: 0.164 and 0.179 against 0.182; $n_f = 8$: 0.513 and 0.673 against 0.768; $n_f = 12$: 0.728 and 1.073 against 0.734. The results scatter around the reference with errors of 1% to 46% that do not shrink in a systematic way.

Interpretation: the error of a conversion at $t_r$ follows the deformation of $f$ at $t_r$. This is R4: $(I - P)(\delta f/f)$ grows as $f$ shears and folds. A low-degree polynomial basis handles linear shear but not folding. A mesh needs cells finer than the filament width. **[derived from measurement]**

Scope of E2: it is mean-field chaos only, without field feedback and without N-body noise. It does not test the N-body case of the paper, where R-v is expected to work. Phase R1 (Debye reproduction) tests that case.

## 7. Comparison summary for the R0 note

| Aspect | Particle-pdf (paper, per step) | R-v |
|---|---|---|
| Steps 0 to about 1000 (laminar, cold) | converts every step; cold $\nabla_u f$ needs regularization | exact Lagrangian, no conversion |
| Number of conversions | 2000 | 20 to 40 (or adaptive) |
| Source term | mesh $\delta F\cdot\nabla_u f$ every step | carried as $\xi$ by the verified kernels |
| Basis | 6D mesh (not feasible here) or the Galerkin basis of Section 2.3 | same choice; the Galerkin basis is recommended for both |
| Mean-field chaos with filamenting $f$ | fails (E2, $T = 1$) | fails the same way |
| N-body chaos, smooth $f$ | works (paper) | expected to work; error grows with $A(T)$ |
| New code | source evaluation, $\hat n$, $h_s$, weight update every step | one projection routine, one assembly kernel, a schedule |
| Extra cost on the tangent run | about 1 to 2 PIC runs (paper, 1D-1V) [paper] | about 1% to 10% [estimated] |

## 8. Kill criteria, open questions, and next steps

Kill criteria for Track R (they apply to R-v and to particle-pdf):

1. Chaos type. On Twin-S, the growth rate of the median (not maximum) $|\xi|$ in the selected beam after step 1200 does not fall when the particle count doubles (`p_nt` 8 to 16, or `p_nr` 2 to 4) and stays within 20% of its value. That indicates mean-field chaos, as in E2.
2. Resolvability. At $t_r = 1500$, the degree-4 Galerkin conversion of the beam changes a held-out moment (not in $V$, for example $\langle x^2u_x^2\rangle$ or $\cos(k\zeta)$ moments) by more than 10% of its Lagrangian derivative, and degree 6 does not halve the difference.
3. Interval drift. On Twin-S at $N_t = 2000$, the R-v columns for $T = 10$, 25, and 50 differ by more than 3 times their statistical spread (from 4 to 8 particle-seed replicas), or the median cosine to the smoothed FD column of P7 step 5 stays below 0.9.
4. Short-horizon check. A reset at step 500 changes the AD column at $N_t = 500$ by more than 5% (the Lagrangian tangent is verified there, so the reset alone must not move it).
5. Cold sheet. More than 10% of the weight of a converted node lies in nodes whose momentum covariance has a smallest eigenvalue below $(0.55\,\Delta u_\text{eff})^2$ for the chosen basis; this is the E1 failure mode.

Open questions:

1. Which species carries the field-tangent growth after step 1500: `elec_from_N` or the background electrons? It decides the subset $\mathcal C$. Measure the per-species $\infty$-norm and median of $\xi$ with `diag_tangent_growth.py` extended per species.
2. Is the median growth rate of the beam close to the $\infty$-norm rate 0.02? It sets $T$.
3. Does the PSATD current correction need the exact $\rho$ row of the basis, or is the $O(\text{basis})$ mismatch harmless? The exact row is cheap, so I recommend it.
4. Is the Vay pusher volume-preserving to the precision that matters? It does not affect (b) or (c), which are conservative. It only rules out the local form (a).
5. The conversion assumes that each converted node is warm. The separatrix region (trapped against untrapped `elec_from_N`) is a sharp edge in phase space; the selection $\sigma(u_z)$ of the loss sits on it. E1b suggests degree 4 to 6, but this is untested in 3V.

Recommended changes to the Track R phases (for the user):

1. R1: add to the Debye reproduction an R-v run with $T \in \{1, 5, 25\}$, and the Galerkin converter next to the paper's mesh converter.
2. R2: run the chaos-type test (kill criterion 1) on Twin-XS and Twin-S before building any GPU kernel. It needs only the existing code and two short GPU runs.
3. R3: build R-v first (Section 5.2). A per-step particle-pdf source evaluation is not needed, because R-v with $T = 1$ is that method.
