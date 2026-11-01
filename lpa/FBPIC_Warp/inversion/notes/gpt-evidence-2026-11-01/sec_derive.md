# Gauge-projected tangent (GPT): formal derivation for relativistic Vlasov-Maxwell

| Item | Value |
|---|---|
| Task key | `derive` |
| Date | 2026-11-01 |
| Scope | Forward mode only. Twin-S (cold plasma, EV-injected bunch). |
| Scripts | `scratchpad/mt/work_derive/` (full path in Section 8). CPU, NumPy, `python3 -I`. |

Labels: **[derived]** derived here; **[measured]** measured with the named script; **[lit]** from the literature (citation given); **[R0]** from the R0 design note `inversion/notes/r0-design-note-2026-10-31.md`; **[estimated]** an estimate with stated inputs.

## 0. Verdict

1. **Continuum: GPT is correct.** The $L^2(f)$ projection onto gradient fields keeps $\delta f$ exactly. It therefore keeps $\delta\rho$, $\delta\mathbf J$, every weighted moment, and $\mathrm{Cov}(z,\xi)$. The removed part stays a gauge under the linearized Vlasov-Maxwell flow, so the removal never changes a later result [derived, Section 2].
2. **Cold sheets: GPT does not stop the growth.** On a sheet, GPT keeps the full normal component and only removes the density-preserving tangential part. On a 1-D sheet that part is at most one-dimensional. On any sheet, the minimal (Wasserstein) norm itself grows with the stretching factor when the source changes the density along the sheet. Measured on a chaotic standard-map sheet: $\lVert P\xi\rVert/\lVert\xi\rVert \ge 0.995$ at every step while $\lVert\xi\rVert$ grows by $10^4$ [derived, Section 3; measured, D2]. The doubling-map success needs a source that only relabels an invariant measure. That is a special case.
3. **Discrete: GPT inherits the Lagrangian noise.** For every $\psi$ in the basis span, the projected tangent reproduces the particle Lagrangian sum $\sum_iw_i\xi_i\cdot\nabla\psi(z_i)$ to rounding, including its $e^{\lambda t}/\sqrt N$ noise [derived; measured, D1]. The empirical measure has no gauge at all. Every discrete "gauge removal" is a truncation by the finite basis, that is, a coarse-graining at the basis scale. A per-cell (broken) basis keeps the cell-mean tangent and therefore keeps the coherent doubling-map growth [measured, D1].
4. **What GPT adds over R-v.** GPT keeps displacements, so its charge and current rows stay independent on a cold node. It represents $\delta\mathbf v \neq 0$ on a cold sheet, where the R-v weight conversion is inconsistent [derived, Section 4.4; R0 A.4]. It is a cold-safe Galerkin coarse-graining. It is not a scale-free derivative.
5. **Measured end-to-end (D4).** With projections every 1 to 3 steps on two chaotic toys (a cold sheet and the warm R0 E2 blob), GPT is stable but wrong in the chaotic phase: errors of 45% to 97% of the reference, which change with the basis size and $T$. This repeats the R0 R-v result [measured, D4].
6. **Recommendation.** Do not build GPT as the fix for Twin-S chaos. If the team wants a cold-safe coarse-grained derivative to compare with FD at matched scale, GPT with a conforming basis is the better-posed choice than R-v. Section 7 gives the cost; Section 9 gives the one test that decides it.

## 1. Setting and notation

Per species $s$, phase-space point $z = (\mathbf x, \mathbf u)\in\mathbb R^6$, $\mathbf u = \mathbf p/(m_sc)$, $\gamma = \sqrt{1+u^2}$, $\mathbf v = c\mathbf u/\gamma$. The characteristic flow $\Phi_{t_0\to t}$ preserves phase-space volume (R0 A.1). Markers carry $(w_i, z_i)$ and the exact forward tangent $(\dot w_i, \xi_i)$, $\xi_i = \partial z_i/\partial\theta$. The empirical measure and its tangent are

$$
\mu = \sum_iw_i\delta_{z_i},\qquad
\mu' = \sum_i\dot w_i\delta_{z_i} - \nabla_z\cdot\Big(\sum_iw_i\xi_i\delta_{z_i}\Big).
$$

In the continuum, $\delta f = \delta f_w - \nabla_z\cdot(f\xi)$. GPT acts only on $\xi$. The weight part $\delta f_w$ (EV birth monopole $\dot W'$, density parameters) is untouched.

Metric. Let $\mathsf M = \mathrm{diag}(\ell_x^{-2}I_3, \ell_u^{-2}I_3)$ fix the units of $\lVert\xi\rVert$. Define $\langle a, b\rangle_f = \int f\,a\cdot\mathsf Mb\,dz$. The projection depends on $\mathsf M$. The preserved quantities (Section 2) do not.

Gauge set: $\mathcal G_f = \{\eta : \nabla_z\cdot(f\eta) = 0 \text{ in } \mathcal D'\}$, equivalently $\int f\,\eta\cdot\nabla\psi = 0$ for all smooth $\psi$.

## 2. Continuum theory

**T1 (orthogonal decomposition) [derived].** The $\langle\cdot,\cdot\rangle_f$-orthogonal complement of $\{\mathsf M^{-1}\nabla\phi\}$ is $\mathcal G_f$, because $\langle\eta, \mathsf M^{-1}\nabla\phi\rangle_f = \int f\eta\cdot\nabla\phi$. Hence $\xi = \mathsf M^{-1}\nabla\phi + \eta$, $\eta\in\mathcal G_f$, with $\phi$ the solution of the weighted elliptic problem

$$
-\nabla_z\cdot\big(f\,\mathsf M^{-1}\nabla_z\phi\big) = -\nabla_z\cdot(f\xi).
$$

Its minimizer is the minimal-norm representative of $\delta f$. For $\mathsf M = I$ this is the tangent vector of $f$ in the Otto-Wasserstein geometry, and $\lVert\nabla\phi\rVert_{L^2(f)} = \lVert\delta f\rVert_{\dot H^{-1}(f)}$ [lit: F. Otto, Comm. PDE 26 (2001); Ambrosio, Gigli, Savaré, *Gradient Flows*, Birkhäuser 2008, Ch. 8; from memory, not re-checked].

**T2 (what is kept) [derived].** For any $\psi\in C^1$, $\int\psi\,\delta f = \int f\,\xi\cdot\nabla\psi$, and the $\eta$ part contributes $\int f\eta\cdot\nabla\psi = 0$. So GPT keeps every linear functional of $\delta f$:

1. $\delta\rho_g^m = q\int\delta f\,S_g(z,r)e^{-im\varphi}$ and $\delta\mathbf J_g^m = q\int\delta f\,\mathbf v(\mathbf u)S_g e^{-im\varphi}$, for every grid node and mode.
2. Every weighted beam moment $\int\chi(z)\,z^\alpha\,\delta f$ with a smooth selection weight $\chi$.
3. The spreads. $\delta\sigma_k^2 = \delta\langle z_k^2\rangle - 2\langle z_k\rangle\delta\langle z_k\rangle$ (plus the charge-normalization terms) uses only $\int z_k^2\delta f$, $\int z_k\delta f$, $\int\delta f$. In particle form this is $2\,\mathrm{Cov}_w(z_k, \xi_k)$; in the continuum it is invariant.

So the residual Jacobian is invariant under GPT in the continuum. A hard selection cut ($\chi$ an indicator) is covered as the limit of smooth $\chi$.

**T3 (the gauge stays a gauge) [derived].** Let $\eta_0\in\mathcal G_{f(t_0)}$ and propagate it with the tangent equations with zero field tangents: $\eta(t) = D\Phi_{t_0\to t}\,\eta_0$. The pushforward identity (R0 A.5) gives

$$
\nabla\cdot\big(f(t)\,\eta(t)\big) = \Phi_{\#}\big(\nabla\cdot(f(t_0)\eta_0)\big) = 0 .
$$

So $\eta$ deposits $\delta\rho = 0$ and $\delta\mathbf J = 0$ at all later times, the field tangents stay zero, and the zero-field assumption is self-consistent. By linearity, removing $\eta$ at $t_0$ changes no later $\delta f$, field tangent, or residual. This is the precise sense of "the gauge part changes nothing macroscopic".

**T3 exception: ionizing parents [derived].** The EV birth source is $G = B(|\mathbf E(\mathbf x)|)\,f_{\rm N}$ (rate times parent density). A parent gauge $\eta$ changes it by $-\nabla\cdot(Bf_{\rm N}\eta) = -f_{\rm N}\,\eta\cdot\nabla B \ne 0$. So a gauge of the nitrogen ions is not a gauge of the coupled system. Do not project the `nitrogen` markers. Ions move little; leave all ions Lagrangian.

**T4 (minimal norm) [derived].** $\lVert P\xi\rVert_f\le\lVert\xi\rVert_f$ with equality if and only if $\xi$ is already a gradient. GPT therefore reduces the norm only by the gauge content. It does not bound $\lVert P\xi\rVert$ by anything independent of the flow. Section 3 shows that on a sheet the bound grows.

## 3. Cold sheets

### 3.1 Projection on a measure supported on a sheet

Let $f = n\,\mathcal H^k\llcorner\mathcal M$: a density $n$ on a $k$-dimensional submanifold $\mathcal M\subset\mathbb R^6$. Cases on Twin-S [R0 4.1]: the background electrons, $k = 3$ (sheet $\mathbf u = \mathbf U(\mathbf x, t)$, folded after crossing); the `elec_from_N` population, $k = 4$ (labels: parent position and birth time).

$L^2(f)$ contains only fields on $\mathcal M$. Split $\xi = \xi_T + \xi_N$ ($\mathsf M$-orthogonal, tangent and normal to $\mathcal M$).

**C1 [derived].** The normal derivative of $\phi$ on $\mathcal M$ is free (extend $\phi$ off $\mathcal M$ arbitrarily). So the normal part of a gradient field is arbitrary, and the tangential part is $\mathrm{grad}_{\mathcal M}(\phi|_{\mathcal M})$ in the induced metric. Hence

$$
P\xi = \xi_N + \Pi_{\mathcal M}\xi_T,\qquad
\mathcal G_f = \{\eta_T \text{ tangent} : \mathrm{div}_{\mathcal M}(n\eta_T) = 0,\ n\eta_T\cdot\nu = 0 \text{ on }\partial\mathcal M\},
$$

with $\Pi_{\mathcal M}$ the $L^2(n\,dA)$ Helmholtz projection onto tangential gradients on $\mathcal M$. **GPT never touches the normal component.**

**C2: dimension of the gauge [derived].**

1. $k = 1$, open arc (a finite filament): every tangential field is $\partial_s$ of its antiderivative, so $\mathcal G_f = \{0\}$ and $P = I$.
2. $k = 1$, closed curve: $\mathcal G_f = \{C/n\}$, one dimension. In 1D-1V this covers the whole cold background: GPT is the identity up to one scalar.
3. $k\ge2$: $\mathcal G_f$ is infinite-dimensional (density-weighted solenoidal fields on $\mathcal M$). On the EV manifold the birth front $t_b = t$ is part of $\partial\mathcal M$, with inflow, so the no-flux condition applies there.

**C3: single-stream background equals linearized cold fluid [derived].** Use labels $\mathbf x$ on $\mathcal M = \{\mathbf u = \mathbf U(\mathbf x)\}$. Write $\xi = (\xi_x, \xi_u)$. Then

$$
\delta f = \delta n\,\delta(\mathbf u - \mathbf U) - n\,\delta\mathbf U\cdot\nabla_u\delta(\mathbf u - \mathbf U),\qquad
\delta n = -\nabla_x\cdot(n\xi_x),\qquad
\delta\mathbf U = \xi_u - (\xi_x\cdot\nabla_x)\mathbf U .
$$

These are the Eulerian density and velocity perturbations of the linearized cold relativistic fluid, written from a Lagrangian displacement. GPT keeps $(\delta n, \delta\mathbf U)$ and so $\delta\mathbf J = q(\delta n\,\mathbf V + n\,D\mathbf v(\mathbf U)\,\delta\mathbf U)$. Its gauge is $\xi_x = \boldsymbol\eta$ with $\nabla\cdot(n\boldsymbol\eta) = 0$ and $\xi_u = (\boldsymbol\eta\cdot\nabla)\mathbf U$, the density-preserving relabelings. So, while single-stream, the projected tangent is the minimal-norm Lagrangian displacement of the exact linearized cold-fluid perturbation. After orbit crossing the fluid picture fails, but C1 still holds stream by stream in phase space.

**C4: the growth on a Lagrangian sheet is tangential [derived].** The background starts at rest with $\mathbf A = 0$, so its canonical momentum is zero and $\mathcal M_0$ is a Lagrangian submanifold of the canonical form $\omega = d\mathbf P\wedge d\mathbf x$, $\mathbf P = m c\mathbf u + q\mathbf A$. The flow is symplectic, so $\mathcal M_t$ stays Lagrangian. For a tangent vector $\tau$ of $\mathcal M$, $\omega(\xi, \tau)$ depends only on $\xi_N$, and the homogeneous part of $\omega(\xi, F\tau_0)$ is conserved. When $\mathcal M_t$ aligns with the unstable directions (stretching of $F = \partial z/\partial\mathbf a$), the normal component stays bounded, and the growth is in $\xi_T$. The EV manifold is not Lagrangian (children are born at the ion momentum inside the laser, where $\nabla\times\mathbf A\neq0$), so this argument does not cover it.

Measured on a cold standard-map sheet (D2): $\lVert\xi_N\rVert$ stays between 0.6 and 1.7 for 14 steps, while $\lVert\xi_T\rVert$ grows from 0.8 to $1.0\cdot10^4$ [measured].

**C5: the minimal norm grows with the stretching [derived].** Write $\xi_T = F\zeta$ with $\zeta$ the label-space displacement and $n_0$ the label density. The tangential part of $\delta f$ is $\Phi_\#(\delta n_a)$, $\delta n_a = -\nabla_a\cdot(n_0\zeta)$. With the pulled-back metric $g = F^{\mathsf T}\mathsf MF$,

$$
\lVert\Pi_{\mathcal M}\xi_T\rVert^2 = \sup_\chi\frac{\big(\int\chi\,\delta n_a\,da\big)^2}{\int n_0\,\nabla_a\chi\cdot g^{-1}\nabla_a\chi\,da}.
$$

Stretching by $e^{\lambda t}$ along a sheet direction shrinks $g^{-1}$ by $e^{-2\lambda t}$ in that direction. If $\delta n_a$ varies along it, the supremum grows as $e^{\lambda t}$. Example: $x = e^{\lambda t}a$ on $a\in[0,1]$, $n_0 = 1$, $\delta n_a = \sin 2\pi a$; then the transport field is $f^{-1}\int_0^x\delta f = O(e^{\lambda t})$ and $\lVert\Pi\xi_T\rVert = O(e^{\lambda t})$. To realize a density change along a long filament, mass must move along the filament, over a distance that grows with its length. Moving across folds would be cheap, but on an exact sheet that is a normal displacement, which makes a dipole layer and not a density change along the sheet.

Consequences:

1. The hope "the $\dot H^{-1}(f)$ norm damps under phase mixing" holds for a smooth (warm) $f$, where transport across filaments is allowed. On an exact cold sheet it fails. It returns only after coarse-graining $f$, which reintroduces the R0 smoothing scale [derived; R0 Section 2].
2. The doubling map works because its source is a pure relabeling of an invariant measure: $\delta n_a\equiv0$. A Lorentz-force source $\delta\mathbf a\,\Delta t$ has $\delta n_a\ne0$ in general.
3. Measured (D2, sheet with 1-D gauge): $\lVert P\xi\rVert/\lVert\xi\rVert$ is 0.995 to 1.000 at every step; both grow by $10^4$ over 14 steps.
4. Measured (D3): $\delta n_a$ itself is not bounded. Its median grows from 0.86 to 6.0 over 9 steps, and its 99th percentile from 7 to $2.5\cdot10^3$, with spikes at folds. A label-space integration by parts ("weights in Lagrangian coordinates") is therefore no fix either: with $N = 2\cdot10^4$ its Monte Carlo spread is $2.6\cdot10^3$ at step 9, against 0.93 for the plain Lagrangian sum.

**C6: the cold-fluid question [derived].** "Does $P\xi$ equal the linearized cold-fluid perturbation?" Yes, in the sense of C3: the same $\delta f$, so the same $(\delta n, \delta\mathbf U)$ while single-stream, and the same per-stream phase-space perturbation after crossing. But that equality does not make the result small. The linearized cold fluid itself develops $e^{\lambda t}$ gradients when the characteristics are chaotic, and its representation by displacements has the norm of C5.

## 4. Discrete algorithm

### 4.1 Definition and exact properties

Basis $\{\psi_k\}_{k=1}^m$; columns $A_{ik} = \mathsf M^{-1}\nabla\psi_k(z_i)\in\mathbb R^6$. Projection:

$$
c = \arg\min_c\sum_iw_i\Big\lVert\xi_i - \sum_kc_kA_{ik}\Big\rVert_{\mathsf M}^2,\qquad
Gc = b,\quad G_{kl} = \sum_iw_i\nabla\psi_k\cdot\mathsf M^{-1}\nabla\psi_l,\quad b_k = \sum_iw_i\,\xi_i\cdot\nabla\psi_k(z_i),
$$

and $\xi_i\leftarrow(P\xi)_i = \sum_kc_kA_{ik}$.

**D-T1 (exact discrete preservation) [derived].** The normal equations give $\sum_iw_i\nabla\psi\cdot(\xi_i - P\xi_i) = 0$ for every $\psi\in\mathrm{span}\{\psi_k\}$. So every such $\psi$ keeps its particle derivative $\sum_iw_i\xi_i\cdot\nabla\psi(z_i)$ to rounding. A functional outside the span is not kept in general.

**D-T2 (noise inheritance) [derived].** $P\xi = AG^{-1}b$, and $b_k$ is the particle Lagrangian derivative of $\int\psi_k\,d\mu$. In a chaotic phase $b_k$ carries the full pathwise noise, about $e^{\lambda t}\lVert\nabla\psi_k\rVert/\sqrt{N_{\rm eff}}$. GPT reproduces that noise exactly on the span. It cannot make any span moment more accurate at the projection time. It can only stop the out-of-span part from feeding later steps. Measured on the doubling map (D1, $\psi = \cos2\pi x$ in the span): the GPT and Lagrangian values agree to all printed digits, for example $2.54\cdot10^4$ RMS at $N = 2\cdot10^4$, $n = 20$, where the exact answer is 0.

**D-T3 (no discrete gauge) [derived].** For distinct $z_i$, $\nabla\cdot(\mu\eta) = 0$ forces $w_i\eta_i = 0$, so $\eta = 0$. The continuum gauge (T3) explains what is safe to drop, but in the discrete system the dropped part does change out-of-span observables. GPT is a Galerkin truncation; its error is a coarse-graining error set by the basis.

**D-T4 (broken bases fail) [derived; measured, D1].** A per-cell polynomial basis contains $\mathbb 1_{\rm cell}\nabla(\text{linear})$, that is, any cell-constant vector. It keeps each cell's mean tangent, which carries coherent growth. On the doubling map the broken projection equals the Lagrangian value for every observable. A conforming basis (continuous across cells) is necessary for the continuum gauge to be visible. On a compact bunch a global constant is a gradient (of a linear function), so even a conforming basis keeps a uniform shift; only closed supports have constant gauges.

### 4.2 Basis for FBPIC

Use a conforming basis on the $(z, r)$ node mesh with the deposit shape functions $S_g$ (linear, `inversion/twin_config.py:234`). Per node $g$:

1. Field rows (12 real): $\mathrm{Re}/\mathrm{Im}\,[S_ge^{-im\varphi}]$ and $\mathrm{Re}/\mathrm{Im}\,[v_\ell S_ge^{-im\varphi}]$ for $\ell\in\{r, t, z\}$, $m\in\{0,1\}$, written in Cartesian $(x, y)$ with $\cos\varphi = x/r$, $\sin\varphi = y/r$. With these, D-T1 keeps $\delta\rho_g^m$ and $\delta\mathbf J_g^m$ exactly.
2. Enrichment (27 real): $S_g\cdot\{u_k, u_ku_l\}\cdot\{1, \cos\varphi, \sin\varphi\}$. These give the span the local linear and quadratic momentum structure that the spreads need.
3. Global loss rows: $\chi(z)\cdot\{1, z_k, z_k^2, \ldots\}$ for the 13 base sums. With them, the loss column is continuous across a projection (a free consistency check).

About 39 functions per node, 4 active nodes per particle (linear shape), so 156 active columns per particle [derived].

The Gram matrix $G$ depends only on positions, not on the parameter. For $K$ parameters, factor once and solve $K$ right sides [derived].

### 4.3 Which particles

1. Project electrons only: `elec_from_N` (the bunch, first) and, optionally, `electrons`.
2. Keep `nitrogen` and `helium` Lagrangian (T3 exception; ions move little).
3. Keep markers within one cell of the injection front, the window back, and the axis Lagrangian (R0 3.6).
4. Projecting only the bunch is consistent: D-T1 holds species by species if each species has its own coefficient vector.

### 4.4 Field-tangent consistency (no pulse at a projection)

FBPIC uses the curl-free current correction, which needs $\delta\rho_{\rm next} - \delta\rho_{\rm prev}$ (R0 3.5). A projection must keep the next $\delta\mathbf J^{n+1/2}$ and $\delta\rho^{n+1}$, not only $\delta\rho^n$. The removed part of $\xi$ is the large part, so any row that is not kept injects a large spurious field. Two exact placements exist [derived]:

1. **Inside the step (preferred).** Project $\xi^{n+1/2}$ after the first half position push and before the $J$ deposit (`stepper.py:545`). Rows: $\nabla_z(v_\ell S_ge^{-im\varphi})(z^{n+1/2})$ for $J$, and the $\rho^{n+1}$ rows pulled back through the field-free second half drift, $D^{\mathsf T}_{\rm drift}\nabla_x(S_ge^{-im\varphi})(\mathbf x^{n+1})$, with $D_{\rm drift} = \begin{pmatrix}I & \tfrac{\Delta t}{2}D\mathbf v\\ 0 & I\end{pmatrix}$. $\delta\rho_{\rm prev}$ is unchanged. This needs a hook inside `fbpic/autodiff/stepper.py` and invalidates the P7 cache key.
2. **At the step boundary (no autodiff edit).** Project $(\xi_x^n, \xi_u^{n-1/2})$ in the `driver.jvp` callback. The next $J$ deposit sees $\xi$ after the Vay tangent, which mixes $\xi_x$ through $\nabla\mathbf E$ and $\nabla\mathbf B$. The rows must be pulled back through the frozen-field half-step Jacobian $L_i$: $L_i^{\mathsf T}\nabla(\ldots)$. Forming $L_i$ costs 6 frozen-field tangent evaluations per projected particle.

With either placement, $\delta\rho$, $\delta\mathbf J$ and so all field tangents are continuous across the projection to the solver tolerance.

**Cold nodes [derived].** On a cold node, $\nabla(v_\ell S_g) = (v_\ell\nabla_xS_g,\ S_g\nabla_uv_\ell)$ differs from $v_{0\ell}\nabla S_g$ by the $u$-part. The charge and current rows stay independent, so $\delta\mathbf v\ne0$ is representable. This removes the R-v cold-node inconsistency (R0 A.4, Section 3.5 item 4).

### 4.5 Conditioning

1. On an exact sheet, some enrichment columns vanish (for example $\nabla[(u_k - U_k)^2] = 0$ on $\mathcal M$). $G$ is singular. Use a constrained solve: field rows and loss rows as exact constraints (KKT system or a two-stage projection), and a truncated eigen-solve or a small ridge for the enrichment block. A ridge on the field rows breaks D-T1 at $O(\epsilon)$ and produces a pulse of size $O(\epsilon\lVert(I-P)\xi\rVert)$.
2. Nodes with few markers. The background has 16 markers per cell (`p_nr` 2, `p_nt` 8, `p_nz` 1, `configs/twin_s.json`), so about 64 markers and 384 equations per node against 39 unknowns. Merge or drop enrichment where a node has fewer than about 3 times its unknowns.
3. Scale. Use $\ell_x = \Delta z$ and $\ell_u$ equal to the bunch momentum spread in each direction. Scale the basis functions to unit RMS gradient per node before the solve.

### 4.6 Algorithm (one projection)

1. Sort projected markers by cell (already done by `sort_all`).
2. Evaluate the 156 active gradient columns per marker; accumulate $G$ and $b$ per node block with a per-cell batched GEMM (no atomics).
3. Solve the block-sparse system ($3\times3$ node stencil) with the constraints of 4.5.
4. Overwrite $\xi_i$ with $(P\xi)_i$. Leave $\dot w_i$ unchanged.
5. Log $\lVert\xi\rVert$, $\lVert(I-P)\xi\rVert/\lVert\xi\rVert$, the field-row residual, and the loss-column jump (should be rounding).

## 5. Regrowth and the projection interval $T$

Between projections the exact tangent kernels run unchanged. Two regimes [derived]:

1. **Gauge-dominated regime** (smooth $f$, $\lVert P\xi\rVert$ bounded). After a projection the out-of-span part regrows by at most $e^{\lambda T}$. The next $b$ has pathwise noise $\approx e^{\lambda T}\sigma_0$, with $\sigma_0$ the noise of a non-growing tangent. Independent errors over $N_p = L/T$ projections add to $\varepsilon\approx\sqrt{L/T}\,e^{\lambda T}\sigma_0$ plus a truncation bias that grows with $N_p$. The minimum is at $T^\ast = 1/(2\lambda)$.
2. **Sheet regime** (cold $f$, C5). $\lVert P\xi\rVert$ itself grows as $e^{\lambda t}$ in the continuum. Only the truncation keeps the discrete value bounded, so the result is a coarse-grained derivative at the basis scale, as for R-v. $T$ then trades noise against bias, and both depend on the basis.

Twin-S numbers [estimated, rates from R0 `timeline_check.out`, $\omega_{p0}\Delta t = 0.0266$]:

| Window | $\lambda$ per step | $T^\ast = 1/(2\lambda)$ | $e^{\lambda T^\ast}$ | Lagrangian growth over the window |
|---|---|---|---|---|
| steps 1200 to 1500 | 0.0139 | 36 | 1.65 | $e^{4.2} = 65$ |
| steps 1500 to 1800 | 0.0212 | 24 | 1.65 | $e^{6.4} = 580$ |
| P7 average | 0.019 | 26 | 1.65 | $4\cdot10^6$ over steps 1200 to 2000 |

In regime 1, $T = 25$ from step 1100 to 2000 gives 36 projections and $\varepsilon\approx\sqrt{36}\cdot1.65\,\sigma_0\approx10\,\sigma_0$, against $4\cdot10^6\sigma_0$ for the raw tangent [estimated]. That gain exists only if Twin-S is in regime 1, which C4 and C5 argue against for the cold background and leave open for the 4-D EV manifold.

## 6. What GPT can and cannot fix

| Growth type | Continuum $\delta f$ | GPT result |
|---|---|---|
| Relabeling of an invariant measure (doubling map, free streaming of an $x$-uniform kick) | bounded, here 0 | Removed exactly in the continuum [derived]. Discretely only with a conforming basis and only for the out-of-span part; span moments keep the Lagrangian noise [measured, D1]. |
| N-body chaos on a smooth $f$ (paper's Debye case) | smooth, bounded | Removed up to the truncation error; same regime where R-v works [derived; R0 Section 5]. |
| Mean-field chaos, warm $f$ (filamenting smooth $f$) | bounded in $\dot H^{-1}$ under mixing | Partly removed; bias grows as $f$ filaments below the basis scale [derived]. |
| Mean-field chaos on a cold sheet (Twin-S background; probably the bunch) | $\dot H^{-1}(f)$ norm grows as the stretching (C5) | Not removed. Normal part untouched (C1); tangential minimal norm grows (C5). Only the basis truncation bounds it [derived; measured, D2]. |
| Collective physical instability (hosing, two-stream, wake amplitude response) | grows | Kept, correctly, because $\delta f$ and the field tangents are preserved (T2, T3) [derived]. |

GPT never removes a growth of $\delta f$. It removes only representation growth. On Twin-S the representation growth is mostly tangential stretching of sheets, and on a sheet the minimal representation grows too.

Note on the R0 open question (N-body or mean-field): for the cold background, the analysis above does not depend on it. A cold sheet has no smooth-$f$ regime.

## 7. Cost per projection on Twin-S [estimated]

Inputs: 800 by 50 cells (41k nodes), $N_m = 2$; 39 functions per node, 156 active per marker; FP64 peak about 1.4 TFLOP/s (RTX 6000 Ada, R0 4.5); bunch about $2\cdot10^6$ `elec_from_N` markers in about 300 cells, all `elec_from_N` about $10^7$; background about $6.4\cdot10^5$ markers.

| Item | Bunch only ($2\cdot10^6$) | All `elec_from_N` ($10^7$) |
|---|---|---|
| Gram assembly, $156^2\times6\times2\approx2.9\cdot10^5$ flop per marker | $5.8\cdot10^{11}$ flop, 0.4 s at peak, about 1 to 2 s achieved | $2.9\cdot10^{12}$ flop, 2 s at peak, about 5 to 10 s achieved |
| Unknowns, block-sparse solve | about $4\cdot10^4$; about 110 MB; sparse Cholesky about 0.1 s | up to $1.6\cdot10^6$; about 4.5 GB assembled; use matrix-free CG instead (about $3.7\cdot10^3$ flop per marker per iteration, 100 to 300 iterations, 5 to 30 s) |
| Boundary placement (4.4 item 2) | plus 6 frozen-field tangent evaluations per marker | same |
| Per run, $T = 25$, steps 1100 to 2000 (36 projections) | 36 to 72 s against a 298 s gradient: +12% to +25% | +60% to +120% |
| Extra parameters | one Gram factorization; $K$ right sides | same |

The projection does not change the per-step cost of the exact tangent ($c_1 = 2.47$ as built, R0 6.1). So GPT cannot beat one-sided FD on cost; its only possible gain is accuracy at matched scale.

## 8. Evidence

All scripts are in `inversion/notes/gpt-evidence-2026-11-01/work_derive/`, with outputs next to them.

**D1, `d1_doubling.py` (`d1_doubling.out`).** Doubling map $x\to2x+\theta$; exact $d\langle\psi\rangle/d\theta = 0$; $\xi = 2^n - 1$.

| Particles, $N$, $n$ | Lagrangian, $\psi$ in span | GPT conforming (Fourier, $K = 4$) | GPT broken (8 cells, degree 2) |
|---|---|---|---|
| random, $2\cdot10^4$, 10 | 25.3 | 25.3 | 25.3 |
| random, $2\cdot10^4$, 20 | $2.54\cdot10^4$ | $2.54\cdot10^4$ | $2.54\cdot10^4$ |
| lattice with jitter 0.3, $2\cdot10^4$, 15 | 803 | 803 | 803 |
| exact lattice, $2\cdot10^4$, 20 | $8\cdot10^{-8}$ | $8\cdot10^{-8}$ | $8\cdot10^{-8}$ |

(RMS over 5 seeds.) For $\psi = \cos(2\pi\cdot5x)$, outside the span, the conforming GPT value is 9 to 50 times smaller than the Lagrangian one only because the projected field has no content at that wavenumber; the broken value equals the Lagrangian one. The exact lattice is quiet for every estimator because the doubling map permutes it.

**D2, `d2_sheet_norms.py 2.0 14` (`d2_K2.out`).** Standard map, $K = 2$, closed cold sheet $p_0 = 0.3$, $2^{22}$ labels.

| Step | $\lVert\xi\rVert$ | $\lVert\xi_N\rVert$ | $\lVert\xi_T\rVert$ | $\lVert P\xi\rVert$ (continuum) | stretching $\lVert F\rVert$ |
|---|---|---|---|---|---|
| 1 | 1.00 | 0.61 | 0.79 | 1.00 | 14 |
| 6 | 21.2 | 1.11 | 21.2 | 21.1 | 625 |
| 10 | 447 | 1.41 | 447 | 445 | $1.4\cdot10^4$ |
| 14 | $1.02\cdot10^4$ | 1.57 | $1.02\cdot10^4$ | $1.02\cdot10^4$ | $3.2\cdot10^5$ |

The label lattice resolves the curve to step 10 ($\max\lvert F\rvert\Delta u\le0.76$); later rows are lattice values, not continuum values. At step 14 the lattice Lagrangian value ($-3.091$) agrees with a lattice FD at $h = 10^{-7}$ ($-3.059$). The growth of $\lVert\xi\rVert$ and $\lVert P\xi\rVert$ is the same in the resolved rows.

**D3, `d3_label_density.py` (`d3_label_density.out`).** Label-space density change on the same sheet, and the Monte Carlo spread at $N = 2\cdot10^4$ (200 draws).

| Step | $\lvert\delta n_a\rvert$ median | 99th percentile | Lagrangian mean $\pm$ SD | Label IBP mean $\pm$ SD | Reference |
|---|---|---|---|---|---|
| 2 | 0.86 | 7.0 | $0.128\pm0.006$ | $0.128\pm0.006$ | 0.128 |
| 6 | 3.2 | 645 | $0.441\pm0.095$ | $0.80\pm71$ | 0.444 |
| 9 | 6.0 | 2480 | $-1.16\pm0.93$ | $174\pm2600$ | $-1.20$ |

**D4, `d4_gpt_resets.py` (`d4_sheet.out`, `d4_blob.out`).** Discrete GPT with projections every $T$ steps, $N = 2\cdot10^4$, 5 seeds. Sheet: as D2, reference from the $2^{22}$ lattice. Blob: warm Gaussian of R0 E2 ($K = 1.5$), reference from $4\cdot10^6$ random particles.

| Case, step $n_f$ | Reference | Lagrangian | Conf. $K_f = 4$, $T = 1$ | Conf. $K_f = 8$, $T = 1$ | Conf. $K_f = 8$, $T = 3$ | Broken $16^2$, $T = 1$ |
|---|---|---|---|---|---|---|
| sheet, 6 | $+0.444$ | $+0.50\pm0.05$ | $+0.71\pm0.07$ | $+0.47\pm0.13$ | $+0.47\pm0.10$ | $+0.53\pm0.07$ |
| sheet, 9 | $-1.202$ | $-1.58\pm0.81$ | $-0.03\pm0.10$ | $-0.34\pm0.15$ | $-0.41\pm0.36$ | $-0.37\pm0.49$ |
| blob, 8 | $+0.776\pm0.004$ | $+0.75\pm0.11$ | $+0.75\pm0.06$ | $+0.75\pm0.06$ | $+0.81\pm0.03$ | $+0.77\pm0.03$ |
| blob, 12 | $+0.90\pm0.06$ | $+0.63\pm1.24$ | $+0.29\pm0.05$ | $+0.50\pm0.39$ | $+0.46\pm0.29$ | $+1.36\pm0.29$ |

Entries are mean $\pm$ SD over 5 seeds (the SE of a mean is SD$/\sqrt5$). The blob reference carries its SE. R0 E2b gave 0.734 for the blob at $n_f = 12$ with $4\cdot10^6$ particles, consistent within two SE.

Reading [measured]:

1. Before strong chaos (sheet step 6, blob step 8), the conforming projection with $K_f = 8$ is unbiased within noise. $K_f = 4$ on the sheet is biased by $+0.27$ (truncation). On the sheet at step 6 the $K_f = 8$ projection doubles the SD (0.13 against 0.05): projecting a noisy $b$ into smooth fields that are then stretched can amplify noise.
2. In the chaotic phase (sheet step 9, blob step 12), every projected variant is stable (SD 0.05 to 0.49 against 0.81 to 1.24) and wrong: errors of 0.40 to 1.17, that is, 66% to 97% of the reference on the sheet and 45% to 68% on the blob. The error changes with $K_f$ and $T$ and does not decrease in a systematic way. This is the R0 E2 pattern for R-v ("stable but wrong", R0 Section 5), now for GPT.
3. The fraction of $\lVert\xi\rVert^2$ removed per projection is 1% to 47%. Little of the tangent is gauge even on the warm blob.

## 9. Open items

1. **Decisive test, if anyone continues.** Run the conforming GPT (4.2, bunch only, in-step placement) on the 1D pilot `r0/work_tests/r2_pilot.py` and on Twin-S at $T\in\{12, 25, 50\}$, and plot the residual column against the CRN FD column at $h\in\{0.0025, 0.01, 0.02\}$. GPT is useful only if its column matches FD at some $h$ with a spread smaller than the FD local-secant spread (15% to 28% at $h = 0.01$, R0 Section 2). This is a matched-scale comparison, not a check of a scale-free derivative.
2. **Not derived here:** the normal-component bound for the non-Lagrangian EV manifold (C4); the size of $\delta n_a$ for the $k = 3$ and $k = 4$ sheets of Twin-S (D3 covers only $k = 1$).
3. **No mean field in D2 to D4.** The toys have prescribed forces. T3 covers the field coupling in the continuum; the discrete field consistency (4.4) is derived, not measured.
