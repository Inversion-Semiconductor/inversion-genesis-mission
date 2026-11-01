# Track R, Phase R0: design note and go/no-go recommendation

| Item | Value |
|---|---|
| Date | 2026-10-31 |
| Status | Phase R0 complete. Recommendation: no-go for Track R as the route to multi-parameter optimization. Waiting for the user decision (proposed D35, Section 10). |
| Scope | Differentiable FBPIC inversion, Twin-S, $E_L$ first; Track R of plan Section 7 (D33); user decision D34 (test Track R feasibility before the FD fallback). |
| Inputs | Five R0 derivation sections (`sec_vm`, `sec_gradf`, `sec_rv`, `sec_multi`, `sec_tests`), two critiques (`crit_physics`, `crit_feasibility`), the paper, the plan, and the worktree code. |

Evidence labels: **[paper]** Chung, Bond, Cyr, Freund, JCP 400, 108969 (2020), OSTI 1570259, checked on the page images; **[derived]** derived in R0 (derivations in Appendix A); **[measured]** measured on CPU with the named script; **[estimated]** an estimate with stated inputs; **[plan]** `DIFFERENTIABLE_INVERSION_PLAN.md`; **[code]** worktree code at `path:line`, relative to `lpa/FBPIC_Warp/`. A measurement by an R0 agent other than this note carries the agent key, for example [measured, `crit_physics`].

Base directory of all R0 files: `inversion/notes/r0-evidence-2026-10-31/`, written `r0/` below. Appendix C lists the full paths.

## 1. Decision for the user

**Decision.** Close Track R at Phase R0, and take the multi-parameter path on finite-difference-type Jacobians?

**Recommendation: yes.** The R0 evidence does not support Track R for Twin-S, on accuracy or on cost.

1. Accuracy. Twin-S at $N_t = 2000$ has no scale-free derivative to compute. Its residual map is rough down to $h = 10^{-7}$, and local secants at $h \le 0.01$ vary by 15% to 44% of the column norm after the curvature is removed [measured, Section 2]. Every Track R estimator is a momentum-smoothed (coarse-grained) derivative whose value depends on its smoothing scale. All R0 tests with trapping or phase mixing showed $O(1)$ errors that depend on that scale.
2. Cost. A forward Track R variant that is credible on the cold Twin-S plasma costs at best about one primal run per parameter, the same as one-sided FD [estimated, Section 6]. The adjoint saves cost only in the pure-weight form, which fails on cold particles.
3. FD already works. Central FD at $h = 0.02$ recovered $E_L$ and $z_f$ in Phase P3 (5 to 8 Jacobians per solve) [plan 12.5]. Central and forward FD at $h = 0.0025$ to $0.02$ pass the guarded usability gate of Section 7.3 on the P7 solution slice; the AD column fails it [measured].

**Consequences if the user accepts.**

1. No R1 to R4 work. This saves an estimated 30 to 50 days to a $K$-batched forward capability, which would at best tie FD.
2. Stage A Phases P8 and P9 use the D32 branch 3 (central FD at $h = 0.02$). The verified AD stays as a short-horizon tool ($N_t \le 500$) and as the reference for code checks.
3. For $K \ge 3$, a Stage A comparison of one-sided FD against a model-based derivative-free least-squares solver (DFO-LS type) sets the multi-parameter Jacobian method (1 to 2 days, a few GPU-h).
4. Stage B moves first to the global-search decision of plan Section 2.6. Exact gradients do not fix it [plan 12.5].
5. One diagnostic is still worth 2 to 3 GPU-h and no new code: the particle-count scaling of the FD roughness (Section 7.1, Q1). It sets the smallest useful FD step. It is also a necessary condition for any later reopening of Track R.

**If the user rejects the recommendation**, Sections 7 and 8 give the restructured test path. Its first steps are Twin-S-native and cheap. Its expected best outcome is parity with FD.

## 2. Feasibility verdict in brief

**What the evidence supports.**

1. The dipole identity is exact for the relativistic electromagnetic force: $-\delta\mathbf a\cdot\nabla_u f = -\nabla_u\cdot(f\,\delta\mathbf a)$, with $\delta\mathbf a = \frac{q}{mc}(\delta\mathbf E + \mathbf v\times\delta\mathbf B)$ [derived, A.1; confirmed by both critics]. The weight form of the sensitivity is therefore the score-function (likelihood-ratio) estimator, and the Lagrangian tangent is the pathwise estimator of the same quantity. The weight form removes the stretching factor $D\Phi$ of the flow [derived, A.2, A.5].
2. The paper's method works for warm, near-equilibrium plasmas. An R0 pilot reproduces the Debye case: particle-pdf gives $11.79 \pm 0.35$ (non-collocated) and $12.86 \pm 0.48$ (collocated) against a continuum slope of $12.4 \pm 0.2$ [measured, `sec_tests`, `r0/work_tests/debye_pdf.py`].
3. Variant R-v is well defined, and R-v with reset interval $T = 1$ step is the collocated particle-pdf method. The existing tangent state already carries the weight tangent [derived, code; Section 5].

**What the evidence does not support.**

1. A regular, scale-free continuum derivative for Twin-S. The P7 FD secants follow a Hölder-0.73 roughness law over $h = 10^{-3}$ to $10^{-7}$ [measured, `r0/work_note/holder_check.py`]. Between $h = 10^{-2}$ and $10^{-3}$ the column changes by 17%, and $\langle u_x\rangle$ by a factor 2.4. On the 41-point solution slice, local central secants differ from the local cubic-fit derivative by 31% to 44% of the column norm at $h = 0.0025$ and by 15% to 28% at $h = 0.01$ [measured, `r0/work_note/detrended_spread.py`].
2. Accuracy of the weight form on Twin-S. The background is a cold sheet (codimension 3 in $\mathbf u$), and the EV electrons lie on a 4-D manifold (codimension 2). By Liouville both stay sheets. Their score $\nabla_u\ln f$ does not exist, so any scheme must smooth $f$ in momentum at a width $\sigma_u$. That width is then a coarse-graining scale like the FD step $h$, not a small numerical parameter [derived, A.3]. Every R0 test with trapping or mixing gave $\sigma_u$-dependent $O(1)$ errors (Appendix B), and an integrable pendulum test failed at the paper's relative resolution after 3 to 6 rotations [measured, `crit_physics`].
3. The claim that the P7 growth is N-body chaos. The fastest growth ($0.80\,\omega_{p0}$, steps 1500 to 1800) occurs after the laser leaves the plasma (step 1520), while the bunch crosses the density downramp [derived, `r0/work_note/timeline_check.py`]. The mechanism is unidentified, because P7 logs only the maximum over all species (`fbpic/autodiff/driver.py:118-136`).
4. A cost advantage. Section 6 gives the numbers.

**Main risk.** A smoothed score estimator is bounded by construction: its estimate is at most $\mathrm{osc}(\psi)/\sigma_u$ per unit source time [derived, A.3]. A Track R column therefore always looks regular. Its value is still a coarse-grained derivative at an uncontrolled scale. Boundedness is not evidence of accuracy, and the paper's own validations compare with coarse-grained references (FV lower bound, p33; a $\Delta v_0 = 0.1$ moving-window average of a visibly rough brute-force curve, Fig. 19b, p49) [paper].

**What would change the verdict.** Two results together: (a) the FD roughness on Twin-S falls as $N^{-1/2}$ when the particle count grows (Q1), so a smooth continuum response exists; and (b) a parameter or point where every FD step fails the guarded gate, so that a different derivative method has a use.

## 3. Method for FBPIC

This section records the method as it would be built. It is the reference if the user continues (Section 7) or reopens Track R later.

### 3.1 Sensitivity equations

Notation: $\mathbf u = \mathbf p/(mc)$, $\gamma = \sqrt{1 + u^2}$, $\mathbf v = c\,\mathbf u/\gamma$, $\mathbf a_s = \frac{q_s}{m_sc}(\mathbf E + \mathbf v\times\mathbf B)$, $\delta(\cdot) = \partial(\cdot)/\partial\theta$. For species $s \in$ {`electrons`, `helium`, `nitrogen`, `elec_from_N`} [derived, `sec_vm` 2.2]:

$$
\partial_t\delta f_s + \mathbf v\cdot\nabla_x\delta f_s + \mathbf a_s\cdot\nabla_u\delta f_s = -\nabla_u\cdot\big(f_s\,\delta\mathbf a_s\big) + \delta G_s,
\qquad \delta\mathbf a_s = \frac{q_s}{m_sc}\big(\delta\mathbf E + \mathbf v\times\delta\mathbf B\big),
$$

$$
\partial_t\delta\mathbf B = -\nabla\times\delta\mathbf E,\quad
\partial_t\delta\mathbf E = c^2\nabla\times\delta\mathbf B - \delta\mathbf J/\varepsilon_0,\quad
\delta\rho = \sum_s q_s\!\int\!\delta f_s\,d^3u,\quad
\delta\mathbf J = \sum_s q_s\!\int\!\mathbf v\,\delta f_s\,d^3u .
$$

This is the relativistic electromagnetic form of paper Eq. (23a) and (25), p12. $\mathbf v\times\delta\mathbf B$ has no $\delta\mathbf v$ factor, because $\mathbf u$ is a phase-space coordinate. Sources:

1. $E_L$ and $z_f$ enter only through the initial fields: $\delta f_s(0) = 0$, $(\delta\mathbf E, \delta\mathbf B)(0) = F_0/(2E_L)$ for $E_L$ [plan 4.8]. There is no volume source $\partial_\theta\mathcal V_1$.
2. EV births: $G_s = \sum_{i,n}B_{i,n}\,\delta(t - t_n)\,\delta(\mathbf x - \mathbf x_i)\,\delta(\mathbf u - \mathbf u_i)$. Then $\delta G$ has a monopole part $\dot B_{i,n}$ (the existing birth weight tangent $\dot W'$) and a dipole part carried by the parent ion displacement $\xi_i$ [derived, `sec_vm` 2.2].
3. Future density parameters ($n_0$, $F_L$) change marker weights directly; they are weight sources by construction [plan 2.1].

### 3.2 What the dipole identity means here

The identity of Section 2 has five consequences [derived, A.1 to A.6]:

1. Over one step, the source $-\nabla_u\cdot(f\,\delta\mathbf a\,\Delta t)$ is the $\delta f$ of a one-step Lagrangian momentum kick from a zero tangent. The existing Vay tangent already computes it (`fbpic/autodiff/tangent.py:582-599`, terms `econst * dE`).
2. The Lagrangian form keeps the dipole on the marker as $\xi$ and propagates it with $\dot\xi = (\partial\Phi/\partial z)\,\xi + \delta\Phi$; the factor $\partial\Phi/\partial z$ stretches it. The weight form converts the dipole at once into a scalar, $\Delta\hat W_p = -W_p\,\Delta t\,\delta\mathbf a_p\cdot\nabla_u\ln f$, and the flow carries scalars without a Jacobian factor.
3. In weak form both are the same integral $\int f\,\delta\mathbf a\cdot\nabla_u\psi_t$, with $\psi_t = \phi\circ\Phi_{t\to T}$ the pulled-back observable. The pathwise form differentiates $\psi_t$, which filaments. The score form differentiates $f$, which needs $\nabla_u\ln f$.
4. For a cold sheet, $\nabla_u\ln f$ does not exist. Smoothing $f$ at width $\sigma_u$ replaces $\nabla_u\psi_t$ by a $\sigma_u$-difference of $\psi_t$, that is, a local FD in momentum at every source time.
5. Adjoint: the dual of the weight is the adjoint value $\lambda$, and the dual of $\xi$ is $\nabla\lambda$. A transposed particle-pdf adjoint carries values and does not stretch; reverse-mode AD of the PIC step carries $\nabla\lambda$ and stretches like the forward tangent (Section 6.2).

### 3.3 Discrete placement in the FBPIC step

The step order is in `fbpic/autodiff/stepper.py:310-436`: gather at $t^n$, Vay push $u^{n-1/2}\to u^{n+1/2}$ with its tangent (call at `:502`), half position push, EV and its tangent, J deposit at $t^{n+1/2}$ (`deposit_tangent` call at `:545`), second half push, `rho_next` deposit, curl-free current correction and PSATD, window shift and damping [code]. A weight update fits right after the Vay tangent, $\hat W^{n+1/2} = \hat W^{n-1/2} - \Delta t\,h^n$, as in paper Fig. 4, p21 (weights at half steps). The J deposit then uses centered weights. The half-step lag of `rho_next` is absorbed by the curl-free correction, which uses only $\delta\rho_{next} - \delta\rho_{prev}$ [derived, `sec_vm` 2.7, 3.3].

An R-v reset runs at a step boundary, after `step` returns. `driver.jvp` already calls `callback(state, tangents, decisions)` after every step (`fbpic/autodiff/driver.py:182`), and `value_and_grad` forwards it (`driver.py:249`) [code]. A prototype therefore needs no edit of `fbpic/autodiff/`, which keeps the P7 step 5 cache valid (its key is the hash of `fbpic/autodiff/*.py`, plan 12.10).

### 3.4 Term-by-term map to code

| Term | Discrete form | Code | Status |
|---|---|---|---|
| Characteristics (primal) | functional step | `fbpic/autodiff/stepper.py:310-436` | Reused |
| Laser seed | $F_0/(2E_L)$ after `interp2spect` | `energy_seed` `fbpic/autodiff/laser.py:92`, `seed_tangents` `driver.py:92` | Reused |
| Maxwell tangent: transforms, curl-free correction, PSATD, filter, shift, damping | linear field operators | `FieldTangents.apply` `fbpic/autodiff/field_tangents.py:109`; `stepper.py:449, 561` | Reused |
| Gather of $\delta\mathbf E$, $\delta\mathbf B$ at markers | shape sum | `gather_tangent` `fbpic/autodiff/tangent.py:494` | Reused; with $\xi = 0$ it returns the field tangent at the marker |
| Source kick $\delta\mathbf a\,\Delta t$ (dipole form) | Vay tangent | `tangent.py:582-599` in `vay_tangent` (`:698`); `vay_ion_tangent` (`:773`) | Reused |
| Lagrangian transport of $\xi$ between resets | gather, Vay, position tangents | `tangent.py:494, 698, 829` | Reused, exact JVP of the discrete step (P4 to P6 verified) |
| Weight tangent $\dot w$ storage and transport | per-marker scalar | `TANGENT_PARTICLE_NAMES` `fbpic/autodiff/state.py:105`; `sort_all` and `_realign_tangents` `stepper.py:137, 291` | Reused; injected markers get 0 |
| $\delta\rho$, $\delta\mathbf J$ from $\dot w$ and $\xi$ | deposit tangent | `deposit_tangent` `tangent.py:1076`, `dw` terms at `:994`, `:1014` | Reused for correctness. Modified for cost: a dense-cell split is needed (P7 profile: 81 + 40 ms against 24 + 14 ms primal, plan 12.10 step 3) |
| EV birth monopole $\dot W'$ and parent $\xi$ copy | birth tangent | `ionize_tangent` `fbpic/autodiff/ionization.py:690`; copy at `:762-764` | Reused |
| Observable contraction (13 base sums) | $\sum_i(\dot w_i\phi_a + w_i\xi_i\cdot\nabla\phi_a)$ | `ResidualValue.contract` `fbpic/autodiff/loss.py:264-282`, `t["w"]` at `:273` | Reused |
| Reset (conversion of $\xi$ into $\Delta w$) | Galerkin solve $Gc = b$ (A.4) | none | New: `project_tangents`, first as a `driver.jvp` callback in `inversion/` |
| Right side $b$ for $\rho$ and $\mathbf J$ rows | Lagrangian moment derivatives | `deposit_tangent` with $\dot w = 0$, raw output before the spectral transform | Modified access only (raw grid output) |
| Gram matrix $G$ | per-slab outer products | none | New: chunked per-slab FP64 GEMM with centered orthogonal polynomials; no atomics (Section 4.5) |
| Cold-node detector | least-squares residual of the $\rho$ and $\mathbf J$ rows per node | none | New (A.4) |
| Per-species and quantile tangent growth | norms per species, median, 90th percentile | `tangent_norms` `driver.py:118-136` gives only the maximum over species | New callback |
| Full phase-space mesh of paper Eq. (36), (41), (42) | 6-D deposits | none | Not recommended (Section 4) |

### 3.5 Charge and current consistency

1. Twin-S uses the curl-free correction (`inversion/twin_config.py:236`; kernel `fbpic/fields/cuda_methods.py:141`). It replaces the longitudinal part of $\hat{\mathbf J}$ by the value that $\delta\rho_{next} - \delta\rho_{prev}$ requires. Gauss's law then holds for the deposited $\delta\rho$ [code, derived, `sec_vm` 3.3].
2. The correction cannot tell physical charge from projection noise. Any non-neutral weight change becomes a real longitudinal $\delta\mathbf E$. The reset must keep $\delta\rho$ on every node: the Galerkin basis contains the $\rho$ deposit test functions $S_g(z,r)\,e^{-im\varphi}$, $m \in \{0, 1\}$ [derived, `sec_rv` 2.3].
3. The correction does not touch the transverse current. A reset that changes $\delta\mathbf J_\perp$ injects a spurious pulse into PSATD. The basis must also contain the current rows $v_k(\mathbf u)\,S_g\,e^{-im\varphi}$, $k = x, y, z$ (9 real rows per node) [derived, `crit_physics` F7; accepted, A.4].
4. With current rows, a cold node gives collinear rows ($\psi_J = v_0\psi_\rho$) but an inconsistent right side when $\delta\mathbf v \ne 0$. This is the cold-limit theorem (A.2) in matrix form. The per-node least-squares residual is the cold-sheet detector [derived].
5. The $k = 0$ component of the correction is set to zero (`fbpic/fields/spectral_grid.py:140-144`) [code]. With charge-neutral resets this does not matter.

### 3.6 Boundaries

1. Front injection: injected plasma is cold and $\theta$-independent, so $\delta f = 0$ on inflow; `_realign_tangents` sets $\dot w = 0$ and $\xi = 0$ (`stepper.py:291-307`) [code].
2. Back removal: $\delta f$ leaves with its markers. The selection weight is 0 at the window back [plan 2.3], so the removal decision does not enter $J$ [derived].
3. Axis: converted markers carry no position tangent, so the $1/r$ terms of the gather and deposit tangents do not occur for them. Bins near $r = 0$ hold few markers; merge them or keep those markers Lagrangian [derived, `sec_vm` 5].
4. Outer radius: reflective at 20 µm; a converted marker keeps its weight [code, `inversion/twin_config.py:233`].
5. EV newborns: keep them Lagrangian until the next reset, so that the parent-ion dipole is converted with the bunch, not with a separate birth score [derived, `sec_vm` 5 item 6].

## 4. $\nabla_u f$ regularization and the phase-space mesh

### 4.1 The distribution in Twin-S

1. Background (`electrons`, and the He and N L-shell electrons): loaded cold (`inversion/twin_config.py:272-275`, no thermal momentum) [code]. Before orbit crossing it is $n\,\delta^3(\mathbf u - \mathbf U(\mathbf x,t))$. In the laser, $\Delta z\,\partial_z u_x \approx a_0k_0\Delta z \approx 1.6$ per cell, so a mesh cannot resolve momentum structure finer than about 0.6 there [derived, `sec_gradf` G.2.4].
2. `elec_from_N`: born with the parent ion momentum (`ionization.py:326-366`) at discrete steps. In the continuum the labels are the ion position and the birth time, so $f_N$ lives on a 4-D manifold in 6-D phase space [derived, `sec_gradf` G.2.3]. By Liouville it stays a 4-D manifold; phase mixing fills a volume only after coarse-graining [derived, `crit_physics` F5].
3. The bunch completes 0.3 to 0.8 betatron periods in the growth window (steps 1200 to 1800), so it is not phase mixed even in the coarse sense [derived, `crit_physics` F2, F5].

### 4.2 Options

| Option | Bias | Variance | Cost on Twin-S | Verdict |
|---|---|---|---|---|
| Collocated phase-space mesh (paper Eq. (36), (41), (42), (44)) | Warm: $O(\Delta v^2/v_T^2)$. Cold: not a derivative. On a sheet only tangential differences are visible, and the lost source is $\sum_{\hat n_c = 0}\mathcal H_c$ times $\langle\psi\rangle$ [derived, A.3] | like PIC | 6-D dense: $1.4\cdot10^{12}$ cells, 11 TB [estimated] | Rejected. Cold beam: wrong sign at every $\Delta v$ at $T = 10$ [measured, `sec_gradf` G.4]; cold one-step $\delta J$ 0% to 0.6% of exact [measured, `sec_vm` 2.4] |
| Non-collocated or adaptive sparse band mesh (paper Section 8.4) | $O(\sigma_u^2/\ell_\psi^2)$, no small parameter after mixing [derived, A.3] | $\propto 1/\sigma_u$ | 27 to 85 run-eq per run; 4 to 9 restricted to the bunch after step 1000 [estimated, `sec_gradf` G.6.3] | Rejected on cost |
| Dipole pairs at $\mathbf u\pm\sigma_u\hat{\mathbf e}$ (mesh-free) | same as band mesh; a local FD of $\psi_t$ at step $\sigma_u$ | SE $\times\sigma_u$ constant [measured, `sec_gradf` G.5] | 4 to 6 run-eq at $K = 1$ [estimated] | Coarse-grained derivative; tracks the FD step with $\sigma_u \approx 2h$ to $5h$ [measured, `sec_gradf` G.4] |
| Gaussian moment score per bin | exact for $\delta\langle\mathbf u\rangle$ and charge; factors 0.4 to 4.5 wrong for skewed or bimodal bins [measured, `sec_vm` 2.5] | $\propto 1/\sigma$ | cheap | Rejected |
| First-moment 6-D projection per bin | drops intra-bin $\mathrm{Cov}(\xi, \mathbf z)$, which carries 6 of the 11 residual sensitivities (all spreads) [derived, A.4] | low | cheap | Rejected (`crit_physics` F6, accepted) |
| Galerkin moment conversion at resets (R-v) | exact on the span $V$; error $\langle(I-P)\delta f/f, (I-P)\psi\rangle_f$ (A.4) | equal to Lagrangian in E1 at $\sigma \ge 0.01$ [measured, `sec_rv` 6.1] | 0.13 s (degree 4) to 1.8 s (degree 6) per reset for 2 M markers [estimated] | Chosen if Track R continues; fails silently on exact sheets without current rows |

### 4.3 Chosen option (only if Track R continues)

1. No phase-space mesh. Use the Galerkin conversion of A.4 at resets, applied to the bunch-relevant `elec_from_N` markers only (variant R-v-s). The background and the ions stay Lagrangian.
2. Basis per converted species: the $\rho$ rows and the current rows of Section 3.5; per longitudinal slab of about 1 µm, centered orthogonal polynomials of degree $p \in \{2, 4\}$ in $(x, y, u_x, u_y, u_z)$ (21 or 126 functions); the 13 loss base sums as global rows. With the loss rows in $V$, the loss column is continuous across a reset to rounding.
3. The regularization scale is the triple (reset interval $T$, degree $p$, slab width). Report every Track R column as a function of this scale, as FD columns are reported against $h$.

### 4.4 Bias and variance

1. Warm $f$: the bias moves onto $f$ and is $O(\sigma_u^2/v_T^2)$, independent of the roughness of $\psi_t$ [derived, A.3, Eq. (A4)]. This is the regime of all paper tests ($\Delta v/v_T = 0.08$ to 0.19, p25 and p44) [paper].
2. Cold $f$: the bias is $O(\sigma_u^2/\ell_\psi(t)^2)$, with $\ell_\psi$ the momentum scale of the pulled-back observable. Chaos shrinks it as $e^{-\lambda(T - t)}$; with the P7 rates, sources at step 1200 see $\ell_\psi \lesssim 0.154\,e^{-11}$ [derived, `sec_gradf` G.3.3]. No affordable $\sigma_u$ is small.
3. Measured on a cold 1D beam before chaos ($T = 10$): 10% bias needs $\sigma_u \approx 10^{-3}$, where the standard error with $8\cdot10^4$ particles is still 17% [measured, `sec_gradf` G.4, `r0/work_gradf/small_sigma.py`].
4. Variance of score estimators grows as $1/\sigma_u$ per kicked direction [measured, `sec_gradf` G.5]. The best bias-variance balance gives an error of about $N^{-1/3}$, but only if $\ell_\psi$ exists [derived].
5. The mesh-transfer bias of Eq. (42) is bounded in expectation by $\lVert\psi\rVert_\infty\lVert\tilde{\mathcal H} - \mathcal H\rVert_1 = O(\Delta^2)$ for smooth $f$. The roughness of $\psi$ enters the variance, not the bias [derived, `crit_physics` F10, accepted; A.3]. The relative bias can still be large when $\delta A$ is a small result of cancellation.

### 4.5 Cost on Twin-S

1. Gram assembly: a converted marker touches about 277 rows (2 slabs $\times$ 126 monomials, 12 field rows, 13 loss rows; more with current rows). Atomic assembly into one $126^2$ block per slab is not viable. A chunked per-slab FP64 GEMM costs about $9.5\cdot10^4$ flop per marker at degree 4 and $1.3\cdot10^6$ at degree 6. At about 1.4 TFLOP/s FP64 on the RTX 6000 Ada, a reset costs 0.13 s to 1.8 s for 2 M markers, 5 times more for all 10 M `elec_from_N` [estimated, `crit_feasibility` 1].
2. At 20 to 40 resets, the overhead is 1% to 25% of the 298 s gradient for 2 M markers [estimated].
3. Between resets the converted markers get a new $\xi$ from the next Vay kick, so they need `deposit_tangent` again. The cheap production `deposit_fused` path (`fbpic/particles/deposition/cuda_methods_fused.py:471`) applies only to $T = 1$ [code, derived; `crit_feasibility` 1].

## 5. Variant R-v against particle-pdf

| Aspect | Particle-pdf (paper, per step) | R-v / R-v-s |
|---|---|---|
| Relation | collocated particle-pdf = R-v with $T = 1$ [derived, `sec_rv` 2.5, `sec_tests` 3] | general $T$; $T \to \infty$ is particle-exact |
| Steps 0 to about 1100 (laminar, cold) | must convert every step from step $n_c = 450$, where $\nabla_u f$ is singular | exact Lagrangian, verified to step 500 [plan 12.10] |
| Conversions | 2000 | 20 to 100 |
| Exactness between conversions | not applicable | exact JVP of the discrete map [plan, P4 to P6] |
| Mesh | 6-D (infeasible) or the Galerkin basis | Galerkin basis |
| Cold sheet | fails (A.2) | fails silently without current rows; detectable with them |
| Filamenting $f$ (E2 standard map) | wrong: 0.39, 0.21, $-0.17$ against 0.74; no convergence in mesh or $T$ [measured, `sec_rv` 6.3] | same failure |
| Integrable phase mixing (pendulum) | 15% to 56% error after 5.8 rotations at the paper's relative resolution [measured, `crit_physics` F5] | same mechanism (R4 grows as $f$ shears) |
| Self-check | none | $T$ sweep |
| Cost per parameter | pure weights: $c_1 \approx 0.4$ to 0.8 [estimated], inaccurate on cold particles | as built $c_1 = 2.47$; after a deposit-tangent rewrite about 1.0 to 1.3 [estimated] |

Recommendation: if Track R continues, build R-v-s (Galerkin, bunch only, with current rows) as a callback prototype. Do not build a per-step particle-pdf source; R-v with $T = 1$ is that method. Neither variant removes the sheet and filament failure: that failure is a property of $f$, not of the variant [derived, `sec_rv` 3.4, `crit_physics` F5].

## 6. Multi-parameter path

### 6.1 Forward, $K$-batched

Cost per Jacobian in primal-run equivalents, including the residual run: $C_F(K) = 1 + c_0 + c_1K$. One-sided FD costs $1 + K$, central FD $1 + 2K$. Break-even with one-sided FD needs $c_1 < 1 - c_0/K$ [derived].

| Variant | $c_1$ | Basis | Accurate on cold Twin-S? |
|---|---|---|---|
| Lagrangian AD as built | 2.47 | measured, P7 step 3 | not applicable (chaotic at $N_t = 2000$) |
| R-v / R-v-s as built ($T > 1$) | 2.47 plus reset cost | measured inputs | unknown; at best coarse-grained |
| R-v-s after a dense-cell deposit-tangent rewrite | 1.0 to 1.3 | estimated, `crit_feasibility` 2 | same |
| Lagrangian background plus per-step weight-form bunch ($T = 1$ on the bunch) | about 0.8 to 1.3 | estimated: background tangent about 0.2 EV-run-eq (P5: tangent step 1.96 times a 9.2 ms no-EV primal step against a 43 ms EV step, `inversion/results/p5_tangent_cost.json`), plus a weight deposit of the EV electrons, a gather, and the field tangent | no: collocated weights on a codimension-2 sheet (A.2, A.3) |
| Pure weights everywhere | 0.4 to 0.8, $c_0 = 2$ to 8 | estimated, `sec_multi` 2.3 | no: zero current response in cold cells [derived, measured `sec_vm` 2.4] |

Correction to `crit_feasibility`: the bound "$c_1 \ge 0.96$ for every variant that keeps Lagrangian tangents" uses the no-EV primal step. A Lagrangian background alone costs about 0.2 EV-run-equivalents [derived above]. The conclusion still holds: the only variants with $c_1$ clearly below 1 are the ones that are wrong on cold particles. The best credible forward variant ties one-sided FD; at $K = 25$ the pure-weight best case would save at most about 20% (20.5 against 26 runs) [estimated, `r0/work_crit_feasibility/cost_compare.out`].

GPU detail: the deposit tangent keeps a per-thread matrix of $48K$ doubles for J with $N_m = 2$ (`tangent.py:888`), so $K$-batching spills registers beyond small $K$. Tile over $K$ in groups of 2 to 4 [code, estimated].

### 6.2 Adjoint and the Lyapunov question

1. Continuous adjoint [derived, A.6]: a scalar $\lambda_s(\mathbf x, \mathbf u, t)$ transported backward along the same characteristics with source $q_s\mathbf v\cdot\boldsymbol\alpha$, plus adjoint fields $(\boldsymbol\alpha, \boldsymbol\beta)$ that obey Maxwell's equations backward with sources $\int\lambda_s\nabla_u f_s\,d^3u$ and $\int\lambda_s(\nabla_u f_s\times\mathbf v)\,d^3u$. Both critics re-derived the signs and factors.
2. The coupling has two continuum forms: $\int\lambda\,\nabla_u f$ and $-\int f\,\nabla_u\lambda$. The particle-pdf adjoint must use the first. It is the exact transpose of the forward particle-pdf recurrence, so its propagator has the same singular values: it is free of Lyapunov growth exactly when the forward scheme is [derived]. A 1D-1V test confirmed the transpose to $9.7\cdot10^{-16}$ and bounded adjoint values (RMS 1.39 to 1.59) while $\xi_v$ grew to $2.9\cdot10^{11}$ [measured, `sec_multi` 5.2].
3. Reverse-mode AD of the PIC step (plan Phase P11) carries $\nabla\lambda = (\partial\Phi/\partial z)^T\nabla\phi$ on the markers and grows like the forward tangent. P11 is not a remedy for the P7 finding [derived].
4. The adjoint inherits the forward accuracy exactly. In the same 1D test, after trapping, the forward and adjoint estimate was biased by about +24 on a reference of 5 to 41, mesh-dependent [measured, `sec_multi` 5.3]. That reference is itself a quadratic fit over $\pm 37\%$ of $\theta$, and its CRN FD moves from 14.0 to 0.14 between $h = 0.02$ and 0.005 at $t = 85$ [measured, `crit_feasibility` 3]. So the adjoint is regular, and its accuracy is undefined in the same way as the forward one.

### 6.3 Memory and checkpointing

1. Forward: one FP64 weight per marker per parameter (0.08 GB at 10 M markers) plus 0.013 GB of field tangents; $K = 25$ fits easily on Twin-S. At production size the field tangents (about 3.8 GB per parameter) limit one 48 GB GPU to about 7 parameters per pass [estimated, `sec_multi` 2.3].
2. Adjoint: Twin-S checkpoints of 0.5 to 0.8 GB; 45 checkpoints on the GPU give 2.44 forward sweeps in total, 100 in host RAM give 1.95 [estimated, binomial schedule, `sec_multi` 6.2].
3. Replay determinism: FP64 atomics make repeated runs differ by $7.5\cdot10^{-13}$ after 10 steps [plan 12.10 step 2]. With growth of 10 per 120 steps, a replayed segment of $L$ steps deviates by about $10^{-13}\cdot10^{L/120}$; segments must stay below about 600 steps, or the deposits must be made deterministic (sorted per-cell accumulation) [derived]. Production has no deterministic deposit mode [plan 6.3 rule 7].

### 6.4 Crossover

The adjoint needs 11 adjoint sets (rank of the $11\times13$ map from the base sums) plus one replay. With $c_1^{adj} = 1.0$ to $1.3\,c_1$ and replay $R/c_1 = 1.5$ to 3, the crossover is $K^\ast \approx 13$ to 18 [estimated, `sec_multi` 4.2]. This holds only for the pure-weight form. A hybrid with a Lagrangian background also needs reverse mode for the background, that is, the P11 engine (not built), checkpointing, and deterministic deposits. The adjoint effort is 15 to 25 days on top of a working forward scheme [estimated].

### 6.5 Alternatives, ranked for the multi-parameter goal

| Rank | Method | Cost per Jacobian | Evidence | Main risk |
|---|---|---|---|---|
| 1 | Central FD at $h = 0.02$; one-sided FD at $h = 0.01$ to 0.02 | $1 + 2K$; $1 + K$; perfectly parallel over 2 GPUs | Stage A recovery in 5 to 8 Jacobians [plan 12.5]; gate pass on the solution slice [measured, Section 7.3] | curvature at $h \ge 0.05$ (cos 0.33); parameter-specific $h$ |
| 2 | Model-based derivative-free least squares (DFO-LS or POUNDERS type) | $K + 1$ initial runs, then about 1 to 2 runs per iteration (from the literature, not checked here) | none on this problem | iteration count at $K = 25$ unknown |
| 3 | Surrogate (GP or quadratic) on the global design | a few hundred runs for $K = 25$ [estimated] | addresses the Stage B global search | model quality in 25-D |
| 4 | Track R, R-v-s | about one primal run per parameter at best | no positive trapping evidence | parity with FD at best, 30 to 50 days |
| 5 | Short-horizon AD | $1 + c_1K$ | exact to step 500 | observable is at step 2000; not usable |
| 6 | Reverse-mode AD (P11) | 11 sweeps | none | inherits the chaos (6.2 item 3) |

Recommendation: rank 1 now; rank 2 compared with rank 1 on Stage A before $K \ge 3$; rank 3 as part of the Stage B global-search decision.

## 7. Test design if Track R continues, and the guarded usability metric

The R0 critics showed that R1 (Debye) tests the implementation, not the hypothesis, and that R2a (1D ponderomotive, uniform plateau) can pass while Twin-S fails and fail while Twin-S works [`crit_feasibility` 4, `crit_physics` F9; accepted]. The order below therefore puts Twin-S-native tests first.

### 7.1 Phase R0b: Twin-S-native diagnostics (no `fbpic/autodiff` edits)

| ID | Test | Numeric criterion | Cost |
|---|---|---|---|
| Q1 | FD roughness against particle count. At $\hat\theta = (0.588, 0.4)$, free EV, $N_t = 2000$: central FD at $h \in \{10^{-2}, 3\cdot10^{-3}, 10^{-3}, 3\cdot10^{-4}, 10^{-4}\}$ for particle multiplier $m = 1$ (`p_nr` 2, `p_nt` 8) and $m = 4$ (`p_nr` 4, `p_nt` 16). $A_m(h) = \mathrm{median}_j\,h\,\lvert D_{m,j}(h) - D_{m,j}(10^{-2})\rvert$ | Continue only if $A_4/A_1 \le 0.6$ at every $h \le 10^{-3}$ ($N^{-1/2}$ predicts 0.5). Kill if $A_4/A_1 \ge 0.85$ at every $h \le 10^{-3}$ (continuum roughness). Between: add 4 injection seeds per $m$ at $h = 10^{-2}$ and test the seed spread of the secant | 22 runs, about 2 to 3 GPU-h [estimated] |
| Q2 | Per-species growth: one $K = 1$ run with a callback that logs, every 50 steps, the median and the 90th-percentile $\lvert\xi_u\rvert$ per species, and the step at which each top-1% marker crosses $u_z = 1$ | Kill if the 90th-percentile norm of `electrons` grows by $\ge 10\times$ per 200 steps after step 1200 (the hybrid cannot keep the background Lagrangian) | 5 min GPU, 0.5 day code |
| Q3 | R-v-s callback prototype (CuPy, `inversion/`), Galerkin basis of Section 4.3, resets from step 1100 every $T \in \{10, 25, 50, 100\}$, degree $p \in \{2, 4\}$ | (a) a single reset at step 500 changes the $N_t = 500$ column by $\le 5\%$; (b) the per-node cold residual covers $\le 10\%$ of the converted weight; (c) columns at $T = 25$ and 50, and at $p = 2$ and 4, agree within 2 seed SE + 10% (plateau); (d) the 7.3 gate passes per seed on the solution slice against a matched-scale reference; (e) after one reset at step 1500, the field-tangent norm grows by $< 10\times$ per 300 steps (no collective mode, `crit_physics` F8) | 4 to 7 days, about 4 GPU-h |

### 7.2 R1 and R2 (only after R0b passes)

| ID | Test | Numeric criterion |
|---|---|---|
| R1-1 to R1-9 | Debye reproduction as in `sec_tests` 4.3 | $\lvert\bar{\mathcal J} - 619.875\rvert \le 0.5$, $\sigma_{\mathcal J} \in [0.17, 0.28]$; particle-pdf within 2 SE + 5% of $12.4 \pm 0.2$; $\log\sigma$ against $\log N$ slope in $[-0.6, -0.4]$ over $N = 10^4$ to $10^6$; $\sigma \le 5\%$ of the slope at $N = 10^5$; projection of seed Eq. (66) reproduces Eq. (65) within 5% $L^2$ |
| R1-T | Warm driven-trapping test (`sec_multi` 5) with matched-scale references: local quadratic fits over windows $\pm\ell$, $\ell \in \{0.005, 0.01, 0.02\}$, 16 CRN seeds | The reference is stable (changes $< 10\%$) over at least a factor 2 in $\ell$; else report "no target" and stop. The particle-pdf and R-v columns change by $< 2$ SE + 10% over a factor 3 in $\Delta v$ and lie within 2 SE + 20% of the stable reference |
| R2a' | `sec_tests` R2a model plus a downramp to $n_0/16$, a density cliff, and driver exit (amplitude to 0 over $2\,k_p^{-1}$) | (a) pathology: median-tangent growth exponential over $\ge 2$ decades, $\lambda \ge 0.05\,\omega_p$, onset time logged against the exit; (b') curvature-aware regularity: fit $D_h = D_0 + ch^2$ over $h \in \{0.005, 0.01, 0.02, 0.04\}$; $D_0$ changes by $< 2$ SE + 10% when the largest $h$ is dropped, for $\ge 5$ of 6 observables; (b'') roughness amplitude falls by $\le 0.6$ from 20 to 80 particles per cell; (e) a $\sigma_u$ or $T$ plateau over a factor 3 with bias $\le 10\%$ and noise $\le 15\%$ of $\lvert D_0\rvert$ |
| R2b | 1D electromagnetic, resolved laser ($\omega_0/\omega_p = 15$), 1D-2V; mandatory | same as R2a'; R-v-s must pass where pure particle-pdf fails ($u_y = a_y$ sheet) |
| R2-cost | marginal cost of a second parameter | operation count $c_1 \le 1.3$ |

GO for R3 needs all of: Q1 continue, Q2 no kill, Q3 (a) to (e), R1-T, R2a' and R2b pass, and a stated use: a parameter or slice point where every FD step fails the 7.3 gate. Without that use, Track R has no advantage over FD even when it works.

### 7.3 Guarded usability metric (replaces the D32 $\delta$ bar)

Why $\delta$ alone fails. At a least-squares solution, $\delta = \lvert(J_A - J)^Tr^\ast\rvert/\lvert J_A^TJ\rvert$ is invariant to the scale of $J_A$ and is small when $\lVert r^\ast\rVert/\lVert J\rVert$ is small. On P7, $\lVert r_w\rVert/\lVert J_{sm}\rVert = 0.013$, so the AD column met $\delta = 3.5\cdot10^{-3}$ with cosine 0.25. 19.5% of random 11-vectors pass $\delta \le 0.005$ [derived, measured, `sec_tests` 6.1, 6.2]. $\delta$ says where the iteration stops, not whether it gets there.

Definition. For a candidate Jacobian $J_A$ ($11\times K$) and a reference $J_{ref}$ at a stated scale $\ell$ (default: the derivative of a per-residual cubic fit over a 41-point $\pm 0.05$ slice, evaluated at each slice point):

$$
\kappa = \lVert I - J_A^+J_{ref}\rVert_2 \quad (K = 1:\ \kappa = \lvert 1 - \cos/\rho\rvert,\ \rho = \lVert J_A\rVert/\lVert J_{ref}\rVert), \qquad
\delta = \lVert (J_A^TJ_{ref})^{-1}(J_A - J_{ref})^Tr_w^\ast\rVert_\infty .
$$

$\kappa$ is the linear contraction factor of Gauss-Newton with $J_A$ (A.7).

Gate. $J_A$ is usable on a slice when all of these hold:

1. median $\kappa \le 0.5$, 90th-percentile $\kappa \le 0.8$, and $\kappa < 1$ at every point;
2. median per-column cosine to $J_{ref}$ $\ge 0.85$; the minimum is reported;
3. $\delta \le 0.005$ at the solution (D32, unchanged);
4. checked on the solution slice and on the false-minimum and far-field slices (plan P7 step 5);
5. for a stochastic column, checked on the ensemble mean and on each single seed at the production $N$.

Calibration on the P7 solution slice (17 centers within $\pm 0.02$ of 0.588, local cubic-fit reference) [measured, `r0/work_note/gate_calibration.py`]:

| Column | $\kappa$ median / p90 / max | cos median / min | Gate |
|---|---|---|---|
| Central FD, $h = 0.0025$ | 0.21 / 0.42 / 0.53 | 0.92 / 0.79 | pass |
| Central FD, $h = 0.01$ | 0.12 / 0.25 / 0.26 | 0.96 / 0.86 | pass |
| Central FD, $h = 0.02$ | 0.11 / 0.17 / 0.18 | 0.99 / 0.94 | pass |
| Forward FD, $h = 0.005$ | 0.19 / 0.45 / 0.51 | 0.90 / 0.79 | pass |
| Forward FD, $h = 0.02$ | 0.18 / 0.60 / 0.70 | 0.85 / 0.62 | pass (marginal cosine) |
| AD ($N_t = 2000$) | 1.00 / 1.00 / 1.00 | 0.21 / $-0.68$ | fail |
| Random 11-vectors, $2\cdot10^4$ | | | 0 pass |

Limits. The gate shows parity with FD, not superiority, because its reference is a large-scale secant and FD passes it by construction [`crit_feasibility` 5, accepted]. The false-minimum and far-field slices are incomplete (76 of 123 P7 step 5 runs missing, about 6.3 GPU-h) [plan 12.10]. The cosine bar is lower than the 0.9 of `sec_tests` 6.3, because forward FD at $h = 0.01$ to 0.02 has median cosines 0.85 to 0.87 on this slice; for $K > 1$ the matrix $\kappa$ covers the leakage between columns.

## 8. Risks, kill criteria, and effort

### 8.1 Risks if Track R continues

| Risk | Effect | Mitigation |
|---|---|---|
| Continuum roughness (Q1 kill) | no target derivative exists | stop at Q1 |
| Background carries the chaos (Q2 kill) | R-v-s cannot keep it Lagrangian; weights cannot represent it | stop at Q2 |
| Sheet and fold bias | coarse-grained, scale-dependent column that looks regular | plateau tests in $T$, $p$, $\sigma_u$; matched-scale references |
| Collective growth (F8) | weights and field tangents grow anyway | Q3 (e) |
| Cost | parity with FD at best | measure $c_1$ in Q3; rewrite the deposit tangent only after Q3 passes |
| Cache invalidation | an edit of `fbpic/autodiff/*.py` invalidates the P7 step 5 cache | callbacks in `inversion/` until P7 step 5 ends |

### 8.2 Kill criteria (any one stops Track R)

1. Q1: $A_4/A_1 \ge 0.85$ at every $h \le 10^{-3}$.
2. Q2: the `electrons` 90th-percentile tangent norm grows by $\ge 10\times$ per 200 steps after step 1200.
3. Q3: no plateau, that is, the columns at $T = 25$ and 50, or at $p = 2$ and 4, differ by more than 2 seed SE + 10%.
4. Q3: the cold-node residual covers more than 10% of the converted weight.
5. R1-T or R2a': the reference has no stable scale range, or the method misses it by more than 2 SE + 20%.
6. Cost: measured $c_1 > 1.3$ after the deposit-tangent rewrite.
7. Use: no parameter or point where every FD step fails the 7.3 gate.

### 8.3 Effort

| Phase | Recommended path (Track R closed) | Track R continued |
|---|---|---|
| Q1 roughness diagnostic | 0.5 to 1 day, 2 to 3 GPU-h | same |
| One-sided FD and DFO-LS comparison on Stage A | 1 to 2 days, a few GPU-h | in parallel, same |
| Q2, Q3 (R0b) | | 5 to 8 days, about 4 GPU-h |
| P7 step 5 completion | 0.5 day, 6.3 GPU-h (gate references) | same, before any `fbpic/autodiff` edit |
| R1 with R1-T | | 1 to 2 days |
| R2a' and R2b | | 7 to 9 days |
| R3 port, deposit-tangent rewrite, re-verification | | 11 to 20 days |
| $K$ tiling | | 2 to 4 days |
| R4 | | 3 to 5 days |
| Total | about 2 to 3.5 days plus about 10 GPU-h | about 30 to 50 days to a $K$-batched forward capability (uncertainty about $\pm 50\%$); adjoint plus 15 to 25 days and the P11 engine |

All effort figures are [estimated] from the section and critic numbers.

## 9. Critic findings and resolutions

| ID | Finding (severity) | Resolution | Check done for this note |
|---|---|---|---|
| P-F1 | Paper evidence does not transfer; boundedness is not accuracy (blocking) | Accepted. Section 2 states that Track R targets a $\sigma_u$-coarse-grained derivative and makes a plateau the first test | Read p10, p29 (Eq. (80), (81)), p33 (Eq. (92), FV lower bound), p34 (Fig. 11b), p49 (criterion unknown; Fig. 19b compares with a $\Delta v_0 = 0.1$ moving average of a rough brute-force curve) |
| P-F2 | Growth window is a mean-field transient, not established N-body chaos (blocking) | Accepted in part. The mechanism is labeled unidentified. Rejected part: the betatron comparison assumes the bunch carries the maximum; the maximum is over all species, and background electrons in the downramp oscillate at $\omega_{p,loc} = 0.67$ to $1.0\,\omega_{p0}$, close to the measured rates. Q2 decides | `r0/work_note/timeline_check.py`: laser exits at step 1520, bunch at about 1760; rates 0.52, 0.80, 0.13 $\omega_{p0}$ from plan 12.10 |
| P-F3 | FD response is Hölder-rough (major) | Accepted, with a caveat: below $h = 10^{-4}$ the FD runs also change removal counts (plan 12.10 step 4), so the small-$h$ exponent mixes mechanisms. Q1 is the decisive test | `r0/work_note/holder_check.py`: $A \propto h^{0.73}$; 17.4% column change; $\langle u_x\rangle$ ratio 2.43 |
| P-F4 | Rate-based chaos criterion misclassifies the paper's case (major) | Accepted. Replaced by the amplitude test Q1 | Fig. 11b, p34: $W_1$ saturation $\propto N^{-1/2}$ |
| P-F5 | Smoothness of $f$ is the real condition; warmth test cannot detect folds (major) | Accepted. The N-body/mean-field dichotomy and the covariance warmth test are dropped; plateau and cold-residual tests replace them | Liouville argument re-derived; pendulum numbers taken from `crit_physics` (not re-run) |
| P-F6 / C-6 | First-moment projection drops all 6 spread sensitivities (major) | Accepted. Galerkin basis with degree $\ge 2$ and the 13 loss rows (Section 4.3) | Re-derived: $\delta\sigma^2 = 2\,\mathrm{Cov}(z, \xi)$; the projection changes a bin's second moment only through third moments |
| P-F7 | Basis lacks current rows (major) | Accepted. Current rows added; per-node residual is the cold detector (Section 3.5) | Curl-free kernel corrects only the longitudinal part (`fbpic/fields/cuda_methods.py:141`) |
| P-F9 | R2a omits the downramp, cliff, and driver exit (major) | Accepted. R2a' (Section 7.2) | Timeline check above |
| P-F8 | Collective growth survives the weight form (minor) | Accepted. Q3 (e) | |
| P-F10 | Mesh-transfer bias is not a roughness bias in expectation (minor) | Accepted (Section 4.4 item 5) | Re-derived in A.3 |
| P-F11 | `sec_multi` reference is heavily smoothed (minor) | Accepted (Section 6.2 item 4) | |
| C-1 | Cost claims (2.5 run-eq; $c_1$ 0.4 to 1.8; resets 1% to 10%) (blocking) | Accepted, with one correction: a Lagrangian background alone costs about 0.2 EV-run-eq, not 0.96; the credible variants still tie FD at best (Section 6.1). The 2.5 run-eq figure is removed | `tangent.py:582-599` adds the kick to every marker every step; `p5_tangent_cost.json` (9.2 against 18.0 ms per step without EV) |
| C-2 | Twin-S response is scale-dependent; R2 rule (b) confuses curvature with roughness (blocking) | Accepted, with revised numbers: after removing the curvature of a cubic or quintic fit, the roughness is 31% to 44% at $h = 0.0025$ and 15% to 28% at $h = 0.01$ (critic: 41% to 54%, which includes 21% to 32% curvature). Rule (b) replaced by the curvature-aware rule (b') | Re-ran `slice_secant_spread.py`, `r2b_on_twins.py`, `onesided_gate.py` (identical output); new `detrended_spread.py` |
| C-3 | Driven-trapping reference is scale-dependent; kill criterion 0 ill-posed (major) | Accepted. R1-T uses matched-scale references | |
| C-4 | R1 and R2 cannot decide for Twin-S (major) | Accepted. Twin-S-native R0b first; R1 shortened with R1-T; R2b mandatory | |
| C-5 | Gate shows parity only; per-point cosine can fail a perfect method (major) | Accepted. Median-based gate with a local reference (Section 7.3). Refinement: against a local reference the per-point cosines of FD are 0.62 to 0.94 at minimum, not 0.25 to 0.48 (the critic compared every center with the fixed 0.588 column); forward FD sits at the cosine bar | `gate_calibration.py` |
| C-7 | Gram assembly cost (major) | Accepted (Section 4.5) | Monomial counts $\binom{9}{4} = 126$, $\binom{11}{6} = 462$ checked |
| C-8 | Code claims true; callbacks missed; max-only norms; register spill (minor) | Accepted | `driver.py:150-182, 249`; `driver.py:118-136`; `tangent.py:888` read |
| C-9 | Total effort 30 to 45 days (major) | Accepted; widened to 30 to 50 days with the R2a' changes | |
| C-10 | Simpler alternatives already work (major) | Accepted (Section 6.5). DFO-LS iteration counts are not known for this problem | |

## 10. Proposed plan changes

### 10.1 New decision D35 (if the user accepts)

| D35 | Close Track R at Phase R0 (design note `inversion/notes/r0-design-note-2026-10-31.md`). The multi-parameter path uses FD-type Jacobians: central FD at $h = 0.02$ in Stage A (D32 branch 3), then a comparison of one-sided FD and a model-based derivative-free least-squares solver before $K \ge 3$. Run the particle-count roughness diagnostic Q1. Track R reopens only if Q1 shows roughness that falls as $N^{-1/2}$ and a parameter exists for which every FD step fails the usability gate. Supersedes D33. | User, date |

### 10.2 Replacement text for the Track R block (plan Section 7, after "Variant R-v")

> Phase R0 result (2026-10-31). Design note: `inversion/notes/r0-design-note-2026-10-31.md`. Verdict: no-go for Track R as the route to multi-parameter optimization (D35).
>
> 1. The source of the linearized Vlasov equation is a phase-space dipole, $-\nabla_u\cdot(f\,\delta\mathbf a)$, also for the relativistic magnetic term. The weight form is a score-function estimator. It needs $\nabla_u\ln f$, which does not exist for the cold Twin-S background or for the EV manifold. A momentum smoothing scale is therefore required, and the result is a coarse-grained derivative at that scale.
> 2. Twin-S at $N_t = 2000$ has no scale-free derivative: the FD roughness follows $h^{0.73}$ over $h = 10^{-3}$ to $10^{-7}$, and local secants at $h \le 0.01$ differ from the local smooth derivative by 15% to 44% of the column norm.
> 3. All R0 tests with trapping or phase mixing gave scale-dependent $O(1)$ errors. Only the warm Debye case reproduced the paper.
> 4. A forward variant that is credible on cold particles costs about one primal run per parameter, the same as one-sided FD. The adjoint saves cost only in the pure-weight form, which fails on cold particles. Reverse-mode AD of the PIC step (P11) inherits the chaos.
> 5. FD at $h = 0.0025$ to 0.02 passes the guarded usability gate (Section 6.4) on the P7 solution slice. The AD column fails it.
>
> Reopening condition: the roughness diagnostic Q1 of the note shows roughness that falls as $N^{-1/2}$, and a parameter exists for which every FD step fails the gate. Phases R1 to R4 below are not run.

Also change the Reason paragraph: replace "This is the Lyapunov instability of the N-body particle dynamics, not a code error." with "This is a dynamical instability of the particle trajectories, not a code error. Its mechanism is not identified: the fastest growth ($0.80\,\omega_{p0}$, steps 1500 to 1800) occurs after the laser leaves the plasma (step 1520), while the bunch crosses the density downramp, and P7 logs only the maximum over all species." Make the same change in Section 12.10, conclusion item 2.

Also change Open problem 4: replace "Evidence for it here: FD secants at $h = 10^{-2}$ and $10^{-3}$ agree at $N_t = 2000$." with "The evidence here is weak: FD secants at $h = 10^{-2}$ and $10^{-3}$ differ by 17% of the column norm, and the roughness follows $h^{0.73}$ (R0 note, Section 2)."

### 10.3 Replacement text for the D32 recovery gate (plan Section 6.4)

Replace the paragraph "Recovery gate (D32) ... measured by the fixed-point shift $\delta$ at the T0 solution." and branch 1 with:

> Recovery gate (D32, guarded on 2026-10-31). The question is whether a Jacobian $J_A$ is usable by dogbox. The reference $J_{ref}$ is the derivative of a per-residual cubic fit over a 41-point $\pm 0.05$ slice, at each slice point. Define $\kappa = \lVert I - J_A^+J_{ref}\rVert_2$ (for $K = 1$, $\lvert 1 - \cos/\rho\rvert$ with $\rho = \lVert J_A\rVert/\lVert J_{ref}\rVert$), the Gauss-Newton contraction factor, and the fixed-point shift $\delta = \lVert(J_A^TJ_{ref})^{-1}(J_A - J_{ref})^Tr_w^\ast\rVert_\infty$ at the solution. $J_A$ is usable when, on the solution, false-minimum, and far-field slices: the median $\kappa \le 0.5$, the 90th-percentile $\kappa \le 0.8$, and $\kappa < 1$ at every point; the median cosine of each column to $J_{ref}$ is at least 0.85; and $\delta \le 0.005$ at the solution. A stochastic column must pass for its ensemble mean and for each single seed. $\delta$ alone is not a gate: it is invariant to the scale of $J_A$, and 19.5% of random columns pass $\delta \le 0.005$ at the T0 solution. The gate shows parity with FD, not superiority.
>
> 1. AD usable: $J_{AD}$ passes the gate. Phase P9 uses the AD Jacobian. (On Twin-S at $N_t = 2000$, $J_{AD}$ fails: median $\kappa = 1.00$, median cosine 0.21.)

Keep branches 2 to 4 unchanged.

### 10.4 Other text changes

1. Phase P11: add "Reverse-mode AD of the PIC step carries $\nabla\lambda$ on the markers and grows like the forward tangent after trapping (R0 note, Section 6.2). P11 does not remedy the chaotic long-horizon tangents."
2. Section 9, risk row "Chaotic particle dynamics after trapping": change the mitigation to "FD-type Jacobians (D32 branch 3, D35); Track R closed at R0".
3. Phase P7, step 4: add "Log the tangent norms per species, with the median and the 90th percentile, in addition to the maximum (driver callback)."
4. Section 12: add a subsection "12.11 Track R Phase R0" that records the verdict, the Q1 result when it exists, and D35.

## References

Paper (read on the page images where cited):

1. Chung, S. W., Bond, S. D., Cyr, E. C., Freund, J. B., "Regular sensitivity computation avoiding chaotic effects in particle-in-cell plasma methods", J. Comput. Phys. 400, 108969 (2020), [OSTI 1570259](https://www.osti.gov/biblio/1570259). Local copy `/mnt/work/snidhan/projects/applied-research/solvers/fbpic/literature/Chung2020_regular_sensitivity_PIC_OSTI1570259.pdf`. Sections and PDF pages used: Section 2.3 commutability, p10 ("so long as sensitivity in the continuum limit is stable"); Eq. (23a), (24), (25), (26), (27), p12; Eq. (28), (29), p13; Section 3.2.2, Eq. (30), (31), p14; Eq. (32) to (37), p15; Section 3.2.3, Eq. (38) to (42), p16 (Eq. (41) text says $S_x = S_v = b_l$ but uses $b_{l+1}$); Eq. (43), p17; Eq. (44) to (48), p18; Fig. 4, p21; Section 6.1, Eq. (76) to (81), p29; Eq. (92), (93), p33; Fig. 11, p34; Section 7.1 cost, p35; Section 8.4 and Fig. 19, p44 to p49; Section 9 discussion, p49 to p50 (criterion for Eq. (92) unknown; adjoint as future work).

Other literature (not re-read for this note unless stated):

2. Bartsch, Knopf, Scheurer, Weber, "Controlling a Vlasov-Poisson plasma by a particle-in-cell method based on a Monte Carlo framework", [arXiv:2304.02083](https://arxiv.org/abs/2304.02083). Adjoint of controlled Vlasov-Poisson with $\lambda\,\partial_vf$ coupling; summary only (`sec_multi`).
3. Wang, Hu, Blonigan, "Least squares shadowing sensitivity analysis of chaotic limit cycle oscillations", JCP 267 (2014), [arXiv:1204.0159](https://arxiv.org/abs/1204.0159). Summary only.
4. Chandramoorthy, Wang, "A computable realization of Ruelle's formula for linear response of statistics in chaotic systems", [arXiv:2002.04117](https://arxiv.org/abs/2002.04117). Score-type regularization on attractors; summary only.
5. Péraud, Hadjiconstantinou, "Adjoint-based deviational Monte Carlo methods for phonon transport calculations", Phys. Rev. B 91, 235321 (2015), [OSTI 1370983](https://www.osti.gov/pages/biblio/1370983). Cited by the paper, p50.
6. Cartis, Fiala, Marteau, Roberts, "Improving the flexibility and robustness of model-based derivative-free optimization solvers", ACM TOMS (2019), [arXiv:1804.00154](https://arxiv.org/abs/1804.00154); software [DFO-LS](https://github.com/numericalalgorithmsgroup/dfols). From memory; not checked for this note.
7. Griewank, Walther, "Algorithm 799: revolve", ACM TOMS 26 (2000). Binomial checkpointing; from memory.
8. Higuera, Cary, "Structure-preserving second-order integration of relativistic charged particle trajectories in electromagnetic fields", Phys. Plasmas 24, 052104 (2017). Cited for the volume preservation of the Vay pusher; not verified.

## Appendix A. Derivations

### A.1 Dipole identity [derived]

With $\mathbf v = c\,\mathbf u/\gamma = c\,\nabla_u\gamma$, $\partial v_j/\partial u_i = (c/\gamma)(\delta_{ij} - u_iu_j/\gamma^2)$ is symmetric. Then $\nabla_u\cdot(\mathbf v\times\mathbf B) = \epsilon_{ijk}B_k\,\partial v_j/\partial u_i = 0$, and $\nabla_u\cdot\mathbf E = 0$ because $\mathbf E$ does not depend on $\mathbf u$. Hence $\nabla_u\cdot\delta\mathbf a = 0$ and

$$
-\delta\mathbf a\cdot\nabla_u f = -\nabla_u\cdot(f\,\delta\mathbf a).
$$

With $\nabla_x\cdot\mathbf v = 0$, the flow in $(\mathbf x, \mathbf u)$ preserves volume, so constant marker weights are exact for relativistic $\mathbf v(\mathbf u)$ without a Jacobian factor.

### A.2 Score form and the cold limit [derived]

Over one step, $\delta f = -\nabla_u\cdot(f\,\xi_u)$ with $\xi_u = \delta\mathbf a\,\Delta t$. For a smooth $f$, $-\nabla_u\cdot(f\xi_u) = -f\,\xi_u\cdot\nabla_u\ln f$ (because $\nabla_u\cdot\xi_u = 0$), so $\Delta\hat W_p = -W_p\,\Delta t\,\delta\mathbf a_p\cdot\nabla_u\ln f(\mathbf z_p)$. With collocated equal weights, $\hat n \approx f/W$, and paper Eq. (42) gives $h_s \approx W_s\,\delta\mathbf a\cdot\nabla_u\ln\tilde f$, with $\tilde f$ the momentum-smoothed $f$: the collocated scheme is a kernel-density score.

Cold-limit theorem. In a cell where all markers of a species have velocity $\mathbf v_0$, any charge-neutral weight change gives $\delta\mathbf J = q\mathbf v_0\sum_p\Delta\hat W_p = 0$. The exact response is $q\,n\,\delta\mathbf v \ne 0$. The Lorentz source is charge neutral (a $u$-divergence), so a correct weight scheme gives zero current response in a cold cell, and a non-neutral scheme creates spurious charge. Measured one-step $\delta J/\delta J_{exact}$: 0.000 to 0.006 cold; warm values follow $\sigma^2/(\sigma^2 + \Delta v^2/4)$ [measured, `sec_vm` 2.4, `r0/work_vm/cold_limit_pdf.py`].

### A.3 Weak form and bias [derived]

Let $A = \int\phi\,f(T)\,dz$ and $\psi_t = \phi\circ\Phi_{t\to T}$. Volume preservation gives

$$
\delta A = \int\psi_0\,\delta f_0\,dz + \int_0^T\!dt\!\int dz\,\big[f\,\delta\mathbf a\cdot\nabla_u\psi_t + \psi_t\,S_b\big]. \tag{A1}
$$

For $f = \sum_pw_p\delta(z - z_p)$ the first source term is the Lagrangian chain rule; $\nabla_u\psi_t = (\partial\Phi_{t\to T}/\partial\mathbf u)^T\nabla\phi$ grows as $e^{\lambda(T-t)}$. Replace $\nabla_uf$ by $\nabla_u(K_\sigma *_u f)$ with a symmetric kernel of variance $\sigma_u^2$:

$$
\delta A_\sigma - \delta A = \int_0^T\!dt\!\int dz\,f\,\big[K_\sigma *_u(\delta\mathbf a\cdot\nabla_u\psi_t) - \delta\mathbf a\cdot\nabla_u\psi_t\big] \tag{A2}
$$
$$
= \frac{\sigma_u^2}{2}\int_0^T\!dt\!\int dz\,f\,\nabla_u^2(\delta\mathbf a\cdot\nabla_u\psi_t) + O(\sigma_u^4). \tag{A3}
$$

For a smooth $f$ the kernel moves onto $f$:

$$
\delta A_\sigma - \delta A = -\frac{\sigma_u^2}{2}\int_0^T\!\!\int\psi_t\,\delta\mathbf a\cdot\nabla_u\nabla_u^2f + O(\sigma_u^4), \tag{A4}
$$

so the warm bias is $O(\sigma_u^2/v_T^2)$ and only $\lVert\psi_t\rVert_\infty$ enters. For a cold $f$, $\nabla_u^3f$ is not integrable, only (A3) remains, and the relative bias is $(\sigma_u/\ell_\psi)^2$. In (A2), $K_\sigma *_u\nabla_u\psi_t$ is bounded by $\mathrm{osc}(\psi_t)/\sigma_u$ up to a kernel constant, so the smoothed estimate cannot grow; a symmetric dipole pair at $\mathbf u\pm\sigma_u\hat{\mathbf e}$ evaluates the central difference $[\psi_t(\mathbf u + \sigma_u\hat{\mathbf e}) - \psi_t(\mathbf u - \sigma_u\hat{\mathbf e})]/(2\sigma_u)$.

Collocated scheme on a sheet. Combining paper Eq. (41), (42), (44): $\delta A_{coll} = -\sum_t\Delta t\,\Delta\mathbf x\Delta\mathbf v\sum_{c:\hat n_c > 0}\mathcal H_c\,\langle\psi_t\rangle_c$, with $\langle\psi\rangle_c$ the sampler average in cell $c$. On a uniform sheet the $\langle\psi\rangle_c$ are equal; the gradient stencil of Eq. (36) is one cell wider than the $\hat n$ stencil of Eq. (41), so cells with $\mathcal H_c \ne 0$, $\hat n_c = 0$ drop out, and the result is the lost source times $\langle\psi\rangle$, not a derivative. The normal derivative cannot be observed from samples on the sheet.

Transfer bias (correction of `sec_gradf` G.3.3). In expectation $\sum_c\mathcal H_c\langle\psi\rangle_c = \int\psi\,\tilde{\mathcal H}\,dz$ with $\tilde{\mathcal H} = \sum_c(\mathcal H_c/\hat n_c)\,b_c\,\hat n$. The bias $\int\psi(\tilde{\mathcal H} - \mathcal H)$ is bounded by $\lVert\psi\rVert_\infty\lVert\tilde{\mathcal H} - \mathcal H\rVert_1 = O(\Delta^2)$ for a smooth $f$.

### A.4 Galerkin conversion at a reset [derived]

The particle measure $\mu_\theta = \sum_iw_i\delta_{z_i}$ has $\mu' = \sum_i\dot w_i\delta_{z_i} - \nabla_z\cdot(\sum_iw_i\xi_i\delta_{z_i})$. At a reset choose the basis $\{\psi_a\}$ and set

$$
\Delta w_i = w_i\sum_ac_a\psi_a(z_i),\qquad Gc = b,\qquad G_{ab} = \sum_{i\in\mathcal C}w_i\psi_a(z_i)\psi_b(z_i),\qquad b_a = \sum_{i\in\mathcal C}w_i\,\xi_i\cdot\nabla_z\psi_a(z_i), \tag{R2}
$$

then $(\dot w_i, \xi_i) \to (\dot w_i + \Delta w_i, 0)$. This is the minimum-$\sum\Delta w^2/w$ change that reproduces every moment in $V = \mathrm{span}\{\psi_a\}$. For any $\psi = \Psi a + r$ with $r$ $w$-orthogonal to $V$:

$$
\sum_i\Delta w_i\psi(z_i) - \sum_iw_i\xi_i\cdot\nabla\psi(z_i) = -\sum_iw_i\xi_i\cdot\nabla r(z_i), \tag{R3}
$$
$$
\lvert\epsilon_{reset}(\psi)\rvert = \Big\lvert\Big\langle(I-P)\tfrac{\delta f}{f},\,(I-P)\psi\Big\rangle_f\Big\rvert \le \Big\lVert(I-P)\tfrac{\delta f}{f}\Big\rVert_f\lVert(I-P)\psi\rVert_f . \tag{R4}
$$

The relevant $\psi$ is the response of the final observable to a weight perturbation at the reset, not $\phi$ itself. The error grows as $f$ shears and folds, because $(I-P)\,\delta f/f$ grows.

Basis rows. Charge rows $S_g(z,r)e^{-im\varphi}$ keep $\delta\rho_g$; current rows $v_k(\mathbf u)S_ge^{-im\varphi}$ keep $\delta\mathbf J_g$. A reset changes $\delta\mathbf J_g$ by $q\sum_i\Delta w_i\mathbf v_iS_g - q\sum_iw_i(\delta\mathbf v_iS_g + \mathbf v_i\,\xi_{x,i}\cdot\nabla S_g)$, which is zero only if $\mathbf vS_g \in V$. In a cold node $\psi_J = v_{0k}\psi_\rho$ while $b_J = v_{0k}b_\rho + q\sum w\,\delta v_kS_g$, so the system is inconsistent exactly when $\delta\mathbf v \ne 0$; the least-squares residual flags it. Spread residuals: $\delta\sigma_k^2 = 2\,\mathrm{Cov}(z_k, \xi_k)$; a first-moment projection $\Delta w_p = w_p(\mathbf z_p - \bar{\mathbf z})^TC^{-1}\bar\xi$ changes a bin's second moment only through its third moments, so a degree-2 or higher basis is required.

### A.5 Why a reset removes the stretching [derived]

For any smooth map $\Phi$, $\Phi_\#(-\nabla\cdot P) = -\nabla\cdot\Phi_\#(D\Phi\,P)$. The Lagrangian form carries $D\Phi$ on each dipole; after one integration by parts at the reset the weight form has absorbed it, and the flow carries scalars without a Jacobian. Between resets $\xi$ restarts from 0 and grows at most by $e^{\lambda T}$; with $\lambda = 0.02$ per step this is 1.65 at $T = 25$ and 2.72 at $T = 50$.

### A.6 Adjoint [derived]

With multipliers $\lambda_s$, $\varepsilon_0\boldsymbol\alpha$, $\boldsymbol\beta/\mu_0$ and terminal $\lambda_e(T) = \phi$:

$$
\partial_t\lambda_s + \mathbf v\cdot\nabla_x\lambda_s + \mathbf a_s\cdot\nabla_u\lambda_s = q_s\,\mathbf v\cdot\boldsymbol\alpha, \tag{A5}
$$
$$
\partial_t\boldsymbol\alpha = c^2\nabla\times\boldsymbol\beta + \sum_s\frac{q_s}{\varepsilon_0m_sc}\int\lambda_s\nabla_uf_s\,d^3u,\qquad
\partial_t\boldsymbol\beta = -\nabla\times\boldsymbol\alpha + \sum_s\frac{\mu_0q_s}{m_sc}\int\lambda_s(\nabla_uf_s\times\mathbf v)\,d^3u, \tag{A6}
$$

and $dG/d\theta_j = \varepsilon_0\int\boldsymbol\alpha(0)\cdot\mathbf e_j(0) + \mu_0^{-1}\int\boldsymbol\beta(0)\cdot\mathbf b_j(0) + \sum_s\int\lambda_s(0)g_{j,s}(0) + \int_0^T\sum_s\int\lambda_s\partial_{\theta_j}S_s$. Since $\nabla_u\times\mathbf v = 0$, $\int\lambda(\nabla_uf\times\mathbf v) = -\int f(\nabla_u\lambda\times\mathbf v)$. The discrete particle-pdf adjoint is the transpose of the forward linear recurrence $Y^{n+1} = M_nY^n$; $\lVert M_{n_2-1}\cdots M_{n_1}\rVert_2 = \lVert M_{n_1}^T\cdots M_{n_2-1}^T\rVert_2$, so it grows exactly as much as the forward scheme.

### A.7 Gauss-Newton contraction and the fixed-point shift [derived]

Linearize $r(\theta) = r^\ast + J(\theta - \theta^\ast)$ at the least-squares solution ($J^Tr^\ast = 0$). Gauss-Newton with $J_A$ gives $e_{k+1} = (I - J_A^+J)e_k - J_A^+r^\ast$. The contraction factor is $\kappa = \lVert I - J_A^+J\rVert_2$; for $K = 1$, $\kappa = \lvert 1 - \cos/\rho\rvert$. The fixed point is $e_\infty = -(J_A^TJ)^{-1}(J_A - J)^Tr^\ast$; for $K = 1$ its size is the D32 $\delta$. Near the solution $\delta \approx (\lVert r^\ast\rVert/\lVert J\rVert)\lvert\cos(J_A, r^\ast)\rvert/\lvert\cos(J_A, J)\rvert$, independent of $\lVert J_A\rVert$. Check on P7: $\kappa_{AD} = 1 - 0.25/355 = 0.9993$.

## Appendix B. R0 evidence index

| Test | Regime | Result | Source |
|---|---|---|---|
| Debye pilot | warm, N-body chaos, no trapping | particle-pdf 11.79 and 12.86 against 12.4; SD 3.5 times the paper's | [measured, `sec_tests` 2.3] |
| Debye twins | warm | $\lambda = 0.18, 0.11, 0.077$ for $N = 10^3, 10^4, 10^5$ ($\propto N^{-0.18}$) | [measured, `sec_tests` 2.2; `crit_physics` F4] |
| Cold one-step | cold, 1D-1V | $\delta J/\delta J_{exact} \le 0.006$ | [measured, `sec_vm` 2.4] |
| Gaussian moment score | skewed and bimodal bins | $\delta J$ factor 0.40 and 4.49 | [measured, `sec_vm` 2.5] |
| Cold beam, prescribed overlapping waves, $T = 10$ | cold, before chaos | collocated wrong sign at every $\Delta v$; pairs need $\sigma = 10^{-3}$ for 10% | [measured, `sec_gradf` G.4] |
| Same, $T = 30$ | cold, chaotic | reference itself moves from $-0.98$ to $+0.44$ with $h$; pairs track $h$ with $\sigma \approx 2h$ to $5h$ | [measured, `sec_gradf` G.4] |
| E1, E1b Galerkin conversion | warm, two-stream | unbiased for $\sigma \ge 0.01$; exact sheet fails silently | [measured, `sec_rv` 6.1, 6.2] |
| E2 standard map | filamenting $f$ | 0.39, 0.21, $-0.17$ against 0.74 | [measured, `sec_rv` 6.3] |
| Driven trapping, 1D-1V | warm, trapping | +24 bias against a scale-dependent reference; transpose exact | [measured, `sec_multi` 5; `crit_feasibility` 3] |
| R2 pilot, $t = 75$ | cold, EV trapping | exact column cos $-0.28$; single-loading FD itself irregular | [measured, `sec_tests` 2.4] |
| Pendulum | warm, integrable | 15% to 56% error at $T = 40$ at the paper's relative resolution | [measured, `crit_physics` F5] |
| P7 FD roughness | Twin-S | $A \propto h^{0.73}$ | [measured, `r0/work_note/holder_check.py`] |
| P7 slice roughness | Twin-S | 31% to 44% at $h = 0.0025$, 15% to 28% at $h = 0.01$, after curvature removal | [measured, `r0/work_note/detrended_spread.py`] |
| Gate calibration | Twin-S | FD passes, AD fails, random fails | [measured, `r0/work_note/gate_calibration.py`] |

## Appendix C. Files

Scripts written for this note (CPU only; run from `lpa/FBPIC_Warp` with `python3 -I <script> inversion/runs/p7_fe7188bf9100 inversion/results/p7_jacobian.json` where arguments apply):

1. `inversion/notes/r0-evidence-2026-10-31/work_note/detrended_spread.py` and `detrended_spread.out`: curvature against roughness on the P7 solution slice.
2. `inversion/notes/r0-evidence-2026-10-31/work_note/gate_calibration.py` and `gate_calibration.out` (takes only the cache directory): guarded gate calibration.
3. `inversion/notes/r0-evidence-2026-10-31/work_note/holder_check.py` and `holder_check.out` (no arguments): Hölder fit of the plan 12.10 FD table.
4. `inversion/notes/r0-evidence-2026-10-31/work_note/timeline_check.py` and `timeline_check.out` (no arguments): Twin-S laser and bunch timeline and growth rates.
5. `inversion/notes/r0-evidence-2026-10-31/work_note/rerun_*.out`: re-runs of the `crit_feasibility` scripts (identical to the originals).

R0 inputs: `r0/sec_vm.md`, `r0/sec_gradf.md`, `r0/sec_rv.md`, `r0/sec_multi.md`, `r0/sec_tests.md`, `r0/crit_physics.md`, `r0/crit_feasibility.md`, and their `r0/work_*` directories, all under `inversion/notes/r0-evidence-2026-10-31/`.
