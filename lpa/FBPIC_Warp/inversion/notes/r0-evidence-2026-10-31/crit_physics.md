# R0 critique: math and physics correctness

Critic key: `crit_physics`. Date: 2026-11-01. Scope: sections `sec_vm.md`, `sec_gradf.md`, `sec_rv.md`, `sec_multi.md`, `sec_tests.md`, and the central Track R claim.

Labels: **[paper]** Chung, Bond, Cyr, Freund, JCP 400, 108969 (2020), checked on the page images (PDF pages 9 to 20, 28 to 34, 46 to 50). **[derived]** derived here. **[measured]** by a CPU script in `inversion/notes/r0-evidence-2026-10-31/work_crit_physics/`. **[plan]** `DIFFERENTIABLE_INVERSION_PLAN.md`. **[code]** worktree code. **[estimated]** stated basis.

## 0. Verdict

The linear algebra in the five sections is correct. I re-derived the dipole identity, the weak form, the kernel-bias formula, the moment score, the Galerkin projection and its error bound, the continuous adjoint, and the usability metric $\kappa$. I found no sign or factor error that changes a conclusion.

The central physics claim is not supported. A particle-pdf, R-v, or adjoint estimate is bounded by construction, because it is a $\sigma_u$-smoothed score-function estimator. Boundedness is therefore no evidence of accuracy. Accuracy needs a density $f$ that is smooth at the kernel scale over the whole horizon. The paper shows this only for near-equilibrium Maxwellian plasmas. Twin-S violates it in three independent ways: cold sheets, a short non-equilibrium transient in a density downramp, and a Hölder-rough FD response. Every R0 experiment with trapping or mixing gives an $O(1)$, $\sigma_u$-dependent error. My integrable test (no chaos) also fails at the paper's relative mesh resolution after 3 to 6 rotations; it converges only with a mesh that shrinks as $1/t$.

Recommendation: no-go for Track R as specified, unless one cheap GPU test (Section 3, F3) shows that the Twin-S roughness falls as $N^{-1/2}$. Even then, the expected product is a coarse-grained derivative that is no better defined than an FD secant at a matched scale.

## 1. Checked and confirmed

1. Dipole identity [derived]. $\partial v_j/\partial u_i = (c/\gamma)(\delta_{ij} - u_iu_j/\gamma^2)$ is symmetric, so $\nabla_u\cdot(\mathbf v\times\delta\mathbf B) = \epsilon_{ijk}\,\delta B_k\,\partial v_j/\partial u_i = 0$. Equivalently $\mathbf v = c\nabla_u\gamma$, so $\nabla_u\times\mathbf v = 0$. Hence $-\delta\mathbf a\cdot\nabla_u f = -\nabla_u\cdot(f\,\delta\mathbf a)$ with $\delta\mathbf a = \frac{q}{mc}(\delta\mathbf E + \mathbf v\times\delta\mathbf B)$. All five sections state this correctly.
2. Paper algorithm [paper]. Eq. (25) p12, Eq. (29) p13, Eq. (34) and (36) p15, Eq. (41) and (42) p16, Eq. (43) p17, Eq. (44) p18, Eq. (51)-(54a) p19-20 match `sec_vm.md` Section 1. Eq. (54a) reads $d\hat W_{exist}/dt = -h_{exist}$ on the image. One paper inconsistency: Eq. (41) says "using $S_x = S_v = b_l$" but then uses $b_{l+1}$ (p16).
3. Score form [derived]. With equal collocated weights, $\hat n \approx f/W$ and $h_s \approx W\,\delta\mathbf a\cdot\nabla_u\ln\tilde f$, so $\Delta\hat W_p = -W_p\Delta t\,\delta\mathbf a_p\cdot\nabla_u\ln\tilde f$ (`sec_vm.md` 2.3). The Gaussian score $-C^{-1}(\mathbf u - \bar{\mathbf u})$ gives $\Delta W_p = W_p\,\bar\xi^TC^{-1}(\mathbf z_p - \bar{\mathbf z})$ with the correct sign.
4. Weak form and bias [derived]. With $\partial_t\psi + \mathbf v\cdot\nabla_x\psi + \mathbf a\cdot\nabla_u\psi = 0$, $\frac{d}{dt}\int\psi\,\delta f = \int f\,\delta\mathbf a\cdot\nabla_u\psi$ (`sec_gradf.md` G1). Smoothing $f$ by a symmetric kernel gives G2 and G3 with the stated sign.
5. Galerkin conversion [derived]. $\sum_i\Delta w_i\psi_b(z_i) = (Gc)_b = b_b$, and the residual identity R3 and bound R4 of `sec_rv.md` hold.
6. Adjoint [derived]. From the Lagrangian with multipliers $\lambda$, $\varepsilon_0\boldsymbol\alpha$, $\boldsymbol\beta/\mu_0$: variation in $g$ gives A1; in $\mathbf e$ gives $\partial_t\boldsymbol\alpha = c^2\nabla\times\boldsymbol\beta + \frac{q}{\varepsilon_0 mc}\int\lambda\nabla_u f$; in $\mathbf b$, using $(\mathbf v\times\mathbf b)\cdot\mathbf g = \mathbf b\cdot(\mathbf g\times\mathbf v)$, gives $\partial_t\boldsymbol\beta = -\nabla\times\boldsymbol\alpha + \frac{\mu_0 q}{mc}\int\lambda(\nabla_u f\times\mathbf v)$. This matches `sec_multi.md` A2. The transposed recurrence has the same singular values as the forward one, so "regular iff forward is regular" is correct.
7. Metric [derived]. For $K = 1$, $J_A^+J = \cos/\rho$, so $\kappa = |1 - \cos/\rho|$; P7 AD gives $1 - 0.25/355 = 0.9993$ (`sec_tests.md` 6.1).
8. Code citations used for physics [code]. Vay kick terms `econst * dEx` at `fbpic/autodiff/tangent.py:582-599`; newborn copies parent $\xi$ and takes $\dot W'$ at `fbpic/autodiff/ionization.py:762-764`; weight tangent `w` at `fbpic/autodiff/state.py:105`; loss reads `t["w"]` at `fbpic/autodiff/loss.py:273`; `current_correction="curl-free"` at `inversion/twin_config.py:236`; all species loaded with no thermal momentum at `inversion/twin_config.py:272-275`.

## 2. What the dipole identity means for correctness [derived]

The identity makes particle-pdf one member of a known estimator family. The Lagrangian tangent is the pathwise estimator. The weight form is the likelihood-ratio (score) estimator. Both are exact for a smooth $f$. They differ in variance: pathwise variance grows as $e^{2\lambda t}$; score variance grows as $1/\sigma^2$ of the local momentum width.

The likelihood-ratio method has one hard precondition. The parameter must act through a density with $\theta$-independent support. A cold stream $n\,\delta(\mathbf u - \mathbf U(\mathbf x,t))$ and the 4-D EV manifold have no such density. Their score does not exist. Every working scheme must therefore replace $f$ by $K_\sigma * f$. In weak form (G2) this replaces $\nabla_u\psi_t$ by a $\sigma_u$-difference of the pulled-back observable. The result is bounded by $\mathrm{osc}(\psi)/\sigma_u$ per unit source time for any dynamics.

Consequence. "No exponential growth" is guaranteed by construction and proves nothing. The paper's claim of regularity, "so long as sensitivity in the continuum limit is stable" (p10; Eq. (81) p29; Eq. (92) p33; p49), is a statement about the limit $N \to \infty$ at fixed smooth $f$. It is not a statement about a $\sigma_u$-smoothed cold $f$. The only meaningful gate is agreement with a reference across a plateau in $\sigma_u$. No R0 experiment with trapping found such a plateau.

## 3. Findings, ranked

### F1. The paper's evidence does not transfer to Twin-S (blocking)

Claim attacked: the plan Track R block (plan line 858) and all five sections treat the continuum stability assumption, Eq. (81) and (92), as plausible for LWFA.

Evidence:

1. Debye (p20 to p34): $f$ is Maxwellian and relaxes to a function of the energy, so $f$ never filaments. The chaos is N-body: $\overline{\Delta x_p}$ grows "at an N-dependent rate" and $W_1(\Delta n_N)$ saturates as $N^{-1/2}$ (Fig. 11a, 11b, p34).
2. The paper checks Eq. (92) with a finite-volume solver whose "dissipative upwind scheme and TVD flux limiter" give only "a lower bound for $W_1(\Delta n)$" (p33). That check is itself coarse-grained.
3. The forced sheath (Section 8.4) is validated against "a $\Delta v_0 = 0.1$ moving window averaged" brute-force curve at three points (Fig. 19b, p49). Its sources are Maxwellian fluxes (Eq. (97), (113)).
4. The paper states that no criterion for Eq. (92) is known (p49).
5. Every R0 test with trapping or mixing fails at $O(1)$: `sec_gradf.md` G.4 (cold beam: no $\sigma$-plateau at $T = 30$; the estimate tracks the FD step with $\sigma \approx 2h$ to $5h$); `sec_rv.md` E2 (0.39, 0.21, $-0.17$ against 0.74, no convergence in $M$ or $T$); `sec_multi.md` 5.3 (bias about 0.9 times the reference RMS after trapping, mesh dependent). The only passing test is the Debye reproduction (`sec_tests.md` 2.3).
6. My integrable test (F5) fails without any chaos.

The section verdicts ("conditional", "feasible") are more optimistic than their own data.

### F2. The Twin-S growth window is a mean-field transient, not established N-body chaos (blocking)

Claim attacked: plan line 858, "This is the Lyapunov instability of the N-body particle dynamics", repeated in the briefing and used by `sec_rv.md` 3.4 as the favourable case.

Evidence [derived from `inversion/configs/twin_s.json` and plan 12.10 step 4; `work_crit_physics/timeline.py`, `lambda_vs_N.py`]:

1. Density is a Gaussian, peak at 38 µm, FWHM 28 µm, cut at 66 µm where $n = n_0/16$ (`inversion/twin_config.py:83-94`). The laser centre passes the peak at step 960 and leaves the plasma at step 1520. The bunch, about 12 µm behind, leaves at about step 1760 to 1820.
2. Growth rate of $\lVert\partial u_x/\partial E_L\rVert_\infty$: 0.52 $\omega_{p0}$ (steps 1200 to 1500), 0.80 $\omega_{p0}$ (1500 to 1800), 0.13 $\omega_{p0}$ (1800 to 2000). The fastest growth occurs after the laser has left the plasma, with the bunch in the downramp ($n/n_0 = 0.45 \to 0$).
3. After step 1800 the position tangent growth is ballistic. Free drift gives $\Delta(\partial x/\partial E_L) \approx c\,|\partial u_x/\partial E_L|\,T/\gamma = 1.4\cdot10^5$ m/J at $\gamma = 12$; P7 shows $1.1\cdot10^5$.
4. The betatron frequency is $\omega_\beta = \omega_{p,loc}/\sqrt{2\gamma} = 0.07$ to $0.39\,\omega_{p0}$. Over steps 1200 to 1800 the bunch completes 0.3 to 0.8 betatron periods. A rate of 0.5 to 0.8 $\omega_{p0}$ is 2 to 10 times the betatron frequency and exceeds the local $\omega_p$ late in the window.

Interpretation [derived, not proven]. The growth cannot come from betatron chaos or phase mixing of the bunch. The candidates are separatrix (X-point) stretching at trapping, wave breaking of the driverless wake in the downramp after step 1520, the transit of the 66 µm density cliff, and N-body or grid noise. The first three are mean-field and fold cold sheets. These are the regimes where every R0 toy fails. Also, P1 reports 44387 background-electron macroparticles in the window at $N_t$ although the window lies past $z_{hi}$ (plan 12.3 step 8); if these are not guard-region artifacts, background electrons were injected, which supports wave breaking (not checked).

Fix: label the P7 mechanism "unidentified" in the plan and the design note. Add a timing diagnostic: per-species, per-quantile tangent growth plus the step when each high-growth particle crosses $u_z = 1$.

### F3. The FD evidence for a regular continuum is weak; the response is Hölder-rough (major)

Claim attacked: "FD secants at $h = 10^{-2}$ and $10^{-3}$ agree" as the evidence for Eq. (81) (plan Track R item 4; all sections).

Evidence [measured from the plan 12.10 step 4 table, `work_crit_physics/fd_roughness.py`]. Define the rough amplitude $A(h) = h\,|D(h) - D(10^{-2})|$ per residual. Median $A$: $5.6\cdot10^{-3}$, $1.9\cdot10^{-3}$, $4.1\cdot10^{-4}$, $6.1\cdot10^{-5}$, $7.1\cdot10^{-6}$ at $h = 10^{-3}$ to $10^{-7}$. A fit gives $A \propto h^{0.73}$ ($h^{0.81}$ over $10^{-7}$ to $10^{-4}$). The residual map behaves like a Hölder-0.75 function over four decades, with no scale at which the secant converges. Extrapolated to $h = 10^{-2}$, the rough part is about 4.4 against a median secant of 37.5, about 12%. The column at $h = 10^{-3}$ differs from $h = 10^{-2}$ by 17% RMS.

Meaning. The data fit both an N-body roughness ($\propto N^{-1/2}$, vanishing in the continuum) and a continuum roughness from sheet folds (independent of $N$), as in `sec_gradf.md` G.4 at $T = 30$. Only the first case gives particle-pdf a well-defined target.

Fix (decisive and cheap; existing primal code, no new kernels): repeat the central-FD $h$-scan ($h = 10^{-2}$ to $10^{-6}$) at the nominal $\theta$ with 4 times the macroparticles (`p_nt` 8 to 32, or `p_nr` 2 to 8), about 10 primal runs at about 6 min each. Go only if $A(h)$ drops by about 2 at every $h$. If $A(h)$ does not change, the roughness is continuum physics, and no derivative method can deliver more than a coarse-grained secant.

### F4. The chaos-type kill criterion of `sec_rv.md` misclassifies the paper's own case (major)

Claim attacked: `sec_rv.md` Section 8 kill criterion 1: if the median growth rate stays within 20% when the particle count doubles, the chaos is mean-field.

Evidence [measured from `sec_tests.md` 2.2; `work_crit_physics/lambda_vs_N.py`]. The Debye pilot, which is N-body chaos, gives $\lambda = 0.18, 0.11, 0.077$ for $N = 10^3, 10^4, 10^5$, so $\lambda \propto N^{-0.18}$. Doubling $N$ lowers $\lambda$ by 12%, inside the 20% band. The criterion would label the paper's own case as mean-field. Chung use the amplitude, not the rate: $W_1$ saturation $\propto N^{-1/2}$ (Fig. 11b, p34).

Fix: replace the rate criterion with the amplitude test of F3, or with the seed-to-seed spread of a fixed-$h$ secant against $N$.

### F5. Integrable phase mixing alone breaks a fixed-mesh particle-pdf; neither regime of the EV bunch is smooth (major)

Claims attacked: `sec_rv.md` 3.4 and table 7 ("N-body chaos, smooth $f$: works"; the dichotomy N-body against mean-field); `sec_vm.md` 2.4 ("after trapping, betatron and synchrotron phase mixing fill a volume in $u$; the trapped bunch is the population that the weight form can represent"); `sec_vm.md` 6 hybrid warmth test $\lambda_{min}(C_u) \ge 4\sigma_m^2$.

Evidence:

1. Liouville [derived]. The flow preserves phase-space volume and maps a 4-D manifold to a 4-D manifold. The EV bunch, born cold on a 4-D manifold (`sec_gradf.md` G.2.3), stays a sheet at all times. Phase mixing fills a volume only after coarse-graining, and that coarse-graining is the $\sigma_u$ bias.
2. Timescale [derived, F2]. The bunch completes less than one betatron period in the growth window. It is not phase mixed even in the coarse sense; it is close to a cold sheet.
3. Integrable test [measured, `work_crit_physics/pendulum_pdf.py`, `pendulum_pdf.log`, `pendulum_summary.log`]. Model: pendulum in a prescribed field, $\dot x = v$, $\dot v = -\theta\sin x$, warm Gaussian blob ($\sigma = 0.25$) inside the island, source $\delta a = -\sin x$, observables $\langle v^2\rangle$, $\langle\cos x\rangle$, $\langle S(v)\rangle$. There is no chaos: the median Lagrangian $|\xi|$ grows linearly (3.0, 4.8, 10.6, 21.8 at $T = 5, 10, 20, 40$). The Lagrangian tangent at $N = 10^6$ matches CRN FD at $h = 0.003$ within one SE at every $T$. The collocated scheme (Eq. (36), (41), (42), (44); $N = 4\cdot10^5$) has these errors, normalised by the RMS of the reference over $T$:

   | Mesh | $\Delta v/\sigma$, $\Delta x/\sigma$ | $T = 20$ (2.9 rotations) | $T = 40$ (5.8 rotations) |
   |---|---|---|---|
   | $\Delta v = 0.1$, $N_x = 64$ | 0.40, 0.39 | 0.11 to 0.24 | 0.23 to 0.56 |
   | $\Delta v = 0.05$, $N_x = 64$ | 0.20, 0.39 | 0.08 to 0.19 | 0.15 to 0.43 |
   | $\Delta v = 0.025$, $N_x = 128$ | 0.10, 0.20 | 0.02 to 0.06 | 0.08 to 0.16 |

   At $T \le 10$ the error is below 0.08 at every mesh. At $T = 40$ the bounce-phase spread across the blob is only 3 rad, yet a mesh at the paper's relative resolution ($\Delta v/v_T = 0.08$ to 0.19) has 15% to 56% error, and some observables have the wrong sign. The error grows with time at a fixed mesh and falls only when both $\Delta x$ and $\Delta v$ shrink, so the mesh must shrink about as $1/t$; in 6D that costs $2^6$ per halving. The particle-pdf SE is also 2.5 to 4 times the Lagrangian SE at equal $N$. In an integrable flow the exact tangent is better in both bias and variance.
4. Bin covariance [derived]. A folded sheet has a large coarse covariance in a bin. The warmth test of `sec_vm.md` 6 passes for a sheet and therefore cannot detect the failure mode.

Correct condition [derived]: the weight form is accurate when $f$ stays smooth at the mesh scale for the whole horizon. This holds when $f$ is close to a function of the mean-field invariants (Debye, sheath). It fails for any non-equilibrium population in an anharmonic potential after about one phase-mixing time, chaotic or not. The trapped EV bunch is such a population.

### F6. The first-moment 6D projection of `sec_vm.md` drops the sensitivity of six residuals, not two (major)

Claim attacked: `sec_vm.md` 6: $\Delta\hat W_p = W_p(\mathbf z_p - \bar{\mathbf z})^TC_6^{-1}\bar\xi$ drops $\mathrm{Cov}(\xi,\mathbf z)$, which "carries the regular sensitivity of the bin's spread (energy spread, emittance), which enters residuals 3 and 4".

Evidence [plan 12.10 step 4 table]. Six of the 11 residuals are spreads: $\log\sigma_x$, $\log\sigma_{u_x}$, $\log\sigma_y$, $\log\sigma_{u_y}$, $\log\sigma_\zeta$, $\log(\sigma_{u_z}/\langle u_z\rangle)$. A first-moment projection at every reset (every 50 to 100 steps) replaces the regular part of $\mathrm{Cov}(\xi,\mathbf z)$ by zero for all of them. If bins are larger than the bunch, the projection keeps only $\delta\langle\mathbf z\rangle$.

Fix: drop the first-moment projection. Use the Galerkin form of `sec_rv.md` (R2) with degree at least 2 and the 13 loss base sums as rows, which keeps all second moments exactly.

### F7. The Galerkin basis of `sec_rv.md` lacks current rows; adding them turns the silent cold-sheet failure into a detectable one (major)

Claim attacked: `sec_rv.md` 2.3 basis (field rows $S_g e^{-im\varphi}$ for $\rho$ only) and 2.4 ("Field part 1 of the basis sets $\Delta\rho = 0$ exactly"), and E1 ("fails silently").

Problem [derived]. A reset changes $\delta\mathbf J_g$ by $q\sum_i\Delta w_i\mathbf v_iS_g - q\sum_iw_i(\delta\mathbf v_iS_g + \mathbf v_i\,\xi_{x,i}\cdot\nabla S_g)$, which is nonzero unless $\mathbf v S_g \in V$. The curl-free correction (`fbpic/fields/cuda_methods.py:142-175`) replaces only the longitudinal part. The transverse jump enters PSATD directly as a spurious $\delta\mathbf E, \delta\mathbf B$ pulse at every reset.

Fix [derived]: add rows $\psi = v_k(\mathbf u)\,S_g(z,r)\,e^{-im\varphi}$ ($k = x, y, z$; 9 real rows per node). Then $\delta\mathbf J$ is continuous across a reset. In a cold cell, $v_k = v_{0k}$ for all particles, so $\psi_{J} = v_{0k}\psi_\rho$: the rows are collinear, while $b_{J} = v_{0k}b_\rho + q\sum w\,\delta v_k S_g$. The system is inconsistent exactly when $\delta\mathbf v \ne 0$, which is the cold-limit theorem of `sec_vm.md` 2.4. A least-squares residual per node then flags the cold-sheet failure, which is silent in E1.

### F8. Collective growth is not removed by the weight form (minor)

Claim attacked: the plan Track R item 2, "No trajectory tangent is formed, so no Lyapunov growth occurs", and `sec_multi.md` 4.1.

Problem [derived]. The weight form removes the stretching factor $D\Phi$ only. If the linearized Vlasov-Maxwell operator has a growing mode, the weights and the field tangents grow too. Candidates: beam-plasma instabilities of the bunch, or numerical Cherenkov instability of a relativistic bunch in lab-frame PSATD (Twin-S has no Galilean option, `inversion/twin_config.py:221-239`). P7 gives no sign of it: in steps 1800 to 2000 the field tangent grows by 8, as $\partial x/\partial E_L$ does, so the field growth looks slaved to the particle tangents.

Fix: in R3, one R-v reset at step 1500, then log the field-tangent norm. Continued exponential growth means a collective mode, which no Track R variant removes.

### F9. The R2a test omits the Twin-S growth regime (major)

Claim attacked: `sec_tests.md` 5.1: R2a is "the P7 failure in miniature".

Problem [derived, F2]. R2a has a ramp-up to a uniform plateau and a non-evolving driver ($\gamma_g = 10$). Twin-S grows fastest after its driver exits the plasma, in a downramp that ends at a cliff. R2a can pass while Twin-S fails. In addition, R2a is 1D-1V, and a test particle in a stationary 1D wake is integrable (`sec_tests.md` 8.3), so the 1D growth mechanism differs from 3D.

Fix: add a downramp to $n_0/16$, a density cliff, and driver exit to R2a. Record when the growth starts relative to the exit. Use the F3 amplitude-against-$N$ test inside R2.

### F10. The mesh-transfer bias claim of `sec_gradf.md` is not correct in expectation (minor)

Claim attacked: `sec_gradf.md` G.3.3 caveat: the transfer of Eq. (42) "smooths $\psi_t$ on the mesh scale ... adds a bias that depends on the roughness of $\psi_t$, even for a warm $f$".

Problem [derived]. In expectation, $\sum_c\mathcal H_c\langle\psi\rangle_c = \int\psi\,\tilde{\mathcal H}\,dz$, where $\tilde{\mathcal H}$ is the mesh-interpolated source. The bias is $\int\psi(\tilde{\mathcal H} - \mathcal H)$, bounded by $\lVert\psi\rVert_\infty\lVert\tilde{\mathcal H} - \mathcal H\rVert_1 = O(\Delta^2)$ for smooth $f$. Roughness of $\psi$ enters the variance, not the bias. The measured 2 to 4 SE deficit at $T = 30$ needs another cause: ratio bias of $\mathcal H/\hat n$ at $O(1/N_{cell})$, lost support, or statistics.

### F11. The `sec_multi.md` trapping reference is heavily smoothed and its chaos type is unknown (minor)

The reference slope is a quadratic fit over $\theta \in [0.05, 0.11]$ around $\theta_0 = 0.08$, a range of $\pm 37\%$ (`sec_multi.md` 5.2). "The macroscopic response is regular" therefore holds at that scale only. $\lambda(N)$ and the amplitude against $N$ were not measured, so the 90% bias cannot be attributed to mean-field or N-body dynamics. Run the F3 amplitude test there; it costs CPU minutes.

## 4. Answer to the central question

1. Is the continuum sensitivity regular after trapping? Not shown. The FD response is Hölder 0.75 over $h = 10^{-7}$ to $10^{-3}$ (F3). The growth window is a downramp and exit transient at 2 to 10 times the betatron frequency (F2). The EV bunch is a cold 4-D sheet that is not phase mixed (F5). The paper offers no criterion and validates only coarse-grained responses of Maxwellian plasmas (F1).
2. Does particle-pdf, R-v, or the adjoint avoid exponential growth? Yes, trivially: smoothed score estimators are bounded by $\mathrm{osc}(\psi)/\sigma_u$. Collective modes would still grow (F8).
3. Is the bounded answer correct? Only if $f$ is smooth at the kernel scale for the whole horizon. Twin-S fails this in both regimes of the bunch (unmixed sheet, or folded sheet), and probably in the background during the downramp. Every R0 test with trapping or mixing, plus my integrable test, shows $O(1)$ errors that depend on $\sigma_u$.

## 5. Strongest no-go case

A weight-form sensitivity is a likelihood-ratio estimator. It exists only for a density with $\theta$-independent support that is smooth at the kernel scale. Twin-S has cold background sheets and a 4-D EV manifold, which Liouville keeps as sheets. Its growth occurs in a short mean-field transient (downramp, driver exit, density cliff) at 2 to 10 times the betatron frequency. Its FD response is Hölder-rough over four decades of $h$. Under these conditions, any particle-pdf, R-v, or adjoint estimate is a $\sigma_u$-coarse-grained derivative. Its value moves at $O(1)$ with $\sigma_u$, as every trapping test in R0 shows. It cannot grow, so it looks regular. But it is not better defined than an FD secant at a matched step, and the forward forms do not beat one-sided FD on cost.

## 6. Files

1. `work_crit_physics/timeline.py`: Twin-S laser and bunch positions, local density, rates per interval, ballistic check.
2. `work_crit_physics/fd_roughness.py`: Hölder fit of the P7 FD secants.
3. `work_crit_physics/lambda_vs_N.py`: $\lambda(N)$ slope of the Debye pilot and betatron phase advance.
4. `work_crit_physics/pendulum_pdf.py`, `pendulum_pdf.log`, `pendulum_pdf.json`, `pendulum_summary.py`, `pendulum_summary.log`: integrable phase-mixing test of the collocated particle-pdf scheme. Run as `python3 -I -u pendulum_pdf.py 400000` (about 8 min on one core). The log label "dv=0.03" is $\Delta v = 0.025$ (print rounding).
