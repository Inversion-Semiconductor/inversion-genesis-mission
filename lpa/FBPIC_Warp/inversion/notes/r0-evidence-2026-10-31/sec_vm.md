# R0 section VM: Linearized relativistic EM Vlasov-Maxwell sensitivity mapped to FBPIC

Author: derivation agent `vm`, R0 study of Track R, 2026-10-31.
Tags: [paper] = Chung, Bond, Cyr, Freund, JCP 400, 108969 (2020), OSTI PDF; [derived] = derived here; [measured] = measured here with the script named; [estimated] = estimate with stated basis; [plan] = `DIFFERENTIABLE_INVERSION_PLAN.md`; [code] = worktree code.
Code paths are relative to `lpa/FBPIC_Warp/`. Scratch scripts are in `inversion/notes/r0-evidence-2026-10-31/work_vm/`.

## 0. Result in brief

1. The particle-pdf weight source is the score-function (likelihood-ratio) form of the per-step Lagrangian momentum kick: $\Delta\hat W_p = -W_p\,\Delta t\,\delta\mathbf F_p\cdot\nabla_u \ln f$. Chung's collocated mesh scheme is a kernel-density estimate of $\nabla_u \ln f$ [derived].
2. A weight-only representation cannot carry the linear response of a cold, single-stream species. In a cold cell every charge-neutral weight change gives zero current change, but the exact response is $q n\,\delta v \neq 0$ [derived]. Chung's scheme on a cold 1D beam gives 0.0 to 0.6% of the exact $\delta J$ and creates spurious charge [measured].
3. Every Twin-S species is loaded cold (`inversion/twin_config.py:249-275`, no thermal momentum). Before phase mixing, `elec_from_N` lies on a low-dimensional manifold in $u$. The pure particle-pdf method therefore loses the background wake response and the laser-plasma response, which drive everything else [derived].
4. For warm populations the mesh score has a smoothing bias $\delta J_{pdf}/\delta J \approx \sigma_u^2/(\sigma_u^2 + \Delta u^2/4)$ [measured, 1D]. A cheap moment (Gaussian) score is exactly charge neutral and exact for $\delta\langle u\rangle$, but it is biased by factors 0.4 to 4.5 for skewed or bimodal bins under relativistic $v(u)$ [measured, 1D].
5. Charge conservation: Twin-S uses the curl-free spectral correction (`inversion/twin_config.py:236`). It absorbs any weight-created charge into a longitudinal current and keeps Gauss's law for the deposited $\delta\rho$. No field code changes. The weight projection must be charge neutral, or the correction turns projection noise into a real $\delta\mathbf E$ [derived].
6. Recommended form for R3: variant R-v as a hybrid. Keep Lagrangian tangents on cold, laminar markers. Every $P \approx 50$ to 100 steps, project $\xi$ into weights only in bins where the local 6D covariance is well conditioned. The full 6D Chung mesh costs an estimated 20 to 40 run-equivalents per gradient. The hybrid costs about the present 3.5 run-equivalents [estimated].
7. Verdict for this topic: conditional. Three cheap measurements decide it (Section 7).

## 1. The paper's particle-pdf algorithm, exactly

I read PDF pages 5 to 21 and checked every equation below against the page images. The OCR drops the minus sign in Eq. (54a) (page 20); the image reads $d\hat W_{exist}/dt = -h_{exist}$.

### 1.1 Model and sensitivity equation [paper]

The model is 1D-1V or 3V Vlasov-Poisson, Eq. (1a)-(1c), page 5. PIC markers carry $f_N = \sum_p W_p S_x(x - x_p) S_v(v - v_p)$, Eq. (3), with constant weights, Eq. (5a). Differentiation first gives Eq. (23a), page 12:

$$
\partial_t\,\delta f + v\cdot\nabla_x \delta f + \frac{q}{m} E\cdot\nabla_v \delta f + \frac{q}{m}\,\delta E\cdot\nabla_v f + \partial_\theta \mathcal V_1 = 0 .
$$

The last two terms form the source $\mathcal H = \frac{q}{m}\delta E\cdot\nabla_v f + \partial_\theta\mathcal V_1$, Eq. (25). Then $\partial_t \delta f_M = -\mathcal H$ along the unperturbed characteristics, Eq. (29), page 13.

### 1.2 Sensitivity particles [paper]

1. $\delta f \approx \sum_{s=1}^M \hat W_s S_x(x - \hat x_s) S_v(v - \hat v_s)$, Eq. (21), page 11. The sensitivity particles can be distinct from the PIC markers.
2. They move exactly like PIC particles in the primal fields, Eq. (27a)-(27b), page 12. Their weights do not feed back into the primal field, so the force on them needs no extra solve (page 20).
3. Three schemes, Section 3.2.4, Fig. 3, page 17: collocated (sensitivity particles are the PIC markers, interpolation coefficients of Eq. (8) are reused), non-collocated (distinct particles), and particle addition (new particles each step).

### 1.3 Weight evolution, density, and source [paper]

1. Sensitivity charge: $\delta\rho_i = \frac{q}{\Delta x}\sum_s \hat W_s b_{l+1}(\hat\xi_{s,i}) + \delta\rho_{ext,i}$, Eq. (34) and (51b), page 15 and 19. This is the primal charge deposit with $W_p \to \hat W_s$.
2. Field sensitivity: the discrete Poisson system on $\delta\rho$ with problem sources, Eq. (33a)-(33b) and (51c)-(51d).
3. Source on a phase-space mesh, Eq. (36) and (52), page 15 and 19:
$$
(\delta E\cdot\nabla_v f)_{i,i_v} = \sum_{p=1}^N \frac{W_p}{\Delta x\,\Delta v}\, b_{l+1}(\xi_{p,i})\; \delta E_i\cdot\nabla_v b_{l+2}(\zeta_{p,i_v}),
$$
with $S_x = b_l$ and $S_v = b_{l+1}$ for $f$; the paper uses $l = 1$.
4. Marker number density on the same mesh, Eq. (41) and (53a): $\hat n_{i,i_v} = \frac{1}{\Delta x\Delta v}\sum_s b_{l+1}(\hat\xi_{s,i})\, b_{l+1}(\hat\zeta_{s,i_v})$.
5. Source strength per particle, Eq. (42) and (53b): $h_s = \sum_{i,i_v} (\mathcal H_{i,i_v}/\hat n_{i,i_v})\, b_{l+1}(\hat\xi_{s,i})\, b_{l+1}(\hat\zeta_{s,i_v})$.
6. Support condition, Eq. (43), page 17: $\hat n_{i,i_v} > 0$ wherever $\mathcal H_{i,i_v} \neq 0$. If it fails, add particles.
7. Weight evolution, Eq. (44) and (54a): $d\hat W_s/dt = -h_s$ (collocated and non-collocated). Particle addition, Eq. (48) and (54b): $\hat W_{new} = -\int_{t^*}^{t^*+\Delta t} h_{new}\,dt$.
8. Redistribution onto $M_{new}$ particles with modified B-splines $\tilde b$, Eq. (49)-(50) and (55)-(56), page 19 and 20.

### 1.4 Time integration [paper]

Fig. 4, page 21, is a second-order leapfrog. Positions at integer steps, Eq. (51a). Velocity half kick to $k$, then the weight update $\hat W^{k+1/2} = \hat W^{k-1/2} - h^k\Delta t$ (collocated and non-collocated) or the addition of $\hat W_{new} = -h^k_{new}\Delta t$. If $M^k > M_{max}$, redistribute. Then the second velocity half kick. Weights therefore live at half steps, like velocities.

## 2. The sensitivity system in FBPIC variables

### 2.1 Primal system [derived]

FBPIC uses $\mathbf u = \mathbf p/(m c)$, $\gamma = \sqrt{1 + u^2}$, $\mathbf v = c\mathbf u/\gamma$. For species $a$ (`electrons`, `helium`, `nitrogen` with EV, `elec_from_N`):

$$
\partial_t f_a + \mathbf v\cdot\nabla_x f_a + \mathbf F_a\cdot\nabla_u f_a = G_a, \qquad \mathbf F_a = \frac{q_a}{m_a c}\,(\mathbf E + \mathbf v\times\mathbf B).
$$

$G_a$ is the EV birth source; it is nonzero only for `elec_from_N` (and as a charge change of `nitrogen`). Maxwell's equations close the system with $\rho = \sum_a q_a\int f_a\,d^3u$ and $\mathbf J = \sum_a q_a\int \mathbf v f_a\,d^3u$.

The flow in $(\mathbf x, \mathbf u)$ is volume preserving:

$$
\nabla_x\cdot\mathbf v = 0, \qquad \nabla_u\cdot(\mathbf v\times\mathbf B) = \epsilon_{ijk} B_k\,\frac{\partial v_j}{\partial u_i} = 0,
$$

because $\partial v_j/\partial u_i = (c/\gamma)(\delta_{ij} - u_i u_j/\gamma^2)$ is symmetric. $\nabla_u\cdot\mathbf E = 0$ trivially. So $f$ is constant along characteristics in $(\mathbf x, \mathbf u)$, and constant marker weights are exact. This holds for relativistic $\mathbf v(\mathbf u)$ with no Jacobian factor.

### 2.2 Linearized system [derived]

Differentiate with respect to $\theta$. With $\delta(\cdot) = \partial(\cdot)/\partial\theta$:

$$
\partial_t \delta f_a + \mathbf v\cdot\nabla_x \delta f_a + \mathbf F_a\cdot\nabla_u \delta f_a = -\,\delta\mathbf F_a\cdot\nabla_u f_a + \delta G_a, \qquad \delta\mathbf F_a = \frac{q_a}{m_a c}\,(\delta\mathbf E + \mathbf v\times\delta\mathbf B),
$$

$$
\partial_t\delta\mathbf B = -\nabla\times\delta\mathbf E, \quad \partial_t\delta\mathbf E = c^2\nabla\times\delta\mathbf B - \delta\mathbf J/\varepsilon_0, \quad \delta\rho = \sum_a q_a\!\int\!\delta f_a\,d^3u, \quad \delta\mathbf J = \sum_a q_a\!\int\!\mathbf v\,\delta f_a\,d^3u .
$$

The term $\mathbf v\times\delta\mathbf B$ has no $\delta\mathbf v$ factor: $\mathbf v$ is a phase-space coordinate, not a state variable. The field equations are the same linear operators as the primal ones, so `FieldTangents.apply` (`fbpic/autodiff/field_tangents.py:109`) covers them unchanged.

$\theta$-derivative sources:

1. Laser energy $E_L$. The laser enters only through the initial fields. So $\delta f_a(0) = 0$ and $(\delta\mathbf E, \delta\mathbf B)(0) = F_0/(2E_L)$, after damping and `interp2spect` [plan 4.8]. Code: `energy_seed` (`fbpic/autodiff/laser.py:92`) and `seed_tangents` (`fbpic/autodiff/driver.py:92`). There is no volume source $\partial_\theta\mathcal V_1$ for $E_L$. The focal position $z_f$ behaves the same way with $t_{z_f} = \mathcal L[\partial\mathcal E/\partial z_f]$ [plan 4.8].
2. EV births. In the EV model the births are a sum over ions $i$ and emission steps $n$:
$$
G(\mathbf x, \mathbf u, t) = \sum_{i,n} B_{i,n}\,\delta(t - t_n)\,\delta(\mathbf x - \mathbf x_i)\,\delta(\mathbf u - \mathbf u_i),
$$
with $B_{i,n}$ = `birth_w` (`fbpic/autodiff/ionization.py:157-162`). Then
$$
\delta G = \sum_{i,n}\delta(t - t_n)\Big[\dot B_{i,n}\,\delta\delta - B_{i,n}\big(\xi_{x,i}\cdot\nabla_x + \xi_{u,i}\cdot\nabla_u\big)\delta\delta\Big].
$$
The first part is a scalar weight at the birth point: the existing birth tangent $\dot w = \dot W'$ (`fbpic/autodiff/ionization.py:690-764`). The second part is a dipole carried by the parent ion's Lagrangian tangent $\xi_i$. The existing code copies $\xi_i$ to the newborn, which is the Lagrangian form of the same dipole.
3. Future density parameters change marker weights directly [plan 2.1]. These are weight sources by construction.

### 2.3 The source is a phase-space dipole [derived]

Because $\nabla_u\cdot\delta\mathbf F_a = 0$ (Section 2.1, same proof),

$$
-\,\delta\mathbf F_a\cdot\nabla_u f_a = -\nabla_u\cdot\big(f_a\,\delta\mathbf F_a\big).
$$

Over one step $\Delta t$ this equals $-\nabla_u\cdot(f\,\xi_u)$ with $\xi_u = \delta\mathbf F\,\Delta t$. That is the $\delta f$ of a one-step Lagrangian momentum tangent, which `vay_tangent` computes from a zero input tangent. The Lagrangian form keeps $\xi$ on the marker and transports it with the linearized flow $\dot\xi = (\partial\Phi/\partial z)\,\xi + \delta\Phi$, $\Phi = (\mathbf v, \mathbf F)$. The factor $\partial\Phi/\partial z$ stretches $\xi$ at the Lyapunov rate (P7: about $0.019$ per step, $\times 10$ per 121 steps). The weight form converts the dipole into a scalar at once:

$$
\delta f = -f\,\xi_u\cdot\nabla_u\ln f \quad\Rightarrow\quad \Delta\hat W_p = -W_p\,\Delta t\;\delta\mathbf F_p\cdot \mathbf s_p, \qquad \mathbf s = \nabla_u \ln f .
$$

The scalar is transported unchanged along characteristics, so $\partial\Phi/\partial z$ never multiplies it. This is the only mechanism that removes the exponential growth.

Consequences:

1. $\mathbf s = \nabla_u\ln f$ is the score function of the local momentum distribution. The weight form is the likelihood-ratio estimator of Monte Carlo sensitivity analysis. The Lagrangian form is the pathwise estimator.
2. Chung's collocated scheme is a kernel-density score. With collocated markers of equal weight, $\hat n \approx f/W$, so $h_s = \sum (\mathcal H/\hat n)\,b\,b \approx W_s\,\delta\mathbf F\cdot\nabla_u\ln\tilde f$, where $\tilde f$ is $f$ smoothed by the momentum shapes. Collocated particle-pdf equals variant R-v with reset period $P = 1$, up to where $\delta\mathbf E$ is sampled (mesh node or marker).
3. In a chaotic flow the true $\nabla_u f$ also develops exponentially fine filaments. The mesh smoothing removes them. The regular sensitivity is therefore the sensitivity of the coarse-grained $f$. Its validity rests on the paper's assumption that the continuum response of the observable is stable; the paper states no criterion for it [paper, Section 2.3, page 10; plan Track R item 4].

### 2.4 The cold limit [derived, measured]

Claim. Take a spatial cell in which all markers of a species have the same velocity $\mathbf v_0$ (cold, single stream). Any weight change $\Delta\hat W_p$ that is charge neutral in the cell gives zero current change: $\delta\mathbf J = q\sum_p \Delta\hat W_p\,\mathbf v_0 = q\,\mathbf v_0\sum_p\Delta\hat W_p = 0$. The exact response is $\delta\mathbf J = q\,n\,\delta\mathbf v \neq 0$. The Lorentz source is charge neutral in the continuum (it is a $u$-divergence). So a correct weight scheme gives zero current response in a cold cell, and a non-neutral scheme creates spurious charge instead.

Test. `cold_limit_pdf.py` implements Eq. (36), (41), (42), (44) with $l = 1$ in 1D-1V for one step of a uniform $\delta E$ ($q = m = \Delta t = 1$, exact $\delta J = n = 1$, exact $\delta n = 0$), 200 000 markers, $\Delta v = 0.1$:

| Case | $\delta J/\delta J_{exact}$ | $\delta n$ | $\mathcal H$ mass outside $\hat n$ support |
|---|---|---|---|
| Cold, collocated, $v_0$ at a cell center | 0.0000 | $4\cdot10^{-13}$ | 0 |
| Cold, collocated, $v_0$ at $0.2\,\Delta v$ from a cell center | 0.0060 | +0.20 | 2% |
| Cold, collocated, $v_0$ on a cell edge | 0.0000 | -1.2 | 8% |
| Warm, collocated, $\sigma = 0.25\,\Delta v$ | 0.220 | $10^{-4}$ | 0 |
| Warm, $\sigma = 0.5\,\Delta v$ | 0.505 | $10^{-5}$ | 0 |
| Warm, $\sigma = \Delta v$ | 0.799 | $3\cdot10^{-6}$ | 0 |
| Warm, $\sigma = 2\,\Delta v$ | 0.940 | $10^{-6}$ | 0 |
| Warm, $\sigma = 4\,\Delta v$ | 0.983 | $10^{-5}$ | 0 |
| Cold PIC, separate sensitivity markers uniform in $\pm 3\Delta v$ | 1.0004 | $2\cdot10^{-13}$ | 0 |

[measured, `cold_limit_pdf.py`]. The warm rows follow $\sigma^2/(\sigma^2 + \Delta v^2/4)$ within 0.02; $\Delta v^2/4$ is the variance of the cell-averaged quadratic shape $b_2$. This is the smoothing bias of the kernel score.

The last row shows that separate sensitivity markers spread in $u$ recover the one-step response. They do not follow the cold fluid, however. Their spread $\Delta u$ separates them in $z$ by about $\Delta u\,c\,t$ (nonrelativistic part). To keep this below $c/\omega_p = 1.88\ \mu$m over the 2000 steps ($\omega_p T = 53$) needs $\Delta u \lesssim 0.02$ [estimated]. The laser quiver spans $u_\perp \in [-4, 4]$, so a mesh at that resolution needs about 400 cells per transverse $u$ axis [estimated]. This is not practical.

Twin-S facts. `electrons`, `helium`, and `nitrogen` are loaded with no thermal momentum (`inversion/twin_config.py:249-275`). `elec_from_N` markers are born with the parent ion momentum (`fbpic/autodiff/ionization.py:326-366`). Before trapping and phase mixing, the electrons in one cell lie near a curve in $u$ parametrized by the birth phase. The score is then undefined across the curve, and the cold-limit claim applies to that direction. After trapping, betatron and synchrotron phase mixing fill a volume in $u$; the trapped bunch is the population that the weight form can represent.

### 2.5 Moment (Gaussian) score [derived, measured]

A cheap alternative to the mesh replaces $\tilde f$ in each bin by a Gaussian with the bin's weighted mean $\bar{\mathbf u}$ and covariance $C$: $\mathbf s_p = -C^{-1}(\mathbf u_p - \bar{\mathbf u})$. Properties for a bin-uniform kick $\xi$:

1. $\sum_p\Delta\hat W_p = 0$ exactly: charge neutral per bin.
2. $\sum_p\Delta\hat W_p\,\mathbf u_p = \xi\sum_p W_p$ exactly, for any $f$.
3. $\sum_p\Delta\hat W_p\,\mathbf v(\mathbf u_p) = \xi\cdot\mathbb E[\partial\mathbf v/\partial\mathbf u]$ only for Gaussian $f$ (Stein's lemma).

`gauss_score.py` (1D, $v = u/\sqrt{1+u^2}$, 400 000 markers, uniform kick):

| Bin distribution | $\sum\Delta\hat W$ | $\delta\sum Wu$ rel. | $\delta J_v$ rel. | spurious $\delta\,\mathrm{var}(u)$ | rms $\Delta\hat W/W$ |
|---|---|---|---|---|---|
| Gaussian, $u_0 = 2$, $\sigma = 0.5$ | $10^{-15}$ | 1.0000 | 1.0025 | $-3\cdot10^{-3}$ | 2.0 |
| Bimodal $\pm 2$, $\sigma = 0.3$ | $10^{-18}$ | 1.0000 | 4.49 | $10^{-5}$ | 0.50 |
| Lognormal (skewed) | $10^{-16}$ | 1.0000 | 0.40 | +2.9 | 0.77 |
| Near cold, $u_0 = 4$, $\sigma = 10^{-3}$ | $5\cdot10^{-10}$ | 1.0000 | 1.0000 | $4\cdot10^{-6}$ | 1000 |

[measured, `gauss_score.py`]. The moment score is unusable in bins that mix streams (for example trapped bunch and untrapped electrons of the same species) or that are skewed. It amplifies weight noise as $1/\sigma$. A per-bin mixture model (for example a split by $u_z$) is a possible repair; I did not test it.

### 2.6 Quasi-cylindrical geometry [derived]

1. Markers are 3D Cartesian; deposit and gather project onto modes $m = 0, 1$ with $e^{+im\varphi}$ and $e^{-im\varphi}$ [plan 4.7]. A scalar weight needs no mode treatment: the production deposit with $w \to \hat W$ projects it.
2. $\delta f$ has all azimuthal harmonics, not only $m \le 1$. The mode truncation applies to fields only, as in the primal.
3. The score needs the local momentum distribution at the marker. Use bins in $(z, r, \varphi)$ with Cartesian $\mathbf u$, because the ponderomotive outflow makes the Cartesian $u$-distribution depend on $\varphi$. At least 4 sectors are needed to resolve $m = 1$; I suggest 8 [estimated].
4. $\nabla_u\ln f$ does not contain the spatial volume factor $2\pi r$, so the momentum score has no axis singularity. A spatial score $\nabla_x\ln f$ (needed only to project $\xi_x$, Section 6) does contain it and is noisy near $r = 0$.
5. Background markers per cell are $p_{nz}p_{nr}p_{nt} = 16$, so a $(z, r, \varphi)$ cell with 8 sectors holds 2 background markers. A 6D covariance needs at least about 30 markers per bin. Bins must therefore span several cells, or the score must be built only for the dense `elec_from_N` bunch [estimated].

### 2.7 Discrete step and staggering [derived]

The FBPIC step order is in `fbpic/autodiff/stepper.py:310-436`: gather at $t^n$ (`:348`), Vay push $u^{n-1/2} \to u^{n+1/2}$ and its tangent (`:364`), half position push (`:366`), EV and its tangent (`:377`), J deposit at $t^{n+1/2}$ (`:392`), second half push, `rho_next` deposit at $t^{n+1}$ (`:402`), current correction and PSATD (`:405-414`), window shift and damping (`:427`).

The weight update fits right after the Vay tangent: $\hat W^{n+1/2} = \hat W^{n-1/2} - \Delta t\,h^n$, with $h^n$ from $\delta\mathbf F$ gathered at $t^n$. This matches Fig. 4 of the paper (weights at half steps). The J deposit at $t^{n+1/2}$ then uses $\hat W^{n+1/2}$, which is centered. `rho_next` at $t^{n+1}$ also uses $\hat W^{n+1/2}$, so $\delta\rho$ lags by half a weight increment. Section 3 shows why this is harmless with the curl-free correction.

## 3. Charge conservation and the current correction

### 3.1 Continuum [derived]

Integrate the linearized Vlasov equation over $\mathbf u$ and sum over species:

$$
\partial_t\delta\rho + \nabla\cdot\delta\mathbf J = \sum_a q_a\!\int\!\delta G_a\,d^3u - \sum_a q_a\!\int\!\nabla_u\cdot(f_a\,\delta\mathbf F_a)\,d^3u = 0 .
$$

The second integral vanishes because it is a $u$-divergence. The first vanishes because each EV birth adds the same weight to the electron and to the ion charge `w_times_level` at the same point [plan 5.2]. In the code the electron birth tangent is $\dot W'$ and the ion charge tangent is $w\sum_\ell \ell\dot P_\ell - \dot W$ at the same position (`fbpic/autodiff/ionization.py:676`, `:690`). So the EV source is locally neutral in the discrete deposit too.

### 3.2 Discrete weights break continuity [derived]

With time-varying weights,

$$
\frac{\delta\rho^{n+1} - \delta\rho^n}{\Delta t} + \nabla\cdot\delta\mathbf J^{n+1/2} = \frac{q}{\Delta t}\sum_p\Delta\hat W_p\,S(\mathbf x - \mathbf x_p) + (\text{deposit truncation}).
$$

The first term is zero only if the weight increments are neutral at the deposit-node level. Chung's mesh score is not neutral when Eq. (43) fails (cold rows of Section 2.4: $\delta n = 0.2$ and $-1.2$). The moment score is neutral per bin but not per node, because $\Delta\hat W_p$ correlates with the position inside the bin when $\bar{\mathbf u}$ varies across the bin (laser quiver varies over 16 cells per wavelength).

### 3.3 What the curl-free correction does [code, derived]

Twin-S sets `current_correction="curl-free"` (`inversion/twin_config.py:236`). The production kernel (`fbpic/fields/cuda_methods.py:142-175`; functional twin `fbpic/autodiff/fields.py:195-220`) computes

$$
F = -\frac{1}{k^2}\Big(\frac{\hat\rho_{next} - \hat\rho_{prev}}{\Delta t} + k_z\,i\hat J_z + k_r(\hat J_+ - \hat J_-)\Big), \quad \hat J_\pm \mathrel{+}= \pm\tfrac12 k_r F, \quad \hat J_z \mathrel{-}= i k_z F,
$$

which replaces the longitudinal part of $\hat{\mathbf J}$ by the value that the change of $\hat\rho$ requires. PSATD then keeps $\nabla\cdot\mathbf E = \rho/\varepsilon_0$ for the deposited $\rho$.

The tangent applies the same operator to $(\delta\rho_{prev}, \delta\rho_{next}, \delta\mathbf J)$ (`stepper.py:409-414`). So:

1. No field code changes. The weight-created charge appears in $\delta\rho_{next} - \delta\rho_{prev}$, and the correction adds the matching longitudinal $\delta\mathbf J$. Gauss's law holds for the deposited $\delta\rho$, and $\delta\mathbf E$ is consistent with it.
2. The half-step lag of Section 2.7 is absorbed for the same reason: only $\delta\rho_{next} - \delta\rho_{prev}$ enters the longitudinal current.
3. The correction cannot tell a physical charge change from projection noise. Any non-neutral part of $\Delta\hat W$ becomes a real longitudinal $\delta\mathbf E$. The weight update must therefore be neutral at the node level. Options: (a) build the score with the deposit shapes as bin memberships; (b) a lumped node correction $\Delta\hat W_p \leftarrow \Delta\hat W_p - W_p\sum_i S_i(\mathbf x_p)\lambda_i$ with $\lambda_i = \sum_p\Delta\hat W_p S_i(\mathbf x_p)/\sum_p W_p S_i(\mathbf x_p)$, per mode $m = 0, 1$ with the phase factor; this is first-order neutral, not exact; (c) an exact solve with the marker mass matrix (sparse SPD, about $4\times10^4$ nodes per mode).
4. The $k = 0$ component of $F$ is set to zero (`fbpic/fields/spectral_grid.py:144`). A net weight-created charge at $k = 0$ cannot be absorbed. With the discrete Hankel transform $k_r > 0$, so this probably never applies (not checked).
5. The exchange-step `rho_prev` redeposit (`stepper.py:283`) uses the current weights. This resets $\delta\rho_{prev}$ to the deposited value and is consistent.
6. Cross-deposition is not used by Twin-S. It builds $\mathbf J$ from the $\rho$ deposits under constant weights, so it would need an explicit weight-change charge term.

## 4. Mapping to the code

### 4.1 Reused as-is [code]

| Operation | Function | Use in the weight form |
|---|---|---|
| Laser seed | `energy_seed` (`fbpic/autodiff/laser.py:92`), `seed_tangents` (`fbpic/autodiff/driver.py:92`) | Unchanged initial condition |
| Field operators: transforms, current correction, PSATD, filter, shift, damping | `FieldTangents.apply` (`fbpic/autodiff/field_tangents.py:109`), called at `stepper.py:414, 430` | Unchanged |
| Gather of $\delta\mathbf E, \delta\mathbf B$ at markers | `gather_tangent` (`fbpic/autodiff/tangent.py:494`) | With zero position tangents it returns the field tangent at the marker |
| Momentum kick $\xi_u = (\partial u^{n+1/2}/\partial F)\,\delta F$ | `vay_tangent` (`tangent.py:698`), `vay_ion_tangent` (`tangent.py:773`) | Gives the one-step kick to project; ions stay Lagrangian |
| Position tangent | `position_tangent` (`tangent.py:829`) | Lagrangian markers only; zero for weight markers |
| Deposit with $\xi$ and $\dot w$ | `deposit_tangent` (`tangent.py:1076`) | Lagrangian markers; it already takes the weight tangent `dw` (`tangent.py:994, 1014`) |
| Deposit of weights only | production `deposit_fused(particles, grids, fieldtype, weight)` (`fbpic/particles/deposition/cuda_methods_fused.py:471`) | For markers with $\xi = 0$: call with `weight` $= \hat W_k$ into the tangent grids; cost of a primal deposit, with the multi-lane option for dense cells |
| EV tangent and births | `ionize_tangent` (`fbpic/autodiff/ionization.py:690`) | Birth weight $\dot W'$ is already a weight source |
| Sort, exchange, removal, injection, realignment | `sort_all`, `_exchange`, `_realign_tangents` (`stepper.py:137, 223, 291`) | Weights travel with markers; injected markers get 0 |
| Observable contraction | `ResidualValue.contract` (`fbpic/autodiff/loss.py:264`) | Already uses the weight tangent `t["w"]` |

### 4.2 New kernels [derived]

1. N1, bin moments (shared by all $K$ tangents). Per bin and per weight-form species: $\sum W$, $\sum W\mathbf u$, $\sum W\mathbf u\mathbf u^T$ (10 scalars); for the 6D projection of Section 6 also $\mathbf x$ moments and cross terms (28 scalars). Sorted, one thread per cell, like the fused deposit.
2. N2, projection (per tangent). Per marker: read the bin moments, regularize $C + \sigma_m^2 I$, compute $\Delta\hat W_p = W_p\,\xi_p\cdot C^{-1}(\mathbf z_p - \bar{\mathbf z})$ (Section 6), add it to $\hat W$, and zero $\xi$ and the `inv_gamma` tangent.
3. N3, neutrality correction (per tangent, Section 3.3 item 3b): one weight deposit and one gather.
4. N4, marker classification: a per-marker flag (Lagrangian or weight), set from the bin conditioning and the local tangent growth.
5. Only for the full Chung scheme: N5 6D key and sort, N6 sparse $\hat n$ deposit, N7 sparse $\mathcal H$ deposit with $\nabla_u b_3$, N8 interpolation of $\mathcal H/\hat n$ to markers.

### 4.3 Cost [measured by P7, estimated here]

Reference numbers from plan Section 12.10, step 3 and Section 8: primal run 86 s (43 ms per step on average); $K = 1$ gradient 298 s (3.47 run-equivalents); at step 1200 with 9.8M `elec_from_N`, the tangent step takes 228 ms, of which the J and `rho_next` deposit tangents take 81 and 40 ms; the primal J and `rho_next` deposits take 24 and 14 ms.

| Kernel | Frequency | Estimated cost at 10M markers |
|---|---|---|
| N1 bin moments (10 or 28 scalars) | every step (P = 1) or every $P$ steps (R-v) | 1 to 3 primal J deposits (24 to 72 ms) |
| N2 projection | same | about 1 gather (10 to 20 ms) per tangent |
| N3 neutrality correction | same | about 1 rho deposit + 1 gather (25 to 35 ms) per tangent |
| Weight-only deposit instead of `deposit_tangent` for weight markers | every step | saves up to 121 - 38 = 83 ms per step per tangent |
| Full Chung N5 to N8 | every step | $\hat n$: 8 spatial x 27 momentum = 216 updates per marker; $\mathcal H$: 8 x 64 = 512; interpolation: 216 reads. About $10^{10}$ scattered updates per step: 0.7 to 1.5 s per step, plus a sparse mesh of up to $2\cdot10^9$ entries (up to 32 GB) |

Totals for $K = 1$ [estimated]:

1. Collocated particle-pdf with a moment score at $P = 1$: about 230 to 280 ms per late step, so about 3.5 to 4.5 run-equivalents per gradient. Each further parameter adds about 1.3 to 1.8 run-equivalents (N1 is shared).
2. Hybrid R-v with $P = 50$ to 100: N1 to N3 cost below 2 ms per step when amortized. The gradient stays at about 3.5 run-equivalents, or about 2.5 if the weight markers use the cheap deposit.
3. Full Chung 6D mesh: 20 to 40 run-equivalents per gradient, and the memory may not fit. This is more than central FD (2 runs per parameter) up to $K \approx 10$ to 20.

The cost of the full mesh is the decisive negative for the literal paper algorithm in 3V. The paper's own cost figure (1.9 to 3 PIC runs) is for 1D-1V, where the momentum stencil has 3 or 4 cells instead of 27 or 64 [paper, plan Track R item 5].

## 5. Boundaries

1. Moving-window injection at the front. Injected plasma is cold and $\theta$-independent, ahead of the laser. So $\delta f = 0$ on the inflow boundary: injected markers get $\hat W = 0$ and $\xi = 0$. This is what `_realign_tangents` does now (`stepper.py:291-307`) [code, derived].
2. Removal at the back. $\delta f$ leaves with its markers; no boundary term is needed in the weight form. The removal count is a $\theta$-dependent decision, but the observable weight $\sigma_i$ is zero at the window back [plan 2.3], so it has no effect on $J$. The field tangents in the damping region are handled by the existing damping operator [derived].
3. Window shift of fields: `shift_spect_grid` on the tangent fields (`stepper.py:427`), unchanged [code].
4. Axis $r = 0$. Weight markers carry no position tangent, so the $1/r$ and $1/r^2$ terms of the gather and deposit tangents [plan 4.7] do not occur for them. The deposit of $\hat W$ uses the production Ruyten-corrected shape and axis branch. The momentum score has no axis factor (Section 2.6). Bins at small $r$ hold few markers, so the score there is noisy. Merge the first radial bins or fall back to Lagrangian form there [derived].
5. Outer radius. Twin-S has reflective particle boundaries at $r_{max} = p_{rmax} = 20\ \mu$m, four laser waists out (`inversion/twin_config.py:233`). A reflected marker keeps its weight; its $\xi$ (if Lagrangian) needs the reflection map. The existing tangent code freezes this decision [code].
6. EV births. The newborn weight $\dot W'$ is regular if $\delta\mathbf E$ at the ion and the ion tangents are regular. Ions are heavy (N, $m/q = 14 m_p/5e$) and cold; their Lagrangian tangents should stay bounded, but this is not measured per species. The newborn also inherits the parent $\xi$ (Section 2.2 item 2). If the newborn is in weight form, this dipole must be projected with the score of the birth source $G$, not of the electron population; $G$ sits at $u \approx u_{ion}$ and is cold. Two options: keep newborns Lagrangian until the next reset (recommended), or drop the ion-displacement part (error of the order of the ion tangent, not measured) [derived].
7. The emission threshold $\epsilon$ remains a frozen decision in both forms [plan 5.3]. P7 step 6 found that $\epsilon \to 0$ does not change the chaotic picture [plan 12.10], so this is not the limiting issue.

## 6. Variant R-v seen from the Vlasov-Maxwell side

Projection at a reset [derived]. For a marker set with Lagrangian tangents $\xi_p = (\xi_{x,p}, \xi_{u,p})$, the represented sensitivity is $\delta f = -\nabla_z\cdot\big(\sum_p W_p\,\xi_p\,S(\mathbf z - \mathbf z_p)\big)$, $\mathbf z = (\mathbf x, \mathbf u)$. Per bin, write the bin-mean tangent $\bar\xi$ and the 6D covariance $C_6$ of $\mathbf z$. First-moment matching asks for $\sum_p\Delta\hat W_p = 0$ and $\sum_p\Delta\hat W_p(\mathbf z_p - \bar{\mathbf z}) = \sum_p W_p\,\xi_p$. The minimum-norm solution of the form $W_p\,\beta\cdot(\mathbf z_p - \bar{\mathbf z})$ is

$$
\Delta\hat W_p = W_p\,(\mathbf z_p - \bar{\mathbf z})^T C_6^{-1}\,\bar\xi .
$$

This is the moment score of Section 2.5 in 6D. It keeps the bin charge and the bin dipole moments ($\delta\rho$ and $\delta\langle\mathbf u\rangle$ at bin scale). It drops the intra-bin covariance $\mathrm{Cov}(\xi, \mathbf z)$. That covariance is the stretched, chaotic part of the tangent, so dropping it is the regularization. It also carries the regular sensitivity of the bin's spread (energy spread, emittance), which enters residuals 3 and 4 of the loss [plan 2.4]. Matching second moments would keep it but needs fourth moments of $f$ per bin. I did not derive the conditioning of that system.

Relation to the paper. With $P = 1$, $\xi = (0, \delta\mathbf F\,\Delta t)$ at the projection, $\bar\xi$ is nearly uniform in the bin, and the formula reduces to the collocated particle-pdf update with a moment score. With $P > 1$, R-v projects only every $P$ steps, so the cost of N1 to N3 is amortized (Section 4.3). The growth of $\xi$ between resets is at most $10^{P/121}$: about 2.6 for $P = 50$ and 6.7 for $P = 100$ [derived from the P7 rate].

Hybrid rule. At each reset, project only markers in bins where (a) the bin holds enough markers (about 30 or more) and (b) $C_6$ is well conditioned relative to the shape variance, for example $\lambda_{min}(C_u) \ge 4\sigma_m^2$, which keeps the bias of Section 2.4 below 6%. Keep all other markers Lagrangian. Cold, laminar background markers then keep their exact Lagrangian tangents, and the warm trapped bunch moves to weight form. Ions stay Lagrangian.

The hybrid fails if chaotic markers sit in cold or multi-stream bins. Examples: background electrons that cross the bubble sheath, or a trapped bunch that is still thin in $u$ when its tangents start to grow. P7 gives only the maximum over all species of the tangent norms (`fbpic/autodiff/driver.py:118-136`), so this is not known yet.

## 7. Feasibility, kill criteria, and open questions

### 7.1 Verdict for this topic: conditional

1. Literal particle-pdf for all Twin-S species: infeasible. The cold background and cold newborn manifolds cannot be represented by weights (Section 2.4), and the full 6D mesh is too expensive (Section 4.3).
2. Hybrid R-v with a moment projection on warm bins: possible, at about the present gradient cost. Its accuracy is unknown. It depends on where the chaos lives and on the warmth of that population.

### 7.2 Measurements that decide it (cheap, before R1 and R2)

1. Per-species tangent growth on Twin-S to step 2000 (one $K = 1$ run, about 5 minutes on GPU): extend `tangent_norms` to report each species and quantiles, not only the maximum.
2. Warmth and modality of the markers that carry the growth: at steps 1200, 1500, and 2000, dump the bins of the top 1% tangent-norm markers. Report $\lambda_{min}(C_u)^{1/2}$, the marker count per bin, and a bimodality coefficient.
3. A projection test on the real state: at step 1000 (tangents still accurate), project $\xi$ of the warm bins with the 6D moment formula. Compare $\delta\rho$, $\delta\mathbf J$, and the 13 loss base-sum tangents before and after the projection.

### 7.3 Kill criteria (measurable)

1. Background chaos: if the `electrons` species (cold background) shows tangent growth of at least $\times 10$ per 200 steps after step 1200 in the 90th-percentile norm, the hybrid cannot keep it Lagrangian, and the weight form cannot represent it. No-go.
2. Cold chaotic carriers: if more than 20% of the tangent growth (by contribution to the 13 base-sum tangents at step 1500) sits in bins with $\lambda_{min}(C_u)^{1/2} < 2\sigma_m$ or fewer than 30 markers, the projection bias exceeds about 20% there. No-go for the hybrid as specified.
3. Projection fidelity at step 1000: if the 6D moment projection changes any of the 13 base-sum tangents by more than 10%, or changes $\lVert\delta\mathbf J\rVert$ by more than 10%, the projection is not good enough to repeat every 50 to 100 steps.
4. Spurious charge: if after the node-neutrality correction the longitudinal part of $\delta\mathbf E$ caused by weight-created charge exceeds 5% of $\lVert\delta\mathbf E\rVert$ over 200 steps, the method injects noise faster than it removes chaos.
5. Cost: if the marginal cost per parameter exceeds 2 run-equivalents (the cost of central FD), the method loses to FD at every $K$.
6. Final usability: R4 of the plan (fixed-point shift $\delta \le 0.005$ against the smoothed column at $N_t = 2000$).

### 7.4 Open questions

1. Where does the chaos live (species, trapped or untrapped, sheath)? Measurement 7.2 item 1 answers it.
2. Is the trapped bunch warm in all 3 momentum directions at the bin scale? The longitudinal spread may be small early after trapping.
3. Bin design: a $(z, r, \varphi)$ bin needs about 30 markers for a stable 6D covariance. The background has 2 markers per 8-sector cell. Bins must be coarser than the deposit cells, which makes node-level neutrality harder (Section 3.3).
4. Spread sensitivities: the first-moment projection drops $\mathrm{Cov}(\xi, \mathbf z)$ at each reset. Is the sensitivity of the energy spread and the emittance recovered later by weight transport, or lost? A 1D test (R2) can answer it.
5. Newborn dipole: is the ion-displacement part of $\delta G$ negligible? Compare the deposited $\delta\rho$ of births with and without the copied parent $\xi$ on Twin-XS.
6. Statistical error: the weight form has an error of order $N^{-1/2}$ amplified by $1/\sigma_u$ (Section 2.5). Is it below the model-error whitening $\Delta$ of the residuals [plan 2.4]? R2 and R4 measure it.
7. The paper gives no stability criterion for the continuum response. The only evidence here is that FD secants at $h = 10^{-2}$ and $10^{-3}$ agree at $N_t = 2000$ [plan 12.10].

## 8. Files

1. `inversion/notes/r0-evidence-2026-10-31/work_vm/cold_limit_pdf.py`: Chung Eq. (36), (41), (42), (44) in 1D-1V, cold and warm one-step response.
2. `inversion/notes/r0-evidence-2026-10-31/work_vm/gauss_score.py`: moment-score projection, charge, current, and variance response.

Both run on CPU with `python3 -I <script>` in under 2 minutes.
