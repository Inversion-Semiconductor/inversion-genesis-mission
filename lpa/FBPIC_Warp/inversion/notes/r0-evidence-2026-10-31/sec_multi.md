# R0 section: Multi-parameter scaling and an adjoint particle-pdf method

Author: R0 derivation agent `multi`. Date: 2026-10-31. Status: study note, not reviewed.

Evidence labels: **[paper]** Chung, Bond, Cyr, Freund, JCP 400, 108969 (2020), OSTI 1570259, cited by section, equation, and PDF page. **[derived]** derived in this note. **[measured]** measured by this agent with the script named. **[plan]** taken from `lpa/FBPIC_Warp/DIFFERENTIABLE_INVERSION_PLAN.md`. **[code]** read in the worktree at `path:line`. **[estimated]** an estimate with stated inputs. **[lit]** other literature, listed in Section 9.

## 0. Answer in brief

1. Forward scaling. In the particle-pdf method, all $K$ parameters share the markers, the phase-space mesh of $\nabla_u f$, and its interpolation to the markers. Per parameter, only one scalar weight per marker, one field-sensitivity gather, one weight deposit, and one field solve remain. The estimated marginal cost is $c_1 \approx 0.4$ to $0.8$ primal steps per parameter on Twin-S, against a measured $2.47$ for the Lagrangian tangent with EV (Section 2). The fixed cost $c_0$ of the phase-space mesh is the largest unknown (estimated 1 to 10 primal steps). **[estimated]**
2. Adjoint. The continuous adjoint of the linearized Vlasov-Maxwell system is a scalar $\lambda(x,u,t)$ that is constant along the same characteristics, plus adjoint fields $(\alpha,\beta)$ that obey Maxwell's equations backward in time. The coupling can be written with $\lambda\,\nabla_u f$ or, after integration by parts, with $f\,\nabla_u\lambda$. The particle-pdf adjoint must use the first form. That form is the exact discrete transpose of the forward particle-pdf scheme. **[derived]**
3. Lyapunov growth. The transposed particle-pdf adjoint has the same propagator norm as the forward particle-pdf scheme, so it is regular exactly when the forward scheme is regular. It never forms $\nabla\lambda$. In contrast, the adjoint of the particle-exact (Lagrangian) tangent carries $\nabla\lambda$ on the markers and grows like the forward tangent. Reverse-mode AD of the PIC step (Phase P11) therefore does not solve the chaos problem. A 1D-1V driven-trapping test confirms the transpose to $10^{-15}$ and shows bounded adjoint values while the Lagrangian tangent grows by $10^{9}$ (Section 5.2). **[derived, measured]**
4. Warning from the same test. After trapping, the collocated particle-pdf result (and therefore its adjoint) is regular but biased: about +24 on a reference of 5 to 41, with a velocity-mesh dependence and no visible convergence at $5\cdot10^4$ markers. Before trapping it agrees within 1% to 7%. The prototype is crude (Section 5.3), but this accuracy question, not the cost of $K$, is the first risk of Track R. **[measured]**
5. Jacobian count. Dogbox needs the $11 \times K$ residual Jacobian. Forward mode needs $K$ weight sets. Adjoint mode needs 11 adjoint sets (the rank of the $11 \times 13$ map from the 13 particle sums). The estimated crossover is $K^\ast \approx 13$ to $18$. At $K = 25$ the adjoint saves about 1.5 to 2 times, at the price of checkpointing (about one extra primal replay) and a large engineering effort. **[estimated]**
6. Recommendation. First, R1 must show that particle-pdf converges to a smoothed brute-force reference after trapping (Section 5.3). If it does, build the forward, $K$-batched, collocated or marker-augmented particle-pdf first (R3), and write it as an explicit linear recurrence in the weights so that its transpose exists. Add the adjoint only when $K$ exceeds about 15, or when the per-parameter cost $c_1$ turns out larger than estimated. R3 must measure $c_0$ and $c_1$ early. Kill criteria are in Section 7.3.

## 1. Model, notation, and the shared structure

Let $f_s(\mathbf{x},\mathbf{u},t)$ be the distribution of species $s$, with normalized momentum $\mathbf{u} = \mathbf{p}/(m_s c)$, $\gamma = \sqrt{1+u^2}$, and $\mathbf{v} = c\,\mathbf{u}/\gamma$. Define the momentum-space force $\mathbf{F}_s = (q_s/(m_s c))(\mathbf{E} + \mathbf{v}\times\mathbf{B})$. The Vlasov-Maxwell system is

$$
\partial_t f_s + \mathbf{v}\cdot\nabla_x f_s + \mathbf{F}_s\cdot\nabla_u f_s = S_s, \qquad
\partial_t \mathbf{B} = -\nabla\times\mathbf{E}, \qquad
\partial_t \mathbf{E} = c^2\nabla\times\mathbf{B} - \frac{1}{\varepsilon_0}\sum_s q_s\!\int \mathbf{v} f_s\, d^3u .
$$

$S_s$ is the EV birth source of `elec_from_N` (plan Section 5.1, with $\epsilon = 0$ in the continuum).

Identity 1 **[derived]**. $\nabla_u\cdot\mathbf{F}_s = 0$. The electric part does not depend on $\mathbf{u}$. For the magnetic part, $\partial_{u_i}(\mathbf{v}\times\mathbf{B})_i = \epsilon_{ijk}\,(\partial v_j/\partial u_i)\,B_k$, and $\partial v_j/\partial u_i = (c/\gamma)(\delta_{ij} - u_iu_j/\gamma^2)$ is symmetric in $(i,j)$, so the contraction with $\epsilon_{ijk}$ is zero. With $\nabla_x\cdot\mathbf{v} = 0$, the phase-space flow is incompressible. This confirms the hint: the sensitivity source $-(q/(mc))(\mathbf{e} + \mathbf{v}\times\mathbf{b})\cdot\nabla_u f$ equals $-\nabla_u\cdot\big(f\,(q/(mc))(\mathbf{e} + \mathbf{v}\times\mathbf{b})\big)$, a phase-space dipole.

The tangent system for parameter $\theta_j$, with $g_j = \partial_{\theta_j} f$, $\mathbf{e}_j = \partial_{\theta_j}\mathbf{E}$, and $\mathbf{b}_j = \partial_{\theta_j}\mathbf{B}$, is the relativistic electromagnetic form of [paper] Eq. (23a) to (25), p12:

$$
\partial_t g_j + \mathbf{v}\cdot\nabla_x g_j + \mathbf{F}\cdot\nabla_u g_j = -\frac{q}{mc}(\mathbf{e}_j + \mathbf{v}\times\mathbf{b}_j)\cdot\nabla_u f + \partial_{\theta_j} S, \tag{T1}
$$
$$
\partial_t \mathbf{e}_j = c^2\nabla\times\mathbf{b}_j - \frac{1}{\varepsilon_0}\sum_s q_s\!\int\mathbf{v}\,g_{j,s}\,d^3u, \qquad \partial_t\mathbf{b}_j = -\nabla\times\mathbf{e}_j, \tag{T2}
$$

with $g_j(0) = 0$ (plasma independent of $\theta$ under D21) and $(\mathbf{e}_j,\mathbf{b}_j)(0) = $ the laser seed $t_j$ of plan Section 4.8.

What the $K$ parameters share **[derived]**. The left side of (T1) is the same for every $j$: one set of characteristics carries all $K$ weight sets ([paper] Section 3.2.1, p11: "Extension to an array of parameters is straightforward"). The source factors as $\mathbf{e}_j(\mathbf{x})\cdot\mathbf{D}(\mathbf{x},\mathbf{u}) + \mathbf{b}_j(\mathbf{x})\cdot\mathbf{D}'(\mathbf{x},\mathbf{u})$, with $\mathbf{D} = (q/(mc))\nabla_u f$ and $\mathbf{D}' = (q/(mc))\,\nabla_u f\times\mathbf{v}$. $\mathbf{D}$ and $\mathbf{D}'$ do not depend on $j$. In the discrete scheme, the expensive phase-space objects (the mesh of $\nabla_u f$ in [paper] Eq. (52), p19, the marker density $\hat n$ in Eq. (53a), p20, and the per-marker interpolants of $\mathbf{D}/\hat n$) are computed once per step. Only the mesh fields $\mathbf{e}_j$ and $\mathbf{b}_j$ differ per parameter.

## 2. Forward scaling (task item 1)

### 2.1 Per-step work, split into shared and per-parameter parts

Collocated scheme ([paper] Section 3.2.4, p17, Eq. (44) and (54a), p18 and p20): the sensitivity markers are the plasma markers, and only the weights evolve, $d\hat W_{s,j}/dt = -h_{s,j}$.

| Work item | Shared or per parameter | Existing code that covers it |
|---|---|---|
| Primal step (push, sort, EV, deposits, fields) | shared | `fbpic/autodiff/stepper.py:310-439` |
| Phase-space mesh of $\nabla_u f$, $\hat n$, and per-marker $\mathbf{D}/\hat n$ | shared ($c_0$) | none; new kernel |
| `spect2interp` of $\mathbf{e}_j, \mathbf{b}_j$; gather at markers | per parameter | `gather_tangent` with zero position tangents (`fbpic/autodiff/tangent.py:260-542`) already computes $\sum_c S_c\,\dot F_{m,c}$; it is $K$-batched |
| $h_{s,j} = \mathbf{e}_j\cdot\mathbf{D}_s + \mathbf{b}_j\cdot\mathbf{D}'_s$; weight update | per parameter (about 12 FMAs) | none; trivial kernel |
| Deposit of $J$ and $\rho$ with weights $\hat W_{s,j}$ | per parameter | the primal deposit with $w$ replaced; or `deposit_tangent` with $\dot x = \dot u = 0$ and $\dot w = \hat W$ (`tangent.py:873-1113`, weight term at `:994`, `:1014`) |
| Current correction, PSATD push, window shift, damping of $\mathbf{e}_j,\mathbf{b}_j$ | per parameter | the existing field tangents (plan Section 4.6; `stepper.py:401-436`) |

The per-parameter state is one FP64 weight per marker (8 B) plus one field-tangent set. The Lagrangian tangent stores 112 B per marker per parameter (plan Section 8). **[plan, code, derived]**

### 2.2 Measured inputs

- Primal Twin-S EV run: 85.9 s and 86.0 s for 2000 steps, 43 ms per step on average (plan Section 12.10, step 3). **[plan, measured by P7]**
- $K = 1$ Lagrangian gradient: 299 s, 3.47 run-equivalents. The tangent adds 2.47 primal per parameter. Without EV, Phase P5 measured 0.96 to 1.2 (`inversion/results/p5_tangent_cost.json`: 9.2 ms against 18.0 ms per step on Twin-S). **[plan]**
- Profile at step 1200 (9.8 M `elec_from_N`): 228 ms per tangent step. The J deposit tangent takes 81 ms and the `rho_next` deposit tangent 40 ms; the primal deposits take 24 ms and 14 ms (plan Section 12.10). **[plan]**
- The task statement gives "1.9 to 3x PIC" for particle-pdf: [paper] Section 7.1, p35, measured 2.96 (non-collocated), 2.10 (collocated), and 1.89 (particle addition) times PIC for one parameter in 1D-1V. Table 1 (p37) gives the source-evaluation and number-density stages as $O[(C_2 l d + l^{2d})N]$ and $O[(C_3 l + l^{2d}) d N]$. With $d = 3$ these terms grow as $l^{6}$, so the 1D-1V ratio does not carry over to 3D-3V. **[paper, derived]**

### 2.3 Cost model

Let $P$ be one primal step. The per-parameter marginal cost $c_1$ is the gather of 6 field components for $N_m$ modes, one $J$ and $\rho$ deposit, the field-tangent solve, and the weight update. On Twin-S at the step-1200 peak, the primal deposits take 38 ms of a primal step of roughly 60 ms. A weight-only deposit, unbatched, therefore costs about 0.6 P. Gather and field operations add about 0.1 to 0.2 P. A $K$-batched deposit kernel shares particle loads, shape factors, and phase factors, so its marginal cost is the accumulators and atomics only. My estimate is $c_1 \approx 0.4$ to $0.8$ P, central 0.6 P. **[estimated]**

GPU detail 1 **[code, derived]**. The existing deposit tangent keeps a per-thread matrix of $4\cdot(\text{components})\cdot K \times 2$ FP64 values (`tangent.py:888`). For $J$ with $N_m = 2$ that is $48K$ doubles. At $K = 8$ it is 384 doubles per thread, far above the 255-register limit, so the kernel spills to local memory. A $K$-batched weight deposit must tile over $K$ (for example, groups of 2 to 4 per launch) or move the accumulators to shared memory. $c_1$ is therefore not constant in $K$.

GPU detail 2 **[estimated]**. At production size the field part of $c_1$ is no longer small. One complex-by-real DHT of a $2592\times1080$ array over 5 modes costs $4 N_r^2 N_z N_m = 6.0\cdot10^{10}$ flops. A step needs about 10 such transforms per field set. The RTX 6000 Ada has a 1:64 FP64 rate (about 1.4 TFLOP/s), so FP64 transforms take about 0.43 s per step per parameter. FBPIC has an FP32 Hankel path (`fbpic/fields/spectral_transform/hankel.py:216-220`), but plan Section 3.2 requires FP64 Hankel for differentiable runs. At production, $c_1$ is then of order 0.5 to 1 P from the field solves alone (`work_multi/cost_model.py`).

Fixed cost $c_0$ **[estimated]**. The phase-space mesh deposit with linear $x$ shapes and quadratic-derivative $u$ shapes touches $2^3\cdot3^3 = 216$ mesh points per marker and 3 gradient components, against about 48 accumulations per marker in the primal $J$ deposit. If both are atomic-bound, the mesh deposit costs about 10 primal deposits, so $c_0$ could reach 5 to 10 P without a sparse or per-cell (sorted) design. A per-cell kernel density estimate of $\nabla_u f$ at the markers, which avoids a global 6D mesh, could bring $c_0$ to 1 to 2 P. The mesh design belongs to another R0 section; here $c_0$ is a parameter.

Cost per Jacobian evaluation, in primal runs:

$$
C_F(K) = (1 + c_0) + K\,c_1 .
$$

| $K$ | Particle-pdf, $c_0 = 2$, $c_1 = 0.6$ | Particle-pdf, $c_0 = 8$, $c_1 = 0.8$ | Lagrangian tangent, $c_1 = 2.47$ (measured) | Lagrangian, faster deposit ($c_1 = 1.2$) | One-sided FD ($K+1$) | Central FD ($2K$) |
|---|---|---|---|---|---|---|
| 1 | 3.6 | 9.8 | 3.5 | 2.2 | 2 | 2 |
| 2 | 4.2 | 10.6 | 5.9 | 3.4 | 3 | 4 |
| 5 | 6.0 | 13.0 | 13.4 | 7.0 | 6 | 10 |
| 10 | 9.0 | 17.0 | 25.7 | 13.0 | 11 | 20 |
| 25 | 18.0 | 29.0 | 62.8 | 31.0 | 26 | 50 |

**[estimated, from `work_multi/cost_model.py`]**

Reading of the table. Per extra parameter, particle-pdf weights are 3 to 6 times cheaper than the Lagrangian tangent. The Lagrangian tangent is unusable after trapping anyway (plan Section 12.10). The relevant competitor is FD: FD secants at $h = 10^{-2}$ and $10^{-3}$ agree on Twin-S at $N_t = 2000$. Particle-pdf beats one-sided FD only when $c_0 < K(1 - c_1)$, that is, for $K \gtrsim 5$ if $c_0 \approx 2$, and only for $K \gtrsim 40$ if $c_0 \approx 8$. FD runs are also independent and run in parallel on two GPUs. Hence the value of Track R for multi-parameter work depends first on $c_0$, and second on whether the particle-pdf noise is lower than the FD noise at a usable $h$.

Memory, forward **[estimated]**. Twin-S: 0.08 GB of weights and 0.013 GB of field tangents per parameter at 10 M markers; $K = 25$ needs about 2.3 GB. Production: 0.48 GB of weights (60 M markers) and about 3.8 GB of field tangents per parameter (spectral 1.57 GB, interp 1.34 GB, $J$ and $\rho$ 0.90 GB, at 44.8 MB per complex array). With a primal of about 10 to 15 GB, one 48 GB GPU holds about 7 parameters. $K = 25$ needs 4 passes on one GPU, each repeating the primal and $c_0$, or 2 passes on the two GPUs. The memory limit at production comes from the field tangents, not from the particle weights.

## 3. Adjoint of the linearized Vlasov-Maxwell system (task item 2)

### 3.1 Continuous adjoint

Take one linear functional of the final electron distribution, $G = \int \phi(\mathbf{x},\mathbf{u})\, g(\mathbf{x},\mathbf{u},T)\, dz$, with $dz = d^3x\,d^3u$. Use multipliers $\lambda_s$ for (T1), $\varepsilon_0\boldsymbol{\alpha}$ for the $\mathbf{e}$ equation, and $\boldsymbol{\beta}/\mu_0$ for the $\mathbf{b}$ equation. Integrate by parts in $t$, $\mathbf{x}$, and $\mathbf{u}$, and use Identity 1. Collecting the coefficients of $g$, $\mathbf{e}$, and $\mathbf{b}$ gives **[derived]**:

Adjoint transport, backward from $T$ (one equation per species, same characteristics as the primal):
$$
\partial_t\lambda_s + \mathbf{v}\cdot\nabla_x\lambda_s + \mathbf{F}_s\cdot\nabla_u\lambda_s = q_s\,\mathbf{v}\cdot\boldsymbol{\alpha}(\mathbf{x},t), \qquad \lambda_{e}(T) = \phi, \quad \lambda_{s\neq e}(T) = 0. \tag{A1}
$$
Along a characteristic, $\lambda_s(t) = \lambda_s(T) - \int_t^T q_s\,\mathbf{v}\cdot\boldsymbol{\alpha}\,dt'$. The value is transported; it is not stretched.

Adjoint Maxwell, backward from $\boldsymbol{\alpha}(T) = \boldsymbol{\beta}(T) = 0$:
$$
\partial_t\boldsymbol{\alpha} = c^2\nabla\times\boldsymbol{\beta} + \sum_s\frac{q_s}{\varepsilon_0 m_s c}\int\lambda_s\,\nabla_u f_s\,d^3u, \qquad
\partial_t\boldsymbol{\beta} = -\nabla\times\boldsymbol{\alpha} + \sum_s\frac{\mu_0 q_s}{m_s c}\int\lambda_s\,(\nabla_u f_s\times\mathbf{v})\,d^3u. \tag{A2}
$$
With the energy-weighted inner product $\int(\varepsilon_0\boldsymbol{\alpha}\cdot\mathbf{e} + \boldsymbol{\beta}\cdot\mathbf{b}/\mu_0)\,dx$, the Maxwell operator is skew-adjoint, so the adjoint fields obey the same Maxwell equations, run backward in time, with sources.

Coupling, two exact continuum forms **[derived]**:
$$
\int\lambda\,\nabla_u f\,d^3u = -\int f\,\nabla_u\lambda\,d^3u, \qquad \int\lambda\,(\nabla_u f\times\mathbf{v})\,d^3u = -\int f\,(\nabla_u\lambda\times\mathbf{v})\,d^3u .
$$
The second identity uses $\nabla_u\times\mathbf{v} = 0$ ($\mathbf{v} = c\nabla_u\gamma$). The left forms need $\nabla_u f$ (the primal) and values of $\lambda$. The right forms need $\nabla_u\lambda$ on the support of $f$. Section 4 shows that this choice decides whether the adjoint is regular.

Gradient **[derived]**:
$$
\frac{dG}{d\theta_j} = \underbrace{\varepsilon_0\!\int\boldsymbol{\alpha}(0)\cdot\mathbf{e}_j(0)\,dx + \frac{1}{\mu_0}\!\int\boldsymbol{\beta}(0)\cdot\mathbf{b}_j(0)\,dx}_{\text{laser seed}} + \underbrace{\sum_s\int\lambda_s(0)\,g_{j,s}(0)\,dz}_{\text{loaded plasma}} + \underbrace{\int_0^T\!\!\sum_s\int\lambda_s\,\partial_{\theta_j}S_s\,dz\,dt}_{\text{explicit sources}} .
$$

Literature check **[lit]**. Bartsch, Knopf, Scheurer, and Weber (arXiv:2304.02083) derive the same structure for magnetically controlled Vlasov-Poisson: a backward transport of $\lambda$ along the characteristics (their Eq. (31a), Section 3.7) with a reaction term that contains $\lambda\,\partial_v f$, and a gradient (their Eq. (33)) that contains $\partial_v f\,\lambda$. Their particle discretization (Section 4, Eq. (46)) evaluates $\partial_v f$ on the grid by finite differences of the forward solution and does not integrate by parts onto $\partial_v\lambda$. They do not discuss chaos. Albi, Dimarco, Ferrarese, and Pareschi (arXiv:2610.26110) project a similar optimality system onto particles. Neither treats Maxwell, relativistic momenta, or chaotic sensitivity.

### 3.2 Terminal condition from the 11-residual loss

The 11 whitened residuals depend on the particles only through 13 sums $S_a = \sum_i w_i\,\phi_a(z_i)$, with $\phi_a \in \{\sigma, \sigma\xi_k, \sigma\xi_k^2\}$ and $\sigma$ the quintic smoothstep in $u_z$ (plan Section 2.3; Section 12.10 "Design"; `fbpic/autodiff/loss.py:162-215`, `:284-352`). Let $C = L^{-1}\partial v/\partial S$ be the $11\times13$ host matrix. Then:

1. Gauss-Newton rows: $\lambda_i(T) = \sum_a C_{ia}\,\phi_a$ for $i = 1..11$. Row $i$ of $J_r$ is $dr_i/d\theta$. At most $\operatorname{rank} C \le 11$ adjoint sets are needed.
2. Gradient only: $\lambda_J(T) = 2\sum_i r_i\,\lambda_i(T)$, one adjoint set.

$C$ depends on the final residuals, so the adjoint starts after the forward pass, as usual. Each $\phi_a$ is smooth: a polynomial times a smoothstep of width $4\delta_u$. On collocated markers, $\lambda_i(T)$ is evaluated pointwise at the markers, without a mesh.

### 3.3 Source terms of the gradient for this problem

1. Laser parameters ($E_L$, $z_f$, and later duration, spot size, Zernike terms). The direct method sets only the initial fields (plan Section 4.8), so $\partial_\theta S = 0$ after $t = 0$ and $g_j(0) = 0$. The gradient is $\langle\hat{\boldsymbol\alpha}(0), \mathcal{T}\mathcal{D}\,t_j\rangle$, where $\hat{\boldsymbol\alpha}(0)$ is the spectral adjoint field after the transposed `begin`, and the bracket is the paired-real inner product of `contract` (`fbpic/autodiff/driver.py:201-222`). Because $t_j = \mathcal{L}[\partial\mathcal{E}/\partial\theta_j]$ with $\mathcal{L}$ linear, define $\Lambda = \mathcal{L}^T\mathcal{D}^T\mathcal{T}^T\hat{\boldsymbol\alpha}(0)$ on the laser-profile samples. Then $dG/d\theta_j = \langle\Lambda, \partial\mathcal{E}/\partial\theta_j\rangle$. One adjoint gives the derivative for every laser parameter at the cost of one host profile derivative each. $\Lambda$ is also a physical map: the sensitivity of the residual to the initial laser field. **[derived]**
2. Density parameters ($n_0$, $F_L$; plan Section 2.1). Loaded markers contribute $\sum_p\lambda_p(0)\,\partial w_p/\partial\theta_j$. Injected markers contribute $\sum_p\lambda_p(t_p^{inj})\,\partial w_p/\partial\theta_j$ at their injection step. **[derived]**
3. EV births. In the $\epsilon = 0$ continuum, $S_e(z,t) = \sum_\ell \nu_\ell(E_{rf},\gamma)\,n_\ell(z,t)$, where $n_\ell$ is the ion density in charge state $\ell$ and $\nu_\ell$ the ADK rate. Discretely, an electron marker $b$ is born at its parent ion with weight $W'_b$ (plan Section 5.2). The transpose is a co-state on each ion: $\mu_{W,i}$ receives $\lambda_e$ of the newborn marker at its birth step. The ion co-states $(\mu_{P_5},\mu_{P_6},\mu_{P_7},\mu_W)$ then run backward through the transpose of the EV update (plan Section 5.1). Through $\partial p_\ell/\partial E$ they add an adjoint-field source at the ion position, deposited with the gather shape. Through the ion charge $w_{tl} = w\bar Z - W$ they receive the adjoint $J$ and $\rho$ fields, gathered with the deposit shape. The local transposes exist as the Warp-generated VJP of `ev_ionize` (`fbpic/autodiff/ionization.py:97`). **[derived, code]**
4. Ion motion. N and He ions are heavy and, on the evidence so far, not chaotic. They can keep Lagrangian tangents in the forward scheme and reverse-mode co-tangents in the adjoint (VJP clones `fbpic/autodiff/particles.py:93`, `:144`). A newborn electron inherits the ion displacement $\xi_I$ (plan Section 5.3). In weight form this is a birth dipole $-\nabla_z\cdot(S_e\,\xi_I)$ that must be projected at birth (an R-v type projection). Its size is unknown. Open question O3. **[derived]**

### 3.4 Discrete form: the transpose of the forward particle-pdf scheme

The collocated forward scheme is a linear recurrence in $Y^n = (\hat W^n, \hat F^n)$, where $\hat F$ holds the spectral field sensitivities of every mode:

$$
\hat W^{n+1} = \hat W^n - \Delta t\, G_n\,\mathcal{I}\hat F^n, \qquad \hat F^{n+1} = \mathcal{M}\,\hat F^n + \mathcal{B}\,A_n\,\hat W^{n} .
$$

(The exact placement of the deposit within the FBPIC step, between the two half position pushes, changes the indices but not the structure.)

$\mathcal{I}$ is `spect2interp`. $G_n$ (markers by mesh) holds the gather shape times $\mathbf{D}_s/\hat n$ and $\mathbf{D}'_s/\hat n$. $A_n$ is the weight deposit with the deposit shape (Ruyten-corrected $S_r$, `deposition/cuda_methods_fused.py`; plan Section 4.7). $\mathcal{B}$ is `interp2spect`, the filter, and the current correction. $\mathcal{M}$ is the PSATD push, window shift, and damping. Every coefficient depends only on the primal trajectory, never on $Y$. The adjoint recurrence is the transpose, run from $n = N_t - 1$ down to 0:

$$
\mu_F^{n} = \mathcal{M}^T\mu_F^{n+1} - \Delta t\,\mathcal{I}^T G_n^T\lambda^{n+1}, \qquad \lambda^{n} = \lambda^{n+1} + A_n^T\mathcal{B}^T\mu_F^{n+1} .
$$

What this needs **[derived, code]**:

1. $G_n^T\lambda$: a deposit of $\lambda_s\mathbf{D}_s/\hat n$ (6 components per mode) with the gather shape and the conjugate azimuthal phase. A Warp reference exists: the VJP of the gather clone with respect to the field grids (`fbpic/autodiff/particles.py:196-206`).
2. $A_n^T\mu$: a gather of the adjoint $J$ and $\rho$ grids with the deposit shape, times $q\,\mathbf{v}_s$. It is not the primal gather, because FBPIC deposits and gathers with different radial shapes. A reference exists: the VJP of the deposition clone with respect to the weights (`fbpic/autodiff/deposition.py:50`).
3. $\mathcal{M}^T$, $\mathcal{B}^T$, $\mathcal{I}^T$: per-mode transposes of the PSATD $(k_z, k_r)$ matrices and the current correction, the transposed DHT ($M^T$, not $M^{-1}$), the conjugate FFT, a reversed window shift with zero fill at the back, and the diagonal damping and filter (self-transposes). The design of these transposes is at plan commit `07dcc6c` (plan Section 4, Phase P11).
4. Structural events. A marker removed at the window back has $\lambda = 0$ before its removal step in the backward sweep. A marker injected at the front drops its $\lambda$ at injection, except for the density terms of Section 3.3. A newborn EV electron passes its $\lambda$ to the parent ion co-state.

Verification advantage **[derived]**. Forward and adjoint are linear maps with identical coefficients. The dot-product identity $\langle\lambda(T), Y_{fwd}(T)\rangle = \langle\mu(0), Y(0)\rangle$ must hold to rounding, even at $N_t = 2000$ and even when the dynamics is chaotic. This is a strict L1-type check that FD cannot give after trapping. It requires that the forward and backward sweeps see bitwise-identical coefficients (Section 6.3).

[paper] Section 9, p49 to p50, anticipates this direction: "For cases with many parameters-of-interest, joining these methods with adjoint-based methods would be valuable. Since the adjoint equation is similar to its original governing equation, the particle discretization of the adjoint is expected to be similarly effective." The paper does not demonstrate an adjoint.

## 4. Is the adjoint free of Lyapunov growth? (task item 3)

### 4.1 Three adjoints with different growth

Let $\Phi(t,T)$ be the flow map of the characteristics, and $\Lambda$ its leading finite-time Lyapunov exponent. In the PIC model $\Lambda$ includes the N-body and grid-noise part, measured at about 0.019 per step after step 1200 on Twin-S (plan Section 12.10).

1. Adjoint of the particle-exact tangent (reverse-mode AD of the PIC step; Phase P11). The co-states on the markers are $(\mu_x,\mu_u) = \nabla_z\lambda$, that is, $\nabla\lambda(z,t) = (\partial\Phi(T;z,t)/\partial z)^T\nabla\phi$. The backward propagator is the transpose of the forward Lagrangian propagator, with the same singular values. The co-states grow like $e^{\Lambda(T-t)}$, the same $10^{11}$ factor as the forward tangent. Reverse mode computes the transpose of the same chaotic Jacobian, so it is equally unusable at $N_t = 2000$. **[derived]** Consequence for the plan: Phase P11 as designed is not a remedy for the P7 finding.
2. Continuous adjoint in the $f\,\nabla_u\lambda$ form. $\lambda$ is a scalar pulled back along chaotic characteristics. Its values stay bounded, but $\nabla\lambda$ filaments and grows like $e^{\Lambda_{mf}(T-t)}$, where $\Lambda_{mf}$ is the exponent of the mean-field characteristics. On markers, the PIC exponent $\Lambda$ applies instead. Any discretization that differentiates $\lambda$ along markers inherits this growth. A mesh estimate of $\nabla_u\bar\lambda$ from cell averages is bounded by $\lVert\lambda\rVert_\infty/\Delta u$, but it is a different, coarse-grained operator, with an $O(1)$ bias wherever $\lambda$ varies below $\Delta u$. **[derived]**
3. Particle-pdf adjoint in the $\lambda\,\nabla_u f$ form (Section 3.4). It needs only values $\lambda_s$ on markers and the mesh $\nabla_u f$. Write one forward step as $Y^{n+1} = M_n Y^n$. The adjoint step is $\mu^n = M_n^T\mu^{n+1}$. For every $n_1 < n_2$, $\lVert M_{n_2-1}\cdots M_{n_1}\rVert_2 = \lVert M_{n_1}^T\cdots M_{n_2-1}^T\rVert_2$. So the adjoint grows at most as much as the forward particle-pdf system, and in the same singular directions. **[derived]**

Statement **[derived]**. The transposed particle-pdf adjoint is free of Lyapunov growth exactly when the forward particle-pdf scheme is. The conditions are the forward conditions, plus one:

1. The continuum sensitivity is Lyapunov stable for the observables ([paper] Eq. (81), p29; Eq. (92), p33). The paper states that no criterion for this is known ([paper] Section 9, p49). Evidence here: FD secants at $h = 10^{-2}$ and $10^{-3}$ agree on Twin-S at $N_t = 2000$ (plan Section 12.10).
2. The coarse-grained $\nabla_u f$ is bounded. The primal $f$ also filaments forward in time. The fine-grained $\nabla_u f$ grows, but the mesh estimate is bounded by about $f/\Delta u$. $\Delta u$ is therefore the regularization scale for both directions, with the same bias. For the cold plasma and the thin trapped bunch, $|\nabla_u f|/f \sim 1/\Delta u$, so the bias question of the cold-plasma section applies unchanged to the adjoint.
3. Each phase-space cell where $\mathbf{D} \neq 0$ holds enough markers ([paper] Eq. (43), p17). The adjoint deposit $G_n^T\lambda$ divides by the same $\hat n$ as the forward. Its noise is the cell average of $\lambda$ values, with variance about $\operatorname{Var}_{cell}(\lambda)/N_{cell}$. The values $\lambda_s$ are rough across neighboring markers, because neighbors at time $t$ end at different places at $T$, but they stay bounded by $\lVert\phi\rVert_\infty$ plus the integrated field source. The noise therefore scales like $N_{cell}^{-1/2}$, not like $e^{\Lambda t}$.
4. Additional adjoint condition: the coefficients of the backward sweep must equal those of the forward sweep. With replay from checkpoints this needs deterministic deposits (Section 6.3).

Duality summary **[derived]**. Forward Lagrangian displacement $\xi$ is dual to the adjoint co-tangent $\nabla\lambda$; both stretch. Forward weight $\hat W$ is dual to the adjoint value $\lambda$; both are transported without stretching. The phase-space dipole of the hint can be carried either by a displacement or by a weight; particle-pdf chooses the weight in forward mode and the value in adjoint mode, and puts the derivative on the mesh $\nabla_u f$ in both.

Relation to chaotic-sensitivity literature **[lit]**. Conventional tangent and adjoint methods for chaotic systems give derivatives that grow exponentially with the horizon (Wang, Hu, Blonigan, arXiv:1204.0159). Shadowing methods (LSS, NILSS, NILSAS) target long-time averages of ergodic systems and need an optimization over the trajectory. The LWFA problem is a transient and has no time average, as the paper notes for its own cases ([paper] Section 1, p3). The space-split sensitivity (S3) method of Chandramoorthy and Wang (arXiv:2002.04117) also regularizes the unstable contribution through the gradient of the log density on the attractor. This is the same device as the particle-pdf use of $\nabla_u f$, in a different setting. I found no adjoint particle-pdf method for chaotic PIC in the literature.

### 4.2 Jacobian count

| Mode | Solves per dogbox Jacobian | Shared per solve set |
|---|---|---|
| Forward particle-pdf | $K$ weight sets in one pass | primal, $c_0$ |
| Adjoint particle-pdf | 11 adjoint sets (rank of $C$), in one backward sweep if memory allows | recording pass, replay, $c_0$ |
| Adjoint, gradient only | 1 adjoint set | same |

The gradient-only adjoint suits L-BFGS-B, which failed in Phase P3 (plan Section 2.5), so it does not replace the Jacobian for dogbox. A matrix-free Gauss-Newton with Krylov products costs one forward and one adjoint per product; with 11 residuals and $K \le 25$, the explicit Jacobian is cheaper.

Per-solve cost **[estimated]**. Per adjoint set, the backward step does one gather of 4 components per mode (adjoint $J$, $\rho$) and one deposit of 6 components per mode. Per forward parameter it is one gather of 6 and one deposit of 4. Deposits are atomic-bound, so I estimate $c_1^{adj} \approx 1.0$ to $1.5\,c_1$.

Cost per Jacobian:

$$
C_A = 1 + n_{sweep}(R + c_0) + 11\,c_1^{adj},
$$

with $R$ the replay cost per backward sweep (Section 6) and $n_{sweep}$ the number of backward sweeps (more than 1 when the 11 adjoint field sets do not fit in GPU memory). Crossover with $C_F(K)$ for one sweep: $K^\ast = 11\,c_1^{adj}/c_1 + R/c_1$. With $c_1^{adj}/c_1 = 1$ to 1.3 and $R/c_1 = 1.5$ to 3: $K^\ast \approx 13$ to 18. The crossover does not depend on $c_0$ when $n_{sweep} = 1$. At production with $n_{sweep} = 2$ and $K$-passes of 7, `cost_model.py` gives $K^\ast \approx 15$. At $K = 25$: $C_A \approx 12$ to 18 against $C_F \approx 18$ to 32. **[estimated]**

## 5. Evidence from a 1D-1V test (supports item 3)

### 5.1 Setup

Script: `work_multi/adjoint_pdf_1d1v.py` (full path in Section 8). Driven 1D-1V electrostatic PIC: periodic box $L = 4\pi$ ($k_0 = 0.5$), 64 cells, CIC deposit and gather, FFT Poisson, leapfrog, Maxwellian electrons ($v_{th} = 1$), neutralizing ions, $\Delta t = 0.1\,\omega_p^{-1}$. The external driver is $E_{ext} = \theta\sin(k_0x - \omega t)$ with $\omega = 1.3$ (near the Bohm-Gross frequency) and $\theta = 0.08$; it traps electrons, so the particle dynamics is chaotic. QoI: kinetic energy $J(t) = \sum_p w_p v_p^2/2$. Methods:

1. Particle-exact tangent $(\xi_x,\xi_v)$, including the field tangent from $\delta\rho$.
2. Collocated particle-pdf forward ([paper] Eq. (44), (52), (53)): phase-space mesh $64\times48$ over $|v| \le 6$, CIC in $x$, quadratic B-spline in $v$, $\hat n$ normalization.
3. The exact transpose of item 2, run backward from $\lambda(T) = v^2/2$ on the markers, with stored primal markers.
4. Reference: central FD with common random numbers (same initial markers for $\theta\pm h$), $h \in \{0.02, 0.01, 0.005\}$, 4 seeds.

### 5.2 Results

Run: $N = 5\cdot10^4$ markers, 2000 steps ($T = 200\,\omega_p^{-1}$), 4 seeds. Outputs: `adjoint_pdf_1d1v_N5e4_Nt2000.out`, `slice_ref_N5e4_Nt2000.out` (13-point slice $\theta \in [0.05, 0.11]$, 4 seeds, slope of a quadratic fit at $\theta_0$), `pdf_mesh_scan_N5e4_Nt2000.out` (velocity mesh $N_v = 24$ and 96, seeds 1 and 2). A repeat at $N = 2\cdot10^4$, 1500 steps, is in `adjoint_pdf_1d1v_N2e4_Nt1500.out`. **[measured]**

1. Transpose. The adjoint gradient equals the forward particle-pdf gradient to $9.7\cdot10^{-16}$ relative at $N = 5\cdot10^4$ (36.591386453 both) and $1.2\cdot10^{-16}$ at $N = 2\cdot10^4$.
2. Boundedness. The adjoint values stay bounded over the whole backward sweep: RMS 1.39 at $T$, 1.42 at step 1000, 1.59 at step 0; maximum 9.6 to 17. The terminal values are RMS 1.39 and maximum 9.6. The forward weight RMS grows from 4.3 (step 100) to 42 (step 1999) per marker weight, roughly linearly, not exponentially.
3. Chaos of the Lagrangian tangent. The RMS of $\xi_v$ grows from 43 at $t = 25$ to $2.9\cdot10^{11}$ at $t = 195$, a rate of about 0.13 per $\omega_p^{-1}$ (0.013 per step; Twin-S: 0.019 per step). The particle-exact derivative becomes useless after $t \approx 40$.
4. Accuracy. Table of $dJ/d\theta$ (mean $\pm$ standard deviation over seeds):

| $t\,\omega_p$ | rms $\xi_v$ | Particle-exact | Particle-pdf, $N_v = 48$ | pdf $N_v = 24$ (seed 1, 2) | pdf $N_v = 96$ (seed 1, 2) | CRN FD, $h = 0.02$ | Smoothed slice (reference) |
|---|---|---|---|---|---|---|---|
| 5 | 1.4 | $4.48 \pm 0.09$ | $4.41 \pm 0.10$ | 4.19, 4.39 | 4.27, 4.49 | $4.48 \pm 0.09$ | $4.48 \pm 0.09$ |
| 25 | 43 | $38.2 \pm 0.9$ | $39.2 \pm 0.3$ | 34.9, 35.7 | 39.4, 40.8 | $38.2 \pm 0.2$ | $37.5 \pm 0.2$ |
| 35 | $1.6\cdot10^2$ | $37 \pm 8$ | $31.4 \pm 2.2$ | 31.7, 35.0 | 30.1, 33.8 | $24.9 \pm 1.9$ | $24.3 \pm 1.5$ |
| 45 | $1.6\cdot10^3$ | $47 \pm 81$ | $30.8 \pm 2.1$ | 31.1, 35.5 | 17.6, 20.2 | $4.7 \pm 3.7$ | $5.1 \pm 3.5$ |
| 65 | $1.2\cdot10^4$ | $-280 \pm 540$ | $59.1 \pm 1.8$ | 34.1, 45.7 | 44.8, 55.0 | $41.2 \pm 3.6$ | $40.8 \pm 3.3$ |
| 85 | $1.2\cdot10^5$ | $(1.1 \pm 0.7)\cdot10^4$ | $44.9 \pm 3.9$ | 40.0, 48.3 | 1.8, 4.1 | $14.0 \pm 3.7$ | $15.4 \pm 1.8$ |
| 125 | $1.2\cdot10^7$ | $(-5 \pm 5)\cdot10^5$ | $59.4 \pm 5.8$ | 40.3, 47.3 | 13.3, 49.3 | $23.6 \pm 3.0$ | $23.5 \pm 1.4$ |
| 195 | $2.9\cdot10^{11}$ | $(-7 \pm 11)\cdot10^9$ | $47.3 \pm 7.8$ | 32.6, 50.4 | 33.4, 22.8 | $31.8 \pm 4.1$ | $30.4 \pm 0.8$ |

### 5.3 Reading of the test

1. The macroscopic response is regular. CRN FD at $h = 0.02$ matches the 13-point smoothed slope within about 1 standard deviation at every time. This is the continuum-stability assumption of [paper] Eq. (92), p33, holding in this test. **[measured]**
2. The transposed adjoint is exact and bounded, as derived in Section 4.1. Regularity in the sense "no exponential growth" holds for both the forward weights and the adjoint values. **[measured]**
3. Accuracy after trapping fails in this prototype. Before trapping ($t \le 25$), particle-pdf agrees with the reference within about 1% to 7%. After trapping ($t \ge 35$), the $N_v = 48$ result is biased high by +7 to +36 against a reference of 5 to 41: mean bias about +24, about 0.9 times the RMS of the reference. The seed scatter (2 to 8) is smaller than the bias, so the error is systematic. The bias changes with the velocity mesh. $N_v = 24$ is smoother but also wrong. $N_v = 96$ (about half as many markers per cell) is closer at some times and further at others, and its two seeds differ by up to 36. No convergence toward the reference is visible at this $N$. The $N = 2\cdot10^4$ repeat shows the same bias. **[measured]**
4. Interpretation **[derived, not proven]**. The particle-pdf source uses a coarse-grained $\nabla_v f$. In the weak form, $dG/d\theta = \iint\lambda\,H$, the coarse-grained source is accurate only if $\lambda$ is smooth on the scale $\Delta v$. After trapping, $\lambda$ is pulled back along mixing characteristics and varies below $\Delta v$, so $\int\lambda\,\overline{\nabla_v f} \neq \int\lambda\,\nabla_v f$. Refining $\Delta v$ reduces this bias but raises the $N_{cell}^{-1/2}$ noise. This is the same mechanism as condition 2 of Section 4.1, and it applies equally to the forward scheme and the adjoint.
5. Limits of the test. It is a quick prototype: collocated markers only, first-order weight integration, no particle addition or redistribution ([paper] Section 3.2.4, p17 to p19), one $N$ per mesh, and a thermal (not cold) plasma. The paper's tests (Debye shielding and sheaths) have no trapping vortices. The result is a warning that R1 must reproduce or refute with a careful implementation and a joint $(N, \Delta v)$ convergence study. It is not a proof that the method fails.

Consequence for this section. Multi-parameter scaling and the adjoint are secondary to this accuracy question. The adjoint inherits exactly the accuracy of the forward particle-pdf (item 1), so no adjoint work should start before R1 and R2 show convergence after trapping.

## 6. Memory and checkpointing (task item 4)

### 6.1 What the backward sweep needs

The collocated particle-pdf adjoint at step $n$ needs the primal markers $(\mathbf{x},\mathbf{u},\gamma^{-1},w)$ at that step: 64 B per marker. It also needs the EV ion arrays (32 B per ion) and, for ion co-tangents, the primal fields at the ions (48 B per ion, or the primal field arrays). It does not need the primal fields for the electron part, because the adjoint field equations have $\theta$-independent coefficients. The phase-space mesh of $\nabla_u f$ is recomputed from the markers in the backward sweep. **[derived]**

Storing every step is not possible: Twin-S needs about 0.4 to 0.7 GB per step, about 1 TB for 2000 steps; production needs 3.8 GB per step, 7.7 TB. Checkpointing with replay is required. **[estimated]**

### 6.2 Checkpoint schedules

The host has 125 GB RAM (114 GB available) and 1.3 TB free disk on `/mnt/work`; the GPUs have 48 GB each (`free -g`, `df -h`, 2026-10-31; plan Section 8). For $c$ checkpoints and $l$ steps to reverse, the Griewank-Walther binomial schedule needs $t(l,c) = r\,l - \binom{c+r}{c+1}$ forward steps, with $r$ the smallest integer such that $\binom{c+r}{c} \ge l$. **[lit, computed in `cost_model.py`]**

| Case | Checkpoint size | Store | Checkpoints $c$ | Total forward sweeps | Replay $R$ per backward sweep |
|---|---|---|---|---|---|
| Twin-S | 0.5 to 0.8 GB (10 M markers at peak) | GPU, about 35 GB free | 45 | 2.44 | about 1.4 |
| Twin-S | same | host RAM, 60 GB | 100 | 1.95 | about 1.0 |
| Production | 5 GB (given) | host RAM, 100 GB | 20 | 2.99 | about 2.0 |
| Production | 5 GB | disk, 1.3 TB | 250 | about 1.9 | about 0.9, if disk bandwidth allows |

Transfers: a 5 GB checkpoint over PCIe 4 x16 takes about 0.2 s, small against an estimated production step of 0.5 to 1 s (60 M markers at about 7 ns per marker-step from Twin-S, plus 0.43 s of FP64 DHTs). Disk bandwidth on `/dev/sda2` is unknown; at 0.5 GB/s, 250 checkpoint writes and reads would take about 2 h and dominate. **[estimated]**

GPU memory during the backward sweep at production: 11 adjoint field sets of at least 1.6 to 3.8 GB each (18 to 42 GB), plus 0.48 GB of $\lambda$ per set, plus the primal working state and the current replay segment. This does not fit with 11 sets on one 48 GB GPU, so $n_{sweep} = 2$ on one GPU, or one sweep with 6 and 5 sets split over the two GPUs (each GPU replays the primal). On Twin-S all 11 sets take about 1 GB and fit easily. **[estimated]**

### 6.3 Replay determinism under chaos (a GPU-specific risk)

The deposits use FP64 atomics, so the summation order changes between runs. Two repeated $K = 1$ runs differ by up to $7.5\cdot10^{-13}$ relative after 10 steps (plan Section 12.10, step 2). With the measured growth of 10 per 120 steps after step 1200, a replayed segment of $L$ steps deviates from the recording by about $10^{-13}\cdot10^{L/120}$. A full replay from step 0 to 2000 would produce a different realization (deviation of order $10^{-2}$ relative). The forward and adjoint coefficients then differ, and the dot-product identity fails. **[derived from plan data]**

Rules:

1. Keep replay segments short: $L \le 600$ steps keeps the deviation below about $10^{-8}$. Binomial schedules with long first segments violate this; prefer uniform segments of at most a few hundred steps after step 1000.
2. Better: make the replay deterministic. Sort by cell (already done before each deposit, `stepper.py:137-160`) and accumulate per cell in a fixed order. The Phase P4 deposit tangent already runs one thread per cell over its particles (plan Section 12.10, step 3); a weight deposit with the same structure and no cross-thread atomics on the same cell would be bitwise reproducible.
3. Reverse integration of the primal (plan Section 11, item 4) is not usable after trapping: round-off grows by the same $10^{11}$ factor, and EV births, removal, and injection are irreversible.

## 7. Recommendation (task item 5)

### 7.1 Path

0. Precondition for every item below: R1 shows particle-pdf convergence after trapping (Section 5.3, kill criterion 0). Without it, neither the forward nor the adjoint particle-pdf is worth building.
1. Main path: forward, $K$-batched particle-pdf (collocated, with added markers where the cold-plasma section requires them). It needs no checkpointing, reuses the field tangents and `gather_tangent`, and its per-parameter state is one weight array. It covers Stage B ($K = 2$) and the first multi-parameter targets ($K = 3$ to 5) and remains competitive to $K \approx 13$ to 18.
2. Write the forward scheme as an explicit linear recurrence in the weights with coefficients that depend only on the stored or replayable primal (Section 3.4). Then the adjoint is a mechanical transpose and can be verified by the dot-product identity.
3. Adjoint particle-pdf: only when $K > 15$ to 20 (Zernike terms, a free-form density lineout), or when measured $c_1$ is much larger than estimated. It then gives all laser and density parameters from 11 adjoint sets, with the laser sensitivity map $\Lambda$ as a by-product.
4. Hybrid in time, optional: before trapping (steps below about 1000) the Lagrangian tangent is accurate (plan Section 12.10). The forward scheme can start in Lagrangian form and project into weights at a fixed step (variant R-v). The transpose of that projection is $\mu = \nabla_z\bar\lambda$ evaluated on the mesh at the projection step, then exact reverse mode before it. This is only worth it if the projection is cheaper than particle-pdf over the early steps; the R-v section decides.
5. Do not pursue Phase P11 (reverse-mode AD of the PIC step) for the chaos problem. It transposes the same chaotic Jacobian (Section 4.1, item 1).

### 7.2 What R1 and R2 must show early

1. R1, first and decisive: repeat the driven-trapping test of Section 5 with a careful implementation (paper's Debye case first, then trapping), with a joint scan of $N$ ($10^5$ to $10^7$) and $\Delta v$, collocated and with particle addition. Required: convergence of the particle-pdf column to the smoothed slice after trapping, with an error that falls as $N$ grows and $\Delta v$ shrinks together.
2. R1, adjoint: the transposed adjoint equals the forward particle-pdf gradient to $10^{-12}$ (already shown in the prototype). Also run 3 to 5 parameters in one batched pass (driver amplitude, frequency, wavenumber, phase, plasma density) to check that the shared mesh is reused and that each column has the expected error.
3. R2 (1D cold beam with trapping): the same three-way comparison for a cold distribution, with collocated and with added markers, for both the forward and the adjoint direction. The adjoint inherits the $\nabla_u f$ bias exactly, so one study covers both.
4. Measure $c_0$ and $c_1$ on the GPU early in R3, on Twin-S, before the EV and window details: a prototype phase-space mesh kernel and a $K$-batched weight deposit at $K = 1, 2, 4, 8$. These two numbers decide the multi-parameter value of Track R against FD.
5. Measure the variance of the particle-pdf column against the variance of FD at $h = 10^{-2}$ on Twin-S (seeds or marker resampling). The method has a multi-parameter advantage only if its noise is not larger than FD noise at equal cost.

### 7.3 Kill criteria for the multi-parameter case

0. Applies to all of Track R: in the R1 driven-trapping test, the particle-pdf column error against the smoothed slice after trapping does not fall below 20% of the reference RMS at the largest affordable $N$ (at least $10^7$ markers in 1D-1V) for any $\Delta v$, with and without particle addition. The prototype of Section 5 is at about 90%.
1. On Twin-S, $C_F(5) = 1 + c_0 + 5c_1 > 6$ (one-sided FD at $K = 5$), with an optimized mesh kernel. Then particle-pdf has no cost advantage over FD for $K \le 5$.
2. On Twin-S, the particle-pdf column error against the smoothed FD column exceeds the FD scatter between $h = 10^{-2}$ and $10^{-3}$ by more than 2 times, after the regularization scale $\Delta u$ is tuned in R2.
3. In R1 or R2, the forward particle-pdf weight norm or the adjoint value norm grows by more than 10 times over the horizon while the FD response stays bounded. That would show that the scheme itself, not only the particles, is unstable.
4. In R1, the dot-product identity of the transposed adjoint fails above $10^{-10}$ with deterministic coefficients. That would indicate that the forward scheme is not a fixed linear map (for example, the coefficients depend on the weights).
5. For the adjoint only: replay determinism cannot be reached on the GPU (rule 2 of Section 6.3) and the dot-product identity fails above $10^{-6}$ at $N_t = 2000$.

### 7.4 Cost of the work **[estimated]**

1. Forward batched particle-pdf, beyond the $K = 1$ R3 work: 2 to 4 days for $K$-batching, tiling over $K$ in the deposit, and the driver.
2. Adjoint particle-pdf on top of a working forward scheme: 15 to 25 days. Items: field-operator transposes (PSATD, current correction, DHT, FFT, window), 2 transposed particle kernels, EV co-states and the birth coupling, checkpoint and replay with deterministic deposits, ion co-tangents, and verification. Uncertainty is large.

## 8. Open questions

1. O1. Size of $c_0$ in quasi-cylindrical 3D-3V geometry with $N_m = 2$: a sparse or per-cell density estimate of $\nabla_u f$ may avoid a global 6D mesh.
2. O2. Is the continuum response Lyapunov stable for the 11 residuals beyond Twin-S, for example at production size and with 5 modes? The FD secant agreement is the only evidence.
3. O3. Size of the EV birth dipole from ion displacement tangents (Section 3.3, item 4).
4. O4. Disk bandwidth of `/mnt/work` for production checkpoints.
5. O5. Whether dogbox tolerates a Jacobian with $N^{-1/2}$ noise of a given level; the D32 fixed-point shift should be re-measured on the particle-pdf Jacobian (plan Section 6.4).
6. O6. Whether the adjoint should use a smoothed terminal condition (wider $\delta_u$) to reduce $\operatorname{Var}_{cell}(\lambda)$; this changes the loss.

Files of this study:

1. `inversion/notes/r0-evidence-2026-10-31/work_multi/cost_model.py` and `cost_model.out`: cost, checkpoint, and memory model.
2. `inversion/notes/r0-evidence-2026-10-31/work_multi/adjoint_pdf_1d1v.py` with outputs `adjoint_pdf_1d1v_N5e4_Nt2000.out` and `adjoint_pdf_1d1v_N2e4_Nt1500.out`: the 1D-1V test. Run as `python3 -I adjoint_pdf_1d1v.py N Nt`.
3. `inversion/notes/r0-evidence-2026-10-31/work_multi/slice_ref.py` (`slice_ref_N5e4_Nt2000.out`, `.npy`): the smoothed brute-force reference.
4. `inversion/notes/r0-evidence-2026-10-31/work_multi/pdf_mesh_scan.py` (`pdf_mesh_scan_N5e4_Nt2000.out`): the velocity-mesh scan.

## 9. References

1. Chung, Bond, Cyr, Freund, "Regular sensitivity computation avoiding chaotic effects in particle-in-cell plasma methods", JCP 400, 108969 (2020). Read in full: Sections 2, 3, 7, 9 (PDF pages 5 to 20, 35 to 38, 46 to 50); Section 6 (p28 to p35) for Eq. (78) to (92). Equations checked against the page images.
2. Bartsch, Knopf, Scheurer, Weber, "Controlling a Vlasov-Poisson plasma by a Particle-In-Cell method based on a Monte Carlo framework", [arXiv:2304.02083](https://arxiv.org/abs/2304.02083). Adjoint Eq. (31a), gradient Eq. (33), particle adjoint Section 4, Eq. (46). Read through the arXiv HTML via a summarizer, not in full.
3. Knopf, Weber, "Optimal control of a Vlasov-Poisson plasma by fixed magnetic field coils", Appl. Math. Optim. (2020), [arXiv:1710.09619](https://arxiv.org/abs/1710.09619). Not read in full.
4. Albi, Dimarco, Ferrarese, Pareschi, "Full-horizon magnetic control of the Vlasov-Poisson system for plasma confinement", [arXiv:2610.26110](https://arxiv.org/abs/2610.26110). Abstract only.
5. Einkemmer, Li, Wang, Yang, "Suppressing instability in a Vlasov-Poisson system by an external electric field through constrained optimization", JCP (2020), [PDF](https://yunany.github.io/Publications/20_JCP_Vlasov.pdf). Semi-Lagrangian discrete adjoint. Search summary only.
6. Wang, Hu, Blonigan, "Least squares shadowing sensitivity analysis of chaotic limit cycle oscillations", JCP 267 (2014), [arXiv:1204.0159](https://arxiv.org/abs/1204.0159). Search summary only.
7. Chandramoorthy, Wang, "A computable realization of Ruelle's formula for linear response of statistics in chaotic systems", [arXiv:2002.04117](https://arxiv.org/abs/2002.04117). Search summary only.
8. Peraud, Hadjiconstantinou, "Adjoint-based deviational Monte Carlo methods for phonon transport calculations", Phys. Rev. B 91, 235321 (2015), [OSTI 1370983](https://www.osti.gov/pages/biblio/1370983). Cited by the paper (Section 9, p50) as the adjoint precedent for particle methods. Search summary only.
9. Griewank, Walther, "Algorithm 799: revolve", ACM TOMS 26 (2000). Binomial checkpointing formula, from memory; not re-read.
