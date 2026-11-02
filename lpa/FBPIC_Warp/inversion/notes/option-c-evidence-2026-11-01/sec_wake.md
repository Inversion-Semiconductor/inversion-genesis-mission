# Option C on the 1D cold wake with trapping (task key: wake)

## Verdict

**Option C fails on the 1D cold wake with trapping.** No noise level $\sigma$ meets the goal. Where the mean-field LR estimator is stable and accurate ($\sigma \ge 0.3$ with $N \ge 4$), the noise has destroyed the physics (mean bunch momentum 6.8 instead of 40). Where the noise is weak enough to keep the physics ($\sigma \le 0.03$), the estimator diverges exponentially, by $10^{24}$ at $t = 75$. The divergence is mainly a sampling-noise feedback through the weight deposit, with a rate that falls with $N$ and rises steeply as $\sigma$ falls. The remaining $N$-independent part of the rate approaches the chaotic tangent rate (about 0.23) as $\sigma \to 0$. Even at $\sigma = 0.01$ the gradient bias against the deterministic model is 12 to 92 percent at $t = 75$. The estimator itself is correct: at $\sigma = 0.3$ to $1$ and $N = 4$ it matches the CRN FD of the noisy model within 2 SE in 23 of 24 comparisons (six observables, $t = 30$ and $75$; measured). Single-run noise is 15 percent or less only for $\log Q$ at $\sigma = 1$. The estimator costs 1.4 to 2.4 primal runs per parameter in the NumPy harness.


## 1. Setup

### 1.1 Harness

The harness is `work_wake/wake_lr.py`. It keeps the physics and push of `r0/work_tests/r2_pilot.py` and `mt/work_proto/r2_tan.py`: 1D relativistic electrostatic wake, prescribed ponderomotive driver $A = \tfrac12 a_0^2 e^{-\xi^2/2}$, EV ionization injection, $c = \omega_p = k_p = 1$, $\theta = a_0 = 2.5$, $t_f = 75$, 3000 steps. Observables are the six r2 observables (log trapped charge $\log Q$, mean momentum $\langle u\rangle$, centroid $\langle\xi\rangle$, $\log\sigma_u$, $\log\sigma_\xi$, $\log(\sigma_u/\langle u\rangle)$), with the smooth $u$ window at $u_c = 3$.

Three changes make common random numbers (CRN) possible and the code about 20 times faster. All references use the same code, so the comparison is consistent.

1. The domain front is $x = 80$. Plasma electrons ahead of $x_d + 8$ are not pushed and are left out of the deposit together with their own ion charge (they are at rest, $E = 0$ there, and the driver force there is below $10^{-13}$).
2. EV births follow a $\theta$-independent schedule: every 4 steps, every injector ion with $|x - x_d| < 1.5$ emits weight $r\,(1 - e^{-\Gamma\,4\Delta t})$. Particle slots are fixed in advance, so each particle has a stable slot index.
3. The noise for step $n$ is `default_rng([seed, n]).standard_normal` over the active slot range. The active range is $\theta$-independent, so CRN is exact.

Particle number: plasma $5N$ per cell, injector ions $N$ per cell, $N \in \{1, 4, 16\}$. $N = 4$ equals the r2 pilot resolution (20 per cell). At $N=1$ there are $1.6\times10^4$ plasma and $7.2\times10^3$ beam particles.

### 1.2 Noise model and why

Noise: $u_{n+1} = u_n + \Delta t\, f_1(x_n, u_n; E_n, \theta) + \sigma\sqrt{\Delta t}\, z_n$, $z_n \sim N(0,1)$, on every electron (plasma and bunch) with $\xi = x - x_d < 6$. Ahead of $\xi = 6$ there is no noise and no push. Reasons:

1. **Noise only behind the driver front.** Ahead of the driver the plasma is at rest and $\delta F = 0$ (the driver tail at $\xi = 6$ is $e^{-18}$ of the peak). Noise there only pre-heats the plasma. The weight increment dropped at $\xi \ge 6$ is at most $1.1\times10^{-7}$ per step (measured, field `drop` in every LR output).
2. **No drag (pure Langevin, not OU).** An OU drag $-\nu u$ bounds the spread at $\sigma/\sqrt{2\nu}$ only after $t \gtrsim 1/\nu$. To bound heating inside $t_f = 75$ needs $\nu \gtrsim 0.05$, which damps the wake amplitude by $e^{-\nu t} \approx e^{-2}$ over the run (derived). Drag toward a local mean flow needs the mean flow as a field, which is a phase-space average that option C wants to avoid. So drag removes physics faster than it saves it here.
3. **Noise on the plasma, not only on the bunch.** The bunch gradient is carried mostly by the wake-field tangent $\delta E$, which comes from the plasma response. In LR form the plasma response must be a weight deposit, so the plasma must carry noise. The alternative (noise on the bunch only, plasma kept pathwise) was tested and fails because the plasma pathwise tangent itself grows (Section 3.4).
4. **1D causality would allow a back cut.** In 1D, $E(x)$ depends only on charge ahead of $x$, so electrons that slip behind the bunch cannot affect it. This does not carry over to the 3D electromagnetic case, so no back cut was used.

Heating scale: a plasma electron that has been behind the driver for a time $\tau$ has a random momentum spread $\sigma\sqrt{\tau}$; at $\tau = 70$ this is $0.08$, $0.25$, $0.84$, $2.5$, $8.4$ for $\sigma = 0.01, 0.03, 0.1, 0.3, 1$ (derived). The cold wake has momentum amplitude of order 1 to 5, and the trapping threshold sits near $u \approx 2$ to $3$.

## 2. Estimators (derived)

### 2.1 Mean-field LR with a deterministic position update

The discrete scheme has a random $u$ update and a deterministic $x$ update, $x_{n+1} = \Psi(x_n, u_{n+1}; \theta) = x_n + \Delta t\, u_{n+1}/\gamma(u_{n+1}, A(x_n;\theta))$. The position update depends on $\theta$ through $A$ in $\gamma$, so a pure weight method does not cover it. Fix the realized momentum sequence $u_{1:n}$. Then $x_n(\theta; u_{1:n})$ is a deterministic function, and the path density is
$$p(u_{1:n};\theta) = \prod_k \mathcal N\!\left(u_{k+1};\ \Phi(x_k(\theta), u_k; E_k, \theta),\ \sigma^2\Delta t\right).$$
Differentiation gives the exact identity
$$\frac{d}{d\theta}\mathbb E[g(x_n,u_n)] = \mathbb E\!\left[g\,\omega_n + \partial_x g\ \delta x_n\right],\qquad \omega_{k+1} = \omega_k + \frac{\delta\Phi_k\, z_k}{\sigma\sqrt{\Delta t}},\qquad \delta x_{k+1} = \partial_x\Psi\,\delta x_k + \partial_\theta\Psi,$$
with $\delta\Phi_k = \partial_x\Phi\,\delta x_k + \partial_E\Phi\,\delta E_k + \partial_\theta\Phi$ at fixed $u_k$. The position tangent has $\delta u \equiv 0$, so $\partial_x\Psi = 1 + \Delta t\,\partial_A v\,\partial_x A$ is nonzero only inside the driver and $\delta x$ cannot grow chaotically. In code each particle carries $\delta W_i = W_i\omega_i$ (weight tangent) and $\delta x_i$, with
$$\delta W_i \mathrel{+}= W_i\,\delta f_{1,i}\, z_i\sqrt{\Delta t}/\sigma,$$
where $\delta f_{1,i}$ is the change of the two-stage mean force at fixed $(x_i, u_i)$ (the `df1` of `r2_tan.py` with $\delta u = 0$).

Mean-field closure: $\delta E_k$ is the field of $\delta\rho = \delta\rho_{\rm ion} - \sum_i (\delta W_i S - W_i\,\delta x_i\,S')$. As $N\to\infty$ this is the linearized McKean-Vlasov response (the field is a functional of the law, not of the particle positions). At finite $N$ the closure has an $O(1/N)$ bias and, more important, it feeds sampling noise back into the weights (Section 3.1).

EV births: a particle born with weight $W_b(\theta)$ starts with $\delta W_b = dW_b/d\theta$ from the deterministic birth tangent ($\delta\rho_{\rm ion}$ gets the same deposit). Births are deterministic given $\theta$, so this term is exact and has no noise.

### 2.2 Joint-measure LR

With the field kept as an implicit function of all particle states, the only explicit $\theta$ dependence of the $N$-particle transition kernel is the driver term and the field of the birth-weight change at fixed positions, $\delta E^W$. The score is
$$S = \sum_i\sum_k \frac{\partial^{\rm expl}_\theta\Phi_{i,k}\ z_{i,k}}{\sigma\sqrt{\Delta t}},\qquad \frac{d}{d\theta}\mathbb E[J] = \mathbb E[(J - b)\,S] + \mathbb E\!\left[\textstyle\sum_b \partial_{W_b}J\ \delta W_b\right],$$
exact for finite $N$, with any constant baseline $b$ (here the leave-one-out ensemble mean). There is no weight deposit, so no feedback. The cost is that $\mathrm{Var}(S) = \sum_i\int |\partial_\theta^{\rm expl}F_i|^2/\sigma^2\,dt \propto N$ while $\mathrm{Var}(J) \propto 1/N$ for moment observables, so the single-run variance of $(J-b)S$ is $O(1)$ in $N$: it does not fall with particle number (derived).

### 2.3 Variance of the mean-field weights

Without feedback, $\mathrm{Var}\,\omega_i(t) = \int_0^t \mathbb E|\delta F_i|^2\,dt'/\sigma^2$, so $\omega_{\rm rms} \propto \sigma^{-1}$, independent of $N$, growing like $\sqrt t$ for bounded $\delta F$ (derived). With the weight deposit in the field, the deposit carries sampling noise of variance $\mathrm{Var}(\delta E_{\rm noise}) \approx \sum_{j} W_j^2\,\mathrm{Var}\,\omega_j \approx (Q_L^2/N_L)\,\mathrm{Var}\,\omega$ ($Q_L$, $N_L$: charge and number of noisy electrons ahead of the point). This noise is not adapted away by the particle's own increment, so
$$\frac{d\,\mathrm{Var}\,\omega}{dt} \approx \frac{\mathbb E|\delta F_{\rm signal}|^2}{\sigma^2} + \frac{Q_L^2}{N_L\,\sigma^2}\,\mathrm{Var}\,\omega ,$$
an exponential instability of the estimator with rate $\lambda_{\rm fb} \sim Q_L^2/(N_L\sigma^2)$ (estimated). It is a sampling-noise feedback, not chaos: it vanishes as $N\to\infty$ at fixed $\sigma$. A second, $N$-independent growth comes from the true linear response if $\delta F_{\rm signal}$ itself grows (collective amplification).

## 3. Results

All numbers below are measured with `work_wake/wake_lr.py` at $a_0 = 2.5$. FD references come from `work_wake/fd_batch.py` (CRN: the same seed at $a_0 \pm h$, $h \in \{0.02, 0.04\}$, plus a cubic fit over $|h| \le 0.04$). Tables come from `work_wake/analyze.py` and `work_wake/tables.py`; full tables are in `work_wake/out/tables_t{30.0,45.0,75.0}.md`. Ensembles in the tables below: FD $N=1$ 16 seeds (12 for $\sigma = 1$, $0.01$); FD $N = 4$ 8 to 9 seeds; LR $N = 1$ 16 seeds; LR $N = 4$ 9 seeds. The files in `work_wake/out/tables_*.md` were regenerated after the job pool was stopped and include 1 to 2 more seeds (LR $N = 4$: 11, FD $N = 4$: 10); all values move within 1 SE and no conclusion changes. The pool was stopped early because the shared machine ran at load 80 to 110 on 32 cores, which slowed each run about 20 times; the planned $N = 16$ ensembles were not run. Seeds of LR and FD are disjoint. "SD/|g|" is the single-run standard deviation of an estimator divided by the reference gradient; this is the noise of a gradient from ONE simulation.

### 3.1 Weight variance: scaling with sigma, N and time

Seed 1, $t \le 30$, plasma weight rms $\omega_{\rm rms}$, from `work_wake/growth.py` (runs in `work_wake/out/grow/`). "No feedback" sets the weight part of the deposit to zero (diagnostic only, env `NOFB=1`).

| $\sigma$ | $N$ | growth rate of $\ln\omega_{\rm rms}$, $t\in[10,30]$, with / without feedback | $\omega_{\rm rms}(30)$ with / without | $\sigma\,\omega_{\rm rms}(30)$ without |
|---|---|---|---|---|
| 0.01 | 1 | 2.95 / 0.027 | $2\times10^{33}$ / 65 | 0.65 |
| 0.01 | 4 | 1.44 / 0.027 | $1.9\times10^{15}$ / 64 | 0.64 |
| 0.03 | 1 | 0.94 / 0.031 | $1.3\times10^{10}$ / 23 | 0.68 |
| 0.03 | 4 | 0.41 / 0.030 | $6.7\times10^{4}$ / 23 | 0.68 |
| 0.03 | 16 | 0.30 / 0.030 | $7.6\times10^{3}$ / 23 | 0.68 |
| 0.1 | 1 | 0.27 / 0.038 | $1.9\times10^{3}$ / 7.8 | 0.78 |
| 0.1 | 4 | 0.094 / 0.037 | 41 / 7.7 | 0.77 |
| 0.1 | 16 | 0.11 / 0.036 | 50 / 7.7 | 0.77 |
| 0.3 | 1 | 0.089 / 0.030 | 9.3 / 2.5 | 0.76 |
| 0.3 | 4 | 0.051 / 0.027 | 5.1 / 2.3 | 0.69 |
| 0.3 | 16 | 0.055 / 0.026 | 4.9 / 2.2 | 0.67 |
| 1.0 | 1, 4, 16 | 0.023 to 0.029 / 0.018 to 0.022 | 0.57 to 0.67 / 0.49 to 0.56 | 0.49 to 0.56 |

Findings (measured):

1. Without feedback, $\sigma\,\omega_{\rm rms}$ is constant (0.5 to 0.8) over two decades of $\sigma$ and independent of $N$. This confirms $\omega_{\rm rms} \propto \sigma^{-1}$ and slow, sub-linear growth in time (Section 2.3).
2. With feedback, the growth is exponential and its rate falls with $N$. At $\sigma = 0.03$ the rates fit $\lambda \approx 0.23 + 0.7/N$ (estimated from three points). The $N$-dependent part is the sampling-noise feedback of Section 2.3. Its size grows steeply as $\sigma$ falls: $0.7/N$ at $\sigma = 0.03$, about $0.17/N$ at $\sigma = 0.1$, about $0.04/N$ at $\sigma = 0.3$ (estimated).
3. The $N$-independent floor (0.23 at $\sigma = 0.03$, 0.1 at $\sigma = 0.1$, 0.05 at $\sigma = 0.3$) is the growth of the true linear response of the noisy model. At small $\sigma$ it approaches the trapped-tangent growth rate of the deterministic pilot (about 0.23 $\omega_p$). This is expected: as $\sigma \to 0$, $\delta f/f$ of the noisy model resolves the same fine phase-space structure that makes the Lagrangian tangent grow, so the weights must grow at the same rate (derived, qualitative).

Full horizon, ensemble median over seeds (`work_wake/wtime.py`):

| $\sigma$ | $N$ | plasma $\omega_{\rm rms}$ at $t = 10, 20, 30, 45, 60, 75$ | bunch $\omega_{\rm rms}$ at $t = 75$ | $\max|\delta E|$ at $t = 75$ |
|---|---|---|---|---|
| 0.03 | 1 | 155, $2.5\times10^6$, $4\times10^{10}$, $2\times10^{17}$, $2\times10^{24}$, $5\times10^{30}$ | $5\times10^{30}$ | $3\times10^{30}$ |
| 0.1 | 1 | 6.1, 88, $1.1\times10^3$, $4.4\times10^4$, $1.5\times10^6$, $1.8\times10^8$ | $1.8\times10^8$ | $1.0\times10^8$ |
| 0.1 | 4 | 5.0, 36, 147, 948, $7.1\times10^3$, $2.2\times10^4$ | $3.3\times10^4$ | $3.5\times10^3$ |
| 0.3 | 1 | 1.5, 4.9, 10, 25, 116, 228 | 241 | 126 |
| 0.3 | 4 | 1.5, 4.1, 5.7, 7.8, 9.3, 17 | 22 | 4.0 |
| 1.0 | 1 | 0.34, 0.56, 0.68, 0.85, 1.06, 1.32 | 1.9 | 0.74 |
| 1.0 | 4 | 0.34, 0.53, 0.60, 0.66, 0.70, 0.76 | 1.35 | 0.33 |

So the estimator is bounded only for $\sigma = 1$ at any $N$ and for $\sigma = 0.3$ at $N \ge 4$. At $\sigma = 0.1$ it still grows by a factor of 150 from $t = 30$ to $75$ at $N = 4$. Two $N = 16$ runs at $\sigma = 0.1$ to $t = 75$ were started but did not finish because the machine was overloaded (load average near 100 on 32 cores).

### 3.2 Agreement with the noisy-model FD (reference i)

$z = |{\rm LR} - {\rm FD}|/\sqrt{{\rm SE}_{\rm LR}^2 + {\rm SE}_{\rm FD}^2}$, FD = cubic fit of the noisy model at the same $\sigma$ and $N$ (for $\sigma = 1$, $N = 4$ the FD is at $N = 1$).

$t = 30$ (trapping done, before the strong tangent growth):

| $\sigma$ | $N$ | quantity | $\log Q$ | $\langle u\rangle$ | $\langle\xi\rangle$ | $\log\sigma_u$ | $\log\sigma_\xi$ | $\log\sigma_u/\langle u\rangle$ |
|---|---|---|---|---|---|---|---|---|
| 0.3 | 4 | LR | 1.67 ± 0.13 | 2.10 ± 0.31 | -1.11 ± 0.48 | 0.56 ± 0.08 | -0.14 ± 0.49 | 0.22 ± 0.09 |
| | | FD | 1.68 ± 0.02 | 2.47 ± 0.03 | -1.04 ± 0.05 | 0.64 ± 0.02 | -0.24 ± 0.05 | 0.24 ± 0.02 |
| | | z | 0.1 | 1.2 | 0.2 | 0.9 | 0.2 | 0.2 |
| | | SD/abs(g) | 0.23 | 0.38 | 1.4 | 0.37 | 6.2 | 1.2 |
| 1.0 | 4 | LR | 1.06 ± 0.012 | 0.174 ± 0.007 | -0.69 ± 0.11 | 0.086 ± 0.006 | -0.018 ± 0.004 | 0.051 ± 0.006 |
| | | FD | 1.06 ± 0.011 | 0.179 ± 0.017 | -0.72 ± 0.05 | 0.088 ± 0.004 | -0.020 ± 0.004 | 0.052 ± 0.003 |
| | | z | 0.3 | 0.3 | 0.2 | 0.3 | 0.3 | 0.2 |
| | | SD/abs(g) | 0.035 | 0.12 | 0.45 | 0.22 | 0.59 | 0.32 |
| 0.1 | 4 | LR | 0.76 ± 0.54 | 3.7 ± 3.3 | -1.8 ± 0.8 | 0.54 ± 0.43 | 4.4 ± 4.3 | 0.17 ± 0.72 |
| | | FD | 1.34 ± 0.02 | 5.23 ± 0.14 | -1.00 ± 0.03 | 0.39 ± 0.03 | -1.06 ± 0.25 | -0.14 ± 0.04 |
| | | SD/abs(g) | 1.2 | 1.9 | 2.4 | 3.3 | 12 | 16 |

$t = 75$ (end of run):

| $\sigma$ | $N$ | quantity | $\log Q$ | $\langle u\rangle$ | $\langle\xi\rangle$ | $\log\sigma_u$ | $\log\sigma_\xi$ | $\log\sigma_u/\langle u\rangle$ |
|---|---|---|---|---|---|---|---|---|
| 0.3 | 1 | LR | -5.8 ± 2.4 | 17 ± 7 | 104 ± 54 | 3.6 ± 1.1 | -3.3 ± 3.7 | 0.95 ± 0.34 |
| | | FD | 1.27 ± 0.04 | 3.74 ± 0.17 | 5.3 ± 0.7 | 0.97 ± 0.02 | -0.17 ± 0.05 | 0.42 ± 0.02 |
| | | z | 2.9 | 2.0 | 1.8 | 2.3 | 0.8 | 1.6 |
| 0.3 | 4 | LR | 0.72 ± 0.58 | 3.7 ± 1.4 | 6.7 ± 7.2 | 1.07 ± 0.31 | 0.13 ± 0.27 | 0.52 ± 0.13 |
| | | FD | 1.24 ± 0.04 | 3.93 ± 0.17 | 4.9 ± 0.3 | 1.00 ± 0.02 | 0.003 ± 0.03 | 0.42 ± 0.01 |
| | | z | 0.9 | 0.2 | 0.2 | 0.2 | 0.5 | 0.8 |
| | | SD/abs(g) | 1.4 | 1.1 | 4.4 | 0.94 | 240 | 0.91 |
| 1.0 | 4 | LR | 0.994 ± 0.009 | 0.23 ± 0.03 | 1.06 ± 0.12 | 0.049 ± 0.004 | -0.011 ± 0.005 | 0.021 ± 0.002 |
| | | FD ($N$=1) | 1.00 ± 0.014 | 0.19 ± 0.09 | 0.74 ± 0.26 | 0.057 ± 0.017 | 0.024 ± 0.013 | 0.035 ± 0.016 |
| | | z | 0.4 | 0.5 | 1.1 | 0.5 | 2.5 | 0.8 |
| | | SD/abs(g) | 0.028 | 0.54 | 0.49 | 0.18 | 0.61 | 0.21 |
| 0.1 | 4 | LR | -15 ± 73 | 651 ± 2500 | 1270 ± 2100 | -16 ± 47 | -312 ± 290 | -33 ± 110 |
| | | SD/abs(g) | 140 | 370 | 1400 | 250 | 1900 | 25000 |
| 0.03 | 1 | LR | $\sim 10^{24}$ to $10^{26}$ (diverged) | | | | | |
| 0.01 | 1 | LR (4 runs) | $\sim 10^{29}$ to $10^{30}$ (diverged; $10^{11}$ already at $t = 30$) | | | | | |

Findings (measured):

1. The estimator is unbiased where it is stable. At $\sigma = 0.3$, $N = 4$ and at $\sigma = 1$, $N = 4$, all 12 comparisons at $t = 30$ and 11 of 12 at $t = 75$ are within 2 SE. The one miss ($\log\sigma_\xi$ at $\sigma = 1$, $t = 75$, $z = 2.5$) compares $N = 4$ LR with $N = 1$ FD, for a gradient of size 0.02. This validates the derivation of Section 2.1, including births and the $\delta x$ term.
2. At $\sigma = 0.3$, $N = 1$, the LR mean misses FD by up to 2.9 SE at $t = 75$. The weights are already feedback-amplified there (Section 3.1), so the ensemble is heavy-tailed and the $O(1/N)$ closure error is not small.
3. Single-run noise meets the 15 percent target only for $\log Q$ at $\sigma = 1$ (3 to 4 percent). At $\sigma = 0.3$, $N = 4$ it is 23 to 38 percent at $t = 30$ and 90 to 140 percent at $t = 75$ for the well-defined observables.
4. Noise falls faster than $N^{-1/2}$ when $N$ removes feedback (for $\log Q$ at $t = 30$, $\sigma = 0.3$: 1.0 at $N = 1$, 0.23 at $N = 4$) and close to $N^{-1/2}$ once feedback is gone ($\sigma = 1$: 0.099 to 0.035).

### 3.3 Bias against the deterministic model (reference ii) and physics distortion

Relative gradient bias $|g_\sigma - g_0|/|g_0|$, both from CRN FD at $h = 0.04$, $N = 1$. The last row shows how ill-defined $g_0$ itself is: the spread between $h = 0.02$ and $h = 0.04$ for $\sigma = 0$.

| $\sigma$ | $\log Q$ | $\langle u\rangle$ | $\langle\xi\rangle$ | $\log\sigma_u$ | $\log\sigma_\xi$ | $\log\sigma_u/\langle u\rangle$ |
|---|---|---|---|---|---|---|
| $t = 30$: 0.01 | 0.04 | 0.05 | 0.02 | 0.23 | 0.16 | 0.39 |
| $t = 30$: 0.1 | 0.22 | 0.14 | 0.04 | 0.57 | 4.8 | 0.77 |
| $t = 30$: 0.3 | 0.49 | 0.57 | 0.03 | 1.1 | 1.8 | 1.9 |
| $t = 30$: $g_0$ spread | 0.04 | 0.05 | 0.02 | 0.36 | 0.23 | 0.56 |
| $t = 75$: 0.01 | 0.13 | 0.92 | 0.24 | 0.12 | 0.69 | 0.15 |
| $t = 75$: 0.03 | 0.20 | 1.10 | 0.21 | 0.68 | 4.1 | 0.24 |
| $t = 75$: 0.1 | 0.28 | 2.8 | 0.82 | 0.32 | 0.07 | 0.98 |
| $t = 75$: 0.3 | 0.44 | 1.4 | 1.3 | 0.26 | 1.1 | 0.60 |
| $t = 75$: 1.0 | 0.55 | 1.0 | 1.05 | 0.95 | 0.96 | 0.99 |
| $t = 75$: $g_0$ spread | 0.16 | 1.18 | 0.16 | 0.15 | 0.63 | 0.41 |

Physics distortion at $t = 75$, ensemble mean at $a_0 = 2.5$, $N = 1$:

| $\sigma$ | trapped charge $Q$ (u > 3) | $\langle u\rangle$ | $\langle\xi\rangle$ | $\sigma_u$ |
|---|---|---|---|---|
| 0 | 0.0277 | 40.2 | -17.5 | 26.8 |
| 0.01 | 0.0245 (-12%) | 44.3 | -14.5 | 25.3 |
| 0.03 | 0.0198 (-29%) | 49.9 | -11.0 | 22.4 |
| 0.1 | 0.0126 (-55%) | 37.6 | -11.4 | 23.8 |
| 0.3 | 0.0216 (-22%) | 6.8 | -27.2 | 7.6 |
| 1.0 | 0.0544 (+96%) | 8.4 | -30.4 | 4.2 |

Findings (measured unless labeled):

1. **No $\sigma$ gives bias $\le$ 10 percent and noise $\le$ 15 percent together.** At $t = 30$, $\sigma = 0.01$ gives bias of 2 to 5 percent on $\log Q$, $\langle u\rangle$, $\langle\xi\rangle$, but there the weights grow at rate 2.9 at $N = 1$ and 1.4 at $N = 4$ ($10^{33}$ and $10^{15}$ at $t = 30$). At $t = 75$ even $\sigma = 0.01$ moves trapped charge by 12 percent and the $\log Q$ gradient by 13 percent.
2. **Trapping is the most noise-sensitive process.** Noise of $\sigma = 0.01$ (momentum spread 0.08 after 70 plasma periods) already removes 12 percent of the trapped charge. Trapping selects electrons near a separatrix, so any diffusion across it changes the trapped population. In 3D LWFA the same process (ionization-injected electrons crossing the trapping threshold) sets the bunch charge (estimated).
3. **At $\sigma \ge 0.3$ the wake no longer accelerates** ($\langle u\rangle$ falls from 40 to 7). The regime where LR is stable is a different physical system.
4. At $t = 75$ the deterministic gradient $g_0$ itself changes by 15 to 120 percent between $h = 0.02$ and $0.04$ at $N = 1$. A 10 percent bias target against $g_0$ is therefore only meaningful for $\log Q$, $\langle\xi\rangle$, $\log\sigma_u$ here.

### 3.4 Variants that do not help

1. **Noise on the bunch only, plasma pathwise** (seed 1, $N = 1$, runs in `work_wake/out/probe/lrb_*.json`, measured). The plasma pathwise tangent median grows to 123 to 265 at $t = 75$ for $\sigma = 0.03$ to $0.3$, the same as without noise (110 to 159). The bunch weights reach $10^5$ and the gradient reaches $10^4$ to $10^5$. The plasma in this wake is itself multi-stream after $t \approx 45$, so it cannot stay pathwise.
2. **Pathwise tangent of the noisy model** (8 seeds, `work_wake/out/path/`). Noise does not stop the per-realization tangent from growing: the median trapped-particle tangent at $t = 75$ is 33, 36, 38 for $\sigma = 0, 0.1, 0.3$. At $\sigma = 0.3$ the ensemble mean is consistent with FD ($\log Q$: $1.5 \pm 1.0$ vs $1.27$) but with single-run SD near 200 percent. At $\sigma \le 0.1$ the mean is $10^2$ to $10^3$ times the FD value.
3. **Drag (OU)**: rejected by the argument in Section 1.2 (derived, not run).

### 3.5 Joint-measure LR

The joint estimator $(J - b)S + \partial_W J\,\delta W_b$ was accumulated in every LR run with noise on all species. It has no feedback and never diverges, and its mean is consistent with FD where its noise allows a test (for example $\log Q$ at $\sigma = 1$, $t = 75$: $1.03 \pm 0.28$ vs $1.00 \pm 0.014$, measured). Its single-run SD/|g| at $t = 30$ is 10 to 15 for $\log Q$ and $\langle u\rangle$ at $\sigma = 0.3$ for both $N = 1$ and $N = 4$, and 1 to 4 for $\log Q$ at $\sigma = 1$. It does not fall with $N$, as derived in Section 2.2 (measured: $\sigma = 0.3$, $\log Q$, 9.8 at $N = 1$ and 15 at $N = 4$). Reaching 15 percent would need about $10^4$ runs at $\sigma = 0.3$ (estimated, $(12/0.15)^2$). It is never the right choice for a single-simulation gradient. It is the right choice only when $N$ is small and many independent runs exist anyway (shot-to-shot ensembles), where its finite-$N$ unbiasedness matters (derived).

### 3.6 Cost per parameter

Per extra parameter the LR estimator needs one weight-tangent deposit, one field-tangent solve, one gather of $\delta E$, and one evaluation of $\delta f_1$ per particle. It skips the particle tangent push. Measured wall times, $N = 1$, $\sigma = 0.3$, seed 7, one run each on the loaded machine (`work_wake/out/timing/`): primal 26 to 28 s, exact pathwise tangent 63 s (2.4 primal), mean-field LR 89 s (3.4 primal), LR plus joint score 107 s (4.1 primal). An earlier unloaded single run gave the same ordering (1.0, 2.2, 3.6 s). So in the NumPy harness the LR tangent costs 1.4 to 2.4 primal runs per parameter, more than the exact tangent, because of the extra masking and noise bookkeeping. In FBPIC the field-tangent solve and the deposit dominate, so the cost per parameter is close to one forward tangent, that is of order one simulation, not well below it (estimated). The $N$ needed to suppress the feedback adds to this: at $\sigma = 0.3$ the estimator needs $N \ge 4$ in this harness, and the $N$ needed grows like $\sigma^{-2}$ (estimated from Section 3.1).

## 4. Conclusion

1. **Correctness (measured).** The mixed estimator of Section 2.1 (Girsanov weights for momentum, a non-growing position tangent for the explicit $A$ dependence of the position update, deterministic birth tangents) matches the CRN FD of the noisy model within 2 SE where its variance is finite.
2. **Variance (measured and derived).** Without feedback the weights grow like $\sigma^{-1}\sqrt t$, as designed. With the self-consistent weight deposit, the finite-$N$ estimate of $\delta\rho$ feeds sampling noise back into the weights. The result is exponential growth with rate $\approx \lambda_{\rm phys}(\sigma) + c(\sigma)/N$, $c(\sigma)$ rising steeply as $\sigma$ falls. As $\sigma \to 0$ the physical part tends to the chaotic tangent growth rate. So option C does not escape exponential growth; it trades chaos for a noise level, and the trade is unfavourable here.
3. **Bias (measured).** Trapping is too sensitive to momentum noise. There is no window between "noise small enough to keep the physics" ($\sigma \lesssim 0.01$) and "noise large enough for a bounded, single-run estimate" ($\sigma \gtrsim 0.3$ at $N = 4$, $\sigma \approx 1$ for 15 percent noise). The gap is a factor of 30 to 100 in $\sigma$, that is $10^3$ to $10^4$ in $N$ by the $\sigma^{-2}$ scaling (estimated).
4. **What would have to change for C to work (not tested).** (a) A variance-reduction step on the weights, such as periodic regression of $\omega$ onto a smooth function of $(\xi, u)$ (weight smoothing as in collisional $\delta f$ codes). This reintroduces a phase-space mesh and its bias, and in the trapped region $\delta f/f$ is filamented, so the bias may be large. (b) Noise placed only in degrees of freedom that do not decide trapping. In this wake every electron that matters passes the trapping separatrix, so no such placement was found.

## Files

- `work_wake/wake_lr.py`: harness (primal, exact pathwise tangent, mean-field LR, joint LR).
- `work_wake/fd_batch.py`: CRN FD of the noisy model for one seed.
- `work_wake/analyze.py`, `work_wake/tables.py`, `work_wake/growth.py`, `work_wake/wtime.py`, `work_wake/show.py`: analysis.
- `work_wake/out/fd/`, `work_wake/out/lr/`, `work_wake/out/path/`, `work_wake/out/grow/`, `work_wake/out/probe/`: raw run outputs (JSON).
- `work_wake/out/tables_t30.0.md`, `work_wake/out/tables_t45.0.md`, `work_wake/out/tables_t75.0.md`, `work_wake/out/analyze_t*.txt`: full tables.
