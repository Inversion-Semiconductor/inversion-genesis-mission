# R0 section: go/no-go tests for R1 (Debye reproduction) and R2 (1D cold-beam trapping)

Author: R0 derivation agent "tests". Date: 2026-10-31. Status: design plus CPU pilot measurements.

Claim labels: **[paper]** from Chung, Bond, Cyr, Freund, JCP 400, 108969 (2020), OSTI 1570259, checked against the page images; **[derived]** worked out here; **[measured]** by a pilot script in `r0/work_tests/` (full paths in Section 9); **[estimated]** an engineering estimate.

## 0. Summary

1. R1 is feasible and cheap. A NumPy pilot of the Debye case reproduces the paper's QoI statistics and its chaotic particle-exact sensitivity in minutes on CPU. Measured: $\bar{\mathcal J} = 619.65 \pm 0.04$ against $619.875$ (Table B.3), and $\sigma_{\mathcal J} = 0.210$ against $0.2223$. A first particle-pdf implementation gives a regular sensitivity that agrees with the continuum slope $12.4 \pm 0.2$ (two Vlasov meshes and PIC secants): non-collocated $11.79 \pm 0.35$, collocated $12.86 \pm 0.48$ at $N = 10^5$. Its seed-to-seed SD (10% of the slope) is about 3.5 times the paper's Fig. 9b value; R1 must find why. The remaining R1 work is that variance check, the $N^{-1/2}$ test over two decades, the cost profile, and variant R-v.
2. R2 is feasible as a 1D relativistic electrostatic wake with a prescribed ponderomotive driver and expected-value (EV) ionization injection. The pilot traps an EV beam at $t \approx 19$ to $25\,\omega_p^{-1}$. After trapping, the trajectory tangents grow exponentially at $\lambda \approx 0.23\,\omega_p$ (constant over 3.3 decades, $t = 40$ to 75). At $t_f = 75$ the exact derivative column has the wrong sign for 5 of 6 beam observables and fails the guards ($\cos = -0.28$, $\kappa = 1.06$), while the FD secant at $h = 0.01$ stays regular ($\cos = 0.987$ against the slice fit). This is the P7 failure in miniature, at about 30 s per run on one CPU core. At $t_f = 58$ the failure is only partial, so R2 uses $t_f = 75$.
3. The D32 fixed-point shift $\delta$ alone cannot gate R3. At a least-squares solution, $\delta$ is almost independent of the magnitude of $J_{AD}$; on P7 it is small because $\lVert r_w\rVert / \lVert J_{sm}\rVert = 0.013$. In a null test, 19.5% of random 11-vectors pass $\delta \le 0.005$. Section 6 adds a Gauss-Newton contraction bound $\kappa \le 0.5$ and a cosine bound $\cos \ge 0.9$, checked at the solution and at two off-solution points. The P7 AD column then fails ($\kappa = 0.999$, $\cos = 0.25$). FD columns with $h$ from $10^{-4}$ to 0.02 pass; the over-curved $h = 0.05$ secant fails ($\cos = 0.33$). No random column passes in $2\cdot10^5$ trials.
4. Effort: R1 about 2 days, R2 about 5 days (R2b, an optional 1D electromagnetic variant, adds 2 to 3 days).

## 1. What the paper demonstrates (Sections 4 to 7, PDF pages 20 to 38)

### 1.1 Debye-shielding setup [paper, Section 4, pp. 20 to 23]

| Item | Value | Source |
|---|---|---|
| Model | 1D-1V electrostatic Vlasov-Poisson, PIC discretization (11a to 11f) | p. 8, p. 21 |
| Ions | Uniform, stationary, $n_i = n_{e0}$ | p. 21 |
| Electrons | $f(x,v,0) = \frac{n_{e0}}{\sqrt{2\pi}v_T}\exp(-v^2/2v_T^2)$ (57); $v_p(0) = v_T\gamma_p$, $\gamma_p \sim \mathcal N(0,1)$ (58) | p. 21 |
| Domain | Periodic, $L = 20\lambda_D$ (59) | p. 22 |
| External charge | $\rho_{ext} = -\frac{2q_e}{L} + \frac{2q_e}{\sqrt{2\pi}w_q}\exp\!\big[-\frac{(x-x_c)^2}{2w_q^2}\big]$ (60), $w_q = 0.1L$, $x_c = L/2$ | p. 22 |
| Discretization | $N = 10^5$ particles, $N_m = 64$ mesh points, $\Delta t = 0.05/\omega_p$ | p. 22 |
| Shapes | $S_x = b_1$, $S_v = b_2$; interpolation $b_{l+1} = b_2$; source uses $\nabla_v b_3$ (A.2) | p. 15, p. 51 |
| Time integration | Second-order kick-drift-kick (Fig. 4) | p. 21 |
| Parameter | $\theta = v_T/v_0$, nominal $\theta = 1.5$; $v_0 = \lambda_D\omega_p$ | p. 22 to 23 |
| QoI | $\mathcal J = \frac1{t_f}\int_0^{t_f}\!\!\int\!\!\int (x-x_c)^2 f\,dx\,dv\,dt$ (61); discrete $\mathcal J_D = \frac{1}{N_t}\sum_k\sum_p W_p (x_p^k - x_c)^2$ (62); $t_f = 150/\omega_p$ | pp. 22 to 23, Fig. 7 caption |
| QoI bounds | $T\to\infty$: $L^3/12 = 666.7$; $T\to 0$: $L^3/12 - L^2/6 + 2w_q^2 = 608.0$ | p. 23 |
| Particle-pdf source | $\partial_\theta\mathcal V_1 = (v^2/v_T^2 - 1)\frac{n_{e0}}{\sqrt{2\pi v_T^2}}\exp(-v^2/2v_T^2)\,\delta(t)$ (65), an initial condition for $\partial f$ | p. 23 |
| Particle-exact source | $\partial_\theta v_p = \gamma_p$ at $k = 1$ (66), same random numbers | p. 23 |
| Particle-pdf mesh | Non-collocated scheme; $N_m = 64$ velocity cells on $[-9v_0, 9v_0]$; $\Delta t$, $\Delta x$ as PIC; $M = 2\cdot10^5$ sensitivity particles uniform in $x$ and $v$ (support condition (43)) | p. 25 |
| Particle-pdf QoI sensitivity | $(\partial\mathcal J/\partial\theta)_D = \frac1{N_t}\sum_k\sum_s \hat W_s(\hat x_s^k - x_c)^2$ (63) | p. 23 |

The scheme per step is (51a) to (54a) [paper, p. 19 to 20]: push sensitivity particles like PIC particles; deposit $\partial\rho$ from $\hat W_s$ (51b); solve for $\partial\phi$, $\partial E$ (51c, 51d); evaluate $(\partial E\cdot\nabla_v f)_{i,i_v} = \sum_p \frac{W_p}{\Delta x\Delta v} b_{l+1}(\xi_{p,i})\,\partial E_i\,\nabla_v b_{l+2}(\zeta_{p,i_v})$ (52); deposit $\hat n$ of the sensitivity particles (53a); gather $h_s = \sum \frac{\mathcal H}{\hat n} b\,b$ (53b); update $d\hat W_s/dt = -h_s$ (54a).

### 1.2 References and results to match [paper, Section 5, pp. 24 to 27; Appendix B, pp. 52 to 53]

1. Accuracy of particle-exact against FD (67) to (69): $\epsilon(\Delta\theta)$ follows $\mathcal O(\Delta\theta)$ only for $\Delta\theta$ that shrinks with time. The prediction range $\Delta\theta_p$ at $\epsilon_p = 10^{-6}$ falls from about $10^{-6}$ at $t\omega_p = 0.1$ to about $10^{-14}$ near $t\omega_p = 100$ to 150 (Fig. 6b, read from the plot). "The perturbation needs to be machine precision after about 100 plasma periods" (p. 24).
2. Brute force: 1000 PIC runs with $\theta_k$ uniform on $[0.5, 3.5]$, all with the same initial particle distribution, plus a second-order flux-limited finite-volume (FV) Vlasov solution with $\Delta x = 0.02\lambda_D$, $\Delta v = 0.018v_0$, Strang splitting, Courant number 0.5 (70) (p. 24). Fig. 7: $\mathcal J(\theta)$ rises from about 610 at $\theta = 0.5$ to about 641 at 3.5 and is about 620 at 1.5. The particle-pdf slope at 1.5 matches the FV sensitivity of (23a); the particle-exact slope is a near-vertical line (inset, p. 26).
3. Sensitivity distribution: Fig. 8 compares $\partial f$ from particle-pdf and FV, colour range about $-1$ to $0.5$, with the negative core near $v = 0$ and positive bands near $|v| \approx 3$ to $4$ (p. 26).
4. Statistics: Fig. 9a, 10^4 seeds; the particle-exact standard deviation grows exponentially after $t\omega_p \approx 10^2$ and reaches about $10^{23}$ to $10^{24}$ at $t\omega_p \approx 700$ (read from the plot); the particle-pdf standard deviation tracks that of $\mathcal J_D$. Fig. 9b, 10^3 seeds, $N_p = 10^4$ to $10^6$: all three particle-pdf variants fall as $N^{-1/2}$; collocated has the largest constant (p. 27).
5. Table B.3 (p. 53), $N = 10^5$, $t\omega_p = 150$, 10^4 seeds: $\mathcal J$: $\mu = 619.875$, $\sigma = 0.2223$ ($N_m = 64$, $\omega_p\Delta t = 0.05$); 619.838, 0.2226 ($N_m = 256$); 619.700, 0.2205 ($\omega_p\Delta t = 0.01$). Particle-exact $\partial_\theta\mathcal J$: $\mu = 26.4$, $\sigma = 984$; 9.7, $3.15\cdot10^3$; 9.16, 997. Fig. B.21: particle-exact $\sigma$ falls with $N$ (about $2\cdot10^5$ at $N = 10^4$ to about 40 at $10^6$, read from the plot) but stays far above $\sigma_{\mathcal J}$.

### 1.3 Non-commutability metrics [paper, Section 6, pp. 28 to 35]

The paper splits $\Delta n_N/\Delta\theta = \Delta n/\Delta\theta + \Delta\epsilon_N/\Delta\theta$ (77), with $\Delta\epsilon_N \sim \Delta\theta\,e^{\lambda t}$ (78) and $\lim_{N\to\infty}\Delta\epsilon_N = 0$ (79). Particle-pdf is stable when the continuum perturbation stays Lyapunov stable, $\Delta n \sim \Delta\theta$ (81), (92). The tests use $\overline{\Delta x_p}$ (83, 86b) and the Wasserstein-1 distance $W_1(\Delta n_N)$ (85) at $\theta = 1.5$, $\Delta\theta = 10^{-9}$, $N \in \{10^2, \ldots, 10^5\}$. Fig. 11a: $\overline{\Delta x_p}$ grows to the $L/4$ limit at an $N$-dependent rate; $W_1$ departs from $\overline{\Delta x_p}$ at $\overline{\Delta x_p} = L/N$ and saturates. Fig. 11b: the $W_1$ saturation at $t\omega_p = 450$ scales as $N^{-1/2}$ (about $0.3\lambda_D$ at $N = 10^2$ to $0.012\lambda_D$ at $10^5$, read from the plot). Fig. 11c: the FV value converges to about $2\Delta\theta$. Summary (93): $W_1(\Delta n_{FV}) \sim \Delta\theta \lesssim W_1(\Delta n) < W_1(\Delta n_N) \sim N^{-1/2} \ll \overline{\Delta x_p} \sim L$.

### 1.4 Cost [paper, Section 7, pp. 35 to 38]

Total time relative to PIC on one Xeon E5-2695 core, $N = 10^5$: 2.96 non-collocated, 2.10 collocated, 1.89 particle addition (p. 35). The extra cost is in the source evaluation (52), the $\hat n$ deposit (53a), and the source weights (53b) (Fig. 12, p. 36). Operation counts scale with $N$, not with the phase-space mesh (Tables 1 and 2, pp. 37 to 38); the paper flags the memory of a dense phase-space mesh as an unsolved issue (p. 38).

## 2. Pilot measurements (CPU, NumPy, float64)

### 2.1 R1 baseline [measured, `debye_pilot.py`]

My implementation: quadratic B-spline (b_2) deposit and gather, 3-point Laplacian solved by FFT, centered-difference $E$, kick-drift-kick, uniform random positions (the paper does not state the position loading).

| Quantity | Pilot | Paper |
|---|---|---|
| Peak-to-minimum of $\phi_{ext}$ | 3.41 $k_BT_0/e$ (discrete) | about 3.2 (Fig. 5b, read) |
| $\bar{\mathcal J}$, 24 seeds, $N = 10^5$ | 619.65 $\pm$ 0.043 (SE) | 619.875 (Table B.3) |
| $\sigma_{\mathcal J}$ | 0.210 | 0.2223 |
| Wall time, one run, 3000 steps, one core | 6.9 s | not given |
| CRN secant $(\mathcal J(\theta+h)-\mathcal J(\theta-h))/2h$, 12 seeds | $h = 0.1$: 12.43 $\pm$ 0.13; $h = 0.03$: 12.39 $\pm$ 0.30 | Fig. 7 slope about 13 (read) |

The mean differs from Table B.3 by 0.23, about 5 standard errors. The paper's own $\Delta t$ change moves the mean by 0.175 (Table B.3), so discretization details explain a difference of this size. R1 tolerances below allow 0.5.

### 2.2 R1 particle chaos [measured, `debye_pilot.py twin`]

Twin runs at $\theta = 1.5$ and $1.5 + 10^{-9}$ with the same random numbers. $\overline{\lvert\Delta x_p\rvert}$:

| $N$ | Growth window | Rate $\lambda$ [$\omega_p$] | Saturation |
|---|---|---|---|
| $10^3$ | $t = 25$ to 100 | 0.18 | 4.7 to 4.9 ($L/4 = 5$) |
| $10^4$ | $t = 75$ to 175 | 0.11 | 3.9 to 4.0 at $t = 275$ (still rising) |
| $10^5$ | $t = 100$ to 250 | 0.077 | 0.23 at $t = 275$ (still rising) |

The rate falls with $N$ and the saturation is $L/4$, as in Fig. 11a. The twin derivative of the instantaneous $J(t)$ at $N = 10^5$ is 27 at $t = 50$, $2\cdot10^4$ at $t = 150$, and $8\cdot10^7$ at $t = 250$.

### 2.3 R1 particle-pdf first cut [measured, `debye_pdf.py`]

Non-collocated scheme on the paper's mesh (64 $x$ cells, 64 $v$ cells on $[-9, 9]$), $M = 2N$ sensitivity particles uniform in $(x, v)$, initial weights from (65), explicit update of (54a). Collocated scheme: the plasma particles carry the weights, with initial weights $W_p(\gamma_p^2 - 1)/\theta$ (the score of (58)). The continuum reference is the semi-Lagrangian Vlasov solver with central FD. Estimates of $\partial\mathcal J/\partial\theta$ at $\theta = 1.5$, $t_f = 150$:

| Estimate | $N$ | Seeds | Mean | SD | SE |
|---|---|---|---|---|---|
| Vlasov FD, $256\times512$, $h = 0.02$ | n/a | n/a | 12.17 | n/a | n/a |
| Vlasov FD, $256\times512$, $h = 0.1$ | n/a | n/a | 12.50 | n/a | n/a |
| Vlasov FD, $512\times1024$, $h = 0.02$ | n/a | n/a | 12.41 | n/a | n/a |
| PIC CRN secant, $h = 0.1$ | $10^5$ | 12 | 12.43 | 0.47 | 0.13 |
| PIC CRN secant, $h = 0.03$ | $10^5$ | 12 | 12.39 | 1.06 | 0.30 |
| Particle-pdf, non-collocated | $10^4$ ($M = 2\cdot10^4$) | 24 | 14.18 | 5.43 | 1.11 |
| Particle-pdf, non-collocated | $10^5$ ($M = 2\cdot10^5$) | 12 | 11.79 | 1.22 | 0.35 |
| Particle-pdf, collocated | $10^5$ | 12 | 12.86 | 1.67 | 0.48 |
| Particle-exact (twin, instantaneous $\partial_\theta J(t)$ at $t = 150$, one seed) | $10^5$ | 1 | $2\cdot10^4$ | n/a | n/a |

Findings:

1. Both particle-pdf variants are regular and agree with the continuum value within 2 SE + 5%. The two Vlasov meshes give 12.17 and 12.41 (2% apart), and their $\bar{\mathcal J}$ is 619.68 and 619.70. The PIC CRN secants (12.4) and my reading of Fig. 7 (about 13) agree. I take $12.4 \pm 0.2$ as the continuum slope.
2. The SD falls by 4.45 for a tenfold $N$ (slope $-0.65$, with an uncertainty of about 0.1 from 12 to 24 seeds), close to $N^{-1/2}$.
3. The SD at $N = 10^5$ is 10% of the slope. Fig. 9b shows about 0.35 for the non-collocated variant (read from the plot), so my first cut is about 3.5 times noisier than the paper. The cause is not known. Candidates: the explicit weight update, the $\hat n$ division in cells with few sensitivity particles, and the source-mesh noise. R1 must find it (criterion R1-6).
4. Cost in NumPy, 100 steps at $N = 10^5$: 12× a PIC step (non-collocated) and 5× (collocated). In the loaded ensemble, one non-collocated run took 1833 s against about 7 s for PIC unloaded; that wall time is not a clean ratio, because 60 or more processes shared 32 cores.

### 2.4 R2 pilot [measured, `r2_pilot.py`]

The pilot is the R2a model of Section 5.1 (lab frame, NumPy, linear shapes, Gauss's law integrated from the unperturbed front). Driver scan at $t = 60$:

| $a_0$ | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 |
|---|---|---|---|---|---|
| Born weight (EV electrons) | 0.012 | 0.047 | 0.093 | 0.129 | 0.147 |
| Born weight with $u > 8$ | $8.6\cdot10^{-5}$ | $6.2\cdot10^{-3}$ | $1.8\cdot10^{-2}$ | $2.7\cdot10^{-2}$ | $2.8\cdot10^{-2}$ |
| Peak $\lvert E\rvert$ | 0.61 | 0.90 | 1.18 | 1.44 | 1.68 |
| Wall time, one core | 18 s | 22 s | 23 s | 24 s | 24 s |

At $a_0 = 2.5$ births occur at $t \approx 12$ to 19 (about $1.4\cdot10^5$ EV macroparticles) and trapping at $t \approx 19$ to 25 (about $5.3\cdot10^4$ with $u > 3$). Some plasma electrons are also injected at the entrance ramp; the observables exclude them, as Twin-S excludes non-`elec_from_N` species.

Twin runs ($a_0 = 2.5$ and $2.5 + 10^{-9}$; $10^{-7}$ gives the same numbers to 2 or 3 digits). Mean per-particle sensitivities:

| $t$ | 25 | 40 | 46 | 52 | 58 |
|---|---|---|---|---|---|
| Plasma $\overline{\lvert\partial x/\partial a_0\rvert}$ | 0.33 | 1.10 | 2.42 | 14.1 | 42.5 |
| EV $\overline{\lvert\partial x/\partial a_0\rvert}$ | 1.65 | 12.6 | 24.0 | 43.5 | 101 |
| Trapped $\overline{\lvert\partial u/\partial a_0\rvert}$ | 2.9 | 13.7 | 18.4 | 26.5 | 34.5 |
| Exact $\partial\langle u\rangle/\partial a_0$ (selection $u_c = 3$, $\delta_u = 0.3$) | 2.5 | 7.0 | 12.3 | 10.4 | $-20.5$ |

The local growth rate from $t = 46$ to 58 is 0.24 (plasma) and 0.12 (EV) per $\omega_p^{-1}$, and it increases with time. The trapped-momentum tangent grows about linearly, as expected from acceleration. The exact $\partial\langle u\rangle/\partial a_0$ changes sign after $t \approx 46$.

Slice at $t = 58$ (21 points over $a_0 = 2.45$ to 2.55, one deterministic loading), derivatives at 2.5:

| Observable | Cubic-fit slope | FD $h = 10^{-2}$ | FD $10^{-3}$ | FD $10^{-4}$ | Exact (twin $10^{-9}$) |
|---|---|---|---|---|---|
| $\log Q$ | 1.89 | 1.95 | 2.02 | $-5.66$ | 2.72 |
| $\langle u\rangle$ | $-1.38$ | $-2.71$ | $-4.07$ | 177 | $-20.5$ |
| $\langle\xi\rangle$ | $-14.3$ | $-16.5$ | $-23.6$ | 152 | $-40.2$ |
| $\log\sigma_u$ | 0.748 | 0.765 | 0.774 | $-1.63$ | 0.983 |
| $\log\sigma_\xi$ | 0.232 | 0.303 | 0.421 | $-3.92$ | 0.866 |
| $\log(\sigma_u/\langle u\rangle)$ | 0.797 | 0.863 | 0.920 | $-8.00$ | 1.72 |

1. The FD at $h = 10^{-4}$ straddles a discrete birth-threshold jump (the $\epsilon$ cut), the 1D analog of the P7 crossings with $K(h) > 0$.
2. The selected beam spans several wake buckets ($\langle\xi\rangle \approx -17$, $\sigma_\xi \approx 15.5$). R2 should add a smooth $\xi$ window on the first bucket.
3. Seed ensemble (8 jittered loadings at $a_0 = 2.48$ and 2.52): seed-to-seed SD of $v$ is (0.012, 0.29, 0.28, 0.0044, 0.0068, 0.015). Whitened by these SDs, the per-seed secant noise is about 25 per component, against a whitened reference column norm of 243. A whitened column with 5% SE therefore needs about 25 seeds per $h$. Per-observable 5% SE on the weak $\langle u\rangle$ slope would need about $10^4$ seeds; report its SE instead.
4. Whitened column metrics against the cubic fit, with a synthetic target at $a_0^\ast = 2.53$ ($\lVert r_w\rVert = 7.4$): exact column $\cos = 0.944$, $\rho = 1.61$, $\kappa = 0.415$, $\delta = 0.011$; FD $h = 0.01$: $\cos = 0.999$, $\kappa = 0.04$; FD $h = 10^{-4}$: $\cos = -0.69$, $\kappa = 1.13$.

So at $t = 58$ the pilot shows the mechanism (growing tangents, an erratic exact derivative of the weak observables) but not the full Twin-S failure: the whitened exact column still passes the Section 6 guards, because the dominant whitened components ($\log Q$, $\log\sigma_u$) are still nearly regular.

Extension to $t = 75$ (3000 steps; same model; whitening still by the $t = 58$ seed SDs, which is an approximation):

| $t$ | 40 | 50 | 60 | 65 | 70 | 75 |
|---|---|---|---|---|---|---|
| Plasma $\overline{\lvert\partial x/\partial a_0\rvert}$ | 1.1 | 7.0 | 82 | 250 | 666 | 2310 |
| EV $\overline{\lvert\partial x/\partial a_0\rvert}$ | 12.6 | 34 | 144 | 425 | 1110 | 2890 |

A log-linear fit gives $\lambda = 0.221$ ($t = 40$ to 58) and 0.228 ($t = 58$ to 75) per $\omega_p^{-1}$ for the plasma electrons, and 0.11 and 0.20 for the EV electrons. The rate is constant over 3.3 decades, so the growth is exponential, not polynomial.

| Observable at $t = 75$ | Cubic-fit slope | FD $h = 10^{-2}$ | FD $5\cdot10^{-3}$ | FD $10^{-3}$ | Exact (twin $10^{-9}$) |
|---|---|---|---|---|---|
| $\log Q$ | 2.30 | 2.56 | 1.96 | 3.29 | 0.29 |
| $\langle u\rangle$ | $-14.0$ | $-23.3$ | $-2.05$ | $-46.5$ | 67.1 |
| $\langle\xi\rangle$ | $-21.8$ | $-29.3$ | $-4.87$ | $-24.5$ | 111.5 |
| $\log\sigma_u$ | 0.767 | 0.885 | 0.647 | 1.02 | $-0.477$ |
| $\log\sigma_\xi$ | 0.226 | 0.573 | $-0.443$ | $-0.494$ | $-8.63$ |
| $\log(\sigma_u/\langle u\rangle)$ | 1.12 | 1.46 | 0.699 | 2.18 | $-2.15$ |

Whitened column metrics against the cubic fit: exact $\cos = -0.28$, $\rho = 4.8$, $\kappa = 1.06$, $\delta = 0.044$; FD $h = 0.01$: $\cos = 0.987$, $\kappa = 0.18$, $\delta = 0.0045$; FD $5\cdot10^{-3}$: $\cos = 0.889$, $\kappa = 0.087$; FD $10^{-3}$: $\cos = 0.930$, $\kappa = 0.39$.

At $t = 75$ the pilot reproduces the P7 failure in miniature: the exact column has the wrong sign for 5 of 6 observables, as P7 had for 6 of 11, while the large-$h$ secant stays regular for the whitened column. The single-loading slices are rough for $\langle u\rangle$ and $\langle\xi\rangle$ (cubic-fit residual 0.10 and 0.084), so the R2 reference must be an ensemble. **R2 uses $t_f = 75$.** The pathology needs about $\lambda(t_f - t_{trap}) \approx 10$; Twin-S has about 15 (Section 8, item 4).

## 3. What the dipole identity means for the tests [derived]

1. For relativistic $\mathbf v = \mathbf u/\gamma$, $\partial_{u_i}(\mathbf v\times\mathbf B)_i = \epsilon_{ijk}B_k(\delta_{ij}/\gamma - u_iu_j/\gamma^3) = 0$. So $\nabla_u\cdot(\delta\mathbf E + \mathbf v\times\delta\mathbf B) = 0$, and the Track R source is a divergence: $-q(\delta\mathbf E + \mathbf v\times\delta\mathbf B)\cdot\nabla_u f = -\nabla_u\cdot\big(f\,q(\delta\mathbf E + \mathbf v\times\delta\mathbf B)\big)$. In R1, $\delta E$ does not depend on $v$, so $\mathcal H = -\partial_v\big(f\,\tfrac{q}{m}\partial E\big)$ exactly. In the R2 ponderomotive model the single-particle flow $\dot x = u/\gamma$, $\dot u = -E - \partial_x A/(2\gamma)$ is Hamiltonian with $H = \gamma(x,u,t) - \phi(x,t)$, $\gamma = \sqrt{1 + u^2 + A}$, $E = -\partial_x\phi$. Then $\partial_x(u/\gamma) + \partial_u(-\partial_xA/2\gamma) = -u\partial_xA/2\gamma^3 + u\partial_xA/2\gamma^3 = 0$, so $\nabla_z\cdot\partial_\theta\dot z = 0$ and the source is $-\nabla_z\cdot(f\,\partial_\theta\dot z)$ with both an $x$ and a $u$ component inside the driver.
2. Consequence for R-v. A dipole carried by a displacement $\xi$ is the Lagrangian tangent: $\delta f = -\nabla_z\cdot(f\xi)$. Over one step, $\xi_u = \Delta t\,\tfrac{q}{m}\partial E$ and $\xi_x = \mathcal O(\Delta t^2)$, and the projected source is $-\partial_v(f\,\Delta t\,\tfrac qm\partial E)$. The deposit of $W_p\,\partial E\,\partial_v S_v$ is eq. (52). So **R-v with projection interval $T_p = \Delta t$ is the particle-pdf method** (collocated if the weights stay on the plasma particles). R-v differs only for $T_p > \Delta t$: it transports the dipole along characteristics with the linearized flow, including the stretching $\partial\dot z/\partial z\,\xi$, before it converts it to a scalar weight. The tests must therefore sweep $T_p$ and expect a flat error for $\lambda T_p \lesssim 1$ and growth beyond.
3. Initial-condition check. In R1 the Lagrangian seed is $\xi_v(0) = \gamma_p = v_p/\theta$ (66). Projection gives $-\partial_v(f_0 v/\theta) = (v^2/\theta^2 - 1)f_0/\theta$, which is (65). R1 must reproduce (65) from the projection of (66) to within the mesh error; this is a unit test of the projection kernel.
4. Conservation checks. $\int\mathcal H\,dv = 0$ at each $x$, and $\int \partial_\theta f_0\,dv = 0$. So $\sum_s \hat W_s(t)$ must stay at 0 to within its noise, and the $v$-moment gives $\int v\,\mathcal H\,dv = -\tfrac qm\partial E\,n$, the linearized momentum source. The mesh division $h = \mathcal H/\hat n$ (42) does not preserve either moment exactly, so their drift measures the discretization error.
5. Projection consistency. Just before and just after a projection, the observable sensitivity can be computed two ways: $\sum_p W_p\nabla J(z_p)\cdot\xi_p$ and $\sum_s \hat W_s J(\hat z_s)$. In the continuum they are equal ($\int J\,\delta f = \int f\,\xi\cdot\nabla J$). Their difference per projection is the projection error and is cheap to log.
6. Why the weight form is regular. In the continuum, $\int f\,\xi\cdot\nabla J$ is finite even when $\lvert\xi\rvert \sim e^{\lambda t}$, because $\xi$ changes sign across folded filaments and cancels against smooth $f$. With $N$ particles the cancellation is incomplete, and the error is about $N^{-1/2}e^{\lambda t}$; this is (78) and (91). A scalar $\delta f$ is transported on the same characteristics without stretching; only its gradients filament. So the observable error stays near $N^{-1/2}$ plus the mesh bias of $\nabla_u f$.
7. Cold-beam consequence. A cold stream has $f \propto \delta(u - U(x))$. Its dipole is a momentum displacement of a delta function, which the weight form can represent only with a smoothed $\tilde f = f * K_{\sigma_u}$. In R1-like ES problems, $\partial E$ does not depend on $u$, so the computed source is the exact source convolved in $u$ with $K_{\sigma_u}$. The bias is then $\mathcal O(\sigma_u^2)$ where the map from source momentum to the observable is smooth, and $\mathcal O(\sigma_u)$ where the smoothed dipole straddles the trapping separatrix. R2 test (e) must measure this exponent.
8. A laminar cold plasma has regular Lagrangian tangents. R-v can therefore keep exact tangents for laminar particles and project only the particles that stretch (the trapped beam). This "selective" variant, R-v-s, avoids the $\sigma_u$ bias on the cold background, which is the dominant concern for Twin-S (Track R open problem 1). R2 must test it.

## 4. R1 design: Debye reproduction

### 4.1 Implementation

1. Language: NumPy, float64, CPU. The pilot (`debye_pilot.py`, `debye_pdf.py`) is the starting point. `np.bincount` deposits are deterministic, so particle-exact runs are reproducible to the bit. PyTorch adds nothing here. Warp is excluded by the task.
2. Particle-exact: hand-written tangent of the kick-drift-kick step (19a to 19h), seed (66). Validate against twin FD at $\Delta\theta = 10^{-9}$ up to $t\omega_p = 50$.
3. Particle-pdf: non-collocated (the paper's main variant), collocated, and optionally particle addition. Use the paper's mesh: 64 $x$ cells, 64 $v$ cells on $[-9, 9]$, $M = 2\cdot10^5$.
4. Variant R-v: Lagrangian tangent from item 2, projected every $T_p$ steps onto the sensitivity-particle weights by $\delta\hat W_s \mathrel{+}= -\Delta x\Delta v\,[\nabla_z\cdot(f\xi)]/\hat n$ evaluated as in (53b), then $\xi \leftarrow 0$.
5. Continuum reference: a semi-Lagrangian Vlasov solver (Strang splitting, spectral shifts in $x$ and $v$) on $256\times512$ and $512\times1024$ meshes, $v \in [-12, 12]$, with central FD at $h = 0.02$. The continuum map is smooth, so FD of the Vlasov solution is a valid reference. The pilot has this solver (`run_vlasov`).
6. Brute-force reference: CRN secant ensembles of PIC at $h \in \{0.1, 0.03\}$ and the 1000-point $\theta$ scan of Fig. 7 (7 s per run).

### 4.2 Runs

| Run set | Count | CPU time (one core) |
|---|---|---|
| PIC baseline, $N = 10^5$ | 64 seeds | 7.5 min |
| Brute-force scan, 1000 $\theta$, one seed | 1000 | 2 h |
| CRN secants, $h \in \{0.1, 0.03\}$, $N = 10^5$ | 64 seeds × 2 h × 2 | 30 min |
| Particle-exact + twin FD, $N \in \{10^3, 10^4, 10^5\}$, to $t\omega_p = 450$ | 3 × 4 | 15 min |
| Particle-pdf (non-collocated, collocated), $N \in \{10^4, 10^5, 10^6\}$ | 32 seeds × 3 × 2 | about 40 h with the NumPy pilot; about 4 h after vectorizing the phase-space deposit (see 4.4) |
| R-v, $T_p \in \{1, 10, 100, 1000\}$ steps, $N = 10^5$ | 16 seeds × 4 | about 4 h |
| Vlasov reference, two meshes, 3 runs each | 6 | about 10 min ($256\times512$) and 25 min ($512\times1024$) per set of 3 on a loaded machine [measured] |

On 30 cores the whole R1 campaign fits in about 2 h of wall time with the vectorized deposit [estimated].

### 4.3 Pass criteria

| ID | Test | Pass |
|---|---|---|
| R1-1 | Baseline QoI | $\lvert\bar{\mathcal J} - 619.875\rvert \le 0.5$ and $\sigma_{\mathcal J} \in [0.17, 0.28]$ at $N = 10^5$ |
| R1-2 | Particle-exact is correct | Relative difference to twin FD at $\Delta\theta = 10^{-9}$ below $10^{-5}$ up to $t\omega_p = 50$ |
| R1-3 | Particle-exact is chaotic | Twin $\overline{\lvert\Delta x_p\rvert}$ grows exponentially with $\lambda(N)$ falling with $N$; $\sigma$ of $\partial_\theta\mathcal J_D$ at $t\omega_p = 150$ at least $10^3\sigma_{\mathcal J}$ (paper: 984 against 0.22) |
| R1-4 | Continuum reference | Vlasov FD on two meshes agree within 2%; CRN secants at $h = 0.1$ and 0.03 agree with it within 2 SE + 2% |
| R1-5 | Particle-pdf accuracy | Ensemble mean of $(\partial\mathcal J/\partial\theta)_D$ at $N = 10^5$ within 2 SE + 5% of the R1-4 reference, for non-collocated and collocated |
| R1-6 | Particle-pdf statistics | $\log\sigma$ against $\log N$ has slope in $[-0.6, -0.4]$ over $N = 10^4$ to $10^6$ (32 seeds each); $\sigma$ at $N = 10^5$ at most 5% of the reference slope |
| R1-7 | Variance stability in time | $\sigma$ of the particle-pdf estimate of the instantaneous $\partial_\theta J(t)$ grows less than 3× from $t\omega_p = 10$ to 450 |
| R1-8 | Conservation | $\lvert\sum_s\hat W_s(t)\rvert$ stays within 3 standard deviations of its seed-to-seed scatter |
| R1-9 | Initial condition by projection | Projection of the seed (66) reproduces (65) on the mesh with relative $L^2$ error below 5% |
| R1-10 | R-v | Error against R1-4 flat (within 2 SE) for $\lambda T_p \le 1$; report where it departs |
| R1-11 | Cost | Per-stage time profile like Fig. 12; time ratio to PIC reported; O(N) scaling with exponent in $[0.9, 1.1]$ |

R1-1 to R1-9 must pass to start R2. R1-10 and R1-11 are reported. If R1-5 fails while R1-1 to R1-4 pass, the particle-pdf implementation is wrong; stop and debug before R2.

### 4.4 Cost note

The NumPy pilot spends about 12× a PIC step per particle-pdf step (non-collocated) and 5× (collocated) [measured, 100 steps, $N = 10^5$]. The paper reports 2.96× and 2.10× for compiled code. The excess is the 12-entry phase-space `bincount` and the fancy-index gathers. R1-11 therefore checks scaling and the stage profile, not an absolute ratio. An absolute ratio for FBPIC is an R3 question.

## 5. R2 design: 1D trapping prototype

### 5.1 Model (R2a)

1D relativistic electrostatic PIC with a prescribed cycle-averaged ponderomotive driver. Units $k_p^{-1}$, $\omega_p^{-1}$, $m_ec$, $n_0$; electron charge $-1$.

1. Equations: $\gamma = \sqrt{1 + u^2 + A}$, $A = \langle a^2\rangle = \tfrac{a_0^2}{2}\exp(-\xi^2/2s^2)$, $\xi = x - x_{d0} - \beta_g t$; $\dot x = u/\gamma$; $\dot u = -E - \partial_x A/(2\gamma)$; $\partial_x E = n_{ion} - n_e$ with $E = 0$ at the unperturbed front (exact in 1D). Absorbing back boundary.
2. Driver: $s = 1$ (Twin-S: $c\tau/2.355 \approx 1.1\,k_p^{-1}$), $\gamma_g = 10$, $x_{d0} = -4$. The driver does not evolve. $a_0$ is the parameter ($E_L \propto a_0^2$); nominal $a_0 = 2.5$.
3. Plasma: cold, linear ramp over $[0, 3]$, then uniform; 20 macroparticles per cell; regular loading plus a seeded sub-cell jitter for ensembles.
4. Injection: 3% dopant on $[8, 14]$, 4 ions per cell, EV emission at rest at the ion position with $\Delta W = W_{ion}\,r\,(1 - e^{-\Gamma\Delta t})$, $\Gamma = 10\,e^{-8/a}$, threshold $\epsilon = 10^{-4}$, and a static positive charge added at birth (neutral pair). $a_0$ enters both the wake and the birth weights, as $E_L$ does in Twin-S.
5. Resolution: $\Delta x = \Delta t = 0.025$ (Twin-S: $\omega_p\Delta t = 0.0266$, $\Delta t$ = one cell); lab frame $x \in [-6, 95]$; $t_f = 75$ (3000 steps; Twin-S: 2000 steps). The pilot shows that $t_f = 58$ is too short to reproduce the failure (Section 2.4).
6. Observables (analog of the 11 Twin-S residuals): EV electrons only, quintic smoothstep selection on $u$ (Section 2.3 of the plan) with $u_c = 3$, $\delta_u = 0.3$, times a quintic window on the first bucket ($-2\pi \le \xi \le 0$, edge width 0.5; the pilot had no $\xi$ window and selected several buckets); $v = (\log Q, \langle u\rangle, \langle\xi\rangle, \log\sigma_u, \log\sigma_\xi, \log(\sigma_u/\langle u\rangle))$ with the plan's floors. Whitening: $\Sigma = \mathrm{diag}(s_j^2)$, with $s_j$ the seed-to-seed standard deviation of $v_j$ at the nominal point (a statistical stand-in for the model-error weights). Synthetic target $v_y = \bar v(a_0^\ast = 2.53)$.

R2b (optional, only if R2a passes and the user wants it before R3): 1D electromagnetic PIC with a resolved laser ($\omega_0/\omega_p = 15$, as Twin-S) and a 1D-2V phase space. In 1D the cold plasma has $u_y = a_y$ exactly, so $\nabla_{u_y}f$ is singular everywhere in the laser. R2b is the realistic stress test of the $\sigma_u$ bias and of R-v-s; R2a has no transverse momentum.

### 5.2 Implementation

1. PyTorch, float64, CPU (GPU optional). Write the step as a pure function of the state $(x, u, W, \text{ion charge}, \text{birth bookkeeping})$ and $a_0$. Then `torch.func.jvp` over one window of $T_p$ steps returns the Lagrangian tangent and the EV weight tangent with no hand-written tangent code. R-v is a loop of windows: call `jvp`, project $(\xi_x, \xi_u)$ into weight tangents, set $\xi = 0$, continue. Particle-pdf (collocated) is the same loop with $T_p = 1$ (Section 3, item 2). `torch.func.jvp` through `index_add` and `scatter_add` works in the installed PyTorch 2.7.1 and matches FD [measured, 5-particle check]. Birth decisions (the $\epsilon$ cut) are computed from the primal and stay frozen inside `jvp`.
2. Phase-space mesh: dense, restricted to a window $x \in [x_d - 25, x_d + 3]$ that moves with the driver; $u \in [-3, 80]$; $\Delta u = \sigma_u/2$. At $\sigma_u = 0.02$ this is about $1120\times8300 \approx 9\cdot10^6$ cells (75 MB). Ahead of the driver $f$ is unperturbed and $\delta E = 0$, so the source vanishes there. Behind the window the source does not vanish (the wake persists), but in 1D the field at $x$ depends only on the charge ahead of $x$ ($E(x) = -\int_x^{x_{front}}\rho\,dx'$), so sensitivity behind the window cannot reach the selected first-bucket beam unless particles overtake the window edge. Log the weight that crosses the back edge forward; it must stay below 1% of the selected weight. [derived]
3. Non-collocated particle-pdf as in R1, with $\nabla_x f$ added inside the driver (Section 3, item 1).
4. Deterministic runs: no atomics on CPU, so twin FD at $h = 10^{-9}$ is a valid check of the `jvp` tangent.

### 5.3 Tests

**(a) Pathology and Lyapunov rate.** Run the exact `jvp` tangent at nominal $a_0$ (no projection). Record per-particle $\lvert\xi\rvert$ every 40 steps. Report the finite-time Lyapunov exponent distribution over $t \in [t_{trap}, t_f]$ and the rate of the median and the 90th percentile. Compare with the twin FD at $h = 10^{-9}$ and $10^{-7}$. Also compute the exact derivative column of the 6 residuals at $t_f$ and its column metrics against the (b) reference. Pass (pathology reproduced): median-tangent growth fits $e^{\lambda t}$ better than $t^p$ ($p \le 3$) over at least 2 decades, with $\lambda \ge 0.05\,\omega_p$, and the exact column has $\kappa > 0.5$ or $\cos < 0.5$ against the reference. If not, raise the stress: extend to $t_f = 80$, raise the dopant to 10%, or lower $\gamma_g$ to 5. If the pathology still does not appear, R2a cannot test Track R; move to R2b. The pilot meets this pass at $t_f = 75$ with one deterministic loading ($\lambda = 0.23$ over 3.3 decades; exact column $\cos = -0.28$, $\kappa = 1.06$); R2 must confirm it on the seed ensemble and with the first-bucket window, which the pilot did not have.

**(b) Continuum regularity (the key assumption (81)).** CRN ensemble FD: $D_h = \frac1S\sum_s [v(a_0+h; s) - v(a_0-h; s)]/2h$ with $h \in \{0.01, 0.02, 0.04\}$ ($\delta a_0/a_0 = 0.4\%$ to 1.6%, which spans the Twin-S FD steps $h = 0.01$ to 0.02 in $\hat E_L$). Also a 41-point slice of $\bar v$ over $\pm0.05$ with a cubic fit, as P7. Pass: $D_{0.01}$, $D_{0.02}$, $D_{0.04}$ agree within 2 combined SE + 10% for at least 5 of 6 observables, and the cubic-fit slope agrees with $D_{0.02}$. Reference column: $D_{0.02}$ with its SE. Seed count: 32 seeds per $h$. Reason: at $t = 58$ the whitened per-seed secant noise at $h = 0.02$ is about 25 per component against a whitened column norm of 243 [measured], so 32 seeds give a column SE of about $25\sqrt6/(243\sqrt{32}) \approx 4\%$. Re-measure the noise at $t = 75$ with the first 8 seeds and raise the count if the column SE exceeds 5%. Whitening uses the $t = 75$ seed SDs.

**(c) Particle-pdf** (non-collocated, as in R1) against the (b) reference, at $\sigma_u$ from (e). Per-observable relative error with SE, and the Section 6 column metrics.

**(d) R-v and R-v-s.** R-v with $T_p \in \{1, 8, 32, 128, 512\}$ steps; R-v-s projects only EV electrons with $u > 1$ (the trapped and trapping set) and keeps exact tangents for the plasma. Same metrics as (c), plus the projection-consistency log (Section 3, item 5) and $\lambda T_p$ for each run.

**(e) Bias against $\sigma_u$.** $\sigma_u \in \{0.02, 0.05, 0.1, 0.2, 0.5, 1, 2\}$ at fixed $N$, 16 seeds each, for (c) and (d). Fit $\lvert\text{bias}\rvert = C\sigma_u^p$ using the ensemble mean minus the reference. Also run an adaptive width $\sigma_u \propto$ local spread (Silverman rule). Pass: $p \ge 1$ and a $\sigma_u^\ast$ exists with bias $\le 0.1\lvert D_{ref}\rvert$ and noise $\le 0.15\lvert D_{ref}\rvert$ per observable.

**(f) $N^{-1/2}$ convergence.** Plasma 5, 20, 80 per cell and dopant 1, 4, 16 per cell, at $\sigma_u^\ast$, 16 seeds each. Pass: slope of $\log\sigma$ against $\log N$ in $[-0.65, -0.35]$; $\sigma(t_f)/\sigma(t_{trap}) \le 3$ (no exponential variance growth).

**(g) Cost.** Wall time per step of each method relative to the primal step in the same code, the per-stage profile, the phase-space mesh memory, and the marginal cost of a second parameter (a second $a_0$-like parameter, for example the driver length $s$). Report operation counts per stage as in Table 1.

### 5.4 Runs and time

A primal run at $t_f = 75$ takes about 30 s on one unloaded core (pilot: 22 s for 2320 steps with about $2\cdot10^5$ particles) [measured]. Sensitivity runs cost 5 to 12 primal runs in NumPy-style code (R1 pilot ratio) [estimated for R2].

| Set | Runs | CPU time |
|---|---|---|
| (a) exact `jvp` tangent + twins at $10^{-9}$, $10^{-7}$ | 3 | 5 min |
| (b) CRN ensemble: 32 seeds × ($h \in \{0.01, 0.02, 0.04\}$, ±) + 32 centers; 41-point slice | 265 | 2.2 h |
| (b) the same at the two off-solution points $a_0 \in \{2.40, 2.65\}$, $h = 0.02$ only | 128 | 1.1 h |
| (e) $\sigma_u$ sweep: 7 widths × 2 methods (particle-pdf, R-v-s) × 16 seeds | 224 | about 15 h |
| (d) $T_p$ sweep: 5 values × 16 seeds | 80 | about 5 h |
| (f) $N$ sweep: 3 sizes (1×, 4×, 16×) × 2 methods × 16 seeds | 96 | about 40 h (dominated by 16×) |
| (g) profile and second-parameter runs | 10 | 1 h |

Total about 65 h of CPU, which is about 2.5 h of wall time on 30 cores, or less on one GPU with PyTorch [estimated]. The (f) set at 16× can drop to 8× if time is short; the slope then rests on one decade.

## 6. Go/no-go for R2 to R3, and a guarded usability metric

### 6.1 What $\delta$ measures [derived]

Gauss-Newton with an approximate Jacobian $J_A$ iterates $\theta_{k+1} = \theta_k - J_A^+ r(\theta_k)$. Linearize $r(\theta) = r^\ast + J(\theta - \theta^\ast)$ about the least-squares solution $\theta^\ast$, where $J^Tr^\ast = 0$. With $e_k = \theta_k - \theta^\ast$:

$$
e_{k+1} = (I - J_A^+J)\,e_k - J_A^+r^\ast .
$$

The fixed point is $e_\infty = -(J_A^TJ)^{-1}J_A^Tr^\ast = -(J_A^TJ)^{-1}(J_A - J)^Tr^\ast$. For $K = 1$ its size is the D32 metric $\delta = \lvert(J_A - J)^Tr^\ast\rvert / \lvert J_A^TJ\rvert$. The contraction factor is $\kappa = \lVert I - J_A^+J\rVert_2$; for $K = 1$, $\kappa = \lvert 1 - J_A^TJ/J_A^TJ_A\rvert = \lvert 1 - \cos/\rho\rvert$ with $\rho = \lVert J_A\rVert/\lVert J\rVert$.

$\delta$ is invariant to the scale of $J_A$. Near the solution $\delta \approx (\lVert r^\ast\rVert/\lVert J\rVert)\,\lvert\cos(J_A, r^\ast)\rvert/\lvert\cos(J_A, J)\rvert$. So a large $J_A$ does not inflate the denominator, as the task statement suggests; $\delta$ is small because $\lVert r^\ast\rVert/\lVert J\rVert$ is small at a near-zero-residual solution. $\delta$ says where the iteration would stop, not whether it gets there. A garbage $J_A$ gives $\kappa \approx 1$ (stall) or $\kappa > 1$ (divergence; dogbox then shrinks the trust radius and stalls).

### 6.2 Evaluation on the P7 data [measured, `guard_metrics.py` on `inversion/results/p7_jacobian.json`]

Reference: the smoothed column at the T0 solution ($\hat E_L = 0.588$, $\lVert r_w\rVert/\lVert J_{sm}\rVert = 0.013$).

| Column | $\delta$ | $\cos$ | $\rho$ | $e_{col}$ | $\kappa$ |
|---|---|---|---|---|---|
| AD | 3.5e-3 | 0.25 | 355 | 355 | 0.999 |
| Free FD, $h = 0.02$ | 2.1e-4 | 0.999 | 0.92 | 0.090 | 0.087 |
| Free FD, $h = 0.01$ | 5.8e-4 | 0.976 | 1.17 | 0.29 | 0.165 |
| Free FD, $h = 0.005$ | 8.6e-4 | 0.96 | 1.37 | 0.50 | 0.30 |
| Frozen FD, $h = 10^{-3}$ | 1.2e-3 | 0.93 | 1.62 | 0.78 | 0.43 |
| Frozen FD, $h = 10^{-4}$ | 2.1e-3 | 0.97 | 1.85 | 0.92 | 0.48 |
| Frozen FD, $h = 10^{-5}$ | 4.8e-3 | 0.79 | 2.88 | 2.18 | 0.73 |
| Free FD, $h = 0.05$ | 1.4e-2 | 0.33 | 0.53 | 0.97 | 0.39 |

At the false minimum ($\hat E_L = 0.545$, reference frozen FD at $h = 0.01$ because the smoothed slice is incomplete): AD has $\delta = 0.020$, $\cos = 0.33$, $\rho = 78$, $\kappa = 0.996$; FD at $h = 10^{-3}$ has $\cos = 0.98$, $\kappa = 0.11$.

Null test: 11-vectors with random direction and magnitude log-uniform over $10^{-2}$ to $10^3\lVert J_{sm}\rVert$. $\delta \le 0.005$ passes 19.5%; $\kappa \le 0.5$ and $\cos \ge 0.9$ pass 0 of $2\cdot10^5$.

### 6.3 The guarded metric

A column $J_A$ (one per parameter; the $K$-parameter forms use $\kappa = \lVert I - J_A^+J_{ref}\rVert_2$ and the per-column cosines) is **usable** when, against the smoothed reference $J_{ref}$:

1. $\kappa \le 0.5$ at the solution and at two off-solution points (the P7 false-minimum and far-field centers, or $a_0 \in \{2.40, 2.65\}$ in R2). Reason: $\kappa$ is the linear convergence factor; 0.5 at least halves the error per iteration, so a 0.04 basin reaches 0.005 in 3 iterations. FD columns with $h \ge 10^{-4}$ reach 0.09 to 0.48 on P7.
2. $\cos(J_A, J_{ref}) \ge 0.9$ for every column. Reason: for $K = 1$, $\kappa \le 0.5$ allows $\cos = 0.75$ with $\rho = 0.5$; for $K > 1$ a bad direction leaks into the other parameters. 0.9 is passed by every FD column with $h \in [10^{-4}, 0.02]$ and failed by AD (0.25) and by the over-curved $h = 0.05$ secant (0.33).
3. $\delta \le 0.005$ (D32, unchanged) at the solution. It bounds the stopping bias once 1 and 2 hold.
4. Reported, not gated: $\rho$ (implied within $[0.6, 2]$ by 1 and 2), $e_{col}$, and the per-residual errors. The reference itself has $e_{col} \approx 0.1$ to 0.5 (FD at $h = 0.005$ to 0.02 against the cubic fit), so a stricter $e_{col}$ bar would test the reference, not the method.
5. Statistical margin: for a stochastic column (particle-pdf, R-v), apply 1 to 3 to the ensemble mean and also to each single seed at the production $N$. A single seed is what the optimizer sees.

### 6.4 R2 to R3 decision

| Outcome | Condition | Decision |
|---|---|---|
| GO | (a) reproduces the pathology; (b) passes; the best of particle-pdf, R-v, R-v-s passes Section 6.3 per seed at the production $N$; (e) $p \ge 1$ with a usable $\sigma_u^\ast$; (f) passes; (g) marginal cost per parameter at most 2 primal steps by operation count | Start R3 with that variant |
| GO, R-v-s only | Only R-v-s passes | R3 builds on the P4 to P6 kernels plus a projection kernel; no full particle-pdf |
| HOLD | (a) fails in R2a and R2b | Track R untested; the user decides whether to go to R3 on R1 evidence alone |
| NO-GO, method | (b) passes, but no variant passes Section 6.3, or the noise at $\sigma_u^\ast$ needs more than 16× the primal particles | Stop Track R; the FD branch (D32 branch 3) stays |
| NO-GO, physics | (b) fails: ensemble secants at $h = 0.01$ to 0.04 disagree | The continuum response is not regular at the observable level; no derivative method can help; stop Track R |

Reason for the cost bar [derived from measured P7 costs]. A central-FD Jacobian costs $2K + 1$ runs per dogbox iteration. A forward-mode Jacobian costs $1 + cK$ run-equivalents, where $c$ is the marginal cost of one parameter. P7 step 3 measured $3.47$ run-equivalents for $K = 1$ with Lagrangian tangents, so $c \approx 2.47$ for the existing kernels: they do not beat central FD for any $K$. The paper's particle-pdf totals (1.89 to 2.96 PIC runs for one parameter, Section 1.4) give $c \approx 0.9$ to 2.0. In particle-pdf, the deposit of $f$ and $\nabla_u f$ is shared by all parameters; per parameter only $\delta\rho$ or $\delta\mathbf J$, the field tangent, the source product, and the weight gather remain. So $c \le 2$ is the break-even with central FD, and $c \le 1$ with one-sided FD. A Track R method that is accurate but has $c > 2$ helps only through gradient quality (no $h$ to tune, no birth-threshold jumps), not through cost; the user should know this before R3.

## 7. Effort and what each result decides

| Phase | Effort | Decides |
|---|---|---|
| R1 | 2 days (pilot covers baseline, twins, first particle-pdf) | Whether our reading of the particle-pdf algorithm is right and reproduces the paper; the R-v equivalence at $T_p = \Delta t$; the stage cost profile. A failure here stops Track R at low cost. |
| R2a | 5 days (model exists in the pilot; adds `jvp` tangents, particle-pdf with a moving phase-space mesh, R-v, R-v-s, ensembles) | Whether continuum regularity holds for trapping observables; which variant to build in R3; the $\sigma_u$ bias law; the noise per $N$; the guarded usability verdict |
| R2b (optional) | 2 to 3 days | Whether the $u_y$ singularity of a laser-driven cold plasma breaks particle-pdf but not R-v-s |

## 8. Open questions

1. The paper does not state the position loading or the exact Poisson stencil; the 0.23 offset in $\bar{\mathcal J}$ may come from either. It does not matter for slopes.
2. Fig. 9b does not say whether $M$ scales with $N$. The pilot used $M = 2N$ at $N = 10^4$ and saw a larger spread than Fig. 9b; R1 must run both $M = 2N$ and fixed $M = 2\cdot10^5$.
3. The source of the 1D chaos is not identified. A test particle in a stationary 1D wake is integrable (one degree of freedom in the wake frame), so the growth must come from the coupling: beam loading, the plasma electrons that cross the beam, and the ramp-injected plasma electrons. The pilot's plasma-electron tangents grow faster (0.23) than the EV tangents (0.11 to 0.20). R2 should log which particles carry the growth, because R-v-s (Section 3, item 8) assumes that the background plasma stays laminar. If the background plasma is the chaotic set, R-v-s must project it too and loses its advantage.
4. Twin-S growth: 10× per 120 steps is $\lambda \approx 0.72\,\omega_p$ ($\omega_p\Delta t = 0.0266$), and the growth lasts about 21 $\omega_p^{-1}$ after trapping, so $\lambda T \approx 15$. The R2 pilot has $\lambda \approx 0.23$ and $\lambda T \approx 10$ at $t_f = 75$; R1 has 0.08 to 0.18. A method can look better in R2 than it will be in Twin-S if its error depends on $\lambda T_p$. Report $\lambda T_p$, not $T_p$, and prefer methods whose error is flat in $\lambda T_p$.
5. The ponderomotive model has an $x$-component of the source inside the driver. FBPIC has none (the laser is a field), but it has $u_\perp$ dipoles from the laser. R2b covers that case.
6. The reference uncertainty (about 0.1 to 0.5 in $e_{col}$ on P7) limits how precisely any method can be judged at $N_t = 2000$ on Twin-S. R4 should use more slice points or more seeds of the stochastic referee, not a tighter bar.

## 9. Files

Pilot directory: `inversion/notes/r0-evidence-2026-10-31/work_tests/`. All runs are CPU-only NumPy (`python3 -I`).

1. `debye_pilot.py`: R1 PIC baseline (`base`), twin divergence (`twin`), CRN secant (`secant`). Results: `base_N1e5.jsonl`, `out_twin_*.json`, `out_secant_*.json`.
2. `debye_pdf.py`: particle-pdf (non-collocated and collocated) and the semi-Lagrangian Vlasov reference. Results: `results_all.jsonl` (keys `N`, `M`, `col`, and `NX` for Vlasov); summary script `r1_analyze.py`.
3. `r2_pilot.py`: R2a model (`scan`, `twin`, `obs`). Results: `twin_a25_h1e-9.json`, `twin_a25_h1e-7.json`, `twin_a20_h1e-9.json`, `twin_a25_h1e-9_t75.json`, `results_all.jsonl` (keys `a0`, `seed`), `results_t75.jsonl`; analysis `r2_analyze.py`.
4. `guard_metrics.py`: Section 6 metrics and the null test on `inversion/results/p7_jacobian.json` (read only).

Code references in the worktree (read only): the D32 metric is `inversion/p7_jacobian.py:429-431` (`_delta`); the smoothed column is the cubic fit at `inversion/p7_jacobian.py:416`; the P7 numbers are `DIFFERENTIABLE_INVERSION_PLAN.md:1816-1844`; the Track R block is `DIFFERENTIABLE_INVERSION_PLAN.md:856-886`; the D32 branches are `DIFFERENTIABLE_INVERSION_PLAN.md:658-662`. Kernels that R-v and R-v-s would reuse in R3: `fbpic/autodiff/particles.py:22-145` (push tangents), `fbpic/autodiff/deposition.py:40-158` (deposit tangents), `fbpic/autodiff/ionization.py:237` (`ev_tangent`, the EV weight source), `fbpic/autodiff/loss.py:170` (`observables`). PyTorch 2.7.1 `torch.func.jvp` supports `index_add` and `scatter_add` [measured, a 5-particle check against FD].
