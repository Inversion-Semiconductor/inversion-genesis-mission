# Option C on chaotic toys (task key: toy)

Work directory: `inversion/notes/option-c-evidence-2026-11-01/work_toy/` (below: `work_toy/`).
All runs are CPU NumPy, `python -I`. Labels: [derived], [measured: script], [literature: citation], [estimated].

## 0. Verdict

**Negative for the stated goal.** Option C is a correct, unbiased estimator of the noisy model's gradient in every test. In a self-consistent plasma, however, its mean-field form does not avoid exponential growth, and the noise it needs changes the gradient at $O(1)$.

1. Correctness [measured]. Full-LR means agree with the reference within 2 SE in 30 doubling-map and 15 standard-map comparisons, except 3 doubling-map cases ($|z|\le2.9$) that do not repeat across seed sets. In the two-stream model, all 180 mean-field LR, joint LR and tangent means are within 2 SE of FD, except one heavy-tailed tangent case ($z=-2.3$).
2. Independent particles (maps) [measured]. The weight variance grows linearly, $\omega_{\rm rms}\propto\sqrt n/\sigma$, while the exact tangent reaches $10^{17}$ to $10^{23}$. This is the regime where option C works as advertised. Its SE scales as $1/\sigma$; at $\sigma=0.03$ it needs about 86 times more particles than CRN FD of the same noisy model, so it pays off only for $K\gtrsim40$ parameters.
3. Self-consistent plasma [derived and measured]. Coupling the weights to the field tangent adds a feedback: the weighted deposit has sampling noise $\propto\omega_{\rm rms}/\sqrt N$, which feeds back into the weights. The variance grows like $e^{\lambda t}$ with $\lambda=c_{\rm shot}/(N\sigma^2)$, measured in a stable cold plasma with the predicted $1/N$ law. On top of this, the linear instability makes the true $\partial_\theta F$ large. In the best case tried ($\sigma=0.03$, $N=524288$, $T=100$) the weights still grow at half the tangent's rate and overtake it only when both are 200 to 1100 times noisier than FD.
4. Bias [measured]. Before saturation, the noisy gradient converges to the deterministic one at roughly $O(\sigma^2)$. After saturation, the noise levels at which the weights stay usable ($\sigma\ge0.03$) change the sign of $\partial_\theta m_1$ and $\partial_\theta s_1$. In mixed phase space (standard map) there is no $\sigma\to0$ limit at all.
5. Joint-measure weight [derived and measured]. It is unbiased for finite $N$ and its variance stops growing after the explicit $\theta$-dependence ends. It is the best LR variant in the toy, but only because $\theta$ there enters through a weak, short seed. For $\theta=E_L$ the explicit dependence is the whole laser, so it does not transfer [estimated].
6. Resolved fast forces [measured]. The relative variance grows like $\omega_0^4$ (quiver toy), which matters directly for a laser-resolved code [estimated: Section 5].

The decisive FBPIC check, if anyone pursues option C further, is the one-step random-weight measurement of $\lambda$ in Section 5, item 1.

## 1. Estimators and what they cost in variance

### 1.1 Noisy model and the discrete Girsanov weight [derived]

Discrete noisy push for particle $i$ (Euler-Maruyama in $v$, then drift):
$$v_i^{n+1} = v_i^n + F_\theta(x_i^n, v_i^n; \mu^n, t_n)\,\Delta t + \sigma\sqrt{\Delta t}\,\xi_i^n,\qquad x_i^{n+1} = x_i^n + v_i^{n+1}\Delta t,\qquad \xi_i^n\sim\mathcal N(0,1).$$
For one particle with a given force field, the transition density of $v^{n+1}$ is Gaussian. Its $\theta$-score is
$$\partial_\theta \log p = \frac{(v^{n+1}-v^n-F\Delta t)\,\partial_\theta F\,\Delta t}{\sigma^2\Delta t} = \frac{\partial_\theta F\,\sqrt{\Delta t}\,\xi^n}{\sigma}.$$
So $\omega^n=\sum_{m<n}\sigma^{-1}\partial_\theta F^m\sqrt{\Delta t}\,\xi^m$ and $\partial_\theta\mathbb E[g(z^n)]=\mathbb E[g(z^n)\,\omega^n]=\mathbb E[(g-b)\,\omega^n]$ for any constant $b$. This is exact for the time-discrete noisy model, not only as $\Delta t\to0$ [derived; literature: Glynn, Commun. ACM 33(10):75, 1990]. For maps $z' = T_\theta(z)+\sigma\xi$ the increment is $\partial_\theta T(z)\,\xi/\sigma$.

$\omega^n$ is a martingale, so $\operatorname{Var}\omega^n=\sum_m \Delta t\,\mathbb E|\partial_\theta F^m|^2/\sigma^2$. It grows linearly in time when $\partial_\theta F$ is bounded and does not depend on $\omega$ [derived]. The estimator variance per particle is about $\operatorname{Var}(g)\operatorname{Var}(\omega)$.

A useful reading: $\operatorname{Var}\omega(T)\approx (\delta v_\theta)^2/(\sigma^2 T)$ when $\partial_\theta F$ is steady over $T$, where $\delta v_\theta=\partial_\theta F\,T$ is the velocity response per unit $\theta$ and $\sigma\sqrt T$ is the noise-induced velocity spread. The weights are small only when the noise spread exceeds the per-unit-$\theta$ response [derived].

### 1.2 Mean-field weight with field coupling: a feedback growth rate [derived, then measured]

In a self-consistent system $\partial_\theta F_i = q(\partial_\theta E_{\rm ext} + \delta E[\delta\rho])(x_i)$ with $\delta\rho = q\sum_j W_j\omega_j S(x-x_j)$. With $N$ particles, $\delta E = \delta E_{\rm true}+\eta$, where $\eta$ is the sampling error of the weighted deposit. For independent weights, $\mathbb E\eta^2 = c_{\rm shot}\operatorname{Var}(\omega)/N$. Then
$$\frac{d}{dt}\operatorname{Var}\omega \;\ge\; \frac{\mathbb E|\partial_\theta F_{\rm true}|^2}{\sigma^2} + \lambda\operatorname{Var}\omega,\qquad \lambda=\frac{\mathbb E|q\eta|^2}{\sigma^2\operatorname{Var}\omega}=\frac{c_{\rm shot}}{N\sigma^2}.$$
So the mean-field weight variance grows exponentially at rate $\lambda$, even in a stable plasma. For a 1D periodic box of length $L$, unit mean density, and point charges, $c_{\rm shot}=L^2/12$ (field of one charge is a zero-mean sawtooth with $\langle H^2\rangle=1/12$). CIC shape, the spectral cut, and the gather reduce $c_{\rm shot}$ by a constant factor. In words: $\lambda$ is the force variance that the PIC shot noise would produce if each macroparticle carried a random unit charge, divided by the injected momentum diffusivity. The Lagrangian tangent has the opposite dependence: its rate is the Lyapunov exponent, which $N$ changes little (10 to 30 percent per factor 4 in Twin-S, from the briefing).

### 1.3 Joint-measure weight [derived]

For finite $N$ the full path density is $\prod_{n,i}\varphi(\xi_i^n)$ with $\xi_i^n$ defined through the actual (finite-$N$) field. At fixed paths, the field depends on $\theta$ only explicitly. The score is one scalar,
$$\Omega=\sum_{n,i}\sigma^{-1}\,\partial_\theta^{\rm expl}F_i^n\,\sqrt{\Delta t}\,\xi_i^n,\qquad \partial_\theta\mathbb E[J]=\mathbb E[(J-b)\,\Omega].$$
It is unbiased for every $N$, needs no field tangent, and has no feedback. Its variance is $\operatorname{Var}(J)\,N\,I_1$ with $I_1=\sum_n\Delta t\,\mathbb E|\partial^{\rm expl}_\theta F|^2/\sigma^2$ per particle. If $J$ fluctuates like $1/N$ (independent particles), the product is $O(I_1)$ and does not fall with $N$. If $J$ fluctuates collectively ($\operatorname{Var}J=O(1)$, for example noise-seeded instabilities), it grows like $N$.

The mean-field weight has variance about $\operatorname{Var}(g)\operatorname{Var}(\omega)/N$, so it gains a factor $N$ when $\lambda T\lesssim1$. It has a propagation-of-chaos bias with respect to the finite-$N$ system, expected to be $O(1/N)$ for smooth interactions [derived; not resolved here: mf agrees with finite-$N$ FD within SE at $N=32768$ to $524288$].

When to use which [derived]: the joint weight suits a weak, short explicit $\theta$-dependence (a seed, a short kick) and few particles that see it. The mean-field weight suits many particles and a strong explicit dependence, provided $\lambda T\lesssim 1$. For an LPA with $\theta=E_L$, the explicit dependence at fixed particle paths is the vacuum-propagated laser tangent acting on every background electron. That makes $N I_1$ very large, so the joint weight is not a candidate there [estimated].

### 1.4 Resolved fast forces [derived]

The weight increment uses $\partial_\theta F$ itself, not its time average. For $F=\theta A(x)\cos(\omega_0 t)$, $I_1\approx\int A^2\cos^2/\sigma^2\,dt$ does not depend on $\omega_0$. The useful signal, the ponderomotive response, scales as $A^2/\omega_0^2$. Relative variance at fixed particle count therefore grows like $\omega_0^4$. Pathwise tangents of quiver motion are bounded and cheap.

### 1.5 Bias of the noisy model [derived; literature]

The noisy model's derivative is the derivative of a different objective. For a uniformly expanding map with smooth $f_0$ and $g$, linear response holds [literature: Ruelle, Nonlinearity 22:855, 2009; Hairer and Majda, Nonlinearity 23:909, 2010]. Gaussian noise then acts on smooth densities as $1+\tfrac12\sigma^2\partial_x^2+O(\sigma^4)$, so the noisy derivative converges to the deterministic one as $O(\sigma^2)$ [derived; measured in Section 2.5]. For mixed phase space (islands), the deterministic derivative need not exist at any useful scale, and the noisy derivative depends on $\sigma$ at $O(1)$.

## 2. Chaotic maps

### 2.1 Setup [measured: `work_toy/maps_lr.py`, tables in `work_toy/out/maps_tables.txt`]

- Doubling map (case G of `testA_doubling.py`): $x'=2x+0.3\sin(2\pi x)/2\pi+\theta s(x)+\sigma\xi \pmod 1$, $s=(0.5+\sin(2\pi x+0.7))/2\pi$, $\theta_0=0$, $f_0=1+0.5\cos(2\pi x-1)$, $\psi=\cos 2\pi x+0.5\sin 4\pi x$. Weight increment $s(x)\xi/\sigma$.
- Reference: the noisy transfer operator with its exact $\theta$-tangent on Fourier coefficients (4096 nodes, 600 modes; the 8192/1200 check differs by at most $1.5\times10^{-12}$). At $\sigma=0$ it gives 0.047332, equal to the earlier `testA_doubling.py` reference.
- Standard map, $K_0=1.5$, noise on the kick only: $p'=p+K\sin x+\sigma\xi$, $x'=x+p'$, Gaussian blob at $(2,1)$, $\phi=\cos x+0.5\sin p$. Weight increment $\sin(x)\xi/\sigma$.
- Reference for the standard map: CRN central FD of the noisy model at $h=0.03$. At $h=0.1$ the FD has an $O(h^2)$ bias of 12 to 35 SE (for example $\sigma=0.03$, $n=20$ and 40). At $h=0.01$ it agrees with $h=0.03$ but is 2-3 times noisier.
- $N=2\times10^7$ particles (40 batches of $5\times10^5$). SE is the batch scatter. Times $n\in\{5,10,20,40,80\}$.

### 2.2 Agreement [measured]

| map, $\sigma$ | $n$ | reference | full LR | LR, window $W=5$ | exact tangent (mean) |
|---|---|---|---|---|---|
| doubling, 0.3 | 80 | 0.00921 | 0.0087 ± 0.0008 | 0.0093 ± 0.0002 | $1.3\times10^{21}$ |
| doubling, 0.1 (3 seed sets) | 80 | 0.03952 | 0.0391 ± 0.0025; 0.0394 ± 0.0024; 0.0440 ± 0.0024 | 0.0389 ± 0.0006; 0.0392 ± 0.0005; 0.0395 ± 0.0005 | $\sim10^{20}$ |
| doubling, 0.03 | 80 | 0.04655 | 0.0325 ± 0.0068 | 0.0445 ± 0.0016 | $1.2\times10^{20}$ |
| doubling, 0.01 | 80 | 0.04724 | 0.0107 ± 0.0217 | 0.0410 ± 0.0049 | $2.4\times10^{20}$ |
| standard, 0.3 | 80 | 0.0122 ± 0.0047 | 0.0151 ± 0.0041 | -0.0018 ± 0.0009 (biased) | $1.1\times10^{14}$ |
| standard, 0.1 | 80 | 0.1438 ± 0.0043 | 0.1527 ± 0.0102 | -0.0317 ± 0.0029 (biased) | $-2.3\times10^{15}$ |
| standard, 0.03 | 80 | 0.3571 ± 0.0039 | 0.3268 ± 0.0374 | -0.0886 ± 0.0112 (biased) | $-3.4\times10^{14}$ |

- Full LR, doubling map: 30 comparisons (6 runs, 5 times). The rms $z$ is 1.29; 3 have $|z|>2$; the largest is 2.9. The outliers ($\sigma=0.1$, $n=40$: $-2.3$; replicate 1, $n=5$: $+2.9$; $\sigma=0.03$, $n=80$: $-2.1$) do not repeat across the three $\sigma=0.1$ seed sets. The first four runs share seeds across $\sigma$, so their $z$ values are correlated.
- Full LR, standard map: 15 comparisons, rms $z$ 0.83, largest $|z|$ 1.5.
- The exact Lagrangian tangent is useless after about 10 steps: rms $2\times10^{23}$ (doubling) and $2\times10^{17}$ to $2.5\times10^{19}$ (standard map) at $n=80$.

### 2.3 Variance against time, $\sigma$ and $N$ [measured; derived]

- Prediction: $\omega_{\rm rms}(n)=\sqrt{n\,\mathbb E s^2}/\sigma$ with $\mathbb E s^2=0.019$, so $\omega_{\rm rms}(80)=1.233/\sigma$. Measured at $\sigma=0.1$: 3.18, 4.44, 6.23, 8.78, 12.39 at $n=5,\dots,80$ (prediction 12.33 at $n=80$). The growth is $\sqrt n$, not exponential, while the tangent grows like $2^n$.
- SE of the full LR at $n=80$ with $2\times10^7$ particles: 0.0008, 0.0025, 0.0068, 0.0217 for $\sigma=0.3, 0.1, 0.03, 0.01$. This is $\propto\sigma^{-1}$. Standard map: 0.004, 0.010, 0.037 for $\sigma=0.3, 0.1, 0.03$.
- Particles are independent here, so SE $\propto N^{-1/2}$.

### 2.4 Windowed weights [measured]

Keeping only the last $W$ increments bounds the variance (doubling map: SE 0.0002 to 0.0055, flat in $n$). It is unbiased only if the noisy dynamics forgets the perturbation within $W$ steps. The doubling map mixes in about 5 steps, and windowing works there. The standard map at $K=1.5$ (islands) does not, and $W=5,10,20$ are biased by 3 to 137 SE at every $\sigma$. Windowing is not a general fix. It needs a per-problem memory test.

### 2.5 Bias of the noisy gradient as $\sigma\to0$ [measured]

- Doubling map, $n=80$ (transfer operator): noisy minus deterministic derivative is $-7.9\times10^{-6}$, $-8.8\times10^{-5}$, $-7.8\times10^{-4}$, $-7.8\times10^{-3}$, $-3.8\times10^{-2}$ for $\sigma=0.003, 0.01, 0.03, 0.1, 0.3$. This is $\approx-0.88\sigma^2$ for $\sigma\le0.03$, 1.7 percent of the gradient at $\sigma=0.03$.
- Standard map, $n=80$: the deterministic FD secant depends on $h$: 0.700, 0.697, 0.387 at $h=0.1, 0.03, 0.01$ ($n=40$: -0.697, -0.651, -0.404). The noisy derivatives are 0.357, 0.144, 0.012 for $\sigma=0.03, 0.1, 0.3$, each consistent across $h$. There is no visible limit as $\sigma\to0$. The noise level defines the objective at $O(1)$.

### 2.6 Cost on maps [measured; derived]

LR costs one multiply-add per particle, step, and parameter. Compare it with CRN FD of the same noisy model, which costs 2 runs per parameter. At $\sigma=0.03$ (standard map, $n=80$) the SE ratio LR/FD($h=0.03$) is $0.037/0.004\approx9$. LR needs about 86 times more particles for the same SE, so it is cheaper only for $K\gtrsim43$ parameters. At $\sigma=0.3$ the ratio is about 1 and LR wins for any $K$. CRN FD below the noise scale is also poor: SE 0.01 at $h=0.01$ and 0.1 at $h=0.001$, because the pathwise derivative explodes.

## 3. 1D-1V self-consistent two-stream plasma

### 3.1 Setup [measured: `work_toy/es1d_lr.py`]

- Units $\omega_p=1$, $q=-1$, $m=1$. Periodic box $L=10\pi$ ($k_1=0.2$), 128 cells, CIC, spectral Poisson with modes $m>24$ removed, $\Delta t=0.1$.
- Two cold beams at $v=\pm1$, quiet start, fixed ions. Modes $m=1..4$ are unstable. All randomness comes from the momentum noise.
- $\theta$ enters only through an external seed field, like the laser: $E_{\rm ext}=\theta\,\epsilon\,[\sin 2k_1x+\sin(3k_1x+1)]\,e^{-((t-4)/1.5)^2}$, $\theta_0=1$, $\epsilon=10^{-3}$ (ensembles) or $10^{-2}$ (long run).
- Observables: $K=\langle v^2\rangle$, and the mean $m_1$ and variance $s_1$ of beam 1. These are weighted beam moments, as in the inversion.
- Estimators from one noisy run per seed: mean-field LR "mf" (weights deposit $\delta\rho$, field tangent closes the loop), joint LR "jnt" (scalar $\Omega$, leave-one-out baseline over seeds), exact pathwise tangent "tan" (including the deposit tangent), and CRN central FD at $h=0.3, 0.1, 0.03$ (reference: $h=0.3$, lowest SE; $h=0.1$ and $0.03$ agree within 1.3 SE).
- Check: the pathwise tangent equals CRN FD at $h=10^{-5}$ and $10^{-6}$ to 4 to 6 digits up to $t=30$ (`es1d_lr.py verify`).

### 3.2 Shot-noise feedback in a stable cold plasma [measured: `work_toy/lam_test.py`, `work_toy/out/lam_test.log`]

Same code with both beams at rest ($V_0=0$, stable Langmuir oscillations). The fitted growth rate of $\operatorname{Var}\omega$ for $t\ge20$ is compared with the measured ratio $\lambda=\langle\delta E[\omega]^2\rangle/(\sigma^2\operatorname{Var}\omega)$ and with $L^2/(12N\sigma^2)$.

| $\sigma$ | $N$ | growth of $\operatorname{Var}\omega$ | $\lambda$ measured | $L^2/(12N\sigma^2)$ | $\omega_{\rm rms}(50)$ | tangent rms (50) |
|---|---|---|---|---|---|---|
| 0.03 | 32768 | 1.07 | 1.00 | 2.79 | $4.7\times10^{10}$ | 0.0053 |
| 0.03 | 131072 | 0.345 | 0.289 | 0.697 | 81 | 0.0038 |
| 0.03 | 524288 | 0.072 | 0.068 | 0.174 | 0.68 | 0.0035 |
| 0.1 | 32768 | 0.067 | 0.058 | 0.251 | 0.16 | 0.024 |
| 0.1 | 131072 | 0.042 | 0.037 | 0.063 | 0.091 | 0.014 |
| 0.1 | 524288 | 0.026 | 0.023 | 0.016 | 0.062 | 0.012 |

- At $\sigma=0.03$ the measured $\lambda$ falls by 3.5 and 4.2 per factor 4 in $N$, as $1/N$ predicts, and the fitted growth rate follows it. The prefactor is 0.36 to 0.41 of the point-charge value because of CIC, the spectral cut, and the gather.
- At $\sigma=0.1$ the physical part of $\delta E$ dominates and the $1/N$ law is masked.
- In this stable, non-chaotic plasma the exact tangent stays below 0.03, while the weights grow exponentially. This is the cost of the mean-field coupling, and it is independent of chaos.

### 3.3 Ensembles, $\epsilon=10^{-3}$, 96 seeds, $T\le60$ [measured: `work_toy/analyze_es.py` on `work_toy/out/es_*.json`]

Medians of $\omega_{\rm rms}$ and tangent rms, and per-seed SD of the estimators for $K$. For $m_1$ and $s_1$ the pattern is the same at $\sigma=0.03$; at $\sigma=0.1$, $t=60$ the mf and tan SDs are closer (about 1.2 to 1).

| case | $t$ | $\omega_{\rm rms}$ | tangent rms | SD mf | SD jnt | SD tan | SD FD($h=0.3$) |
|---|---|---|---|---|---|---|---|
| $\sigma=0.1$, $N=32768$ | 20 | 0.41 | 0.13 | $2.4\times10^{-3}$ | $2.7\times10^{-2}$ | $1.7\times10^{-3}$ | $1.7\times10^{-3}$ |
| | 60 | 10.7 | 18.1 | 0.15 | $2.5\times10^{-2}$ | $6.9\times10^{-3}$ | $2.9\times10^{-3}$ |
| $\sigma=0.1$, $N=131072$ | 20 | 0.38 | 0.12 | $1.8\times10^{-3}$ | $1.8\times10^{-2}$ | $1.4\times10^{-3}$ | $1.4\times10^{-3}$ |
| | 60 | 3.95 | 18.4 | $2.5\times10^{-2}$ | $2.9\times10^{-2}$ | $3.4\times10^{-3}$ | $1.5\times10^{-3}$ |
| $\sigma=0.03$, $N=131072$ | 20 | 4.9 | 0.24 | $9.2\times10^{-3}$ | $9.8\times10^{-2}$ | $5.6\times10^{-3}$ | $5.6\times10^{-3}$ |
| | 30 | 100 | 5.4 | 0.34 | 0.48 | 0.10 | $3.6\times10^{-2}$ |
| | 40 | 836 | 39 | 3.6 | 0.40 | 0.28 | $4.6\times10^{-2}$ |
| | 60 | $3.7\times10^4$ | $1.1\times10^3$ | 190 | 0.43 | 1.8 | $4.6\times10^{-2}$ |

Agreement with the FD reference: every mf, jnt and tan mean is within 2 SE of FD($h=0.3$) for all 3 observables and 5 times in all three cases (largest $|z|$: mf 1.9, jnt 2.0, tan 1.6). At $\sigma=0.03$, $t=60$ this is agreement with a useless SE: mf gives $K'=-33\pm19$ against $-0.023\pm0.005$.

Reading:
- $\sigma=0.1$ heats the beams strongly ($K$ rises from 1.0 to 1.6 by $t=60$). The tangent grows only to about 18 and stays the better estimator.
- $\sigma=0.03$, $N=131072$: the weights grow at least as fast as the tangent ($\omega_{\rm rms}$ rate 0.19 against 0.17 between $t=40$ and 60), and start 20 times higher, because the linear instability makes $\partial_\theta F$ large during $t=15$ to 30. The mean-field LR is worse than the exact tangent at every time.
- Going from $N=32768$ to $131072$ at $\sigma=0.1$ cuts the mf SD at $t=60$ by 6 (0.15 to 0.025): a factor 2 from $1/\sqrt N$, the rest from the smaller $\lambda$.
- The joint LR has a nearly constant per-seed SD after the driver ends (0.4 to 0.5 at $\sigma=0.03$), because $\Omega$ stops growing. It is the best of the three at $\sigma=0.03$, $t=60$, but still 10 times noisier than FD. It works here only because $\theta$ enters through a weak, short seed.

### 3.4 Best case for the mean-field weight: $\sigma=0.03$, $N=524288$, $\epsilon=10^{-2}$, $T=100$, 32 seeds [measured: `work_toy/run_es_long.sh`, `work_toy/out/es_long_summary_ref0.1.txt`]

This case has $\lambda\approx0.07$ (Section 3.2) and a stronger seed, so the tangent should explode long before the weights do. Reference: FD at $h=0.1$. At $t=20$ the $h=0.3$ secant of $K$ differs from $h=0.1$ by 13 SE because the response is nonlinear over $\pm30$ percent; for $t\ge40$ the two step sizes agree within 1.3 SE except $m_1$ at $t=40$ (3.4 SE).

| $t$ | $\omega_{\rm rms}$ | tangent rms | SD mf ($K$ / $s_1$) | SD tan ($K$ / $s_1$) | SD jnt ($K$ / $s_1$) | SD FD $h=0.1$ ($K$ / $s_1$) |
|---|---|---|---|---|---|---|
| 20 | 24 | 3.0 | 0.034 / 0.089 | 0.015 / 0.018 | 0.54 / 3.3 | 0.015 / 0.017 |
| 40 | 168 | 153 | 0.33 / 0.75 | 0.12 / 0.29 | 2.4 / 2.4 | 0.072 / 0.071 |
| 60 | 725 | $3.4\times10^3$ | 1.6 / 2.5 | 0.95 / 2.4 | 2.8 / 1.8 | 0.043 / 0.060 |
| 80 | $3.0\times10^3$ | $6.6\times10^4$ | 22 / 35 | 22 / 71 | 2.4 / 6.3 | 0.11 / 0.16 |
| 100 | $1.35\times10^4$ | $5.7\times10^5$ | 98 / 130 | 150 / 550 | 2.8 / 5.0 | 0.086 / 0.11 |

- Agreement: all 45 mf, jnt and tan means are within 2 SE of FD($h=0.1$), except tan $s_1$ at $t=100$ ($z=-2.3$, heavy-tailed). Largest $|z|$: mf 1.7, jnt 1.8.
- The weights are not bounded. Between $t=40$ and 100, $\omega_{\rm rms}$ grows at 0.073 per $\omega_p^{-1}$ and the tangent rms at 0.137. The weights grow at half the tangent rate. Shot-noise feedback alone gives $\lambda/2\approx0.035$; the rest is collective amplification.
- The mean-field LR overtakes the exact tangent only after $t\approx80$. There both have per-seed SD 200 to 1100 times that of FD at $h=0.1$, so neither is usable.
- The joint weight keeps a flat SD (2.4 to 6.3) after the seed ends. It is the only LR variant with time-bounded variance here, but it is still 20 to 65 times noisier than FD per seed, which means 400 to 4000 times more seeds for the same SE.

### 3.5 Bias of the noisy gradient as $\sigma\to0$ [measured: `work_toy/es_det.py`, `work_toy/out/es_det_N131k.log`, fd-only ensembles]

Noisy-model gradient (CRN FD over 96 seeds, $N=131072$, $\epsilon=10^{-3}$; $h=0.1$, because at $\sigma\le0.01$ the $h=0.3$ secant differs from $h=0.1$ by up to 10 SE) against the deterministic quiet-start model. Entries are $\partial_\theta m_1$ (beam-1 mean velocity). $K$ and $s_1$ also deviate at $O(1)$ for $\sigma\ge0.03$ at $t=40$; for example $\partial_\theta s_1=-0.03\pm0.03$ at $\sigma=0.03$ against 0.15 to 0.19 deterministic (`work_toy/out/es_fd_summary.txt`, `work_toy/out/es_ens_summary.txt`).

| $\sigma$ | $t=20$ | $t=30$ | $t=40$ | $t=60$ |
|---|---|---|---|---|
| 0.1 | -0.0034 ± 0.0005 | -0.018 ± 0.003 | -0.008 ± 0.004 | -0.007 ± 0.004 |
| 0.03 | -0.0236 ± 0.0009 | -0.121 ± 0.030 | +0.070 ± 0.030 | -0.009 ± 0.031 |
| 0.01 | -0.0279 ± 0.0004 | +0.043 ± 0.011 | -0.130 ± 0.009 | -0.011 ± 0.014 |
| 0.003 | -0.0285 ± 0.0001 | +0.106 ± 0.003 | -0.106 ± 0.010 | +0.027 ± 0.015 |
| 0, FD $h=0.1/0.03/0.01/0.003$ | -0.0286 (all $h$) | 0.127 / 0.133 / 0.134 / 0.134 | -0.130 / -0.128 / -0.093 / -0.178 | 0.156 / -0.033 / 0.454 / 0.729 |

- Before saturation ($t=20$) the noisy gradient converges to the deterministic one. The relative bias is 88, 18, 2.4 and 0.4 percent at $\sigma=0.1, 0.03, 0.01, 0.003$, roughly $\sigma^2$ (a factor 7 per factor 3 in $\sigma$).
- Just after saturation ($t=30$, 40) the gradient changes sign between $\sigma=0.03$ and the deterministic value. Only $\sigma\le0.01$ is close to deterministic.
- At $t=60$ the deterministic secant is rough (0.156, -0.033, 0.454, 0.729 for $h=0.1$ to 0.003). Moving the quiet-start positions by $10^{-3}$ of a spacing does not change the secants (4 loadings agree to 3 digits). The roughness is in $\theta$, not in the initial state.
- The mean-field LR has usable variance only at $\sigma\ge0.03$ for $N\le5\times10^5$ (Sections 3.2 to 3.4). In this toy that noise level changes the sign of the post-saturation gradient.

### 3.6 Cost [measured: `work_toy/cost_es.py`; derived]

- Wall time for 50 steps, $N=131072$, one process on a shared, loaded node (3 repeats, medians; scatter about 30 percent): plain 2.7 s, plus mean-field LR 3.5 s, plus exact tangent 5.4 s. One mean-field weight set costs about 0.3 of a run per parameter (one deposit, one field solve, one gather, one multiply-add). It reuses the noise draws of the base run. $K$ parameters cost about $0.3K$ runs.
- The joint weight costs one reduction per step per parameter, which is negligible.
- These are costs per sample. The useful measure is cost for a target SE. Section 2.6 and Table 3.3 show that the SE, not the per-step cost, decides the comparison.

## 4. Resolved fast force (quiver analog)

[measured: `work_toy/quiver_lr.py`, `work_toy/out/quiver_s0.05.log`]

Non-interacting particles at rest, $x\sim\mathcal N(1,0.3^2)$, force $\theta A e^{-x^2/2}g(t)\cos(\omega_0 t)$ with a Gaussian envelope $g$ of width 5, $A=1$, $\sigma=0.05$, $\Delta t=0.005$, $6\times10^5$ particles. Observable $\langle v\rangle$ at $t=30$.

| $\omega_0$ | tangent (= FD) | LR | $\omega_{\rm rms}$ | LR relative SE |
|---|---|---|---|---|
| 2 | 0.0998 | 0.092 ± 0.009 | 17.9 | 0.09 |
| 4 | 0.0256 | 0.030 ± 0.005 | 19.7 | 0.19 |
| 8 | 0.0058 | 0.005 ± 0.006 | 20.3 | 1.0 |
| 16 | 0.0014 | 0.006 ± 0.012 | 20.3 | 8.6 |

LR agrees within 1 SE everywhere. The signal falls as $\omega_0^{-2}$ while $\omega_{\rm rms}$ stays at about 20. The particle count for a fixed relative SE therefore grows like $\omega_0^4$, as derived in Section 1.4. The pathwise tangent has SE $4\times10^{-4}$ or less at every $\omega_0$.

## 5. What this means for FBPIC [estimated]

These are extrapolations from the toys. None was run on FBPIC.

1. Shot-noise feedback. The mean-field weights in FBPIC will grow at least like $e^{\lambda t}$ with $\lambda=\langle|q(\eta_E+v\times\eta_B)|^2\rangle/(\sigma^2\operatorname{Var}\omega)$. Bounded weights need $\lambda T\lesssim1$. Bias control needs $\sigma^2T\le\beta^2\Delta u^2$ ($\Delta u$ a physical momentum spread, $\beta\ll1$). Together: $\langle|\delta F_\eta|^2\rangle T^2\lesssim\beta^2\Delta u^2$ per unit weight variance. The random-weight force accumulated coherently over the run must be below a small fraction of the bunch spread. In the 1D toy, $N=5.2\times10^5$ at $\sigma=0.03$ still gave $\lambda T\approx4$ at $T=60$ and 7 at $T=100$. A cheap, decisive FBPIC check: give every macroparticle of a Twin-S snapshot an i.i.d. unit weight, deposit $\delta\rho,\delta J$, advance the field tangent one step, gather $\delta F$, and compute $\lambda$ for the $\sigma$ allowed by the bias budget.
2. Resolved laser quiver. Every background electron that crosses the laser gets weight increments from $\partial_{\ln E_L}$ of the full quiver force. With typical values $a_0\approx2$, $\omega_0/\omega_p\approx20$, $\omega_p\tau_L\approx3.5$ (not taken from Twin-S), $\int|\partial F|^2dt\approx700\,(mc\,\omega_p)^2/\omega_p$. With $\sigma^2=10^{-3}$ (a 1 $mc$ noise spread over $1000/\omega_p$), $\operatorname{Var}\omega\approx7\times10^5$ per background electron before any feedback. The useful ponderomotive signal is smaller by $(\omega_p/\omega_0)^2$. Section 4 confirms this scaling.
3. Collective amplification. During wake formation and injection the true $\partial_\theta F$ is large. By Cauchy-Schwarz, $\operatorname{Var}\omega\ge(\delta v_\theta)^2/(\sigma^2T)$, where $\delta v_\theta=\int\partial_\theta F\,dt$ is the Eulerian momentum response per unit $\theta$. For $\theta=\ln E_L$ and a bunch whose energy gain changes by tens of $mc$, this requires a noise spread of tens of $mc$, which destroys the bunch physics. A smaller noise gives large weights.
4. Joint weight. At fixed particle paths, $\partial_{E_L}$ of the field is the vacuum laser tangent acting on every background electron. $N I_1$ is then enormous. Not a candidate.
5. Bias. The toys show $O(\sigma^2)$ bias only where the deterministic gradient exists at small scales (uniformly hyperbolic map, two-stream before saturation). After saturation, and in mixed phase space, the noisy gradient differs from every deterministic secant at $O(1)$. The noise level becomes a model choice that the inversion must fix and justify.

## 6. Files

All under `work_toy/`:

- `maps_lr.py`: doubling and standard maps with noise; LR, windowed LR, exact tangent, CRN FD; doubling-map transfer-operator reference (`pf` mode).
- `analyze_maps.py` -> `out/maps_tables.txt`; raw `out/dbl_s*.json`, `out/std_s*.json`, `out/dbl_pf.json`.
- `es1d_lr.py`: 1D-1V two-stream with momentum noise; mean-field LR with field-tangent coupling, joint LR, exact tangent with deposit tangent, CRN FD; `verify`, `diag`, `ens` modes. Environment variables `ES_V0`, `ES_EPS`, `ES_TREC`.
- `lam_test.py` -> `out/lam_test.log`: feedback rate in a stable cold plasma.
- `analyze_es.py` -> `out/es_ens_summary.txt`, `out/es_s0.03_N131k_summary.txt`, `out/es_fd_summary.txt`, `out/es_long_summary.txt`, `out/es_long_summary_ref0.1.txt` (`REF_H=0.1`).
- `es_det.py` -> `out/es_det_N131k.log`: deterministic FD at 5 step sizes and 5 loadings.
- `run_maps*.sh`, `run_es.sh`, `run_es_long.sh`: drivers. `out/diag_*.log`: single-seed time series.
- `quiver_lr.py` -> `out/quiver_s0.05.log`.
- `cost_es.py` -> `out/cost_es.log`.
