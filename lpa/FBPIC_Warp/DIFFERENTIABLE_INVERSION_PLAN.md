# Differentiable Inversion Plan for FBPIC-Warp

| Item | Value |
|---|---|
| Status | Draft, revised after two reviews, a scope change, and the switch to forward mode (Section 12.1). Nothing is implemented. |
| Date | 2026-10-26 |
| Branch | `snidhan/dev-autodiff` (base commit `92352b8`) |
| Scope | One GPU. Lab frame. Two parameters ($E_L$, $z_f$). Forward-mode tangents. Twin experiment with noise-free synthetic data. |
| Owner | The implementing agent writes all code: kernels, tangents, algorithms, scaffolding, and tests. |
| Production code | Unchanged. The differentiable path is a separate stepper in `fbpic/autodiff` (D17). |
| Decisions | D1 to D25 in Section 10. D1, D2, and D21 to D25 are user decisions. The others are planner defaults that the user can change. |

## How to use this plan

1. Read Sections 1 to 5 before you write code. They define the problem, the state, the stepper, and the ionization model.
2. Execute the phases in Section 7 in order. Do not start a phase before the previous phase passes its acceptance criteria.
3. For each phase, record the evidence in Section 12. Record commands, numbers, the GPU, and the commit hash.
4. Invoke the `learning-mode-gpu` skill in delivery mode before you write or edit a Warp kernel.
5. Ask the user before every commit. Propose a Conventional Commits message.
6. Follow `lpa/FBPIC_Warp/CLAUDE.md`. Rules 1, 4, 5, 6, and 8 apply directly. Rule 3 applies only if you change a production kernel, which this plan does not require.
7. Stop and ask the user when a phase says so, or when a measured result contradicts this plan. Record the result and the suspect assumption in Section 12 first.

## Contents

- [Notation](#notation)
- [Abbreviations](#abbreviations)
- [1. Purpose and scope](#1-purpose-and-scope)
- [2. Problem formulation](#2-problem-formulation)
- [3. Twin configuration](#3-twin-configuration)
- [4. Architecture](#4-architecture)
- [5. Expected-value ionization](#5-expected-value-ionization)
- [6. Verification ladder](#6-verification-ladder)
- [7. Rollout plan](#7-rollout-plan)
- [8. Memory and performance budget](#8-memory-and-performance-budget)
- [9. Risks and mitigations](#9-risks-and-mitigations)
- [10. Decisions](#10-decisions)
- [11. Future work](#11-future-work)
- [12. Execution results](#12-execution-results)
- [Appendix A. Unverified claims and required probes](#appendix-a-unverified-claims-and-required-probes)
- [Appendix B. File map](#appendix-b-file-map)
- [Appendix C. Research basis](#appendix-c-research-basis)

## Notation

| Symbol | Meaning | Units | Code name |
|---|---|---|---|
| $\theta = (E_L, z_f)$ | Laser energy, laser focal position (D21) | J, m | `laser_energy_J`, `laser_focal_position_m` |
| $\Theta$ | Box of admissible parameters | - | `config["bounds"]` |
| $\hat\theta$ | Normalized parameters, $\hat\theta_i = (\theta_i - \theta_i^{lo})/(\theta_i^{hi} - \theta_i^{lo})$ | - | `theta_hat` |
| $\theta^\ast$ | Hidden truth of the twin experiment | - | `theta_star` |
| $s_n$ | Complete state after $n$ steps (Section 4.2) | - | `StepState` |
| $\mathcal{D}_n$ | Discrete decisions of step $n$: sort permutations, removal ranges, injection counts, emission parents (Section 4.5) | - | `StepDecisions` |
| $\Phi$ | One step, $s_{n+1}, \mathcal{D}_n = \Phi(s_n)$ | - | `step` |
| $N_t$ | Steps from $t = 0$ to the exit snapshot | - | `num_steps` |
| $H$ | Observable map from $s_{N_t}$ to the beam at exit | - | `observable` |
| $y$ | Synthetic measured data (Section 2.2) | - | `target` |
| $J$ | Objective: the loss of Section 2.4, or a check functional of Section 6.2 | - | `objective` |
| $\dot s_n[j]$ | Tangent state, $\partial s_n / \partial\theta_j$ | - | `TangentState` |
| $K$ | Number of tangents, one per parameter (2) | - | `K` |
| $N_m$ | Azimuthal modes (2: $m = 0, 1$) (D23) | - | `Nm` |
| $N_z^a$, $N_r$ | Allocated z rows and radial cells | cells | `fld.Nz`, `fld.Nr` |
| $N_p$ | Macroparticles of all species | - | `Ntot` |
| $P_\ell$ | Population fraction of nitrogen ions at charge state $\ell \in \{5,6,7\}$ | - | `pop5`, `pop6`, `pop7` |
| $W_i$ | Pending birth weight of ion $i$ (Section 5) | - | `w_pending` |
| $\bar Z_i$ | Mean charge state, $5P_5 + 6P_6 + 7P_7$ | - | - |
| $a_\ell$, $q_\ell$, $c_\ell$ | ADK prefactor, field exponent, exponent prefactor (negative) of transition $\ell \to \ell+1$ | - | `adk_prefactor`, `adk_power`, `adk_exp_prefactor` |
| $p_\ell$ | One-step transition probability of $\ell \to \ell+1$ | - | `p_ion` |
| $z_R$ | Rayleigh length, $\pi w_0^2/\lambda_0$ | m | - |

## Abbreviations

| Abbreviation | Meaning |
|---|---|
| AD | Automatic differentiation |
| ADK | Ammosov-Delone-Krainov tunnel ionization rate |
| CRN | Common random numbers |
| DHT | Discrete Hankel transform (dense matrix product in FBPIC) |
| EV | Expected value (deterministic fractional ionization) |
| FD | Finite differences |
| FFT | Fast Fourier transform |
| JVP | Jacobian-vector product (forward mode) |
| MMD | Maximum mean discrepancy |
| PSATD | Pseudo-spectral analytical time domain field solver |
| RFF | Random Fourier features |
| SE | Standard error |
| VJP | Vector-Jacobian product (reverse mode) |

## 1. Purpose and scope

### 1.1 Purpose

Compute exact gradients of a beam-level objective with respect to two laser parameters, the laser energy and the focal position, through FBPIC-Warp (D21).
Use these gradients to recover hidden parameters from synthetic data in a twin experiment.

Success claim. A pass of Phase P9 shows that the inversion machinery recovers $E_L$ and $z_f$ in the compact, lab-frame, Gaussian-laser twin with the EV model. Transfer to the production stochastic model (truth variant T2) and to the production configuration is a separate result. This plan reports T2 but does not claim it.
This is the first goal of the project statement "Laser Wakefield Accelerator Condition Reconstruction": recover 3 to 5 known synthetic parameters. This plan starts with 2. The density parameters follow in Section 11.

$$
\hat\theta = \arg\min_{\theta \in \Theta} \; \mathcal{L}\big(H(F(\theta)),\, y\big).
$$

$F$ is the PIC forward model, and $H$ maps the simulated bunch to the observable. The targets are noise-free, so the plan uses no prior term (D22).

### 1.2 In scope

1. Single-GPU, lab-frame, 2-mode twin configurations at two sizes (Section 3).
2. A functional differentiable stepper with recorded discrete decisions (Section 4).
3. Hand-written tangent (JVP) kernels for gather, push, and deposition, checked against the existing VJPs (Section 4.7).
4. Expected-value ionization of the nitrogen K shell, with tangents (Section 5).
5. A forward-mode driver that returns $\partial J / \partial \theta$ for both parameters in one pass (Section 4.10).
6. A differentiable observable and loss (Section 2.4).
7. A local identifiability study before the gradient work (Phase P1).
8. A verification ladder from single operators to full-horizon gradients (Section 6).
9. Recovery of $\theta^\ast$ with a bounded quasi-Newton optimizer (Phase P9).

### 1.3 Out of scope

- Multiple GPUs, MPI, and NCCL exchanges of tangents (D1).
- The boosted frame and back-transformed diagnostics (D3).
- The LASY antenna laser (D4).
- CUDA graphs. The differentiable path runs eagerly (D9).
- Reverse mode, checkpointing, and FFT and DHT adjoints (D25). The design is kept at commit `07dcc6c`.
- Reverse-time trajectory reconstruction.
- Changes to production stepping, ionization, sorting, or buffer handling (D17).
- Experimental data, GPSR targets, and multi-shot jitter (Section 11).

### 1.4 Research basis in brief

Appendix C lists the sources.

1. In reverse mode, the memory problem is the forward trajectory, not the adjoint state. Forward mode stores no trajectory: memory is one primal state plus one tangent state per parameter. With 2 parameters, forward mode is cheaper and simpler. Reverse mode with checkpointing is the route for many parameters.
2. All field operators are linear with fixed coefficients, and no parameter enters them.
3. Pathwise AD through `random < p` returns zero for the change in electron count. No published PIC code differentiates field ionization. Expected-value ionization makes the populations smooth in the fields. Emission decisions remain discrete (Section 5).
4. Gradients through long, nonlinear runs can grow without bound. Every AD gradient must agree with FD before an optimizer uses it.

### 1.5 Evidence status

| Item | Status |
|---|---|
| Per-operator VJPs for push, gather, deposition, and pointwise field operators | Verified: 122 tests on GPU, 118 on CPU with 4 skipped, commit `24fb4c7` |
| Tangent kernels for gather, push, and deposition | Not implemented |
| One-step, horizon, and parameter gradients | Not measured |
| Memory and run time of a forward-mode gradient pass | Not measured |
| Expected-value ionization | Not implemented |
| Identifiability of the two parameters in the twin | Not measured |

## 2. Problem formulation

### 2.1 Parameters and their entry points

| Parameter | Production bounds | Enters the state through | Derivative |
|---|---|---|---|
| $E_L$ | 1.5 to 3.5 J | All six initial interp fields (Er, Et, Ez, Br, Bt, Bz), set by `add_laser_direct` | Fields scale as $\sqrt{E_L}$: tangent $= F_0 / (2E_L)$ (Section 4.8) |
| $z_f$ | 2.5 to 4.5 mm | All six initial interp fields | Analytic profile derivative pushed through the full `add_laser_direct` map; check with host FD of that map |

$n_0$ and $F_L$ are fixed at their nominal values (D21). Particle weights therefore do not depend on $\theta$. The density derivatives below are kept for the future density parameters (Section 11):

| Future parameter | Production bounds | Enters the state through | Derivative |
|---|---|---|---|
| $n_0$ | 0.5 to 8.0 $\times 10^{18}$ cm$^{-3}$ | Weights of injected or preloaded particles of every species | $\partial w / \partial n_0 = w / n_0$, exact |
| $F_L$ | 0.8 to 3.0 mm | Weights of injected or preloaded particles with $z < z_c$ | Closed form below |

The twin density is the template's `BilateralSuperGaussian`, multiplied by a fixed indicator:

$$
n(z) = n_0\, S(z)\, \mathbb{1}_{[z_{lo}, z_{hi}]}(z), \quad S(z) = \exp\!\Big(-\ln 2 \,\big(4 (z - z_c)^2 / F^2\big)^{\beta/2}\Big),
$$

with $F = F_L$ for $z < z_c$ and $F = F_R$ otherwise. Then

$$
\frac{\partial S}{\partial F_L} = S \,\ln 2\, \beta \, \frac{(4 (z-z_c)^2/F_L^2)^{\beta/2}}{F_L}, \quad z < z_c .
$$

$z_c$, $\beta$, $F_R$, $z_{lo}$, and $z_{hi}$ are fixed (D6). Choose $z_{lo}$ so that $S(z_{lo}; F_L) \ge 10^{-12}$ at the fixed $F_L$ (and at $F_L^{lo}$ once $F_L$ is inverted). Then no weight underflows, and the set of particles does not depend on $\theta$.

Gradients are reported in both units: $\partial J/\partial\hat\theta_i = (\theta_i^{hi} - \theta_i^{lo})\, \partial J/\partial\theta_i$. The FD harness and the optimizer use $\hat\theta$.

### 2.2 Twin experiment and truth variants

1. Choose $\theta^\ast$ inside $\Theta$ (Section 3.4).
2. Generate the target $y$ from a truth run (variants below).
3. Start the optimizer from a different $\hat\theta_0$ and recover $\hat\theta^\ast$.

Gradient checks (Phases P5 to P7) use one configuration for AD and FD, because they test the derivative of one discrete map.
Recovery runs (Phases P3 and P9) use the truth variants to avoid an inverse crime. Every variant keeps the inversion grid, $\Delta t$, $N_t$, `exchange_period`, and the `n_move` sequence, so that $H$ is evaluated at the same time. Every variant uses a different NumPy injection seed and `p_nz = 2` instead of 1.

| Variant | Ionization model | Role |
|---|---|---|
| T0 | EV with the inversion settings ($K_e$, $\epsilon$) | Control. L6 applies. |
| T1 | EV with immediate emission ($\epsilon = 0$), or $\epsilon = 10^{-6}$ if the births do not fit Twin-S memory | Main test. L6 applies. |
| T2 | Production stochastic ionization (Philox, tracked ions) | Model error. Report only. |

If T0 passes and T1 fails, stop. Report the recovered-parameter bias together with the emission-delay results of Phase P8, and ask the user.

### 2.3 Observable H

The twin runs in the lab frame (D3). The observable is the state of the species `elec_from_N` at one fixed step $N_t$.
Choose $N_t$ so that the window has passed $z_{hi}$ for every $\theta \in \Theta$. $z_{hi}$ does not depend on $\theta$.

The selection weight is a quintic smoothstep on $u_z$:

$$
\sigma_i = S_5\big((u_{z,i} - u_c + 2\delta_u)/(4\delta_u)\big), \quad S_5(t) = 6t^5 - 15t^4 + 10t^3 \text{ on } [0,1],
$$

with $S_5 = 0$ for $t \le 0$ and $S_5 = 1$ for $t \ge 1$. $\sigma_i$ is exactly 0 below $u_c - 2\delta_u$, so particles removed at the window back do not contribute. Phase P1 sets $u_c$ and $\delta_u$ (Section 3.4). The effective weight is $w_i\sigma_i$. There are no hard cuts and no quantile crops.

### 2.4 Loss

$$
J(\theta) = \alpha_Q \big(\log(Q + Q_f) - \log(Q_y + Q_f)\big)^2 + \alpha_M \sum_k \frac{(m_k - m_{k,y})^2}{s_k^2} + \alpha_D\, \mathrm{MMD}^2_{\mathrm{RFF}}.
$$

- $Q = e \sum_i w_i \sigma_i$ is the selected charge. $Q_f = 10^{-3} Q_y$.
- $m_k = \sum_i w_i\sigma_i f_k(\xi_i) / (\sum_i w_i \sigma_i + Q_f/e)$ are the weighted first and second moments of $\xi = (x, u_x, y, u_y, z - z_{win}, u_z)$.
- $s_k$ are fixed scales from the target: the target standard deviation for first moments and its square for second moments.
- $\mathrm{MMD}^2_{\mathrm{RFF}} = \lVert \mu - \mu_y \rVert^2$, with $\mu = \sum_i w_i\sigma_i \phi(\tilde\xi_i) / (\sum_i w_i\sigma_i + Q_f/e)$. The denominator has the same floor as the moments. $\phi$ has $D = 512$ random Fourier features, seed 1234, bandwidth 1.0. $\tilde\xi$ is $\xi$ standardized with the frozen target mean and standard deviation, never with statistics of the current sample.
- The weight gradient is
  $$
  \frac{\partial\, \mathrm{MMD}^2}{\partial (w_i\sigma_i)} = \frac{2\,\langle \mu - \mu_y,\; \phi(\tilde\xi_i) - \mu\rangle}{\sum_k w_k \sigma_k + Q_f/e}.
  $$
- Empty and nearly empty beams. If every $\sigma_i = 0$, then $\mu = 0$, $m_k = 0$, $\mathrm{MMD}^2 = \lVert\mu_y\rVert^2$, and the charge term is $\alpha_Q(\log Q_f - \log(Q_y + Q_f))^2$. The loss and every gradient stay finite. The trapping checks of Section 3.4 do not cover every point that the optimizer can visit, so the loss must not rely on a positive selected charge.
- No prior term. The targets are noise-free (D22). A prior, as in noisy recovery (Section 11), depends on $\hat\theta$ directly; the objective must then also return that direct gradient.
- Defaults: $\alpha_Q = 1$, $\alpha_M = 1/K$ for $K$ moments, $\alpha_D = 1$.

Implement the loss and its analytic gradient in CuPy or Warp (D11). Test every term with FD on random particle sets (L1). Also test the loss and its gradient on an empty selection (all $\sigma_i = 0$) and on a nearly empty selection (one particle with $\sigma_i = 10^{-6}$). Both must be finite and match FD.

## 3. Twin configuration

### 3.1 Why a new configuration

The production template `IonizationInjectionSimulation` cannot serve as the twin:

- It hard-codes `use_mpi=True`, which forces `n_order=32` on one GPU.
- It forces the LASY antenna, which evaluates the laser on the host every step.
- It cannot freeze `beta_window`, and `num_steps` moves with $n_0$ and $F_L$.
- He and N start neutral, so every plasma electron is born by ionization.
- The exit observable is assembled from back-transformed slices, with hard cuts.

Build a standalone FBPIC-level configuration module instead (D5). Base it on `docs/source/example_input/ionization_script.py` and on `tests/test_gpu_end_to_end.py::_lwfa`.

### 3.2 Configuration choices

| Item | Twin value | Reason |
|---|---|---|
| Frame | Lab (`gamma_boost=None`) | One exit snapshot. No slice assembly. (D3) |
| Modes | $N_m = 2$ | Mode 0 holds the wake and mode 1 the linearly polarized laser (D23). Production kernels support $N_m = 2$ for the primal. The tangent kernels are generic over $N_m$. The twin drops the $m = 2$ content of a linearly polarized laser. |
| Stencil | `n_order=-1` | Single-rank PSATD. `n_guard=64`, `exchange_period=14`. (D2) |
| Shape | Linear | The tangent kernels and the reference VJP clones cover linear shapes only. |
| Field solver | Standard PSATD, curl-free correction, `filter_currents=True`, `use_true_rho=False` | Lab frame. All are linear in the fields, so their tangents reuse them. |
| Boundaries | z open, r reflective | No PML tangent is needed. |
| Laser | `GaussianLaser`, `method='direct'` | Parameters enter only the initial fields. (D4) |
| Helium | He$^{2+}$ ions with fixed charge $2e$ and no ionizer; electrons loaded | No background births. (D7, D15) |
| Nitrogen | N$^{5+}$ ions; L-shell electrons loaded | Only K-shell transitions remain. (D7) |
| K-shell electrons | Separate species `elec_from_N` | The observable and EV births act on this species only. |
| Ionization model | EV for N$^{5+}\to$N$^{6+}\to$N$^{7+}$ | Section 5. (D7) |
| Particles per cell | `p_nz=1`, `p_nr=2`, `p_nt=8` for every species | FBPIC requires `p_nt >= 4 Nm` (`fbpic/main.py:950`), which is 8 for $N_m = 2$. The Phase P1 aliasing check confirms it. (D16) |
| Density | `BilateralSuperGaussian` times the fixed indicator $[z_{lo}, z_{hi}]$ | No underflow, $\theta$-independent particle set. (D6) |
| Plasma start | Set through `dens_func` only | `p_zmin` limits only the initial load. The injector uses `dens_func`. |
| Moving window | $v = c$ | `n_move` is independent of $\theta$. |
| Steps | Fixed $N_t$ for all $\theta$ | No $\theta$-dependent step count. |
| Hankel precision | `float64` | FP32 Hankel gradients are not validated. |
| CUDA graphs | Off (`use_cuda_graphs=False`) in every differentiable run | Eager launches keep the stepper simple. Graphs are a later optimization. (D9) |
| RNG | Fixed NumPy seed for injection angles | Identical injection sequence in AD and FD runs. |

### 3.3 Sizes

| Size | Purpose | Grid | Steps | Particles (estimate) |
|---|---|---|---|---|
| Twin-XS | Operator, one-step, and short-horizon checks | 96 x 32 cells, as in `_lwfa`, with the twin density preloaded so that the laser meets the up-ramp within 20 steps | 50 to 300 | 0.1 to 0.5 M |
| Twin-S | Horizon sweep, identifiability, recovery | About 800 x 50 physical cells | About 1800 | 2 to 6 M (about 1.9 M loaded at `p_nt=8`, plus EV births) |

Twin-M, a larger twin closer to production physics, is deferred (D24, Section 11).

Twin-S keeps the compact physics of `ionization_script.py`: $a_0 \approx 4$, $w_0 = 5$ µm, $\tau = 16$ fs, $\lambda_0 = 0.8$ µm, so $z_R \approx 98$ µm.

- $a_0 \propto \sqrt{E_L}$ with fixed $\tau$, $w_0$, and $\lambda_0$, from the template formula `peak_power = E / tau / sqrt(pi/2)`.
- $z_f$ is measured from $z_c$. Nominal $z_f - z_c = +0.53\, z_R$, bounds nominal $\pm 0.44\, z_R$. These ratios match production (2.5 to 4.5 mm, $z_c = 2.3$ mm, $z_R = 2.26$ mm).
- $n_0$ and $F_L$ are fixed at nominal values (D21).
- Record the plasma length in units of $z_R$. Production is about 0.9. Twin-S is at most about 0.5, which weakens $z_f$ sensitivity. Phase P1 measures this.

EV births add particles. Phase P3 measures their count. Section 8 scales the memory rows with it.

### 3.4 Fixed choices and targets

Store all fixed values in `inversion/configs/<size>.json`. Commit the file. A later change invalidates every L4 result recorded for that file. Re-run those checks after a change.

1. Twin physics (fill in Phase P1, freeze before Phase P2): $z_c$, $\beta$, $F_R$, $z_{lo}$, $z_{hi}$, the He to N ratio, the laser $z_0$, nominal $\theta$, bounds, $N_t$, and seeds.
2. Selection: choose $u_c$ at the minimum of the nominal $u_z$ histogram between the trapped peak and the low-energy population. Let $u_{lo}$ be the 99th weighted percentile of the low-energy population and $u_{hi}$ the 1st percentile of the trapped peak. Set $\delta_u = \min(u_c - u_{lo},\, u_{hi} - u_c)/4$.
3. Trapping check: at the nominal $\theta$, at the 4 corners of $\Theta$, and at the 5 optimizer starts, require at least 1000 macroparticles with $\sigma_i > 0.5$ and $Q \ge 0.1\,Q(\text{nominal})$. If a point fails, shrink the bounds of the parameter that causes it, record the new $\Theta$, and tell the user.
4. Optimizer starts: 5 points from a seeded Latin hypercube in $[0.1, 0.9]^2$, each at least 0.25 from $\hat\theta^\ast$ in the infinity norm.
5. Targets: write them with `inversion/make_target.py` to `inversion/targets/<size>_<variant>_Nt<N>.npz`. Keys: `Q`, `moments`, `rff_mean`, `standardization`, `s_k`, `theta_star`, `seed`, `p_nz`, `N_t`, `variant`, `config_sha256`, `git_hash`. `value_and_grad` refuses a target whose `config_sha256` or `N_t` does not match.
6. Noise: none. Every target is a noise-free simulation output (D22). Noisy recovery is future work (Section 11).

## 4. Architecture

The main engine is forward mode (D25). Each step advances the primal state and $K = 2$ tangent states together, one tangent per parameter. The gradient is the inner product of the loss gradient with the final tangents. No tape, checkpoint, or replay is needed. The reverse-mode design (checkpointed one-step tape, layout check, FFT and DHT adjoints) is kept at commit `07dcc6c` of this file for the future many-parameter case (Section 11).

### 4.1 Data flow

```text
init (host, once):
  sim = build_twin(theta, size, config)          # production objects for grids and coefficients only
  s_0 = begin(sim)                               # laser fields, prologue, empty or preloaded plasma
  t_j = laser_tangents(sim, theta)               # j in {E_L, z_f}; Section 4.8
  sdot_0[j] = seed(t_j)                          # damping and interp2spect of t_j; particle tangents zero

one pass, no tape:
  for n in 0 .. N_t-1:
      s_{n+1}, sdot_{n+1}, D_n = step(s_n, sdot_n)   # primal and both tangents together
  J, g_s = objective(s_{N_t})                    # g_s = dJ/ds_{N_t}, analytic (Section 2.4)
  dJ/dtheta_j = <g_s, sdot_{N_t}[j]>             # paired-real inner product over fields and particles
```

### 4.2 Step state and tangent state

`StepState` is a container of device arrays and host scalars. `TangentState` holds the tangents of the differentiable arrays of `StepState`, with a leading dimension $K$.

| Part | Primal contents | Tangent contents | Size per entry |
|---|---|---|---|
| Fields | Spectral `Ep, Em, Ez, Bp, Bm, Bz` and `rho_prev` for each mode | Same arrays, $K$ copies | $16\, N_z^a N_r$ B per array |
| Particles, every species | `x, y, z, ux, uy, uz, inv_gamma, w`, tracker `id` | `x, y, z, ux, uy, uz, inv_gamma, w`, $K$ copies | 72 B primal, 64 B per tangent |
| N ions | `pop5, pop6, pop7, w_pending` | Same, $K$ copies | 32 B per tangent |
| Sorting, every species | `prefix_sum`, `prefix_sum_shift`, `sorted` flag | None | $4 N_z (N_r+1)$ B |
| Window and grid | `moving_win.zmin`, `t_last_move`, `comm._zmin_global_domain`, `fld.interp[m].zmin` and `zmax` | None | scalars |
| Injection, every species | `z_inject`, `z_end_plasma`, `nz_inject` | None | scalars |
| Tracking, every species | `next_attributed_id` | None | scalars |
| Clock, RNG | `time`, `iteration`, NumPy RNG state | None | scalars |

Rules:

1. Keep `inv_gamma` as its own variable. The Vay push reads the stored old value, and the position push and J deposit read the new value.
2. `w_times_level` is derived, not state: $w\bar Z - W$ for N ions (Section 5.1). Its tangent is $w\sum_\ell \ell\,\dot P_\ell - \dot W$. He ions carry a fixed charge $2e$.
3. The interp E and B are derived from the spectral fields at the start of each step (`spect2interp`). Their tangents come from the same transform applied to the spectral tangents.
4. Give every species a tracker (`track()`). L2 and the FD harness match particles by ID.
5. The weight tangent is zero for every species except `elec_from_N`, whose births carry $\dot W$ (Section 5.3). Keep the array for all species, for the future density parameters.

### 4.3 The stepper

`fbpic/autodiff/stepper.py` holds `begin` and `step`. Production `fbpic/main.py` stays unchanged (D17).

- `begin(sim) -> StepState` replicates the `Simulation.step` prologue exactly once (`fbpic/main.py:475-518`): injection position setup, damping of the interp E and B, `interp2spect` of E and B, the first particle exchange, and the `rho_prev` deposit.
- `step(state, tangents=None) -> (state, tangents, decisions)` mirrors the loop body at `fbpic/main.py:533-687` for the twin configuration, in the same order:
  1. exchange iteration (`iteration % exchange_period == 0`): removal, injection, `rho_prev` deposit;
  2. `spect2interp` of E and B; gather;
  3. Vay momentum push; first half position push;
  4. EV ionization and emission (Section 5), at the production slot of `handle_elementary_processes` (`main.py:618-620`);
  5. sort; J deposit; second half position push; sort; `rho_next` deposit;
  6. curl-free current correction; PSATD push; `push_rho`;
  7. moving-window shift; open-z damping (`exchange_and_damp_EB`).
- After each primal operation, `step` applies the tangent of that operation to all $K$ tangents (Section 4.4). With `tangents=None`, it is the plain forward step.

Rules:

1. The primal operations call production kernels where they exist. No tape is active, so forward-only kernels are allowed. Production kernels support $N_m = 2$.
2. When a primal operation updates an array in place, first copy the inputs that its tangent needs into a per-step stash. Examples: `ux, uy, uz, inv_gamma` before the Vay push, and positions before each half push.
3. The primal result must not depend on whether tangents are computed. L2t checks this.
4. The step-0 diagnostic J deposit of production (`main.py:561-562`) is not reproduced. The J deposit of the same step overwrites it before any field push.
5. Exchange cadence and window shifts come from the `iteration` and window scalars stored in the state.

Use `step` without tangents for every FD run and every EV truth run. Production `Simulation.step` is the physics reference for L2 and the stochastic model for L5 and truth variant T2.

### 4.4 Operations and their tangents

| Production operation | Tangent | Source | Reference for L1 |
|---|---|---|---|
| `spect2interp`, `interp2spect` (FFT and DHT) | Same operator on the tangent arrays | Production `SpectralTransformer` methods | Linearity check |
| Curl-free current correction, PSATD push, `push_rho`, filter, `erase`, `divide_by_volume` | Same operator on the tangent arrays | Production or `fields.py` functional twins | Linearity check |
| Moving-window spectral shift, open-z damping | Same operator on the tangent arrays | Production or `fields.py` twins | Linearity check |
| Gather E, B at particles | `gather_tangent` | New kernel (Section 4.7) | Dot product with the 5-mode gather VJP; `check_jvp` |
| Vay momentum push, electrons and He | `vay_tangent` | New kernel | Dot product with `push_momenta` VJP; `check_jvp` |
| Vay momentum push, N ions (charge $(w\bar Z - W)e/w$) | `vay_tangent` with the charge tangent | New kernel (Phase P6) | Dot product with the Warp-generated VJP of `push_ion_momenta_ev`; `check_jvp` |
| Position half push (x2) | `position_tangent` | New kernel | Dot product with `push_positions` VJP; `check_jvp` |
| Deposit J, `rho_next`, and the exchange-step `rho_prev` | `deposit_tangent` | New kernel (Section 4.7) | Dot product with the 5-mode deposition VJP; `check_jvp` |
| EV ionization and emission | `ev_tangent` | New kernel (Section 5.3) | Dot product with the Warp-generated VJP of the EV kernel; `check_jvp` |
| Sort before each deposit | The primal permutation applied to the tangent arrays | `cupy.take` | Exact |
| Exchange-step removal and injection | Section 4.9 | Host and CuPy | Exact |

The field side is linear with $\theta$-independent coefficients (Section 1.4). Its tangent is the operator itself, so no new field code is needed.

### 4.5 Discrete decisions

The primal step makes every discrete decision: cell indices, the axis branch, sort permutations, removal ranges, injection counts, and emissions. The tangents follow these decisions and never change them. Forward mode therefore computes the conditional derivative of the step with the decisions fixed.

Rules:

1. Sort with the production key `ir_upper + iz_upper*(Nr+1)` (`cuda_sorting.py:94`) and a stable sort (`cupy.argsort` on the key). Apply the same permutation to the primal and to every tangent array.
2. Assign slots for emission and injection with an exclusive prefix sum over the parent index or batch. Never use an atomic slot counter.
3. `StepDecisions` records only what the FD harness needs (Section 6.3): the `n_move` sequence, the injected count per species and event, and the emitting parent IDs per step.

Frozen decisions also define the FD reference (Section 6.3). With the decisions fixed, the forward map is piecewise smooth in $\theta$, and the tangents are its exact derivative.

### 4.6 Field tangents

Every field operation in the step is linear in the field arrays, and no parameter enters its coefficients. The tangent of each operation is the same operation applied to the tangent arrays.

1. Call the operator once per tangent, or once on a batched array where the operator allows it.
2. Phase P4 checks linearity for each operator: $\lVert A(x + \alpha y) - Ax - \alpha Ay\rVert \le 10^{-13}\,(\lVert Ax\rVert + \lvert\alpha\rvert\lVert Ay\rVert)$ for 3 random draws.
3. The interp tangents of E and B are derived from the spectral tangents at the start of each step, like the primal.

### 4.7 Particle tangent kernels

Warp 1.18 generates reverse-mode code only. The tangent kernels are hand-written, generic over $N_m$ as a compile-time constant, and process all $K$ tangents in one launch.

- Gather. For mode $m$, the gathered value is $G_m = \sum_c S_c(z, r) F_{m,c}$ over the cells $c$ of the particle. Its tangent is $\dot G_m = \sum_c S_c \dot F_{m,c} + \sum_c (\partial_z S_c\,\dot z + \partial_r S_c\,\dot r) F_{m,c}$, with $\dot r = (x\dot x + y\dot y)/r$. Differentiate the mode factor and the Cartesian rotation with $\dot\varphi = (x\dot y - y\dot x)/r^2$, $d e^{im\varphi} = i m \dot\varphi\, e^{im\varphi}$.
- Deposition. The tangent of $\rho_{m,c} = \sum_i q\, w_i S_c(z_i, r_i)\, e^{-im\varphi_i}$ is $\sum_i q\,\big[\dot w_i S_c + w_i(\partial_z S_c\,\dot z_i + \partial_r S_c\,\dot r_i) - i m\, w_i S_c\,\dot\varphi_i\big] e^{-im\varphi_i}$. J adds the velocity tangent $\dot v = c(\dot u\,\gamma^{-1} + u\,\dot{(\gamma^{-1})})$, with $\dot{(\gamma^{-1})} = -\gamma^{-3}\, u\cdot\dot u$. Differentiate the Ruyten-corrected radial shape, not the gather shape. Deposit with atomics into $K$ tangent grids.
- Vay push. Differentiate the production `push_p_vay` sequence line by line in a `@wp.func` that carries each value with its $K$ tangents.
- Position push. $\dot x = (c\,\Delta t/2)(\dot u\,\gamma^{-1} + u\,\dot{(\gamma^{-1})})$.
- Linear shapes have a derivative in position that jumps at cell crossings. The cell, the Ruyten index, and the axis branch are fixed inside one step.
- Guard the axis. For $r$ below the production axis threshold, use the production branch ($\cos\varphi = 1$, $\sin\varphi = 0$) and set $\dot\varphi = 0$. The L3 harness masks these particles.
- Validate each kernel at $N_m = 5$ by a dot product with the existing VJP clones: $e = \lvert\langle J u, v\rangle - \langle u, J^T v\rangle\rvert / (\lVert Ju\rVert\lVert v\rVert + \lVert u\rVert\lVert J^T v\rVert) \le 10^{-12}$ for 3 random draws. Then run `check_jvp` at $N_m = 2$ and $N_m = 5$.

### 4.8 Laser parameter tangents

With the direct method, the laser sets all six initial interp fields once, on the host, before `begin`:

$$
F_0(\theta) = \mathcal{L}\big[\mathcal{E}(\cdot\,; E_L, z_f)\big].
$$

$F_0$ holds Er, Et, Ez, Br, Bt, Bz for every mode, on the local layout with guard cells. $\mathcal{L}$ is the complete linear map of `add_laser_direct` (`fbpic/lpa_utils/laser/direct_injection.py:13-106`):

1. `get_laser_Er_Et`: sample the profile and do the azimuthal decomposition;
2. copy Er and Et into a host `Fields` on the grid without guard cells;
3. `calculate_laser_fields`: `interp2spect`, the kz filter $(1 - \sin^4(k_z \Delta z/2))$ on Ep and Em, Ez from $\nabla\cdot E = 0$, B from the curl with `propag_direction`, then `spect2interp` of E and B;
4. add all six components into rows `[iz_in_array, iz_in_array + Nz_local)`. On one rank `iz_in_array = n_guard = 64`, and the guard rows receive zero.

`begin` then applies open-z damping $\mathcal{D}$ and `interp2spect` $\mathcal{T}$. Both are linear. The initial tangents are

$$
\dot s_0^{EB}[j] = \mathcal{T}\mathcal{D}\, t_j, \qquad \dot s_0^{\text{particles}}[j] = 0, \qquad \dot\rho_{prev,0}[j] = 0.
$$

The `rho_prev` tangent is zero because the particle tangents are zero at $t = 0$. The laser tangents are:

- $t_{E_L} = F_0/(2E_L)$. This is exact, because $\mathcal{L}$ is linear and $\mathcal{E} \propto \sqrt{E_L}$, and the initial fields hold only the laser. Store $F_0$ at init.
- $t_{z_f} = \mathcal{L}[\partial\mathcal{E}/\partial z_f]$. For the Gaussian, the transverse profile is $T = \exp(-r^2/(w_0^2 D))/D$ with $D = 1 + i\,p\,(z - z_f)/z_R$, so $\partial T/\partial z_f = T\,(-ip/z_R)\,\big(r^2/(w_0^2 D^2) - 1/D\big)$. Write a host class `GaussianLaserDzf(LaserProfile)` whose `E_field` returns this derivative and that exposes `propag_direction`. Write a non-mutating helper `laser_initial_fields(sim, profile)` that runs steps 1 to 4 into new zero arrays. Call it with `GaussianLaserDzf`, and store the result on the device before `begin`.

Probes (Phase P5):

1. `laser_initial_fields(sim, GaussianLaser(theta))` equals the six arrays that `add_laser_direct` writes into zeroed fields, within $10^{-14}$ relative.
2. $t_{z_f}$ matches a host central FD in $z_f$ of `laser_initial_fields`, for all six components and all modes, with $h \in \{10^{-2}, 10^{-3}, 10^{-4}\}$ of the normalized range, within $10^{-6}$ relative.
3. $\dot s_0^{EB}[E_L]$ equals $s_0^{EB}/(2E_L)$ within $10^{-14}$ relative, where $s_0^{EB}$ is the six spectral E and B arrays after `begin`.

### 4.9 Structural events

Exchange iterations remove particles at the back of the window, inject new particles at the front, and redeposit `rho_prev`.

- Removal: drop the tangents of the removed range with the primal arrays.
- Injection: append zero tangents. Positions, angles, momenta, and (under D21) weights of injected particles do not depend on $\theta$. For the future density parameters (Section 11), the injected weight tangent is $\dot w_i = \partial w_i/\partial\theta_j$ from the closed forms of Section 2.1.
- `rho_prev` redeposit: `deposit_tangent` from the particles and their tangents on that step.
- `w_times_level` and its tangent are computed immediately before each deposit, including the exchange-step `rho_prev` deposit.

### 4.10 Driver

`fbpic/autodiff/driver.py` runs one pass of `step` with $K = 2$ tangents (D25).

1. Build the tangent seeds (Section 4.8).
2. Run $N_t$ steps. Keep only the current primal state, the current tangents, and the per-step stash.
3. Evaluate the objective and its analytic state gradient $g_s$ at $s_{N_t}$. Contract it with each final tangent.

Driver API:

```python
result = value_and_grad(theta, objective, config)
# result.J: float
# result.grad: dJ/dtheta, shape (2,), float64, host
# result.grad_hat: dJ/dtheta_hat = diag(theta_hi - theta_lo) @ grad
# result.info: GPU and host memory peaks, timings, per-step norms of each tangent
# objective(state) -> (J, g_s); the Section 2.4 loss or a Section 6.2 functional

state, tangents = jvp(theta, config, seeds, n_steps)
# full tangent state for field-level checks such as L4w
```

## 5. Expected-value ionization

### 5.1 Model

Each N ion carries populations $P_5, P_6, P_7$ with $P_5 + P_6 + P_7 = 1$ and a pending birth weight $W$. At injection, $P = (1, 0, 0)$ and $W = 0$.
For each step, compute the rest-frame field $E$ and the Lorentz factor $\gamma$ with the production formulas (`warp_inline_functions.py:14-76`):

$$
x_\ell = \frac{1}{\gamma}\, a_\ell\, E^{q_\ell} \exp(c_\ell/E), \quad p_\ell = 1 - \exp(-x_\ell) = -\mathrm{expm1}(-x_\ell), \quad \ell \in \{5, 6\}.
$$

Compute $p_\ell$ with `expm1` and the survival factor $1 - p_\ell$ directly as $\exp(-x_\ell)$. Forming $1 - p_\ell$ from $p_\ell$ loses precision near saturation.

Use the old populations, which matches production's limit of one level per step:

$$
\Delta_{56} = P_5\, p_5, \quad \Delta_{67} = P_6\, p_6,
$$
$$
P_5' = P_5 - \Delta_{56}, \quad P_6' = P_6 + \Delta_{56} - \Delta_{67}, \quad P_7' = P_7 + \Delta_{67}, \quad W' = W + w(\Delta_{56} + \Delta_{67}).
$$

Status of the model:

- For prescribed fields and a prescribed ion trajectory, these updates give the exact expected populations of production's one-transition-per-step process. Deterministic field-ionization models have published precedent (DiIorio et al., Appendix C).
- In the coupled simulation, the fields respond to the electrons. Ionization, trajectories, and trapping then become correlated. The complete simulation is therefore a deterministic rate-equation PIC model, not the exact expectation of stochastic PIC. L5 measures the difference.
- Mean-charge closure: $P_5, P_6, P_7$ share one position and momentum, and the ion is pushed with the mean charge $w_\text{tl}$. Different charge states feel different forces, and the pending weight changes the charge used for the force. Phase P3 checks this closure with an immobile-ion comparison.

### 5.2 Emission and charge neutrality

Run the update and the emission at the production slot: after the Vay push and the first half position push, before the J deposit.

- Each ion with $W > \epsilon w$ emits one electron into `elec_from_N` in the same step. The electron gets weight $W$ and copies `x, y, z, ux, uy, uz, inv_gamma` from the parent ion. Then $W$ is reset to 0 for the emitting ions only.
- Ions with $W \le \epsilon w$ keep $W$ for later steps. An ion that never crosses the threshold never emits.
- Deposit and push each N ion with `w_times_level` $= w\bar Z - W$. This equals $5w$ plus the total weight the ion has emitted. Pending births are therefore neutral, and an emission raises the ion charge by $W$ at the position and in the step of the new electron. This matches production, which changes the ion level and creates the electron in the same call.
- Merge per parent ion, not per cell. A per-cell merge averages over the azimuth and destroys the $m = 1$ content of a linearly polarized laser.
- Defaults (D12): $K_e = 1$ (emission checked every step), $\epsilon = 10^{-4}$. Do not use $K_e > 1$: emission copies the parent momentum, so a delay of $K_e - 1$ steps shifts the birth phase (22.5 degrees per step in Twin-S) and adds up to about $0.9\, a_0$ of residual transverse momentum.
- The threshold $\epsilon$ causes the same kind of delay, even with $K_e = 1$. An ion that gains $10^{-6} w$ per step reaches $\epsilon w = 10^{-4} w$ only after about 100 steps. Its electrons are then released at a later position and laser phase, and during the delay they miss acceleration and do not carry current. Charge stays conserved, but the birth history changes. A small delayed weight can still change the trapped charge or the emittance if it sits at a sensitive birth phase. $\epsilon$ is therefore a model parameter. Section 5.5 validates it against immediate emission.
- Grow `elec_from_N` by reallocation at each emission, as production does. Assign slots with an exclusive prefix sum in parent order. Use the parent index to copy tangents (Section 5.3). Record the emitting parent IDs in `StepDecisions` for the FD harness.
- The emission kernel takes an optional `emit_mask` (by parent ion ID). When the mask is given, it replaces the threshold test. Frozen-decision FD runs use it (Section 6.3).

### 5.3 Tangent

Use closed-form derivatives. The ADK probability has

$$
\frac{\partial p_\ell}{\partial E} = (1 - p_\ell)\, x_\ell \left( \frac{q_\ell}{E} - \frac{c_\ell}{E^2} \right), \qquad \frac{\partial p_\ell}{\partial \gamma}\Big|_E = -(1 - p_\ell)\,\frac{x_\ell}{\gamma}.
$$

- Evaluate the factor $1 - p_\ell$ as $\exp(-x_\ell)$, as in Section 5.1. Then $\log(1 - p_\ell) = -x_\ell$ exactly, and no term is singular or loses precision at saturation.
- Guard $\sqrt{E^2}$ at $E = 0$.
- Compute $\dot E$ and $\dot\gamma$ from $E = \sqrt{|\gamma \mathbf{E} + \mathbf{u} \times c\mathbf{B}|^2 - (\mathbf{u}\cdot\mathbf{E})^2}$ and $\gamma = \sqrt{1 + u^2}$, with the gathered field tangents and the ion momentum tangents. Then $\dot p_\ell = \partial_E p_\ell\,\dot E + \partial_\gamma p_\ell\,\dot\gamma$.
- Population tangents: $\dot\Delta_{56} = \dot P_5 p_5 + P_5 \dot p_5$, $\dot\Delta_{67} = \dot P_6 p_6 + P_6 \dot p_6$, then the update of Section 5.1 applied to the tangents. The ion weight $w$ is constant, so $\dot W' = \dot W + w(\dot\Delta_{56} + \dot\Delta_{67})$.
- Charge tangent: $\dot w_\text{tl} = w\sum_\ell \ell\,\dot P_\ell - \dot W$.
- At an emission, the new electron copies the tangents of `x, y, z, ux, uy, uz, inv_gamma` from its parent, and its weight tangent is $\dot W$ before the reset. Then $\dot W$ is reset to 0 with $W$.
- The emission threshold is a decision. The tangent does not differentiate it (Section 6.3). These derivatives are local, with the emission decisions fixed. The threshold still changes the birth time and the particle count, so smooth population equations alone do not make the beam objective smooth.
- Reference for L1: write the EV primal also as a Warp kernel with `enable_backward=True`. Warp then generates its VJP, and the dot-product test of Section 4.7 compares the hand-written tangent with it.

### 5.4 Stochastic referee

EV replaces macroparticle shot noise with its mean. Phase P8 measures how much this changes the gradient.

1. For each parameter $i$, use $h_i \in \{0.05, 0.1\}$ of the normalized range. Do not use the Section 6 steps ($10^{-2}$ to $10^{-4}$): they are noise-dominated for a discrete-birth model.
2. For seeds $j = 1..M$, run production stochastic ionization (`ions.track()`, `rng_seed`, `rng_stream`) at $\theta \pm h_i e_i$ with the same seed (CRN). Record $D_{ij} = J_s(\theta + h, j) - J_s(\theta - h, j)$.
3. Compute $g_S = \mathrm{mean}_j(D_{ij})/(2h)$ and $\mathrm{SE} = \mathrm{std}_j(D_{ij})/(2h\sqrt{M})$. Report the correlation of $J_s(\theta+h, j)$ with $J_s(\theta-h, j)$.
4. Start with $M = 8$. Double $M$, up to 64, until $\mathrm{SE} < 0.1 \max(|g_{EV,i}|, 0.01 \lVert\nabla J_{EV}\rVert_\infty)$.
5. Compute the EV secant $g_{EV}^h$ at the same $h$. Report $(g_S - g_{EV}^h)/\mathrm{SE}$ and $|g_S - g_{EV}^h|/|g_{EV}^h|$. Report $|g_{EV}^h - g_{EV}^{AD}|$ separately as EV truncation error.
6. Split the bias. Compare the linear statistics $Q$ and $S_{ab} = \sum_i w_i\sigma_i x^a u^b$ (seed means and secants) with EV values: this is the dynamics bias. Then compare $\mathbb{E}[J_s]$ with $J$ evaluated on the seed-mean statistics: this is the Jensen gap of the nonlinear loss.

### 5.5 Immediate-emission reference

The reference sets $\epsilon = 0$: every ion with $W > 0$ emits a weighted electron in every step. This gives the exact birth history of the rate-equation model of Section 5.1. It costs one new electron per ionizing ion per step. Run it on Twin-XS and on a reduced Twin-S in which the nitrogen occupies a short region, so that the births fit in memory.

1. Compare $\epsilon \in \{10^{-4}, 10^{-5}, 10^{-6}\}$ at $K_e = 1$ with the reference, at the nominal $\theta$ and two corners.
2. Birth history: the weighted distribution of the birth phase $\xi_b = z_b - c\,t_b$ in bins of $\lambda_0/16$, and the weighted mean and maximum delay between a fractional increment and its release.
3. Beam: trapped charge $Q$, normalized transverse emittance, every $m_k$, $\mathrm{MMD}^2_{\mathrm{RFF}}$, and the full loss $J$. For each quantity, compute the SE from 16 stochastic seeds.
4. Sensitivities: the forward-mode gradient of the full loss, relative to $\lVert g_{ref}\rVert_\infty$.
5. Adopt an $\epsilon$ only if every beam difference is below 2 SE, every gradient difference is below 5%, and the differences decrease as $\epsilon$ decreases. Otherwise report the results and ask the user. The options are a smaller $\epsilon$ or immediate emission.

## 6. Verification ladder

### 6.1 Levels

Run each level on GPU. Run each test file in its own process (`CLAUDE.md` rule 5).

| Level | Test | Pass criterion |
|---|---|---|
| L0 | Existing autodiff and fused tests | All pass, unchanged (122 GPU tests in `test_autodiff_*`) |
| L1 | Tangent kernels, EV kernels, ion push, field-operator linearity, loss terms, laser helper | Tangent kernels: dot product with the reference VJP, $e \le 10^{-12}$ for 3 random draws (Section 4.7), and `check_jvp` with the normalization and adjacent-epsilon pair rule of `check_vjp` (`verification.py:157-173`). Field operators: linearity within $10^{-13}$ (Section 4.6), at the Twin-S shape and at 2912 x 1080, modes $m = 0, 1$, all three DHT orders. Loss terms: `check_vjp` on the analytic gradient. |
| L2 | `step` parity with production | Twin-XS and Twin-S, pre-ionized N (no EV): `begin` plus 300 `step` calls without tangents against production `sim.step(300)` in one call from the same initial state. Fields within $\max(2\cdot10^{-11}, 10 d_{AA})$ relative in the infinity norm, where $d_{AA}$ is the difference of two production runs. Particle counts and IDs equal; particle arrays matched by ID within the same bound. |
| L2t | Primal unchanged by tangents | `step` with and without tangents, 300 steps on Twin-XS. Primal arrays within $\max(2\cdot10^{-11}, 10 d_{AA})$; counts and IDs equal. |
| L3 | One full step | Random direction on all of $s_n$. Tangent of `step` against central FD of `step` with frozen decisions. `check_jvp` normalization and pair rule at $10^{-6}$. Steps: one ordinary step, one exchange step, one step with emissions (Phase P6). Harness rules in Section 6.3. |
| L4 | Parameter gradient, horizon sweep | Error definition and pass rule in Section 6.4. Horizons in Section 6.4. |
| L4w | Linear-wakefield derivative (physics check, no FD) | Section 6.5. Analytic: within the tolerances of the upstream test (8% of max for Ez, 11% for Er). Self-consistency: within $10^{-3}$ of max. |
| L5 | EV vs stochastic | Section 5.4. Report only. The user decides acceptance. |
| L6 | Twin recovery | From the 5 starts, recover $\hat\theta^\ast$ within 0.02 in every component. Targets are noise-free (D22). |

### 6.2 Check functional for L3 and L4

Until the full loss is built (Phase P7), and whenever `elec_from_N` holds fewer than 1000 selected macroparticles, use

$$
J_{chk}(s) = \sum_{s} \sum_i w_i\, T(z_i - z_{win})\, \big(a_s\cdot\xi_i + (b_s\cdot\xi_i)^2\big) + \sum_m \lVert P \odot \hat E_m \rVert^2 .
$$

- $\xi_i$ is the standardized $(x, y, z - z_{win}, u_x, u_y, u_z)$ with fixed scales.
- $a_s$, $b_s$ (one pair per species) and the positive spectral mask $P$ are fixed random arrays with seed 0.
- $T$ is a smooth taper: 0 within 10 cells of the left window edge, 1 beyond 20 cells. Particle removal then does not change $J_{chk}$.
- $J_{chk}$ does not depend on particle order, needs no target, and is defined with and without ionization, from $N_t = 1$.
- Report the field term, the background-particle term, and the `elec_from_N` term separately. Each must pass.

### 6.3 FD rules

1. Build every FD run with the same `inversion/configs/<size>.json`: same $N_t$, $\Delta t$, window speed, `exchange_period`, $K_e$, $\epsilon$, and NumPy seed.
2. In the base run, record the `n_move` sequence and the injected count of every species at every injection event. Each $\pm h$ run asserts that both are equal. A count mismatch shifts the NumPy stream and invalidates the FD value.
3. Frozen-decision FD is the AD reference. In the base run, record the emitting parent IDs at each step. In each $\pm h$ run, pass them as `emit_mask` and skip the threshold test. Key the set by tracker ID.
4. Free FD keeps the thresholds active. Report free minus frozen as the jump contribution. It is not a small correction with a fixed size. Each emission threshold that crosses inside $[\theta - h, \theta + h]$ adds $\Delta J_k/(2h)$, where $\Delta J_k$ is the jump of $J$ at that crossing. One crossing therefore contributes $O(\epsilon w/h)$, which grows as $h$ decreases. Example: for $q(\theta) = w(\epsilon + \theta)\,\mathbb{1}_{\theta > 0}$ at $\theta = 0$, frozen AD gives 0 and free central FD gives $w/2 + \epsilon w/(2h)$. Away from crossings, free and frozen FD agree once $h$ is small enough. AD claims only the conditional derivative with the decisions fixed.
5. Threshold checks (Phase P6 and Phase P8):
   - Crossing probe: build a one-ion fixture with a prescribed field ramp, so that the emission threshold crosses at a known $\theta_c$. Confirm that free FD minus frozen FD equals $\Delta J/(2h)$ when $\theta_c$ is inside $[\theta - h, \theta + h]$, and is zero otherwise.
   - $\epsilon$ convergence: Phase P8 step 2 compares the full loss and its forward-mode gradient with the immediate-emission reference as $\epsilon$ decreases (Section 5.5).
6. L3 harness:
   - Run the base step once and store its decisions.
   - Use a fixture with even `p_nz` and `p_nr`. The cell coordinate is $z_{cell} = (z - z_{min})/\Delta z - 0.5$ (`cuda_sorting.py:72-73`). With `p_nz = 1`, particles are loaded at $z_{min} + (i + 0.5)\Delta z$, so $z_{cell}$ is an integer and every resting particle sits on a shape boundary. The same holds in r for `p_nr = 1`. With an even count, $z_{cell} = i/2 - 1/4$.
   - Give `check_jvp` an optional `direction_masks` argument. Mask out every particle whose fractional cell coordinate in z or r, at gather, J deposit, or `rho_next` deposit, is within $10^{-3}$ of an integer. The shape kernels take the cell from `ceil(position)`, so a crossing cannot be frozen.
   - A field perturbation moves unmasked particles too. Record a branch signature for each particle at each stage: the cell indices and the axis branch at gather, at J deposit, and at `rho_next` deposit, and the emission flag. Recompute the signature for every FD evaluation. If it differs from the base, reject that FD point and use a smaller $\epsilon$.
   - Test the particle inputs and the field inputs as separate blocks, each with its own random direction.
7. Measure the run-to-run noise $\sigma_J$ by running the base $\theta$ three times. Use only steps $h$ with $2h\lvert g_j\rvert > 100\,\sigma_J$.

### 6.4 L4 error, pass rule, and horizons

- Error per component: $e_j(h) = \lvert g^{AD}_j - g^{FD}_j(h)\rvert / \max(\lvert g^{AD}_j\rvert, \lvert g^{FD}_j(h)\rvert, 10^{-6}\lVert g^{FD}(h)\rVert_\infty)$, in $\hat\theta$, with $h \in \{10^{-2}, 10^{-3}, 10^{-4}\}$.
- Pass: two adjacent $h$ give $e_j \le 10^{-4}$ for every $j$. A component also passes when $\lvert g^{AD}_j\rvert$ and $\lvert g^{FD}_j\rvert$ are both below $10^{-6}\lVert g\rVert_\infty$.
- Report $e_j(h)$ for every $h$ and horizon, the growth of $\lVert\dot s_n[j]\rVert$ for each tangent, and 1D slices of $J$ over $\pm 0.05$ in each $\hat\theta_i$.
- Horizons: $N_t \in \{1, 10, 50\}$ and $\{n_c + 10, n_c + 50, n_c + 200\}$ must pass. $\{1000, \text{full}\}$ are reported. $n_c$ is the first step at which the laser peak reaches the position where $S(z) = 0.1$, at the nominal $\theta$. On Twin-XS, $n_c \le 20$.
- Agreement between batched and single tangents (Phase P7 step 2) shows that the driver is consistent. It does not show that the derivative is correct.
- Recovery gate. At the end of Phase P7, the user chooses one branch for Phase P9 and records it in Section 12:
  1. Verified: L4 passes at the full horizon on Twin-S, for $J_{chk}$ and for the loss. Phase P9 results carry the exact-gradient claim.
  2. FD fallback: the user approves central-FD gradients (4 forward runs per gradient) or a shorter horizon at which L4 passes. Phase P9 results state which one.
  3. Exploratory: Phase P9 uses AD gradients that did not pass L4 at the full horizon. Phase P9 results make no exact-gradient claim and are labeled exploratory.

### 6.5 Linear-wakefield derivative check (L4w)

Suggested by Rémi Lehe (FBPIC). Upstream `tests/test_linear_wakefield.py` compares the simulated Ez and Er at the end of a linear wakefield run with the linear-theory formulas. Those formulas (test docstring and functions `Ez`, `Er`) scale as $a_0^2$, so the analytic derivative is exact and closed-form:

$$
\frac{\partial E_z}{\partial a_0} = \frac{2 E_z^{an}}{a_0}, \qquad \frac{\partial E_r}{\partial a_0} = \frac{2 E_r^{an}}{a_0}.
$$

The check differentiates the full PIC cycle with respect to the laser amplitude and compares the result with physics, not only with FD.

1. Configuration: the upstream test with `Nm=2`. That is a linearly polarized `GaussianLaser` in mode 1 and the wake in mode 0, $a_0 = 0.01$, $w_0 = 20$ µm, $c\tau = 6$ µm, $n_e = 8\times10^{24}$ m$^{-3}$, 800 x 120 cells, 1500 steps, electrons only, laser added with `add_laser_pulse` (direct method). It matches D4 and D23, and it needs no ionization, loss, or trapping.
2. Parameter: $a_0$. Because $a_0 \propto \sqrt{E_L}$, the tangent seed is $\mathcal{T}\mathcal{D}\, t_{a_0}$ with $t_{a_0} = F_0/a_0$ (Section 4.8), and $\partial/\partial E_L = (a_0/2E_L)\,\partial/\partial a_0$.
3. Output: $E_z$ and $E_r$ on the grid at the last step, combined over modes as in `compare_fields` (mode 0 plus twice the real part of mode 1). One `jvp` run gives the whole derivative field $\partial E/\partial a_0$ on the grid, so the checks compare full fields.
4. Analytic check: compare with $2E^{an}/a_0$, within the upstream tolerances (`atol` 0.08 x max for Ez, 0.11 x max for Er). This error is dominated by the PIC-versus-theory error of the forward run.
5. Self-consistency check: compare with $2E^{sim}/a_0$, within $10^{-3}$ of max. Nonlinear corrections are of relative order $a_0^2 = 10^{-4}$, so this check is sharper than item 4. It isolates the derivative error from the discretization error.
6. Central FD in $a_0$ with the Section 6.4 rule, as a third reference.

## 7. Rollout plan

Effort estimates are agent working days, with large uncertainty because the long-horizon behavior is unknown. The phases total about 38 to 56 days.

Milestones. Stop and report to the user after each one.

1. M1, Phases P0 to P3 (about 13 to 20 days): one frozen Twin-S configuration, identifiability, the forward EV model, and an FD recovery from a displaced start. M1 shows that the problem can be solved.
2. M2, Phases P4 and P5 (about 11 to 15 days): a verified short-horizon tangent gradient on Twin-XS, including L4w. M2 shows that the gradient is exact.
3. Plan Phases P6 to P9 again with the measurements of M1 and M2.

### Phase P0. Baseline (0.5 day)

1. Activate the `fbpic-warp` environment. Confirm Warp 1.18.0 and CuPy 14.2.
2. From `lpa/FBPIC_Warp`, run:
   ```bash
   for f in tests/test_autodiff_*.py tests/test_fused_particle_operations.py tests/test_gpu_end_to_end.py; do python -m pytest "$f" -q; done
   ```
3. If a GPU other than `cuda:0` is used, cherry-pick commit `8f27245` from `snidhan/dev-nccl`. It removes a stray GPU 0 context.
4. Record the GPU model, total memory, free memory, and host memory.

Acceptance: L0 passes.

### Phase P1. Twin configurations, trapping, and identifiability (3 to 5 days)

1. Write `inversion/twin_config.py`: `build_twin(theta, size, config, ionization in {"none", "ev", "stochastic"}, plasma_start in {"injected", "preloaded"}, p_nz=1)`. With "none", load N as fixed-charge 5+ ions without an ionizer. With "stochastic", use production `make_ionizable` with `level_start=5`, `track()`, `rng_seed`, and `rng_stream`.
2. Write the twin density with the indicator $[z_{lo}, z_{hi}]$ and the closed-form weight derivatives.
3. Build Twin-XS and Twin-S. Write `inversion/configs/<size>.json` (Section 3.4).
4. Aliasing check: before the laser reaches the plasma, compute $\lvert\rho_m\rvert/\lvert\rho_0\rvert$ for $m = 1$ over the plasma. Require it below $10^{-10}$.
5. Weight check: all particle weights are positive. Weights do not depend on $\theta$ (D21).
6. Trapping: run Twin-S with stochastic ionization at the nominal $\theta$. Set $u_c$ and $\delta_u$ (Section 3.4). Run the trapping check at the 4 corners and at the 5 starts.
7. Identifiability:
   - For each parameter $i$, run Twin-S forward-only at $\hat\theta^\ast \pm 0.05\,e_i$ with stochastic ionization, the same seed for each pair, and 4 seeds (32 runs).
   - Compute $v = (\log Q, m_k)$ and the sensitivity $S = \partial v/\partial\hat\theta$ by central differences. Record the seed mean and its SE.
   - Run the T2 truth at $\theta^\ast$. Set $\Sigma = \mathrm{diag}(\text{seed variance of } v + (v_{truth} - v_{model})^2)$, with each entry at least $(0.02\, s_k)^2$.
   - Form $G = S^T \Sigma^{-1} S$. Record its eigenvalues, $\mathrm{cond}(G)$, the weakest eigenvector, and $\sqrt{\mathrm{diag}\, G^{-1}}$.
   - Stop and ask the user if $\mathrm{cond}(G) > 10^4$ or any $\sqrt{(G^{-1})_{ii}} > 0.02$. Options: a longer plasma, a different observable, or one parameter.
8. Record Twin-S charge, energy, run time, and peak memory.

Acceptance: steps 4 to 7 pass, or the user has decided how to proceed. The config files are frozen.

### Phase P2. Functional stepper without ionization (4 to 6 days)

1. Implement `StepState`, `TangentState`, and `StepDecisions` (Sections 4.2 and 4.5).
2. Implement `begin` and `step` without tangents for `ionization="none"` and $N_m = 2$, with production kernels and the stable sort (Sections 4.3 and 4.5).
3. Run L2 on Twin-XS and on Twin-S.

Acceptance: L2 passes. Production code is unchanged.

### Phase P3. Forward EV model, minimal loss, and FD recovery gate (5 to 8 days)

This phase shows that the inversion problem can be solved before any tangent code exists. It uses forward runs only.

1. Implement the EV primal in `fbpic/autodiff/ionization.py`: populations, ADK, emission, `emit_mask`, and the N-ion push with the charge $w_\text{tl}$ (Section 5). Add EV to `step` and its arrays to `StepState`.
2. Open-loop parity: freeze the gathered fields and ion momenta of a set of ions over 200 steps. Give each ion a prescribed field so that $p_5$ and $p_6$ each cover about $10^{-3}$ to 0.3 over the set. Run the EV kernel and 32 Philox seeds of the production kernel. Pass if $\sum_i w_i P_\ell$ for $\ell = 5,6,7$ and the cumulative birth weight each satisfy $\lvert EV - \text{mean}\rvert \le 3\,\mathrm{SE}$.
3. Charge test: deposit rho from the N ions and `elec_from_N` before and after the EV call. The difference must be below $10^{-12}$ relative, on emission steps and on steps without emission.
4. Invariant tests on Twin-S with EV. Production removes each species at the window back independently, so a trapped electron can outlive its parent ion. Test the closed cohort first, then the open window:
   1. Closed cohort: take the N ions of the first injection event and the electrons whose recorded parent is one of them. Before any of these particles reaches the left window edge, check $P_5 + P_6 + P_7 = 1$ to $10^{-14}$ and $\sum_{e} w_e + \sum_i W_i = \sum_i w_i(\bar Z_i - 5)$ over the cohort, to $10^{-12}$ relative.
   2. Open window, after 3 exchange periods: keep test-only cumulative ledgers from the recorded removal ranges, per species: the removed electron weight, and the removed ions' $\sum w(\bar Z - 5)$ and $\sum W$. Let $B$ be the cumulative emitted weight. Check $B = \sum_{\text{live}} w_e + \sum_{\text{removed}} w_e$ and $B + \sum_{\text{live}} W + \sum_{\text{removed}} W = \sum_{\text{live}} w(\bar Z - 5) + \sum_{\text{removed}} w(\bar Z - 5)$, each to $10^{-12}$ relative.
   3. Each ion's populations follow its tracker ID through sorts.
5. Run Twin-S forward with EV at the nominal $\theta$ and at two opposite corners. Record the `elec_from_N` count and peak memory. Update Section 8.
6. Immediate-emission feasibility: run the reference of Section 5.5 forward on Twin-XS and on the reduced Twin-S. Record the birth count and peak memory.
7. Minimal loss: implement the charge and moment terms of Section 2.4 with $\alpha_D = 0$, forward only, and `inversion/make_target.py`. Confirm a finite loss on an empty and a nearly empty selection.
8. Mean-charge closure: run Twin-XS with EV twice, with the N ions mobile and immobile. Compare $Q$, every $m_k$, and the minimal loss. Report the differences in units of the SE from 16 stochastic seeds. If any difference exceeds 2 SE, ask the user.
9. Generate the T0 target at $\theta^\ast$ on Twin-S (Section 2.2).
10. Slices: evaluate the minimal loss on 13 points over $\hat\theta^\ast_i \pm 0.3$ for each parameter (26 runs). Record the basin shape and any plateaus or jumps.
11. FD recovery gate:
    - Start from the optimizer start of Section 3.4, item 4, that is farthest from $\hat\theta^\ast$.
    - Run `scipy.optimize.minimize(method="L-BFGS-B")` on $\hat\theta$, bounds $[0, 1]$, with central-FD gradients at $h = 0.02$ in $\hat\theta$ and the thresholds active (4 forward runs per gradient).
    - If L-BFGS-B stops on a line-search failure, restart once from its last point with Powell (derivative-free).
    - Record iterations, forward runs, wall time, the path of $\hat\theta$, $J$, and the FD gradients along the path.

Acceptance (gate): the recovery from the displaced start reaches $\hat\theta^\ast$ within 0.02 in every component, with the T0 target and the minimal loss. If the gate fails, stop and ask the user before Phase P4. Options: a different observable, narrower bounds, a longer plasma, or one parameter.

### Phase P4. Tangent kernels (6 to 8 days)

1. Add `check_jvp` to `fbpic/autodiff/verification.py`. It mirrors `check_vjp`: central FD of the primal kernel along a random direction, with the same normalization and adjacent-epsilon pair rule.
2. Implement `gather_tangent`, `vay_tangent`, `position_tangent`, and `deposit_tangent` in `fbpic/autodiff/tangent.py`, generic over $N_m$ (Section 4.7).
3. Run L1 for each kernel: the dot-product test against the existing VJP clones at $N_m = 5$, and `check_jvp` at $N_m = 2$ and $N_m = 5$.
4. Run the linearity check for every field operator that `step` uses (Section 4.6), at the Twin-S shape, modes $m = 0, 1$, and all three DHT orders.

Acceptance: L1 passes for every kernel and operator.

### Phase P5. Tangent step and driver, without ionization (5 to 7 days)

1. Add tangents to `step` for `ionization="none"`, with the stash and the permutation of tangent arrays (Sections 4.3 to 4.5).
2. Implement the structural events (Section 4.9).
3. Implement the laser tangents and run their three probes (Section 4.8).
4. Implement `value_and_grad` and `jvp` (Section 4.10). The objective is a parameter.
5. Write the FD harness `inversion/verify_param_gradient.py` (Section 6.3).
6. Run L2t on Twin-XS.
7. Run L3 on Twin-XS: one ordinary step and one exchange step.
8. Run L4w, the linear-wakefield derivative check (Section 6.5).
9. Run L4 on Twin-XS with $J_{chk}$ at the pass horizons.

Acceptance: probes, L2t, L3, L4w, and L4 (pass horizons) pass on Twin-XS without ionization.

### Phase P6. EV tangents (3 to 5 days)

1. Implement the tangents of Section 5.3, the N-ion push tangent, and the Warp-generated reference VJPs. Run L1 on each kernel, with `ux, uy, uz` and the gathered fields as inputs.
2. Add the EV tangents to `step`. Run L3 on an emission step and L4 on Twin-XS at the pass horizons.
3. Run the crossing probe of Section 6.3, rule 5.

Acceptance: steps 1 to 3 pass.

### Phase P7. Full loss and the full horizon sweep (4 to 6 days)

1. Add the MMD term and the analytic state gradient of every loss term (Section 2.4). Extend the targets with `rff_mean`. Run L1 on each loss term, including the empty and nearly empty selections.
2. Run `value_and_grad` on Twin-XS at $N_t = 200$ once with $K = 2$ tangents and twice with one tangent each. The gradients must agree within $10^{-12}$ relative.
3. Measure peak GPU memory, host memory, and time per gradient on Twin-S. Record the cost of one gradient in forward-run equivalents next to the 4 runs of central FD. Update Section 8.
4. Run the full L4 sweep on Twin-S with $J_{chk}$, and with the loss where the selected beam has at least 1000 macroparticles at every FD point.

Acceptance: step 2 passes. L4 results for all horizons are reported to the user, and the user has chosen a recovery branch (Section 6.4).

### Phase P8. Model checks (3 to 4 days)

1. Run L5 (Section 5.4) on Twin-S at the nominal $\theta$.
2. Emission-delay check: run the comparison of Section 5.5 against the immediate-emission reference.

Acceptance: L5 and the emission-delay check are reported to the user.

### Phase P9. Twin recovery (3 to 5 days)

0. Confirm the recovery branch chosen at the end of Phase P7 (Section 6.4). Label every result with it.
1. Write `inversion/recover.py` with `scipy.optimize.minimize(method="L-BFGS-B")` on $\hat\theta$, bounds $[0, 1]$, and `value_and_grad(...).grad_hat`.
2. Generate the T1 and T2 targets (Section 2.2). T0 exists from Phase P3.
3. Run L6 noise-free on Twin-S for T0 and T1 from the 5 starts. Record iterations, gradient evaluations, and wall time. Compare them with the FD recovery of Phase P3. Report T2.

Acceptance: L6 passes on Twin-S for T0 and T1. Results are recorded.

### Phase P10. Documentation (1 day)

1. Add `docs/warp-porting-guide/09-differentiable-step-and-inversion.md` with contracts, measured memory, and gradient accuracy.
2. Update the next steps in doc 08 and Section 12 of this plan.

### Phase P11. Reverse-mode engine (optional, deferred)

Implement reverse mode only when the parameter count grows beyond about 5 to 10 (Section 11), or if the user asks. Start from the design at commit `07dcc6c` of this file: checkpointing with a one-step tape, the layout check, and the FFT and DHT adjoints. The tangent kernels of this plan then serve as an exact reference for the new VJPs.

## 8. Memory and performance budget

Estimates. Phases P1, P3, and P7 replace them with measurements.

| Quantity | Formula or source | Twin-S ($N_m = 2$, `p_nt = 8`) |
|---|---|---|
| Field array | $16 N_z^a N_r$ B | about 0.9 MB |
| Primal state | 7 field arrays x $N_m$, plus about 90 B per particle (plus 32 B per N ion) | about 0.42 GB |
| EV births | measured in Phase P3 | up to about 1.7 x background (estimate) |
| Tangent state, $K = 2$ | 2 x (7 field arrays x $N_m$, plus 64 B per particle, plus 32 B per N ion) | about 0.6 GB |
| Per-step stash | inputs of in-place primal operations | below one primal state, about 0.4 GB |
| Device peak | primal + tangents + stash + production workspace | about 1.5 to 3 GB, plus the forward workspace. Phase P7 measures it. |
| Forward time per step | doc 07: 0.31 s for 1852 x 300 x 5 with about 57 M particles | well below 0.1 s |
| Gradient cost | 1 primal + 2 tangents in one pass | about 3 to 4 forward runs (estimate) |
| Central FD, for comparison | 2 runs per parameter | 4 forward runs |

At 2 parameters, forward mode costs about as much as central FD, subject to the Phase P7 measurement. Forward mode removes the FD step-size truncation error. It does not remove the emission-threshold discontinuities: its derivatives are conditional on the discrete decisions (Section 4.5). L4 checks it against frozen-decision FD.

The host has two NVIDIA RTX 6000 Ada GPUs with 49140 MiB each (`nvidia-smi`, 2026-10-30). The plan uses one. Memory is not a constraint for Twin-S in forward mode.
The CuPy pool and the Warp allocator compete for device memory. Measure the peak of both.

## 9. Risks and mitigations

| Risk | Effect | Mitigation |
|---|---|---|
| Weak identifiability ($z_f$ against $E_L$, the two inverted parameters) | Recovery is ill-posed | Phase P1 step 7 and the FD recovery gate of Phase P3, before any tangent work. User decides. |
| Gradient growth over long horizons (trapping sensitivity) | AD disagrees with FD at long horizons | L4 sweep, $\lVert\dot s_n\rVert$ report. User decides. |
| Forward map of AD differs from the map that made the states | Consistent but wrong gradient | One shared stepper (D17). L2 against production. |
| Hand-written tangent kernels are wrong | Wrong gradient without error | Dot product with existing or generated VJPs, `check_jvp`, L3 |
| Tangent computation changes the primal | Gradient of a different trajectory | Per-step stash, L2t |
| Tangents not permuted with the primal | Tangents attached to the wrong particles | One permutation for primal and tangents (Section 4.5), L3 on an exchange step |
| EV model bias | Gradient of a different model than the stochastic code | L5, T2 truth. User decides. |
| Delayed emission changes the birth history | Wrong trapped charge, emittance, or sensitivities | Immediate-emission reference (Section 5.5) |
| Mean-charge ion closure | Wrong ion trajectories | Immobile-ion comparison (Phase P3) |
| Charge non-conservation from pending births | Spurious fields at the injection point | Neutral pending deposit (Section 5.2), charge test in Phase P3 |
| Azimuthal aliasing at low `p_nt` | Spurious mode-1 fields | `p_nt = 8 = 4 N_m`, aliasing check in Phase P1 |
| Inverse crime | Over-optimistic recovery | Truth variants T0, T1, T2 |
| Emission-threshold jumps | Free FD differs from AD by $O(\epsilon w/h)$ per crossing | Frozen-decision FD as the reference, crossing probe, $\epsilon$ convergence (Section 6.3) |
| Long-horizon gradient fails L4 | Recovery uses an unverified gradient | Recovery gate with three branches (Section 6.4) |

## 10. Decisions

| ID | Decision | Status |
|---|---|---|
| D1 | One GPU only. No MPI or NCCL in the differentiable path. | User, 2026-10-26 |
| D2 | `use_mpi=False`, `n_order=-1` for the twin. | User intent (single GPU), planner choice of stencil |
| D3 | Lab frame first. Boosted frame later. | Planner default |
| D4 | Direct-method `GaussianLaser`. LASY antenna later. | Planner default |
| D5 | Standalone twin configuration, not the production template. | Planner default |
| D6 | Fix the whole density profile: $n_0$, $F_L$, $z_c$, $\beta$, $F_R$, and the indicator. | Planner default, narrowed by D21 |
| D7 | Pre-ionized He$^{2+}$ and N$^{5+}$. EV for the N K shell only. | Planner default |
| D8 | Superseded by D25. Checkpointing with a one-step tape is the future reverse-mode design. | Superseded 2026-10-30 |
| D9 | Eager differentiable path. No CUDA graphs. | Planner default |
| D10 | Frozen-decision central FD is the reference for every gradient. | Planner default |
| D11 | No new Python dependency. Loss in CuPy or Warp. | Global work rules |
| D12 | EV emission per parent ion each step: $K_e = 1$, $\epsilon = 10^{-4}$, neutral pending deposit. | Planner default, validated against immediate emission in Phase P8 |
| D13 | Hand-written tangent kernels. Existing and Warp-generated VJPs serve as references in dot-product tests. | Planner default after D25 |
| D14 | L-BFGS-B on the 2 normalized parameters, 5 Latin-hypercube starts. | Planner default |
| D15 | He as fixed-charge 2+ ions without an ionizer. | Planner default |
| D16 | `p_nt = 8` for every species. | FBPIC rule `p_nt >= 4 Nm` with $N_m = 2$ |
| D17 | Production stepping is unchanged. The differentiable path uses its own functional stepper for gradients, FD, and EV truth runs. | Planner default after review |
| D18 | Truth variants T0, T1, T2 with the same grid and $N_t$. | Planner default |
| D19 | Quintic smoothstep selection and log charge with floor $Q_f$. | Planner default |
| D20 | Identifiability gate in Phase P1 before gradient work. | Planner default |
| D21 | Invert $E_L$ and $z_f$ only. Fix $n_0$ and $F_L$. | User, 2026-10-30 |
| D22 | Noise-free targets. No artificial noise and no prior. | User, 2026-10-30 |
| D23 | Two azimuthal modes ($m = 0, 1$). | User, 2026-10-30 |
| D24 | Defer Twin-M. | User, 2026-10-30 |
| D25 | Forward mode is the main engine, with one tangent per parameter. Reverse mode is future work. | User, 2026-10-30 |

## 11. Future work

1. Boosted frame with the back-transformed observable computed from the final state (smooth selection).
2. LASY antenna path: a GPU envelope interpolator, then tangents of the unsorted antenna deposition.
3. Multi-GPU tangents: tangent arrays follow the same NCCL exchanges as the primal. A reverse-mode version needs the adjoint of each exchange.
4. Reverse-time reconstruction of particle trajectories within short segments, with logs of irreversible operations.
5. Experimental realism: drop z from $H$, GPSR reconstructions as targets with their uncertainty, and multi-shot jitter.
6. Hybrid ML: gradient-enhanced surrogate training, and simulation-based inference as a warm start.
7. More parameters, with reverse mode (Phase P11) once forward cost grows too large: $n_0$ and $F_L$ first (Section 2.1, injection-weight terms of Section 4.9), then pulse duration, spot size, Zernike terms, and a free-form density lineout.
8. Noisy recovery: matched likelihood $r^T\Sigma_n^{-1}r$, a Gaussian prior with its direct gradient returned by the objective, and a sandwich covariance $C = H^{-1} S^T \Sigma_n^{-1} S H^{-1}$ with $H = S^T\Sigma_n^{-1}S + \Lambda$ for the predicted spread.
9. Twin-M, a larger twin closer to production physics.
10. Five azimuthal modes, to keep the $m = 2$ content of a linearly polarized laser.
11. A weighted sliced-Wasserstein (SWD) loss term.

## 12. Execution results

Record results here, one subsection per phase. Include commands, numbers, tolerances, the GPU, and the commit hash.

### 12.1 Plan review record

An adversarial review (4 lenses, then one skeptic per serious finding, 38 agents) checked the first draft against the code on 2026-10-26. It returned 34 serious findings: 29 confirmed, 5 partly confirmed, none refuted. This revision applies them. The main changes:

1. One functional stepper serves the sweep, the replay, FD, and EV truth runs. Production code stays unchanged (D17).
2. Discrete decisions are recorded and replayed, and a layout check guards the seed (Section 4.5).
3. The laser term uses the complete `add_laser_direct` map with all six field components (Section 4.8).
4. EV emission is per ion and per step, with a neutral pending deposit (Section 5.2).
5. `p_nt = 20` prevents mode-4 aliasing (D16).
6. A check functional and the Twin-XS size make L3 and L4 meaningful before the loss exists (Sections 6.2 and 3.3).
7. An identifiability gate runs in Phase P1 (D20).
8. Acceptance criteria have numeric thresholds and normalizations (Section 6).
9. Truth variants T0, T1, and T2 address the inverse crime (Section 2.2).

A second review on 2026-10-30 returned 4 correctness findings and 5 validation findings. All were checked against the plan and the code and confirmed. This revision applies them:

1. The jump contribution of free FD is $O(\epsilon w/h)$ per crossing, not $O(\epsilon)$. A crossing probe and an $\epsilon$-convergence check of the full loss and gradient were added (Section 6.3, Phase P7).
2. The objective returns a direct parameter gradient, so the prior reaches the gradient. Phase P6 checks it with $\lambda = 1$ (Section 4.10).
3. The layout check aligns IDs first, then checks every continuous state component, and rejects. It has a rejection test (Section 4.5).
4. The MMD mean has the $Q_f$ floor. The empty and nearly empty beams have a defined loss and are tested (Section 2.4).
5. A recovery gate with three branches follows Phase P6 (Section 6.4).
6. The ionization invariants use a closed cohort, then removal ledgers (Phase P5).
7. Noisy recovery uses the matched likelihood and prior, and a sandwich covariance for the predicted spread (Section 3.4, L6).
8. The L3 harness uses even `p_nz` and `p_nr`, branch signatures, and separate particle and field blocks (Section 6.3).
9. $K_c$ comes from measured peaks, with ordered Twin-S fallbacks (Sections 4.10 and 8).

Scope change by the user on 2026-10-30 (D21 to D24): invert $E_L$ and $z_f$ only, use noise-free targets, use 2 azimuthal modes, and defer Twin-M. This removes the prior and the noisy-recovery items 2 and 7 above from the current scope; Section 11 keeps their design.


Switch to forward mode by the user on 2026-10-30 (D25). With 2 parameters, forward mode needs no tape, checkpoint, replay, layout check, or FFT and DHT adjoints, and it fits in a few GB. Review items 3 and 9 above apply only to the deferred reverse mode. The emission-threshold, empty-beam, recovery-gate, invariant, and L3 items still apply.

A Section 5 review on 2026-10-30 (Sol) supported the population model and challenged the delayed emission. This revision applies it: the model is called a deterministic rate-equation PIC model; the threshold delay is stated; an immediate-emission reference replaces the $\epsilon = 10^{-6}$ reference (Section 5.5); an immobile-ion check tests the mean-charge closure; and $1 - p_\ell$ is evaluated as $\exp(-x_\ell)$.

A scope review on 2026-10-30 (Sol) recommended an inversion baseline before the derivative work. This revision adds the FD recovery gate (Phase P3) with the forward EV model and a minimal loss, moves the tangent phases after it (P4 to P6), adds milestones M1 and M2 and a narrow success claim, corrects the Section 8 wording on threshold discontinuities, and moves the SWD term to future work.

Phase numbers in the review records above refer to the plan at the time of each review.

_No implementation phase has run yet._

## Appendix A. Unverified claims and required probes

| Claim | Probe | Phase |
|---|---|---|
| The $k=0$ columns of the $p=m$ and $p=m+1$ components stay zero for $m \ge 1$ | Compare spect with `interp2spect(spect2interp(spect))` after 500 steps | P2 |
| Every field operator used by `step` is linear in the field arrays | Linearity check (Section 4.6) | P4 |
| Tangent kernels generic over $N_m$ agree with the 5-mode VJP clones | Dot-product test (Section 4.7) | P4 |
| `elec_from_N` traps a beam in Twin-S with 2 modes, `p_nt = 8`, at all checked points | Phase P1 steps 5 and 6 | P1 |
| $E_L$ and $z_f$ are locally identifiable in Twin-S | Phase P1 step 7 | P1 |
| EV births match the stochastic marginals in open loop | Phase P3 step 2 | P3 |
| The device peak of Twin-S stays below about 3 GB plus the forward workspace | Measure in `value_and_grad` | P7 |

## Appendix B. File map

| Path | Change |
|---|---|
| `fbpic/autodiff/state.py` | New: `StepState`, `TangentState`, `StepDecisions` |
| `fbpic/autodiff/stepper.py` | New: `begin`, `step` with optional tangents |
| `fbpic/autodiff/tangent.py` | New: `gather_tangent`, `vay_tangent`, `position_tangent`, `deposit_tangent` |
| `fbpic/autodiff/verification.py` | Add `check_jvp` and the dot-product test |
| `fbpic/autodiff/ionization.py` | New: EV populations, ADK, emission, tangents, reference VJP kernel |
| `fbpic/autodiff/particles.py` | Add `push_ion_momenta_ev` |
| `fbpic/autodiff/laser.py` | New: `laser_initial_fields`, `GaussianLaserDzf` |
| `fbpic/autodiff/driver.py` | New: `value_and_grad`, `jvp` |
| `fbpic/autodiff/loss.py` | New: observable, loss, check functional, analytic gradients |
| `inversion/twin_config.py` | New: `build_twin`, density closed forms |
| `inversion/configs/*.json` | New: frozen twin choices (Section 3.4) |
| `inversion/make_target.py` | New: target generation (T0, T1, T2) |
| `inversion/identifiability.py` | New: Phase P1 step 7 |
| `inversion/verify_param_gradient.py` | New: AD vs FD harness, horizon sweep |
| `inversion/recover.py` | New: L-BFGS-B recovery driver |
| `tests/test_autodiff_tangent.py`, `test_autodiff_ionization.py`, `test_autodiff_stepper.py`, `test_autodiff_laser.py`, `test_autodiff_loss.py` | New tests (L1 to L3) |
| `docs/warp-porting-guide/09-differentiable-step-and-inversion.md` | New chapter |

`inversion/` is a new directory under `lpa/FBPIC_Warp`. It holds research drivers and `verify_*.py` scripts, not library code. No production module changes.

## Appendix C. Research basis

| Topic | Source | Use in this plan |
|---|---|---|
| Phase-space reconstruction with differentiable transport | Roussel et al., PRL 130, 145001 (2023); PRAB 27, 094601 (2024) | Problem framing; later GPSR targets |
| Reverse-mode memory in accelerator PIC | Dhamrait et al., PASC '26, arXiv:2605.27282 | A full tape does not fit; reason for forward mode at small $K$ |
| Checkpointing | Griewank and Walther, ACM TOMS 26 (2000); Aupy et al., SISC 38 (2016) | Future reverse mode (Phase P11) |
| Reverse-time reconstruction | Li et al., ApJS 270, 36 (2024); Greener, PNAS 122 (2025) | Future work; segment length limits |
| Discrete events in adjoint Monte Carlo | Yang, Silantyev, Caflisch, JCP 488 (2023); Caflisch and Yang (2024) | Score terms; EV rationale |
| Deterministic ionization in PIC | DiIorio et al., arXiv:2603.17885 (2026) | EV populations |
| AD with discrete randomness | Arya et al., NeurIPS 2022 | Stochastic referee, future estimator |
| Gradient quality over long horizons | Metz et al., arXiv:2111.05803 | L4 horizon sweep |
| Approximate adjoints | Dwight and Brezillon, AIAA J. 44 (2006) | Verify the exact adjoint before shortcuts |
| Forward-mode PIC | Qiang, PRAB 26, 024601 (2023) | Precedent for the main engine |
| Continuous Vlasov adjoints | Bartsch et al., SICON 62 (2024); Albi et al., arXiv:2610.26110 | Future macroscopic adjoint research |
