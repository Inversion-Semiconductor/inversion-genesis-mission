# Differentiable Inversion Plan for FBPIC-Warp

| Item | Value |
|---|---|
| Status | Draft, revised after an adversarial review (Section 12.1). Nothing is implemented. |
| Date | 2026-10-26 |
| Branch | `snidhan/dev-autodiff` (base commit `92352b8`) |
| Scope | One GPU. Lab frame. Four parameters. Twin experiment with synthetic data. |
| Owner | The implementing agent writes all code: kernels, adjoints, algorithms, scaffolding, and tests. |
| Production code | Unchanged. The differentiable path is a separate stepper in `fbpic/autodiff` (D17). |
| Decisions | D1 to D20 in Section 10. D1 and D2 are user decisions. The others are planner defaults that the user can change. |

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
| $\theta = (E_L, z_f, n_0, F_L)$ | Laser energy, laser focal position, peak neutral density, up-ramp FWHM | J, m, cm$^{-3}$, m | `laser_energy_J`, `laser_focal_position_m`, `peak_neutral_density_cm3`, `density_fwhm_left_m` |
| $\Theta$ | Box of admissible parameters | - | `config["bounds"]` |
| $\hat\theta$ | Normalized parameters, $\hat\theta_i = (\theta_i - \theta_i^{lo})/(\theta_i^{hi} - \theta_i^{lo})$ | - | `theta_hat` |
| $\theta^\ast$ | Hidden truth of the twin experiment | - | `theta_star` |
| $s_n$ | Complete state after $n$ steps (Section 4.2) | - | `StepState` |
| $\mathcal{D}_n$ | Discrete decisions of step $n$: sort permutations, removal ranges, injection counts, emission parents (Section 4.5) | - | `StepDecisions` |
| $\Phi$ | One step, $s_{n+1}, \mathcal{D}_n = \Phi(s_n)$ | - | `forward_step` (no tape), `taped_step` (tape) |
| $N_t$ | Steps from $t = 0$ to the exit snapshot | - | `num_steps` |
| $H$ | Observable map from $s_{N_t}$ to the beam at exit | - | `observable` |
| $y$ | Synthetic measured data (Section 2.2) | - | `target` |
| $J$ | Objective: the loss of Section 2.4, or a check functional of Section 6.2 | - | `objective` |
| $\lambda_n$ | Adjoint state, $\partial J / \partial s_n$ | - | `adj` |
| $N_m$ | Azimuthal modes (5) | - | `Nm` |
| $N_z^a$, $N_r$ | Allocated z rows and radial cells | cells | `fld.Nz`, `fld.Nr` |
| $N_p$ | Macroparticles of all species | - | `Ntot` |
| $K_c$ | Steps per checkpoint segment | steps | `segment_length` |
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

Compute exact gradients of a beam-level objective with respect to four laser and gas parameters through FBPIC-Warp.
Use these gradients to recover hidden parameters from synthetic data in a twin experiment.
This is the first goal of the project statement "Laser Wakefield Accelerator Condition Reconstruction": recover 3 to 5 known synthetic parameters.

$$
\hat\theta = \arg\min_{\theta \in \Theta} \; \mathcal{L}\big(H(F(\theta)),\, y\big) + \lambda\, R(\theta).
$$

$F$ is the PIC forward model, $H$ maps the simulated bunch to the observable, and $R$ penalizes departure from independently measured values.

### 1.2 In scope

1. Single-GPU, lab-frame, 5-mode twin configurations at three sizes (Section 3).
2. A functional differentiable stepper with recorded discrete decisions (Section 4).
3. Hand-written adjoints for cuFFT and the cuBLAS DHT (Section 4.6).
4. Expected-value ionization of the nitrogen K shell, with adjoints (Section 5).
5. A checkpointed reverse driver that returns $\partial J / \partial \theta$ (Section 4.10).
6. A differentiable observable and loss (Section 2.4).
7. A local identifiability study before the adjoint work (Phase P1).
8. A verification ladder from single operators to full-horizon gradients (Section 6).
9. Recovery of $\theta^\ast$ with a bounded quasi-Newton optimizer (Phase P8).

### 1.3 Out of scope

- Multiple GPUs, MPI, and NCCL adjoints (D1).
- The boosted frame and back-transformed diagnostics (D3).
- The LASY antenna laser (D4).
- Reverse CUDA graphs. The differentiable path runs eagerly (D9).
- Reverse-time trajectory reconstruction (D8).
- Changes to production stepping, ionization, sorting, or buffer handling (D17).
- Experimental data, GPSR targets, and multi-shot jitter (Section 11).

### 1.4 Research basis in brief

Appendix C lists the sources.

1. The memory problem is the forward trajectory, not the adjoint state. A full tape does not fit. Checkpointing with a one-step tape is the exact, proven method.
2. All field operators are linear with fixed coefficients, and no parameter enters them.
3. Pathwise AD through `random < p` returns zero for the change in electron count. No published PIC code differentiates field ionization. Expected-value ionization makes the charge a smooth function of $\theta$ (Section 5).
4. Gradients through long, nonlinear runs can grow without bound. Every AD gradient must agree with FD before an optimizer uses it.

### 1.5 Evidence status

| Item | Status |
|---|---|
| Per-operator VJPs for push, gather, deposition, and pointwise field operators | Verified: 122 tests on GPU, 118 on CPU with 4 skipped, commit `24fb4c7` |
| FFT and DHT adjoints | Not implemented |
| One-step, horizon, and parameter gradients | Not measured |
| Memory and run time of a reverse pass | Not measured |
| Expected-value ionization | Not implemented |
| Identifiability of the four parameters in the twin | Not measured |

## 2. Problem formulation

### 2.1 Parameters and their entry points

| Parameter | Production bounds | Enters the state through | Derivative |
|---|---|---|---|
| $E_L$ | 1.5 to 3.5 J | All six initial interp fields (Er, Et, Ez, Br, Bt, Bz), set by `add_laser_direct` | Fields scale as $\sqrt{E_L}$: tangent $= F_0 / (2E_L)$ (Section 4.8) |
| $z_f$ | 2.5 to 4.5 mm | All six initial interp fields | Analytic profile derivative pushed through the full `add_laser_direct` map; check with host FD of that map |
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

$z_c$, $\beta$, $F_R$, $z_{lo}$, and $z_{hi}$ are fixed (D6). Choose $z_{lo}$ so that $S(z_{lo}; F_L^{lo}) \ge 10^{-12}$. Then no weight underflows, and the set of particles does not depend on $\theta$.

Gradients are reported in both units: $\partial J/\partial\hat\theta_i = (\theta_i^{hi} - \theta_i^{lo})\, \partial J/\partial\theta_i$. The FD harness and the optimizer use $\hat\theta$.

### 2.2 Twin experiment and truth variants

1. Choose $\theta^\ast$ inside $\Theta$ (Section 3.4).
2. Generate the target $y$ from a truth run (variants below).
3. Start the optimizer from a different $\hat\theta_0$ and recover $\hat\theta^\ast$.

Gradient checks (Phases P4 to P6) use one configuration for AD and FD, because they test the derivative of one discrete map.
Recovery runs (Phase P8) use three truth variants to avoid an inverse crime. Every variant keeps the inversion grid, $\Delta t$, $N_t$, `exchange_period`, and the `n_move` sequence, so that $H$ is evaluated at the same time. Every variant uses a different NumPy injection seed and `p_nz = 2` instead of 1.

| Variant | Ionization model | Role |
|---|---|---|
| T0 | EV with the inversion settings ($K_e$, $\epsilon$) | Control. L6 applies. |
| T1 | EV with $K_e = 1$, $\epsilon = 10^{-6}$ | Main test. L6 applies. |
| T2 | Production stochastic ionization (Philox, tracked ions) | Model error. Report only. |

If T0 passes and T1 fails, stop. Report the recovered-parameter bias together with the merge bias of Phase P7, and ask the user.

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
J(\theta) = \alpha_Q \big(\log(Q + Q_f) - \log(Q_y + Q_f)\big)^2 + \alpha_M \sum_k \frac{(m_k - m_{k,y})^2}{s_k^2} + \alpha_D\, \mathrm{MMD}^2_{\mathrm{RFF}} + \lambda R(\hat\theta).
$$

- $Q = e \sum_i w_i \sigma_i$ is the selected charge. $Q_f = 10^{-3} Q_y$.
- $m_k = \sum_i w_i\sigma_i f_k(\xi_i) / (\sum_i w_i \sigma_i + Q_f/e)$ are the weighted first and second moments of $\xi = (x, u_x, y, u_y, z - z_{win}, u_z)$.
- $s_k$ are fixed scales from the target: the target standard deviation for first moments and its square for second moments.
- $\mathrm{MMD}^2_{\mathrm{RFF}} = \lVert \mu - \mu_y \rVert^2$, with $\mu = \sum_i w_i\sigma_i \phi(\tilde\xi_i) / \sum_i w_i\sigma_i$. $\phi$ has $D = 512$ random Fourier features, seed 1234, bandwidth 1.0. $\tilde\xi$ is $\xi$ standardized with the frozen target mean and standard deviation, never with statistics of the current sample.
- $\mu$ is linear in the normalized weights, so $\mathrm{MMD}^2$ is quadratic in them. The weight gradient is
  $$
  \frac{\partial\, \mathrm{MMD}^2}{\partial (w_i\sigma_i)} = \frac{2\,\langle \mu - \mu_y,\; \phi(\tilde\xi_i) - \mu\rangle}{\sum_k w_k \sigma_k}.
  $$
- $R(\hat\theta) = \sum_i (\hat\theta_i - \hat\theta^{pr}_i)^2/\hat\sigma_i^2$. $\hat\theta^{pr}$ is the prior center, not the optimizer start.
- Defaults: $\alpha_Q = 1$, $\alpha_M = 1/K$ for $K$ moments, $\alpha_D = 1$, $\lambda = 0$ except in noisy L6 runs.

Implement the loss and its analytic gradient in CuPy or Warp (D11). Test every term with FD on random particle sets (L1).

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
| Modes | $N_m = 5$ | The autodiff gather and deposition clones need exactly 5 modes. |
| Stencil | `n_order=-1` | Single-rank PSATD. `n_guard=64`, `exchange_period=14`. (D2) |
| Shape | Linear | Only linear shapes have autodiff clones. |
| Field solver | Standard PSATD, curl-free correction, `filter_currents=True`, `use_true_rho=False` | Lab frame. Autodiff twins exist for all of these. |
| Boundaries | z open, r reflective | No PML adjoint is needed. |
| Laser | `GaussianLaser`, `method='direct'` | Parameters enter only the initial fields. (D4) |
| Helium | He$^{2+}$ ions with fixed charge $2e$ and no ionizer; electrons loaded | No background births. (D7, D15) |
| Nitrogen | N$^{5+}$ ions; L-shell electrons loaded | Only K-shell transitions remain. (D7) |
| K-shell electrons | Separate species `elec_from_N` | The observable and EV births act on this species only. |
| Ionization model | EV for N$^{5+}\to$N$^{6+}\to$N$^{7+}$ | Section 5. (D7) |
| Particles per cell | `p_nz=1`, `p_nr=2`, `p_nt=20` for every species | FBPIC requires `p_nt >= 4 Nm` (`fbpic/main.py:950`). With `p_nt=4`, each ring deposits a mode-4 charge as large as its mode-0 charge. Hard floor: `p_nt > 2(Nm-1) = 8`. (D16) |
| Density | `BilateralSuperGaussian` times the fixed indicator $[z_{lo}, z_{hi}]$ | No underflow, $\theta$-independent particle set. (D6) |
| Plasma start | Set through `dens_func` only | `p_zmin` limits only the initial load. The injector uses `dens_func`. |
| Moving window | $v = c$ | `n_move` is independent of $\theta$. |
| Steps | Fixed $N_t$ for all $\theta$ | No $\theta$-dependent step count. |
| Hankel precision | `float64` | FP32 Hankel gradients are not validated. |
| CUDA graphs | Off (`use_cuda_graphs=False`) in every differentiable run | Graph launches never reach the tape. (D9) |
| RNG | Fixed NumPy seed for injection angles | Identical injection sequence in AD and FD runs. |

### 3.3 Sizes

| Size | Purpose | Grid | Steps | Particles (estimate) |
|---|---|---|---|---|
| Twin-XS | Operator, one-step, and short-horizon checks | 96 x 32 cells, as in `_lwfa`, with the twin density preloaded so that the laser meets the up-ramp within 20 steps | 50 to 300 | 0.1 to 0.5 M |
| Twin-S | Horizon sweep, identifiability, recovery | About 800 x 50 physical cells | About 1800 | 5 to 15 M (about 4.8 M loaded at `p_nt=20`, plus EV births) |
| Twin-M | Recovery closer to production physics | Chosen in Phase P1 to fit Section 8 | Up to about 10,000 | Up to about 30 M |

Twin-S keeps the compact physics of `ionization_script.py`: $a_0 \approx 4$, $w_0 = 5$ µm, $\tau = 16$ fs, $\lambda_0 = 0.8$ µm, so $z_R \approx 98$ µm.

- $a_0 \propto \sqrt{E_L}$ with fixed $\tau$, $w_0$, and $\lambda_0$, from the template formula `peak_power = E / tau / sqrt(pi/2)`.
- $z_f$ is measured from $z_c$. Nominal $z_f - z_c = +0.53\, z_R$, bounds nominal $\pm 0.44\, z_R$. These ratios match production (2.5 to 4.5 mm, $z_c = 2.3$ mm, $z_R = 2.26$ mm).
- $n_0$ and $F_L$ have bounds of nominal $\pm 20\%$.
- Record the plasma length in units of $z_R$. Production is about 0.9. Twin-S is at most about 0.5, which weakens $z_f$ sensitivity. Phase P1 measures this.

EV births add particles. Phase P5 measures their count. Section 8 scales the memory rows with it.

### 3.4 Fixed choices and targets

Store all fixed values in `inversion/configs/<size>.json`. Commit the file. A later change invalidates every L4 result recorded for that file. Re-run those checks after a change.

1. Twin physics (fill in Phase P1, freeze before Phase P2): $z_c$, $\beta$, $F_R$, $z_{lo}$, $z_{hi}$, the He to N ratio, the laser $z_0$, nominal $\theta$, bounds, $N_t$, and seeds.
2. Selection: choose $u_c$ at the minimum of the nominal $u_z$ histogram between the trapped peak and the low-energy population. Let $u_{lo}$ be the 99th weighted percentile of the low-energy population and $u_{hi}$ the 1st percentile of the trapped peak. Set $\delta_u = \min(u_c - u_{lo},\, u_{hi} - u_c)/4$.
3. Trapping check: at the nominal $\theta$, at the 16 corners of $\Theta$, and at the 5 optimizer starts, require at least 1000 macroparticles with $\sigma_i > 0.5$ and $Q \ge 0.1\,Q(\text{nominal})$. If a point fails, shrink the bounds of the parameter that causes it, record the new $\Theta$, and tell the user.
4. Optimizer starts: 5 points from a seeded Latin hypercube in $[0.1, 0.9]^4$, each at least 0.25 from $\hat\theta^\ast$ in the infinity norm.
5. Targets: write them with `inversion/make_target.py` to `inversion/targets/<size>_<variant>_Nt<N>.npz`. Keys: `Q`, `moments`, `rff_mean`, `standardization`, `s_k`, `theta_star`, `seed`, `p_nz`, `N_t`, `variant`, `config_sha256`, `git_hash`. `value_and_grad` refuses a target whose `config_sha256` or `N_t` does not match.
6. Noise (noisy L6 only): additive Gaussian noise of $0.02\, s_k$ on first moments; 2% multiplicative noise on $Q$ and on variances. Set $\alpha_D = 0$. Draw $\hat\theta^{pr} = \hat\theta^\ast + \mathcal{N}(0, 0.2^2)$ once with a fixed seed, and set $\hat\sigma = 0.2$.

## 4. Architecture

### 4.1 Data flow

```text
init (host, once):
  sim = build_twin(theta, size, config)          # production objects for grids and coefficients only
  s_0 = begin(sim)                               # laser fields, prologue, empty or preloaded plasma
  store laser tangents t_{E_L}, t_{z_f}          # Section 4.8

forward sweep (no tape):
  for n in 0 .. N_t-1:
      if n % K_c == 0: save checkpoint C_n       # GPU or pinned-host tier
      s_{n+1}, D_n = forward_step(s_n)
  J, lambda_{N_t} = objective_and_seed(s_{N_t})

backward sweep (last segment first):
  restore C_k; replay K_c steps with forward_step; keep s_k .. s_{k+K_c-1} and D_k .. D_{k+K_c-1}
  for n = k+K_c-1 down to k:
      with tape: s'_{n+1} = taped_step(s_n, decisions=D_n)
      check layout of s'_{n+1} against the state that produced lambda_{n+1}   # Section 4.5
      tape.backward(seed = lambda_{n+1})  ->  lambda_n
      accumulate the injection-weight terms of step n into dJ/dtheta          # Section 4.9
  after step 0: add the laser term and the t = 0 weight term                  # Sections 4.8, 4.9
```

### 4.2 Step state

`StepState` is a self-contained container of device arrays and host scalars. `forward_step` never mutates its input. It writes new arrays (ping-pong buffers are allowed). A stored state therefore stays valid without a deep copy, as long as nothing writes into it.

| Part | Contents | Size |
|---|---|---|
| Fields | Spectral `Ep, Em, Ez, Bp, Bm, Bz` and `rho_prev` for each mode | $16\, N_z^a N_r$ B per array |
| Particles, every species | `x, y, z, ux, uy, uz, inv_gamma, w`, tracker `id` | 72 B per particle |
| N ions | `pop5, pop6, pop7, w_pending` | 32 B per ion |
| Sorting, every species | `prefix_sum` (int32, $N_z(N_r+1)$), `prefix_sum_shift`, `sorted` flag | $4 N_z (N_r+1)$ B per species |
| Window and grid | `moving_win.zmin`, `t_last_move`, `comm._zmin_global_domain`, `fld.interp[m].zmin` and `zmax` for every mode | scalars |
| Injection, every species | `z_inject`, `z_end_plasma`, `nz_inject` | scalars |
| Tracking, every species | `next_attributed_id` | scalars |
| Clock | `time`, `iteration` | scalars |
| RNG | NumPy RNG state (`np.random.get_state()`) | about 2.5 KB |

Rules:

1. Keep `inv_gamma` as its own variable. The Vay push reads the stored old value, and the position push and J deposit read the new value.
2. `w_times_level` is derived, not state: $w (\bar Z - W/w) = w\bar Z - W$ for N ions (Section 5.1). He ions carry a fixed charge $2e$.
3. The interp E and B are derived from the spectral fields at the start of each step (`spect2interp`).
4. Give every species a tracker (`track()`). L2 and the layout check match particles by ID.
5. Restore from the openPMD checkpoint is not used. It stores interp fields only, and it omits `rho_prev`, the RNG, and the window state.

### 4.3 The functional stepper

`fbpic/autodiff/stepper.py` holds `begin`, `forward_step`, and `taped_step`. Production `fbpic/main.py` stays unchanged (D17).

- `begin(sim) -> StepState` replicates the `Simulation.step` prologue exactly once (`fbpic/main.py:475-518`): injection position setup, damping of the interp E and B, `interp2spect` of E and B, the first particle exchange, and the `rho_prev` deposit. It then converts the state into a `StepState`.
- `forward_step(state) -> (state, decisions)` mirrors the loop body at `fbpic/main.py:533-687` for the twin configuration, in the same order:
  1. exchange iteration (`iteration % exchange_period == 0`): removal, injection, `rho_prev` deposit;
  2. `spect2interp` of E and B; gather;
  3. Vay momentum push; first half position push;
  4. EV ionization and emission (Section 5), at the production slot of `handle_elementary_processes` (`main.py:618-620`);
  5. sort; J deposit; second half position push; sort; `rho_next` deposit;
  6. curl-free current correction; PSATD push; `push_rho`;
  7. moving-window shift; open-z damping (`exchange_and_damp_EB`).
- `taped_step(state, decisions)` is the same code path with an active `wp.Tape` and with `decisions` replayed instead of recomputed.

The step-0 diagnostic J deposit of production (`main.py:561-562`) is not reproduced. The J deposit of the same step overwrites it before any field push.
Exchange cadence and window shifts come from the `iteration` and window scalars stored in the state. No counter local to a segment or a replay is used.

`forward_step` and `taped_step` share one implementation. Use `forward_step` for the forward sweep, the segment replay, every FD run, and every EV truth run. Production `Simulation.step` is the physics reference for L2 and the stochastic model for L5 and truth variant T2.

### 4.4 Taped operations

| Production operation | Differentiable operation | Exists |
|---|---|---|
| Gather E, B at particles | `get_gather_kernel()` clone | Yes |
| Vay momentum push, electrons | `push_momenta` | Yes |
| Vay momentum push, He ions (fixed $2e$) | `push_momenta` with charge $2e$ | Yes |
| Vay momentum push, N ions (charge $(w\bar Z - W)e/w$) | `push_ion_momenta_ev` | No (Phase P5) |
| Position half push (x2) | `push_positions` | Yes |
| EV ionization and emission | `fbpic/autodiff/ionization.py` | No (Phase P5) |
| Sort before each deposit | Recorded permutation applied out of place with `permute_float64` (Section 4.5) | Partly |
| Deposit J and `rho_next`, and the exchange-step `rho_prev` | `deposit_five_modes` with the recorded prefix sum | Yes |
| `erase`, `divide_by_volume`, `filter` | `fields.py` twins | Yes |
| `interp2spect`, `spect2interp` | Adapters of Section 4.6 | No (Phase P3) |
| Curl-free current correction (standard) | `correct_currents_curlfree_standard` | Yes |
| PSATD push (standard), `push_rho` | `push_eb_standard`, `push_rho` | Yes |
| Moving-window spectral shift | `shift_spectral` | Yes |
| Open-z damping | `damp_eb_left`, `damp_eb_right` | Yes |
| Exchange-step removal and injection | Structural events (Section 4.9) | No (Phase P4) |

Rules for the taped step:

1. Write every operation out of place into a new array.
2. Never call `sort_particles` or `rearrange_particle_arrays` on arrays that a tape holds. The sorting buffer rotation overwrites saved arrays in the same call.
3. Tape guard. Inside an active tape, call only `fbpic.autodiff` kernels and `record_func` adapters. A production launch (`warp_kernel`, compiled with `enable_backward=False`, `fbpic/utils/warp.py:147`) goes through `wp.launch` with `record_tape=True`. The tape records it, and backward skips it with only a warning, so its gradient path is silently zero. Implement the guard in `fbpic/autodiff/step.py`:
   - Record `start = len(tape.launches)` before the step.
   - After the step, read each non-callable entry `L` in `tape.launches[start:]`. Read `L[0].options.get("enable_backward")`, else `L[0].module.options.get("enable_backward")`.
   - If the value is `False`, raise `RuntimeError` with `L[0].key`. Accept callables (`record_func` entries).
   - Launch integer-only decision kernels (cell index, prefix sum) with `record_tape=False`, as `particle_cells_from_prefix` does (`autodiff/deposition.py:237-244`).
   - Assert that `sim.use_cuda_graphs` is `False`. Launches inside a `_gpu_section` never reach the tape.
4. Call `tape.backward` only after the `with tape:` block exits, as `verification.py:94-111` does.
5. Refresh `DifferentiableArrays` views after any reallocation. Views are keyed by pointer.
6. Call `get_runtime().use()` at the start of the driver and of every test that mixes CuPy and Warp.
7. `taped_step` must give the same $s_{n+1}$ with and without an active tape, given the same decisions.

### 4.5 Discrete decisions and the layout invariant

The cotangent $\lambda_{n+1}$ is indexed in the particle order of the state that produced it. The output of `taped_step(s_n)` must have the same count and order, or the seed lands on the wrong particles without any error.

`StepDecisions` records, per species:

1. each sort permutation (int32) and its prefix sum, for the J sort and the `rho_next` sort, and the exchange-step sort;
2. the removal range `(i_min, i_max)` on exchange iterations;
3. the injected batch count;
4. for `elec_from_N`, the parent ion index of every new electron.

Rules:

1. `forward_step` computes and returns the decisions. `taped_step` uses them and recomputes none.
2. Sort with the production key `ir_upper + iz_upper*(Nr+1)` (`cuda_sorting.py:94`) and a stable sort (`cupy.argsort` on the key).
3. Assign slots for emission and injection with an exclusive prefix sum over the parent index or batch. Never use an atomic slot counter.
4. Before seeding step $n$, compare $s'_{n+1}$ with the state that produced $\lambda_{n+1}$: equal `Ntot` per species, equal IDs, `max|dx|`, `max|dy|`, `max|dz|` below $10^{-6}$ of the cell size, and `ux, uy, uz, w` within $10^{-9}$ relative. Run this check on every step in Phases P4 to P6, and at every segment boundary afterwards.
5. At an inner segment boundary, $\lambda_{k+K_c}$ comes from the replay of segment $k+1$, which starts from checkpoint $C_{k+K_c}$ of the sweep. The replay of segment $k$ produces its own $s_{k+K_c}$. Atomics rounding can change a decision between the two. If the check fails and the ID sets match, permute $\lambda_{k+K_c}$ by ID. Otherwise raise an error that names the step and the species.

Frozen decisions also define the FD reference (Section 6.3). With the decisions fixed, the forward map is piecewise smooth in $\theta$, and AD computes its exact derivative.

### 4.6 Spectral transform adjoints

Use the paired-real inner product of the existing autodiff contract (doc 08).

| Operation | Adjoint | Note |
|---|---|---|
| FFT along z (`CUFFT_FORWARD`, unnormalized) | $N_z^a \times$ `inverse_transform` | `inverse_transform` already divides by $N_z^a$ |
| `inverse_transform` | `transform` $/ N_z^a$ | |
| DHT forward, $G = F M$ | $\bar F = \bar G M^T$: `dgemm` with `transa=1` on `d_M` | Not the stored inverse matrix |
| DHT inverse | `dgemm` with `transa=1` on `d_invM` | |
| `rt_to_pm` | $\tfrac12$ `pm_to_rt` | Conjugate transpose of a 2 x 2 complex map |
| `pm_to_rt` | $2$ `rt_to_pm` | |

Rules:

1. Build each adapter from the differentiable layout clones in `fbpic/autodiff/fields.py:23-40` (`copy_2d_to_1d`, `copy_1d_to_2d`, `copy_2dc_to_2dr`, `copy_2dr_to_2dc`, `rt_to_pm`, `pm_to_rt`), launched normally so that the tape records them. Register only the cuFFT and `dgemm` calls with `tape.record_func`. The production `transform`, `inverse_transform`, `interp2spect_*`, and `spect2interp_*` methods launch forward-only copy kernels and would trip the tape guard.
2. If a production method must be reused, wrap it in a context manager `fbpic.autodiff.tape.suspended()` that sets `wp._src.context.runtime.tape` to `None` and restores it in a `finally` block. Use `suspended()` only when the adjoint of the wrapped call is registered.
3. Accumulate through scratch. `dgemm` and cuFFT work on split-real or 1D scratch, not on the interleaved `vec2d` layout. For each backward: copy the output gradient to scratch, run `dgemm` (`transa=1`, `beta=0`) or cuFFT into a second scratch, then add the result into the input gradient with a kernel.
4. For $m \ge 1$, the forward matrices for $p = m$ and $p = m+1$ have a zero column 0. The adjoint inherits this. Test all modes $m = 0..4$ and all three DHT orders.

The whole field side is linear. After Phase P4 passes, an optional optimization replaces the field part of the tape with a tape-free chain of hand-written transposes.

### 4.7 Particle adjoint details

- Linear shapes make gather and deposition derivatives in position jump at cell crossings. The cell, the Ruyten index, and the axis branch are fixed inside one step.
- The deposition adjoint is not the gather operator. Deposition has a Ruyten term and gather does not. Use the existing deposition VJP.
- Write squares as `x * x`, not `x ** 2.0`, in every new differentiable kernel. Warp's `pow` adjoint produces a NaN exponent adjoint for negative bases.
- Avoid mutable loop-carried locals in differentiable kernels. Put such loops in a `@wp.func`, as the `shift_spectral` fix in doc 08 does.

### 4.8 Laser parameter gradient

With the direct method, the laser sets all six initial interp fields once, on the host, before `begin`:

$$
F_0(\theta) = \mathcal{L}\big[\mathcal{E}(\cdot\,; E_L, z_f)\big].
$$

$F_0$ holds Er, Et, Ez, Br, Bt, Bz for every mode, on the local layout with guard cells. $\mathcal{L}$ is the complete linear map of `add_laser_direct` (`fbpic/lpa_utils/laser/direct_injection.py:13-106`):

1. `get_laser_Er_Et`: sample the profile and do the azimuthal decomposition;
2. copy Er and Et into a host `Fields` on the grid without guard cells;
3. `calculate_laser_fields`: `interp2spect`, the kz filter $(1 - \sin^4(k_z \Delta z/2))$ on Ep and Em, Ez from $\nabla\cdot E = 0$, B from the curl with `propag_direction`, then `spect2interp` of E and B;
4. add all six components into rows `[iz_in_array, iz_in_array + Nz_local)`. On one rank `iz_in_array = n_guard = 64`, and the guard rows receive zero.

`begin` then applies open-z damping $\mathcal{D}$ and `interp2spect` $\mathcal{T}$. Both are linear.

$$
\frac{\partial J}{\partial \theta_j} \mathrel{+}= \big\langle \mathcal{D}^T \mathcal{T}^T \lambda_0^{EB},\; t_j \big\rangle,
$$

summed over the six components and all modes, with the paired-real inner product. The tangents are:

- $t_{E_L} = F_0/(2E_L)$. This is exact, because $\mathcal{L}$ is linear and $\mathcal{E} \propto \sqrt{E_L}$, and the initial fields hold only the laser. Store $F_0$ at init.
- $t_{z_f} = \mathcal{L}[\partial\mathcal{E}/\partial z_f]$. For the Gaussian, the transverse profile is $T = \exp(-r^2/(w_0^2 D))/D$ with $D = 1 + i\,p\,(z - z_f)/z_R$, so $\partial T/\partial z_f = T\,(-ip/z_R)\,\big(r^2/(w_0^2 D^2) - 1/D\big)$. Write a host class `GaussianLaserDzf(LaserProfile)` whose `E_field` returns this derivative and that exposes `propag_direction`. Write a non-mutating helper `laser_initial_fields(sim, profile)` that runs steps 1 to 4 into new zero arrays. Call it with `GaussianLaserDzf`, and store the result on the device before `begin`.

Probes (Phase P4):

1. `laser_initial_fields(sim, GaussianLaser(theta))` equals the six arrays that `add_laser_direct` writes into zeroed fields, within $10^{-14}$ relative.
2. $t_{z_f}$ matches a host central FD in $z_f$ of `laser_initial_fields`, for all six components and all modes, with $h \in \{10^{-2}, 10^{-3}, 10^{-4}\}$ of the normalized range, within $10^{-6}$ relative.
3. The $E_L$ term equals $\langle \lambda_0^{EB}, s_0^{EB}\rangle/(2E_L)$, where $s_0^{EB}$ is the six spectral E and B arrays after `begin`. Exclude `rho_prev` and particles.

### 4.9 Density parameter gradient and structural events

Exchange iterations remove particles at the back of the window, inject new particles at the front, and redeposit `rho_prev`.

- Removal: a removed particle no longer affects the state. Its adjoint at the removal step is zero. The replay restores it from the checkpoint and propagates its adjoint backward from that step.
- Injection: positions, angles, and momenta do not depend on $\theta$. Weights do. At each injection event,
  $$
  \frac{\partial J}{\partial \theta_j} \mathrel{+}= \sum_{i \in \text{batch}} \bar w_i \, \frac{\partial w_i}{\partial \theta_j},
  $$
  summed over every injected species: He ions, N ions, and electrons. Compute $\partial w_i / \partial \theta_j$ on the host with the closed forms of Section 2.1. Store the batch index range per event.
- `w_times_level` is computed inside the taped step from $w$, $P_\ell$, and $W$ immediately before each deposit, including the exchange-step `rho_prev` deposit. The tape then carries its cotangent into $\bar w_i$. For an injected N ion, $P = (1,0,0)$ and $W = 0$, so $\bar w_i$ includes $5\,\overline{w_\text{tl}}_i$.
- `rho_prev` redeposit: a deposition from the particles on that step, with the existing deposition VJP.
- Particles loaded at $t=0$ (Twin-XS): add $\sum_i \bar w_i(0)\, \partial w_i(0)/\partial\theta_j$ after step 0.

### 4.10 Checkpointing and the reverse driver

`fbpic/autodiff/driver.py` implements two-level checkpointing first (D8):

1. Forward sweep: save `StepState` every $K_c$ steps. Put checkpoints on the GPU while they fit. Otherwise use pinned host memory.
2. Backward sweep, for each segment in reverse order: restore the checkpoint, replay $K_c$ steps with `forward_step`, and keep the states and decisions. Then run `taped_step` and `backward` for each step in reverse order.
3. Choose $K_c \approx \sqrt{N_t}$. Memory is $(N_t/K_c + K_c)$ states, $K_c$ decision records, and one step tape.
4. If two levels do not fit (Twin-M), implement binomial checkpointing (Griewank-Walther Revolve) over the GPU and host tiers.

Driver API:

```python
result = value_and_grad(theta, objective, config, segment_length=None)
# result.J: float
# result.grad: dJ/dtheta, shape (4,), float64, host
# result.grad_hat: dJ/dtheta_hat = diag(theta_hi - theta_lo) @ grad
# result.info: GPU and host memory peaks, timings, tape sizes, per-segment norms of lambda
# objective(state) -> (J, seed cotangents); the Section 2.4 loss or a Section 6.2 functional
```

## 5. Expected-value ionization

### 5.1 Model

Each N ion carries populations $P_5, P_6, P_7$ with $P_5 + P_6 + P_7 = 1$ and a pending birth weight $W$. At injection, $P = (1, 0, 0)$ and $W = 0$.
For each step, compute the rest-frame field $E$ and the Lorentz factor $\gamma$ with the production formulas (`warp_inline_functions.py:14-76`):

$$
x_\ell = \frac{1}{\gamma}\, a_\ell\, E^{q_\ell} \exp(c_\ell/E), \quad p_\ell = 1 - \exp(-x_\ell) = -\mathrm{expm1}(-x_\ell), \quad \ell \in \{5, 6\}.
$$

Use the old populations, which matches production's limit of one level per step:

$$
\Delta_{56} = P_5\, p_5, \quad \Delta_{67} = P_6\, p_6,
$$
$$
P_5' = P_5 - \Delta_{56}, \quad P_6' = P_6 + \Delta_{56} - \Delta_{67}, \quad P_7' = P_7 + \Delta_{67}, \quad W' = W + w(\Delta_{56} + \Delta_{67}).
$$

### 5.2 Emission and charge neutrality

Run the update and the emission at the production slot: after the Vay push and the first half position push, before the J deposit.

- Each ion with $W > \epsilon w$ emits one electron into `elec_from_N` in the same step. The electron gets weight $W$ and copies `x, y, z, ux, uy, uz, inv_gamma` from the parent ion. Then $W$ is reset to 0 for the emitting ions only.
- Ions with $W \le \epsilon w$ keep $W$ for later steps. An ion that never crosses the threshold never emits.
- Deposit and push each N ion with `w_times_level` $= w\bar Z - W$. This equals $5w$ plus the total weight the ion has emitted. Pending births are therefore neutral, and an emission raises the ion charge by $W$ at the position and in the step of the new electron. This matches production, which changes the ion level and creates the electron in the same call.
- Merge per parent ion, not per cell. A per-cell merge averages over the azimuth and destroys the $m = 1$ and $m = 2$ content of a linearly polarized laser.
- Defaults (D12): $K_e = 1$ (emission checked every step), $\epsilon = 10^{-4}$. Do not use $K_e > 1$: emission copies the parent momentum, so a delay of $K_e - 1$ steps shifts the birth phase (22.5 degrees per step in Twin-S) and adds up to about $0.9\, a_0$ of residual transverse momentum.
- Grow `elec_from_N` by reallocation at each emission, as production does. Assign slots with an exclusive prefix sum in parent order. Record the parent index of every new electron in `StepDecisions`.
- The emission kernel takes an optional `emit_mask` (by parent ion ID). When the mask is given, it replaces the threshold test. Frozen-decision FD runs use it (Section 6.3).

### 5.3 Adjoint

Use closed-form derivatives. Do not rely on code generation through `exp` and `pow` chains.

$$
\frac{\partial p_\ell}{\partial E} = (1 - p_\ell)\, x_\ell \left( \frac{q_\ell}{E} - \frac{c_\ell}{E^2} \right), \qquad \frac{\partial p_\ell}{\partial \gamma}\Big|_E = -(1 - p_\ell)\,\frac{x_\ell}{\gamma}.
$$

- $\log(1 - p_\ell) = -x_\ell$ exactly, so no term is singular at saturation.
- Guard $\sqrt{E^2}$ at $E = 0$.
- Chain $E = \sqrt{|\gamma \mathbf{E} + \mathbf{u} \times c\mathbf{B}|^2 - (\mathbf{u}\cdot\mathbf{E})^2}$ and $\gamma = \sqrt{1 + u^2}$ back into the gathered fields and the ion momentum.
- Between emissions, $\partial\, w_\text{tl}/\partial W = -1$ and $\partial\, w_\text{tl}/\partial P_\ell = w\ell$.
- At an emission, scatter the cotangents of the new electron's position, momentum, `inv_gamma`, and weight into the parent ion and into $W$, using the recorded parent index. The cotangent of the new weight and the $-1$ path into $W$ combine on the same $W$.
- The emission threshold is a decision. Its derivative is not differentiated (Section 6.3).

### 5.4 Stochastic referee

EV replaces macroparticle shot noise with its mean. Phase P7 measures how much this changes the gradient.

1. For each parameter $i$, use $h_i \in \{0.05, 0.1\}$ of the normalized range. Do not use the Section 6 steps ($10^{-2}$ to $10^{-4}$): they are noise-dominated for a discrete-birth model.
2. For seeds $j = 1..M$, run production stochastic ionization (`ions.track()`, `rng_seed`, `rng_stream`) at $\theta \pm h_i e_i$ with the same seed (CRN). Record $D_{ij} = J_s(\theta + h, j) - J_s(\theta - h, j)$.
3. Compute $g_S = \mathrm{mean}_j(D_{ij})/(2h)$ and $\mathrm{SE} = \mathrm{std}_j(D_{ij})/(2h\sqrt{M})$. Report the correlation of $J_s(\theta+h, j)$ with $J_s(\theta-h, j)$.
4. Start with $M = 8$. Double $M$, up to 64, until $\mathrm{SE} < 0.1 \max(|g_{EV,i}|, 0.01 \lVert\nabla J_{EV}\rVert_\infty)$.
5. Compute the EV secant $g_{EV}^h$ at the same $h$. Report $(g_S - g_{EV}^h)/\mathrm{SE}$ and $|g_S - g_{EV}^h|/|g_{EV}^h|$. Report $|g_{EV}^h - g_{EV}^{AD}|$ separately as EV truncation error.
6. Split the bias. Compare the linear statistics $Q$ and $S_{ab} = \sum_i w_i\sigma_i x^a u^b$ (seed means and secants) with EV values: this is the dynamics bias. Then compare $\mathbb{E}[J_s]$ with $J$ evaluated on the seed-mean statistics: this is the Jensen gap of the nonlinear loss.

## 6. Verification ladder

### 6.1 Levels

Run each level on GPU. Run each test file in its own process (`CLAUDE.md` rule 5).

| Level | Test | Pass criterion |
|---|---|---|
| L0 | Existing autodiff and fused tests | All pass, unchanged (122 GPU tests in `test_autodiff_*`) |
| L1 | New operators: FFT, DHT, EV kernels, ion push, permutation, loss terms, laser helper | `check_vjp` with its normalization and adjacent-epsilon pair rule (`verification.py:157-173`). Adjoint identity $e = \lvert\langle Ax,y\rangle - \langle x,A^T y\rangle\rvert / \max(\lVert Ax\rVert\lVert y\rVert + \lVert x\rVert\lVert A^T y\rVert, \text{tiny}) \le 10^{-12}$ for 3 random draws. Transforms at the Twin-S shape and at 2912 x 1080, all modes, all three DHT orders. |
| L2 | `forward_step` parity with production | Twin-XS and Twin-S, pre-ionized N (no EV): `begin` plus 300 `forward_step` calls against production `sim.step(300)` in one call from the same initial state. Fields within $\max(2\cdot10^{-11}, 10 d_{AA})$ relative in the infinity norm, where $d_{AA}$ is the difference of two production runs. Particle counts and IDs equal; particle arrays matched by ID within the same bound. |
| L2r | Restore | Runs A and B: straight 350 steps. Run C: restore the state saved at step 300, then 50 steps. Choose step 300 as an exchange iteration after a nonzero `n_move`. Pass if $d(C,A) \le \max(2\cdot10^{-11}, 10\, d(B,A))$ for every array and counts and IDs match exactly. |
| L3 | One full step | Random cotangent on all of $s_{n+1}$. `check_vjp` normalization and pair rule at $10^{-6}$. Steps: one ordinary step, one exchange step, one step with emissions (Phase P5). Harness rules in Section 6.3. |
| L4 | Parameter gradient, horizon sweep | Error definition and pass rule in Section 6.4. Horizons in Section 6.4. |
| L5 | EV vs stochastic | Section 5.4. Report only. The user decides acceptance. |
| L6 | Twin recovery | Noise-free: from the 5 starts, recover $\hat\theta^\ast$ within 0.02 in every component. Noisy: the recovered spread is within 2x the spread $\sqrt{\mathrm{diag}(G^{-1})}$ predicted by the Phase P1 sensitivity matrix. |

### 6.2 Check functional for L3 and L4

Until the loss is built (Phase P6), and whenever `elec_from_N` holds fewer than 1000 selected macroparticles, use

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
4. Free FD keeps the thresholds active. Report free minus frozen as the jump contribution. It is $O(\epsilon)$ and does not vanish as $h \to 0$.
5. L3 harness: run the base step once and store its decisions. Give `check_vjp` an optional `direction_masks` argument. Mask out every particle whose fractional cell coordinate in z or r, at gather, J deposit, or `rho_next` deposit, is within $10^{-3}$ of an integer. The shape kernels take the cell from `ceil(position)`, so a crossing cannot be frozen.
6. Measure the run-to-run noise $\sigma_J$ by running the base $\theta$ three times. Use only steps $h$ with $2h\lvert g_j\rvert > 100\,\sigma_J$.

### 6.4 L4 error, pass rule, and horizons

- Error per component: $e_j(h) = \lvert g^{AD}_j - g^{FD}_j(h)\rvert / \max(\lvert g^{AD}_j\rvert, \lvert g^{FD}_j(h)\rvert, 10^{-6}\lVert g^{FD}(h)\rVert_\infty)$, in $\hat\theta$, with $h \in \{10^{-2}, 10^{-3}, 10^{-4}\}$.
- Pass: two adjacent $h$ give $e_j \le 10^{-4}$ for every $j$. A component also passes when $\lvert g^{AD}_j\rvert$ and $\lvert g^{FD}_j\rvert$ are both below $10^{-6}\lVert g\rVert_\infty$. This covers $n_0$ and $F_L$ before any plasma is injected.
- Report $e_j(h)$ for every $h$ and horizon, the growth of $\lVert\lambda_n\rVert$, and 1D slices of $J$ over $\pm 0.05$ in each $\hat\theta_i$.
- Horizons: $N_t \in \{1, 10, 50\}$ and $\{n_c + 10, n_c + 50, n_c + 200\}$ must pass. $\{1000, \text{full}\}$ are reported. $n_c$ is the first step at which the laser peak reaches the position where $S(z) = 0.1$, at the nominal $\theta$. On Twin-XS, $n_c \le 20$.
- For horizons past the pass set, decide with the user how to proceed (shorter horizon, smoother objective, or FD fallback).

## 7. Rollout plan

Effort estimates are agent working days, with large uncertainty because the long-horizon behavior is unknown.

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
3. Build Twin-XS, Twin-S, and a first Twin-M candidate. Write `inversion/configs/<size>.json` (Section 3.4).
4. Aliasing check: before the laser reaches the plasma, compute $\lvert\rho_m\rvert/\lvert\rho_0\rvert$ for $m = 1..4$ over the plasma. Require it below $10^{-10}$.
5. Weight check: at the 16 corners of $\Theta$, every injection event gives the same `Ntot` per species, and all weights are positive.
6. Trapping: run Twin-S with stochastic ionization at the nominal $\theta$. Set $u_c$ and $\delta_u$ (Section 3.4). Run the trapping check at the 16 corners and at the 5 starts.
7. Identifiability:
   - For each parameter $i$, run Twin-S forward-only at $\hat\theta^\ast \pm 0.05\,e_i$ with stochastic ionization, the same seed for each pair, and 4 seeds (32 runs).
   - Compute $v = (\log Q, m_k)$ and the sensitivity $S = \partial v/\partial\hat\theta$ by central differences. Record the seed mean and its SE.
   - Run the T2 truth at $\theta^\ast$. Set $\Sigma = \mathrm{diag}(\text{seed variance of } v + (v_{truth} - v_{model})^2)$, with each entry at least $(0.02\, s_k)^2$.
   - Form $G = S^T \Sigma^{-1} S$. Record its eigenvalues, $\mathrm{cond}(G)$, the weakest eigenvector, and $\sqrt{\mathrm{diag}\, G^{-1}}$.
   - Stop and ask the user if $\mathrm{cond}(G) > 10^4$ or any $\sqrt{(G^{-1})_{ii}} > 0.02$. Options: a longer plasma (Twin-M), a different observable, or fewer parameters.
8. Record Twin-S charge, energy, run time, and peak memory. Choose the Twin-M grid with Section 8.

Acceptance: steps 4 to 7 pass, or the user has decided how to proceed. The config files are frozen.

### Phase P2. Functional stepper without ionization (4 to 6 days)

1. Implement `StepState`, save, and restore with the GPU and pinned-host tiers (Section 4.2).
2. Implement `begin` and `forward_step` for `ionization="none"` with production-equivalent kernels and recorded decisions (Sections 4.3 and 4.5). Use the existing autodiff kernels where they exist. Use production kernels inside `suspended()` only for the untaped path.
3. Run L2 on Twin-XS and on Twin-S.
4. Run L2r.

Acceptance: L2 and L2r pass. Production code is unchanged.

### Phase P3. Spectral adjoints (3 to 4 days)

1. Implement the FFT, inverse FFT, DHT, and inverse DHT adapters in `fbpic/autodiff/spectral.py` (Section 4.6).
2. Compose the `interp2spect` and `spect2interp` adapters for scalar and vector fields.
3. Run L1 for these adapters at both shapes, all modes, and all three DHT orders.
4. Switch `forward_step` to the adapters, and re-run L2 on Twin-XS.

Acceptance: L1 passes, and L2 still passes.

### Phase P4. Taped step and the minimal driver, without ionization (6 to 8 days)

1. Implement `taped_step` for `ionization="none"`, with replayed decisions and the tape guard (Section 4.4).
2. Implement the structural events and the injection-weight term (Section 4.9).
3. Implement the laser term and run its three probes (Section 4.8).
4. Implement `value_and_grad` with two-level checkpointing and the layout check (Sections 4.5 and 4.10). The objective is a parameter.
5. Write the FD harness `inversion/verify_param_gradient.py` (Section 6.3).
6. Add an injection-term probe: at one injection event, scale only the injected N-ion weights by $(1 + h)$, and check the term against central FD of $J_{chk}$. Repeat for electrons only.
7. Run L3 on Twin-XS: one ordinary step and one exchange step.
8. Run L4 on Twin-XS with $J_{chk}$ at the pass horizons.

Acceptance: probes, L3, and L4 (pass horizons) pass on Twin-XS without ionization.

### Phase P5. Expected-value ionization (6 to 9 days)

1. Implement the EV state, the ADK kernel, emission, `emit_mask`, and `push_ion_momenta_ev` (Section 5) in `fbpic/autodiff/ionization.py`. Add EV to `forward_step` and to `StepState` save and restore.
2. Open-loop parity: freeze the gathered fields and ion momenta of a set of ions over 200 steps. Give each ion a prescribed field so that $p_5$ and $p_6$ each cover about $10^{-3}$ to 0.3 over the set. Run the EV kernel and 32 Philox seeds of the production kernel. Pass if $\sum_i w_i P_\ell$ for $\ell = 5,6,7$ and the cumulative birth weight each satisfy $\lvert EV - \text{mean}\rvert \le 3\,\mathrm{SE}$.
3. Charge test: deposit rho from the N ions and `elec_from_N` before and after the EV call. The difference must be below $10^{-12}$ relative, on emission steps and on steps without emission.
4. Invariant test on Twin-S with EV, after 3 exchange periods: $P_5 + P_6 + P_7 = 1$ to $10^{-14}$; the charge of `elec_from_N` plus $\sum_i W_i$ equals $\sum_i w_i(\bar Z_i - 5)$; each ion's populations follow its tracker ID through sorts.
5. Implement the adjoints of Section 5.3. Run L1 on each kernel, with `ux, uy, uz` and the gathered fields as differentiable inputs.
6. Add EV to `taped_step`. Run L3 on an emission step and L4 on Twin-XS at the pass horizons.
7. Run Twin-S forward with EV at the nominal $\theta$ and at two opposite corners. Record the `elec_from_N` count and peak memory. Update Section 8.

Acceptance: steps 2 to 6 pass.

### Phase P6. Loss and the full horizon sweep (5 to 7 days)

1. Implement the loss module and `inversion/make_target.py` (Sections 2.4 and 3.4). Run L1 on each loss term.
2. Run `value_and_grad` on Twin-XS at $N_t = 200$ with `segment_length` in $\{5, 13, 14, 40\}$. The gradients must agree within $10^{-10}$ relative. This covers segment lengths that do not divide $N_t$ and boundaries on and off exchange steps.
3. Measure peak GPU memory, host memory, and time per gradient on Twin-S.
4. Run the full L4 sweep on Twin-S with $J_{chk}$, and with the loss where the selected beam has at least 1000 macroparticles at every FD point.
5. Implement Revolve if Twin-M does not fit two levels.

Acceptance: step 2 passes. L4 results for all horizons are reported to the user.

### Phase P7. Model checks (3 to 4 days)

1. Run L5 (Section 5.4) on Twin-S at the nominal $\theta$.
2. Merge check: compare $\epsilon \in \{10^{-4}, 10^{-5}\}$ at $K_e = 1$ against $\epsilon = 10^{-6}$, forward-only, at the nominal $\theta$ and two corners. Compare $Q$ and every $m_k$. Adopt $\epsilon = 10^{-4}$ only if every difference is below 2 SE of 16 stochastic seeds.
3. Optional: add the weighted SWD term.

Acceptance: L5 and the merge check are reported to the user.

### Phase P8. Twin recovery (3 to 5 days)

1. Write `inversion/recover.py` with `scipy.optimize.minimize(method="L-BFGS-B")` on $\hat\theta$, bounds $[0, 1]$, and `value_and_grad(...).grad_hat`.
2. Generate targets for T0, T1, and T2 (Section 2.2).
3. Run L6 noise-free on Twin-S for T0 and T1 from the 5 starts. Record iterations, gradient evaluations, and wall time. Report T2.
4. Run L6 with noise (Section 3.4, item 6) for T1, with 5 noise realizations from one start.
5. Run L6 on Twin-M if Phase P6 shows that it fits.

Acceptance: L6 passes on Twin-S for T0 and T1. Results are recorded.

### Phase P9. Documentation (1 day)

1. Add `docs/warp-porting-guide/09-differentiable-step-and-inversion.md` with contracts, measured memory, and gradient accuracy.
2. Update the next steps in doc 08 and Section 12 of this plan.

### Phase P10. Forward-mode baseline (optional, 5 to 8 days)

Warp 1.18 has no forward-mode code generation. Implement hand-written tangent kernels for gather, push, deposit, and EV only if the user asks, or if L4 shows that reverse mode needs a second exact reference. Linear field operators reuse the production transforms on tangent arrays.

## 8. Memory and performance budget

Estimates. Phases P1, P5, and P6 replace them with measurements.

| Quantity | Formula or source | Twin-S | Twin-M |
|---|---|---|---|
| Field array | $16 N_z^a N_r$ B | about 0.9 MB | about 30 MB |
| Field checkpoint | 7 arrays x $N_m$ | about 32 MB | about 1 GB |
| Particle checkpoint | about 90 B per particle (plus 32 B per N ion) | about 1 GB | about 3 GB |
| EV births | measured in Phase P5 | up to about 1.7 x background (estimate) | measure |
| One-step tape | about 0.7 KB per particle plus field versions | about 4 to 10 GB | about 20 GB |
| Recorded permutations | about 2.1 sorts per step x 4 B x $N_p$ x $K_c$ | about 2 to 4 GB | about 30 GB (recompute or host tier) |
| Two-level storage | $(N_t/K_c + K_c)$ states | about 85 x 1.05 GB = 90 GB: checkpoints on the pinned-host tier, replay states (about 42 GB) on the GPU | Revolve over GPU and host |
| Forward time per step | doc 07: 0.31 s for 1852 x 300 x 5 with about 57 M particles | well below 0.1 s | measure |
| Gradient cost | 1 sweep + 1 replay + 1 taped step and backward per step | about 4 to 6 forward runs | about 4 to 6 forward runs |

If host memory is insufficient for Twin-S, reduce `p_nt` to 12 (not below 9) or limit the plasma radius with `p_rmax`. Record the choice.
DHT cost grows as $N_z^a N_r^2$. Keep Twin-M at moderate $N_r$.
The CuPy pool and the Warp allocator compete for device memory. Measure the peak of both.

## 9. Risks and mitigations

| Risk | Effect | Mitigation |
|---|---|---|
| Weak identifiability ($z_f$ against $E_L$) | Recovery is ill-posed | Phase P1 step 7 before any adjoint work. User decides. |
| Gradient growth over long horizons (trapping sensitivity) | AD disagrees with FD at long horizons | L4 sweep, $\lVert\lambda_n\rVert$ report. User decides. |
| Forward map of AD differs from the map that made the states | Consistent but wrong gradient | One shared stepper (D17). L2 against production. |
| Particle order mismatch | Seed on wrong particles, no error | Recorded decisions and the layout check (Section 4.5) |
| Silent zero gradients from production launches on the tape | Wrong gradient without error | Tape guard (Section 4.4 rule 3) |
| EV model bias | Gradient of a different model than the stochastic code | L5, T2 truth. User decides. |
| Charge non-conservation from pending births | Spurious fields at the injection point | Neutral pending deposit (Section 5.2), charge test in Phase P5 |
| Azimuthal aliasing at low `p_nt` | Spurious mode-4 fields | `p_nt = 20`, aliasing check in Phase P1 |
| Atomics non-determinism | Replay differs from the sweep by rounding | Backward uses replayed states. Layout check with ID fallback. |
| Private Warp API (`wp._src.context.runtime.tape`) | Breaks on a Warp upgrade | Keep Warp at 1.18.0 |
| Inverse crime | Over-optimistic recovery | Truth variants T0, T1, T2 |
| Memory of Twin-S and Twin-M | Does not fit | Pinned-host tier, Revolve, lower `p_nt` |

## 10. Decisions

| ID | Decision | Status |
|---|---|---|
| D1 | One GPU only. No MPI or NCCL adjoint. | User, 2026-10-26 |
| D2 | `use_mpi=False`, `n_order=-1` for the twin. | User intent (single GPU), planner choice of stencil |
| D3 | Lab frame first. Boosted frame later. | Planner default |
| D4 | Direct-method `GaussianLaser`. LASY antenna later. | Planner default |
| D5 | Standalone twin configuration, not the production template. | Planner default |
| D6 | Vary $F_L$ only. Keep $z_c$, $\beta$, $F_R$, and the indicator fixed. | Planner default |
| D7 | Pre-ionized He$^{2+}$ and N$^{5+}$. EV for the N K shell only. | Planner default |
| D8 | Checkpointing with a one-step tape. No reverse-time reconstruction. | Planner default |
| D9 | Eager differentiable path. No CUDA graphs. | Planner default |
| D10 | Reverse mode is the main engine. Frozen-decision central FD is the reference. Forward mode is optional. | User preference for Warp reverse mode |
| D11 | No new Python dependency. Loss in CuPy or Warp. | Global work rules |
| D12 | EV emission per parent ion each step: $K_e = 1$, $\epsilon = 10^{-4}$, neutral pending deposit. | Planner default, checked in Phase P7 |
| D13 | Generated adjoints inside the one-step tape. Tape-free field transposes only as a later optimization. | Planner default |
| D14 | L-BFGS-B on normalized parameters, 5 Latin-hypercube starts. | Planner default |
| D15 | He as fixed-charge 2+ ions without an ionizer. | Planner default |
| D16 | `p_nt = 20` for every species. | Planner default (FBPIC rule `p_nt >= 4 Nm`) |
| D17 | Production stepping is unchanged. The differentiable path uses its own functional stepper for the sweep, the replay, FD, and EV truth runs. | Planner default after review |
| D18 | Truth variants T0, T1, T2 with the same grid and $N_t$. | Planner default |
| D19 | Quintic smoothstep selection and log charge with floor $Q_f$. | Planner default |
| D20 | Identifiability gate in Phase P1 before adjoint work. | Planner default |

## 11. Future work

1. Boosted frame with the back-transformed observable computed from the final state (smooth selection, one adjoint seed).
2. LASY antenna path: a GPU envelope interpolator, then adjoints of the unsorted antenna deposition.
3. Multi-GPU adjoint: reverse NCCL exchanges, where the adjoint of a copy exchange is an add exchange.
4. Reverse-time reconstruction of particle trajectories within short segments, with logs of irreversible operations.
5. Experimental realism: drop z from $H$, GPSR reconstructions as targets with their uncertainty, and multi-shot jitter.
6. Hybrid ML: gradient-enhanced surrogate training, and simulation-based inference as a warm start.
7. More parameters: pulse duration, spot size, Zernike terms, and a free-form density lineout.

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

_No implementation phase has run yet._

## Appendix A. Unverified claims and required probes

| Claim | Probe | Phase |
|---|---|---|
| The $k=0$ columns of the $p=m$ and $p=m+1$ components stay zero for $m \ge 1$ | Compare spect with `interp2spect(spect2interp(spect))` after 500 steps | P2 |
| cuFFT and FP64 `dgemm` are deterministic on one GPU | Run a transform twice and compare bitwise | P3 |
| About 0.7 KB of tape per particle per step | Measure Warp and CuPy peaks for one taped step | P4 |
| `elec_from_N` traps a beam in Twin-S with 5 modes, `p_nt = 20`, at all checked points | Phase P1 steps 5 and 6 | P1 |
| The four parameters are locally identifiable in Twin-S | Phase P1 step 7 | P1 |
| EV births match the stochastic marginals in open loop | Phase P5 step 2 | P5 |
| Two-level checkpointing fits Twin-S with the pinned-host tier | Measure in `value_and_grad` | P6 |
| Generated helper adjoints (`push_p_vay`, `add_linear_gather_for_mode`, `Sr_linear`) are correct in composed use | L3 | P4 |

## Appendix B. File map

| Path | Change |
|---|---|
| `fbpic/autodiff/state.py` | New: `StepState`, `StepDecisions`, save, restore, tiers |
| `fbpic/autodiff/stepper.py` | New: `begin`, `forward_step`, `taped_step` |
| `fbpic/autodiff/tape.py` | New: `suspended()`, the tape guard |
| `fbpic/autodiff/spectral.py` | New: FFT and DHT adapters with `record_func` |
| `fbpic/autodiff/ionization.py` | New: EV populations, ADK, emission, adjoints |
| `fbpic/autodiff/particles.py` | Add `push_ion_momenta_ev` |
| `fbpic/autodiff/laser.py` | New: `laser_initial_fields`, `GaussianLaserDzf` |
| `fbpic/autodiff/driver.py` | New: checkpoint schedules, `value_and_grad` |
| `fbpic/autodiff/loss.py` | New: observable, loss, check functional, analytic gradients |
| `inversion/twin_config.py` | New: `build_twin`, density closed forms |
| `inversion/configs/*.json` | New: frozen twin choices (Section 3.4) |
| `inversion/make_target.py` | New: target generation (T0, T1, T2) |
| `inversion/identifiability.py` | New: Phase P1 step 7 |
| `inversion/verify_param_gradient.py` | New: AD vs FD harness, horizon sweep |
| `inversion/recover.py` | New: L-BFGS-B recovery driver |
| `tests/test_autodiff_spectral.py`, `test_autodiff_ionization.py`, `test_autodiff_stepper.py`, `test_autodiff_laser.py`, `test_autodiff_loss.py` | New tests (L1 to L3) |
| `docs/warp-porting-guide/09-differentiable-step-and-inversion.md` | New chapter |

`inversion/` is a new directory under `lpa/FBPIC_Warp`. It holds research drivers and `verify_*.py` scripts, not library code. No production module changes.

## Appendix C. Research basis

| Topic | Source | Use in this plan |
|---|---|---|
| Phase-space reconstruction with differentiable transport | Roussel et al., PRL 130, 145001 (2023); PRAB 27, 094601 (2024) | Problem framing; later GPSR targets |
| Reverse-mode memory in accelerator PIC | Dhamrait et al., PASC '26, arXiv:2605.27282 | A full tape does not fit; checkpointing needed |
| Checkpointing | Griewank and Walther, ACM TOMS 26 (2000); Aupy et al., SISC 38 (2016) | Section 4.10 |
| Reverse-time reconstruction | Li et al., ApJS 270, 36 (2024); Greener, PNAS 122 (2025) | Future work; segment length limits |
| Discrete events in adjoint Monte Carlo | Yang, Silantyev, Caflisch, JCP 488 (2023); Caflisch and Yang (2024) | Score terms; EV rationale |
| Deterministic ionization in PIC | DiIorio et al., arXiv:2603.17885 (2026) | EV populations |
| AD with discrete randomness | Arya et al., NeurIPS 2022 | Stochastic referee, future estimator |
| Gradient quality over long horizons | Metz et al., arXiv:2111.05803 | L4 horizon sweep |
| Approximate adjoints | Dwight and Brezillon, AIAA J. 44 (2006) | Verify the exact adjoint before shortcuts |
| Forward-mode PIC | Qiang, PRAB 26, 024601 (2023) | Optional baseline |
| Continuous Vlasov adjoints | Bartsch et al., SICON 62 (2024); Albi et al., arXiv:2610.26110 | Future macroscopic adjoint research |
