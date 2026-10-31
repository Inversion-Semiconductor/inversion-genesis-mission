# Differentiable Inversion Plan for FBPIC-Warp

| Item | Value |
|---|---|
| Status | Phases P0 to P3 done; milestone M1 reached with the Stage A gate (Section 12.5). Revised on 2026-10-30 with the optimizer of Section 2.5 (D26) and the one-parameter Stage A (Section 2.6, D28). |
| Date | 2026-10-26 |
| Branch | `snidhan/dev-autodiff` (base commit `92352b8`) |
| Scope | One GPU. Lab frame. Stage A: one parameter ($E_L$) from Phase P3 to Phase P9. Stage B: both parameters ($E_L$, $z_f$). Forward-mode tangents. Twin experiment with noise-free synthetic data. |
| Owner | The implementing agent writes all code: kernels, tangents, algorithms, scaffolding, and tests. |
| Production code | Unchanged. The differentiable path is a separate stepper in `fbpic/autodiff` (D17). |
| Decisions | D1 to D28 in Section 10. D1, D2, and D21 to D28 are user decisions. The others are planner defaults that the user can change. |

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
9. Recovery of $\theta^\ast$ with a bounded trust-region least-squares solver (Section 2.5, Phase P9), first for one parameter (Stage A, Section 2.6).

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
Recovery runs (Phases P3 and P9) use the truth variants to avoid an inverse crime. Every variant keeps the inversion grid, $\Delta t$, $N_t$, `exchange_period`, and the `n_move` sequence, so that $H$ is evaluated at the same time. Every variant except T0′ uses a different NumPy injection seed and `p_nz = 2` instead of 1. T0′ uses exactly the inversion settings, so $J(\theta^\ast)$ is at round-off level: a recovery on T0′ tests convergence alone, without model bias.

| Variant | Ionization model | Role |
|---|---|---|
| T0′ | EV with exactly the inversion settings (seed, `p_nz = 1`, $K_e$, $\epsilon$) | Convergence control. L6 applies. |
| T0 | EV with the inversion settings ($K_e$, $\epsilon$) | Bias report: the `p_nz` model error. Report the offset from $\hat\theta^\ast$. |
| T1 | EV with immediate emission ($\epsilon = 0$). If the $\epsilon = 0$ births do not fit in memory (Phase P8 step 3), use the reduced Twin-S for T1 and for its T0 control. | Main test. Report the offset from $\hat\theta^\ast$ (L6). |
| T2 | Production stochastic ionization (Philox, tracked ions) | Model error. Report only. |

If T0′ and T0 pass and the T1 offset exceeds 0.02 in a component, stop. Report the recovered-parameter bias together with the emission-delay results of Phase P8, and ask the user.

### 2.3 Observable H

The twin runs in the lab frame (D3). The observable is the state of the species `elec_from_N` at one fixed step $N_t$.
Choose $N_t$ so that the window has passed $z_{hi}$ for every $\theta \in \Theta$. $z_{hi}$ does not depend on $\theta$.

The selection weight is a quintic smoothstep on $u_z$:

$$
\sigma_i = S_5\big((u_{z,i} - u_c + 2\delta_u)/(4\delta_u)\big), \quad S_5(t) = 6t^5 - 15t^4 + 10t^3 \text{ on } [0,1],
$$

with $S_5 = 0$ for $t \le 0$ and $S_5 = 1$ for $t \ge 1$. $\sigma_i$ is exactly 0 below $u_c - 2\delta_u$, so particles removed at the window back do not contribute. Phase P1 sets $u_c$ and $\delta_u$ (Section 3.4). The effective weight is $w_i\sigma_i$. There are no hard cuts and no quantile crops.

### 2.4 Loss

The loss is a whitened nonlinear least-squares objective over 11 residuals, plus an optional distribution term:

$$
J(\theta) = \lVert r_w \rVert^2 + \alpha_D\, \mathrm{MMD}^2_{\mathrm{RFF}}, \qquad r_w = L^{-1}\big(v(\theta) - v_y\big), \qquad \Sigma = L L^T .
$$

The observables $v$ are:

1. $\log(Q + Q_f)$. $Q = e \sum_i w_i \sigma_i$ is the selected charge, and $Q_f = 10^{-3} Q_y$.
2. The 4 weighted first moments $\langle\xi_k\rangle = \sum_i w_i\sigma_i \xi_{k,i} / (\sum_i w_i \sigma_i + Q_f/e)$ of $x$, $u_x$, $z - z_{win}$, and $u_z$, with $\xi = (x, u_x, y, u_y, z - z_{win}, u_z)$. The loss leaves out $\langle y\rangle$ and $\langle u_y\rangle$: by symmetry they carry no information on $\theta$ (sensitivity near 0), and they jitter by up to 4 model-error units when $\hat\theta$ changes by 0.003. In Phase P3 that jitter stalled the trust-region solver (Section 12.5).
3. $\log(\sigma_k + f_k)$ for $x$, $u_x$, $y$, $u_y$, and $z - z_{win}$. $\sigma_k^2 = \langle\xi_k^2\rangle - \langle\xi_k\rangle^2$ is the central second moment, with the same weights and floor. The floor $f_k = 10^{-3}$ times the target spread keeps $v$ finite when the selected beam vanishes.
4. The log relative energy spread, $\log\big((\sigma_{u_z} + f_{u_z}) / (\langle u_z\rangle + f_\mu)\big)$, with $f_\mu = 10^{-3} \langle u_z\rangle_y$.

Central moments replace the raw second moments $\langle\xi_k^2\rangle$ of the first draft. A raw second moment equals $\sigma_k^2 + \langle\xi_k\rangle^2$, so it counted the means of $z - z_{win}$ and $u_z$ again, by factors of about 170 and 40 (Section 12.5). Log scales make the spreads and the charge dimensionless and treat relative changes equally.

Whitening. $\Sigma$ is the residual covariance. In the twin, $\Sigma = \mathrm{diag}(\Delta^2)$, a model-error term: the mean over the nominal $\theta$ and the 4 corners of $(v_{truth} - v_{model})^2$, with the truth settings against the inversion settings. These points keep knowledge of $\theta^\ast$ out of the weights. One point is not enough: at the nominal point alone, $\Delta$ of the relative energy spread was 70 times smaller than the model error at $\theta^\ast$, and that one residual made up 99.6% of $J(\theta^\ast)$ (Section 12.5). The EV twin is deterministic, so no sampling term is needed. For experimental data, $\Sigma$ comes from shot-to-shot scatter plus a model-error term (Section 11).

- $\mathrm{MMD}^2_{\mathrm{RFF}} = \lVert \mu - \mu_y \rVert^2$, with $\mu = \sum_i w_i\sigma_i \phi(\tilde\xi_i) / (\sum_i w_i\sigma_i + Q_f/e)$. The denominator has the same floor as the moments. $\phi$ has $D = 512$ random Fourier features, seed 1234, bandwidth 1.0. $\tilde\xi$ is $\xi$ standardized with the frozen target mean and standard deviation, never with statistics of the current sample. Use it only in a fine stage after the moment fit (Section 2.5).
- The weight gradient is
  $$
  \frac{\partial\, \mathrm{MMD}^2}{\partial (w_i\sigma_i)} = \frac{2\,\langle \mu - \mu_y,\; \phi(\tilde\xi_i) - \mu\rangle}{\sum_k w_k \sigma_k + Q_f/e}.
  $$
- Empty and nearly empty beams. If every $\sigma_i = 0$, then the first moments and $\sigma_k$ are 0, every $\log$ term takes its floor value, and $\mathrm{MMD}^2 = \lVert\mu_y\rVert^2$. The loss and every gradient stay finite. The trapping checks of Section 3.4 do not cover every point that the optimizer can visit, so the loss must not rely on a positive selected charge.
- No prior term. The targets are noise-free (D22). A prior, as in noisy recovery (Section 11), depends on $\hat\theta$ directly; the objective must then also return that direct gradient.
- Defaults: $\alpha_D = 0$ until Phase P7.

Implement the loss and its analytic gradient in CuPy or Warp (D11). Test every term with FD on random particle sets (L1). Also test the loss and its gradient on an empty selection (all $\sigma_i = 0$) and on a nearly empty selection (one particle with $\sigma_i = 10^{-6}$). Both must be finite and match FD.

### 2.5 Optimizer

The optimizer follows the literature note `inversion/notes/optimizer-literature-2026-10-30.md` (D26). It uses 5 techniques:

1. Trust-region Gauss-Newton or Levenberg-Marquardt on the residual vector $r_w$, not a quasi-Newton method on the scalar $J$: `scipy.optimize.least_squares(method="dogbox")` with bounds $[0, 1]^2$. The model Hessian $J_r^T J_r$ exists from the first iteration, and the trust radius bounds every step. L-BFGS-B failed here in Phase P3: its first step crossed the box, and a secant pair from a flat region sent the next step to the opposite corner (Section 12.5).
2. A global design, then a local solve. Evaluate a 25-point grid or Sobol design over $[0, 1]^2$. Start the local solver from the 2 or 3 best design points that are well separated. A bounded local method cannot leave a boundary local minimum, and the design shows whether one exists.
3. Loss design (Section 2.4): central moments, log scales, and whitened residuals.
4. FD steps. Use fixed steps, $h = 0.02$ for $E_L$ and $h = 0.05$ for $z_f$; the FD Jacobian is temporary until the JVP Jacobian of item 5. Use one-sided second-order stencils $(x, x + h, x + 2h)$ within $h$ of a bound. Keep $h$ above the width of the jumps that emission thresholds and cell crossings cause. Removing jumps in the model (zero-weight creation, window-edge tapering) is a separate decision for Phase P6.
5. JVP Jacobians in the same driver. After the JVP gate of the note (Section 3.2 there: comparison with central FD at several $h$ and with a smoothed Jacobian), the `jac` callable becomes 1 primal run with $K$ tangents, $K = 1$ in Stage A and $K = 2$ in Stage B (Sections 2.6 and 4.10). Before that gate, the driver uses the batched FD Jacobian, which reuses the cached center run.

### 2.6 Staged scope: one parameter first

The work runs in two stages. Stage A takes one scalar parameter through every milestone, end to end. Stage B adds the second parameter after Stage A works.

Stage A (D28):

1. Parameter: $E_L$. $z_f$ stays at $z_f^\ast$ in the model and in every target. The tangent seed of $E_L$ is exact and simple: every initial field scales as $\sqrt{E_L}$, so the seed is $F_0 / (2E_L)$ (Section 4.8). The $z_f$ seed needs the profile derivative and `GaussianLaserDzf`, which move to Stage B.
2. Phases P3 to P9 apply with one tangent ($K = 1$). Write the tangent kernels and the driver generic over $K$ (Section 4.7), but run the Stage A tests, the L4 sweeps, and the recovery with the $E_L$ tangent only.
3. The loss, the weights $\Sigma$, the targets (T0′, T0, T1, T2), and the optimizer of Sections 2.2 to 2.5 do not change. The optimizer works on one coordinate, with a 10-point design in place of the 25-point design.
4. Milestones: M1 ends with the Stage A FD gate (Phase P3 step 15). M2 is the $K = 1$ tangent gradient. Phases P6 to P9 follow for $E_L$, and P9 compares the JVP recovery with the FD recovery of P3.

Stage B starts after Stage A passes Phase P9:

1. Add the $z_f$ tangent (Section 4.8) and run the $K = 2$ checks.
2. Solve the global search problem first. The two-parameter FD gate of Phase P3 failed because the basin of $\theta^\ast$ has a radius of about 0.04 in $\hat\theta$, while the 25-point design has a spacing of 0.2 (Section 12.5). Options: a denser design near the valley, narrower bounds, or observables with a wider basin.
3. Then repeat the P3 two-parameter gate and P9 with both parameters.

Reason: in Phase P3 the 2-D landscape had a false local minimum at $\hat\theta \approx (0.51, 0.28)$ next to a narrow true basin. Exact gradients do not remove a false minimum. One parameter separates the gradient work from the global-search problem.

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

`fbpic/autodiff/driver.py` runs one pass of `step` with $K$ tangents, one per parameter (D25): $K = 1$ in Stage A and $K = 2$ in Stage B (Section 2.6).

1. Build the tangent seeds (Section 4.8).
2. Run $N_t$ steps. Keep only the current primal state, the current tangents, and the per-step stash.
3. Evaluate the objective and its analytic state gradient $g_s$ at $s_{N_t}$. Contract it with each final tangent.

Driver API:

```python
result = value_and_grad(theta, objective, config)
# result.J: float
# result.grad: dJ/dtheta, shape (K,), float64, host
# result.grad_hat: dJ/dtheta_hat = diag(theta_hi - theta_lo) @ grad
# result.info: GPU and host memory peaks, timings, per-step norms of each tangent
# objective(state) -> (J, g_s); the Section 2.4 loss or a Section 6.2 functional
# objective may instead return (r_w, dr_w/ds): residuals and their state gradients,
# so that result.jacobian = dr_w/dtheta_hat feeds the Section 2.5 driver

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
| L6 | Twin recovery | From the design-based starts and the 5 Latin-hypercube starts (Section 2.5), recover $\hat\theta^\ast$ within 0.02 in every free component on T0′ and T0: $\hat E_L$ in Stage A, both components in Stage B. On T0′, also report convergence diagnostics that are free of model bias: the spread of the multistart final points ($\le 2\cdot10^{-3}$), the final $\lVert\hat g\rVert_\infty$, and $\hat\theta - \hat\theta^\ast$ in units of $\sqrt{\mathrm{diag}\,G^{-1}}$. On T1, report the offset. Targets are noise-free (D22). |

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
10. Slices: evaluate the loss on 13 points over $\hat\theta^\ast_i \pm 0.3$ for each parameter (26 runs). Record the basin shape and any plateaus or jumps.
11. Recovery gate prerequisites (Sections 2.4 and 2.5):
    - Switch to the loss of Section 2.4. Recompute $\mathrm{cond}(G)$ and $\sqrt{\mathrm{diag}\,G^{-1}}$ at $\theta^\ast$ with the new observables and the Phase P1 method, and check them against the P1 limits.
    - Use the FD steps $h = 0.02$ for $E_L$ and $h = 0.05$ for $z_f$, with one-sided stencils within $h$ of a bound (Section 2.5).
12. Generate the T0′ target (Section 2.2) and confirm that $J(\theta^\ast)$ on T0′ is at round-off level. Gate part (a), landscape:
    - $z_f$ edge test: at the point where L-BFGS-B stopped on the $z_f$ bound, evaluate the one-sided FD derivative in $z_f$ at $h = 0.02$ and $h = 0.1$, with the old and the new loss.
    - Evaluate a 25-point grid or Sobol design over $[0, 1]^2$. The design runs do not depend on the target, so evaluate the loss on T0′ and on T0 from the same runs.
    - Pass if the best design point lies in the $\theta^\ast$ basin, and the loss of every other basin is well above the loss floor at $\theta^\ast$.
13. Gate part (b), recovery on T0′: first take fine slices at $\hat\theta^\ast_i \pm 0.05$ on T0′ and confirm that the minimum is at $\hat\theta^\ast$. Then run `least_squares(method="dogbox")` with the batched FD Jacobian, which reuses the cached center run. Start from the 2 or 3 best well-separated design points. Use the stopping rule of the note (step norm, cost reduction, evaluation budget). Pass if the best result is within 0.02 of $\hat\theta^\ast$ in every component.
14. Report the final loss, the parameter error, $G^{-1}$ at the solution, and a run on T0′ from the optimizer start farthest from $\hat\theta^\ast$. Also run the T0 recovery from the best design start, and report its offset from $\hat\theta^\ast$ as the model bias of `p_nz`. The farthest-start run and the T0 recovery are reported only; they are not part of the gate.

15. Stage A gate (Section 2.6): the one-parameter recovery of $\hat E_L$ with $\hat z_f = \hat z_f^\ast$ (`python fd_recovery.py --target T0p one_E_L`). Use a 10-point design at 0.05, 0.15, ..., 0.95, then dogbox from the 2 best design points that are at least 0.2 apart. Pass if the best result is within 0.02 of $\hat E_L^\ast$. Also run $z_f$ alone, and the T0 recoveries of both as the bias report.

Acceptance (gate): the Stage A gate (step 15) passes on T0′. The two-parameter gate of steps 12 and 13 belongs to Stage B. Its first result (Section 12.5) is recorded: the 25-point design does not resolve the basin of $\theta^\ast$. The previous text of this acceptance was: parts (a) and (b) pass with the T0′ target and the Section 2.4 loss. If the gate fails, stop and ask the user before Phase P4. Options: a different observable, narrower bounds, a longer plasma, or one parameter.

The first version of this gate (one L-BFGS-B run from the farthest start, raw moments) failed in its design, not in the problem: see Section 12.5.

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

Stage A runs this phase for $E_L$ alone, with $\hat z_f = \hat z_f^\ast$ (Section 2.6). Steps 1 and 3 give the Stage A form first. Stage B uses the two-parameter form in the same steps.

0. Confirm the recovery branch chosen at the end of Phase P7 (Section 6.4). Label every result with it.
1. Write `inversion/recover.py` with the Section 2.5 driver.
   - Stage A: the 10-point 1-D design in $\hat E_L$ of Phase P3 step 15, then `least_squares(method="dogbox")` on $r_w$ from the 2 best design points that are at least 0.2 apart, bounds $[0, 1]$. The JVP Jacobian (1 primal run and 1 tangent) replaces the FD Jacobian after it passes the JVP gate of Section 2.5, item 5.
   - Stage B: the 25-point design, then dogbox from the 2 or 3 best design points, with 1 primal run and 2 tangents, after the global-search decision of Section 2.6.
2. Generate the T1 and T2 targets (Section 2.2). T0′ and T0 exist from Phase P3.
3. Run L6 noise-free on Twin-S or on the reduced Twin-S, as Phase P8 step 3 decides. Stage A starts: the design-based starts and the $\hat E_L$ coordinates of the 5 starts of Section 3.4, with $\hat z_f = \hat z_f^\ast$. Stage B starts: the design-based starts and the 5 Latin-hypercube starts. T0′ is the tight convergence check, with the diagnostics of the L6 row. T0 must pass within 0.02. Run the same recoveries on T1 and report its offset from $\hat\theta^\ast$ as model bias. Record iterations, gradient evaluations, and wall time. Compare them with the FD recovery of Phase P3. Report T2.

Acceptance: L6 passes for T0′ and T0. The T1 offset is recorded and reported to the user, together with the emission-delay results of Phase P8.

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
| D14 | L-BFGS-B on the 2 normalized parameters, 5 Latin-hypercube starts. | Superseded by D26 on 2026-10-30 |
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
| D26 | Bounded trust-region `least_squares(method="dogbox")` on the whitened residuals, from a 25-point global design, with multistart (Section 2.5). Supersedes D14. | User, 2026-10-30 |
| D27 | Add the T0′ control target with exactly the inversion settings. Gate (b) and L6 test convergence on T0′; T0 and T1 recoveries report model bias. Drop ECNoise (fixed FD steps) and the bootstrap term of $\Sigma$. Drop $\langle y\rangle$ and $\langle u_y\rangle$ from the loss (11 observables). | User, 2026-10-30 |
| D28 | Stage A: take one parameter ($E_L$, $z_f$ fixed at $z_f^\ast$) through Phases P3 to P9 with one tangent, then add $z_f$ in Stage B (Section 2.6). | User, 2026-10-30 |

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

An adversarial plan review on 2026-10-30 (Fable) had 35 candidate findings: 33 survived verification, 0 were refuted, 2 were partly refuted, and 8 were downgraded. Finding R1 (the T0 target mixes `p_nz` model bias with convergence) is applied in this revision as the T0′ control target (D27). The other fixes go into a separate plan commit after Phase P3.

Phase numbers in the review records above refer to the plan at the time of each review.

### 12.2 Phase P0. Baseline

Run on 2026-10-30 at commit `74c21bd`, branch `snidhan/dev-autodiff`.

| Item | Value |
|---|---|
| Environment | Conda `fbpic-warp`: Warp 1.18.0, CuPy 14.2.0, NumPy 2.2.6, SciPy 1.17.1 |
| GPU | Physical GPU 1, NVIDIA RTX 6000 Ada Generation, 49140 MiB total, 48496 MiB free at start |
| Device selection | `CUDA_VISIBLE_DEVICES=1`. The process sees the GPU as `cuda:0`, so commit `8f27245` is not needed. |
| Host memory | 125 GiB total, 115 GiB available |
| Sandbox note | GPU access needs a process outside the agent sandbox. |

Command, from `lpa/FBPIC_Warp`:

```bash
export CUDA_VISIBLE_DEVICES=1 OPENPMD_VERIFY_HOMOGENEOUS_EXTENTS=0 PYTHONPATH="$PWD"
for f in tests/test_autodiff_*.py tests/test_fused_particle_operations.py tests/test_gpu_end_to_end.py; do python -m pytest "$f" -q; done
```

| Test file | Result |
|---|---|
| `test_autodiff_arrays.py` | 4 passed |
| `test_autodiff_deposition.py` | 17 passed |
| `test_autodiff_fields.py` | 94 passed |
| `test_autodiff_particles.py` | 17 passed |
| `test_fused_particle_operations.py` | 139 passed |
| `test_gpu_end_to_end.py` | 2 passed |

L0 passes. The autodiff files hold 132 tests, not the 122 of Section 1.5. Tests were added after commit `24fb4c7`. This difference does not change the plan.

### 12.3 Phase P1. Twin configurations, trapping, and identifiability

Run on 2026-10-30 on physical GPU 1 (RTX 6000 Ada), code on top of commit `9a65f19`. The P1 commit holds the code and the results.

Files: `inversion/twin_config.py`, `inversion/observable.py`, `inversion/forward.py`, `inversion/identifiability.py`, `inversion/verify_twin_p1.py`, `inversion/configs/twin_s.json`, `inversion/configs/twin_xs.json`, and `inversion/results/p1_twin_s.json`.

Frozen Twin-S physics (`configs/twin_s.json`):

| Item | Value |
|---|---|
| Grid | 800 x 50 cells, window [-30, 10] µm, rmax 20 µm, $\Delta t = \Delta z / c$, $N_m = 2$, `n_order=-1`, `n_guard=64`, `exchange_period=14`, float64 Hankel |
| Laser | $\lambda_0 = 0.8$ µm, $w_0 = 5$ µm, $\tau = 16$ fs (FBPIC field duration), $z_0 = -10$ µm, linear polarization along x; $z_R = 98.2$ µm |
| $E_L$ | nominal 0.27 J ($a_0 = 4.0$), bounds [0.162, 0.378] J ($a_0$ 3.10 to 4.73). The ratios 0.6 and 1.4 match production (1.5, 2.5, 3.5 J). |
| $z_f$ | nominal 90.0 µm ($z_c + 0.53 z_R$), bounds [46.8, 133.2] µm ($\pm 0.44 z_R$) |
| Density | $n_{e0} = 8\times10^{24}$ m$^{-3}$ peak electron density, N fraction 0.03 of the atoms, $z_c = 38$ µm, $\beta = 2$, $F_L = F_R = 28$ µm, indicator [10, 66] µm, $S(z_{lo}) = S(z_{hi}) = 0.0625$. The window injects all plasma (`start = injected`). |
| Plasma length | FWHM $= 0.29\, z_R$, indicator extent $= 0.57\, z_R$ |
| Particles | `p_nz=1`, `p_nr=2`, `p_nt=8`, every species tracked |
| $N_t$ | 2000. The left window edge is at 69.95 µm, past $z_{hi}$. |
| Seeds | NumPy 0, ionization Philox 20261030 stream 1, Latin hypercube 2045 |
| $\hat\theta^\ast$ | (0.6, 0.4). This is a planner choice: $\theta^\ast$ = (0.2916 J, 81.36 µm). |
| Selection | $u_c = 6.246$, $\delta_u = 0.0385$ ($u_{lo} = 6.075$, $u_{hi} = 6.400$; low-energy peak at 1.0, trapped peak at 11.2) |
| Optimizer starts | (0.273, 0.503), (0.164, 0.323), (0.750, 0.723), (0.690, 0.123), (0.502, 0.795). The farthest from $\hat\theta^\ast$ is (0.164, 0.323), at 0.436 in the infinity norm. |

Twin-XS (`configs/twin_xs.json`): 96 x 32 cells, window [-10, 10] µm, rmax 12 µm, the same laser with $z_0 = -3$ µm, preloaded density with $z_c = 8$ µm, $F = 8$ µm, indicator [0, 16] µm, $N_t = 300$. The laser peak reaches $S = 0.1$ after 18 steps ($n_c \le 20$). Twin-XS traps no beam ($u_z \le 4.6$). Its selection ($u_c = 2.0$, $\delta_u = 0.25$) is a convention for the operator and closure checks.

Interpretation choices:

1. The moments are the 6 first moments and the 6 raw diagonal second moments of $\xi$, so $K = 12$. The plan does not name the second moments; $s_k$ (standard deviation, and its square) implies diagonal moments.
2. Step 7 needs 16 runs for 2 parameters (2 parameters x 2 signs x 4 seeds), not 32. The 32 in the plan dates from the 4-parameter scope. I also ran $\hat\theta^\ast$ with the same 4 seeds for the seed variance and $v_{model}$.
3. The $\Sigma$ floor of $\log Q$ uses the scale 1, so the floor is $0.02^2$.

Commands, from `lpa/FBPIC_Warp/inversion`, with `CUDA_VISIBLE_DEVICES=1`:

```bash
python verify_twin_p1.py twin_xs
python verify_twin_p1.py twin_s
python identifiability.py selection trapping sensitivity report
```

Each production run is one `Simulation.step(N_t)` call. A first attempt split the runs into 100-step calls. Each call forces a particle exchange at its first iteration, which changes the physics, so that attempt was discarded.

Step 4, aliasing, $\max|\rho_1| / \max|\rho_0|$ per species (limit $10^{-10}$):

| Size | When | Electrons | He | N |
|---|---|---|---|---|
| Twin-XS | t = 0, preloaded | 1.9e-15 | 1.4e-15 | 1.3e-15 |
| Twin-S | 300 steps, laser fields set to zero, injected plasma | 2.4e-15 | 2.5e-15 | 3.0e-15 |

Step 5, weights: all weights are positive. At the corners (0, 0) and (1, 1), the weights of equal tracker IDs are identical (maximum difference 0.0), and the attributed ID counts are equal (Twin-XS after 60 steps; Twin-S after 1300 steps, 896000 IDs per species).

Step 6, trapping (stochastic ionization, limits: at least 1000 macroparticles with $\sigma > 0.5$ and $Q \ge 0.1\,Q_{nom}$):

| Point | $\hat\theta$ | $n(\sigma > 0.5)$ | $Q$ [pC] | $Q/Q_{nom}$ | Pass |
|---|---|---|---|---|---|
| nominal | (0.5, 0.5) | 40612 | 15.51 | 1.000 | yes |
| corner | (0, 0) | 35607 | 9.40 | 0.606 | yes |
| corner | (1, 0) | 32536 | 13.30 | 0.858 | yes |
| corner | (0, 1) | 20183 | 4.24 | 0.274 | yes |
| corner | (1, 1) | 69686 | 28.01 | 1.807 | yes |
| start 0 | (0.273, 0.503) | 46731 | 14.77 | 0.952 | yes |
| start 1 | (0.164, 0.323) | 39825 | 12.18 | 0.786 | yes |
| start 2 | (0.750, 0.723) | 50722 | 20.21 | 1.303 | yes |
| start 3 | (0.690, 0.123) | 28162 | 12.67 | 0.817 | yes |
| start 4 | (0.502, 0.795) | 62162 | 20.32 | 1.310 | yes |

The bounds stay unchanged.

Step 7, identifiability ($h = 0.05$, Philox seeds 101 to 104, T2 truth with NumPy seed 7, Philox seed 999, and `p_nz = 2`):

| Quantity | Value |
|---|---|
| $Q_y$ (seed mean at $\theta^\ast$) | 15.06 pC |
| Eigenvalues of $G$ | 2.215e4, 4.827e4 |
| $\mathrm{cond}(G)$ | 2.18 (limit $10^4$) |
| Weakest eigenvector | (-0.659, -0.752) in ($\hat E_L$, $\hat z_f$) |
| $\sqrt{\mathrm{diag}\, G^{-1}}$ | 0.0056, 0.0059 (limit 0.02) |

The gate passes. The strongest normalized sensitivities $S/\sqrt{\Sigma}$ are $\langle u_z^2\rangle$ (143.6) and $\langle\zeta^2\rangle$ (-89.4) for $E_L$, and $\langle y^2\rangle$ (-120.7) and $\langle u_y^2\rangle$ (-83.2) for $z_f$. Section `identifiability` of `results/p1_twin_s.json` holds $S$, its SE, $\Sigma$, and $G$.

Step 8, Twin-S at the nominal $\theta$ (stochastic):

| Item | Value |
|---|---|
| Selected charge | 15.5 pC, 40612 macroparticles with $\sigma > 0.5$ |
| Selected energy | weighted mean 5.8 MeV, standard deviation 1.8 MeV, maximum 12.6 MeV |
| Particles at $N_t$ | 165717 `elec_from_N`; 44387 electrons, 24000 He, 24000 N (in the guard region) |
| Run time | 22.5 s for 2000 steps (11 ms per step) |
| Device memory | CUDA mempool high-water mark 0.49 GB used, 0.64 GB reserved; 1.13 GB device memory in use at the end, including the context |

Acceptance: steps 4 to 7 pass. The config files are frozen.

### 12.4 Phase P2. Functional stepper without ionization

Run on 2026-10-30 on physical GPU 1 (RTX 6000 Ada), code on top of commit `227304f`.

Files: `fbpic/autodiff/state.py` (`StepState`, `TangentState`, `StepDecisions`, `EVArrays`), `fbpic/autodiff/stepper.py` (`begin`, `step`, `stable_sort`), and `tests/test_autodiff_stepper.py` (L2). No production module changed.

Design:

1. The production `Simulation` owns the field arrays, the particle arrays, and the window, injection, tracking, and clock scalars. `StepState` refers to it and adds the EV arrays and the exchange bookkeeping. `step` updates the state in place and returns the same object.
2. `step` calls the production methods in the order of `main.py:533-687`. Before each deposit, `stable_sort` computes the production key with `get_cell_idx_per_particle`, sorts it with `cupy.argsort(kind="stable")`, rebuilds the prefix sum with the production kernels, and permutes every particle array. The production sort is also a stable radix sort on the same key, so both give the same permutation.
3. `begin` performs the prologue and the first exchange with its `rho_prev` deposit. A `step` at an exchange iteration skips the exchange if `begin` already did it at that iteration, as production does at `i_step == 0`.
4. Production removal and injection do not expose a permutation. After each exchange, the EV arrays are realigned by tracker ID (Phase P3).
5. `StepDecisions` records `n_move`, the injected count per species at each exchange, and the emitting parent IDs.

Command, from `lpa/FBPIC_Warp`, with `CUDA_VISIBLE_DEVICES=1`:

```bash
python -m pytest tests/test_autodiff_stepper.py -q -s
```

L2 results (pre-ionized N, nominal $\theta$). The ratio is the stepper difference divided by $\max(2\cdot10^{-11}, 10\, d_{AA})$; it must be at most 1.

| Case | Largest ratio | Array | Stepper difference | $d_{AA}$ |
|---|---|---|---|---|
| Twin-XS, 300 steps | 0.103 | electrons x | 5.5e-5 | 5.4e-5 |
| Twin-S, 300 steps | 0.149 | spectral mode 0 `rho_prev` | 3.3e-12 | 2.2e-12 |
| Twin-S, 1000 steps | 0.149 | spectral mode 0 Bz | 1.8e-10 | 1.2e-10 |

These numbers are from the run on the committed code. An earlier run of the same test, before formatting and the EV hooks, gave largest ratios of 0.198, 0.123, and 0.143.

Particle counts, tracker IDs, the iteration, and the window position are equal in all cases. L2 passes.

Deviation from L2 as written. $d_{AA}$ is the largest difference over the 3 pairs of 3 production runs, not the difference of one pair. On Twin-XS ($a_0 = 4$ at $\Delta z = 0.21$ µm), one electron bifurcates between production runs. Two production runs can then differ by 7.7e-4 in the infinity norm of electron uy after 300 steps, and a different pair by 1e-5. One pair therefore underestimates the run-to-run spread. Evidence, from 7 runs in one process:

1. After 50 steps, every pair (production with production, stepper with production, stepper with stepper) differs by about 1e-7 in electron uy and by about 1e-11 in the spectral fields.
2. After 300 steps, one production run and two stepper runs take the same branch (7.7e-4 from the others), and the other runs agree within 1e-5. The stepper runs are distributed like the production runs.

The suspect assumption is that one production pair bounds the run-to-run noise of Twin-XS. Twin-S stays at rounding level (about 1e-10) and passes with either definition.

The 1000-step Twin-S case is an addition. At 300 steps the laser has not reached the Twin-S plasma (it reaches $z_{lo}$ after about 400 steps), so the 300-step case tests the field solver and injection only.

Acceptance: L2 passes on Twin-XS and Twin-S. Production code is unchanged.

### 12.5 Phase P3. Forward EV model, minimal loss, and FD recovery gate

Run on 2026-10-30 on physical GPU 1 (RTX 6000 Ada), code on top of commit `1d9272e`.

Files: `fbpic/autodiff/ionization.py` (EV kernels `ev_ionize` and `copy_ev_electrons`, state management), `push_ion_momenta_ev` in `fbpic/autodiff/particles.py`, the EV hooks of `fbpic/autodiff/stepper.py`, and in `inversion/`: `verify_ev_open_loop.py`, `verify_ev_invariants.py`, `ev_runs.py`, `make_target.py`, `verify_loss_minimal.py`, `verify_ev_closure.py`, `fd_recovery.py`, `ev_point.py`, `target_covariance.py`, `notes/optimizer-literature-2026-10-30.md`.

Implementation notes:

1. Warp 1.18 has no `expm1`. `ev_ionize` computes $p_\ell = -\mathrm{expm1}(-x_\ell)$ with Kahan's form $(u-1)\,x/\log u$, $u = e^{x}$, which is accurate to a few ulp. The ADK coefficients come from the production `Ionizer.initialize_ADK_parameters`.
2. Emission slots are the exclusive `cumsum` of the emission flags in parent order. The new electrons take their tracker IDs from the production `generate_new_ids`, so electron $k$ of a step has the parent `emitted_parent_ids[k]`.
3. For each deposit, the N ions temporarily take the weight $w\bar Z - W$ and the charge $e$. The ion push uses the charge $e\,(w\bar Z - W)/w$.
4. After each production exchange, the EV arrays are realigned by tracker ID. Injected ions start at $P = (1, 0, 0)$, $W = 0$.

Step 1: implemented (see the files above).

Step 2, open loop (`python verify_ev_open_loop.py`): 2400 ions in 60 field groups, 200 steps, Twin-S $\Delta t$, 32 Philox seeds. Over the set, $p_5$ covers 1.0e-3 to 0.90 and $p_6$ covers 1.1e-5 to 0.30.

| Quantity | EV | Seed mean | SE | (EV - mean)/SE |
|---|---|---|---|---|
| $\sum w P_5$ | 1.5606e7 | 1.5518e7 | 8.9e4 | +0.99 |
| $\sum w P_6$ | 4.2095e7 | 4.2029e7 | 1.4e5 | +0.48 |
| $\sum w P_7$ | 6.7963e7 | 6.8117e7 | 1.1e5 | -1.42 |
| Birth weight (emitted + pending) | 1.78020e8 | 1.78262e8 | 1.4e5 | -1.69 |

Pass (limit 3 SE). Also: $\max|P_5 + P_6 + P_7 - 1| = 1.6\cdot10^{-15}$, and $\max W/w = 8.9\cdot10^{-5} \le \epsilon$.

Steps 3 and 4 (`python verify_ev_invariants.py`, Twin-S at $\theta^\ast$, 2000 steps):

| Check | Result | Limit |
|---|---|---|
| Charge before and after the EV call, 8 steps (2 without emission, 6 with 10447 to 12893 emissions) | at most 7.0e-16 relative | 1e-12 |
| Closed cohort: the 76000 ions of the first injection (iteration 14) and their electrons, at iterations 450 to 750; no cohort particle removed | $\max|P_5+P_6+P_7-1| = 2.2\cdot10^{-15}$; weight balance at most 8.3e-16 relative | 1e-14; 1e-12 |
| Open window with removal ledgers, iterations 450 to 2000 | emitted-weight balance at most 6.0e-16; total balance at most 8.9e-16 | 1e-12 |
| Populations follow IDs through 2 sorts and 2 exchanges (11200 ions injected at each, 11184 removed at iteration 1008) | equal; injected ions start at (1, 0, 0, 0) | exact |

Pass. The cohort emitted 1.24M electrons from 76000 ions, about 16 per ion: with $\epsilon = 10^{-4}$, an ion emits again each time its pending weight passes $\epsilon w$.

Step 5, EV runs of Twin-S (`python ev_runs.py twin_s <point>`, $\epsilon = 10^{-4}$, $K_e = 1$):

| Point | Births | `elec_from_N` at $N_t$ | $Q$ EV [pC] | $Q$ stochastic [pC] (P1) | Run time | Mempool peak used / reserved |
|---|---|---|---|---|---|---|
| nominal (0.5, 0.5) | 13.16M | 6.66M | 15.50 | 15.51 | 88.8 s | 2.62 / 3.96 GB |
| corner (0, 0) | 11.88M | 6.43M | 9.45 | 9.40 | 87.6 s | 2.50 / 3.59 GB |
| corner (1, 1) | 16.30M | 7.52M | 28.03 | 28.01 | 101.4 s | 3.02 / 4.46 GB |

The EV births are about 6 times the loaded background (2.69M macroparticles over the run), not up to 1.7 times as Section 8 estimated. The stochastic run leaves 0.17M `elec_from_N`; EV leaves 40 times more. An EV run takes about 4 times as long as a production stochastic run (22 s). Memory is not a constraint: the peak stays below 5 GB of 48 GB.

Step 6, immediate emission ($\epsilon = 0$):

| Case | Births | `elec_from_N` at $N_t$ | Run time | Mempool peak used / reserved |
|---|---|---|---|---|
| Twin-XS, nominal | 2.53M | 0.49M | 6.7 s | 0.64 / 0.77 GB |
| Reduced Twin-S (N only in $z_c \pm 2$ µm), nominal | 13.75M | 5.97M | 224 s | 3.27 / 4.97 GB |

Both fit in memory. The reduced Twin-S keeps the plasma neutral: its electron density drops by the nitrogen share outside the nitrogen region. Its selected charge is 4.81 pC. The device-memory reading of that run (13.3 GB) includes a concurrent process, so the mempool marks are the relevant numbers.

Step 7, minimal loss (`python verify_loss_minimal.py`, host only): on an empty selection the loss equals the closed form $\alpha_Q(\log Q_f - \log(Q_y+Q_f))^2 + \alpha_M\sum_k m_{k,y}^2/s_k^2$ (48.297, relative error below 1e-12). On one particle with $\sigma = 10^{-6}$, the loss (48.297) and the moments are finite. Pass. `make_target.py` writes the target with the keys of Section 3.4; `load_target` refuses a target whose `config_sha256` or $N_t$ does not match.

Step 8, mean-charge closure (`python verify_ev_closure.py <size>`). The SE is that of the mean of 16 Philox seeds of production stochastic ionization. Twin-XS runs at the nominal $\theta$ against a target made from its mobile EV beam; Twin-S runs at $\theta^\ast$ against T0 and reuses the P1 seeds 101 to 104.

| Quantity | Twin-XS: (mobile - immobile)/SE | Twin-XS: (EV mobile - seed mean)/SE | Twin-XS: (EV immobile - seed mean)/SE |
|---|---|---|---|
| Q_pC | -3.45 | -0.23 | +3.22 |
| <x> | +0.64 | -0.48 | -1.12 |
| <ux> | +0.52 | -0.26 | -0.78 |
| <y> | -0.62 | -1.90 | -1.28 |
| <uy> | -0.57 | -2.03 | -1.45 |
| <zeta> | +3.15 | +0.74 | -2.41 |
| <uz> | +4.34 | +0.66 | -3.68 |
| <x^2> | +1.70 | -1.95 | -3.65 |
| <ux^2> | +2.37 | -2.58 | -4.95 |
| <y^2> | -0.68 | +0.91 | +1.60 |
| <uy^2> | -0.42 | +0.78 | +1.21 |
| <zeta^2> | -2.75 | -0.65 | +2.11 |
| <uz^2> | +4.07 | +0.54 | -3.53 |
| loss | -3.02 | -3.74 | -0.72 |

| Quantity | Twin-S: (mobile - immobile)/SE | Twin-S: (EV mobile - seed mean)/SE | Twin-S: (EV immobile - seed mean)/SE |
|---|---|---|---|
| Q_pC | -1.40 | +0.87 | +2.27 |
| <x> | -0.12 | +0.02 | +0.14 |
| <ux> | -0.25 | -0.35 | -0.09 |
| <y> | +0.33 | -0.81 | -1.14 |
| <uy> | +0.06 | -0.61 | -0.67 |
| <zeta> | +1.09 | +0.49 | -0.59 |
| <uz> | +0.02 | +0.88 | +0.86 |
| <x^2> | -0.28 | -1.65 | -1.37 |
| <ux^2> | -0.07 | -0.62 | -0.55 |
| <y^2> | -0.51 | +1.09 | +1.60 |
| <uy^2> | -0.10 | +0.37 | +0.47 |
| <zeta^2> | +1.36 | +0.52 | -0.84 |
| <uz^2> | -0.10 | +0.99 | +1.09 |
| loss | -1.40 | -1.83 | -0.43 |

Result: Twin-S passes (largest difference 1.40 SE). Twin-XS fails the 2 SE rule: mobile and immobile ions differ by up to 4.34 SE (in $\langle u_z\rangle$, $\langle u_z^2\rangle$, $Q$, $\langle\zeta\rangle$, and the loss), with relative differences of 1 to 3%. The loss row of Twin-XS compares against a target made from the mobile EV beam, so its EV-mobile entry is not meaningful.

Interpretation. The mobile-against-immobile difference measures the effect of ion motion, not the error of the mean-charge closure. On Twin-XS, EV with mobile ions agrees with the stochastic seed mean in $Q$, $\langle\zeta\rangle$, $\langle u_z\rangle$, and $\langle u_z^2\rangle$ within 0.74 SE; EV with immobile ions does not (2.4 to 4.9 SE). Two Twin-XS moments of EV-mobile, $\langle u_y\rangle$ and $\langle u_x^2\rangle$, are 2.0 and 2.6 SE from the seed mean. The plan asks the user to decide when a difference exceeds 2 SE. Status: reported to the user on 2026-10-30. The user judged that the FD recovery gate does not depend on this result, because the T0 target and the model use the same EV closure. The closure question stays open for the T2 recovery of Phase P9.

Step 9, T0 target (`python make_target.py twin_s T0`): `inversion/targets/twin_s_T0_Nt2000.npz`. EV at $\theta^\ast$ with NumPy seed 7 and `p_nz = 2`. $Q_y = 14.80$ pC, $Q_f = 1.48\cdot10^{-2}$ pC, 3.43M macroparticles with $\sigma > 0.5$, 25.3M births, 283 s, mempool peak 5.17 GB used and 7.42 GB reserved. The target was made from the working tree on top of commit `1d9272e`; its `git_hash` key records `1d9272e`, which does not yet hold the EV code. The P3 commit holds that code unchanged.

Noise floor (`python fd_recovery.py noise`): 3 EV runs at $\theta^\ast$ give $J = 1.5116\cdot10^{-3}$ with a standard deviation of $2.1\cdot10^{-12}$. The EV twin is deterministic to round-off; the nonzero $J$ at $\theta^\ast$ is the model error of `p_nz = 1` against the `p_nz = 2` target.

Step 10, slices (`python fd_recovery.py slices`, 26 runs, minimal loss). $J$ at $\hat\theta^\ast_i + d$, $d = -0.30, -0.25, \dots, +0.30$:

| Loss | Coordinate | $J$ |
|---|---|---|
| Raw second moments | $E_L$ | 2.299, 1.541, 0.917, 0.415, 0.227, 0.019, **0.002**, 0.050, 0.206, 0.483, 0.708, 1.115, 1.586 |
| Raw second moments | $z_f$ | 0.076, 0.037, 0.051, 0.060, 0.016, 0.019, **0.002**, 0.008, 0.022, 0.064, 0.105, 0.224, 0.376 |
| Central second moments | $E_L$ | 0.107, 0.077, 0.041, 0.021, 0.017, **0.0008**, 0.0013, 0.0015, 0.006, 0.018, 0.024, 0.047, 0.060 |
| Central second moments | $z_f$ | 0.065, 0.031, 0.042, 0.032, 0.008, 0.004, **0.0013**, 0.008, 0.025, 0.044, 0.060, 0.117, 0.190 |

Both slices have one basin around $\hat\theta^\ast$. The $z_f$ slice is flat and not monotone below $\hat z_f = 0.25$ with both losses. With central moments, the $E_L$ minimum moves to $\hat E_L \approx 0.55$: the model error of `p_nz = 1` shifts the minimum. The raw-moment loss is larger in $E_L$ because the raw terms count the means of $\zeta$ and $u_z$ again. $Q$ rises monotonically from 12.6 to 19.3 pC over the $z_f$ slice and from 14.0 to 15.6 pC over the $E_L$ slice.

First gate attempt (plan as written: L-BFGS-B, raw-moment loss, central FD with $h = 0.02$, from the farthest start). It failed in its design. The evaluations were:

| Evaluation | $\hat\theta$ | $J$ | FD gradient |
|---|---|---|---|
| 1 | (0.164, 0.323) | 5.274 | (-37.8, +4.5) |
| 2 | (1, 0) | 1.433 | (+6.4, +1.3) |
| 3 | (0.889, 0) | 0.915 | (+5.8, +2.1) |
| 4 | (0, 0) | 10.37 | (-61.3, +0.2) |
| 5 | (0.603, 0) | 0.072 | (+0.67, +0.29) |
| 6 | (0, 0) | 10.37 | (-61.3, +0.2) |
| 7 | (0.521, 0) | 0.157 | (-2.08, -0.39) |

The first step went to the corner (1, 0). L-BFGS-B scales its first step by the gradient only, and the projection onto the box moved $z_f$ to its bound. A secant pair from the flat region then sent the step to the opposite corner. At $\hat z_f = 0$ the $z_f$ derivative is positive, so the bounded method stays at the bound. The run was stopped after 7 evaluations (`results/p3_fd_recovery.json`, key `recovery_lbfgsb_stopped`). A trust-region `least_squares` run (method `trf`, central moments, not whitened) from the same start reached (0.432, 0.231) with $J = 0.023$ after 3 accepted steps; it was stopped when the user approved the new gate (key `recovery_trf_central_stopped`). A literature review (`inversion/notes/optimizer-literature-2026-10-30.md`) gave the optimizer of Section 2.5 and the loss of Section 2.4 (D26).

Step 11, prerequisites of the new gate:

1. Weights (`python target_covariance.py run <point>` for the nominal point and the 4 corners, then `build`). $\Sigma = \mathrm{diag}(\Delta^2)$, with $\Delta$ the RMS over the 5 points of the truth-setting run minus the model run (Section 2.4). The 4 corner runs at `p_nz = 2` take 345 to 351 s each. The RMS model error per observable:

| Observable | RMS $\Delta$ | Observable | RMS $\Delta$ |
|---|---|---|---|
| $\log(Q+Q_f)$ | 4.46e-2 | $\log\sigma_x$ | 2.71e-2 |
| $\langle x\rangle$ [m] | 6.52e-8 | $\log\sigma_{u_x}$ | 2.11e-2 |
| $\langle u_x\rangle$ | 1.74e-2 | $\log\sigma_y$ | 4.47e-2 |
| $\langle \zeta\rangle$ [m] | 2.65e-8 | $\log\sigma_{u_y}$ | 3.28e-2 |
| $\langle u_z\rangle$ | 1.83e-1 | $\log\sigma_\zeta$ | 4.12e-2 |
| | | $\log(\sigma_{u_z}/\langle u_z\rangle)$ | 1.46e-2 |

   The first build used the nominal point alone, plus a Poisson bootstrap ($B = 256$) of the T0 rerun. With it, $J(\theta^\ast) = 1128$ on T0, and 1124 of that came from one residual: the log relative energy spread, with a whitened value of $-33.5$. Its $\Delta$ at the nominal point was $3.8\cdot10^{-4}$, but the model error at $\theta^\ast$ is $2.5\cdot10^{-2}$. The bootstrap term was 3 to 30 times smaller than $\Delta$ for 10 of the 13 observables. The user removed the bootstrap and the ECNoise step (D27). With the 5-point $\Delta$, $J(\theta^\ast) = 6.8$ on T0, and every whitened residual is within $\pm 1.7$. The T0 rerun reproduces the target observables to $2\cdot10^{-11}$ (relative), so the code of this commit reproduces T0. File: `inversion/targets/twin_s_T0_Nt2000_whitening.npz`.
2. Identifiability with the Section 2.4 observables (`python identifiability.py report_loss`, the P1 runs and method):

| Weights | cond(G) | $\sqrt{\mathrm{diag}\,G^{-1}}$ ($E_L$, $z_f$) | P1 limits (1e4; 0.02) |
|---|---|---|---|
| Diagonal P1 rule (seed variance plus $(v_{T2} - v)^2$, 2% floor) | 2.21 | (0.0125, 0.0089) | pass |
| Weights $\Sigma$ of item 1 | 1.41 | (0.0123, 0.0110) | pass |

   Both rows use the final 11 observables (item 3).
3. Observables $\langle y\rangle$ and $\langle u_y\rangle$ removed (D27). Their P1 sensitivities are small: at most 0.3 for $\langle y\rangle$ and 1.0 for $\langle u_y\rangle$ in units of the P1 noise per unit $\hat\theta$, against 13 to 59 (in at least one parameter) for the other observables, and their jitter is 1.14 and 0.72 model-error units (RMS of the residual about a quadratic fit along the fine slices). In the 13-observable dogbox run, the rejected trial step at (0.5150, 0.2735) had $J = 43.6$: $\langle y\rangle$ gave 16.2 and $\langle u_y\rangle$ gave 3.3, while the other 11 terms stayed at 24.0. Each of the other 11 observables changes by 1.5 to 9 jitter units over $\Delta\hat\theta = 0.02$ in at least one parameter.

Step 12, T0′ and gate part (a):

1. T0′ (`python make_target.py twin_s T0p`): EV at $\theta^\ast$ with the config seed and `p_nz = 1`, 381 s, $Q = 15.06$ pC. The cached FD run at $\hat\theta^\ast$ matches it to $1.7\cdot10^{-12}$ in $Q$ and to $1.7\cdot10^{-9}$ in the observables, after the moments are rescaled to the $Q_f$ of T0′ (`ev_point.py` forms them with the $Q_f$ of T0; the rescale is exact). $J(\theta^\ast) = 4.7\cdot10^{-15}$ on T0′.
2. $z_f$ edge test at (0.603082, 0), one-sided second-order stencil (`python fd_recovery.py edge`):

| $h$ | $J$ at $\hat z_f = 0, h, 2h$ (new loss, T0) | $dJ/d\hat z_f$ new loss | $dJ/d\hat z_f$ old loss |
|---|---|---|---|
| 0.02 | 116.5, 132.5, 142.7 | +948 | +0.318 |
| 0.1 | 116.5, 110.8, 86.6 | +34.5 | +0.123 |

   At $h = 0.02$ the bound is a local minimum in $z_f$ with both losses. At $h = 0.1$ the values fall from 0.1 to 0.2, but the quadratic stencil still gives a positive slope. The point has $J = 116.5$, against 6.8 at $\theta^\ast$ (T0).
3. 25-point design (`python fd_recovery.py --target T0p design`), $J$ on T0′; rows $\hat E_L$, columns $\hat z_f$ = 0.1, 0.3, 0.5, 0.7, 0.9:

| $\hat E_L$ | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| 0.1 | 1195 | 1986 | 3490 | 4332 | 6103 |
| 0.3 | 237 | 335 | 1296 | 2411 | 3570 |
| 0.5 | 107 | **25.7** | 172 | 1113 | 2131 |
| 0.7 | 190 | 115 | **35.1** | 180 | 1388 |
| 0.9 | 330 | 453 | 275 | 153 | 645 |

   The grid minima are (0.5, 0.3), (0.7, 0.5), and (0.9, 0.7), along one diagonal valley. On T0 the values differ by less than 10% away from $\theta^\ast$. The script scored part (a) as a pass, because the best grid point lies within one grid step of $\theta^\ast$. That score is wrong: step 13 shows that (0.5, 0.3) lies in a different basin.
4. Fine slices on T0′ (`python fd_recovery.py --target T0p fine`), $J$ at offsets $-0.05, -0.04, -0.02, -0.01, 0, +0.01, +0.02, +0.04, +0.05$:
   - $E_L$: 15.1, 21.2, 11.9, 2.7, 0, 1.4, 5.3, 18.4, 26.5.
   - $z_f$: 28.1, 23.7, 7.8, 1.8, 0, 1.2, 4.8, 15.1, 18.3.

   The minimum is at $\hat\theta^\ast$ in both coordinates. The $E_L$ slice has a second dip at $-0.05$.

Step 13, gate part (b) with two parameters, on T0′: failed.

1. 13 observables, FD steps (0.02, 0.05). From (0.5, 0.3): $J$ = 29.9, 27.9, 27.0 at (0.5117, 0.2810), then `xtol` stop after 7 residual and 3 Jacobian evaluations, error (0.088, 0.119). From (0.9, 0.7): stop at (0.898, 0.689) with $J = 157$. The third start was stopped. Cause: the $\langle y\rangle$ and $\langle u_y\rangle$ jitter of step 11, item 3.
2. 11 observables. From (0.5, 0.3): $J$ = 25.7, then 23.73 at (0.5062, 0.2860), a rejected step to (0.5112, 0.2853) with 23.86, then 23.79 at (0.5075, 0.2859). The run was stopped when the user chose Stage A (D28).
3. False minimum. The cached runs near (0.512, 0.281) give $J$ = 23.7 to 23.9 over a patch of size 0.01. $J$ rises on every side: 28.2 and 39.3 at $\hat E_L \mp 0.02$, and 38.2 and 27.8 at $\hat z_f \mp 0.05$. The residuals there do not jitter. $\langle x\rangle$, $\langle u_x\rangle$, and $\langle\zeta\rangle$ change sign at points other than those where the spread terms do, and the spreads have a minimum in $E_L$ near $\theta^\ast$.
4. Basin radius. The radius is the distance from $\hat\theta^\ast$ at which $J$ on T0′ reaches the level of the false minimum, about 24. On the fine slices and the step 10 slices this happens at an offset of about 0.04 in each coordinate: $J$ = 18.4 to 21.2 at $\pm 0.04$ in $E_L$, 15.1 and 23.7 at $\pm 0.04$ in $z_f$, and 32.4 at $(0.6, 0.3)$. The 25-point design has a spacing of 0.2, so no design point lies in this basin.

Step 15, Stage A gate (one parameter, the other at $\hat\theta^\ast$; `python fd_recovery.py --target T0p one_E_L one_z_f`). The 10-point designs on T0′:

- $E_L$: 3249, 1850, 974, 456, 105, **15.1**, 26.5, 126, 291, 346 at 0.05, 0.15, ..., 0.95.
- $z_f$: 192, 70.3, 100, 28.1, **18.3**, 96.8, 307, 971, 1465, 1819.

| Target, parameter | Start | Final | Error | $J$ final | Evaluations (residual, Jacobian) | $\sqrt{G^{-1}}$ at the solution | Wall time |
|---|---|---|---|---|---|---|---|
| T0′, $E_L$ | 0.55 (design) | 0.5453 | 0.055 | 12.6 | 3, 3 | 0.0044 | 723 s |
| T0′, $E_L$ | 0.85 (design) | **0.60000** | **1.5e-5** | 1.6e-5 | 5, 5 | 0.0081 | 1251 s |
| T0′, $E_L$ | 0.164 (farthest, report) | 0.60013 | 1.3e-4 | 5.3e-4 | 6, 6 | 0.0081 | 1645 s |
| T0′, $z_f$ | 0.45 (design) | 0.40009 | 8.8e-5 | 2.7e-4 | 5, 5 | 0.0107 | 1080 s |
| T0′, $z_f$ | 0.15 (design) | 0.40013 | 1.3e-4 | 8.6e-4 | 8, 8 | 0.0107 | 1615 s |
| T0′, $z_f$ | 0.795 (farthest, report) | 0.40007 | 6.8e-5 | 1.7e-4 | 6, 6 | 0.0107 | 1648 s |
| T0, $E_L$ | 0.55 (design) | 0.5541 | 0.046 | 9.5 | 3, 2 | 0.0060 | 360 s |
| T0, $E_L$ | 0.164 (farthest) | 0.5878 | 0.012 | 3.8 | 6, 5 | 0.0074 | 895 s |
| T0, $z_f$ | 0.35 (design) | 0.3843 | 0.016 | 2.6 | 7, 7 | 0.0115 | 1611 s |
| T0, $z_f$ | 0.795 (farthest) | 0.3854 | 0.015 | 2.6 | 7, 6 | 0.0114 | 1431 s |

Result: the Stage A gate passes. The best $E_L$ recovery on T0′ ends at $\hat E_L = 0.60000$ (error $1.5\cdot10^{-5}$, limit 0.02). The $z_f$ recovery also passes from all three starts. The best design point of $E_L$, 0.55, lies in a 1-D false minimum at $\hat E_L \approx 0.545$ ($J = 12.6$); the second design start avoids it, so the multistart is necessary. On T0, the `p_nz` model bias is $-0.012$ in $\hat E_L$ and $-0.015$ to $-0.016$ in $\hat z_f$, within 0.02. The T0 bias run with one design start stopped in the same 1-D false minimum, so the T0 $E_L$ bias comes from the farthest start. At the solution, $\sqrt{G^{-1}}$ = 0.0081 in $\hat E_L$ and 0.0107 in $\hat z_f$ (normalized units), against the 0.02 tolerance. Each gradient-based solve took 5 to 8 Jacobians, about 15 to 27 min on one GPU (3 runs of about 90 s per iteration).

Provenance. The targets and the weights are not committed (`*.npz` is ignored). SHA-256 prefixes: T0 `90ee4adeb6e6`, T0′ `88783bf73c69`, weights `719cbc0e9ce2`. All 311 cached FD runs in `inversion/runs/p3` carry the simulation-code hash `804032cdc3e2`, which is the hash of `forward.py`, `twin_config.py`, `ionization.py`, `stepper.py`, and `particles.py` in this commit. T0′ was made with that code. The T0 rerun with that code reproduces T0 to $2\cdot10^{-11}$, although the `git_hash` key of T0 records `1d9272e`.

Two-parameter conclusion (Stage B input, Section 2.6): the FD gradient and the trust-region solver converge inside a basin. The 2-D failure comes from the global search, which exact gradients do not fix.

Independent code review (a workflow of 3 reviewers with distinct lenses and 1 adversarial verifier per finding, 14 agents): 11 findings, 6 confirmed (all minor after verification), 5 refuted.

| Finding (confirmed) | Disposition |
|---|---|
| `ev_ionize` forms $P_5' = P_5 - P_5 p_5$, not $P_5 e^{-x_5}$ as Section 5.1 requires. The difference is at ulp level in absolute terms, but it is a relative error near saturation, and the P6 tangent will use $e^{-x}$. | Fix after the gate, so that all gate runs use one model. |
| The emission record keeps one device array of parent IDs per emitting step (8 B per birth), which inflates the reported mempool peaks by up to about 4%. | Fix after the gate: make the record optional. |
| The minimal loss uses raw second moments $\langle\xi_k^2\rangle$ scaled by the target variance. For $\zeta$ and $u_z$, whose means are large, these terms weigh the mean residual again, by factors of about 173 and 40. The loss is still minimal at the target. | Reported to the user. Central second moments are the better reading of Section 2.4. The slices are also evaluated offline with central moments, from the stored moments. |
| The FD cache key holds only $\hat\theta$, so a stale entry could be reused. | Fixed: each run records hashes of the config, the target, and the EV code, and a mismatch raises. |
| The target `git_hash` records HEAD without a dirty marker. | Recorded above; fix with a dirty marker after the gate. |
| The Powell fallback evaluated the full 5-point FD stencil for each function value. | Fixed before the recovery run: Powell uses 1 run per point. |

Refuted findings: missing checks in `_check_supported` for features the twin does not use; the deposit weight swap is not exception-safe (a failed step leaves the state invalid anyway); the open-loop check covers only ions at rest; and the FD stencil leaves $[0, 1]$ by $h$ at a bound (the plan prescribes this setup, and the loss stays finite). A third finding, a NaN in the generated VJP at zero field, was first refuted because no P3 caller differentiates there. It is real for Phase P6: `ev_tangent` runs on every ion at every step, so Section 5.3 gets a guard for $x_\ell = 0$.

## Appendix A. Unverified claims and required probes

| Claim | Probe | Phase |
|---|---|---|
| The $k=0$ columns of the $p=m$ and $p=m+1$ components stay zero for $m \ge 1$ | Compare spect with `interp2spect(spect2interp(spect))` after 500 steps | P2 |
| Every field operator used by `step` is linear in the field arrays | Linearity check (Section 4.6) | P4 |
| Tangent kernels generic over $N_m$ agree with the 5-mode VJP clones | Dot-product test (Section 4.7) | P4 |
| `elec_from_N` traps a beam in Twin-S with 2 modes, `p_nt = 8`, at all checked points | Phase P1 steps 5 and 6 | P1 |
| $E_L$ and $z_f$ are locally identifiable in Twin-S | Phase P1 step 7 | P1 |
| EV births match the stochastic marginals in open loop | Phase P3 step 2 | P3 |
| The device peak of `value_and_grad` on Twin-S stays near the Section 8 estimate (about 5 GB used, 7.5 to 8 GB reserved), well below 48 GB. The earlier "below about 3 GB" is already false for the primal (T0: 5.17 GB used, 7.42 GB reserved). | Measure in `value_and_grad` | P7 |
| The raw second moments of the first loss draft give a well-conditioned loss | Closed on 2026-10-30: false. They counted the means of $z - z_{win}$ and $u_z$ again (factors 170 and 40). Section 2.4 now uses central moments and log scales. | P3 |

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
| `inversion/recover.py` | New: Section 2.5 recovery driver (design, then `least_squares` dogbox; FD Jacobian, then the JVP Jacobian) |
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
