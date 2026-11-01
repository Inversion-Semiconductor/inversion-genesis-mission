# Differentiable Inversion Plan for FBPIC-Warp

| Item | Value |
|---|---|
| Status | Phases P0 to P5 done; milestone M1 reached with the Stage A gate (Section 12.5); milestone M2 reached with deviations (Sections 12.6 and 12.7). Revised on 2026-10-30 with the optimizer of Section 2.5 (D26) and the one-parameter Stage A (Section 2.6, D28). Phases P6 to P9 re-planned on 2026-10-31 (Section 12.8, D29 to D32). Phase P6 done on 2026-10-31, with one accepted deviation (Section 12.9). Phase P7 in progress, checkpointed on 2026-10-31 (Section 12.10). Phase P7 found that exact tangents are unusable at the full horizon because the trapping dynamics is chaotic; Track R (Section 7, D33) started on 2026-10-31. On 2026-10-31 the user paused the FD recovery and Phases P8 and P9 to test Track R first (D34). Phase R0 done on 2026-11-01: the study recommends no-go for Track R, and the user decision is open (proposed D35). |
| Date | 2026-10-26 |
| Branch | `snidhan/dev-autodiff` (base commit `92352b8`) |
| Scope | One GPU. Lab frame. Stage A: one parameter ($E_L$) from Phase P3 to Phase P9. Stage B: both parameters ($E_L$, $z_f$). Forward-mode tangents. Twin experiment with noise-free synthetic data. |
| Owner | The implementing agent writes all code: kernels, tangents, algorithms, scaffolding, and tests. |
| Production code | Unchanged. The differentiable path is a separate stepper in `fbpic/autodiff` (D17). |
| Decisions | D1 to D34 in Section 10. D1, D2, and D21 to D34 are user decisions. The others are planner defaults that the user can change. |

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
| Tangent kernels for gather, push, and deposition | Verified (L1): Phase P4, Section 12.6 |
| One-step, horizon, and parameter gradients | L3 verified on Twin-XS; L4 verified for $N_t \le 30$ on Twin-XS with small $h$ (Section 12.7) |
| Memory and run time of a forward-mode gradient pass | Not measured |
| Expected-value ionization | Forward model verified in Phase P3 (Section 12.5); tangents verified by L1, L3, and short-horizon L4 in Phase P6 (Section 12.9) |
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
| T1 | EV with immediate emission ($\epsilon = 0$). If the $\epsilon = 0$ births do not fit in memory (Phase P8 step 1), use the reduced Twin-S for T1 and for its T0 control. | Main test. Report the offset from $\hat\theta^\ast$ (L6). |
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
- Defaults: $\alpha_D = 0$ in Stage A (D29). The MMD term enters in Stage B or later.

Implement the loss and its analytic gradient in CuPy or Warp (D11). Test every term with FD on random particle sets (L1). Also test the loss and its gradient on an empty selection (all $\sigma_i = 0$) and on a nearly empty selection (one particle with $\sigma_i = 10^{-6}$). Both must be finite and match FD.

### 2.5 Optimizer

The optimizer follows the literature note `inversion/notes/optimizer-literature-2026-10-30.md` (D26). It uses 5 techniques:

1. Trust-region Gauss-Newton or Levenberg-Marquardt on the residual vector $r_w$, not a quasi-Newton method on the scalar $J$: `scipy.optimize.least_squares(method="dogbox")` with bounds $[0, 1]^2$. The model Hessian $J_r^T J_r$ exists from the first iteration, and the trust radius bounds every step. L-BFGS-B failed here in Phase P3: its first step crossed the box, and a secant pair from a flat region sent the next step to the opposite corner (Section 12.5).
2. A global design, then a local solve. Evaluate a 25-point grid or Sobol design over $[0, 1]^2$. Start the local solver from the 2 or 3 best design points that are well separated. A bounded local method cannot leave a boundary local minimum, and the design shows whether one exists.
3. Loss design (Section 2.4): central moments, log scales, and whitened residuals.
4. FD steps. Use fixed steps, $h = 0.02$ for $E_L$ and $h = 0.05$ for $z_f$; the FD Jacobian is temporary until the JVP Jacobian of item 5. Use one-sided second-order stencils $(x, x + h, x + 2h)$ within $h$ of a bound. Keep $h$ above the width of the jumps that emission thresholds and cell crossings cause. Removing jumps in the model (zero-weight creation, window-edge tapering) is a separate decision for Phase P6.
5. JVP Jacobians in the same driver. After the JVP gate of the note (Section 3.2 there: comparison with central FD at several $h$ and with a smoothed Jacobian), the `jac` callable becomes 1 primal run with $K$ tangents, $K = 1$ in Stage A and $K = 2$ in Stage B (Sections 2.6 and 4.10). Before that gate, the driver uses the batched FD Jacobian, which reuses the cached center run. In Stage A, the user chooses the Jacobian at the end of Phase P7 from the usability study (Section 6.4, recovery gate): the AD Jacobian, a blend of AD and smoothed columns, or the FD Jacobian.

### 2.6 Staged scope: one parameter first

The work runs in two stages. Stage A takes one scalar parameter through every milestone, end to end. Stage B adds the second parameter after Stage A works.

Stage A (D28):

1. Parameter: $E_L$. $z_f$ stays at $z_f^\ast$ in the model and in every target. The tangent seed of $E_L$ is exact and simple: every initial field scales as $\sqrt{E_L}$, so the seed is $F_0 / (2E_L)$ (Section 4.8). The $z_f$ seed needs the profile derivative and `GaussianLaserDzf`, which move to Stage B.
2. Phases P3 to P9 apply with one tangent ($K = 1$). Write the tangent kernels and the driver generic over $K$ (Section 4.7), but run the Stage A tests, the L4 sweeps, and the recovery with the $E_L$ tangent only.
3. The loss, the weights $\Sigma$, the targets (T0′, T0, T1, T2), and the optimizer of Sections 2.2 to 2.5 do not change, except that $\alpha_D = 0$ throughout Stage A (D29). The optimizer works on one coordinate, with a 10-point design in place of the 25-point design.
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
| Particles per cell | `p_nz=1`, `p_nr=2`, `p_nt=8` for every species | FBPIC recommends `p_nt >= 4 Nm` in a docstring (`fbpic/main.py:963-964`) but does not enforce it. That is 8 for $N_m = 2$. The Phase P1 aliasing check confirms it. (D16) |
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
| Twin-S | Horizon sweep, identifiability, recovery | About 800 x 50 physical cells | About 1800 | about 2 M loaded (1.9 M at `p_nt=8`), plus 6.4 to 7.5 M live `elec_from_N` from EV births (Phase P3); about 8 M live at peak |

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

The main engine is forward mode (D25). Each step advances the primal state and $K$ tangent states together, one tangent per parameter: $K = 1$ ($E_L$) in Stage A and $K = 2$ in Stage B (Section 2.6). The code is generic over $K$. The gradient is the inner product of the loss gradient with the final tangents. No tape, checkpoint, or replay is needed. The reverse-mode design (checkpointed one-step tape, layout check, FFT and DHT adjoints) is kept at commit `07dcc6c` of this file for the future many-parameter case (Section 11).

### 4.1 Data flow

```text
init (host, once):
  sim = build_twin(theta, size, config)          # production objects for grids and coefficients only
  s_0 = begin(sim)                               # laser fields, prologue, empty or preloaded plasma
  t_j = laser_tangents(sim, theta)               # j in {E_L} (Stage A), {E_L, z_f} (Stage B); Section 4.8
  sdot_0[j] = seed(t_j)                          # damping and interp2spect of t_j; particle tangents zero

one pass, no tape:
  for n in 0 .. N_t-1:
      s_{n+1}, sdot_{n+1}, D_n = step(s_n, sdot_n)   # primal and all K tangents together
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

1. Keep `inv_gamma` as its own variable. The Vay push reads the stored old value, and the position push and J deposit read the new value. Treat its tangent the same way: `vay_tangent` writes the tangent of the new `inv_gamma` line by line, and `position_tangent` and `deposit_tangent` read that stored tangent. They do not recompute $-\gamma^{-3}\, u\cdot\dot u$. The VJP clones treat `inv_gamma` as an independent input (`fbpic/autodiff/particles.py:29, 53`, `fbpic/autodiff/deposition.py:60`), so a recomputed tangent breaks the L1 dot product. The closed form is an optional probe only.
2. `w_times_level` is derived, not state: $w\bar Z - W$ for N ions (Section 5.1). Its tangent is $w\sum_\ell \ell\,\dot P_\ell - \dot W$. He ions carry a fixed charge $2e$.
3. The interp E and B are derived from the spectral fields at the start of each step (`spect2interp`). Their tangents come from the same transform applied to the spectral tangents.
4. Give every species a tracker (`track()`). L2 and the FD harness match particles by ID.
5. The weight tangent is zero for every species except `elec_from_N`, whose births carry $\dot W'$, the pending-weight tangent after the EV update (Section 5.3). Keep the array for all species, for the future density parameters.

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
2. When a primal operation updates an array in place, first copy the inputs that its tangent needs into a per-step stash. Example: `ux, uy, uz, inv_gamma` before the Vay push. The position increment depends only on `u` and `inv_gamma` (`push/cuda_methods.py:62-65`), so positions need no stash. Keep the gathered-field tangents in a work buffer from the gather to `vay_tangent` and `ev_tangent`.
3. The primal result must not depend on whether tangents are computed. L2t checks this.
4. The step-0 diagnostic J deposit of production (`main.py:561-562`) is not reproduced. The J deposit of the same step overwrites it before any field push.
5. Exchange cadence and window shifts come from the `iteration` and window scalars stored in the state.

Use `step` without tangents for every FD run and every EV truth run. Production `Simulation.step` is the physics reference for L2 and the stochastic model for L5 and truth variant T2.

### 4.4 Operations and their tangents

| Production operation | Tangent | Source | Reference for L1 |
|---|---|---|---|
| `spect2interp`, `interp2spect` (FFT and DHT) | Same operator on the tangent arrays | Production `SpectralTransformer` methods | Linearity check |
| Curl-free current correction, PSATD push, `push_rho`, filter, `erase`, `divide_by_volume` | Same operator on the tangent arrays | Production or `fields.py` functional twins | Linearity check |
| Moving-window spectral shift of E, B, `rho_prev`, and J (`moving_window.py:144, 186-191`), open-z damping | Same operator on the tangent arrays | Production or `fields.py` twins | Linearity check |
| Gather E, B at particles | `gather_tangent` | New kernel (Section 4.7) | Dot product with the 5-mode gather VJP; `check_jvp` |
| Vay momentum push, electrons and He | `vay_tangent` | New kernel | Dot product with `push_momenta` VJP; `check_jvp` |
| Vay momentum push, N ions (charge $(w\bar Z - W)e/w$) | `vay_tangent` with the charge tangent from before the EV call (Section 5.3) | New kernel (Phase P6) | Dot product with the Warp-generated VJP of `push_ion_momenta_ev`; `check_jvp` |
| Position half push (x2) | `position_tangent` | New kernel | Dot product with `push_positions` VJP; `check_jvp` |
| Deposit J, `rho_next`, and the exchange-step `rho_prev` | `deposit_tangent` with the charge tangent current at the deposit (Section 5.3) | New kernel (Section 4.7) | Dot product with the 5-mode deposition VJP; `check_jvp` |
| EV ionization and emission | `ev_tangent` | New kernel (Section 5.3) | Dot product with the Warp-generated VJP of the EV kernel; `check_jvp` |
| Sort before each deposit | The primal permutation applied to the tangent arrays | `cupy.take` | Exact |
| Exchange-step removal and injection | Section 4.9 | Host and CuPy | Exact |

The field side is linear with $\theta$-independent coefficients (Section 1.4). Its tangent is the operator itself, so no new field code is needed.

### 4.5 Discrete decisions

The primal step makes every discrete decision: cell indices, the axis branch, sort permutations, removal ranges, injection counts, and emissions. The tangents follow these decisions and never change them. Forward mode therefore computes the conditional derivative of the step with the decisions fixed.

Rules:

1. Sort with the production key `ir_upper + iz_upper*(Nr+1)` (`cuda_sorting.py:94`) and a stable sort (`cupy.argsort` on the key). Apply the same permutation to the primal and to every tangent array. The stepper `sort_all` must return its permutation; today it discards it (`stepper.py:124-130`). Before `exchange_all_particles` in `step`, assert `sp.sorted` for every species. Otherwise `get_stay_range_gpu` runs the production radix sort (`particle_buffer_handling.py:250-252`) with a permutation that the tangents never see. `begin` exchanges unsorted particles; this is harmless because the particle tangents are zero there (Section 4.8). `rebuild_open_boundary_gpu`, `push_x`, and `ionize_and_emit` clear the flag.
2. Assign slots for emission and injection with an exclusive prefix sum over the parent index or batch. Never use an atomic slot counter.
3. `StepDecisions` records what the FD harness needs (Section 6.3): the `n_move` sequence; the injected count and the removed count per species and exchange event (`n_stay` from `get_stay_range_gpu`, before the rebuild); and the emitting parent IDs per step, keyed by (step, parent ID). The AD run (`value_and_grad`) is the recorder. The FD harness replays its decisions.
4. Cell indices and the axis branch are frozen in the tangent by construction, within each operation. FD runs do not freeze them, and no stepper mode does. With linear shapes, $J(\theta)$ is $C^0$ with a slope jump at every (particle, stage) cell crossing, so a crossing inside $[\theta - h, \theta + h]$ adds an $O(1)$ error to central FD, not $O(h^2)$. Section 6.3 rule 8 and Section 6.4 handle this.

The frozen emission decisions and the asserted counts (`n_move`, injected, removed) define the FD reference (Section 6.3). FD runs do not freeze cell indices or the axis branch (rule 4). The forward map is therefore piecewise smooth in $\theta$, with slope jumps at cell crossings. The tangents are its exact derivative on each smooth piece, and the FD reference agrees with them only where no cell crossing lies inside $[\theta - h, \theta + h]$.

### 4.6 Field tangents

Every field operation in the step is linear in the field arrays, and no parameter enters its coefficients. The tangent of each operation is the same operation applied to the tangent arrays.

1. Call the operator once per tangent, or once on a batched array where the operator allows it.
2. Phase P4 checks linearity for each operator: $\lVert A(x + \alpha y) - Ax - \alpha Ay\rVert \le 10^{-12}\,(\lVert Ax\rVert + \lvert\alpha\rvert\lVert Ay\rVert)$ for 3 random draws. A tighter bound fails on rounding alone: a dense 1080-long fp64 DHT has worst-case rounding $n\epsilon \approx 2.4\cdot10^{-13}$, and cuBLAS may change the reduction order between calls.
3. The interp tangents of E and B are derived from the spectral tangents at the start of each step, like the primal.

### 4.7 Particle tangent kernels

Warp 1.18 generates reverse-mode code only. The tangent kernels are hand-written, generic over $N_m$ as a compile-time constant, and process all $K$ tangents in one launch.

- Gather. For mode $m$, the gathered value is $G_m = \sum_c S_c(z, r) F_{m,c}$ over the cells $c$ of the particle. Its tangent is $\dot G_m = \sum_c S_c \dot F_{m,c} + \sum_c (\partial_z S_c\,\dot z + \partial_r S_c\,\dot r) F_{m,c}$, with $\dot r = (x\dot x + y\dot y)/r$. Differentiate the mode factor and the Cartesian rotation with $\dot\varphi = (x\dot y - y\dot x)/r^2$. The gather uses $e^{-im\varphi}$ (`phase_step = (cos, -sin)`, `gathering/cuda_methods_fused.py:177`), so $d e^{-im\varphi} = -i m \dot\varphi\, e^{-im\varphi}$.
- Deposition. Differentiate the fused kernel body line by line (`deposition/cuda_methods_fused.py`), as for the Vay push: $r$, $\cos\varphi$, $\sin\varphi$, the phase factor, $v_r$ and $v_\theta$, the Ruyten-corrected radial shape $S_r$ (not the gather shape), the fold sign, and the atomics into $K$ tangent grids. The code conventions are: the deposit phase is $e^{+im\varphi}$ (`phase_step = (cos, +sin)`, `:211`), and $v_r = \cos\varphi\, v_x + \sin\varphi\, v_y$, $v_\theta = \cos\varphi\, v_y - \sin\varphi\, v_x$ (`:207-208`). So J carries the rotation terms $\partial_\varphi v_r = v_\theta$ and $\partial_\varphi v_\theta = -v_r$ in addition to the mode-factor term $+i m\,\dot\varphi$ and the Cartesian velocity tangent $\dot v = c(\dot u\,\gamma^{-1} + u\,\dot{(\gamma^{-1})})$. Read $\dot{(\gamma^{-1})}$ from the stored tangent (Section 4.2, rule 1). The deposition clone computes $\cos\varphi$ and $\sin\varphi$ inside its backward-enabled kernel (`fbpic/autodiff/deposition.py:73-80`), so the L1 dot product fails if a term is missing.
- Vay push. Differentiate the production `push_p_vay` sequence line by line in a `@wp.func` that carries each value with its $K$ tangents. It writes the tangent of the new `inv_gamma`.
- Position push. $\dot x = (c\,\Delta t/2)(\dot u\,\gamma^{-1} + u\,\dot{(\gamma^{-1})})$, with $\dot{(\gamma^{-1})}$ the stored tangent from `vay_tangent`.
- Linear shapes have a derivative in position that jumps at cell crossings. The cell, the Ruyten index, and the axis branch are fixed within each operation. Gather, J deposit, and `rho_next` deposit each recompute them.
- Guard the axis. Production tests `radius != 0.0` exactly (`gathering/cuda_methods_fused.py:120`, `deposition/cuda_methods_fused.py:192`). At $r = 0$, use the production branch ($\cos\varphi = 1$, $\sin\varphi = 0$) and set $\dot r = 0$ and $\dot\varphi = 0$. Loaded particles never sit at $r = 0$; the real hazard is the $1/r^2$ factor in $\dot\varphi$ at small $r > 0$. The L3 harness masks particles at $r = 0$. The radial guard-cell fold (`ir_lower < 0`, `gathering/cuda_methods_fused.py:151-154`) is smooth for $0 < r < 0.5\,\Delta r$; its crossing at $r = 0.5\,\Delta r$ is covered by the cell-crossing mask of Section 6.3, rule 6.
- Validate each kernel at $N_m = 5$ by a dot product with the existing VJP clones: $e = \lvert\langle J u, v\rangle - \langle u, J^T v\rangle\rvert / (\lVert Ju\rVert\lVert v\rVert + \lVert u\rVert\lVert J^T v\rVert) \le 10^{-12}$ for 3 random draws. Then run `check_jvp` at $N_m = 2$ and $N_m = 5$. The gather fixture must include particles with $r \in (0, 0.5\,\Delta r)$, which take the `ir_lower < 0` guard-cell branch: the current fixture has $r \ge 1.17\,\Delta r$ (`tests/test_autodiff_particles.py:229-231`) and never takes it. Test $r = 0$ particles separately, not in the dot product: the clone's VJP is NaN there (adjoint of `wp.sqrt` at 0, as in Section 5.3), and `check_vjp` asserts finite gradients. For these particles, assert that the hand tangent is finite and uses the production branch with $\dot r = \dot\varphi = 0$.

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

Probes (Phase P5). Stage A runs probes 1 and 3. Probe 2 and `GaussianLaserDzf` belong to Stage B (Section 2.6).

1. `laser_initial_fields(sim, GaussianLaser(theta))` equals the six arrays that `add_laser_direct` writes into zeroed fields, within $10^{-14}$ relative.
2. $t_{z_f}$ matches a host central FD in $z_f$ of `laser_initial_fields`, for all six components and all modes, with $h \in \{10^{-3}, 10^{-4}, 10^{-5}\}$ of the normalized range, within $10^{-6}$ relative to the maximum over the array. At $h = 10^{-2}$ the FD truncation error is about $5\cdot10^{-5}$ of the maximum, so the probe would fail by construction.
3. $\dot s_0^{EB}[E_L]$ equals $s_0^{EB}/(2E_L)$ within $10^{-14}$ relative, where $s_0^{EB}$ is the six spectral E and B arrays after `begin`.

### 4.9 Structural events

Exchange iterations remove particles at the back of the window, inject new particles at the front, and redeposit `rho_prev`.

- Removal: drop the tangents of the removed range with the primal arrays. Removal happens at the guard edge, 96 cells (`n_inject` 32 plus `nz_damp` 64) left of `comm._zmin_global_domain` (`particle_buffer_handling.py:262-275`, `boundary_communicator.py:259-264`). It is a $\theta$-dependent decision: record the removed count (Section 4.5, rule 3), and slice the tangents with the same `[i_min, i_max)` as the primal, or realign them by tracker ID as `ionization.realign` does.
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
- Guard $x_\ell = 0$. Production returns $x = 0$ at $E = 0$ (`warp_inline_functions.py:70-71`, `fbpic/autodiff/ionization.py:87-88`). The formula above then evaluates $0\cdot(-\infty + \infty) = \mathrm{NaN}$, and $\dot E = \dot{(E^2)}/(2E)$ is $0/0$. Warp's generated VJP of `ev_ionize` is NaN there too (adjoint of `wp.sqrt` at 0), so the dot product cannot serve as the reference at $E = 0$. When $x_\ell = 0$, set $\dot p_\ell = 0$ and skip $\dot E$. This covers $E = 0$ (the step-0 gather, which reads the damped arrays, and `rj >= rmax_gather`, `gathering/cuda_methods_fused.py:139`) and the underflow band $E < \lvert c_\ell\rvert/745$. Compute $\dot E$ only when $E > 0$. One NaN in $\dot p$ reaches $\dot W$, $\dot w_\text{tl}$, and `deposit_tangent`, and poisons every field tangent within one step. Exclude $E = 0$ ions from the L1 dot product. For these ions, the L1 harness asserts that every output tangent is finite and that $\dot p_\ell = 0$, so $\dot P' = \dot P$ and $\dot W' = \dot W$.
- Compute $\dot E$ and $\dot\gamma$ from $E = \sqrt{|\gamma \mathbf{E} + \mathbf{u} \times c\mathbf{B}|^2 - (\mathbf{u}\cdot\mathbf{E})^2}$ and $\gamma = \sqrt{1 + u^2}$, with the gathered field tangents and the ion momentum tangents. Then $\dot p_\ell = \partial_E p_\ell\,\dot E + \partial_\gamma p_\ell\,\dot\gamma$.
- Population tangents: $\dot\Delta_{56} = \dot P_5 p_5 + P_5 \dot p_5$, $\dot\Delta_{67} = \dot P_6 p_6 + P_6 \dot p_6$, then the update of Section 5.1 applied to the tangents. The ion weight $w$ is constant, so $\dot W' = \dot W + w(\dot\Delta_{56} + \dot\Delta_{67})$.
- Charge tangent: $\dot w_\text{tl} = w\sum_\ell \ell\,\dot P_\ell - \dot W$. Compute it from the current $(P, W)$ and their tangents immediately before each use (Section 4.9). Within a step this gives two values. The value from before the EV call feeds `vay_tangent` of the N ions: `charge_weights` runs inside `push_ion_momenta` (`stepper.py:270`, `ionization.py:381`) before `ionize_and_emit` (`stepper.py:278`), as production `push_p_ioniz_gpu` reads the level before `handle_elementary_processes`. The same value feeds the exchange-step `rho_prev` deposit, which runs in `_exchange` before the EV call and after the realignment (`stepper.py:216, 219, 259`). The value from after the EV call feeds the J and `rho_next` deposits (`stepper.py:286, 293`). The EV update is out of place (`ionization.py:454`), so replace the EV tangent arrays out of place too; they need no stash entry.
- At an emission, the new electron copies the tangents of `x, y, z, ux, uy, uz, inv_gamma` from its parent, and its weight tangent is $\dot W'$ after the update and before the reset (`birth_w` is the post-update pending weight, `ionization.py:157, 162`). Then $\dot W$ is reset to 0 with $W$.
- The emission threshold is a decision. The tangent does not differentiate it (Section 6.3). These derivatives are local, with the emission decisions fixed. The threshold still changes the birth time and the particle count, so smooth population equations alone do not make the beam objective smooth.
- Reference for L1: write the EV primal also as a Warp kernel with `enable_backward=True`. Warp then generates its VJP, and the dot-product test of Section 4.7 compares the hand-written tangent with it. Of the two EV kernels in `ionization.py`, only `ev_ionize` is backward-enabled (`:96`); `copy_ev_electrons` is not (`:171`). The dot product covers the smooth map $(u, E, B, w, P, W) \to (P', W', w_\text{birth})$ with the emission flags fixed. Test the copy tangent as an exact index gather: slot assignment, reallocation, and ID generation have no derivative.

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
3. Beam: trapped charge $Q$, normalized transverse emittance, every $m_k$, $\mathrm{MMD}^2_{\mathrm{RFF}}$ (Stage B or later, D29), and the full loss $J$. For each quantity, compute the SE from 16 stochastic seeds.
4. Sensitivities: the forward-mode gradient of the full loss, relative to $\lVert g_{ref}\rVert_\infty$.
5. Adopt an $\epsilon$ only if every beam difference is below 2 SE, every gradient difference is below 5%, and the differences decrease as $\epsilon$ decreases. Otherwise report the results and ask the user. The options are a smaller $\epsilon$ or immediate emission.

## 6. Verification ladder

### 6.1 Levels

Run each level on GPU. Run each test file in its own process (`CLAUDE.md` rule 5).

| Level | Test | Pass criterion |
|---|---|---|
| L0 | Existing autodiff and fused tests | All pass, unchanged (122 GPU tests in `test_autodiff_*`) |
| L1 | Tangent kernels, EV kernels, ion push, field-operator linearity, loss terms, laser helper | Tangent kernels: dot product with the reference VJP, $e \le 10^{-12}$ for 3 random draws (Section 4.7), and `check_jvp` in vector form: $\lVert J d - \mathrm{FD}(d)\rVert_\infty / \max(\lVert J d\rVert_\infty, \lVert \mathrm{FD}(d)\rVert_\infty, \lVert \sum_b \lvert J_b d_b\rvert\rVert_\infty, 10^{-12})$, where $b$ runs over the input blocks, with the tolerance ($10^{-6}$, or $10^{-10}$ for linear kernels), the epsilon list, and the adjacent-epsilon pair rule of `check_vjp` (`verification.py:130, 157-173`). A random output cotangent, as in `check_vjp`, would dilute an error in $k$ of $n$ outputs by $\sqrt{k/n}$. Field operators: linearity within $10^{-12}$ (Section 4.6), at the Twin-S shape and at 2912 x 1080, modes $m = 0, 1$, all three DHT orders. Loss terms: `check_vjp` on the analytic gradient. |
| L2 | `step` parity with production | Twin-XS and Twin-S, pre-ionized N (no EV): `begin` plus 300 `step` calls without tangents against production `sim.step(300)` in one call from the same initial state. Fields within $\max(2\cdot10^{-11}, 10 d_{AA})$ relative in the infinity norm, where $d_{AA}$ is the maximum difference over 3 pairs of production runs (`tests/test_autodiff_stepper.py:11`). Particle counts and IDs equal; particle arrays matched by ID within the same bound. |
| L2t | Primal unchanged by tangents | (a) Bitwise, in one run: add `_hook` calls (`stepper.py:323-327`) before and after each tangent-only section, snapshot every primal array there, and require equality. Today the hook runs only at `before_ionization` and `after_ionization` (`stepper.py:274, 279`). Tangent operations run as separate launches after each primal operation, so this check is deterministic and sees a stash or swap leak of any size. (b) Smoke test: `step` with and without tangents, 300 steps on Twin-XS and on Twin-S, primal arrays within $\max(2\cdot10^{-11}, 10 d_{AA})$; counts and IDs equal. On Twin-XS this bound is about $5\cdot10^{-4}$ relative and cannot see a small leak, which is why (a) exists. |
| L3 | One full step | Random direction on all of $s_n$. Tangent of `step` against central FD of `step` with frozen decisions. `check_jvp` normalization and pair rule at $10^{-6}$. Steps on Twin-XS: one ordinary step; exchange steps 14 and 154 (154 is the first exchange with removal); one step with emissions (Phase P6). Harness rules in Section 6.3. |
| L4 | Parameter gradient, horizon sweep | Error definition and pass rule in Section 6.4. Horizons in Section 6.4. |
| L4w | Linear-wakefield derivative (physics check, no FD) | Section 6.5. Analytic: within the tolerances of the upstream test (8% of max for Ez, 11% for Er). Self-consistency: within $10^{-3}$ of max, and the relative residual scales as $a_0^2$ between $a_0 = 0.01$ and $0.005$ (Section 6.5, item 5). |
| L5 | EV vs stochastic | Section 5.4. Report only. The user decides acceptance. |
| L6 | Twin recovery | From the design-based starts and the 5 Latin-hypercube starts (Section 2.5), recover $\hat\theta^\ast$ within 0.02 in every free component on T0′ and T0: $\hat E_L$ in Stage A, both components in Stage B. On T0′, also report convergence diagnostics that are free of model bias: the spread of the multistart final points ($\le 2\cdot10^{-3}$), the final $\lVert\hat g\rVert_\infty$, and $\hat\theta - \hat\theta^\ast$ in units of $\sqrt{\mathrm{diag}\,G^{-1}}$. On T1, report the offset. Targets are noise-free (D22). |

### 6.2 Check functional for L3 and L4

Until the full loss is built (Phase P7), and whenever `elec_from_N` holds fewer than 1000 selected macroparticles, use

$$
J_{chk}(s) = \sum_{s} \sum_i w_i\, T(z_i - z_{win})\, \big(a_s\cdot\xi_i + (b_s\cdot\xi_i)^2\big) + \sum_m \lVert P \odot \hat E_m \rVert^2 .
$$

- $z_{win} :=$ `comm._zmin_global_domain`, the left edge of the physical domain.
- $\xi_i$ is the standardized $(x, y, z - z_{win}, u_x, u_y, u_z)$ with fixed scales. Fix the scales in code in `fbpic/autodiff/loss.py` and record them with every result, because they are outside `config_sha256`.
- $a_s$, $b_s$ (one pair per species) are fixed random arrays with seed 0. The positive spectral mask $P$ is a fixed random array with seed 0, band-limited to mode 0 and $\lvert k_z\rvert < k_0/3$. Without the band limit the laser sets the field term (the plasma share is about 4 to 6% on Twin-XS), and the field check would test the particle-to-field chain only to about $2\cdot10^{-3}$.
- $T$ is a smooth taper in $\zeta = z - z_{win}$: $T = 0$ for $\zeta \le 10\,\Delta z$, which includes the damping and injection cells left of $z_{win}$, and $T$ rises smoothly to 1 at $\zeta = 20\,\Delta z$. Removal happens 96 cells left of $z_{win}$ (Section 4.9), where $T = 0$, so a removal flip changes only the nonlocal field effect of the removed charge, not the particle term.
- $J_{chk}$ does not depend on particle order, needs no target, and is defined with and without ionization, from $N_t = 1$.
- Report the field term, the background-particle term, and the `elec_from_N` term separately. Each must pass. A term that is identically zero (for example `elec_from_N` under `ionization="none"`) is skipped, not failed.

### 6.3 FD rules

1. Build every FD run with the same `inversion/configs/<size>.json`: same $N_t$, $\Delta t$, window speed, `exchange_period`, $K_e$, $\epsilon$, and NumPy seed.
2. In the base run (the AD run of rule 3), record the `n_move` sequence, and the injected count and the removed count of every species at every exchange event. Each $\pm h$ run asserts that all three are equal. A count mismatch shifts the NumPy stream or the removal set and invalidates the FD value.
3. Frozen-decision FD is the AD reference. The AD run (`value_and_grad`) records `StepDecisions` (Section 4.5, rule 3); a separate primal base run could flip a few threshold decisions through nondeterministic atomics. Write the emitting parent IDs of every step to a named file for each base $\theta$, keyed by step iteration (about 105 MB on Twin-S); the stepper looks masks up by step (`ionization.py:407`). FD points run as separate processes; each $\pm h$ run loads the file, passes the IDs as `emit_mask`, and skips the threshold test. Electron IDs follow slot order (`ionization.py:457, 488-489`), so match across runs by (step, parent ID). Free FD runs record their own decisions.
4. Free FD keeps the thresholds active. Report free minus frozen as the jump contribution. It is not a small correction, and on the twins it does not shrink as $h$ decreases. Each emission threshold that crosses inside $[\theta - h, \theta + h]$ adds $\Delta J_k/(2h)$, where $\Delta J_k$ is the jump of $J$ at that crossing. Example: for $q(\theta) = w(\epsilon + \theta)\,\mathbb{1}_{\theta > 0}$ at $\theta = 0$, frozen AD gives 0 and free central FD gives $w/2 + \epsilon w/(2h)$. In the twins the packet weight is pinned at about $\epsilon w$ and the $\theta$ dependence sits in the release time $t_k(\theta)$. The frozen derivative then misses the birth-time term by a Riemann-sum error of relative order $\tau_e/T_\varphi$, where $\tau_e$ is the interval between emissions (2 to 4 steps) and $T_\varphi$ is the laser-phase time scale over which the ionization rate varies (a quarter laser period, 4 steps on Twin-S). On the twins this ratio is of order 1. With about $10^7$ births, crossings are dense in $\hat\theta$ at every plan $h$, so free minus frozen is a finite bias at every plan $h$. Free and frozen FD agree only for $h$ below the crossing spacing, about $10^{-6}$ or less. AD claims only the conditional derivative with the decisions fixed. L4 acceptance therefore uses frozen FD only. The $\epsilon$ sweep of Section 5.5 is the referee for the birth-time term: at $\epsilon = 0$, frozen equals free.
5. Threshold checks (Phase P6 and Phase P8):
   - Crossing probe: build a one-ion fixture with a prescribed field ramp, so that the emission threshold crosses at a known $\theta_c$. Set the fields through `_hook(state, "before_ionization")` (`stepper.py:274, 323-327`), which runs after the gather and push and can overwrite the gathered E and B each step. Confirm that free FD minus frozen FD equals $\Delta J/(2h)$ when $\theta_c$ is inside $[\theta - h, \theta + h]$, and is zero otherwise.
   - $\epsilon$ convergence: Phase P7 step 6 compares the forward-mode gradient with the immediate-emission reference on the reduced Twin-S, and Phase P8 step 3 compares the beam (Section 5.5).
6. L3 harness:
   - Run the tangent step once and store its decisions (rule 3).
   - Use a fixture with even `p_nz` and `p_nr`. The cell coordinate is $z_{cell} = (z - z_{min})/\Delta z - 0.5$ (`cuda_sorting.py:72-73`). With `p_nz = 1`, particles are loaded at $z_{min} + (i + 0.5)\Delta z$, so $z_{cell}$ is an integer and every resting particle sits on a shape boundary. The same holds in r for `p_nr = 1`. With an even count, $z_{cell} = i/2 - 1/4$.
   - Give `check_jvp` an optional `direction_masks` argument. Mask out every particle whose fractional cell coordinate in z or r, at gather, J deposit, or `rho_next` deposit, is within $10^{-3}$ of an integer. The shape kernels take the cell from `ceil(position)`, so a crossing cannot be frozen.
   - A field perturbation moves unmasked particles too. Record a branch signature for each particle at each stage: the cell indices, the axis branch, the `rj >= rmax_gather` branch (`gathering/cuda_methods_fused.py:139`), and the radial clamp at $N_r - 1$, at gather, at J deposit, and at `rho_next` deposit, and the emission flag. Recompute the signature for every FD evaluation. If it differs from the base, reject that FD point and use a smaller $\epsilon$.
   - Test the particle inputs and the field inputs as separate blocks, each with its own random direction.
7. Measure the run-to-run noise $\sigma_J$ as the maximum pairwise spread of at least 5 base runs. The FD noise error is $\sqrt 2\,\sigma_J/(2h\lvert g_j\rvert)$. Use the tolerance $\tau_j(h) = \max(10^{-4},\ 10\sqrt 2\,\sigma_J/(2h\lvert g_j\rvert))$ in place of $10^{-4}$ in Section 6.4, and reject any $h$ whose second term exceeds $10^{-3}$. (The earlier rule $2h\lvert g_j\rvert > 100\,\sigma_J$ admitted a noise error 140 times the pass threshold.) No deterministic deposition mode exists, so $\sigma_J > 0$ on Twin-XS. On Twin-S, the run-to-run noise is at rounding level (standard deviation $2.1\cdot10^{-12}$ over 3 runs of the P3 loss, `fd_recovery.py` `stage_noise`), and the rule rarely binds.
8. Cell crossings (Section 4.5, rule 4). In every FD run, record the per-stage production cell key (`cuda_sorting.py:72-77`) per tracker ID: all particles on Twin-XS (about 180 MB), sampled IDs on Twin-S. Report $K(h)$, the number of (particle, stage) keys that differ between the $+h$ and $-h$ runs, next to $e_j(h)$. On Twin-S, report the sampled count and the sample fraction; the $K(h) = 0$ rule of Section 6.4 then applies to the sampled count, and the result states this caveat. Past $n_c$, smooth-looking slices still carry deterministic roughness: the Twin-S fine slices show second differences of about $10^{-4}$ absolute (7% of $J$), against a run-to-run standard deviation of $2\cdot10^{-12}$. Estimate this roughness from the second differences of a fine slice along each coordinate; it sets the smallest useful $h$ there.

### 6.4 L4 error, pass rule, and horizons

- Error per component: $e_j(h) = \lvert g^{AD}_j - g^{FD}_j(h)\rvert / \max(\lvert g^{AD}_j\rvert, \lvert g^{FD}_j(h)\rvert, 10^{-6}\lVert g^{FD}(h)\rVert_\infty)$, in $\hat\theta$, with $h \in \{10^{-2}, 10^{-3}, 10^{-4}\}$.
- Pass: two adjacent $h$ with $K(h) = 0$ (Section 6.3, rule 8) give $e_j \le \tau_j(h)$ for every $j$. A component also passes when $\lvert g^{AD}_j\rvert$ and $\lvert g^{FD}_j\rvert$ are both below $10^{-6}\lVert g\rVert_\infty$.
- When no two adjacent usable $h$ (those not rejected by Section 6.3, rule 7) have $K(h) = 0$, the FD reference is invalid: each crossing adds an $O(1)$ error, and the aggregate grows as $\sqrt{K}$. Then a component passes if $g^{AD}_j$ lies within the spread of $g^{FD}_j(h)$ over the $h$ values, or within the smoothed slope of a $\pm 0.05$ slice with at least 41 points. Report such a pass as "FD reference invalid ($K > 0$)", not as a failure, so that a correct AD is not labeled exploratory.
- Report $e_j(h)$ and $K(h)$ for every $h$ and horizon, the growth of $\lVert\dot s_n[j]\rVert$ for each tangent, and 1D slices of $J$ over $\pm 0.05$ in each $\hat\theta_i$.
- Horizons: $N_t \in \{1, 10, 50\}$ and $\{n_c + 10, n_c + 50, n_c + 200\}$ must pass. $\{1000, \text{full}\}$ are reported; on Twin-XS (at most 300 steps) this is $\{300\}$. $n_c$ is the first step at which the laser peak reaches the position where $S(z) = 0.1$, at the nominal $\theta$. Compute $n_c$ once per configuration with a script and store it in `inversion/results/n_c_<size>.json` with the `config_sha256` of the configuration. Do not add it to `inversion/configs/<size>.json`: that changes `config_sha256`, and `load_target` would then refuse the T0 target (`make_target.py:81-82`). On Twin-XS, $n_c \le 20$; on Twin-S it is about 450.
- Agreement between batched and single tangents (Phase P7 step 2) shows that the driver is consistent. It does not show that the derivative is correct.
- Recovery gate (D32). At the end of Phase P7, the user chooses one branch for Phase P9 from the Jacobian usability study (Phase P7 steps 5 and 6) and records it in Section 12. On Twin-S, cell crossings make $K(h) > 0$ at every usable $h$ at the full horizon, so L4 there is reported, not passed. The question is whether the AD Jacobian is usable by dogbox, measured by the fixed-point shift $\delta$ at the T0 solution.
  1. AD usable: $\delta \le 0.005$, a quarter of the L6 tolerance. Phase P9 uses the AD Jacobian. Results claim exact derivatives of the frozen-decision map, verified by strict L4 at short horizons, and usability at the full horizon.
  2. Blend: $J = a\,J_{AD} + (1 - a)\,J_{sm}$, with $a$ set from the measured disagreement (literature note, Section 3.2 item 4). Phase P9 results state $a$.
  3. FD fallback: the central-FD Jacobian at $h = 0.02$ (3 runs per dogbox iteration). AD results are reported only.
  4. Exploratory: Phase P9 uses an AD Jacobian with $\delta > 0.005$. Results make no gradient-quality claim and are labeled exploratory.

### 6.5 Linear-wakefield derivative check (L4w)

Suggested by Rémi Lehe (FBPIC). Upstream `tests/test_linear_wakefield.py` compares the simulated Ez and Er at the end of a linear wakefield run with the linear-theory formulas. Those formulas (test docstring and functions `Ez`, `Er`) scale as $a_0^2$, so the analytic derivative is exact and closed-form:

$$
\frac{\partial E_z}{\partial a_0} = \frac{2 E_z^{an}}{a_0}, \qquad \frac{\partial E_r}{\partial a_0} = \frac{2 E_r^{an}}{a_0}.
$$

The check differentiates the full PIC cycle with respect to the laser amplitude and compares the result with physics, not only with FD.

0. Run the upstream `Nm=2` forward test first: call `test_linear_wakefield(Nm=2, show=False)` directly, in its own process. pytest runs only `Nm=1` (`tests/test_linear_wakefield.py:57`), and the `__main__` block runs all three cases with `show=True`, so the `Nm=2` tolerances are not yet exercised in this fork.
1. Configuration: the upstream test with `Nm=2`. That is a `GaussianLaser` polarized along y (`theta_pol = pi/2`, `tests/test_linear_wakefield.py:109`) in mode 1 and the wake in mode 0, $a_0 = 0.01$, $w_0 = 20$ µm, $c\tau = 6$ µm, $n_e = 8\times10^{24}$ m$^{-3}$, 800 x 120 cells, 1500 steps, electrons only, laser added with `add_laser_pulse` (direct method). It matches D4 and D23, and it needs no ionization, loss, or trapping. Keep `theta_pol = pi/2`: `compare_fields` is valid only in the $\theta = 0$ plane (`:160-166`), where a y-polarized laser has no Er or Ez. With the twin's x polarization (`theta_pol = 0`), the laser Er (about $4\cdot10^{10}$ V/m) swamps the $7\cdot10^6$ V/m wake. L4w needs its own builder, not `build_twin`: the stepper rejects `use_cuda_graphs=True` (the `Simulation` default), a missing moving window, and untracked species (`stepper.py:40-57`), and `build_twin` has a fixed 4-species set. L4w is kink-free (quiver 1.3 nm, wake push 0.2 nm, against a 12.5 nm cell margin), so it says nothing about cell crossings (Section 6.3, rule 8).
2. Parameter: $a_0$. Because $a_0 \propto \sqrt{E_L}$, the tangent seed is $\mathcal{T}\mathcal{D}\, t_{a_0}$ with $t_{a_0} = F_0/a_0$ (Section 4.8), and $\partial/\partial E_L = (a_0/2E_L)\,\partial/\partial a_0$.
3. Output: $E_z$ and $E_r$ on the grid at the last step, combined over modes as in `compare_fields` (mode 0 plus twice the real part of mode 1). One `jvp` run gives the whole derivative field $\partial E/\partial a_0$ on the grid, so the checks compare full fields.
4. Analytic check: compare with $2E^{an}/a_0$, within the upstream tolerances (`atol` 0.08 x max for Ez, 0.11 x max for Er). This error is dominated by the PIC-versus-theory error of the forward run.
5. Self-consistency check: compare with $2E^{sim}/a_0$, within $10^{-3}$ of max. Nonlinear corrections are of relative order $a_0^2 = 10^{-4}$, so this check is sharper than item 4. It isolates the derivative error from the discretization error. Write $E = c_2 a_0^2 + c_4 a_0^4$. The residual $\partial E/\partial a_0 - 2E^{sim}/a_0$ is $2c_4 a_0^3$; relative to $\max\lvert 2E^{sim}/a_0\rvert$ it is $(c_4/c_2)\,a_0^2$, with an unverified ratio $c_4/c_2$ (about $-1/4$ from the relativistic correction). Run $a_0 \in \{0.01, 0.005\}$ and require this relative residual to drop by a factor between 3 and 5.
6. Central FD in $a_0$ with the Section 6.4 rule, as a third reference.

## 7. Rollout plan

Effort estimates are agent working days, with large uncertainty because the long-horizon behavior is unknown. Phases P0 to P5 are done (Section 12). The remaining Stage A phases, P6 to P9, total about 14.5 to 21.5 days (Section 12.8). Stage B adds about 6 to 10 days (rough estimate, re-planned after Stage A).

Milestones. Stop and report to the user after each one.

1. M1, Phases P0 to P3 (about 13 to 20 days): one frozen Twin-S configuration, identifiability, the forward EV model, and an FD recovery from a displaced start. M1 shows that the problem can be solved.
2. M2, Phases P4 and P5 (about 12.5 to 17 days): a verified short-horizon gradient without ionization on Twin-XS, including L4w. M2 does not verify the EV tangents and birth copies, exchange steps with births, the loss gradient, Twin-S scale, or trapping horizons; Phases P6 and P7 cover them.
3. M3, Phases P6 and P7 (about 9.5 to 13.5 days): the gradient through ionization, the Twin-S horizon sweep to step 2000, and the Jacobian usability study that sets the recovery branch. These phases were re-planned on 2026-10-31 with the measurements of M1 and M2 (Section 12.8).
4. M4, Phases P8 and P9 (about 5 to 8 days): the model checks and the Stage A recovery.
5. Stage B (below) starts only after Stage A passes Phase P9.

Stage A (Section 2.6) runs P4 to P9 with the $E_L$ tangent only ($K = 1$). Every step below that names $z_f$ or a second tangent is marked Stage B.

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

1. Add `check_jvp` to `fbpic/autodiff/verification.py`: central FD of the primal kernel along a random direction, compared in the vector form of Section 6.1 (L1), with the epsilon list and adjacent-epsilon pair rule of `check_vjp`.
2. Implement `gather_tangent`, `vay_tangent`, `position_tangent`, and `deposit_tangent` in `fbpic/autodiff/tangent.py`, generic over $N_m$ (Section 4.7). `vay_tangent` writes the `inv_gamma` tangent that the other two read (Section 4.2, rule 1).
3. Extend the gather VJP fixture with particles at $r \in (0, 0.5\,\Delta r)$, and add the separate $r = 0$ check (Section 4.7).
4. Run L1 for each kernel: the dot-product test against the existing VJP clones at $N_m = 5$, and `check_jvp` at $N_m = 2$ and $N_m = 5$.
5. Run the linearity check for every field operator that `step` uses (Section 4.6), at the Twin-S shape, modes $m = 0, 1$, and all three DHT orders. Also run the $k = 0$ round-trip probe of Appendix A, row 1, which Phase P2 did not run.

Acceptance: L1 passes for every kernel and operator.

### Phase P5. Tangent step and driver, without ionization (6.5 to 9 days)

1. Add tangents to `step` for `ionization="none"`, with the stash and the permutation of tangent arrays (Sections 4.3 to 4.5). Make `sort_all` return its permutation, and assert `sp.sorted` before each exchange.
2. Implement the structural events (Section 4.9), including the removed-count record of Section 4.5, rule 3.
3. Implement the $E_L$ laser tangent and run probes 1 and 3 (Section 4.8). The $z_f$ tangent and probe 2 are Stage B.
4. Implement `value_and_grad` and `jvp` (Section 4.10). The objective is a parameter.
5. Write `fbpic/autodiff/loss.py` with $J_{chk}$ and its analytic state gradient (Section 6.2): $z_{win}$, the fixed scales, and the band-limited mask $P$. Run L1 on it.
6. Write the FD harness `inversion/verify_param_gradient.py` (Section 6.3), including the decision file of rule 3, the 5-run $\sigma_J$ of rule 7, and the cell-key record of rule 8.
7. Run L2t on Twin-XS and Twin-S: the bitwise in-run check and the two-run smoke test (Section 6.1).
8. Run L3 on Twin-XS: one ordinary step and exchange steps 14 and 154.
9. Write the L4w builder (`use_cuda_graphs=False`, tracked species, moving window as the stepper requires; about 0.5 day). Run L4w, the linear-wakefield derivative check (Section 6.5), starting with its item 0.
10. Run L4 on Twin-XS with $J_{chk}$ at the pass horizons. Report $K(h)$.

Acceptance: the probes, L2t (Twin-XS and Twin-S), L3, L4w, and L4 (pass horizons) pass without ionization; L3 and L4 on Twin-XS.

### Phase P6. EV tangents (3.5 to 5.5 days)

0. Before any EV tangent, commit these changes:
   - The three deferred M1 fixes: change `ev_ionize` (`fbpic/autodiff/ionization.py:154`) from $P_5' = P_5 - P_5 p_5$ to $P_5' = P_5 e^{-x_5}$, as Section 5.1 requires; make the emission record optional; add a dirty marker to the target `git_hash`.
   - FD caches keyed by code hash (D30): `runs/fd_<code_sha256[:12]>` in place of the fixed `runs/p3` (`fd_recovery.py:43`), and `ev_point` with the size and the target as arguments. Mark `runs/p3` read-only as P3 evidence.
   - Cell-key sampling for Section 6.3 rule 8 in `verify_param_gradient.py`: tracker IDs divisible by 256, records on the host. Full records do not fit in device memory on Twin-S.
1. Implement the tangents of Section 5.3 (with the $x_\ell = 0$ guard and the pre- and post-update charge tangents), the N-ion push tangent, and the Warp-generated reference VJPs. Run L1 on each kernel, with `ux, uy, uz` and the gathered fields as inputs, and the separate $E = 0$ check of Section 5.3. Test the emission copy as an exact index gather.
2. Add the EV tangents to `step`. Run L3 on a Twin-XS emission step.
3. Strict L4 gate on Twin-XS with EV and frozen emissions, at $N_t \in \{20, 28\}$ and $h \in \{10^{-6}, 10^{-7}, 10^{-8}\}$. First confirm that births occur by step 20. Twin-XS is chaotic beyond $N_t \approx 30$ (Section 12.7), so longer Twin-XS horizons are not used.
4. Twin-S report: compute and store $n_c$ (about 450), then run L4 with $J_{chk}$ and frozen emissions at $N_t \in \{n_c + 10, n_c + 50\}$, with $h$ down to $10^{-8}$ and sampled $K(h)$. This step is a gate only if two adjacent $h$ have a sampled $K(h) = 0$ and $e(h)$ follows the $h^2$ truncation trend.
5. Run the crossing probe of Section 6.3, rule 5.

Acceptance: steps 1, 2, 3, and 5 pass. Step 4 is reported.

### Phase P7. Loss Jacobian, horizon sweep, and Jacobian usability (6 to 8 days)

1. Implement the residual objective `objective(state) -> (r_w, dr_w/ds)` over the 11 residuals of Section 2.4, with $\alpha_D = 0$ (D29). The driver contracts each residual gradient with the tangent. Run L1 on every residual term, including the empty and nearly empty selections.
2. Run `jvp` on Twin-XS at $N_t \le 10$ once with $K = 2$ tangents and twice with one tangent each. Set the tolerance from two repeated $K = 1$ runs, because deposit atomics differ between runs (`CLAUDE.md` rule 4). In Stage A the second tangent is a synthetic seed (a fixed random direction on the initial fields); it tests that the code is correct for any $K$. Stage B repeats the check with the real $z_f$ tangent.
3. Measure peak GPU memory, host memory, and time per gradient on Twin-S with EV. Record the cost in run-equivalents next to the 3 runs per FD dogbox iteration (center and $\pm h$). Update Section 8.
4. Horizon sweep on Twin-S with $J_{chk}$ at $N_t \in \{1, 10, 50, n_c + 10, n_c + 50, n_c + 200, 1000, 2000\}$. Report $e(h)$, the sampled $K(h)$, and the growth of the field and particle tangents up to step 2000. Run the loss part at $N_t = 2000$ only: the loss is defined only where the selected beam has at least 1000 macroparticles, and the targets exist only at $N_t = 2000$.
5. Jacobian usability on T0 at $N_t = 2000$:
   - Run `value_and_grad` at every point of three 41-point slices over $\pm 0.05$ in $\hat E_L$: around the T0 solution ($\hat E_L \approx 0.588$), around the false minimum ($\hat E_L \approx 0.545$), and in the far field ($\hat E_L = 0.75$). Each run returns $r_w$ and the AD column $\partial r_w/\partial \hat E_L$.
   - Smoothed column: fit each of the 11 residuals separately on its slice. Also compute central FD at $h \in \{0.005, 0.01, 0.02, 0.05\}$.
   - Frozen-emission central FD at small $h$ at the same points. It separates code errors (AD against frozen FD) from the birth-time bias (frozen against free FD, Section 6.3 rule 4).
   - Report the distribution of the AD column error against the smoothed column (median, 90th percentile, and cosine), and the dogbox fixed-point shift $\delta = \lvert (J_{AD} - J_{sm})^T r_w\rvert / \lvert J_{AD}^T J_{sm}\rvert$ at the T0 solution.
   - Cost: 123 gradient runs, about 6 to 8 h on one GPU.
6. Before the branch choice, run the $\epsilon = 0$ gradient comparison on the reduced Twin-S (Section 5.5, item 4). If the frozen-against-free difference dominates the error of step 5, the remedy is $\epsilon \to 0$ or zero-weight creation (Section 11), not a fallback to FD.

Acceptance: steps 1 and 2 pass. Steps 3 to 6 are reported. The user chooses the recovery branch with the step 5 and step 6 numbers (Section 6.4, D32).

### Phase P8. Model checks (3 to 4 days)

Phase P8 runs after M3, from the committed P6 code (D31). Set `CUDA_VISIBLE_DEVICES` for every process: the stray GPU 0 context fix (`8f27245`) is not on this branch.

1. Birth count and memory on the full Twin-S at `p_nz = 2` (the T1 setting), at $\epsilon \in \{10^{-5}, 10^{-6}, 0\}$. The reduced Twin-S gave 215 births per ion at $\epsilon = 0$, which extrapolates to about 380 M births for T1 on the full Twin-S, about 21 GB used and about 1.5 times that reserved. If $\epsilon = 0$ does not fit in memory, run T1 on the reduced Twin-S, not at $\epsilon = 10^{-6}$: births per ion are bounded by the number of emitting steps, so a small nonzero $\epsilon$ saves little.
2. L5 (Section 5.4) for $E_L$: write the CRN referee script, run the production stochastic seeds (22.5 s per run, about 0.2 to 1.6 h for $M = 8$ to 64), and compare with the EV secant and with the AD gradient (Section 5.4, items 4 and 5).
3. The beam part of the emission-delay check (Section 5.5, items 1 to 3), without MMD (D29), on the reduced Twin-S.

Acceptance: L5, the emission-delay check, and the step 1 birth counts and mempool peaks are reported to the user. The T1 configuration for Phase P9 is recorded.

### Phase P9. Twin recovery (2 to 4 days)

Stage A runs this phase for $E_L$ alone, with $\hat z_f = \hat z_f^\ast$ (Section 2.6). Steps 1 and 3 give the Stage A form first. Stage B uses the two-parameter form in the same steps.

0. Confirm the recovery branch chosen at the end of Phase P7 (Section 6.4). Label every result with it.
1. Write `inversion/recover.py` with the Section 2.5 driver.
   - Stage A: the 10-point 1-D design in $\hat E_L$ of Phase P3 step 15, then `least_squares(method="dogbox")` on $r_w$ from the 2 best design points that are at least 0.2 apart, bounds $[0, 1]$. Use the Jacobian of the chosen branch: AD (1 primal run and 1 tangent), the blend, or the FD Jacobian.
   - Stage B: the 25-point design, then dogbox from the 2 or 3 best design points, with 1 primal run and 2 tangents, after the global-search decision of Section 2.6.
2. Generate the T1 and T2 targets (Section 2.2). T0′ and T0 exist from Phase P3.
3. Run L6 noise-free on Twin-S or on the reduced Twin-S, as Phase P8 step 1 decides. Stage A starts: the design-based starts and the $\hat E_L$ coordinates of the 5 starts of Section 3.4, with $\hat z_f = \hat z_f^\ast$. Stage B starts: the design-based starts and the 5 Latin-hypercube starts. T0′ is the tight convergence check, with the diagnostics of the L6 row. T0 must pass within 0.02. Run the same recoveries on T1 and report its offset from $\hat\theta^\ast$ as model bias. Record iterations, run-equivalents, and wall time. Compare them with the FD recovery of Phase P3. Report T2.

Acceptance: L6 passes for T0′ and T0. The T1 offset is recorded and reported to the user, together with the emission-delay results of Phase P8.

### Stage B. Add $z_f$ (after Stage A passes Phase P9; about 6 to 10 days)

Re-plan this block with the Stage A results before it starts.

1. Implement `GaussianLaserDzf`, the $z_f$ laser tangent, and probe 2 (Section 4.8).
2. Run L4 for the $z_f$ tangent on Twin-XS and on Twin-S at the horizons that Stage A passed. Repeat Phase P7 step 2 with the real $z_f$ tangent.
3. Choose the global stage of the 2-D recovery (Section 2.6, Stage B item 2). The two-parameter gate of Phase P3 (step 13) stopped in a false local minimum near $\hat\theta \approx (0.51, 0.28)$, and the true basin has a radius of about 0.04, which the 25-point design cannot resolve (Section 12.5). Options: a denser design near the valley, narrower bounds, observables with a wider basin, or a $z_f$ scan at the Stage A $E_L$ solution followed by the 2-D local solve. The user decides. Then repeat the two-parameter P3 gate on T0′.
4. Run L6 in 2-D on T0′, T0, and T1.

Acceptance: L4 passes for both tangents, and L6 passes in 2-D on T0′ and T0.

### Track R. Regular sensitivities by the particle-pdf method (research, started 2026-10-31)

Reason. Phase P7 step 4 showed that the exact tangent is accurate up to about step 500 on Twin-S. After step 1200 it grows about tenfold every 120 steps and reaches $1.7\cdot10^{11}$ per J at step 2000. At $N_t = 2000$, the AD residual column is 10 to 1000 times the smooth FD secant and has the opposite sign for 6 of the 11 residuals (Section 12.10). This is the Lyapunov instability of the N-body particle dynamics, not a code error. Chung, Bond, Cyr, and Freund (JCP 400, 108969, 2020) avoid it: they differentiate the Vlasov equation first and then discretize the sensitivity distribution $\delta f$ with particles (D33).

Method in brief:

1. $\delta f$ obeys the linearized Vlasov equation. In the relativistic electromagnetic form, $\partial_t \delta f + \mathbf{v}\cdot\nabla_x \delta f + q(\mathbf{E} + \mathbf{v}\times\mathbf{B})\cdot\nabla_p \delta f = -q(\delta\mathbf{E} + \mathbf{v}\times\delta\mathbf{B})\cdot\nabla_p f - \partial_\theta \mathcal{V}$.
2. Sensitivity particles move on the unperturbed characteristics, like the plasma particles. Their weights carry $\delta f$ and change by the right-hand side. No trajectory tangent is formed, so no Lyapunov growth occurs.
3. $\nabla_p f$ comes from a deposit of the plasma particles onto a phase-space mesh. $\delta\mathbf{E}$ and $\delta\mathbf{B}$ come from $\delta\rho$ and $\delta\mathbf{J}$ through the existing linear field tangents (Section 4.6).
4. EV births are a source of $f$. Their sensitivity is a weight source at the birth point, which the EV tangent $\dot W$ already gives (Section 5.3).
5. The paper shows the method only for 1D-1V electrostatic plasmas with Maxwellian velocities (Debye shielding, a sheath). The error falls as $N^{-1/2}$, like PIC. The cost is 1.9 to 3 PIC runs. The paper reads the result as consistent with the continuum limit, not exact for the particle model.

Open problems for LWFA:

1. Cold plasma and a thin trapped bunch make $\nabla_p f$ nearly singular. The momentum shape width becomes a smoothing parameter with a bias.
2. The phase-space mesh spans $(z, r)$, the azimuthal modes, and three momenta. It needs a sparse or adaptive representation.
3. The relativistic, electromagnetic, quasi-cylindrical form is untested.
4. The key assumption is that the continuum dynamics is Lyapunov stable for the observable. The paper states that no criterion is known. Evidence for it here: FD secants at $h = 10^{-2}$ and $10^{-3}$ agree at $N_t = 2000$.
5. The moving window, injection, and removal act as sources and sinks of $\delta f$ at the boundaries.

Variant R-v, to evaluate in Phase R0: keep the verified Lagrangian tangents while they are accurate, then project them periodically into weight form, $\delta f = -\nabla_z\cdot(f\,\xi)$ on a phase-space mesh, and reset the trajectory perturbation $\xi$ to zero. This variant reuses the kernels of Phases P4 to P6.

Phases. Stop and report to the user after each one. The research session leads this track with subagents.

- R0. Study (2 to 4 days). Derive the linearized Vlasov-Maxwell sensitivity for the FBPIC discretization: quasi-cylindrical modes, PSATD, Vay push, moving window, and EV births. Specify the phase-space mesh, the momentum shape, and the source evaluation. Compare the particle-pdf method with variant R-v. Output: a design note in `inversion/notes/` and a decision for the user.
- R1. Reproduction (2 to 3 days). Build a standalone 1D-1V electrostatic prototype that reproduces the Debye-shielding result of the paper: particle-exact against particle-pdf against a smoothed brute-force reference, and the $N^{-1/2}$ statistics.
- R2. Cold-beam prototype (3 to 5 days). Test the regularization of $\nabla_p f$ and the continuum-stability assumption on a cold beam with trapping in a wave, in 1D. Compare the particle-pdf method and variant R-v.
- R3. Implementation (8 to 15 days, rough). Add the chosen method to `fbpic/autodiff` for $E_L$ ($K = 1$), on Twin-XS and then Twin-S. Reuse the field tangents and the EV weight source. Production code stays unchanged (D17). Use `learning-mode-gpu` delivery mode for the kernels.
- R4. Verification (3 to 5 days). On Twin-S at $N_t = 2000$, compare the regular sensitivity column with the smoothed column of Phase P7 step 5 at the same points. Apply the fixed-point shift bar $\delta \le 0.005$ (D32). At $N_t \le 500$, compare with the verified Lagrangian tangent; expect agreement within the $N^{-1/2}$ statistical error, not to rounding.

Acceptance: R4 passes. Phase P9 may then use the regular Jacobian. Until then, Phases P8 and P9 use the branch chosen at the end of Phase P7. Effort: about 18 to 32 days, with large uncertainty.

Phase R0 result (2026-11-01, D34). A multi-agent study did R0: five derivations, two adversarial critiques, and one synthesis. The design note is `inversion/notes/r0-design-note-2026-10-31.md`; its inputs and CPU scripts are in `inversion/notes/r0-evidence-2026-10-31/`. The note recommends no-go for Track R as the route to multi-parameter optimization. The user decision is open (proposed D35, note Section 10). The main findings:

1. The algebra holds. The source of the linearized Vlasov equation is a phase-space dipole, also for the relativistic magnetic term, so the weight form is a score-function estimator. The adjoint is the exact transpose of the forward weight form.
2. The weight form needs $\nabla_u \ln f$. This quantity does not exist for the cold Twin-S background or for the EV electrons. A momentum smoothing scale $\sigma_u$ is therefore necessary, and the result is a coarse-grained derivative at that scale.
3. Twin-S at $N_t = 2000$ has no scale-free derivative. The FD roughness follows $h^{0.73}$ over $h = 10^{-3}$ to $10^{-7}$. On the P7 solution slice, local secants at $h \le 0.01$ differ from the local smooth derivative by 15% to 44% of the column norm.
4. A forward variant that is credible on cold particles costs about one primal run per parameter, the same as one-sided FD.
5. The $\delta$ bar of D32 alone is not a valid gate: 19.5% of random columns pass it. The note gives a guarded gate. Central FD at $h = 0.0025$ to $0.02$ passes the guarded gate on the P7 solution slice, and the AD column fails it.

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
| Primal state | 7 field arrays x $N_m$, plus about 90 B per particle (plus 32 B per N ion) | measured in Phase P3: mempool 2.62 GB used, 3.96 GB reserved (5.17 and 7.42 GB for the T0 run at `p_nz = 2`) |
| EV births | measured in Phase P3 | 11.9 to 16.3 M births, 4 to 6 x the 2.69 M loaded background; 6.4 to 7.5 M live `elec_from_N` at $N_t$ (6.66 M at the nominal $\theta$) |
| Tangent state, $K$ tangents | $K$ x (7 field arrays x $N_m$, plus 112 B per particle, plus 32 B per N ion). The 112 B are the 64 B state tangents plus 48 B of gathered-field tangents held between the separate gather and push kernels (Section 4.7). | about 0.9 GB per tangent (8 M live particles): 0.9 GB in Stage A, 1.8 GB in Stage B |
| Per-step stash | primal `u` and `inv_gamma` before the Vay push (Section 4.3, rule 2); the gathered-field tangents are counted in the tangent-state row | about 0.3 to 0.45 GB |
| Device peak | primal + tangents + stash + production workspace | measured in Phase P7 (Section 12.10): mempool 3.78 GB used, 6.85 GB reserved for a $K = 1$ gradient on Twin-S with EV |
| Forward run time | measured in Phase P3: 88.8 s per Twin-S EV run at the nominal $\theta$ (86 to 101 s over the checked points) | One run saturates the GPU: 25 runs in parallel batches took 2158 s (86 s each), about the same as serial. Count cost in serial run-equivalents of about 90 s. |
| Gradient cost | 1 primal + $K$ tangents in one pass | Stage A, measured in Phase P7: 3.47 run-equivalents, 298 s (the deposit tangents take half of the step). Stage B: about 5 to 6 run-equivalents (extrapolated) |
| Central FD, for comparison | 2 runs per parameter | Stage A: 2 run-equivalents, about 180 s. Stage B: 4, about 360 s |
| Phase P7 sweep and Phase P9 L6 | from the rows above | about 1.8 to 3.5 h for the P7 step 4 sweep; 10 to 30 h serial for L6 |

At 1 or 2 parameters, forward mode costs about as much as central FD, subject to the Phase P7 measurement. Forward mode removes the FD step-size truncation error. It does not remove the emission-threshold discontinuities: its derivatives are conditional on the discrete decisions (Section 4.5). L4 checks it against frozen-decision FD.

The host has two NVIDIA RTX 6000 Ada GPUs with 49140 MiB each (`nvidia-smi`, 2026-10-30). The plan uses one. Memory is not a constraint for Twin-S in forward mode.
The CuPy pool and the Warp allocator compete for device memory. Measure the peak of both.

## 9. Risks and mitigations

| Risk | Effect | Mitigation |
|---|---|---|
| Weak identifiability ($z_f$ against $E_L$, the two inverted parameters) | Recovery is ill-posed | Phase P1 step 7 and the FD recovery gate of Phase P3, before any tangent work. User decides. |
| Gradient growth over long horizons (trapping sensitivity) | AD disagrees with FD at long horizons | L4 sweep, $\lVert\dot s_n\rVert$ report. User decides. |
| FD reference invalid past $n_c$ (cell crossings of linear shapes) | Central FD carries an $O(1)$ error per crossing, so a correct AD can look wrong | Record cell keys and report $K(h)$; tight pass rule only at $K = 0$ (Sections 6.3 and 6.4) |
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
| Emission-threshold jumps | Free FD differs from AD by a finite birth-time bias at every plan $h$ | Frozen-decision FD as the only L4 reference, crossing probe, $\epsilon$ sweep as referee (Section 6.3, rule 4) |
| Removal at the window back is a $\theta$-dependent decision | Tangents attached to removed or wrong particles; FD runs that remove different sets | Removed count recorded (Section 4.5, rule 3) and asserted in every FD run (Section 6.3, rule 2); tangents sliced with the primal range (Section 4.9); L3 on Twin-XS exchange step 154 |
| Long-horizon gradient fails L4 | Recovery uses an unverified gradient | Recovery gate with three branches (Section 6.4) |
| Chaotic particle dynamics after trapping (Lyapunov growth of the tangents) | The exact AD Jacobian is unusable at $N_t = 2000$ (Section 12.10) | Track R regular sensitivities (D33); the P7 branch choice meanwhile (D32) |

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
| D16 | `p_nt = 8` for every species. | FBPIC docstring recommendation `p_nt >= 4 Nm` (`fbpic/main.py:963-964`, not enforced) with $N_m = 2$ |
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
| D29 | Stage A loss: the 11 whitened residuals with $\alpha_D = 0$. MMD moves to Stage B or later, so Phase P9 uses the same loss as the P3 FD recovery. | User, 2026-10-31 |
| D30 | FD caches are keyed by code hash. `runs/p3` is frozen as P3 evidence; Phase P9 compares with the recorded P3 results. | User, 2026-10-31 |
| D31 | Phase P8 runs after M3 from committed code, not in parallel with P6 and P7, because it imports the EV code that P6 edits. | User, 2026-10-31 |
| D32 | The recovery branch is chosen from the P7 Jacobian usability study: AD usable at a fixed-point shift $\delta \le 0.005$, blend, FD fallback, or exploratory (Section 6.4). Replaces the full-horizon L4 pass as the bar for using AD. | User, 2026-10-31 |
| D33 | Start Track R (Section 7): regular sensitivities by the particle-pdf method of Chung et al. (2020), as the response to the chaotic long-horizon tangents. The research session leads it with subagents. Until Phase R4 passes, Phases P8 and P9 use the branch chosen at the end of Phase P7. | User, 2026-10-31 |
| D34 | Do not run the FD recovery now. Pause Phases P8 and P9, defer the P7 branch choice (D32), and skip the rest of the P7 step 5 study (about 6.3 GPU-h). Test the feasibility of Track R first, because the goal is multi-parameter optimization. Phase R0 runs as a multi-agent study and must also cover the multi-parameter cost and an adjoint form. | User, 2026-10-31 |

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

### 12.6 Phase P4. Tangent kernels

Run on 2026-10-31 on physical GPU 1 (RTX 6000 Ada, 49140 MiB), Warp 1.18.0, code on top of commit `6e855cb`. Stage A scope: the kernels are generic over $K$ and $N_m \le 5$; the tests use $K = 3$ for the three dot-product draws and $K = 1$ for `check_jvp`.

Files:

1. `fbpic/autodiff/tangent.py`: `gather_tangent`, `vay_tangent`, `position_tangent`, `deposit_tangent`. The gather and deposition kernels are compile-time factories over ($N_m$, $K$). The Vay and position kernels loop over $K$ at run time.
2. `fbpic/autodiff/verification.py`: `check_jvp` (vector form of Section 6.1, epsilon list and pair rule of `check_vjp`, optional `direction_masks`) and `dot_product_error`.
3. `fbpic/autodiff/field_tangents.py`: `FieldTangents`, $K$ copies of every spectral and interp field array. `apply` runs a production field method once per tangent, with the tangent arrays swapped into the production grid objects.
4. Tests: `tests/test_autodiff_tangent.py` (L1 of the four kernels), `tests/test_autodiff_field_tangents.py` (operator linearity and the $k = 0$ probe). The gather VJP fixture in `tests/test_autodiff_particles.py` has two new particles at $r = 0.21\,\Delta r$ and $0.43\,\Delta r$.

Design:

1. Particle tangents are $(K, N)$ arrays, so that row $k$ is coalesced. Field tangents are $(K, N_z, N_r)$ complex arrays.
2. The gather tangent differentiates the production gather line by line: $r$, $\cos\varphi$, $\sin\varphi$ (written as $x/r$ and $y/r$, with $d(1/r) = -\dot r/r^2$), the linear weights including the guard-cell fold, the phase $e^{-im\varphi}$ through its `cmul` recursion, and the Cartesian rotation. The cells, the fold, the radial clamp, and the `rmax_gather` branch come from the primal.
3. The deposition tangent uses the production work split: one thread per sorted cell, particles in reverse order, a local matrix of $4 \times$ components $\times K$ complex entries, then atomics into the tangent grids. It differentiates $r$, $\cos\varphi$, $\sin\varphi$, the phase $e^{+im\varphi}$, $v_r$ and $v_\theta$ (the rotation terms follow from $\dot{\cos\varphi}$ and $\dot{\sin\varphi}$), the Ruyten-corrected radial shape (slope $-1 + \beta(1 - 2u)$, with the fold sign), the axial shape, the weight, and the stored `inv_gamma` tangent.
4. The Vay tangent (`vay_jvp`) carries each line of `push_p_vay` with its tangent and writes the tangent of the new `inv_gamma`. The position tangent reads that stored tangent (Section 4.2, rule 1).
5. At $r = 0$, the kernels keep the production branch ($\cos\varphi = 1$, $\sin\varphi = 0$) and set $\dot r = \dot\varphi = 0$.

Command, from `lpa/FBPIC_Warp`, with `CUDA_VISIBLE_DEVICES=1`:

```bash
python -m pytest tests/test_autodiff_tangent.py -q -s
python -m pytest tests/test_autodiff_field_tangents.py -q -s
python -m pytest tests/test_autodiff_particles.py -q
```

L1 results, tangent kernels. The dot-product error is the largest of 3 draws, computed in scaled variables (inputs and outputs divided by their characteristic scales), against the reverse-mode clones at $N_m = 5$. The `check_jvp` error is the largest over 3 directions of the best normalized error; the pair rule holds for every direction.

| Kernel | Dot product, largest $e$ (limit $10^{-12}$) | `check_jvp` at $N_m = 2$ | `check_jvp` at $N_m = 5$ |
|---|---|---|---|
| Gather, positions / E grids / B grids / all | 3.0e-17 / 1.1e-17 / 1.4e-17 / 1.7e-17 | positions 6.2e-10, fields 1.7e-15 (linear) | positions 2.3e-10, fields 1.4e-15 (linear) |
| Vay push, electrons ($a_0$-scale fields), He$^{2+}$ | 4.1e-17, 2.1e-17 | electrons 6.1e-11, He 9.8e-10 (no modes) | - |
| Position push | 3.4e-17 | 4.2e-14 (no modes) | - |
| Deposit $\rho$, positions / weights / grids / all | 1.7e-17 / 1.2e-17 / 0 / 9.0e-18 | 7.3e-11 | 4.5e-10 |
| Deposit $J$, positions / weights / momenta and `inv_gamma` / grids / all | 2.4e-17 / 3.8e-18 / 6.1e-18 / 0 / 1.3e-17 | 1.2e-10 | 1.2e-10 |

The `check_jvp` reference is the production forward kernel (fused gather, `push_p_gpu`, `push_x_gpu`, fused deposition), not the clone. The gather fixture includes particles in the guard-cell fold ($r = 0.08$, $0.21$, $0.43\,\Delta r$) and one beyond `rmax_gather`. The deposition fixture includes 3 particles at $r = 0.3\,\Delta r$, in the fold. The separate $r = 0$ tests pass: the tangents are finite, and they equal the tangents with $\dot x = \dot y = 0$ for those particles (exact for gather, within $10^{-14}$ of the maximum for deposition, which uses atomics).

L1 results, field operators (Section 4.6). Each operator runs through `FieldTangents.apply` on 3 slots that hold $x$, $y$, and $x + \alpha y$, for 3 random draws. The table gives the largest normalized residual over all arrays, modes $m = 0, 1$, and draws (limit $10^{-12}$). The vector transforms use the DHT orders $m \pm 1$, the scalar transforms order $m$.

| Operator | Twin-S grid (800 x 50 physical cells) | 2912 x 1080 physical cells |
|---|---|---|
| `spect2interp` E, B | 4.0e-16, 4.2e-16 | 1.1e-15, 9.1e-16 |
| `interp2spect` E, B, J, `rho_prev`, `rho_next` | 4.6e-16 to 8.1e-16 | 1.6e-15 to 2.5e-15 |
| `spect2partial_interp` E, `partial_interp2spect` B | 3.1e-16, 3.4e-16 | 7.4e-16, 6.2e-16 |
| `damp_EB_open_boundary` | 1.3e-16 | 1.5e-16 |
| `divide_by_volume` rho, J | 1.6e-16, 1.7e-16 | 1.4e-16, 1.6e-16 |
| `filter_spect` J, `rho_next`, `rho_prev` | 1.7e-16 to 1.9e-16 | 1.7e-16 to 2.5e-16 |
| `correct_currents` (curl-free) | 2.6e-16 | 1.7e-16 |
| `push` (PSATD and `push_rho`) | 2.7e-16 | 2.9e-16 |
| `shift_spect_grid` (3 cells) | 2.6e-16 | 2.7e-16 |
| `exchange_and_damp_EB` | 5.4e-16 | 1.5e-15 |

The primal spectral arrays are bitwise unchanged after all swapped operations.

$k = 0$ round-trip probe (Appendix A, row 1). Twin-S, nominal $\theta$, `ionization="none"`, 500 steps, then $\mathrm{interp2spect}(\mathrm{spect2interp}(s))$ against $s$ for E and B (`inversion/results/p4_k0_round_trip.json`):

1. The round-trip error is at most $1.3\cdot10^{-15}$ of the array maximum for every array, except mode 1 `Bp` ($2.6\cdot10^{-14}$) and mode 1 `Ep` ($1.7\cdot10^{-12}$).
2. For $m = 1$, the $k_r$ index 0 column of `Ep`, `Ez`, `Bp`, and `Bz` is exactly 0. The `Em` and `Bm` columns are not zero, and their round-trip error is $9\cdot10^{-16}$. The claim of Appendix A holds.

Acceptance: L1 passes for every kernel and operator. The existing L0 particle tests pass with the extended gather fixture (17 tests). Production code is unchanged.

### 12.7 Phase P5. Tangent step and driver, without ionization

Run on 2026-10-31 on physical GPU 1 (RTX 6000 Ada), code on top of commit `7c5aa8f`. Stage A: one tangent ($K = 1$, $E_L$), `ionization="none"`.

Status: the probes, L2t, L3, and the J_chk L1 pass. L4 on Twin-XS fails at the plan steps $h \in \{10^{-2}, 10^{-3}, 10^{-4}\}$ from $N_t = 10$. L4w passes the analytic and self-consistency checks, but the $E_r$ residual ratio (2.67) is outside 3 to 5. The user accepted Phase P5 with these deviations on 2026-10-31 (see the end of this section).

Files:

1. `fbpic/autodiff/stepper.py`: `step(state, tangents)`. `sort_all` returns its permutations and applies them to the tangents. `_exchange` asserts `sp.sorted` for every species (not in `begin`), realigns the tangents by tracker ID, and records the removed count. Hooks `stage:*` and `tangent_*:before/after`; `state.hooks["*"]` receives every hook name.
2. `fbpic/autodiff/state.py`: `TangentState.zeros`, `StepDecisions.removed`.
3. `fbpic/autodiff/driver.py`: `jvp`, `value_and_grad`, `seed_tangents`, `contract`. `fbpic/autodiff/laser.py`: `laser_initial_fields`, the $E_L$ and $a_0$ seeds. `fbpic/autodiff/loss.py`: $J_{chk}$ with its analytic state gradient.
4. `inversion/builders.py`, `inversion/state_snapshot.py`, `inversion/verify_one_step.py` (L3), `inversion/verify_param_gradient.py` (L4, $n_c$), `inversion/verify_linear_wakefield.py` (L4w), and three diagnostics (`diag_tangent_growth.py`, `diag_tangent_cost.py`, `diag_l4w_noise.py`).
5. Tests: `tests/test_autodiff_laser.py` (probes 1 and 3), `tests/test_autodiff_loss.py` (J_chk L1), `tests/test_autodiff_tangent_step.py` (L2t).

Design:

1. After each primal operation, `step` applies its tangent: gather tangent, Vay tangent (from a stash of $u$ and `inv_gamma` taken before the push), position tangent, the sort permutation, and the deposit tangents. Each deposit tangent runs the kernel of Section 12.6 and then the production post-operations (divide by volume, `interp2spect`, filter) on the swapped tangent arrays. The field push, the window shift with the recorded `n_move`, and `exchange_and_damp_EB` also run on the swapped tangent arrays.
2. Exchange: production removal and injection do not expose a permutation. The tangents of kept particles move to their new slots by tracker ID; injected particles get zero tangents. The removed count is (particles before) minus (particles after minus injected).
3. Seeds (Section 4.8): the $E_L$ seed $F_0/(2E_L)$ goes into the interp tangents, and the prologue field operations of `begin` (damping, `interp2spect`) run on it.
4. API deviation: `value_and_grad(theta, objective, build, n_steps, seeds, bounds)` and `jvp(theta, build, seeds, n_steps)` take a builder `build(theta) -> (sim, species, profile)` in place of `config`. The builder keeps `fbpic.autodiff` independent of `inversion/`. The objective may return a dict of terms; the residual-Jacobian form is not implemented (Phases P7 and P9 need it).

Commands, from `lpa/FBPIC_Warp`, with `CUDA_VISIBLE_DEVICES=1`:

```bash
python -m pytest tests/test_autodiff_laser.py -q -s
python -m pytest tests/test_autodiff_loss.py -q -s
python -m pytest tests/test_autodiff_tangent_step.py -q -s
python inversion/verify_one_step.py
python inversion/verify_param_gradient.py nc twin_xs
python inversion/verify_param_gradient.py l4 twin_xs
python inversion/verify_param_gradient.py l4 twin_xs --steps 70 --slice-points 0 --h 1e-4 1e-5 1e-6 1e-7 1e-8 --tag _small_h
python inversion/verify_linear_wakefield.py forward
python inversion/verify_linear_wakefield.py check
```

Probes (Section 4.8). Probe 1: `laser_initial_fields` equals the fields that `add_laser_direct` writes, with a difference of exactly 0 on Twin-XS and Twin-S. Probe 3: the $E_L$ tangent of the six spectral fields after `begin` equals $s_0/(2E_L)$ within $4.2\cdot10^{-16}$ (Twin-XS) and $6.3\cdot10^{-16}$ (Twin-S). The scale of each difference is the largest entry of the same field (E or B) over all modes and components. For the x-polarized laser, mode 0 and the mode 1 `Ep` hold rounding noise only ($10^{-13}$ against $10^{4}$ V/m), so their own maximum is no scale. The `rho_prev` and particle tangents are exactly 0.

J_chk L1. The analytic gradients of the particle term (positions, momenta, weights, all) and of the field term match central FD with the pair rule: best errors $\le 1.1\cdot10^{-9}$ and $\le 2\cdot10^{-13}$. The empty species gives $J = 0$ and empty gradients. The fixed choices are recorded with every L4 result (`CheckFunctional.record`).

L2t (Section 6.1), Twin-XS and Twin-S, 300 steps:

1. (a) Bitwise in-run check: every tangent section (gather, Vay, 2 position pushes, 3 sorts, 3 deposits, field push, window, realign) is enclosed by hooks that snapshot every primal array and scalar. All 3063 sections per run leave the primal bitwise unchanged.
2. (b) Smoke test, with and without tangents, bound $\max(2\cdot10^{-11}, 10\,d_{AA})$ with $d_{AA}$ from 3 production runs: largest ratio 0.058 (Twin-XS, electron `uz`) and 0.13 (Twin-S, spectral mode 0 `Bz`). Counts and IDs are equal.

L3 (Section 6.1), Twin-XS with `p_nz = 2`, nominal $\theta$ (`inversion/results/p5_l3_twin_xs.json`). `state_snapshot.py` restores the full state; a restored step reproduces the first step to $9.7\cdot10^{-16}$. Each block (particles, fields) has 2 random directions. Particles within $10^{-3}$ of a cell boundary at any stage are masked. An epsilon whose branch signature (cell keys, axis and `rmax_gather` branches, by tracker ID) or decisions differ from the base is rejected.

| Step | Decisions | Particle block, best $e$ | Field block, best $e$ | Masked electrons |
|---|---|---|---|---|
| 30 (ordinary) | no exchange | 1.4e-9, 4.2e-10 | 2.2e-11, 2.0e-10 | 354 |
| 14 (exchange) | 29696 injected per species | 1.1e-9, 1.4e-9 | 1.6e-13, 1.3e-13 | 233 |
| 154 (first removal) | 5620 electrons and 9728 ions removed | 6.0e-9, 7.8e-9 | 1.4e-12, 2.3e-12 | 824 |

All 12 directions pass the pair rule at $10^{-6}$. Deviation: the error is the vector form of Section 6.1 over the whole output, with each output array divided by its largest primal value. A first version normalized each output array by its own norm. It failed on the N-ion `inv_gamma` ($\approx 1$, tangent about $10^{-12}$), whose central difference is dominated by $\mathrm{ulp}(1)/(2\epsilon)$. With the field perturbations, cell crossings reject $\epsilon \ge 10^{-5}$ (rule 6), so $\epsilon = 10^{-8}$ was added to the list.

L4 (Section 6.4), Twin-XS, $J_{chk}$, $n_c = 18$ (`inversion/results/n_c_twin_xs.json`), pass horizons $\{1, 10, 28, 50, 68, 218\}$, report horizon 300 (`inversion/results/p5_l4_twin_xs.json`). Error $e$ of the total $J_{chk}$ and $K(h)$ at the plan steps:

| $N_t$ | $g^{AD}$ ($\hat\theta$) | $e$ at $h = 10^{-2}$ / $10^{-3}$ / $10^{-4}$ | $K(h)$ | Verdict (Section 6.4) |
|---|---|---|---|---|
| 1 | 3.8e8 | 1.4e-6 / 1.4e-8 / 3.0e-10 | 16 / 1 / 0 | FD reference invalid ($K > 0$); $g^{AD}$ just outside the FD spread |
| 10 | 6.9e8 | 2.7e-5 / 2.0e-3 / 4.2e-4 | 2414 / 230 / 21 | FD reference invalid; fail |
| 28 | -5.5e10 | 1.1 / 0.91 / 1.4 | 57243 / 12565 / 2003 | FD reference invalid; fail |
| 50 | 2.1e14 | 1.0 / 1.0 / 1.0 | 3.6e5 / 1.7e5 / 7.4e4 | FD reference invalid; fail |
| 68 | -2.6e14 | 1.0 / 1.0 / 1.0 | 8.7e5 / 4.9e5 / 2.7e5 | FD reference invalid; fail |
| 218 | -4.0e12 (field term only) | 1.0 / 1.0 / 1.0 | about 7e6 to 1e7; removal counts differ | fail |
| 300 (report) | -2.3e12 (field term only) | 1.0 / 1.0 / 1.0 | about 7e6 to 1e7; removal counts differ | fail |

The field and background terms behave like the total. `elec_from_N` is identically zero and is skipped. The plasma leaves the window before step 218, so the background term is 0 there. Rule 2: every $\pm h$ run changes the removed electron count at iteration 112, and repeated base runs at the same $\theta$ differ at iteration 196 or 224 (one electron on the removal boundary). At $N_t = 1$, all key flips are resting electrons on shape boundaries (`p_nz = 1`, Section 6.3 rule 6).

Small-step diagnostic (not a plan gate; `inversion/results/p5_l4_twin_xs_small_h.json`). Where $K(h) = 0$, AD matches FD:

| $N_t$ | $e$ with $K(h) = 0$ | $e$ at $h = 10^{-5}$ / $10^{-6}$ ($K$) |
|---|---|---|
| 1 | 1.1e-9 to 5.7e-7 ($h = 10^{-5}$ to $10^{-8}$) | 1.1e-9 (0) / 8.0e-9 (0) |
| 10 | 6.0e-8, 1.3e-6 ($h = 10^{-7}, 10^{-8}$) | 6.4e-11 (1) / 5.6e-9 (1) |
| 20 | 4.7e-6, 2.2e-5 ($h = 10^{-7}, 10^{-8}$; $\tau = 1.9\cdot10^{-4}$ at $10^{-8}$) | 0.23 (22) / 1.7e-7 (1) |
| 28 | 2.2e-6 ($h = 10^{-8}$ only) | 0.24 (234) / 0.14 (27) |
| 30 | 1.8e-3 ($h = 10^{-8}$, $\tau = 7.1\cdot10^{-3}$) | 1.0 / 1.0 |
| 40 to 70 | no $h \ge 10^{-8}$ has $K(h) = 0$ | about 1 |

Tangent growth on Twin-XS: the largest $|\partial u_x/\partial E_L|$ is 65 J$^{-1}$ at step 10, $1.2\cdot10^6$ at step 30, and $6\cdot10^8$ at step 50, a factor of about 10 every 5 steps. The largest values belong to electrons in the laser at $r = 1$ to 3 µm, not near the axis. $\sigma_J$ (5 base runs) grows from 0 at $N_t = 1$ to 1.6e-3 at 28 and 19 at 68.

Conclusion: the tangent is the exact derivative of the frozen-decision map wherever central FD can resolve it ($N_t \le 30$). The L4 failure at the plan steps comes from the fixture. Suspect assumption: "Twin-XS is a usable L4 fixture up to $n_c + 200$ steps." Twin-XS resolves $\lambda_0$ with 3.85 cells ($\Delta z = 0.208$ µm, $c\Delta t = \Delta z$, $\omega_0\Delta t \approx 1.6$) at $a_0 \approx 4$. Its discrete particle map is strongly chaotic (Lyapunov rate about $\ln 10/5$ per step). Cell crossings then appear for every $h \ge 10^{-7}$ after about 20 steps, and the derivative grows beyond what fp64 FD can check after about 30 steps.

Twin-S comparison (`inversion/results/p5_tangent_growth_twin_s.json`, 16 cells per $\lambda_0$, 1000 steps): $|\partial u_x/\partial E_L|$ rises by about 10 per 50 steps while the laser approaches the plasma, rises by $10^2$ between steps 450 and 625, and then saturates near $10^6$ J$^{-1}$ ($|\partial x/\partial E_L|$ near 3 m/J). Twin-S has no runaway, but a few particles are very sensitive. This is the trapping-horizon risk of Phase P7.

L4w (Section 6.5; `inversion/results/p5_l4w.json`):

1. Item 0: the upstream `test_linear_wakefield(Nm=2, show=False)` passes in its own process (13 s).
2. Own builder: the upstream configuration with `use_cuda_graphs=False` and a tracked electron species; the laser seed is $F_0/a_0$.

| Check | $E_z$ | $E_r$ | Limit |
|---|---|---|---|
| Analytic, $a_0 = 0.01$: $\partial E/\partial a_0$ against $2E^{an}/a_0$ | 0.074 of max | 0.065 of max | 0.08, 0.11: pass |
| Analytic, $a_0 = 0.005$ | 0.074 | 0.065 | pass |
| Self-consistency against $2E^{sim}/a_0$, $a_0 = 0.01$ / 0.005 | 3.8e-5 / 9.3e-6 | 2.0e-4 / 7.4e-5 | $10^{-3}$: pass |
| Residual ratio from $a_0 = 0.01$ to 0.005 | 4.03 | **2.67** | 3 to 5: $E_z$ passes, $E_r$ fails |
| Run-to-run noise (2 runs), relative to max, $a_0 = 0.01$ / 0.005 | 2.4e-5 / 1.3e-5 | 3.4e-4 / 2.0e-4 | - |
| Central FD in $a_0$, $e$ at $h = 10^{-2}$ / $10^{-3}$ / $10^{-4}$ | 8.6e-4 / 3.9e-3 / 2.7e-2 | 1.2e-2 / 5.3e-2 / 0.46 | $e \le \tau$ at each $h$, but rule 7 rejects all $h$ (noise term $\ge 9.9\cdot10^{-3}$) |

The analytic error of the derivative equals the forward PIC-versus-theory error (0.0741 against 0.0741 for $E_z$), as expected. Sampled $K(h)$ (tracker IDs divisible by 256) is 0 at all $h$, and the counts match: L4w is kink-free. The $E_r$ self-consistency residuals at both amplitudes are below the run-to-run noise of $E_r$ (stepper atomics noise after 1500 steps), so the $a_0^2$ scaling of the $E_r$ residual is not resolved. For $E_z$, the residuals are at the noise level, and the ratio is 4.03.

Tangent cost (`inversion/results/p5_tangent_cost.json`; best of 2 interleaved runs, without the first 20 steps):

| Configuration | Primal step | Step with $K = 1$ | Ratio |
|---|---|---|---|
| Twin-XS, 300 steps | 8.4 ms | 16.7 ms | 1.98 |
| Twin-S, 1000 steps, no EV | 9.2 ms | 18.0 ms | 1.96 |
| L4w, 1500 steps | 6.1 ms | 13.3 ms | 2.18 |

One tangent costs 1.0 to 1.2 primal steps. A gradient costs about 2 to 2.2 forward-run equivalents, inside the Section 8 estimate. Device memory of the Twin-S tangent run, 1000 steps: 2.9 GB in use.

Regression: L2 (`tests/test_autodiff_stepper.py`) passes with the changed stepper (largest ratios 0.11, 0.13, and 0.17 for Twin-XS 300, Twin-S 300, and Twin-S 1000 steps). The EV path of the changed stepper runs (Twin-XS, 60 steps). The stepper change alters the hash of `stepper.py`, so the cached P3 FD runs in `inversion/runs/p3` no longer match their provenance check (`ev_point.py`); the primal is unchanged (L2t).

Acceptance: accepted by the user on 2026-10-31, with deviations. The probes, L2t (Twin-XS and Twin-S), and L3 pass. L4 at the pass horizons fails at the plan steps for the fixture reason above; the gradient is verified where central FD resolves it ($N_t \le 30$, small $h$, $K(h) = 0$). L4w fails one criterion (the $E_r$ residual ratio), which the noise floor limits. The full-horizon L4 sweep stays on Twin-S in Phase P7, where the trapping sensitivity above is the main risk. The alternative, a Twin-XS variant with 16 cells per $\lambda_0$, was not chosen.

### 12.8 Re-plan of Phases P6 to P9 after M2

Date: 2026-10-31. The user approved decisions D29 to D32. A first draft was checked by three reviewers (facts, method, schedule); the approved version applies their findings. Measurements that changed the plan:

| # | Measurement (source) | Effect |
|---|---|---|
| 1 | Without EV, one tangent step costs 1.96 times a primal step on Twin-S (12.7). The cost with EV is not measured; an EV run takes 88.8 s, against about 18 s without EV. | Phase P7 step 3 measures it. Per dogbox iteration, AD needs about 2 run-equivalents and FD needs 3. |
| 2 | Twin-XS is chaotic at 3.85 cells per $\lambda_0$. FD resolves the derivative only up to $N_t = 30$, and AD matches FD there (12.7). | Twin-XS is the strict short-horizon gate (P6 step 3). Twin-S long horizons are reported. |
| 3 | Without EV, the Twin-S particle tangent grows by about $6\cdot10^2$ between steps 450 and 625, then stays near $10^6$ per J up to step 1000; the field tangent still grows at step 1000 (12.7). | Steps 1000 to 2000 and the EV case are measured in P7 step 4. The most sensitive particles stay linear only over about $10^{-7}$ in $\hat E_L$, against dogbox steps of $10^{-3}$ to $10^{-1}$, so P7 step 5 tests usability, not exactness. |
| 4 | The 1-D loss has one false minimum at $\hat E_L \approx 0.545$ to 0.55. FD at $h = 0.02$ stopped in it from the best design start (12.5). | Multistart stays mandatory. P7 step 5 includes a slice around it. |
| 5 | The P5 stepper change broke the provenance of the P3 FD cache (12.7), and P6 step 0 will break it again. `fd_recovery.py` hard-codes `runs/p3`. | D30 and P6 step 0. |

The draft is kept at `inversion/notes/replan-p6-p9-2026-10-31.md`.

### 12.9 Phase P6. EV tangents

Run on 2026-10-31 on physical GPU 1 (RTX 6000 Ada), Warp 1.18.0, code on top of commit `1c0f458`. Stage A: one tangent ($K = 1$, $E_L$).

#### Step 0. Deferred fixes, FD caches, and cell-key sampling

Changes:

1. `ev_ionize` (`fbpic/autodiff/ionization.py`) forms $P_5' = P_5 e^{-x_5}$ and $P_6' = P_6 e^{-x_6} + \Delta_{56}$, with the survival factors computed directly (Section 5.1). The P6 form is the same correction for the second transition; it was not in the M1 finding, but Section 5.1 requires it for both.
2. The emission record (`ledger["emitted_first_id"]`, 8 B per birth on the device) is optional: `attach(..., record_emissions=False)`. Only `verify_ev_invariants.py` turns it on.
3. `make_target.git_hash()` appends `-dirty` when tracked files differ from HEAD.
4. FD caches are keyed by code hash (D30): `fd_recovery.py` writes to `runs/fd_<code_sha256[:12]>/`, and `ev_point.py` takes the size and the target as arguments and stores the $Q_f$ that formed its moments. `runs/p3/` is read-only (mode `a-w`); its 311 runs carry the code hash `804032cdc3e2`.
5. `verify_param_gradient.py`: the cell keys of Section 6.3 rule 8 are recorded on the host. Option `--key-stride 256` records only the tracker IDs divisible by 256 and reports the sampled and total counts.

Checks after the change:

| Check | Result | Limit or previous value |
|---|---|---|
| EV open loop (`python verify_ev_open_loop.py`) | pass; $z$ = +0.99, +0.48, -1.42, -1.69 (unchanged to the printed digits); $\max\lvert P_5+P_6+P_7-1\rvert = 9.3\cdot10^{-15}$ over 200 steps | 3 SE; previous sum error $1.6\cdot10^{-15}$ |
| EV invariants on Twin-S (`python verify_ev_invariants.py`) | pass; cohort population error at most $2.0\cdot10^{-15}$; charge, ledgers, and ID follow-up pass | $10^{-14}$, $10^{-12}$ |

The larger population-sum error of the open loop is expected: $e^{-x}$ and $-\mathrm{expm1}(-x)$ are rounded separately, so $P_5' + \Delta_{56}$ differs from $P_5$ by a few ulp in each step, as a random walk. The invariant limit of Phase P3 still holds with a margin of 5.

Commit: `94d9ffb`.

#### Steps 1 to 5

Status: steps 1, 2, and 5 pass. Step 3 passes at $N_t = 20$; at $N_t = 28$ the user accepted it as a deviation on 2026-10-31 (below). Step 4 is reported.

Files:

1. `fbpic/autodiff/ionization.py`: the kernel `ev_tangent` (closed-form tangent of `ev_ionize` with the emission flags fixed, $K$ tangents per thread, out of place), the device functions `E_amplitude_jvp` and `adk_exponent_jvp`, and the host functions `ionize_tangent` and `charge_weight_tangents`. `ionize_and_emit(state, keep_inputs=True)` keeps the old EV arrays and the emission flags for the tangent.
2. `fbpic/autodiff/tangent.py`: `vay_ion_tangent`, the tangent of `push_ion_momenta_ev`.
3. `fbpic/autodiff/state.py`, `fbpic/autodiff/stepper.py`: the EV tangents live in `tangents.particles["nitrogen"]` as `pop5`, `pop6`, `pop7`, `w_pending`, so that every sort and realignment carries them. Hooks `tangent_ev:before` and `:after`.
4. `inversion/verify_one_step.py` and `inversion/state_snapshot.py` (`--ionization ev`), `inversion/verify_param_gradient.py` (`--ionization ev`, `--horizons`, `--key-stride`), `inversion/verify_crossing_probe.py`.
5. Tests: `tests/test_autodiff_ionization.py` (L1, 13 tests); `tests/test_autodiff_tangent_step.py` gets the EV case of L2t (a).

Design:

1. EV tangent (Section 5.3). $\dot E$ comes from $\dot{(E^2)}$ of `get_E_amplitude` differentiated line by line, with $\dot\gamma = u\cdot\dot u/\gamma$. Where $x_\ell = 0$, $\dot x_\ell = 0$ and $\dot E$ is not formed. With $s_\ell = e^{-x_\ell}$: $\dot p_\ell = s_\ell \dot x_\ell$, $\dot P_5' = s_5(\dot P_5 - P_5\dot x_5)$, $\dot P_6' = s_6(\dot P_6 - P_6\dot x_6) + \dot\Delta_{56}$, $\dot P_7' = \dot P_7 + \dot\Delta_{67}$, $\dot W' = \dot W + w(\dot\Delta_{56} + \dot\Delta_{67})$. An emitting ion moves $\dot W'$ into the birth-weight tangent and resets $\dot W$ to 0. The ion weight $w$ is constant (D21).
2. Ion push tangent. The Vay push reads $\mathbf E$ only as $c_E\mathbf E$ and $\mathbf B$ only as $c_B\mathbf B$, and both coefficients are proportional to the charge $e\,w_\text{tl}/w$. The charge tangent therefore enters as the field tangents $\dot{\mathbf E} + (\dot w_\text{tl}/w_\text{tl})\mathbf E$ and $\dot{\mathbf B} + (\dot w_\text{tl}/w_\text{tl})\mathbf B$ of the fixed-charge tangent `vay_jvp`.
3. Charge tangents. $\dot w_\text{tl} = w\sum_\ell \ell\dot P_\ell - \dot W$ from the current EV tangents: before the EV call for the ion push and the exchange-step `rho_prev` deposit, after it for the J and `rho_next` deposits. The ion deposit tangent uses the weight $w_\text{tl}$, the weight tangent $\dot w_\text{tl}$, and the charge $e$, as the primal deposit does.
4. Births. The new electrons take the tangents of $x, y, z, u_x, u_y, u_z$, `inv_gamma` of their parents, and the weight tangent $\dot W'$, by an index gather over `flatnonzero(emit)`. This is the parent order, which is the slot order of the primal.

Commands, from `lpa/FBPIC_Warp`, with `CUDA_VISIBLE_DEVICES=1`:

```bash
python -m pytest tests/test_autodiff_ionization.py -q -s
python -m pytest tests/test_autodiff_tangent_step.py -q -s -k bitwise
python inversion/verify_one_step.py --ionization ev --steps 20 28
python inversion/verify_param_gradient.py l4 twin_xs --ionization ev --steps 28 --horizons 20 28 --h 1e-6 1e-7 1e-8 --slice-points 0
python inversion/verify_param_gradient.py l4 twin_xs --ionization ev --steps 28 --horizons 28 --h 3e-8 1e-8 3e-9 1e-9 --slice-points 0 --tag _small_h
python inversion/verify_param_gradient.py nc twin_s
python inversion/verify_param_gradient.py l4 twin_s --ionization ev --steps 500 --horizons 460 500 --h 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 --slice-points 0 --key-stride 256
python inversion/verify_crossing_probe.py
```

Step 1, L1 (`tests/test_autodiff_ionization.py`). The EV fixture has 40 ions with rest-frame fields from $10^{11}$ V/m (two ions in the underflow band, $x = 0$ with $E > 0$) to $1.8\cdot10^{13}$ V/m ($x_5$ up to 43), random populations on the simplex, $W$ up to $10^{-4}w$, and half of the ions emitting. The ion-push fixture has 9 ions with $w_\text{tl}/w$ from 5 to 7.

| Kernel | Dot product with the Warp VJP, largest $e$ of 3 draws (limit $10^{-12}$) | `check_jvp`, best errors of 3 directions |
|---|---|---|
| `ev_tangent`: momenta / fields / populations and $W$ / all | 3.5e-17 / 5.2e-17 / 1.1e-17 / 2.4e-17 | 2.1e-10, 8.6e-11, 4.0e-10 |
| `vay_ion_tangent`: momenta / fields / $w_\text{tl}$ / all | 2.5e-17 / 5.6e-17 / 3.0e-17 / 2.4e-17 | 1.5e-9, 7.1e-10, 6.6e-10 |

The reference VJPs are those of `ev_ionize` (with `use_mask = 1`, so the emission branch is fixed) and of `push_ion_momenta_ev`. Separate checks: at $E = 0$ (6 ions), every output tangent is finite and $\dot P' = \dot P$, $\dot W' = \dot W$ or the birth tangent, exactly. The birth copy is an exact index gather: the old electron tangents are unchanged, and the new ones equal the parent tangents and $\dot W'$ bit for bit.

Step 2, EV tangents in `step`.

1. L2t (a) with EV: every tangent section, including `tangent_ev`, leaves the primal bitwise unchanged over 300 steps on Twin-XS and Twin-S (3363 sections per run). The Twin-XS ions leave the window before step 300; `ionize_tangent` then does nothing.
2. L3 (`inversion/results/p6_l3_twin_xs.json`), Twin-XS with `p_nz = 2`, nominal $\theta$. The base step records the emitting parent IDs; the tangent step and every FD step reuse them as `emit_mask`. The particle block also perturbs the EV arrays of the ions (not the ion weight, whose tangent is 0 under D21).

| Step | Decisions | Particle block, best $e$ | Field block, best $e$ | Masked particles (electrons, `elec_from_N`) |
|---|---|---|---|---|
| 20 (ordinary) | 3285 emissions | 8.6e-9, 5.0e-9 | 2.0e-12, 2.7e-12 | 278, 318 |
| 28 (exchange) | 4348 emissions | 1.7e-8, 2.2e-8 | 3.5e-12, 2.9e-12 | 292, 660 |

All 8 directions pass the pair rule at $10^{-6}$. The worst output of the particle block is the pending weight $W$ of the ions, with the $h^2$ truncation trend from $\epsilon = 10^{-3}$ to $10^{-5}$.

Step 3, strict L4 on Twin-XS with EV and frozen emissions (`inversion/results/p6_l4_twin_xs.json`). Births occur from step 0 (376 at step 0, 55366 up to step 28). Every base and FD run reuses the emitting parent IDs of the AD run at every step. The counts of rule 2 match in every run.

| $N_t$ | Term | $g^{AD}$ ($\hat\theta$) | $e$ at $h = 10^{-6}$ / $10^{-7}$ / $10^{-8}$ | $K(h)$ | Verdict |
|---|---|---|---|---|---|
| 20 | field | 1.13e4 | 5.5e-8 / 5.4e-8 / 8.4e-7 | 3 / 0 / 0 | pass |
| 20 | background | -1.24e8 | 1.4e-8 / 1.1e-5 / 4.1e-5 | 3 / 0 / 0 | pass |
| 20 | `elec_from_N` | 2.18e7 | 1.0e-8 / 3.3e-9 / 5.2e-8 | 3 / 0 / 0 | pass |
| 20 | total | -1.02e8 | 3.9e-8 / 1.3e-5 / 4.4e-5 | 3 / 0 / 0 | pass |
| 28 | field | -6.47e4 | 1.5e-2 / 1.0e-6 / 9.0e-7 | 50 / 2 / 0 | deviation (below) |
| 28 | background | -6.51e10 | 3.9e-2 / 2.4e-5 / 5.9e-6 | 50 / 2 / 0 | deviation |
| 28 | `elec_from_N` | 6.70e7 | 2.5e-4 / 4.7e-6 / 3.5e-7 | 50 / 2 / 0 | deviation |
| 28 | total | -6.50e10 | 3.9e-2 / 2.4e-5 / 5.9e-6 | 50 / 2 / 0 | deviation |

$\tau = 10^{-4}$ at every entry, except $1.3\cdot10^{-4}$ to $1.6\cdot10^{-4}$ at $N_t = 20$, $h = 10^{-8}$. At $N_t = 28$ only one plan $h$ has $K = 0$, so the pair rule of Section 6.4 is not met, and the fallback rule (the AD value inside the FD spread) fails for the background and the total term because all FD values lie on one side. Small-step diagnostic (`p6_l4_twin_xs_small_h.json`, not a plan gate): at $h = 3\cdot10^{-8}$, $10^{-8}$, $3\cdot10^{-9}$, $10^{-9}$, $K(h) = 0$ at all four, and $e \le 1.2\cdot10^{-5}$ for every term (total: 8.0e-7, 7.5e-6, 7.2e-6, 1.2e-5; $\tau = 10^{-4}$, $1.8\cdot10^{-4}$ for the field term at $10^{-9}$). Deviation accepted by the user on 2026-10-31: $N_t = 28$ passes on the small-step evidence. The cause is the Twin-XS fixture of Section 12.7, not the EV tangent.

Step 4, Twin-S report (`inversion/results/n_c_twin_s.json`: $n_c = 450$; `inversion/results/p6_l4_twin_s.json`). $J_{chk}$, frozen emissions, 1.64M births up to step 500, sampled cell keys (tracker IDs divisible by 256: 5.9M of 1.52G particle-stage keys per FD pair). AD costs 26.4 ms per step.

| $N_t$ | Total $e$ at $h = 10^{-2}$ / $10^{-3}$ / $10^{-4}$ / $10^{-5}$ / $10^{-6}$ / $10^{-7}$ / $10^{-8}$ | Sampled $K(h)$ |
|---|---|---|
| 460 ($n_c + 10$) | 3.0e-2 / 4.7e-3 / 1.8e-3 / 3.0e-4 / 3.3e-4 / 3.2e-5 / 1.2e-5 | 107635 / 12493 / 1495 / 291 / 5 / 0 / 1 |
| 500 ($n_c + 50$) | 1.8e-1 / 2.9e-2 / 5.1e-3 / 3.9e-4 / 2.9e-4 / 1.8e-6 / 9.5e-8 | 270153 / 32885 / 3605 / 579 / 22 / 0 / 1 |

The sampled $K(h)$ is 0 only at $h = 10^{-7}$, so this step is not a gate (Phase P6 step 4). Every term passes the fallback rule of Section 6.4 ("FD reference invalid ($K > 0$)": the AD value lies within the FD spread). The error falls with $h$ together with $K(h)$; it does not follow an $h^2$ trend, because the cell crossings, not the truncation, set it. The field, background, and `elec_from_N` terms behave like the total (largest $e$ at $h \le 10^{-7}$: 3.4e-5).

Step 5, crossing probe (`inversion/results/p6_crossing_probe.json`). Twin-XS, 20 steps. The hook `before_ionization` sets the gathered fields of every ion to 0, except one probe ion, which sees $E_x = a E_0 \min(1, (n+1)/20)$ with $E_0 = 5\cdot10^{12}$ V/m. The objective is the `elec_from_N` term of $J_{chk}$. Bisection gives the crossing $a_c = 0.939923718941973$, where the probe ion's emission steps change from (18, 19) to (17, 18, 19), with the jump $\Delta J = -1.477\cdot10^{-2}$. At $a_0 = a_c \pm 3\cdot10^{-4}$:

| Case | Prediction | Result |
|---|---|---|
| $a_c$ inside $[a_0 - h, a_0 + h]$ ($h = 10^{-3}$, $5\cdot10^{-4}$) | free − frozen $= (\Delta J + \mathrm{sign}(d)(g_{above} - g_{below})(\lvert d\rvert - h))/(2h)$ | relative error 1.9e-4 to 8.1e-4 (limit $10^{-3}$); against $\Delta J/(2h)$ alone, 0.4% to 1.5% |
| $a_c$ outside ($h = 2\cdot10^{-4}$, $10^{-4}$, $10^{-5}$) | free − frozen $= 0$ | at most $6.3\cdot10^{-12}$ relative (limit $10^{-9}$) |

$g_{above} = 1.2211$ and $g_{below} = 1.5294$ are the frozen slopes on the two branches. The second term is the $O(1)$ slope difference over the part of the stencil past $a_c$; Section 6.3 rule 5 states only the leading term $\Delta J/(2h)$. Pass.

Acceptance: steps 1, 2, and 5 pass; step 3 passes with the accepted deviation at $N_t = 28$; step 4 is reported. Production code is unchanged.

### 12.10 Phase P7. Loss Jacobian, horizon sweep, and Jacobian usability

Run on 2026-10-31 on physical GPU 1 (RTX 6000 Ada), code on top of commit `4c5ff56`. Stage A: one tangent ($K = 1$, $E_L$), $\alpha_D = 0$ (D29).

Files:

1. `fbpic/autodiff/loss.py`: `ResidualObjective`, `ResidualValue`, `observables`. `fbpic/autodiff/driver.py`: `value_and_grad` accepts a residual objective and returns `residuals`, `jacobian`, and `jacobian_hat` ($J = \lVert r_w\rVert^2$, gradient $2J_r^Tr_w$).
2. `inversion/residual_objective.py` (the objective of a twin target), `inversion/verify_batched_tangents.py` (step 2), `inversion/p7_jacobian.py` (steps 3, 5, 6), and `--loss-target` in `inversion/verify_param_gradient.py` (step 4).
3. Tests: `tests/test_autodiff_residual.py`.

Design. The 11 whitened residuals depend on the particles only through 13 weighted sums: $S_0 = \sum w\sigma$, $A_k = \sum w\sigma\xi_k$, and $B_k = \sum w\sigma\xi_k^2$. With $D = S_0 + Q_f/e$, $m_k = A_k/D$, $M_k = B_k/D$, and $s_k = \sqrt{\max(M_k - m_k^2, 0)}$, every observable of Section 2.4 is a function of these sums. The objective computes the sums and the $11 \times 13$ host Jacobian $L^{-1}\,\partial v/\partial(\text{sums})$, and contracts each tangent into the 13 sum tangents with 13 device reductions. It does not form the 11 particle gradients, which would take about 4 GB at 7M particles. Where $s_k = 0$ (an empty selection), $\dot s_k := 0$; every sum tangent is then 0 as well, because $\sigma' = 0$ outside the selection band.

Step 1, L1 (`python -m pytest tests/test_autodiff_residual.py -q -s`). The reference is the independent host code of Phase P3 (`observable.beam_statistics`, `whitened_residuals`), with a random lower-triangular $L$ so that the off-diagonal path is tested.

| Check | Result | Limit |
|---|---|---|
| Residuals against the P3 host code (2000 particles) | 3.5e-14 relative | 1e-11 |
| Residuals at the target beam | below 1e-9 | 1e-9 |
| `check_jvp`, normal beam, 3 directions | best errors 1.1e-9, 1.3e-9, 1.5e-9; pair rule holds | 1e-6 |
| Empty selection (all $\sigma = 0$) | residuals finite and equal to the host code; contraction exactly 0; FD exactly 0 | finite, FD |
| Nearly empty selection (one particle with $\sigma = 10^{-6}$) | residuals finite; best errors 1.5e-9, 1.6e-9, 4.4e-10 | 1e-6 |
| Explicit sum gradients against the contraction | 1.1e-15 | 1e-12 |

Step 2, $K = 2$ (`python inversion/verify_batched_tangents.py`; `inversion/results/p7_batched_tangents.json`). Twin-XS with EV, nominal $\theta$, $N_t = 10$ (7867 births). Seeds: $E_L$, and a synthetic seed (random interp fields, seed 0, $10^{-3}$ of the largest laser field). The difference of every tangent array (spectral fields per mode; particle and EV arrays by tracker ID) is relative to the largest entry of that array.

| Seed | $K = 2$ row against the $K = 1$ run | Repeated $K = 1$ runs ($d_{rep}$) | Limit $\max(10 d_{rep}, 10^{-13})$ |
|---|---|---|---|
| $E_L$ | 5.5e-13 (electron `inv_gamma`) | 7.5e-13 | 7.4e-12: pass |
| synthetic | 6.6e-13 (electron `inv_gamma`) | 6.6e-13 | 6.6e-12: pass |

The $K = 2$ rows differ from the $K = 1$ runs by no more than two $K = 1$ runs differ from each other (deposit atomics).

Step 3, cost (`python inversion/p7_jacobian.py cost`; `inversion/results/p7_jacobian.json`, key `cost`). Twin-S with EV at $\hat\theta = (0.588, 0.4)$, 2000 steps, T0 residual objective, $K = 1$. One primal run and one gradient run, twice, interleaved, each in its own process.

| Quantity | Primal EV run | `value_and_grad` ($K = 1$) |
|---|---|---|
| Wall time of the steps | 85.9 s, 86.0 s | 299.4 s, 298.1 s |
| CUDA mempool high-water mark, used / reserved | 2.53 / 3.42 to 3.49 GB | 3.78 / 6.64 to 6.85 GB |
| Device memory in use at the end | 3.9 to 4.0 GB | 7.1 to 7.3 GB |
| Host peak RSS | 1.42 GB | 1.02 to 1.22 GB |
| Births | 12.68M | 12.68M |

A gradient costs 3.47 run-equivalents, against 3 runs (center and $\pm h$) for one central-FD dogbox iteration in one parameter. Section 8 estimated 2 to 2.5; without EV, Phase P5 measured 1.96 (Section 12.7). Profile of 100 tangent steps at step 1200 (9.8M `elec_from_N`, synchronized hooks): 228 ms per step, of which the J deposit tangent takes 81 ms and the `rho_next` deposit tangent 40 ms, against 24 ms and 14 ms for the primal deposits. The deposit tangent of Phase P4 runs one thread per cell and loops over the particles of the cell; the EV births pack `elec_from_N` densely into few cells, so this loop serializes. A faster deposit tangent (for example, a split of dense cells) would change a verified Phase P4 kernel and is left as a follow-up. The gradient also does not need a second residual run: the residuals come from the same pass.

Step 4, horizon sweep (`python inversion/verify_param_gradient.py l4 twin_s --ionization ev --steps 2000 --horizons 1 10 50 460 500 650 1000 2000 --h 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 --slice-points 0 --key-stride 256 --loss-target T0 --tag _sweep`; `inversion/results/p7_sweep_twin_s.json`). Twin-S with EV at the nominal $\theta$, $J_{chk}$, frozen emissions (13.16M births), sampled cell keys (stride 256). The AD run took 157 ms per step. Total $J_{chk}$:

| $N_t$ | $g^{AD}$ ($\hat\theta$) | $e$ at $h = 10^{-2}$ / $10^{-3}$ / $10^{-4}$ / $10^{-5}$ / $10^{-6}$ / $10^{-7}$ | Sampled $K(h)$ |
|---|---|---|---|
| 1 | 2.5e-45 | (rounding noise; the laser has not reached the particles) | 0 at every $h$ |
| 10 | 5.4e-44 | (rounding noise) | 0 |
| 50 | -22.8 | 3.0e-6 / 1.4e-5 / 1.4e-5 / 5.0e-3 / 5.0e-3 / 0.40 | 56 / 7 / 2 / 0 / 0 / 0 |
| 460 | 2.14e9 | 3.0e-2 / 4.7e-3 / 1.8e-3 / 3.0e-4 / 3.3e-4 / 3.1e-5 | 107637 / 12492 / 1495 / 292 / 5 / 0 |
| 500 | 2.70e9 | 0.18 / 2.9e-2 / 5.1e-3 / 3.9e-4 / 2.9e-4 / 1.8e-6 | 270155 / 32884 / 3605 / 580 / 22 / 0 |
| 650 | 1.97e10 | 0.43 / 0.53 / 0.42 / 0.26 / 5.3e-2 / 5.7e-3 | 2.3M / 443k / 59k / 8113 / 688 / 57 |
| 1000 | 3.41e10 | 0.23 / 0.33 / 0.30 / 7.7e-2 / 2.5e-2 / 1.2e-2 | 19.7M / 8.0M / 2.1M / 415k / 57k / 5892 |
| 2000 | -2.26e13 | 1.0 at every $h$ | 109M / 59M / 20.7M / 5.1M / 942k / 166k |

At $N_t = 50$, rule 7 rejects $h \le 10^{-5}$ (noise term above $10^{-3}$); the usable $h$ agree to $1.4\cdot10^{-5}$ with $K > 0$. At $N_t = 460$ and 500, AD matches FD at the only $h$ with $K = 0$ (as in Phase P6 step 4). From $N_t = 650$, $e$ still falls with $h$, but $K(h) > 0$ at every $h$. Rule 2: from iteration 854 ($h = 10^{-2}$) to 1148 ($h = 10^{-7}$), the removed counts of the FD runs differ from those of the AD run, so FD at $N_t = 1000$ (for $h \ge 10^{-5}$) and at $N_t = 2000$ (every $h$) is not a frozen-decision reference. The five base runs reproduce every decision of the AD run.

Tangent growth (infinity norms per J of $E_L$, every 100 steps in the record):

| Step | 300 | 450 | 500 | 600 | 650 | 800 | 1000 | 1200 | 1500 | 1800 | 2000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $\lvert\partial u_x/\partial E_L\rvert$ | 4.5 | 2.8e3 | 2.3e4 | 1.1e6 | 2.9e6 | 1.1e6 | 1.2e6 | 2.3e6 | 1.5e8 | 8.7e10 | 1.7e11 |
| $\lvert\partial x/\partial E_L\rvert$ [m/J] | 8.1e-7 | 1.4e-4 | 2.8e-3 | 7.8e-2 | 0.37 | 5.8 | 6.8 | 22 | 53 | 1.6e4 | 1.3e5 |
| spectral $\lvert\partial E_p/\partial E_L\rvert$ | 0.77 | 164 | 376 | 1.2e3 | 1.9e3 | 3.8e3 | 4.7e3 | 5.6e3 | 2.3e4 | 2.0e6 | 1.6e7 |

Up to step 1200 the particle tangent saturates near $10^6$ per J, as without EV (Section 12.7). After step 1200 it grows by about a factor of 10 every 120 steps. The field tangent follows from step 1500.

Loss part at $N_t = 2000$ (T0 target, nominal $\theta$, $J = 204.7$). AD column against central FD ($\hat\theta$ units):

| Residual | AD | FD at $h = 10^{-2}$ / $10^{-3}$ / $10^{-4}$ / $10^{-5}$ / $10^{-6}$ / $10^{-7}$ |
|---|---|---|
| $\log(Q+Q_f)$ | 17.7 | 6.8 / 4.9 / 20.1 / 33.4 / -8.9 / 77.4 |
| $\langle x\rangle$ | 1.86e3 | 24.1 / 24.0 / 12.6 / 30.1 / -159 / 33.6 |
| $\langle u_x\rangle$ | 1.80e4 | 5.8 / 14.1 / -18.4 / -32.0 / -230 / 409 |
| $\langle\zeta\rangle$ | 447 | -75.6 / -75.4 / -99.0 / -147 / -13.4 / -92.7 |
| $\langle u_z\rangle$ | 58.3 | 5.4 / 4.0 / -22.7 / -42.8 / -13.8 / -149 |
| $\log\sigma_x$ | -950 | 38.3 / 39.3 / 30.7 / 98.6 / -23.1 / -62.9 |
| $\log\sigma_{u_x}$ | -3.65e3 | 37.5 / 43.1 / 35.8 / 78.6 / -80.1 / -140 |
| $\log\sigma_y$ | -648 | 38.9 / 46.6 / 59.3 / 37.4 / 21.2 / 38.4 |
| $\log\sigma_{u_y}$ | -3.60e3 | 38.9 / 46.2 / 57.5 / 13.5 / -17.0 / -15.4 |
| $\log\sigma_\zeta$ | -283 | 41.6 / 51.0 / 60.4 / 112 / 60.9 / 64.1 |
| $\log(\sigma_{u_z}/\langle u_z\rangle)$ | 1.06e3 | 22.2 / 33.8 / 72.3 / 190 / 126 / 494 |
| $J$ | 1.18e5 | -3.2e3 / -3.6e3 / -4.4e3 / -7.0e3 / -1.5e3 / -3.2e3 |

FD at $h = 10^{-2}$ and $10^{-3}$ agree with each other for most residuals (the smooth secant). Below $h = 10^{-4}$, FD scatters (roughness and changed removals). The AD column is 10 to 1000 times larger than the secant and has the opposite sign for 6 of 11 residuals and for $J$. The frozen-decision derivative at $N_t = 2000$ is dominated by the exponentially growing tangents after step 1200. Section 12.8 (row 3) anticipated this; step 5 measures what it means for dogbox.

Step 5, Jacobian usability on T0 at $N_t = 2000$ (`python inversion/p7_jacobian.py usability`, then `report`; `inversion/results/p7_jacobian.json`, key `usability`). Status: partial, checkpointed on 2026-10-31 when the session was shut down. Done: the three center runs (AD column, frozen and free central FD at small $h$, and the particle-contribution diagnostic), the complete 41-point slice around the T0 solution, 5 of 41 points of the false-minimum slice, and 1 of 41 of the far-field slice. Not done: the remaining 76 slice points (about 6.3 h at 298 s each). The completed runs are cached in `inversion/runs/p7_fe7188bf9100/`, and the stage resumes from them.

Solution slice ($\hat E_L = 0.538$ to 0.638, $\hat z_f = 0.4$; the slice minimum is $J = 3.74$ at 0.588):

| Quantity | Value |
|---|---|
| Column error $\lVert J_{AD} - J_{sm}\rVert / \lVert J_{sm}\rVert$ over the 41 points | median 2010, 90th percentile 7103 |
| Cosine of $J_{AD}$ and $J_{sm}$ | median -0.14, minimum -0.68 |
| Fixed-point shift $\delta$ at the T0 solution (0.588) | $3.5\cdot10^{-3}$ |
| Gradient $\partial J/\partial\hat E_L$ at 0.588, AD / smoothed | -1.34e4 / -11.7 |

$J_{sm}$ is the derivative of a cubic least-squares fit of each residual over the slice. At 0.588, the AD column is (-1962, 17299, -36696, 5924, -335, -7841, -28407, -5581, -11810, 564, -4400), and the smoothed column is (1.7, -108, -76, 4.8, 20, -31, -12, -30, -16, -36, -19).

Central FD at the three centers, relative to the AD column ($\lVert\mathrm{FD} - J_{AD}\rVert/\lVert J_{AD}\rVert$) and as a column norm:

| Center | $\lVert J_{AD}\rVert$ | Frozen FD, $h = 10^{-2}$ / $10^{-3}$ / $10^{-4}$ / $10^{-5}$: norm (relative to AD) | Free FD, $h = 10^{-3}$ / $10^{-4}$ / $10^{-5}$: norm |
|---|---|---|---|
| T0 solution (0.588), $J = 3.74$ | 5.2e4 | 172, 239, 273, 424 (0.998 to 0.999) | 239, 273, 425 |
| False minimum (0.545), $J = 12.0$ | 2.0e4 | 252, 222, 129, 262 (0.996 to 0.999) | 222, 129, 262 |
| Far field (0.75), $J = 104$ | 3.8e7 | 64, 67, 173, 423 (1.000) | 67, 173, 423 |

1. Frozen against free (birth-time term, Section 6.3 rule 4): frozen and free FD agree to 3 digits at every $h$ at all three centers. The birth-time term is negligible at these $h$.
2. AD against frozen FD (code against chaos): they differ by 99.6% to 100% at every $h$ down to $10^{-5}$. With the tangent growth of step 4 ($\lvert\partial x/\partial E_L\rvert \approx 10^5$ m/J at step 2000), the frozen-decision map is linear only over a range far below $h = 10^{-5}$, so frozen FD cannot check the AD column there. The AD column is the derivative of a chaotic map; it does not describe the response of the residuals to any usable step.
3. Particle contributions (center runs): the AD column is a sum over particles. For most residuals, 50% of $\sum_i\lvert c_{ji}\rvert$ comes from $10^4$ to $7\cdot10^4$ particles, and 90% from $2\cdot10^5$ to $4\cdot10^5$ of the 1.7M selected particles. The top 10 particles hold 1% to 25%. The error is spread over the beam, not carried by a few outliers, so clipping outliers does not remove it.
4. The value $\delta = 3.5\cdot10^{-3}$ meets the "AD usable" bar of Section 6.4 ($\delta \le 0.005$) formally, but $\delta$ is a poor indicator here. At the solution $r_w$ is small, so $\delta$ is small for any column; it does not see that $J_{AD}$ has the wrong direction (cosine below 0 at most points) and a magnitude 100 to $10^5$ times too large away from the solution. A dogbox iteration from the false minimum or the far field with $J_{AD}$ would take steps of about $10^{-3}$ to $10^{-6}$ of the secant step, in a direction that is wrong about half the time.

Step 6, $\epsilon$ comparison on the reduced Twin-S (`python inversion/p7_jacobian.py epsilon`; key `epsilon`). $\hat\theta = (0.588, 0.4)$, residuals against T0 ($J = 1829$, because the reduced configuration does not match T0; the columns, not $J$, are compared).

| $\epsilon$ | Births | $\lVert J_{AD} - J_{AD}^{\epsilon=0}\rVert / \lVert J_{AD}^{\epsilon=0}\rVert$ | Cosine to $\epsilon = 0$ | $\partial J/\partial\hat E_L$ (AD) | Run time | Mempool used / reserved |
|---|---|---|---|---|---|---|
| $10^{-4}$ | 0.91M | 3.12 | 0.32 | 1.30e5 | 93 s | 0.87 / 1.17 GB |
| $10^{-5}$ | 1.11M | 3.04 | -0.40 | 2.63e6 | 101 s | 0.93 / 1.34 GB |
| $10^{-6}$ | 1.33M | 1.58 | -0.46 | 1.13e6 | 111 s | 0.99 / 1.41 GB |
| 0 | 13.71M | 0 | 1 | -4.85e6 | 426 s | 4.58 / 8.99 GB |

At $\epsilon = 0$ (frozen equals free), central FD at $h = 10^{-2}$ and $10^{-3}$ differs from the AD column by 99.99% and 99.98%. The AD columns do not converge as $\epsilon \to 0$, and the $\epsilon = 0$ AD column disagrees with its own FD, as on the full Twin-S. So the frozen-against-free difference does not dominate the step 5 error, and $\epsilon \to 0$ or zero-weight creation is not the remedy (Phase P7 step 6). The error comes from the chaotic sensitivity of the trajectories at the full horizon.

Status of Phase P7 (2026-10-31): in progress, checkpointed, final state of the M3 session (commit `1119983` and the commit that adds this conclusion). Steps 1 and 2 pass. Steps 3, 4, and 6 are reported. Step 5 is partial (solution slice and centers complete). P7 is not accepted.

Conclusion of milestone M3 (supported by the data above):

1. The forward-mode gradient through the EV model is correct where FD can check it. L1, L3, and short-horizon L4 on Twin-XS pass (Section 12.9), and on Twin-S the AD gradient matches frozen-decision FD up to $N_t = 500$ ($n_c + 50$), with $e = 1.8\cdot10^{-6}$ to $3.1\cdot10^{-5}$ at the $h$ with $K = 0$ (steps 4 and Section 12.9 step 4).
2. Past step 1200 the particle dynamics is chaotic: the tangents grow by about a factor of 10 every 120 steps, to $1.7\cdot10^{11}$ per J at step 2000. This is the Lyapunov instability of the particle trajectories, not a code error and not the emission threshold: frozen and free FD agree to 3 digits, and $\epsilon \to 0$ does not change the picture (step 6).
3. At $N_t = 2000$ the exact frozen-decision Jacobian is unusable for the recovery. Its residual column is 10 to 1000 times the smooth FD secant, with a median column error of 2010 and a median cosine of -0.14 along the solution slice; the error is spread over $10^4$ to $4\cdot10^5$ particles. The bar $\delta \le 0.005$ of Section 6.4 is met only formally ($\delta = 3.5\cdot10^{-3}$ at the T0 solution, where $r_w$ is small).
4. Response: Track R (Section 7, D33), regular sensitivities by the particle-pdf method of Chung et al. (2020), started by the user on 2026-10-31. Until Track R passes its Phase R4, Phases P8 and P9 use the branch chosen at the end of Phase P7. The M3 recommendation for that branch is the FD fallback (branch 3), with the AD results reported only.

To close Phase P7, a successor must:

1. Read `inversion/notes/m3-state-2026-10-31.md` (rules, resume commands, open issues).
2. Check `nvidia-smi`, then run the rest of step 5 on one GPU: `python inversion/p7_jacobian.py usability` from `lpa/FBPIC_Warp`, with `CUDA_VISIBLE_DEVICES` set and the `fbpic-warp` environment. It reuses the 47 cached runs in `inversion/runs/p7_fe7188bf9100/` and runs the 76 remaining slice points (about 6.3 h). Do not edit `fbpic/autodiff/*.py`, `forward.py`, `twin_config.py`, or `residual_objective.py` before it ends: the cache key is their hash.
3. Run `python inversion/p7_jacobian.py report`, and add the false-minimum and far-field slice rows (column error, cosine, smoothed and FD columns) to step 5 above.
4. Give the user the step 5 and step 6 numbers. The user chooses the recovery branch of Section 6.4 (D32) and records it here; then mark Phase P7 accepted and commit locally.


## Appendix A. Unverified claims and required probes

| Claim | Probe | Phase |
|---|---|---|
| The $k=0$ columns of the $p=m$ and $p=m+1$ components stay zero for $m \ge 1$ | Closed in Phase P4 (Section 12.6): after 500 Twin-S steps, the $k_r$ index 0 column of mode 1 `Ep`, `Ez`, `Bp`, `Bz` is exactly 0. | P4 |
| Every field operator used by `step` is linear in the field arrays | Closed in Phase P4 (Section 12.6): all residuals below $3\cdot10^{-15}$ | P4 |
| Tangent kernels generic over $N_m$ agree with the 5-mode VJP clones | Closed in Phase P4 (Section 12.6): dot products below $4\cdot10^{-17}$ | P4 |
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
| Regular PIC sensitivities avoiding chaos | Chung, Bond, Cyr, Freund, JCP 400, 108969 (2020); OSTI 1570259 | Track R method |
| Gradients through chaotic or stiff simulators | Suh et al., ICML 2022, arXiv:2202.00817 | Why averaged exact gradients have high variance |
