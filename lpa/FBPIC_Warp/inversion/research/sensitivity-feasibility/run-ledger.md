# Run ledger: what was tried, what passed, and what failed

Investigation: sensitivity feasibility ([current.md](current.md)). Decisions cited as U01 to U27 are in [decisions.md](decisions.md). Checkpoint 2026-11-01 12:45 PST.

This ledger covers the whole investigation, 2026-10-26 to 2026-11-01. It links the existing experiment records instead of copying them: plan Section 12 (Phases P0 to P7), the milestone notes, `inversion/results/*.json`, the run logs `inversion/runs_*.log`, and the technical record Sections 12 and 13.

How it was built. Part A and Part C were compiled on 2026-11-01 by two read-only reader agents (workflow `checkpoint-run-ledger`) from the plan, notes, logs, results, caches, and `git`; they recomputed the cache code hashes. Part B was written in session 2553145c from the run outputs that it analyzed. Evidence tags: [M] measured from a saved output, [R] reported in plan or note text, [E] estimated. No run was repeated for this ledger.

Contents:

1. Part A: milestone phases P0 to P7 (2026-10-30 and 2026-10-31).
2. Part B: research-track studies and the GPU diagnostics of 2026-11-01.
3. Part C: reusable assets, operating pitfalls, and numbers to plan with.

One correction to an earlier statement in session 2553145c: host CPU contention slowed the GPU runs by at most 2.6 times (measured; Part C, "Operations and agent workflow"). The "about 20 times" figure applies only to some CPU agent jobs, and no clean measurement of it exists.

Resolution label note: label 8 (`p_nr` 8, `p_nt` 16) and label 16r (`p_nr` 16, `p_nt` 8) have the same total number of particles per cell. Because `p_nt` has no measured effect (labels 1 and 2t agree), the pair isolates `p_nr`; see finding 5.

## Part A. Milestone phases P0 to P7

Scope: the FBPIC-Warp forward-mode inversion work on branch `snidhan/dev-autodiff`, Phases P0 to P7 (milestones M1, M2, M3).
Work of 2026-11-01 (Q1 studies, `runs/q1`, `results/q1_*.json`, `runs_q1_*.log`) is in the research record and is not repeated here.
All paths are relative to `lpa/FBPIC_Warp/` in the worktree `dev-autodiff`.

### Source keys and evidence tags

| Key | Source |
|---|---|
| `PL:n` | `DIFFERENTIABLE_INVERSION_PLAN.md`, line `n` |
| `N1`, `N2`, `N3` | `inversion/notes/m1-state-2026-10-30.md`, `m2-state-2026-10-31.md`, `m3-state-2026-10-31.md` |
| `RP` | `inversion/notes/replan-p6-p9-2026-10-31.md` |
| `R:file#key` | `inversion/results/file`, JSON key |
| `L:file` | `inversion/file` (run log, git-ignored) |
| `C:dir` | `inversion/runs/dir` (run cache, git-ignored) |

Tags: **[M]** measured, which means read in this session from a saved output, log, cache, or `git`. **[R]** reported, which means taken from plan or note text and not re-read from a raw output. **[E]** estimated.

Common environment (PL:1050-1055, N2 rules) [R]: physical GPU 1, RTX 6000 Ada, 49140 MiB; `CUDA_VISIBLE_DEVICES=1`; Conda `fbpic-warp`; Warp 1.18.0, CuPy 14.2.0, NumPy 2.2.6, SciPy 1.17.1. GPU runs execute outside the agent sandbox.

Common configuration (PL:1081-1098; `inversion/configs/*.json`) [M for grid and particle keys]:

| Size | Grid | Particles per cell | \(N_t\) | Laser and plasma | EV \(\epsilon\) |
|---|---|---|---|---|---|
| Twin-S | 800 x 50, \(N_m = 2\), 16 cells per \(\lambda_0\) | `p_nz=1`, `p_nr=2`, `p_nt=8` (T0 truth uses `p_nz=2`) | 2000 | \(a_0 = 4\) at \(E_L = 0.27\) J; \(\hat\theta^\ast = (0.6, 0.4)\) | \(10^{-4}\) |
| Twin-XS | 96 x 32, \(N_m = 2\), 3.85 cells per \(\lambda_0\) | `p_nz=1`, `p_nr=2`, `p_nt=8`; L3 uses `p_nz=2` | 300 | preloaded plasma; traps no beam | \(10^{-4}\) |

Twin-S particle scale [M from `R:p3_ev_runs.json`, R for loaded count]: about 2.69M loaded macroparticles over a run (PL:1254); 13.16M EV births and 6.66M live `elec_from_N` at \(N_t\) at the nominal point; 12.68M births at \(\hat\theta = (0.588, 0.4)\) (`L:runs_p7_cost.log`); 1.7M selected beam particles in the P7 center runs (PL:1848, [R]).

### 1. Ledger

Status values: passed, failed, partial, stopped, deviation accepted by the user, reported (a report step with no pass rule).
Failure meaning: **impl** = implementation defect; **approach** = unsuitable method or fixture; **gate** = gate or metric design issue.

#### 1.1 Milestone M1 (P0 to P3), 2026-10-30, P3 gate finished about 03:14 on 2026-10-31

| ID | Phase, step | Question | Configuration | Script or command | Output | Status | Decisive numbers | Failure meaning | Tag, source |
|---|---|---|---|---|---|---|---|---|---|
| M1-01 | P0 L0 | Do the existing autodiff, fused, and end-to-end tests pass? | no twin; commit `74c21bd` | `python -m pytest <file> -q` per file | plan text only | passed | 132 autodiff, 139 fused, 2 end-to-end tests pass | none | [R] PL:1045-1073; no log saved |
| M1-02 | P1 (first try) | Can production runs be split into 100-step calls? | Twin-S | `verify_twin_p1.py` (early form) | discarded | failed, abandoned | each call forces an exchange at its first iteration, which changes the physics | impl (harness) | [R] PL:1114 |
| M1-03 | P1 steps 4 to 6 | Aliasing, weights, trapping at nominal, 4 corners, 5 starts | Twin-XS and Twin-S, stochastic ionization | `verify_twin_p1.py twin_xs`, `twin_s` | `R:p1_twin_s.json#trapping`, `L:runs_p1.log`, `C:p1` | passed | \(\max\lvert\rho_1\rvert/\max\lvert\rho_0\rvert \le 3.0\cdot10^{-15}\); trapping \(Q/Q_{nom}\) 0.274 to 1.807, \(n(\sigma>0.5) \ge 20183\) | none | [M] trapping; [R] aliasing PL:1116-1123 |
| M1-04 | P1 step 7 | Are \(E_L\) and \(z_f\) identifiable? | Twin-S, \(h = 0.05\), Philox seeds 101 to 104, T2 truth `p_nz=2` | `identifiability.py selection trapping sensitivity report` | `R:p1_twin_s.json#identifiability` | passed | \(\mathrm{cond}(G) = 2.18\); \(\sqrt{\mathrm{diag}\,G^{-1}} = (0.0056, 0.0059)\), limit 0.02 | none | [M] |
| M1-05 | P1 step 8 | Nominal Twin-S cost | Twin-S stochastic, 2000 steps | `verify_twin_p1.py twin_s` | `R:p1_twin_s.json#nominal_run` | reported | 22.5 s per run; 15.5 pC; mempool peak 0.49 GB | none | [R] PL:1154-1162 |
| M1-06 | P2 L2 | Does the functional stepper reproduce production? | Twin-XS 300 steps; Twin-S 300 and 1000 steps; no ionization | `pytest tests/test_autodiff_stepper.py` | plan text only | passed, with a planner deviation | largest ratios 0.103, 0.149, 0.149; \(d_{AA}\) is the largest of 3 production pairs | gate (one pair underestimates Twin-XS run-to-run spread, 7.7e-4 vs 1e-5) | [R] PL:1186-1203; no log saved |
| M1-07 | P3 step 2 | EV open loop against 32 Philox seeds | 2400 ions, 200 steps | `verify_ev_open_loop.py` | plan text only | passed | \(\lvert z\rvert \le 1.69\) (limit 3) | none | [R] PL:1224-1233 |
| M1-08 | P3 steps 3, 4 | Charge, cohort, ledger, ID invariants | Twin-S at \(\theta^\ast\), 2000 steps | `verify_ev_invariants.py` | `L:runs_p3_invariants.log` | passed | all 4 checks true, "PASS"; population error \(2.2\cdot10^{-15}\) | none | [M] log; [R] numbers PL:1237-1244 |
| M1-09 | P3 step 5 | EV run cost and births | Twin-S nominal, corner (0,0), corner (1,1); \(\epsilon = 10^{-4}\) | `ev_runs.py twin_s <point>` | `R:p3_ev_runs.json` | reported | births 13.16M / 11.88M / 16.30M; 88.8 / 87.6 / 101.4 s; EV \(Q\) within 1% of stochastic | none | [M] |
| M1-10 | P3 step 6 | Immediate emission (\(\epsilon = 0\)) fits? | Twin-XS nominal; reduced Twin-S (N in \(z_c \pm 2\) µm) | `ev_runs.py` | `R:p3_ev_runs.json#twin_xs:immediate`, `#twin_s_reduced:immediate` | reported | 2.53M births, 6.7 s; 13.75M births, 224 s | none | [M] |
| M1-11 | P3 step 7 | Minimal loss finite on empty and near-empty beams | host only | `verify_loss_minimal.py` | plan text only | passed | closed form 48.297, relative error < 1e-12 | none | [R] PL:1265 |
| M1-12 | P3 step 8 | EV mean-charge closure within 2 SE | Twin-XS nominal; Twin-S at \(\theta^\ast\) vs T0; 16 Philox seeds | `verify_ev_closure.py <size>` | `R:p3_closure.json` | Twin-S passed; Twin-XS failed, deviation accepted by the user | max \(\lvert\Delta\rvert\)/SE: Twin-S 1.40, Twin-XS 4.34 | approach (mobile vs immobile ions measure ion motion, not closure error); open for P9 T2 | [M] numbers; [R] decision PL:1305 |
| M1-13 | P3 step 9 | T0 target | Twin-S EV at \(\theta^\ast\), `p_nz=2`, NumPy seed 7 | `make_target.py twin_s T0` | `inversion/targets/twin_s_T0_Nt2000.npz` | done | SHA-256 `90ee4adeb6e6`; \(Q_y = 14.80\) pC; 25.3M births; 283 s | none | [M] hash; [R] rest PL:1307 |
| M1-14 | P3 noise floor | Is the EV twin deterministic? | 3 runs at \(\theta^\ast\) | `fd_recovery.py noise` | `R:p3_fd_recovery.json#noise` | passed | \(J = 1.5116\cdot10^{-3}\), \(\sigma_J = 2.1\cdot10^{-12}\) | none | [M] |
| M1-15 | P3 step 10 | 1-D slices of the minimal loss | 26 runs, offsets \(\pm 0.30\) | `fd_recovery.py slices` | `R:p3_fd_recovery.json#slices`, `#slices_central` | reported | one basin; central-moment \(E_L\) minimum moves to about 0.55 | none | [M] |
| M1-16 | P3 gate, first try | L-BFGS-B recovery as planned (D14) | raw moments, central FD \(h = 0.02\), start (0.164, 0.323) | `fd_recovery.py` (L-BFGS-B) | `R:p3_fd_recovery.json#recovery_lbfgsb_stopped`, `L:runs_p3_lbfgsb.log` | stopped after 7 evaluations | first step to corner (1, 0); \(\hat z_f\) pinned at 0; best \(J = 0.072\) at (0.603, 0) | approach | [M] |
| M1-17 | P3 gate, trf | Trust region from the same start | `least_squares(trf)`, central moments, unwhitened | `fd_recovery.py` | `R:p3_fd_recovery.json#recovery_trf_central_stopped`, `L:runs_p3_trf_central.log` | stopped (gate redesigned) | (0.432, 0.231), \(J = 0.0225\) after 3 accepted steps | none (not a failure) | [M] |
| M1-18 | P3 step 11, first build | Whitening from 1 point plus Poisson bootstrap (\(B = 256\)), with ECNoise | 13 observables | `target_covariance.py build` (early form), `fd_recovery.py edge design` | `L:runs_gate.log`, `R:p3_fd_recovery.json#gate_design_13obs` | failed, replaced | \(J(\theta^\ast) = 1128\) on T0; 1124 from the energy-spread residual (whitened \(-33.5\)); gate (a) false | gate (weights too strict by about 70) | [M] \(J\), gate flag; [R] split PL:1349 |
| M1-19 | P3 step 11 | 5-point model-error whitening | T0, nominal, 4 corners at `p_nz=2` vs `p_nz=1` | `target_covariance.py run <point>`, `build` | `inversion/targets/twin_s_T0_Nt2000_whitening.npz`, `L:runs_cov_*.log` | done | SHA-256 `719cbc0e9ce2`; \(J(\theta^\ast) = 6.77\) on T0; residuals within \(\pm 1.7\) | none | [M] hash, \(J\) (`L:runs_gate3.log`); [R] \(\pm 1.7\) |
| M1-20 | P3 step 11 item 2 | Identifiability with the 11 loss observables | P1 runs | `identifiability.py report_loss` | `R:p1_twin_s.json#identifiability_loss_observables` | passed | \(\mathrm{cond}(G) = 1.41\), \(\sqrt{\mathrm{diag}\,G^{-1}} = (0.0123, 0.0110)\) | none | [M] |
| M1-21 | P3 step 12 | T0′ control target | Twin-S EV at \(\theta^\ast\), `p_nz=1` | `make_target.py twin_s T0p` | `inversion/targets/twin_s_T0p_Nt2000.npz` | done | SHA-256 `88783bf73c69`; \(J(\theta^\ast) = 4.7\cdot10^{-15}\) | none | [M] |
| M1-22 | P3 gate (a) edge | Is the \(z_f = 0\) bound a local minimum? | (0.603082, 0), one-sided stencil, \(h = 0.02, 0.1\) | `fd_recovery.py edge` | `R:p3_fd_recovery.json#gate_edge` | reported | new loss \(dJ/d\hat z_f = +948\) (\(h = 0.02\)), \(+34.5\) (\(h = 0.1\)); \(J = 116.5\) | none | [M] |
| M1-23 | P3 gate (a) design | 25-point design finds the basin? | 5x5 grid 0.1 to 0.9 on T0′ | `fd_recovery.py --target T0p design` | `R:p3_fd_recovery.json#gate_design_T0p`, `L:runs_gate3.log` | scored passed, but wrong | grid minima (0.5, 0.3) \(J = 25.7\), (0.7, 0.5), (0.9, 0.7) | gate (criterion "within one grid step" accepts a point in another basin) | [M] |
| M1-24 | P3 gate (b), 13 obs | 2-D dogbox converges? | T0′, FD steps (0.02, 0.05), `x_scale="jac"`, `xtol=2e-3` | `fd_recovery.py --target T0p design fine dogbox` | `R:p3_fd_recovery.json#gate_dogbox_T0p_13obs`, `L:runs_gate2.log` | failed | (0.5, 0.3) -> (0.5117, 0.2810), \(J = 26.98\), error (0.088, 0.119); (0.9, 0.7) -> \(J = 157\); third start stopped | approach (\(\langle y\rangle\), \(\langle u_y\rangle\) jitter) | [M] |
| M1-25 | P3 gate (b), 11 obs | 2-D dogbox after D27 | as M1-24, 11 observables | as M1-24 | `L:runs_gate3.log` only | stopped (user chose Stage A, D28) | 25.75 -> 23.73 at (0.5062, 0.2860), rejected 23.86, 23.79 at (0.5075, 0.2859) | approach (false minimum, basin radius about 0.04 vs grid spacing 0.2) | [M] log; no results key |
| M1-26 | P3 step 15, Stage A | 1-D recovery of \(E_L\) and \(z_f\) | T0′ and T0; 10-point design; dogbox, FD steps (0.02, 0.05) | `fd_recovery.py --target T0p one_E_L one_z_f`; `--target T0 ...` | `R:p3_fd_recovery.json#gate_one_E_L_T0p`, `#gate_one_z_f_T0p`, `#gate_one_E_L_T0`, `#gate_one_z_f_T0`; `L:runs_one.log` | passed | \(E_L\): start 0.85 -> 0.599985 (error \(1.46\cdot10^{-5}\), 5 Jacobians, 1251 s); start 0.55 -> 0.5453, \(J = 12.55\) (false minimum); farthest 0.164 -> 0.60013. \(z_f\): all 3 starts within \(1.3\cdot10^{-4}\). T0 bias \(-0.012\) (\(E_L\), farthest start), \(-0.016\) (\(z_f\)) | none | [M] |
| M1-27 | P3 code review | Independent review of P3 code | 14 agents | workflow | PL:1419-1430 | 6 minor confirmed, 5 refuted | 2 fixed before the gate, 3 deferred to P6 step 0, 1 reported | impl (minor) | [R] |

#### 1.2 Milestone M2 (P4, P5), 2026-10-31

| ID | Phase, step | Question | Configuration | Script or command | Output | Status | Decisive numbers | Failure meaning | Tag, source |
|---|---|---|---|---|---|---|---|---|---|
| M2-01 | P4 L1 kernels | Gather, Vay, position, deposit tangents correct? | fixtures, \(N_m = 2, 5\), \(K = 3\) and 1 | `pytest tests/test_autodiff_tangent.py` | `C:p4/tangent_l1.log` | passed | "34 passed, 1 skipped"; dot product \(\le 4.1\cdot10^{-17}\) | none | [M] log; [R] table PL:1461-1467 |
| M2-02 | P4 L1 field operators | Field operators linear through `FieldTangents.apply`? | Twin-S grid and 2912 x 1080 | `pytest tests/test_autodiff_field_tangents.py` | `C:p4/field_linearity.log` | passed | "2 passed"; residual \(\le 2.5\cdot10^{-15}\) | none | [M] log; [R] values |
| M2-03 | P4 \(k = 0\) probe | Round trip of spectral E, B | Twin-S nominal, no ionization, 500 steps | test in `test_autodiff_field_tangents.py` | `R:p4_k0_round_trip.json` | passed | round trip \(\le 1.7\cdot10^{-12}\) (mode 1 `Ep`) | none | [R] PL:1488-1491; JSON present |
| M2-04 | P5 probes 1, 3; J_chk L1 | Laser seed and check functional correct? | Twin-XS, Twin-S | `pytest tests/test_autodiff_laser.py`, `test_autodiff_loss.py` | plan text only | passed | probe 1 difference 0; probe 3 \(\le 6.3\cdot10^{-16}\); J_chk best \(\le 1.1\cdot10^{-9}\) | none | [R] PL:1530-1532 |
| M2-05 | P5 L2t | Tangents leave the primal unchanged | Twin-XS, Twin-S, 300 steps | `pytest tests/test_autodiff_tangent_step.py` | `C:p5/l2t_bitwise.log`, `C:p5/l2t_smoke.log` | passed | 3063 sections bitwise; smoke ratios 0.058, 0.13 | none | [M] "2 passed" each; [R] ratios |
| M2-06 | P5 L3 | One-step tangent vs FD | Twin-XS `p_nz=2`, steps 30, 14, 154 | `python inversion/verify_one_step.py` | `R:p5_l3_twin_xs.json`, `C:p5/l3_twin_xs.log` | passed, with a deviation | best \(e\) \(10^{-13}\) to \(7.8\cdot10^{-9}\); 12 directions pass at \(10^{-6}\) | gate (per-array normalization failed on N-ion `inv_gamma`; replaced by vector form) | [M] log; [R] deviation PL:1547 |
| M2-07 | P5 L4, plan \(h\) | Full-parameter gradient vs FD on Twin-XS | Twin-XS, no ionization, \(n_c = 18\), \(N_t \in \{1, 10, 28, 50, 68, 218\}\), report 300, \(h \in \{10^{-2}, 10^{-3}, 10^{-4}\}\) | `verify_param_gradient.py nc twin_xs`; `l4 twin_xs` | `R:p5_l4_twin_xs.json`, `R:n_c_twin_xs.json`, `C:p5/l4_twin_xs.log` | failed, deviation accepted by the user | \(K(h) > 0\) from \(N_t = 10\); \(e \approx 1\) from \(N_t = 28\); removal counts differ at iteration 112 | approach (Twin-XS chaotic at 3.85 cells per \(\lambda_0\)) | [M] |
| M2-08 | P5 L4, small \(h\) | Does AD match FD where \(K(h) = 0\)? | as M2-07, \(N_t \le 70\), \(h = 10^{-4}\) to \(10^{-8}\) | `... l4 twin_xs --steps 70 --h 1e-4 ... 1e-8 --tag _small_h` | `R:p5_l4_twin_xs_small_h.json`, `C:p5/l4_twin_xs_small_h.log` | reported (diagnostic) | \(e = 6\cdot10^{-8}\) (\(N_t = 10\)), \(2.2\cdot10^{-6}\) (28), \(1.8\cdot10^{-3}\) (30); no \(K = 0\) \(h\) for \(N_t \ge 40\) | none | [M] files; [R] summary PL:1563-1572 |
| M2-09 | P5 tangent growth | Twin-S tangent growth without EV | Twin-S, 1000 steps | `diag_tangent_growth.py` | `R:p5_tangent_growth_twin_s.json` | reported | \(\lvert\partial u_x/\partial E_L\rvert\) saturates near \(10^6\) J\(^{-1}\) after step 625 | none | [R] PL:1578; JSON present |
| M2-10 | P5 L4w | Linear-wakefield derivative | upstream linear wakefield, \(a_0 = 0.01, 0.005\), 1500 steps | `verify_linear_wakefield.py forward`, `check` | `R:p5_l4w.json`, `R:p5_l4w_noise.json` | failed one criterion, deviation accepted by the user | \(E_r\) residual ratio 2.67 (limit 3 to 5); \(E_z\) 4.03 | gate (residuals below run-to-run noise \(3.4\cdot10^{-4}\)) | [M] |
| M2-11 | P5 tangent cost | Cost of one tangent | Twin-XS 300, Twin-S 1000 no EV, L4w 1500 | `diag_tangent_cost.py` | `R:p5_tangent_cost.json` | reported | step ratio 1.98, 1.96, 2.18 | none | [M] |
| M2-12 | P5 L2 regression | Changed stepper still matches production | as M1-06 | `pytest tests/test_autodiff_stepper.py` | `C:p5/l2_regression.log` | passed | "3 passed"; ratios 0.11, 0.13, 0.17 | none | [M] log; [R] ratios |

#### 1.3 Milestone M3 (P6, P7), 2026-10-31

| ID | Phase, step | Question | Configuration | Script or command | Output | Status | Decisive numbers | Failure meaning | Tag, source |
|---|---|---|---|---|---|---|---|---|---|
| M3-01 | P6 step 0 | Deferred M1 fixes keep EV checks valid | survival \(e^{-x}\), optional record, dirty marker, FD cache by code hash, key stride | `verify_ev_open_loop.py`, `verify_ev_invariants.py` | `L:runs_p6_invariants.log` | passed | invariants "PASS"; open-loop \(z\) unchanged; population sum error \(9.3\cdot10^{-15}\) | none | [M] log; [R] open loop PL:1642 |
| M3-02 | P6 step 1 L1 | EV and ion-push tangents | 40-ion and 9-ion fixtures | `pytest tests/test_autodiff_ionization.py` | plan text only | passed | 13 tests; dot product \(\le 5.6\cdot10^{-17}\); `check_jvp` \(\le 1.5\cdot10^{-9}\) | none | [R] PL:1681-1688 |
| M3-03 | P6 step 2 L2t (a) | EV tangents leave the primal unchanged | Twin-XS, Twin-S, 300 steps | `pytest ... -k bitwise` | plan text only | passed | 3363 sections bitwise | none | [R] PL:1692 |
| M3-04 | P6 step 2 L3 | One EV step vs FD | Twin-XS `p_nz=2`, steps 20 (3285 births), 28 (exchange, 4348 births) | `verify_one_step.py --ionization ev --steps 20 28` | `R:p6_l3_twin_xs.json`, `L:runs_p6_l3.log` | passed | 8 of 8 directions; best \(e\) \(2.0\cdot10^{-12}\) to \(2.2\cdot10^{-8}\) | none | [M] |
| M3-05 | P6 step 3 strict L4 | EV gradient vs frozen-emission FD | Twin-XS EV, \(N_t \in \{20, 28\}\), \(h \in \{10^{-6}, 10^{-7}, 10^{-8}\}\); 55366 births to step 28 | `verify_param_gradient.py l4 twin_xs --ionization ev --steps 28 --horizons 20 28 --h 1e-6 1e-7 1e-8 --slice-points 0` | `R:p6_l4_twin_xs.json`, `L:runs_p6_l4_xs.log` | \(N_t = 20\) passed; \(N_t = 28\) failed, deviation accepted by the user | \(N_t = 28\): \(K = 50 / 2 / 0\); total \(e = 2.4\cdot10^{-5}\), \(5.9\cdot10^{-6}\) at \(10^{-7}, 10^{-8}\); fallback rule "FAIL" for background and total | gate (pair rule needs 2 adjacent \(K = 0\) \(h\); fixture chaos) | [M] |
| M3-06 | P6 step 3 small \(h\) | Same at smaller \(h\) | \(N_t = 28\), \(h = 3\cdot10^{-8}\) to \(10^{-9}\) | `... --h 3e-8 1e-8 3e-9 1e-9 --tag _small_h` | `R:p6_l4_twin_xs_small_h.json`, `L:runs_p6_l4_xs_small_h.log` | passed (diagnostic) | \(K = 0\) at all 4 \(h\); \(e \le 1.2\cdot10^{-5}\) | none | [M] |
| M3-07 | P6 step 4 | Twin-S report past \(n_c\) | Twin-S EV, \(n_c = 450\), \(N_t \in \{460, 500\}\), \(h = 10^{-2}\) to \(10^{-8}\), key stride 256; 1.64M births | `verify_param_gradient.py nc twin_s`; `l4 twin_s --ionization ev --steps 500 --horizons 460 500 ... --key-stride 256` | `R:p6_l4_twin_s.json`, `R:n_c_twin_s.json`, `L:runs_p6_l4_s.log` | reported (not a gate) | sampled \(K = 0\) only at \(h = 10^{-7}\); total \(e\) at \(10^{-7}\): \(3.2\cdot10^{-5}\) (460), \(1.8\cdot10^{-6}\) (500); 26.4 ms per AD step | none | [M] |
| M3-08 | P6 step 5 | Crossing probe, free minus frozen FD | Twin-XS, 20 steps, one probe ion | `verify_crossing_probe.py` | `R:p6_crossing_probe.json`, `L:runs_p6_crossing.log` | passed | \(a_c = 0.9399237\); jump \(\Delta J = -1.477\cdot10^{-2}\); relative error \(1.9\cdot10^{-4}\) to \(8.1\cdot10^{-4}\) inside, \(\le 6.3\cdot10^{-12}\) outside | none | [M] |
| M3-09 | P7 step 1 L1 | Residual objective and contraction | 2000-particle fixtures, empty and near-empty selection | `pytest tests/test_autodiff_residual.py` | plan text only | passed | 6 tests; residuals vs P3 host code \(3.5\cdot10^{-14}\) | none | [R] PL:1749-1758 |
| M3-10 | P7 step 2 | \(K = 2\) equals two \(K = 1\) runs | Twin-XS EV, \(N_t = 10\), 7867 births; seeds \(E_L\) and synthetic | `verify_batched_tangents.py` | `R:p7_batched_tangents.json` | passed | \(5.5\cdot10^{-13}\) and \(6.6\cdot10^{-13}\) vs limits \(7.4\cdot10^{-12}\), \(6.6\cdot10^{-12}\) | none | [M] |
| M3-11 | P7 step 3 | Gradient cost with EV | Twin-S EV, \(\hat\theta = (0.588, 0.4)\), 2000 steps, T0 residuals, 12.68M births | `p7_jacobian.py cost` | `R:p7_jacobian.json#cost`, `L:runs_p7_cost.log`, `C:p7_fe7188bf9100/cost_*` | reported | 3.47 run-equivalents (299 s vs 86 s); mempool 3.78 / 6.85 GB; deposit tangents 81 + 40 of 228 ms per step | none (deposit tangent serializes on dense cells; follow-up) | [M] ratio, memory; [R] profile PL:1779 |
| M3-12 | P7 step 4 | Horizon sweep to \(N_t = 2000\) | Twin-S EV nominal, 13.16M births, \(N_t \in \{1, 10, 50, 460, 500, 650, 1000, 2000\}\), \(h = 10^{-2}\) to \(10^{-7}\), stride 256, loss vs T0 at 2000 | `verify_param_gradient.py l4 twin_s --ionization ev --steps 2000 --horizons ... --loss-target T0 --tag _sweep` | `R:p7_sweep_twin_s.json`, `L:runs_p7_sweep.log` | reported | match to \(N_t = 500\); from 650, \(K > 0\) at every \(h\); at 2000, \(e = 1.0\) at every \(h\); removal counts differ from iteration 854 to 1148; tangent \(\lvert\partial u_x/\partial E_L\rvert = 1.7\cdot10^{11}\) at step 2000; AD loss column 10 to 1000 times the FD secant, opposite sign in 6 of 11 | approach (chaotic trajectory map at full horizon; not a code defect) | [M] \(e\), \(K\); [R] growth table PL:1798-1802 |
| M3-13 | P7 step 6 | Does \(\epsilon \to 0\) fix the AD column? | reduced Twin-S, \(\hat\theta = (0.588, 0.4)\), \(\epsilon \in \{10^{-4}, 10^{-5}, 10^{-6}, 0\}\); FD at \(\epsilon = 0\), \(h = 10^{-2}, 10^{-3}\) | `p7_jacobian.py epsilon` | `R:p7_jacobian.json#epsilon`, `L:runs_p7_epsilon.log`, `C:p7_fe7188bf9100/eps*` | reported | column vs \(\epsilon = 0\): 3.12, 3.04, 1.58; AD at \(\epsilon = 0\) vs its FD: 0.99986, 0.99976 relative | approach (not the remedy) | [M] |
| M3-14 | P7 step 5 | Is the AD Jacobian usable for dogbox at \(N_t = 2000\)? | Twin-S EV vs T0; 3 slices of 41 points over \(\pm 0.05\) at 0.588, 0.545, 0.75; centers with frozen and free FD \(h = 10^{-2}\) to \(10^{-5}\) | `p7_jacobian.py usability`, then `report` | `R:p7_jacobian.json#usability`, `L:runs_p7_usability.log`, `C:p7_fe7188bf9100/center_*`, `grad_*` | partial, then stopped by D34 | solution slice complete: column error median 2010, p90 7103; cosine median \(-0.144\); \(\delta = 3.50\cdot10^{-3}\); \(\partial J/\partial\hat E_L\) AD \(-1.34\cdot10^4\) vs smoothed \(-11.7\); false-minimum slice 5 of 41; far-field 1 of 41 | approach (chaos), plus gate (\(\delta\) metric) | [M] |
| M3-15 | P7 acceptance | Choose the recovery branch (D32) | n/a | n/a | PL:1862, PL:986 | not reached; deferred by D34 | M3 recommendation: FD fallback, AD reported only | n/a | [R] |

### 2. Failed or abandoned attempts

| # | Attempt | Conditions | Observed failure | Known cause | Replacement | Tag, source |
|---|---|---|---|---|---|---|
| F1 | Production runs split into 100-step calls | P1, Twin-S | physics changed | each `step()` call forces an exchange at its first iteration | one `Simulation.step(N_t)` call per run | [R] PL:1114 |
| F2 | \(d_{AA}\) from one production pair in L2 | P2, Twin-XS 300 steps | bound too tight | one electron bifurcates; pairs differ by 7.7e-4 or 1e-5 | largest difference over 3 pairs | [R] PL:1198-1203 |
| F3 | L-BFGS-B, raw second moments (D14) | P3, central FD \(h = 0.02\), start (0.164, 0.323) | first step to corner (1, 0); \(\hat z_f\) pinned at 0; stopped after 7 evaluations | first step scaled by gradient only, then projected; positive \(z_f\) slope at the bound; raw moments double-weight \(\zeta\), \(u_z\) means (factors 173, 40) | dogbox `least_squares` on whitened residuals with multistart (D26); central moments | [M] `R:p3_fd_recovery.json#recovery_lbfgsb_stopped`, `L:runs_p3_lbfgsb.log`; [R] PL:1334, PL:1425 |
| F4 | `least_squares(trf)`, unwhitened central moments | P3, same start | not a failure; stopped at (0.432, 0.231), \(J = 0.0225\) | superseded by the user-approved gate | dogbox gate | [M] `#recovery_trf_central_stopped` |
| F5 | ECNoise step selection and Poisson bootstrap term (\(B = 256\)) in \(\Sigma\); \(\Delta\) from 1 point | P3 step 11, 13 observables, T0 | \(J(\theta^\ast) = 1128\); 1124 from one residual; gate (a) false | \(\Delta\) at nominal \(3.8\cdot10^{-4}\) vs \(2.5\cdot10^{-2}\) model error at \(\theta^\ast\); bootstrap 3 to 30 times smaller than \(\Delta\) for 10 of 13 observables | fixed FD steps \(h = (0.02, 0.05)\); \(\Sigma = \mathrm{diag}(\Delta^2)\) from 5 points (D27); \(J(\theta^\ast) = 6.77\) | [M] `L:runs_gate.log`, `L:runs_gate3.log`; [R] PL:1349, N1 |
| F6 | 13-observable loss with \(\langle y\rangle\), \(\langle u_y\rangle\) | P3 gate (b), T0′ | dogbox `xtol` stop at (0.5117, 0.2810), \(J = 26.98\); rejected step \(J = 43.6\) | jitter of \(\langle y\rangle\) (16.2 of 43.6) and \(\langle u_y\rangle\) (3.3); sensitivities \(\le 1.0\) noise units | 11 observables (D27) | [M] `#gate_dogbox_T0p_13obs`, `L:runs_gate2.log`; [R] PL:1358 |
| F7 | Gate (a) scoring rule | P3, 25-point design, T0′ | scored pass, but (0.5, 0.3) is in another basin | criterion "best grid minimum within one grid step of \(\theta^\ast\)" ignores basin radius | judged by fine slices and step 13; no automated fix recorded | [M] `#gate_design_T0p.gate_a_pass = true`; [R] PL:1381 |
| F8 | 2-D FD recovery (\(E_L\), \(z_f\)) with dogbox | P3 gate (b), 11 obs, T0′, start (0.5, 0.3) | stalls at \(J \approx 23.7\) near (0.507, 0.286) | false minimum near (0.51, 0.28), size about 0.01; true basin radius about 0.04; design spacing 0.2 | Stage A: \(E_L\) alone, \(K = 1\) (D28); \(z_f\) in Stage B | [M] `L:runs_gate3.log`; [R] PL:1388-1393 |
| F9 | Single best-design start in 1-D | P3 Stage A, \(E_L\) start 0.55 | ends at 0.5453, \(J = 12.55\) | 1-D false minimum at \(\hat E_L \approx 0.545\) | second design start (0.85) and multistart, now mandatory | [M] `#gate_one_E_L_T0p.runs.design_0.550` |
| F10 | EV closure on Twin-XS within 2 SE | P3 step 8 | 4.34 SE (mobile vs immobile) | effect of ion motion, not closure error; EV-mobile matches stochastic within 0.74 SE in key moments | user: does not block M1; open for P9 T2 | [M] `R:p3_closure.json`; [R] PL:1305 |
| F11 | FD cache keyed by \(\hat\theta\) only, then hard-coded `runs/p3` | P3, P5 | stale-entry risk; P5 stepper change broke provenance | key lacked code hash | provenance hashes (P3); `runs/fd_<code_sha256[:12]>/`, `runs/p3` read-only (D30, P6 step 0) | [R] PL:1426, PL:1606, PL:1635 |
| F12 | L3 with per-array normalization | P5, Twin-XS | failed on N-ion `inv_gamma` | central difference dominated by \(\mathrm{ulp}(1)/(2\epsilon)\) | vector form, each output divided by its primal max; \(\epsilon = 10^{-8}\) added | [R] PL:1547 |
| F13 | Twin-XS as L4 fixture to \(n_c + 200\) | P5, \(h \in \{10^{-2}, 10^{-3}, 10^{-4}\}\) | \(K(h) > 0\) from \(N_t = 10\); \(e \approx 1\) from 28 | 3.85 cells per \(\lambda_0\), \(a_0 \approx 4\): chaotic map, tangent \(\times 10\) per about 5 steps | small-\(h\) diagnostic; Twin-XS strict gate only for \(N_t \le 30\); Twin-S long horizons reported. A 16-cell Twin-XS was not chosen | [M] `R:p5_l4_twin_xs.json`; [R] PL:1576, PL:1608 |
| F14 | L4w \(E_r\) residual ratio | P5 | 2.67 (limit 3 to 5) | residuals below \(E_r\) run-to-run noise | accepted as deviation | [M] `R:p5_l4w.json#check.self_ratio_Er` |
| F15 | P6 strict L4 at \(N_t = 28\) | Twin-XS EV, \(h \in \{10^{-6}, 10^{-7}, 10^{-8}\}\) | only one \(h\) with \(K = 0\); fallback rule fails for background and total | Twin-XS fixture chaos, not the EV tangent | small-\(h\) evidence (\(K = 0\) at 4 \(h\), \(e \le 1.2\cdot10^{-5}\)); deviation accepted by the user | [M] `L:runs_p6_l4_xs.log`, `L:runs_p6_l4_xs_small_h.log`; [R] PL:1715 |
| F16 | Full-horizon AD tangent (P7 step 4) | Twin-S EV, \(N_t = 2000\) | tangents \(\times 10\) per 120 steps after step 1200, \(1.7\cdot10^{11}\) J\(^{-1}\) at 2000; \(e = 1\) at every \(h\) | Lyapunov instability of particle trajectories; frozen and free FD agree to 3 digits, so not thresholds | Track R (D33), then Q1 studies of 2026-11-01 (research record) | [M] `R:p7_sweep_twin_s.json`; [R] PL:1867 |
| F17 | Fixed-point shift \(\delta \le 0.005\) as the "AD usable" bar (D32) | P7 step 5, solution slice | \(\delta = 3.5\cdot10^{-3}\) passes formally while cosine median is \(-0.14\) and column error median 2010 | \(r_w\) is small at the solution, so \(\delta\) is small for any column | no replacement metric recorded; column error and cosine reported | [M] `R:p7_jacobian.json#usability.slices.solution`; [R] PL:1849 |
| F18 | \(\epsilon \to 0\) or zero-weight creation as the remedy | P7 step 6, reduced Twin-S | AD columns do not converge; \(\epsilon = 0\) AD differs from its FD by 99.98% | the error is chaos, not birth time | none needed | [M] `L:runs_p7_epsilon.log` |
| F19 | P7 step 5 slices | 123 planned runs | 47 done: 3 centers, solution slice 41 of 41, false minimum 5 of 41, far field 1 of 41; 76 not run (about 6.3 GPU-h [E]) | session ended on 2026-10-31; then D34 skipped the rest | resume command in N3; paused by D34 | [M] `C:p7_fe7188bf9100` (44 `grad_*.json` + 3 `center_*.json`); [R] PL:986 |

### 3. Code states

#### 3.1 Commit per phase

The "base" commit is the code the plan says the run sat on. The "evidence commit" holds the code and results. Commit times come from `git log` [M]. File times of logs come from `ls` [M], but they may not be exact run times.

| Phase | Plan base (source) | Evidence commit | Commit time | Note |
|---|---|---|---|---|
| P0 | `74c21bd` (PL:1047) | `9a65f19` (plan record) | 2026-10-30 18:00 | no test log saved |
| P1 | on top of `9a65f19` (PL:1077) | `227304f` | 2026-10-30 18:22 | `results/p1_twin_s.json`; extended in `41e471a` |
| P2 | on top of `227304f` (PL:1168) | `1d9272e` | 2026-10-30 18:24 | |
| P3 | on top of `1d9272e` (PL:1211) | `41e471a` | 2026-10-31 03:17 | All P3 runs used the working tree before this commit. Its simulation-code hash (`804032cdc3e2`) equals the hash at `41e471a` [M] |
| Plan fixes | n/a | `6e855cb` | 2026-10-31 11:51 | Fable review fixes, Stage A scope |
| P4 | on top of `6e855cb` (PL:1434) | `7c5aa8f` | 2026-10-31 12:22 | |
| P5 | on top of `7c5aa8f` (PL:1497) | `84a9426` | 2026-10-31 12:53 | Stepper change; simulation-code hash becomes `6a2fa798102c` [M] |
| Re-plan | n/a | `1c0f458` | 2026-10-31 15:26 | D29 to D32 |
| P6 step 0 | n/a | `94d9ffb` | 2026-10-31 15:36 | `L:runs_p6_invariants.log` time is 15:33, before this commit |
| P6 steps 1 to 5 | "on top of `1c0f458`" (PL:1626); in fact after `94d9ffb` by log times 15:37 to 15:48 | `4c5ff56` | 2026-10-31 16:23 | |
| Track R docs | n/a | `5de9dca` | 2026-10-31 22:34 | D33 |
| P7 | on top of `4c5ff56` (PL:1739) | `1119983` (checkpoint), `b3e03f2` (conclusion) | 2026-10-31 22:37 | All P7 runs share cache `p7_fe7188bf9100`, which equals the hash at `1119983` [M] |
| Research checkpoint | n/a | `a0d1bd2` | 2026-11-01 02:34 | D34 recorded; out of scope |

Targets [M, from the `.npz` keys]: T0 and T0′ both record `git_hash` = `1d9272e…` with no dirty marker. T0′ was made after the EV code existed; the dirty marker came only in P6 step 0. T0 has `p_nz=2`; T0′ has `p_nz=1`; both have \(N_t = 2000\).

#### 3.2 Code hashes recomputed from `git show` (read-only) [M]

FD cache hash (`ev_point.code_sha256`: `forward.py`, `twin_config.py`, `fbpic/autodiff/{ionization,stepper,particles}.py`):

| Commit | `1d9272e` | `41e471a`, `6e855cb`, `7c5aa8f` | `84a9426` | `94d9ffb` | `4c5ff56`, `1119983`, working tree |
|---|---|---|---|---|---|
| Hash prefix | `e22c7e65c402` | `804032cdc3e2` | `6a2fa798102c` | `8cafa90f0be3` | `5083190be4e5` |

P7 cache hash (`p7_jacobian.code_dir`: `forward.py`, `twin_config.py`, `residual_objective.py`, `fbpic/autodiff/*.py`): `4c5ff56` gives `83f8ef7650d7`; `1119983`, `b3e03f2`, `a0d1bd2`, and the working tree give `fe7188bf9100`. The hashed files have no uncommitted changes.

#### 3.3 Cache directories (`inversion/runs/`, git-ignored)

| Directory | Contents [M] | Code key | Phase |
|---|---|---|---|
| `p1/` | 43 `.npz`: corners, starts, \(\pm h\) seeds 101 to 104, 16 `star_s*` seeds, `nominal`, `t2_truth`; 453 MB | none | P1 |
| `p3/` (mode read-only) | 311 JSON + 330 logs. 309 JSON carry `code_sha256` `804032cdc3e2`, `config_sha256` `ee2a49844003`, `target_sha256` `90ee4adeb6e6` (T0). 2 noise-floor JSON (`p0.600000_0.400000.json`, `_rep1.json`) have no `provenance` key | `804032cdc3e2` | P3 FD runs |
| `p3_twin_s_*.npz`, `p3_twin_xs_immediate.npz` | 11 field and particle dumps for covariance and EV runs, about 3.0 GB | none | P3 |
| `p4/` | `tangent_l1.log`, `field_linearity.log` | none | P4 |
| `p5/` | 8 logs (L2t, L3, L4, L4w, L2 regression) | none | P5 |
| `p7_fe7188bf9100/` | 59 JSON + logs: 3 `center_*`, 44 `grad_*` (40 solution, 4 false-minimum), 2 `cost_grad_*`, 2 `cost_primal_*`, 4 `eps_grad_*`, 4 `eps0_fd_*` | `fe7188bf9100` | P7 steps 3, 5, 6 |
| `fd_<hash>/` | none exist | would be `fd_5083190be4e5` now | no FD run after `94d9ffb` |
| `q1/` | 2026-11-01 work | see research record | out of scope |

Preservation risk [M]: `inversion/runs/`, `inversion/targets/*.npz`, and all `inversion/runs_*.log` files are git-ignored (`inversion/.gitignore`: `runs/`, `*.log`; root `.gitignore`: `*.npz`). They exist only on this disk. Committed evidence is the plan, the notes, and `inversion/results/*.json`.

### 4. Evidence gaps and discrepancies

1. No saved log for P0 L0, P2 L2, P3 open loop and minimal loss, P5 probes and J_chk L1, P6 L1 and L2t (a) with EV, or P7 L1. These rows are [R] only.
2. The 2-D 11-observable dogbox run (M1-25) has no `results/p3_fd_recovery.json` key; only `L:runs_gate3.log` holds it.
3. `R:p3_fd_recovery.json#gate_one_E_L_T0.pass` is `false`, because its `best` entry is the design start (0.55) in the false minimum. The plan reports the T0 \(E_L\) bias from the farthest start (0.5878). T0 is a report-only target.
4. The plan says all 311 `runs/p3` runs carry hash `804032cdc3e2` (PL:1415). Two noise-floor JSON have no provenance key.
5. The plan gives 345 to 351 s per `p_nz=2` covariance run (PL:1338). `L:runs_cov_corner*.log` show 701 to 971 s; nominal and T0 show 351 and 345 s. Concurrent GPU use is a possible cause; this was not checked.
6. The plan says P6 ran "on top of `1c0f458`" (PL:1626). Log times put steps 1 to 5 after `94d9ffb`. File times are only indirect evidence.
7. Values that I did not re-read from raw output: P1 aliasing and weights, P2 ratios, P4 table values, P5 L2t ratios, P5 growth numbers, P7 step 3 profile, and the P7 step 4 growth table.


## Part B. Research-track studies and GPU diagnostics, 2026-10-31 to 2026-11-01

### B1. Research-track studies, 2026-10-31 to 2026-11-01

All studies ran as multi-agent workflows from session 2553145c, CPU only (NumPy, `python3 -I`), with no edits to `fbpic/autodiff`. Agent scripts have no commit identity; their outputs are in the evidence directories named below.

| Study | Authorizing decision | Agents | Question | Outcome | Failure meaning | Evidence |
|---|---|---|---|---|---|---|
| Track R Phase R0: particle-pdf method (Chung et al. 2020) for FBPIC | U19, U20 | 5 derivation, 2 critics, 1 synthesis (workflow `track-r-r0-feasibility`, about 1.5 h) | Can the paper's \(\delta f\) weight method give regular sensitivities on Twin-S? | No-go recommended (proposed D35; open) | Unsuitable approach for a cold plasma: the weight form needs \(\nabla_u \ln f\), which a cold sheet lacks; credible variants cost about one run per parameter | [R0 note](../../notes/r0-design-note-2026-10-31.md), [r0-evidence](../../notes/r0-evidence-2026-10-31/), finding 3 |
| Gauge-projected (Wasserstein) tangent | U23 | 4 explore agents (derivation, prototype, alternatives, literature); stopped by the agent before critics | Does projecting \(\xi\) onto gradient fields remove chaotic growth? | Rejected | Unsuitable approach: on a cold sheet the explosion is not relabeling; on finite particles every gauge removal is a basis truncation | [gpt-evidence](../../notes/gpt-evidence-2026-11-01/), finding 6 |
| Label-space integration by parts (LIBP) | U23 (found by the alternatives agent) | inside the study above | Exact cold-sheet identity for the along-sheet tangent | Exact in 1D; raw summand heavy-tailed; a stretched-region hybrid cuts the per-particle standard deviation 7 to 8 times | Not rejected; needs second-order tangents; returns the pointwise continuum derivative | `gpt-evidence/sec_alt.md` Section 3 |
| Option C: noise-regularized likelihood-ratio (Girsanov) weights | U24 | 4 (theory, literature, toy, wake), 2 critics, 1 synthesis (workflow `stochastic-lr-gradient-study`, about 11 h including an 8 h synthesis) | Can per-particle score weights on a noisy model give the macroscopic gradient from one run? | Stopped by the user (U26) | Unsuitable approach at physical noise: correct for the noisy model, but no noise level keeps both the physics and the variance acceptable; growth moves to the field tangent; information floor \(\mathrm{SE}/\lvert\delta m\rvert \gtrsim s_g/(u_n\sqrt{N_{\mathrm{eff}}})\) | [option-c-evidence](../../notes/option-c-evidence-2026-11-01/), finding 8 |

### B2. GPU diagnostics, 2026-11-01

Environment for all rows: conda env `fbpic-warp`, Warp 1.18.0, CuPy 14.2.0, Python 3.11.16, RTX 6000 Ada, FP64, one GPU per process via `CUDA_VISIBLE_DEVICES`. Point: \(\hat E_L = 0.588\), \(\hat z_f = 0.4\), EV with free thresholds, \(\epsilon = 10^{-4}\). Run directory `inversion/runs/q1/` (git-ignored). Commands and code identity: technical record Sections 12.2 and 12.7.

| Run family | Decision | Runs | Result | Status | Finding |
|---|---|---|---|---|---|
| FD scan at 1x (`p_nr` 2, `p_nt` 8), \(h = 10^{-2}\) to \(10^{-4}\), 7 snapshots | U21 | 11, about 90 s each | Secants at \(h \le 3\cdot10^{-3}\) deviate 23% to 66% from \(h = 10^{-2}\) (steps 1520 to 2000); roughness already present at step 1520 | complete | 4 |
| Injection-seed spread at 1x, \(h = 10^{-2}\) | U21 | 9 | Spread 3.5% to 4.1%, cosine at least 0.999 | complete | 4 |
| FD scan at 4x (`p_nr` 4, `p_nt` 16) | U21 | 11, about 325 s each (up to 637 s under host load) | Roughness at \(h = 10^{-3}\) to \(3\cdot10^{-3}\) falls to 4% to 9% (steps 1520 to 2000); \(h = 10^{-2}\) Jacobian has cosine \(-0.69\) to 1x | complete | 4, 5 |
| Seed spread at 4x | U21 | 1 of 9 | none | stopped by the agent to save time | 4 |
| Per-species tangent growth at 1x and 4x | U21 | 2 (305 s, 1299 s) | Growth from about step 1450 in the medians; 4x cuts rates by only 10% to 30% | complete | 4 |
| Radial convergence: labels 2r, 2t, 8 | U21 (agent follow-up, reported to the user) | 9 | Only `p_nr` matters; 2 to 4 reverses the Jacobian; 4 to 8 shrinks it 2.7 times | complete | 5 |
| Radial convergence: label 16r (`p_nr` 16, `p_nt` 8) | started before U25; last point finished under it | 3, 1250 to 1630 s | `p_nr` 8 and 16 agree at step 2000: cosine 0.998, relative vector difference 9%, norms within 6% | complete | 5 |
| Exact AD column at 1x and at `p_nr` 8 | U25 | 2 (2438 s at `p_nr` 8) | At `p_nr` 8 the AD column is about \(10^4\) times the FD column at step 2000 and 9 to 12 times at step 1200; the 1x column differs 63% from the cached P7 column | complete | 7 |
| FD at `p_nr` 8, \(h = 3\cdot10^{-3}, 10^{-3}\) | U25 | 4, 677 to 1058 s | Direction stable (cosine 0.96 to 0.99); norm within 5% to 14%, relative vector difference 18% to 29% | complete | 7 |
| Seed spread at `p_nr` 8, \(h = 10^{-2}\) | U25 | 4, 655 to 682 s | Spread 4.0% to 4.6%, cosine 1.000 | complete | 7 |

Not run: P7 step 5 remaining 76 slice points (paused by U20); option C Gate B (stopped by U26); any run at a different grid (`Nr`) or time step.


## Part C. Reusable assets, pitfalls, and numbers to plan with

Scope: worktree `dev-autodiff/lpa/FBPIC_Warp`, branch `snidhan/dev-autodiff`. All paths below are relative to `lpa/FBPIC_Warp` unless they start with `/`.
Evidence labels: **M** = measured (I inspected the saved output, log, file, or git state in this pass). **R** = reported (plan, note, finding, or transcript text; not rechecked). **E** = estimated (my arithmetic or inference).
This file does not repeat the scientific findings 1 to 8 (`inversion/research/sensitivity-feasibility/findings/`); it points to them.

---

### 1. Reusable assets

#### 1.1 Frozen inputs: configurations, targets, whitening

| Asset | What it holds | Tracked in Git? | Key facts | Label, source |
|---|---|---|---|---|
| `inversion/configs/twin_s.json` | Twin-S: `Nz` 800, `Nr` 50, `Nm` 2, `num_steps` 2000, `p_nz` 1, `p_nr` 2, `p_nt` 8, `theta_star_hat` (0.6, 0.4), `selection.u_c` 6.246, `delta_u` 0.0385, `ev.epsilon` 1e-4, `K_e` 1, 5 optimizer starts | yes | file sha256 `a489fc1b...`; `config_sha256()` (canonical) `ee2a4984...`. These two hashes differ by design; do not compare them. | M: file read, `sha256sum`, `results/n_c_twin_s.json` key `config_sha256` |
| `inversion/configs/twin_xs.json` | Twin-XS: `Nz` 96, `Nr` 32, preloaded plasma, 300 steps, selection by convention only | yes | file sha256 `77c5eb96...`; `config_sha256` `1ee8ee92...` | M: file read, `results/n_c_twin_xs.json` |
| `inversion/targets/twin_s_T0_Nt2000.npz` | T0 target: `Q`, `Q_f`, 12 `moments`, `standardization` (2, 6), `s_k`, `theta_star`, `seed` 7, `p_nz` 2, `n_selected` 3433801, empty `rff_mean` | **no** (root `.gitignore:9` `*.npz`) | sha256 `90ee4ade...`; `git_hash` key `1d9272e` (that commit lacked the EV code; target made from a dirty tree) | M: npz keys, `git check-ignore`; R: plan line 1307 |
| `inversion/targets/twin_s_T0p_Nt2000.npz` | T0p (T0' control): inversion settings, `p_nz` 1, `seed` -1, `n_selected` 1668833 | **no** | sha256 `88783bf7...` | M: npz keys |
| `inversion/targets/twin_s_T0_Nt2000_whitening.npz` | 11-residual whitening: `L` (11, 11), `Sigma`, `Delta_model`, `Delta_points` (5, 11), `floor_sigma`, `floor_mean_uz`, `v_target` | **no** | sha256 `719cbc0e...`; \(\Sigma = \mathrm{diag}(\Delta^2)\) over nominal and 4 corners | M: npz keys; R: m1 note "Later changes", D27 |
| `inversion/results/n_c_twin_{s,xs}.json` | Cell-crossing horizon \(n_c\): 450 (Twin-S), 18 (Twin-XS), with `dt_s` | yes | Twin-S `dt_s` \(1.668\cdot10^{-16}\) s | M |

#### 1.2 Driver scripts in `inversion/` (research drivers)

| Script | What it does | Inputs | Outputs and cache location | Label, source |
|---|---|---|---|---|
| `twin_config.py` | `load_config`, `config_sha256`, `theta_from_hat`/`hat_from_theta`, `build_twin` (single-GPU lab-frame `Simulation`), density closed forms | config JSON, \(\theta = (E_L, z_f)\) | none; part of P7 cache key | M: docstring, defs |
| `builders.py` | `twin_builder(size or config, ionization, **kw)` returns `build(theta) -> (sim, species, profile)` for the driver | config name or dict | none | M: file read |
| `forward.py` | `run_production` (production `Simulation.step`), `build_ev_state`, `run_ev`, `device_memory()` (CUDA mempool high-water marks) | config, \(\theta\) | none; part of P3 FD and P7 cache keys | M |
| `observable.py` | Selection (quintic smoothstep on \(u_z\)), beam moments, central moments, 11 loss observables, `whitened_residuals` | particle arrays (NumPy or CuPy) | none | M |
| `residual_objective.py` | Builds `fbpic.autodiff.loss.ResidualObjective` from target + whitening + selection | config, variant (`T0`, `T0p`) | none; part of P7 cache key | M |
| `make_target.py` | Writes T0 or T0p target | `twin_s T0` or `twin_s T0p` | `targets/twin_s_<variant>_Nt2000.npz` | M: docstring |
| `target_covariance.py` | Model-error whitening; stages `run <point>`, `build` | P3 FD cached model runs + truth runs | `runs/p3_twin_s_<point>_particles.npz`, `targets/..._whitening.npz` | M: docstring, `runs/` listing |
| `identifiability.py` | P1: `selection`, `trapping`, `sensitivity`, `report`, `report_loss` (production stochastic runs) | twin_s | `runs/p1/*.npz` (453 MB, 44 files), `results/p1_twin_s.json` | M: listing, `du` |
| `ev_runs.py` | P3 steps 5 and 6: EV runs at nominal, corners, immediate emission (Twin-XS, reduced Twin-S); `reduced_config()` | size, point | `runs/p3_twin_*_*.npz`, `results/p3_ev_runs.json` | M |
| `ev_point.py` | One EV loss evaluation; `code_sha256()` over `forward.py`, `twin_config.py`, `ionization.py`, `stepper.py`, `particles.py` | size, target, \(\hat\theta_0\), \(\hat\theta_1\), out.json | JSON with J, Q, moments, births, wall, `provenance` | M: lines 25 to 52 |
| `fd_recovery.py` | P3 FD stages: `noise`, `slices`, `recover`, `lsq`, `edge`, `design`, `fine`, `dogbox`, `farthest`; `--target`, `--starts`; up to `MAX_PARALLEL = 5` children | config, target | cache `runs/fd_<code_sha256[:12]>/p<t0>_<t1><tag>.json`; result `results/p3_fd_recovery.json` | M: lines 20 to 75 |
| `verify_param_gradient.py` | L4 harness: AD vs frozen-decision central FD, \(\sigma_J\) from 5 base runs, sampled cell keys \(K(h)\) (stride 256), `nc` stage | size, `--ionization ev`, `--steps`, `--horizons`, `--h`, `--key-stride`, `--loss-target`, `--tag` | `results/p5_l4_*.json`, `p6_l4_*.json`, `p7_sweep_twin_s.json`, `n_c_*.json` | M: docstring; R: plan 12.10 step 4 command |
| `verify_one_step.py` | L3: tangent of one `step` vs central FD of `step` from a snapshot; frozen `emit_mask` with `--ionization ev` | Twin-XS, step list | `results/p5_l3_twin_xs.json`, `p6_l3_twin_xs.json` | M |
| `state_snapshot.py` | `snapshot`/`restore` of a stepper state incl. NumPy RNG | state | none | M |
| `verify_batched_tangents.py` | P7 step 2: \(K = 2\) vs two \(K = 1\) runs, tolerance from repeat runs | Twin-XS, 10 steps | `results/p7_batched_tangents.json` | M |
| `verify_crossing_probe.py` | Free FD minus frozen FD at one emission threshold (one-ion ramp) | Twin-XS | `results/p6_crossing_probe.json` | M |
| `verify_ev_open_loop.py`, `verify_ev_invariants.py`, `verify_ev_closure.py`, `verify_loss_minimal.py`, `verify_twin_p1.py`, `verify_linear_wakefield.py` | P1 and P3 model checks; L4w (linear wakefield, \(dE/da_0 = 2E/a_0\)) | as in docstrings | `results/p3_closure.json`, `p5_l4w.json` | M: docstrings |
| `diag_tangent_cost.py`, `diag_tangent_growth.py`, `diag_l4w_noise.py` | Per-step primal and \(K = 1\) tangent timing; tangent norm growth; L4w noise | size, steps | `results/p5_tangent_cost.json`, `p5_tangent_growth_twin_s.json`, `p5_l4w_noise.json` | M |
| `p7_jacobian.py` | P7 stages `cost`, `usability`, `epsilon`, `report`; each run a child process | config, T0, \(\hat E_L\) | cache `runs/p7_<sha[:12]>/` (`grad_<x>`, `center_<x>`, `cost_*`, `eps_grad_*`, `eps0_fd_*`); result `results/p7_jacobian.json` (keys `cost`, `epsilon`, `usability`, `code_dir`) | M: lines 60 to 100, cache listing |
| `q1_roughness.py` | Free-threshold EV runs with \(r_w\) at snapshots 1200, 1400, 1520, 1640, 1760, 1880, 2000; stages `scan <int>`, `seeds <int>`, `points <label>`, `report`; \(X_0 = 0.588\), \(H = (10^{-2}, 3\cdot10^{-3}, 10^{-3}, 3\cdot10^{-4}, 10^{-4})\), seeds 1, 2, 3 | label (see 1.4), seed, \(x\) | cache `runs/q1/m<label>_<default or s<seed>>_<x:.6f>.json` + `.log`; `results/q1_roughness.json` (labels 1 and 4 only) | M: file read |
| `q1_tangent_growth.py` | `driver.jvp` with EV at \(\hat E_L = 0.588\); per-species median, p90, max of \(\lvert\partial u/\partial\hat E_L\rvert\), \(\lvert\partial x/\partial\hat E_L\rvert\) every 50 steps; AD residual column `ad_columns.<step>.jac_hat` at the Q1 snapshots | label, optional suffix | `results/q1_tangent_growth_m<label><suffix>.json` (`m1`, `m4` without AD columns; `m1_ad`, `m8_ad` with them) | M: file read, result keys |

#### 1.3 Library modules in `fbpic/autodiff/`

| Module | Role | Label, source |
|---|---|---|
| `state.py` | `StepState`, `TangentState` (`zeros`), `StepDecisions` (`removed`), `EVArrays` | M: docstring |
| `stepper.py` | `begin`, `step(state, tangents)`, `sort_all` (stable sort, permutations applied to tangents), hooks `stage:*`, `tangent_*:before/after`, `"*"` | M; R: m2 note "P5 files" |
| `driver.py` | `jvp`, `value_and_grad`, `seed_tangents`, `contract`, `tangent_norms`; takes a `build(theta)` callable and `callback(state, tangents, decisions)` | M |
| `tangent.py` | Hand-written JVP kernels: gather, Vay push (electron and ion with charge tangent), position push, fused deposit tangent | M |
| `field_tangents.py` | `FieldTangents`: reuses production linear field operators on swapped tangent arrays | M |
| `ionization.py` | EV ionization (ADK, populations P5 to P7, pending weight W, emission at \(W' > \epsilon w\)), `ev_tangent`, `ionize_tangent`, `charge_weight_tangents` | M; R: m3 note "P6 design" |
| `laser.py` | `laser_initial_fields`, `energy_seed` (\(F_0/(2E_L)\)), `amplitude_seed` | M |
| `loss.py` | `CheckFunctional` (\(J_{chk}\)), `ResidualObjective`, `ResidualValue.contract` | M |
| `particles.py`, `deposition.py`, `fields.py`, `arrays.py`, `verification.py` | FP64 operator VJPs, `check_jvp`, `check_vjp`, `dot_product_error` (L1 references) | M |
| Tests | `tests/test_autodiff_*.py` (12 files: arrays, deposition, fields, field_tangents, ionization, laser, loss, particles, residual, stepper, tangent, tangent_step) | M: `ls` |

#### 1.4 Run caches and their keys (all under `inversion/runs/`, git-ignored by `inversion/.gitignore` line 1)

| Cache | Key | Contents | Valid now? | Label, source |
|---|---|---|---|---|
| `runs/p1/` | point name, seed | 44 production-run npz (nominal, 4 corners, 5 starts, 16 star seeds, 16 derivative seeds, `t2_truth`), 453 MB | no code key | M: listing |
| `runs/p3/` | \(\hat\theta\) (+ `_rep1`, `_rep2`); provenance in each JSON: `config_sha256`, `target_sha256`, `code_sha256` `804032cdc3e2...` | 309 + 2 FD run JSON + logs (P3 evidence, frozen by D30) | frozen evidence; current simulation-code hash is `5083190be4e5`, so `fd_recovery.py` would not reuse it | M: recomputed hash; R: D30, `fd_recovery.py` docstring |
| `runs/fd_<sha[:12]>/` | `ev_point.code_sha256()` + \(\hat\theta\) | none exists yet | a new FD campaign starts at `fd_5083190be4e5/` | M: `ls runs/fd_*` empty |
| `runs/p3_twin_*.npz` | point name | EV particle dumps for whitening and EV checks, 31 to 459 MB each (about 3 GB total) | no code key | M: `du` |
| `runs/p4/`, `runs/p5/` | none | L1, L2, L2t, L3, L4, L4w logs | logs only | M |
| `runs/p7_fe7188bf9100/` | sha256 of `forward.py`, `twin_config.py`, `residual_objective.py`, sorted `fbpic/autodiff/*.py` | 3 centers (0.545, 0.588, 0.75), 44 `grad_<x>` (40 of the solution slice 0.538 to 0.638, 4 of the false-minimum slice), 2 `cost_grad`, 2 `cost_primal`, 4 `eps_grad`, 4 `eps0_fd` | **yes**: recomputed key is `fe7188bf9100` today | M: recomputed hash, listing |
| `runs/q1/` | label, seed tag, \(x\) only (no code or config hash) | 52 JSON + logs: `m1` 20, `m4` 12, `m8` 11, `m2r` 3, `m2t` 3, `m16r` 3 | valid only while `worker_point` is unchanged | M: listing; R: technical record 12.2 "Code identity" |

Q1 label to particle layout (`q1_roughness.py` `MULTIPLIERS`, `EXTRA_RESOLUTIONS`), all `p_nz` 1:

| Label | `p_nr` x `p_nt` | Particles per cell vs label 1 | `elec_from_N` at step 2000 | Selected at step 2000 | Label, source |
|---|---|---|---|---|---|
| 1 | 2 x 8 | 1x | 6.52M | 1.68M | M: `runs/q1/m1_default_0.588000.json` |
| 2r | 4 x 8 | 2x | 13.05M | 3.35M | M |
| 2t | 2 x 16 | 2x | 13.04M | 3.37M | M |
| 4 | 4 x 16 | 4x | 26.10M | 6.69M | M |
| 8 | 8 x 16 | 8x | 52.19M | 13.41M | M |
| 16r | 16 x 8 | **8x** (same count as label 8) | 52.20M | 13.41M | M |

#### 1.5 Other evidence directories (CPU studies)

| Path | Contents | Label, source |
|---|---|---|
| `inversion/notes/r0-evidence-2026-10-31/`, `gpt-evidence-2026-11-01/`, `option-c-evidence-2026-11-01/` | Agent sections and `work_*` CPU NumPy scripts and outputs for Track R, gauge-projected tangent, option C | M: listing; R: technical record 12.3 |
| `inversion/notes/optimizer-literature-2026-10-30.md` | Optimizer literature review (trust-region GN, design then local solve) | R: m1 note |
| `inversion/research/sensitivity-feasibility/decisions.md` | Dated decision log reconstructed from transcripts (new, untracked status not checked) | M: file exists |

---

### 2. Pitfalls and operating rules learned

#### 2.1 Environment and GPU

| # | Rule or pitfall | Label, source |
|---|---|---|
| 1 | Activate `source ~/anaconda3/etc/profile.d/conda.sh && conda activate fbpic-warp`; `export PYTHONPATH=$PWD` from `lpa/FBPIC_Warp`. Stack: Python 3.11.16, Warp 1.18.0, CuPy 14.2.0, CUDA 12.9, driver 13.2. The system `python3` has no NumPy. | R: m3 note Rules 2; technical record 12.1. M: system `python3` import failed |
| 2 | Always set `CUDA_VISIBLE_DEVICES` per process. The stray GPU 0 context fix `8f27245` lives on `snidhan/dev-nccl`, not on this branch. With one visible device the process sees it as `cuda:0`. | R: plan lines 703, 824, 1053; memory `inversion-milestone-workflow.md` line 55 |
| 3 | GPU processes must run outside the agent sandbox. | R: plan 12.2 "Sandbox note"; m3 note Rules 1 |
| 4 | Check `nvidia-smi` before a campaign. Host: 2 x RTX 6000 Ada (49140 MiB each), 125 GiB RAM, 32 cores. | R: plan Section 8, 12.2. M: `nproc` = 32 |
| 5 | One run saturates a GPU. Parallel processes on one GPU give no speedup (25 runs in parallel batches: 2158 s, 86 s each). Use one process per GPU; use both GPUs for independent points. | R: plan Section 8 line 917 |
| 6 | Run each test file in its own process; the root `pytest.ini` does not collect GPU tests. Format with Black 25.1.0, lint with Ruff 0.11.11. | R: `CLAUDE.md` rules 5, 8; m2 note Rules 6 |
| 7 | Measure memory with `forward.device_memory()` (CUDA stream-ordered mempool high-water marks). `cupy.get_default_memory_pool().total_bytes()` returns 0 here, so the Q1 field `pool_bytes_end` is useless. | M: every `runs/q1/*.json` has `pool_bytes_end` 0; `forward.py` lines 17 to 29 |

#### 2.2 Branch and Git

| # | Rule or pitfall | Label, source |
|---|---|---|
| 1 | `snidhan/dev-autodiff` is based on `snidhan/dev-main`, not `origin/main`. Never run `sync-push` on it: it rebases onto `origin/main` and removes the base. | M: `git merge-base --is-ancestor snidhan/dev-main HEAD` true. R: memory `differentiable-inversion-plan.md` line 12 |
| 2 | Pushes are plain fast-forwards and need user approval. Commits are local per phase and need user approval. | R: memory `differentiable-inversion-plan.md` line 13; m2, m3 notes |
| 3 | State now: HEAD `a0d1bd2` is 1 commit ahead of `origin/snidhan/dev-autodiff` (`b3e03f2`). Uncommitted: `inversion/q1_tangent_growth.py` (+21, -6), research record edits, `CLAUDE.md`, `README.md`; untracked: `results/q1_tangent_growth_m{1,8}_ad.json`, option C evidence, findings 7 and 8. | M: `git branch -vv`, `git status`, `git diff --stat` |
| 4 | Targets and whitening `.npz` are ignored by the root `.gitignore` (`*.npz`), and `inversion/runs_*.log` are ignored by `inversion/.gitignore` (`*.log`). A clean checkout has neither. Back them up outside Git; a regenerated target differs by GPU-atomic noise and has a new `target_sha256`, so cached FD runs stop matching. | M: `git check-ignore -v`; `ev_point.provenance` includes `target_sha256` |
| 5 | The T0 and T0p `git_hash` key is `1d9272e`, which did not hold the EV code; the targets were made from a dirty tree. The dirty-marker fix came later (P6 step 0) and the targets were not remade. | R: plan line 1307; replan Section 2 P6 step 0. M: npz `git_hash` |
| 6 | `D17`: production stepping is unchanged. New code goes only in `fbpic/autodiff/` and `inversion/`. If a production kernel must change, stop and ask. | R: plan Section 10 D17; m3 note Rules 4 |

#### 2.3 Caches and invalidation

| # | Rule or pitfall | Label, source |
|---|---|---|
| 1 | P7 cache key = sha256 of `forward.py`, `twin_config.py`, `residual_objective.py`, and every `fbpic/autodiff/*.py`. Any edit (even a comment or a formatter pass) starts a new empty cache. Key today: `fe7188bf9100`. | M: `p7_jacobian.py` lines 68 to 80, recomputed hash |
| 2 | `p7_jacobian.code_dir()` calls `mkdir` on every call. After an edit, `report` silently reads a new empty directory. | M: `p7_jacobian.py` line 79 |
| 3 | FD cache key (`ev_point.code_sha256`) covers only `forward.py`, `twin_config.py`, `ionization.py`, `stepper.py`, `particles.py`. Loss changes do not invalidate it (the loss is recomputed from stored Q and raw moments), so `beam_statistics` must stay unchanged. | M: `ev_point.py` lines 25 to 39 |
| 4 | `runs/p3` is frozen evidence (code hash `804032cdc3e2`). The P5 and P6 stepper changes broke its provenance. P9 compares with the recorded P3 results, not by reuse. | R: D30; m2 note Findings 4; replan Section 1 row 5 |
| 5 | `runs/q1` has no code or config hash in its key or records. Editing `worker_point`, `_config`, the objective, or the stepper does not invalidate it. Record the code state by hand before new Q1 runs. The 1x and 4x scans used an earlier `q1_roughness.py` revision. | M: `q1_roughness.py` `_path`, record fields; R: technical record 12.2 |
| 6 | `q1_roughness.py scan` and `seeds` cast the label to `int`; string labels (`2r`, `2t`, `8`, `16r`) need `points <label>` or a direct `_child(label, seed, x)` call. `report` analyzes only labels 1 and 4. | M: `q1_roughness.py` lines 200 to 212, 159 to 199 |
| 7 | The Q1 runs at labels other than 1 compare against the T0 target and whitening made at `p_nr` 2 (`residual_objective(load_config(SIZE), TARGET)`). Their center residuals carry resolution bias; only differences in \(\theta\) are meaningful. | M: `q1_roughness.py` `worker_point`; R: finding 5 |

#### 2.4 Numerics and FD references

| # | Rule or pitfall | Label, source |
|---|---|---|
| 1 | No bitwise reproducibility: deposition uses GPU atomics; repeated runs differ by rounding. Particle IDs and ionization levels must still match. | R: `CLAUDE.md` rule 4 |
| 2 | Consequence on Twin-XS: one electron bifurcates between production runs; runs differ by \(7.7\cdot10^{-4}\) in the infinity norm. Use the largest of 3 pairwise differences as the noise floor, not one pair. | R: plan lines 1198 to 1203 |
| 3 | Consequence at the full horizon: two 1x AD columns at step 2000 at the same point differ by 63% of the norm (cosine 0.78). Atomic noise amplified by the tangent growth is the stated interpretation (not tested). Do not treat one full-horizon AD column as reproducible. | R: finding 7 "Reproducibility check" |
| 4 | Set tolerances from repeated runs: \(K = 2\) vs \(K = 1\) uses \(\max(10\,d_{rep}, 10^{-13})\); FD uses \(\sigma_J\) = largest pairwise spread of 5 base runs and \(\tau(h) = \max(10^{-4}, 10\sqrt2\,\sigma_J/(2h\lvert g\rvert))\). | M: `verify_batched_tangents.py` docstring, `verify_param_gradient.py` docstring |
| 5 | Frozen-decision FD (emissions of the AD run reused as `emit_mask`) is the only valid AD reference (D10). On Twin-S at \(N_t = 2000\), frozen and free FD agree to 3 digits at \(h = 10^{-2}\) to \(10^{-5}\), so the birth-time term is negligible there; \(\epsilon \to 0\) does not fix the AD error. | R: plan 12.10 step 5 item 1, step 6 |
| 6 | FD at \(N_t \ge 1000\) is not a frozen-decision reference: removed counts at the window back differ from the AD run from iteration 854 (\(h = 10^{-2}\)) to 1148 (\(h = 10^{-7}\)). | R: plan 12.10 step 4; m3 note Open issue 3 |
| 7 | Cell crossings: tight pass only where sampled \(K(h) = 0\). Twin-S reaches \(K(h) > 0\) at every \(h\) from \(N_t = 650\). | R: plan 12.10 step 4 table |
| 8 | Twin-XS (3.85 cells per \(\lambda_0\)) is chaotic: FD resolves the derivative only to about \(N_t = 30\). Use it only for short-horizon gates (\(N_t \le 30\), small \(h\)). | R: m2 note results 5; replan row 2 |
| 9 | Exact frozen-decision AD at \(N_t = 2000\) is unusable on Twin-S at every tested resolution (findings 1, 7). The fixed-point shift \(\delta\) passes formally (\(3.5\cdot10^{-3}\)) but is a poor indicator because \(r_w\) is small at the solution. | R: plan 12.10 step 5 item 4; findings 1, 7 |
| 10 | `p_nr` 2 (the frozen Twin-S config) is under-resolved: the Jacobian reverses direction from `p_nr` 2 to 4 and converges within about 10% only at `p_nr` 8. All Twin-S gradient references before finding 5 belong to `p_nr` 2. | R: finding 5 |
| 11 | Selection ramp at intermediate snapshots: the smoothstep selection uses \(u_c = 6.246\) fixed at step 2000. At step 1200 the 1x run selects 274590 particles against 1.68M at step 2000, and the \(\log Q\) residual is \(-66.6\) (label 8: \(-68.0\)). Before step 1520 the residual values are dominated by the selection cut; use only \(\theta\)-differences, and do not read step-1200 or step-1400 secants as beam physics without care. | M: `runs/q1/m1_default_0.588000.json` and `m8_default_0.588000.json`, keys `snaps.<step>.n_selected`, `r_w[0]` |
| 12 | Gradient cost is dominated by the deposit tangent (one thread per cell, serial over dense EV cells): 121 of 228 ms per step at step 1200. A split of dense cells changes a verified P4 kernel (L1, L3, L4 again). | R: plan 12.10 step 3; m3 note Open issue 2 |
| 13 | EV births are 4 to 6 times the loaded background (11.9M to 16.3M per Twin-S run); at \(\epsilon = 0\) the reduced Twin-S needs 13.7M births and 8.99 GB reserved. Size memory for births, not for the loaded plasma. | R: plan Section 8; M: `results/p3_ev_runs.json` births |
| 14 | 2-D recovery on T0p failed from the 25-point design (false minimum near (0.51, 0.28), basin radius about 0.04, design spacing 0.2). Multistart stays mandatory; Stage A (E_L only) is the active scope (D28). | R: m1 note "Outcome" |
| 15 | L-BFGS-B on raw moments jumps to the corner (1, 0) on the first step. Use trust-region least squares on whitened residuals (D26). | R: m1 note "P3 results so far" |

#### 2.5 Operations and agent workflow

| # | Rule or pitfall | Label, source |
|---|---|---|
| 1 | **pkill self-match.** A background watcher ran `pkill -f "q1_roughness.py seeds 4"`. The watcher's own shell command line contained that pattern, so `pkill` killed the watcher before its `report` step ran. Kill by PID from `pgrep`, or use a pattern that cannot match itself (for example `q1_roughnes[s].py`). | R: session transcript `2553145c...jsonl`, tool call `toolu_018iHy7h...` and the assistant text "My stop command matched its own command line and killed itself" |
| 2 | **Host CPU contention.** Parallel agent workflows ran CPU NumPy ensembles (60 or more processes on 32 cores, load average near 100). GPU runs slowed by up to 2.6 times (label 2t: 578 s vs 218 s; label 4: 749 s vs 325 s; label 8: 1463 s vs 655 s). CPU studies ran about 20 times slower; that factor is an agent statement, not a saved measurement. Do not run CPU studies while timing GPU runs; record the load average with each timing. | M: `runs/q1/*.json` `wall_s`. R: `option-c-evidence-2026-11-01/sec_wake.md` line 102; `r0-evidence-2026-10-31/sec_tests.md` line 103; transcript text "the last study ran about 20x slow" |
| 3 | **Daylight-saving change.** On 2026-11-01 at 02:00 PDT the clock went back to 01:00 PST. Local times 01:00 to 02:00 occur twice. In `runs/q1`, `m2r_default_0.578000.json` (01:00:08 -0800) is later than `m2r_default_0.588000.json` (01:54:28 -0700). Commits show both offsets (`b3e03f2` -0700, `a0d1bd2` -0800). Use UTC or offset-aware stamps (`ls --time-style=+%FT%T%z`); do not sort runs by local wall time. | M: file mtimes with offsets, `git log --date=iso` |
| 4 | Run directories and logs are git-ignored (`inversion/runs/`, `*.log`, `*.npz`). Findings cite them by path; they survive only on this disk. | M: `git check-ignore -v` |
| 5 | Stopped campaigns leave gaps: only 1 of 9 label-4 seed runs finished (no 4x seed spread); P7 step 5 has 76 slice points missing (skipped by D34). | R: technical record 12.2; finding 4; D34 |
| 6 | Before a consequential run, record code state, dirty changes, configuration, environment, command, and output paths in the experiment record. | R: `CLAUDE.md` "Research continuity" |
| 7 | Gates: stop at each phase gate and ask. Never start P8 or P9 without the user (D34 pauses them). | R: m3 note Rules 5; D34 |

#### 2.6 Plan risk table (Section 9) items still open

| Risk | State at this checkpoint | Label, source |
|---|---|---|
| Gradient growth over long horizons | Realized: chaos after step 1200; AD unusable at \(N_t = 2000\) | R: plan Section 9 last row, 12.10 |
| FD reference invalid past \(n_c\) | Realized: \(K(h) > 0\) at every \(h\) from \(N_t = 650\) | R: plan 12.10 step 4 |
| Emission-threshold jumps | Small at tested \(h\) (frozen = free to 3 digits) | R: plan 12.10 step 5 |
| EV model bias (L5, T2) | Not tested: P8 not run | R: D34 |
| Delayed emission (immediate-emission reference, T1) | Not settled: P8 step 1 not run | R: replan P8 |
| Mean-charge ion closure | Twin-XS fails 2 SE (4.3 SE); user deferred to P9/T2 | R: m1 note "P3 results so far" |

---

### 3. Numbers to plan with

#### 3.1 Forward (primal) run cost, Twin-S, 2000 steps, EV, one GPU

| Label (`p_nr` x `p_nt`) | Wall time per run, unloaded host | Wall time under contention | Label, source |
|---|---|---|---|
| 1 (2 x 8) | 86 to 90 s (P7 cost 85.9, 86.0 s; Q1 86 to 87 s; P3 88.8 s nominal, 87.6 to 101 s corners) | 89 to 90 s (log) | M: `runs/p7_fe7188bf9100/cost_primal_*.json`, `runs/q1/m1_*.json` `wall_s`, `results/p3_ev_runs.json` |
| 2r (4 x 8) | 167 to 168 s | none seen | M: `runs/q1/m2r_*.json` |
| 2t (2 x 16) | 218 to 225 s | 578 s | M: `runs/q1/m2t_*.json` |
| 4 (4 x 16) | 325 to 334 s | up to 749 s | M: `runs/q1/m4_*.json`; `runs_q1_scan_m4.log` |
| 8 (8 x 16) | 655 to 709 s | up to 1463 s (1526 s in log incl. start-up) | M: `runs/q1/m8_*.json`; `runs_q1_conv_gpu1.log` |
| 16r (16 x 8) | not measured unloaded | 1251 to 1633 s (shared host) | M: `runs/q1/m16r_*.json` |
| Production stochastic (`Simulation.step`), label 1 | 22.5 s (11 ms per step) | n/a | R: plan line 1161 |
| Twin-XS EV immediate, 300 steps | 6.7 s | n/a | M: `results/p3_ev_runs.json` `twin_xs:immediate` |
| T0 target run (`p_nz` 2) | 283 s | n/a | R: plan line 1307 |

Scaling: wall time grows about linearly with particle count (label 1 to 8: 87 s to about 680 s, factor 7.8). Label 16r has the same count as label 8 but ran about 2 times longer; contention is a likely cause (not isolated). E: arithmetic from the table.

#### 3.2 Tangent cost multiple

| Quantity | Value | Label, source |
|---|---|---|
| \(K = 1\) tangent step / primal step, no EV | 1.96 (Twin-S, 17.97 vs 9.17 ms), 1.98 (Twin-XS), 2.18 (L4w) | M: `results/p5_tangent_cost.json` |
| Gradient (`value_and_grad`, \(K = 1\), EV, residual objective), label 1 | 298 to 300 s, 3.47 run-equivalents | M: `runs/p7_fe7188bf9100/cost_grad_*.json`, `grad_*.json` `wall_s`; `runs_p7_cost.log` |
| AD per step with EV | 26.4 ms (to step 500, P6); 157 ms (P7 sweep with \(J_{chk}\) and cell keys); 228 ms at step 1200 (profile, synchronized hooks) | R: plan 12.9 step 4, 12.10 steps 3 and 4 |
| `jvp` with per-species callbacks (Q1) | label 1: 305 s and 345 s (with AD columns); label 4: 1299 s; label 8: 2438 s (GPU 0, shared host) | M: `results/q1_tangent_growth_m{1,1_ad,4,8_ad}.json` `wall_s` |
| AD-to-primal multiple by label | 1: 3.5 to 4.0; 4: about 3.9; 8: about 3.4 to 3.7 (contended runs) | E: ratios of the rows above |
| Central FD per parameter | 2 runs; one dogbox iteration with center is 3 runs | R: plan Section 8; 12.10 step 3 |
| Reduced Twin-S gradient | \(\epsilon = 10^{-4}\): 93 s; \(10^{-5}\): 101 s; \(10^{-6}\): 111 s; \(\epsilon = 0\): 426 s | M: `runs/p7_fe7188bf9100/eps_grad_*.json` |
| P7 usability center run (AD + frozen FD at 4 \(h\) + free FD at 3 \(h\) + contributions) | about 1483 to 1521 s per center | M: `runs_p7_usability.log` |

#### 3.3 Memory (label 1 unless stated)

| Quantity | Value | Label, source |
|---|---|---|
| Primal EV run: mempool used / reserved high | 2.53 / 3.42 to 3.49 GB; device in use 3.9 to 4.0 GB; host RSS 1.42 GB | M: `runs/p7_fe7188bf9100/cost_primal_*.json` `memory` |
| Gradient \(K = 1\): mempool used / reserved high | 3.77 to 3.79 / 6.21 to 7.15 GB; device in use 6.7 to 7.6 GB; host RSS 1.0 to 1.2 GB | M: `runs/p7_fe7188bf9100/grad_*.json`, `center_*.json` |
| T0 run at `p_nz` 2 | 5.17 / 7.42 GB | R: plan line 1307, Section 8 |
| Reduced Twin-S gradient at \(\epsilon = 0\) | 4.58 / 8.99 GB | M: `eps_grad_0.json` |
| Twin-S tangent state | about 0.9 GB per tangent (8M live particles) | E: plan Section 8 |
| Labels 2r to 16r | not recorded (`pool_bytes_end` is 0) | M |
| Label 8 gradient | below 48 GB (the AD run finished); about 30 GB if the label-1 figure scales linearly with particles | M: run finished; E: scaling |
| Cell-key records for L4 | Twin-XS full: about 180 MB; Twin-S sampled stride 256: 5.9M of 1.52G keys per FD pair | R: plan lines 648, 1717 |
| Disk | `runs/p1` 453 MB; `runs/p3_*.npz` about 3 GB; `runs/q1` 420 KB; `runs/p7_*` 1.1 MB | M: `du -sh` |

#### 3.4 Campaign-level wall times

| Campaign | Cost | Label, source |
|---|---|---|
| Rest of P7 step 5 (76 slice points) | about 6.3 h on one GPU | R: m3 note Resume; plan line 1874 |
| Full P7 step 5 (123 gradient runs) | about 6 to 8 h | R: replan P7 step 5 |
| P7 step 4 sweep | about 1.8 to 3.5 h | R: plan Section 8 |
| L6 recovery (P9) | 10 to 30 h serial | R: plan Section 8 |
| L5 stochastic referee (P8) | about 0.2 to 1.6 h for 8 to 64 seeds at 22.5 s | R: plan line 827 |
| Q1 study (planned) | about 22 runs, 2 to 3 GPU-h | R: Track R finding line 81 (estimate) |
| One label-8 FD secant (3 runs) on 2 GPUs | about 0.4 h unloaded | E: 3 x 690 s over 2 GPUs |
| One label-8 seed spread (3 seeds x 2 points) | about 1.2 h on 2 GPUs | E: 6 x 690 s over 2 GPUs |

---

### 4. Sources I could not check

1. The "about 20 times" CPU slowdown has no saved measurement; I found only the agent statement in the transcript and the 1833 s vs 7 s note in `sec_tests.md` (not a clean ratio).
2. Unloaded wall time for label 16r was never measured.
3. Memory for labels 2r to 16r was not recorded.
4. I did not open `runs/p4`, `runs/p5` logs, `runs/p1` npz contents, or any `work_*` script under the notes directories.
5. I did not verify whether the emission-record fix of P6 step 0 changes the reported mempool numbers (plan line 1424 says up to about 4%).
