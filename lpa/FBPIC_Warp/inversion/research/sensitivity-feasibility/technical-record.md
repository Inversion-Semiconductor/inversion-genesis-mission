# Technical record: full-horizon sensitivity feasibility

Checkpoint: 2026-11-01. Links resolve from this directory. Position: [current.md](current.md). Brief: [reader-brief.md](reader-brief.md).

## 1. Source state inspected today

Source: orchestrator preflight [source-manifest.json](/home/snidhan/.cache/research-workflow-qa/2026-11-01/source-manifest.json) (host `snidhan-dt`, captured 2026-11-01T07:47:09Z). This session had no shell. It did not run Git. The session-start hook reported the same HEAD and a dirty tree.

| Item | Value |
|---|---|
| Repository | `snidhan-dt:/mnt/work/snidhan/projects/applied-research/solvers/fbpic/inversion-genesis-mission-worktrees/dev-autodiff` |
| Branch and HEAD | `snidhan/dev-autodiff`, `b3e03f2e32b194a21c5e93e8669624577966d9d0` |
| Tracked change | `lpa/FBPIC_Warp/DIFFERENTIABLE_INVERSION_PLAN.md` modified: status line, header, R0 result block in Track R, row D34. No staged changes. |
| Untracked | `r0-design-note-2026-10-31.md`, the `r0-evidence-2026-10-31/` directory, `inversion/q1_roughness.py`, `inversion/q1_tangent_growth.py` |
| Manifest scope | The manifest lists 11 unrelated cuDSS files and a second checkout (`cudss_multi_dof`). They are outside this investigation and were not used. |
| Hash status | SHA-256 values in the manifest are taken as reported. This session did not recompute them. The two `q1_*` scripts are not in the manifest file list, so they have no recorded hash. |

This is the inspected state. It is not the tested state of any historical run. Historical runs are in Section 3.

## 2. Inspected sources and coverage

Requested scope: the whole investigation. Coverage labels: Full = read end to end. Bounded = selected line ranges. Spot = only named values checked.

| Source | Coverage | Used for |
|---|---|---|
| [DIFFERENTIABLE_INVERSION_PLAN.md](../../../DIFFERENTIABLE_INVERSION_PLAN.md) (211 KB, about 1950 lines) | Bounded: lines 88 to 270 (purpose, loss, optimizer, stages); 856 to 897 (Track R and R0 result); 1396 to 1431 (P3 recovery); 1737 to 1877 (P7); heading list; grep of decisions D27 to D34 and gate text. Not read: Sections 3 to 6 (except grep hits), 8 to 9, 12.1 to 12.4, 12.6 to 12.9 bodies, appendices. | Question, decisions, P7 numbers, Track R text |
| [m3-state-2026-10-31.md](../../notes/m3-state-2026-10-31.md) | Full | Resume instructions, P6 and P7 status |
| [m2-state-2026-10-31.md](../../notes/m2-state-2026-10-31.md) | Bounded (first 40 lines) | P5 results, rules |
| [replan-p6-p9-2026-10-31.md](../../notes/replan-p6-p9-2026-10-31.md) | Full | Old P6 to P9 plan, decisions A to D |
| [m1-state-2026-10-30.md](../../notes/m1-state-2026-10-30.md) | Not read (in manifest only) | Gap |
| [optimizer-literature-2026-10-30.md](../../notes/optimizer-literature-2026-10-30.md) | Not read (in manifest only) | Gap; dogbox and DFO context taken from the plan |
| [r0-design-note-2026-10-31.md](../../notes/r0-design-note-2026-10-31.md) (65 KB) | **Full** (lines 1 to 535, bounded reads covering the whole file) | Main synthesis |
| [sec_vm.md](../../notes/r0-evidence-2026-10-31/sec_vm.md), [sec_gradf.md](../../notes/r0-evidence-2026-10-31/sec_gradf.md), [sec_rv.md](../../notes/r0-evidence-2026-10-31/sec_rv.md), [sec_multi.md](../../notes/r0-evidence-2026-10-31/sec_multi.md), [sec_tests.md](../../notes/r0-evidence-2026-10-31/sec_tests.md) | Bounded: heading lists and the result-in-brief section of each (first 14 to 20 lines). Bodies, derivations, and 1D experiment tables not read. | Cross-check of design note claims |
| [crit_physics.md](../../notes/r0-evidence-2026-10-31/crit_physics.md) | Bounded: Sections 0 to 2, 4, 5 (verdict, confirmations, central question, strongest no-go). Findings F1 to F11 read only through the design note's resolution table. | Independent critique |
| [crit_feasibility.md](../../notes/r0-evidence-2026-10-31/crit_feasibility.md) | Bounded: Sections 0, 3, 8, 9, 10. | Independent critique |
| Saved outputs | See Section 4 | Numerical claims |

Not inspected: the 31 `work_tests/out_secant_*.json` files, `results_all.jsonl`, `cold_beam_bias.json`, `ref_scan*.json`, `e1*_results.json`, `e2*_results.json`, `pendulum_pdf.json`, `cost_model.out`, `pdf_mesh_scan*.out`, `slice_ref*.out`, `slice_scale.out`, `r2b_on_twins.out`, `rerun_*.out`, all Python scripts except the two `q1_*` headers, all P7 cached run JSON files, and the Chung 2020 PDF. Claims that rest on these are "report read".

## 3. Historical tested state of the key measurements

| Measurement | Tested code identity (reported) | Provenance gaps |
|---|---|---|
| P3 FD recovery (Twin-S, EV) | Simulation-code hash `804032cdc3e2` of `forward.py`, `twin_config.py`, `ionization.py`, `stepper.py`, `particles.py` at the commit; 311 cached runs in `inversion/runs/p3`; targets T0 `90ee4adeb6e6`, T0' `88783bf73c69` (plan 12.5). | The T0 target records `git_hash 1d9272e` with no dirty marker. Targets are not committed. |
| P7 steps 1 to 6 | Code on top of commit `4c5ff56` (plan 12.10). Cache directory `inversion/runs/p7_fe7188bf9100/`, named by a hash of `fbpic/autodiff/*.py`, `forward.py`, `twin_config.py`, `residual_objective.py`. Physical GPU 1 (RTX 6000 Ada), FP64. | Whether the working tree was dirty during the runs is not stated in what I read. Commit `1119983` is the P7 checkpoint. The cache hash was not recomputed. Seeds: the P7 runs use the Twin-S injection seed of the config. The seed value was not read. |
| R0 CPU scripts | CPU only, NumPy, run by R0 agents on 2026-10-31. | No commit identity. The `work_*` directories are untracked. No environment, no seeds listed in the files I read, except those printed in some .out files. |
| `work_note/*.out`, except `holder_check.out` and `timeline_check.out` | Computed from `inversion/results/p7_jacobian.json` and `inversion/runs/p7_fe7188bf9100` (design note Appendix C). `gate_calibration.out` and `detrended_spread.out` are in this group. | The P7 results JSON is from commit `1119983` or later. The exact JSON checksum is not in the manifest. |
| `work_note/timeline_check.out` | Computed model from `work_note/timeline_check.py`. Its inputs are typed by hand: the Twin-S configuration values (`:3` to `:7`) and the P7 tangent norms at steps 1200, 1500, 1800 and 2000 (`:16`). The script reads no P7 JSON and no cached run. It assumes a bunch lag of 12 um behind the laser (`:11`, `:14`). | The growth rates are arithmetic on copied plan norms. The laser and bunch timeline depends on the assumed lag. Typing errors are not checked here. It does not identify the growth mechanism. |
| `work_note/holder_check.out` | Computed transformation of an FD table typed by hand from the plan step 4 table (`work_note/holder_check.py:1` to `:15`). It uses no P7 JSON or cached run. | It does not independently corroborate the GPU measurements. The plan table is the measurement source. Typing errors are not checked here. |

The CLAUDE.md of the package asks for fixed-seed comparison and per-step timing for kernel changes. No kernel change occurs in this checkpoint.

## 4. Saved outputs inspected

| Output | Values checked | Supports |
|---|---|---|
| [holder_check.out](../../notes/r0-evidence-2026-10-31/work_note/holder_check.out) | \(A(h)\) five values; 0.73 and 0.81 exponents; 0.174; 2.43. Input table typed from the plan (see Section 3). | Roughness claims, as arithmetic on copied plan values only |
| [detrended_spread.out](../../notes/r0-evidence-2026-10-31/work_note/detrended_spread.out) | 41 points; degree 3 and 5; four \(h\) | 15% to 44% range |
| [gate_calibration.out](../../notes/r0-evidence-2026-10-31/work_note/gate_calibration.out) | All nine rows | Solution-slice contraction and cosine statistics (partial calibration). It does not certify the full proposed gate; false-minimum and far-field slices are incomplete. |
| [timeline_check.out](../../notes/r0-evidence-2026-10-31/work_note/timeline_check.out) | \(\omega_{p0} = 1.595\cdot10^{14}\) s\(^{-1}\); laser exit step 1520; rates 0.52, 0.80, 0.13 \(\omega_{p0}\) | Mechanism not identified. Inputs typed by hand; assumed 12 um bunch lag (see Section 3 lineage row). |
| [slice_secant_spread.out](../../notes/r0-evidence-2026-10-31/work_crit_feasibility/slice_secant_spread.out) | Six \(h\) rows | Raw spread |
| [onesided_gate.out](../../notes/r0-evidence-2026-10-31/work_crit_feasibility/onesided_gate.out) | Nine stencil rows | Reference dependence |
| [cost_compare.out](../../notes/r0-evidence-2026-10-31/work_crit_feasibility/cost_compare.out) | Cost table, wall-time table | Cost estimates |
| [adjoint_pdf_1d1v_N5e4_Nt2000.out](../../notes/r0-evidence-2026-10-31/work_multi/adjoint_pdf_1d1v_N5e4_Nt2000.out) | Transpose \(9.71\cdot10^{-16}\); FD table | Adjoint and trapping bias |
| [pendulum_summary.log](../../notes/r0-evidence-2026-10-31/work_crit_physics/pendulum_summary.log) | Error table and bounce periods | Integrable failure |
| [small_sigma_T10.log](../../notes/r0-evidence-2026-10-31/work_gradf/small_sigma_T10.log) | Three rows, columns unlabeled | Cold-beam bias (partly) |
| [p7_jacobian.json](../../results/p7_jacobian.json) | Grep of three keys: 2010.41, 7103.41, \(-0.144\) | P7 usability |

Level reached for all: "Saved output inspected". No "Check rerun" or "Experiment rerun" level was reached in this checkpoint.

## 5. Path reconciliation

| Source shorthand | Actual path |
|---|---|
| `r0/` | `inversion/notes/r0-evidence-2026-10-31/` |
| `sec_vm`, `sec_gradf`, `sec_rv`, `sec_multi`, `sec_tests`, `crit_*` | `.md` files in that directory |
| `scratchpad/r0/work_gradf/` (sec_gradf header) | `inversion/notes/r0-evidence-2026-10-31/work_gradf/` |
| `r0/work_note/*.py` | `inversion/notes/r0-evidence-2026-10-31/work_note/` |
| `inversion/runs/p7_fe7188bf9100` | exists; cache of P7 runs |
| `inversion/results/p7_jacobian.json` | exists |
| `inversion/results/p5_tangent_cost.json` (design note 6.1) | Not checked. |
| Chung 2020 local PDF | `/mnt/work/snidhan/projects/applied-research/solvers/fbpic/literature/Chung2020_regular_sensitivity_PIC_OSTI1570259.pdf` (outside the worktree; not opened) |

Several saved log files are empty (`ref_scan.log`, `ref_scan_T10.log`, `errors.log`, `errors75.log`; 0 bytes in the manifest). Their JSON companions exist.

## 6. Key problem setup (for continuation)

1. Twin-S: about 800 by 50 physical cells, about 1800 to 2000 steps, 2 azimuthal modes, nitrogen K-shell EV ionization, lab frame, one GPU (plan 3.3). All species load cold (`inversion/twin_config.py:272-275`, per the design note; code not re-read).
2. Parameters: \(\hat E_L\) and \(\hat z_f\), normalized to \([0,1]\). Stage A uses \(\hat E_L\) only, with \(\hat z_f = 0.4\) fixed. Nominal solution \(\hat E_L = 0.588\) (T0) and 0.600 (T0').
3. Observable: `elec_from_N` state at \(N_t = 2000\), weights \(\sigma_i\) a quintic smoothstep on \(u_z\). Loss \(J = \lVert r_w\rVert^2\) with 11 whitened residuals (plan 2.4). The 11 residuals depend on 13 weighted sums.
4. FD steps: \(h = 0.02\) for \(E_L\) and 0.05 for \(z_f\) in the P3 recovery (plan 2.5 item 4). The optimizer is `scipy.optimize.least_squares(method="dogbox")`.
5. Truth variants: T0' uses the inversion settings, so it tests convergence only. T0 uses `p_nz = 2` and a different injection seed, so its offset is model bias. The "references fitted to the same slice" caveat applies to the gate, not to these truth variants.
6. P3 recovery numbers (plan 12.5): T0' \(E_L\) from design start 0.85 ends at 0.60000 (error \(1.5\cdot10^{-5}\)); \(z_f\) recovers to \(\le 1.3\cdot10^{-4}\) from three starts. A design start at 0.55 stopped in a false minimum at \(\hat E_L \approx 0.545\) (\(J = 12.6\)). The 2D gate failed: the basin radius is about 0.04 and the 25-point design spacing is 0.2. T0 model bias: \(-0.012\) in \(\hat E_L\) and \(-0.015\) to \(-0.016\) in \(\hat z_f\). Each solve took 5 to 8 Jacobians, 15 to 27 minutes.

## 7. Limitations by topic

1. Particle counts: P7 used about 1.9 M loaded macroparticles (`p_nt = 8`) plus 12.68 M EV births. The proposed particle-count comparison uses 1x and 4x. Section 11 inventories new tangent-growth files at those multipliers. Their scientific contents and the roughness diagnostic remain unreviewed in this update.
2. Dimensionless perturbation: all \(h\) values are in \(\hat E_L\). Conversion to physical energy needs the plan's bounds. It was not extracted.
3. Measurements versus estimates: P7 step 3 timings, P7 slice statistics, and the R0 CPU outputs are measurements. The Track R cost figures, the 30 to 50 day effort, the Gram assembly cost, the memory figures, and the DFO-LS behavior are estimates or literature claims.
4. One nominal point and one slice: roughness statistics come from one center (\(h\) scan) and one 41-point slice. No other parameter or point is covered.
5. Debye test: it is a warm near-equilibrium 1D case. The design note and critic both say it tests the implementation, not the Twin-S hypothesis.
6. R0 CPU toys (1D-1V, standard map, pendulum): they are surrogates. They do not include the quasi-cylindrical geometry or PSATD. The R2 pilot in `sec_tests.md` includes a simplified EV ionization injection (`sec_tests.md:10`, `:105` to `:107`, `:236`). The standard-map and pendulum tests have no births. Birth models differ by test.
7. Run provenance for the R0 CPU scripts is unknown beyond the file timestamps in the manifest. Treat their outputs as reported-and-inspected, not reproducible by commit.

## 8. Requirements for reopening Track R (source conditions, not a decision)

From design note Sections 2, 7, 8.2:

1. Q1 shows \(A_4/A_1 \le 0.6\) at every \(h \le 10^{-3}\). Kill if \(\ge 0.85\).
2. Q2: the 90th-percentile tangent norm of `electrons` does not grow by \(10\times\) per 200 steps after step 1200.
3. A parameter or slice point exists where every FD step fails the guarded gate.
4. Q3 plateau tests in reset interval \(T\), basis degree \(p\), and slab width. Cold-node residual below 10% of converted weight.
5. Measured \(c_1 \le 1.3\) after a deposit-tangent rewrite.
6. A matched-scale reference stable over a factor 2 in \(\ell\).

## 9. Restart context (reference only)

These commands come from [m3-state](../../notes/m3-state-2026-10-31.md) and the design note. They are **not** authorized. D34 pauses the P7 step 5 completion.

```bash
# Reference only. Not authorized. From lpa/FBPIC_Warp, GPU 1, env fbpic-warp.
python inversion/p7_jacobian.py usability   # 76 remaining slice points, about 6.3 h
python inversion/q1_roughness.py scan 1     # proposed Q1, about 22 runs in total
```

Rules to keep: never push or `sync-push` on this branch; do not edit `fbpic/autodiff/*.py`, `forward.py`, `twin_config.py`, or `residual_objective.py` while the P7 cache must stay valid; production modules stay unchanged (D17).

## 9a. Position detail moved from the top layers (QA correction 2026-11-01)

This section holds detail that `current.md` and `reader-brief.md` formerly repeated.

### Intent attribution

The pause comes from plan decision D34, recorded as a user decision dated 2026-10-31 (`DIFFERENTIABLE_INVERSION_PLAN.md:986`; the plan lists D21 to D34 as user decisions at line 11). The checkpoint prompt for this investigation only summarized that recorded decision ("The current source plan says the user paused recovery"). The prompt is not a new human statement. The earlier text "the user also stated in this session" was unsupported and is withdrawn. The documentation checkpoint and this QA task are authorized by explicit requests, and they authorize writing this directory only.

### Three separate claims

| Claim | Status | Evidence |
|---|---|---|
| Implementation agreement | AD matches FD on Twin-S at \(N_t = 460\) and 500 where the zero-flip step exists. L1 and L3 checks are reported as passing in the plan. | Plan 12.9, 12.10 step 4 |
| Useful sensitivity | Solution-slice thresholds met by FD columns only, on one slice and one parameter, against a reference fitted to the same slice. Not met by AD at \(N_t = 2000\). Not tested for any Track R method on Twin-S. Full gate not satisfied (false-minimum and far-field slices incomplete). | [gate_calibration.out](../../notes/r0-evidence-2026-10-31/work_note/gate_calibration.out) |
| Actual recovery | FD only, one parameter at a time, on target T0' (inversion settings). The two-parameter FD gate failed on global search. No AD or Track R recovery exists. | Plan 12.5 |

### Older resume instructions and their status

| Older instruction | Where | Status now |
|---|---|---|
| "To close P7": run the remaining 76 step 5 slice points (about 6.3 h) | Plan 12.10 end; [m3-state](../../notes/m3-state-2026-10-31.md) | Superseded by D34. Reference only. |
| "Then the user chooses the recovery branch (D32)" | Plan 12.10; m3-state | Deferred by D34. |
| Phases P8 and P9 as planned | Plan Section 7; [replan](../../notes/replan-p6-p9-2026-10-31.md) | Paused by D34. |
| "Stop when P7 has reported. Do not start P8 or P9." | m3-state Rules | Consistent with D34. |
| Design note Section 8.3 lists "P7 step 5 completion, 0.5 day, 6.3 GPU-h" in the recommended path | [design note](../../notes/r0-design-note-2026-10-31.md) | Conflicts with the D34 skip. Running it would need a new user decision. |

### Class of each item

| Class | Items |
|---|---|
| Recorded user decisions | D34, D33, D32 (branch chosen from the P7 study, then deferred by D34). |
| Proposed by agent or reviewer, not accepted | D35 (close Track R at R0, use FD-type Jacobians). Replacement plan text (design note Section 10). Tests Q1, Q2, Q3. One-sided FD versus DFO-LS. Completing P7 step 5. Plan wording corrections. |
| Paused (D34) | FD recovery. P8, P9. P7 branch choice. 76 of 123 step 5 slice runs. |
| Open | Closure recommendation (proposed D35). |

The scripts `inversion/q1_roughness.py` and `inversion/q1_tangent_growth.py` were untracked in the original preflight. No Q1 output appeared in the 2026-11-01 07:47:09 UTC inventory. That absence is an earlier inventory observation. It is superseded for the tangent-growth outputs inventoried at 08:43:07 UTC (Section 11). Completion of the particle-count roughness diagnostic remains unestablished by this metadata update.

### Source date inconsistency

The plan says Phase R0 was done on 2026-11-01. The design note and most section files are dated 2026-10-31. `crit_physics.md` is dated 2026-11-01. This record does not resolve it.

### Plan text not yet corrected

The plan still contains the Track R "Reason" sentence that calls the growth the Lyapunov instability of the N-body dynamics. It also contains open problem 4 ("FD secants at \(h = 10^{-2}\) and \(10^{-3}\) agree"). The R0 study shows a 17.4% column-norm difference (from the hand-typed plan table) and an unidentified mechanism. The source files stay unchanged. See [finding 3](findings/2026-11-01-track-r-feasibility-study.md).

### QA correction log (2026-11-01)

Issues I1 to I8 of `/home/snidhan/.cache/research-workflow-qa/2026-11-01/independent-research-review.md` were checked against source lines. Each was supported. The variance-versus-standard-error wording issue was also supported (finding 3). No source file, plan, script, or result was changed.

## 10. Verification limits

1. No experiment ran. No check reran.
2. No shell was available. Git state comes from the orchestrator manifest and the session-start hook.
3. The plan was read in bounded sections. The synthesis note was read in full. The five derivation files and two critiques were read in bounded sections, not end to end.
4. No hash was recomputed. Date labels conflict between files (Section "Date inconsistency" in current.md).
5. Reading the two `q1_*` scripts (headers only) does not show that they run.


## 11. Evidence-inventory freshness update (metadata only)

The original source inventory was captured at 2026-11-01 07:47:09 UTC. It did not contain the two tangent-growth outputs below.

A later inventory observed both files at 2026-11-01 08:43:07 UTC. Their canonical paths, hashes, and metadata are retained in [evidence-inventory.json](evidence-inventory.json).

| Output | Particle multiplier | Normalized point | Saved rows |
|---|---:|---:|---:|
| `inversion/results/q1_tangent_growth_m1.json` | 1 | 0.588 | 40 |
| `inversion/results/q1_tangent_growth_m4.json` | 4 | 0.588 | 40 |

These fields were copied from the supplied inventory. No scientific contents were interpreted for this update.

These are tangent-growth outputs. File presence does not establish completion of the proposed particle-count roughness diagnostic or its scaling criterion.

No experiment or scientific analysis ran for this update. The files' tested code states, scientific validity, and implications remain unreviewed here.

The historical conclusions and dated findings are unchanged. D34 remains the latest recorded pause, and D35 remains a proposal.

This observation does not establish new user intent or authorize research execution. The original Git preflight remains historical metadata.

## 12. Checkpoint 2026-11-01, session 2553145c (new runs and method studies)

This section adds the work after the documentation-QA checkpoint above. Sections 1 to 11 stay as written. The earlier reader-brief content (roughness, gate calibration, Track R) is retained in findings 1 to 3 and Section 9a.

### 12.1 Source state at this checkpoint

| Item | Value |
|---|---|
| Branch and HEAD | `snidhan/dev-autodiff`, `b3e03f2e32b194a21c5e93e8669624577966d9d0` (unchanged) |
| Tracked changes | `DIFFERENTIABLE_INVERSION_PLAN.md` (D34 and R0 block, this session); `lpa/FBPIC_Warp/CLAUDE.md` and repository `README.md` (research-continuity links, written by another workflow, not this session) |
| Untracked, relevant | `inversion/q1_roughness.py` (sha256 `5f541a87...bd537`), `inversion/q1_tangent_growth.py` (`43016269...7b`), `inversion/results/q1_*.json`, `inversion/notes/r0-*`, `inversion/notes/gpt-evidence-2026-11-01/`, `inversion/notes/option-c-evidence-2026-11-01/`, this directory |
| Ignored | `inversion/runs/q1/` (per-point run JSON and logs) |
| Environment | conda env `fbpic-warp`, Python 3.11.16, Warp 1.18.0, CuPy 14.2.0, CUDA 12.9 toolkit, driver 13.2; 2 × RTX 6000 Ada (48 GB), one GPU per process via `CUDA_VISIBLE_DEVICES`; FP64 |
| Config | `inversion/configs/twin_s.json` sha256 `a489fc1b...55e2`; particle counts overridden in memory per run label (`q1_roughness.py` `MULTIPLIERS`, `EXTRA_RESOLUTIONS`) |

`fbpic/autodiff/*.py`, `forward.py`, `twin_config.py`, and `residual_objective.py` were not edited, so the P7 cache key stays valid.

### 12.2 Runs (experiment record)

| Family | Command (from `lpa/FBPIC_Warp`, `PYTHONPATH=$PWD`) | GPU | Outputs | Status |
|---|---|---|---|---|
| FD scan 1x | `python inversion/q1_roughness.py scan 1`, then `seeds 1` | 0 | `runs/q1/m1_default_*`, `m1_s{1,2,3}_*`; log `inversion/runs_q1_scan_m1.log` | 11 + 9 runs complete, about 90 s each |
| FD scan 4x | `python inversion/q1_roughness.py scan 4`, then `seeds 4` | 1 | `runs/q1/m4_default_*`, `m4_s1_0.588000.json`; log `inversion/runs_q1_scan_m4.log` | 11 scan runs complete; seeds stopped by the agent after 1 of 9 |
| Report | `python inversion/q1_roughness.py report` | none | `inversion/results/q1_roughness.json` (sha256 `927c3b48...6b02`) | complete |
| Tangent growth | `python inversion/q1_tangent_growth.py 1`, then `4` | 0 | `inversion/results/q1_tangent_growth_m{1,4}.json`; log `inversion/runs_q1_tangent.log` | complete (305 s and 1299 s) |
| Radial convergence | `python inversion/q1_roughness.py points {2r,2t,8}` | 0 and 1 | `runs/q1/m{2r,2t,8}_default_*`; logs `inversion/runs_q1_conv_gpu{0,1}.log` | complete |
| Radial convergence 16r | `q._child('16r', None, x)` for the three points | 0 and 1 | `runs/q1/m16r_default_*`; logs `inversion/runs_q1_conv16_gpu{0,1}.log` | running at this checkpoint |

Code identity: the 1x and 4x scans and seeds ran with an earlier revision of `q1_roughness.py` that lacked `EXTRA_RESOLUTIONS` and the `points` stage; the worker function was unchanged. The hash above is the current file. No commit records either revision.

Per-run fields saved: residuals at 7 snapshot steps, selected-particle counts, `elec_from_N` count, wall time. Particle-count multiplier labels: 1 = (`p_nr` 2, `p_nt` 8); 2r = (4, 8); 2t = (2, 16); 4 = (4, 16); 8 = (8, 16); 16r = (16, 8). All at `p_nz` 1, \(\hat E_L = 0.588\), \(\hat z_f = 0.4\), EV with free thresholds, \(\epsilon = 10^{-4}\).

### 12.3 Method-study provenance

| Study | Agents | Evidence location | Status |
|---|---|---|---|
| Gauge-projected tangent (workflow `macroscopic-tangent-study`, run `wf_17967109-4bb`) | derivation, prototype, alternatives, literature; critics and synthesis not run | [gpt-evidence-2026-11-01](../../notes/gpt-evidence-2026-11-01/) | Stopped by the agent after the derivation showed no fix on cold sheets. The prototype agent's partial outputs are in `work_proto/`; it wrote no section. |
| Option C (workflow `stochastic-lr-gradient-study`, run `wf_5aaf83a6-ec7`) | theory, literature, toy, wake, 2 critics, synthesis | [option-c-evidence-2026-11-01](../../notes/option-c-evidence-2026-11-01/) holds a snapshot of the theory and literature sections and `work_theory/`. Toy and wake outputs were in the session scratchpad (`lr/`) at this checkpoint. | Running. Copy the remaining outputs into the evidence directory when the workflow ends. |

All agent scripts are CPU NumPy, run with `python3 -I`, without commit identity or recorded seeds beyond what their outputs print. Downloaded literature PDFs (`lr/lit_dl/`) were not copied.

### 12.4 Verification at this checkpoint

1. Saved output inspected and analyzed: all `runs/q1` JSON files used in findings 4 and 5; both tangent-growth JSON files; `q1_roughness.json`.
2. Report read: agent sections `gpt-evidence/sec_derive.md` (Sections 0 to 2, bounded read), `sec_alt.md` (Sections 0 and 1), `sec_lit.md` (summary); `option-c-evidence/sec_theory.md` and `sec_lit.md` (structured summaries only).
3. Paper check: Chung et al. (2020) pages 3, 4, 9, 10, 34, 49, and 50 were read as images.
4. Not rerun: any agent script; any earlier finding's check.

### 12.5 User statements recorded in this session (2026-11-01)

These are conversation statements, not plan decisions.

| Statement (quoted or close paraphrase) | Meaning recorded |
|---|---|
| "let's do that no?" (in reply to the particle-count test with a plasma-exit observable) | Authorizes the Q1 runs |
| "you can keep focusing on forward mode for now please" | Forward mode only for now |
| "We need to come up with a novel way of differentiating through particle trajectory macroscopically somehow, that is the inherent novelty of this work I feel like" | Research direction |
| "let's not use A right now, let's focus on C with subagents deriving, evaluating, reviewing with a concrete goal of capturing the macrosopic gradients from particles trajectory" | FD-type Jacobians set aside for now; option C study authorized |

The agent's proposal to run FD recovery for Stage A in parallel was not accepted. D35 stays a proposal.

### 12.6 Restart context (reference only)

When the option C workflow ends: read its note (`lr/option-c-note.md` in the session scratchpad), copy `lr/` into `inversion/notes/option-c-evidence-2026-11-01/`, and add a finding. When the `16r` runs end: compare the step-2000 secant with `p_nr` 8 (finding 5 table). Neither step authorizes Twin-S implementation work.
