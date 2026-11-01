---
research_workflow: 1
investigation: "Full-horizon sensitivity feasibility for FBPIC-Warp laser-parameter inversion"
updated: "2026-11-01T02:10:00-08:00"
source_revision:
  repository: "snidhan-dt:/mnt/work/snidhan/projects/applied-research/solvers/fbpic/inversion-genesis-mission-worktrees/dev-autodiff"
  head: "b3e03f2e32b194a21c5e93e8669624577966d9d0"
  dirty: true
  snapshot: null
current_position: "current.md"
technical_record: "technical-record.md"
explainer: "methods-explainer.md"
evidence_inventory: "evidence-inventory.json"
findings:
  - "findings/2026-10-31-ad-tangent-at-full-horizon.md"
  - "findings/2026-11-01-finite-range-roughness-and-usability-gate.md"
  - "findings/2026-11-01-track-r-feasibility-study.md"
  - "findings/2026-11-01-particle-count-roughness-and-tangent-growth.md"
  - "findings/2026-11-01-radial-particle-convergence.md"
  - "findings/2026-11-01-macroscopic-gradient-methods.md"
verification_scope: "New GPU run outputs inspected and analyzed in this session. Agent study reports read; their CPU scripts not rerun. No check of earlier findings rerun."
user_decision_status: "mixed"
sources:
  - path: "../../results/q1_roughness.json"
    kind: "measurements"
    status: "inspected"
  - path: "../../results/q1_tangent_growth_m1.json"
    kind: "measurements"
    status: "inspected"
  - path: "../../results/q1_tangent_growth_m4.json"
    kind: "measurements"
    status: "inspected"
  - path: "../../runs/q1/"
    kind: "measurements"
    status: "inspected"
    note: "Git-ignored run directory; per-point JSON files."
  - path: "../../notes/r0-design-note-2026-10-31.md"
    kind: "report"
    status: "inspected"
  - path: "../../notes/gpt-evidence-2026-11-01/sec_derive.md"
    kind: "derivation"
    status: "inspected"
  - path: "../../notes/gpt-evidence-2026-11-01/sec_alt.md"
    kind: "derivation"
    status: "inspected"
  - path: "../../notes/gpt-evidence-2026-11-01/sec_lit.md"
    kind: "literature"
    status: "reported"
  - path: "../../notes/option-c-evidence-2026-11-01/sec_theory.md"
    kind: "derivation"
    status: "reported"
  - path: "../../notes/option-c-evidence-2026-11-01/sec_lit.md"
    kind: "literature"
    status: "reported"
  - path: "/mnt/work/snidhan/projects/applied-research/solvers/fbpic/literature/Chung2020_regular_sensitivity_PIC_OSTI1570259.pdf"
    kind: "literature"
    status: "inspected"
    note: "Pages 3, 4, 9, 10, 34, 49, 50 read as images in this session."
---

# Can the Twin-S beam response be differentiated at the full horizon? Reader brief

## Question and current answer

The project recovers laser parameters from a synthetic electron beam with a gradient-based optimizer. It needs the Jacobian of 11 beam residuals with respect to laser energy at \(N_t = 2000\) steps, in forward mode, and later for many parameters.

Exact derivatives do not work at this horizon. The exact per-particle tangent grows collectively in the density downramp from about step 1450. With 4x more particles the growth rate falls by only 10% to 30%, and the step-2000 tangents are as large or larger ([finding 4](findings/2026-11-01-particle-count-roughness-and-tangent-growth.md)). The reason is explained with a worked example in the [methods explainer](methods-explainer.md), Section 2.

## A resolution problem underneath

The Twin-S Jacobian depends on the number of radial macroparticles per cell. From `p_nr` 2 to 4 its direction reverses (cosine \(-0.69\) at step 2000). From 4 to 8 its direction holds (cosine 0.93) but its norm falls from 199 to 73. The observables move by less than 0.7 whitened units ([finding 5](findings/2026-11-01-radial-particle-convergence.md)). All earlier Twin-S gradients used `p_nr` 2. A run at `p_nr` 16 was pending at this checkpoint.

## Methods under study

The goal is a macroscopic gradient from particle trajectories. Local averaging of tangents and the gauge-projected (Wasserstein) tangent do not stop the growth on a cold plasma. Option C adds a small momentum noise and carries Girsanov weights \(d\omega_i = \sigma^{-1}\,\delta F_i\cdot dW_i\) on unperturbed paths. It is exact for the noisy model and needs no \(\nabla_u f\). It faces laser-quiver variance, a noise feedback loop, an untested regime after trapping, and a cost of about 0.8 to 2.5 runs per parameter (estimate). An earlier 1D test found its bias against the noiseless model to be 20% to 64% ([finding 6](findings/2026-11-01-macroscopic-gradient-methods.md)). Its evaluation was still running.

## Decision status

The user chose on 2026-11-01 to focus on option C in forward mode and to set FD aside for now. No plan decision records this yet. D35 remains a proposal. Open: whether the target is a noise-regularized gradient or the pointwise continuum gradient, and which resolution defines the Twin-S reference. See [current.md](current.md).
