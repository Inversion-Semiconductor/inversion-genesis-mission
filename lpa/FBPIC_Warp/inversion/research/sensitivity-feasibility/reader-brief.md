---
research_workflow: 1
investigation: "Full-horizon sensitivity feasibility for FBPIC-Warp laser-parameter inversion"
updated: "2026-11-01T12:45:00-08:00"
source_revision:
  repository: "snidhan-dt:/mnt/work/snidhan/projects/applied-research/solvers/fbpic/inversion-genesis-mission-worktrees/dev-autodiff"
  head: "b3e03f2e32b194a21c5e93e8669624577966d9d0"
  dirty: true
  snapshot: null
current_position: "current.md"
decision_log: "decisions.md"
research_mode: "brainstorm"
run_ledger: "run-ledger.md"
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
  - "findings/2026-11-01-converged-model-ad-versus-fd.md"
  - "findings/2026-11-01-option-c-likelihood-ratio-no-go.md"
verification_scope: "GPU run outputs of 2026-11-01 inspected and analyzed. Agent study reports read through summaries and selected sections; their CPU scripts not rerun. User decisions quoted from the session transcripts. No experiment or check rerun for this checkpoint."
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
  - path: "../../results/q1_tangent_growth_m8_ad.json"
    kind: "measurements"
    status: "inspected"
  - path: "../../notes/option-c-evidence-2026-11-01/option-c-note.md"
    kind: "report"
    status: "inspected"
    note: "Sections 1, 2, 3.6, 4.5, 5 read."
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

On a converged model a well-defined macroscopic Jacobian exists, and exact AD does not recover it. At 8 radial macroparticles per cell (`p_nr` = 8), central FD secants keep their direction from \(h = 10^{-2}\) to \(10^{-3}\) (cosine 0.96 to 0.99) and agree across three injection seeds to 4%. The exact forward-mode AD column is about \(10^4\) times larger at step 2000, with cosine 0.26 to 0.41 ([finding 7](findings/2026-11-01-converged-model-ad-versus-fd.md)). The reason is collective tangent growth in the density downramp, which more particles do not stop ([finding 4](findings/2026-11-01-particle-count-roughness-and-tangent-growth.md)); the [methods explainer](methods-explainer.md) explains it with a worked example.

## Resolution matters

The Jacobian converges only at `p_nr` \(\ge 8\): it reverses direction from `p_nr` 2 to 4, shrinks 2.7 times from 4 to 8, and changes by about 10% from 8 to 16 ([finding 5](findings/2026-11-01-radial-particle-convergence.md)). All earlier Twin-S gradients, including the R0 roughness, used `p_nr` = 2.

## Methods studied

Three routes to a macroscopic gradient were studied in 1D and toy models. The particle-pdf method needs \(\nabla_u f\), which a cold plasma lacks (finding 3). The gauge-projected tangent cannot bound growth on a cold sheet (finding 6). Option C, momentum noise with per-particle Girsanov weights, is correct for the noisy model but has no usable noise window: noise small enough to keep the trapped charge within 5% leaves single-run errors 56 to 150 times the gradient ([finding 8](findings/2026-11-01-option-c-likelihood-ratio-no-go.md)). All method evidence is 1D, CPU only.

## Decision status

The research mode is brainstorm, set by the user on 2026-11-01 with "Discuss first" ([U27](decisions.md)). Earlier user decisions keep forward mode only (U22), set FD-type Jacobians aside for now (U24), and stop option C (U26). Open proposals: a mechanism study of where the macroscopic sensitivity and the tangent growth live, `p_nr` = 8 as the reference resolution, a split-horizon method, and D35. What was tried and why each attempt failed is in the [run ledger](run-ledger.md). See [current.md](current.md).
