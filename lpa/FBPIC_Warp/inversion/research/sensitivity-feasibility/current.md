---
investigation: "sensitivity-feasibility"
question: "How can a useful gradient of the 11 Twin-S beam residuals with respect to laser parameters be computed at N_t = 2000, in forward mode, for later multi-parameter recovery?"
goal: "Differentiate through particle trajectories macroscopically (U23), as the basis of multi-parameter laser recovery from beam data (U20)."
research_mode: "brainstorm"
mode_source: "U27"
decision_log: "decisions.md"
authorized_actions:
  - "Discuss the next research direction against the converged reference (U27)."
constraints_in_force:
  - "Forward mode only; no reverse mode or checkpointing for now (U22)."
  - "FD-type Jacobians set aside for now (U24)."
  - "Option C stopped at the 1D stage; no fbpic/autodiff implementation (U26)."
  - "One GPU only; no MPI or NCCL in the differentiable path (U02, U05b, plan D1)."
  - "Production stepping unchanged; the differentiable path uses its own stepper (plan D17, planner default)."
  - "Never sync-push snidhan/dev-autodiff; it is based on snidhan/dev-main (U07)."
open_decisions:
  - "Run a mechanism study (which particles and which time window carry the macroscopic response and the tangent growth). Proposed by the agent on 2026-11-01; the user chose to discuss first (U27)."
  - "Adopt p_nr = 8 as the Twin-S reference resolution. Proposed by the agent on 2026-11-01 (finding 5)."
  - "D35: close Track R and use FD-type Jacobians. Proposed in the R0 note, 2026-11-01. U24 set FD aside for now; Track R work is not active."
  - "Split-horizon method: exact tangent until a cut time, then a macroscopic treatment of the chaotic tail (tail restarts along the tangent, or a collective-mode shadowing projection). Proposed by the agent on 2026-11-01; not studied."
open_questions:
  - "Q1: When does the exact AD column first depart from FD on the converged model, and is the early departure carried by a few threshold particles or by many?"
  - "Q2: Is the late tangent growth confined to a few collective unstable directions (NILSS-style shadowing feasible)?"
  - "Q3: How much of the final FD response is already present in the bunch state at a candidate cut time?"
  - "Q4: Why is the full-horizon AD column not reproducible between runs (63% difference at 1x)? GPU-atomic rounding amplified by the growth is the hypothesis."
paused:
  - "FD recovery, Phases P8 and P9, P7 branch choice, 76 of 123 P7 step 5 slice runs (U20, plan D34)."
latest_findings:
  - "findings/2026-11-01-option-c-likelihood-ratio-no-go.md"
  - "findings/2026-11-01-converged-model-ad-versus-fd.md"
  - "findings/2026-11-01-radial-particle-convergence.md"
  - "findings/2026-11-01-particle-count-roughness-and-tangent-growth.md"
  - "findings/2026-11-01-macroscopic-gradient-methods.md"
evidence_boundary: "GPU run outputs of 2026-11-01 inspected and analyzed; agent study reports read through their summaries and selected sections; no agent script rerun; earlier phases from plan Section 12 and the run ledger."
last_checkpoint: "2026-11-01T12:45:00-08:00"
---

# Current position: full-horizon sensitivity feasibility

Brief: [reader-brief.md](reader-brief.md). Decisions: [decisions.md](decisions.md). What was run and what failed: [run-ledger.md](run-ledger.md). Methods: [methods-explainer.md](methods-explainer.md). Provenance and branches: [technical-record.md](technical-record.md) Sections 12 and 13.

On a converged model (`p_nr` = 8; the Jacobian is within about 10% of `p_nr` = 16), a macroscopic Jacobian exists. Central FD secants keep their direction from \(h = 10^{-2}\) to \(10^{-3}\) (cosine 0.96 to 0.99) and agree across three injection seeds to 4% ([finding 7](findings/2026-11-01-converged-model-ad-versus-fd.md)). Exact forward AD misses it by about \(10^4\) at step 2000 and by 9 to 12 times at step 1200. The tangent growth is collective, and more particles do not stop it ([finding 4](findings/2026-11-01-particle-count-roughness-and-tangent-growth.md)). All earlier Twin-S gradients used `p_nr` = 2, which gives the wrong Jacobian direction ([finding 5](findings/2026-11-01-radial-particle-convergence.md)).

Three macroscopic methods were studied and rejected for this problem: the particle-pdf method (finding 3), the gauge-projected tangent (finding 6), and option C, noise-regularized likelihood-ratio weights (finding 8, stopped by U26). The research mode is brainstorm (U27): the user asked to discuss the next direction before any new study. The saved plan still lists Track R phases R1 to R4 and the P8 and P9 recovery; U20, U24, and U26 prevail over that plan text.

The next useful discussion is the choice between a mechanism study and a split-horizon method design, both open proposals above.
