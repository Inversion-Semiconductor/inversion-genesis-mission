# Current position: full-horizon sensitivity feasibility

Checkpoint 2026-11-01 02:10 PST (session 2553145c). Brief: [reader-brief.md](reader-brief.md). Methods: [methods-explainer.md](methods-explainer.md). Evidence: [technical-record.md](technical-record.md) Section 12. The earlier 2026-11-01 documentation-QA checkpoint is preserved in technical-record Sections 1 to 11.

**Question.** How can a useful gradient of the 11 Twin-S beam residuals with respect to laser parameters be computed at \(N_t = 2000\), in forward mode, for later multi-parameter recovery?

**Conclusion.** Exact derivatives are ruled out at this horizon. The exact tangent grows collectively from about step 1450, and 4x particles reduce the rate by only 10% to 30% ([finding 4](findings/2026-11-01-particle-count-roughness-and-tangent-growth.md)). The Twin-S Jacobian itself is not converged in radial particles: its direction reverses from `p_nr` 2 to 4, and its norm falls 2.7 times from 4 to 8 ([finding 5](findings/2026-11-01-radial-particle-convergence.md)). Candidate macroscopic-gradient methods are under study; none is validated ([finding 6](findings/2026-11-01-macroscopic-gradient-methods.md)).

**Latest user intent (2026-11-01, this conversation).** Keep the focus on forward mode. Develop a novel way to differentiate particle trajectories macroscopically. Set FD-type Jacobians aside "right now" and study option C (noise-regularized likelihood-ratio weights) with subagents that derive, evaluate, and review it.

| Class | Items |
|---|---|
| Recorded plan decisions | D32, D33, D34 (plan file). No new plan decision since D34. |
| User intent, not yet a plan decision | Focus on option C; forward mode only; FD set aside for now. |
| Proposed, not accepted | D35 (close Track R, use FD-type Jacobians). |
| Running at this checkpoint | Option C study (toy and 1D-wake prototypes, critics, synthesis); `p_nr` = 16 convergence runs. |
| Paused (D34) | FD recovery, P8, P9, rest of P7 step 5. |

**Next.** Review the option C note and the `p_nr` = 16 result. Then decide the target derivative (noise-regularized or pointwise) and the reference resolution for Twin-S.
