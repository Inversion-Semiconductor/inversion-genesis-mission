# Finding 8 (2026-11-01): option C, noise-regularized likelihood-ratio weights, is correct but has no usable noise window in 1D

Status: current. The study recommended stopping option C before any `fbpic/autodiff` implementation. **User decision, 2026-11-01:** "Yes, stop C" (this conversation). Option C stops at the 1D stage; Gate B is not run.
Source: option C study (workflow `stochastic-lr-gradient-study`, run `wf_5aaf83a6-ec7`, finished 2026-11-01 about 11:50 PST): [note](../../../notes/option-c-evidence-2026-11-01/option-c-note.md), sections `sec_theory`, `sec_lit`, `sec_toy`, `sec_wake`, critiques `crit_math`, `crit_evidence`, all in [option-c-evidence](../../../notes/option-c-evidence-2026-11-01/).
Verification level: the note's Sections 1, 2, 3.6, 4.5, and 5 were read; the sections and critiques were read through their structured summaries. No agent script was rerun by this checkpoint. All evidence is 1D or toy, CPU only.

## Conclusion and tested conditions

1. Correctness holds. The per-particle Girsanov weights \(\Delta\omega_i = \delta\mathbf a_i\cdot\boldsymbol\zeta_i\sqrt{\Delta t}/\sigma\), closed through the field tangent, give an unbiased gradient of the noisy model, also on a cold plasma and with ionization births. In the 1D wake, where the weights are bounded (\(\sigma \ge 0.3\), \(N = 4\)), they matched common-random-number FD of the noisy model within 2 standard errors in 23 of 24 comparisons (`sec_wake`).
2. No noise window exists on the 1D cold wake. The trapped charge stays within 5% only for a noise level \(u_n \lesssim 0.07\). At \(u_n = 0.2\) the single-run error is 56 to 150 times the gradient (\(N = 4\)). Noise that bounds the weights (\(u_n \ge 0.67\)) changes the trapped charge by \(-27\%\) to \(-55\%\) and still leaves a single-run error of 2 to 39 times the gradient (note Section 1, item 2).
3. The growth moves from trajectories to the field level. Leave-one-out charge-neutral increments (added by the evidence critic) remove the sampling-feedback floor that the wake section had read as chaos; the remaining rate falls roughly as \(1/N\). At \(\sigma = 0.01\) it is still 0.31 and 0.15 \(\omega_p\) at \(N = 4\) and 16. In the cold two-stream toy, the FD gradient decays while the weights grow 560 times (`crit_math`).
4. The cost goal fails. One weight set costs 2.1 to 2.9 primal runs in NumPy, and the variance needs 16 to 600 times more runs than one FD seed for equal standard error at \(u_n \ge 0.67\), more than \(10^4\) times at \(u_n = 0.2\).
5. A general limit was derived: an information floor \(\mathrm{SE}/\lvert\delta m\rvert \gtrsim s_g/(u_n\sqrt{N_{\mathrm{eff}}})\) for score-type estimators on cold, thin bunches (note Section 3.4, Appendix A.3). Sequential Monte Carlo work suggests the only known cure is marginalization (Rao-Blackwellization), which needs a phase-space basis (note Section 5, item 4).

## Literature and novelty

Every ingredient is published: Warren and Allen (PRL 109, 250601, 2012); Baños (Ann. IHP 54, 2018); Crisan and McMurray (PTRF 171, 2018); Ni (J. Nonlinear Sci. 35, 90, 2025); Brunner, Valeo, Krommes (Phys. Plasmas 6, 4504, 1999); Poyiadjis, Doucet, Singh (Biometrika 98, 2011). No paper found builds the linearized Vlasov-Fokker-Planck-Maxwell solver from per-particle Girsanov weights closed through a self-consistent field tangent. The study calls the publishable content a characterization with a negative outcome.

## Limits

- The 1D wake uses a cycle-averaged driver, a bunch momentum spread of about 27 (large against Twin-S), and a different injection geometry. The note proposes a Twin-S "Gate B" (one tangent run at `p_nr` 8 to step 1520 with a logging callback, about 1 GPU-hour, no kernel) that would test the information floor natively. It was not run.
- In the 1D wake at \(t = 45\), all pointwise estimators (option C and the exact tangent on the same noise paths) deviate from FD by up to 3 standard errors. The note leaves this unresolved.

## Consequence

No forward-mode method studied so far meets all four goal criteria of the option C brief. The open questions of [finding 6](2026-11-01-macroscopic-gradient-methods.md) remain: which derivative is the target, and how to obtain it on the converged Twin-S model of [finding 7](2026-11-01-converged-model-ad-versus-fd.md).
