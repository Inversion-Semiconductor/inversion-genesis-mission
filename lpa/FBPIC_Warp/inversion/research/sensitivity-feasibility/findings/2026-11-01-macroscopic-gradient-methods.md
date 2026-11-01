# Finding 6 (2026-11-01): methods for a macroscopic gradient from particle trajectories, status at this checkpoint

Status: in progress. The option C evaluation (toy and 1D-wake prototypes, two critics, synthesis) was still running at this checkpoint.
Sources: agent study sections, 2026-11-01. Gauge-projected tangent study (stopped by the agent after its derivation, literature, and alternatives sections): [gpt-evidence](../../../notes/gpt-evidence-2026-11-01/). Option C study (running): [option-c-evidence](../../../notes/option-c-evidence-2026-11-01/), a partial snapshot of `sec_theory.md`, `sec_lit.md`, and `work_theory/`.
Verification level: agent reports read; their CPU scripts were not rerun by this checkpoint. Paper statements were checked on the page images of Chung et al. (2020) in this session.

## Research intent (user, 2026-11-01, this conversation)

1. "you can keep focusing on forward mode for now please."
2. "We need to come up with a novel way of differentiating through particle trajectory macroscopically somehow, that is the inherent novelty of this work I feel like."
3. "let's not use A right now, let's focus on C with subagents deriving, evaluating, reviewing with a concrete goal of capturing the macroscopic gradients from particles trajectory." Here A is FD-type Jacobians at a matched scale and C is the noise-regularized likelihood-ratio method below.

The goal given to the option C study: a forward-mode gradient of beam moments from one run plus one weight set per parameter, which (1) does not grow exponentially under chaotic trajectories, (2) works on a cold plasma and a thin bunch, (3) matches a trusted reference within statistical error, and (4) costs well under one run per extra parameter.

## Conclusions so far

1. Local averaging of the exact tangent fails. On the doubling map every particle has the same tangent \(2^n\), so a local mean keeps the explosion. Chung et al. (2020) state that naive averaging of particle sensitivities does not commute with \(N\to\infty\) (p. 4), that periodic redistribution may not remove the problem (p. 49), and Fig. 11(d) (p. 34) shows mesh-interpolated quantities grow at the particle rate.
2. The gauge-projected tangent (GPT, the \(L^2(f)\) projection of \(\xi\) onto gradient fields, the Otto-Wasserstein tangent) is exact in the continuum but does not bound the growth on a cold sheet. On a sheet the relabeling gauge is small; the exploding along-sheet part changes the density along the sheet and is not gauge. On finite particles, every gauge removal is a basis truncation, that is, coarse-graining. Measured on a chaotic standard-map sheet: the projection keeps at least 99.5% of the tangent norm while the norm grows \(10^4\) times (`gpt-evidence/sec_derive.md` Section 0, test D2). Literature: the ingredients are known; no periodic projection against chaos in PIC or N-body work was found (`gpt-evidence/sec_lit.md`).
3. Label-space integration by parts (LIBP) is an exact cold-sheet identity. In 1D its raw per-particle summand is heavy-tailed; a hybrid that applies it only where the sheet is stretched cuts the per-particle standard deviation by 7 to 8 times and stays unbiased at \(T = 10, 15\) (`gpt-evidence/sec_alt.md` Section 3). It needs second-order tangents. It returns the pointwise continuum derivative, which is useful only where that derivative is smooth at the optimizer's scale.
4. Option C, noise-regularized likelihood-ratio weights, is exact for the noisy model: \(\mathbb E[\omega\,g] = \int g\,\delta f\) for the linearized Vlasov-Fokker-Planck equation. It needs no \(\nabla f\) and works on a cold plasma for \(\sigma > 0\) (`option-c-evidence/sec_theory.md`). A mixed identity holds for any adapted split of the force tangent between paths and weights:
   \[
   \frac{\partial}{\partial\theta}\,\mathbb E[g] = \mathbb E\big[\omega\,g + \xi\cdot\nabla g\big].
   \]
5. Option C has three measured or derived obstacles (`option-c-evidence/sec_theory.md`):
   - The laser quiver force makes pure weights infeasible in the laser region (variance about \(7.6\cdot10^4/u_n^2\), derived for \(a_0 = 4\)). The laser region must stay pathwise through the mixed identity.
   - The plain mean-field estimator has its own exponential noise loop, rate about \(0.25/(n_m\sigma^2)\) (1D measurement: \(\mathrm{Var}\,\omega = 1.7\cdot10^{37}\) at \(\sigma = 0.01\), \(N = 6400\)). Charge-neutral increments recovered the reference gradient at \(\sigma = 0.01\) in a cold laminar 1D test (0.477 ± 0.020 against 0.485). After trapping this is untested.
   - Credible hybrids cost about 0.8 to 2.5 primal runs per parameter (estimate). Goal criterion 4 is not met.
6. An earlier 1D test of the same idea ("NLR" in `gpt-evidence/sec_alt.md` Section 2.4, cold beam in prescribed chaotic waves) found it unbiased for the noisy system (12 of 12 cases within 2.3 combined standard errors) but biased against the noiseless system without a plateau: \(-20\%\) to \(-31\%\) at \(\sigma = 0.01\), \(-42\%\) to \(-64\%\) at \(\sigma = 0.03\), at \(T = 10\). Its variance exceeded FD of the same noisy model at small \(\sigma\).
7. Literature for option C: every mathematical ingredient is published (Warren and Allen, PRL 2012; Crisan and McMurray, PTRF 2018; Baños 2018; Ni, JNS 2025). The combination of per-particle Girsanov weights closed through self-consistent field tangents, used against chaos on a cold PIC plasma, was not found (`option-c-evidence/sec_lit.md`).

## Limits

- All method evidence is 1D or toy-model, CPU only. None was tested on Twin-S.
- The 1D wake pilot uses a cycle-averaged driver; it cannot show the laser-quiver obstacle.
- The option C toy and wake evaluations, the critics, and the synthesis were not finished at this checkpoint.
- The Twin-S reference gradient for any validation is not available until the radial convergence of [finding 5](2026-11-01-radial-particle-convergence.md) is settled.

## Open questions for the user

- Whether the target is the derivative of a noise-regularized model (option C), with the noise level as a stated physical smoothing scale, or the pointwise continuum derivative (LIBP direction).
- The D35 proposal (close Track R, use FD-type Jacobians) remains open. The user set FD aside "right now" without a recorded plan decision.
