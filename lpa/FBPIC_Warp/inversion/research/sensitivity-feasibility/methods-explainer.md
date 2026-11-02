# Methods explainer: differentiating a chaotic particle simulation

Checkpoint 2026-11-01. Purpose: explain the techniques, algorithms, and key results of this investigation for a reader who knows CFD and numerical methods but not this project. Each section links its evidence. Worked examples marked "illustrative" were constructed for explanation; they are not project measurements.

## 1. What is differentiated

The simulation maps laser parameters \(\theta\) (first the normalized laser energy \(\hat E_L\)) to the state of an ionization-injected electron bunch after \(N_t = 2000\) steps. The loss uses 11 whitened residuals \(r_w\) built from weighted beam moments: charge, centroids, spreads, and relative energy spread (plan Section 2.4). An optimizer (trust-region dogbox) needs the Jacobian \(\partial r_w/\partial\theta\).

Forward-mode AD carries, for each macroparticle \(i\), the tangent
\[
\xi_i = \left(\frac{\partial x_i}{\partial\theta},\; \frac{\partial u_i}{\partial\theta}\right),
\]
plus the tangents of all fields. The moment tangents are weighted sums of \(\xi_i\). The hand-written kernels are verified against FD to about \(10^{-5}\) at \(N_t \le 500\) ([finding 1](findings/2026-10-31-ad-tangent-at-full-horizon.md)).

## 2. Why exact tangents explode (illustrative example)

Take the doubling map \(x_{n+1} = 2x_n \bmod 1\) with \(x_0(\theta) = x_0 + \theta\). One particle has
\[
\frac{\partial x_n}{\partial\theta} = 2^n .
\]
Put \(N\) particles uniformly on \([0,1)\). The map keeps a uniform cloud uniform, so a beam average such as \(\bar c = N^{-1}\sum_i\cos(2\pi x_i)\) does not depend on \(\theta\). The exact derivative of the finite-\(N\) average is
\[
\frac{\partial\bar c}{\partial\theta} = \frac{1}{N}\sum_i -2\pi\sin(2\pi x_{n,i})\,2^n ,
\]
a sum of terms of size \(2^n\) with nearly random signs. It does not cancel, so its size is about \(2^n/\sqrt N\). That is the pattern of the Twin-S AD column at step 2000: 10 to 1000 times too large, with wrong signs ([finding 1](findings/2026-10-31-ad-tangent-at-full-horizon.md)).

Important detail: every particle has the same tangent \(2^n\). The explosion is coherent, not a random difference between neighbours. The crowd's sensitivity is zero only because the stretch moves particles along a uniform distribution.

In Twin-S, the per-species medians of \(\lvert\partial u/\partial\hat E_L\rvert\) grow from about step 1450, in the density downramp, and 4x more particles reduce the rates only by 10% to 30% ([finding 4](findings/2026-11-01-particle-count-roughness-and-tangent-growth.md)).

## 3. Two ways to represent a perturbation

| Representation | What each particle carries | Cold plasma | Chaos |
|---|---|---|---|
| Move particles (Lagrangian tangent, our AD) | \(\xi_i\) | Works: a shifted spike is shifted particles | Explodes after trapping |
| Reweight particles (Eulerian, \(\delta f\) weights) | a weight change \(\delta w_i\) on an unperturbed path | Fails without smoothing | No path separation, so no explosion |

The two are linked by \(\delta f = -\nabla_z\cdot(f\,\xi)\) with \(z = (x, u)\). A pure displacement along a smooth distribution gives \(\delta f = 0\), which is why the weight form avoids the explosion.

Why the weight form fails on a cold plasma: at each position all electrons share one velocity \(v\), so a charge-neutral weight change gives
\[
\delta J = q\sum_i \delta w_i\,v = q\,v\sum_i\delta w_i = 0 ,
\]
while the true response to a push is a velocity change, which is a current. The slope \(\nabla_u\ln f\) the method needs does not exist for a spike in momentum space. Smoothing the spike to a width \(\sigma_u\) makes the answer depend on \(\sigma_u\) ([finding 3](findings/2026-11-01-track-r-feasibility-study.md); R0 note Appendix A.2 and A.3).

## 4. What a finite-difference secant measures

A central secant \(D(h) = [r(\theta+h) - r(\theta-h)]/(2h)\) averages the response over a window of width \(2h\). It is a derivative at the scale \(h\), not a pointwise derivative. On Twin-S at the current resolution, \(D(10^{-2})\) is reproducible across injection seeds to about 4%, while secants at \(h \le 3\cdot10^{-3}\) differ from it by 23% to 66% ([finding 4](findings/2026-11-01-particle-count-roughness-and-tangent-growth.md)).

## 5. Particle-pdf method (Chung, Bond, Cyr, Freund, JCP 400, 2020)

Algorithm: sensitivity particles move on the unperturbed characteristics; their weights carry \(\delta f\) and change at the rate \(-q(\delta E + v\times\delta B)\cdot\nabla_u f\), with \(\nabla_u f\) from a phase-space mesh; the weights deposit \(\delta\rho\) and \(\delta J\), which drive the field tangents. Shown in the paper only for warm 1D-1V electrostatic plasmas (Debye shielding, sheath). Phase R0 found it a no-go for Twin-S as specified: the cold background and thin bunch make \(\nabla_u f\) singular, and credible variants cost about one run per parameter ([finding 3](findings/2026-11-01-track-r-feasibility-study.md); [R0 design note](../../notes/r0-design-note-2026-10-31.md)).

## 6. Gauge-projected tangent (studied, not adopted)

Idea: particles are interchangeable, so \(\xi\) and \(\xi + \eta\) give the same \(\delta f\) whenever \(\nabla_z\cdot(f\,\eta) = 0\) (a relabeling). Project \(\xi\) onto gradient fields in the particle-weighted norm, by least squares over particles:
\[
\min_c\ \sum_i w_i\,\Big\lVert \xi_i - \sum_k c_k\,\nabla\psi_k(z_i)\Big\rVert^2 .
\]
In the continuum this keeps \(\delta f\) and every beam moment and removes only the relabeling part; it is the Otto-Wasserstein tangent. It handles the doubling map correctly in the continuum. Why it was not adopted: on a cold sheet the relabeling part is small and the explosion is a real change of density along the sheet; on finite particles, every "gauge removal" is a basis truncation, that is, coarse-graining ([finding 6](findings/2026-11-01-macroscopic-gradient-methods.md), item 2).

## 7. Option C: noise-regularized likelihood-ratio weights (studied, no-go in 1D)

Model change: add a small controlled momentum noise \(\sigma\,dW\) to the particle push. The expected observable \(\mathbb E[J]\) of the noisy model is smooth in \(\theta\). The noise level is an explicit smoothing scale, with a physical reading (temperature, collisions, shot-to-shot jitter).

Algorithm (mean-field form):
1. Push particles as usual, plus the noise kick; record each kick \(dW_i\).
2. Compute the force tangent \(\delta F_i = q(\delta E + v_i\times\delta B)\) at each particle from the existing field tangents.
3. Update a weight per particle: \(d\omega_i = \sigma^{-1}\,\delta F_i\cdot dW_i\).
4. Deposit \(\delta\rho\) and \(\delta J\) from \(q\,w_i\,\omega_i\); advance the field tangents (linear, already implemented), including the laser initial-field tangent and the EV birth-weight source.
5. Moment tangents are \(\mathbb E[\omega\,g]\) for each moment function \(g\).

This is the paper's method with \(\nabla_u\ln f\) replaced by the particle's own noise increment (Girsanov). It needs no mesh and works on a cold plasma for \(\sigma > 0\). The mixed identity \(\partial_\theta\mathbb E[g] = \mathbb E[\omega\,g + \xi\cdot\nabla g]\) allows any part of the force tangent to be carried by paths instead of weights, for example in the laser region.

Trade-off: bias grows with \(\sigma\) (the noise changes the physics); variance grows roughly as \(\int\lvert\delta F\rvert^2/\sigma^2\,dt\), linearly in time unless collective modes amplify it. Result of the 1D study: correct for the noisy model, but no noise level is both physics-preserving and low-variance; the growth moves from trajectories to the field tangent, and an information floor \(\mathrm{SE}/\lvert\delta m\rvert \gtrsim s_g/(u_n\sqrt{N_{\mathrm{eff}}})\) limits score-type estimators on cold, thin bunches ([finding 8](findings/2026-11-01-option-c-likelihood-ratio-no-go.md)).

## 8. Resolution caveat

The Twin-S Jacobian depends strongly on the radial macroparticle count `p_nr`: its direction reverses from 2 to 4, its norm falls 2.7 times from 4 to 8, and it changes by about 10% from 8 to 16 ([finding 5](findings/2026-11-01-radial-particle-convergence.md)). At `p_nr` = 8 the FD Jacobian is a well-defined target, and exact AD misses it by about \(10^4\) ([finding 7](findings/2026-11-01-converged-model-ad-versus-fd.md)).

## Literature pointers

- Chung, Bond, Cyr, Freund, "Regular sensitivity computation avoiding chaotic effects in particle-in-cell plasma methods", JCP 400, 108969 (2020). Local PDF: `/mnt/work/snidhan/projects/applied-research/solvers/fbpic/literature/Chung2020_regular_sensitivity_PIC_OSTI1570259.pdf`. Method p. 11 to 19; averaging and redistribution remarks p. 4 and p. 49; Fig. 11(d) p. 34.
- Gauge-projection prior art: [gpt-evidence/sec_lit.md](../../notes/gpt-evidence-2026-11-01/sec_lit.md).
- Likelihood-ratio and Malliavin prior art for option C: [option-c-evidence/sec_lit.md](../../notes/option-c-evidence-2026-11-01/sec_lit.md).
