# Optimization Methods for Laser-Parameter Inversion in Ionization-Injection LWFA

Date: 2026-10-30

**Summary.** Replace L-BFGS-B with a bounded trust-region Gauss-Newton (GN) or Levenberg-Marquardt (LM) solver on the 13-residual vector, and start it from a small global design over [0,1]^2. First, change the residual definitions to central second moments, because the raw second moments count the means of z and uz 40 to 170 times. For 3 to 5 parameters and noisy experimental data, add exact forward-mode Jacobians, a global Bayesian optimization (BO) stage, and simulation-based posterior estimation.

Abbreviations: FD (finite difference), JVP (Jacobian-vector product, forward-mode tangent), GP (Gaussian process), DFO (derivative-free optimization), PIC (particle-in-cell), LWFA (laser-wakefield acceleration), NPE (neural posterior estimation), SBC (simulation-based calibration).

Evidence labels used in this report:
- **Standard practice**: wide use in production or in the accelerator community.
- **Established research**: peer-reviewed methods with published benchmarks, but no LWFA inversion precedent.
- **Emerging**: few or no precedents for PIC or LWFA.

All run counts that do not have a citation are the author's estimates.

---

## 1. Ranked methods for the current 2-parameter twin experiment

The table uses these conditions: 90 s per run, 5 GPUs, deterministic expected-value (EV) ionization, 13 residuals, and the box [0,1]^2.

| Rank | Method (maturity) | Forward runs per iteration | Why it fits | Main risk |
|---|---|---|---|---|
| 1 | Bounded trust-region GN/LM on the residual vector: `scipy.optimize.least_squares`, `method='dogbox'` or `'trf'`, with a batched central-FD Jacobian [1, 2, 3, 5, 6, 7] (standard practice) | 4 FD runs in 1 batch plus the cached center, and 1 serial run for each trial step. About 3 min wall time per iteration. After the JVP exists: 1 primal and 2 tangents. | It uses G = J^T W J (cond about 2) from the first iteration. The trust radius limits the step. It keeps no secant memory and uses the least-squares structure. | It is a local method. FD columns are wrong when the stencil crosses a jump, with an error of about delta/(2h) = 25*delta at h = 0.02. |
| 2 | Global initial design plus multistart: Sobol or 5x5 grid now, APOSMM later [28, 29, 30] (standard practice) | 10 to 25 runs, done once (2 to 5 batches). After that, the local-method cost for each start. | It tests directly whether the z_f edge is a separate basin. It seeds the local solvers and checks identifiability without an optimizer. | A grid spacing of 0.25 can miss a narrow basin. Grid cost grows quickly with dimension. |
| 3 | DFO-LS, a model-based derivative-free GN method with restarts [15, 16, 17] (established research) | 3 runs to build the first model (1 batch), or fewer if seeded from an `EvaluationDatabase`. After that, about 1 serial run per iteration. | It has the lowest total run count of the least-squares DFO options. An `rhoend` above the jump scale averages over the bumps. Restarts can move the iterate off an edge minimum. | Evaluation is serial, so 4 of 5 GPUs are idle unless several instances run. The default `rhoend` of 1e-8 wastes runs on bumps. |
| 4 | Implicit filtering, GN variant with a decreasing stencil scale h [22, 23, 24, 25] (established research) | 5 runs in 1 batch, plus 1 to 3 serial line-search runs. About 40 to 100 runs in total (estimate). | A large initial h filters cell-crossing bumps. It tests the pinned bound at a large scale, and it fits the 5-GPU batch. | It needs more runs than DFO-LS on smooth problems. The GN mode needs a custom loop because scikit-quant ImFil accepts only a scalar objective. |
| 5 | Batch GP-BO on log(loss) or on the residual vector (composite BO), with BoTorch or Xopt [31, 32, 33, 34, 35, 36] (scalar: standard practice; composite: established research) | q = 5 per batch (about 90 s). About 20 to 60 runs in total at 2-D (estimate). | Jalas et al. used BO with FBPIC on ionization injection, with laser energy and focus as inputs. An outward gradient at a bound cannot pin BO. | Final convergence is slow. Jumps distort a stationary GP unless the GP learns a noise term. |
| 6 | TuRBO trust-region BO as a bridge to GN [41, 42, 43] (established research) | q = 5 per batch. | Its bounded trust region prevents corner-to-corner jumps, and its restarts handle several basins. | TuRBO was designed for high dimension. Its tolerances need retuning for small budgets, and it has no published LWFA use. |
| 7 | POUNDERS, with quadratic residual models [18, 19, 20, 21] (established research) | 3 to 6 runs to start, then about 1 serial run per iteration. | Quadratic residual models capture curvature in z_f that linear models miss. | Evaluation is serial. Interpolation can fit the bumps. Bound support in the Python version is not confirmed. TAO adds a PETSc dependency. |
| 8 | Py-BOBYQA with `seek_global_minimum=True` on the scalar loss [24, 26, 27] (established research) | 5 runs to start (1 batch), then about 1 serial run per iteration. | It is a fallback when the loss stops being a sum of squares. | It ignores the residual structure. It performs poorly on surfaces with many shallow minima [24]. |
| 9 | Current L-BFGS-B, repaired with noise-aware FD steps and loss scaling [8 to 14] (established research) | 5 runs in the FD batch, plus 1 or more serial line-search runs. | It needs the smallest code change. | It discards the residual structure. For least squares, Shi et al. use FD Levenberg-Marquardt instead [12]. |
| Not ranked | Adam on a differentiable simulator [63, 64] (established research, but not for this problem) | 1 gradient run per iteration. The 6-D phase-space reconstruction used 3000 iterations [64]. | It suits fits of many network weights, not 2 scalars. | It needs hours to days at 90 to 450 s per run. It gives no curvature for uncertainty. |

---

## 2. Diagnosis of the observed L-BFGS-B failure

### 2.1 Oversized first step

L-BFGS-B starts from a scaled-identity Hessian, B0 = theta*I. The author recalls theta = 1 at the first iteration [13], but the fetched text does not confirm this. The first step goes to the generalized Cauchy point, which is the first local minimizer along the projected-gradient path [13]. With a gradient norm of about 38 in a unit box, that path ends at a corner. This matches the observation.

Nocedal and Wright recommend the scaling H0 = (s^T y / y^T y) I, but only after a secant pair exists [14, Section 6.1]. SciPy L-BFGS-B has no option to set H0. If you keep L-BFGS-B, scale the loss so that the first Cauchy step is about the box size.

The literature recommends a structural fix instead. GN builds its model Hessian from J^T W J at the first iteration, and the trust radius limits the step [1, 2, 6]. Accelerator optics fitting (LOCO) went through the same change [5]. Plain GN in LOCO had difficulty converging from far initial guesses and found degenerate solutions. Huang et al. added Levenberg-Marquardt damping and parameter constraints to fix this.

### 2.2 Bad secant curvature from a flat region

A secant pair (s, y) from a flat or bumpy region gives a small or wrong y. The quasi-Newton update then predicts too little curvature, and the next step goes to the opposite corner. FD noise makes this worse. A residual jump delta inside the stencil gives a gradient error of about delta/(2h), which is 25*delta at h = 0.02.

The literature gives three fixes for this failure:
1. Measure the noise level eps_f with ECNoise [8]. Then set h from eps_f and a curvature estimate [9, 10]. The adaptive-interval method of Shi et al. needs eps_f as an input [10].
2. Use a noise-tolerant quasi-Newton method with a recovery mechanism [11]. Skip the update when s^T y is small relative to ||s||*||y||. SciPy does not expose this option, so it needs a custom driver.
3. Use a least-squares method. Shi et al. compare FD-based methods and use FD Levenberg-Marquardt for least squares [12]. GN keeps no secant memory, so this failure cannot occur.

### 2.3 Focal parameter pinned at a bound

The current data do not distinguish three possible causes:
- (a) The edge region (loss about 0.9) is a separate local basin or descent region.
- (b) A bump or an FD artifact gives a local outward slope, but the large-scale trend points inward.
- (c) The raw-moment weighting distorts the landscape and makes or deepens the edge region.

Bounded local methods cannot fix cause (a). The reflective strategy in `'trf'` keeps iterates feasible, but it does not prevent convergence to a boundary local minimum [1]. The literature fix is a global stage: a space-filling design with multistart [28, 30], or BO [31, 32]. Implicit filtering evaluates the gradient at a large stencil scale first, where the large-scale trend can override a local outward slope [22, 23]. DFO-LS and Py-BOBYQA restarts with a larger radius give a weaker version of the same fix [15, 17, 26, 27].

Use this diagnostic before the next optimizer run:
1. Compute the projected gradient in z_f at the bound with FD at h = 0.02 and h = 0.1.
2. If the two signs differ, the bound pin is a small-scale effect (cause b).
3. Evaluate the 25-point design. It shows whether a separate edge basin exists (cause a).
4. Repeat steps 1 to 3 after the residual change in Section 4 to test cause (c).

---

## 3. Recommendations

### 3.1 Next run: 2-parameter twin experiment

Do the steps in this sequence:

1. **Redefine the residuals** as Section 4 describes. Recompute G = J^T W J and cond(G) at the truth, because the value of about 2 used the old residual set.
2. **Measure the noise.** Run ECNoise with 7 to 9 runs along each coordinate, near the truth and near the z_f bound [8]. Set h for each coordinate from eps_f [9, 10]. If you skip this step, use h = 0.02 in E_L and h = 0.05 in z_f. These values are the author's choice: 0.05 is the upper limit of the z_f bump width that the current slices show.
3. **Run the global design.** Evaluate a 25-point Sobol design or a 5x5 grid over [0,1]^2 in 5 batches (about 7.5 min). As an option, run a 21x21 grid once (441 runs, about 2.2 h on 5 GPUs) to get a ground-truth landscape for method comparison.
4. **Run the local solve** from the best design point, with these settings:

```python
res = scipy.optimize.least_squares(
    fun=residuals_w,          # whitened 13-vector; caches each run by x
    x0=x_best_design,
    jac=fd_jacobian_batch,    # 4 perturbed runs as 1 batch; reuses cached center
    bounds=([0.0, 0.0], [1.0, 1.0]),
    method="dogbox",          # n = 2, full-rank J; "trf" is the alternative
    x_scale="jac",            # set explicitly; the default changed in SciPy 1.16.0
    loss="linear",            # "soft_l1" or "huber" for experimental data
    xtol=2e-3, ftol=1e-4, gtol=1e-8,
    max_nfev=20,
)
```

   - `least_squares` calls `fun` and `jac` separately. The Jacobian callable must therefore reuse the cached center run, or each iteration costs 1 extra run [1].
   - Within h of a bound, use the one-sided second-order stencil (x, x+h, x+2h) inside the box. A central stencil that crosses a bound is a silent error.
   - Do not use the SciPy default step for `diff_step`. It is about 6e-6, which is too small for jump noise. A user-supplied `diff_step` is relative to |x0| and collapses near x = 0 [1].
   - SciPy 1.16.0 and later accept a `workers` map for parallel `'3-point'` differences. A custom callable is still cleaner, because it controls the absolute step and the bounds.
5. **Run a multistart.** Repeat step 4 from the 2 or 3 best design points that are far from each other. GN uses all GPUs for each batch, so run the starts in sequence. As an alternative, run 3 to 5 DFO-LS instances at the same time, with 1 GPU each [15, 17].
6. **Apply the stopping rule.** Stop when any one condition is true:
   - The accepted step norm is below 2e-3 in normalized units.
   - The relative cost reduction is below 1e-4, or below the ECNoise level of the cost.
   - The loss is at the floor that the truth gives. That floor is about 0.0015 for the current loss; recompute it after the residual change.
   - The budget of 20 function evaluations (about 15 iterations) is used.

   The tolerance values are the author's choices. They are not values from the literature.
7. **Report the results.** Report the final loss, the parameter error against the truth, and G^-1 at the solution. In the noiseless twin, Sigma is a scaling choice. G^-1 is therefore a sensitivity metric, not a posterior covariance.

Expected cost: 25 design runs, plus about 5 to 10 iterations for each start at about 3 min each. This is less than 1 h of wall time in total (estimate).

### 3.2 Later: exact forward-mode gradients (emerging)

Forward mode is the correct AD mode for 2 to 5 parameters and 13 residuals. One augmented run gives the full residual Jacobian. Two PIC precedents exist:
- JAX-in-Cell (1D3V electromagnetic) drives a damped Newton update on 1 parameter with a single forward-mode JVP. It converges in 6 iterations [51].
- Qiang et al. use dual-number forward mode in an MPI-parallel PIC code. They verify the emittance sensitivities to 7 parameters against central FD, and they run BFGS matching [52].

Neither precedent covers LWFA or ionization injection. The JVP cost of 1.5 to 3 primal runs per direction is the author's estimate, so measure it.

Gate the JVP before you use it:
1. Compare the JVP Jacobian with central FD at h = 0.005, 0.01, 0.02 and 0.05. Use points away from known jumps and points near them.
2. Also compare it with a randomized-smoothing Jacobian (Section 4.3) [59, 60, 61].
3. Accept the JVP only where it agrees with FD within the ECNoise level.
4. Where they disagree, blend J = alpha*J_JVP + (1 - alpha)*J_smoothed, with alpha set from the measured disagreement [55].
5. Measure the tangent growth over the full LWFA propagation length. Per-particle PIC sensitivities can grow chaotically through the Lyapunov instability [53]. Moment averaging probably damps this growth, but this is not demonstrated.

Frozen-decision tangents give the slope of one smooth branch. The many particle-level jumps add up to a smooth trend, for example in Q(E_L), and frozen tangents miss that trend [54, 55]. This bias does not decrease with more particles.

After the gate, the `jac` callable becomes 1 primal run with 2 tangents, and the GN driver stays the same. The JVP can also supply derivative observations to a gradient-enhanced GP [38, 39, 40]. Give the derivative outputs a larger noise variance than the values.

### 3.3 Later: 3 to 5 parameters

- **Global stage.** Use a Sobol design of about 10*d runs, then APOSMM in libEnsemble with a DFO-LS or POUNDERS local solver [28, 30]. As an alternative, use batch BO or TuRBO [33, 41]. Full grids are not practical above about 3 parameters.
- **Use the curvature.** Use G from the JVP or FD to set GP length scales and correlations, or use the GN model as a prior mean [44, 45, 46]. Hanuka's kernel uses the Hessian of log(f - f(inf)), not G. Apply the log transform first.
- **Multi-fidelity.** Use coarser FBPIC-Warp runs as the low fidelity, as Irshad et al. did with FBPIC [48, 49]. Do not use Wake-T: the published FBPIC/Wake-T multitask result used an externally injected beam [47]. First check the correlation between fidelities on 10 to 20 paired runs. Ionization-injection thresholds can move with resolution.
- **Degeneracy.** Correlated parameters cause large degenerate steps. LOCO needed LM damping and penalty constraints for this problem [5]. Expect z_f to become degenerate with new parameters such as density or spot size [66].

### 3.4 Later: noisy experimental data and multi-shot jitter

- Use `loss='soft_l1'` or `'huber'` with `f_scale` from 1 to 3 in whitened units [1]. GN assumes small residuals at the optimum, so keep LM damping when the model has errors [7].
- Build Sigma from shot-to-shot scatter and add a model-error term (Section 4.2).
- Use several shots for each setting. Jalas et al. used 6 shots for each setting and gave each shot to the surrogate separately [32].
- **Posterior, in order of increasing cost:**
  1. Laplace approximation from G at the GN solution: 0 extra runs.
  2. MCMC with emcee on an emulator of the residuals, with jitter absorbed into per-shot nuisance parameters, as Los et al. did [66]. Train the emulator on the optimization archive (about 50 to 200 runs, estimate).
  3. Amortized NPE with sbi [68, 70]. This needs about 1e3 to 1e4 runs (25 to 250 GPU-h, estimate). Check the result with SBC and coverage tests. Bethke et al. recovered z_focus and a0 from PIConGPU spectra and found multimodal posteriors [71]. Their case used downramp injection, not ionization injection.
- Direct MCMC on PIC is not feasible at 90 s per run.
- If Monte Carlo ionization returns, stochastic automatic differentiation gives unbiased derivatives that include the jump term [56, 57]. This is emerging work: the reference implementation is a Julia prototype, and no GPU PIC port exists. EV ionization gives exact derivatives of the EV mean-field model, not of the expected Monte Carlo output, because beam loading is nonlinear in the ionized charge.

---

## 4. Loss-design recommendations

### 4.1 Central moments instead of raw moments

The raw second moment satisfies <z^2> = var(z) + mean(z)^2. The current loss therefore counts the z and uz means again inside the second-moment residuals, by factors of 40 to 170. Use these residuals instead:
1. Keep log(Q + Qf), with Qf near the charge noise floor.
2. Keep the first moments of x, ux, y, uy, z - z_window and uz.
3. Replace the raw second moments with log(sigma) for each coordinate.
4. Use the relative energy spread log(sigma_uz / mean(uz)), because energy spread scales multiplicatively.
5. Add only the correlation coefficients you need, for example corr(z, uz).

Add a floor to each sigma, because log(sigma) is undefined when the beam vanishes near the box corners.

### 4.2 Residual weighting and whitening

The efficient weight in the generalized method of moments is the inverse of the moment covariance [72]. Whiten with the Cholesky factor: Sigma = L L^T and r_w = L^-1 r.

- **Twin experiment.** Set the residual scales from a macroparticle bootstrap of the target run. This needs 0 extra runs. Treat the result as a scale proxy only. The EV twin is deterministic, and macroparticles are not independent samples.
- **Experimental data.** Build Sigma from shot-to-shot scatter and add a model-error term. A Sigma from macroparticle sampling alone is too small and gives too much weight to moments that the model cannot match.
- **Small samples.** A full covariance from few samples is noisy. Shrink it toward the diagonal when you have fewer than about 3x13 samples.

After whitening, cond(J^T W J) is an identifiability measure. This is standard Fisher-information reasoning. The geometry of nonlinear least squares explains the boundary and canyon effects that GN and LM meet [6].

Pakes and Pollard prove asymptotic normality for criterion functions with no smoothness assumptions [73]. Their result is statistical and asymptotic. It does not make the optimization landscape smooth.

### 4.3 Non-smoothness

1. **Classify the discontinuities.** A linear (CIC) shape gives C0 kinks at cell crossings. A cubic B-spline shape gives C2 deposition. Frozen-index tangents are exact almost everywhere for both shapes. True jumps come from these sources:
   - macroparticle creation at a nonzero weight threshold;
   - removal of particles at the moving-window edge or at boundaries;
   - hard beam-selection cuts.
2. **Remove jumps in the model design where possible:**
   - Create ionized macroparticles with zero weight at the moment of creation.
   - Taper weights to zero before window-edge removal.
   - Replace hard cuts with sigmoid weights of a set width.

   These changes modify the forward model, so check the physics against the original code [54, 55].
3. **Smooth the jumps that remain.** A central FD with step h is exactly the derivative averaged over a box of half-width h. Keep h above the jump width until the final stage.
   - Use a coarse-to-fine continuation: large h with first moments only, then smaller h with second moments, at production resolution [62].
   - Mobahi and Fisher give only a local linearization argument, not a global guarantee [62].
   - Always finish at production resolution, because ionization-injection thresholds depend on resolution.
4. **Randomized smoothing as an alternative.** Use one batch of 5 per iteration: the center point and 2 antithetic Gaussian pairs, with sigma from 0.03 to 0.05 in z_f. Fit the residual Jacobian by least squares [59, 60, 61].
   - The convergence rates of Duchi et al. assume convexity and do not apply to this problem [59].
   - In 2-D, the gain over FD is a controlled smoothing scale, not a lower cost.

### 4.4 Distribution-level losses (later fine stage)

You can replace the moments with a weighted MMD [74], a Sinkhorn divergence [75] or a sliced Wasserstein distance [76, 77] between macroparticle clouds. Use the EV charges as weights, and standardize each coordinate first.
- These losses need 0 extra runs, because they post-process the same particle output.
- Keep log-charge as a separate residual, because normalized weights ignore the total charge.
- For experimental data, compare only the measured projections. On an energy spectrum, the 1-D W2 distance is the L2 distance between quantile functions.
- Anneal the kernel bandwidth from large to small. Gradients vanish when the two distributions do not overlap.
- Roussel et al. use an image-based loss with KDE-smoothed projections in a differentiable beam reconstruction [63].
- Use moments for the coarse stage and distribution losses only for the fine stage.

---

## 5. References

**Least squares and trust regions**
1. SciPy developers, `scipy.optimize.least_squares` documentation. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
2. M. A. Branch, T. F. Coleman, Y. Li, "A subspace, interior, and conjugate gradient method for large-scale bound-constrained minimization problems," SIAM J. Sci. Comput. 21(1):1-23, 1999. https://doi.org/10.1137/S1064827595289108
3. J. J. More, "The Levenberg-Marquardt algorithm: implementation and theory," in G. A. Watson (ed.), Numerical Analysis, Lecture Notes in Mathematics 630, Springer, 1978, pp. 105-116. https://doi.org/10.1007/BFb0067700
4. J. Safranek, "Experimental determination of storage ring optics using orbit response measurements," Nucl. Instrum. Methods Phys. Res. A 388(1-2):27-36, 1997. https://doi.org/10.1016/S0168-9002(97)00309-4
5. X. Huang, J. Safranek, G. Portmann, "LOCO with Constraints and Improved Fitting Technique," ICFA Beam Dyn. Newsl. 44:60-69, 2007. https://www.osti.gov/biblio/957446-loco-constraints-improved-fitting-technique
6. M. K. Transtrum, B. B. Machta, J. P. Sethna, "The geometry of nonlinear least squares with applications to sloppy models and optimization," Phys. Rev. E 83, 036701, 2011. https://arxiv.org/abs/1010.1449
7. M. K. Transtrum, J. P. Sethna, "Improvements to the Levenberg-Marquardt algorithm for nonlinear least-squares minimization," arXiv:1201.5885, 2012. https://arxiv.org/abs/1201.5885

**Noise-aware finite differences and quasi-Newton methods**

8. J. J. More, S. M. Wild, "Estimating computational noise," SIAM J. Sci. Comput. 33(3):1292-1314, 2011. https://doi.org/10.1137/100786125 (code: https://github.com/POptUS/CNoise)
9. J. J. More, S. M. Wild, "Estimating derivatives of noisy simulations," ACM Trans. Math. Softw. 38(3):19:1-19:21, 2012. https://doi.org/10.1145/2168773.2168777
10. H.-J. M. Shi, Y. Xie, M. Q. Xuan, J. Nocedal, "Adaptive finite-difference interval estimation for noisy derivative-free optimization," SIAM J. Sci. Comput. 44(4):A2302-A2321, 2022. https://arxiv.org/abs/2110.06380
11. A. S. Berahas, R. H. Byrd, J. Nocedal, "Derivative-free optimization of noisy functions via quasi-Newton methods," SIAM J. Optim. 29(2):965-993, 2019. https://arxiv.org/abs/1803.10173
12. H.-J. M. Shi, M. Q. Xuan, F. Oztoprak, J. Nocedal, "On the numerical performance of finite-difference-based methods for derivative-free optimization," Optim. Methods Softw. 38(2):289-311, 2023. https://arxiv.org/abs/2102.09762
13. R. H. Byrd, P. Lu, J. Nocedal, C. Zhu, "A limited memory algorithm for bound constrained optimization," SIAM J. Sci. Comput. 16(5):1190-1208, 1995. https://doi.org/10.1137/0916069
14. J. Nocedal, S. J. Wright, Numerical Optimization, 2nd ed., Springer, 2006 (Section 6.1).

**Derivative-free local methods**

15. C. Cartis, J. Fiala, B. Marteau, L. Roberts, "Improving the flexibility and robustness of model-based derivative-free optimization solvers," ACM Trans. Math. Softw. 45(3):32, 2019. https://arxiv.org/abs/1804.00154
16. C. Cartis, L. Roberts, "A derivative-free Gauss-Newton method," Math. Program. Comput. 11:631-674, 2019. https://arxiv.org/abs/1710.11005
17. DFO-LS user guide. https://numericalalgorithmsgroup.github.io/dfols/build/html/userguide.html
18. S. M. Wild, "Chapter 40: POUNDERS in TAO: solving derivative-free nonlinear least-squares problems with POUNDERS," in T. Terlaky, M. F. Anjos, S. Ahmed (eds.), Advances and Trends in Optimization with Engineering Applications, MOS-SIAM Series on Optimization 24, SIAM, 2017, pp. 529-539. https://doi.org/10.1137/1.9781611974683.ch40
19. PETSc/TAO users manual, TAO chapter. https://petsc.org/release/manual/tao/
20. S. M. Wild, R. G. Regis, C. A. Shoemaker, "ORBIT: Optimization by radial basis function interpolation in trust-regions," SIAM J. Sci. Comput. 30(6):3197-3219, 2008. https://doi.org/10.1137/070691814
21. J. Larson, M. Menickelly, S. M. Wild, IBCDFO software repository. https://github.com/POptUS/IBCDFO
22. C. T. Kelley, Implicit Filtering, SIAM Software, Environments, and Tools 23, 2011. https://doi.org/10.1137/1.9781611971903
23. P. Gilmore, C. T. Kelley, "An implicit filtering algorithm for optimization of functions with many local minima," SIAM J. Optim. 5(2):269-285, 1995. https://doi.org/10.1137/0805015
24. W. Lavrijsen, A. Tudor, J. Muller, C. Iancu, W. de Jong, "Classical optimizers for noisy intermediate-scale quantum devices," IEEE QCE 2020. https://arxiv.org/abs/2004.03004
25. scikit-quant ImFil documentation. https://scikit-quant.readthedocs.io/en/latest/imfil.html
26. C. Cartis, L. Roberts, O. Sheridan-Methven, "Escaping local minima with derivative-free methods: a numerical investigation," Optimization 71(8):2343-2373, 2022. https://arxiv.org/abs/1812.11343
27. Py-BOBYQA user guide. https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/userguide.html

**Global stage and multistart**

28. J. Larson, S. M. Wild, "Asynchronously parallel optimization solver for finding multiple minima," Math. Program. Comput. 10(3):303-332, 2018. https://doi.org/10.1007/s12532-017-0131-4
29. J. Larson, M. Menickelly, S. M. Wild, "Derivative-free optimization methods," Acta Numerica 28:287-404, 2019. https://arxiv.org/abs/1904.11585
30. libEnsemble documentation, APOSMM generator. https://libensemble.readthedocs.io/en/main/examples/aposmm.html

**Bayesian optimization**

31. R. J. Shalloo et al., "Automation and control of laser wakefield accelerators using Bayesian optimization," Nat. Commun. 11, 6355, 2020. https://doi.org/10.1038/s41467-020-20245-6
32. S. Jalas, M. Kirchen, P. Messner, P. Winkler, L. Huebner, J. Dirkwinkel, M. Schnepp, R. Lehe, A. R. Maier, "Bayesian Optimization of a Laser-Plasma Accelerator," Phys. Rev. Lett. 126, 104801, 2021. https://doi.org/10.1103/PhysRevLett.126.104801
33. R. Roussel et al., "Bayesian optimization algorithms for accelerator physics," Phys. Rev. Accel. Beams 27, 084801, 2024. https://doi.org/10.1103/PhysRevAccelBeams.27.084801
34. M. Balandat et al., "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization," NeurIPS 2020. https://arxiv.org/abs/1910.06403
35. R. Roussel, C. Mayes, A. Edelen, A. Bartnik, "Xopt: A simplified framework for optimization of accelerator problems using advanced algorithms," Proc. IPAC'23, THPL164. https://doi.org/10.18429/JACoW-IPAC2023-THPL164 (TuRBO controllers: https://xopt.xopt.org/)
36. R. Astudillo, P. I. Frazier, "Bayesian Optimization of Composite Functions," ICML 2019, PMLR 97:354-363. https://arxiv.org/abs/1906.01537
37. R. Astudillo, P. I. Frazier, "Bayesian Optimization of Function Networks," NeurIPS 2021. https://arxiv.org/abs/2112.15311
38. J. Wu, M. Poloczek, A. G. Wilson, P. I. Frazier, "Bayesian Optimization with Gradients," NIPS 2017. https://arxiv.org/abs/1703.04389
39. D. Eriksson, K. Dong, E. H. Lee, D. Bindel, A. G. Wilson, "Scaling Gaussian Process Regression with Derivatives," NeurIPS 2018. https://arxiv.org/abs/1810.12283
40. GPyTorch documentation, "GP Regression with Derivative Information (2D)." https://docs.gpytorch.ai/en/stable/examples/08_Advanced_Usage/Simple_GP_Regression_Derivative_Information_2d.html
41. D. Eriksson, M. Pearce, J. R. Gardner, R. Turner, M. Poloczek, "Scalable Global Optimization via Local Bayesian Optimization," NeurIPS 2019. https://arxiv.org/abs/1910.01739
42. BoTorch tutorial, "Trust Region Bayesian Optimization (TuRBO)." https://botorch.org/docs/tutorials/turbo_1
43. S. Muller, A. von Rohr, S. Trimpe, "Local policy search with Bayesian optimization," NeurIPS 2021. https://arxiv.org/abs/2106.11899
44. A. Hanuka et al., "Physics model-informed Gaussian process for online optimization of particle accelerators," Phys. Rev. Accel. Beams 24, 072802, 2021. https://arxiv.org/abs/2009.03566
45. J. Kaiser, C. Xu, A. Eichler, A. Santamaria Garcia, "Cheetah: Bridging the gap between machine learning and particle accelerator physics with high-speed, differentiable simulations," Phys. Rev. Accel. Beams 27, 054601, 2024. https://arxiv.org/html/2401.05815
46. T. Boltz et al., "Leveraging prior mean models for faster Bayesian optimization of particle accelerators," Sci. Rep., 2025. https://arxiv.org/abs/2403.03225 (journal: https://www.nature.com/articles/s41598-025-95297-z)

**Multi-fidelity optimization for laser-plasma accelerators**

47. A. Ferran Pousa et al., "Bayesian optimization of laser-plasma accelerators assisted by reduced physical models," Phys. Rev. Accel. Beams 26, 084601, 2023. https://doi.org/10.1103/PhysRevAccelBeams.26.084601
48. F. Irshad, S. Karsch, A. Doepp, "Multi-objective and multi-fidelity Bayesian optimization of laser-plasma acceleration," Phys. Rev. Research 5, 013063, 2023. https://doi.org/10.1103/PhysRevResearch.5.013063
49. F. Irshad et al., "Pareto Optimization and Tuning of a Laser Wakefield Accelerator," Phys. Rev. Lett. 133, 085001, 2024. https://doi.org/10.1103/PhysRevLett.133.085001
50. optimas documentation and repository. https://optimas.readthedocs.io/en/latest/ and https://github.com/optimas-org/optimas

**Differentiable PIC and derivatives through discrete events**

51. L. Ma, R. Jorge, H. Lu, A. Tran, C. Woolford, "JAX-in-Cell: A Differentiable Particle-in-Cell Code for Plasma Physics Applications," arXiv:2512.12160, 2025. https://arxiv.org/html/2512.12160v1
52. J. Qiang, Y. Hao, A. Qiang, J. Wan, "A multi-language auto-differentiation module and its application to a parallel particle-in-cell code on distributed computers," arXiv:2511.21839, 2025. https://arxiv.org/html/2511.21839
53. S. W. Chung, S. D. Bond, E. C. Cyr, J. B. Freund, "Regular sensitivity computation avoiding chaotic effects in particle-in-cell plasma methods," J. Comput. Phys. 400, 108969, 2020. https://www.osti.gov/pages/biblio/1570259-regular-sensitivity-computation-avoiding-chaotic-effects-particle-cell-plasma-methods
54. S. P. Bangaru, J. Michel, K. Mu, G. Bernstein, T.-M. Li, J. Ragan-Kelley, "Systematically Differentiating Parametric Discontinuities," ACM Trans. Graph. 40(4), Article 107, 2021. https://doi.org/10.1145/3450626.3459775
55. H. J. T. Suh, M. Simchowitz, K. Zhang, R. Tedrake, "Do Differentiable Simulators Give Better Policy Gradients?," ICML 2022, PMLR 162:20668-20696. https://arxiv.org/abs/2202.00817
56. G. Arya, M. Schauer, F. Schafer, C. Rackauckas, "Automatic Differentiation of Programs with Discrete Randomness," NeurIPS 2022, pp. 10435-10447. https://arxiv.org/abs/2210.08572
57. StochasticAD.jl documentation. https://gaurav-arya.github.io/StochasticAD.jl/stable/
58. S. Mohamed, M. Rosca, M. Figurnov, A. Mnih, "Monte Carlo Gradient Estimation in Machine Learning," JMLR 21(132):1-62, 2020. https://arxiv.org/abs/1906.10652

**Smoothing and continuation**

59. J. C. Duchi, P. L. Bartlett, M. J. Wainwright, "Randomized Smoothing for Stochastic Optimization," SIAM J. Optim. 22(2):674-701, 2012. https://arxiv.org/abs/1103.4296
60. Q. Le Lidec, F. Schramm, L. Montaut, C. Schmid, I. Laptev, J. Carpentier, "Leveraging Randomized Smoothing for Optimal Control of Nonsmooth Dynamical Systems," arXiv:2203.03986, 2022. https://arxiv.org/abs/2203.03986
61. F. Petersen, C. Borgelt, A. Mishra, S. Ermon, "Generalizing Stochastic Smoothing for Differentiation and Gradient Estimation," arXiv:2410.08125, 2024. https://arxiv.org/abs/2410.08125
62. H. Mobahi, J. W. Fisher III, "On the Link between Gaussian Homotopy Continuation and Convex Envelopes," EMMCVPR 2015, LNCS 8932, pp. 43-56. https://doi.org/10.1007/978-3-319-14612-6_4

**Differentiable beam inference**

63. R. Roussel et al., "Phase Space Reconstruction from Accelerator Beam Measurements Using Neural Networks and Differentiable Simulations," Phys. Rev. Lett. 130, 145001, 2023. https://doi.org/10.1103/PhysRevLett.130.145001
64. R. Roussel et al., "Efficient six-dimensional phase space reconstructions from experimental measurements using generative machine learning," Phys. Rev. Accel. Beams 27, 094601, 2024. https://arxiv.org/html/2404.10853v2
65. R. Watt, B. O'Shea, "A differentiable simulation package for performing inference of synchrotron-radiation-based diagnostics," J. Synchrotron Rad. 31(2):409-419, 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC10914167/

**Uncertainty quantification and simulation-based inference**

66. E. E. Los et al., "A Bayesian Framework to Investigate Radiation Reaction in Strong Fields," arXiv:2406.19420, 2024. https://arxiv.org/html/2406.19420v1
67. A. Doepp, C. Eberle, S. Howard, F. Irshad, J. Lin, M. Streeter, "Data-driven science and machine learning methods in laser-plasma physics," High Power Laser Sci. Eng. 11, e55, 2023. https://doi.org/10.1017/hpl.2023.47
68. K. Cranmer, J. Brehmer, G. Louppe, "The frontier of simulation-based inference," PNAS 117(48):30055-30062, 2020. https://doi.org/10.1073/pnas.1912789117
69. J. Brehmer, G. Louppe, J. Pavez, K. Cranmer, "Mining gold from implicit models to improve likelihood-free inference," PNAS 117(10):5242-5249, 2020. https://doi.org/10.1073/pnas.1915980117
70. J. Boelts, M. Deistler et al., "sbi reloaded: a toolkit for simulation-based inference workflows," J. Open Source Softw. 10(108), 7754, 2025. https://doi.org/10.21105/joss.07754
71. F. Bethke, R. Pausch, P. Stiller, A. Debus, M. Bussmann, N. Hoffmann, "Invertible Surrogate Models: Joint surrogate modelling and reconstruction of Laser-Wakefield Acceleration by invertible neural networks," arXiv:2106.00432, 2021. https://arxiv.org/abs/2106.00432

**Loss design**

72. L. P. Hansen, "Large Sample Properties of Generalized Method of Moments Estimators," Econometrica 50(4):1029-1054, 1982. https://doi.org/10.2307/1912775
73. A. Pakes, D. Pollard, "Simulation and the Asymptotics of Optimization Estimators," Econometrica 57(5):1027-1057, 1989. https://doi.org/10.2307/1913622
74. F.-X. Briol, A. Barp, A. B. Duncan, M. Girolami, "Statistical Inference for Generative Models with Maximum Mean Discrepancy," arXiv:1906.05944, 2019. https://arxiv.org/abs/1906.05944
75. J. Feydy, T. Sejourne, F.-X. Vialard, S. Amari, A. Trouve, G. Peyre, "Interpolating between Optimal Transport and MMD using Sinkhorn Divergences," AISTATS 2019, PMLR 89:2681-2690. https://arxiv.org/abs/1810.08278
76. K. Nadjahi, A. Durmus, U. Simsekli, R. Badeau, "Asymptotic Guarantees for Learning Generative Models with the Sliced-Wasserstein Distance," NeurIPS 2019. https://arxiv.org/abs/1906.04516
77. E. Bernton, P. E. Jacob, M. Gerber, C. P. Robert, "On parameter estimation with the Wasserstein distance," Information and Inference: A Journal of the IMA, 2019. https://arxiv.org/abs/1701.05146