# R0 section: $\nabla_u f$ and the phase-space mesh for a cold plasma and a thin trapped bunch

Author: R0 derivation agent (topic key `gradf`), 2026-10-31.
Claim labels: **[paper]** Chung, Bond, Cyr, Freund, JCP 400, 108969 (2020), checked against the page images; **[derived]** worked out here; **[measured]** by a script in `scratchpad/r0/work_gradf/`; **[estimated]** an order-of-magnitude estimate with stated inputs; **[plan]** taken from `DIFFERENTIABLE_INVERSION_PLAN.md`.
Scratch directory: `inversion/notes/r0-evidence-2026-10-31/work_gradf/`.

## G.0 Result in brief

1. In a cold plasma, $f$ is a sum of sheets $n_k\,\delta(\mathbf u - \mathbf U_k(\mathbf x,t))$, and the trapped EV bunch is a codimension-2 manifold in the 6-D phase space. $\nabla_u f$ is a derivative of a delta function. A mesh or kernel of momentum width $\sigma_u$ returns a dipole layer of thickness $\sigma_u$ with amplitude $\propto n/\sigma_u^{c+1}$, where $c$ is the codimension in $\mathbf u$ (3 for the background, 2 for the bunch). [derived]
2. The dipole identity in the briefing is exact, including the magnetic term: $\delta\mathbf a\cdot\nabla_u f = \nabla_u\cdot(f\,\delta\mathbf a)$. In weak form, every smoothed particle-pdf estimate equals the Lagrangian tangent with $\nabla_u\psi_t$ replaced by a $\sigma_u$-smoothed derivative of the pulled-back observable $\psi_t = \phi\circ\Phi_{t\to T}$. [derived]
3. For a warm plasma the bias can be moved onto $f$ and is $O(\sigma_u^2/v_T^2)$, independent of how rough $\psi_t$ is. This is the regime of every test in the paper. For a cold plasma the bias cannot be moved onto $f$. It is $O(\sigma_u^2/\ell_\psi^2)$, where $\ell_\psi(t)$ is the momentum scale of $\psi_t$, and $\ell_\psi$ shrinks with phase mixing and chaos. Then $\sigma_u$ is a coarse-graining scale, like the FD step $h$, and not a small numerical parameter. [derived] In a 1D cold-beam test that crosses a separatrix, the bias fell much more slowly than $\sigma_u^2$ even before chaos: 10% accuracy needed $\sigma_u \approx 10^{-3}$ of the velocity scale. [measured]
4. The collocated scheme of the paper cannot work on a cold sheet: sensitivity particles on the sheet see only tangential derivatives of $\psi$. A 1D-1V experiment confirms this. It also shows that for a cold beam the smoothed estimate depends on $\sigma_u$ at O(1), exactly as the FD secant of the cold continuum depends on $h$. For a warm beam all estimators agree. [derived, measured]
5. A full 6-D phase-space mesh on Twin-S is impossible as a dense array (about $10^{12}$ cells). A sparse, band-limited mesh fits in memory (about 2 to 6 GB) but costs an estimated 27 to 85 run-equivalents per 2000-step run on the full domain, and 4 to 9 when restricted to the bunch-relevant electrons after step 1000. Mesh-free dipole pairs cost an estimated 4 to 6 run-equivalents for $K = 1$. Central FD costs 2. [estimated]

Verdict for this topic: **conditional**. Track R on Twin-S needs off-sheet sampling (dipole pairs or an adaptive band mesh). It also needs evidence, from Phase R2, that a $\sigma_u$-plateau exists, that is, a range of $\sigma_u$ in which the smoothed sensitivity does not change and agrees with the smooth FD secant. Without that plateau, Track R computes an FD secant in disguise, at higher cost than FD.

## G.1 What the paper computes

The sensitivity source is $\mathcal H = \tfrac{q}{m}\,\partial\lfloor\mathbf E\rfloor\cdot\nabla_{\mathbf v} f + \partial_\theta\mathcal V_1$ (eq. 25, PDF p12). The sensitivity distribution is carried by particles with weights $\hat W_s$ (eq. 21, p11), which move like PIC particles (eq. 27, p12) and change by $d\hat W_s/dt = -h_s$ (eq. 44, p18). [paper]

The steps that involve $\nabla_v f$ are as follows. [paper]

1. Mesh source (Section 3.2.2, eqs. 30 to 37, PDF p14 to p15). The derivative is put on the velocity shape function of each PIC particle, not on a histogram:
   $$(\partial\lfloor\mathbf E\rfloor\cdot\nabla_{\mathbf v} f)_{\mathbf i,\mathbf i_v} = \sum_{p=1}^{N} \frac{W_p}{\Delta\mathbf x\,\Delta\mathbf v}\, b_{l+1}(\boldsymbol\xi_{p,\mathbf i})\;\partial\lfloor\mathbf E\rfloor_{\mathbf i}\cdot\nabla_{\mathbf v} b_{l+2}(\boldsymbol\zeta_{p,\mathbf i_v}) \qquad (36)$$
   This is a deposit of one smoothed dipole per PIC particle, with momentum width $\Delta v$.
2. Sampler density (Section 3.2.3, eq. 41, p16): $\hat n_{\mathbf i,\mathbf i_v} = (\Delta\mathbf x\Delta\mathbf v)^{-1}\sum_s b_{l+1}(\hat{\boldsymbol\xi}_{s,\mathbf i})\, b_{l+1}(\hat{\boldsymbol\zeta}_{s,\mathbf i_v})$.
3. Per-particle source (eq. 42, p16): $h_s = \sum_{\mathbf i,\mathbf i_v} (\mathcal H_{\mathbf i,\mathbf i_v}/\hat n_{\mathbf i,\mathbf i_v})\, b_{l+1}(\hat{\boldsymbol\xi}_{s,\mathbf i})\, b_{l+1}(\hat{\boldsymbol\zeta}_{s,\mathbf i_v})$.
4. Support condition (eq. 43, p17): $\hat n_{\mathbf i,\mathbf i_v} > 0$ wherever $\mathcal H_{\mathbf i,\mathbf i_v}\ne 0$. The paper adds particles where it fails (particle addition, eqs. 45 to 48, p18) and redistributes onto a mesh with modified B-splines (eqs. 49, 50, p19; 55, 56, p20).
5. Adaptive scheme (Section 8.4, p44 to p46; Algorithm 1, p47): add sensitivity particles in every cell with $|\mathcal H_{\mathbf i,\mathbf i_v}|\ge\mathcal H_{min}$ and $\hat n_{\mathbf i,\mathbf i_v}\le\hat n_{min}$; redistribute when $M > M_{max}$. In the forced-sheath test the velocity domain is 8 times larger than in Section 8.3, and about twice as many sensitivity particles as PIC particles are needed (Fig. 18c, p48).

Velocity resolution in the paper's tests: Debye case $\Delta v = 18 v_0/64 = 0.28 v_0$ with $v_T = 1.5 v_0$, so $\Delta v = 0.19 v_T$ (p25); sheath $\Delta v_e = 0.078\,v_{T,e}$ (p44). Every test is a Maxwellian, 1D-1V, electrostatic plasma. [paper]

Cost (Section 7, p35 to p37): 2.96, 2.10, and 1.89 PIC-equivalents for the non-collocated, collocated, and particle-addition schemes, measured in 1D-1V with $N = 10^5$ (Fig. 12, p36). Table 1 (p37) gives the scaling: source evaluation $O[(C_2 l d + l^{2d} d^2)N]$, number density $O[(C_3 l + l^{2d}) d N]$, source strength $O[(l+1)^{2d} N]$, with $d$ the space dimension. [paper]

Note on notation: Appendix A (p51) labels $b_0$ as the top hat and $b_1$ as the triangle (Fig. A.20a), but writes $\partial_v b_3$ as a piecewise-linear function with support 3 (eq. A.2), which is the derivative of the quadratic spline. The index convention is therefore not consistent. The relevant fact for this section is that the velocity-gradient stencil in (36) is one cell wider than the $\hat n$ stencil in (41). [paper, checked on p15, p16, p51]

## G.2 The distribution $f$ in Twin-S

### G.2.1 Scales

| Quantity | Value | Source |
|---|---|---|
| $n_{e0}$ | $8\times10^{24}$ m$^{-3}$, $n/n_c = 4.6\times10^{-3}$ | [plan] `inversion/configs/twin_s.json`; $n_c$ [derived] |
| $\omega_p$, $c/\omega_p$, $\lambda_p$ | $1.60\times10^{14}$ s$^{-1}$, 1.88 µm, 11.8 µm | [derived] |
| $\Delta z$, $\Delta r$, $\Delta t$ | 0.05 µm ($\lambda_0/16$), 0.4 µm, 0.167 fs | [derived] from `twin_s.json` |
| $\omega_p\Delta t$; 2000 steps | 0.0266; $53\,\omega_p^{-1}$ | [derived] |
| Tangent growth after step 1200 | ×10 per 120 steps ($\approx 0.019$ per step $= 0.71\,\omega_p$) | [plan] Section 12.10 step 4 |
| $k_p w_0$; blowout radius $2\sqrt{a_0}/k_p$ | 2.66; 7.5 µm | [derived], $a_0 \approx 4$ |
| Laser quiver gradient per $z$ cell, $a_0 k_0\Delta z$ | 1.57 | [derived] |
| Selected particles at $N_t$ | 1.7 M of 6.66 M `elec_from_N` | [plan] Section 12.10 step 5 |
| Trapped peak, low peak in $u_z$ | 11.2, 1.0; $u_c = 6.25$, $\delta_u = 0.0385$ | [measured by P1] `inversion/results/p1_twin_s.json` |
| Gap around $u_c$ | low population 99th percentile 6.07, trapped 1st percentile 6.40 | same file |

### G.2.2 Background electrons

The background (loaded He and N L-shell electrons, `p_nz=1, p_nr=2, p_nt=8`) starts cold and at rest. As long as no orbits cross, it is one stream:
$$f_{bg}(\mathbf x,\mathbf u,t) = n(\mathbf x,t)\,\delta^3\big(\mathbf u - \mathbf U(\mathbf x,t)\big).$$
Inside the laser, $u_x \approx a(\mathbf x,t)$ oscillates on $\lambda_0/2\pi$, so $\mathbf U$ changes by $a_0 k_0\Delta z \approx 1.6$ per $z$ cell. With $k_p w_0 = 2.66$ and $a_0 \approx 4$, the wake is in or near blowout. Orbits cross at the sheath and at the back of the bubble, so there $f_{bg}$ becomes multi-stream: $f_{bg} = \sum_k n_k\,\delta^3(\mathbf u - \mathbf U_k)$, with caustics where $n_k$ is integrably singular. [derived]

In the quasi-cylindrical frame, the laser part of $f_{bg}$ is $\delta(u_r - U_r - a\cos\theta)\,\delta(u_\theta + a\sin\theta)\,\delta(u_z - U_z)$. Its $\theta$ dependence is inside a delta function, so it has no finite azimuthal mode expansion. A truncation at $m \le 1$ gives a smeared, non-positive function. The phase-space mesh must resolve $\theta$ (or $x, y$) explicitly; the two field modes do not carry over to $f$. [derived]

### G.2.3 EV electrons and the trapped bunch

EV emission copies `x, y, z, ux, uy, uz, inv_gamma` from the parent ion (plan Section 5.2; `fbpic/autodiff/ionization.py:326-360`, `elec_ux[j] = ion_ux[i]` at line 360). Ions are nearly at rest, so electrons are born with $\mathbf u \approx 0$ at the ion positions, at discrete steps that sample the laser phase every $22.5°$. In the continuum,
$$f_N(\mathbf x,\mathbf u,t) = \int dt_b \int d^3X\; \dot W(\mathbf X,t_b)\;\delta^3\big(\mathbf x - \mathbf x(t;\mathbf X,t_b)\big)\,\delta^3\big(\mathbf u - \mathbf u(t;\mathbf X,t_b)\big).$$
The labels are the ion position and the birth time, so $f_N$ lives on a 4-D manifold in 6-D phase space. At a fixed $\mathbf x$, the electrons form a curve in $\mathbf u$ (one point per birth time), so $\nabla_u f_N$ is singular in two momentum directions. [derived]

At $N_t$ the selected bunch is broad in projection: $\langle u_z\rangle \approx 12.7$, and the P1 moment scales (`inversion/identifiability.py:206-213`, `moment_scales`, the spreads of the selected beam) are 3.9 in $u_z$ and 1.07 in $u_x$ (`p1_twin_s.json`, keys `identifiability/v_model` and `scales`; P1 runs, before the EV model). "Thin" therefore means thin in the normal directions of the manifold, not narrow in the projected moments. The birth-phase spread fills the projections like a temperature, but it does not give $f_N$ a 6-D density. [derived, measured by P1]

### G.2.4 What a mesh or kernel estimate returns

Let $K_\sigma$ be a symmetric momentum kernel with variance $\sigma_u^2$ per direction (B-spline or Gaussian). For one stream,
$$K_\sigma *_u f = n\,K_\sigma(\mathbf u - \mathbf U), \qquad \nabla_u (K_\sigma *_u f) = n\,\nabla K_\sigma(\mathbf u - \mathbf U),$$
a dipole layer of thickness $\sigma_u$ and amplitude $n/\sigma_u^{c+1}$, with $c$ the codimension in $\mathbf u$ (3 for the background, 2 for $f_N$). A spatial shape of width $\Delta x$ adds the sheet slope inside a cell: the effective width is $\sigma_{eff}^2 \approx \sigma_u^2 + |\Delta x\,\partial_x\mathbf U|^2/6$ for a linear spline. In the laser region $\Delta z\,\partial_z u_x \approx 1.6$, so the mesh cannot represent any momentum structure finer than about 0.6 there, whatever $\Delta u$ is. [derived]

The result is not an approximation to a function. It is a regularized distribution, and its only meaning is through its pairing with a test function, which Section G.3 works out.

## G.3 Dipole identity, weak form, and the bias

### G.3.1 Identity

Write the relativistic Vlasov equation in FBPIC units ($\mathbf u = \mathbf p/mc$):
$$\partial_t f + \mathbf v\cdot\nabla_x f + \mathbf a\cdot\nabla_u f = 0,\qquad \mathbf v = c\,\mathbf u/\gamma,\quad \mathbf a = \frac{q}{mc}(\mathbf E + \mathbf v\times\mathbf B).$$
Since $\mathbf v = c\,\nabla_u\gamma(\mathbf u)$, $\nabla_u\times\mathbf v = 0$. With $\mathbf B$ independent of $\mathbf u$,
$$\nabla_u\cdot(\mathbf v\times\mathbf B) = \mathbf B\cdot(\nabla_u\times\mathbf v) - \mathbf v\cdot(\nabla_u\times\mathbf B) = 0 .$$
So $\nabla_u\cdot\mathbf a = 0$ and $\nabla_u\cdot\delta\mathbf a = 0$ with $\delta\mathbf a = \tfrac{q}{mc}(\delta\mathbf E + \mathbf v\times\delta\mathbf B)$. The linearized equation is
$$\partial_t\delta f + \mathbf v\cdot\nabla_x\delta f + \mathbf a\cdot\nabla_u\delta f = -\delta\mathbf a\cdot\nabla_u f + S_b = -\nabla_u\cdot(f\,\delta\mathbf a) + S_b,$$
with $S_b$ the EV birth source. [derived; confirms the briefing hint]

Over one step, $-\nabla_u\cdot(f\,\delta\mathbf a\,\Delta t)$ is the weight form of a momentum kick $\delta\mathbf u = \delta\mathbf a\,\Delta t$ applied to every particle. The Vay tangent applies exactly this kick (`econst * dEx` and the `dtau` terms in `fbpic/autodiff/tangent.py:582-599`) and then carries it as a trajectory perturbation, which stretches. The EV source $S_b$ is already in weight form: the birth weight tangent $\dot W'$ (plan Section 5.3; `ionization.py`, `ionize_tangent`) is a monopole at the birth point and needs no $\nabla_u f$. [derived]

### G.3.2 Weak form

Let the observable be $A = \int\phi\,f(T)\,dz$ with $z = (\mathbf x,\mathbf u)$. Let $\Phi_{t\to T}$ be the characteristic flow of the unperturbed fields, and $\psi_t = \phi\circ\Phi_{t\to T}$. The flow preserves phase-space volume ($\nabla_x\cdot\mathbf v = 0$, $\nabla_u\cdot\mathbf a = 0$), so $\frac{d}{dt}\int\psi_t\,\delta f_t\,dz$ equals the source paired with $\psi_t$. Integration by parts in $\mathbf u$ gives
$$\delta A = \int\psi_0\,\delta f_0\,dz + \int_0^T\!dt\!\int dz\,\Big[f\,\delta\mathbf a\cdot\nabla_u\psi_t + \psi_t S_b\Big]. \qquad (G1)$$
$\delta\mathbf a$ depends on $\delta f$ through $\delta\rho$, $\delta\mathbf J$ and the field tangents; (G1) holds with that $\delta\mathbf a$ by Duhamel's principle. For $f = \sum_p w_p\,\delta(z - z_p)$, the first source term is $\sum_p w_p\,\delta\mathbf a_p\cdot\nabla_u\psi_t(z_p)$, which is the chain rule of the Lagrangian tangent. Its factor $\nabla_u\psi_t = (\partial\Phi_{t\to T}/\partial\mathbf u)^T\nabla\phi$ grows as $e^{\lambda(T-t)}$ on chaotic orbits. This is the P7 explosion seen from the observable side. [derived]

Replace $\nabla_u f$ by $\nabla_u(K_\sigma *_u f)$, the exact-transfer limit of eq. (36). Because $\nabla_u\cdot\delta\mathbf a = 0$, the same steps give
$$\delta A_\sigma - \delta A = \int_0^T\!dt\!\int dz\; f\,\Big[K_\sigma *_u(\delta\mathbf a\cdot\nabla_u\psi_t) - \delta\mathbf a\cdot\nabla_u\psi_t\Big] \qquad (G2)$$
$$\phantom{\delta A_\sigma - \delta A} = \frac{\sigma_u^2}{2}\int_0^T\!dt\!\int dz\; f\,\nabla_u^2\big(\delta\mathbf a\cdot\nabla_u\psi_t\big) + O(\sigma_u^4), \qquad (G3)$$
if $\delta\mathbf a\cdot\nabla_u\psi_t$ is $C^2$ in $\mathbf u$ on the scale $\sigma_u$. (G2) needs no assumption on $f$; the $\mathbf v\times\delta\mathbf B$ term is included. [derived]

### G.3.3 Warm against cold

For a smooth $f$, move the kernel onto $f$ instead:
$$\delta A_\sigma - \delta A = -\int_0^T\!dt\!\int dz\;\psi_t\,\delta\mathbf a\cdot\nabla_u\big(K_\sigma * f - f\big) = -\frac{\sigma_u^2}{2}\int\!\!\int\psi_t\,\delta\mathbf a\cdot\nabla_u\nabla_u^2 f + O(\sigma_u^4). \qquad (G4)$$
For a Maxwellian the relative bias is $O(\sigma_u^2/v_T^2)$, and only $\lVert\psi_t\rVert_\infty$ enters. The roughness of $\psi_t$, that is, the chaos, does not enter the bias. It enters only the variance, because $\psi_t$ is sampled pointwise by the sensitivity particles. This is why the method works in the paper ($\Delta v/v_T = 0.08$ to 0.19). [derived]

Caveat: (G2) and (G4) assume exact transfer of the source to the sensitivity particles. The mesh transfer of eq. (42) divides by $\hat n$ and interpolates back. In weak form it pairs $\mathcal H_c$ with the sampler average of $\psi_t$ over cell $c$ (G5 below), which smooths $\psi_t$ on the mesh scale $(\Delta\mathbf x, \Delta\mathbf v)$. That adds a bias that depends on the roughness of $\psi_t$, even for a warm $f$. Dipole pairs avoid it because each pair sits exactly at its parent. [derived]

For a cold $f$, $\nabla_u^3 f$ is not integrable, so (G4) is not available. Only (G3) remains, and its size is set by $\psi_t$:
$$\frac{|\delta A_\sigma - \delta A|}{|\delta A|} \sim \Big(\frac{\sigma_u}{\ell_\psi(t)}\Big)^2,\qquad \ell_\psi(t) = \text{momentum scale of }\psi_t\text{ at the sheet}.$$
At $t = T$, $\ell_\psi$ is the scale of $\phi$: the selection edge $4\delta_u = 0.154$ in $u_z$, or the polynomial moments. Backward in time, phase mixing in the bucket shrinks $\ell_\psi$ algebraically, and chaos shrinks it as $e^{-\lambda(T-t)}$. With the P7 data, sources at step 1200 see $\ell_\psi \lesssim 0.154\times e^{-11}$ to $0.154\times e^{-15}$: the infinity norm of $\partial u_x/\partial E_L$ grows by $7.4\times10^4$ from step 1200 to 2000, and the rate of 10 per 120 steps gives $4.6\times10^6$. Then no affordable $\sigma_u$ is small, and $\delta A_\sigma$ is a different quantity from $\delta A$: the derivative of the observable coarse-grained in momentum on the scale $\sigma_u$. [derived, with the P7 rate from plan Section 12.10]

Interpretation. A symmetric dipole pair at $\mathbf u\pm\sigma_u\hat{\mathbf e}$ evaluates $[\psi_t(\mathbf u + \sigma_u\hat{\mathbf e}) - \psi_t(\mathbf u - \sigma_u\hat{\mathbf e})]/2\sigma_u$, a central difference of the pulled-back observable. The particle-pdf regularization is therefore a local FD in momentum, at step $\sigma_u$, applied at every source time. The FD secant in $\theta$ is a global FD at step $h$. Both are bounded by $\mathrm{osc}(\psi)/\text{step}$ and so never explode. Both depend on their step when the response has structure on that scale. [derived]

Does the bias vanish in an integrated sense? Two cases do vanish. First, at $t \to T$ with $\phi$ a polynomial of degree at most 2 in $\mathbf u$ and $\delta\mathbf a$ independent of $\mathbf u$, $\nabla_u^2(\delta\mathbf a\cdot\nabla_u\phi) = 0$. Second, the selection edge $S_5$ sits in the gap between the populations (6.07 to 6.40 in $u_z$), so $f\,\nabla^3 S_5$ is small at $t \approx T$. Neither holds for earlier sources, because $\psi_t = \phi\circ\Phi$ is not polynomial. There is no general cancellation of (G3). [derived]

Charge sensitivity. For $\log Q$, $\psi_t$ is close to the indicator of the pre-image of the selection, the trapping region. Its gradient is a surface delta, and (G1) becomes a flux of the sheet across the trapping boundary. If the sheet crosses the boundary transversally, the smoothed flux has $O(\sigma_u^2)$ bias. Ionization injection traps electrons that are marginal by construction, so the sheet runs nearly tangent to the boundary, and the flux has the form $n\,|\delta\mathbf a|/|\text{relative slope}|$. There $\sigma_u$ is again a regularization of a near-singular quantity, not a small correction. [derived]

### G.3.4 The collocated scheme on a sheet

Combine eqs. (41), (42), (44): the weights of collocated particles move along the characteristics, so $\delta A = \sum_s\hat W_s(T)\,\phi(z_s(T)) = -\sum_t\Delta t\sum_s h_s(t)\,\psi_t(z_s(t))$. Insert (42):
$$\delta A_{coll} = -\sum_t\Delta t\;\Delta\mathbf x\Delta\mathbf v\sum_{c:\,\hat n_c>0}\mathcal H_c\,\langle\psi_t\rangle_c,\qquad \langle\psi\rangle_c = \frac{\sum_s b_c(z_s)\,\psi(z_s)}{\sum_s b_c(z_s)} . \qquad (G5)$$
The estimate pairs the mesh dipole $\mathcal H_c$ with the sampler average of $\psi$ in each cell. Through $\sum_c\nabla b(\zeta_{p,c})\langle\psi\rangle_c$, it measures $\nabla_u\psi$ only by differences of $\langle\psi\rangle_c$ between neighbouring momentum cells. [derived]

If every sampler lies on the sheet, two cases occur.

1. Uniform stream ($\mathbf U$ constant in a cell): the momentum cells around the sheet hold the same particles, so the $\langle\psi\rangle_c$ are equal. Because the gradient stencil is one cell wider than the $\hat n$ stencil (G.1), the outer cells have $\mathcal H_c\ne0$ and $\hat n_c = 0$; they violate (43) and drop out. Since the derivative stencil sums to zero over all cells, $\delta A_{coll} = +\sum_t\Delta t\,\langle\psi_t\rangle\sum_{c:\,\hat n_c=0}\mathcal H_c\,\Delta\mathbf x\Delta\mathbf v$: the lost source times the mean of $\psi$. This is an artifact, not a derivative.
2. Sloped stream: the cells hold different sheet points, and the differences of $\langle\psi\rangle_c$ measure $\psi$ along the sheet. That is a tangential derivative, scaled by the sheet slope.

The normal derivative, which the momentum kick needs, cannot be observed from samples on the sheet. Any working scheme for a cold plasma must place sensitivity particles off the sheet, at momentum offsets of order $\sigma_u$: non-collocated samplers, adaptive addition in the $\sigma_u$-band, or dipole pairs. [derived]

### G.3.5 Consequence for variant R-v

The R-v projection is $\delta f = -\nabla_z\cdot(f\,\boldsymbol\xi)$ with $\boldsymbol\xi = (\delta\mathbf x, \delta\mathbf u)$. On a sheet, split $\boldsymbol\xi$ into a tangential part $\boldsymbol\xi_\parallel$ and a normal part $\boldsymbol\xi_\perp$ (normal space of dimension 3 for the background, 2 for the bunch). [derived]

1. $-\nabla_z\cdot(f\boldsymbol\xi_\parallel)$ is a monopole on the sheet, a relabelling density change $-\nabla_\parallel\cdot(n\boldsymbol\xi_\parallel)$. On-sheet weights can carry it, but it needs a divergence along an irregularly sampled sheet (`p_nz = 1`).
2. $-\nabla_z\cdot(f\boldsymbol\xi_\perp)$ is a dipole layer. Only off-sheet weights can carry it, by (G5).

The simplest exact-to-first-order projection is one pair per particle at $z_p \pm \sigma\hat{\boldsymbol\xi}_p$ with weights $\pm w_p|\boldsymbol\xi_p|/2\sigma$, in a scaled metric (for example $\mathbf x$ in units of $c/\omega_p$). It needs no mesh and covers both parts. R-v on a cold plasma is therefore the same scheme as the dipole pairs of G.5, with the Lagrangian tangent used between re-seeds; the reset interval sets the pair count (G.6.3). The $\sigma$-dependence of G.3.3 and G.4 applies unchanged. [derived]

## G.4 1D-1V experiment

Script: `work_gradf/cold_beam_bias.py` (estimators, $T = 30$), `work_gradf/short_horizon.py` ($T = 10$), `work_gradf/ref_scan.py` and `ref_scan_T.py` (fine scans of $J(A)$), `work_gradf/analyze_scan.py`, `work_gradf/small_sigma.py`, `work_gradf/chaos.py`. Outputs: `cold_beam_bias.log`, `short_T10.json`, `ref_scan.json`, `ref_scan_T10.json`, `small_sigma_T10.json`. Run with `python3 -I`.

Model. Test particles in a prescribed field, $x$ periodic on $[0, 2\pi)$:
$$\dot x = v,\qquad \dot v = -A\sin x - B\sin(x - \Omega t),\qquad A = 1,\ B = 0.2,\ \Omega = 2.2 .$$
The two resonances overlap, so orbits near the separatrices are chaotic, as in trapping. The parameter is $\theta = A$, and $\delta a = -\sin x$ exactly. There is no self-consistent field, so this tests only the $\nabla_u f$ question, not the field feedback. Leapfrog with $\Delta t = 0.05$. Observables at $T$: $J_1 = \langle S_5\rangle$ (a smooth selection on $v$, edge $v_c = 2$, transition width 0.4), $J_2 = \langle S_5 v\rangle$, $J_3 = \langle v^2\rangle$. Cold beam: $v_0 = 1$, $x_0$ uniform. Warm control: $v_0 = 1 + 0.3\,\mathcal N(0,1)$. The Lagrangian tangent at $T = 30$ has median $|\xi| \approx 10^2$ and maximum $3\times10^7$ (`chaos.py`). [measured]

Estimators.

1. Reference: $J(A)$ on 25 values $A \in [0.94, 1.06]$, $N = 4\times10^5$ stratified particles, two seeds; central differences at several $h$ and cubic fits.
2. Lagrangian tangent (particle-exact), mean of $\nabla\phi\cdot\xi$.
3. Dipole pairs: every 4 steps, each base particle emits a pair at $v\pm\sigma$ with weights $\pm\,\delta a\,4\Delta t/2\sigma$; pairs move in the unperturbed field. This is the weak form (G1) with a two-point kernel.
4. Collocated mesh scheme of eqs. (36), (41), (42): linear shape in $x$, derivative of the quadratic spline in $v$ for $\mathcal H$, linear shape in $v$ for $\hat n$ and the transfer.

Results at $T = 10$ (before strong mixing). The reference is unambiguous here: the central differences at $h = 0.005$ to $0.06$ agree to 3 digits for the cold beam, and the two seeds agree exactly (stratified sampling). Mean ± one standard error. [measured: `short_T10.json`, `ref_scan_T10.json`, `small_sigma_T10.json`]

| Estimator | Cold $J_1$ | Cold $J_3$ | Warm $J_1$ | Warm $J_3$ |
|---|---|---|---|---|
| Reference (central differences, $h = 0.005$ to 0.06) | 0.291 | 3.35 | 0.034 to 0.050 | 1.46 to 1.49 |
| Lagrangian tangent, $N = 2\times10^5$ | 0.25 ± 0.05 | 3.17 ± 0.19 | -0.02 ± 0.07 | 1.55 ± 0.23 |
| Pairs, $\sigma = 10^{-4}$, $N = 4000$ | -0.08 ± 0.22 | 5.8 ± 2.4 | 0.01 ± 0.22 | -0.7 ± 2.5 |
| Pairs, $\sigma = 0.01$ | 0.36 ± 0.09 | 4.30 ± 0.38 | -0.06 ± 0.10 | 1.01 ± 0.44 |
| Pairs, $\sigma = 0.03$ | 0.65 ± 0.06 | 5.42 ± 0.29 | 0.10 ± 0.06 | 1.24 ± 0.28 |
| Pairs, $\sigma = 0.1$ | -0.30 ± 0.03 | 0.85 ± 0.18 | 0.054 ± 0.024 | 1.31 ± 0.18 |
| Pairs, $\sigma = 0.2$ | -0.12 ± 0.02 | 1.61 ± 0.13 | 0.071 ± 0.017 | 1.50 ± 0.13 |
| Pairs, $\sigma = 0.4$ | 0.06 ± 0.01 | 2.14 ± 0.09 | 0.050 ± 0.011 | 1.45 ± 0.09 |
| Collocated mesh, $\Delta v = 0.025$, $N = 10^5$ | -0.61 ± 0.05 | -4.10 ± 0.32 | 0.029 ± 0.024 | 1.38 ± 0.22 |
| Collocated mesh, $\Delta v = 0.05$ | -0.13 ± 0.03 | -0.98 ± 0.18 | 0.023 ± 0.013 | 1.37 ± 0.13 |
| Collocated mesh, $\Delta v = 0.1$ | -0.22 ± 0.01 | -1.16 ± 0.08 | 0.021 ± 0.007 | 1.29 ± 0.08 |
| Collocated mesh, $\Delta v = 0.2$ | -0.14 ± 0.01 | -0.68 ± 0.05 | 0.012 ± 0.004 | 1.16 ± 0.05 |

Small-$\sigma$ follow-up, cold beam, $T = 10$, $N = 8\times10^4$ base particles (8 seeds of $10^4$; `small_sigma.py`, `small_sigma_T10.json`):

| $\sigma$ | $J_1$ | $J_2$ | $J_3$ | Bias of $J_3$ |
|---|---|---|---|---|
| 0.01 | 0.436 ± 0.023 | 1.078 ± 0.049 | 4.01 ± 0.09 | +20% (7.5 SE) |
| 0.003 | 0.402 ± 0.043 | 0.987 ± 0.093 | 3.97 ± 0.17 | +19% (3.6 SE) |
| 0.001 | 0.301 ± 0.050 | 0.777 ± 0.107 | 3.70 ± 0.17 | +10% (2.1 SE) |
| Reference | 0.291 | 0.705 | 3.347 | |

The pair estimate converges as $\sigma \to 0$ in this regular regime, but much more slowly than $\sigma^2$: the $J_1$ bias is +50%, +38%, +3% and the $J_3$ bias +20%, +19%, +10% for $\sigma = 0.01, 0.003, 0.001$. A bias of 10% needs $\sigma \approx 10^{-3}$, that is, 1/400 of the selection width and 1/1000 of the beam velocity scale. There the standard error with $8\times10^4$ particles is still 17% for $J_1$. [measured]

Results at $T = 30$ (after mixing; the Lagrangian tangent is useless here). [measured: `cold_beam_bias.log`, `ref_scan.json`]

| Estimator | Cold $J_1$ | Cold $J_3$ | Warm $J_1$ | Warm $J_3$ |
|---|---|---|---|---|
| Reference central difference, $h = 0.005$ | -0.98 | -10.6 | 0.17 | 1.36 |
| $h = 0.01$ | -0.46 | -6.4 | 0.12 | 0.97 |
| $h = 0.02$ | -0.41 | -3.3 | 0.13 | 1.05 |
| $h = 0.04$ | 0.09 | -0.04 | 0.09 | 0.92 |
| $h = 0.06$ | 0.44 | 1.89 | 0.13 | 1.06 |
| Lagrangian tangent, $N = 2\times10^4$ | -142 ± 85 | -1650 ± 1200 | 41 ± 66 | -770 ± 480 |
| Pairs, $\sigma = 0.01$, $N = 4000$ | -1.23 ± 0.64 | -13.2 ± 3.4 | 0.61 ± 0.57 | 5.6 ± 3.2 |
| Pairs, $\sigma = 0.03$ | -0.77 ± 0.23 | -7.3 ± 1.3 | 0.20 ± 0.21 | 2.1 ± 1.2 |
| Pairs, $\sigma = 0.1$ | 0.14 ± 0.08 | -1.84 ± 0.45 | 0.20 ± 0.07 | 1.33 ± 0.44 |
| Pairs, $\sigma = 0.3$ | 0.19 ± 0.03 | 0.35 ± 0.16 | 0.088 ± 0.025 | 0.64 ± 0.16 |
| Collocated mesh, $\Delta v = 0.05$, $N = 2\times10^4$ | 0.11 ± 0.12 | -3.1 ± 0.8 | -0.01 ± 0.07 | 0.34 ± 0.62 |
| Collocated mesh, $\Delta v = 0.1$ | 0.16 ± 0.06 | -0.73 ± 0.41 | 0.05 ± 0.04 | 0.33 ± 0.33 |
| Collocated mesh, $\Delta v = 0.2$ | 0.08 ± 0.03 | -0.36 ± 0.23 | 0.044 ± 0.021 | 0.21 ± 0.18 |

Findings.

1. Collocated scheme on a cold beam: wrong sign for $J_1$, $J_2$, $J_3$ at every $\Delta v$ at $T = 10$, where the truth is unambiguous, and no convergence as $\Delta v$ decreases. The lost-source fraction (cells with $\mathcal H \ne 0$, $\hat n = 0$) is 3 to 5% for the cold beam and 0.1 to 0.3% for the warm beam. This confirms (G5). On the warm beam at $T = 10$, the collocated scheme agrees with the reference within 2 standard errors at $\Delta v \le 0.05$ ($\Delta v/v_T \le 0.17$, the paper's regime), is low by about 6 standard errors at $\Delta v = 0.2$ ($0.67\,v_T$), and moves toward the reference as $\Delta v$ falls. [measured]
2. Pairs on a cold beam at $T = 10$: the true derivative is smooth and the Lagrangian tangent is unbiased, yet the pair estimate is off by +28% ($J_3$, 2.5 standard errors) at $\sigma = 0.01$, +62% at $\sigma = 0.03$, and -75% at $\sigma = 0.1$. The bias is not monotone in $\sigma$ and is not $O(\sigma^2)$. With $8\times10^4$ particles it falls to 3 to 10% only at $\sigma = 10^{-3}$, where the standard error is 5 to 17%. So even before chaos, the cold beam needs a $\sigma$ two orders of magnitude below the selection width, at a variance cost $\propto 1/\sigma^2$. The cause is (G3): the beam crosses the separatrix of the first wave, and there $\psi_t$ varies in $v$ on scales below 0.01 off the sheet. Trapping in LWFA is the same geometry. [measured, interpretation derived]
3. Pairs on a warm beam ($v_T = 0.3$): consistent with the reference within 2 standard errors for $\sigma = 0.03$ to 0.4 at $T = 10$ and for $\sigma \le 0.1$ at $T = 30$. At $\sigma = 0.3 \approx v_T$ and $T = 30$, the estimate is low by about 2 combined standard errors. This is the plateau that (G4) predicts. [measured]
4. Cold beam at $T = 30$: the reference itself depends on the scale. $J(A)$ has seed-independent wiggles of amplitude about 0.02 in $J_1$ on the scale $\Delta A \approx 0.01$ (`ref_scan.json`): folds of the cold sheet sweep through the selection window as $A$ changes. The central difference moves from -0.98 at $h = 0.005$ to +0.44 at $h = 0.06$. The pair estimate follows the same sequence with $\sigma$: $J_3$ is -13.2, -7.3, -1.84, 0.35 for $\sigma = 0.01, 0.03, 0.1, 0.3$, against -10.6, -6.4, -3.3, -0.04, 1.89 for $h = 0.005$ to 0.06. In this model, $\sigma$ plays the role of an FD step with $\sigma \approx 2h$ to $5h$. The warm reference has no such wiggles (0.09 to 0.17 for $J_1$). [measured]
5. The pair variance law is as predicted: standard error $\propto 1/\sigma$ (G.5). [measured]
6. The warm collocated scheme at $T = 30$ is low by 2 to 4 combined standard errors for $J_3$ at $\Delta v = 0.1$ and 0.2. The mesh transfer (42) replaces $\psi$ by its cell average over the samplers, which smooths $\psi$ on the mesh scale in $x$ and $v$, even for a warm $f$. (G4) holds only in the exact-transfer limit, which pairs reach and the mesh transfer does not. The significance is moderate; treat this as a warning for R1. [measured, interpretation derived]

Limits of the experiment. The fields are prescribed. There is no field feedback, no N-body chaos (the chaos is single-particle, from two overlapping waves), one dimension, and an observable with a sharper selection (width 0.4 for a beam spread of about 2) than the twin. It tests the $\nabla_u f$ question only. It shows that a cold beam crossing a separatrix has no small $\sigma$, even in a regular regime.

## G.5 Regularization options

| Option | Bias | Variance constant | Chaotic stretching | Cold-plasma verdict |
|---|---|---|---|---|
| Fixed-width phase-space histogram, collocated (paper Sections 3.2.2 to 3.2.4) | Warm: $O(\Delta v^2/v_T^2)$ (G4), plus a transfer bias from cell-averaging $\psi$ (measured low by 2 to 4 standard errors at $T = 30$). Cold: not a derivative; lost source times $\langle\psi\rangle$ plus a tangential derivative (G5) | Low, like PIC | None: weights are scalars | Fails [derived, measured] |
| Same, non-collocated samplers filling the $\sigma_u$-band | $O(\Delta v^2/\ell_\psi^2)$ by (G3); no small parameter after mixing | $\propto \mathrm{osc}(\psi)/\Delta v$ per sampler pair; requires $\hat n\Delta\mathbf x\Delta\mathbf v \ge 1$ in every band cell | None | Works as a coarse-grained derivative; band sampling in 6-D is the cost driver |
| Adaptive mesh (Section 8.4, Algorithm 1) | Same as the line above | Same; adds particles only where $\lvert\mathcal H\rvert\ge\mathcal H_{min}$ | None | Best mesh variant; needs a sparse 6-D hash mesh and 6-D redistribution (eq. 50) |
| Per-particle kernel density estimate | Same as the histogram with width $\sigma_u$ | Higher: ratio of two noisy KDEs; neighbour search in 6-D | None | Collocated KDE fails for the same reason as (G5); non-collocated KDE equals the pair method at higher cost |
| Local cold-fluid closure, $f\approx\sum_k n_k\delta(\mathbf u - \mathbf U_k)$, Eulerian $\delta n_k$, $\delta\mathbf U_k$ | Zero $\sigma_u$-bias where single-stream | Fluid fields, no particle noise of their own | None in Eulerian form, but $\delta\mathbf U = \xi_u - (\xi_x\cdot\nabla)\mathbf U$ is large where $\nabla\mathbf U$ is steep | Fails at orbit crossing (bubble sheath, bunch) and caustics; works only where the Lagrangian tangent already works [derived] |
| Dipole pairs at $\mathbf u\pm\sigma_u\hat{\mathbf e}$ (mesh-free) | (G3): $O(\sigma_u^2/\ell_\psi^2)$; equals a local FD of $\psi$ at step $\sigma_u$ | $\propto\mathrm{osc}(\psi)/\sigma_u$ after a pair decorrelates; measured SE $\times\sigma$ constant | None while weights are scalars; pair separation grows, but each pair contributes at most $\mathrm{osc}(\psi)/\sigma_u$ | Works as a coarse-grained derivative; count growth needs re-seeding (G.6) |
| Momentum kicks on the particle itself | None | Exponential: $e^{\lambda(T-t)}$ | Yes: this is the Lagrangian tangent | Not a regularization |
| Physical temperature added to the model, then the warm scheme | Model bias (a different primal), derivative bias $O(\Delta v^2/v_T^2)$ | Like PIC | None | Needs $\Delta v\ll v_T$ and $|\Delta x\,\partial_x\mathbf U|\ll v_T$; the laser region has $1.6$ per cell, so a usable $v_T$ changes the physics [derived] |

The measured variance law: for the cold beam at $T = 30$, the pair standard error times $\sigma$ is $0.0064, 0.0068, 0.0077, 0.0076$ for $\sigma = 0.01, 0.03, 0.1, 0.3$ ($N = 4000$ base particles), so the variance constant is $\propto 1/\sigma$ as predicted. [measured, `cold_beam_bias.log`]

Bias against variance. With relative bias $(\sigma/\ell_\psi)^2$ and relative noise $(\ell_\psi/\sigma)N^{-1/2}$, the error is smallest at $\sigma/\ell_\psi \sim N^{-1/6}$ and is then $\sim N^{-1/3}$. For $N = 1.7$ M selected particles, $N^{-1/3} \approx 0.008$. This holds only if $\ell_\psi$ exists, that is, if $\psi_t$ is smooth on some scale. [derived]

## G.6 Dimensionality and cost on Twin-S

### G.6.1 Dimensions

1. Space: $(z, r, \theta)$ with $\theta$ resolved (G.2.2). Use at least 8 $\theta$ cells, matching `p_nt = 8`, or Cartesian $(x, y)$.
2. Momenta: all three. For $E_L$, $\delta\mathbf a$ has a $u_z$ part ($\delta E_z$ of the $m = 0$ wake), a $u_r$ part ($\delta E_r - c\,\delta B_\theta$), and a $u_x$ part (laser, $m = 1$). The residuals contain $\langle u_x\rangle$, $\sigma_{u_x}$, $\sigma_{u_y}$, $\langle u_z\rangle$, $\sigma_{u_z}$ (plan Section 2.4).
3. Resolution: $\Delta u_z \le 0.05$ to resolve the selection edge $4\delta_u = 0.154$; $\Delta u_\perp \approx 0.05$ for $\sigma_{u_x}\approx 1$. Ranges: $u_z\in[0, 15]$, $u_\perp\in[-3, 3]$.
4. Where $\Delta z\,\partial_z\mathbf U$ exceeds $\Delta u$ (the laser region), the mesh cannot resolve the sheet. Keep the Lagrangian tangent there; it is accurate (verified to step 500). [derived]

### G.6.2 Cell counts and memory

| Mesh | Cells | Memory at 8 B per value |
|---|---|---|
| Dense, full window: $800\times50\times8\times300\times120\times120$ | $1.4\times10^{12}$ | 11 TB |
| Dense, bunch box 5 µm × 2 µm × 8 $\theta$ cells | $1.7\times10^{10}$ | 138 GB |
| Sparse $\sigma_u$-band, background: $4\times10^4$ $(z,r)$ cells × 8 × 27 × 1 to 3 streams | $0.9$ to $2.6\times10^7$ | 0.3 to 0.8 GB |
| Sparse band, EV electrons near the axis: about $2\times10^4$ config cells × $10^3$ to $3\times10^3$ momentum cells along the birth curve | $2$ to $6\times10^7$ | 0.6 to 2 GB |

Each sparse cell holds a key, $\hat n$, $\mathcal H$ for $K$ parameters, and $h$: about 32 B at $K = 1$. With a hash table at load 0.5, the total is about 3 to 6 GB. Memory is not the constraint on the 48 GB GPU. [estimated]

### G.6.3 Time

Inputs [plan]: primal 88.8 s per run (44 ms per step); $K = 1$ gradient 298 s (3.47 run-equivalents); at step 1200, 9.8 M `elec_from_N` and 228 ms per tangent step, of which the deposit tangents take 121 ms against 38 ms for the primal deposits.

Sparse band mesh, per step [estimated]:

1. $\mathcal H$ deposit: every PIC electron touches $2^3\times3^3 = 216$ cells (linear in space, quadratic-derivative in momentum).
2. $\hat n$ deposit and $h_s$ gather: every sampler touches $2^6 = 64$ cells each. Samplers number 2 to 3 times the PIC particles (paper Fig. 18c shows about 2 times).
3. With 12.5 M electrons: $2.7\times10^9 + (2.5\text{ to }3.7)\times10^7\times128 \approx 6$ to $7.5\times10^9$ hashed updates per step.
4. At an assumed 2 to $5\times10^9$ hashed float64 atomic updates per second (not measured on this GPU), that is 1.2 to 3.8 s per step, or 27 to 85 run-equivalents per 2000-step run.
5. Restricted to the bunch-relevant electrons (about 2 M, the pre-image of the selection plus a margin, found from tracker IDs of a primal run) and to steps 1000 to 2000: a factor of about 0.08, so 2 to 7 run-equivalents, plus about 1.7 for the Lagrangian part before step 1000 and about 0.5 for sampler pushes and deposits. Total 4 to 9 run-equivalents at $K = 1$.

Cross-check against the paper: Table 1 (p37) scales the source evaluation with $l^{2d}d^2$. From $d = 1$ to $d = 3$ at $l = 2$ this term grows from 4 to 576. The paper measured 1.9 to 3 PIC-equivalents in 1D-1V, so the naive extrapolation to 6-D is of order $10^2$ PIC-equivalents for the full domain. This agrees in order of magnitude with item 4. [estimated from paper]

Dipole pairs, mesh-free [estimated]:

1. Keep the Lagrangian tangent up to a projection step $t_p$ (variant R-v), for example 1000.
2. After $t_p$, give each bunch-relevant electron 4 companions (pairs along $u_z$ and $u_r$, which carry the $m = 0$ wake kicks) or 6 (all three axes). Accumulate the kicks into the companion weights. When a pair has stretched beyond a bound or after $\tau$ steps, freeze its weights and start a new pair at the parent.
3. Count: $6\times2\,\mathrm M\times(1000/\tau)$. With $\tau = 120$ (one decade of P7 growth), about $10^8$ companions, 6 to 7 GB at $K = 1$.
4. Time: a companion costs about one primal particle (gather, push, scalar deposit), about 3.5 ns per particle per step from the primal (44 ms for 12.5 M electrons). The count grows linearly to $10^8$, so the mean is about $5\times10^7$: about 175 ms per step, or 2 run-equivalents for 1000 steps. Add about 1.7 for the Lagrangian part before $t_p$ and about 0.5 to 1.5 for the field tangents after $t_p$. Total: about 4 to 6 run-equivalents at $K = 1$.

Multi-parameter scaling. Sampler or companion positions do not depend on the parameter; only weights, $\mathcal H$, and the field tangents do. A batched $K$-weight deposit, as in the existing deposit tangent (`tangent.py:873`, one `values` matrix with $K$ columns), keeps the marginal cost per parameter below the first. A plausible model is $c_0 + c_1 K$ with $c_0 \approx 3$ and $c_1 \approx 1$ to 2 run-equivalents (the measured $K = 1$ AD gradient already costs 2.47 run-equivalents above the primal), against $2K$ for central FD and $K + 1$ for one-sided FD. Forward-mode Track R beats central FD only for $K > 3$ and only if $c_1 \approx 1$; it never beats one-sided FD on cost. Its value must come from accuracy and smoothness, not cost. A many-parameter cost advantage needs an adjoint form, which the paper names as future work (p50). [estimated]

## G.7 Recommendation, kill criteria, open questions

Recommendation for the design note.

1. Drop the collocated mesh scheme for electrons. It is wrong for cold sheets by (G5) and by the experiment.
2. Do not build a 6-D phase-space mesh for the gradient. Represent the source by dipole pairs (off-sheet samplers) in the bunch-relevant set after the projection step, and keep Lagrangian tangents where they are regular (laser region, background ahead of the bunch, all steps before $t_p$). Use a mesh only if pair counts must be merged; then a sparse band mesh restricted to the bunch is the only affordable form.
3. Treat $\sigma_u$ as a coarse-graining scale and report every Track R column as a function of $\sigma_u$, like FD columns as a function of $h$.
4. In Phase R2, add a cold-beam test with self-consistent fields. Its pass criterion is a $\sigma_u$-plateau: over at least a factor 3 in $\sigma_u$, the smoothed column changes by less than its standard error and matches the smooth FD secant of the same model.

Kill criteria (measurable; any one makes Track R a no-go for Twin-S).

1. R2 cold-beam model: no $\sigma_u$ range of at least a factor 3 in which the pair or band-mesh estimate of every observable changes by less than 2 standard errors.
2. R2: on the plateau, the estimate differs from the smooth FD secant of the same model by more than 20% or 2 standard errors, whichever is larger.
3. Twin-S at $N_t = 2000$ (R4): the Track R column at three $\sigma_u$ values spanning a factor 10 changes by more than the difference between FD at $h = 10^{-2}$ and $10^{-3}$, or its cosine with the smoothed column of P7 step 5 is below 0.9.
4. Cost: a measured $K = 1$ Track R gradient above 10 run-equivalents (about 900 s), or a GPU peak above 40 GB.
5. Variance: the standard error from 4 independent sampler seeds exceeds 10% of the column norm at the plateau $\sigma_u$.

Open questions.

1. Does Twin-S have a $\sigma_u$-plateau? The FD secants at $h = 10^{-2}$ and $10^{-3}$ agree for most residuals (plan Section 12.10 step 4), which suggests a smooth macroscopic response at those scales. The 1D cold beam has no such plateau. The birth-time spread of ionization injection may smooth the folds in 3-D; this is not shown.
2. What is the map between $\sigma_u$ and $h$? A first guess is $\sigma_u \approx h\,|\partial\mathbf u/\partial\hat\theta|_{\text{regular}}$. P7 reports only infinity norms of the tangents, which outliers dominate. A median per species, from `diag_tangent_growth.py`, would give the regular scale.
3. Are background tangents regular after step 1200? If yes, only the bunch-relevant set needs the weight form, and the cost drops to the restricted estimates. P7 reports the norm over all particles only.
4. Re-seeding of pairs: the frozen companion weights still deposit $\delta\rho$ and $\delta\mathbf J$. Their deposit after decorrelation is noise with amplitude $\propto 1/\sigma_u$; its effect through the field tangents on the column is not quantified.
5. Hashed float64 atomic throughput on the RTX 6000 Ada sets the band-mesh cost to within a factor of 3. It is not measured.
6. The trapping-boundary flux for marginal (tangent) crossings: is it regularized by the birth-phase spread, or does it need $\sigma_u$?
