# Density-profile fitting (`fludat_fit`)

Fit the parameterized density profiles of `inversion_fbpic` to lineouts of the gas-jet
density cubes produced by [`../postproc`](../postproc/) (`fludat_proc`), and compare
how well each profile family reproduces a lineout.

```
density cube (fludat_proc)                              inversion_fbpic.lib.density_profiles
  density[z_m, x_mm, pressure_bar]                        AsymmetricSine, GenericConicalTarget, ...
            │                                                          │
            ▼ dataset                                                  ▼ families
   NozzleDataset.lineout(x, pressure, angle)              ProfileFamily(parameter_space, build)
   (density along an oblique line in (z, x))
            └──────────────────────┬───────────────────────────────────┘
                                   ▼ fitting
                        FittingScheme.fit(lineout, family) -> FitResult
                        (MultiStartLocalFit: bounded L-BFGS-B, Latin-hypercube starts)
                                   │
                                   ▼ goodness_of_fit
                        compare_families -> ranked FamilyComparison
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
     explore_fits            fit_statistics          evaluate_models
  sliders for x, pressure,   a distribution of     learning: TrainingSet of
  angle; fit + save the      condition points;     (conditions, local fit) ->
  selected point             per-family report     models scored by direct /
                                                   indirect loss
```

A lineout is selected by its physical conditions `(x_mm, pressure_bar, angle_deg)`: the
transverse offset of the lineout line at `z = 0`, the backing pressure, and the angle of
the line to the `z` axis. Those conditions travel with every fit result, so a set of
fits is already a table of `conditions -> profile parameters`; learning that mapping is
future work and nothing here has to change for it beyond adding a new `FittingScheme`.

## Installation and running

The package lives in [`fludat_fit/`](fludat_fit/). It imports `inversion_fbpic`
(installed in the `inv-fbpic` conda env) and `fludat_proc` (found automatically in the
sibling `../postproc` checkout; no install needed). Run the command-line module with
`python -m` from this directory, or `pip install -e .` to register the console scripts below.

```bash
conda run -n inv-fbpic python -m fludat_fit.explore_fits ../postproc/data/density_field/htu_dens_7_0.h5
conda run -n inv-fbpic python -m fludat_fit.fit_statistics \
    ../postproc/data/density_field/htu_dens_7_0.h5 --angle-range -10 10 --samples 32
conda run -n inv-fbpic python -m fludat_fit.explore_fits --list-families
```

| Module | Console script | Purpose |
|---|---|---|
| `fludat_fit.explore_fits` | `explore-fits` | Interactive fit of the lineout selected by x / pressure / angle sliders |
| `fludat_fit.fit_statistics` | `fit-statistics` | Fit a distribution of condition points and report per-family performance |
| `fludat_fit.evaluate_models` | `evaluate-models` | Build a training set and evaluate `conditions -> parameters` models |
| `fludat_fit.generate_profile_configs` | `generate-profile-configs` | Fit a JSON template of profile families at given `(x, p, angle)` points |
| `fludat_fit.plot_profile_configs` | `plot-profile-configs` | Plot every profile of a filled config on one figure, with vlines at their centres |

Library modules: `dataset` (oblique lineouts of a cube), `lineout`, `families`, `fitting`,
`goodness_of_fit`, `sampling` (condition boxes), `plotting`, `cli_common` (shared
command-line options), `profile_reconstruction` (rebuilding real `inversion_fbpic`
profiles from a filled config, shared by `generate_profile_configs` and
`plot_profile_configs`), and the `learning` subpackage (model evaluation framework).

Requirements: Python ≥ 3.10, NumPy ≥ 2.0, SciPy, Matplotlib, attrs, `inversion_fbpic`,
`fludat_proc`. Tests: `conda run -n inv-fbpic python -m pytest` from this directory.

## Datasets and the lineout angle

A density cube has axes `(z, x, pressure)`. The third physical parameter, `angle_deg`,
is the angle of the **lineout line** to the `z` axis in the `(z, x)` plane, following
the affine-path form of `inversion_fbpic`'s `InterpolateFromH5Profile`:

```
z_m(t)  = cos(angle) * t              x_mm(t) = x_mm + 1000 * sin(angle) * t
```

with the path parameter `t` in metres and the angle positive towards `+x`. `angle = 0`
is a plain lineout along `z` at fixed `x`. `NozzleDataset.lineout` follows that class's
conventions exactly: the path is clipped to the part inside the cube (the range of `t`
over which both `z` and `x` stay on their grids), it is sampled with one point per
participating grid point (`n_z` for an axial line, `n_z + n_x` otherwise), and the
density is trilinear in `(z, x, pressure)`. The returned `Lineout.z` is `t`, the
longitudinal coordinate FBPIC sees. `NozzleDataset.h5_profile_kwargs(conditions)` gives
the matching `InterpolateFromH5Profile` arguments (`lineout_axis`, `interpolation_points`),
and the explorer's *Save* writes them next to each fit.

For `--method linear` the trilinear path is vectorised and fast enough for slider use;
other methods fall back to the per-point `fludat_proc` interpolation, which is slow.

### Options shared by both scripts

| Argument | Default | Description |
|---|---|---|
| `hdf5_path` | — | The density cube. |
| `--method` | `linear` | `(x, pressure)` interpolation method of `fludat_proc`. |
| `--z-bounds` | jet support + padding | Fit window along the lineout [mm]. |
| `--cutoff-ratio` | `5e-3` | Peak fraction that defines the jet support. |
| `--padding` | `0.25` | Vacuum kept on each side, as a fraction of the support width. |
| `--max-points` | `1000` | Resample longer lineouts to this many samples (`0`: keep all). |
| `--families` | all | Family names; see `--list-families`. |
| `--starts` | `8` (`4` in the explorer) | Latin-hypercube starting points per family, plus the heuristic guess. |
| `--optimizer` | `L-BFGS-B` | Any bounded `scipy.optimize.minimize` method (`Powell`, `TNC`, ...). |
| `--seed` | `0` | Seed of the start sample. |
| `--rank-by` | `bic` | `bic`, `aic`, `nrmse`, `rmse`, `sse`, `integrated_relative_error`, `max_abs_error`, `r_squared`. |

The fit window matters: by default the lineout is trimmed to where the density exceeds
`--cutoff-ratio` of its peak, plus `--padding` of vacuum on each side, so profiles are
also required to vanish outside the jet without the far vacuum dominating the residual.

## `explore_fits`

```bash
python -m fludat_fit.explore_fits htu_dens_7_0.h5 --x 1.0 --pressure 20 --angle 5 \
    --families conical:supergaussian generalized_lorentzian_sum[2] --live \
    --output-dir fits --species H
python -m fludat_fit.explore_fits cube.h5 --x 0.5 -o explorer.png     # static figure
```

Sliders for x, pressure and the lineout angle select the lineout line, whose density is
redrawn immediately. **Fit** fits the families and overlays the `--top` best curves with
their residuals and a ranked table; **Save** writes the ranked results as JSON and the
best profile as an FBPIC YAML config to `--output-dir`. After each fit the previous
parameters of every family are used as warm starts, so small slider moves refit quickly.

| Argument | Default | Description |
|---|---|---|
| `--x`, `--pressure`, `--angle` | lowest, lowest, `0` | Initial slider values. |
| `--angle-range` | `-45 45` | Angle slider range [deg] (at most ±89). |
| `--top` | `3` | Ranked model curves drawn. |
| `--live` | off | Refit on every slider change instead of on *Fit*. |
| `--fit` | off | Fit the initial point before showing the window. |
| `--output-dir` | `fits` | Where *Save* writes `<family>_x..mm_p..bar_a..deg.json/.yaml`. |
| `--nominal-density`, `--species` | fitted amplitude, none | Values written into the YAML config. |
| `-o`, `--output` | — | Fit the initial point and save the figure instead of showing it. |

## `fit_statistics`

```bash
python -m fludat_fit.fit_statistics htu_dens_7_0.h5 \
    --x-range 0 3 --pressure-range 5 40 --angle-range -10 10 --samples 32 --workers 4 \
    --json stats.json --csv stats.csv --plot stats.png --param-stats stats_params.json
python -m fludat_fit.fit_statistics cube.h5 --pressure-range 5 35 --sampling grid --samples 9
```

Samples `--samples` condition points over the selected ranges (default: the cube's x and
pressure extents with the angle fixed at 0; ranges reaching outside are clamped with a
warning), fits every family at every point, and reports per family: how often it ranks first, its mean rank, the median /
90th percentile / maximum NRMSE, the median integrated relative error and BIC, the
success rate and the time per fit. Families are ordered by the median of the ranking
metric. The plot shows the NRMSE distribution and rank-first frequency per family and
NRMSE against one condition for the best families.

| Argument | Default | Description |
|---|---|---|
| `--x-range`, `--pressure-range` | cube extent | `MIN MAX`; equal values fix the axis. |
| `--angle-range` | `0 0` | Lineout angle range [deg], at most ±89. |
| `--samples` | `16` | Number of condition points (a grid rounds up to a full lattice). |
| `--sampling` | `lhs` | `lhs`, `random`, or `grid`. |
| `--sample-seed` | `0` | Seed of the point sample. |
| `--workers` | `1` | Processes fitting points in parallel (lineouts are extracted first, so the cubes stay in the main process). |
| `--json` | — | Summary plus every fit; `--include-profiles` adds each serialized FBPIC config. |
| `--csv` | — | One row per (point, family): conditions, rank, metrics, amplitude and the fitted parameters. |
| `--param-stats` | — | Per family, per fitted parameter (plus `amplitude`): median and a `--confidence`-level confidence window across every fitted point, as JSON. Worth checking before committing to a long run at full `--samples`. |
| `--confidence` | `0.95` | Confidence level for `--param-stats`. |
| `--plot`, `--show` | — | Save / show the summary figure; `--plot-axis` picks the condition on the scatter's x axis. Each family's label includes its free-parameter count, e.g. `generalized_lorentzian_sum[3] (13)`. |
| `-q`, `--quiet` | off | Suppress per-point progress on stderr. |

The CSV is the raw material for a `conditions -> parameters` mapping: filter it to one
family and you have `(x_mm, pressure_bar, angle_deg) -> theta` rows.

## Library use

```python
from fludat_fit import (
    LineoutConditions, MultiStartLocalFit, NozzleDataset, compare_families,
    fit_family, get_families, load_lineout,
)

lineout = load_lineout("cube.h5", x_mm=1.0, pressure_bar=12.5).trimmed()
# or along a line tilted 5 degrees from the z axis:
dataset = NozzleDataset.load("cube.h5")
lineout = dataset.lineout(LineoutConditions(x_mm=1.0, pressure_bar=12.5, angle_deg=5.0)).trimmed()
dataset.h5_profile_kwargs(lineout.conditions)   # InterpolateFromH5Profile arguments

comparison = compare_families(lineout, scheme=MultiStartLocalFit(n_starts=16))
print(comparison.table())
best = comparison.best                       # FitResult, ranked by BIC by default
best.parameters                              # {"center": ..., "fwhm": ..., ...}
best.amplitude, lineout.density_units        # peak scale in cube units
best.model_density(lineout.z)                # fitted curve in cube units
profile = best.build_profile(species="H", ionization=0, p_nz=2, p_nr=2, p_nt=4)
profile.to_yaml_file("best.yaml")            # ready for an FBPIC config

result = fit_family(lineout, get_families(["asymmetric_sine"])[0])
```

`build_profile` converts the fitted amplitude to `nominal_density` in m^-3 when the
cube units are recognised (`m^-3`, `cm^-3`, `mm^-3`); otherwise pass `nominal_density`
explicitly. Density profiles evaluate at `r = 0` only; all supported profiles are
functions of `z` alone.

## Profile families

Each family wraps one `inversion_fbpic` profile class. The family declares bounds
derived from the lineout (positions inside the window, lengths from 1/200 of the jet
support to twice the window, searched in log scale) and builds the actual profile
object for every evaluation, so the fitted curve is exactly what FBPIC would load.

The profile *shape* is what a family parameterizes; the overall *amplitude* is solved
in closed form at every evaluation (variable projection) and reported separately.
Families therefore fix intrinsic amplitudes (`gauss_peak`, the first Lorentzian `A`) to
one, and the amplitude counts as one extra parameter in the information criteria.

| Family name | Class | Parameters (shape) |
|---|---|---|
| `asymmetric_sine` | `AsymmetricSine` | `peak_z0`, `upramp_length`, `downramp_length` |
| `smooth_sine_flattop` | `SmoothSineFlattop` | `center`, `flattop_width`, `upramp_length`, `downramp_length` |
| `gaussian_plus_triangle` | `GaussianPlusTriangle` | `gauss_z0`, `gauss_sigma`, `tri_z0`, `tri_left_width`, `tri_right_width`, `tri_height` |
| `generalized_gaussian_plus_triangle` | `GeneralizedGaussianPlusTriangle` | as above with `gauss_alpha`, `gauss_beta` (`gauss_peak = 1`) |
| `bilateral_supergaussian` | `BilateralSuperGaussian` | `center`, `fwhm_left`, `beta_left`, `fwhm_right`, `beta_right` — a supergaussian with an independent width and shape exponent on each side of `center` (peak is always exactly 1) |
| `generalized_lorentzian_sum[n]`, n = 1, 2, 3 | `GeneralizedLorentzianSum` | per term `c_i`, `w_i`, `b_i`, `m_i`; `A_0 = 1`, `A_i ∈ [-1, 1]` |
| `power_law_flattop` | `PowerLawFlattop` | `center`, `flattop_width`, `transition_length`, `transition_exponent`, `skew_rate` |
| `conical:<main>` | `GenericConicalTarget` | `center`, the main-type parameters, `skew_rate` |
| `conical:<main>+<fringe>` | `GenericConicalTarget` | plus `fringe_center_offset`, `fringe_relative_height`, `fringe_*` |

Main types: `supergaussian`, `cosine_squared_flattop`, `power_law_flattop`,
`lorentzian_flattop`; fringe types: `supergaussian`, `cosine_squared`, `power_law`,
`lorentzian`. The registry includes the four fringe-less conical families plus
`conical:supergaussian+cosine_squared` and `conical:lorentzian_flattop+lorentzian`; any
other combination is one constructor call away:

```python
from fludat_fit.families import GenericConicalTargetFamily, GeneralizedLorentzianSumFamily
extra = [
    GenericConicalTargetFamily("power_law_flattop", "supergaussian", fringe_side="right"),
    GeneralizedLorentzianSumFamily(n_terms=4),
]
comparison = compare_families(lineout, [*get_families(), *extra])
```

Centred profiles (`center`) replace the classes' `start_position`, and the cosine
flattop's `ramp_length` becomes `ramp_fraction * fwhm` so the class constraint is a plain
bound. Note that the `_FiniteSkewedProfile` classes multiply by a Gaussian envelope in
their default `finite_supergaussian` skew mode even at zero skew, so e.g. a fitted
super-Gaussian `fwhm` is the class parameter, not the curve's measured FWHM.

Not fitted: `InterpolateFromH5Profile` (it reads the same cubes the lineouts come from)
and `ExampleDensityProfile` (documented as not for practical use).

`families.FixedParameterFamily(family, parameter_name, fixed_value)` wraps any family
with one named parameter pinned to a fixed value: it is removed from the searched
`ParameterSpace` and merged back in wherever the wrapped family builds a profile or
evaluates its relative density, so every *other* parameter is still fitted normally,
free to compensate. `generate_profile_configs` uses this to pin every family's position
parameter to one shared, data-driven centroid (see below); it is equally usable
anywhere else a parameter's correct value is known externally and should not vary per fit.

### Plotting a filled config (`plot_profile_configs`)

```bash
python -m fludat_fit.plot_profile_configs data/fit_cfgs/profile_parameters_x1mm_p20bar_a0deg.json
python -m fludat_fit.plot_profile_configs data/fit_cfgs/profile_parameters_x*.json -o data/fit_cfgs
```

Reconstructs every profile of a filled config with the real `inversion_fbpic` classes
(`center` substitutes for `start_position` on conical profiles, as the template
documents) and draws each one over its own `get_z_extent()` — its full defined support,
not just the region it was fitted against — on one figure, each with a dashed vertical
line at its `center` or `centroid` (the same concept under two names, depending on the
class). Showing the full extent matters: a profile can fit well where it was compared to
data while still carrying an isolated, badly-placed bump elsewhere (what
`max_deviation_weight` above discourages during fitting; this plot is how you check it
actually did). `InterpolateFromH5Profile` has no builder or centre here: its curve is the
density cube's own lineout, converted to m^-3 and sampled over the union of every other
profile's extent, drawn as the reference the others were fitted against, with no line.
One PNG is written per config, named after it.

Each reconstructed profile is translated back out of its own left-edge-zero output frame
into the frame it was actually fit in, where every profile shares one common anchor — the
lineout's own crossing of `z_m = 0`. Without this, profiles would overlay in a coordinate
frame the data itself does not use, making the comparison meaningless; with it, every
dashed vline coincides at that one shared anchor, exactly as before the re-basing was
introduced in `generate_profile_configs`.

## Generating FBPIC profile configs (`generate_profile_configs`)

`data/fit_cfgs/profile_parameters_template.json` is a hand-written schema for a
downstream FBPIC run script: one physical point (`density_file`, `x_mm`, `p_bar`,
`angle`) plus several named profiles, each `null` until filled in. `generate_profile_configs`
fills every profile's `null` fields by fitting its matching family (by `class`, and for
`GenericConicalTarget` by `main_profile_type`/`fringe_profile_type`) to the lineout at
each requested point, and writes one config per point. A profile with no `null` in its
`kwargs` (such as `h5_direct`) is copied through unchanged since it needs no fit.

Every fitted profile shares one `center`/`centroid` (the same concept under two names):
the point along *that* lineout where it crosses the gas target's physical `z_m = 0`
symmetry plane (`dataset.path_for(conditions).t_at_physical_z(0.0)`), which by
construction of the lineout path is always `t = 0`, regardless of `x_mm` or `angle`. This
is *not* the density-weighted centroid of the lineout's own data: the target is
symmetric about `z_m = 0`, but an oblique line (or an axial one off `x = 0`) samples that
symmetric field along an asymmetric path, so the sampled data's own weighted centroid is
generally offset from `z_m = 0` even though the target's true centre is not. Fitting each
family to its own data-weighted centroid would reproduce that sampling artefact rather
than the target's real geometry, and the families would disagree with each other besides.

That anchor is pinned *during* fitting via `families.FixedParameterFamily` (`center` on
`GenericConicalTarget`; the dominant term's `c_0` on `GeneralizedLorentzianSum`), not
translated afterwards — every other parameter is still free to compensate, so the fit
only pays for the constraint it actually needs. `GenericConicalTarget`'s `center` *is*
its centroid, so this pins it exactly. `GeneralizedLorentzianSum` has no single centre
parameter; its reported `centroid` is the same fixed anchor by definition, not a
density-weighted average recomputed from the fitted terms, which would generally
disagree with it once there is more than one term.

Every written-out profile is then rigidly re-based into its own frame where its
`get_z_extent()` starts at exactly `0.0` — the same convention `InterpolateFromH5Profile`'s
`centering_mode="left"` already uses for `h5_direct`. A profile's support is generally
asymmetric about its pinned anchor (skew, fringes, satellite terms), so this shift differs
per profile even within one config. After the shift the anchor is no longer at `0`; the
*reported* `center`/`centroid` is updated to the anchor's new position in that profile's
own frame (`-z_extent[0]` measured before the shift), which is what a consumer needs in
order to place the laser focus correctly once it has instantiated the profile in its own
left-based frame. `GeneralizedLorentzianSum`'s `density_cutoff_ratio` is recorded
explicitly in the written `kwargs` for the same reason: `get_z_extent()` depends on it, so
a reconstruction that silently fell back to the class's own default instead of the value
actually used at fit time would no longer land exactly on `0`.

```bash
python -m fludat_fit.generate_profile_configs htu_dens_7_0.h5 \
    data/fit_cfgs/profile_parameters_template.json \
    --point 1.0 20.0 0.0 --point 2.0 20.0 30.0 --point 3.0 20.0 -30.0 \
    --output-dir data/fit_cfgs
```

| Argument | Default | Description |
|---|---|---|
| `hdf5_path`, `template` | — | The density cube and the template JSON. |
| `--point X_MM P_BAR ANGLE_DEG` | — | Repeatable; one config is written per point. |
| window options | jet support + padding | As for the other scripts. |
| `--starts`, `--optimizer`, `--seed` | `16`, `L-BFGS-B`, `0` | Local-fit settings (more starts than the other scripts by default, since a handful of points is cheap to fit well). |
| `--output-dir` | the template's directory | Where the filled configs are written. |
| `--density-file` | `hdf5_path` relative to `--output-dir` | Overrides the recorded `density_file`, e.g. to keep a path relative to a different location. |

## Learning `conditions -> parameters` (`fludat_fit.learning`)

The end goal is a model that predicts a family's parameters directly from
`(x_mm, pressure_bar, angle_deg)`. The `learning` subpackage is the evaluation framework
for such models; it does not care what the model is.

**Data.** `build_training_set` samples conditions, extracts the lineouts, and fits the
family to each with the local multi-start optimiser. Everything lives in one *reference*
`ParameterSpace` per family (the envelope of the per-lineout bounds), so a
`TrainingSet` exposes `features` (conditions normalised to `[0, 1]`) and `targets` (the
local optima as unit-cube vectors). It splits (`split`, `k_folds`), saves and loads as
`.npz`, and hands out each sample's `FitObjective`.

**Objectives.** Both score a predicted unit-cube vector `u` per sample:

| Objective | Loss | Needs local fits | Gradient |
|---|---|---|---|
| `DirectObjective` | the local optimiser's own normalised SSE of the profile built from `u` against the lineout (`FitObjective`) | no | central finite differences in the unit cube |
| `IndirectObjective` | `mean_k w_k (u_k - u*_k)^2` against the local optimum `u*` | yes | analytic |

A model trained on the direct loss learns to minimise the fit error itself; the indirect
loss is a plain regression on the local optimiser's answers.

**Models.** Anything with `fit(training_set, objective)` and `predict(features)` is a
`ParameterModel`. `fit` may use `objective.loss` and `objective.gradient` (a
gradient-based model trains on either loss through them), or ignore the objective (a
lookup). Baselines: `constant`, `knn3` (needs targets), `poly1` / `poly2` (sigmoid-squashed
polynomial trained by L-BFGS-B on whichever objective, the template for differentiable
models). A `ModelFactory` (zero-argument callable) supplies fresh instances per fold.

**Evaluation.** `evaluate_model` trains on one set and scores on another: the fit
quality of the predictions (`GoodnessOfFit` through the same `FitObjective`), the ratio
to the local fit's NRMSE, the unit-cube parameter error (when targets exist), and with
`refine=True` how many objective evaluations a local optimiser needs when warm-started
from the prediction versus the cold multi-start fit. `compare_models` runs every model
under every objective with k-fold cross-validation (or one split) and reports a ranked
table, JSON, per-sample CSV rows and a figure.

```python
from fludat_fit import NozzleDataset, get_families, MultiStartLocalFit
from fludat_fit.sampling import ConditionRanges
from fludat_fit.learning import (
    DirectObjective, IndirectObjective, TrainingSet, build_training_set,
    compare_models, get_model_factories,
)

dataset = NozzleDataset.load("htu_dens_7_0.h5")
family = get_families(["conical:supergaussian"])[0]
ranges = ConditionRanges.from_dataset(dataset, x_range=(0, 3), angle_range=(-10, 10))
training_set = build_training_set(
    dataset, ranges.sample(64), family, ranges=ranges, scheme=MultiStartLocalFit(), workers=4
)
training_set.save("htu_sg.npz")            # reuse: TrainingSet.load("htu_sg.npz")

class MyModel:                              # any framework: torch, sklearn, ...
    name = "mine"
    def fit(self, training_set, objective):
        # training_set.features (N, 3), objective.loss(u, training_set), objective.gradient(...)
        ...
    def predict(self, features):
        return ...                          # (N, d) in [0, 1]

comparison = compare_models(
    {"mine": MyModel, **get_model_factories()}, training_set,
    [DirectObjective(), IndirectObjective()], folds=4, refine=True,
)
print(comparison.table())
best = comparison.ranked()[0]               # ModelEvaluation
best.test_set.space.from_unit(best.predictions[0])   # physical parameters of one prediction
```

### `evaluate_models`

```bash
python -m fludat_fit.evaluate_models htu_dens_7_0.h5 --family conical:supergaussian \
    --x-range 0 3 --angle-range -10 10 --samples 64 --workers 4 \
    --training-set htu_sg.npz --refine --json models.json --csv models.csv --plot models.png
python -m fludat_fit.evaluate_models --training-set htu_sg.npz --models poly1 knn3 \
    --objectives direct --folds 0 --test-fraction 0.3
```

| Argument | Default | Description |
|---|---|---|
| `hdf5_path`, `--method`, window options | | As for the other scripts; the cube is optional when `--training-set` exists. |
| `--family` | `conical:supergaussian` | Family whose parameters are learned. |
| `--training-set` | — | `.npz` cache: loaded when present, otherwise built and saved. |
| `--x-range`, `--pressure-range`, `--angle-range` | cube extent, angle `0` | Sampled condition box. |
| `--samples`, `--sampling`, `--sample-seed` | `32`, `lhs`, `0` | Condition points. |
| `--starts`, `--optimizer`, `--workers` | `8`, `L-BFGS-B`, `1` | Local fits for the targets. |
| `--no-targets` | off | Skip local fits (direct objective only; no reference or indirect metrics). |
| `--models` | all built-in | `constant`, `knn3`, `poly1`, `poly2`. |
| `--objectives` | both | `direct`, `indirect`. |
| `--folds`, `--test-fraction`, `--seed` | `4`, `0.25`, `0` | Cross-validation folds (`0`: one split). |
| `--refine` | off | Warm-start a local optimiser from each prediction and report the savings. |
| `--json`, `--csv`, `--plot`, `--show`, `-t` | — | Outputs. |

Table columns: `nrmse med/p90` of the predicted fits, `ratio med` (predicted / local-fit
NRMSE), `indirect` (median unit-cube squared error), `refined` (NRMSE after polishing)
and `evals%` (objective evaluations of the polish relative to the cold fit).

## Fitting schemes and goodness of fit

`MultiStartLocalFit` optimises `FitObjective` over the family's unit cube from
`n_starts` Latin-hypercube points plus the family's heuristic initial guess, with
`scipy.optimize.minimize` (`L-BFGS-B` by default). `FitObjective` adds three terms:

- The weighted, normalised sum of squared residuals (unit-free: it is divided by the
  weighted sum of squared data). This alone was the whole objective originally.
- `max_deviation_weight` (default `1.0`) times a smooth stand-in for the squared
  *relative* worst-point error, `(max(|residual|) / peak) ** 2` (a weighted power mean
  of `|residual| / peak`, not the exact max, so gradients stay well behaved for the
  finite-difference line search `L-BFGS-B` uses). The default weight is calibrated so
  this term matches the sum-of-squares term exactly when `model - data` is a nonzero
  constant over flat data, independent of the number of points: a plain sum-of-squares
  fit can leave an isolated, badly-missed region far from the bulk of the data (its
  share of the *sum* is small next to everywhere else); this term penalises that worst
  region directly. Pass `max_deviation_weight=0.0` to drop it.
- `extent_weight` (default `1.0`) times the squared excess of the built profile's own
  `get_z_extent()` width over `allowed_extent_ratio` (default `2.0`) times the fit
  window's width — zero unless that ratio is exceeded. The first two terms are only
  ever evaluated inside the fit window, so a family like `generalized_lorentzian_sum`
  can fit the window perfectly while one term's shape parameters give it an enormous,
  numerically negligible tail outside it (nothing before this term would notice).
  `get_z_extent()` is the same quantity `plot_profile_configs` draws over, so a fit
  this term accepts will not later turn out to carry a wildly disproportionate tail.
  Unlike the worst-point term, there is no flat-data identity calibrating
  `allowed_extent_ratio`; since the fit window is already padded around the jet's
  support, a built profile at most twice as wide as that window is a threshold choice,
  not a derivation. Pass `extent_weight=0.0` to drop it.

Parameter vectors that cannot build a profile score a large constant. Options:
`method`, `seed`, `options` (solver options), `fit_amplitude=False` to pin the amplitude
to the data peak, and `Lineout.weights` for weighted fits.

`GoodnessOfFit` reports `sse`, `rmse`, `nrmse` (÷ peak), `r_squared`, `max_abs_error`
(÷ peak), `integrated_relative_error` (`∫|model − data| dz / ∫|data| dz`, the metric of
`fludat_proc.convergence`), and Gaussian-residual `aic`/`bic`. Families are ranked by
`bic` by default so that extra parameters have to earn their keep.

### Adding a family or a scheme

- **Family**: subclass `ProfileFamily`, set `name` and `profile_class`, implement
  `parameter_space(summary)` and `build(parameters, *, nominal_density, **overrides)`;
  optionally `initial_guess(summary)`. Register it in `family_registry` or pass it to
  `compare_families` directly.
- **Scheme**: any object with a `name` and `fit(lineout, family) -> FitResult`
  satisfies `FittingScheme`. A model that predicts parameters from
  `LineoutConditions` would return the predicted `theta` with the closed-form
  amplitude (`fitting.optimal_amplitude`) and a `GoodnessOfFit.compute(...)`, and plugs
  into `compare_families`, the JSON report and the plots unchanged.
- **Learned model**: implement `fit(training_set, objective)` / `predict(features)` (see
  above) and pass a factory to `compare_models`; no framework code changes.

## Development

```bash
conda run -n inv-fbpic python -m pytest      # from this directory
pre-commit run --files fludat_fit/*.py tests/*.py
```

The tests build small synthetic cubes; the real HTU cubes under
`../postproc/data/density_field/` (about 0.5 GB each, loaded fully into memory) are the
intended inputs for the two scripts.
