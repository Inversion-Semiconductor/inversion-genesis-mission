# Exploration FBPIC

This package runs staged Bayesian exploration campaigns for the ionization-injection
FBPIC template. It uses Xopt's `BayesianExplorationGenerator` to learn the response
surface rather than optimizing a prematurely chosen scalar objective.

The campaign data model is deliberately independent of Xopt. Every run records the
complete physical parameter vector, including parameters held at nominal values in
earlier stages. A later stage can therefore activate additional dimensions and reuse
all compatible completed simulations as observations on the nominal slice.

## Install

Use the FBPIC-compatible Python environment on the target machine:

```bash
cd lpa/Exploration_FBPIC
python -m pip install -e ../Simulation_FBPIC
python -m pip install -e '.[nersc,dev]'
```

`libensemble` and `mpi4py` are only needed for NERSC dispatch. The base package
includes Xopt, scalar storage, and the local single-run workflow.

The package installs two commands: `explore-fbpic` for local evaluation, design,
inspection, and plotting; and `explore-fbpic-libe` for resource-managed
libEnsemble campaigns. Run `pytest -q` from this directory after installation to
verify the campaign tooling and its saved-result contract.

## Campaign configuration

[runs/initial_study/campaign.yaml](runs/initial_study/campaign.yaml) is the
production campaign template. It contains:

- every physical input accepted by `IonizationInjectionSimulation`;
- nominal values and reviewed, conservative bounds;
- linear or logarithmic optimization coordinates via `log_scale` on positive parameters;
- readable scaled input coordinates via `linear_scale` (default `1`), with physical
  values restored before FBPIC execution;
- active dimensions for each study stage;
- fixed numerical/execution settings and the scalar diagnostic outputs used for exploration.

The configuration fingerprint is written to every `run_manifest.json`. Change bounds
or active stages as needed; do not change the frozen template, numerical settings,
or analysis settings within a campaign. Begin a new campaign when those must change.

`run_root` supports environment-variable expansion. The production configuration
uses `${PSCRATCH}/ionization_injection_exploration`; set `PSCRATCH` before local
commands or replace `run_root` in a copied configuration with an absolute local
directory. Do not use the production run root for an unrelated campaign because
existing manifests with a different fingerprint cause preflight to fail.

### Adapting a new template

Most template updates require only a new campaign YAML: set `template_script` and,
when needed, `template_class`; update the physical parameter names, nominal values,
bounds, stages, and fixed execution settings. The generated entrypoint imports the
module named by `template_script`'s filename and the configured class (default:
`IonizationInjectionSimulation`). The executor, campaign store, scalar extraction,
and Slurm dispatch remain unchanged when that class accepts the physical and
hyperparameter mappings, provides a `.run()` method, and writes the saved-analysis
contract described below.

If the template moves to a configuration-based constructor or CLI, adapt
`FBPICRunExecutor._write_entrypoint` to write the template's per-run configuration
from the manifest and invoke its new run method. The rest of the exploration workflow
can remain unchanged as long as the saved datapoint contract stays compatible.
Update `extract_features` only when that output contract changes.

### Saved analysis results

The template calculates its diagnostics during the configuration-based run and
writes a `ConfigContainer` to `results/datapoint.json` by default. The executor
loads the diagnostic result dictionaries from that file; it does not reload raw
openPMD particles or repeat selection, cropping, or calculations. Parameter metadata
is ignored, and nested containers are supported.

Set `execution.hyperparameters.results_directory` to use another results directory;
relative paths are resolved inside each run directory. JSON, YAML, and HDF5 datapoints
are supported. HDF5 uses the template's `/results` group. Keep exactly one
`datapoint` config in that directory so stale files in another format cannot be
selected accidentally.

Every returned feature must be a unique, finite numeric scalar, and all names in
`analysis.exploration_features` must be present. Missing or invalid saved results
fail extraction rather than triggering fresh analysis. Diagnostic targets and
options belong to the template; particle filters and cropping are controlled by
`execution.hyperparameters.analysis_select` and `analysis_central_fraction`.
Legacy `analysis.species`, `uz_min`, `central_fraction`, and `longitudinal_bins`
settings are no longer used by the executor. Raw particle dumps are not required
for feature extraction.

## Stage 1

Stage 1 varies 11 controls: plasma density; both density widths and ramp locations;
laser energy, duration, third-order spectral phase, spot size, and focal position;
and fourth-order astigmatism. It retains all 33 physical inputs in each manifest.
The production stage seeds 64 evaluations: the nominal point, low/high anchors for
each active parameter, and a Sobol fill:

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_1 initial-design \
  --count 64 --seed 0 --output initial_design.json
```

The template's diagnostics save the fixed 33-scalar moment descriptor and mean
kinetic energy. The executor copies these saved scalars into `result.json`, alongside
stdout/stderr logs and an append-only `campaign_runs.parquet` record under the
configured `run_root`.

## Local smoke evaluation

After validating the bounds and environment, run one point:

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_1 evaluate \
  --point '{"laser_energy_J": 2.5}'
```

This creates an isolated run directory, writes the full physical vector to
`run_manifest.json`, launches `run_fbpic.py`, and reads its saved diagnostic
results. It is expected to consume normal FBPIC resources.

Local `evaluate` does not impose an execution timeout. For NERSC/libEnsemble
campaigns, set `execution.simulation_timeout_s` to terminate and record an
individual simulation that exceeds its allocation.

Inspect the reusable successful records and the exact feature/parameter columns
used for the selected stage:

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_1 status
```

Each run writes its durable `result.json` before the aggregate Parquet table is
updated. Rebuild that table after an interrupted local run or filesystem issue:

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_1 rebuild-store
```

## Virtual-waist shifted scalars

When `analysis.virtual_waist_shift.enabled` is true, successful runs also write
`results_shifted.json`. The full beam is drifted to the configured plane
downstream of a virtual waist defined by the central mean-`uz` slice containing
the configured charge fraction. The native diagnostic result remains in
`result.json`; live exploration and the default rebuild use the shifted features.

The shift runs on the worker CPU. It is dominated by openPMD HDF5 loading and
charge-weighted NumPy reductions; transferring each completed particle dump to a
GPU would add more overhead than the two-coordinate drift saves. The code can be
ported to CuPy if particles become available on-device, but that is not the
current FBPIC diagnostic path.

To shift all successful runs in an existing NERSC campaign and rebuild a compact
shifted table, run:

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_2 shift-results \
  --run-root /pscratch/path/to/campaign
explore-fbpic runs/initial_study/campaign.yaml --stage stage_2 rebuild-store \
  --run-root /pscratch/path/to/campaign \
  --output /pscratch/path/to/campaign/campaign_runs_shifted.parquet
```

`shift-results` reports every analysis failure and exits nonzero if any successful
run could not produce a shifted record. Shifted rebuilds retain native `no_beam`
records so feasibility modeling continues to include physical no-beam outcomes.

## Cached scalar surrogates

`train-surrogate` trains independent one-output GP models from an archived
campaign Parquet and caches each model under its scalar name:

```text
stage2_gp_cache/
  var_total_beam_charge_pc/
    metadata.json
    training.parquet
    model_state.pt
```

`training.parquet` freezes the model-coordinate inputs and one selected output;
`model_state.pt` stores the fitted BoTorch/GPyTorch state dictionary; and
`metadata.json` records source and artifact hashes, the source campaign
fingerprint, ordered inputs/bounds, GP constructor settings, and package versions.
The cache uses a bounded 128-inducing-point approximate GP rather than an exact
GP, so loading and posterior scans remain fast as the campaign grows. The cache
loader rebuilds the architecture from the snapshot and applies the state
dictionary without fitting again.

Train charge first, or train every scalar feature in the input Parquet:

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_2 train-surrogate \
  --data /path/to/campaign_runs.parquet \
  --output stage2_gp_cache \
  --variable total_beam_charge_pc

explore-fbpic runs/initial_study/campaign.yaml --stage stage_2 train-surrogate \
  --data /path/to/campaign_runs.parquet \
  --output stage2_gp_cache \
  --variable all
```

`--variable all` skips any existing `var_<feature>` directory by default, so a
later invocation resumes an interrupted multi-scalar cache build. Pass
`--overwrite` to retrain and replace those existing caches.

The input Parquet must contain exactly one historical campaign fingerprint. This
allows post-analysis of archived campaign data even after the live YAML has evolved.

## Compact campaign assessment

The aggregate `campaign_runs.parquet` contains the complete parameter vector,
run status, and extracted scalar diagnostics for every run. It is sufficient to
assess sampling coverage and surrogate predictiveness; raw FBPIC diagnostics do
not need to leave the compute system. Copy this one file from NERSC, or run the
following directly beside the campaign output:

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_1 analyze-campaign \
  --output stage_1_analysis
```

For a downloaded Parquet file, pass it directly rather than setting `PSCRATCH`:

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_1 analyze-campaign \
  --data /path/to/campaign_runs.parquet --output stage_1_analysis
```

This writes `campaign-analysis.json` with compatibility/run-health counts,
unit-cube coverage distances, per-feature five-fold held-out Xopt GP errors,
and a data-size heuristic. `campaign-projection.png` contains PCA scatter
projections of the normalized active inputs and scalar-response descriptor.
The held-out GP is deliberately fitted from the scalar table, so this
command does not read particle dumps or submit simulations.

Plot the current charge GP across focal position and peak plasma density, with
all attempted simulations overlaid. The remaining Stage 1 inputs are held at
their nominal values for the GP slice; white markers have no charge diagnostic
and were not included in the GP training data.

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_1 plot-charge-slice \
  --data /path/to/campaign_runs.parquet --output charge-gp-slice.png
```

Use `plot-feature-slice` for another configured exploration feature, such as
the third longitudinal mean-momentum bin:

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_1 plot-feature-slice \
  --feature longitudinal_mean_uz_02 --data /path/to/campaign_runs.parquet \
  --output longitudinal-mean-uz-02-gp-slice.png
```

Plot the separate beam-existence classifier used to prevent feature exploration
from repeatedly proposing confidently no-beam conditions. The final panel shows
binary classification entropy, so high values identify the injection boundary.
The probability panel marks 5%, 10%, 15%, 20%, 25%, and 30% contours. The
campaign PCA projection also marks each observed no-beam point with a blue `X`
with feasibility bands: pink at 20-30%, orange at 10-20%, red at 5-10%, and
maroon below 5%. Green `X` markers identify measured beams where the classifier
predicts less than 30% feasibility. Beam-present markers use a black-to-blue
charge scale.

Write additional PC1/3, PC1/4, PC2/3, and PC2/4 views to inspect whether the
apparent no-beam cluster persists outside the default PC1/PC2 projection:

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_1 \
  plot-input-pca-projections --data /path/to/campaign_runs.parquet \
  --output input-pca-projections
```

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_1 plot-feasibility-slice \
  --data /path/to/campaign_runs.parquet --output beam-feasibility-slice.png
```

`stages.<name>.minimum_beam_feasibility` controls the adaptive exploration
floor. Candidates at or above the floor retain Xopt's feature-acquisition order;
below-floor candidates only backfill an otherwise incomplete batch. Stage 2 uses
`0.10` to suppress confidently no-beam regions without excluding the injection
boundary.

Preview a batch without creating run directories or submitting work. The JSON
log records the Xopt generator order, each candidate's feasibility, whether it
meets the configured floor, marginal relative combined GP uncertainty, and the
final selected points. The uncertainty is an independent diagnostic; Xopt uses
joint batch posterior-variance optimization, so return order is not a simple
per-candidate uncertainty ordering.

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_1 preview-batch \
  --data /path/to/campaign_runs.parquet --count 32 \
  --minimum-beam-feasibility 0.10 --output batch-preview.json
```

The rebuild scans `sim_*/result.json` below the configured `run_root`; it does not
rerun FBPIC. Local `evaluate` automatically chooses the next numeric `sim_` ID.
Pass `--run-id` when an external dispatcher needs a predetermined, unique ID.

## Inspecting density ramps

Plot the four lower/upper combinations of `density_fwhm_left_m` and
`density_left_ramp_edge_percent` without running FBPIC. The latter places the
1% density point at a fraction of the available distance between the left FWHM
point and the simulation edge; the template derives beta from it, FWHM, and the
density center. All other physical parameters remain at their campaign nominal
values. `--simulation-half-length` is in metres and sets
`density_center_location_m`, so the figure spans the left half of the simulation
from zero to the density peak:

For the initial study, the requested 10% to 100% range is remapped before beta
is derived: at the minimum FWHM it remains 10% to 100%, while at the maximum
FWHM it becomes 55% to 100%. This suppresses overly steep large-FWHM ramps.

```bash
explore-fbpic runs/initial_study/campaign.yaml plot-density-ramps \
  --simulation-half-length 5e-3 \
  --output density-ramp-extremes.png
```

The command also writes `density-ramp-extremes-parameter-space.png`: 16 sampled
boundary points from the derived FWHM/beta region, with the direct
`beta = 1` to `4` rectangle overlaid for comparison. Use
`--parameter-space-output` to choose a different path.

Plot the nominal full density profile with lightweight random profiles sampled
from a stage's active campaign parameters:

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_2 \
  plot-density-envelope --random-samples 30 --seed 7 \
  --output density-profile-envelope.png
```

## NERSC/libEnsemble

Use a single Slurm allocation and let libEnsemble assign one GPU or GPU/MPI team per
FBPIC evaluation. The persistent driver runs Xopt's Bayesian exploration generator
on the libEnsemble manager and uses resource-set workers to launch FBPIC. It starts
with the campaign's deterministic initial design, records the completed scalar
outputs, then asks Xopt for later candidates as GPU resource sets become available.

A production submission starts one manager and 32 four-GPU simulation workers. Do
not dispatch background `explore-fbpic evaluate`
processes from the batch script: libEnsemble's `MPIExecutor` owns the FBPIC MPI
launch and GPU binding. The reproducible NERSC smoke fixtures live under
`tests/smoke_campaigns/`, and the four-GPU OOM repro lives under
`tests/memory_stress/`; neither belongs under production `runs/`.

Install the NERSC extra before running the driver:

```bash
python -m pip install --no-cache-dir libensemble
```

Then submit from the campaign directory:

```bash
sbatch submit_nersc.sh
```

For a site-specific allocation, invoke the installed driver directly after arranging
the required manager and worker ranks:

```bash
explore-fbpic-libe campaign.yaml --stage stage_1
```

`runs/initial_study/submit_nersc.sh` is the authoritative Perlmutter/libEnsemble
submission format. Its node count, `SLURM_EXACT` settings, and 33-rank `srun`
launch are coupled to one manager-hosted generator and 32 four-GPU workers; keep
that allocation pattern intact when changing the production campaign.

Before Stage 2 production, submit the isolated continuation test:

```bash
sbatch submit_stage_2_test_nersc.sh
```

It requests four GPU nodes for 30 minutes and starts four four-GPU workers. The
script copies production `campaign_runs.parquet` into
`$PSCRATCH/ionization_injection_stage_2_test/history`; all test artifacts are
written to its separate `runs` directory, leaving the production run root
unchanged.

To exercise feasibility-gated adaptive point generation in the original Stage 1
parameter space without modifying its production history, submit:

```bash
sbatch submit_stage_1_continue_test_nersc.sh
```

This uses the same four-node, 30-minute allocation and copies only
`campaign_runs.parquet` into its own history root. Because its
`initial_design_size` is zero, all four submitted points are generated by Xopt
then filtered through the 10% feasibility policy.

After the test completes successfully, Stage 2 production uses 128 concurrent
four-GPU simulations across 128 GPU nodes:

```bash
sbatch submit_stage_2_nersc.sh
```

The production script currently requests a 30-hour regular-QoS allocation.
Confirm the account, QoS, and wallclock against the approved NERSC allocation
before submission.

### Single-case NERSC checks

The production campaign's nominal single-case entrypoints are grouped under
`runs/initial_study/nominal/`. Each submission script runs the colocated
`run_nominal.py`, which loads the parent `runs/initial_study/campaign.yaml` and
evaluates its Stage 1 nominal point with inspection diagnostics. Submit the
variant matching the requested GPU count:

```bash
sbatch runs/initial_study/nominal/submit_nominal_nersc.sh
sbatch runs/initial_study/nominal/submit_nominal_2gpu_nersc.sh
sbatch runs/initial_study/nominal/submit_nominal_4gpu_nersc.sh
```

The reproducible four-GPU memory-stress case is deliberately kept with test
artifacts. It combines extrema observed in distinct Stage 0 OOM cases, then
runs against the same production campaign:

```bash
sbatch tests/memory_stress/submit_memory_stress_4gpu_nersc.sh
```

All single-case scripts write scheduler output and FBPIC artifacts under
`$PSCRATCH`; update their allocation account, QoS, wall time, and run labels as
needed for the target NERSC environment. They are direct `srun` launches and do
not use the production libEnsemble allocation format.

### Continuation

Set `continuation: true` in a stage to resume an interrupted campaign from its
existing `run_root`. The driver loads successful records with the same stage and
campaign fingerprint, then dispatches only the unspent portion of
`max_evaluations`. A continuation stage may still define its own
`initial_design_size`: that one-time, stage-local seed design runs before
adaptive proposals and is offset by already completed runs in the same stage.
This is useful when a continuation activates new controls, since prior-stage data
cover those controls only at their nominal values. Initial-design points bypass
the feasibility gate; subsequent adaptive proposals apply it. Failed and timed-out
evaluations consume budget but are excluded from GP training. Keep all numerical,
analysis, template, and parameter settings unchanged when resuming, since changing
them changes the fingerprint and prevents continuation in that run root.

With `continuation: false`, the libEnsemble driver refuses to start when `run_root`
already contains any `sim_*` run directory. This prevents different campaigns from
accidentally sharing output. Local `evaluate` calls may be repeated; each still
refuses to overwrite an existing run directory. The provided `stage_2` uses
`continuation: true`, which permits it to reuse the Stage 1 run root and data.
`stage_2_test` demonstrates isolated continuation instead: its `history_root`
contains copied scalar history for model training while its stage-specific
`run_root` receives only test artifacts.

Stages cannot be run in reverse: successful records from a later stage that vary a
parameter inactive in the requested stage cause the driver to fail before it starts
libEnsemble, regardless of that stage's `continuation` setting.

For the production driver, configure libEnsemble with:

- `exploration_fbpic.libensemble_adapter.run_fbpic_simulation` as `sim_f`;
- an `FBPICRunExecutor`, stage name, and active parameter names in `sim_specs['user']`;
- Xopt's `BayesianExplorationGenerator` as the persistent generator;
- a scalar `status_code` and your chosen descriptor feature fields in `sim_specs['out']`.

The adapter is intentionally small because site-specific libEnsemble resource
configuration depends on requested nodes, ranks per FBPIC case, and Perlmutter
binding policy. Each worker should call the executor only after libEnsemble has
allocated its assigned GPUs.

## Retaining and reusing data

Keep raw diagnostics while the output feature set is evolving. Before deleting large
HDF5 files, re-extract the agreed scalar schema across all runs and retain
`campaign_runs.parquet`, per-run manifests, `result.json`, and selected plots. Keep
raw audit cases: nominal, successful/extreme cases, failures, and random samples.

To expand to `stage_2`, use the same campaign configuration and full table. The
Xopt bridge loads successful prior records using the stage's active columns; older
points remain valid training data at their fixed nominal values.

## Tests

```bash
pytest -q
```
