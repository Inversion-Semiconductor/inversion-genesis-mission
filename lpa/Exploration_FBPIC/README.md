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
campaign fingerprint into Xopt, skips the initial design, and dispatches only the
unspent portion of `max_evaluations`. Failed and timed-out evaluations consume
budget but are excluded from GP training. Keep all numerical, analysis, template,
and parameter settings unchanged when resuming, since changing them changes the
fingerprint and prevents continuation in that run root.

With `continuation: false`, the libEnsemble driver refuses to start when `run_root`
already contains any `sim_*` run directory. This prevents different campaigns from
accidentally sharing output. Local `evaluate` calls may be repeated; each still
refuses to overwrite an existing run directory. The provided `stage_2` uses
`continuation: true`, which permits it to reuse the Stage 1 run root and data.

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
