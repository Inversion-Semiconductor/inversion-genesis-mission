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

## Campaign configuration

[runs/initial_study/campaign.yaml](runs/initial_study/campaign.yaml) is the source of
truth. It contains:

- every physical input accepted by `IonizationInjectionSimulation`;
- nominal values and reviewed, conservative bounds;
- active dimensions for each study stage;
- fixed numerical/execution settings, and descriptor extraction settings.

The configuration fingerprint is written to every `run_manifest.json`. Change bounds
or active stages as needed; do not change the frozen template, numerical settings,
or analysis settings within a campaign. Begin a new campaign when those must change.

### Adapting a new template

Most template updates require only a new campaign YAML: set `template_script`, update
the physical parameter names, nominal values, bounds, stages, and fixed execution
settings. The executor, campaign store, scalar extraction, and Slurm dispatch remain
unchanged when the template still accepts the physical and hyperparameter mappings,
provides a `.run()` method, and writes the same final openPMD particle diagnostic.

If the template moves to a configuration-based constructor or CLI, adapt
`FBPICRunExecutor._write_entrypoint` to write the template's per-run configuration
from the manifest and invoke its new run method. The rest of the exploration workflow
can remain unchanged as long as the diagnostic path, particle species, and openPMD
layout stay compatible. Update `extract_features` only when that output contract or
the desired scalar calculations change.

## Stage 1

Stage 1 varies eight physically important controls: plasma density, dopant fraction,
density length/position, laser energy, duration, spot size, and focal position. It
retains all 29 physical inputs in each manifest. The suggested first batch is nominal
plus low/high anchors through that point and a Sobol fill:

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_1 initial-design \
  --count 24 --seed 0 --output initial_design.json
```

The existing FBPIC utilities reduce each final diagnostic to the fixed 33-scalar
moment descriptor plus optional skewness and excess kurtosis. Each completed run also
gets `result.json`, stdout/stderr logs, and an append-only `campaign_runs.parquet`
record under `data/ionization_injection_exploration/`.

## Local smoke evaluation

After validating the bounds and environment, run one point:

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_1 evaluate \
  --point '{"laser_energy_J": 2.5}'
```

This creates an isolated run directory, writes the full physical vector to
`run_manifest.json`, launches `run_fbpic.py`, and extracts the final particle
diagnostic. It is expected to consume normal FBPIC resources.

## NERSC/libEnsemble

Use a single Slurm allocation and let libEnsemble assign one GPU or GPU/MPI team per
FBPIC evaluation. The persistent driver runs Xopt's Bayesian exploration generator
on the libEnsemble manager and uses resource-set workers to launch FBPIC. It starts
with the campaign's deterministic initial design, records the completed scalar
outputs, then asks Xopt for later candidates as GPU resource sets become available.

[runs/nersc_smoke/submit_nersc.sh](runs/nersc_smoke/submit_nersc.sh) is the
reference Perlmutter launch. It starts one manager and eight simulation workers;
the smoke campaign gives every worker a two-GPU resource set. Do not dispatch
background `explore-fbpic evaluate` processes from the batch script: libEnsemble's
`MPIExecutor` owns the FBPIC MPI launch and GPU binding.

Install the NERSC extra before running the driver:

```bash
python -m pip install --no-cache-dir libensemble
```

Then submit from the campaign directory:

```bash
sbatch submit_nersc.sh
```

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
