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

## Stage 1

Stage 1 varies eight physically important controls: plasma density, dopant fraction,
density length/position, laser energy, duration, spot size, and focal position. It
retains all 29 physical inputs in each manifest. The suggested first batch is nominal
plus low/high anchors through that point and a Sobol fill:

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_1 \
  --features total_beam_charge_pc mean_uz cov_x_x cov_y_y cov_uz_uz \
  initial-design --count 24 --seed 0 --output initial_design.json
```

The existing FBPIC utilities reduce each final diagnostic to the fixed 33-scalar
moment descriptor plus optional skewness and excess kurtosis. Each completed run also
gets `result.json`, stdout/stderr logs, and an append-only `campaign_runs.parquet`
record under `data/ionization_injection_exploration/`.

## Local smoke evaluation

After validating the bounds and environment, run one point:

```bash
explore-fbpic runs/initial_study/campaign.yaml --stage stage_1 \
  evaluate --point '{"laser_energy_J": 2.5}'
```

This creates an isolated run directory, writes the full physical vector to
`run_manifest.json`, launches `run_fbpic.py`, and extracts the final particle
diagnostic. It is expected to consume normal FBPIC resources.

## NERSC/libEnsemble

Use a single Slurm allocation and let libEnsemble assign one GPU or GPU/MPI team per
FBPIC evaluation. [runs/initial_study/submit_nersc.sh](runs/initial_study/submit_nersc.sh)
provides allocation settings and a safe initial-design command.

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
