# lume-fbpic

A [LUME](https://github.com/lume-science) model for fbpic plasma-accelerator simulations, wrapping
[`inversion_fbpic`](../lpa/Simulation_FBPIC): laser-driven (LWFA), beam-driven (PWFA), or a beam
through external fields. With it you can:

- **run** an fbpic simulation through `get()` / `set()`, with the config's fields as inputs and the
  run's statistics, moment descriptor and final particles as outputs;
- **archive** a model with its config, actions and results, and **load** it back.

## Install

```bash
pip install -e ../lpa/Simulation_FBPIC      # inversion_fbpic, which this package wraps
pip install -e .                            # lume_fbpic
pip install -e .[dev]                       # also pytest and pytest-mock
```

## Quick start

An `FBPICSimulator` holds the config (hyperparameters, and lists of lasers, density profiles,
particle beams and external fields); the scripts in `docs/examples/` build one for each case.

```python
from lume_fbpic.actions import FinalParticlesAction, LaserFieldAction, make_stat_actions
from lume_fbpic.model import LUMEFBPICModel

simulator = ...   # an FBPICSimulator; docs/examples/lwfa.py builds one for a laser-driven run
actions = [   # the model's inputs and outputs; docs/examples build the lists for each case
    LaserFieldAction(name="laser_energy", field_name="energy"),
    *make_stat_actions(),                 # charge_pc, energy_mean_mev, energy_std_mev
    FinalParticlesAction(name="final_particles", read_only=True),
]
model = LUMEFBPICModel(simulator, actions)      # fills in the units (J, pC, MeV) of the actions
model.set({"laser_energy": 2.5})          # applies the value, then runs fbpic
model.set({})                             # runs fbpic again, changing nothing
model.get(["charge_pc", "energy_mean_mev"])

model.archive("run.h5", save_final_particles=True)
model = LUMEFBPICModel.from_archive("run.h5")      # actions come back with it
```

- `set()` runs the simulation. `LUMEFBPICModel(simulator, actions, dummy_run=True)` (or
  `from_archive(..., dummy_run=True)`) makes it only update parameters.
- The **actions** are the model's inputs and outputs: laser and density fields, Zernike
  coefficients, the dopant fraction, particle-beam fields and charge (`make_beam_actions()`), the
  components of a uniform external field, run statistics, the 33-scalar moment descriptor and the final
  particles.
- An action's `unit` is left out and the model fills it in: for an action on a config field, from
  the unit tagged in brackets in that field's documentation (`[J]`, `[m]`, `[m^-3]`, ...); for a
  statistic, from `lume_fbpic.units.STAT_UNITS`; for a moment-descriptor feature, from
  `lume_fbpic.descriptor.features()`. A `unit` given to the action, `None` included, is kept.
- An archive (`model.archive()` and `LUMEFBPICModel.from_archive()`, lume-base's names; the
  format is `lume_fbpic.archive`'s) holds the config (as native HDF5 groups, written by
  `SerializableConfig.to_hdf5()`), the actions, the output values and optionally the final
  particles (as a `ParticleGroup`). It holds no paths. Output directories are decided when a run
  starts, and an input data file a config reads (such as a gas-jet density table) is recorded by
  basename and md5: pass `input_dirs` to `archive()` and `from_archive()` to say where those
  files are. A file that is not in `input_dirs` when archiving is embedded in the archive (with a
  warning over 10 MB), and a loaded archive uses the embedded copy. A referenced file missing
  from `input_dirs` raises `FileNotFoundError`, and one with a different md5 gives a warning.

## Layout

| Path | Purpose |
|------|---------|
| `lume_fbpic/model.py` | `LUMEFBPICModel`: the LUME model |
| `lume_fbpic/simulator.py` | `FBPICSimulator`: wraps `inversion_fbpic`'s `Simulation` (lasers, density profiles, particle beams, external fields); the lifecycle (`configure()`, `run()`, `reset()`), the results, and the input files of a loaded archive |
| `lume_fbpic/archive.py` | Archives of a model, behind `LUMEFBPICModel.archive()` and `from_archive()`: `save()`, `load()`, and the `fingerprint()` of the config, which with the actions names one by default |
| `lume_fbpic/archive_config.py` | How an archive stores the config: native HDF5 with no paths, input data files by basename and md5 |
| `lume_fbpic/actions.py` | The input and output actions; `make_beam_actions()`, `make_descriptor_actions()` and `make_stat_actions()` |
| `lume_fbpic/descriptor.py` | The moment descriptor of a bunch as `build_dataset.py` computes it: the selection, the features, their names and units |
| `lume_fbpic/units.py` | The default units of the actions: `STAT_UNITS` and the config-docstring unit tags |
| `lume_fbpic/utils.py` | fbpic's openPMD output to a `ParticleGroup` and back (`particle_group()`, `normalized_phase_space()`), the run statistics (`STATISTICS`) and `ELECTRON_MC2_EV` |
| `lume_fbpic/density_profiles.py` | Density profiles for the examples (`LinearRampFlattop`, `GeneralizedGaussianProfile`, `UpDownRampProfile`), and the atom density of a profile (`atom_density()`, `nominal_density()`) |
| `docs/examples/` | runnable examples: LWFA, two beam-driven PWFA setups (an illustrative matched-beam case, `pwfa_gaussian.py`, and CLARA FEBE energy doubling, which plot their dumps with `pwfa_dumps.py`), ionization injection, and a boosted-frame LWFA with an injected bunch (`lwfa_injected_beam.py`, fbpic's `boosted_frame_script.py`) |
| `docs/htu-twin/` | The HTU twin demo, with an archived run as the twin's source (own README and `pyproject.toml`) |
| `tests/` | Tests for this package; `downramp_actions.py` is the HTU downramp action set their fixtures use, and `archive_helpers.py`, `bunches.py` and `reference_descriptor.py` are helpers they share |

## Running the examples

Each example writes its output (fbpic's `diags/`, and any plots) into the current directory, so run
it from where you want that output:

```bash
examples=path/to/lume-fbpic/docs/examples
mkdir -p run && cd run
python $examples/pwfa_gaussian.py --preset quick
```

`pwfa_gaussian.py` and `pwfa_clara_febe.py` take options (`--help`); the others run as they are.
Each example runs on a GPU when there is one (`use_cuda`, True by default) and on the CPU otherwise.

### CPU cores

On the CPU, fbpic uses every core. To limit the cores a run uses, start it under `taskset`, with
`OMP_WAIT_POLICY=passive` so that idle OpenMP threads sleep instead of spinning:

```bash
OMP_WAIT_POLICY=passive taskset -c 0-7 python $examples/pwfa_gaussian.py   # cores 0 to 7
```

fbpic's thread pools size themselves to the cores `taskset` allows, so no thread counts need
setting.

The run times recorded in the examples' docstrings give the cores they were run on.

### HTU twin demo

`docs/htu-twin/` uses an archived run as the source of the HTU transport twin, serves archives and
the twin as EPICS PVs (via `lume-pva`), and has Phoebus displays for them. It is its own package,
`lume_fbpic_twin`, so that `lume_fbpic` needs neither the twin nor `lume-pva`; its README says how
to install and run it.

## Tests

```bash
pytest
```

The HTU twin demo's tests are skipped unless the twin is installed; `docs/htu-twin/README.md`
lists them.

## Relationship to `inversion_fbpic`

This package depends on `inversion_fbpic` (from the sibling
[`Simulation_FBPIC`](../lpa/Simulation_FBPIC) directory) as an external dependency and does not
modify or vendor that code.
