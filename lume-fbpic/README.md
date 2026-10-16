# lume-fbpic

A [LUME](https://github.com/lume-science) model for fbpic plasma-accelerator simulations: laser-driven
(LWFA, wrapping [`inversion_fbpic`](../lpa/Simulation_FBPIC)) and beam-driven (PWFA, built directly
on fbpic), in the way [`lume-cheetah`](https://github.com/lume-science/lume-cheetah) wraps
Cheetah. With it you can:

- **run** an fbpic simulation through `get()` / `set()`;
- **archive** it, **load** it back, and **serve** its outputs as EPICS PVs (via `lume-pva`);
- use its bunch as the **source of the HTU transport twin**, with a selector
  that switches between several archived runs;
- turn the **existing NERSC runs** (`runs/initial_sample`) into archives that serve the same way.

## Install

```bash
pip install -e ../lpa/Simulation_FBPIC      # inversion_fbpic, which this package wraps
pip install -e .                            # lume_fbpic
pip install -e .[dev]                       # also pytest and pytest-mock
pip install -e docs/htu-twin[serve]         # lume_fbpic_twin, the HTU twin demo, with lume-pva
```

The HTU twin code (`TwinStage`, the archive selector, the PV server) is its own package,
`lume_fbpic_twin` in `docs/htu-twin`, so that `lume_fbpic` needs neither the twin (`htu`, torch,
Cheetah) nor `lume-pva`. Its `serve` extra installs the `lume-pva` commit that `geecs-lume-twin`
pins, so the two installs agree.

## Quick start

An `LWFASimulator` holds the config (hyperparameters, laser, density profiles); the scripts in
`docs/examples/` build one for each case.

```python
from lume_fbpic.actions import FinalParticlesAction, LaserFieldAction, make_stat_actions
from lume_fbpic.model import LUMEFBPICModel

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

- `set()` runs the simulation, as `lume-cheetah` and `lume-impact` do. `dummy_run=True` makes it
  only update parameters.
- The **actions** are the model's inputs and outputs: laser and density fields, Zernike
  coefficients, the dopant fraction, run statistics, the 33-scalar moment descriptor and the final
  particles.
- An action's `unit` is left out and the model fills it in: for an action on a config field, from
  the unit tagged in brackets in that field's documentation (`[J]`, `[m]`, `[m^-3]`, ...); for a
  statistic, from `lume_fbpic.units.STAT_UNITS`; for a moment-descriptor feature, from
  `lume_fbpic.descriptor.features()`. A `unit` given to the action, `None` included, is kept.
  The units are served as the PVs' display units.
- An archive (`model.archive()` and `LUMEFBPICModel.from_archive()`, lume-base's names; the
  format is `lume_fbpic.archive`'s) holds the config (as native HDF5 groups, written by
  `SerializableConfig.to_hdf5()`), the actions, the output values and optionally the final
  particles (as a `ParticleGroup`). It holds no paths. Output directories are decided when a run
  starts, and an input data file a config reads (such as a gas-jet density table) is recorded by
  basename and md5: pass `input_dirs` to `archive()` and `from_archive()` to say where those
  files are. A file that is not in `input_dirs` when archiving is embedded in the archive (with a
  warning over 10 MB), and a loaded archive uses the embedded copy. A referenced file missing
  from `input_dirs` raises `FileNotFoundError`, and one with a different md5 gives a warning.

To serve the outputs as PVs, or to feed the HTU twin, see `docs/htu-twin/README.md`.

## Layout

| Path | Purpose |
|------|---------|
| `lume_fbpic/model.py` | `LUMEFBPICModel`: the LUME model |
| `lume_fbpic/simulator.py` | `BaseSimulator`: the lifecycle (`configure()`, `run()`, `reset()`), the results, and the input files of a loaded archive |
| `lume_fbpic/lwfa_simulator.py` | `LWFASimulator`: LWFA, wraps `inversion_fbpic`'s `Simulation` |
| `lume_fbpic/pwfa_simulator.py` | `PWFASimulator`: beam-driven PWFA, built directly on fbpic |
| `lume_fbpic/archive.py` | Archives of a model, behind `LUMEFBPICModel.archive()` and `from_archive()`: `save()`, `load()`, and the `fingerprint()` that names one by default |
| `lume_fbpic/archive_config.py` | How an archive stores the config: native HDF5 with no paths, input data files by basename and md5 |
| `lume_fbpic/pwfa_config.py` | PWFA configs: `PWFAGrid` and the electron bunches (`FlatTopBunch`, `GaussianBunch`) |
| `lume_fbpic/actions.py` | The input and output actions; `make_descriptor_actions()`, `make_stat_actions()` and `make_pwfa_actions()` |
| `lume_fbpic/descriptor.py` | The moment descriptor of a bunch as `build_dataset.py` computes it: the selection, the features, their names and units |
| `lume_fbpic/units.py` | The default units of the actions: `STAT_UNITS` and the config-docstring unit tags |
| `lume_fbpic/utils.py` | fbpic's openPMD output to a `ParticleGroup` and back (`particle_group()`, `normalized_phase_space()`), the run statistics (`STATISTICS`) and `ELECTRON_MC2_EV` |
| `lume_fbpic/density_profiles.py` | Density profiles for the examples (`LinearRampFlattop`, `GeneralizedGaussianProfile`, `UpDownRampProfile`), and the atom density of a profile (`atom_density()`, `nominal_density()`) |
| `docs/examples/` | runnable examples: LWFA, two beam-driven PWFA setups (FACET-II type, and CLARA FEBE energy doubling, which plot their dumps with `pwfa_dumps.py`), ionization injection |
| `docs/htu-twin/` | The HTU twin demo, with the LPA archive as the twin's source (own README and `pyproject.toml`), in three parts: |
| &nbsp;&nbsp;`lume_fbpic_twin/` | The package: `twin_stage.py`, `TwinStage` and `build_chain()`, the HTU twin with the LPA bunch as its source, and its helpers (bunch moments, ballistic drift); `archive_selector.py`, `ArchiveSelector`, several archived runs behind one enum variable (`LPA_Archive`); `serve.py`, serve archives as EPICS PVs, optionally with the twin |
| &nbsp;&nbsp;`twin/` | `lpa_to_twin.py`: an example running the LPA-to-twin chain; `initial_sample_archives.py`: builds the archives (from the NERSC dataset) that the twin demo serves |
| &nbsp;&nbsp;`phoebus/` | Phoebus displays for the twin chain and the scripts that generate them (own README) |
| `tests/` | Tests for this package; `downramp_actions.py` is the HTU downramp action set their fixtures use, and `archive_helpers.py`, `bunches.py` and `reference_descriptor.py` are helpers they share |

Run the examples from this directory so their output lands in `diags/`.

## Tests

```bash
pytest
```

The twin tests need `htu` importable and `lume-pva` installed; without them they are skipped.

## Status and limits

- Archives reconstructed from `runs/initial_sample` hold the recorded outputs but no particles. To
  feed the twin, a bunch is built from the moment descriptor (`--synthesize-bunch`): an
  approximation, not the real bunch.
- Served values are recorded results; nothing re-runs the simulation when an input changes.
- `lume-pva` currently runs against `lume-base` 0.5.0, although it declares `>= 0.6.0`.

## Relationship to `inversion_fbpic`

This package depends on `inversion_fbpic` (from the sibling
[`Simulation_FBPIC`](../lpa/Simulation_FBPIC) directory) as an external dependency and does not
modify or vendor that code.
