# The lume-fbpic schema

How an fbpic simulation is described in `lume-fbpic`: the config that defines the run, the actions
that make it a LUME model, and the archive that stores both with the results. The last section is
a recipe for transcribing an fbpic script.

The schema is defined by code; this page summarizes it. The field-by-field reference is in the
docstrings of the `inversion_fbpic` config classes (`lpa/Simulation_FBPIC/inversion_fbpic/lib/`),
and of `lume_fbpic.actions` and `lume_fbpic.archive`.

## 1. The config

An `FBPICSimulator` (`lume_fbpic/simulator.py`) holds five config parts, all `inversion_fbpic`
`SerializableConfig` objects, frozen (`attrs`), with units in brackets in their field docstrings:

| Part | Type | Classes |
|---|---|---|
| `hyparams` | one | `SimulationHyperparameters`: grid (`zmin`, `zmax`, `nz`, `rmax`, `nr`, `nm`), boost (`gamma_boost`), moving window (`beta_window`), run length (`simulation_length`, `right_buffer`), diagnostics (`number_dumps`, `field_diagnostics`, `write_period`, `save_directory`), `use_cuda`, `use_mpi`, `random_seed`, `r_boundary` |
| `lasers` | list | `GaussianLaserPulse`, `LasyLaserPulse` (with Zernike aberrations) |
| `densities` | list | `inversion_fbpic`: `SmoothSineFlattop`, `PowerLawFlattop`, `BilateralSuperGaussian`, `AsymmetricSine`, `GaussianPlusTriangle`, `GeneralizedGaussianPlusTriangle`, `GeneralizedLorentzianSum`, `GenericConicalTarget`, `InterpolateFromH5Profile` (a gas-jet table), `ExpressionDensityProfile` (a formula), `ModifiedDensityProfile` (with `MatchedRadialModifier`); `lume_fbpic.density_profiles`: `LinearRampFlattop`, `GeneralizedGaussianProfile`, `UpDownRampProfile` |
| `beams` | list | `GaussianParticleBeam`, `ExpressionParticleBeam` (a formula), `FileParticleBeam`, `OpenPMDParticleBeam` |
| `external_fields` | list | `UniformExternalField`, `ExpressionExternalField` |

Besides the config, the simulator takes `target_species`, the electron species its outputs
describe (a density profile's `elec_name` or a beam's `particle_name`), and an optional
`working_directory`, where fbpic writes `diags/`.

`configure()`, which every run calls first, requires `hyparams`; a laser or a beam; a density
profile or `hyparams.simulation_length`; and a `target_species` that the config writes.

### How `inversion_fbpic`'s `Simulation` fills in the rest

These are not in the config, so a transcription has to match them:

- **Run length:** the window moves `simulation_length` (or, if it is None, the density profiles'
  combined z extent) plus `right_buffer` (if None, the window length `zmax - zmin`).
- **Moving window:** `beta_window`, if None, is the fastest of each laser's mean group velocity in
  the plasma and each beam's `sqrt(1 - 1/gamma0^2)`. Set it to reproduce a script's `v_window`.
- **Time step:** `dz / c` in the lab frame; `min(rmax / (2 gamma_boost nr), dz) / c` boosted.
  There is no setting for it.
- **Boost:** with `gamma_boost`, the Galilean frame, the boosted density positions and the
  back-transformed lab-frame dumps are set up automatically.
- **Plasma species:** a profile with `species` (for example `"He"`) adds the ions as well as the
  electrons, ionizable from `ionization` up; `species=None` is a bare-electron plasma.
  `nominal_density` is the fully-ionized plasma electron number density, so for a neutral gas of
  atom density `n`, set `nominal_density = n * levels`: `nominal_density()` in
  `lume_fbpic.density_profiles` does this, and `atom_density()` is its inverse.

## 2. The actions

A `LUMEFBPICModel(simulator, actions)` (`lume_fbpic/model.py`) is the LUME model. Its variables are
its actions (`lume_fbpic/actions.py`): writable ones change the config, read-only ones read a run's
results. `set()` applies the values and runs the simulation (`dummy_run=True` only applies them);
`get()` reads; `reset()` returns to the config and results the model started with.

**Inputs** (writable). Each replaces the frozen config object it changes:

| Action | Sets | Unit |
|---|---|---|
| `HyperparameterFieldAction(field_name)` | a field of `hyparams` | from the field's docstring |
| `LaserFieldAction(field_name, laser_index=0)` | a field of a laser; `energy` and `a0` stay consistent | from the field's docstring |
| `ZernikeCoefficientAction(coefficient)` | one Zernike coefficient of a `LasyLaserPulse` | rad |
| `DensityFieldAction(density_index, field_name)` | a field of a density profile | from the field's docstring |
| `DopantFractionAction(host_index, dopant_index, dopant_atoms_per_molecule=1, host_atoms_per_molecule=1)` | a two-gas mixture's dopant fraction, keeping the neutral density | 1 |
| `BeamFieldAction(beam_index, field_name)` | a field of a beam | from the field's docstring |
| `BeamTotalChargeAction(beam_index)` | a beam's charge, through `n_physical_particles` | C |
| `ExternalFieldAction(field_index, field_type)` | one component of a `UniformExternalField` | V/m or T |

`make_beam_actions(simulator, beam_index=0)` makes a beam's set, named after its `particle_name`
(`witness_gamma`, `witness_sigma_r`, `witness_charge`, ...).

**Outputs** (read-only), from the last dump of `target_species`:

| Action | Value | Unit |
|---|---|---|
| `make_stat_actions()`: `charge_pc`, `energy_mean_mev`, `energy_std_mev` | the run statistics | pC, MeV |
| `make_descriptor_actions(uz_min=30, central_fraction=0.95)` | the 33-value moment descriptor `build_dataset.py` writes (`lpa/Simulation_FBPIC/inversion_fbpic/scripts/data_parsing/`): 3 momentum centroids, 21 covariances, 8 longitudinal `uz` features, the charge | per feature |
| `FinalParticlesAction` | the particles, as an openPMD-beamphysics `ParticleGroup` | |

Outputs are NaN (or None) before a run. An action's `unit`, when left out, is filled in by the
model as the tables say; a unit given explicitly is kept.

## 3. The archive

`model.archive(path)` writes, and `LUMEFBPICModel.from_archive(path)` reads back, one HDF5 file (or
a group of one), layout version 3 (`lume_fbpic/archive.py`):

| Part | Contents |
|---|---|
| root attributes | `dataType` and `software` (`lume-fbpic`), `format_version` (3), `config_kind` (`fbpic`), `target_species` |
| `hyparams`, `lasers`, `densities`, `beams`, `external_fields` | the config as native HDF5 (`lume_fbpic/archive_config.py`); a list is a group with the attribute `list` and one numbered group per object |
| `actions` | one numbered group per action: its class (module and name), its parameters, and for a read-only scalar its value when saved |
| `inputs` | the input files the config reads (a gas-jet table, for example), by basename and md5; embedded when not found in `input_dirs` at save |
| `parameters` | optional metadata, such as a dataset record |
| `final_particles`, `stats` | with `save_final_particles=True`: the particles (openPMD, so the file opens with `ParticleGroup(h5=...)`) and the run statistics |

An archive holds no paths: output paths are stored by basename, and input files by basename and
md5, found again through `input_dirs` or the embedded copy. fbpic's own `diags/` are not archived.
`save()`'s default name hashes `archive.fingerprint(simulator)` (the config, the target species
and the input files' md5s) with the actions, so the same model gets the same name.

## 4. Transcribing an fbpic script

1. **Grid and run:** copy the script's box, cell counts and modes into `SimulationHyperparameters`.
   Set `gamma_boost` if the script boosts. Match its interaction length with `simulation_length`
   and `right_buffer`, and its `v_window` with `beta_window` (see "How `Simulation` fills in the
   rest"). A custom `dt` cannot be reproduced.
2. **Laser:** a `GaussianLaserPulse` for fbpic's `GaussianLaser` (convert its `tau` to the FWHM
   `tau_fwhm` with `inversion_fbpic.utils.simulation_setup_tools`), or a `LasyLaserPulse`.
3. **Plasma:** a named profile class if one has the shape, or an `ExpressionDensityProfile` with
   the script's `dens_func` written as a formula (and its z extent). Choose `species` and
   `ionization` for the ions, and `nominal_density` as above.
4. **Beams:** `GaussianParticleBeam` for `add_particle_bunch_gaussian`, `ExpressionParticleBeam`
   for `add_particle_bunch`, the file and openPMD beams for the others.
5. **Diagnostics:** `number_dumps`, `field_diagnostics`, and a `particle_select` (beam) or
   `elec_select` (plasma) for a script's `select`.
6. **Model:** an `FBPICSimulator` with a `target_species`, and the actions to vary and read.
7. **Check:** run both, small, and compare a dump.

The examples in `docs/examples/` are worked transcriptions:

| Example | Transcribes | Shows |
|---|---|---|
| `lwfa.py` | fbpic's `lwfa_script.py` | the simplest case; matching the run length and window |
| `lwfa_injected_beam.py` | fbpic's `boosted_frame_script.py` | a boost, a formula density with ions, an injected bunch; checked against the original in every lab-frame dump |
| `ionization_injection.py` | HTU's ionization-injection runscript (`runs/initial_sample`) | two gases, a dopant fraction, a LASY laser with Zernike actions |
| `pwfa_gaussian.py` | none: a matched beam-driven case | Gaussian beams and their actions |
| `pwfa_clara_febe.py` | Zhang et al., Plasma Phys. Control. Fusion 68, 045051 (2026) | a paper's tables as a config |

## Limits

- `ionization_injection.py` transcribes HTU's runscript. Its default grid is coarsened to run on a
  CPU, and it has not been run at the runscript's production grid.
- The seven HTU `initial_sample` runs are archives rebuilt from their recorded results
  (`docs/htu-twin/twin/initial_sample_archives.py`): config and outputs, no particles.
- A config file referenced by path (`ModifiedDensityProfile`'s base profile), or a directory input,
  cannot be archived.
