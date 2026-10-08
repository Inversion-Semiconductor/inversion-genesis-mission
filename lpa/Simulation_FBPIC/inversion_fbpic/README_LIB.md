# `inversion_fbpic.lib` Guide

`inversion_fbpic.lib` is a configuration-oriented wrapper around FBPIC. It turns a laser-plasma interaction into validated, serializable components, then constructs and runs the corresponding FBPIC simulation. It centralizes the physical and operational choices that must remain consistent through three main classes:

- `Simulation` (paired with `SimulationHyperparameters`)
  - Uses constituent `_DensityProfile`s and `_LaserPulse`s, as well as the hyperparameters, to drive the base FBPIC solver
  - Many of the checks/helpers perform tasks that users of base FBPIC typically do manually
- `_DensityProfile` (with concrete implementations)
  - Contributes a collection of methods and fields that are common among implementations
  - Retains information, such as longitudinal extent and nominal plasma density, which `Simulation` uses to prepare an FBPIC run with minimal manual calculation
- `_LaserPulse` (also with concrete implementations)
  - Similar contributions as `_DensityProfile`
  - Makes circular/elliptical polarization trivial without LASY
  - Takes in total energy or amplitude $a_0$, derives the other for ease-of-use

## `lib` Code Map

- `serializable_config.py`: configuration protocol and registry management.
- [datapoint.py](lib/datapoint.py): shared dictionary-backed datapoints and curated input parameters.
- [diagnostics.py](lib/diagnostics.py): the abstract post-analysis lifecycle and runtime completion tracking.
- [config_container.py](lib/config_container.py): ordered collections of heterogeneous configurations.
- `density_core.py`, `density_profiles.py`, and `density_modifiers.py`: common plasma behavior, public shapes, and composition.
- `laser.py`: laser validation and FBPIC profile construction; `LasyLaserPulse` remains unimplemented.
- `simulation.py`: FBPIC assembly, execution, diagnostics, and completion hashing.
- `commented_yaml.py`: comment-aware YAML I/O.
- `_density_implementations/` and `_doc_management/`: numerical profile internals and generated editor-facing documentation.

# Usage

## Assemble And Run A Simulation

Build one `SimulationHyperparameters`, one or more density profiles, and one or more laser pulses, then pass them to `Simulation`. The wrapper validates this assembly before it constructs FBPIC.

```python
from pathlib import Path

from inversion_fbpic.lib.density_profiles import ExampleDensityProfile
from inversion_fbpic.lib.laser import GaussianLaserPulse
from inversion_fbpic.lib.simulation import Simulation, SimulationHyperparameters

hyperparameters = SimulationHyperparameters(
    zmin=-200e-6,
    zmax=0.0,
    rmax=200e-6,
    nz=1024,
    nr=300,
    nm=2,
    use_mpi=False,
    number_dumps=2,
)
density = ExampleDensityProfile(
    nominal_density=1.0e24,
    p_nz=2,
    p_nr=2,
    p_nt=4,
    length=1.0e-3,
)
laser = GaussianLaserPulse(
    energy=5.0,
    z0=-50e-6,
    wavelength=8.0e-7,
    tau_fwhm=3.8e-14,
    cep=0.0,
    waist=2.8e-5,
    focal_position=3.0e-3,
    polarization=0.0,
)

simulation = Simulation(elements=[hyperparameters, density, laser])
simulation.setup_simulation(working_directory=Path("runs/example"))
simulation.run_simulation()
```

`setup_simulation()` resolves runtime paths, creates the FBPIC object, adds particle species and lasers, configures the moving window, diagnostics, checkpoints, and restart behavior. `run_simulation()` advances the derived interaction time. Setup and execution have filesystem side effects; the wrapper writes only from rank zero when MPI is enabled.

## Choose Physical And Runtime Parameters

- `SimulationHyperparameters` controls the FBPIC grid, timestep, boundaries, boosted-frame settings, moving window, diagnostics, restart behavior, and random seed. Values use SI units. `nz` is checked for FFT-friendly factorization; `save_directory` is relative to `working_directory` unless absolute.
- Density profiles represent relative $n(z, r)$; `nominal_density` supplies the physical scale in $m^{-3}$. `get_z_extent()` contributes to the interaction length, and the `Simulation` instance will use this to determine the full interaction length. `get_r_extent()` or `p_rmax` bounds radial plasma loading.
- `species` and `ionization` select the ion model. Ionization `0` starts neutral; a negative value means fully ionized. Electron and ion diagnostic names and selection maps determine recorded particle populations. In laboratory-frame runs, diagnostic names must be unique across density components.
- Laser pulses require exactly one of energy or $a_0$; the wrapper derives and records the other. Concrete pulse types specify the envelope and optical parameters, then construct the FBPIC laser profile.
- Use density `plot()`, `plot_z_profile()`, and `plot_r_profile()` to inspect or record the plasma profile before launching a run. The standard plots report electron density by default.

## Use Configuration Files

Objects can be assembled in Python and serialized to record a run, or loaded from mappings, YAML/JSON text, individual files, or a directory of component files. Directory loading considers `.yaml`, `.json`, and compatible component types. Nested references resolve relative to their containing configuration file, keeping a run directory portable.

All configurations use a tagged representation:

```yaml
config_type: density_profile
subclass: sine_squared_bump
parameters:
  nominal_density: 1.0e24
  length: 5.0e-6
  p_nz: 1
  p_nr: 1
  p_nt: 1
```

`config_type` selects the component domain, `subclass` selects its model, and `parameters` contains user inputs rather than derived state. `commented_yaml.py` preserves user comments before PyYAML parsing and restores them, or injects docstring-derived descriptions, on output. Files therefore remain both machine-loadable and practical to inspect between runs.

## Store Curated Inputs And Manage Multiple Configs

`_Datapoint` is the shared dictionary-backed configuration domain. `Parameters`
stores curated inputs, while `_Diagnostic` provides an abstract lifecycle for
future post-analysis datapoints. Neither base has a concrete serialization tag;
use `Parameters` or an application-defined concrete diagnostic.

```python
from inversion_fbpic.lib.config_container import ConfigContainer
from inversion_fbpic.lib.datapoint import Parameters
from inversion_fbpic.lib.serializable_config import SerializableConfig

inputs = {"laser_energy": 5.0, "nominal_density": 1.0e24}
parameters = Parameters(data=inputs)
configs = ConfigContainer(configs=[parameters])

inputs["laser_energy"] = 6.0
assert parameters.data["laser_energy"] == 6.0
assert configs.configs[0] is parameters

configs.to_yaml_file("configs.yaml")
loaded = SerializableConfig.from_file("configs.yaml")
```

`Parameters(data=...)` requires a dictionary for Python construction and retains
that exact object. Read and update values through `data`; no schema, copying,
read-only wrapper, or automatic loading of dictionaries inside it is applied.
It uses `config_type: datapoint`, `subclass: parameters`, and stores the dictionary
under `parameters.data`. An omitted serialized `data` field uses the existing
empty-dictionary loading fallback. NumPy values, tuples, complex numbers, and
paths use the standard serializer's built-in conversions, not a lossless codec
for arbitrary Python objects.

`ConfigContainer(configs=[...])` accepts an ordered list of live configurations,
tagged payload dictionaries, YAML/JSON strings, or individual config file paths.
Entries are resolved using `SerializableConfig.from_any()`, including relative
paths in a containing file or a `resolving_paths_relative_to()` context. After
construction, manage live configuration objects through `configs`.

Containers retain object references, order, and duplicates, and may hold nested
containers without flattening. Saving embeds all resolved configs inline, even
those originally loaded from files. Reloading reconstructs separate objects;
shared identity between repeated entries is not preserved. Containers do not
scan directories or run/inspect their contents. Datapoints and containers are not
responsible for launching a simulation themselves.

`Simulation.elements` accepts `ConfigContainer`, `Parameters`, and concrete
`_Diagnostic` alongside the required hyperparameters, densities, and lasers.
Nested containers are traversed depth-first in their declared order. The
simulation retains supporting objects as `simulation.parameters` and
`simulation.diagnostics`; parameter dictionaries remain metadata and are not
interpreted as additional components. Required physical components must still
be supplied, and duplicate hyperparameters remain invalid.

```python
simulation = Simulation(
  elements=ConfigContainer(
    configs=[hyperparameters, density, laser, Parameters(data=inputs)]
  )
)
```

Inline element payloads are resolved to live objects, so later parameter edits
and analysis results are included when saving the simulation. Elements supplied
as file paths retain their existing reference behavior. The configuration hash
still describes only the physical hyperparameters, densities, and lasers, not
attached metadata, analysis results, or container nesting.

## Define A Post-Analysis _Datapoint

Import `_Diagnostic` from `inversion_fbpic.lib.diagnostics`, subclass it, and
decorate the concrete class with
`@attrs.define(kw_only=True, slots=False)`, declare a unique `SUBCLASS` tag, and
implement `_analyze(self) -> dict[str, Any]`. Declare any analysis inputs as
normal init fields so they can be saved and reloaded. Import concrete classes
before loading their tagged configurations to register them.

Set the class attribute `RUN_BEFORE_SIMULATION: ClassVar[bool]` in the concrete
diagnostic. `True` schedules `analyze()` at the end of `setup_simulation()`;
`False` (the inherited default) schedules it after stepping at the end of
`run_simulation()`, before recording the completion hash. This is a class policy,
not a constructor argument or serialized parameter. Attach diagnostic instances
directly to `elements` or include them in nested containers.

Automatic analysis runs on the simulation's write rank (rank zero with MPI).
Within a phase it follows depth-first element order. Repeated references are not
deduplicated. Setup skipped because it is already initialized or hashed, and
runs skipped because they are hashed, do not invoke analysis. Forced runs with
`skip_if_hashed=False` invoke the appropriate phase normally. A pre-analysis
failure leaves `is_setup` false; a post-analysis failure prevents recording the
completion hash. Errors propagate to the caller. Analysis implementations use
their own declared inputs; no simulation argument is passed to `_analyze()`.

- `_analyze()` returns the desired result dictionary; `analyze()` assigns it to
  `data` without copying and marks the read-only `analysis_complete` property
  true only after success.
- Every `analyze()` call reruns the implementation. An exception or non-dict
  return leaves completion false and does not replace the previous data.
  Subclass mutations and external side effects are not rolled back.
- Construction, loading, and serialization never run analysis automatically.
  Serialization before successful analysis logs a warning through the diagnostics
  module logger and still saves the current inputs and data. This applies to
  dictionary, JSON, YAML, file output, and diagnostics nested inside containers.
- Completion is runtime-only and excluded from saved payloads. Reloaded
  diagnostics retain saved results but start incomplete, so serialization warns
  again until `analyze()` succeeds in that instance. Example templates do not
  warn because they do not serialize an instance.

No concrete beam-charge or field-analysis diagnostic is supplied yet.

## Demos

The [demos](../demos) directory contains runnable examples. Start with the core wrapper demos; the `miscellaneous/` entries are related density-modeling workflows rather than minimal simulation templates.

- [Building blocks](../demos/demo_building_blocks/README.md): generates a library of YAML components, selects active components in a directory, and runs `Simulation(elements=cfg_active)`.
- [Density profiles](../demos/demo_densities): generates and plots YAML configurations for some concrete density-profile types.
- [Laser pulses](../demos/demo_lasers): generates and plots Gaussian-laser YAML configurations for supported polarization variants.
- [Downramp simulation](../demos/demo_downramp_simulation/README.md): script-assembled hydrogen flattop/downramp LPA simulation, with recorded configuration, diagnostics, and density movie output.
- [Ionization simulation](../demos/demo_ionization_simulation/README.md): helium plasma with nitrogen doping, showing species-specific macroparticle settings and ionization injection.

## Skipping Runs if Complete

The wrapper hashes normalized hyperparameters, density profiles, and laser pulses. If the diagnostics directory contains the same hash, `setup_simulation()` and `run_simulation()` skip the matching completed configuration; component order does not affect the hash. The marker records configuration identity, not simulation-output validation.

# Architecture

## Component Model

The public configuration classes inherit shared domain behavior rather than repeating FBPIC setup logic:

- `SerializableConfig` provides JSON/YAML I/O, relative-path handling, example generation, and tagged type dispatch.
- `_Datapoint` shares dictionary-backed data storage; `Parameters` retains curated inputs and `_Diagnostic` wraps explicit analysis with runtime completion tracking.
- `ConfigContainer` resolves and retains an ordered collection of configs, then serializes its contents inline.
- `_DensityProfile` supplies particle-loading settings, species and ionization handling, diagnostic selections, plotting, plasma-wavelength calculation, and FBPIC species creation. Concrete profiles only define spatial shape and extent.
- `_DensityModifier` transforms a density function. `ModifiedDensityProfile` applies modifiers in order while inheriting loading and species settings from its base profile.
- `_LaserPulse` enforces the energy/$a_0$ contract, derives the companion value, and defines the interface for physical extents and FBPIC profile construction.
- `Simulation` owns component assembly and translates the configuration model into FBPIC calls.

It also resolves nested containers, retains `Parameters` as metadata, and runs
attached diagnostics at the selected lifecycle phase.

## Policies Embedded In `Simulation`

This wrapper incorporates simulation policy as well as object wiring, things that are typically done manually using base FBPIC. It derives interaction length from the union of density-profile extents plus `right_buffer`; when `beta_window` is unset, it estimates the moving-window velocity from the first laser wavelength and summed on-axis plasma density; and it configures a boosted Galilean frame for all relevant components when `gamma_boost` is active.

## Registry And Serialization

`SerializableConfig` maintains a two-level registry: `config_type` maps to a domain base class, then `subclass` maps to a concrete class in that domain. Importing `inversion_fbpic.lib` imports the public domains and registers their concrete types, so `from_dict()`, `from_yaml()`, and `from_file()` reconstruct components without caller-specific dispatch code.

The type tags and serialized parameter names are part of the persisted run interface. Unknown, missing, or duplicate tags fail early; changing them can make existing configurations unloadable. Derived `attrs` fields are excluded from serialization, while input fields are retained for reconstruction.

_Datapoint `data` is deliberately an init field so curated inputs and completed
analysis results can both be saved. _Diagnostic completion status is a non-init
runtime field and is not included.

## Internal Maintenance Layers

`_density_implementations/` isolates the numerical mechanics behind public profiles, including conical targets (WIP) and HDF5 interpolation. This keeps `density_profiles.py` a stable catalogue for simulation assembly while model-specific code handles interpolation, composition, and validation.

`_doc_management/` parses the `attrs` configuration hierarchy without importing it, merges inherited docstrings and `Args:` entries, and generates `.pyi` stubs with explicit keyword-only constructors. Pylance can therefore display inherited required parameters and documentation alongside subclass fields. Run `tools/sync_config_docstrings.py --check` to detect stub/documentation drift.

## Testing

`tests/test_lib/test_serializable_config.py`, `test_config_yaml.py`, and `test_config_yaml_limitations.py` cover tagged loading, path resolution, YAML comments, and serialization. `test_density.py`, `test_density_profiles.py`, and `test_laser.py` cover model validation and physical-profile behavior. `test_simulation.py` covers assembly, derived grid values, diagnostic periods, hashing, skip behavior, and a lightweight setup-and-step integration path. `tests/test_doc_management/` covers MRO documentation merging and generated stubs.

[test_datapoint.py](../tests/test_lib/test_datapoint.py) covers dictionary identity,
result storage, analysis completion, and serialization warnings.
[test_config_container.py](../tests/test_lib/test_config_container.py) covers
mixed sources, ordered and nested containers, inline serialization, and relative
file references.
[test_simulation.py](../tests/test_lib/test_simulation.py) additionally covers
supporting elements, lifecycle scheduling, skipped and failed analysis, and MPI
write-rank behavior using a mocked FBPIC backend.