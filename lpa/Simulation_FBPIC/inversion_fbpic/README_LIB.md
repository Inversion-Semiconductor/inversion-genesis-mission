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
- `laser.py`: laser validation and FBPIC profile construction. `GaussianLaserPulse` builds analytic FBPIC profiles; `LasyLaserPulse` wraps `utils.laser.HighOrderLasyLaser` (super-Gaussian transverse profile with Zernike aberrations), writes a LASY HDF5 file on rank zero during `Simulation.setup_simulation()`, and emits it through FBPIC's laser antenna.
- `simulation.py`: FBPIC assembly, execution, diagnostics, and completion hashing.
- `commented_yaml.py`: comment-aware YAML I/O.
- `_density_implementations/`, `_laser_implementations/`, and `_doc_management/`: density profile internals, laser pulse implementations, and generated editor-facing documentation.

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
- Laser pulses require exactly one of energy or $a_0$; the wrapper derives and records the other. Concrete pulse types specify the envelope and optical parameters, then construct the FBPIC laser profile. `LasyLaserPulse` accepts energy only: its $a_0$ is measured numerically at focus when the LASY file is built and reported as `out_a0`, so it is `null` in configurations written before setup.
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

Resolution is eager: without a containing file or explicit context, relative
paths use the current working directory at container construction time.
`Simulation` consumes the resolved objects and does not rebase their paths.
For programmatic construction against an existing config directory, use:

```python
with SerializableConfig.resolving_paths_relative_to("cfg"):
  components = ConfigContainer(configs=["density.yaml", "laser.yaml"])
simulation = Simulation(elements=[hyperparameters, components])
```

If editing `configs` after construction, add live configuration objects rather
than unresolved path strings or payloads.

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
`run_simulation()`, after recording the successful-stepping hash. This is a class policy,
not a constructor argument or serialized parameter. Attach diagnostic instances
directly to `elements` or include them in nested containers.

Automatic analysis runs on the simulation's write rank (rank zero with MPI).
Within a phase it follows depth-first element order. Repeated references are not
deduplicated. A matching hash skips FBPIC setup and stepping but still reruns
the corresponding pre- and post-analysis phases. Repeating setup on an already
initialized, unhashed instance remains a no-op. Forced runs with
`skip_if_hashed=False` invoke the appropriate phase normally. A pre-analysis
failure leaves a new setup incomplete; a post-analysis failure preserves the
completed simulation hash, so retrying runs analysis without repeating stepping.
Errors propagate to the caller. Analysis implementations use
their own declared inputs; no simulation argument is passed to `_analyze()`.

Diagnostic elements are attached automatically when the simulation is assembled.
Implementations access their simulation through `attached_simulation`; callers
may also use `attach(simulation)` explicitly. Attachments are runtime-only and
excluded from JSON, YAML, and HDF5 configurations, avoiding circular serialized
references. A saved diagnostic reloads unattached, while loading a simulation
automatically attaches its diagnostic elements to the new simulation. An instance
cannot be attached to two different simulations; create a separate instance for
each simulation.

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

## Compute The Final Beam Moment Descriptor

`_ParticleDiagnostic` extends `_Diagnostic` with the shared `selection` field,
recording-name resolution, latest-output discovery, and particle/weight
concatenation. Its `load_particles()` method returns an `(N, 6)` particle array
and matching `(N,)` macro-weights without changing `data` or completion state.
Subclass it and implement `_analyze()` to reuse this input pipeline for another
particle analysis. The base remains abstract and has no concrete registry tag.

`MomentDescriptorDiagnostic` inherits this gathering behavior and stores the result of
`compute_moment_descriptor()` in `data`. It runs after stepping, produces 33
features with the default spline settings, and includes `total_beam_charge_pc`.

```python
import attrs

from inversion_fbpic.lib.diagnostics import MomentDescriptorDiagnostic

density = attrs.evolve(density, elec_name="electrons")
moments = MomentDescriptorDiagnostic(selection=("elec_name", "electrons"))
simulation = Simulation(elements=[hyperparameters, density, laser, moments])
simulation.setup_simulation(working_directory=Path("runs/example"))
simulation.run_simulation()
features = moments.data
```

The required `selection` is a Literal-tagged tuple with these supported forms:

```python
("elec_name", ["elec_name_1", "elec_name_2"])
("elec_name", "elec_name_1")
("ion_name", ["ion_name_1", "ion_name_2"])
("ion_name", "ion_name_1")
("all_of_species", "e")
("all_of_species", "He")
```

Explicit names must match the corresponding `elec_name` or `ion_name` fields
on the attached simulation's density profiles. `("all_of_species", "e")`
includes all recorded electrons, including bare-electron profiles.
`("all_of_species", "He")` includes recorded ions from all helium profiles;
use another atomic symbol to select another ion species. Profiles without a
recording name are excluded. Unknown names or no matching recordings raise
an error.

The selected particle arrays and macro-weights are concatenated before computing
one descriptor. Duplicate recording names are loaded only once, so they do not
double-count particles or charge. Outer tuples reload correctly from the list
representation used in JSON, YAML, and HDF5. This replaces the previous
single-string `species` constructor argument.

At each
analysis call, the diagnostic searches the resolved `save_directory/hdf5` and
selects the highest numerical iteration in a `data########.h5` filename, not the
newest modification time or lexical filename order. Output files need not exist
at construction time. Missing output or invalid beam data raises an error without
replacing prior results or marking analysis complete.

The constructor exposes `longitudinal_mode` (OFF=0, MOMENTS=1, SPLINE=2),
`longitudinal_bins`, `include_total_weight`, and `include_higher_moments` using
the function's defaults. All particles of the selected recordings are analyzed without
additional momentum selection or central cropping. `config_type` remains
`datapoint` and its concrete `subclass` tag is `moment_descriptor`. Call
`moments.analyze()` explicitly to recompute outside the simulation lifecycle;
cached `run_simulation()` calls already refresh results from existing output.


## Demos

The [demos](../demos) directory contains runnable examples. Start with the core wrapper demos; the `miscellaneous/` entries are related density-modeling workflows rather than minimal simulation templates.

- [Building blocks](../demos/demo_building_blocks/README.md): generates a library of YAML components, selects active components in a directory, and runs `Simulation(elements=cfg_active)`.
- [Density profiles](../demos/demo_densities): generates and plots YAML, JSON, or HDF5 configurations for some concrete density-profile types.
- [Laser pulses](../demos/demo_lasers): generates and plots Gaussian-laser YAML, JSON, or HDF5 configurations for supported polarization variants.
- [Downramp simulation](../demos/demo_downramp_simulation/README.md): script-assembled hydrogen flattop/downramp LPA simulation, with recorded configuration, diagnostics, and density movie output.
- [Ionization simulation](../demos/demo_ionization_simulation/README.md): helium plasma with nitrogen doping, showing species-specific macroparticle settings and ionization injection.
- [LASY laser simulation](../demos/demo_lasy_laser_simulation/README.md): hydrogen flattop driven by an aberrated super-Gaussian `LasyLaserPulse`, showing the rank-0 LASY build, antenna emission, and the numerically measured `a0`.

The density demo uses [create_density_configs.py](../demos/demo_densities/create_density_configs.py)
and [plot_density_configs.py](../demos/demo_densities/plot_density_configs.py);
the laser demo uses [create_laser_configs.py](../demos/demo_lasers/create_laser_configs.py)
and [plot_laser_configs.py](../demos/demo_lasers/plot_laser_configs.py).
All four scripts accept `--format json` or `--format hdf5` to write or read
JSON or native HDF5 configs; omit the option for the existing YAML workflow.
All formats can coexist in each demo's `cfg` directory. JSON plots go to
`plots/json` and HDF5 plots to `plots/hdf5`, leaving YAML plots unchanged.
Re-running a generator explicitly
replaces its existing HDF5 configurations while preserving unrelated data.

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

### Native HDF5 configurations

`config.to_hdf5_file(path)` writes a native configuration into `/config` in an
HDF5 file. `SerializableConfig.from_file(path)` and `from_any(path)` recognize
`.h5` and `.hdf5` extensions, case-insensitively. To select a different group,
use `to_hdf5_file(path, group_path="/metadata/config")` and
`from_hdf5_file(path, group_path="/metadata/config")`.

For embedding in an already-open file, pass an empty `h5py.Group` to
`config.to_hdf5(group)`. Read it with `SerializableConfig.from_hdf5(group)` or
`from_any(group)`. These methods never close caller-owned handles. File writers
open in append mode and preserve unrelated datasets and metadata. Replacing a
previously written configuration requires `overwrite=True`; unrelated occupied
groups cannot be replaced, even with that flag. Encoding is staged before any
existing configuration is replaced, so unsupported values leave it intact.

The versioned schema retains `config_type`, `subclass`, and `parameters` as native
datasets/groups, rather than an opaque JSON document. Mapping keys are escaped
when necessary. Tags distinguish mappings, ordered sequences, empty containers,
and `None`; homogeneous numeric lists use native array datasets. `None` is stored
as a tagged null dataset (no shape or value), with a float64 placeholder dtype
regardless of the optional parameter's type. Values follow the existing
`to_dict()` semantics: NumPy arrays and tuples become lists, complex
values become real/imaginary pairs, and YAML comments and original NumPy dtypes
are not preserved. `include_nones` and deserialization `overrides` behave as in
the text serializers. Filesystem-backed group handles also supply `source_file`
and the anchor for portable relative input-file paths.
Writes to relative-open handles also work without an anchor; paths retain their
existing representation (absolute input paths stay absolute) instead of guessing
the file's original directory. For portable relative paths, open the file with an
absolute filename or wrap writes in
`SerializableConfig.resolving_paths_relative_to(directory)`.
Reads from relative-open handles still require that context with the original
file directory, so later working-directory changes cannot silently select the
wrong input files.
All serialized configs and examples include `git_hash`, cached from [git_hash.txt](git_hash.txt) when `serializable_config` is imported. Later commits or builds do not change a running process's provenance; restart to capture a new revision. Serialization never calls Git or preserves an incoming config's hash.

Builds/installs and Git hooks update the Git-ignored file, which is bundled in distributions. **Existing clones must re-run `pre-commit install`** to add the new stages. Amend/rebase are covered by `post-rewrite`; `git reset` is not. After resets or for uninstalled source use, run [tools/record_git_hash.py](../../../tools/record_git_hash.py) (or rebuild/reinstall) before importing. Runtime does not validate Git HEAD.

Unavailable provenance warns once at import and stays `null`, even with `include_nones=False`. It identifies committed code, not local edits; legacy configs remain supported.

The cached revision contributes to `Simulation.config_hash()`: a new process using a different revision invalidates completed-run skipping, while an existing process's hash stays stable.

## Internal Maintenance Layers

`_density_implementations/` isolates the numerical mechanics behind public profiles, including conical targets and HDF5 interpolation. This keeps `density_profiles.py` a stable catalogue for simulation assembly while model-specific code handles interpolation, composition, and validation.

`_laser_implementations/` fulfills a similar role in storing lengthy and specialized laser pulse implementations, while `laser.py` holds the base class and the basic profiles.

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