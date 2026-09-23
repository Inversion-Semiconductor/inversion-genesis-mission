# Building-Blocks Demo

This example shows how to assemble an FBPIC simulation by copying YAML
building blocks in and out of a directory. Each file in `cfg_active` becomes
one simulation element — densities, lasers, and hyperparameters — without
maintaining a separate top-level `Simulation` YAML.

The physics setup matches [`demo_downramp_simulation`](../demo_downramp_simulation/) (H flattop
with matched radial channel, N2 doped bump, Gaussian laser) but uses a coarse
grid so runs finish quickly. This is a usage demo and not physical.

## Directories

| Directory | Purpose |
|-----------|---------|
| `cfg_archive/` | Library of example YAML building blocks (generated) |
| `cfg_active/` | Active selection copied from the archive for the current run |
| `cfg_active/deps/` | Dependency YAML files referenced by composed profiles (not loaded as elements in the `Simulation`) |

`cfg_archive` holds every example profile and alternative component. Only the
files you copy into `cfg_active` participate in the simulation.

## Files

| File | Purpose |
|------|---------|
| `create_archive_yamls.py` | Generate example YAML files in `cfg_archive` and populate a default `cfg_active` |
| `run_sim.py` | Build and run a simulation from all YAML files in `cfg_active` |

## Archive contents

After running `create_archive_yamls.py`, `cfg_archive` contains:

- `simulation_hyperparameters.yaml` — coarse grid and diagnostics settings
- `smooth_sine_flattop.yaml`, `matched_radial_modifier.yaml`, `modified_density_profile.yaml` — H gas channel
- `doped_asymmetric_sine.yaml` — N2 injection bump
- `example_density_profile.yaml` — standalone alternative density (not in the default active set)
- `gaussian_linear.yaml`, `gaussian_left_circular.yaml` — laser pulse options

No `simulation.yaml` is written. Pass the active directory directly to
`Simulation(elements=cfg_active)`.

## Usage

1. Generate the archive and default active config:

   ```bash
   python create_archive_yamls.py
   ```

2. Customize the simulation by adding or removing files in `cfg_active`:

   ```bash
   # Add a second laser pulse
   cp cfg_archive/gaussian_left_circular.yaml cfg_active/

   # Swap in a different density profile
   cp cfg_archive/example_density_profile.yaml cfg_active/
   rm cfg_active/modified_density_profile.yaml
   rm -r cfg_active/deps
   ```

   Rules:

   - Keep exactly one `simulation_hyperparameters.yaml`.
   - Include at least one density profile and one laser pulse.
   - `modified_density_profile.yaml` references base and modifier files under
     `cfg_active/deps/`. Keep that subdirectory (or update the paths in the
     modified profile) when swapping density components.

3. Run the simulation:

   ```bash
   python run_sim.py
   ```

   This writes profile plots to `plots/`, records grid settings in
   `cfg_active/grid_parameters.yaml`, and saves diagnostics under `diags/`.

## How it works

`run_sim.py` passes `cfg_active` as the `elements` argument to
`Simulation`. The loader reads every `*.yaml` file in that directory and sorts
each object into hyperparameters, densities, or lasers. Nested path references
(for example in `modified_density_profile.yaml`) resolve relative to
`cfg_active`.
