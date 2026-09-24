This example runs a Bayesian optimization of an FBPIC LPA model
using Optimas. Each trial renders `sim_template.py` with a proposed parameter
set, runs the simulation, and analyzes the particle diagnostics.

The simulation follows the flattop-and-downramp setup in
[`demo_downramp_simulation`](../demo_downramp_simulation/). The template uses a 2048 x 500,
three-mode grid, a boost factor of 4, and two diagnostic dumps per trial (beginning and end).

## Files

| File | Purpose |
|------|---------|
| `run_optimas.py` | Configures the search space, tracked beam metrics, logging, and Optimas exploration. |
| `sim_template.py` | Defines one parameterized FBPIC simulation for each Optimas trial. |

## Run the optimization

Activate an environment containing `inversion_fbpic`, FBPIC, Optimas, MPI,
and the analysis dependencies. Run from this directory:

```bash
cd lpa/Simulation_FBPIC/demos/demo_optimas
python run_optimas.py
```

Do not start this command with `mpirun`. The shared Optimas launcher detects
available GPUs and assigns trial workers; without an available GPU it runs one
trial at a time in CPU mode. The default exploration performs 80 evaluations
and logs progress to `optimization.log`.

## Search parameters

| Parameter | Range | Description |
|------|-------|-------------|
| `laser_energy` | 4-6 J | Laser pulse energy. |
| `laser_waist` | 20-40 um | Laser waist. |
| `focal_position` | 1-6 mm | Laser focal position. |
| `tau_fwhm_fs` | 25-50 fs | Laser pulse duration. |
| `flattop_plasma_density_e18` | 0.4-2.0 | Flattop plasma density in $10^{18}$ cm$^{-3}$. |
| `downramp_length` | 20-100 um | Downramp length. |
| `downramp_height` | 0.1-1.0 | Downramp density as a fraction of flattop density. |
| `downramp_position` | 250 um-1 mm | Downramp position. |

## Analysis and outputs

For every trial, the analysis reads `diags/hdf5` and records `charge`,
`charge_weighted`, `energy_mean`, `energy_med`, and `energy_mad`. The
maximized objective `f` favors high charge in a narrow momentum window, low
momentum deviation, low relative laser energy, and low normalized transverse
emittance. Trials without a qualifying beam receive an objective of zero.

Optimas writes the optimization history and per-trial run directories in the
current directory. Each trial also records its generated configuration in
`cfgs/`, its density-profile plot in `plots/`, and its FBPIC diagnostics in
`diags/`.

## Customize the exploration

Change `MAX_EVALS` or the `VaryingParameter` definitions in
`run_optimas.py` to adjust the evaluation budget or search space. Edit the
fixed simulation resolution and runtime settings near the top of
`sim_template.py` when a different fidelity is required.
