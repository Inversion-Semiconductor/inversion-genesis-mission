# Single-Simulation Demo

This example builds and runs an FBPIC LPA simulation with
`inversion_fbpic.lib`. It demonstrates `Parameters` for curated physical inputs,
`MomentDescriptorDiagnostic` for automatic beam analysis, and `ConfigContainer`
for saving both records together in one HDF5 results file. Simulation configs
are also saved for reference and later re-running without the original script.

The setup uses an 800 nm Gaussian laser (4.5 J, 40 fs, 30 um waist) in a
helium flattop plasma with 3% nitrogen doping. The flattop
length is calculated from the requested 430 MeV target energy and the laser's
estimated `a0`.

Furthermore, since we are most interested in the statistics of the injected electrons, primarily sourced from nitrogen, the particles-per-cell is increased for this species and decreased for the helium species.

## File

| File | Purpose |
|------|---------|
| `run_simulation.py` | Defines and records physical inputs, runs FBPIC, and exports the combined electron-beam moment descriptor |

## Run the demo

Activate an environment containing `inversion_fbpic`, FBPIC, MPI, and the
plotting dependencies. Output paths are anchored to this demo directory,
independent of the process working directory. A typical invocation is:

```bash
cd lpa/Simulation_FBPIC/demos/demo_ionization_simulation
python run_simulation.py
```

The script detects whether it was launched with multiple MPI ranks. For a
multi-rank run, use your MPI launcher, for example:

```bash
mpirun -n 2 python run_simulation.py
```

## Outputs

The run writes the following artifacts in the demo directory:

- `results.h5` contains one `ConfigContainer` under `/config`. Its first entry
   is the physical `Parameters` record; its second is the completed
   `MomentDescriptorDiagnostic` for the combined helium and nitrogen electrons.
- `cfgs/` contains records of the hyperparameters, calculated grid
   parameters (YAML format), laser, and both density profiles in the selected
   format.
- `plots/density_profiles.png` shows the longitudinal flattop and downramp
   density profiles.
- `diags/` contains the FBPIC diagnostic dumps.
- `plots/stills/` and `plots/rho.mp4` contain charge-density frames and the
   generated rho movie.
- `plots/beam_summary.txt` and `plots/beam_analysis.png` retain the existing
   beam-summary and plotting workflow.

## Read The Results

```python
from inversion_fbpic.lib.config_container import ConfigContainer

results = ConfigContainer.from_hdf5_file("results.h5")
physical_parameters, beam_moments = results.configs
print(physical_parameters.data["laser_energy_j"])
print(beam_moments.data["total_beam_charge_pc"])
```

The physical record includes the hard-coded target energy, laser energy,
wavelength, waist, pulse duration, plasma density, dopant fraction, and geometry,
plus useful derived values such as `a0` and the calculated flattop length.
Numeric keys carry their SI units, with energy also labelled explicitly as eV
or J. The beam record contains the default 33-feature descriptor: momentum
centroids, the 6D covariance entries, longitudinal slice features, and beam
charge in pC.

The selector `("all_of_species", "e")` gathers both recorded electron
populations. It analyzes the particles retained by their existing `uz >= 10`
diagnostic filters. `results.h5` is written after analysis on rank zero, before
the optional movie and plotting work, and can be overwritten by rerunning the
demo. No raw particle arrays are copied into this results file.

Reloading preserves `beam_moments.data`, but runtime completion status and the
simulation attachment are deliberately not persisted.

## Customize the setup

Edit the constants at the start of `run_simulation.py` to change the target
energy, laser, plasma density, or dopant fraction. The script recalculates the
flattop length whenever the target energy, laser, or flattop density changes.

The simulation saves a hash representation of itself; if it has already run successfully, it will not attempt to re-run if only non-simulation-related items are altered (e.g., if post-processing is modified within the same script).

When stepping is skipped because a matching hash exists, automatic analysis
still reruns against the last recorded particle dump and refreshes `results.h5`.
The simulation hash is saved before post-analysis. If analysis fails, retrying
the demo skips the completed simulation and retries analysis rather than stepping
again. The results file is replaced only after successful analysis.
