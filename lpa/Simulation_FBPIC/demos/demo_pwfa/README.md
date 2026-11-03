# Beam-driven downramp injection

This demonstration models PWFA in a hydrogen plasma driven by a relativistic electron bunch. The density downramp lowers the wake phase velocity and can trap plasma electrons behind the driver.

The downramp is represented by the sum of a flattop profile and a delayed, lower-density profile. The demonstration exercises the serializable `GaussianParticleBeam` API and records the driver (`driver_electrons`) plus both plasma-profile species (`electrons_flattop` and `electrons_downramp`) with FBPIC particle diagnostics.

## Run

Run from this directory so diagnostics and generated configuration files remain local:

```bash
cd lpa/Simulation_FBPIC/demos/demo_pwfa
python run_simulation.py
```

For MPI, use an MPI launcher:

```bash
mpirun -n 2 python run_simulation.py
```

## Outputs

- `cfgs/simulation.h5` — complete serializable simulation configuration
- `plots/plasma_density.png` — longitudinal and radial plasma-density plot
- `plots/rho.mp4` and `plots/eme.mp4` — field-energy and charge-density movies
- `plots/beam_summary.txt` and `plots/beam_analysis.png` — combined plasma-electron beam analysis
- `diags/` — field and particle diagnostics

The grid and macroparticle count are intentionally coarse. Increase them before using the case for quantitative beam-quality studies.
