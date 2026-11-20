"""A lab-frame, beam-driven plasma wakefield accelerator (PWFA): an illustrative case.

A Gaussian drive bunch (3 nC, gamma 1.957e4, about 10 GeV) drives a nonlinear "blowout" wake in a
4e16 cm^-3 plasma, and a Gaussian witness bunch (100 pC, the same gamma) 140 um behind it is
accelerated in the wake's ion cavity. The parameters are this example's own design, chosen so that
the run is sound: both beams are resolved by the grid, both are matched to the cavity's focusing,
and the run can be long enough to accelerate. They are not a particular machine's.

The design:

- Plasma: 4e16 cm^-3 (k_p^-1 = 26.6 um, lambda_p = 167 um), bare electrons (`species=None`: no ion
  species; the ions are an immobile background), from a sharp step (a `LinearRampFlattop` with a
  1 nm ramp) at the front of the box, so that the beams reach it at once.
- Driver: sigma_r = 0.4 / k_p = 10.6 um and sigma_z = 1.2 / k_p = 31.9 um, so dense
  (8 times the plasma) and short enough to blow the plasma electrons out.
- Witness: sigma_r = 2 um, sigma_z = 6 um, 140 um behind the driver's centre (k_p zeta = -5.27),
  where its tail stays ahead of the cavity's back. Its dumps keep |x|, |y| < 20 um
  (`WITNESS_ACCEPTANCE`; see "Checks").
- Matching: in the cavity the ions focus an electron of gamma with the betatron function
  beta_m = sqrt(2 gamma) / k_p (5.3 mm here; its betatron wavelength is 2 pi beta_m = 33 mm). A
  bunch keeps its size when its normalized emittance is
  `matched_emittance(sigma_r, gamma) = sigma_r^2 k_p sqrt(gamma / 2)`: 14.9 mm mrad for the
  witness and about 420 mm mrad for the driver. The driver's is far larger than a real driver's
  (an angular spread of about 2 mrad, harmless at 10 GeV); matching it keeps the driver's size,
  and so the wake, steady. The driver's head, in plasma not yet blown out, is not focused and
  erodes slowly.
- Grid: the box is two plasma wavelengths long (334 um) and 4 / k_p (106 um) in radius, past the
  cavity (about 2.3 / k_p); dz is about 1 um (`NZ` = 336) and dr = 0.5 um, so the witness spans 4
  radial and 6 longitudinal cells, and the driver 21 and 32. `inversion_fbpic` warns that 336's
  largest prime factor, 7, is not optimal for the FFT; the results below were run with it. The
  time step is one cell of light travel, dz / c, the only one `SimulationHyperparameters` gives
  in the lab frame; the window moves at c (`beta_window=1.0`).
- Plasma macroparticles: 2 x 2 x 4 per cell (see "Checks").
- Length: `plasma_length` is the distance the witness travels in the plasma; `PLASMA_LENGTHS`
  holds three presets, `quick` (the default), `standard` and `long` (one betatron wavelength). The
  run lasts that plus the witness's distance to the plasma's start, and half a cell of
  `right_buffer` keeps the step count from rounding down.
- `random_seed` is set so a run repeats. `use_cuda` is True by default: a GPU when there is one,
  the CPU otherwise.

The model's inputs are the plasma density (`plasma_density`) and each bunch's fields from
`make_beam_actions()` (`driver_charge`, `witness_sigma_r`, ...); its charge and energy outputs
describe the witness.

Running it (`python pwfa_gaussian.py [--length MM | --preset NAME]`) prints the witness's charge,
mean energy and energy spread, plots the last dump (`plot_results()`: the plasma's and the
bunches' electron density, the longitudinal field `E_z` as a map and as a lineout near the axis,
and the transverse field `E_r - c B_theta`, with the bunches' centres marked), and analyzes the
witness (`analyze_witness()`, with `inversion_fbpic.utils.analysis`). Needs matplotlib.

Results, from runs on the CPU (8 threads held to 8 cores, no GPU), with the default plasma
macroparticles. The standard and long runs were made before `WITNESS_ACCEPTANCE` was added, and
the cut was applied to their dumps afterwards; a quick run with the cut gave the same dumps as one
without it (to 2e-13), as the cut changes only what is written.

    preset               quick      standard    long
    plasma length        2 mm       10 mm       33.3 mm
    steps                2283       10333       33507
    run time             10 min     47 min      2 h 29 min
    witness energy gain  37.6 MeV   185 MeV     576 MeV
    energy spread (rms)  4.6 MeV    23.5 MeV    76.3 MeV
    witness emittance    15.0       15.0        15.0      (x, mm mrad; 14.9 in y)
    witness sigma_x      2.03 um    1.99 um     1.96 um
    driver energy loss   18 MeV     86 MeV      266 MeV
    driver sigma_x       10.9 um    13.5 um     25.8 um   (10.6 um at the start)

Checks, on those runs:

- The witness keeps its size and its emittance over the whole long run, one betatron wavelength:
  15.0 to 15.04 mm mrad in x in every dump, and slice by slice it stays matched, beta / beta_m =
  0.96 to 1.01 at the end. Its energy spread is a correlated chirp: the witness sits on the slope
  of E_z ahead of the spike at the cavity's back (100 pC does not flatten it), so its tail gains
  more than its head.
- `WITNESS_ACCEPTANCE`: a Gaussian tail reaches where the plasma sheath closes the cavity on the
  axis (k_p zeta = -6.4; the witness's centre is at -5.27), and a macroparticle there is
  expelled. In the standard and long runs one of the 20,000 was (from 4.95 sigma_z behind the
  centre, to r = 198 um by the end of the long run): 0.005 pC, out of the dumps and so out of
  every statistic, plot and analysis. Kept, it alone would raise the long run's projected x
  emittance to 17.9 mm mrad and sigma_x to 2.33 um. The cut changes only what is written, not
  the run.
- `witness_summary.txt`'s alpha^2 + beta gamma is 1.003 to 1.006 at the end of every run. That
  check (`inversion_fbpic.utils.analysis`) is 1 + 2 alpha^2: its warning flags a bunch off its
  waist (|alpha| > 0.07), not unreliable Twiss parameters.
- E_z at the witness's centre is about -18 GV/m, steady from 0.5 mm into the plasma (the wake's
  build-up) over the quick and standard runs, falling to -15.8 GV/m by the end of the long run as
  the driver's head erodes (its head is not blown out and is not focused; the driver's size and
  emittance grow throughout, faster as the run goes on). Once the wake has formed, the witness's
  gain is within 1.5% of E_z over its particles, weighted by their charge and integrated over the
  dumps: 33.7 against 33.2 MeV in quick (from 0.46 mm), 170.5 against 168.8 MeV in standard
  (from 1.03 mm), 518.9 against 513.0 MeV in long (from 3.33 mm). The dumps are too far apart to
  integrate across the wake's build-up.
- The focusing gradient at the witness, d(E_r - c B_theta)/dr over its first 8 radial cells, is
  0.85 to 1.20 times the ion column's e n_p / (2 epsilon_0) in every dump after the wake has
  formed (0.5 mm).
- Plasma macroparticles: the quick run at 4 x 4 x 4 per cell (18 min) agrees with 2 x 2 x 4: the
  witness's gain 37.3 against 37.6 MeV, spread 4.59 against 4.57 MeV, E_z at the witness -18.0
  against -17.9 GV/m, and the driver's energy loss within 0.3%. The witness's emittance, 14.6
  against 15.0 mm mrad, differs from the start (its particles are drawn from a different random
  sequence) and is kept unchanged in both.
"""

from __future__ import annotations

import argparse
import math
import pathlib

import numpy
from scipy.constants import c, e, epsilon_0, m_e, pi

from inversion_fbpic.lib.particle_beam import GaussianParticleBeam
from inversion_fbpic.lib.simulation import SimulationHyperparameters
from inversion_fbpic.utils import analysis
from lume_fbpic.actions import (
    DensityFieldAction,
    FinalParticlesAction,
    make_beam_actions,
    make_stat_actions,
)
from lume_fbpic.density_profiles import LinearRampFlattop
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.simulator import FBPICSimulator
import pwfa_dumps  # docs/examples/pwfa_dumps.py, this directory

N_PLASMA = 4.0e16 * 100**3  # [m^-3]
OMEGA_P = numpy.sqrt(N_PLASMA * e**2 / (m_e * epsilon_0))
K_P = OMEGA_P / c
LAMBDA_P = 2.0 * pi / K_P

GAMMA = 1.957e4  # both bunches
DRIVE_SIGMA_R = 0.4 / K_P
DRIVE_SIGMA_Z = 1.2 / K_P
DRIVE_Q = 3.0e-9  # [C]
DRIVE_MACROPARTICLES = 50_000
WITNESS_SIGMA_R = 2.0e-6
WITNESS_SIGMA_Z = 6.0e-6
WITNESS_Q = 1.0e-10  # [C]
WITNESS_MACROPARTICLES = 20_000
# The witness's dumps keep |x|, |y| below this [m], 10 sigma_r: a macroparticle from deep in
# the tail, expelled at the cavity's back, would otherwise dominate the rms statistics.
# It also works around a faulty check in inversion_fbpic: `analysis.print_beam_summary()` warns
# "Twiss parameters may be unreliable!" when alpha^2 + beta gamma is not 1, but that sum is
# 1 + 2 alpha^2 (the identity is beta gamma - alpha^2 = 1), so it warns for any |alpha| > 0.07.
# The expelled macroparticle raised the long run's alpha_x to -0.11 and set it off.
WITNESS_ACCEPTANCE = 20.0e-6
# The witness behind the driver [m]: its Gaussian tail (to about 4 sigma_z) stays ahead of where
# the sheath closes the cavity on the axis (k_p zeta = -6.4 behind the driver); at 150 um the
# tail reached it and was expelled
TRAILING_DISTANCE = 140.0e-6
# The ion cavity's matched betatron function and the betatron wavelength [m]
BETA_MATCHED = math.sqrt(2.0 * GAMMA) / K_P
BETATRON_WAVELENGTH = 2.0 * pi * BETA_MATCHED

DOMAIN_LENGTH = 2.0 * LAMBDA_P
DOMAIN_RADIUS = 4.0 / K_P
NZ = 336  # dz of about 1 um; 2^4 3 7 (inversion_fbpic warns that 7 is not optimal)
NR = int(numpy.rint(DOMAIN_RADIUS / 0.5e-6))  # dr of 0.5 um: 4 cells a witness radius
PLASMA_PPC = (2, 2, 4)  # z, r, theta
# The driver's centre is 4 sigma_z behind the box's front, where the plasma starts
DRIVE_Z = DOMAIN_LENGTH - 4.0 * DRIVE_SIGMA_Z
WITNESS_Z = DRIVE_Z - TRAILING_DISTANCE
# How far the witness travels in plasma [m], by preset
PLASMA_LENGTHS = {
    "quick": 2.0e-3,
    "standard": 1.0e-2,
    "long": BETATRON_WAVELENGTH,
}
NUMBER_DUMPS = 11  # the start, then every tenth of the run


def analyze_witness(
    directory: str | pathlib.Path, *, output: str | pathlib.Path = "."
) -> list[pathlib.Path]:
    """Analyze the witness of the last dump of the run in `directory` (its working directory)
    with `inversion_fbpic.utils.analysis`, and return the files written to `output`:

    - `witness_summary.txt`: `print_beam_summary()` of `analyze_beam()`: charge, energy and its
      spread, emittance, Twiss parameters and sizes.
    - `witness_analysis.png`: `plot_beam_analysis()`: the current, energy against z and the phase
      spaces.

    Needs matplotlib.
    """
    import matplotlib

    matplotlib.use("Agg")
    diagnostics = build_model(dummy_run=True).simulator.diagnostics_directory(
        pathlib.Path(directory)
    )
    pwfa_dumps.dumps(diagnostics)  # raises if there are none
    # x, y, z, ux, uy, uz, w, q and the time series
    beam = analysis.load_beam_data(diagnostics, "witness")
    summary = analysis.analyze_beam(*beam[:8])
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = [output / "witness_summary.txt", output / "witness_analysis.png"]
    analysis.print_beam_summary(summary, files[0])
    analysis.plot_beam_analysis(
        *beam[:7], summary, supertitle="Witness", save_path=files[1], show=False
    )
    return files


def build_model(
    *,
    dummy_run: bool = False,
    plasma_length: float = PLASMA_LENGTHS["quick"],
    plasma_ppc: tuple[int, int, int] = PLASMA_PPC,
    use_cuda: bool = True,
) -> LUMEFBPICModel:
    """The model of the module docstring: the witness travels `plasma_length` [m] in plasma, with
    `plasma_ppc` plasma macroparticles per cell (z, r, theta)."""
    dz = DOMAIN_LENGTH / NZ
    hyparams = SimulationHyperparameters(
        zmin=0.0,
        zmax=DOMAIN_LENGTH,
        nz=NZ,
        rmax=DOMAIN_RADIUS,
        nr=NR,
        nm=1,
        use_mpi=False,
        use_cuda=use_cuda,
        number_dumps=NUMBER_DUMPS,
        # POSIT: the window moves at c and the witness with it: it reaches the plasma's start
        # (the box's front) after `DOMAIN_LENGTH - WITNESS_Z`
        simulation_length=plasma_length + DOMAIN_LENGTH - WITNESS_Z,
        right_buffer=0.5 * dz,
        beta_window=1.0,
        field_diagnostics=["rho", "E", "B"],
        r_boundary="reflective",
        smoother_passes=1,
        smoother_compensator=False,
        random_seed=0,
    )
    plasma = LinearRampFlattop(
        nominal_density=N_PLASMA,
        species=None,
        ionization=-1,
        p_nz=plasma_ppc[0],
        p_nr=plasma_ppc[1],
        p_nt=plasma_ppc[2],
        ramp_start=DOMAIN_LENGTH,
        ramp_length=1.0e-9,
    )
    driver = GaussianParticleBeam(
        n_physical_particles=DRIVE_Q / e,
        gamma0=GAMMA,
        sig_r=DRIVE_SIGMA_R,
        sig_z=DRIVE_SIGMA_Z,
        sig_gamma=1.0,
        n_emit=matched_emittance(DRIVE_SIGMA_R, GAMMA),
        n_macroparticles=DRIVE_MACROPARTICLES,
        tf=0.0,
        zf=DRIVE_Z,
        particle_name="driver",
    )
    witness = GaussianParticleBeam(
        n_physical_particles=WITNESS_Q / e,
        gamma0=GAMMA,
        sig_r=WITNESS_SIGMA_R,
        sig_z=WITNESS_SIGMA_Z,
        sig_gamma=1.0,
        n_emit=matched_emittance(WITNESS_SIGMA_R, GAMMA),
        n_macroparticles=WITNESS_MACROPARTICLES,
        tf=0.0,
        zf=WITNESS_Z,
        particle_name="witness",
        particle_select={
            k: [-WITNESS_ACCEPTANCE, WITNESS_ACCEPTANCE] for k in ("x", "y")
        },
    )
    simulator = FBPICSimulator(
        hyparams,
        densities=[plasma],
        beams=[driver, witness],
        target_species="witness",
    )
    actions = [
        DensityFieldAction(
            name="plasma_density", density_index=0, field_name="nominal_density"
        ),
        *make_beam_actions(simulator, 0),
        *make_beam_actions(simulator, 1),
        *make_stat_actions(),
        FinalParticlesAction(name="final_particles", read_only=True),
    ]
    return LUMEFBPICModel(simulator, actions, dummy_run=dummy_run)


def matched_emittance(sigma_r: float, gamma: float) -> float:
    """The normalized emittance [m] that keeps a bunch of rms size `sigma_r` [m] and Lorentz
    factor `gamma` at that size in the ion cavity: `sigma_r^2 k_p sqrt(gamma / 2)`, from
    `sigma_r = sqrt(beta_m eps_n / gamma)` with `beta_m = sqrt(2 gamma) / k_p`."""
    return sigma_r**2 * K_P * math.sqrt(gamma / 2.0)


def plot_results(
    directory: str | pathlib.Path, *, output: str | pathlib.Path = "."
) -> list[pathlib.Path]:
    """Plot the last dump of the run in `directory` (its working directory) and return the files
    written to `output`:

    - `electron_density.png`: the electron density of the plasma and the bunches over the
      plasma's, on a log scale from 1e-2, so that the cavity, the plasma and the bunches show.
    - `longitudinal_field.png`: `E_z` [GV/m]; an electron is accelerated where it is negative.
    - `longitudinal_field_lineout.png`: `E_z` along the first radial cell.
    - `transverse_force.png`: `E_r - c B_theta` [GV/m], minus the transverse force on a
      relativistic electron per unit charge: positive focuses electrons.

    The two field maps share a linear colour scale of +/- 1.5 E_0 (E_0 = m_e c omega_p / e,
    19.2 GV/m), the same in every run: the spike where the sheath closes the cavity on the axis,
    up to about 200 GV/m in a few cells, would otherwise set the scale and wash out the rest of
    the wake. Values beyond it take the end colours.

    The horizontal axis is `k_p zeta`, with `zeta = z - z_driver` (zero at the drive bunch's
    centre, the wake behind it at negative values) and the radial one is `k_p r`; dashed lines
    mark the driver's and the witness's centres, labelled above the plot. Needs matplotlib.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    from matplotlib.colors import LogNorm, Normalize

    snapshot = pwfa_dumps.read_dump(
        pwfa_dumps.dumps(
            build_model(dummy_run=True).simulator.diagnostics_directory(
                pathlib.Path(directory)
            )
        )[-1]
    )
    zeta, radius = K_P * snapshot["zeta"], K_P * snapshot["r"]
    witness = snapshot["particles"]["witness"]
    centres = {
        "driver": 0.0,
        "witness": K_P
        * (
            numpy.average(witness["z"], weights=witness["weight"])
            - snapshot["z_driver"]
        ),
    }
    # rho is the charge density of the electrons, the plasma's and the bunches' (no ions)
    density = numpy.clip(-snapshot["rho"] / e / N_PLASMA, 1.0e-2, None)
    extent = [zeta[0], zeta[-1], radius[0], radius[-1]]
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = []

    def mark_centres(axis):
        for k, x in centres.items():
            axis.axvline(x, color="k", linestyle="--", linewidth=0.8)
            # x in data, y in axes coordinates: just above the plot
            axis.text(
                x,
                1.01,
                k,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
            )

    def field_map(name, data, label, cmap, norm, extend="neither"):
        figure, axis = pyplot.subplots()
        image = axis.imshow(
            data, extent=extent, cmap=cmap, origin="lower", norm=norm, aspect="auto"
        )
        mark_centres(axis)
        axis.set(xlabel=r"$k_p \zeta$", ylabel=r"$k_p r$")
        figure.colorbar(
            image, ax=axis, orientation="horizontal", label=label, extend=extend
        )
        figure.tight_layout()
        files.append(output / f"{name}.png")
        figure.savefig(files[-1], dpi=120)
        pyplot.close(figure)

    # 1.5 E_0 [GV/m]
    v = 1.5 * m_e * c * OMEGA_P / e / 1.0e9

    field_map(
        "electron_density",
        density,
        r"$n_e / n_p$",
        "viridis",
        LogNorm(1.0e-2, max(density.max(), 1.0)),
    )
    field_map(
        "longitudinal_field",
        snapshot["E_z"],
        r"$E_z$ [GV/m] (< 0 accelerates electrons)",
        "RdBu",
        Normalize(-v, v),
        "both",
    )
    field_map(
        "transverse_force",
        snapshot["F_r"],
        r"$E_r - c B_\theta$ [GV/m] (> 0 focuses electrons)",
        "RdBu",
        Normalize(-v, v),
        "both",
    )

    figure, axis = pyplot.subplots()
    axis.plot(zeta, snapshot["E_z"][0])
    mark_centres(axis)
    axis.set(xlabel=r"$k_p \zeta$", ylabel=r"$E_z$ near the axis [GV/m]")
    figure.tight_layout()
    files.append(output / "longitudinal_field_lineout.png")
    figure.savefig(files[-1], dpi=120)
    pyplot.close(figure)
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    lengths = parser.add_mutually_exclusive_group()
    lengths.add_argument(
        "--length", type=float, help="the witness's travel in plasma [mm]"
    )
    lengths.add_argument(
        "--preset",
        choices=sorted(PLASMA_LENGTHS),
        default="quick",
        help="a preset length (default: quick)",
    )
    args = parser.parse_args()
    model = build_model(
        plasma_length=(
            args.length * 1e-3
            if args.length is not None
            else PLASMA_LENGTHS[args.preset]
        )
    )
    model.set({})  # runs the simulation (an empty set changes nothing first)
    print(model.get(["charge_pc", "energy_mean_mev", "energy_std_mev"]))
    directory = model.simulator.working_directory
    for path in plot_results(directory, output=directory) + analyze_witness(
        directory, output=directory
    ):
        print(f"wrote {path}")
