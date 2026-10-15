"""PWFA energy doubling at CLARA FEBE: lume_fbpic setup of the simulations in Zhang et al.,
"Enabling energy-doubling at CLARA FEBE: high-quality beam generation in plasma wakefield
acceleration", Plasma Phys. Control. Fusion 68, 045051 (2026),
https://doi.org/10.1088/1361-6587/ae5e08, published under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). The parameters are taken from its tables 1 and 2
and equation (4); the implementation here is new. Where a value differs from the paper's, a
`# paper:` comment gives the paper's.

A 150 pC, 250 MeV drive bunch and a trailing witness bunch, 90 um behind, cross a helium plasma
cell with an up-ramp, a plateau and a down-ramp; the witness is meant to reach 500 MeV in 20 cm.
`build_model(witness="longer")` is the paper's 10 pC witness, `"shorter"` the 5 pC one (Table 2).

The defaults are coarser than the paper's: dz = 1.5 um (paper: 0.30 um) and 2 x 2 x 4 plasma
macroparticles per cell (paper: 4 x 4 x 4). A 20 cm run at these settings is 133,333 steps. At
the paper's resolution it is 666,667 steps on a grid five times as long, with four times the plasma
macroparticles per cell, which is better run on a GPU (`use_cuda=True`); `--dz 0.30 --ppc 4 4 4`
gives it.

The 20 cm run, measured as the paper does (with its 5-sigma particle cut) against the paper's
values read off its Figures 6a and 7 (approximate):

                              paper                          this example
    500 MeV reached at        about 17 cm                    15.7 cm
    energy at 10 cm           about 372 MeV                  377 MeV
    spread at 6 cm / end      2.1 % / 5.8 % (at 17 cm)       2.2 % / 5.0 % (at 17 cm)
    charge                    10 pC to 6 cm, then 8.6 pC     10 pC to 6 cm, then 8.57 pC
    emittance peak / end      16 at 7 cm / 6.5 mm mrad       13.1 at 6.7 cm / 4.74 mm mrad
    transverse size, end      about 3.1 um                   about 3.0 um

Chosen here, as the paper's text does not give them:

- Positions: the window is z = -150 um to 0, the plasma begins at z = 0, and the driver is
  centred at -30 um.
- The helium is a fully ionized, immobile bare electron plasma (`species=None`).
- The cell is 20 cm long for both cases, and the 250 MeV is taken as the total energy.
- Macroparticles of the bunches: 100000 for the driver, 20000 for the witness.

The model's mean energy is kinetic, 0.511 MeV below the total energy the paper quotes. Its
charge, energy and spread outputs cover all the witness's particles; the emittance, size and
charge in the table are computed from the dumps with the cut.

`python pwfa_clara_febe.py --steps N` runs the first N steps (default: the whole cell). `--plot`
plots the run when it ends and `--plot-only DIR` plots an earlier run from its dumps: the
witness's size, energy, spread, charge and emittance along the run, and snapshots of the beam
density and the wakefield.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy
from scipy.ndimage import gaussian_filter

from inversion_fbpic.utils.distributions import COORD_NAMES
from lume_fbpic.actions import make_pwfa_actions
from lume_fbpic.density_profiles import UpDownRampProfile
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.pwfa_config import GaussianBunch, PWFAGrid
from lume_fbpic.pwfa_simulator import PWFASimulator
from lume_fbpic.utils import ELECTRON_MC2_EV, particle_group
import pwfa_dumps  # docs/examples/pwfa_dumps.py, this directory

CELL_LENGTH = 0.20  # [m]; paper: a 20 cm plasma cell
N_MAX_CM3 = 4.2e16  # paper: plateau density 4.2e16 cm^-3
GAMMA = 250.0e6 / ELECTRON_MC2_EV  # paper: 250 MeV, taken here as the total energy
SEPARATION = 90.0e-6  # paper: driver-witness separation (Table 2)
DRIVER_POSITION = -30.0e-6
RAMP_LENGTH = 0.01  # [m]; paper: 1 cm up- and down-ramps, L_up = L_down

# Where `plot_results` takes its snapshots [m]: in the initial stage, a little way up the ramp (a
# fraction of it, so the density is between the two stages, as in the paper's 3.6e16 of 2.8 -> 4.2e16),
# and on the plateau; and the off-axis wakefield at the paper's z = 17.97 mm, at r = 15 um.
SNAPSHOT_Z_INITIAL = 2.5e-3
SNAPSHOT_RAMP_FRACTION = 0.57
SNAPSHOT_Z_PLATEAU = 0.10
OFFAXIS_Z = 17.97e-3
OFFAXIS_R = 15.0e-6

WINDOW_LENGTH = 150.0e-6  # the paper's window, longitudinal and transverse (Table 1)
DZ = 1.5e-6  # paper (Table 1): 0.30 um, i.e. 500 cells; here 100 cells and 5x fewer steps
PLASMA_PPC = (
    2,
    2,
    4,
)  # paper (Table 1): (4, 4, 4) plasma macroparticles per cell (z, r, theta)

# Table 2: the two witness cases; the values are the paper's.
CASES = {
    "longer": dict(
        n0_cm3=2.8e16,
        z_up=0.05,
        z_down=0.15,
        charge=10.0e-12,
        sig_z=10.0e-6,
        sig_r=14.0e-6,
        n_emit=5.0e-6,
    ),
    "shorter": dict(
        n0_cm3=1.9e16,
        z_up=0.048,
        z_down=0.14,
        charge=5.0e-12,
        sig_z=2.0e-6,
        sig_r=7.0e-6,
        n_emit=2.0e-6,
    ),
}


def build_model(
    *,
    witness: str = "longer",
    dummy_run: bool = False,
    dz: float = DZ,
    n_steps: int | None = None,
    plasma_ppc: tuple[int, int, int] = PLASMA_PPC,
    use_cuda: bool = False,
    write_period: int | None = None,
    write_plasma: bool = False,
) -> LUMEFBPICModel:
    """The paper's setup at the module's coarser resolution (see the module docstring): `dz` [m]
    sets the longitudinal cell size (the window stays 150 um, so `nz = 150 um / dz`, and the time
    step is `dz / c`), `plasma_ppc` the plasma macroparticles per cell (z, r, theta). The paper's
    own are `dz=0.30e-6` and `plasma_ppc=(4, 4, 4)`. `n_steps` defaults to the whole 20 cm cell at
    that `dz`; `write_period` to an eightieth of the steps (80 dumps for `plot_results`).
    """
    case = CASES[witness]
    nz = int(round(WINDOW_LENGTH / dz))
    n_steps = n_steps if n_steps is not None else int(round(CELL_LENGTH / dz))
    grid = PWFAGrid(
        zmin=-WINDOW_LENGTH,
        zmax=0.0,
        nz=nz,  # paper: 500 (dz = 0.30 um); here 150 um / dz
        rmax=150.0e-6,  # paper: 150 um transverse window
        nr=200,  # paper: 200 (dr = 0.75 um)
        nm=1,  # paper: m = 0 only
        n_steps=n_steps,
        write_period=write_period or max(n_steps // 80, 1),
        write_plasma=write_plasma,
        r_boundary="reflective",  # paper: reflective in r
        z_boundary="open",  # paper: open in z
        use_cuda=use_cuda,
        random_seed=0,
    )
    plasma = UpDownRampProfile(
        nominal_density=N_MAX_CM3 * 1.0e6,
        species=None,
        ionization=-1,
        p_nz=plasma_ppc[0],  # paper: 4
        p_nr=plasma_ppc[1],  # paper: 4
        p_nt=plasma_ppc[2],  # paper: 4
        base_fraction=case["n0_cm3"]
        / N_MAX_CM3,  # n0 / n_max: paper 2.8 / 4.2 or 1.9 / 4.2
        z_up=case["z_up"],
        up_length=RAMP_LENGTH,
        z_down=case["z_down"],
        down_length=RAMP_LENGTH,
        z_start=0.0,
        z_end=CELL_LENGTH,
    )
    driver = GaussianBunch(
        charge=150.0e-12,  # paper: 150 pC
        gamma=GAMMA,  # paper: 250 MeV
        sig_gamma=0.01 * GAMMA,  # paper: 1 % energy spread
        sig_r=70.0e-6,  # paper: 70 um
        sig_z=10.0e-6,  # paper: 10 um
        n_emit=5.0e-6,  # paper: 5 mm mrad
        n_macroparticles=100_000,  # not in the paper
        zf=DRIVER_POSITION,
    )
    witness_bunch = GaussianBunch(
        charge=case["charge"],  # paper: 10 pC (longer) or 5 pC (shorter)
        gamma=GAMMA,  # paper: 250 MeV
        sig_gamma=0.01 * GAMMA,  # paper: 1 % energy spread
        sig_r=case["sig_r"],  # paper: 14 um or 7 um
        sig_z=case["sig_z"],  # paper: 10 um or 2 um
        n_emit=case["n_emit"],  # paper: 5 or 2 mm mrad
        n_macroparticles=20_000,  # not in the paper
        zf=DRIVER_POSITION - SEPARATION,  # paper: 90 um behind the driver
    )
    simulator = PWFASimulator(
        grid, plasma, driver, witness_bunch, target_species="witness"
    )
    return LUMEFBPICModel(simulator, make_pwfa_actions(simulator), dummy_run=dummy_run)


def plot_results(
    directory: str | pathlib.Path,
    *,
    witness: str = "longer",
    output: str | pathlib.Path = ".",
) -> list[pathlib.Path]:
    """Plot the run in `directory` (its working directory, holding `diags/hdf5`) from its dumps,
    and return the files written to `output`:

    - `transverse_size.png`: the witness's transverse size along the run.
    - `beam_quality_with_ramp_and_500MeV_markers.png`: its energy, energy spread, charge and
      emittance along the run, with the plasma ramps marked and where the energy doubles.
    - `beam_density_and_wakefield.png`: the density of both beams over the plasma density, and the
      on-axis wakefield, at three places along the plasma: the initial stage, partway up the ramp
      and the plateau.
    - `offaxis_transverse_wakefield.png`: the transverse wakefield at r = 15 um along the bunch
      at z = 17.97 mm, with the two bunches' extents.

    The first two are computed from the dumps as the paper does, within its 5-sigma particle cut:
    particles with `r^2 / sigma_r^2 + xi^2 / sigma_z^2 > 25` about the bunch centre, for the initial
    sigmas of the witness, count as lost. The beam density is estimated from the macroparticles
    in a slab |y| < 4 um. Only the run in `directory` is plotted. Needs matplotlib.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    case = CASES[witness]
    simulator = build_model(witness=witness, dummy_run=True).simulator
    paths = pwfa_dumps.dumps(simulator.diagnostics_directory(pathlib.Path(directory)))
    rows = numpy.array([_dump_metrics(path, case) for path in paths]).T
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plasma = simulator.plasma
    density = plasma.build_density_function()

    def plasma_density(z):  # [m^-3]
        return float(
            plasma.nominal_density * density(numpy.array([z]), numpy.array([0.0]))[0]
        )

    return [
        _plot_transverse_size(pyplot, rows, output),
        _plot_beam_quality(pyplot, rows, plasma, witness, output),
        _plot_beam_density_and_wakefield(
            pyplot, paths, plasma, plasma_density, witness, output
        ),
        _plot_offaxis_transverse_wakefield(pyplot, paths, witness, output),
    ]


def _beam_density(
    snapshot: dict,
    zeta_edges,
    x_edges,
    species=("driver", "witness"),
    slab: float = 4.0e-6,
):
    """The density [m^-3] of `species` in the slab |y| < `slab`, on a (zeta, x) grid [um], from the
    macroparticles, lightly smoothed."""
    counts = numpy.zeros((len(zeta_edges) - 1, len(x_edges) - 1))
    for name in species:
        beam = snapshot["particles"][name]
        inside = numpy.abs(beam["y"]) < slab
        counts += numpy.histogram2d(
            (beam["z"][inside] - snapshot["z_driver"]) * 1.0e6,
            beam["x"][inside] * 1.0e6,
            bins=[zeta_edges, x_edges],
            weights=beam["weight"][inside],
        )[0]
    volume = (
        numpy.diff(zeta_edges)[:, None]
        * numpy.diff(x_edges)[None, :]
        * 1.0e-12
        * 2.0
        * slab
    )
    return gaussian_filter(counts / volume, 1.0)


def _dump_metrics(path: pathlib.Path, case: dict) -> list[float]:
    """The witness in one dump, within the paper's 5-sigma cut: [distance (cm), energy (MeV),
    energy spread (%), charge (pC), emittance (mm mrad), transverse size (um)].

    The energy is the total energy, the emittance the mean of the x and y normalized emittances
    and the transverse size the rms of x, all beamphysics's statistics of the particles inside
    the cut."""
    dump = pwfa_dumps.read_dump(path)
    witness = dump["particles"]["witness"]
    x, y, z = (witness[axis] for axis in "xyz")
    inside = (
        ((x - numpy.median(x)) ** 2 + (y - numpy.median(y)) ** 2) / case["sig_r"] ** 2
        + ((z - numpy.median(z)) / case["sig_z"]) ** 2
    ) <= 25.0
    particles = particle_group(
        numpy.column_stack([witness[name] for name in COORD_NAMES])[inside],
        witness["weight"][inside],
    )
    return [
        dump["distance"] * 100.0,
        particles["mean_energy"] / 1.0e6,
        100.0 * particles["sigma_energy"] / particles["mean_energy"],
        particles.charge * 1.0e12,
        0.5 * (particles["norm_emit_x"] + particles["norm_emit_y"]) * 1.0e6,
        particles["sigma_x"] * 1.0e6,
    ]


def _plot_beam_density_and_wakefield(
    pyplot, paths, plasma, plasma_density, witness: str, output: pathlib.Path
) -> pathlib.Path:
    """Beam density over plasma density (top) and the on-axis wakefield (bottom), at three
    places along the plasma."""
    from matplotlib.colors import LogNorm

    distances = numpy.array([pwfa_dumps.distance(path) for path in paths])
    z_ramp = plasma.z_up + SNAPSHOT_RAMP_FRACTION * plasma.up_length
    zeta_edges, x_edges = numpy.linspace(-150, 40, 191), numpy.linspace(-60, 60, 121)
    floor = 1.0e-3  # cells with less density than this show the colour of the floor
    figure, axes = pyplot.subplots(
        2, 3, figsize=(15, 7), gridspec_kw={"height_ratios": [2.2, 1]}, sharex="col"
    )
    for column, z in enumerate((SNAPSHOT_Z_INITIAL, z_ramp, SNAPSHOT_Z_PLATEAU)):
        path = paths[int(numpy.argmin(numpy.abs(distances - z)))]
        snapshot = pwfa_dumps.read_dump(path)
        n_plasma = plasma_density(snapshot["distance"])
        ratio = _beam_density(snapshot, zeta_edges, x_edges) / n_plasma
        top = axes[0, column]
        image = top.pcolormesh(
            0.5 * (zeta_edges[1:] + zeta_edges[:-1]),
            0.5 * (x_edges[1:] + x_edges[:-1]),
            numpy.clip(ratio, floor, None).T,
            cmap="viridis",
            norm=LogNorm(floor, max(ratio.max(), 1.1 * floor)),
            shading="auto",
        )
        figure.colorbar(image, ax=top, label="$n_b / n_p$", pad=0.01)
        top.set_ylabel("x [um]")
        top.set_title(
            f"z = {snapshot['distance'] * 1e3:.1f} mm, n_p = {n_plasma * 1e-6:.2e} cm$^{{-3}}$",
            fontsize=9,
        )
        bottom = axes[1, column]
        bottom.plot(snapshot["zeta"] * 1.0e6, snapshot["E_z"][0], "r-")
        bottom.axhline(0, color="gray", lw=0.5)
        bottom.set(xlabel=r"$\zeta$ [um]", ylabel=r"on-axis $E_z$ [GV/m]")
        bottom.set_xlim(zeta_edges[0], zeta_edges[-1])
    figure.suptitle(
        "Beam density over plasma density, and the on-axis wakefield, along the plasma-density "
        f"ramp ({witness} witness)",
        fontsize=10,
    )
    figure.tight_layout()
    file = output / "beam_density_and_wakefield.png"
    figure.savefig(file, dpi=120)
    pyplot.close(figure)
    return file


def _plot_beam_quality(
    pyplot, rows, plasma, witness: str, output: pathlib.Path
) -> pathlib.Path:
    """Energy, spread, charge and emittance along the run, with the ramps and the doubling."""
    distance_cm, energy, spread, charge, emittance, _ = rows
    figure, axes = pyplot.subplots(2, 2, figsize=(10, 7.5))
    panels = (
        (energy, "Energy (MeV)"),
        (spread, "Energy spread (%)"),
        (charge, "Charge (pC)"),
        (emittance, "Emittance (mm mrad)"),
    )
    for axis, (values, ylabel) in zip(axes.flat, panels):
        axis.plot(distance_cm, values, "b-")
        axis.set(xlabel="Propagating distance (cm)", ylabel=ylabel, xlim=(0, 20))
        for ramp in (
            plasma.z_up,
            plasma.z_up + plasma.up_length,
            plasma.z_down,
            plasma.z_down + plasma.down_length,
        ):
            axis.axvline(ramp * 100.0, color="0.85", lw=0.8, zorder=0)
    doubled = 2.0 * energy[0]
    axes[0, 0].axhline(doubled, color="gray", ls=":")
    peak = int(numpy.argmax(energy)) + 1
    if (
        energy.max() >= doubled
    ):  # the energy is rising, so interpolate where it first crosses
        axes[0, 0].axvline(
            numpy.interp(doubled, energy[:peak], distance_cm[:peak]),
            color="gray",
            ls=":",
        )
    figure.suptitle(
        f"{witness} witness, ramping plasma; within the paper's 5-sigma particle cut",
        fontsize=9,
    )
    figure.tight_layout()
    file = output / "beam_quality_with_ramp_and_500MeV_markers.png"
    figure.savefig(file, dpi=120)
    pyplot.close(figure)
    return file


def _plot_offaxis_transverse_wakefield(
    pyplot, paths, witness: str, output: pathlib.Path
) -> pathlib.Path:
    """The transverse wakefield at r = 15 um along the bunch, at z = 17.97 mm."""
    distances = numpy.array([pwfa_dumps.distance(path) for path in paths])
    snapshot = pwfa_dumps.read_dump(
        paths[int(numpy.argmin(numpy.abs(distances - OFFAXIS_Z)))]
    )
    row = int(numpy.argmin(numpy.abs(snapshot["r"] - OFFAXIS_R)))
    figure, axes = pyplot.subplots(figsize=(9, 4))
    axes.plot(snapshot["zeta"] * 1.0e6, snapshot["F_r"][row], "b-")
    axes.axhline(0, color="gray", lw=0.5)
    axes.set(
        xlabel=r"$\zeta$ [um]",
        ylabel=r"$E_r - cB_\theta$ at r = %.0f um [GV/m]" % (snapshot["r"][row] * 1e6),
        xlim=(-150, 40),
    )
    for name, colour in (("driver", "k"), ("witness", "g")):
        beam = snapshot["particles"][name]
        zeta = (beam["z"] - snapshot["z_driver"]) * 1.0e6
        centre = numpy.average(zeta, weights=beam["weight"])
        sigma = numpy.sqrt(numpy.cov(zeta, aweights=beam["weight"]))
        axes.axvspan(
            centre - sigma,
            centre + sigma,
            color=colour,
            alpha=0.12,
            label=f"{name} (+/- 1 sigma_z)",
        )
    axes.legend(fontsize=8)
    axes.set_title(
        f"Off-axis transverse wakefield at z = {snapshot['distance'] * 1e3:.2f} mm "
        f"({witness} witness)",
        fontsize=10,
    )
    figure.tight_layout()
    file = output / "offaxis_transverse_wakefield.png"
    figure.savefig(file, dpi=120)
    pyplot.close(figure)
    return file


def _plot_transverse_size(pyplot, rows, output: pathlib.Path) -> pathlib.Path:
    """The witness's transverse size along the run."""
    figure, axes = pyplot.subplots(figsize=(5.5, 3.6))
    axes.plot(rows[0], rows[5], "b-")
    axes.set(
        xlabel="Propagating distance (cm)", ylabel="Transverse size (um)", xlim=(0, 20)
    )
    figure.tight_layout()
    file = output / "transverse_size.png"
    figure.savefig(file, dpi=150)
    pyplot.close(figure)
    return file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--witness", choices=sorted(CASES), default="longer")
    parser.add_argument(
        "--steps",
        type=int,
        help="number of steps (default: the whole cell, CELL_LENGTH / dz)",
    )
    parser.add_argument(
        "--dz",
        type=float,
        default=DZ * 1e6,
        help="longitudinal cell size [um] (the paper's: 0.30)",
    )
    parser.add_argument(
        "--ppc",
        type=int,
        nargs=3,
        default=PLASMA_PPC,
        metavar=("NZ", "NR", "NT"),
        help="plasma macroparticles per cell (the paper's: 4 4 4)",
    )
    parser.add_argument("--plot", action="store_true", help="plot the run when it ends")
    parser.add_argument(
        "--plot-only",
        type=pathlib.Path,
        metavar="DIR",
        help="do not run; plot the dumps of the earlier run in DIR (its working directory)",
    )
    args = parser.parse_args()

    if args.plot_only is not None:
        for path in plot_results(args.plot_only, witness=args.witness):
            print(f"wrote {path}")
        raise SystemExit

    dz = args.dz * 1e-6
    model = build_model(
        witness=args.witness,
        dz=dz,
        n_steps=args.steps,
        plasma_ppc=tuple(args.ppc),
    )
    model.set({})  # runs the simulation (an empty set changes nothing first)
    print(model.get(["charge_pc", "energy_mean_mev", "energy_std_mev"]))
    if args.plot:
        for path in plot_results(
            model.simulator.working_directory, witness=args.witness
        ):
            print(f"wrote {path}")
