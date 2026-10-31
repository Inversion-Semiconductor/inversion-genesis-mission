"""Twin configurations for the differentiable inversion (plan Sections 3 and 7).

`build_twin` builds a single-GPU, lab-frame FBPIC `Simulation` from a frozen
JSON file in `inversion/configs/`. The parameters are the laser energy E_L and
the laser focal position z_f (D21). Everything else is fixed by the file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.constants import c, e, epsilon_0, m_e, m_p

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
PARAMETER_NAMES = ("laser_energy_J", "laser_focal_position_m")
IONIZATION_MODES = ("none", "ev", "stochastic")
PLASMA_STARTS = ("injected", "preloaded")


def load_config(size: str) -> dict:
    """Load `inversion/configs/<size>.json`."""
    with open(CONFIG_DIR / f"{size}.json") as handle:
        return json.load(handle)


def config_sha256(config: dict) -> str:
    """Hash of the canonical JSON text of a configuration."""
    text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


# Parameters
# ----------


def bounds(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper bounds of theta = (E_L [J], z_f [m])."""
    theta = config["theta"]
    return np.asarray(theta["lo"], float), np.asarray(theta["hi"], float)


def theta_from_hat(config: dict, theta_hat) -> np.ndarray:
    lo, hi = bounds(config)
    return lo + np.asarray(theta_hat, float) * (hi - lo)


def hat_from_theta(config: dict, theta) -> np.ndarray:
    lo, hi = bounds(config)
    return (np.asarray(theta, float) - lo) / (hi - lo)


def a0_from_energy(energy, lambda0, w0, tau):
    """Peak a0 of a Gaussian pulse of energy `energy` [J].

    Uses the template formulas `peak_power = E / tau / sqrt(pi/2)` and
    `I = P / (pi w0^2 / 2)`. `tau` is the FBPIC field duration (1/e of the
    field), so a0 scales as sqrt(energy) at fixed tau, w0, and lambda0.
    """
    peak_power = energy / tau / np.sqrt(np.pi / 2.0)
    peak_intensity = peak_power / (np.pi * w0**2 / 2.0)
    peak_field = np.sqrt(2.0 * peak_intensity / (c * epsilon_0))
    return peak_field / (m_e * c**2 * (2 * np.pi / lambda0) / e)


def energy_from_a0(a0, lambda0, w0, tau):
    """Inverse of `a0_from_energy`."""
    return a0**2 / a0_from_energy(1.0, lambda0, w0, tau) ** 2


def rayleigh_length(config: dict) -> float:
    laser = config["laser"]
    return np.pi * laser["w0"] ** 2 / laser["lambda0"]


# Density
# -------


def density_shape(z, plasma: dict):
    """Relative density S(z) times the indicator of [z_lo, z_hi] (Section 2.1).

    S is the bilateral super-Gaussian of the template: F_L and beta for
    z < z_c, F_R and beta otherwise. S(z_c) = 1.
    """
    z = np.asarray(z, float)
    zc, beta = plasma["z_c"], plasma["beta"]
    fwhm = np.where(z < zc, plasma["F_L"], plasma["F_R"])
    s = np.exp(-np.log(2.0) * (4.0 * (z - zc) ** 2 / fwhm**2) ** (beta / 2.0))
    inside = (z >= plasma["z_lo"]) & (z <= plasma["z_hi"])
    return np.where(inside, s, 0.0)


def density_shape_dF_L(z, plasma: dict):
    """Closed-form dS/dF_L (Section 2.1); zero for z >= z_c and outside."""
    z = np.asarray(z, float)
    zc, beta, fl = plasma["z_c"], plasma["beta"], plasma["F_L"]
    s = density_shape(z, plasma)
    x = (4.0 * (z - zc) ** 2 / fl**2) ** (beta / 2.0)
    return np.where(z < zc, s * np.log(2.0) * beta * x / fl, 0.0)


def density_shape_dn0(z, plasma: dict):
    """d(n0 S)/d n0 divided by n0, so that dw/dn0 = w / n0 (Section 2.1)."""
    return density_shape(z, plasma) / plasma["n_e0"]


def species_densities(plasma: dict) -> dict:
    """Peak densities [m^-3] of He2+, N5+, and the loaded electrons."""
    n_atoms = plasma["n_e0"] / (
        2.0 * (1.0 - plasma["nitrogen_fraction"]) + 5.0 * plasma["nitrogen_fraction"]
    )
    n_he = (1.0 - plasma["nitrogen_fraction"]) * n_atoms
    n_n = plasma["nitrogen_fraction"] * n_atoms
    return {"He": n_he, "N": n_n, "e": 2.0 * n_he + 5.0 * n_n}


def make_dens_func(plasma: dict):
    def dens_func(z, r):
        return density_shape(z, plasma)

    return dens_func


# Simulation
# ----------


@dataclass
class Twin:
    """A built twin: the simulation, its species, and its settings."""

    sim: object
    electrons: object
    helium: object
    nitrogen: object
    elec_from_N: object
    config: dict
    theta: np.ndarray
    ionization: str
    laser_profile: object

    @property
    def species(self) -> dict:
        return {
            "electrons": self.electrons,
            "helium": self.helium,
            "nitrogen": self.nitrogen,
            "elec_from_N": self.elec_from_N,
        }


def laser_profile(config: dict, theta):
    """The direct-method Gaussian laser for theta = (E_L, z_f)."""
    from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser

    laser = config["laser"]
    a0 = a0_from_energy(theta[0], laser["lambda0"], laser["w0"], laser["tau"])
    return GaussianLaser(
        a0,
        laser["w0"],
        laser["tau"],
        laser["z0"],
        zf=float(theta[1]),
        theta_pol=laser["theta_pol"],
        lambda0=laser["lambda0"],
    )


def build_twin(
    theta,
    size: str | dict,
    ionization: str = "none",
    plasma_start: str | None = None,
    p_nz: int = 1,
    seed: int | None = None,
    rng_seed: int | None = None,
    use_cuda_graphs: bool = False,
) -> Twin:
    """Build a twin simulation at theta = (E_L [J], z_f [m]).

    Args:
        theta: Laser energy and focal position, in SI units.
        size: Configuration name (`twin_xs`, `twin_s`) or a loaded config.
        ionization: "none" loads N as fixed-charge 5+ ions without an
            ionizer. "ev" also loads fixed-charge ions; the autodiff stepper
            adds the expected-value model. "stochastic" uses production
            `make_ionizable` with `level_start=5`, a tracker, `rng_seed`,
            and `rng_stream`.
        plasma_start: "injected" or "preloaded". Defaults to the config value.
            Both use the same `dens_func`; "injected" loads nothing at t = 0.
        p_nz: Particles per cell along z (truth variants use 2).
        seed: NumPy seed for the injection angles. Defaults to the config
            inversion seed.
        rng_seed: Philox seed of the stochastic ionizer. Defaults to the
            config value.
        use_cuda_graphs: Production CUDA graphs. Off for every
            differentiable run (D9).

    Returns:
        The built `Twin`. Every species is tracked.
    """
    from fbpic.lpa_utils.laser import add_laser_pulse
    from fbpic.main import Simulation
    from fbpic.utils.random_seed import set_random_seed

    config = load_config(size) if isinstance(size, str) else size
    if ionization not in IONIZATION_MODES:
        raise ValueError(f"ionization must be one of {IONIZATION_MODES}")
    plasma_start = plasma_start or config["plasma"]["start"]
    if plasma_start not in PLASMA_STARTS:
        raise ValueError(f"plasma_start must be one of {PLASMA_STARTS}")
    theta = np.asarray(theta, float)
    grid, plasma, seeds = config["grid"], config["plasma"], config["seeds"]

    set_random_seed(seeds["numpy"] if seed is None else seed)
    dz = (grid["zmax"] - grid["zmin"]) / grid["Nz"]
    sim = Simulation(
        grid["Nz"],
        grid["zmax"],
        grid["Nr"],
        grid["rmax"],
        grid["Nm"],
        grid["dt_cells"] * dz / c,
        zmin=grid["zmin"],
        n_order=grid["n_order"],
        n_guard=grid["n_guard"],
        exchange_period=grid["exchange_period"],
        use_cuda=True,
        boundaries={"z": "open", "r": "reflective"},
        particle_shape="linear",
        filter_currents=True,
        current_correction="curl-free",
        hankel_precision=grid["hankel_precision"],
        use_cuda_graphs=use_cuda_graphs,
        verbose_level=0,
    )

    dens = species_densities(plasma)
    dens_func = make_dens_func(plasma)
    if plasma_start == "injected":
        # Nothing at t = 0: the window injects the whole plasma.
        p_zmin = max(plasma["z_lo"], grid["zmax"])
    else:
        p_zmin = plasma["z_lo"]
    load = {
        "dens_func": dens_func,
        "p_nz": p_nz,
        "p_nr": plasma["p_nr"],
        "p_nt": plasma["p_nt"],
        "p_zmin": p_zmin,
        "p_zmax": plasma["z_hi"],
        "p_rmax": plasma["p_rmax"],
    }
    n_load, e_load = load, load
    n_range = plasma.get("nitrogen_z_range")
    if n_range is not None:
        # Reduced variant (Phase P3 step 6): nitrogen only inside n_range.
        # The electron density follows, so that the plasma stays neutral.
        def in_range(z):
            return (np.asarray(z) >= n_range[0]) & (np.asarray(z) <= n_range[1])

        f_n = 5.0 * dens["N"] / dens["e"]
        n_load = {**load, "dens_func": lambda z, r: dens_func(z, r) * in_range(z)}
        e_load = {
            **load,
            "dens_func": lambda z, r: dens_func(z, r) * (1.0 - f_n + f_n * in_range(z)),
        }
    electrons = sim.add_new_species(q=-e, m=m_e, n=dens["e"], **e_load)
    helium = sim.add_new_species(q=2 * e, m=4.0 * m_p, n=dens["He"], **load)
    nitrogen = sim.add_new_species(q=5 * e, m=14.0 * m_p, n=dens["N"], **n_load)
    elec_from_N = sim.add_new_species(q=-e, m=m_e)
    for species in (electrons, helium, nitrogen, elec_from_N):
        species.track(sim.comm)
    if ionization == "stochastic":
        nitrogen.make_ionizable(
            "N",
            target_species=elec_from_N,
            level_start=5,
            rng_seed=seeds["ionization_rng"] if rng_seed is None else rng_seed,
            rng_stream=seeds["ionization_stream"],
        )

    profile = laser_profile(config, theta)
    add_laser_pulse(sim, profile, method="direct")
    sim.set_moving_window(v=c)
    return Twin(
        sim,
        electrons,
        helium,
        nitrogen,
        elec_from_N,
        config,
        theta,
        ionization,
        profile,
    )


def species_arrays(species, names=("x", "y", "z", "ux", "uy", "uz", "w")) -> dict:
    """Host copies of the first Ntot entries of particle arrays."""
    import cupy

    n = int(species.Ntot)
    out = {name: cupy.asnumpy(getattr(species, name)[:n]) for name in names}
    if species.tracker is not None:
        out["id"] = cupy.asnumpy(species.tracker.id[:n])
    return out
