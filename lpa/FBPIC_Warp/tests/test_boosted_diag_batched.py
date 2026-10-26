# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It checks that the batched GPU path of BackTransformedParticleDiagnostic
(`gpu_batched=True`) writes the same openPMD data as the legacy path.

- A single boosted-frame simulation (laser ionization of nitrogen, with
  tracked electrons and tracked, ionizable ions) carries legacy and batched
  diagnostics side by side, for several `select` rules. Every h5 dataset
  must be byte-identical between each pair.
- Two separate simulations, one with only legacy diagnostics and one with
  only batched diagnostics, must end with identical particle arrays
  (i.e. the batched path does not change when the particles are sorted).
"""

import math
import os

import h5py
import numpy as np
import pytest
from scipy.constants import c, m_e, m_p, e

from fbpic.main import Simulation
from fbpic.utils.cuda import cuda_installed
from fbpic.lpa_utils.external_fields import ExternalField
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.openpmd_diag import BackTransformedParticleDiagnostic

pytestmark = pytest.mark.skipif(not cuda_installed, reason="requires a GPU")

# Selection rules on the lab-frame quantities. The cold plasma has a lab-frame
# uz of about +-1e-16 (rounding), so the bounds at 0 test the device prefilter
# close to the selection boundary.
SELECT_CASES = {
    "none": None,
    "lower": {"uz": [0.0, None]},
    "upper": {"uz": [None, 0.0]},
    "both": {"uz": [-0.05, 0.05]},
    "mixed": {"uz": [0.0, None], "x": [None, 0.0]},
}
# Cases that also run with a copy floor of 1 row (forces the blocking fallback)
TINY_FLOOR_CASES = ("none", "lower")
GAMMA_BOOST = 2.0


def make_simulation():
    """
    Build a small boosted-frame simulation: a Gaussian laser (external
    fields) ionizes nitrogen atoms. Electrons and ions are tracked.

    Returns
    -------
    sim, species, params: the Simulation, a dict of the species to output,
        and a dict of the parameters needed by the diagnostics
    """
    np.random.seed(0)
    from fbpic.utils.cuda import cupy

    cupy.random.seed(0)

    zmax_lab = 20.0e-6
    zmin_lab = 0.0e-6
    Nr = 3
    rmax = 10.0e-6
    Nm = 2
    p_zmin, p_zmax = 5.0e-6, 15.0e-6
    n_atoms = 0.2
    boost = BoostConverter(GAMMA_BOOST)
    beta_boost = np.sqrt(1.0 - 1.0 / GAMMA_BOOST**2)
    zmin, zmax = boost.static_length([zmin_lab, zmax_lab])
    p_zmin, p_zmax = boost.static_length([p_zmin, p_zmax])
    (n_atoms,) = boost.static_density([n_atoms])
    p_nz = int(2 * GAMMA_BOOST * (1 + beta_boost) * 2)

    a0 = 1.8
    lambda0_lab = 0.8e-6
    (lambda0,) = boost.copropag_length([lambda0_lab], beta_object=1.0)
    ctau = 10.0 * lambda0
    z0 = -2 * ctau
    omega = 2 * np.pi * c / lambda0
    E0 = a0 * m_e * c * omega / e

    def laser_func(F, x, y, z, t, amplitude, length_scale):
        return F + amplitude * math.cos(2 * np.pi * (z - c * t) / lambda0) * math.exp(
            -((z - c * t - z0) ** 2) / ctau**2
        )

    dz = lambda0 / 16.0
    dt = dz / c
    Nz = int((zmax - zmin) / dz) + 1
    N_step = int((2.0 * 40.0 * lambda0 + zmax - zmin) / (dz * (1 + beta_boost))) + 1
    (uz_m,) = boost.longitudinal_momentum([0.0])
    (v_plasma,) = boost.velocity([0.0])

    sim = Simulation(
        Nz,
        zmax,
        Nr,
        rmax,
        Nm,
        dt,
        zmin=zmin,
        v_comoving=v_plasma,
        use_galilean=False,
        boundaries={"z": "open", "r": "reflective"},
        use_cuda=True,
    )
    plasma = dict(
        p_nz=p_nz,
        p_nr=1,
        p_nt=4,
        p_zmin=p_zmin,
        p_zmax=p_zmax,
        p_rmin=0.0,
        p_rmax=100.0e-6,
        continuous_injection=False,
        uz_m=uz_m,
    )
    elec = sim.add_new_species(q=-e, m=m_e, n=2 * n_atoms, **plasma)
    ions = sim.add_new_species(q=0, m=14.0 * m_p, n=n_atoms, **plasma)
    ions.make_ionizable(element="N", level_start=2, target_species=elec)
    elec.track(sim.comm)
    ions.track(sim.comm)
    sim.set_moving_window(v=v_plasma)
    sim.external_fields = [
        ExternalField(laser_func, "Ex", E0, 0.0),
        ExternalField(laser_func, "By", E0 / c, 0.0),
    ]
    T_sim_lab = (2.0 * 40.0 * lambda0_lab + zmax_lab - zmin_lab) / c
    params = dict(
        zmin_lab=zmin_lab,
        zmax_lab=zmax_lab,
        dt_snapshots_lab=T_sim_lab / 5.0,
        N_step=N_step,
    )
    return sim, {"elec": elec, "ions": ions}, params


def make_diag(sim, species, params, write_dir, select, gpu_batched):
    return BackTransformedParticleDiagnostic(
        params["zmin_lab"],
        params["zmax_lab"],
        v_lab=0.0,
        dt_snapshots_lab=params["dt_snapshots_lab"],
        Ntot_snapshots_lab=6,
        gamma_boost=GAMMA_BOOST,
        period=50,
        fldobject=sim.fld,
        select=select,
        species=species,
        comm=sim.comm,
        write_dir=write_dir,
        gpu_batched=gpu_batched,
    )


def read_datasets(write_dir):
    """Return {file/dataset path: numpy array} for all h5 files of a diag."""
    datasets = {}
    hdf5_dir = os.path.join(write_dir, "hdf5")
    for name in sorted(os.listdir(hdf5_dir)):
        with h5py.File(os.path.join(hdf5_dir, name), "r") as f:

            def visit(path, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets[name + "/" + path] = obj[()]

            f.visititems(visit)
    return datasets


def assert_same_datasets(dir_ref, dir_new, rounding_noise=()):
    """
    Assert byte-identical datasets (same names, dtypes, shapes and bytes).
    Datasets whose path ends with an element of `rounding_noise` only need
    to agree within 1e-9 of their largest absolute value.

    Returns
    -------
    The total number of particles (elements of the "position/z" datasets)
    """
    ref = read_datasets(dir_ref)
    new = read_datasets(dir_new)
    assert sorted(ref) == sorted(new)
    n_particles = 0
    for path, data in ref.items():
        other = new[path]
        assert data.dtype == other.dtype, path
        assert data.shape == other.shape, path
        if path.endswith(tuple(rounding_noise)) and data.size > 0:
            scale = np.max(np.abs(data))
            np.testing.assert_allclose(other, data, rtol=0, atol=1.0e-9 * scale)
            continue
        assert np.asarray(data).tobytes() == np.asarray(other).tobytes(), path
        if path.endswith("position/z"):
            n_particles += data.size
    return n_particles


def particle_state(species):
    """Copy the final particle arrays of a species (on the host)."""
    state = {
        name: np.array(getattr(species, name))
        for name in ("x", "y", "z", "ux", "uy", "uz", "w", "inv_gamma")
    }
    if species.tracker is not None:
        state["id"] = np.array(species.tracker.id)
    if species.ionizer is not None:
        state["level"] = np.array(species.ionizer.ionization_level)
        state["w_times_level"] = np.array(species.ionizer.w_times_level)
    return state


def test_batched_matches_legacy_same_simulation(tmp_path):
    """Legacy and batched diagnostics in one simulation write equal data."""
    sim, species, params = make_simulation()
    pairs = []
    for case, select in SELECT_CASES.items():
        legacy_dir = str(tmp_path / ("legacy_" + case))
        legacy = make_diag(sim, species, params, legacy_dir, select, False)
        sim.diags.append(legacy)
        variants = [("batched", 256, True)]
        if case in TINY_FLOOR_CASES:
            variants.append(("tiny", 1, False))
        for label, floor, adaptive in variants:
            batched_dir = str(tmp_path / (label + "_" + case))
            batched = make_diag(sim, species, params, batched_dir, select, True)
            batched.copy_floor = floor
            batched.adaptive_copy = adaptive
            sim.diags.append(batched)
            pairs.append((case, label, legacy_dir, batched_dir, batched))

    sim.step(params["N_step"], use_true_rho=True)
    # Write the slices registered after the last periodic flush
    # (and, for the batched diagnostics, the slices still in flight)
    for diag in sim.diags:
        diag.flush_to_disk()

    counts = {}
    for case, label, legacy_dir, batched_dir, batched in pairs:
        n_particles = assert_same_datasets(legacy_dir, batched_dir)
        assert n_particles > 0, (case, label)
        counts[case] = n_particles
        if label == "tiny":
            fallbacks = sum(s.fallbacks for s in batched._staging.values())
            print("Blocking fallback copies (%s): %d" % (case, fallbacks))
            assert fallbacks > 0, case
    print("Particles written per case: %s" % counts)
    # The selections remove particles, and both sides of uz = 0 are populated
    for case in ("lower", "upper", "both", "mixed"):
        assert counts[case] < counts["none"], case
    assert counts["lower"] + counts["upper"] <= counts["none"]


def test_batched_does_not_change_simulation(tmp_path):
    """Runs with only legacy or only batched diagnostics end with equal particles."""
    final = {}
    for gpu_batched in (False, True):
        sim, species, params = make_simulation()
        for case in ("none", "lower"):
            write_dir = str(tmp_path / ("%s_%s" % (gpu_batched, case)))
            sim.diags.append(
                make_diag(
                    sim, species, params, write_dir, SELECT_CASES[case], gpu_batched
                )
            )
        sim.step(params["N_step"], use_true_rho=True)
        for diag in sim.diags:
            diag.flush_to_disk()
        final[gpu_batched] = {name: particle_state(sp) for name, sp in species.items()}

    # Any change in the sort sequence permutes the particle arrays, which the
    # exact comparisons detect. Deposition uses GPU atomics, so two identical
    # runs can differ by rounding: here, only uy (about 1e-25, zero up to
    # rounding) differs, by about 1e-38 in repeated legacy runs.
    for name in final[False]:
        ref, new = final[False][name], final[True][name]
        assert sorted(ref) == sorted(new)
        for quantity in ref:
            assert ref[quantity].shape == new[quantity].shape, (name, quantity)
            if quantity == "uy":
                scale = np.max(np.abs(ref[quantity]))
                np.testing.assert_allclose(
                    new[quantity], ref[quantity], rtol=0, atol=1.0e-9 * scale
                )
            else:
                assert np.array_equal(ref[quantity], new[quantity]), (name, quantity)
    for case in ("none", "lower"):
        assert_same_datasets(
            str(tmp_path / ("False_" + case)),
            str(tmp_path / ("True_" + case)),
            rounding_noise=("momentum/y",),
        )
