"""Tests for beam-driven (PWFA) runs on `FBPICSimulator`: their actions and archives, short real
runs, the plasma profile of the CLARA example, and the two PWFA examples."""

from __future__ import annotations

import attrs
import h5py
import numpy
import pytest
from scipy.constants import e

from inversion_fbpic.lib.particle_beam import ExpressionParticleBeam
from inversion_fbpic.lib.serializable_config import SerializableConfig
from lume_fbpic import archive
from lume_fbpic.actions import make_beam_actions
from lume_fbpic.density_profiles import UpDownRampProfile
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.simulator import FBPICSimulator
from lume_fbpic.utils import ELECTRON_MC2_EV
from reference_descriptor import reference_descriptor
from short_runs import short_pwfa_model, short_pwfa_simulator, uniform_beam_charge

import pwfa_clara_febe as clara_example  # docs/examples, on the path from conftest.py
import pwfa_gaussian as pwfa_gaussian_example


@pytest.fixture
def pwfa_model(tmp_path) -> LUMEFBPICModel:
    """`short_pwfa_model(tmp_path)` (a flat-top driver, no witness), `dummy_run`, so nothing
    runs."""
    return short_pwfa_model(tmp_path, dummy_run=True)


def test_configure_rejects_a_target_species_that_does_not_exist(pwfa_model):
    pwfa_model.simulator.target_species = "witness"

    with pytest.raises(ValueError, match="target_species"):
        pwfa_model.simulator.configure()


def test_the_model_gets_the_pwfa_actions(pwfa_model):
    assert {"driver_gamma", "driver_density", "driver_radius", "plasma_density"} <= set(
        pwfa_model.supported_variables
    )
    assert "witness_gamma" not in pwfa_model.supported_variables
    assert "laser_energy" not in pwfa_model.supported_variables


def test_the_descriptor_actions_work_on_a_pwfa_simulator(pwfa_model, particle_group):
    from lume_fbpic.actions import make_descriptor_actions

    model = LUMEFBPICModel(
        pwfa_model.simulator, make_descriptor_actions(uz_min=1.0), dummy_run=True
    )
    pwfa_model.simulator.final_particles = particle_group

    values = model.get(["descriptor_mean_uz", "descriptor_total_beam_charge_pc"])

    expected = reference_descriptor(particle_group, uz_min=1.0, central_fraction=0.95)
    assert values == {
        "descriptor_mean_uz": expected["mean_uz"],
        "descriptor_total_beam_charge_pc": expected["total_beam_charge_pc"],
    }


def test_a_witness_adds_witness_actions(pwfa_model):
    simulator = pwfa_model.simulator
    simulator.beams = [
        *simulator.beams,
        ExpressionParticleBeam(
            gamma0=500.0,
            n=1.0e23,
            p_zmin=0.0,
            p_zmax=1.0e-6,
            p_rmax=1.0e-6,
            particle_name="witness",
        ),
    ]
    simulator.target_species = "witness"

    model = LUMEFBPICModel(
        simulator, make_beam_actions(simulator, beam_index=1), dummy_run=True
    )

    assert {"witness_gamma", "witness_density"} <= set(model.supported_variables)


def test_setting_actions_replaces_the_frozen_configs(pwfa_model):
    original_driver = pwfa_model.simulator.beams[0]

    pwfa_model.set({"driver_gamma": 1000.0, "plasma_density": 2.0e24})

    assert pwfa_model.simulator.beams[0] is not original_driver
    assert pwfa_model.simulator.beams[0].gamma0 == 1000.0
    assert pwfa_model.simulator.densities[0].nominal_density == 2.0e24
    assert pwfa_model.get(["driver_gamma"]) == {"driver_gamma": 1000.0}


def test_beam_actions_come_back_from_an_archive(pwfa_model, tmp_path):
    archive.save(pwfa_model, tmp_path / "p.h5")

    restored = archive.load(tmp_path / "p.h5")

    action = restored.supported_variables["driver_gamma"]
    assert type(action).__name__ == "BeamFieldAction"
    assert action.beam_index == 0 and action.field_name == "gamma0"


def test_archive_round_trips_a_pwfa_model(pwfa_model, tmp_path):
    simulator = pwfa_model.simulator
    simulator.hyparams = attrs.evolve(simulator.hyparams, save_directory="scan_07")
    archive.save(pwfa_model, tmp_path / "p.h5")

    with h5py.File(tmp_path / "p.h5") as f:
        assert f.attrs["config_kind"] == "fbpic"
    restored = archive.load(tmp_path / "p.h5")

    assert type(restored.simulator) is FBPICSimulator

    def dicts(config):
        return {
            k: [x.to_dict() for x in v] if isinstance(v, list) else v.to_dict()
            for k, v in config.items()
        }

    assert dicts(restored.simulator.config()) == dicts(pwfa_model.simulator.config())
    assert restored.get(["driver_gamma"]) == pwfa_model.get(["driver_gamma"])


def test_a_short_run_writes_the_driver_and_reads_it_back(tmp_path):
    simulator = short_pwfa_simulator(tmp_path)

    simulator.run()

    driver = simulator.beams[0]
    assert simulator.final_particles is not None
    assert (tmp_path / "diags" / "hdf5" / "data00000002.h5").exists()
    # the charge is the bunch's to rounding; 3 steps of wake change its energy by about 2e-7
    assert simulator.stats["charge_pc"] == pytest.approx(
        uniform_beam_charge(driver) * 1e12, rel=1e-12
    )
    assert simulator.stats["energy_mean_mev"] == pytest.approx(
        (driver.gamma0 - 1) * ELECTRON_MC2_EV / 1e6, rel=1e-6
    )


def test_the_start_middle_and_end_of_a_run_are_dumped_and_the_end_read_back(tmp_path):
    simulator = short_pwfa_simulator(tmp_path)  # 3 steps, 3 dumps

    simulator.run()

    directory = tmp_path / "diags" / "hdf5"
    assert sorted(path.name for path in directory.glob("data*.h5")) == [
        "data00000000.h5",
        "data00000002.h5",
        "data00000003.h5",
    ]
    with h5py.File(directory / "data00000003.h5") as f:
        z = f["data/3/particles/driver/position/z"][:]
    numpy.testing.assert_array_equal(simulator.final_particles.z, z)


def test_a_run_without_plasma_output_dumps_only_the_beams(tmp_path):
    simulator = short_pwfa_simulator(tmp_path)  # the plasma has no elec_name

    simulator.run()

    with h5py.File(tmp_path / "diags" / "hdf5" / "data00000002.h5") as f:
        assert sorted(f["data/2/particles"]) == ["driver"]


def test_the_gaussian_example_has_a_witness_and_gaussian_actions():
    model = pwfa_gaussian_example.build_model(dummy_run=True)
    simulator = model.simulator
    h = simulator.hyparams

    names = set(model.supported_variables)
    assert {
        "plasma_density",
        "driver_charge",
        "driver_sigma_z",
        "driver_position",
        "witness_charge",
    } <= names
    assert "driver_density" not in names
    assert simulator.target_species == "witness"
    # dz of about 1 um and dr of 0.5 um: the witness spans 4 radial and 6 longitudinal cells
    dz, dr = (h.zmax - h.zmin) / h.nz, h.rmax / h.nr
    assert (h.nz, h.nr) == (336, 213)
    assert simulator.beams[1].sig_r / dr == pytest.approx(4.0, rel=0.01)
    assert simulator.beams[1].sig_z / dz == pytest.approx(6.0, rel=0.01)
    assert h.dt == pytest.approx(dz / 299792458.0, rel=1e-12)
    # by default the witness travels the quick preset's 2 mm in plasma, which starts at the box's
    # front
    witness_to_plasma = h.zmax - simulator.beams[1].zf
    assert h.simulation_length - witness_to_plasma == pytest.approx(2.0e-3)
    assert simulator.densities[0].ramp_start == h.zmax

    model.set({"witness_position": 1.0e-4})
    assert simulator.beams[1].zf == 1.0e-4


def test_the_gaussian_examples_beams_are_matched_to_the_ion_column():
    from scipy.constants import epsilon_0, m_e

    simulator = pwfa_gaussian_example.build_model(dummy_run=True).simulator
    n_p = simulator.densities[0].nominal_density
    k_p = numpy.sqrt(n_p * e**2 / (m_e * epsilon_0)) / 299792458.0

    for beam in simulator.beams:
        # sigma_r = sqrt(beta_m eps_n / gamma) with beta_m = sqrt(2 gamma) / k_p
        beta_m = numpy.sqrt(2.0 * beam.gamma0) / k_p
        assert beam.n_emit == pytest.approx(
            beam.sig_r**2 * beam.gamma0 / beta_m, rel=1e-12
        )
    assert pwfa_gaussian_example.PLASMA_LENGTHS["long"] == pytest.approx(
        2.0 * numpy.pi * numpy.sqrt(2.0 * simulator.beams[1].gamma0) / k_p, rel=1e-12
    )


def test_the_gaussian_examples_length_is_the_witnesss_travel_in_plasma():
    for name, length in pwfa_gaussian_example.PLASMA_LENGTHS.items():
        h = pwfa_gaussian_example.build_model(
            dummy_run=True, plasma_length=length
        ).simulator.hyparams
        assert h.simulation_length == pytest.approx(
            length + h.zmax - pwfa_gaussian_example.WITNESS_Z
        ), name


def _short_gaussian_example(tmp_path) -> FBPICSimulator:
    """The Gaussian example, 3 steps on a 40 x 16 grid with few macroparticles, on the CPU, in
    `tmp_path`."""
    simulator = pwfa_gaussian_example.build_model().simulator
    simulator.working_directory = tmp_path
    h = simulator.hyparams
    dz = (h.zmax - h.zmin) / 40
    simulator.hyparams = attrs.evolve(
        h,
        nz=40,
        nr=16,
        number_dumps=3,
        # POSIT: the lab-frame step is one cell, dz / c, and the window moves at c; half a cell
        # of right_buffer keeps the step count from rounding down
        simulation_length=3 * dz,
        right_buffer=0.5 * dz,
        use_cuda=False,
    )
    driver, witness = simulator.beams
    simulator.beams = [
        attrs.evolve(driver, n_macroparticles=200),
        attrs.evolve(witness, n_macroparticles=100),
    ]
    return simulator


def test_a_short_run_with_gaussian_bunches_reads_the_witness_back(tmp_path):
    simulator = _short_gaussian_example(tmp_path)

    simulator.run()

    witness = simulator.beams[1]
    assert simulator.stats["charge_pc"] == pytest.approx(
        witness.n_physical_particles * e * 1e12, rel=1e-12
    )
    # the mean of the witness's 100 sampled energies, within 5 standard errors
    assert simulator.stats["energy_mean_mev"] == pytest.approx(
        (witness.gamma0 - 1) * ELECTRON_MC2_EV / 1e6,
        abs=5 * witness.sig_gamma * ELECTRON_MC2_EV / 1e6 / numpy.sqrt(100),
    )
    assert len(simulator.final_particles) == 100


def test_the_gaussian_example_plots_the_fields_of_a_short_run(tmp_path):
    image = pytest.importorskip("matplotlib.image")
    simulator = _short_gaussian_example(tmp_path)
    simulator.run()

    files = pwfa_gaussian_example.plot_results(tmp_path, output=tmp_path / "plots")

    assert [f.name for f in files] == [
        "electron_density.png",
        "longitudinal_field.png",
        "transverse_force.png",
        "longitudinal_field_lineout.png",
    ]
    for f in files:
        assert image.imread(f).ndim == 3, f.name  # a readable image
    with pytest.raises(FileNotFoundError, match="no dumps"):
        pwfa_gaussian_example.plot_results(tmp_path / "plots")


def test_the_gaussian_example_analyzes_the_witness_of_a_short_run(tmp_path):
    image = pytest.importorskip("matplotlib.image")
    simulator = _short_gaussian_example(tmp_path)
    simulator.run()

    files = pwfa_gaussian_example.analyze_witness(
        tmp_path, output=tmp_path / "analysis"
    )

    assert [f.name for f in files] == ["witness_summary.txt", "witness_analysis.png"]
    assert "EMITTANCE" in files[0].read_text()
    assert image.imread(files[1]).ndim == 3  # a readable image
    # the plots read the fields, so the plasma's macroparticles are not written
    with h5py.File(tmp_path / "diags" / "hdf5" / "data00000003.h5") as f:
        assert sorted(f["data/3/particles"]) == ["driver", "witness"]
    with pytest.raises(FileNotFoundError, match="no dumps"):
        pwfa_gaussian_example.analyze_witness(tmp_path / "analysis")


def _up_down_ramp(**changes) -> UpDownRampProfile:
    kwargs = dict(
        nominal_density=4.2e22,
        species=None,
        ionization=-1,
        p_nz=4,
        p_nr=4,
        p_nt=4,
        base_fraction=2.0 / 3.0,
        z_up=0.05,
        up_length=0.01,
        z_down=0.15,
        down_length=0.01,
        z_end=0.2,
    )
    return UpDownRampProfile(**{**kwargs, **changes})


def test_up_down_ramp_has_five_stages_and_no_plasma_outside_the_cell():
    import numpy

    dens = _up_down_ramp().build_density_function()
    z = numpy.array(
        [-1.0e-3, 0.0, 0.04, 0.055, 0.06, 0.1, 0.155, 0.16, 0.18, 0.2, 0.25]
    )

    relative = dens(z, numpy.zeros_like(z))

    numpy.testing.assert_allclose(
        relative, [0, 2 / 3, 2 / 3, 5 / 6, 1, 1, 5 / 6, 2 / 3, 2 / 3, 0, 0], atol=1e-12
    )


def test_up_down_ramp_has_no_end_by_default_and_round_trips_through_yaml():
    import numpy

    profile = _up_down_ramp(z_end=None)

    assert profile.build_density_function()(numpy.array([10.0]), numpy.array([0.0]))[
        0
    ] == pytest.approx(2 / 3)
    loaded = SerializableConfig.from_yaml(profile.to_yaml())
    assert loaded.to_dict() == profile.to_dict()


@pytest.mark.parametrize(
    "changes",
    [{"z_up": -0.1}, {"z_down": 0.055}, {"z_end": 0.155}, {"base_fraction": 1.5}],
)
def test_up_down_ramp_rejects_inconsistent_stages(changes):
    with pytest.raises(ValueError):
        _up_down_ramp(**changes)


@pytest.mark.parametrize(
    "witness, n0_fraction", [("longer", 2.8 / 4.2), ("shorter", 1.9 / 4.2)]
)
def test_the_clara_example_follows_the_papers_tables(witness, n0_fraction):
    model = clara_example.build_model(witness=witness, dummy_run=True)
    simulator = model.simulator
    driver, witness_beam = simulator.beams

    assert simulator.hyparams.nr == 200 and simulator.hyparams.nm == 1
    assert simulator.hyparams.r_boundary == "reflective"  # z is always open
    assert simulator.densities[0].base_fraction == pytest.approx(n0_fraction)
    assert driver.n_physical_particles * e == pytest.approx(150.0e-12)
    assert driver.zf - witness_beam.zf == pytest.approx(90.0e-6)
    assert simulator.target_species == "witness"


@pytest.mark.parametrize(
    "arguments, nz, n_steps, time_step, plasma_ppc",
    [
        (
            {},
            100,
            133_333,
            1.5e-6 / 299792458.0,
            (2, 2, 4),
        ),  # the coarser 20 cm run; the paper: 500, about 1 fs, (4, 4, 4)
        (
            {"dz": 0.30e-6, "plasma_ppc": (4, 4, 4)},
            500,
            666_667,
            0.30e-6 / 299792458.0,
            (4, 4, 4),
        ),  # the paper's
    ],
    ids=["defaults", "the_papers_resolution"],
)
def test_the_clara_example_resolution(arguments, nz, n_steps, time_step, plasma_ppc):
    simulator = clara_example.build_model(dummy_run=True, **arguments).simulator
    h = simulator.hyparams
    plasma = simulator.densities[0]

    assert h.nz == nz
    assert h.dt == pytest.approx(time_step, rel=1e-12)  # dz / c
    # the whole 20 cm cell: the window moves n_steps cells, plus half a cell of right_buffer
    assert round(h.simulation_length / (h.dt * 299792458.0)) == n_steps
    assert h.right_buffer == pytest.approx(0.5 * h.dt * 299792458.0)
    assert (plasma.p_nz, plasma.p_nr, plasma.p_nt) == plasma_ppc
    assert plasma.elec_name is None  # the plasma is not dumped


def _write_witness_dump(directory, iteration, time, *, outlier=False):
    """A fbpic-style dump at `time` [s]: a witness of 1000 equal-weight electrons (with `outlier`,
    one far outside the 5-sigma cut), a driver of 500, and small field arrays."""
    from scipy.constants import c, e, m_e

    rng = numpy.random.default_rng(iteration)
    path = directory / "diags" / "hdf5" / f"data{iteration:08d}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        step = f.create_group(f"data/{iteration}")
        step.attrs["time"] = time
        for name, n, z_centre, sigma_r, sigma_z in (
            ("witness", 1000, -90.0e-6, 14.0e-6, 10.0e-6),
            ("driver", 500, -30.0e-6, 70.0e-6, 10.0e-6),
        ):
            x, y = rng.normal(0.0, sigma_r, n), rng.normal(0.0, sigma_r, n)
            z = rng.normal(z_centre, sigma_z, n)
            ux, uy, uz = (
                rng.normal(0.0, 0.01, n),
                rng.normal(0.0, 0.01, n),
                rng.normal(500.0, 5.0, n),
            )
            if outlier and name == "witness":
                x[0] = 1.0e-3
            particles = step.create_group(f"particles/{name}")
            for axis, values in (("x", x), ("y", y), ("z", z)):
                particles[f"position/{axis}"] = values
            for axis, values in (("x", ux), ("y", uy), ("z", uz)):
                particles[f"momentum/{axis}"] = values * m_e * c
            particles["weighting"] = numpy.full(n, 1.0e-12 / e / n)
        mesh = step.create_group("fields/E")
        mesh.attrs["gridSpacing"] = [0.75e-6, 1.5e-6]
        mesh.attrs["gridGlobalOffset"] = [0.0, -150.0e-6]
        for name in ("r", "t", "z"):
            mesh[name] = rng.normal(size=(1, 12, 100)) * 1.0e8
        step.create_group("fields/B")["t"] = rng.normal(size=(1, 12, 100))
        step["fields/rho"] = rng.normal(size=(1, 12, 100))
    return path


def test_the_dump_metrics_follow_the_papers_cut(tmp_path):
    from beamphysics import ParticleGroup
    from scipy.constants import e

    path = _write_witness_dump(tmp_path, 10, 1.0e-10, outlier=True)

    metrics = clara_example._dump_metrics(path, clara_example.CASES["longer"])

    # The same statistics of the witness without its outlier (particle 0, the only one outside
    # the cut), from beamphysics.
    with h5py.File(path) as f:
        witness = f["data/10/particles/witness"]
        kept = {
            name: witness[name][1:]
            for name in (
                "position/x",
                "position/y",
                "position/z",
                "momentum/x",
                "momentum/y",
                "momentum/z",
                "weighting",
            )
        }
    momentum_ev = 1 / e * 299792458.0  # kg m/s -> eV/c
    particles = ParticleGroup(
        data={
            **{axis: kept[f"position/{axis}"] for axis in "xyz"},
            **{f"p{axis}": kept[f"momentum/{axis}"] * momentum_ev for axis in "xyz"},
            "t": numpy.zeros(999),
            "status": numpy.ones(999, dtype=int),
            "weight": kept["weighting"] * e,
            "species": "electron",
        }
    )
    expected = [
        100.0 * 299792458.0 * 1.0e-10,
        particles["mean_energy"] / 1e6,
        100.0 * particles["sigma_energy"] / particles["mean_energy"],
        particles.charge * 1e12,
        0.5 * (particles["norm_emit_x"] + particles["norm_emit_y"]) * 1e6,
        particles["sigma_x"] * 1e6,
    ]
    # The example converts the momenta with beamphysics's electron rest energy, which differs
    # from CODATA's m_e c^2 / e, used here, by 3.4e-12.
    numpy.testing.assert_allclose(metrics, expected, rtol=1e-11)


def test_plot_results_writes_the_four_plots_from_the_dumps(tmp_path):
    image = pytest.importorskip("matplotlib.image")
    for iteration, time in ((0, 0.0), (10, 1.0e-10), (20, 2.0e-10)):
        _write_witness_dump(tmp_path, iteration, time)
    expected = [
        "transverse_size.png",
        "beam_quality_with_ramp_and_500MeV_markers.png",
        "beam_density_and_wakefield.png",
        "offaxis_transverse_wakefield.png",
    ]

    files = clara_example.plot_results(
        tmp_path, witness="longer", output=tmp_path / "plots"
    )

    assert [f.name for f in files] == expected
    for f in files:
        assert image.imread(f).ndim == 3, f.name  # a readable image
    shorter = clara_example.plot_results(
        tmp_path, witness="shorter", output=tmp_path / "plots_shorter"
    )
    assert [f.name for f in shorter] == expected


def test_plot_results_needs_dumps(tmp_path):
    with pytest.raises(FileNotFoundError, match="no dumps"):
        clara_example.plot_results(tmp_path)
