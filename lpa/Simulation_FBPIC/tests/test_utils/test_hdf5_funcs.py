"""
Unit test which creates a small h5 file and filters out the two electrons with lower energy than the threshold
"""

import os
import tempfile
import numpy as np
import h5py
from pathlib import Path
from scipy.constants import m_e, c, e
from inversion_fbpic.utils.hdf5_funcs import ebeam_extract_particles_only


def uz_for_ke_mev(ke_mev):
    ke_joule = ke_mev * 1e6 * e
    total_e = ke_joule + m_e * c**2
    uz = np.sqrt(total_e**2 - (m_e * c**2) ** 2) / c
    return uz


def create_minimal_fbpic_h5(filename, species_name="electrons"):
    with h5py.File(filename, "w") as f:
        # FBPIC-like structure: data/00000001/particles/electrons/...
        g_data = f.create_group("data")
        g_iter = g_data.create_group("00000001")
        g_particles = g_iter.create_group(f"particles/{species_name}")
        # Minimal arrays: two below and two above 25 MeV
        z = np.array([0.1, 0.2, 0.3, 0.4])
        uz = np.array(
            [
                uz_for_ke_mev(10),  # below threshold
                uz_for_ke_mev(20),  # below threshold
                uz_for_ke_mev(30),  # above threshold
                uz_for_ke_mev(40),  # above threshold
            ]
        )
        g_particles.create_dataset("position/z", data=z)
        g_particles.create_dataset("momentum/z", data=uz)
        g_particles.create_dataset("position/x", data=np.zeros_like(z))
        g_particles.create_dataset("position/y", data=np.zeros_like(z))
        g_particles.create_dataset("momentum/x", data=np.zeros_like(z))
        g_particles.create_dataset("momentum/y", data=np.zeros_like(z))
        g_particles.create_dataset("weighting", data=np.ones_like(z))
        g_particles.attrs["numParticles"] = len(z)


def test_ebeam_extract_filters_and_copies():
    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = os.path.join(tmpdir, "test_fbpic.h5")
        dst_dir = os.path.join(tmpdir, "out")
        os.makedirs(dst_dir, exist_ok=True)
        create_minimal_fbpic_h5(src_file)
        # Only keep electrons with KE >= 25 MeV
        ebeam_extract_particles_only(
            source=Path(src_file),
            destination=Path(dst_dir),
            species_name="electrons",
            min_uz=25.0,
        )
        # Check output file exists
        out_file = os.path.join(dst_dir, "test_fbpic.h5")
        assert os.path.exists(out_file)
        with h5py.File(out_file, "r") as f:
            g = f["data/00000001/particles/electrons"]
            z = g["position/z"][:]
            uz = g["momentum/z"][:]
            # Only the last two should remain (KE >= 25 MeV)
            assert np.allclose(z, np.array([0.3, 0.4]))
            assert np.allclose(uz, np.array(uz_for_ke_mev(30), uz_for_ke_mev(40)))
            # Check numParticles attribute
            assert g.attrs["numParticles"] == 2
        # Clean up (handled by TemporaryDirectory)
