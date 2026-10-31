"""Phase P1 steps 4 and 5: azimuthal aliasing and particle weights.

Usage (from `lpa/FBPIC_Warp/inversion`, one GPU):

    python verify_twin_p1.py twin_xs
    python verify_twin_p1.py twin_s

Aliasing: deposit the charge of each species separately from particles that
no laser has pushed, then require max|rho_1| / max|rho_0| < 1e-10 over the
plasma. Twin-XS is preloaded, so the check runs at t = 0. Twin-S injects its
plasma, so the check runs after 300 steps with the laser fields set to zero:
the window then injects the plasma into a field-free, neutral box.

Weights: all weights are positive, and the weights of the particles with the
same tracker ID agree exactly at two opposite corners of the box Theta.
"""

from __future__ import annotations

import sys

import cupy
import numpy as np
from fbpic.utils.cuda import send_data_to_gpu
from twin_config import build_twin, load_config, theta_from_hat

FIELDS = ("Er", "Et", "Ez", "Br", "Bt", "Bz")


def aliasing(size: str) -> dict:
    config = load_config(size)
    twin = build_twin(config["theta"]["nominal"], config, ionization="none")
    sim = twin.sim
    if config["plasma"]["start"] == "preloaded":
        sim.step(0, show_progress=False)
        steps = 0
    else:
        for grid in sim.fld.interp:
            for name in FIELDS:
                getattr(grid, name)[...] = 0.0
        steps = 300
        sim.step(steps, show_progress=False)
    # `Simulation.step` returns the data to the host; deposit on the GPU.
    send_data_to_gpu(sim)
    out = {"steps": steps}
    for name in ("electrons", "helium", "nitrogen"):
        species = twin.species[name]
        sim.deposit("rho_next", species_list=[species])
        rho0 = np.abs(cupy.asnumpy(sim.fld.interp[0].rho))
        rho1 = np.abs(cupy.asnumpy(sim.fld.interp[1].rho))
        ratio = float(rho1.max() / rho0.max())
        out[name] = {"Ntot": int(species.Ntot), "rho1_over_rho0": ratio}
        assert species.Ntot > 0, f"{name}: no particles loaded"
        assert ratio < 1e-10, f"{name}: aliasing ratio {ratio:.3e}"
    return out


def weights(size: str, steps: int) -> dict:
    config = load_config(size)
    runs = []
    for hat in ((0.0, 0.0), (1.0, 1.0)):
        twin = build_twin(theta_from_hat(config, hat), config, ionization="none")
        twin.sim.step(steps, show_progress=False)
        per_species = {}
        for name in ("electrons", "helium", "nitrogen"):
            sp = twin.species[name]
            n = int(sp.Ntot)
            ids = cupy.asnumpy(sp.tracker.id[:n])
            w = cupy.asnumpy(sp.w[:n])
            per_species[name] = (ids, w, int(sp.tracker.next_attributed_id))
        runs.append(per_species)
        del twin
    out = {"steps": steps}
    for name in ("electrons", "helium", "nitrogen"):
        (ia, wa, na), (ib, wb, nb) = runs[0][name], runs[1][name]
        assert np.all(wa > 0) and np.all(wb > 0), f"{name}: non-positive weight"
        assert na == nb, f"{name}: injected IDs differ ({na} vs {nb})"
        common, ka, kb = np.intersect1d(ia, ib, return_indices=True)
        assert len(common) > 0, f"{name}: no common particles"
        max_diff = float(np.abs(wa[ka] - wb[kb]).max())
        assert max_diff == 0.0, f"{name}: weights differ by {max_diff}"
        out[name] = {
            "Ntot": [len(ia), len(ib)],
            "ids_attributed": na,
            "common_ids": len(common),
            "w_min": float(min(wa.min(), wb.min())),
            "max_abs_weight_difference": max_diff,
        }
    return out


if __name__ == "__main__":
    size = sys.argv[1]
    steps = {"twin_xs": 60, "twin_s": 1300}[size]
    print("aliasing", aliasing(size))
    print("weights", weights(size, steps))
    print("PASS")
