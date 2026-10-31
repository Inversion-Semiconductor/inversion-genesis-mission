"""Simulation builders for the forward-mode driver (`fbpic.autodiff.driver`).

A builder is a callable build(theta) -> (sim, species, profile).
"""

from __future__ import annotations

from twin_config import build_twin, load_config


def twin_builder(size, ionization="none", **kwargs):
    """Builder of a twin configuration (`build_twin`) at theta = (E_L, z_f)."""
    config = load_config(size) if isinstance(size, str) else size

    def build(theta):
        twin = build_twin(theta, config, ionization=ionization, **kwargs)
        return twin.sim, twin.species, twin.laser_profile

    build.config = config
    return build
