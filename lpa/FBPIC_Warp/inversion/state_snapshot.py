"""Snapshot and restore of a stepper state, for one-step checks (L3).

A snapshot copies every CuPy and NumPy array attribute and every scalar
attribute of the objects that `step` changes: the simulation clock, the
communicator and moving window, the field grids, and each species with its
tracker and injector. It also stores the NumPy RNG state, which the
injection uses. `restore` writes the arrays back in place when the shape
is unchanged and rebinds them otherwise (an exchange can grow a buffer).

The check of completeness is empirical: a restored step must reproduce the
first step from the same snapshot to rounding (`verify_one_step.py`).
"""

from __future__ import annotations

import numpy as np

SCALARS = (int, float, bool, str, type(None), np.integer, np.floating, np.bool_)


def _objects(state):
    sim = state.sim
    objects = {
        "sim": sim,
        "state": state,
        "comm": sim.comm,
        "moving_win": sim.comm.moving_win,
    }
    if getattr(state, "ev", None) is not None:
        objects["ev"] = state.ev
    for m, grid in enumerate(sim.fld.interp):
        objects[f"interp{m}"] = grid
    for m, grid in enumerate(sim.fld.spect):
        objects[f"spect{m}"] = grid
    for name, sp in state.species.items():
        objects[f"{name}"] = sp
        objects[f"{name}.tracker"] = sp.tracker
        if sp.injector is not None:
            objects[f"{name}.injector"] = sp.injector
    return objects


def _is_array(value):
    import cupy

    return isinstance(value, (cupy.ndarray, np.ndarray))


def snapshot(state) -> dict:
    """Copy the mutable arrays and scalars of the state."""
    out = {"rng": np.random.get_state(), "objects": {}}
    out["exchanged_source"] = dict(state.sim.fld.exchanged_source)
    for key, obj in _objects(state).items():
        saved = {}
        for attr, value in vars(obj).items():
            if attr in ("hooks",):
                continue
            if _is_array(value):
                saved[attr] = value.copy()
            elif isinstance(value, SCALARS):
                saved[attr] = value
        out["objects"][key] = saved
    return out


def restore(state, snap: dict):
    """Write a snapshot back into the state objects."""
    np.random.set_state(snap["rng"])
    state.sim.fld.exchanged_source.clear()
    state.sim.fld.exchanged_source.update(snap["exchanged_source"])
    objects = _objects(state)
    for key, saved in snap["objects"].items():
        obj = objects[key]
        for attr, value in saved.items():
            if _is_array(value):
                current = getattr(obj, attr, None)
                if (
                    current is not None
                    and type(current) is type(value)
                    and current.shape == value.shape
                    and current.dtype == value.dtype
                ):
                    current[...] = value
                else:
                    setattr(obj, attr, value.copy())
            else:
                setattr(obj, attr, value)
