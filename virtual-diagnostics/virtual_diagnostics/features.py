"""The frozen diagnostic's feature order, independent of model execution."""

import json

import numpy as np

from .artifacts import MANIFESTS


def feature_schema():
    return json.loads((MANIFESTS / "schema.json").read_text())


def assemble_features(*, scalars, gaia, frog, oap, topview):
    """Join aligned batches into (N, 88), in the validated column order.

    Rows must refer to the same shots in every block; timestamps/joins belong
    to acquisition. Scalars use the exact names/units in feature_schema().
    """
    blocks = []
    for name, value, width in [
        ("scalars", scalars, 42),
        ("gaia", gaia, 14),
        ("frog", frog, 8),
        ("oap", oap, 8),
        ("topview", topview, 16),
    ]:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != width or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite with shape (N, {width}).")
        if blocks and len(array) != len(blocks[0]):
            raise ValueError(
                "Feature blocks must have the same number of aligned rows."
            )
        blocks.append(array)
    return np.concatenate(blocks, axis=1)


def assemble_targets(camera1_latents, charge_pc):
    """Join (N, 20) encoded context images and (N,) charge into (N, 21)."""
    z = np.asarray(camera1_latents, dtype=np.float32)
    charge = np.asarray(charge_pc, dtype=np.float32)
    if z.ndim != 2 or z.shape[1] != 20 or charge.shape != (len(z),):
        raise ValueError("Expected camera1_latents (N, 20) and charge_pc (N,).")
    result = np.column_stack([z, charge])
    if not np.isfinite(result).all():
        raise ValueError("Context targets must be finite.")
    return result
