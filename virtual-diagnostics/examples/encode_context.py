"""Encode private prepared image crops into a Kumo context or query table.

Input NPZ: scalars (N,42), gaia (N,304,304), frog (N,459,393),
oap (N,963,1010), topview (N,256,512). A context additionally contains
camera1 (N,189,285) physical intensities and charge_pc (N,). All rows must
be shot-aligned. For a context, pass only the 2,152 training rows.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from virtual_diagnostics.artifacts import Artifacts
from virtual_diagnostics.encoders import GaiaEncoder, ImageEncoder, PCAEncoder
from virtual_diagnostics.features import assemble_features, assemble_targets


def encode_batches(encoder, images, batch_size=8):
    return np.concatenate(
        [
            encoder(
                torch.as_tensor(
                    images[start : start + batch_size, None], dtype=torch.float32
                )
            ).numpy()
            for start in range(0, len(images), batch_size)
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    files = Artifacts(args.artifacts)
    with np.load(args.images, allow_pickle=False) as data:
        features = assemble_features(
            scalars=data["scalars"],
            gaia=encode_batches(GaiaEncoder(files.path("gaia_encoder")), data["gaia"]),
            frog=PCAEncoder(files.path("frog_pca"))(data["frog"]),
            oap=PCAEncoder(files.path("oap_pca"))(data["oap"]),
            topview=encode_batches(
                ImageEncoder(files.path("topview_encoder")), data["topview"]
            ),
        )
        output = {"features": features}
        if args.context:
            if len(features) != 2152:
                raise ValueError(
                    "This frozen Kumo serving configuration requires 2,152 context rows."
                )
            latents = encode_batches(
                ImageEncoder(files.path("camera1_encoder")), data["camera1"]
            )
            output = {
                "train_x": features,
                "train_y": assemble_targets(latents, data["charge_pc"]),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)


if __name__ == "__main__":
    main()
