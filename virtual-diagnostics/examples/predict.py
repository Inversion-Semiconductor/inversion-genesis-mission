"""Use a private prepared context and query; decode/display in separate steps."""

import argparse
from pathlib import Path

import numpy as np
import torch

from virtual_diagnostics.artifacts import Artifacts
from virtual_diagnostics.decoders import ImageDecoder
from virtual_diagnostics.tabular import Kumo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = Artifacts(args.artifacts)
    with np.load(files.path("context"), allow_pickle=False) as data:
        train_x, train_y = data["train_x"], data["train_y"]
    query = np.load(files.path("example_features"), allow_pickle=False)
    with Kumo(train_x, train_y, GPU=args.gpu, weights=files.path("kumo")) as server:
        result = server.predict(query)
    decoder = ImageDecoder(files.path("camera1_decoder"))
    image = decoder(torch.from_numpy(result[:20]).unsqueeze(0))[0, 0].numpy()
    print(
        f"Prediction: {result.shape}; decoded image: {image.shape}; charge: {result[20]:.6f} pC"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, result=result, image=image)
    if args.show:
        import matplotlib.pyplot as plt

        plt.imshow(image, cmap="viridis")
        plt.title(f"Predicted camera-1 image; charge {result[20]:.2f} pC")
        plt.colorbar(label="Reconstructed intensity")
        plt.show()


if __name__ == "__main__":
    main()
