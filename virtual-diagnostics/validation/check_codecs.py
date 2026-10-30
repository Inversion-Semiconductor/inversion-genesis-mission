"""Compare installed codecs against a private fixture from the original models."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from virtual_diagnostics.artifacts import Artifacts
from virtual_diagnostics.decoders import ImageDecoder
from virtual_diagnostics.encoders import GaiaEncoder, ImageEncoder, PCAEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    files = Artifacts(args.artifacts)
    results = {}
    with np.load(args.fixture, allow_pickle=False) as fixture:
        for name, cls in [
            ("camera1", ImageEncoder),
            ("topview", ImageEncoder),
            ("gaia", GaiaEncoder),
        ]:
            encoder = cls(files.path(name + "_encoder"))
            actual = encoder(torch.from_numpy(fixture[name][:, None])).numpy()
            expected = fixture[name + "_latents"]
            # The reference uses macOS CPU kernels; Linux CPU kernels round differently.
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
            results[name] = dict(
                rows=len(actual),
                max_abs_difference=float(np.max(np.abs(actual - expected))),
            )
        decoder = ImageDecoder(files.path("camera1_decoder"))
        actual = decoder(torch.from_numpy(fixture["camera1_latents"]))[:, 0].numpy()
        np.testing.assert_allclose(actual, fixture["decoded"], rtol=2e-5, atol=1e-7)
        results["decoder"] = dict(
            rows=len(actual),
            max_abs_difference=float(np.max(np.abs(actual - fixture["decoded"]))),
        )
        for name in ("frog", "oap"):
            actual = PCAEncoder(files.path(name + "_pca"))(fixture[name])
            np.testing.assert_allclose(
                actual, fixture[name + "_latents"], rtol=1e-6, atol=1e-6
            )
            results[name] = dict(
                rows=len(actual),
                max_abs_difference=float(
                    np.max(np.abs(actual - fixture[name + "_latents"]))
                ),
            )
    report = dict(
        passed=True,
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        results=results,
        oap_input="stored training-mean frame; raw shot acquisition not tested",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
