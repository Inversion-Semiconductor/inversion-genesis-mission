"""Synthetic checks run without private weights, experimental data, or CUDA."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from virtual_diagnostics.artifacts import Artifacts, checked_file
from virtual_diagnostics.decoders import ImageDecoder, _DecoderNetwork
from virtual_diagnostics.encoders import ImageEncoder, PCAEncoder, _ImageNetwork
from virtual_diagnostics.features import (
    assemble_features,
    assemble_targets,
    feature_schema,
)
from virtual_diagnostics.preprocessing import (
    camera1_image,
    frog_image,
    oap_image,
    topview_image,
)


class BlockTests(unittest.TestCase):
    def test_tabular_import_is_independent_of_codecs_and_sdm(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from virtual_diagnostics.tabular import Kumo; "
                "assert 'virtual_diagnostics.encoders' not in sys.modules; "
                "assert 'virtual_diagnostics.decoders' not in sys.modules; "
                "assert 'sdm' not in sys.modules",
            ],
            check=True,
        )

    def test_feature_and_target_order(self):
        blocks = {
            name: np.full((2, width), i)
            for i, (name, width) in enumerate(
                [
                    ("scalars", 42),
                    ("gaia", 14),
                    ("frog", 8),
                    ("oap", 8),
                    ("topview", 16),
                ]
            )
        }
        x = assemble_features(**blocks)
        self.assertEqual(x.shape, (2, 88))
        self.assertEqual(x.dtype, np.float32)
        for i, block in enumerate(feature_schema()["blocks"]):
            np.testing.assert_array_equal(x[:, block["start"] : block["stop"]], i)
        y = assemble_targets(np.arange(40).reshape(2, 20), [7, 9])
        np.testing.assert_array_equal(y[:, -1], [7, 9])
        np.testing.assert_array_equal(y[:, :20], np.arange(40).reshape(2, 20))
        with self.assertRaisesRegex(ValueError, "aligned rows"):
            assemble_features(**{**blocks, "frog": np.zeros((3, 8))})
        for invalid in [np.zeros((2, 7)), np.full((2, 8), np.nan)]:
            with self.assertRaisesRegex(ValueError, "frog"):
                assemble_features(**{**blocks, "frog": invalid})
        with self.assertRaises(ValueError):
            assemble_targets(np.zeros((2, 20)), [1])
        with self.assertRaises(ValueError):
            assemble_targets(np.full((2, 20), np.inf), [1, 2])

    def test_artifact_verification_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            path.write_bytes(b"verified artifact")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch(
                "virtual_diagnostics.artifacts.manifest",
                return_value={
                    "files": {"model": {"filename": path.name, "sha256": digest}}
                },
            ):
                self.assertEqual(Artifacts(directory).path("model"), path.resolve())
            path.write_bytes(b"wrong version")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                checked_file(path, digest)
            path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing artifact"):
                checked_file(path, digest)

    def test_encoder_and_decoder_load_only_their_own_weights(self):
        torch.set_num_threads(2)
        shape, latent = (19, 21), 3
        norm = {
            "unit": torch.tensor(2.0),
            "reference": torch.zeros(1, 1, *shape),
            "scale": torch.tensor(3.0),
            "max_log": torch.tensor(8.0),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.pt"
            for residual in (False, True):
                model = _ImageNetwork(shape, latent, residual)
                torch.save(
                    dict(
                        model=model.state_dict(),
                        shape=shape,
                        latent=latent,
                        residual=residual,
                        norm=norm,
                    ),
                    path,
                )
                encoder = ImageEncoder(path)
                z = encoder(torch.ones(2, 1, *shape))
                self.assertEqual(tuple(z.shape), (2, latent))
                self.assertFalse(z.requires_grad)
                self.assertFalse(any("decoder" in k for k in encoder.state_dict()))
                for value in [
                    torch.zeros(2, 1, 20, 21),
                    torch.full((2, 1, *shape), -1.0),
                    torch.full((2, 1, *shape), float("nan")),
                ]:
                    with self.assertRaises(ValueError):
                        encoder(value)
            model = _DecoderNetwork(shape, latent)
            torch.save(
                dict(model=model.state_dict(), shape=shape, latent=latent, norm=norm),
                path,
            )
            decoder = ImageDecoder(path)
            image = decoder(z)
            self.assertEqual(tuple(image.shape), (2, 1, *shape))
            self.assertTrue(torch.isfinite(image).all() and (image >= 0).all())
            self.assertFalse(any("encoder" in k for k in decoder.state_dict()))
            with self.assertRaises(ValueError):
                decoder(torch.zeros(2, latent + 1))

    def test_pca_coordinates_and_prepared_shape_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pca.npz"
            np.savez(
                path,
                mean=np.ones(8),
                components=np.eye(8),
                shape=[2, 4],
                crop=[0, 4, 0, 2],
            )
            encoder = PCAEncoder(path)
            values = np.arange(16).reshape(2, 2, 4)
            np.testing.assert_array_equal(encoder(values), values.reshape(2, 8) - 1)
            self.assertEqual(encoder(values).dtype, np.float32)
            with self.assertRaises(ValueError):
                encoder(np.zeros((2, 4, 2)))
            with self.assertRaises(ValueError):
                encoder(np.full((1, 2, 4), np.nan))

    def test_image_calibration(self):
        raw = np.full((600, 700), 10.0)
        raw[80:100, 180:200] += 3
        prepared = frog_image(raw, (168, 561, 72, 531))
        self.assertEqual(prepared.shape, (459, 393))
        self.assertEqual(prepared.sum(), 3 * 20 * 20)
        np.testing.assert_array_equal(topview_image(raw), raw[336:592, 128:640])
        np.testing.assert_allclose(oap_image(np.full((963, 1280), 14.0)), 0, atol=1e-12)
        np.testing.assert_array_equal(
            camera1_image(np.full((189, 285), 4094), 2.0), 2.0
        )
        with self.assertRaises(ValueError):
            camera1_image(np.zeros((189, 285)), -1)
        with self.assertRaises(ValueError):
            oap_image(np.zeros((100, 100)))

    def test_manifests_ship_metadata_only(self):
        from virtual_diagnostics.artifacts import MANIFESTS

        data = json.loads((MANIFESTS / "artifacts.json").read_text())
        self.assertEqual(data["zenodo"]["files_visibility"], "restricted")
        self.assertTrue(all(len(v["sha256"]) == 64 for v in data["files"].values()))


if __name__ == "__main__":
    unittest.main()
