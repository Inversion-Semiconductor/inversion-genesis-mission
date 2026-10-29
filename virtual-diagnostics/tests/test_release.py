"""Exercise private-package separation and integrity without experimental files."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

RELEASE = Path(__file__).resolve().parents[1] / "release"
spec = importlib.util.spec_from_file_location("prepare_release", RELEASE / "prepare.py")
prepare_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_release)


class ReleaseTests(unittest.TestCase):
    def test_private_records_are_separate_and_checksummed(self):
        metadata = json.loads((RELEASE / "metadata.json").read_text())
        for index, record in enumerate(metadata["records"].values(), start=1):
            record["version_doi"] = f"10.5281/zenodo.{index}"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "artifacts"
            source.mkdir()
            entries = {"kumo": {"filename": "upstream.pt", "sha256": "0" * 64}}
            for record in metadata["records"].values():
                for key in record["file_keys"]:
                    content = key.encode()
                    (source / key).write_bytes(content)
                    entries[key] = dict(
                        filename=key,
                        sha256=hashlib.sha256(content).hexdigest(),
                        bytes=len(content),
                    )
            with patch(
                "virtual_diagnostics.artifacts.manifest",
                return_value={"files": entries},
            ):
                out = prepare_release.prepare(
                    source, root / "release", metadata, "a" * 40
                )
                for name, record in metadata["records"].items():
                    folder = out / name
                    manifest = json.loads((folder / "manifest.json").read_text())
                    self.assertEqual(set(manifest["files"]), set(record["file_keys"]))
                    self.assertFalse((folder / "upstream.pt").exists())
                    for filename, digest in json.loads(
                        (folder / "checksums.json").read_text()
                    )["files"].items():
                        self.assertEqual(
                            hashlib.sha256(
                                (folder / filename).read_bytes()
                            ).hexdigest(),
                            digest,
                        )
                    dataset = json.loads((folder / "dataset.jsonld").read_text())
                    self.assertEqual(dataset["@type"], "Dataset")
                    self.assertEqual(
                        dataset["identifier"],
                        "https://doi.org/" + record["version_doi"],
                    )
                    self.assertEqual(
                        dataset["citation"]["@id"],
                        "https://doi.org/"
                        + metadata["records"][record["related_record"]]["version_doi"],
                    )
                    self.assertEqual(
                        dataset["maintainer"]["email"], "daniel@inversionsemi.com"
                    )
                    self.assertEqual(
                        dataset["creator"][0]["@id"],
                        "https://orcid.org/0009-0002-0548-0800",
                    )
                    self.assertNotIn(
                        "license", dataset
                    )  # A proposed license is not an effective grant.
                with self.assertRaises(FileExistsError):
                    prepare_release.prepare(source, out, metadata, "a" * 40)
                (source / "context").write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    prepare_release.prepare(source, root / "failed", metadata, "a" * 40)
                self.assertFalse((root / "failed").exists())


if __name__ == "__main__":
    unittest.main()
