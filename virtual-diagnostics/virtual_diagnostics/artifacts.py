"""Resolve versioned local artifacts. Network transfers are always explicit."""

import hashlib
import importlib.util
import json
from pathlib import Path

MANIFESTS = Path(__file__).with_name("manifests")


def manifest():
    return json.loads((MANIFESTS / "artifacts.json").read_text())


def checked_file(path, sha256):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing artifact: {path}. See the artifact manifest and README."
        )
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != sha256:
        raise ValueError(f"Artifact checksum mismatch: {path}")
    return path


class Artifacts:
    """A local copy of the matched diagnostic release; verify each file on use."""

    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        self.manifest = manifest()

    def path(self, name):
        item = self.manifest["files"][name]
        return checked_file(self.directory / item["filename"], item["sha256"])


def verify_sdm():
    """Check the installed upstream source against the validated snapshot."""
    spec = importlib.util.find_spec("sdm")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Install this subproject with the [kumo] extra for GPU inference."
        )
    installed = Path(spec.origin).resolve().parent.parent
    for name, digest in json.loads((MANIFESTS / "sdm_source.json").read_text()).items():
        checked_file(installed / name, digest)
