"""Prepare two local FAIR metadata/data packages; never contact or publish to Zenodo."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from virtual_diagnostics.artifacts import Artifacts, MANIFESTS

HERE = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(artifact_directory, output, metadata, code_revision):
    """Verify artifacts and build disjoint model/context packages atomically."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite release directory: {output}")
    files = Artifacts(artifact_directory)
    selected = [
        key for record in metadata["records"].values() for key in record["file_keys"]
    ]
    if len(selected) != len(set(selected)) or "kumo" in selected:
        raise ValueError(
            "Records must have disjoint files and must not redistribute upstream Kumo weights."
        )
    paths = {key: files.path(key) for key in selected}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "release"
        staging.mkdir()
        for key, record in metadata["records"].items():
            folder = staging / key
            folder.mkdir()
            inventory = {
                name: files.manifest["files"][name] for name in record["file_keys"]
            }
            for name in record["file_keys"]:
                shutil.copyfile(paths[name], folder / inventory[name]["filename"])
            shutil.copyfile(MANIFESTS / "schema.json", folder / "schema.json")
            write_json(
                folder / "manifest.json",
                {
                    "format_version": 1,
                    "release": metadata["release"],
                    "record": key,
                    "version_doi": record["version_doi"],
                    "code_revision": code_revision,
                    "files": inventory,
                    "provenance": metadata["provenance"],
                    "access": metadata["access"],
                    "license": record["license"],
                    "license_status": record["license_status"],
                    "upstream_kumo": files.manifest["files"]["kumo"],
                },
            )
            other = metadata["records"][record["related_record"]]
            dataset = {
                "@context": "https://schema.org",
                "@type": "Dataset",
                "name": record["title"],
                "description": record["description"],
                "version": record["version"],
                "creativeWorkStatus": "Draft",
                "keywords": metadata["keywords"],
                "dateCreated": "2026-10-15",
                "creator": [
                    {
                        "@type": "Person",
                        "name": f"{c['given_name']} {c['family_name']}",
                        "@id": f"https://orcid.org/{c['orcid']}",
                        "affiliation": {
                            "@type": "Organization",
                            "name": c["affiliation"],
                        },
                    }
                    for c in metadata["creators"]
                ],
                "conditionsOfAccess": metadata["access"]["conditions"],
                "maintainer": {
                    "@type": "Person",
                    "@id": "https://orcid.org/" + metadata["creators"][0]["orcid"],
                    "email": metadata["access"]["request_contact"],
                },
                "isBasedOn": metadata["code_repository"] + "/tree/" + code_revision,
                "distribution": [
                    {
                        "@type": "DataDownload",
                        "name": v["filename"],
                        "contentSize": str(v["bytes"]),
                        "encodingFormat": "application/octet-stream",
                        "description": "Format, array schema and SHA-256 are in manifest.json and schema.json.",
                    }
                    for v in inventory.values()
                ],
            }
            if record["version_doi"]:
                dataset["identifier"] = "https://doi.org/" + record["version_doi"]
            if other["version_doi"]:
                dataset["citation"] = {
                    "@type": "Dataset",
                    "@id": "https://doi.org/" + other["version_doi"],
                    "name": other["title"],
                }
            # Proposed licenses are not asserted as an already effective grant.
            if record["license_status"] == "confirmed":
                dataset["license"] = record["license_url"]
            write_json(folder / "dataset.jsonld", dataset)
            shutil.copyfile(HERE / "README.md", folder / "README.md")
            hashes = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(folder.iterdir())
            }
            for item in inventory.values():
                if hashes[item["filename"]] != item["sha256"]:
                    raise ValueError(
                        f"Copied artifact checksum mismatch: {item['filename']}"
                    )
            write_json(
                folder / "checksums.json", {"algorithm": "sha256", "files": hashes}
            )
        write_json(
            staging / "publication-status.json",
            {
                "published": False,
                "ready_for_publication": False,
                "blockers": metadata["publication_blockers"],
                "code_revision": code_revision,
            },
        )
        staging.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(HERE.parent)], cwd=HERE, text=True
    )
    if dirty:
        parser.error(
            "Commit the diagnostic subproject before preparing a version-linked release."
        )
    metadata = json.loads((HERE / "metadata.json").read_text())
    print(prepare(args.artifacts, args.output, metadata, revision))


if __name__ == "__main__":
    main()
