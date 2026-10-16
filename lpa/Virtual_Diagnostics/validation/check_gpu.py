"""Check the installed bundle against the original A100 validation predictions."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import torch
import virtual_diagnostics
import sdm
from virtual_diagnostics.tabular import Kumo
from virtual_diagnostics.decoders import ImageDecoder
from virtual_diagnostics.artifacts import Artifacts, verify_sdm


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--shots", type=int, default=462)
    parser.add_argument("--output", type=Path, default=Path("gpu_check.json"))
    args = parser.parse_args()
    if not 1 <= args.shots <= 462:
        parser.error("--shots must be between 1 and 462")
    files = Artifacts(args.artifacts)
    verify_sdm()
    with np.load(files.path("context"), allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    with np.load(args.fixture, allow_pickle=False) as stored:
        fixture = {key: stored[key] for key in stored.files}
    assert not np.intersect1d(data["train_indices"], fixture["indices"]).size
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    decoder = ImageDecoder(files.path("camera1_decoder"), device="cuda:0")
    rows = []
    status = args.output.with_suffix(".status.json")
    for count in args.gpus:
        for seed in (20261009, 20261010):
            save(status, dict(state="running", GPU=count, seed=seed))
            started = time.monotonic()
            with Kumo(
                data["train_x"],
                data["train_y"],
                GPU=count,
                seed=seed,
                weights=files.path("kumo"),
            ) as server:
                for index, query in enumerate(fixture["queries"][: args.shots]):
                    result = server.predict(query)
                    assert result.shape == (21,) and result.dtype == np.float32
                    np.testing.assert_array_equal(
                        result, fixture[f"targets_s{seed}"][index]
                    )
                    image = (
                        decoder(
                            torch.from_numpy(result[:20]).to("cuda:0").unsqueeze(0)
                        )[0, 0]
                        .cpu()
                        .numpy()
                    )
                    assert (
                        hashlib.sha256(image.tobytes()).hexdigest()
                        == fixture[f"image_sha256_s{seed}"][index]
                    ), index
                assert len(server.workers) == count
                assert all(w["matmul_tf32"] and w["cudnn_tf32"] for w in server.workers)
                assert sum(w["graphs"]["captured_graphs"] for w in server.workers) == 4
                pids = {w["pid"] for w in server.workers}
                processes = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,gpu_uuid",
                        "--format=csv,noheader",
                    ],
                    text=True,
                )
                used = [
                    line.split(",")
                    for line in processes.splitlines()
                    if int(line.split(",")[0]) in pids
                ]
                assert (
                    len(used) == count
                    and len({line[1].strip() for line in used}) == count
                )
                rows.append(
                    dict(
                        GPU=count,
                        seed=seed,
                        validation_shots=args.shots,
                        exact_targets=True,
                        exact_images=True,
                        workers=server.workers,
                        gpu_processes=processes,
                        elapsed_seconds=time.monotonic() - started,
                    )
                )
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in rows[-1].items()
                        if key not in ("workers", "gpu_processes")
                    }
                ),
                flush=True,
            )
    verify_sdm()
    report = dict(
        passed=True,
        cases=rows,
        host=socket.gethostname(),
        job_id=os.environ.get("PBS_JOBID"),
        version=virtual_diagnostics.__version__,
        runtime_sha256={
            str(
                p.relative_to(Path(virtual_diagnostics.__file__).parent)
            ): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(virtual_diagnostics.__file__).parent.rglob("*.py")
        },
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        python=os.sys.version,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        installed_package=str(Path(virtual_diagnostics.__file__).resolve()),
        installed_sdm=str(Path(sdm.__file__).resolve()),
        packages={
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pyarrow",
                "huggingface_hub",
                "safetensors",
                "typing_extensions",
            )
        },
        prediction_api="float32_numpy_vector_21",
        decoding="separate_ImageDecoder_call",
        test_evaluated=False,
        encoders_evaluated=False,
    )
    save(args.output, report)
    save(status, dict(state="complete"))
    print(
        "PASS: installed bundle matches saved targets and every decoded image.",
        flush=True,
    )


if __name__ == "__main__":
    main()
