"""Gate test for a GPU job step: run before every benchmark task.

Checks the numba CUDA driver path (NERSC 580-driver issue), CuPy device
visibility, Warp initialization, and the MPI library. Exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def _text(value) -> str:
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return str(value).rstrip("\x00 ")


def _numba() -> dict:
    import numba
    from numba import cuda

    cuda.synchronize()
    return {
        "version": numba.__version__,
        "device": _text(cuda.get_current_device().name),
    }


def _cupy() -> dict:
    import cupy

    count = cupy.cuda.runtime.getDeviceCount()
    if count != 1:
        raise RuntimeError(f"expected exactly one visible GPU, found {count}")
    properties = cupy.cuda.runtime.getDeviceProperties(0)
    return {
        "version": cupy.__version__,
        "device": _text(properties["name"]),
        "driver": cupy.cuda.runtime.driverGetVersion(),
        "runtime": cupy.cuda.runtime.runtimeGetVersion(),
    }


def _warp() -> dict:
    import warp as wp

    wp.config.quiet = True
    wp.init()
    if wp.get_cuda_device_count() != 1:
        raise RuntimeError("Warp does not see exactly one CUDA device")
    return {"version": wp.config.version, "device": str(wp.get_cuda_device(0).name)}


def _mpi() -> dict:
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    if size != 1:
        raise RuntimeError(f"expected one MPI rank per job step, found {size}")
    return {"library": MPI.Get_library_version().strip().splitlines()[0], "size": size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report: dict = {"python": sys.version.split()[0]}
    failed = False
    for name, check in (
        ("numba_cuda", _numba),
        ("cupy", _cupy),
        ("warp", _warp),
        ("mpi", _mpi),
    ):
        try:
            report[name] = check()
            print(f"ok   {name}: {report[name]}", flush=True)
        except Exception:  # noqa: BLE001 - report every failed check
            failed = True
            report[name] = {"error": traceback.format_exc()}
            print(f"FAIL {name}:\n{report[name]['error']}", flush=True)
    report["ok"] = not failed
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
