"""Persistent, frozen four-member Kumo inference on 1–4 visible CUDA GPUs."""

import atexit
from contextlib import closing, suppress
import gc
import math
import multiprocessing as mp
from multiprocessing.connection import wait
from numbers import Integral, Real
import os
import threading
import time

import numpy as np
import torch

from pathlib import Path

from ..artifacts import checked_file, manifest, verify_sdm

GROUPS = tuple(tuple(range(rank, 21, 4)) for rank in range(4))


def _validate_gpu_count(GPU):
    if isinstance(GPU, bool) or not isinstance(GPU, Integral):
        raise TypeError("GPU must be an integer GPU count, for example GPU=1 or GPU=4.")
    if GPU < 1:
        raise ValueError("GPU must be at least 1; this implementation requires CUDA.")
    if GPU > len(GROUPS):
        raise ValueError(
            "This frozen Kumo configuration supports GPU=1 through GPU=4 (four fixed output groups)."
        )
    try:
        available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception as exc:
        raise RuntimeError(
            "Unable to query CUDA GPUs. Check the NVIDIA driver and PyTorch CUDA installation."
        ) from exc
    if not available:
        raise RuntimeError(
            "No CUDA GPU is available. Check the NVIDIA driver, CUDA-enabled PyTorch, and CUDA_VISIBLE_DEVICES."
        )
    if GPU > available:
        raise RuntimeError(
            f"Requested GPU={GPU}, but only {available} CUDA GPU(s) are visible. Check your allocation and CUDA_VISIBLE_DEVICES."
        )
    return int(GPU)


def _array(value, shape, name):
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must contain finite numeric values with shape {shape}."
        ) from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(
            f"{name} must contain finite numeric values with shape {shape}; got {array.shape}."
        )
    return np.ascontiguousarray(array)


class _GroupRunner:
    @torch.inference_mode()
    def __init__(self, train_x, train_y, rank, groups, seed, tf32, weights):
        from sdm import Task
        from sdm.models import KumoTabular
        from .numerical_serving import CapturedNetwork
        from .serving_adapter import recipe_without_constant_drop

        torch.set_num_threads(2)
        torch.cuda.set_device(rank)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        self.device = torch.device("cuda", rank)
        self.columns = [col for group in groups for col in group]
        self.models = {}
        os.environ["SDM_CHUNK_MEMORY_FRACTION"] = "0.05"
        x = torch.as_tensor(train_x, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(train_y, dtype=torch.float32, device=self.device)
        base = KumoTabular(
            task="regression", size="large", pretrained=False, device="meta"
        )
        network = base.models[Task.regression]
        network.load_state_dict(
            torch.load(weights, map_location=self.device, weights_only=True),
            assign=True,
        )
        with torch.autocast("cuda", dtype=torch.float16):
            for columns in groups:
                model = (
                    base
                    if not self.models
                    else KumoTabular(
                        task="regression", size="large", pretrained=False, device="meta"
                    )
                )
                if self.models:
                    model.models = base.models
                model.fit(
                    x.unsqueeze(0).expand(len(columns), -1, -1),
                    y[:, list(columns)].T.unsqueeze(-1),
                    recipe=recipe_without_constant_drop(model, x),
                    num_estimators=4,
                    estimator_batch_size=4,
                    generator=torch.Generator(self.device).manual_seed(seed),
                )
                model._cache = model._cache.to(self.device)
                for i in range(model._cache["num_batches"]):
                    if not all(
                        len(s.columns["numerical"]) == 88
                        for s in model._cache[i]["x_schemas"]
                    ):
                        raise RuntimeError(
                            "Fitted preprocessing did not preserve all 88 features."
                        )
                self.models[columns] = model
        os.environ["SDM_CHUNK_MEMORY_FRACTION"] = "0.25"
        expected = torch.cat([self.targets(q[None]) for q in x[:3]])
        self.graph = CapturedNetwork(network)
        actual = torch.cat([self.targets(q[None]) for q in x[:3]])
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
        if self.graph.summary()["captured_graphs"] != len(groups):
            raise RuntimeError("Failed to capture every assigned output group.")
        torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def targets(self, query):
        with torch.autocast("cuda", dtype=torch.float16):
            return torch.cat(
                [
                    model.predict(query.unsqueeze(0).expand(len(cols), -1, -1))
                    .numerical.float()
                    .mean(-1)
                    .T
                    for cols, model in self.models.items()
                ],
                dim=1,
            )


def _worker(pipe, rank, groups, train_x, train_y, seed, tf32, weights):
    with closing(pipe), torch.inference_mode():
        try:
            runner = _GroupRunner(train_x, train_y, rank, groups, seed, tf32, weights)
            del train_x, train_y
            gc.collect()
            gc.freeze()  # Isolated to this serving process; released when it exits.
            pipe.send(
                (
                    "ready",
                    dict(
                        rank=rank,
                        columns=runner.columns,
                        graphs=runner.graph.summary(),
                        pid=os.getpid(),
                        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                        cudnn_tf32=torch.backends.cudnn.allow_tf32,
                    ),
                )
            )
            while (values := pipe.recv()) is not None:
                query = torch.from_numpy(values[None]).to(runner.device)
                targets = runner.targets(query)
                pipe.send(("targets", targets.cpu().numpy()[0]))
        except BaseException as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                detail += " Free GPU memory or use more GPUs for this frozen context."
            with suppress(OSError, EOFError):
                pipe.send(("error", f"CUDA GPU {rank}: {detail}"))


class Kumo:
    """Frozen four-member model. GPU defaults to 1; TF32 defaults to enabled.

    Create inside a main guard in scripts because GPU workers use spawn.
    Use close() or a context manager to release all worker GPU resources.
    predict() calls on this instance are serialized to protect graph buffers.
    This serving configuration accepts 2,152 context pairs with 88 inputs and
    21 numerical targets. Returns a float32 NumPy vector with those 21 targets.
    No image models or context data are loaded by this class.
    """

    def __init__(
        self,
        train_x,
        train_y,
        GPU=1,
        seed=20261009,
        *,
        weights,
        tf32=True,
        timeout=120,
        startup_timeout=300,
        verify_artifacts=True,
    ):
        self._lock = threading.RLock()
        self._workers = []
        self._closed = False
        self.GPU = _validate_gpu_count(GPU)
        if not isinstance(tf32, bool):
            raise TypeError("tf32 must be True or False.")
        self.tf32 = tf32
        if (
            isinstance(seed, bool)
            or not isinstance(seed, Integral)
            or not 0 <= seed < 2**63
        ):
            raise ValueError("seed must be an integer in [0, 2**63).")
        for name, value in [("timeout", timeout), ("startup_timeout", startup_timeout)]:
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number of seconds.")
        self.seed = int(seed)
        self.timeout = float(timeout)
        train_x = _array(train_x, (2152, 88), "train_x")
        train_y = _array(train_y, (2152, 21), "train_y")
        if not (train_x != train_x[:1]).any(axis=0).all():
            raise ValueError(
                "Every one of the 88 training features must vary for the validated preprocessing adapter."
            )
        if verify_artifacts:
            try:
                checked_file(weights, manifest()["files"]["kumo"]["sha256"])
                verify_sdm()
            except Exception as exc:
                raise RuntimeError(
                    f"Kumo artifact verification failed: {type(exc).__name__}: {exc}"
                ) from exc
        started = time.monotonic()
        try:
            context = mp.get_context("spawn")
            for rank in range(self.GPU):
                parent, child = context.Pipe()
                groups = GROUPS[rank :: self.GPU]
                process = context.Process(
                    target=_worker,
                    args=(
                        child,
                        rank,
                        groups,
                        train_x,
                        train_y,
                        self.seed,
                        self.tf32,
                        str(Path(weights).expanduser().resolve()),
                    ),
                    name=f"Kumo-GPU-{rank}",
                    daemon=True,
                )
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self._workers.append((process, parent))
            deadline = started + float(startup_timeout)
            self.workers = [
                self._receive(worker, "ready", deadline) for worker in self._workers
            ]
        except BaseException:
            self.close()
            raise
        self.setup_seconds = time.monotonic() - started
        atexit.register(self.close)

    @staticmethod
    def _receive(worker, expected, deadline):
        process, pipe = worker
        remaining = deadline - time.monotonic()
        ready = (
            wait([pipe, process.sentinel], timeout=remaining) if remaining > 0 else []
        )
        if not ready:
            raise TimeoutError(
                f"{process.name} timed out while waiting for {expected}."
            )
        if pipe not in ready:
            raise RuntimeError(
                f"{process.name} exited unexpectedly (exit code {process.exitcode})."
            )
        try:
            kind, value = pipe.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(
                f"{process.name} disconnected while waiting for {expected}."
            ) from exc
        if kind == "error":
            raise RuntimeError(value)
        if kind != expected:
            raise RuntimeError(
                f"{process.name} returned {kind!r}; expected {expected!r}."
            )
        return value

    def predict(self, features):
        """Return a CPU float32 NumPy array of shape (21,), without image decoding.

        Output columns follow train_y. Interpretation and decoding belong to
        the caller, independently of this tabular runtime.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    "This Kumo instance is closed. Create a new instance to predict."
                )
            features = _array(features, (88,), "features")
            try:
                deadline = time.monotonic() + self.timeout
                for _, pipe in self._workers:
                    pipe.send(features)
                targets = np.empty(21, dtype=np.float32)
                for worker, info in zip(self._workers, self.workers):
                    targets[info["columns"]] = self._receive(
                        worker, "targets", deadline
                    )
                if not np.isfinite(targets).all():
                    raise RuntimeError(
                        "Kumo produced non-finite output; check input feature units and ranges."
                    )
                return targets
            except BaseException as exc:
                self.close()
                if isinstance(
                    exc, (BrokenPipeError, EOFError, OSError)
                ) and not isinstance(exc, TimeoutError):
                    raise RuntimeError(
                        "A Kumo GPU worker disconnected; the instance has been closed."
                    ) from exc
                raise

    def close(self):
        """Release all workers, including after a partial initialization failure."""
        with self._lock:
            self._closed = True
            for process, pipe in self._workers:
                with suppress(OSError, EOFError):
                    if process.is_alive():
                        pipe.send(None)
            for process, pipe in self._workers:
                try:
                    process.join(timeout=2)
                    for stop in (process.terminate, process.kill):
                        if not process.is_alive():
                            break
                        stop()
                        process.join(timeout=2)
                    # Best effort: a worker that survives kill() must not stop the
                    # rest from being released or mask the caller's exception.
                    with suppress(ValueError):
                        process.close()
                finally:
                    pipe.close()
            self._workers.clear()
            atexit.unregister(self.close)

    def __enter__(self):
        if self._closed:
            raise RuntimeError("This Kumo instance is closed.")
        return self

    def __exit__(self, *_):
        self.close()
