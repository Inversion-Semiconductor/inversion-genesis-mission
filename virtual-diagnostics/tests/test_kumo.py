"""CPU-only checks for public validation, failure propagation and cleanup."""

from contextlib import nullcontext
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from virtual_diagnostics.tabular.api import (
    GROUPS,
    Kumo,
    _GroupRunner,
    _validate_gpu_count,
    _worker,
)


class KumoErrors(unittest.TestCase):
    def make_server(self, count=1):
        server = object.__new__(Kumo)
        server._lock = threading.RLock()
        server._closed = False
        server.GPU = count
        server.timeout = 1
        server._workers = [(Mock(), Mock()) for _ in range(count)]
        for process, _ in server._workers:
            process.is_alive.return_value = False
        server.workers = [
            dict(columns=[col for group in GROUPS[rank::count] for col in group])
            for rank in range(count)
        ]
        return server

    def test_vector_return_preserves_target_order_for_all_gpu_counts(self):
        expected = np.arange(21, dtype=np.float32) + 0.25
        for count in (1, 2, 3, 4):
            with self.subTest(GPU=count):
                server = self.make_server(count)
                parts = [expected[info["columns"]] for info in server.workers]
                with patch.object(Kumo, "_receive", side_effect=parts * 2) as receive:
                    result = server.predict(np.zeros(88, dtype=np.float32))
                    self.assertIsInstance(result, np.ndarray)
                    self.assertEqual(result.shape, (21,))
                    self.assertEqual(result.dtype, np.float32)
                    np.testing.assert_array_equal(result, expected)
                    result[:] = -1
                    np.testing.assert_array_equal(
                        server.predict(np.zeros(88)), expected
                    )
                    self.assertTrue(
                        all(
                            call.args[1] == "targets" for call in receive.call_args_list
                        )
                    )
                for _, pipe in server._workers:
                    self.assertEqual(pipe.send.call_count, 2)
                    for call in pipe.send.call_args_list:
                        np.testing.assert_array_equal(call.args[0], np.zeros(88))

    def test_worker_never_loads_the_image_decoder(self):
        runner = Mock(device=torch.device("cpu"), columns=list(range(21)))
        runner.targets.return_value = torch.arange(21, dtype=torch.float32).unsqueeze(0)
        pipe = Mock()
        pipe.recv.side_effect = [np.zeros(88, dtype=np.float32), None]
        with (
            patch("virtual_diagnostics.tabular.api._GroupRunner", return_value=runner),
            patch(
                "virtual_diagnostics.decoders.ImageDecoder",
                side_effect=AssertionError("Decoder must stay separate"),
            ) as decoder,
            patch("virtual_diagnostics.tabular.api.gc.freeze"),
            patch("virtual_diagnostics.tabular.api.gc.collect"),
        ):
            _worker(pipe, 0, GROUPS, None, None, 20261009, True, "unused.pt")
        decoder.assert_not_called()
        messages = [call.args[0] for call in pipe.send.call_args_list]
        self.assertEqual([kind for kind, _ in messages], ["ready", "targets"])
        np.testing.assert_array_equal(messages[1][1], np.arange(21, dtype=np.float32))
        pipe.close.assert_called_once()

    def test_nonfinite_vector_closes_the_instance(self):
        server = self.make_server()
        with patch.object(
            Kumo, "_receive", return_value=np.full(21, np.nan, dtype=np.float32)
        ):
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                server.predict(np.zeros(88))
        self.assertTrue(server._closed)

    def test_tf32_argument_and_worker_propagation(self):
        x = np.arange(2152 * 88, dtype=np.float32).reshape(2152, 88)
        with patch(
            "virtual_diagnostics.tabular.api._validate_gpu_count", return_value=1
        ):
            for value in ["True", 1, None]:
                with self.subTest(value=value), self.assertRaises(TypeError):
                    Kumo(x, np.zeros((2152, 21)), weights="unused.pt", tf32=value)
            for kwargs, expected in [({}, True), ({"tf32": False}, False)]:
                process, parent, child, context = Mock(), Mock(), Mock(), Mock()
                process.is_alive.return_value = False
                context.Pipe.return_value = parent, child
                context.Process.return_value = process
                with (
                    patch(
                        "virtual_diagnostics.tabular.api.mp.get_context",
                        return_value=context,
                    ),
                    patch.object(Kumo, "_receive", return_value={}),
                ):
                    with Kumo(
                        x,
                        np.zeros((2152, 21)),
                        weights="unused.pt",
                        verify_artifacts=False,
                        **kwargs,
                    ) as server:
                        self.assertIs(server.tf32, expected)
                        self.assertIs(
                            context.Process.call_args.kwargs["args"][-2], expected
                        )

    def test_invalid_gpu_counts(self):
        for value in [True, False, 1.0, "1", None]:
            with self.subTest(value=value), self.assertRaises(TypeError):
                _validate_gpu_count(value)
        for value in [0, -1]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                _validate_gpu_count(value)

    def test_missing_and_insufficient_devices(self):
        with patch(
            "virtual_diagnostics.tabular.api.torch.cuda.is_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "No CUDA GPU"):
                Kumo(None, None, weights="unused.pt")
        with (
            patch(
                "virtual_diagnostics.tabular.api.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "virtual_diagnostics.tabular.api.torch.cuda.device_count",
                return_value=1,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Requested GPU=2, but only 1"):
                Kumo(None, None, GPU=2, weights="unused.pt")

    def test_model_limit_and_numpy_integer(self):
        with (
            patch(
                "virtual_diagnostics.tabular.api.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "virtual_diagnostics.tabular.api.torch.cuda.device_count",
                return_value=8,
            ),
        ):
            self.assertEqual(_validate_gpu_count(np.int64(4)), 4)
            with self.assertRaisesRegex(ValueError, "four fixed output groups"):
                _validate_gpu_count(5)

    def test_driver_query_error(self):
        with patch(
            "virtual_diagnostics.tabular.api.torch.cuda.is_available",
            side_effect=RuntimeError("driver error"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Unable to query CUDA") as raised:
                _validate_gpu_count(1)
            self.assertIn("driver error", str(raised.exception.__cause__))

    def test_invalid_context_before_worker_launch(self):
        with (
            patch(
                "virtual_diagnostics.tabular.api._validate_gpu_count", return_value=1
            ),
            patch("virtual_diagnostics.tabular.api.mp.get_context") as launch,
        ):
            for x, y in [
                (np.zeros((2, 88)), np.zeros((2, 21))),
                (np.full((2152, 88), np.nan), np.zeros((2152, 21))),
                (np.zeros((2152, 88)), np.zeros((2152, 21))),
            ]:
                with self.assertRaises(ValueError):
                    Kumo(x, y, weights="unused.pt")
            launch.assert_not_called()

    def test_worker_error_exit_and_timeout(self):
        process, pipe = Mock(name="process"), Mock(name="pipe")
        process.name = "Kumo-GPU-0"
        process.exitcode = 9
        pipe.recv.return_value = ("error", "CUDA GPU 0: OutOfMemoryError: no space")
        with patch("virtual_diagnostics.tabular.api.wait", return_value=[pipe]):
            with self.assertRaisesRegex(RuntimeError, "OutOfMemoryError"):
                Kumo._receive((process, pipe), "ready", time.monotonic() + 1)
        with patch(
            "virtual_diagnostics.tabular.api.wait", return_value=[process.sentinel]
        ):
            with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
                Kumo._receive((process, pipe), "ready", time.monotonic() + 1)
        with patch("virtual_diagnostics.tabular.api.wait", return_value=[]):
            with self.assertRaises(TimeoutError):
                Kumo._receive((process, pipe), "ready", time.monotonic() + 1)

    def test_prediction_failure_closes_workers_and_rejects_reuse(self):
        server = self.make_server()
        process, pipe = server._workers[0]
        # Input mistakes must not destroy a healthy instance.
        with self.assertRaises(ValueError):
            server.predict(np.zeros(87))
        self.assertFalse(server._closed)
        pipe.send.side_effect = BrokenPipeError("worker died")
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            server.predict(np.zeros(88))
        process.close.assert_called_once()
        pipe.close.assert_called_once()
        self.assertTrue(server._closed)
        server.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            server.predict(np.zeros(88))

    def test_partial_startup_failure_cleans_started_workers(self):
        x = np.arange(2152 * 88, dtype=np.float32).reshape(2152, 88)
        process = Mock()
        process.is_alive.return_value = False
        parent, child = Mock(), Mock()
        context = Mock()
        context.Pipe.return_value = parent, child
        context.Process.return_value = process
        with (
            patch(
                "virtual_diagnostics.tabular.api._validate_gpu_count", return_value=1
            ),
            patch(
                "virtual_diagnostics.tabular.api.mp.get_context", return_value=context
            ),
            patch.object(Kumo, "_receive", side_effect=RuntimeError("startup failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                Kumo(
                    x, np.zeros((2152, 21)), weights="unused.pt", verify_artifacts=False
                )
        process.close.assert_called_once()
        parent.close.assert_called_once()
        child.close.assert_called_once()

    def test_group_predictions_keep_column_and_quantile_order(self):
        runner = object.__new__(_GroupRunner)
        runner.models = {}
        for columns in GROUPS:
            quantiles = torch.tensor(columns, dtype=torch.float32)[
                :, None, None
            ] + torch.arange(3)
            runner.models[columns] = Mock(
                predict=Mock(return_value=Mock(numerical=quantiles))
            )
        query = torch.arange(88, dtype=torch.float32).unsqueeze(0)
        with patch(
            "virtual_diagnostics.tabular.api.torch.autocast", return_value=nullcontext()
        ):
            actual = runner.targets(query)
        expected = torch.tensor([[col + 1 for group in GROUPS for col in group]])
        torch.testing.assert_close(actual, expected.float(), rtol=0, atol=0)
        for columns, model in runner.models.items():
            torch.testing.assert_close(
                model.predict.call_args.args[0],
                query.unsqueeze(0).expand(len(columns), -1, -1),
            )

    def test_worker_startup_error_reports_device_and_closes_pipe(self):
        pipe = Mock()
        with patch(
            "virtual_diagnostics.tabular.api._GroupRunner",
            side_effect=torch.cuda.OutOfMemoryError("no space"),
        ):
            _worker(pipe, 2, GROUPS, None, None, 20261009, True, "unused.pt")
        kind, message = pipe.send.call_args.args[0]
        self.assertEqual(kind, "error")
        self.assertIn("CUDA GPU 2: OutOfMemoryError: no space", message)
        self.assertIn("Free GPU memory", message)
        pipe.close.assert_called_once()

    def test_shutdown_escalates_only_as_needed_and_is_idempotent(self):
        for alive, stops, joins in [
            ([True, False], [], 1),
            ([True, True, False], ["terminate"], 2),
            ([True, True, True], ["terminate", "kill"], 3),
        ]:
            with self.subTest(stops=stops):
                server = self.make_server()
                process, pipe = server._workers[0]
                process.is_alive.side_effect = alive
                server.close()
                server.close()
                self.assertEqual(
                    [
                        call[0]
                        for call in process.method_calls
                        if call[0] in ("terminate", "kill")
                    ],
                    stops,
                )
                self.assertEqual(process.join.call_count, joins)
                process.close.assert_called_once()
                pipe.send.assert_called_once_with(None)
                pipe.close.assert_called_once()

    def test_expired_deadline_does_not_wait(self):
        with patch("virtual_diagnostics.tabular.api.wait") as wait:
            with self.assertRaises(TimeoutError):
                Kumo._receive((Mock(), Mock()), "targets", time.monotonic() - 1)
            wait.assert_not_called()


if __name__ == "__main__":
    unittest.main()
