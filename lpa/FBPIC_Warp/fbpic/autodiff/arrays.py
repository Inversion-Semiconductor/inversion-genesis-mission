"""Retain borrowed CuPy arrays and shared gradient buffers for operator tapes."""

from typing import Any

import numpy as np
import warp as wp

from fbpic.utils.warp import as_warp_array


class DifferentiableArrays:
    """Own stable Warp views for one operator graph.

    CuPy owns primal storage. This registry retains its owners and allocates
    Warp-owned gradients. Exact aliases share one view and gradient buffer.
    Nonexact views with overlapping storage ranges are rejected for gradients.
    """

    def __init__(self) -> None:
        self._views: dict[tuple, wp.array] = {}

    def view(self, array: Any, *, requires_grad: bool = True) -> wp.array:
        """Borrow FP64 storage without a host transfer.

        Args:
            array: A CuPy float64 or complex128 array on the selected device.
                Integer arrays are accepted when requires_grad is false.
            requires_grad: Allocate a persistent gradient for continuous inputs.

        Returns:
            A stable view retaining the primal owner and its gradient storage.

        Raises:
            TypeError: Requested derivatives use unsupported precision or integers.
            ValueError: Views overlap or use inconsistent gradient policies.
        """
        if requires_grad and array.dtype not in (
            np.dtype(np.float64),
            np.dtype(np.complex128),
        ):
            raise TypeError("Kernel gradients require float64 or complex128 arrays.")
        key = (
            array.device.id,
            array.data.ptr,
            array.shape,
            array.strides,
            array.dtype.str,
        )
        if key in self._views:
            view = self._views[key]
            if view.requires_grad != requires_grad:
                raise ValueError("An existing view has a different gradient policy.")
            return view
        interval = self._storage_range(array)
        for existing in self._views.values():
            if not (requires_grad or existing.requires_grad):
                continue
            other = self._storage_range(existing._cupy_owner)
            if array.device.id == existing.device.ordinal and (
                max(interval[0], other[0]) < min(interval[1], other[1])
            ):
                raise ValueError(
                    "Gradient views require nonoverlapping storage ranges."
                )
        view = as_warp_array(array)
        if requires_grad:
            view.requires_grad = True
        self._views[key] = view
        return view

    @staticmethod
    def _storage_range(array) -> tuple[int, int]:
        pointer = array.data.ptr
        if not array.size:
            return pointer, pointer
        offsets = [
            (count - 1) * stride for count, stride in zip(array.shape, array.strides)
        ]
        return (
            pointer + sum(min(0, offset) for offset in offsets),
            pointer + sum(max(0, offset) for offset in offsets) + array.dtype.itemsize,
        )

    def zero_gradients(self) -> None:
        """Reset allocated gradients on their current Warp stream."""
        for view in self._views.values():
            if view.grad is not None:
                view.grad.zero_()
