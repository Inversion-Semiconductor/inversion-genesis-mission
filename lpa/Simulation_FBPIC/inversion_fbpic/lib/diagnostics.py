"""Abstract simulation analysis datapoints with explicit completion tracking."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, ClassVar

import attrs

from inversion_fbpic.lib.datapoint import _Datapoint

logger = logging.getLogger(__name__)


@attrs.define(kw_only=True, slots=False)
class _Diagnostic(_Datapoint):
    """Abstract datapoint populated by explicit or scheduled analysis.

    Concrete subclasses declare a ``SUBCLASS`` tag and implement ``_analyze``
    to return a dictionary of results. Analysis never runs during construction,
    loading, or serialization. Call ``analyze()`` to replace ``data`` and mark
    this instance complete. Serializing an incomplete instance logs a warning
    but still saves its current data.

    Completion is runtime-only: loading saved results preserves ``data`` but
    starts a new, incomplete instance. Subclasses should declare their analysis
    inputs as normal init fields so those inputs can also be saved and reloaded.

    Subclasses set ``RUN_BEFORE_SIMULATION`` to select automatic analysis at
    the end of simulation setup (True) or after simulation stepping (False,
    the default). This class-level policy is not an instance input or serialized
    parameter.
    """

    RUN_BEFORE_SIMULATION: ClassVar[bool] = False

    _analysis_complete: bool = attrs.field(
        default=False, init=False, repr=False, eq=False
    )

    @property
    def analysis_complete(self) -> bool:
        """Whether the most recent explicit analysis completed successfully."""
        return self._analysis_complete

    @abstractmethod
    def _analyze(self) -> dict[str, Any]:
        """Perform analysis and return the desired result dictionary."""
        raise NotImplementedError

    def analyze(self) -> None:
        """Run analysis and store its results, marking completion only on success.

        Every call reruns ``_analyze()``. If it raises or returns something other
        than a dictionary, completion remains false and this wrapper does not
        replace the previous data. Mutations or external side effects performed
        by subclass code are not rolled back.

        Raises:
            TypeError: If ``_analyze()`` does not return a dictionary.
        """
        self._analysis_complete = False
        results = self._analyze()
        if not isinstance(results, dict):
            raise TypeError(
                f"{type(self).__name__}._analyze() must return a dict, "
                f"got {type(results).__name__}."
            )
        self.data = results
        self._analysis_complete = True

    def to_dict(self, *, include_nones: bool = True) -> dict[str, Any]:
        """Serialize current inputs and results, warning if analysis is incomplete."""
        if not self.analysis_complete:
            logger.warning(
                "%s analysis is incomplete; call analyze() before serializing "
                "to record completed results.",
                type(self).__name__,
            )
        return super().to_dict(include_nones=include_nones)
