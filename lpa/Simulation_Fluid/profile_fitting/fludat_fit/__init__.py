"""Fit parameterized FBPIC density profiles to gas-jet fluid-simulation lineouts.

The pieces, in the order a fit flows through them:

* :mod:`fludat_fit.lineout` - a :class:`~fludat_fit.lineout.Lineout` is one
  ``density(z)`` curve at fixed physical conditions
  (:class:`~fludat_fit.lineout.LineoutConditions`: ``x_mm``, ``pressure_bar``,
  ``angle_deg``), extracted from a ``fludat_proc`` density cube.
* :mod:`fludat_fit.families` - a :class:`~fludat_fit.families.ProfileFamily`
  wraps one ``inversion_fbpic`` density-profile class: it declares a bounded
  :class:`~fludat_fit.parameters.ParameterSpace` and builds the profile from a
  parameter vector.
* :mod:`fludat_fit.fitting` - a :class:`~fludat_fit.fitting.FittingScheme`
  turns ``(lineout, family)`` into a :class:`~fludat_fit.fitting.FitResult`.
  :class:`~fludat_fit.fitting.MultiStartLocalFit` is the bounded L-BFGS-B
  multi-start implementation; a learned model would implement the same protocol.
* :mod:`fludat_fit.goodness_of_fit` - metrics and ranking shared by every scheme.
* :mod:`fludat_fit.dataset` - a :class:`~fludat_fit.dataset.NozzleDataset` samples a
  cube along the oblique line selected by ``(x_mm, pressure_bar, angle_deg)``.
* :mod:`fludat_fit.explore_fits` / :mod:`fludat_fit.fit_statistics` - the
  interactive explorer and the fitting-performance study (command line).
"""

from .dataset import LineoutPath, NozzleDataset, trilinear_density
from .families import (
    DEFAULT_FAMILY_NAMES,
    ProfileFamily,
    family_registry,
    get_families,
)
from .fitting import (
    FamilyComparison,
    FitResult,
    FittingScheme,
    MultiStartLocalFit,
    compare_families,
    fit_family,
    fit_lineouts,
)
from .goodness_of_fit import RANKING_METRICS, GoodnessOfFit, rank_results
from .lineout import (
    Lineout,
    LineoutConditions,
    LineoutSummary,
    extract_lineout,
    extract_lineouts,
    load_lineout,
)
from .parameters import ParameterSpace, ParameterSpec

__all__ = [
    "DEFAULT_FAMILY_NAMES",
    "FamilyComparison",
    "FitResult",
    "FittingScheme",
    "GoodnessOfFit",
    "Lineout",
    "LineoutConditions",
    "LineoutPath",
    "LineoutSummary",
    "MultiStartLocalFit",
    "NozzleDataset",
    "ParameterSpace",
    "ParameterSpec",
    "ProfileFamily",
    "RANKING_METRICS",
    "compare_families",
    "extract_lineout",
    "extract_lineouts",
    "family_registry",
    "fit_family",
    "fit_lineouts",
    "get_families",
    "load_lineout",
    "rank_results",
    "trilinear_density",
]
