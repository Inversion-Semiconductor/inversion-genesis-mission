#!/usr/bin/env python3
"""Fit every profile in ``profile_parameters_template.json`` at given (x, p, angle) points.

Fills in the template's ``null`` fields from local optimisation fits and writes
one config per point next to the template. Every fitted profile shares one
``center``/``centroid`` (the same concept under two names): the point along
*this* lineout where it crosses the physical ``z_m = 0`` plane, i.e.
``dataset.path_for(conditions).t_at_physical_z(0.0)``.

The gas target itself is symmetric about that physical plane, but a lineout
is not generally symmetric about it in its own coordinate: an oblique line
(or an axial one off the ``x = 0`` centreline) samples the symmetric 2D field
along an asymmetric path, so the lineout's *own* density-weighted centroid is
generally offset from ``z_m = 0`` even though the target's true centre is
not. Fitting each family to its own weighted centroid would therefore
reproduce that sampling artefact rather than the target's real geometry, and
would disagree between families besides. Every profile is pinned instead to
the same physical anchor, which for an axial lineout at ``x = 0`` happens to
coincide with the data's own peak, but in general will not.

That position parameter is pinned *during* fitting (via
:class:`~fludat_fit.families.FixedParameterFamily`: ``center`` on
``GenericConicalTarget``, the dominant term's ``c_0`` on
``GeneralizedLorentzianSum``), not translated afterwards, so every other
parameter (``skew_rate``, satellite-term amplitudes and widths, ...) is still
free to best fit the data given that fixed anchor. For ``GenericConicalTarget``
this pins the centroid exactly, since ``center`` *is* this class's centroid.
``GeneralizedLorentzianSum`` has no single centre parameter, but per this
module's premise its reported ``centroid`` is the same fixed anchor, by
definition, not a value computed from the fitted terms.

Every written-out profile is then rigidly re-based into its own frame where
its ``get_z_extent()`` starts at exactly ``0.0`` — the same convention
``InterpolateFromH5Profile``'s ``centering_mode="left"`` already uses for
``h5_direct`` (left unchanged here, since that class positions itself). A
profile's support is generally asymmetric about its anchor (skew, fringes,
satellite terms), so this re-basing shift differs per profile even within one
config. After the shift the anchor is no longer at ``0``; the *reported*
``center``/``centroid`` is updated to the anchor's new position in that
profile's own frame (``-z_extent[0]`` measured before the shift), which is
what a consumer needs to place the laser focus correctly once it has
instantiated the profile in its own left-based frame.

Example (conda env inv-fbpic), from this directory:

    python -m fludat_fit.generate_profile_configs \\
        ../postproc/data/density_field/htu_dens_7_0.h5 \\
        data/fit_cfgs/profile_parameters_template.json \\
        --point 1.0 20.0 0.0 --point 2.0 20.0 30.0 --point 3.0 20.0 -30.0 \\
        --output-dir data/fit_cfgs

Only the template's profile families are fitted (matched by ``class`` and, for
``GenericConicalTarget``, its ``main_profile_type``/``fringe_profile_type``);
``h5_direct`` (or any other profile with no ``null`` in its ``kwargs``) is
copied through unchanged, since it needs no fit.
"""

from __future__ import annotations

if __package__ in (None, ""):  # run as a plain script
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fludat_fit"

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

from .cli_common import MM_PER_M, FitWindow, add_window_arguments, fit_window_from_args
from .dataset import NozzleDataset
from .families import (
    FixedParameterFamily,
    GeneralizedLorentzianSumFamily,
    GenericConicalTargetFamily,
    ProfileFamily,
)
from .fitting import FitResult, MultiStartLocalFit
from .lineout import LineoutConditions
from .profile_reconstruction import center_of

CLASS_TO_FAMILY = {
    "GeneralizedLorentzianSum": None,  # resolved by term count, see _family_for
    "GenericConicalTarget": None,  # resolved by main/fringe type, see _family_for
}
"""Profile ``class`` values this generator knows how to fit."""


def _is_unfilled(value: Any) -> bool:
    """True if ``value`` contains a ``null`` anywhere (dict/list walked recursively)."""
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_is_unfilled(v) for v in value.values())
    if isinstance(value, list):
        return any(_is_unfilled(v) for v in value)
    return False


def _family_for(
    name: str, spec: dict[str, Any], target_centroid: float
) -> ProfileFamily:
    """The :class:`ProfileFamily` matching one template profile entry, with its
    position parameter pinned to ``target_centroid`` (see module docstring)."""
    class_name = spec["class"]
    if class_name == "GeneralizedLorentzianSum":
        n_terms = len(spec["kwargs"]["parameters"])
        family = GeneralizedLorentzianSumFamily(n_terms=n_terms)
        return FixedParameterFamily(family, "c_0", target_centroid)
    if class_name == "GenericConicalTarget":
        kwargs = spec["kwargs"]
        family = GenericConicalTargetFamily(
            main_profile_type=kwargs["main_profile_type"],
            fringe_profile_type=kwargs.get("fringe_profile_type"),
            fringe_side=(kwargs.get("fringe_side") or "both"),
        )
        return FixedParameterFamily(family, "center", target_centroid)
    raise ValueError(
        f"profile {name!r}: don't know how to fit class {class_name!r}; "
        "fill it in by hand or extend _family_for"
    )


def _shift_to_left_edge_zero(profile: Any) -> float:
    """The constant that, added to every position parameter, makes
    ``profile.get_z_extent()[0] == 0`` exactly.

    Matches ``InterpolateFromH5Profile``'s own ``centering_mode="left"``
    convention (used unchanged for ``h5_direct``): a profile's support is
    generally asymmetric about its anchor (skew, fringes, satellite terms), so
    this shift is specific to each profile, computed from its own (as-fitted,
    not yet shifted) ``get_z_extent()``.
    """
    z_min, _z_max = profile.get_z_extent()
    return -z_min


def _lorentzian_sum_kwargs(
    result: FitResult, target_centroid: float
) -> tuple[dict[str, Any], float]:
    """``(kwargs, centroid)`` for a ``GeneralizedLorentzianSum`` template entry.

    ``result`` comes from fitting with the dominant term's ``c_0`` pinned to
    ``target_centroid`` (see :func:`_family_for`). Every term is then shifted
    by the same constant (see :func:`_shift_to_left_edge_zero`), an exact rigid
    translation of the whole profile, so the reported ``centroid`` is that
    same fixed anchor in its new, shifted position — not a density-weighted
    average recomputed from the fitted terms, which would generally disagree
    with it once there is more than one term.

    ``density_cutoff_ratio`` is recorded explicitly (the family's value, not
    necessarily the class's own default) because :func:`_shift_to_left_edge_zero`
    depends on it through ``get_z_extent()``: a reconstruction that silently
    fell back to a different ratio would no longer have its left edge at
    exactly ``0``.
    """
    profile = result.build_profile()
    shift = _shift_to_left_edge_zero(profile)
    parameters = [
        {"A": term.A, "c": term.c + shift, "w": term.w, "b": term.b, "m": term.m}
        for term in profile.parameters
    ]
    kwargs = {
        "parameters": parameters,
        "density_cutoff_ratio": profile.density_cutoff_ratio,
    }
    return kwargs, target_centroid + shift


def _conical_kwargs(
    result: FitResult, template_kwargs: dict[str, Any], target_centroid: float
) -> dict[str, Any]:
    """``kwargs`` for a ``GenericConicalTarget`` template entry, from the fitted parameters.

    ``center`` *is* this class's centroid (``_centered`` sets
    ``start_position`` so that ``profile.centroid == center`` exactly), so
    shifting it by :func:`_shift_to_left_edge_zero` is an exact rigid
    translation of the whole profile, giving it the same (shifted) anchor.
    """
    parameters = result.parameters
    family = result.family
    if isinstance(family, FixedParameterFamily):
        family = family.family
    if not isinstance(family, GenericConicalTargetFamily):
        raise TypeError("conical config generation requires a conical target family")
    shift = _shift_to_left_edge_zero(result.build_profile())
    kwargs: dict[str, Any] = {
        "nominal_density": result.nominal_density_m3(),
        "center": target_centroid + shift,
        "main_profile_type": template_kwargs["main_profile_type"],
        "main_profile_parameters": family._main_parameters(parameters),
    }
    if template_kwargs.get("fringe_profile_type") is not None:
        kwargs.update(
            fringe_profile_type=template_kwargs["fringe_profile_type"],
            fringe_profile_parameters={
                key: parameters[f"fringe_{key}"]
                for key in template_kwargs["fringe_profile_parameters"]
            },
            fringe_center_offset=parameters["fringe_center_offset"],
            fringe_relative_height=parameters["fringe_relative_height"],
            fringe_side=template_kwargs.get("fringe_side") or "both",
        )
    kwargs["skew_rate"] = parameters.get("skew_rate", 0.0)
    return kwargs


def fill_profile(
    name: str, template_spec: dict[str, Any], result: FitResult, target_centroid: float
) -> dict[str, Any]:
    """The filled-in ``profiles[name]`` entry for one fitted profile, re-based
    into its own left-edge-at-zero frame (see :func:`_lorentzian_sum_kwargs`/
    :func:`_conical_kwargs` and the module docstring)."""
    spec = copy.deepcopy(template_spec)
    class_name = spec["class"]
    if class_name == "GeneralizedLorentzianSum":
        kwargs, centroid = _lorentzian_sum_kwargs(result, target_centroid)
        spec["centroid"] = centroid
        spec["kwargs"] = kwargs
    elif class_name == "GenericConicalTarget":
        spec["kwargs"] = _conical_kwargs(result, spec["kwargs"], target_centroid)
    else:  # pragma: no cover - _family_for already rejects this
        raise ValueError(f"unsupported class {class_name!r}")
    return spec


def fill_template(
    template: dict[str, Any],
    dataset: NozzleDataset,
    conditions: LineoutConditions,
    *,
    window: FitWindow,
    scheme: MultiStartLocalFit,
    density_file: str,
    progress: bool = False,
) -> dict[str, Any]:
    """A copy of ``template`` with every field for ``conditions`` filled in."""
    config = copy.deepcopy(template)
    config["density_file"] = density_file
    config["x_mm"] = conditions.x_mm
    config["p_bar"] = conditions.pressure_bar
    config["angle"] = conditions.angle_deg

    lineout = window.apply(dataset.lineout(conditions))
    if lineout.peak <= 0.0:
        raise ValueError(
            f"{conditions.label()}: lineout has no density inside the fit window"
        )
    # The point along this lineout where it crosses the gas target's physical
    # z_m = 0 symmetry plane — the one anchor every profile family must share.
    # Not a density-weighted average of the (possibly asymmetrically sampled)
    # lineout data; see the module docstring.
    target_centroid = dataset.path_for(conditions).t_at_physical_z(0.0)
    if progress:
        print(
            f"  {conditions.label()}: target centroid (z_m=0 plane) = "
            f"{target_centroid * MM_PER_M:.4g} mm",
            file=sys.stderr,
        )

    filled_profiles = {}
    for name, spec in template["profiles"].items():
        if not _is_unfilled(spec.get("kwargs", {})) and "centroid" not in spec:
            filled_profiles[name] = copy.deepcopy(
                spec
            )  # already complete (e.g. h5_direct)
            continue
        family = _family_for(name, spec, target_centroid)
        result = scheme.fit(lineout, family)
        filled_spec = fill_profile(name, spec, result, target_centroid)
        if progress:
            reported = center_of(filled_spec)
            print(
                f"  {conditions.label()}: {name} ({family.family.name}) "
                f"NRMSE={result.goodness.nrmse:.4g} "
                f"center/centroid={reported * MM_PER_M:.4g} mm (its own left-edge-zero frame)",
                file=sys.stderr,
            )
        filled_profiles[name] = filled_spec
    config["profiles"] = filled_profiles
    return config


def _point_stem(conditions: LineoutConditions) -> str:
    return (
        f"profile_parameters_x{conditions.x_mm:g}mm"
        f"_p{conditions.pressure_bar:g}bar_a{conditions.angle_deg:g}deg"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("hdf5_path", type=Path, help="Density cube.")
    parser.add_argument(
        "template", type=Path, help="Template JSON (profile_parameters_template.json)."
    )
    add_window_arguments(parser)
    parser.add_argument(
        "--point",
        dest="points",
        nargs=3,
        type=float,
        action="append",
        metavar=("X_MM", "P_BAR", "ANGLE_DEG"),
        required=True,
        help="A physical parameter point; repeat --point for several.",
    )
    parser.add_argument(
        "--starts",
        type=int,
        default=16,
        help="Local-fit starting points per profile; default 16.",
    )
    parser.add_argument(
        "--optimizer",
        default="L-BFGS-B",
        help="Bounded scipy.optimize.minimize method; default L-BFGS-B.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Start-sample seed; default 0."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the filled configs; default: the template's directory.",
    )
    parser.add_argument(
        "--density-file",
        default=None,
        help="density_file value written into the configs; default: hdf5_path relative to "
        "--output-dir, so the configs and cube can be relocated together.",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="No per-profile progress."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    template = json.loads(args.template.read_text())
    output_dir = args.output_dir or args.template.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    density_file = args.density_file or os.path.relpath(
        args.hdf5_path.resolve(), output_dir.resolve()
    )

    dataset = NozzleDataset.load(args.hdf5_path)
    window = fit_window_from_args(args)
    scheme = MultiStartLocalFit(
        n_starts=args.starts, method=args.optimizer, seed=args.seed
    )

    written = []
    for x_mm, p_bar, angle_deg in args.points:
        conditions = LineoutConditions(
            x_mm=x_mm, pressure_bar=p_bar, angle_deg=angle_deg
        )
        if not args.quiet:
            print(f"# fitting {conditions.label()}", file=sys.stderr)
        config = fill_template(
            template,
            dataset,
            conditions,
            window=window,
            scheme=scheme,
            density_file=density_file,
            progress=not args.quiet,
        )
        path = output_dir / f"{_point_stem(conditions)}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        written.append(path)
        print(f"wrote {path}")

    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
