"""Generate `lpa_bunch.bob`, a Phoebus display of the LPA bunch that the twin server serves.

The display shows the twin's `Source_*` readbacks (the LPA bunch as injected), the 33
moment-descriptor scalars (the covariance as an upper-triangular grid) and the run statistics.
Every PV is `pva://HTU:SIM:<name>`, under the prefix `python -m lume_fbpic_twin.serve --twin`
uses.

A combo box selects among the archives the server was given (the `LPA_Archive` PV).
A button opens `htu_synoptic_lpa.bob`, the twin's synoptic with the LPA tied in, which
`generate_linked_synoptic.py` writes next to this file.

The descriptor's PV names come from `lume_fbpic.descriptor` and the prefix
`make_descriptor_actions()` gives them, so the display follows the code. Run `python generate_lpa_display.py [output.bob]` to regenerate it; do not edit the `.bob`
by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

from inversion_fbpic.utils.distributions import COORD_NAMES, LONGITUDINAL_PROFILE_BINS
from lume_fbpic.actions import DESCRIPTOR_PREFIX
from lume_fbpic.descriptor import covariance_feature, slice_features
from lume_fbpic.utils import STATISTICS
import widgets  # docs/htu-twin/phoebus/widgets.py, this directory

from lume_fbpic_twin.archive_selector import DEFAULT_SELECTOR_NAME
from lume_fbpic_twin.serve import TWIN_PREFIX

COORD_LABELS = ("x [m]", "ux", "y [m]", "uy", "z [m]", "uz")

# Source readbacks that are counts, shown without decimals.
INTEGER_SOURCE = ("Source_NumParticles",)

# (PV name, label) of the twin's source readbacks, in two columns. POSIT: the names of all the
# twin's source variables (`htu.model.SourceVariable`, `SourceIntVariable`); a test checks it.
SOURCE = (
    ("Source_Energy_MeV", "Energy [MeV]"),
    ("Source_EnergySpread_pct", "Energy spread [% rms]"),
    ("Source_Charge_pC", "Charge [pC]"),
    ("Source_NumParticles", "Macroparticles"),
    ("Source_BetaX_mm", "Beta x [mm]"),
    ("Source_BetaY_mm", "Beta y [mm]"),
    ("Source_AlphaX", "Alpha x"),
    ("Source_AlphaY", "Alpha y"),
    ("Source_NormEmitX_um", "Norm. emit. x [um]"),
    ("Source_NormEmitY_um", "Norm. emit. y [um]"),
    ("Source_X_um", "Centroid x [um]"),
    ("Source_Y_um", "Centroid y [um]"),
    ("Source_Xp_mrad", "Pointing x [mrad]"),
    ("Source_Yp_mrad", "Pointing y [mrad]"),
)


def build() -> str:
    """The display's XML."""
    left, grid_label_w, cell_w, cell_h = 20, 70, 108, 22
    width = left + grid_label_w + 6 * cell_w + 20
    root = widgets.display("LPA bunch", width, 700)

    widgets.label(root, "Title", "LPA bunch", left, 10, 300, 28, bold=True, size=16.0)
    widgets.label(
        root,
        "Subtitle",
        "Served by lume_fbpic_twin.serve --twin. The values are the selected run's recorded results.",
        left,
        40,
        width - 2 * left,
        20,
        size=10.0,
    )

    widgets.button(
        root,
        "SynopticButton",
        "HTU synoptic...",
        "htu_synoptic_lpa.bob",
        "The HTU twin synoptic with the LPA bunch tied in (see generate_linked_synoptic.py)",
        width - left - 150,
        12,
        150,
    )

    widgets.label(root, "RunLabel", "LPA run:", left, 70, 70, 24, bold=True)
    widgets.combo(root, "RunSelector", _pv(DEFAULT_SELECTOR_NAME), left + 74, 70, 260)
    widgets.label(
        root,
        "RunNote",
        "Choose a run: its values are loaded here and passed on to the twin as its source.",
        left + 344,
        70,
        width - left - 344 - left,
        24,
        size=10.0,
    )

    y = 108
    widgets.label(
        root,
        "SourceHeader",
        "Twin source (the LPA bunch as injected)",
        left,
        y,
        400,
        22,
        bold=True,
    )
    widgets.rule(root, "SourceRule", left, y + 24, width - 2 * left)
    y += 32
    column_w = (width - 2 * left) // 2
    for index, (pv, text) in enumerate(SOURCE):
        column, row = index % 2, index // 2
        x = left + column * column_w
        widgets.label(root, f"SrcL{index}", text, x, y + row * 26, 150)
        widgets.value(
            root,
            f"SrcV{index}",
            _pv(pv),
            x + 155,
            y + row * 26,
            110,
            precision=0 if pv in INTEGER_SOURCE else 4,
        )
    y += 26 * ((len(SOURCE) + 1) // 2) + 14

    widgets.label(
        root,
        "MomentHeader",
        "Moment descriptor: momentum centroids and charge",
        left,
        y,
        500,
        22,
        bold=True,
    )
    widgets.rule(root, "MomentRule", left, y + 24, width - 2 * left)
    y += 32
    for index, name in enumerate(("mean_ux", "mean_uy", "mean_uz")):
        x = left + index * 2 * cell_w
        widgets.label(root, f"MeanL{index}", name.replace("mean_", "mean "), x, y, 60)
        widgets.value(
            root,
            f"MeanV{index}",
            _pv(DESCRIPTOR_PREFIX + name),
            x + 62,
            y,
            cell_w,
            precision=4,
        )
    widgets.label(
        root, "ChargeL", "Charge [pC]", left + 6 * cell_w - cell_w, y + 26, 90, align=2
    )
    widgets.value(
        root,
        "ChargeV",
        _pv(DESCRIPTOR_PREFIX + "total_beam_charge_pc"),
        left + 6 * cell_w + 2,
        y + 26,
        cell_w,
        precision=2,
    )
    y += 62

    widgets.label(
        root,
        "CovHeader",
        "Covariance of (x, ux, y, uy, z, uz); u = p / (m c)",
        left,
        y,
        500,
        22,
        bold=True,
    )
    widgets.rule(root, "CovRule", left, y + 24, width - 2 * left)
    y += 32
    for column, text in enumerate(COORD_LABELS):
        widgets.label(
            root,
            f"CovCol{column}",
            text,
            left + grid_label_w + column * cell_w,
            y,
            cell_w - 4,
            20,
            bold=True,
            align=1,
        )
    y += 24
    for row, row_text in enumerate(COORD_LABELS):
        widgets.label(
            root,
            f"CovRow{row}",
            row_text,
            left,
            y + row * (cell_h + 2),
            grid_label_w - 4,
            cell_h,
            bold=True,
        )
        for column in range(row, 6):
            widgets.value(
                root,
                f"Cov_{COORD_NAMES[row]}_{COORD_NAMES[column]}",
                _pv(
                    DESCRIPTOR_PREFIX
                    + covariance_feature(COORD_NAMES[row], COORD_NAMES[column])
                ),
                left + grid_label_w + column * cell_w,
                y + row * (cell_h + 2),
                cell_w - 4,
                cell_h,
                exponential=True,
                precision=3,
            )
    y += 6 * (cell_h + 2) + 14

    widgets.label(
        root,
        "SliceHeader",
        "Longitudinal slices along z, tail to head",
        left,
        y,
        560,
        22,
        bold=True,
    )
    widgets.rule(root, "SliceRule", left, y + 24, width - 2 * left)
    y += 32
    widgets.label(root, "SliceMeanL", "mean uz", left, y + 24, grid_label_w - 4)
    widgets.label(root, "SliceRmsL", "rms uz", left, y + 50, grid_label_w - 4)
    for index in range(LONGITUDINAL_PROFILE_BINS):
        mean_name, rms_name = slice_features(index)
        x = left + grid_label_w + index * cell_w
        widgets.label(
            root,
            f"SliceCol{index}",
            f"slice {index}",
            x,
            y,
            cell_w - 4,
            20,
            bold=True,
            align=1,
        )
        widgets.value(
            root,
            f"SliceMean{index}",
            _pv(DESCRIPTOR_PREFIX + mean_name),
            x,
            y + 24,
            cell_w - 4,
            precision=4,
        )
        widgets.value(
            root,
            f"SliceRms{index}",
            _pv(DESCRIPTOR_PREFIX + rms_name),
            x,
            y + 50,
            cell_w - 4,
            precision=4,
        )
    y += 84

    widgets.label(
        root,
        "StatsHeader",
        "Run statistics (blank/NaN for a reconstructed run: no particles)",
        left,
        y,
        560,
        22,
        bold=True,
    )
    widgets.rule(root, "StatsRule", left, y + 24, width - 2 * left)
    y += 32
    for index, (pv, statistic) in enumerate(STATISTICS.items()):
        text = f"{statistic.description} [{statistic.unit}]"
        x = left + index * ((width - 2 * left) // 3)
        widgets.label(root, f"StatL{index}", text, x, y, 175)
        widgets.value(root, f"StatV{index}", _pv(pv), x, y + 24, 140, precision=4)
    y += 60
    root.find("height").text = str(y)
    return widgets.to_xml(root)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    output = Path(argv[0]) if argv else Path(__file__).with_name("lpa_bunch.bob")
    output.write_text(build())
    print(f"wrote {output}")


def _pv(name: str) -> str:
    """The PV `name`, under the prefix `lume_fbpic_twin.serve --twin` uses."""
    return f"pva://{TWIN_PREFIX}{name}"


if __name__ == "__main__":
    main()
