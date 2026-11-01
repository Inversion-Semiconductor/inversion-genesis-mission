"""Residual objective of a twin target (plan Section 2.4, Phase P7).

`residual_objective(config, variant)` builds `fbpic.autodiff.loss.
ResidualObjective` from the target file, the whitening file of Phase P3, and
the selection of the configuration. The residuals equal those of the P3 FD
recovery (`observable.whitened_residuals`), which `tests/test_autodiff_
residual.py` checks.
"""

from __future__ import annotations

import numpy as np

from make_target import load_target
from target_covariance import load_whitening


def residual_objective(config, variant="T0", species="elec_from_N"):
    from fbpic.autodiff.loss import ResidualObjective

    target = load_target("twin_s", variant, config)
    whitening = load_whitening(config)
    sel = config["selection"]
    return ResidualObjective(
        species,
        sel["u_c"],
        sel["delta_u"],
        target["Q"],
        np.asarray(target["moments"], float),
        target["Q_f"],
        np.asarray(whitening["L"], float),
        np.asarray(whitening["floor_sigma"], float),
        float(whitening["floor_mean_uz"]),
    )
