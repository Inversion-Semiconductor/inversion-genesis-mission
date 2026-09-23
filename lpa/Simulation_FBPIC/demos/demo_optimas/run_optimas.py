"""Example Bayesian optimization of an LPA with FBPIC.

This example optimizes an LPA based on downramp injection using FBPIC
simulations.

The FBPIC simulations are performed using the template defined in
`sim_template.py`.

Run using `python run_optimas.py &`.
Do not use `mpirun` here, this script handles GPU allocation on its own.

In addition to the objective `f`, additional parameters are analyzed for each
simulation and included in the optimization history.
"""

import logging
import os
from typing import List

from optimas.core import Parameter

from inversion_fbpic.utils.optimas_template import (
    VaryingParameter,
    run_optimas_exploration,
)
from inversion_fbpic.utils.optimas_analysis import analyze_hofi_simulation

# Customization.
MAX_EVALS = 80

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("optimization.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Create varying parameters.
varying_params: List[VaryingParameter] = [
    VaryingParameter("laser_energy", 4.0, 6.0, default_value=5.0),
    VaryingParameter("laser_waist", 20e-6, 40e-6, default_value=30e-6),
    VaryingParameter("focal_position", 1e-3, 6e-3, default_value=3e-3),
    VaryingParameter("tau_fwhm_fs", 25, 50, default_value=40),
    VaryingParameter("flattop_plasma_density_e18", 0.4, 2.0, default_value=1.0),
    VaryingParameter("downramp_length", 20e-6, 100e-6, default_value=50e-6),
    VaryingParameter("downramp_height", 0.1, 1, default_value=0.1),
    VaryingParameter("downramp_position", 250e-6, 1e-3, default_value=300e-6),
]


# Define additional parameters to analyze.
energy_mean_param: Parameter = Parameter("energy_mean")
energy_med_param: Parameter = Parameter("energy_med")
energy_mad_param: Parameter = Parameter("energy_mad")
charge_param: Parameter = Parameter("charge")
charge_weighted_param: Parameter = Parameter("charge_weighted")
analyzed_parameters: list[Parameter] = [
    energy_mean_param,
    energy_med_param,
    energy_mad_param,
    charge_param,
    charge_weighted_param,
]


# Get the directory where this script is located
script_dir: str = os.path.dirname(os.path.abspath(__file__))
template_path: str = os.path.join(script_dir, "sim_template.py")


def my_analyze_hofi_simulation(*args, **kwargs):
    return analyze_hofi_simulation(*args, file_tree="diags/hdf5", **kwargs)


if __name__ == "__main__":
    run_optimas_exploration(
        analysis_func=my_analyze_hofi_simulation,
        sim_template=template_path,
        varying_parameters=varying_params,
        analyzed_parameters=analyzed_parameters,
        max_evals=MAX_EVALS,
    )
