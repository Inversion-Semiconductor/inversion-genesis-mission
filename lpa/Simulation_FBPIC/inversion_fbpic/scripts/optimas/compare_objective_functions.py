"""Compare objective functions for electron beam distributions.

This script takes two different objective functions (defined in
`utils/optimas_analysis.py` or `legacy_objective_functions.py`) and compares
their values for one or more electron beam distributions. The goal is to find
an objective function that correctly ranks distributions for the chirped
metrology light source.

The script evaluates both objective functions on a list of test cases and
generates a comparison plot showing normalized objective function values.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import inversion_fbpic.utils.analysis as an
from inversion_fbpic.utils.optimas_analysis import (
    analyze_pax_beam as obj_func_1,
)
from legacy_objective_functions import (
    analyze_chirped_beam_normalized_by_slice as obj_func_2,
)

# List of paths to data to be used for comparison
BASE_FOLDER = "../../../../../../inversion-fbpic-runscripts/optimas/metrology/ionization_injection/"
LIST_OF_TEST_CASES: list[str] = [
    BASE_FOLDER + "internal/version_2/attempt_4/sim0290/lab_diags/hdf5",
    BASE_FOLDER + "internal/version_2/attempt_4/sim0301/lab_diags/hdf5",
    BASE_FOLDER + "internal/version_2/downramp_1/sim0077/lab_diags/hdf5",
    BASE_FOLDER + "internal/version_2/downramp_1/sim0139/lab_diags/hdf5",
]

# Set to None to load all particles
SELECTION: Optional[dict[str, list[Optional[float]]]] = {
    "uz": [None, None],
    "z": [None, None],
}

# Which dump to load from diagnostics folder. Set to -1 for final dump
ITERATION_NUMBER: int = -1
# Which species to load from diagnostics folder. Typically "electrons"
SPECIES: str = "n_elec"
# If True, skip generating phase space plots for each test case
SKIP_PHASE_PLOTS: bool = False


def main() -> None:
    """Compare two objective functions across multiple test cases.

    Evaluates both objective functions on each test case in LIST_OF_TEST_CASES,
    optionally generates phase space plots for each case, and creates a summary
    plot comparing the normalized objective function values.
    """
    index: int = 1
    obj_1_vals: np.ndarray = np.zeros(len(LIST_OF_TEST_CASES))
    obj_2_vals: np.ndarray = np.zeros(len(LIST_OF_TEST_CASES))

    for test_case in LIST_OF_TEST_CASES:
        diag_folder = Path(test_case)

        opa_dict_1: dict[str, float] = {}
        obj_func_1(
            simulation_directory=str(diag_folder.parent),
            output_params=opa_dict_1,
            file_tree="hdf5",
            do_storage_cleanup=False,
        )

        opa_dict_2: dict[str, float] = {}
        obj_func_2(
            simulation_directory=str(diag_folder.parent),
            output_params=opa_dict_2,
            file_tree="hdf5",
            do_storage_cleanup=False,
        )

        obj_summary: str = (
            f"Case {index}: OBJ 1 = {opa_dict_1['f']:.2e}, "
            f"OBJ 2 = {opa_dict_2['f']:.2e}"
        )
        obj_1_vals[index - 1] = opa_dict_1["f"]
        obj_2_vals[index - 1] = opa_dict_2["f"]
        index += 1

        if not SKIP_PHASE_PLOTS:
            try:
                x, y, z, ux, uy, uz, w, q, ts = an.load_beam_data(
                    diag_folder,
                    SPECIES,
                    iteration=ITERATION_NUMBER,
                    select=SELECTION,
                )
            except FileNotFoundError:
                print(f"Error: Could not find data in {diag_folder}")
                return
            except Exception as e:
                print(f"Error loading beam data: {e}")
                return

            # Analyze
            analysis = an.analyze_beam(x, y, z, ux, uy, uz, w, q, bins=200)
            an.print_beam_summary(analysis)

            # Plot
            an.plot_beam_analysis(
                x, y, z, ux, uy, uz, w, analysis, supertitle=obj_summary
            )

    # Make a summary plot
    plt.title("Summary")
    plt.xlabel("Case Number")
    plt.ylabel("Normalized Objective Function")

    max_obj_1_val = max(obj_1_vals) if len(obj_1_vals) > 0 else 0.0
    max_obj_1 = max_obj_1_val if max_obj_1_val > 0 else 1.0
    max_obj_2_val = max(obj_2_vals) if len(obj_2_vals) > 0 else 0.0
    max_obj_2 = max_obj_2_val if max_obj_2_val > 0 else 1.0

    plt.plot(np.array(obj_1_vals) / max_obj_1, label="OBJ 1")
    plt.plot(np.array(obj_2_vals) / max_obj_2, label="OBJ 2")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
