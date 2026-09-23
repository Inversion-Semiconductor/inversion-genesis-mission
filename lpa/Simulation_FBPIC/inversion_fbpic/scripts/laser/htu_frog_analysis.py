"""
Script to load HTU FROG data from the specified folder, initialize a Longitudinal profile
using LASY, and calculate the FWHM for each shot in the folder.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from inversion_fbpic.utils.laser import HTULasyLaser

# Parameters to point to folder of saved data, and the particular scan number and date
SUPER_PATH = Path("/Users/cedoss/Desktop/data/HTU_FROG_Data/")
SCAN_NUMBER = 9
YEAR = 2023
MONTH = 7
DAY = 27

# If True, will plot the intensity and phase of the particular index within the scan folder
PLOT_TEST = True
TEST_INDEX = 50

# If True, will bin the data and show trends against the scanned parameter
IS_SCAN = False
BIN_SIZE = 20


if IS_SCAN:
    path_parent = SUPER_PATH / f"{MONTH:02d}-{DAY:02d}-{YEAR:04d}-Scan{SCAN_NUMBER:02d}-30s"
else:
    path_parent = SUPER_PATH / f"{MONTH:02d}-{DAY:02d}-{YEAR:04d}-Scan{SCAN_NUMBER:02d}"

if PLOT_TEST:
    path_test = path_parent / f"Scan{SCAN_NUMBER:03d}_U_FROG_Grenouille_{TEST_INDEX:03d}.txt"
    htu_laser = HTULasyLaser(path_test)
    print(htu_laser.calculate_fwhm_intensity(), "fs FWHM intensity")
    htu_laser.plot_temporal_profile(title=f"{MONTH:02d}-{DAY:02d}-{YEAR:04d}: {path_test.name}")

fwhm_values = []
index_list = []

num_files = len(list(path_parent.glob("Scan*.txt")))
for i in range(num_files):
    path_i = path_parent / f"Scan{SCAN_NUMBER:03d}_U_FROG_Grenouille_{i + 1:03d}.txt"
    print(path_i.name, path_i.exists())
    if path_i.exists():
        htu_laser = HTULasyLaser(path_i)
        fwhm_values.append(htu_laser.calculate_fwhm_intensity())
        index_list.append(i)

if IS_SCAN:
    if fwhm_values:
        # Bin the FWHM values based on the actual shot indices we observed.
        max_index = max(index_list)
        num_bins = max_index // BIN_SIZE + 1

        binned_fwhm_values = [[] for _ in range(num_bins)]
        for shot_idx, fwhm in zip(index_list, fwhm_values):
            bin_idx = shot_idx // BIN_SIZE
            binned_fwhm_values[bin_idx].append(fwhm)

        # Summarize each bin so we can quickly inspect the distribution.
        bin_indices = []
        bin_means = []
        bin_stds = []
        for bin_idx, bin_values in enumerate(binned_fwhm_values):
            if not bin_values:
                continue

            start_shot = bin_idx * BIN_SIZE
            end_shot = start_shot + BIN_SIZE - 1
            bin_array = np.array(bin_values)

            mean_fwhm = bin_array.mean()
            std_fwhm = bin_array.std()

            bin_indices.append(bin_idx)
            bin_means.append(mean_fwhm)
            bin_stds.append(std_fwhm)

            print(
                f"Shots {start_shot}-{end_shot}: "
                f"{len(bin_values)} shots, "
                f"{mean_fwhm:.2f} ± {std_fwhm:.2f} fs"
            )

        if bin_indices:
            plt.errorbar(
                bin_indices,
                bin_means,
                yerr=bin_stds,
                fmt="o-",
                capsize=4,
                label="Mean ± std per bin",
            )
            plt.xlabel(f"Bin index (size={BIN_SIZE})")
            plt.ylabel("FWHM (fs)")
            plt.title(f"{path_parent.name}: FWHM vs bin")
            plt.legend()
            plt.show()

else:
    fwhm_values = np.array(fwhm_values)
    mean_fwhm = fwhm_values.mean()
    std_fwhm = fwhm_values.std()

    print("FWHM Statistics:")
    statistics = f"{mean_fwhm:.2f} +/- {std_fwhm:.2f} fs"
    print(statistics)
    plt.plot(fwhm_values, label=statistics)
    plt.title(f"{path_parent.name}: FWHM intensity values")
    plt.xlabel("shot number")
    plt.ylabel("FWHM (fs)")
    plt.legend()
    plt.show()
