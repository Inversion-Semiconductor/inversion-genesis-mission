"""
Short script that plots simulation performance vs number of GPU's used on AWS.  Simulation performed was of a modest
size using modest GPUs, so results may change depending on how beefy the simulation and/or GPUs are.

Usage:
    python benchmarking_visualization.py
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_gpu_efficiency() -> None:
    """
    Plot GPU efficiency for different grid sizes and hardware setups.

    Returns:
        None
    """
    # Datapoints for running on AWS's g4dn.metal GPU node
    metal_2048_time: np.ndarray = np.array([1.09, 0.61, 0.45, 0.34, 0.30])
    metal_4096_time: np.ndarray = np.array([2.18, 1.38, 0.71, 0.52, 0.41])
    metal_gpu_num: np.ndarray = np.array([1, 2, 4, 6, 8])

    # Datapoint for running on my Macbook Air
    # macbook_time = np.array([2.51])
    # macbook_gpu_num = np.array([0])

    metal_2048_eff: np.ndarray = metal_2048_time[0] / (metal_2048_time * metal_gpu_num)
    metal_4096_eff: np.ndarray = metal_4096_time[0] / (metal_4096_time * metal_gpu_num)

    plt.plot(metal_gpu_num, metal_2048_eff, label="Grid Size 2048x512")
    plt.plot(metal_gpu_num, metal_4096_eff, label="Grid Size 4096x512")
    plt.xlabel("# of GPUs")
    plt.ylabel("GPU efficiency [time for 1 GPU/(N x time for N GPUs)]")
    plt.title("GPU efficiency with FBPIC on AWS's g4dn.metal")
    plt.legend()
    plt.show()


def main() -> None:
    """
    Script entry point for plotting GPU efficiency.

    Returns:
        None
    """
    plot_gpu_efficiency()


if __name__ == "__main__":
    main()
