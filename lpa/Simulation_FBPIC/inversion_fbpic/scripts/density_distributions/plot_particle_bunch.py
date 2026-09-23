"""
Creates a beam and plots the current profile to verify it's triangular.

"""

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Test parameters
    n_macroparticles: int = 10000
    sig_z: float = 1e-3  # 1 mm beam length
    zf: float = 0.0

    # Generate test distribution
    z_uniform: np.ndarray = np.random.uniform(-sig_z, sig_z, n_macroparticles)
    z: np.ndarray = zf + z_uniform

    # Calculate weights for triangular current distribution
    z_min: float = zf - sig_z
    z_max: float = zf + sig_z
    z_normalized: np.ndarray = (z - z_min) / (z_max - z_min)
    current_weights: np.ndarray = z_normalized

    # Normalize weights
    weight_normalization: float = np.sum(current_weights) / n_macroparticles
    w: np.ndarray = current_weights / weight_normalization

    # Create histogram to visualize current distribution
    z_bins: np.ndarray = np.linspace(z_min, z_max, 50)
    hist, bin_edges = np.histogram(z, bins=z_bins, weights=w)
    bin_centers: np.ndarray = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(z, bins=50, alpha=0.7, label="Particle distribution")
    plt.xlabel("z (m)")
    plt.ylabel("Number of particles")
    plt.title("Particle Distribution in z")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(bin_centers, hist, "o-", label="Current profile")
    plt.xlabel("z (m)")
    plt.ylabel("Current (arbitrary units)")
    plt.title("Triangular Current Distribution")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Total charge: {np.sum(w):.2f}")
    print(f"Expected total charge: {n_macroparticles:.2f}")
    print(
        f"Charge conservation error: "
        f"{abs(np.sum(w) - n_macroparticles) / n_macroparticles * 100:.2f}%"
    )
