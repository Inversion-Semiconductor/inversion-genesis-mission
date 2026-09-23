"""
Script to analyze plasma images from HASO measurements and extract horizontal lineouts, including image rotation and
background subtraction.

Usage:
    Download HASO data from the Inversion Shared Google Drive "Data" (Data/HTU/HASO Measurements)
    Place the data in this directory and point the global DATA_PATH to the HASO data.  There has been work over the
      months on the HASO analyzer so the names aren't standardized in Feb and March of 2025.  Use "PlasmaPhaseAnalysis"
      for the February Data.  The March data is much lower signal and I wouldn't trust it as much.
    With DEBUG_PLOTS = True, adjust the BACKGROUND_LEVEL as needed.
    Run with any python interpreter.
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Any, Optional

DATA_PATH: str = (
    "./02192025-HASOData/PlasmaPhaseAnalysis"  # Directory where Haso data is saved
)
OUTPUT_PATH: str = "plasma_lineouts_data.npy"  # Name of output file for lineouts

BACKGROUND_LEVEL: int = 14  # Amount to subtract off of each pixel
DEBUG_PLOTS: bool = False  # Set to True to plot each image as it is processed


class PlasmaLineoutAnalyzer:
    """
    Class to analyze plasma images and extract horizontal lineouts.
    """

    def __init__(self, data_path: str) -> None:
        """
        Initialize the analyzer with the path to the image data.

        Args:
            data_path (str): Path to the directory containing the processed images.
        """
        self.data_path = Path(data_path)

    @staticmethod
    def background_subtract(image, background_level: int, show_plot: bool = False):
        val = background_level
        image[np.where(image < val)] = 0
        image[np.where(image >= val)] = image[np.where(image >= val)] - val
        if show_plot:
            plt.imshow(image)
            plt.show()
        return image

    @staticmethod
    def analyze_single_image_simple(image_path: str) -> Optional[dict[str, Any]]:
        """
        Simple analysis of a single image with linear fitting.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict[str, Any] | None: Analysis results or None if image not found or not enough plasma pixels.
        """
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None

        img = PlasmaLineoutAnalyzer.background_subtract(
            img, background_level=BACKGROUND_LEVEL, show_plot=DEBUG_PLOTS
        )

        print(f"Image shape: {img.shape}")
        print(f"Image dtype: {img.dtype}")
        print(f"Image min/max: {img.min()}/{img.max()}")

        # Create coordinate grids
        height, width = img.shape

        # Find non-zero pixels (plasma region)
        plasma_mask = img > img.max() * 0.1  # Threshold at 10% of max
        plasma_pixels = np.where(plasma_mask)

        if len(plasma_pixels[0]) == 0:
            print("No plasma pixels found!")
            return None

        # Get coordinates of plasma pixels
        y_plasma = plasma_pixels[0]
        x_plasma = plasma_pixels[1]

        print(f"Found {len(y_plasma)} plasma pixels")

        # Fit linear function to plasma pixels
        # We'll fit y = mx + b to find the slope
        if len(x_plasma) > 1:
            # Use polyfit to get slope and intercept
            coeffs = np.polyfit(x_plasma, y_plasma, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            angle = np.degrees(np.arctan(slope))

            print(f"Fitted slope: {slope:.4f}")
            print(f"Fitted intercept: {intercept:.2f}")
            print(f"Calculated angle: {angle:.2f} degrees")

            # Create fitted line for visualization
            x_line = np.linspace(0, width - 1, 100)
            y_line = slope * x_line + intercept

            # Rotate image to make plasma horizontal
            print(f"Detected angle: {angle:.2f} degrees")
            if abs(angle) > 0.5:  # Lower threshold to catch smaller angles
                print(f"Rotating image by {angle:.2f} degrees")
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
                print("Rotation applied successfully")
            else:
                print(f"Angle too small ({angle:.2f}°), no rotation applied")
                rotated_img = img.copy()
                angle = 0.0

            # Find optimal lineout position in rotated image
            row_means = np.mean(rotated_img, axis=1)
            optimal_y = np.argmax(row_means)

            # Extract lineout
            lineout = rotated_img[optimal_y, :]

            return {
                "original_image": img,
                "rotated_image": rotated_img,
                "plasma_pixels": (x_plasma, y_plasma),
                "fitted_line": (x_line, y_line),
                "slope": slope,
                "intercept": intercept,
                "angle": angle,
                "optimal_y": optimal_y,
                "lineout": lineout,
                "filename": Path(image_path).name,
            }
        else:
            print("Not enough plasma pixels for fitting")
            return None

    @staticmethod
    def plot_single_image_analysis(result: Optional[dict[str, Any]]) -> None:
        """
        Plot the analysis results for a single image.

        Args:
            result (dict[str, Any] | None): Analysis results from analyze_single_image_simple.

        Returns:
            None
        """
        if result is None:
            print("No results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(9, 7))

        # Original image with plasma pixels and fitted line
        axes[0, 0].imshow(result["original_image"], cmap="viridis", vmax=20, vmin=0)
        axes[0, 0].scatter(
            result["plasma_pixels"][0],
            result["plasma_pixels"][1],
            c="red",
            s=1,
            alpha=0.5,
            label="Plasma pixels",
        )
        axes[0, 0].plot(
            result["fitted_line"][0],
            result["fitted_line"][1],
            "b-",
            linewidth=2,
            label=f'Fitted line (slope={result["slope"]:.3f})',
        )
        axes[0, 0].set_title(f"Original Image: {result['filename']}")
        axes[0, 0].legend()
        axes[0, 0].axis("off")

        # Rotated image with lineout position
        axes[0, 1].imshow(result["rotated_image"], cmap="viridis")
        axes[0, 1].axhline(
            y=result["optimal_y"],
            color="red",
            linestyle="--",
            label=f'Lineout at y={result["optimal_y"]}',
        )
        axes[0, 1].set_title(f"Rotated Image (angle={result['angle']:.1f}°)")
        axes[0, 1].legend()
        axes[0, 1].axis("off")

        # Lineout plot (mirrored and scaled to mm)
        width = len(result["lineout"])
        mirrored_lineout = result["lineout"][::-1]
        x_mm = np.linspace(0, 5, width)
        axes[1, 0].plot(x_mm, mirrored_lineout, "b-", linewidth=1)
        axes[1, 0].set_xlabel("X Position (mm)")
        axes[1, 0].set_ylabel("Intensity")
        axes[1, 0].set_title("Horizontal Lineout")
        axes[1, 0].set_xlim(0, 5)
        axes[1, 0].grid(True, alpha=0.3)

        # Histogram of plasma pixel distribution
        axes[1, 1].hist(result["plasma_pixels"][1], bins=50, alpha=0.7, color="green")
        axes[1, 1].set_xlabel("X Position (pixels)")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Plasma Pixel Distribution")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def analyze_single_image_test() -> None:
    """
    Test function to analyze a single image.

    Returns:
        None
    """
    data_path = (
        "./02192025-HASOData/PlasmaPhaseAnalysis"
    )
    analyzer = PlasmaLineoutAnalyzer(data_path)
    image_files = glob.glob(str(Path(data_path) / "*_processed.png"))
    if not image_files:
        print("No images found with pattern '*_processed.png'")
        return
    test_image_path = image_files[0]
    print(f"Testing with image: {Path(test_image_path).name}")
    result = analyzer.analyze_single_image_simple(test_image_path)
    if result:
        analyzer.plot_single_image_analysis(result)
        print("\nAnalysis Summary:")
        print(f"Filename: {result['filename']}")
        print(f"Detected angle: {result['angle']:.2f} degrees")
        print(f"Optimal lineout Y position: {result['optimal_y']}")
        print(f"Lineout length: {len(result['lineout'])} pixels")
        print(
            f"Lineout intensity range: {result['lineout'].min():.1f} - {result['lineout'].max():.1f}"
        )
    else:
        print("Analysis failed")


def analyze_all_images_and_plot_lineouts(data_path=DATA_PATH) -> None:
    """
    Analyze all images and plot their lineouts together.

    Returns:
        None
    """
    analyzer = PlasmaLineoutAnalyzer(data_path)
    image_files = glob.glob(str(Path(data_path) / "*_processed.png"))
    if not image_files:
        print("No images found with pattern '*_processed.png'")
        return
    print(f"Found {len(image_files)} images to analyze")
    all_results = []
    for i, image_path in enumerate(sorted(image_files)):
        print(f"Analyzing image {i + 1}/{len(image_files)}: {Path(image_path).name}")
        result = analyzer.analyze_single_image_simple(image_path)
        if result:
            all_results.append(result)
        else:
            print(f"Failed to analyze {Path(image_path).name}")
    if not all_results:
        print("No images were successfully analyzed")
        return
    print(f"Successfully analyzed {len(all_results)} images")
    plot_all_lineouts(all_results)


def plot_all_lineouts(results: list[dict[str, Any]]) -> None:
    """
    Plot lineouts from all images in a single figure.

    Args:
        results (list[dict[str, Any]]): List of analysis results from analyze_single_image_simple.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for i, result in enumerate(results):
        width = len(result["lineout"])
        mirrored_lineout = result["lineout"][::-1]
        x_mm = np.linspace(0, 10.1e-3 * width, width)
        plt.plot(
            x_mm,
            mirrored_lineout,
            color=colors[i],
            linewidth=1.5,
            alpha=0.8,
            label=result["filename"],
        )
    plt.xlabel("X Position (mm)")
    plt.ylabel("Intensity")
    plt.title("Horizontal Lineouts from All Plasma Images")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xlim(min(x_mm), max(x_mm))
    plt.tight_layout()
    plt.show()
    print("\nLineout Analysis Summary:")
    for result in results:
        print(
            f"{result['filename']}: angle={result['angle']:.2f}°, "
            f"max_intensity={result['lineout'].max():.1f}, "
            f"mean_intensity={result['lineout'].mean():.1f}"
        )
    export_lineouts_to_numpy(results, output_path=OUTPUT_PATH)


def export_lineouts_to_numpy(
    results: list[dict[str, Any]], output_path: str
) -> Optional[str]:
    """
    Export all lineouts to a single numpy file.

    Args:
        results (list[dict[str, Any]]): List of analysis results from analyze_single_image_simple.
        output_path (str, optional): Path to save the numpy file (default: current directory).

    Returns:
        str | None: Path to the saved numpy file, or None if no results to export.
    """
    if not results:
        print("No results to export")
        return None
    min_length = min(len(result["lineout"]) for result in results)
    print(f"Truncating all lineouts to {min_length} points for consistency")
    lineouts_array = np.zeros((len(results), min_length))
    filenames = []
    for i, result in enumerate(results):
        mirrored_lineout = result["lineout"][::-1]
        lineouts_array[i, :] = mirrored_lineout[:min_length]
        filenames.append(result["filename"])

    np.save(output_path, lineouts_array)
    print(f"Saved {len(results)} lineouts to: {output_path}")
    print(f"Array shape: {lineouts_array.shape}")
    return output_path


if __name__ == "__main__":
    # analyze_single_image_test()

    analyze_all_images_and_plot_lineouts()
