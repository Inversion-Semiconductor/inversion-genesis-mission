#!/usr/bin/env python3
"""
Part of the workflow in generating movies of the rho_electron FBPIC images.  This script reads .npy files
  produced by the "extract_rho_electron_slice" function of "utils/hdf5_func.py", and outputs a single png
  image for each npy file.  Then, this script calls a `ffmpeg` command to generate a mp4 movie from these
  png images.  Mp4 movie is created in the directory containing the png images.

Usage:
    First, run `data_parsing/extract_hdf5_field.py` on FBPIC output data containing "rho".
    If virtual environment installed using `pip install ./`, can use the entrypoint
      `slideshow-from-npy` to call the script instead of `python PATH/TO/slideshow_from_npy.py`.
    Then run this script with optional command-line arguments.

    Command-line usage:
        python slideshow_from_npy.py [OPTIONS]

    Optional Arguments:
        -d, -dir, --npy-dir PATH
            Path to the directory containing .npy files. If not provided, uses the default
            value defined in the script.

        --vmin FLOAT
            Lower limit on color scale for the PNG images.
            Default: -1e6

        --vmax FLOAT
            Upper limit on color scale for the PNG images. Should be "0" for no electrons.
            Default: 0.05e6

        --rmax FLOAT
            Maximum radial extent to display (in meters).
            Default: 100e-6

    Examples:
        # Use all defaults
        python slideshow_from_npy.py

        # Specify a different directory
        python slideshow_from_npy.py --npy-dir /path/to/npy/files

        # Adjust color scale limits
        python slideshow_from_npy.py --vmin -2e6 --vmax 0.1e6

    Notes:
        A RuntimeError will be raised if `ffmpeg` is not installed in virtual environment
        The following is the command that is run to generate the mp4 movie:
            ffmpeg -framerate 10 -pattern_type glob -i 'rho_*.png' -c:v libx264 -pix_fmt yuv420p output.mp4
"""
import argparse
from pathlib import Path
from scipy.constants import e

try:
    from inversion_fbpic.utils.plotting import plot_from_npy
    from inversion_fbpic.utils.make_movie import make_movie
except ImportError:
    from plotting import plot_from_npy  # type: ignore
    from make_movie import make_movie  # type: ignore

# Default configuration variables
DEFAULT_NPY_DIR: Path = Path("./")

RHO_TO_NUMBER_DENSITY: float = 1e-6 / e
DEFAULT_VMAX: float = 1e6 * RHO_TO_NUMBER_DENSITY
DEFAULT_VMIN: float = -0.05e6 * RHO_TO_NUMBER_DENSITY

DEFAULT_RMAX: float = 100e-6  # Maximum radial extent to display (m)
CHARGE_DENSITY_MOVIE_FRAMERATE: int = 10
FIELD_NAME: str = "rho"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate PNG images and a MP4 movie from a series of .npy files."
    )

    parser.add_argument(
        "-d",
        "-dir",
        "--npy-dir",
        type=Path,
        default=DEFAULT_NPY_DIR,
        help="Path to the directory containing .npy files",
    )

    parser.add_argument(
        "--vmin",
        type=float,
        default=DEFAULT_VMIN,
        help="Lower limit on color scale for the PNG images",
    )

    parser.add_argument(
        "--vmax",
        type=float,
        default=DEFAULT_VMAX,
        help="Upper limit on color scale for the PNG images",
    )

    parser.add_argument(
        "--rmax",
        type=float,
        default=DEFAULT_RMAX,
        help="Maximum radial extent to display (in meters)",
    )

    return parser.parse_args()


def process(args: argparse.Namespace) -> None:
    """
    Reads .npy files in a directory and produces PNG plots for each.  Generates a MP4
    movie from the PNG images and saves the movie to the same directory.

    Args:
        args: Parsed command-line arguments. See module docstring for argument details.
    """
    plot_from_npy(
        args.npy_dir,
        vmin=args.vmin,
        vmax=args.vmax,
        rmax=args.rmax,
        field_name=FIELD_NAME,
        save_path=Path(args.npy_dir),
    )

    movie_file = make_movie(
        images_dir=args.npy_dir,
        image_prefix=FIELD_NAME + "_",
        filename=FIELD_NAME + "_movie",
        framerate=CHARGE_DENSITY_MOVIE_FRAMERATE,
    )
    if movie_file is not None:
        print(f"Saving rho electron movie to {movie_file.name}")
    else:
        print("Failed to create rho electron movie")


def main() -> None:
    """Main entry point for script execution: parses command-line arguments and calls process()"""
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
