#!/usr/bin/env python3
"""Extract and save high energy electrons from a base hdf5 to a trimmed-down hdf5 file

See below for examples using the command line options
    --set           : Specify the given path is instead a directory containing multiple diags folders
    --suffix        : For a set, specifies which folders to consider.  If None, will try all folders
    --species       : Specify the field name to extract from the HDF5 file.
    --min-uz        : Minimum MeV to consider for extraction of particles
    --last          : Only extract the final time dump (last ``*.h5`` in sorted order under hdf5/)

1. Command line - single simulation directory:
   python extract_hdf5_particles.py <diags_folder> --species "electrons" --min-uz 10

2. Command line - multiple simulations in a set folder:
   python extract_hdf5_particles.py --set <set_folder> --species "electrons" --min-uz 10

3. Python interpreter - single simulation (using global variables):
   Set DIAGS_FOLDER, SPECIES, and MIN_UZ variables, then run main()

4. Python interpreter - multiple simulations (using global variables):
   Set SET_FOLDER, SPECIES, and MIN_UZ variables, then run main()

Trimmed hdf5 files are saved to "analysis/<diags_folder_name>/<species>/"
"""
import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Import hdf5_funcs either through virtual environment or local copy
try:
    from inversion_fbpic.utils.hdf5_funcs import ebeam_extract_particles_only
except ImportError:
    from hdf5_funcs import ebeam_extract_particles_only  # type: ignore


# Global variables for use in Python interpreter
SPECIES: str = "n_elec"
SET_SUFFIX: Optional[str] = "*_lab_diags"  # Note: `*` has to be at the beginning
MIN_UZ: Optional[float] = 10
SET_FOLDER: Optional[str] = None
DIAGS_FOLDER: Optional[str] = None


def process_single_simulation(
    diags_folder: Union[str, Path],
    species: str = SPECIES,
    min_uz: Optional[float] = MIN_UZ,
    parent_level: int = 0,
    set_name: str = "",
    last_dump_only: bool = False,
) -> Path | None:
    """Process a single simulation directory.

    Args:
        diags_folder: Path to the diagnostics folder.
        species: Name of the particle species to extract.
        min_uz: Minimum MeV to consider for extraction of particles.
        parent_level: Level of the parent to extract analysis folder to (0 for single, 1 for set)
        set_name: If part of a set, the analysis folder will contain this subdirectory
        last_dump_only: If True, only process the final ``*.h5`` dump (sorted filenames).

    Returns:
        Path to the output directory, or None if the operation failed.
    """
    diags_path = Path(diags_folder)
    if not (diags_path.exists() and diags_path.is_dir()):
        logger.warning("Path `%s` is not a directory, skipping.", diags_path)
        return None

    source = diags_path / "hdf5"
    if not source.is_dir():
        logger.warning("Path `%s` is not a valid hdf5 directory, skipping.", source)
        return None

    destination = (
        diags_path.parents[parent_level]
        / Path("analysis")
        / set_name
        / diags_path.name
        / species
    )
    destination.mkdir(parents=True, exist_ok=True)

    print(f"Processing single simulation: {diags_path}")
    ebeam_extract_particles_only(
        source=source,
        destination=destination,
        species_name=species,
        min_uz=min_uz,
        last_dump_only=last_dump_only,
    )
    return destination


def process_simulation_set(
    set_folder: Union[str, Path],
    species: str = SPECIES,
    min_uz: Optional[float] = MIN_UZ,
    set_suffix: Optional[str] = SET_SUFFIX,
    last_dump_only: bool = False,
) -> list[Path]:
    """Process multiple simulations in a set folder.

    Args:
        set_folder: Path to the set folder containing multiple simulation directories.
        species: Name of the particle species to extract
        min_uz: Minimum MeV to consider for extraction of particles.
        set_suffix: The sub-directory file pattern that will contain the hdf5 directory.
            If None, will check all folders
        last_dump_only: If True, only the final ``*.h5`` dump is extracted per simulation.

    Returns:
        List of paths to the output directories.
    """
    set_path = Path(set_folder)
    if not (set_path.exists() and set_path.is_dir()):
        logger.error("Path `%s` is not a directory.", set_path)
        sys.exit(1)
    if set_suffix is not None and set_suffix[0] != "*":
        logger.error("'set_suffix' `%s` must start with '*'.", set_suffix)
        sys.exit(1)

    output_dirs: list[Path] = []
    print(f"Processing simulation set: {set_folder}")

    for subfolder in set_path.iterdir():
        if not subfolder.is_dir():
            continue
        if set_suffix is not None and not re.search(
            rf'{re.escape(set_suffix.replace("*", ""))}$', subfolder.name
        ):
            continue
        result = process_single_simulation(
            diags_folder=subfolder,
            species=species,
            min_uz=min_uz,
            parent_level=1,
            set_name=set_path.name,
            last_dump_only=last_dump_only,
        )
        if result is not None:
            output_dirs.append(result)

    return output_dirs


def main() -> None:
    """Main function that handles system arguments and/or global variables"""
    # Check if running from command line
    if len(sys.argv) > 1:
        # Command line usage
        parser = argparse.ArgumentParser(
            description="Extract high energy particles from FBPIC HDF5 files",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
            Examples:
              # Single simulation
              python extract_hdf5_particles.py /test_simulation/diags
              
              # Multiple simulations
              python extract_hdf5_particles.py --set /test_simulation_set/
            """,
        )

        # Create mutually exclusive group for single vs set processing
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "diags_folder", nargs="?", help="Path to single simulation diags folder"
        )
        group.add_argument(
            "--set",
            dest="set_folder",
            help="Path to folder containing multiple simulation diags folders",
        )

        parser.add_argument(
            "--suffix",
            dest="set_suffix",
            default=SET_SUFFIX,
            help="Naming convention of diagnostic folders within set folder",
        )

        parser.add_argument(
            "--species",
            default=SPECIES,
            help=f"Name of the charge density field to extract (default: {SPECIES})",
        )

        parser.add_argument(
            "--min-uz",
            type=float,
            default=MIN_UZ,
            help=f"Minimum particle energy (MeV) to consider (default: {MIN_UZ})",
        )

        parser.add_argument(
            "--last",
            action="store_true",
            help="Only extract the final time dump (last *.h5 in sorted order under hdf5/)",
        )

        args = parser.parse_args()

        if args.set_folder:
            # Command line - multiple simulations
            process_simulation_set(
                args.set_folder,
                args.species,
                args.min_uz,
                args.set_suffix,
                last_dump_only=args.last,
            )
        else:
            # Command line - single simulation
            if not args.diags_folder:
                parser.error("diags_folder is required when not using --set")
            process_single_simulation(
                args.diags_folder,
                args.species,
                args.min_uz,
                last_dump_only=args.last,
            )

    else:
        # Python interpreter usage - check global variables
        if SET_FOLDER is not None:
            print(f"Using global SET_FOLDER: {SET_FOLDER}")
            print(f"Using global SPECIES: {SPECIES}")
            print(f"Using global MIN_UZ: {MIN_UZ}")
            process_simulation_set(SET_FOLDER, SPECIES, MIN_UZ)
        if DIAGS_FOLDER is not None:
            print(f"Using global DIAGS_FOLDER: {DIAGS_FOLDER}")
            print(f"Using global SPECIES: {SPECIES}")
            print(f"Using global MIN_UZ: {MIN_UZ}")
            process_single_simulation(DIAGS_FOLDER, SPECIES, MIN_UZ)
        if SET_FOLDER is None and DIAGS_FOLDER is None:
            print("No arguments provided and no global variables set.")
            print("Please either:")
            print("1. Provide command line arguments")
            print(
                "2. Set DIAGS_FOLDER (and optionally SPECIES and MIN_UZ) for single simulation"
            )
            print(
                "3. Set SET_FOLDER (and optionally SPECIES and MIN_UZ) for multiple simulations"
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
