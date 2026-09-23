# Inversion FBPIC

A Python package for plasma simulations using FBPIC, with a focus on density profile generation and analysis.

## Prerequisites

- Python 3.9 or higher
- Anaconda or Miniconda
- For Apple Silicon (M1/M2) Macs: Additional setup required (see Installation section)

## Simulation Location

All the FBPIC run scripts, including for standalone simulations, parameter scans, and Optimas optimizations, are found in a separate repository:
[Inversion-FBPIC-RunScripts](https://github.com/Inversion-Semiconductor/inversion-fbpic-runscripts)

Using scripts in that separate repository still require the python environment for this sub-repository.

## Installation

### Basic Installation

1. Create and activate a new conda environment:
   ```bash
   conda create --name FBPIC
   conda activate FBPIC
   ```

2. Install FBPIC following the [official documentation](https://fbpic.github.io/install/install_local.html)

   Note: For Apple Silicon (M1/M2) Macs, the `mkl` module is not available. Instead, install fftw:
   ```bash
   conda install -c conda-forge pyfftw
   ```

3. Install this package in editable mode:
   ```bash
   pip install -e .
   ```

### Development Setup

To set up the development environment with testing capabilities:

1. Activate your conda environment:
   ```bash
   conda activate FBPIC
   ```

2. Install pytest and setuptools in your conda environment:
   ```bash
   conda install pytest setuptools
   ```

3. Install the package in development mode with testing dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Run the tests:
   ```bash
   pytest tests -v
   ```

The `-v` flag provides verbose output showing each test case.

## Usage

### Density Profiles

The package provides several density profile functions:

- `build_gaussian_profile`: Creates a Gaussian density profile
- `build_gaussian_plus_triangle_z_density_function`: Creates a combined Gaussian and triangular density profile

Example usage:
```python
from inversion_fbpic.density_profiles.downramp_injection import build_gaussian_profile

# Create a Gaussian profile
density = build_gaussian_profile(sigma=3e-6, center_location=10e-6)
```

### Running FBPIC

FBPIC can be run using python once in the properly set up environment.  For example simulations to test running the code, 
see the examples in the directory `simulations/test_simulations/`


### Visualization with openPMD

For visualizing simulation outputs, use the openPMD notebook tool:

1. Install following the instructions at the bottom of the [FBPIC How to Run page](https://fbpic.github.io/how_to_run.html#visualizing-the-simulation-results)

2. Launch the visualization tool:
   ```bash
   openPMD_notebook
   ```

3. If the default diagnostic folder name was changed in the FBPIC launch script, edit the following cell in the generated notebook accordingly:
   ```python
   # Replace the string below, to point to your data
   ts = OpenPMDTimeSeries('./diags/hdf5/')
   ```
   *Note*: The root directory here will be the directory where the openPMD_notebook was launched.


### Visualization with CLI

Within this project are several methods to view the data in a more customizable way.  These scripts are located in the
`scripts/` directory and utilize analysis and hdf5 manipulation tools within the `utils/` folder.

1. (Optional) if your particle data contains more electrons than just the electron beam you are interested in (as is the
case for the downramp injected ebeams in this project), then you will want to use the `ebeam_extract_particles_only()`
function within `hdf5_funcs.py` to remove electrons below a given energy value.  See `utils/extract_ebeam_from_hdf5.py` or
`utils/extract_ebeam_from_set.py` for the command line instructions for this.

2. Call the analysis and visualization functions within the `scripts` directory.  After installing the package with `pip install -e .`, many scripts are available as command-line tools that can be called from anywhere. These scripts support flexible command-line arguments with both long and short flag options.

   **Command-Line Scripts:**
   
   After installation, scripts can be called directly by name. Common shorthand flags include:
   - `-d, --diag-folder`: Path to diagnostics directory
   - `-i, --iteration-number`: Iteration number to load
   - `-s, --species`: Particle species name
   - `-b, --bins`: Number of bins for calculations
   
   **Example usage:**
   ```bash
   plot-ebeam-analysis -d sim0077/lab_diags/hdf5 -s n_elec -i -1
   ```
   
   This command analyzes the electron beam from the specified diagnostics folder, using species `n_elec` and loading the final iteration (`-1`).
   
   **Available scripts include:**
   - `plot-ebeam-analysis`: Outputs beam analysis for a single simulation
   - `visualize-ebeamparams-vs-scan`: Analyzes beam parameter measurements in a 1D parameter scan
   - `energy-at-peak-current`: Calculates energy at peak current location
   - `simple-r56`: Applies R56 transformation and analyzes compression chicane performance
   - `slideshow-from-npy`: Creates PNG images from .npy files for charge density movies
   - `view-output`: Objective evolution and parameter scatter plots for Optimas runs
   - `gaussian-process-evaluation`: Gaussian-process post-analysis of Optimas exploration data
   - `analyze-laser-evolution`: Laser energy and `a0` tracking from OpenPMD field diagnostics
   - And many more (see `scripts/` directory for full list)
   
   For detailed usage information and all available options for each script, see the documentation at the top of each script file or use the `--help` flag:
   ```bash
   plot-ebeam-analysis --help
   ```

   #### Optimas Gaussian process analysis (`gaussian-process-evaluation`)

   Post-processes completed Optimas exploration runs using ``ExplorationDiagnostics``
   GP models. Use ``view-output`` for trial history and parameter correlations; use
   this script for GP contours, model-vs-data checks, and 1D slice plots.

   Point ``-d`` at the optimization folder or at the ``exploration/`` subdirectory.

   ```bash
   # List varying (contour) and analyzed (slice) parameter indices
   gaussian-process-evaluation -d /path/to/optimas/run -p

   # 2D GP contour over two varying parameters
   gaussian-process-evaluation -d /path/to/optimas/run -c -cx 0 -cy 1

   # Compare GP predictions to observations
   gaussian-process-evaluation -d /path/to/optimas/run -e

   # 1D slices: objective on the left, analyzed parameter 0 on the right (maximize)
   gaussian-process-evaluation -d /path/to/optimas/run -s -s2 0 -s2m
   ```

   Common flags: ``-d`` data path, ``-p`` print parameter indices, ``-c`` contour,
   ``-cx`` / ``-cy`` contour axes, ``-e`` evaluate model, ``-s`` 1D curves,
   ``-s1`` / ``-s2`` analyzed-parameter indices, ``-s1m`` / ``-s2m`` maximize
   (rather than minimize) when building slice models.

   #### Laser evolution analysis (`analyze-laser-evolution`)

   Tracks laser energy and engineering ``a0`` along the propagation axis by reading
   **E** and **B** fields from an OpenPMD HDF5 time series (laser-centred bandpass,
   cylindrical energy integration, analytic-signal envelope). Writes a JSON results
   file and a summary ``energy`` / ``a0`` vs. ``z`` plot.

   ```bash
   # Default: ./diags/hdf5, saves laser_evolution_results.json and laser_evolution_plot.png
   analyze-laser-evolution

   # Custom diagnostics directory and output paths
   analyze-laser-evolution -d /path/to/sim/lab_diags/hdf5 \
       --results-file my_results.json --output-plot my_plot.png

   # Per-iteration imshow panels (optional log colour scale)
   analyze-laser-evolution -d /path/to/sim/lab_diags/hdf5 --show-plots --log

   # Re-plot summary from a saved JSON without re-running the analysis
   analyze-laser-evolution -j my_results.json --output-plot my_plot.png
   ```

   Useful options: ``-d`` series path, ``-w`` laser wavelength (default 0.8 µm),
   ``--band-half-width-frac``, ``--rmax-window``, ``--on-axis-rmax``,
   ``--energy-threshold``, ``--start-iteration`` / ``--end-iteration``,
   ``--show-plots``, ``--log``, ``-j`` / ``--from-json``.

### Visualization with Python

The modules `inversion_fbpic.utils.plotting` and `inversion_fbpic.utils.make_movie` let you render every timestep to PNG stills and stitch them into an MP4. `plot_from_hdf5_series` reads OpenPMD diagnostics directly. Pre-extracted `.npy` field arrays (e.g. from `extract-hdf5-field`) can also be plotted with `plot_from_npy`, which accepts the same color-scale and layout options as the HDF5 workflow below, though with limited quantities available for plotting. `make_movie` calls the system `ffmpeg`, which must be on your `PATH`; importing from `make_movie` raises an `RuntimeError` if `ffmpeg` is not installed.

Key options on `plot_from_hdf5_series`:
- `field_name` / `component`: scalar fields (e.g. `"rho"`), vector components (`"E"` + `"z"`), or `"eme"` for electromagnetic energy density. For `"eme"`, set `field_name=None`; the plotter loads all **E** and **B** components and applies a cosine-squared bandpass (`em_bandpass_filter`) to produce six sub-panels per frame. `rho` is automatically scaled to -e/cm^3 and `eme` has units of J/um^3.
- `vminmax`: fixed `(vmin, vmax)`, `"even"` for a symmetric color scale around zero, or `None` to auto-scale each frame.
- `scale` / `cmap`: `"linear"` or `"log"` color norms (a log-linear distribution is used for signed data in the case of `log`); any Matplotlib colormap name.
- `rmax`: optional radial crop in meters (display limits only).

Example: build movies for charge density, one **E** component, and band-filtered EM energy:

```python
from pathlib import Path

from inversion_fbpic.utils.make_movie import make_movie
from inversion_fbpic.utils.plotting import plot_from_hdf5_series

DIAGS_DIR = Path("lab_diags")   # parent of the hdf5/ folder (or the hdf5 folder itself)
PLOTS_DIR = Path("plots")
RMAX = 50e-6                    # display |r| <= RMAX [m]; use None for full grid

# (field_name, component, extra kwargs for plot_from_hdf5_series)
specs = [
      ("rho",  None,    dict(vminmax=(1e14, 2e18),    cmap="magma",  scale="log")),
      ("E",    "z",     dict(vminmax="even",          cmap="bwr",    scale="linear")),
      (None,   "eme",   dict(vminmax=(1e-15, 1e-5),   cmap="magma",  scale="log")),
]

for field_name, component, kwargs in specs:
      stills_dir, prefix = plot_from_hdf5_series(
         series_path=DIAGS_DIR / "hdf5",
         save_path=PLOTS_DIR / "stills",
         field_name=field_name,
         component=component,
         rmax=RMAX,
         font_size=16,
         **kwargs,
      )
      movie_path = make_movie(
         images_dir=stills_dir,
         image_prefix=prefix,
         filename=prefix,
         framerate=10,
      )
      print(f"Wrote {movie_path}")
```

To render additional vector components (e.g. all **E**/**B** axes), add more `(field_name, component, kwargs)` tuples—`plot_from_hdf5_series` returns `(save_dir, file_prefix)` so `make_movie` can glob `file_prefix_*.png` in that directory. After the movie is created, you can delete the intermediate PNGs if you only need the MP4.

### Troubleshooting

If you are having issues with getting openPMD to successfully read data, can try to enforce that openpmd is of the following version:
```shell
conda install -c conda-forge "openpmd-api=0.16.1" "openpmd-viewer=1.11.0”
```


# Setup on AWS

Ideally, the AWS deployer app will create the docker image with the FBPIC python environment, but if the
AWS instance was deployed from the website then this will need to be configured manually.  If that is the case,
here are the steps.

1. Setup ssh key
   1. Copy a `.pem` ssh key to the directory for ssh keys.  Typically on a Mac this would be in `~/.shh/`
   2. Update permissions for the key:  `chmod 400 my-key.pem`
   3. In the AWS deployer app, make sure this key is selected when trying to ssh to the particular instance
2. Configure docker image
   1. Create the docker image: `docker run -d --name DOCKER-IMAGE-NAME --gpus all nvidia/cuda:12.8.0-runtime-ubuntu24.04 tail -f /dev/null`
      2. NOTE: Could also try the newer CUDA 13.0, although it might still be buggy `nvidia/cuda:13.0.0-devel-ubuntu24.04`
   3. Enter the docker image: `docker exec -it DOCKER-IMAGE-NAME bash`
   3. Make `app` directory: `mkdir app`
   4. `cd app`
   5. Use the AWS deployer app to copy the environment setup script to the app folder.  This script is located in 
`inversion-lab/AWS/FBPIC_test/setup_fbpic_cuda12-8_mpich.sh`
      1. NOTE: Currently `mpich` is more reliable than `openmpi`, but environment setup scripts are provided for both
      2. NOTE: If using CUDA 13.0, use the respective `13-0` setup scripts
   6. Run script: `source setup_fbpic_cuda12-8_mpich.sh`
3. Should be all good to go!
   1. Run commands with a `&` at the end to run process in the background.  This ensures that the simulation continues even if the ssh terminal is disconnected
   2. If using `mpich`, then use `mpiexec`.  Likewise, for `openmpi` use `mpirun`
   2. Use `watch nvidia-smi` to check current GPU performance
   2. Use `df -h` to view current storage statistics
