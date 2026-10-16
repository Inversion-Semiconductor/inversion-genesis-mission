# FBPIC-Warp

FBPIC-Warp is a fork of [FBPIC](https://github.com/fbpic/fbpic).
It ports the FBPIC GPU kernels to NVIDIA Warp 1.18.0 with its CUDA 12 build.
The upstream reference is FBPIC commit `dc1cc32ca36ab1ee813d82ea5e9f6b180be0e2a9`.
The CPU path and the physics models are unchanged.
The license is the upstream BSD-3-Clause-LBNL license ([LICENSE.txt](LICENSE.txt), [LEGAL.txt](LEGAL.txt)).

Milestone 1 covers forward execution on one GPU. Its implementation and validation are complete.
Gradient implementation and multi-GPU validation remain deferred.

Read the [baseline changes](docs/baseline-changes/README.md) for scope and current status.

The [solver porting guide](docs/warp-porting-guide/README.md) records the forward port, performance findings, and future autodiff and selective mixed-precision work.

The [validation report](docs/baseline-changes/milestone-1-validation.md) records test results, environment details, and deferred work.

Performance scripts, benchmark plans, and measurement reports belong to the persistent `snidhan/dev-profiling` branch.
That branch inherits solver changes from `main`. Keep performance tooling changes on that branch.
The porting guide retains the measured findings and their validation limits.

## Development environment

The environment pins NVIDIA's CUDA 12 wheel for Linux x86-64 to match the CUDA 12.9 libraries.

1. Create the Conda environment.

   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment.

   ```bash
   conda activate fbpic-warp
   ```

3. Select the environment's CUDA libraries.

   ```bash
   export CUDA_PATH="$CONDA_PREFIX"
   export CUDA_HOME="$CONDA_PREFIX"
   export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
   ```

4. Install this package in editable mode.

   ```bash
   pip install -e ".[dev]"
   ```

   The distribution name remains `fbpic`, so this install also satisfies packages that require `fbpic`.

5. Run the core tests from this repository.

   ```bash
   CUDA_VISIBLE_DEVICES=0 NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 \
   PYTHONPATH="$PWD" MPLBACKEND=Agg python -m pytest -q \
       tests/test_cpu_gpu_deposition.py tests/test_cuda_transform.py \
       tests/test_periodic_plasma_wave.py
   ```

To compare with upstream FBPIC, select the upstream or the port checkout through `PYTHONPATH` in separate processes.

GPU simulations use one shared CUDA stream and enable graph replay by default.
Pass `use_cuda_graphs=False` to `Simulation` for eager execution.
Pass `fuse_particle_kernels=False` to use the per-mode deposition and gathering kernels.
Pass `hankel_precision='float32'` for the faster single-precision Hankel transform.

## Upstream project information

The following information comes from the upstream FBPIC repository.
Use the environment above for this fork; `pip install fbpic` installs upstream FBPIC, not this fork.
Report problems with this fork to the maintainers of this repository, not to the upstream support channels.

# Fourier-Bessel Particle-In-Cell code (FBPIC)

[![DOI](https://zenodo.org/badge/69215997.svg)](https://zenodo.org/badge/latestdoi/69215997)

Online documentation: [http://fbpic.github.io](http://fbpic.github.io)

## Overview

FBPIC is a
[Particle-In-Cell (PIC) code](https://en.wikipedia.org/wiki/Particle-in-cell)
for relativistic plasma physics.

It is especially well-suited for physical simulations of
**laser-wakefield acceleration** and **plasma-wakefield acceleration**, with close-to-cylindrical symmetry.

### Algorithm

The distinctive feature of FBPIC is to use
a **spectral decomposition in
cylindrical geometry** (Fourier-Bessel
decomposition) for the fields. This combines the advantages of **spectral 3D** PIC codes (high accuracy and stability) and
those of **finite-difference cylindrical** PIC codes
(orders-of-magnitude speedup when compared to 3D simulations).
For more details on the algorithm, its advantages and limitations, see
the [documentation](http://fbpic.github.io).

### Language and hardware

FBPIC is written entirely in Python, but uses
[Numba](http://numba.pydata.org/) Just-In-Time compiler for high
performance. In addition, the code can run on **CPU** (with multi-threading)
and on **GPU**. For large simulations, running the
code on GPU can be much faster than on CPU.

### Advanced features of laser-plasma acceleration

FBPIC implements several useful features for laser-plasma acceleration, including:
- Moving window
- Cylindrical geometry (with azimuthal mode decomposition)
- Calculation of space-charge fields at the beginning of the simulation
- Intrinsic mitigation of Numerical Cherenkov Radiation (NCR) from relativistic bunches
- Field ionization module (ADK model)

In addition, FBPIC supports the **boosted-frame** technique (which can
dramatically speed up simulations), and includes:
- Utilities to convert input parameters from the lab frame to the boosted frame
- On-the-fly conversion of simulation results from the boosted frame back to the lab frame
- Suppression of the Numerical Cherenkov Instability (NCI) using the Galilean technique

## Installation

The installation instructions below are for a local computer. For more
details, or for instructions specific to a particular HPC cluster, see
the [documentation](http://fbpic.github.io).

The recommended installation is through the
[Anaconda](https://www.continuum.io/why-anaconda) distribution.
If Anaconda is not your default Python installation, download and install
it from [here](https://www.continuum.io/downloads).

**Installation steps**:

- Install the dependencies of FBPIC. This can be done in two lines:
    ```
    conda install numba scipy h5py mkl
    conda install -c conda-forge mpi4py
    ```
- Download and install FBPIC:
    ```
    pip install fbpic
    ```
    (If you want to run FBPIC through the [PICMI](https://picmi-standard.github.io/)
    interface, you can instead use `pip install fbpic[picmi]`.
    FBPIC currently supports `picmistandard` up to version 0.34.0.)

- **Optional:** in order to run on GPU, install the additional package
    `cupy` — e.g. using CUDA version 11.8. (The command below also automatically installs `cudatoolkit` which is also needed by FBPIC.)
    ```
    conda install cupy cuda-version=11.8
    ```
    (In the above command, you should choose a CUDA version that is compatible with your GPU driver ; see [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#major-components__table-cuda-toolkit-driver-versions) for more info.)

    If you want to use CUDA version 12+ which no longer comes with the `cudatoolkit` package, explicit installation of `cuda-nvcc` and `cuda-nvrtc` is required
    ```
    conda install cupy cuda-version=12.0 cuda-nvcc cuda-nvrtc
    ```

- **Optional:** in order to run on a CPU which is **not** an Intel model, you
need to install `pyfftw`, in order to replace the MKL FFT:
    ```
    conda install -c conda-forge pyfftw
    ```

## Running simulations

Once installed, FBPIC is available as a **Python module** on your
system.

Therefore, in order to run a physical simulation, you will need a **Python
script** that imports FBPIC's functionalities and use them to setup the
simulation. You can find examples of such scripts in the
[documentation](http://fbpic.github.io) or in this repository, in `docs/source/example_input/`.

Once your script is ready, the simulation is run simply by typing:
```
python fbpic_script.py
```
The code outputs HDF5 files, that comply with the
[OpenPMD standard](http://www.openpmd.org/#/start),
 and which can thus be read as such (e.g. by using the
 [openPMD-viewer](https://github.com/openPMD/openPMD-viewer)).

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow of this fork.
Contribute changes to the CPU path or the physics models to [upstream FBPIC](https://github.com/fbpic/fbpic/blob/dev/CONTRIBUTING.md) as well.

## Research & Attribution

FBPIC was originally developed by Remi Lehe at [Berkeley Lab](http://www.lbl.gov/),
and Manuel Kirchen at
[CFEL, Hamburg University](http://lux.cfel.de/). The code also
benefitted from the contributions of Soeren Jalas (CFEL), Kevin Peters (CFEL),
Irene Dornmair (CFEL), Laurids Jeppe (CFEL), Igor Andriyash (Laboratoire d'Optique Appliquee),
Omri Seemann (Weizmann Institute), Daniel Seipt (University of Michigan),
Samuel Yoffe (University of Strathclyde), David Grote (LLNL and LBNL) and
Anton Golovanov (Weizmann Institute).

FBPIC's algorithms are documented in following scientific publications:

* General description of FBPIC and it's algorithm (original paper): [R. Lehe et al., CPC, 2016](http://www.sciencedirect.com/science/article/pii/S0010465516300224) ([arXiv](https://arxiv.org/abs/1507.04790))
* Boosted-frame technique with Galilean algorithm: [M. Kirchen et al., PoP, 2016](https://aip.scitation.org/doi/10.1063/1.4964770) ([arXiv](https://arxiv.org/abs/1608.00215)) and [Lehe et al., PRE, 2016](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.94.053305) ([arXiv](https://arxiv.org/abs/1608.00227))
* Parallel finite-order solver for multi-CPU/GPU simulations: [S. Jalas et al., PoP, 2017](https://aip.scitation.org/doi/abs/10.1063/1.4978569) ([arXiv](https://arxiv.org/abs/1611.05712))
* Parallel finite-order boosted-frame simulations for multi-CPU/GPU simulations: [M. Kirchen et al., PRE, 2020](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.102.013202)
* Tracking of particle spin evolution: [M. J. Quin MSc thesis](https://doi.org/10.11588/heidok.00033704), [Z. Gong et al., MRE 2023](https://doi.org/10.1063/5.0152382)

If you use FBPIC for your research project: that's great! We are
very pleased that the code is useful to you!

If your project even leads to a scientific publication, please consider citing at least FBPIC's original paper. If your project uses the more advanced algorithms, please consider citing the respective publications in addition.
