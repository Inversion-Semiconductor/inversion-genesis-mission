# Gas jet density post-processing

Tools for turning raw simulation density lineouts into symmetrized profiles, building 2D interpolations over `(x, backing pressure)`, and plotting or exporting density fields.

## Pipeline overview

```
raw lineout files          process_symmetric.py          HDF5 density cube
(density_lineouts_raw/)  ──────────────────────────►  (400_um.h5)
                                                          │
                                                          ▼
                                              interpolate_density.py
                                              (library: build field)
                                                          │
                                                          ▼
                                                   plot_density.py
                                              (interactive plots / PNG export)
```

1. **`process_symmetric.py`** — parse raw lineout exports, mirror each profile across `z = 0`, apply a density scale factor, and write one HDF5 density cube per nozzle. It retains per-lineout `.npz` output for compatibility.
2. **`interpolate_density.py`** — load an HDF5 density cube for a nozzle and build a 3D density field `ρ(z, x, p)` with 2D interpolation in the `(x, pressure)` plane.
3. **`plot_density.py`** — visualize the interpolated field as an `(x, z)` map or a `z` lineout, interactively or as a saved figure.
4. **`convergence.py`** — compare raw ANSYS lineouts or full CGNS density fields with a chosen reference resolution and plot their relative integrated differences.
5. **`cgns_to_density_hdf5.py`** — interpolate one unstructured CGNS density field onto a regular `(z, x)` grid and write a compatible single-pressure HDF5 density cube.
6. **`shock_evaluator.py`** — interactively inspect the density-gradient magnitude in an unstructured CGNS field and sample selected scalar fields along a drawn segment.

## Data layout

### Raw input (`process_symmetric.py`)

Raw files are text exports with one or more labeled sections. Each section is a `(z, density)` table:

```
((xy/key/label "x-0-5")
0.002675  0.00097875
0.0027    0.00087528
...
)
```

Organize raw files in one nozzle directory with one file per backing pressure:

```
data/density_lineouts_raw/
└── 400_um/
    ├── 5_bar.txt      # single file containing all x lineouts for 5 bar
    └── 20_bar.txt
```

Use `--hdf5-output` with a single nozzle directory to create the coordinate-indexed density cube.

### Processed output (`interpolate_density.py`, `plot_density.py`)

The canonical interpolation input is one HDF5 file per nozzle:

```
data/density_lineouts/
└── 400_um.h5
```

| Dataset | Shape | Units |
|---|---|---|
| `z_m` | `(n_z,)` | m |
| `x_mm` | `(n_x,)` | mm |
| `pressure_bar` | `(n_pressure,)` | bar |
| `density` | `(n_z, n_x, n_pressure)` | scaled density |

Root attributes include `nozzle`, `density_scale`, and `density_axis_order="z_m,x_mm,pressure_bar"`.
Each `density` axis also has an HDF5 dimension-scale link and label to its
corresponding coordinate dataset, so the mapping remains unambiguous even when
multiple coordinate arrays have the same length.

Per-lineout `.npz` files are still written under the output directory for compatibility, but interpolation and plotting use the HDF5 cube.


---

## `cgns_to_density_hdf5.py`

Convert a directory of pressure-tagged CGNS fields into the same dimension-scaled
HDF5 structure used by `interpolate_density.py`. The input directory combines files named
`<pressure>_bar.cgns` into the output `pressure_bar` axis; for example,
`10_bar.cgns` and `30_bar.cgns` produce pressure coordinates `[10, 30]`.

For the provided Fluent CGNS export, `CoordinateY` maps to `z_m`,
`CoordinateX` maps to `x_mm`, and `CoordinateZ` is identically zero.
Samples with `CoordinateX < 0` are discarded, and the remaining
`CoordinateY >= 0` half-plane is mirrored across `CoordinateY = 0`. The
conversion linearly interpolates these point samples to the requested regular
grid and copies paired `z_m` grid values so the output is exactly mirrored;
default output values outside the simulated domain are zero.

### Usage

```bash
python cgns_to_density_hdf5.py \
    data/density_field_raw/htu_density \
    --output data/density_lineouts/htu_density.h5 \
    --density-units kg/m^3
```

### Key arguments

| Argument | Description |
|---|---|
| `input_dir` | Directory of `<pressure>_bar.cgns` files. |
| `-o`, `--output` | Required compatible HDF5 output path. |
| `--density-units` | Required units for the output density dataset. |
| `--density-scale` | Scale factor applied to source density before writing; defaults to `1`. |
| `--z-points`, `--x-points` | Regular grid resolution; defaults to `1000` and `500`. |
| `--z-bounds`, `--x-bounds` | Optional output bounds in source-coordinate meters. Directory input defaults to the common extent of every field. |

The output stores `source_format`, `source_files`, `source_pressures_bar`,
coordinate mapping, interpolation method, outside-domain policy, source x
selection, and z symmetry operations as root attributes.

---

## `shock_evaluator.py`

Interactively inspect the density-gradient magnitude $|\nabla \rho|$ directly
from one unstructured CGNS field. The file must contain point-aligned `dp-dX`
and `dp-dY` fields under `Base/Zone/FlowSolution.N:1`.

```bash
python shock_evaluator.py \
    data/generic_field_raw/5_bar_htu_all_fields.cgns
```

The heatmap displays $\sqrt{(\mathrm{dp-dX})^2 + (\mathrm{dp-dY})^2}$ on a
logarithmic scale. Enter positive, increasing values in the lower and upper
limit text boxes to rescale it; invalid entries leave the last valid limits in
place. `CoordinateY` is displayed horizontally as `z [mm]`, and `CoordinateX`
vertically as `x [mm]`.

Click and drag with the primary mouse button across the heatmap to define a
segment. The lineout panel in the same window updates immediately with distance
from the segment start in mm. Its checklist contains every valid physical scalar field
in the CGNS solution; `Density` is selected initially. Samples outside the
unstructured mesh are displayed as gaps rather than extrapolated values. One
selected field retains its source units; multiple selected fields are each
scaled to unit magnitude on an arbitrary axis, without shifting their zero
baselines, for direct shape comparison.

Use `--show-color-limit-boxes` to display editable lower and upper logarithmic
heatmap color limits. The controls are hidden by default.

When the selected endpoints are within the mesh and contain the required state
fields, the evaluator treats the first point as upstream and the second as
downstream. It calculates $M_1$ from upstream `Mach`, forms velocities from
`(Axial_Velocity, Radial_Velocity)`, and calculates
$\beta=\arccos(((v_1-v_2)\cdot v_1)/(|v_1|\,|v_1-v_2|))$. A dashed lime line
from the segment midpoint shows the velocity-change direction. The lineout
window reports simulated endpoint ratios and `oblique_*` predictions for
pressure, density, and temperature.

Use `--shock-angle auto` for the endpoint-velocity method described above.
`--shock-angle perp` treats the selected segment as the shock tangent; its
normal defines the shock direction, and the endpoint with the larger `Mach`
value is used as upstream for the predicted oblique-shock ratios. Equal endpoint
Mach values are rejected because that upstream direction is ambiguous.

`--shock-angle manual` first accepts the dragged evaluation segment, then waits
for a third heatmap click. The ray from the segment midpoint to that point
defines the lime shock direction; the drag start remains the upstream state.
For every mode, observed ratios remain drag-end divided by drag-start.

| Argument | Description |
|---|---|
| `cgns_path` | Input unstructured CGNS field file. |
| `--line-samples N` | Samples along each selected segment; defaults to `400`. |
| `--gamma VALUE` | Heat-capacity ratio for oblique-shock predictions; defaults to `1.4`. |
| `--flow-solution NAME` | CGNS flow-solution group; defaults to `FlowSolution.N:1`. |
| `--shock-angle {auto,perp,manual}` | Select velocity-derived, segment-normal, or third-point shock angle; defaults to `auto`. |
| `--x-coordinate NAME` | Transverse coordinate name; defaults to `CoordinateX`. |
| `--z-coordinate NAME` | Axial coordinate name; defaults to `CoordinateY`. |
| `--x-bounds MIN_M MAX_M` | Optional transverse x bounds in source-coordinate metres. |
| `--z-bounds MIN_M MAX_M` | Optional axial z bounds in source-coordinate metres. |

---

## `convergence.py`

Measure density convergence from raw ANSYS lineout exports or full 2D CGNS fields.
The input directory must contain one file per resolution, named by minimum grid
size in mm with a decimal point represented by `_`: for example, `0_1.txt` or
`0_1.cgns` means `0.1 mm`, and `0_075.txt` means `0.075 mm`.

By default, each coarse grid is compared with the finest available resolution,
treated as ground truth. Use `--reference adjacent` to compare each grid with
the immediately finer available grid instead. For each matching lineout label,
the two profiles are linearly interpolated on their shared z domain and the
relative integrated absolute difference is computed:

```
integral(|density_coarse - density_finer| dz)
------------------------------------------------
       integral(|density_finer| dz)
```

The calculation does not mirror profiles across `z = 0`; it evaluates the raw
exported profiles over their common positive-z range. Duplicate z coordinates
are retained only once, using the last listed density value.

For `.cgns` input, each unstructured 2D field is linearly interpolated once
onto one common regular `(x, z)` grid, then all resolution comparisons reuse
those cached grids. The error is the area-integrated absolute density difference
divided by the reference field's integrated absolute density. Use `--x-min`,
`--x-max`, `--z-min`, and `--z-max` to clip the comparison region; for example,
`--x-min 0` restricts the comparison to `x >= 0`.
Use `--interpolation cubic` to opt into Clough-Tocher interpolation; `linear`
remains the default and is typically faster for large CGNS fields.

The CGNS field curve and its CSV metric remain area-integrated errors. Its
summary plot shows two error distributions over the common grid: pointwise
relative errors (excluding cells where the reference density is zero), and
absolute density errors. The latter first calculates the mean and standard
deviation of `abs(density - reference_density)`, then divides both by the peak
absolute reference density. CSV columns prefixed with
`*_peak_normalized_absolute_error` record this second metric. Use
`--field-error local-relative`, `--field-error peak-normalized`, or the default
`--field-error both` to select the CGNS summary-plot curves. The lineout-only
plot is not shown for CGNS input.

### Usage

```bash
# Display aggregate and per-lineout convergence curves interactively
python convergence.py data/density_lineouts_raw/htu_conv_test

# Save both plot views and all metrics to adjacent PNG and CSV files
python convergence.py data/density_lineouts_raw/htu_conv_test \
    --view both \
    --output figures/htu_density_convergence.png

# Compare only with the immediately finer resolution
python convergence.py data/density_lineouts_raw/htu_conv_test \
    --reference adjacent

# Save only the aggregate mean and standard-deviation spread
python convergence.py data/density_lineouts_raw/htu_conv_test \
    --view summary \
    --output figures/htu_density_convergence_summary.png \
    --metrics-output figures/htu_density_convergence_metrics.csv

# Compare resolution-tagged 2D CGNS fields over x >= 0
python convergence.py data/density_fields_raw/htu_conv_test \
    --input-format cgns \
    --x-min 0 \
    --output figures/htu_cgns_convergence.png
```

### Arguments

| Argument | Description |
|---|---|
| `input_dir` | Directory of resolution-tagged `.txt` ANSYS lineouts or `.cgns` fields. |
| `--input-format` | `auto` (default), `lineout`, or `cgns`. Auto selects CGNS when the directory contains `.cgns` files. |
| `--view` | `summary` for mean and standard-deviation spread, `lineouts` for one curve per lineout, or `both` (default for lineout input). CGNS input always uses the summary view. |
| `--reference` | `finest` (default) compares every coarse grid to the finest resolution treated as ground truth; `adjacent` uses the immediately finer grid. |
| `--field-error` | CGNS summary curve: `local-relative`, `peak-normalized`, or `both` (default). It affects only plotting; CSV output retains both metrics. |
| `-o`, `--output` | Save the plot instead of opening an interactive Matplotlib window. |
| `--metrics-output` | CSV destination for every per-lineout metric and its corresponding aggregate statistics. |
| `--x-min`, `--x-max`, `--z-min`, `--z-max` | Optional CGNS comparison bounds in m. |
| `--x-points`, `--z-points` | CGNS common-grid resolution; defaults to `200` in each direction. |
| `--interpolation` | CGNS unstructured interpolation: `linear` (default) or `cubic` Clough-Tocher. |
| `-t`, `--title` | Optional plot title. |

When `--metrics-output` is omitted, the CSV is written alongside `--output`; for
interactive plots it is written to `<input_dir>/convergence_metrics.csv`.

---

## `process_symmetric.py`

Parse raw density lineout files, sort by `z`, mirror the positive-`z` half across `z = 0`, multiply densities by a scale factor, and save one HDF5 density cube for a nozzle.

### Usage

```bash
# Process one nozzle's pressure files into a coordinate-indexed HDF5 cube
python process_symmetric.py 1.0e20 \
    --input-dir data/density_lineouts_raw/400_um \
    --output-dir data/density_lineouts \
    --hdf5-output data/density_lineouts/400_um.h5 \
    --density-units cm^-3

# Process a single raw file
python process_symmetric.py 1.0e20 \
    data/density_lineouts_raw/800_um_conical/20_bar \
    --output-dir data/density_lineouts
```

### Arguments

| Argument | Description |
|---|---|
| `density_scale` | Required. Factor applied to raw densities before saving. |
| `input` | Path to a single raw lineout file. |
| `-i`, `--input-dir` | Directory of raw lineout files (processed recursively). Mutually exclusive with `input`. |
| `-o`, `--output-dir` | Required. Base output directory. |
| `--hdf5-output` | Optional HDF5 output path. Requires `--input-dir` to specify one nozzle directory. |
| `--density-units` | Required with `--hdf5-output`. Units for the `density` dataset, such as `cm^-3` or `m^-3`. |

### Output paths

- **Single file:** `<output-dir>/<stem>/<label>.npz`
- **Directory input:** `<output-dir>/<relative-path>/<label>.npz` (relative path preserved)

Example: `data/density_lineouts_raw/800_um_conical/20_bar` with section label `x-1-0` writes to `data/density_lineouts/800_um_conical/20_bar/x-1-0.npz`.

---

## `interpolate_density.py`

Library module (no CLI) that loads one nozzle HDF5 density cube and builds a `DensityInterpolation` object over `(z, x, backing pressure)`.

For a given `z`, density on the `(x, pressure)` grid is interpolated using either:

- **`linear`** (default) — `RegularGridInterpolator` with linear splines
- **`cubic`** — `RegularGridInterpolator` with cubic splines (requires ≥4 points per axis)
- **`quintic`** — `RegularGridInterpolator` with quintic splines (requires ≥6 points per axis)
- **`pchip`** — `RegularGridInterpolator` with piecewise cubic Hermite splines (requires ≥4 points per axis)

Along `z`, interpolation is linear between tabulated profile points.

### Basic usage

```python
from pathlib import Path
from interpolate_density import build_density_interpolation

field = build_density_interpolation(
    Path("data/density_lineouts/400_um.h5"),
    method="linear",
)

# Single point
rho = field.interpolate(z_value=0.0, x_value=1.0, pressure_value=12.5)

# Profile along z at fixed x and pressure
profile = field.interpolate_along_z(field.z, x_value=1.0, pressure_value=12.5)

# (x, z) map at fixed backing pressure
import numpy as np
x = np.linspace(*field.x_extent, 200)
z = np.linspace(*field.z_extent, 200)
grid = field.interpolate_xz_grid(x, z, pressure_value=12.5)
```

### FBPIC-compatible density function

`build_density_callable` returns a function `(z, r) → ρ` suitable for FBPIC, with fixed `x` and backing pressure:

```python
from interpolate_density import build_density_callable

density = build_density_callable(
    Path("data/density_lineouts/400_um.h5"),
    backing_pressure=12.5,
    x_position=1.0,
)

# FBPIC calls density(z, r); r is ignored
rho = density(z_array, r_array)
```

### Key API

| Function / method | Description |
|---|---|
| `build_density_interpolation(hdf5_path, method=...)` | Load a nozzle HDF5 cube and build the field. |
| `build_density_callable(hdf5_path, pressure, x, ...)` | Return an FBPIC-compatible `(z, r)` callable. |
| `field.interpolate(z, x, pressure)` | Scalar density at one point. |
| `field.interpolate_along_z(z_values, x, pressure)` | Density along a `z` array. |
| `field.interpolate_xz_grid(x, z, pressure)` | `(n_x, n_z)` map at fixed pressure. |
| `field.density_at_z(z)` | `(n_x, n_pressure)` slice at fixed `z`. |
| `field.x_extent`, `field.z_extent`, `field.pressure_extent` | Valid ranges for each axis. |

Query points outside the tabulated ranges raise `ValueError` with a message listing the valid extents.

---

## `plot_density.py`

Build a density interpolation and display it interactively, or save a static figure.

### Usage

```bash
# Interactive (x, z) density map with pressure slider (default)
python plot_density.py data/density_lineouts/400_um.h5

# Interactive lineout along z with pressure and x sliders
python plot_density.py data/density_lineouts/400_um.h5 \
    --plot lineout

# Lineout via build_density_callable (sanity-check for FBPIC integration)
python plot_density.py data/density_lineouts/400_um.h5 \
    --plot callable_lineout

# Save a static (x, z) map at a specific pressure
python plot_density.py data/density_lineouts/400_um.h5 \
    --plot xz_map --pressure 12.5 -o density_map.png
```

When `-o` / `--output` is given, sliders are disabled and the figure is written to disk at 150 dpi. Without `--output`, an interactive matplotlib window opens.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `hdf5_path` | — | HDF5 density cube for one nozzle. |
| `--plot` | `xz_map` | Plot mode: `xz_map`, `lineout`, or `callable_lineout`. |
| `--method` | `linear` | 2D interpolation method: `linear`, `cubic`, `quintic`, or `pchip`. |
| `--pressure` | lowest in dataset | Backing pressure [bar] for static output. |
| `--x` | lowest in dataset | x position [mm] for static lineout output. |
| `-o`, `--output` | — | Save figure to this path instead of showing interactively. |
| `-t`, `--title` | auto | Custom figure title. |

### Plot modes

| Mode | Description |
|---|---|
| `xz_map` | Pseudocolor of density in the `(x, z)` plane. Interactive mode includes a backing-pressure slider. |
| `lineout` | Density vs `z` at fixed `x` and pressure. Interactive mode includes pressure and `x` sliders. |
| `callable_lineout` | Same as `lineout`, but evaluates density through `build_density_callable` rather than calling the field directly. Useful for verifying the FBPIC integration path. |

Axis labels: `z` and `x` are shown in mm; density is labeled `[1e18 cm⁻³]`.

---

## Example end-to-end workflow

```bash
# 1. Symmetrize one nozzle's raw lineouts into HDF5
python process_symmetric.py 1.0e20 \
    --input-dir data/density_lineouts_raw/400_um \
    --output-dir data/density_lineouts \
    --hdf5-output data/density_lineouts/400_um.h5 \
    --density-units cm^-3

# 2. Explore the nozzle data interactively
python plot_density.py data/density_lineouts/400_um.h5

# 3. Export a figure for a report
python plot_density.py data/density_lineouts/400_um.h5 \
    --plot xz_map --pressure 20 -o figures/400_um_20bar.png
```

## Dependencies

- Python 3
- NumPy
- SciPy
- Matplotlib
- h5py
