# Summary

For a general analaysis workflow 
1. Run simulation(s) on AWS.
2. Extract ebeam from h5 data using `utils/extract_rho_electron.py` for a single simulation or `utils/extract_rho_electron_set.py` for a parameter scan
3. Transfer the "analysis" directory to a local machine
   - Rename this folder if necessary, useful if running successive sets of scans (see `aws_set1_simulations/htu_scans/`)
4. Perform analysis on this data using scripts within this directory.  See individual scripts for more information

## Single simulation analysis

Good examples include those contained within `aws_set1_simulations/htu_refined`.

Recommended to use `plot_ebeam_analysis.py` to look at the ebeam phase space and statistics for a single simulation.

Another script, `fit_sinusoid_to_beam.py` can look at sinusoidal shapes in the x-z phase space of an electron beam.

## 1D parameter scan analysis

Good examples include those contained within `aws_set1_simulations/htu_scans`

Use `visualize_ebeamparams_vs_scan.py` to plot how ebeam parameters vary as a particular input parameter is scanned.

Use `summarize_scan_sensitivities.py` to calculate correlations between all ebeam parameters against all scanned parameters.
