# Contributing to FBPIC-Warp

FBPIC-Warp is a fork of [FBPIC](https://github.com/fbpic/fbpic).
Contribute changes to the CPU path or the physics models to [upstream FBPIC](https://github.com/fbpic/fbpic/blob/dev/CONTRIBUTING.md) as well.

## Workflow

1. Create the environment and install the package (see [README.md](README.md)).
2. Create a branch for one change.
3. Make small commits. Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) messages.
4. Format and lint the code before each commit (see "Style and conventions").
5. Run the test suite on a GPU (see "Tests").
6. For a change to a GPU kernel, add the parity and timing evidence to the commit message (see "GPU kernel changes").
7. Open a pull request and describe what changed and why.

## Tests

Run each test file in its own process from the repository root.
Some upstream tests change the working directory, so a single `pytest tests` process can fail for reasons that are not in the code.
Some tests start example scripts in child processes. These need the package installed (`pip install -e .`) or the checkout on `PYTHONPATH`.

```bash
export OPENPMD_VERIFY_HOMOGENEOUS_EXTENTS=0
export PYTHONPATH="$PWD"
for f in tests/test_*.py; do
    python -m pytest "$f" -q || echo "FAILED: $f"
done
```

The scripts in `tests/unautomated/` are manual checks. Do not collect them with pytest.

## GPU kernel changes

A change to a Warp kernel, its launch configuration, or its dispatch needs this evidence:

1. The fused and unfused kernels agree within the tolerances in `tests/test_fused_particle_operations.py`.
2. A fixed-seed simulation agrees with the result before the change.
   GPU atomics make repeated runs differ by about 1e-11 relative (fp64) after 300 steps, so do not expect bitwise equality.
3. The time per step does not increase by more than the run-to-run noise (about 2 %).

## Style and conventions

- Format with [Black](https://black.readthedocs.io/) 25.1.0 and lint with [Ruff](https://docs.astral.sh/ruff/) 0.11.11 (default rules).
  The line length is the Black default (88 characters).
- Document functions and classes with docstrings.
  Upstream FBPIC code uses the [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html); keep that style in upstream files.
- Use lower-case names with underscores for variables and functions (for example `get_filter_array`).
  Use CapWords for classes (for example `Fields`).
- Do not do device-to-host copies in the simulation step. Keep them in diagnostics and tests.
