# Validation

CPU unit tests use synthetic weights/data and check feature ordering, artifact
integrity, image preparation, independent codec imports, output ordering for
GPU=1–4, error propagation, worker cleanup, and private release packaging.

The private codec fixture contains eight selected training/validation images
and original-model outputs. Camera-1, TopView, Gaia and decoder extraction were
checked exactly against the original CPU implementations on macOS. FROG was
checked against saved coordinates. OAP preparation/projection was checked on
its stored training-mean frame; raw OAP shot files were unavailable locally.
Linux CPU checks use small numerical tolerances because CPU kernels differ.

The GPU reference fixture contains 462 validation queries and frozen expected
21-coordinate predictions and image SHA-256 values for two seeds. No test
partition is evaluated. Neither fixture is committed or included in a release.

```bash
python validation/check_codecs.py --artifacts /path/to/artifacts \
  --fixture /path/to/codec_fixture.npz --output outputs/codecs.json
python validation/check_gpu.py --artifacts /path/to/artifacts \
  --fixture /path/to/fixture.npz --gpus 1 2 3 4 --output outputs/gpu_check.json
```

On Polaris, copy this subproject and private artifacts to compute-readable
storage. From the subproject directory, submit the existing PBS script with
your allocation and private paths:

```bash
qsub -A YOUR_PROJECT \
  -v ARTIFACT_DIR=/path/to/artifacts,CHECK_FIXTURE=/path/to/fixture.npz,CODEC_FIXTURE=/path/to/codec_fixture.npz,SDM_SOURCE=/path/to/pinned-sdm \
  validation/polaris.pbs
```

The job installs real package copies in a temporary environment, runs outside
the source directory with Python isolation, uses an empty offline model cache,
and checks every GPU count and both seeds. The SDM source checkout must match
the pinned commit and source hashes. Results go to ignored `outputs/`.
Run history and measured results are recorded separately in `summary.json`.
