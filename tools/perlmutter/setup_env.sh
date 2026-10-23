#!/usr/bin/env bash
# Create the shared FBPIC dependency environment once, on a login node.
#
# Usage: setup_env.sh [--prefix DIR] [--mpi external|conda] [--recreate] [--local]
#   --prefix    env location (default: FBPIC_ENV from the config)
#   --mpi       MPI flavor (default: FBPIC_MPI from the config)
#   --recreate  remove an existing env at the prefix first
#   --local     skip NERSC modules (workstation test of the env spec)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

prefix="$FBPIC_ENV"
mpi="${FBPIC_MPI:-external}"
recreate=0
local_mode=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) prefix="$2"; shift 2 ;;
        --mpi) mpi="$2"; shift 2 ;;
        --recreate) recreate=1; shift ;;
        --local) local_mode=1; shift ;;
        -h | --help) sed -n '2,9p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done
[[ "$mpi" == external || "$mpi" == conda ]] || die "--mpi must be external or conda"

if [[ $local_mode -eq 0 ]]; then
    module load conda
    if command -v showquota >/dev/null 2>&1; then
        log "Global Common quota (the project limit is shared by all members):"
        showquota --cmn "$NERSC_PROJECT" || true
    fi
    # Keep the package cache off the small /global/common/software quota.
    export CONDA_PKGS_DIRS="${SCRATCH:?}/conda-pkgs"
fi
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ -d "$prefix" ]]; then
    if [[ $recreate -eq 1 ]]; then
        log "removing existing env $prefix"
        conda env remove --yes --prefix "$prefix"
    else
        log "env exists at $prefix; use --recreate to rebuild"
        exit 0
    fi
fi

mkdir -p "$(dirname "$prefix")"
log "creating $prefix from env/fbpic-deps.yml"
conda env create --yes --prefix "$prefix" --file "$PERLMUTTER_TOOLS/env/fbpic-deps.yml"

if [[ "$mpi" == external ]]; then
    # NERSC: conda-forge external MPICH plus cray-mpich-abi at run time.
    # conda-forge does not always have an external build that fits the other pins.
    if ! conda install --yes --prefix "$prefix" -c conda-forge mpi4py "mpich=*=external_*"; then
        log "warning: no external MPICH build fits this env; using conda MPICH (one rank per job step)"
        mpi=conda
    fi
fi
if [[ "$mpi" == conda ]]; then
    conda install --yes --prefix "$prefix" -c conda-forge mpi4py mpich
fi
conda clean --all --yes >/dev/null

conda list --explicit --prefix "$prefix" >"$prefix.lock.txt"
echo "$mpi" >"$prefix.mpi"
log "lock file: $prefix.lock.txt; MPI flavor: $mpi"
du -sh "$prefix" >&2 || true

# conda activation hooks (cuda-nvcc) read unset variables; relax nounset for them.
set +u
conda activate "$prefix"
set -u
python - <<'PY'
import importlib

for name in ("numpy", "numba", "cupy", "warp", "mpi4py", "h5py", "attrs", "lasy",
             "openpmd_viewer", "periodictable", "yaml"):
    module = importlib.import_module(name)
    print(f"{name:16s} {getattr(module, '__version__', '?')}")
try:
    import fbpic
except ImportError:
    print("fbpic            not installed (expected)")
else:
    raise SystemExit(f"fbpic must not be installed in the env: {fbpic.__file__}")
PY
log "done. Test a GPU node with: launch.sh --mode smoke --here (inside salloc)"
