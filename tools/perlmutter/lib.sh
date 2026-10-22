# Shared helpers for the Perlmutter FBPIC scripts. Source this file; do not run it.
# shellcheck shell=bash

PERLMUTTER_TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() {
    echo "error: $*" >&2
    exit 1
}

log() {
    echo "[$(date -u +%H:%M:%S)] $*" >&2
}

load_config() {
    local config="${FBPIC_PERLMUTTER_CONFIG:-$HOME/.config/fbpic-perlmutter.sh}"
    [[ -f "$config" ]] || die "missing config $config (copy $PERLMUTTER_TOOLS/config.example.sh)"
    # shellcheck source=/dev/null
    source "$config"
    local name
    for name in FBPIC_SRC FBPIC_REPO_URL UPSTREAM_FBPIC_URL UPSTREAM_FBPIC_SHA FBPIC_ENV FBPIC_RUNS; do
        [[ -n "${!name:-}" ]] || die "$name is not set in $config"
    done
    export FBPIC_PERLMUTTER_CONFIG="$config"
    REPO_GIT="$FBPIC_SRC/inversion-genesis-mission.git"
    UPSTREAM_GIT="$FBPIC_SRC/fbpic.git"
}

# Variant table: branch, solver kind, and FBPIC options live in bench/variants.py.
# The shell layer needs only the default branch and the solver source.
variant_branch() {
    case "$1" in
        numba) echo "snidhan/perl-main-run" ;;
        warp-fp64 | warp-fp32) echo "snidhan/perl-warp-run" ;;
        *) die "unknown variant '$1' (numba, warp-fp64, warp-fp32)" ;;
    esac
}

variant_solver() {
    case "$1" in
        numba) echo "upstream-fbpic" ;;
        warp-fp64 | warp-fp32) echo "lpa/FBPIC_Warp" ;;
        *) die "unknown variant '$1'" ;;
    esac
}

# Steps per mode: N, 0 for the full interaction, or 1/K for a fraction of it.
mode_steps() {
    case "$1" in
        smoke) echo 100 ;;
        kernels | window) echo 1/10 ;;
        full) echo 0 ;;
        *) die "unknown mode '$1' (smoke, kernels, window, full)" ;;
    esac
}

utc_stamp() {
    date -u +%Y%m%dT%H%M%SZ
}
