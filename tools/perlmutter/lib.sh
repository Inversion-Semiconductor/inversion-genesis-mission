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
    for name in FBPIC_WORKTREES FBPIC_REPO_URL UPSTREAM_FBPIC_GIT UPSTREAM_FBPIC_URL \
        UPSTREAM_FBPIC_SHA FBPIC_ENV FBPIC_RUNS; do
        [[ -n "${!name:-}" ]] || die "$name is not set in $config"
    done
    export FBPIC_PERLMUTTER_CONFIG="$config"
    REPO_GIT="$FBPIC_WORKTREES/.bare"
    UPSTREAM_GIT="$UPSTREAM_FBPIC_GIT"
}

# Worktree directory of a branch: the part of the name after the last "/".
worktree_dir() {
    echo "$FBPIC_WORKTREES/${1##*/}"
}

# Resolve a ref to a commit: the local branch first, then origin/<ref>.
resolve_ref() {
    git -C "$REPO_GIT" rev-parse --verify --quiet "$1^{commit}" ||
        git -C "$REPO_GIT" rev-parse --verify --quiet "refs/remotes/origin/$1^{commit}"
}

# Variant table: branch, solver kind, and FBPIC options live in bench/variants.py.
# The shell layer needs only the default branch and the solver source: numba runs
# upstream FBPIC from perl-main-run; every warp-* variant runs lpa/FBPIC_Warp.
variant_known() {
    [[ "$1" =~ ^[a-z0-9-]+$ ]] &&
        grep -qE "^\s*(Variant\()?\"$1\"," "$PERLMUTTER_TOOLS/bench/variants.py"
}

variant_branch() {
    variant_known "$1" || die "unknown variant '$1' (see $PERLMUTTER_TOOLS/bench/variants.py)"
    case "$1" in
        numba) echo "snidhan/perl-main-run" ;;
        warp-*) echo "snidhan/perl-warp-run" ;;
        *) die "variant '$1' has no default branch" ;;
    esac
}

variant_solver() {
    variant_known "$1" || die "unknown variant '$1' (see $PERLMUTTER_TOOLS/bench/variants.py)"
    case "$1" in
        numba) echo "upstream-fbpic" ;;
        warp-*) echo "lpa/FBPIC_Warp" ;;
        *) die "variant '$1' has no solver source" ;;
    esac
}

# Steps per mode: N, 0 for the full interaction, or 1/K for a fraction of it.
mode_steps() {
    case "$1" in
        smoke) echo 100 ;;
        kernels | window) echo 1/10 ;;
        full | tests) echo 0 ;;
        *) die "unknown mode '$1' (smoke, kernels, window, full, tests)" ;;
    esac
}

utc_stamp() {
    date -u +%Y%m%dT%H%M%SZ
}
