#!/usr/bin/env bash
# Create or update the bare clones that launch.sh snapshots runs from.
#
# Usage: sync.sh
# The repository is private: connect with "ssh -A" so that git can use your agent.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

export GIT_LFS_SKIP_SMUDGE=1
mkdir -p "$FBPIC_SRC"

sync_bare() {
    local url="$1" dir="$2"
    if [[ ! -d "$dir" ]]; then
        log "cloning $url"
        git clone --bare "$url" "$dir"
        git -C "$dir" config remote.origin.fetch '+refs/heads/*:refs/heads/*'
    fi
    log "fetching $(basename "$dir")"
    git -C "$dir" fetch --prune origin
}

sync_bare "$FBPIC_REPO_URL" "$REPO_GIT"
sync_bare "$UPSTREAM_FBPIC_URL" "$UPSTREAM_GIT"

git -C "$UPSTREAM_GIT" cat-file -e "$UPSTREAM_FBPIC_SHA^{commit}" ||
    die "upstream commit $UPSTREAM_FBPIC_SHA is missing from $UPSTREAM_FBPIC_URL"

for variant in numba warp-fp64; do
    branch="$(variant_branch "$variant")"
    sha="$(git -C "$REPO_GIT" rev-parse --verify --quiet "$branch^{commit}")" ||
        die "branch $branch is missing; push it first"
    printf '%-24s %s\n' "$branch" "$sha"
done
printf '%-24s %s\n' "upstream fbpic" "$UPSTREAM_FBPIC_SHA"

# Move the detached launcher checkout (see README) to the new harness commit.
launcher="$FBPIC_SRC/launcher"
if [[ -d "$launcher" ]]; then
    if [[ -n "$(git -C "$launcher" status --porcelain)" ]]; then
        log "launcher checkout has local changes; not moving it"
    else
        git -C "$launcher" checkout --quiet --detach "$(variant_branch warp-fp64)"
        log "launcher at $(git -C "$launcher" rev-parse --short HEAD)"
    fi
fi
