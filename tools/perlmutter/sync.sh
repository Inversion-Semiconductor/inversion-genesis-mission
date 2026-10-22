#!/usr/bin/env bash
# Create or update the worktree layout that launch.sh snapshots runs from.
#
# Usage: sync.sh
#
# Layout (same as the workstation):
#   $FBPIC_WORKTREES/.bare     bare clone; origin branches are refs/remotes/origin/*
#   $FBPIC_WORKTREES/.git      the line "gitdir: ./.bare"
#   $FBPIC_WORKTREES/<name>/   one worktree per branch in FBPIC_WORKTREE_BRANCHES
#
# A clean worktree is fast-forwarded to origin/<branch>. A worktree with local
# changes or local commits is left as it is, with a warning.
# The repository is private: connect with "ssh -A" so that git can use your agent.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

export GIT_LFS_SKIP_SMUDGE=1
mkdir -p "$FBPIC_WORKTREES"

if [[ ! -d "$REPO_GIT" ]]; then
    log "cloning $FBPIC_REPO_URL into $REPO_GIT"
    git clone --bare "$FBPIC_REPO_URL" "$REPO_GIT"
fi
[[ -e "$FBPIC_WORKTREES/.git" ]] || echo "gitdir: ./.bare" >"$FBPIC_WORKTREES/.git"
git -C "$REPO_GIT" config --replace-all remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
log "fetching origin"
git -C "$REPO_GIT" fetch --prune origin
git -C "$REPO_GIT" worktree prune

for branch in ${FBPIC_WORKTREE_BRANCHES:-snidhan/perl-main-run snidhan/perl-warp-run}; do
    remote="refs/remotes/origin/$branch"
    dir="$(worktree_dir "$branch")"
    if ! git -C "$REPO_GIT" rev-parse --verify --quiet "$remote" >/dev/null; then
        log "warning: origin has no branch $branch; skipping"
        continue
    fi
    if [[ ! -d "$dir" ]]; then
        if git -C "$REPO_GIT" rev-parse --verify --quiet "refs/heads/$branch" >/dev/null; then
            git -C "$REPO_GIT" worktree add "$dir" "$branch"
            git -C "$dir" branch --quiet --set-upstream-to="origin/$branch"
        else
            git -C "$REPO_GIT" worktree add --track -b "$branch" "$dir" "origin/$branch"
        fi
    fi
    if [[ -n "$(git -C "$dir" status --porcelain)" ]]; then
        log "warning: $dir has uncommitted changes; not updating it"
    elif ! git -C "$dir" merge --quiet --ff-only "origin/$branch" 2>/dev/null; then
        log "warning: $dir has local commits that origin lacks; not updating it"
    fi
    printf '%-24s %s  %s\n' "$branch" "$(git -C "$dir" rev-parse --short=12 HEAD)" "$dir"
done

if [[ ! -d "$UPSTREAM_GIT" ]]; then
    log "cloning $UPSTREAM_FBPIC_URL into $UPSTREAM_GIT"
    mkdir -p "$(dirname "$UPSTREAM_GIT")"
    git clone --bare "$UPSTREAM_FBPIC_URL" "$UPSTREAM_GIT"
fi
git -C "$UPSTREAM_GIT" fetch --quiet origin '+refs/heads/*:refs/heads/*'
git -C "$UPSTREAM_GIT" cat-file -e "$UPSTREAM_FBPIC_SHA^{commit}" ||
    die "upstream commit $UPSTREAM_FBPIC_SHA is missing from $UPSTREAM_FBPIC_URL"
printf '%-24s %s  %s\n' "upstream fbpic" "${UPSTREAM_FBPIC_SHA:0:12}" "$UPSTREAM_GIT"

for variant in numba warp-fp64; do
    branch="$(variant_branch "$variant")"
    resolve_ref "$branch" >/dev/null || die "branch $branch is missing; push it first"
done
