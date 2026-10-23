#!/usr/bin/env bash
# Stage and run one benchmark batch: variants x repeats for one mode.
#
# Usage:
#   launch.sh --mode {smoke,kernels,window,full} [options] {--submit|--here|--local|--stage-only}
#
# Options:
#   --variant V          numba, warp-fp64, or warp-fp32; repeat for several (default: all three)
#   --ref REF            git ref for every variant (default: the variant's perl-* branch)
#   --src-dir DIR        snapshot a working tree instead of a ref (uncommitted changes are recorded)
#   --upstream-src DIR   snapshot an upstream FBPIC working tree for numba instead of UPSTREAM_FBPIC_SHA
#   --steps N|1/K        steps per run; 0 is the full interaction, 1/K a fraction of it
#                        (default: smoke 100, kernels and window 1/10, full 0)
#   --repeats K          repeats per variant; sequential order rotates per repeat (default: 1)
#   --case C             reduced (runscript_00; default, fits one A100-40GB) or production
#                        (campaign nominal + resolution profile; needs more GPU memory)
#   --rng R              shared (ID-based draws, identical in both solvers; default) or original
#   --profile-window A:B nsys capture steps; B may be 'end'; negative A counts from the end
#                        (default for kernels: -300:end, the last 300 steps)
#   --layout L           sequential (default) or concurrent (one GPU per run, up to 4)
#   --qos Q --time T     Slurm QOS and time limit for --submit (default: from the config)
#   --constraint C       Slurm constraint (default: gpu; use 'gpu&hbm80g' for 80 GB A100s)
#   --gpu N              CUDA device for --local (default: 0)
#
# Execution:
#   --submit      write <batch>/job.sbatch and submit it with sbatch
#   --here        run inside the current salloc allocation
#   --local       run on this machine without Slurm or modules (workstation tests)
#   --stage-only  create the batch directory and stop
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

mode=""
variants=()
ref=""
src_dir=""
upstream_src=""
steps=""
repeats=1
window=""
case_name=""
rng="shared"
constraint="gpu"
layout="sequential"
qos="${FBPIC_QOS:-regular}"
time_limit="${FBPIC_TIME:-01:00:00}"
gpu=0
action=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) mode="$2"; shift 2 ;;
        --variant) variants+=("$2"); shift 2 ;;
        --ref) ref="$2"; shift 2 ;;
        --src-dir) src_dir="$(cd "$2" && pwd)"; shift 2 ;;
        --upstream-src) upstream_src="$(cd "$2" && pwd)"; shift 2 ;;
        --steps) steps="$2"; shift 2 ;;
        --repeats) repeats="$2"; shift 2 ;;
        --profile-window) window="$2"; shift 2 ;;
        --case) case_name="$2"; shift 2 ;;
        --rng) rng="$2"; shift 2 ;;
        --constraint) constraint="$2"; shift 2 ;;
        --layout) layout="$2"; shift 2 ;;
        --qos) qos="$2"; shift 2 ;;
        --time) time_limit="$2"; shift 2 ;;
        --gpu) gpu="$2"; shift 2 ;;
        --submit | --here | --local | --stage-only) action="${1#--}"; shift ;;
        -h | --help) sed -n '2,25p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done
[[ -n "$mode" ]] || die "--mode is required"
[[ -n "$action" ]] || die "choose one of --submit, --here, --local, --stage-only"
[[ ${#variants[@]} -gt 0 ]] || variants=(numba warp-fp64 warp-fp32)
[[ -n "$steps" ]] || steps="$(mode_steps "$mode")"
[[ "$steps" =~ ^([0-9]+|1/[1-9][0-9]*)$ ]] || die "--steps must be N or 1/K"
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || die "--repeats must be a positive integer"
[[ "$layout" == sequential || "$layout" == concurrent ]] || die "--layout must be sequential or concurrent"
[[ -z "$ref" || -z "$src_dir" ]] || die "use --ref or --src-dir, not both"
[[ -n "$case_name" ]] || case_name=reduced
[[ "$case_name" == production || "$case_name" == reduced ]] || die "--case must be production or reduced"
[[ "$rng" == shared || "$rng" == original ]] || die "--rng must be shared or original"
if [[ "$mode" == kernels && -z "$window" ]]; then
    window="-300:end"
fi
if [[ -n "$window" ]]; then
    [[ "$window" =~ ^(-?[0-9]+):([0-9]+|end)$ ]] || die "--profile-window must be A:B or A:end"
    [[ "${BASH_REMATCH[2]}" == end || "${BASH_REMATCH[1]}" -lt "${BASH_REMATCH[2]}" ]] ||
        die "--profile-window needs A < B"
fi
if [[ "$qos" == shared && "$layout" == concurrent ]]; then
    die "the shared QOS gives one GPU; use --layout sequential"
fi
if [[ "$action" == here ]]; then
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "--here needs an salloc allocation"
fi
for variant in "${variants[@]}"; do variant_solver "$variant" >/dev/null; done

# Two launches in the same second must not share a batch: mkdir fails if it exists.
mkdir -p "$FBPIC_RUNS"
batch="$FBPIC_RUNS/$(utc_stamp)_${mode}_${case_name}"
suffix=1
until mkdir "$batch" 2>/dev/null; do
    batch="$FBPIC_RUNS/$(utc_stamp)_${mode}_${case_name}_$suffix"
    suffix=$((suffix + 1))
    [[ $suffix -le 100 ]] || die "cannot create a unique batch directory under $FBPIC_RUNS"
done
mkdir -p "$batch"/{src,meta}
log "batch $batch"

# Freeze the harness: every run in the batch uses this copy of tools/perlmutter.
mkdir -p "$batch/tools"
(cd "$PERLMUTTER_TOOLS" && tar --exclude=__pycache__ -cf - .) | tar -xf - -C "$batch/tools"
cp "$FBPIC_PERLMUTTER_CONFIG" "$batch/config.sh"
{
    echo "harness_dir=$PERLMUTTER_TOOLS"
    git -C "$PERLMUTTER_TOOLS" rev-parse HEAD 2>/dev/null | sed 's/^/harness_commit=/' || true
    echo "harness_dirty=$(git -C "$PERLMUTTER_TOOLS" status --porcelain -- . 2>/dev/null | wc -l)"
} >"$batch/meta/harness.txt"

# Snapshot one source tree per commit (or per working tree) and print its directory.
# These run in $(...) subshells, so the directory itself marks a finished snapshot.
snapshot_ref() {
    local sha="$1"
    local dest="$batch/src/${sha:0:12}"
    if [[ ! -d "$dest" ]]; then
        mkdir -p "$dest.tmp"
        local paths=(lpa/Simulation_FBPIC lpa/Exploration_FBPIC)
        if git -C "$REPO_GIT" cat-file -e "$sha:lpa/FBPIC_Warp/fbpic/__init__.py" 2>/dev/null; then
            paths+=(lpa/FBPIC_Warp)
        fi
        git -C "$REPO_GIT" archive "$sha" -- "${paths[@]}" | tar -xf - -C "$dest.tmp"
        mv "$dest.tmp" "$dest"
    fi
    echo "$dest"
}

snapshot_worktree() {
    local dest="$batch/src/worktree"
    if [[ ! -d "$dest" ]]; then
        mkdir -p "$dest.tmp"
        local paths=(lpa/Simulation_FBPIC lpa/Exploration_FBPIC)
        if [[ -f "$src_dir/lpa/FBPIC_Warp/fbpic/__init__.py" ]]; then
            paths+=(lpa/FBPIC_Warp)
        fi
        (cd "$src_dir" && git ls-files -z -co --exclude-standard -- "${paths[@]}" |
            tar --null -T - -cf -) | tar -xf - -C "$dest.tmp"
        mv "$dest.tmp" "$dest"
        git -C "$src_dir" status --short -- "${paths[@]}" >"$batch/meta/worktree_status.txt"
        git -C "$src_dir" diff HEAD -- "${paths[@]}" >"$batch/meta/worktree.diff"
    fi
    echo "$dest"
}

snapshot_upstream() {
    local dest
    if [[ -n "$upstream_src" ]]; then
        dest="$batch/src/upstream-fbpic-worktree"
        if [[ ! -d "$dest" ]]; then
            mkdir -p "$dest.tmp"
            (cd "$upstream_src" && git ls-files -z -co --exclude-standard -- fbpic |
                tar --null -T - -cf -) | tar -xf - -C "$dest.tmp"
            git -C "$upstream_src" rev-parse HEAD >"$batch/meta/upstream_worktree_head.txt"
            git -C "$upstream_src" status --short -- fbpic >"$batch/meta/upstream_worktree_status.txt"
            git -C "$upstream_src" diff HEAD -- fbpic >"$batch/meta/upstream_worktree.diff"
            mv "$dest.tmp" "$dest"
        fi
    else
        dest="$batch/src/upstream-fbpic-${UPSTREAM_FBPIC_SHA:0:12}"
        if [[ ! -d "$dest" ]]; then
            mkdir -p "$dest.tmp"
            git -C "$UPSTREAM_GIT" archive "$UPSTREAM_FBPIC_SHA" -- fbpic | tar -xf - -C "$dest.tmp"
            mv "$dest.tmp" "$dest"
        fi
    fi
    echo "$dest"
}

stage_run() {
    local variant="$1" repeat="$2"
    local run="$batch/${variant}_r${repeat}"
    local source_ref source_sha source_dirty tree
    if [[ -n "$src_dir" ]]; then
        source_ref="worktree:$src_dir"
        source_sha="$(git -C "$src_dir" rev-parse HEAD)"
        source_dirty="$(git -C "$src_dir" status --porcelain -- lpa | wc -l)"
        tree="$(snapshot_worktree)"
    else
        source_ref="${ref:-$(variant_branch "$variant")}"
        source_sha="$(resolve_ref "$source_ref")" ||
            die "unknown ref $source_ref in $REPO_GIT (run sync.sh)"
        local checkout
        checkout="$(worktree_dir "$source_ref")"
        if [[ -d "$checkout" && -n "$(git -C "$checkout" status --porcelain -- lpa 2>/dev/null)" ]]; then
            log "warning: $checkout has uncommitted lpa/ changes; the run uses commit ${source_sha:0:12}" \
                "(use --src-dir $checkout to include them)"
        fi
        source_dirty=0
        tree="$(snapshot_ref "$source_sha")"
    fi
    local solver
    if [[ "$(variant_solver "$variant")" == upstream-fbpic ]]; then
        solver="$(snapshot_upstream)"
    else
        solver="$tree/lpa/FBPIC_Warp"
        [[ -f "$solver/fbpic/__init__.py" ]] ||
            die "$variant needs lpa/FBPIC_Warp, which $source_ref lacks"
    fi
    mkdir -p "$run"/{meta,cache,out,profile}
    {
        printf 'VARIANT=%q\n' "$variant"
        printf 'MODE=%q\n' "$mode"
        printf 'REPEAT=%q\n' "$repeat"
        printf 'STEPS=%q\n' "$steps"
        printf 'PROFILE_WINDOW=%q\n' "$window"
        printf 'SOLVER_DIR=%q\n' "$solver"
        printf 'PRODUCER_DIR=%q\n' "$tree/lpa/Simulation_FBPIC"
        printf 'EXPLORATION_DIR=%q\n' "$tree/lpa/Exploration_FBPIC"
        printf 'CASE=%q\n' "$case_name"
        printf 'RNG=%q\n' "$rng"
        printf 'SOURCE_REF=%q\n' "$source_ref"
        printf 'SOURCE_SHA=%q\n' "$source_sha"
        printf 'SOURCE_DIRTY=%q\n' "$source_dirty"
        printf 'UPSTREAM_FBPIC_SHA=%q\n' "${upstream_src:+worktree:$upstream_src@}$UPSTREAM_FBPIC_SHA"
    } >"$run/meta/task.env"
    echo "$run"
}

: >"$batch/tasks.txt"
for ((r = 1; r <= repeats; r++)); do
    order=("${variants[@]}")
    if [[ "$layout" == sequential ]]; then
        shift_by=$(((r - 1) % ${#variants[@]}))
        order=("${variants[@]:shift_by}" "${variants[@]:0:shift_by}")
    fi
    for variant in "${order[@]}"; do
        stage_run "$variant" "$r" >>"$batch/tasks.txt"
    done
done

{
    printf 'MODE=%q\n' "$mode"
    printf 'LAYOUT=%q\n' "$layout"
    printf 'ACTION=%q\n' "$action"
    printf 'LOCAL_GPU=%q\n' "$gpu"
} >"$batch/batch.env"
printf '%q ' "$0" "$@" >"$batch/meta/launch_command.txt"
echo >>"$batch/meta/launch_command.txt"
[[ -f "$FBPIC_ENV.lock.txt" ]] && sha256sum "$FBPIC_ENV.lock.txt" >"$batch/meta/env_lock.sha256"
log "staged $(wc -l <"$batch/tasks.txt") runs:"
sed 's/^/  /' "$batch/tasks.txt" >&2

case "$action" in
    stage-only)
        echo "$batch"
        ;;
    local | here)
        FBPIC_PERLMUTTER_CONFIG="$batch/config.sh" bash "$batch/tools/run_batch.sh" "$batch"
        echo "$batch"
        ;;
    submit)
        if [[ "$qos" == shared ]]; then
            resources=$'#SBATCH -n 1\n#SBATCH -c 32\n#SBATCH --gpus-per-task=1'
        else
            resources=$'#SBATCH -N 1\n#SBATCH --ntasks-per-node=4\n#SBATCH -c 32\n#SBATCH --gpus-per-node=4'
        fi
        cat >"$batch/job.sbatch" <<EOF
#!/bin/bash
#SBATCH -J fbpic-$mode
#SBATCH -A $NERSC_ACCOUNT
#SBATCH -C $constraint
#SBATCH -q $qos
#SBATCH -t $time_limit
$resources
#SBATCH -o $batch/slurm-%j.out
set -euo pipefail
export FBPIC_PERLMUTTER_CONFIG="$batch/config.sh"
exec bash "$batch/tools/run_batch.sh" "$batch"
EOF
        job="$(sbatch --parsable "$batch/job.sbatch")"
        echo "$job" >"$batch/jobid"
        echo "$(utc_stamp) $job $batch" >>"$FBPIC_RUNS/jobs.log"
        log "submitted job $job"
        echo "$batch"
        ;;
esac
