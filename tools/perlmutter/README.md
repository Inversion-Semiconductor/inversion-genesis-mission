# Perlmutter FBPIC benchmark workflow

These scripts measure upstream FBPIC (numba) against FBPIC-Warp on one Perlmutter A100.
They record per-kernel GPU time with Nsight Systems and the end-to-end time of the producer case.

## Variants and branches

| Variant | Branch | Solver source | FBPIC options |
|---|---|---|---|
| `numba` | `snidhan/perl-main-run` | upstream fbpic `dc1cc32` | upstream defaults |
| `warp-fp64` | `snidhan/perl-warp-run` | `lpa/FBPIC_Warp` | graphs off, fused particle kernels, FP64 Hankel |
| `warp-fp32` | `snidhan/perl-warp-run` | `lpa/FBPIC_Warp` | graphs off, fused particle kernels, FP32 Hankel |

The variant table is in `bench/variants.py`.
Two cases are available (`--case`):

- `production`: the nominal point of `lpa/Exploration_FBPIC/runs/production_study/campaign.yaml` with its resolution profile (`minimum`: 2592 x 1080, Nm=5, gamma_boost=3). Kernel timing uses this case by default.
- `reduced`: `lpa/Simulation_FBPIC/runs/initial_sample/ionization_injection_runscript_00.py`. Other modes use this case by default.

The report starts with a table that compares the grid, particles per cell, physics, step count, particles, and GPU memory of each case.

All variants use ID-based ionization draws by default (`--rng shared`). Each draw is a Philox value of (seed 0, species stream, ion ID, iteration, level), so numba and Warp draw the same numbers for the same ion.
The numba variant uses upstream FBPIC `dc1cc32` plus this feature: `nv-snidhan/fbpic` branch `snidhan/ionization-rng`, commit `0b39054` (`UPSTREAM_FBPIC_URL` and `UPSTREAM_FBPIC_SHA` in the config). The report checks that ion IDs and ionization levels match numba bit for bit.
The Warp variants set linear deposition to 32 threads per block. Upstream FBPIC uses 16 on A100.
The reduced case uses Nz=1500, Nr=300, Nm=5, and gamma_boost=1.5. The producer computes the full step count at run time; for the reduced case it is 11,974 steps.

## How a run works

1. `launch.sh` creates a batch directory under `$FBPIC_RUNS`.
2. It copies `tools/perlmutter` into `<batch>/tools`. All runs in the batch use this copy.
3. It exports the source of each branch with `git archive` into `<batch>/src`. Later fetches do not change a queued batch.
4. Each run directory gets `meta/task.env`, `cache/`, `out/`, and `profile/`.
5. On the GPU node, `run_task.sh` activates the env and sets `PYTHONPATH` to the run's solver and producer sources.
6. `bench/check_node.py` tests numba CUDA, CuPy, Warp, and MPI. A failed check stops the run.
7. `bench/run_case.py` runs the unchanged producer and writes `result.json`.

`result.json` records these timings:

- `imports_s`, `setup_s`, `steps_s`, `analysis_s`, and `end_to_end_s`;
- `ms_per_step_after_warmup`, which excludes the first 200 steps (JIT compilation and module loads);
- per-segment ms/step every 100 steps;
- final particle counts and ionization-level histograms.

## Worktree layout

The scripts use the same layout as the workstation:

```text
~/fbpic/inversion-genesis-mission-worktrees/
  .bare/           bare clone; origin branches are refs/remotes/origin/*
  .git             the line "gitdir: ./.bare"
  perl-main-run/   worktree of snidhan/perl-main-run
  perl-warp-run/   worktree of snidhan/perl-warp-run (the tools run from here)
  main/            worktree of main
  dev-main/        worktree of snidhan/dev-main
~/fbpic/fbpic.git  bare mirror of upstream FBPIC
```

`FBPIC_WORKTREE_BRANCHES` in the config sets the list of worktrees.
`launch.sh` snapshots the committed state of a branch. It warns when that worktree has uncommitted `lpa/` changes.
To run uncommitted changes, add `--src-dir <worktree>`.

## One-time setup (login node)

1. Connect with agent forwarding, so that git can use your GitHub key: `ssh -A perlmutter.nersc.gov`.
2. Create the bare clone and the first worktree:

   ```sh
   mkdir -p ~/fbpic/inversion-genesis-mission-worktrees
   cd ~/fbpic/inversion-genesis-mission-worktrees
   GIT_LFS_SKIP_SMUDGE=1 git clone --bare \
     git@github.com:Inversion-Semiconductor/inversion-genesis-mission.git .bare
   echo "gitdir: ./.bare" > .git
   git config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
   git fetch origin
   GIT_LFS_SKIP_SMUDGE=1 git worktree add perl-warp-run snidhan/perl-warp-run
   git -C perl-warp-run branch --set-upstream-to=origin/snidhan/perl-warp-run
   ```

3. Copy the configuration file, then edit `NERSC_PROJECT` and `NERSC_ACCOUNT`:

   ```sh
   mkdir -p ~/.config
   cp perl-warp-run/tools/perlmutter/config.example.sh ~/.config/fbpic-perlmutter.sh
   ```

4. Define a shortcut. Add it to `~/.bashrc` to keep it:
   `export PM=~/fbpic/inversion-genesis-mission-worktrees/perl-warp-run/tools/perlmutter`.
5. Create the other worktrees and the upstream FBPIC mirror: `$PM/sync.sh`.
6. Check the Global Common quota: `showquota --cmn <project>`. The env needs about 6 GB of the 10 GB project quota.
7. Create the env: `$PM/setup_env.sh`. This step takes about 10 minutes.

## Each session

1. Update the worktrees: `$PM/sync.sh`.
   It fast-forwards each clean worktree to `origin/<branch>`.
   It does not change a worktree with uncommitted changes or local commits.

## Benchmark runs

1. Run a smoke test (100 steps per variant, 30 min debug QOS):

   ```sh
   $PM/launch.sh --mode smoke --submit --qos debug --time 00:30:00
   ```

2. Run the kernel profile (1/10 of the full steps under nsys, capture window 200:end):

   ```sh
   $PM/launch.sh --mode kernels --submit --time 02:00:00
   ```

3. Run the window timing (1/10 of the full steps, three repeats in rotated order, no profiler):

   ```sh
   $PM/launch.sh --mode window --repeats 3 --submit --time 04:00:00
   ```

4. Run the full case (all steps, one GPU per variant on one node):

   ```sh
   $PM/launch.sh --mode full --layout concurrent --submit --time 06:00:00
   ```

5. Make the report on a login node:

   ```sh
   module load conda && conda activate "$FBPIC_ENV"
   cd <batch>/tools && python -m bench.compare <batch> [<batch>...]
   ```

6. Copy the results to CFS: `$PM/archive.sh <batch> [<batch>...]`. Scratch is purged after 8 weeks without access.

Each `launch.sh` call prints the batch directory. `$FBPIC_RUNS/jobs.log` lists every submitted job.
For an interactive test, request a node with `salloc -A <account> -C gpu -q interactive -N 1 -t 01:00:00 --gpus-per-node=4`.
Then use `--here` instead of `--submit`.

## Options

- `--variant V`: run only some variants. Repeat the option for more than one.
- `--ref REF`: use one ref for every variant, for example a commit to test.
- `--src-dir DIR`: snapshot a working tree. `meta/` records the uncommitted changes.
- `--steps N|1/K`: change the step count. `0` runs the full interaction. `1/K` runs that fraction of it.
- `--profile-window A:B`: capture iterations A to B with nsys. `B` can be `end`.
- `--qos shared`: use one GPU of a shared node. Use `--layout sequential` with it.
- `--case production|reduced`: select the case. The default is `production` for `kernels` and `reduced` for other modes.
- `--rng shared|original`: select the ionization draws. The default is `shared`.
- `--constraint 'gpu&hbm80g'`: request 80 GB A100 nodes. Use it for the production case and for full runs. The reduced case uses about 32 GiB at 1,197 steps.
- `--upstream-src DIR`: use an upstream FBPIC working tree for numba, for example to test an uncommitted patch.
- `--stage-only`: create the batch and stop.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `node check failed`, numba CUDA errors or segmentation faults | numba 0.61 and the 580 driver | Make sure `FBPIC_MODULES` contains `numba-cuda-580-patch`. |
| `mpi` check fails | MPI library mismatch | Set `FBPIC_MPI="conda"` and rebuild with `setup_env.sh --recreate --mpi conda`. That is valid for one rank only. |
| `nsys not found` | The cudatoolkit module has no nsys in the job shell | Set `NSYS=/path/to/nsys` before `launch.sh`. |
| `fbpic imported from ...` | An `fbpic` package is installed in the env | Remove it: `pip uninstall fbpic`. |
| No `profile/kernels.json` | The kernel summary failed after the run | Run `python -m bench.kernels <run>` from `<batch>/tools` after `module load cudatoolkit`. |
| `use --ref or --src-dir` | Both options were given | Give only one. |

## Local test (workstation)

1. Create a config file with local paths. Set `FBPIC_WORKTREES` to the local worktree root and `UPSTREAM_FBPIC_GIT` to an upstream FBPIC clone.
2. Build the env with `setup_env.sh --local --mpi conda --prefix <dir>`.
3. Run `launch.sh --mode smoke --local --gpu 0`. Add `--src-dir <worktree>` to test uncommitted changes.
