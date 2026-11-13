# One-Off Campaign Candidates

This workflow runs a manually selected active-parameter vector through the same
Stage 2 campaign executor used by exploration. Each run writes the normal
manifest, generated `run_fbpic.py`, native `result.json`, shifted
`results_shifted.json`, and an isolated shifted `campaign_runs.parquet` under
`${PSCRATCH}/ionization_injection_exploration_evaluations`.

The submitted candidate is [candidate_mock_stage2_016.json](candidate_mock_stage2_016.json).
To submit it on Perlmutter:

```bash
cd /global/u1/c/cedoss/inversion-genesis-mission/lpa/Exploration_FBPIC/runs/initial_study/one_offs
sbatch submit_mock_stage2_016_4gpu_nersc.sh
```

The submission script resolves its workflow files from `SLURM_SUBMIT_DIR`, so
submit it from this `one_offs` directory rather than an absolute script path from
elsewhere.

The script requests one node, four MPI tasks, and four GPUs for a two-hour
`regular` QoS allocation. If Slurm reports a shorter effective limit, inspect the
submitted job rather than FBPIC output:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,Timelimit,ExitCode
```

Set a distinct label before submission to retain another run:

```bash
ONE_OFF_RUN_ID=my_candidate sbatch submit_mock_stage2_016_4gpu_nersc.sh
```

## Sim 728 validation scans

The following candidates preserve every active `sim_000728` value except the
named scan coordinate:

| Candidate | Changed coordinate |
| --- | --- |
| [candidate_728_energy_1p7J.json](candidate_728_energy_1p7J.json) | laser energy = 1.7 J |
| [candidate_728_energy_2p1J.json](candidate_728_energy_2p1J.json) | laser energy = 2.1 J |
| [candidate_728_focus_3p5mm.json](candidate_728_focus_3p5mm.json) | focal position = 3.5 mm |
| [candidate_728_focus_2p0mm.json](candidate_728_focus_2p0mm.json) | focal position = 2.0 mm |

Submit one candidate by setting both variables:

```bash
ONE_OFF_CANDIDATE="$PWD/candidate_728_energy_1p7J.json" \
ONE_OFF_RUN_ID="scan_728_energy_1p7J" \
sbatch submit_mock_stage2_016_4gpu_nersc.sh
```

The campaign validates every candidate against configured Stage 2 bounds before
writing the run directory.