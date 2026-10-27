# HTU twin demo

Runs the HTU transport twin (`htu`) with a `lume-fbpic` run as its source: the LPA bunch goes in
at the start of the line and the twin tracks it through the magnets and screens. The twin and the
LPA archives can be served as EPICS PVs and watched in Phoebus; choosing another archive changes
the twin's source.

## How the pieces fit

An LPA run is an archive (`LUMEFBPICModel.archive()`): the config, the action values and, if saved,
the final particles. The package `lume_fbpic_twin` (in this directory, with its own
`pyproject.toml`) holds the rest: `twin_stage` wraps the twin as a `StagedModel` stage that takes
those final particles as its source beam, and `serve` loads one or more archives, picks the active
one through `archive_selector`, and serves the archive's outputs and the twin's variables as PVs.
The displays in `phoebus/` are laid out for those PVs.

## Files

**`lume_fbpic_twin/`**

- `twin_stage.py`: `TwinStage` wraps the twin's model so that `StagedModel` can hand it the LPA stage's
  `final_particles`. It selects the particles (`uz_min`, `central_fraction`), drifts the bunch
  from the simulation's lab-frame snapshot onto one plane (its mean `z`, or the twin's source
  plane), replaces the twin's source beam, and makes the twin's `Source_*` PVs read-only
  readbacks of the bunch's moments. `build_chain()` returns
  `StagedModel([lpa_model, TwinStage(twin_model)])`.
- `bunch_frame.py`: the selection and the ballistic move onto one plane
  (`ParticleGroup.drift_to_z()`), shared by `TwinStage` and the synthetic bunch.
- `serve.py` (`python -m lume_fbpic_twin.serve`): serves archives as EPICS PVs through `lume-pva`.
  It loads each archive without running anything (`dummy_run=True`) and serves the model's
  variables as `<prefix><name>`: the outputs, and the inputs read-only. With `--twin` it also
  builds the chain and serves the twin's variables, which re-track on each put.
- `archive_selector.py`: `ArchiveSelector` holds several archives and presents the active one as a single
  LPA model, with one writable enum PV (`LPA_Archive`) to switch. A switch reaches the twin like
  a new LPA result. For archives without particles it can build an approximate bunch from the
  moment descriptor.

**`twin/`**

- `lpa_to_twin.py`: a script that runs that chain once, from an archive with the final particles,
  and prints what the twin received and the fraction of the charge that reaches each
  screen.
- `initial_sample_archives.py`: turns the seven runs of the NERSC `initial_sample` dataset into
  archives. They hold the recorded 33 moment-descriptor values and no particles, so the twin gets a
  synthesized bunch from them. Each also holds its dataset record (run name, inputs, outputs and
  descriptor selection, as recorded) as `Parameters`, in `model.simulator.parameters`.

**`phoebus/`**

- Phoebus displays for the twin when it is fed by an LPA run: the twin's synoptic with an LPA strip
  and run selector, and `lpa_bunch.bob` showing the bunch, the 33 descriptor values and the run
  statistics. Two scripts generate them, and the `.bob` files are not in the repository: generate
  them before opening Phoebus (see "Open the display in Phoebus", and the directory's README).

## Install

You need Python 3.11 to 3.13, git, an MPI library (`fbpic` needs `mpi4py`) and a Phoebus 5.x build.
Run these from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e lpa/Simulation_FBPIC      # inversion_fbpic, with fbpic
pip install -e lume-fbpic                # lume_fbpic
pip install -e lume-fbpic/docs/htu-twin  # lume_fbpic_twin: the twin stage, selector and server
pip install -e geecs-lume-twin           # the twin (htu), with Cheetah, lume-pva and p4p
```

The twin installs the `lume-pva` commit it pins (from GitHub), replacing any other `lume-pva` in the
environment.

No EPICS installation is needed. The server is itself the EPICS server: `lume-pva` serves PV Access
through `p4p`, and both `p4p` and `pcaspy` install from pip with the EPICS libraries built in.
Phoebus has its own PV Access client. EPICS base is needed only for its command-line tools, such
as `pvget` (see "Check the PVs").

## Run the twin

From the repository root, with the environment active:

```bash
cd lume-fbpic
python docs/htu-twin/twin/initial_sample_archives.py      # writes ./initial_sample_archives
python -m lume_fbpic_twin.serve initial_sample_archives --twin --synthesize-bunch
```

Wait for `serving N variables` (a few seconds) and leave the terminal open. The first read of a
PV can show 0 for about a second, until the server's first update.

The writable LPA inputs (`laser_energy`, the Zernike coefficients, ...) are served read-only, so
every variable is in the served config: the `lume-pva` the twin pins fails with
`KeyError: 'laser_energy'` if a variable of the model is missing from it.

### Check the PVs (optional)

This step is not needed to run the twin or the displays. From another terminal, with the
environment active, `p4p` (installed with the twin) reads the PVs:

```bash
python -c "from p4p.client.thread import Context; ctx = Context('pva'); [print(name, ctx.get(name)) for name in ('HTU:SIM:LPA_Archive', 'HTU:SIM:STATUS', 'HTU:SIM:Source_Charge_pC')]"
```

On a machine with EPICS base installed, its command-line tools do the same:

```bash
pvget HTU:SIM:LPA_Archive HTU:SIM:STATUS HTU:SIM:Source_Charge_pC
```

`pvget` is part of EPICS base, not of this package or the twin, so it is only there on a system
with EPICS installed.

Expect `sim_0000`, `Idle` and the charge of `sim_0000`'s bunch. `HTU:SIM:STATUS` is `Idle` or `Simulating`, and is
`Simulating` while the twin re-tracks. If your Phoebus settings pin `epics_pva_addr_list`, add
`127.0.0.1` to that list.

### Open the display in Phoebus

The displays are generated, not in the repository. Write them first, from the repository root:

```bash
cd lume-fbpic/docs/htu-twin/phoebus
python generate_lpa_display.py
python generate_linked_synoptic.py ../../../../geecs-lume-twin/display
```

Then start Phoebus from its product directory, with an absolute path to the display:

```bash
java -jar target/product-<version>.jar -resource <repository>/lume-fbpic/docs/htu-twin/phoebus/htu_synoptic_lpa.bob
```

Start the server first, and restart Phoebus after a server restart. Phoebus restores its last
session, so older windows may open too: use the one with the LPA strip along the bottom.

- The "Run:" combo box in the strip selects the active archive. The values and the twin's source
  follow after the twin re-tracks, in a few seconds.
- The "Twin:" LED shows `Idle` (green) or `Simulating` (amber).
- The `SRC` element opens `lpa_bunch.bob`. The other elements are the twin's own controls and
  camera views.

`phoebus/README.md` has the details: the display macros, regenerating the displays, and notes.

To stop, press Ctrl-C in the server terminal and close Phoebus.

### Options

- Archives are files or directories of `.h5` files; their file names (without `.h5`) are the options
  of the `HTU:SIM:LPA_Archive` selector PV. The first one is active at start.
- `--synthesize-bunch` builds an approximate bunch from each archive's recorded moment descriptor.
  It is needed for archives without final particles (such as the reconstructed `initial_sample`
  runs); leave it out for archives saved with `save_final_particles=True`.
- The server is a demo, so the rest is fixed: the PV prefix is `HTU:SIM:` with `--twin` (the
  twin's own, which its displays use), the camera images are 4 times coarser than the twin's
  1024 x 1024, a synthetic bunch has 20000 macroparticles, and a put is acknowledged at once.
  These are constants in `serve.py`.
- Without `--twin`, the server serves just the archives' variables, with the prefix `LPA:SIM:`:
  `python -m lume_fbpic_twin.serve run.h5`.

### Run the chain once

To run the chain once without a server, from an archive saved with `save_final_particles=True`:

```bash
python docs/htu-twin/twin/lpa_to_twin.py run.h5
```

## Limits

- The reconstructed `initial_sample` archives hold no particles, so their bunch is synthesized from
  the moment descriptor: an approximation, not the real bunch.
- Served values are recorded results; nothing re-runs the simulation when an input changes.
- The twin's magnets keep the settings they have for a 100 MeV beam, so a bunch of very different
  energy or large energy spread is mostly lost. The demo shows the hand-off, not matched optics.
