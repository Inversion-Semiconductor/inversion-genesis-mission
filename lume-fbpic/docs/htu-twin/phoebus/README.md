# Phoebus displays for the LPA-fed HTU twin

Displays for the HTU transport twin when its source is a `lume-fbpic` run,
served by `python -m lume_fbpic_twin.serve --twin`. All PVs are `pva://HTU:SIM:...`.

| File | What it is |
|---|---|
| `htu_synoptic_lpa.bob` | The twin's synoptic with the LPA tied in: a bottom strip with the LPA run selector, the twin's source readbacks and whether the twin is simulating, and the `SRC` element opening `lpa_bunch.bob`. Generated; do not edit. |
| `lpa_bunch.bob` | The LPA bunch: run selector, the twin's `Source_*` readbacks, the 33 moment-descriptor scalars (covariance as a grid), run statistics. Generated; do not edit. |
| `generate_linked_synoptic.py` | Builds `htu_synoptic_lpa.bob` from the twin's own `htu_synoptic.bob`, which is left untouched. |
| `generate_lpa_display.py` | Builds `lpa_bunch.bob` from the descriptor action names. |
| `widgets.py` | The Phoebus widgets both generators build the displays from. |

## Run the server

See `../README.md` (`python -m lume_fbpic_twin.serve --twin`).

## Open the display

Start Phoebus with this display as the resource, for example from the Phoebus product directory:

```bash
java -jar target/product-<version>.jar -resource <this directory>/htu_synoptic_lpa.bob
```

Start the server first, and restart Phoebus after any server restart. Phoebus restores your last
session, so older windows may open too; use the one with the LPA strip along its bottom.

- **Twin activity:** at the right end of the strip, a LED and the state's name from
  `HTU:SIM:STATUS`: green and `Idle`, or amber and `Simulating` while the twin re-tracks (a few
  seconds after a put). The PV is lume-pva's, and read-only.
- **Choose a run:** the "Run:" combo box in the strip (or on `lpa_bunch.bob`) selects the active
  archive. The descriptor values and the twin's source follow after the twin re-tracks, in a few
  seconds.
- **`SRC` element:** opens `lpa_bunch.bob`.
- **Other elements:** the twin's own control and camera displays, found through the `TWIN_DISPLAYS`
  macro. It defaults to the twin's `display` directory, written relative to `htu_synoptic_lpa.bob`
  when the twin is in the same repository (as it is here), so the file works on any clone. To use
  another directory, set the macro when opening the display, with a `file:` URL (a plain path takes
  the `?` as part of the file name): `-resource "file:<this directory>/htu_synoptic_lpa.bob?TWIN_DISPLAYS=<twin display directory>"`.
  Both `<...>` are absolute paths.

## Regenerate the displays

```bash
python generate_lpa_display.py
python generate_linked_synoptic.py <twin display directory>
```

Run the second whenever the twin's synoptic changes. The twin's own files are only read.

## Things to know

- The `Source_*` PVs are read-only: the LPA run is the twin's source. They show the bunch's moments.
- The three run statistics (`charge_pc`, `energy_mean_mev`, `energy_std_mev`) read NaN for a
  reconstructed run: it has no particles to compute them from.
- Phoebus waits one second, on its UI thread, for a put reply; a server that acknowledged only after
  a re-track (a few seconds) made every write time out. That is why puts are acknowledged at once by
  default.
- The tests in `tests/test_phoebus_displays.py` check the generators and that the committed
  `lpa_bunch.bob` is up to date. `htu_synoptic_lpa.bob` is built from the twin's own synoptic, so
  regenerate it whenever that changes.
- The twin's `generate_synoptic.py` writes `<rotation_step>90.0</rotation_step>` for its rotated
  elements, but Phoebus's `rotation_step` is an enum (0 none, 1 = 90 degrees, 2 = 180, 3 = 270),
  not degrees, so those elements are not rotated. The fix belongs in the twin, in
  `geecs-lume-twin/display/generate_synoptic.py`: write `1`, then regenerate its `htu_synoptic.bob`.
  `htu_synoptic_lpa.bob` copies the value, so regenerate it too after that fix.
- A macro that a display file defines itself overrides the one given when opening it, so the two
  displays do not define `TWIN_DISPLAYS` and `P`: each reference to them has the default in it
  (`$(P=HTU:SIM:)`).
