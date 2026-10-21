"""Serve an archived `LUMEFBPICModel` as EPICS PVs through `lume-pva`.

`python -m lume_fbpic_twin.serve <archive.h5 | directory> ...` loads the archives
(`LUMEFBPICModel.from_archive()`; a directory means its `*.h5` files) and serves every read-only
scalar action -- the stats and the moment-descriptor features -- over PV Access, each named
`<prefix><action name>`; `lume-pva` adds `<prefix>RESET` and `<prefix>SNAPSHOT`.

The archives are held by an `ArchiveSelector`: one writable enum PV, `<prefix>LPA_Archive`, lists
them by file name (without `.h5`) and putting one makes it the active run. The first is active at
start. The served values and, with `--twin`, the bunch the twin tracks follow the active run. They
must all have the same variables (be made with the same action list). The values are the
archive's: the live results when it holds the final particles, otherwise the recorded output
values (as in the archives made by `docs/htu-twin/twin/initial_sample_archives.py`).
`final_particles` is not served, as `lume-pva` has no handler for a particle group.

The model is always loaded with `dummy_run=True`, so nothing ever starts a simulation. The input
actions (laser energy, ...) are served read-only.

With `--twin` the archive's final particles are the source of the HTU transport twin
(`lume_fbpic_twin.twin_stage.build_chain()`): the twin's own variables (magnets, steering,
chicane, slit, magspec, ...) are served read-write and re-track on every put, its `Source_*` variables read the
LPA bunch read-only, and the archive's outputs are served read-only next to them. The archive must
hold the final particles (`save_final_particles=True`) unless `--synthesize-bunch` is given (below).
`htu` (geecs-lume-twin) must be importable, and the default prefix becomes `HTU:SIM:`, the one the
twin's Phoebus displays use.

A put is acknowledged at once and the served values update when the twin has re-tracked (a few
seconds), because Phoebus waits one second for a put reply on its UI thread and would otherwise report
a write error and freeze. The twin's camera images are made `SCREEN_BINNING` times coarser: the
bunch has far fewer macroparticles than the twin's 1024 x 1024 pixels.

An archive without final particles (one reconstructed from a dataset, as
`docs/htu-twin/twin/initial_sample_archives.py` writes) cannot feed the twin on its own. With
`--synthesize-bunch` a bunch of `SYNTHETIC_BUNCH_PARTICLES` macroparticles is built from its
recorded moment descriptor: an APPROXIMATION of the real bunch, matching the descriptor's means,
variances and main correlations, not its tails. The served descriptor values stay the recorded ones.

Needs `lume-pva` (the `serve` extra of this package); it is imported only when serving, after the
arguments have been checked.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import typing

from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic_twin.archive_selector import DEFAULT_SELECTOR_NAME, ArchiveSelector

# The PV prefix of the archives served alone.
LPA_PREFIX = "LPA:SIM:"
# The PV prefix with `--twin`. POSIT: the twin's own, `htu.config_tool.DEFAULT_PREFIX`, which its
# Phoebus displays use.
TWIN_PREFIX = "HTU:SIM:"
# How many times coarser the twin's camera images are served, in each direction: 1024 x 1024
# becomes 256 x 256 over the same field of view.
SCREEN_BINNING = 4
# The macroparticles of a bunch synthesized from a moment descriptor.
SYNTHETIC_BUNCH_PARTICLES = 20_000


def build_config(
    model,
    *,
    prefix: str = LPA_PREFIX,
    serve_always: set[str] | None = None,
) -> dict[str, typing.Any]:
    """The `lume_pva.runner.Runner` config for `model`: every variable of the model, with the
    writable ones served read-only, apart from those named in `serve_always`, which are served
    in their natural mode. A put is acknowledged at once (`PutMode.Immediate`), and the model is
    served over PV Access only.

    The config holds every variable of the model: the `Runner` of the `lume-pva` the twin pins
    looks up each of them in it."""
    runner = _import_runner()
    config = runner.Runner.generate_config(
        model,
        prefix=prefix,
        put_mode=runner.PutMode.Immediate,
    )
    always = serve_always or set()
    config["variables"] = {
        name: (
            entry
            if name in always or str(entry["mode"]) == "ro"
            else {**entry, "mode": type(entry["mode"]).RO}
        )
        for name, entry in config["variables"].items()
    }
    config["protocol"] = ["pva"]
    return config


def expand_archives(paths: list[Path]) -> list[Path]:
    """The archive files named by `paths`: a file stays as it is, a directory becomes its `*.h5`
    files in name order.

    Raises:
        ValueError: If a path does not exist, or a directory holds no `.h5` file.
    """
    archives: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            found = sorted(path.glob("*.h5"))
            if not found:
                raise ValueError(f"no .h5 archives in {path}")
            archives.extend(found)
        elif path.is_file():
            archives.append(path)
        else:
            raise ValueError(f"no such archive or directory: {path}")
    return archives


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "archives",
        nargs="+",
        type=Path,
        help="archives written by LUMEFBPICModel.archive(), or directories of them",
    )
    parser.add_argument(
        "--twin",
        action="store_true",
        help="feed the archive's final particles to the HTU twin and serve the twin as well",
    )
    parser.add_argument(
        "--synthesize-bunch",
        action="store_true",
        help="with --twin and no final particles in the archive, build an approximate bunch from "
        "the recorded moment descriptor",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    if args.synthesize_bunch and not args.twin:
        parser.error("--synthesize-bunch only applies with --twin")
    try:
        paths = expand_archives(args.archives)
    except ValueError as error:
        parser.error(str(error))
    stems = [path.stem for path in paths]
    if len(set(stems)) != len(stems):
        parser.error(
            f"archive file names must be unique to name the runs: {sorted(stems)}"
        )
    models = {
        path.stem: LUMEFBPICModel.from_archive(path, dummy_run=True) for path in paths
    }
    if args.twin:
        without = [
            name for name, m in models.items() if m.simulator.final_particles is None
        ]
        if without and not args.synthesize_bunch:
            parser.error(
                f"--twin needs the final particles in every archive; without them: {without}. "
                "Write them with save_final_particles=True, or pass --synthesize-bunch to build "
                "an approximate bunch from each recorded descriptor"
            )
        if without:
            logging.getLogger(__name__).warning(
                "serving SYNTHETIC bunches built from the moment descriptor (%d macroparticles) "
                "for %s; they approximate the real bunches",
                SYNTHETIC_BUNCH_PARTICLES,
                without,
            )
    try:
        selector = ArchiveSelector(
            models,
            synthetic_bunch_particles=(
                SYNTHETIC_BUNCH_PARTICLES if args.synthesize_bunch else None
            ),
        )
    except ValueError as error:
        parser.error(str(error))
    serve_always = {DEFAULT_SELECTOR_NAME}
    if args.twin:
        from lume_fbpic_twin.twin_stage import build_chain  # needs htu and torch

        try:
            served = build_chain(selector, screen_binning=SCREEN_BINNING)
        except ValueError as error:
            parser.error(str(error))
        serve_always |= set(served.lume_model_instances[1].supported_variables)
        # A set of the chain, even an empty one, passes the LPA bunch on: do it once now so the
        # twin starts with the LPA bunch as its source.
        served.set({})
    else:
        served = selector
    config = build_config(
        served,
        prefix=TWIN_PREFIX if args.twin else LPA_PREFIX,
        serve_always=serve_always,
    )
    logging.getLogger(__name__).info("serving %d variables", len(config["variables"]))
    _import_runner().Runner(model=served, config=config).run()


def _import_runner():
    """The `lume_pva.runner` module (it holds `Runner` and `PutMode`)."""
    try:
        import lume_pva.runner as runner
    except ImportError as error:
        raise ImportError(
            "serve.py needs the lume-pva package, which is not installed."
        ) from error
    return runner


if __name__ == "__main__":
    main()
