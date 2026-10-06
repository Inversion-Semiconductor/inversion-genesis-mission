# inversion-genesis-mission
Repository for Genesis mission-related code.

| Folder                  | What it contains                                                       | Key tech                                                          |
| ----------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **`lpa/`**| PIC simulations for LPA injectors                                      | FBPIC                                                      |

---

## Quick start

**Prerequisites:** GitHub account with access to this repo, `gh` CLI tool installed and authenticated.

Navigate to the directory you wish to clone this repo on your local machine (these instructions assume macOS)

```bash
# clone (HTTPS method - requires GitHub authentication)
git clone https://github.com/Inversion-Semiconductor/inversion-genesis-mission.git
cd inversion-genesis-mission

# install commit hooks
brew install pre-commit          # if not present
pre-commit install

```

Python baseline: 3.12.x (min 3.11, tests also run on 3.13)

### Recorded Git revision

`pre-commit install` installs the pre-commit checks and the post-commit,
post-checkout, and post-merge hooks configured in [.pre-commit-config.yaml](.pre-commit-config.yaml).
The latter hooks run [tools/record_git_hash.py](tools/record_git_hash.py) to record
the full `HEAD` hash in the generated
[lpa/Simulation_FBPIC/inversion_fbpic/git_hash.txt](lpa/Simulation_FBPIC/inversion_fbpic/git_hash.txt).
Package builds also record `HEAD` automatically: the setuptools `sdist`,
`build_py`, and `editable_wheel` commands refresh the file before packaging.
Both ordinary and editable pip installs from a fresh clone therefore record its
revision without requiring a commit or installed Git hooks. Install/build the
FBPIC subproject, not the separate shared-utilities package at the repo root.
Conda-build recipes that invoke pip or setuptools on this subproject use the
same hooks; simply creating a conda environment does not build the package.

Source distributions include the recording helper and revision file. When Git
metadata is absent (e.g. rebuilding a source distribution), builds preserve the
bundled hash instead of discovering an unrelated enclosing repository. Builds
without Git preserve any existing hash and warn if a checkout cannot refresh it.
Installing a prebuilt wheel does not need Git or regenerate its embedded hash.

The revision file is ignored by Git: a commit cannot contain its own hash,
and recording it must not dirty the checkout or recursively amend commits.
It is included as package data when building `inversion_fbpic`, allowing
serialization from installed packages or copied source trees without Git.
`SerializableConfig` only reads this file and emits `git_hash: null` if it is
missing, unreadable, or empty. For direct execution from an uninstalled clone,
run the recording script once. After subsequent checkouts, either keep Git
hooks installed, rebuild/reinstall, or refresh the file manually to avoid stale
provenance. This does not require modifying tracked files or making commits.

## Development workflow

| Step                        | Command / action                                            |
| --------------------------- | ----------------------------------------------------------- |
| **1. Create ticket branch** | `git switch -c IG-123-short-description`                    |
| **2. Code & commit**        | Hooks run Ruff → Black → clang-format                       |
| **3. Push**                 | `git push -u origin IG-123-short-description`               |
| **4. CI**                   | Branch-name guard + unit tests                              |
| **5. PR**                   | Requires _Code-Owner_ approval; `main` is fast-forward-only |
| **6. Merge**                | PR _Squash & Merge_ →                                       |

Branch names must match `IG-###-short-description`; duplicate ticket IDs are blocked by
`.github/workflows/branch-name.yml`.

## Tooling cheatsheet

- **Poetry** – dependency/packaging (pyproject.toml)
- **Ruff** – linter • **Black** – formatter • **clang-format** – C/C++
- **pytest** – Python tests • **CMake** – C++ builds
- **GitHub Actions** – CI & branch-name guard

## Contributing

1. File/assign a ticket → create branch `IS-###-short-description`.
2. Make sure `pre-commit run --all-files` passes.
3. Open PR; Code-Owner review required.
4. Squash-merge (fast-forward) into main.

## License

© 2026 Inversion Semiconductor.
Internal proprietary research code – redistributing without written permission
is prohibited.

Built with love, vacuum grease, and far too much caffeine. ☕️🔬
