#!/bin/sh
set -eu

for interpreter in python python3; do
    if command -v "$interpreter" >/dev/null 2>&1 && "$interpreter" -c 'import sys; raise SystemExit(sys.version_info < (3, 9))'; then
        exec "$interpreter" lpa/Simulation_FBPIC/tools/sync_config_docstrings.py
    fi
done

printf '%s\n' 'Python 3.9 or newer is required to synchronize config docstrings.' >&2
exit 1