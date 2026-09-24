"""Read sectioned density lineout files exported by ANSYS."""

import re
from pathlib import Path

import numpy as np

SECTION_HEADER = re.compile(r'^\(\(xy/key/label\s+"([^"]+)"\)\s*$')
DATA_LINE = re.compile(r"^\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s*$")


def parse_lineout_file(path: Path) -> dict[str, np.ndarray]:
    """Parse a raw lineout file into labeled (z, density) arrays."""
    sections: dict[str, list[tuple[float, float]]] = {}
    current_label: str | None = None

    with path.open() as file:
        for line in file:
            line = line.rstrip("\n")

            if match := SECTION_HEADER.match(line):
                current_label = match.group(1)
                if current_label in sections:
                    raise ValueError(f"Duplicate lineout section {current_label!r} in {path}")
                sections[current_label] = []
                continue

            if line.strip() == ")":
                current_label = None
                continue

            if current_label is None:
                continue

            if match := DATA_LINE.match(line):
                z, density = map(float, match.groups())
                sections[current_label].append((z, density))

    if not sections:
        raise ValueError(f"No lineout sections found in {path}")

    return {
        label: np.array(points, dtype=np.float64)
        for label, points in sections.items()
    }