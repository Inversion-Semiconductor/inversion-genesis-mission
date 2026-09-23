"""Plot density profiles from YAML configs in cfg/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from inversion_fbpic.lib.density_core import _DensityProfile
from inversion_fbpic.lib.serializable_config import SerializableConfig

SCRIPT_DIR = Path(__file__).resolve().parent
CFG_DIR = SCRIPT_DIR / "cfg"
PLOTS_DIR = SCRIPT_DIR / "plots"


def _is_density_profile_yaml(path: Path) -> bool:
    try:
        config = SerializableConfig.from_file(path)
    except (ValueError, OSError):
        return False
    return isinstance(config, _DensityProfile)


def main() -> None:
    if not CFG_DIR.is_dir():
        raise FileNotFoundError(f"Config directory not found: {CFG_DIR}")

    yaml_paths = sorted(
        path for path in CFG_DIR.glob("*.yaml") if _is_density_profile_yaml(path)
    )
    if not yaml_paths:
        raise FileNotFoundError(f"No density-profile YAML files found in {CFG_DIR}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))

    for yaml_path in yaml_paths:
        name = yaml_path.stem
        print(f"Plotting {name} from {yaml_path}")
        profile = _DensityProfile.from_file(yaml_path)
        assert isinstance(profile, _DensityProfile)
        profile.plot(
            output_path=PLOTS_DIR / f"{name}.png",
            show=False,
            ax=ax,
        )

    ax.legend()
    ax.set_xlabel("z (mm)")
    ax.set_ylabel(r"Density ($\mathrm{cm}^{-3}$)")
    ax.set_title("Combined Density Profiles")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {len(yaml_paths)} plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
