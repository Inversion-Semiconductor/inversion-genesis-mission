"""Plot laser pulse profiles from YAML configs in cfg/."""

from __future__ import annotations

from pathlib import Path

import yaml
import matplotlib.pyplot as plt
from inversion_fbpic.lib.laser import _GaussianTemporalLaserPulse, _LaserPulse

SCRIPT_DIR = Path(__file__).resolve().parent
CFG_DIR = SCRIPT_DIR / "cfg"
PLOTS_DIR = SCRIPT_DIR / "plots"


def _is_laser_pulse_yaml(path: Path) -> bool:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data, dict) and data.get("config_type") == _LaserPulse.CONFIG_TYPE


def main() -> None:
    if not CFG_DIR.is_dir():
        raise FileNotFoundError(f"Config directory not found: {CFG_DIR}")

    yaml_paths = sorted(
        path for path in CFG_DIR.glob("*.yaml") if _is_laser_pulse_yaml(path)
    )
    if not yaml_paths:
        raise FileNotFoundError(f"No laser-pulse YAML files found in {CFG_DIR}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))

    for yaml_path in yaml_paths:
        name = yaml_path.stem
        print(f"Plotting {name} from {yaml_path}")
        pulse = _LaserPulse.from_file(yaml_path)
        assert isinstance(pulse, _GaussianTemporalLaserPulse)
        pulse.plot(
            output_path=PLOTS_DIR / f"{name}.png",
            show=False,
            ax=ax,
        )

    ax.legend()
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Envelope amplitude (a₀)")
    ax.set_title("Combined Laser Envelopes")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {len(yaml_paths)} plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
