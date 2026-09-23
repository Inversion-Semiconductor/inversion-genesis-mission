"""Write example laser-pulse YAML configs for each polarization variant."""

from __future__ import annotations

from pathlib import Path

from scipy.constants import pi

from inversion_fbpic.lib.laser import GaussianLaserPulse, _LaserPulse

SCRIPT_DIR = Path(__file__).resolve().parent
CFG_DIR = SCRIPT_DIR / "cfg"

COMMON_LASER_KWARGS: dict = dict(
    energy=5.0,
    z0=-30e-6,
    wavelength=800e-9,
    tau_fwhm=38e-15,
    cep=0.0,
    waist=28e-6,
    focal_position=3e-3,
)


def build_example_profiles() -> dict[str, _LaserPulse]:
    """Instantiate each example laser pulse variant."""
    return {
        "gaussian_linear": GaussianLaserPulse(
            **COMMON_LASER_KWARGS,
            polarization=0.0,
        ),
        "gaussian_linear_45deg": GaussianLaserPulse(
            **COMMON_LASER_KWARGS,
            polarization=pi / 4,
        ),
        "gaussian_left_circular": GaussianLaserPulse(
            **COMMON_LASER_KWARGS,
            polarization="left",
        ),
        "gaussian_elliptical": GaussianLaserPulse(
            **COMMON_LASER_KWARGS,
            polarization=[pi / 8, pi / 2],
        ),
        "gaussian_antenna": GaussianLaserPulse(
            **COMMON_LASER_KWARGS,
            polarization=0.0,
            method="antenna",
            z0_antenna=0.0,
            v_antenna=0.0,
        ),
    }


def create_laser_yaml_files(output_dir: Path | None = None) -> dict[str, Path]:
    """Write a YAML file for each example laser pulse."""
    output_dir = output_dir or CFG_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name, pulse in build_example_profiles().items():
        path = output_dir / f"{name}.yaml"
        pulse.to_yaml_file(path, include_nones=False)
        paths[name] = path

    return paths


def main() -> None:
    paths = create_laser_yaml_files()
    print(f"Wrote {len(paths)} YAML files to {CFG_DIR}")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
