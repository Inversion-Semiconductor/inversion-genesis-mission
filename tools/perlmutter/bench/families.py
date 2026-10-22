"""Map GPU kernel names from both solvers to common PIC phase families.

Numba kernels appear in nsys as ``cudapy::fbpic::<module path>::<function>``.
Warp kernels appear as ``<function>_<hash>_cuda_kernel_forward``.
Library kernels come from cuBLAS (Hankel GEMM), cuFFT, CuPy, and CUB.
The first matching pattern wins, so specific patterns come first.
"""

from __future__ import annotations

import re

FAMILIES: tuple[tuple[str, str], ...] = (
    ("push_momentum", r"push_p"),
    ("ionization", r"ioniz|adk|copy_ionized|curand|philox"),
    ("deposition_rho", r"deposit_rho|linear_rho|_rho_m\d"),
    ("deposition_J", r"deposit_j|linear_j_|_j_m\d"),
    ("deposition_normalize", r"by_volume|erase_(vector|scalar)"),
    ("gather", r"gather|erase_eb"),
    ("push_position", r"push_x"),
    (
        "particle_buffers",
        r"particle_buffer|copy_particles|split_particles|add_particles|remove_particles|injection",
    ),
    ("sorting", r"cell_idx|sort|prefix_sum|cub::|scan"),
    ("hankel_gemm", r"gemm|xmma|cutlass|cublas"),
    ("hankel_copy", r"copy_2d|copy_1d|2dc_to|2dr_to|hankel"),
    ("fft", r"fft"),
    ("field_push", r"push_eb|push_rho"),
    ("current_correction", r"correct_current|correct_dive|curl_free|cross_deposit"),
    ("filter_and_damp", r"filter|damp|smooth"),
    ("spectral_conversion", r"pm_to_rt|rt_to_pm|spect2interp|interp2spect"),
    ("moving_window", r"shift_spect|move_grid|shift_interp"),
    ("laser_injection", r"antenna|laser"),
    ("cupy_elementwise", r"^cupy_"),
)

_COMPILED = tuple(
    (family, re.compile(pattern, re.IGNORECASE)) for family, pattern in FAMILIES
)


def short_name(name: str) -> str:
    """Strip numba ABI tags and argument lists to keep a readable kernel label."""
    name = re.sub(r"\[abi:[^\]]*\]", "", name)
    name = re.sub(r"\(.*\)$", "", name)
    name = re.sub(r"_[0-9a-f]{6,}_cuda_kernel_forward$", "", name)
    return name.replace("cudapy::", "")


def classify(name: str) -> str:
    """Return the phase family for a kernel name, or "other"."""
    label = short_name(name)
    for family, pattern in _COMPILED:
        if pattern.search(label):
            return family
    return "other"
