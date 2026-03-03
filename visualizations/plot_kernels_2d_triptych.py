from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "report" / "figures"
FILE_STEM = "kerneli_2d_1x3"

MPLCONFIG_DIR = PROJECT_ROOT / ".runtime" / "mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

from matplotlib import pyplot as plt

from style import PALETTE, use_blue_theme


def rbf_kernel(x: np.ndarray, xp: np.ndarray, length_scale: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - xp) ** 2) / (length_scale**2))


def polynomial_kernel(x: np.ndarray, xp: np.ndarray, alpha: float, c: float, degree: int) -> np.ndarray:
    return (alpha * x * xp + c) ** degree


def linear_kernel(x: np.ndarray, xp: np.ndarray, sigma_v2: float, sigma_b2: float) -> np.ndarray:
    return sigma_v2 * x * xp + sigma_b2


def main() -> None:
    use_blue_theme()

    x = np.linspace(-2.5, 2.5, 240)
    x_mesh, xp_mesh = np.meshgrid(x, x)

    k_rbf = rbf_kernel(x_mesh, xp_mesh, length_scale=0.9, amplitude=1.0)
    k_poly = polynomial_kernel(x_mesh, xp_mesh, alpha=0.4, c=1.0, degree=3)
    k_lin = linear_kernel(x_mesh, xp_mesh, sigma_v2=0.8, sigma_b2=0.2)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), constrained_layout=True)

    for ax, kernel_values in zip(axes, [k_rbf, k_poly, k_lin], strict=True):
        contour = ax.contourf(x, x, kernel_values, levels=28, cmap="Blues")
        ax.contour(x, x, kernel_values, levels=10, colors=PALETTE["spine"], linewidths=0.35, alpha=0.6)
        ax.set_xlabel("x")
        ax.set_ylabel("x'")
        cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("k(x, x')", rotation=90)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f"{FILE_STEM}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {pdf_path}")


if __name__ == "__main__":
    main()
