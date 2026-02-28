from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG_DIR = PROJECT_ROOT / ".runtime" / "mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

from matplotlib import pyplot as plt

from style import PALETTE, use_blue_theme


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generise 1x3 prikaz 2D mapa kernel funkcija: RBF, polinomsko, linearno."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "report" / "figures",
        help="Direktorijum za cuvanje PDF figure.",
    )
    parser.add_argument(
        "--filename-stem",
        type=str,
        default="kerneli_2d_1x3",
        help="Osnovno ime izlaznog PDF fajla.",
    )
    return parser


def rbf_kernel(x: np.ndarray, xp: np.ndarray, length_scale: float, amplitude: float) -> np.ndarray:
    sq_dist = (x - xp) ** 2
    return amplitude * np.exp(-0.5 * sq_dist / (length_scale**2))


def polynomial_kernel(x: np.ndarray, xp: np.ndarray, alpha: float, c: float, degree: int) -> np.ndarray:
    return (alpha * x * xp + c) ** degree


def linear_kernel(x: np.ndarray, xp: np.ndarray, sigma_v2: float, sigma_b2: float) -> np.ndarray:
    return sigma_v2 * x * xp + sigma_b2


def main() -> None:
    args = make_parser().parse_args()

    use_blue_theme()

    # Parametri kernela za prikaz u figuri.
    rbf_params = {"length_scale": 0.9, "amplitude": 1.0}
    poly_params = {"alpha": 0.4, "c": 1.0, "degree": 3}
    lin_params = {"sigma_v2": 0.8, "sigma_b2": 0.2}

    x = np.linspace(-2.5, 2.5, 240)
    x_mesh, xp_mesh = np.meshgrid(x, x)

    k_rbf = rbf_kernel(x_mesh, xp_mesh, **rbf_params)
    k_poly = polynomial_kernel(x_mesh, xp_mesh, **poly_params)
    k_lin = linear_kernel(x_mesh, xp_mesh, **lin_params)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), constrained_layout=True)

    plots = [
        (k_rbf, "RBF kernel"),
        (k_poly, "Polinomski kernel"),
        (k_lin, "Linearni kernel"),
    ]

    for ax, (kernel_values, _) in zip(axes, plots, strict=True):
        contour = ax.contourf(x, x, kernel_values, levels=28, cmap="Blues")
        ax.contour(
            x,
            x,
            kernel_values,
            levels=10,
            colors=PALETTE["spine"],
            linewidths=0.35,
            alpha=0.6,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("x'")
        cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("k(x, x')", rotation=90)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / f"{args.filename_stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {pdf_path}")


if __name__ == "__main__":
    main()
