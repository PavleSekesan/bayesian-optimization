from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.special import erf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
MPLCONFIG_DIR = PROJECT_ROOT / ".runtime" / "mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from matplotlib import pyplot as plt

from bayesopt.gaussian_process import GaussianProcessRegressor
from bayesopt.kernels import RBFKernel
from style import PALETTE, use_blue_theme


def true_function(x: np.ndarray) -> np.ndarray:
    t = (x + 4.0) / 8.0
    return ((6.0 * t - 2.0) ** 2) * np.sin(12.0 * t - 4.0) / 8.0 + 0.15 * np.cos(5.0 * x)


def expected_improvement_min(mean: np.ndarray, variance: np.ndarray, best_y: float, xi: float) -> np.ndarray:
    std = np.sqrt(np.maximum(variance, 0.0))
    eps = 1e-12

    improvement = best_y - mean - xi
    z = np.zeros_like(std)

    valid = std > eps
    z[valid] = improvement[valid] / std[valid]

    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z)

    ei = np.zeros_like(std)
    ei[valid] = improvement[valid] * cdf[valid] + std[valid] * pdf[valid]
    return ei


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generiše 2x1 graf: GP posterior (funkcija, očekivanje, varijansa) i EI funkcija."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "report" / "figures",
        help="Direktorijum za čuvanje PDF figure.",
    )
    parser.add_argument(
        "--filename-stem",
        type=str,
        default="gp_ei_2x1",
        help="Osnovno ime izlaznog PDF fajla.",
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=0.01,
        help="Eksploracioni parametar xi za EI (minimizacija).",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()

    use_blue_theme()

    x_min, x_max = -4.0, 4.0
    x_plot = np.linspace(x_min, x_max, 800).reshape(-1, 1)

    # Namerno malo i neuniformno uzorkovanje radi izraženije EI krive.
    x_train = np.array([-3.8, -3.0, -2.2, -1.4, 0.3, 3.6], dtype=np.float64).reshape(-1, 1)
    y_train = true_function(x_train[:, 0])

    gp = GaussianProcessRegressor(
        kernel=RBFKernel(length_scale=0.9, amplitude=1.0),
        mean_value=0.0,
        jitter=1e-8,
    )
    gp.fit(x_train, y_train)

    mean, variance = gp.predict(x_plot)
    std = np.sqrt(variance)
    y_true = true_function(x_plot[:, 0])

    z_95 = 1.96
    lower = mean - z_95 * std
    upper = mean + z_95 * std

    best_observed = float(np.min(y_train))
    ei = expected_improvement_min(mean, variance, best_observed, xi=args.xi)
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10.2, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.08},
    )

    ax_top.plot(
        x_plot[:, 0], y_true, color=PALETTE["true"], linewidth=2.3, label="Prava funkcija"
    )
    ax_top.plot(
        x_plot[:, 0],
        mean,
        color=PALETTE["mean"],
        linestyle="--",
        linewidth=2.1,
        label="Očekivana vrednost GP-a",
    )
    ax_top.plot(
        x_plot[:, 0],
        lower,
        color=PALETTE["bounds"],
        linewidth=1.3,
        label="95% interval poverenja",
    )
    ax_top.plot(x_plot[:, 0], upper, color=PALETTE["bounds"], linewidth=1.3)
    ax_top.fill_between(
        x_plot[:, 0], lower, upper, color=PALETTE["band"], alpha=0.28, linewidth=0
    )
    ax_top.scatter(
        x_train[:, 0],
        y_train,
        color=PALETTE["points"],
        s=30,
        zorder=5,
        label="Uzorci funkcije",
    )
    ax_top.legend(loc="upper left")

    ax_top.set_ylabel("f(x)")
    ax_top.set_xlim(x_min, x_max)

    ax_bottom.plot(
        x_plot[:, 0],
        ei,
        color=PALETTE["mean"],
        linewidth=2.1,
        label="Očekivano poboljšanje (EI)",
    )
    ax_bottom.fill_between(
        x_plot[:, 0], 0.0, ei, color=PALETTE["band"], alpha=0.35, linewidth=0
    )

    ax_bottom.set_xlabel("x")
    ax_bottom.set_ylabel("EI")
    ax_bottom.legend(loc="upper right")

    fig.subplots_adjust(left=0.08, right=0.91, top=0.98, bottom=0.09, hspace=0.12)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / f"{args.filename_stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {pdf_path}")


if __name__ == "__main__":
    main()
