from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

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
    return np.sin(0.9 * x) + 0.25 * np.cos(2.6 * x)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a 1D GP posterior plot with credible bounds in a uniform blue style."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "report" / "figures",
        help="Directory where the generated figure files are saved.",
    )
    parser.add_argument(
        "--filename-stem",
        type=str,
        default="gp_1d_posterior",
        help="Base filename for exported PDF file.",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=11,
        help="Number of training samples used to fit the GP.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible training locations.",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()

    use_blue_theme()

    x_min, x_max = -4.0, 4.0
    x_plot = np.linspace(x_min, x_max, 600).reshape(-1, 1)

    rng = np.random.default_rng(args.seed)
    x_train = np.sort(rng.uniform(
        x_min, x_max, size=args.num_train)).reshape(-1, 1)
    y_train = true_function(x_train[:, 0])

    gp = GaussianProcessRegressor(
        kernel=RBFKernel(length_scale=0.9, amplitude=1.0),
        mean_value=0.0,
        jitter=1e-8,
    )
    gp.fit(x_train, y_train)

    mean, variance = gp.predict(x_plot)
    std = np.sqrt(variance)

    z_95 = 1.96
    lower = mean - z_95 * std
    upper = mean + z_95 * std

    y_true = true_function(x_plot[:, 0])

    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    ax.plot(
        x_plot[:, 0],
        y_true,
        color=PALETTE["true"],
        linewidth=2.6,
        label="Prava funkcija",
    )
    ax.plot(
        x_plot[:, 0],
        mean,
        color=PALETTE["mean"],
        linestyle="--",
        linewidth=2.2,
        label="Očekivana vrednost",
    )
    ax.plot(
        x_plot[:, 0],
        lower,
        color=PALETTE["bounds"],
        linewidth=1.5,
        label="95% interval poverenja",
    )
    ax.plot(
        x_plot[:, 0],
        upper,
        color=PALETTE["bounds"],
        linewidth=1.5,
    )
    ax.fill_between(
        x_plot[:, 0],
        lower,
        upper,
        color=PALETTE["band"],
        alpha=0.35,
        linewidth=0,
    )
    ax.scatter(
        x_train[:, 0],
        y_train,
        color=PALETTE["points"],
        s=32,
        zorder=5,
        label="Uzorci funkcije",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="best")

    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / f"{args.filename_stem}.pdf"

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {pdf_path}")


if __name__ == "__main__":
    main()
