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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bayesopt.gaussian_process import GaussianProcessRegressor
from bayesopt.kernels import RBFKernel
from style import PALETTE, use_blue_theme


def true_function(x: np.ndarray) -> np.ndarray:
    t = (x + 4.0) / 8.0
    return ((6.0 * t - 2.0) ** 2) * np.sin(12.0 * t - 4.0) / 8.0 + 0.15 * np.cos(5.0 * x)


def expected_improvement_min(
    mean: np.ndarray,
    variance: np.ndarray,
    best_y: float,
    xi: float,
) -> np.ndarray:
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
    return np.maximum(ei, 0.0)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generiše 4x2 prikaz evolucije BO procesa: GP levo, EI desno."
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
        default="bo_run_4x2",
        help="Osnovno ime izlaznog PDF fajla.",
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=0.01,
        help="Eksploracioni parametar xi za EI (minimizacija).",
    )
    return parser


def build_gp() -> GaussianProcessRegressor:
    return GaussianProcessRegressor(
        kernel=RBFKernel(length_scale=0.9, amplitude=1.0),
        noise_variance=1e-4,
        mean_value=0.0,
        jitter=1e-8,
    )


def generate_bo_sequence(x_grid: np.ndarray, max_points: int, xi: float) -> tuple[np.ndarray, np.ndarray]:
    x_observed: list[float] = [-3.2]
    y_observed: list[float] = [float(true_function(np.array([-3.2]))[0])]

    for _ in range(max_points - 1):
        gp = build_gp()
        x_train = np.asarray(x_observed, dtype=np.float64).reshape(-1, 1)
        y_train = np.asarray(y_observed, dtype=np.float64)
        gp.fit(x_train, y_train)

        mean, variance = gp.predict(x_grid.reshape(-1, 1))
        ei = expected_improvement_min(mean, variance, best_y=float(np.min(y_train)), xi=xi)

        for x_prev in x_observed:
            ei[np.abs(x_grid - x_prev) < 1e-6] = -np.inf

        if not np.any(np.isfinite(ei)):
            break

        next_x = float(x_grid[int(np.argmax(ei))])
        next_y = float(true_function(np.array([next_x]))[0])

        x_observed.append(next_x)
        y_observed.append(next_y)

    return np.asarray(x_observed, dtype=np.float64), np.asarray(y_observed, dtype=np.float64)


def compute_state(
    x_grid: np.ndarray,
    n_points: int,
    x_observed: np.ndarray,
    y_observed: np.ndarray,
    xi: float,
) -> dict[str, np.ndarray]:
    amplitude = 1.0

    if n_points == 0:
        mean = np.zeros_like(x_grid)
        variance = np.full_like(x_grid, amplitude)
        ei = np.zeros_like(x_grid)
        x_train = np.empty((0,), dtype=np.float64)
        y_train = np.empty((0,), dtype=np.float64)
    else:
        x_train = x_observed[:n_points]
        y_train = y_observed[:n_points]

        gp = build_gp()
        gp.fit(x_train.reshape(-1, 1), y_train)
        mean, variance = gp.predict(x_grid.reshape(-1, 1))

        best_y = float(np.min(y_train))
        ei = expected_improvement_min(mean, variance, best_y=best_y, xi=xi)

    std = np.sqrt(variance)
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std

    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "ei": ei,
        "x_train": x_train,
        "y_train": y_train,
    }


def main() -> None:
    args = make_parser().parse_args()

    use_blue_theme()

    x_min, x_max = -4.0, 4.0
    x_grid = np.linspace(x_min, x_max, 1200)
    y_true = true_function(x_grid)

    snapshots = [0, 1, 4, 10]
    x_observed, y_observed = generate_bo_sequence(x_grid, max_points=max(snapshots), xi=args.xi)
    states = [compute_state(x_grid, n, x_observed, y_observed, args.xi) for n in snapshots]

    ei_max_global = max(float(np.max(state["ei"])) for state in states)
    ei_ylim_max = 1.15 * ei_max_global if ei_max_global > 1e-10 else 1.0

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(13.0, 12.0),
        sharex=True,
        gridspec_kw={"width_ratios": [2.5, 1.5], "hspace": 0.10, "wspace": 0.16},
    )

    ei_color = "#0f5ea8"

    for row_idx, (n_points, state) in enumerate(zip(snapshots, states, strict=True)):
        ax_gp = axes[row_idx, 0]
        ax_ei = axes[row_idx, 1]

        ax_gp.plot(x_grid, y_true, color=PALETTE["true"], linewidth=2.1)
        ax_gp.plot(x_grid, state["mean"], color=PALETTE["mean"], linestyle="--", linewidth=1.9)
        ax_gp.plot(x_grid, state["lower"], color=PALETTE["bounds"], linewidth=1.1)
        ax_gp.plot(x_grid, state["upper"], color=PALETTE["bounds"], linewidth=1.1)
        ax_gp.fill_between(
            x_grid,
            state["lower"],
            state["upper"],
            color=PALETTE["band"],
            alpha=0.24,
            linewidth=0,
        )

        if n_points > 0:
            ax_gp.scatter(state["x_train"], state["y_train"], color=PALETTE["points"], s=22, zorder=5)

        ax_ei.plot(x_grid, state["ei"], color=ei_color, linewidth=1.8)
        ax_ei.fill_between(x_grid, 0.0, state["ei"], color=ei_color, alpha=0.12, linewidth=0)
        ax_ei.set_ylim(0.0, ei_ylim_max)

        ax_gp.text(
            0.02,
            0.92,
            f"n = {n_points}",
            transform=ax_gp.transAxes,
            fontsize=10,
            fontweight="bold",
            color=PALETTE["text"],
            ha="left",
            va="top",
            bbox={"facecolor": "#ffffff", "edgecolor": PALETTE["spine"], "boxstyle": "round,pad=0.2"},
        )

        if n_points == 0:
            ax_ei.text(
                0.96,
                0.90,
                "EI nije definisan\nbez podataka",
                transform=ax_ei.transAxes,
                fontsize=8,
                color=ei_color,
                ha="right",
                va="top",
            )

        ax_gp.set_xlim(x_min, x_max)
        ax_ei.set_xlim(x_min, x_max)

        ax_gp.set_ylabel("f(x)")
        ax_ei.set_ylabel("EI")

    axes[3, 0].set_xlabel("x")
    axes[3, 1].set_xlabel("x")

    legend_handles = [
        Line2D([0], [0], color=PALETTE["true"], linewidth=2.1, label="Prava funkcija"),
        Line2D([0], [0], color=PALETTE["mean"], linestyle="--", linewidth=1.9, label="Očekivana vrednost GP-a"),
        Patch(facecolor=PALETTE["band"], edgecolor=PALETTE["bounds"], label="95% interval poverenja", alpha=0.35),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PALETTE["points"],
            markeredgecolor=PALETTE["points"],
            markersize=6,
            label="Uzorci funkcije",
        ),
        Line2D([0], [0], color=ei_color, linewidth=1.8, label="Očekivano poboljšanje (EI)"),
    ]

    fig.legend(handles=legend_handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995), frameon=True)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.06, hspace=0.14, wspace=0.16)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / f"{args.filename_stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {pdf_path}")


if __name__ == "__main__":
    main()
