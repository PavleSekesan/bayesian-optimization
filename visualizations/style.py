from __future__ import annotations

from matplotlib import pyplot as plt

# Shared blue-forward palette for all paper figures.
PALETTE = {
    "background": "#ffffff",
    "axes_background": "#ffffff",
    "grid": "#d8e4f3",
    "text": "#14304d",
    "spine": "#adc3dc",
    "true": "#0d3b66",
    "mean": "#1f77b4",
    "bounds": "#4f94d4",
    "band": "#9fc5e8",
    "points": "#0b2540",
}


def use_blue_theme() -> None:
    """Apply consistent styling for all generated figures."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["background"],
            "axes.facecolor": PALETTE["axes_background"],
            "axes.edgecolor": PALETTE["spine"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.alpha": 1.0,
            "grid.linewidth": 0.8,
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "text.color": PALETTE["text"],
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": PALETTE["spine"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 320,
        }
    )
