"""Visual dashboards for dataset-shift and retrieval diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def plot_setting(samples: Mapping[str, np.ndarray], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, kind in enumerate(("inter_date", "inter_user")):
        for column, scale in enumerate(("raw", "instance_normalized")):
            values = samples[f"{kind}_{scale}"]
            if len(values):
                axes[row, column].hist(values, bins=30, color="#4472c4", alpha=0.85)
                axes[row, column].axvline(float(np.mean(values)), color="#c00000", linewidth=1.5)
            else:
                axes[row, column].text(0.5, 0.5, "not available (one user)", ha="center", va="center")
            axes[row, column].set_title(f"{kind.replace('_', ' ')} — {scale.replace('_', ' ')}")
            axes[row, column].set_xlabel("window RMS distance")
            axes[row, column].set_ylabel("sampled pairs")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_neighbor_all(
    sample: Mapping[str, np.ndarray],
    path: Path,
    *,
    scope_counts: tuple[int, int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    same = sample["same_user"]
    counts = np.asarray(scope_counts)
    axes[0, 0].bar(("same user", "other user"), counts / max(1, counts.sum()), color=("#4472c4", "#ed7d31"))
    axes[0, 0].set_ylabel("fraction of retrieved neighbors")
    axes[0, 0].set_title("Neighbor user scope")
    for axis, field, title, label in (
        (axes[0, 1], "age", "Neighbor age", "window-date difference (time steps)"),
        (axes[1, 0], "raw", "Lookback distance — raw", "lookback RMS distance"),
        (axes[1, 1], "instance_normalized", "Lookback distance — instance normalized", "lookback RMS distance"),
    ):
        for mask, name, color in ((same, "same user", "#4472c4"), (~same, "other user", "#ed7d31")):
            if mask.any():
                axis.hist(sample[field][mask], bins=40, density=True, histtype="step", linewidth=1.6, label=name, color=color)
        axis.set_title(title)
        axis.set_xlabel(label)
        axis.set_ylabel("density")
        axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_neighbor_per_user(per_user: Mapping[str, np.ndarray], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    same_fraction = per_user["same_fraction"]
    axes[0, 0].hist(same_fraction[np.isfinite(same_fraction)], bins=np.linspace(0.0, 1.0, 21), color="#4472c4")
    axes[0, 0].set_xlim(0.0, 1.0)
    axes[0, 0].set_title("Same-user fraction by query user")
    axes[0, 0].set_xlabel("fraction")
    axes[0, 0].set_ylabel("query users")
    for axis, field, title, label in (
        (axes[0, 1], "age", "Mean age by query user", "mean window-date difference (time steps)"),
        (axes[1, 0], "raw", "Mean raw distance by query user", "mean lookback RMS distance"),
        (axes[1, 1], "instance", "Mean normalized distance by query user", "mean lookback RMS distance"),
    ):
        for group, name, color in (("same", "same user", "#4472c4"), ("other", "other user", "#ed7d31")):
            values = per_user[f"{group}_{field}"]
            values = values[np.isfinite(values)]
            if len(values):
                axis.hist(values, bins=30, density=True, histtype="step", linewidth=1.6, label=name, color=color)
        axis.set_title(title)
        axis.set_xlabel(label)
        axis.set_ylabel("density")
        axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)
