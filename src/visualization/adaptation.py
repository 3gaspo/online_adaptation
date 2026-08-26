"""Plots for readable summaries of fitted online adaptors."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def plot_coefficient_summary(csv_path: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    names = [row["feature"] for row in rows]
    means = np.asarray([float(row["mean_coefficient"]) for row in rows])
    stds = np.asarray([float(row["coefficient_std"]) for row in rows])
    figure, axis = plt.subplots(figsize=(max(6.0, 0.45 * len(names)), 3.8))
    axis.bar(np.arange(len(names)), means, yerr=stds, capsize=2)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
    axis.set_ylabel("rolling coefficient mean ± std")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_gate_importance(csv_path: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    names = [row["feature"] for row in rows]
    means = np.asarray([float(row["mean_importance"]) for row in rows])
    stds = np.asarray([float(row["importance_std"]) for row in rows])
    order = np.argsort(means)
    figure, axis = plt.subplots(figsize=(7.0, max(4.0, 0.28 * len(names))))
    axis.barh(np.asarray(names)[order], means[order], xerr=stds[order], capsize=2)
    axis.set_xlabel("rolling feature importance mean ± std")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
