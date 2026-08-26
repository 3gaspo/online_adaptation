"""Sampled setting shifts and retrieved-neighbor diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.visualization.diagnostics import (
    plot_neighbor_all,
    plot_neighbor_per_user,
    plot_setting,
)


SETTING_EDGE_DATES = 8
SETTING_ALIGNED_DATES = 16
SETTING_USER_PAIRS = 64
SETTING_DATE_USERS = 64
NEIGHBOR_PLOT_SAMPLES = 200_000


def instance_normalize_window(values: np.ndarray, lookback: int, eps: float = 1e-8) -> np.ndarray:
    """Normalize a complete ``L+H`` window with its own lookback statistics."""
    array = np.asarray(values, dtype=np.float32)
    history = array[..., : int(lookback)]
    mean = history.mean(axis=-1, keepdims=True)
    std = np.maximum(history.std(axis=-1, keepdims=True), float(eps))
    return (array - mean) / std


def rms_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Root-mean-square Euclidean distance along the window axis."""
    difference = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
    return np.sqrt(np.mean(np.square(difference, dtype=np.float64), axis=-1))


def neighbor_lookback_distances(
    query_lookbacks: np.ndarray,
    neighbor_lookbacks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and independently instance-normalized lookback RMS distances."""
    query = np.asarray(query_lookbacks, dtype=np.float32)
    neighbors = np.asarray(neighbor_lookbacks, dtype=np.float32)
    if neighbors.shape[:-2] != query.shape[:-1] or neighbors.shape[-1] != query.shape[-1]:
        raise ValueError("neighbor lookbacks must add one neighbor dimension to query lookbacks")
    raw = rms_distance(query[..., None, :], neighbors)
    normalized_query = instance_normalize_window(query, query.shape[-1])
    normalized_neighbors = instance_normalize_window(neighbors, neighbors.shape[-1])
    normalized = rms_distance(normalized_query[..., None, :], normalized_neighbors)
    return raw.astype(np.float32), normalized.astype(np.float32)


def _sample_without_replacement(
    rng: np.random.Generator,
    values: np.ndarray,
    count: int,
) -> np.ndarray:
    size = min(int(count), len(values))
    if size == 0:
        return np.empty(0, dtype=np.int64)
    return np.sort(rng.choice(values, size=size, replace=False).astype(np.int64))


def _distinct_user_pairs(
    rng: np.random.Generator,
    n_users: int,
    count: int,
) -> np.ndarray:
    possible = int(n_users) * (int(n_users) - 1) // 2
    wanted = min(int(count), possible)
    if wanted == 0:
        return np.empty((0, 2), dtype=np.int64)
    if possible <= 4 * wanted:
        pairs = np.asarray(
            [(left, right) for left in range(n_users) for right in range(left + 1, n_users)],
            dtype=np.int64,
        )
        return pairs[rng.choice(len(pairs), size=wanted, replace=False)]
    selected: set[tuple[int, int]] = set()
    while len(selected) < wanted:
        pair = rng.choice(n_users, size=2, replace=False)
        selected.add(tuple(sorted((int(pair[0]), int(pair[1])))))
    return np.asarray(sorted(selected), dtype=np.int64)


def sample_setting_distances(
    dataset: Any,
    *,
    lookback: int,
    horizon: int,
    seed: int,
    edge_dates: int = SETTING_EDGE_DATES,
    aligned_dates: int = SETTING_ALIGNED_DATES,
    user_pairs: int = SETTING_USER_PAIRS,
    date_users: int = SETTING_DATE_USERS,
) -> dict[str, np.ndarray]:
    """Sample inter-date and aligned inter-user distances between ``L+H`` windows."""
    first = int(lookback) - 1
    last = int(dataset.n_dates) - int(horizon) - 1
    if last < first:
        raise ValueError("dataset has no complete L+H windows for setting diagnostics")
    valid_dates = np.arange(first, last + 1, dtype=np.int64)
    edge_size = max(1, int(np.ceil(0.1 * len(valid_dates))))
    rng = np.random.default_rng(int(seed))
    early = _sample_without_replacement(rng, valid_dates[:edge_size], edge_dates)
    late = _sample_without_replacement(rng, valid_dates[-edge_size:], edge_dates)
    aligned = _sample_without_replacement(rng, valid_dates, aligned_dates)
    users = _sample_without_replacement(
        rng, np.arange(dataset.n_users, dtype=np.int64), date_users
    )
    pairs = _distinct_user_pairs(rng, int(dataset.n_users), user_pairs)

    requested_dates = np.unique(np.concatenate((early, late, aligned)))
    offsets = np.arange(-int(lookback) + 1, int(horizon) + 1, dtype=np.int64)
    windows = np.transpose(
        np.asarray(dataset.values)[requested_dates[:, None] + offsets],
        (2, 0, 1),
    )
    positions = {int(date): index for index, date in enumerate(requested_dates)}

    early_windows = windows[users][:, [positions[int(date)] for date in early]]
    late_windows = windows[users][:, [positions[int(date)] for date in late]]
    left_date = early_windows[:, :, None, :]
    right_date = late_windows[:, None, :, :]

    if len(pairs):
        aligned_pos = [positions[int(date)] for date in aligned]
        left_user = windows[pairs[:, 0]][:, aligned_pos, :]
        right_user = windows[pairs[:, 1]][:, aligned_pos, :]
    else:
        shape = (0, len(aligned), int(lookback) + int(horizon))
        left_user = np.empty(shape, dtype=np.float32)
        right_user = np.empty(shape, dtype=np.float32)

    return {
        "inter_date_raw": rms_distance(left_date, right_date).reshape(-1),
        "inter_date_instance_normalized": rms_distance(
            instance_normalize_window(left_date, lookback),
            instance_normalize_window(right_date, lookback),
        ).reshape(-1),
        "inter_user_raw": rms_distance(left_user, right_user).reshape(-1),
        "inter_user_instance_normalized": rms_distance(
            instance_normalize_window(left_user, lookback),
            instance_normalize_window(right_user, lookback),
        ).reshape(-1),
        "early_dates": early,
        "late_dates": late,
        "aligned_dates": aligned,
        "date_users": users,
        "user_pairs": pairs,
    }


def _summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
            "max": float("nan"),
        }
    quantiles = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "max": float(array.max()),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_setting_diagnostics(
    output_dir: str | Path,
    *,
    dataset: Any,
    lookback: int,
    horizon: int,
    seed: int,
) -> dict[str, Path]:
    """Compute and persist per-dataset/setting shift diagnostics."""
    root = Path(output_dir)
    samples = sample_setting_distances(
        dataset,
        lookback=lookback,
        horizon=horizon,
        seed=seed,
    )
    distance_keys = [
        "inter_date_raw",
        "inter_date_instance_normalized",
        "inter_user_raw",
        "inter_user_instance_normalized",
    ]
    rows = []
    sample_rows = []
    for key in distance_keys:
        if key.endswith("_raw"):
            kind, scale = key.rsplit("_", 1)
        else:
            kind = key.removesuffix("_instance_normalized")
            scale = "instance_normalized"
        rows.append({"diagnostic": kind, "scale": scale, **_summary(samples[key])})
        sample_rows.extend(
            {"diagnostic": kind, "scale": scale, "distance": float(value)}
            for value in samples[key]
        )
    summary_path = root / "setting_diagnostics.csv"
    samples_path = root / "setting_diagnostics_samples.csv"
    plot_path = root / "setting_diagnostics.png"
    sampling_path = root / "setting_diagnostics_sampling.json"
    _write_rows(summary_path, rows)
    _write_rows(samples_path, sample_rows)
    sampling_path.write_text(
        json.dumps(
            {
                "window": "L+H",
                "distance": "root_mean_square_euclidean",
                "instance_normalization": (
                    "each full window standardized by its own L-step lookback "
                    "mean and standard deviation"
                ),
                "seed": int(seed),
                "early_dates": samples["early_dates"].tolist(),
                "late_dates": samples["late_dates"].tolist(),
                "aligned_dates": samples["aligned_dates"].tolist(),
                "date_users": samples["date_users"].tolist(),
                "user_pairs": samples["user_pairs"].tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_setting(samples, plot_path)
    return {
        "setting_summary": summary_path,
        "setting_samples": samples_path,
        "setting_sampling": sampling_path,
        "setting_plot": plot_path,
    }


def _neighbor_sample(
    arrays: Mapping[str, np.ndarray],
    *,
    seed: int,
    maximum: int = NEIGHBOR_PLOT_SAMPLES,
) -> dict[str, np.ndarray]:
    shape = arrays["neighbor_user"].shape
    total = int(np.prod(shape))
    rng = np.random.default_rng(int(seed))
    if total <= int(maximum):
        flat = np.arange(total, dtype=np.int64)
    else:
        flat = rng.integers(0, total, size=int(maximum), dtype=np.int64)
    date_index, remainder = np.divmod(flat, shape[1] * shape[2])
    user_index, neighbor_index = np.divmod(remainder, shape[2])
    neighbor_user = np.asarray(
        arrays["neighbor_user"][date_index, user_index, neighbor_index]
    )
    neighbor_date = np.asarray(
        arrays["neighbor_window_dates"][date_index, user_index, neighbor_index]
    )
    retrieval_date = np.asarray(arrays["retrieval_window_dates"][date_index])
    return {
        "same_user": neighbor_user == user_index,
        "age": retrieval_date - neighbor_date,
        "raw": np.asarray(
            arrays["neighbor_distance_raw"][date_index, user_index, neighbor_index]
        ),
        "instance_normalized": np.asarray(
            arrays["neighbor_distance_instance_normalized"][date_index, user_index, neighbor_index]
        ),
    }


def _exact_neighbor_accumulators(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    n_dates, n_users, _ = arrays["neighbor_user"].shape
    fields = ("count", "age", "age2", "raw", "raw2", "instance", "instance2")
    result = {
        f"{group}_{field}": np.zeros(n_users, dtype=np.float64)
        for group in ("same", "other")
        for field in fields
    }
    user_grid = np.arange(n_users, dtype=np.int64)[None, :, None]
    for start in range(0, n_dates, 64):
        stop = min(start + 64, n_dates)
        neighbor_users = np.asarray(arrays["neighbor_user"][start:stop])
        same = neighbor_users == user_grid
        age = (
            np.asarray(arrays["retrieval_window_dates"][start:stop])[:, None, None]
            - np.asarray(arrays["neighbor_window_dates"][start:stop])
        )
        raw = np.asarray(arrays["neighbor_distance_raw"][start:stop])
        instance = np.asarray(arrays["neighbor_distance_instance_normalized"][start:stop])
        for group, mask in (("same", same), ("other", ~same)):
            result[f"{group}_count"] += mask.sum(axis=(0, 2))
            for field, values in (("age", age), ("raw", raw), ("instance", instance)):
                selected = np.where(mask, values, 0.0)
                result[f"{group}_{field}"] += selected.sum(axis=(0, 2), dtype=np.float64)
                result[f"{group}_{field}2"] += np.square(
                    selected, dtype=np.float64
                ).sum(axis=(0, 2))
    return result


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def write_neighbor_diagnostics(
    output_dir: str | Path,
    *,
    arrays: Mapping[str, np.ndarray],
    user_names: list[str],
    seed: int,
) -> dict[str, Path]:
    """Persist extraction-config-specific neighbor summaries and histograms."""
    root = Path(output_dir)
    sample = _neighbor_sample(arrays, seed=seed)
    accumulators = _exact_neighbor_accumulators(arrays)
    per_user: dict[str, np.ndarray] = {}
    total = accumulators["same_count"] + accumulators["other_count"]
    per_user["same_fraction"] = _safe_divide(accumulators["same_count"], total)
    for group in ("same", "other"):
        count = accumulators[f"{group}_count"]
        for field in ("age", "raw", "instance"):
            per_user[f"{group}_{field}"] = _safe_divide(
                accumulators[f"{group}_{field}"], count
            )

    summary_rows = []
    all_count = int(total.sum())
    for group, group_count in (
        ("same_user", int(accumulators["same_count"].sum())),
        ("other_user", int(accumulators["other_count"].sum())),
    ):
        fraction = group_count / all_count if all_count else float("nan")
        summary_rows.append(
            {
                "aggregation": "all_samples",
                "neighbor_scope": group,
                "metric": "scope_fraction",
                "count": group_count,
                "mean": fraction,
                "std": float("nan"),
                "min": float("nan"),
                "q05": float("nan"),
                "q25": float("nan"),
                "median": float("nan"),
                "q75": float("nan"),
                "q95": float("nan"),
                "max": float("nan"),
                "quantiles_from_sample": 0,
            }
        )
    for group, mask in (("same_user", sample["same_user"]), ("other_user", ~sample["same_user"])):
        exact_group = "same" if group == "same_user" else "other"
        exact_count = int(accumulators[f"{exact_group}_count"].sum())
        for metric, field in (
            ("age", "age"),
            ("lookback_distance_raw", "raw"),
            ("lookback_distance_instance_normalized", "instance_normalized"),
        ):
            sampled = _summary(sample[field][mask])
            accumulator_field = (
                "instance" if field == "instance_normalized" else field
            )
            total_sum = float(
                accumulators[f"{exact_group}_{accumulator_field}"].sum()
            )
            total_square = float(
                accumulators[f"{exact_group}_{accumulator_field}2"].sum()
            )
            exact_mean = total_sum / exact_count if exact_count else float("nan")
            exact_variance = (
                max(0.0, total_square / exact_count - exact_mean**2)
                if exact_count
                else float("nan")
            )
            summary_rows.append(
                {
                    "aggregation": "all_samples",
                    "neighbor_scope": group,
                    "metric": metric,
                    "count": exact_count,
                    "mean": exact_mean,
                    "std": float(np.sqrt(exact_variance)),
                    "min": sampled["min"],
                    "q05": sampled["q05"],
                    "q25": sampled["q25"],
                    "median": sampled["median"],
                    "q75": sampled["q75"],
                    "q95": sampled["q95"],
                    "max": sampled["max"],
                    "quantiles_from_sample": int(sampled["count"]),
                }
            )

    for group, group_key in (("same_user", "same"), ("other_user", "other")):
        scope_values = (
            per_user["same_fraction"]
            if group_key == "same"
            else 1.0 - per_user["same_fraction"]
        )
        per_user_metrics = [("scope_fraction", scope_values)]
        per_user_metrics.extend(
            (
                metric,
                per_user[f"{group_key}_{field}"],
            )
            for metric, field in (
                ("age", "age"),
                ("lookback_distance_raw", "raw"),
                ("lookback_distance_instance_normalized", "instance"),
            )
        )
        for metric, values in per_user_metrics:
            finite = values[np.isfinite(values)]
            summary_rows.append(
                {
                    "aggregation": "per_user_average",
                    "neighbor_scope": group,
                    "metric": metric,
                    **_summary(finite),
                    "quantiles_from_sample": int(len(finite)),
                }
            )

    user_rows = []
    for user, name in enumerate(user_names):
        user_rows.append(
            {
                "query_user": user,
                "query_user_name": name,
                "neighbors": int(total[user]),
                "same_user_neighbors": int(accumulators["same_count"][user]),
                "other_user_neighbors": int(accumulators["other_count"][user]),
                "same_user_fraction": per_user["same_fraction"][user],
                "same_user_mean_age": per_user["same_age"][user],
                "other_user_mean_age": per_user["other_age"][user],
                "same_user_mean_raw_distance": per_user["same_raw"][user],
                "other_user_mean_raw_distance": per_user["other_raw"][user],
                "same_user_mean_instance_distance": per_user["same_instance"][user],
                "other_user_mean_instance_distance": per_user["other_instance"][user],
            }
        )

    summary_path = root / "neighbor_diagnostics.csv"
    users_path = root / "neighbor_diagnostics_per_user.csv"
    all_plot_path = root / "neighbor_diagnostics_all_samples.png"
    user_plot_path = root / "neighbor_diagnostics_per_user.png"
    _write_rows(summary_path, summary_rows)
    _write_rows(users_path, user_rows)
    plot_neighbor_all(
        sample,
        all_plot_path,
        scope_counts=(
            int(accumulators["same_count"].sum()),
            int(accumulators["other_count"].sum()),
        ),
    )
    plot_neighbor_per_user(per_user, user_plot_path)
    return {
        "neighbor_summary": summary_path,
        "neighbor_per_user": users_path,
        "neighbor_all_plot": all_plot_path,
        "neighbor_per_user_plot": user_plot_path,
    }
