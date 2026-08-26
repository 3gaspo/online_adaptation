"""Dependency-light checks for online extraction diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile

import numpy as np

from src.results.diagnostics import (
    instance_normalize_window,
    neighbor_lookback_distances,
    sample_setting_distances,
    write_neighbor_diagnostics,
    write_setting_diagnostics,
)


class SyntheticDataset:
    def __init__(self) -> None:
        dates = np.arange(80, dtype=np.float32)
        self.values = np.stack((dates, 2.0 * dates + 10.0, np.sin(dates / 5.0)), axis=1)
        self.user_names = ["a", "b", "c"]

    @property
    def n_dates(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_users(self) -> int:
        return int(self.values.shape[1])


def _neighbor_arrays() -> dict[str, np.ndarray]:
    dates, users, neighbors = 12, 3, 2
    retrieval_window_dates = np.arange(20, 20 + dates, dtype=np.int64)
    query_users = np.arange(users, dtype=np.int64)
    neighbor_users = np.stack(
        (
            np.broadcast_to(query_users[None, :], (dates, users)),
            np.broadcast_to((query_users + 1)[None, :] % users, (dates, users)),
        ),
        axis=-1,
    )
    return {
        "retrieval_window_dates": retrieval_window_dates,
        "neighbor_window_dates": np.broadcast_to(
            retrieval_window_dates[:, None, None] - np.asarray([2, 5])[None, None, :],
            (dates, users, neighbors),
        ).copy(),
        "neighbor_user": neighbor_users,
        "neighbor_distance_raw": np.broadcast_to(
            np.asarray([0.3, 0.7], dtype=np.float32), (dates, users, neighbors)
        ).copy(),
        "neighbor_distance_instance_normalized": np.broadcast_to(
            np.asarray([0.2, 0.6], dtype=np.float32), (dates, users, neighbors)
        ).copy(),
    }


def test_diagnostics() -> None:
    normalized_window = instance_normalize_window(
        np.asarray([[0.0, 2.0, 4.0, 6.0]], dtype=np.float32), lookback=2
    )
    assert np.allclose(normalized_window, [[-1.0, 1.0, 3.0, 5.0]])
    query = np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32)
    neighbors = np.asarray([[[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]]], dtype=np.float32)
    raw, normalized = neighbor_lookback_distances(query, neighbors)
    assert np.allclose(raw, [[0.0, 5.0]])
    assert np.allclose(normalized, 0.0, atol=1e-6)

    dataset = SyntheticDataset()
    sampled = sample_setting_distances(dataset, lookback=8, horizon=4, seed=3)
    assert sampled["inter_date_raw"].size > 0
    assert sampled["inter_user_raw"].size > 0
    single_user = SyntheticDataset()
    single_user.values = single_user.values[:, :1]
    single_user.user_names = single_user.user_names[:1]
    single_sampled = sample_setting_distances(
        single_user, lookback=8, horizon=4, seed=3
    )
    assert single_sampled["inter_user_raw"].size == 0

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        setting = write_setting_diagnostics(
            root, dataset=dataset, lookback=8, horizon=4, seed=3
        )
        neighbors_outputs = write_neighbor_diagnostics(
            root,
            arrays=_neighbor_arrays(),
            user_names=dataset.user_names,
            seed=3,
        )
        assert all(path.is_file() for path in (*setting.values(), *neighbors_outputs.values()))
        summary = list(
            csv.DictReader(neighbors_outputs["neighbor_summary"].open(encoding="utf-8"))
        )
        assert {row["aggregation"] for row in summary} == {
            "all_samples",
            "per_user_average",
        }
        scope_rows = [
            row
            for row in summary
            if row["aggregation"] == "all_samples"
            and row["metric"] == "scope_fraction"
        ]
        assert {row["mean"] for row in scope_rows} == {"0.5"}
        rows = list(
            csv.DictReader(neighbors_outputs["neighbor_per_user"].open(encoding="utf-8"))
        )
        assert {row["same_user_fraction"] for row in rows} == {"0.5"}


if __name__ == "__main__":
    test_diagnostics()
