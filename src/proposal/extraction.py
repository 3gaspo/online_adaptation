"""Reusable window metadata, context forecasts, and neighbor search primitives."""

from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from src.data.neighbors import neighbor_to_query_scale, search_neighbors
from src.proposal.contracts import ExtractionConfig


class ContextForecastCache:
    """Persist only requested all-user context forecasts by retrieval window and K."""

    def __init__(
        self,
        *,
        arrays: dict[str, Any],
        output_dir: str | Path,
        model: torch.nn.Module,
        device: torch.device,
    ) -> None:
        self.arrays = arrays
        self.root = Path(output_dir).expanduser().resolve() / "context_cache"
        self.root.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.device = device
        self.retrieval_window_dates = np.asarray(
            arrays["retrieval_window_dates"], dtype=np.int64
        )
        self.users = int(len(arrays["window_users"]))
        self.horizon = int(arrays["y"].shape[-1])
        self._values: dict[tuple[int, int], np.ndarray] = {}
        self._timing: dict[tuple[int, int], float] = {}

    def get(self, retrieval_index: int, k: int) -> np.ndarray:
        key = (int(retrieval_index), int(k))
        if key in self._values:
            return self._values[key]
        window_date = int(self.retrieval_window_dates[retrieval_index])
        directory = self.root / f"K_{int(k)}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"window_{window_date}.npy"
        if path.is_file():
            value = np.load(path, allow_pickle=False)
            self._values[key] = value
            return value

        query_x = np.asarray(self.arrays["x"][retrieval_index])
        neighbor_x = np.asarray(
            self.arrays["neighbor_x"][retrieval_index, :, : int(k), :]
        )
        neighbor_y = np.asarray(
            self.arrays["neighbor_y"][retrieval_index, :, : int(k), :]
        )
        context = np.concatenate(
            (
                neighbor_to_query_scale(query_x, neighbor_x, neighbor_x),
                neighbor_to_query_scale(query_x, neighbor_x, neighbor_y),
            ),
            axis=-1,
        )
        tick = perf_counter()
        x = torch.as_tensor(
            query_x[:, None, :], dtype=torch.float32, device=self.device
        )
        context_tensor = torch.as_tensor(
            context, dtype=torch.float32, device=self.device
        )
        with torch.inference_mode():
            output = self.model(x, context=context_tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = perf_counter() - tick
        value = output.detach().cpu().squeeze(1).numpy()
        value = np.asarray(value, dtype=np.float32)
        np.save(path, value)
        self._values[key] = value
        self._timing[key] = elapsed
        return value

    def write_timing(self) -> Path:
        path = self.root / "context_forecast_timing.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "retrieval_window_date",
                    "K",
                    "context_forecast_seconds",
                    "users",
                    "amortized_seconds_per_user",
                ),
            )
            writer.writeheader()
            for (retrieval_index, k), seconds in sorted(self._timing.items()):
                writer.writerow(
                    {
                        "retrieval_window_date": int(
                            self.retrieval_window_dates[retrieval_index]
                        ),
                        "K": k,
                        "context_forecast_seconds": seconds,
                        "users": self.users,
                        "amortized_seconds_per_user": seconds / self.users,
                    }
                )
        return path


def _predict(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(x[:, None, :], dtype=torch.float32, device=device)
    with torch.inference_mode():
        value = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value.detach().cpu().squeeze(1).numpy()


def _memmap(path: Path, shape: tuple[int, ...], dtype: Any) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)


def _window_metadata(
    root: Path,
    dataset: Any,
    config: ExtractionConfig,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Store compact metadata/statistics while leaving overlapping windows in source."""
    dates = np.arange(
        config.lookback - 1,
        dataset.n_dates - config.horizon,
        dtype=np.int64,
    )
    timestamps = np.asarray(
        [str(dataset.datetimes[int(date)]) for date in dates], dtype="U64"
    )
    user_ids = np.asarray(list(dataset.user_names), dtype="U128")
    paths = {
        "window_dates": "tables/windows/dates.npy",
        "window_timestamps": "tables/windows/timestamps.npy",
        "window_users": "tables/windows/users.npy",
        "window_mean": "tables/window_statistics/lookback_mean.npy",
        "window_std": "tables/window_statistics/lookback_std.npy",
        "window_constant": "tables/window_statistics/constant.npy",
    }
    arrays: dict[str, np.ndarray] = {
        "window_dates": _memmap(root / paths["window_dates"], dates.shape, np.int64),
        "window_timestamps": _memmap(
            root / paths["window_timestamps"], timestamps.shape, timestamps.dtype
        ),
        "window_users": _memmap(
            root / paths["window_users"], user_ids.shape, user_ids.dtype
        ),
        "window_mean": _memmap(
            root / paths["window_mean"], (len(dates), dataset.n_users), np.float32
        ),
        "window_std": _memmap(
            root / paths["window_std"], (len(dates), dataset.n_users), np.float32
        ),
        "window_constant": _memmap(
            root / paths["window_constant"],
            (len(dates), dataset.n_users),
            np.bool_,
        ),
    }
    arrays["window_dates"][:] = dates
    arrays["window_timestamps"][:] = timestamps
    arrays["window_users"][:] = user_ids
    windows = np.lib.stride_tricks.sliding_window_view(
        dataset.values,
        config.lookback + config.horizon,
        axis=0,
    )
    for start in range(0, len(dates), 256):
        stop = min(start + 256, len(dates))
        lookback = windows[start:stop, :, : config.lookback]
        mean = lookback.mean(axis=-1)
        std = lookback.std(axis=-1)
        arrays["window_mean"][start:stop] = mean
        arrays["window_std"][start:stop] = std
        arrays["window_constant"][start:stop] = std <= 1e-8
    return arrays, paths


def _scoped_neighbors(
    query_features: np.ndarray,
    store_features: np.ndarray,
    store_users: np.ndarray,
    *,
    scope: str,
    k: int,
    metric: str,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if scope == "all":
        return search_neighbors(
            query_features, store_features, k=k, metric=metric, chunk_size=chunk_size
        )
    users = len(query_features)
    distance = np.empty((users, k), dtype=np.float32)
    indices = np.empty((users, k), dtype=np.int64)
    for user in range(users):
        allowed = np.flatnonzero(
            store_users == user if scope == "same_user" else store_users != user
        )
        if len(allowed) < k:
            raise ValueError(
                f"scope={scope} has fewer than K={k} candidates for user {user}"
            )
        local_distance, local_index = search_neighbors(
            query_features[user : user + 1],
            store_features[allowed],
            k=k,
            metric=metric,
            chunk_size=chunk_size,
        )
        distance[user] = local_distance[0]
        indices[user] = allowed[local_index[0]]
    return distance, indices
