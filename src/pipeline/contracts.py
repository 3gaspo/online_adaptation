"""Artifact contracts for online extraction and adaptation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXTRACTION_FORMAT = "online_extraction_v1"
RESULT_FORMAT = "online_adaptation_v1"


def adaptor_parameter_metadata(count: int, definition: str) -> dict[str, Any]:
    """Describe fitted adaptor parameters without counting the backbone."""
    if int(count) < 0:
        raise ValueError("adaptor parameter count must be non-negative")
    return {
        "adaptor": int(count),
        "backbone_included": False,
        "definition": str(definition),
    }


def count_named_parameters(
    parameters: Any,
    *,
    prefixes: tuple[str, ...],
) -> int:
    """Count tensor elements whose names belong to declared adaptor modules."""
    return sum(
        int(parameter.numel())
        for name, parameter in parameters
        if str(name).startswith(prefixes)
    )

EXTRACTION_ARRAYS = (
    "window_dates",
    "window_timestamps",
    "window_users",
    "window_mean",
    "window_std",
    "window_constant",
    "forecast_window_id",
    "forecast_value",
    "retrieval_window_dates",
    "is_evaluation_query",
    "neighbor_window_id",
    "distance",
    "neighbor_distance_raw",
    "neighbor_distance_instance_normalized",
    "candidate_count",
)
def write_array_manifest(
    run_dir: str | Path,
    *,
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    array_paths: Mapping[str, str] | None = None,
) -> Path:
    root = Path(run_dir)
    payload = {
        "format": EXTRACTION_FORMAT,
        "config": dict(config),
        "metadata": dict(metadata),
        "arrays": {
            name: {
                "path": (
                    dict(array_paths)[name]
                    if array_paths is not None
                    else f"features/{name}.npy"
                ),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in arrays.items()
        },
    }
    path = root / "online_extraction_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def load_array_manifest(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    path = root / "online_extraction_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != EXTRACTION_FORMAT:
        raise ValueError(f"unsupported online extraction manifest: {path}")
    return payload


def open_extraction_arrays(
    run_dir: str | Path,
    *,
    dataset: Any | None = None,
) -> dict[str, np.ndarray]:
    root = Path(run_dir).expanduser().resolve()
    manifest = load_array_manifest(root)
    missing = [name for name in EXTRACTION_ARRAYS if name not in manifest["arrays"]]
    if missing:
        raise ValueError(f"online extraction is missing arrays: {missing}")
    arrays: dict[str, Any] = {
        name: np.load(root / values["path"], mmap_mode="r", allow_pickle=False)
        for name, values in manifest["arrays"].items()
    }
    physical = dict(arrays)
    retrieval_dates = np.asarray(physical["retrieval_window_dates"], dtype=np.int64)
    window_dates = np.asarray(physical["window_dates"], dtype=np.int64)
    positions = np.searchsorted(window_dates, retrieval_dates)
    users = int(len(physical["window_users"]))
    neighbors = int(physical["neighbor_window_id"].shape[2])
    if dataset is None:
        return arrays
    config = manifest["config"]
    lookback = int(config["lookback"])
    horizon = int(config["horizon"])
    if dataset.n_users != users or dataset.n_dates < int(window_dates[-1]) + horizon + 1:
        raise ValueError("dataset does not match the extraction window contract")
    windows = np.lib.stride_tricks.sliding_window_view(
        dataset.values,
        lookback + horizon,
        axis=0,
    )
    window_lookback = windows[:, :, :lookback]
    window_horizon = windows[:, :, lookback:]

    class VirtualArray:
        def __init__(self, shape: tuple[int, ...], build: Any) -> None:
            self.shape = shape
            self.ndim = len(shape)
            self.dtype = np.dtype(np.float32)
            self._build = build

        def __getitem__(self, index: Any) -> np.ndarray:
            if isinstance(index, tuple):
                first, rest = index[0], index[1:]
            else:
                first, rest = index, ()
            value = np.asarray(self._build(first))
            return value[(..., *rest)] if rest else value

        def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
            value = np.asarray(self._build(slice(None)), dtype=dtype)
            return value.copy() if copy else value

    def query_values(source: str) -> Any:
        return lambda index: np.asarray(physical[source][positions[index]])

    def neighbor_values(source: np.ndarray) -> Any:
        def build(index: Any) -> np.ndarray:
            ids = np.asarray(physical["neighbor_window_id"][index], dtype=np.int64)
            window_position, window_user = np.divmod(ids, users)
            return np.asarray(source[window_position, window_user])

        return build

    forecast_ids = np.asarray(physical["forecast_window_id"], dtype=np.int64)
    forecast_values = physical["forecast_value"]

    def forecasts(ids: np.ndarray) -> np.ndarray:
        requested = np.asarray(ids, dtype=np.int64)
        locations = np.searchsorted(forecast_ids, requested)
        if np.any(locations >= len(forecast_ids)) or np.any(
            forecast_ids[np.minimum(locations, len(forecast_ids) - 1)] != requested
        ):
            raise ValueError("extraction is missing a required sparse vanilla forecast")
        return np.asarray(forecast_values[locations])

    retrieval_window_ids = (
        positions[:, None] * users + np.arange(users, dtype=np.int64)[None, :]
    )

    def query_forecasts(index: Any) -> np.ndarray:
        return forecasts(retrieval_window_ids[index])

    def neighbor_forecasts(index: Any) -> np.ndarray:
        return forecasts(np.asarray(physical["neighbor_window_id"][index]))

    q_shape = (len(retrieval_dates), users)
    arrays.update(
        {
            "window_lookback": window_lookback,
            "window_horizon": window_horizon,
            "query_mean": VirtualArray(q_shape, query_values("window_mean")),
            "query_std": VirtualArray(q_shape, query_values("window_std")),
            "y": VirtualArray(
                (*q_shape, horizon), lambda index: window_horizon[positions[index]]
            ),
            "vanilla": VirtualArray((*q_shape, horizon), query_forecasts),
            "x": VirtualArray(
                (*q_shape, lookback), lambda index: window_lookback[positions[index]]
            ),
            "neighbor_x": VirtualArray(
                (*q_shape, neighbors, lookback), neighbor_values(window_lookback)
            ),
            "neighbor_t": VirtualArray(
                (*q_shape, neighbors),
                lambda index: window_dates[
                    np.asarray(physical["neighbor_window_id"][index], dtype=np.int64)
                    // users
                ],
            ),
            "neighbor_user": VirtualArray(
                (*q_shape, neighbors),
                lambda index: np.asarray(physical["neighbor_window_id"][index])
                % users,
            ),
            "store_window_count": physical["candidate_count"],
        }
    )

    physical_neighbor_x = neighbor_values(window_lookback)
    physical_neighbor_y = neighbor_values(window_horizon)
    physical_neighbor_pred = neighbor_forecasts

    def scaled_neighbor(index: Any, build: Any) -> np.ndarray:
        query_x = np.asarray(window_lookback[positions[index]])
        neighbor_x = np.asarray(physical_neighbor_x(index))
        value = np.asarray(build(index))
        query_mean = query_x.mean(axis=-1, keepdims=True)[..., None, :]
        query_std = np.maximum(query_x.std(axis=-1, keepdims=True), 1e-8)[..., None, :]
        neighbor_mean = neighbor_x.mean(axis=-1, keepdims=True)
        neighbor_std = np.maximum(neighbor_x.std(axis=-1, keepdims=True), 1e-8)
        return (value - neighbor_mean) / neighbor_std * query_std + query_mean

    scaled_x = VirtualArray(
        (*q_shape, neighbors, lookback), lambda index: scaled_neighbor(index, physical_neighbor_x)
    )
    arrays["neighbor_x_mean"] = VirtualArray(
        (*q_shape, neighbors), lambda index: np.asarray(scaled_x[index]).mean(axis=-1)
    )
    arrays["neighbor_x_std"] = VirtualArray(
        (*q_shape, neighbors), lambda index: np.asarray(scaled_x[index]).std(axis=-1)
    )
    arrays["neighbor_y"] = VirtualArray(
        (*q_shape, neighbors, horizon), lambda index: scaled_neighbor(index, physical_neighbor_y)
    )
    arrays["neighbor_pred"] = VirtualArray(
        (*q_shape, neighbors, horizon), lambda index: scaled_neighbor(index, physical_neighbor_pred)
    )
    return arrays
