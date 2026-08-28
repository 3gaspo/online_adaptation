"""Extraction planning, compact caches, artifacts, and diagnostics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from src.data.neighbors import build_window_batch
from src.pipeline.contracts import write_array_manifest
from src.proposal.contracts import ExtractionConfig
from src.proposal.datastore import candidate_dates, fitting_dates
from src.proposal.date_planning import build_date_plan
from src.proposal.extraction import _memmap, _predict, _scoped_neighbors, _window_metadata
from src.results.diagnostics import (
    neighbor_lookback_distances,
    write_neighbor_diagnostics,
    write_setting_diagnostics,
)


LOGGER = logging.getLogger(__name__)


def _batch_features(
    dataset: Any,
    dates: np.ndarray,
    *,
    config: ExtractionConfig,
    model: torch.nn.Module,
    representation_model: torch.nn.Module | None,
    device: torch.device,
    representation_batch_size: int,
) -> np.ndarray:
    batch = build_window_batch(
        dataset,
        dates,
        lags=config.lookback,
        horizon=config.horizon,
        distance_space=config.distance_space,
        model=model,
        representation_model=representation_model,
        device=device,
        representation_batch_size=representation_batch_size,
    )
    return batch.features.reshape(dataset.n_users, len(dates), -1).transpose(1, 0, 2)


def extract_online_features(
    *,
    dataset: Any,
    model: torch.nn.Module,
    config: ExtractionConfig,
    output_dir: str | Path,
    device: torch.device,
    representation_model: torch.nn.Module | None = None,
    search_chunk_size: int = 512,
    representation_batch_size: int = 512,
) -> dict[str, Path]:
    """Plan unique retrieval windows and materialize only reusable computations."""
    config.validate()
    started = perf_counter()
    root = Path(output_dir).expanduser().resolve()
    arrays, paths = _window_metadata(root, dataset, config)
    metadata_and_statistics_seconds = perf_counter() - started
    window_dates = np.asarray(arrays["window_dates"], dtype=np.int64)
    date_to_position = {int(date): index for index, date in enumerate(window_dates)}
    windows = np.lib.stride_tricks.sliding_window_view(
        dataset.values,
        config.lookback + config.horizon,
        axis=0,
    )
    window_lookback = windows[:, :, : config.lookback]

    date_plan = build_date_plan(
        n_dates=dataset.n_dates,
        n_users=dataset.n_users,
        config=config,
    )
    candidates_per_query = int(date_plan.n_datastore_dates) * (
        1
        if config.retrieval_scope == "same_user"
        else dataset.n_users - 1
        if config.retrieval_scope == "other_users"
        else dataset.n_users
    )
    if candidates_per_query < int(config.max_k):
        raise ValueError(
            f"datastore supplies {candidates_per_query} candidates per query, "
            f"fewer than max_k={config.max_k}"
        )
    first_retrieval = int(date_plan.first_retrieval_date)
    evaluation_query_dates = np.asarray(
        date_plan.evaluation_query_dates, dtype=np.int64
    )

    required = np.zeros(len(window_dates), dtype=np.bool_)
    fitting_by_query: dict[int, np.ndarray] = {}
    for query_date_raw in evaluation_query_dates:
        query_date = int(query_date_raw)
        required[date_to_position[query_date]] = True
        if config.include_fitting_windows:
            selected = fitting_dates(query_date, config=config, plan=date_plan)
            if len(selected) != date_plan.n_fit or int(selected[0]) < first_retrieval:
                raise ValueError(f"query {query_date} lacks its complete fitting grid")
            fitting_by_query[query_date] = selected
            required[[date_to_position[int(date)] for date in selected]] = True
    retrieval_window_dates = window_dates[np.flatnonzero(required)]
    retrieval_index = {
        int(date): index for index, date in enumerate(retrieval_window_dates)
    }
    evaluation_set = set(map(int, evaluation_query_dates))
    r_count = len(retrieval_window_dates)
    u_count = dataset.n_users
    k_count = config.max_k

    retrieval_specs: dict[str, tuple[tuple[int, ...], Any, str]] = {
        "retrieval_window_dates": (
            (r_count,),
            np.int64,
            "tables/retrieval_neighbors/retrieval_window_dates.npy",
        ),
        "is_evaluation_query": (
            (r_count,),
            np.bool_,
            "tables/retrieval_neighbors/is_evaluation_query.npy",
        ),
        "neighbor_window_id": (
            (r_count, u_count, k_count),
            np.int64,
            "tables/retrieval_neighbors/neighbor_window_id.npy",
        ),
        "distance": (
            (r_count, u_count, k_count),
            np.float32,
            "tables/retrieval_neighbors/distance.npy",
        ),
        "neighbor_distance_raw": (
            (r_count, u_count, k_count),
            np.float32,
            "tables/retrieval_neighbors/raw_distance.npy",
        ),
        "neighbor_distance_instance_normalized": (
            (r_count, u_count, k_count),
            np.float32,
            "tables/retrieval_neighbors/instance_distance.npy",
        ),
        "candidate_count": (
            (r_count,),
            np.int64,
            "tables/retrieval_neighbors/candidate_count.npy",
        ),
    }
    for name, (shape, dtype, relative) in retrieval_specs.items():
        arrays[name] = _memmap(root / relative, shape, dtype)
        paths[name] = relative
    arrays["retrieval_window_dates"][:] = retrieval_window_dates
    arrays["is_evaluation_query"][:] = [
        int(date) in evaluation_set for date in retrieval_window_dates
    ]

    representation: np.ndarray | None = None
    representation_ready: np.ndarray | None = None
    representation_lookup: np.ndarray | None = None
    representation_bootstrap_seconds = 0.0
    if config.distance_space not in {"raw", "instance"}:
        representation_required = np.zeros(len(window_dates), dtype=np.bool_)
        for retrieval_date_raw in retrieval_window_dates:
            retrieval_date = int(retrieval_date_raw)
            dates = candidate_dates(retrieval_date, config=config, plan=date_plan)
            representation_required[date_to_position[retrieval_date]] = True
            representation_required[
                [date_to_position[int(date)] for date in dates]
            ] = True
        representation_positions = np.flatnonzero(representation_required)
        bootstrap_started = perf_counter()
        sample = _batch_features(
            dataset,
            window_dates[representation_positions[:1]],
            config=config,
            model=model,
            representation_model=representation_model,
            device=device,
            representation_batch_size=representation_batch_size,
        )
        representation_bootstrap_seconds = perf_counter() - bootstrap_started
        dimension = int(sample.shape[-1])
        paths["representation_window_dates"] = (
            "tables/window_computation/representation_window_dates.npy"
        )
        paths["representation_value"] = (
            "tables/window_computation/representation_value.npy"
        )
        arrays["representation_window_dates"] = _memmap(
            root / paths["representation_window_dates"],
            (len(representation_positions),),
            np.int64,
        )
        arrays["representation_window_dates"][:] = window_dates[
            representation_positions
        ]
        representation = _memmap(
            root / paths["representation_value"],
            (len(representation_positions), u_count, dimension),
            np.float32,
        )
        arrays["representation_value"] = representation
        representation_ready = np.zeros(len(representation_positions), dtype=np.bool_)
        representation[0] = sample[0]
        representation_ready[0] = True
        representation_lookup = np.full(len(window_dates), -1, dtype=np.int64)
        representation_lookup[representation_positions] = np.arange(
            len(representation_positions), dtype=np.int64
        )

    def features(dates: np.ndarray) -> np.ndarray:
        if representation is None:
            return _batch_features(
                dataset,
                dates,
                config=config,
                model=model,
                representation_model=representation_model,
                device=device,
                representation_batch_size=representation_batch_size,
            )
        assert representation_ready is not None and representation_lookup is not None
        window_positions = np.asarray(
            [date_to_position[int(date)] for date in dates], dtype=np.int64
        )
        compact_positions = representation_lookup[window_positions]
        missing = np.unique(compact_positions[~representation_ready[compact_positions]])
        for start in range(0, len(missing), 128):
            selected = missing[start : start + 128]
            selected_dates = np.asarray(arrays["representation_window_dates"])[selected]
            representation[selected] = _batch_features(
                dataset,
                selected_dates,
                config=config,
                model=model,
                representation_model=representation_model,
                device=device,
                representation_batch_size=representation_batch_size,
            )
            representation_ready[selected] = True
        return np.ascontiguousarray(np.asarray(representation[compact_positions]))

    forecast_required = np.zeros(len(window_dates) * u_count, dtype=np.bool_)

    def process(retrieval_date: int) -> None:
        row = retrieval_index[retrieval_date]
        store_dates = candidate_dates(retrieval_date, config=config, plan=date_plan)
        if not len(store_dates):
            raise ValueError(f"retrieval window {retrieval_date} has an empty datastore")
        all_features = features(
            np.concatenate((np.asarray([retrieval_date], dtype=np.int64), store_dates))
        )
        query_features = all_features[0]
        store_by_date = all_features[1:]
        store_features = np.ascontiguousarray(
            np.transpose(store_by_date, (1, 0, 2)).reshape(
                -1, store_by_date.shape[-1]
            )
        )
        store_users = np.repeat(np.arange(u_count, dtype=np.int64), len(store_dates))
        if config.distance_space == "tsrag":
            if representation_model is None:
                raise ValueError("TS-RAG retrieval requires its retriever")
            distance = np.empty((u_count, k_count), dtype=np.float32)
            local_indices = np.empty((u_count, k_count), dtype=np.int64)
            for user in range(u_count):
                allowed = np.flatnonzero(store_users == user)
                d, i = representation_model.search(
                    query_features[user : user + 1],
                    store_features[allowed],
                    top_k=k_count,
                )
                distance[user], local_indices[user] = d[0], allowed[i[0]]
        else:
            distance, local_indices = _scoped_neighbors(
                query_features,
                store_features,
                store_users,
                scope=config.retrieval_scope,
                k=k_count,
                metric=config.distance_metric,
                chunk_size=search_chunk_size,
            )
        local_users, local_date_positions = np.divmod(
            local_indices, len(store_dates)
        )
        selected_positions = np.asarray(
            [
                date_to_position[int(date)]
                for date in store_dates[local_date_positions.reshape(-1)]
            ],
            dtype=np.int64,
        ).reshape(local_date_positions.shape)
        neighbor_ids = selected_positions * u_count + local_users
        retrieval_position = date_to_position[retrieval_date]
        retrieval_ids = retrieval_position * u_count + np.arange(
            u_count, dtype=np.int64
        )
        forecast_required[retrieval_ids] = True
        forecast_required[neighbor_ids.reshape(-1)] = True
        raw_x = np.asarray(window_lookback[selected_positions, local_users])
        retrieval_x = np.asarray(window_lookback[retrieval_position])
        raw_distance, instance_distance = neighbor_lookback_distances(
            retrieval_x, raw_x
        )
        arrays["neighbor_window_id"][row] = neighbor_ids
        arrays["distance"][row] = distance
        arrays["neighbor_distance_raw"][row] = raw_distance
        arrays["neighbor_distance_instance_normalized"][row] = instance_distance
        arrays["candidate_count"][row] = (
            len(store_dates)
            if config.retrieval_scope == "same_user"
            else len(store_dates)
            * (u_count - 1 if config.retrieval_scope == "other_users" else u_count)
        )

    cold_query = int(evaluation_query_dates[0])
    cold_dates = np.unique(
        np.concatenate(
            (
                fitting_by_query.get(cold_query, np.empty(0, dtype=np.int64)),
                np.asarray([cold_query], dtype=np.int64),
            )
        )
    )
    cold_set = set(map(int, cold_dates))
    processing_order = [
        *map(int, cold_dates),
        *(
            int(date)
            for date in retrieval_window_dates
            if int(date) not in cold_set
        ),
    ]
    cold_started = perf_counter()
    cold_retrieval_seconds = 0.0
    for index, retrieval_date in enumerate(processing_order):
        process(retrieval_date)
        if index + 1 == len(cold_dates):
            cold_retrieval_seconds = (
                representation_bootstrap_seconds + perf_counter() - cold_started
            )
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(processing_order):
            LOGGER.info(
                "compact extraction windows=%s/%s date=%s",
                index + 1,
                len(processing_order),
                retrieval_date,
            )

    forecast_ids = np.flatnonzero(forecast_required).astype(np.int64)
    cold_rows = np.asarray(
        [retrieval_index[int(date)] for date in cold_dates], dtype=np.int64
    )
    cold_forecast_ids = np.unique(
        np.concatenate(
            (
                (
                    np.asarray([date_to_position[int(date)] for date in cold_dates])[
                        :, None
                    ]
                    * u_count
                    + np.arange(u_count, dtype=np.int64)[None, :]
                ).reshape(-1),
                np.asarray(arrays["neighbor_window_id"])[cold_rows].reshape(-1),
            )
        )
    )
    cold_forecast_set = set(map(int, cold_forecast_ids))
    forecast_order = np.asarray(
        [
            *map(int, cold_forecast_ids),
            *(int(value) for value in forecast_ids if int(value) not in cold_forecast_set),
        ],
        dtype=np.int64,
    )
    forecast_values_by_order = np.empty(
        (len(forecast_order), config.horizon), dtype=np.float32
    )
    forecast_started = perf_counter()
    for start in range(0, len(cold_forecast_ids), representation_batch_size):
        stop = min(start + representation_batch_size, len(cold_forecast_ids))
        selected = forecast_order[start:stop]
        positions, users = np.divmod(selected, u_count)
        forecast_values_by_order[start:stop] = _predict(
            model,
            np.asarray(window_lookback[positions, users]),
            device,
        )
    cold_forecast_seconds = perf_counter() - forecast_started
    for start in range(
        len(cold_forecast_ids), len(forecast_order), representation_batch_size
    ):
        stop = min(start + representation_batch_size, len(forecast_order))
        selected = forecast_order[start:stop]
        positions, users = np.divmod(selected, u_count)
        forecast_values_by_order[start:stop] = _predict(
            model,
            np.asarray(window_lookback[positions, users]),
            device,
        )
    order = np.argsort(forecast_order)
    paths["forecast_window_id"] = "tables/window_computation/forecast_window_id.npy"
    paths["forecast_value"] = "tables/window_computation/forecast_value.npy"
    arrays["forecast_window_id"] = _memmap(
        root / paths["forecast_window_id"], (len(forecast_order),), np.int64
    )
    arrays["forecast_value"] = _memmap(
        root / paths["forecast_value"],
        (len(forecast_order), config.horizon),
        np.float32,
    )
    arrays["forecast_window_id"][:] = forecast_order[order]
    arrays["forecast_value"][:] = forecast_values_by_order[order]

    for value in arrays.values():
        if hasattr(value, "flush"):
            value.flush()
    setting_outputs = write_setting_diagnostics(
        root,
        dataset=dataset,
        lookback=config.lookback,
        horizon=config.horizon,
        seed=config.seed,
    )
    neighbor_ids = np.asarray(arrays["neighbor_window_id"])
    neighbor_positions, neighbor_users = np.divmod(neighbor_ids, u_count)
    neighbor_outputs = write_neighbor_diagnostics(
        root,
        arrays={
            "retrieval_window_dates": arrays["retrieval_window_dates"],
            "neighbor_user": neighbor_users,
            "neighbor_window_dates": window_dates[neighbor_positions],
            "neighbor_distance_raw": arrays["neighbor_distance_raw"],
            "neighbor_distance_instance_normalized": arrays[
                "neighbor_distance_instance_normalized"
            ],
        },
        user_names=list(dataset.user_names),
        seed=config.seed,
    )
    total_extraction_seconds = perf_counter() - started
    cold_extraction_seconds = (
        metadata_and_statistics_seconds
        + cold_retrieval_seconds
        + cold_forecast_seconds
    )
    metadata = {
        "users": u_count,
        "user_names": list(dataset.user_names),
        "source_windows": int(len(window_dates) * u_count),
        "window_id": "window_date_position * users + user_position",
        "retrieval_window_rows": int(r_count * u_count),
        "evaluation_query_batches": int(len(evaluation_query_dates)),
        "date_plan": date_plan.scientific_dict(),
        "computed_forecasts": int(len(forecast_order)),
        "total_extraction_seconds": total_extraction_seconds,
        "cold_batch": {
            "evaluation_query_date": cold_query,
            "retrieval_windows": int(len(cold_dates)),
            "metadata_and_statistics_seconds": metadata_and_statistics_seconds,
            "retrieval_and_representation_seconds": cold_retrieval_seconds,
            "forecast_seconds": cold_forecast_seconds,
            "extraction_seconds": cold_extraction_seconds,
        },
    }
    manifest = write_array_manifest(
        root,
        config=config.scientific_dict(),
        arrays=arrays,
        metadata=metadata,
        array_paths=paths,
    )
    timing = root / "extraction_timing.json"
    timing.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"manifest": manifest, "timing": timing, **setting_outputs, **neighbor_outputs}
