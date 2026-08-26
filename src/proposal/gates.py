"""Shared soft-Bayes and CatBoost gate computations."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.proposal.contracts import AdapterConfig
from src.proposal.extraction import ContextForecastCache
from src.proposal.ridge import (
    _aggregate_user_metrics,
    _avgy,
    _date_metrics,
    _user_date_metrics,
)


GATE_FEATURE_NAMES = (
    "candidate_delta_mean",
    "candidate_delta_std",
    "retrieved_minus_vanilla_mean",
    "retrieved_minus_vanilla_std",
    "retrieved_residual_mean",
    "retrieved_residual_std",
    "query_mean",
    "query_std",
    "neighbor_lookback_mean",
    "neighbor_lookback_std",
    "same_user_ratio",
    "neighbor_age_mean",
    "neighbor_age_std",
    "distance_mean",
    "distance_std",
    "distance_min",
)


def _candidate(
    arrays: dict[str, np.ndarray],
    date_index: int,
    name: str,
    used_k: int,
    context_cache: ContextForecastCache | None = None,
) -> np.ndarray:
    if name == "cov":
        if context_cache is None:
            return np.asarray(arrays["vanilla"][date_index], dtype=np.float64)
        return np.asarray(
            context_cache.get(date_index, used_k),
            dtype=np.float64,
        )
    if name == "avgy":
        return _avgy(arrays, date_index, used_k)
    raise ValueError(f"unknown gate candidate {name!r}")


def gate_features(
    arrays: dict[str, np.ndarray],
    date_index: int,
    candidate_name: str,
    used_k: int,
    context_cache: ContextForecastCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one feature row and candidate horizon per date-user window."""
    vanilla = np.asarray(arrays["vanilla"][date_index], dtype=np.float64)
    candidate = _candidate(arrays, date_index, candidate_name, used_k, context_cache)
    neighbor_y = np.asarray(
        arrays["neighbor_y"][date_index, :, :used_k], dtype=np.float64
    )
    neighbor_pred = np.asarray(
        arrays["neighbor_pred"][date_index, :, :used_k], dtype=np.float64
    )
    neighbor_x_mean = np.asarray(
        arrays["neighbor_x_mean"][date_index, :, :used_k], dtype=np.float64
    )
    neighbor_x_std = np.asarray(
        arrays["neighbor_x_std"][date_index, :, :used_k], dtype=np.float64
    )
    query_mean = np.asarray(arrays["query_mean"][date_index], dtype=np.float64)
    query_std = np.asarray(arrays["query_std"][date_index], dtype=np.float64)
    distances = np.asarray(
        arrays["distance"][date_index, :, :used_k], dtype=np.float64
    )
    retrieval_window_date = float(arrays["retrieval_window_dates"][date_index])
    neighbor_t = np.asarray(
        arrays["neighbor_t"][date_index, :, :used_k], dtype=np.float64
    )
    neighbor_users = np.asarray(
        arrays["neighbor_user"][date_index, :, :used_k], dtype=np.int64
    )
    users = np.arange(vanilla.shape[0], dtype=np.int64)[:, None]
    candidate_delta = candidate - vanilla
    retrieved_delta = neighbor_y - vanilla[:, None, :]
    retrieved_residual = neighbor_y - neighbor_pred
    features = np.column_stack(
        (
            candidate_delta.mean(axis=1),
            candidate_delta.std(axis=1),
            retrieved_delta.mean(axis=(1, 2)),
            retrieved_delta.std(axis=(1, 2)),
            retrieved_residual.mean(axis=(1, 2)),
            retrieved_residual.std(axis=(1, 2)),
            query_mean,
            query_std,
            neighbor_x_mean.mean(axis=1),
            np.sqrt(
                np.maximum(
                    np.mean(np.square(neighbor_x_std) + np.square(neighbor_x_mean), axis=1)
                    - np.square(neighbor_x_mean.mean(axis=1)),
                    0.0,
                )
            ),
            (neighbor_users == users).mean(axis=1),
            (retrieval_window_date - neighbor_t).mean(axis=1),
            (retrieval_window_date - neighbor_t).std(axis=1),
            distances.mean(axis=1),
            distances.std(axis=1),
            distances.min(axis=1),
        )
    )
    return features.astype(np.float32), candidate


def _advantage(
    target: np.ndarray,
    vanilla: np.ndarray,
    candidate: np.ndarray,
    scale: np.ndarray,
    fit_loss: str,
) -> np.ndarray:
    denominator = np.square(scale) if fit_loss == "nmse" else 1.0
    return np.mean(
        (np.square(target - vanilla) - np.square(target - candidate)) / denominator,
        axis=1,
    ).astype(np.float32)


def _sigmoid_score(score: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 1e-12:
        return np.where(score > 0, 1.0, np.where(score < 0, 0.0, 0.5))
    return 1.0 / (1.0 + np.exp(-np.clip(score / scale, -60.0, 60.0)))


def _catboost(config: AdapterConfig):
    try:
        from catboost import CatBoostRegressor
    except ModuleNotFoundError as exc:  # pragma: no cover - cluster dependency
        raise ModuleNotFoundError("online CatBoost requires the project catboost dependency") from exc
    return CatBoostRegressor(
        iterations=int(config.catboost_iterations),
        depth=int(config.catboost_depth),
        learning_rate=float(config.catboost_learning_rate),
        loss_function="RMSE",
        random_seed=int(config.seed),
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
    )


def _catboost_adaptor_parameter_count(model: Any) -> int:
    """Count fitted CatBoost leaf values, excluding every upstream backbone."""
    return int(np.asarray(model.get_leaf_values()).size)
