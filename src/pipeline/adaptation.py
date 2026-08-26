"""Artifact-producing orchestration for online ridge and gate evaluation."""

from __future__ import annotations

from collections import deque
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from src.pipeline.contracts import (
    RESULT_FORMAT,
    adaptor_parameter_metadata,
    load_array_manifest,
    open_extraction_arrays,
)
from src.proposal.contracts import AdapterConfig, ExtractionConfig
from src.proposal.datastore import fitting_dates
from src.proposal.extraction import ContextForecastCache
from src.proposal.gates import (
    GATE_FEATURE_NAMES,
    _advantage,
    _candidate,
    _catboost,
    _catboost_adaptor_parameter_count,
    _sigmoid_score,
    gate_features,
)
from src.proposal.ridge import (
    LinearStatistics,
    _aggregate,
    _aggregate_user_metrics,
    _date_metrics,
    _iter_rows,
    _predict,
    _solve,
    _user_date_metrics,
    design_feature_names,
    design_tensor,
    parse_method,
)
from src.results.efficiency import write_compute_timing
from src.visualization.adaptation import plot_coefficient_summary, plot_gate_importance

def evaluate_online_linear(
    *,
    extraction_dir: str | Path,
    output_dir: str | Path,
    config: AdapterConfig,
    dataset: Any,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    eval_start_date: int | None = None,
    eval_end_date: int | None = None,
) -> dict[str, Path]:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_started = perf_counter()
    config.validate()
    design_name, formulation, mode = parse_method(config.method)
    arrays = open_extraction_arrays(extraction_dir, dataset=dataset)
    retrieval_window_dates = np.asarray(arrays["retrieval_window_dates"])
    n_dates, n_users, horizon = arrays["y"].shape
    available_neighbors = int(arrays["neighbor_y"].shape[2])
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    context_cache = (
        ContextForecastCache(
            arrays=arrays,
            output_dir=output,
            model=model,
            device=device or torch.device("cpu"),
        )
        if model is not None
        else None
    )
    candidate_ks = (
        tuple(sorted(int(value) for value in config.candidate_k_grid))
        if config.used_k is None
        else (int(config.used_k),)
    )
    candidate_alphas = (
        tuple(float(value) for value in config.alpha_grid)
        if config.tune_alpha
        else (float(config.alpha),)
    )
    selection_enabled = config.tune_alpha or config.used_k is None
    if max(candidate_ks) > available_neighbors:
        raise ValueError(
            f"ridge K selection requires {max(candidate_ks)} neighbors, "
            f"but extraction contains {available_neighbors}"
        )
    rows_per_date = n_users if config.fitting_scope == "all" else 1
    fitting_capacity = config.n_fit * rows_per_date
    validation_dates = max(1, int(np.ceil(config.n_fit * config.validation_ratio)))
    validation_rows = (
        validation_dates * rows_per_date
        if selection_enabled
        else 0
    )
    training_rows = fitting_capacity - validation_rows
    if selection_enabled and training_rows <= 0:
        raise ValueError("validation split leaves no ridge-selection training rows")
    max_k = max(candidate_ks)
    feature_names = design_feature_names(design_name, formulation, max_k)
    convex = formulation == "convex"

    def new_statistics(k: int) -> LinearStatistics:
        return LinearStatistics(
            horizon=horizon,
            features=len(design_feature_names(design_name, formulation, k)),
            mode=mode,
            convex=convex,
        )

    scope_keys = (
        tuple(range(n_users))
        if config.fitting_scope == "same_user"
        else (0,)
    )
    extraction_config = ExtractionConfig(**load_array_manifest(extraction_dir)["config"])
    date_to_index = {
        int(date): index for index, date in enumerate(retrieval_window_dates)
    }
    trackers: dict[int, dict[str, Any]] = {}
    active_date_counts: dict[int, int] = {}
    cached_designs: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    metric_rows: list[dict[str, Any]] = []
    user_metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[np.ndarray] = []
    coefficient_dates: list[int] = []
    selection_rows: list[dict[str, Any]] = []
    cold_adaptation_seconds: float | None = None

    def values(date_index: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        key = (date_index, k)
        if key not in cached_designs:
            x_value, y_value, _ = design_tensor(
                arrays,
                date_index,
                design=design_name,
                formulation=formulation,
                neighbors=k,
                context_cache=context_cache,
            )
            cached_designs[key] = (x_value, y_value)
        return cached_designs[key]

    def update(
        statistics: dict[int, LinearStatistics],
        date_index: int,
        user: int,
        sign: int,
    ) -> None:
        scale = 1.0
        if config.fit_loss == "nmse":
            scale = max(float(arrays["query_std"][date_index, user]), 1e-8)
        for k, candidate_statistics in statistics.items():
            x_value, y_value = values(date_index, k)
            candidate_statistics.update(x_value[user], y_value[user], scale=scale, sign=sign)

    def statistics_bundle() -> dict[int, dict[int, LinearStatistics]]:
        return {
            scope: {k: new_statistics(k) for k in candidate_ks}
            for scope in scope_keys
        }

    def update_date(
        bundle: dict[int, dict[int, LinearStatistics]],
        date_index: int,
        sign: int,
    ) -> None:
        for user in range(n_users):
            scope = user if config.fitting_scope == "same_user" else 0
            update(bundle[scope], date_index, user, sign)

    def reconcile(
        tracker: dict[str, Any],
        desired: tuple[int, ...],
    ) -> None:
        if config.fit_mode == "fixed" and tracker["frozen"]:
            return
        validation_date_count = validation_dates if selection_enabled else 0
        desired_full = set(desired)
        desired_train = set(desired[:-validation_date_count] if validation_date_count else desired)
        desired_validation = set(desired[-validation_date_count:]) if validation_date_count else set()
        for index in tracker["full_dates"] - desired_full:
            update_date(tracker["full"], index, -1)
            active_date_counts[index] -= 1
        for index in desired_full - tracker["full_dates"]:
            update_date(tracker["full"], index, +1)
            active_date_counts[index] = active_date_counts.get(index, 0) + 1
        if selection_enabled:
            for index in tracker["train_dates"] - desired_train:
                update_date(tracker["train"], index, -1)
            for index in desired_train - tracker["train_dates"]:
                update_date(tracker["train"], index, +1)
            for index in tracker["validation_dates"] - desired_validation:
                update_date(tracker["validation"], index, -1)
            for index in desired_validation - tracker["validation_dates"]:
                update_date(tracker["validation"], index, +1)
        tracker["full_dates"] = desired_full
        tracker["train_dates"] = desired_train
        tracker["validation_dates"] = desired_validation
        tracker["frozen"] = config.fit_mode == "fixed" and len(desired) == config.n_fit
        for index in tuple(active_date_counts):
            if active_date_counts[index] == 0:
                active_date_counts.pop(index)
                for k in candidate_ks:
                    cached_designs.pop((index, k), None)

    cold_setup_seconds = perf_counter() - evaluation_started
    for current_index, current_date_raw in enumerate(retrieval_window_dates):
        current_date = int(current_date_raw)
        if not bool(arrays["is_evaluation_query"][current_index]):
            continue
        if eval_start_date is not None and current_date < int(eval_start_date):
            continue
        if eval_end_date is not None and current_date > int(eval_end_date):
            continue
        adaptation_started = perf_counter()
        fitting_date_values = fitting_dates(
            current_date, config=extraction_config, n_fit=config.n_fit
        )
        desired = tuple(
            date_to_index[int(date)]
            for date in fitting_date_values
            if int(date) in date_to_index
        )
        if len(desired) < config.n_fit:
            continue
        grid_key = int(fitting_date_values[-1] % extraction_config.fit_stride)
        tracker = trackers.setdefault(
            grid_key,
            {
                "full": statistics_bundle(),
                "train": statistics_bundle() if selection_enabled else {},
                "validation": statistics_bundle() if selection_enabled else {},
                "full_dates": set(),
                "train_dates": set(),
                "validation_dates": set(),
                "frozen": False,
            },
        )
        reconcile(tracker, desired)
        full_statistics = tracker["full"]
        selection_train = tracker["train"]
        selection_validation = tracker["validation"]
        padded_coefficients: list[np.ndarray] = []
        selected_ks: list[int] = []
        for scope in scope_keys:
            if selection_enabled:
                choices: list[tuple[float, int, float]] = []
                for k in candidate_ks:
                    for alpha in candidate_alphas:
                        candidate_coefficients = selection_train[scope][k].solve(
                            float(alpha)
                        )
                        loss = selection_validation[scope][k].mean_squared_error(
                            candidate_coefficients
                        )
                        choices.append((loss, k, float(alpha)))
                validation_loss, selected_k, selected_alpha = min(
                    choices,
                    key=lambda item: (item[0], item[1], -item[2]),
                )
            else:
                selected_k = candidate_ks[0]
                selected_alpha = candidate_alphas[0]
                validation_loss = float("nan")
            coefficients = full_statistics[scope][selected_k].solve(selected_alpha)
            selected_names = design_feature_names(
                design_name, formulation, selected_k
            )
            padded = (
                np.zeros(len(feature_names), dtype=np.float64)
                if mode == "shared"
                else np.zeros((horizon, len(feature_names)), dtype=np.float64)
            )
            selected_columns = [feature_names.index(name) for name in selected_names]
            if mode == "shared":
                padded[selected_columns] = coefficients
            else:
                padded[:, selected_columns] = coefficients
            padded_coefficients.append(padded)
            selected_ks.append(selected_k)
            selection_row = {
                    "query_date": current_date,
                    "selected_alpha": selected_alpha,
                    "selected_k": selected_k,
                    "validation_loss": validation_loss,
                    "adaptor_parameters": len(selected_names)
                    * (horizon if mode == "horizon" else 1),
                }
            if config.fitting_scope == "same_user":
                selection_row["user_id"] = int(scope)
            selection_rows.append(selection_row)
        coefficients = (
            np.stack(padded_coefficients)
            if config.fitting_scope == "same_user"
            else padded_coefficients[0]
        )
        vanilla = np.asarray(arrays["vanilla"][current_index], dtype=np.float64)
        target = np.asarray(arrays["y"][current_index], dtype=np.float64)
        prediction = np.empty_like(vanilla)
        prediction_scopes = range(n_users) if config.fitting_scope == "same_user" else (0,)
        for position, scope in enumerate(prediction_scopes):
            k = selected_ks[position]
            design_value, _, selected_names = design_tensor(
                arrays,
                current_index,
                design=design_name,
                formulation=formulation,
                neighbors=k,
                context_cache=context_cache,
            )
            selected_columns = [feature_names.index(name) for name in selected_names]
            if config.fitting_scope == "same_user":
                selected_coefficients = coefficients[scope][..., selected_columns]
                prediction[scope] = _predict(
                    vanilla[scope : scope + 1],
                    design_value[scope : scope + 1],
                    selected_coefficients[None, ...],
                    mode=mode,
                    convex=convex,
                    fitting_scope="same_user",
                )[0]
            else:
                selected_coefficients = coefficients[..., selected_columns]
                prediction = _predict(
                    vanilla,
                    design_value,
                    selected_coefficients,
                    mode=mode,
                    convex=convex,
                    fitting_scope="all",
                )
        scale = np.maximum(
            np.asarray(arrays["query_std"][current_index], dtype=np.float64)[:, None],
            1e-8,
        )
        metric_rows.append(
            _date_metrics(current_date, config.method, prediction, target, vanilla, scale)
        )
        user_metric_rows.extend(
            _user_date_metrics(current_date, config.method, prediction, target, vanilla, scale)
        )
        coefficient_rows.append(coefficients)
        coefficient_dates.append(current_date)
        if cold_adaptation_seconds is None:
            cold_adaptation_seconds = (
                cold_setup_seconds + perf_counter() - adaptation_started
            )
    metrics = _aggregate_user_metrics(user_metric_rows)
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    per_date_path = output / "per_date_metrics.csv"
    with per_date_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    per_user_date_path = output / "per_user_date_metrics.csv"
    with per_user_date_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(user_metric_rows[0]))
        writer.writeheader()
        writer.writerows(user_metric_rows)
    context_timing_path = context_cache.write_timing() if context_cache is not None else None
    selection_path = output / "selected_hyperparameters.csv"
    with selection_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)
    trajectory = np.stack(coefficient_rows)
    parameter_count = max(int(row["adaptor_parameters"]) for row in selection_rows)
    trajectory_path = output / "coefficient_trajectory.npy"
    np.save(trajectory_path, trajectory)
    summary_path = output / "coefficient_summary.csv"
    flattened = trajectory.reshape(-1, trajectory.shape[-1])
    mean = flattened.mean(axis=0)
    std = flattened.std(axis=0)
    mean_abs = np.abs(flattened).mean(axis=0)
    importance = mean_abs / max(float(mean_abs.sum()), 1e-12)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("feature", "mean_coefficient", "coefficient_std", "mean_abs_coefficient", "importance"))
        for values_row in zip(feature_names, mean, std, mean_abs, importance, strict=True):
            writer.writerow(values_row)
    plot_coefficient_summary(summary_path, output / "coefficient_summary.png")
    adaptation_seconds = perf_counter() - evaluation_started
    assert cold_adaptation_seconds is not None
    timing_summary = write_compute_timing(
        output,
        extraction_timing=Path(extraction_dir) / "extraction_timing.json",
        adaptation_seconds=adaptation_seconds,
        evaluation_samples=len(user_metric_rows),
        cold_adaptation_seconds=cold_adaptation_seconds,
        method=config.method,
    )
    result_path = output / "result_manifest.json"
    result_path.write_text(
        json.dumps(
            {
                "format": RESULT_FORMAT,
                "method": config.method,
                "adapter_config": config.scientific_dict(),
                "parameters": adaptor_parameter_metadata(
                    parameter_count,
                    "maximum fitted linear coefficients used per evaluation query user",
                ),
                "feature_names": feature_names,
                "mode": mode,
                "formulation": formulation,
                "evaluation": {
                    "first_query_date": int(coefficient_dates[0]),
                    "last_query_date": int(coefficient_dates[-1]),
                    "dates": len(coefficient_dates),
                },
                "hyperparameter_selection": {
                    "enabled": selection_enabled,
                    "alpha_enabled": config.tune_alpha,
                    "k_enabled": config.used_k is None,
                    "split": "chronological_oldest_train_newest_validation",
                    "selected_k_counts": {
                        str(k): sum(int(row["selected_k"]) == k for row in selection_rows)
                        for k in candidate_ks
                    },
                    "selected_alpha_counts": {
                        str(alpha): sum(
                            float(row["selected_alpha"]) == float(alpha)
                            for row in selection_rows
                        )
                        for alpha in (
                            config.alpha_grid
                            if config.tune_alpha
                            else candidate_alphas
                        )
                    },
                },
                "files": {
                    "metrics": metrics_path.name,
                    "per_date_metrics": per_date_path.name,
                    "per_user_date_metrics": per_user_date_path.name,
                    "selected_hyperparameters": selection_path.name,
                    "coefficient_trajectory": trajectory_path.name,
                    "coefficient_summary": summary_path.name,
                    "coefficient_plot": "coefficient_summary.png",
                    "compute_timing": timing_summary.name,
                    "context_forecast_timing": (
                        str(context_timing_path.relative_to(output))
                        if context_timing_path is not None
                        else None
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "metrics": metrics_path,
        "per_date": per_date_path,
        "per_user_date": per_user_date_path,
        "selection": selection_path,
        "trajectory": trajectory_path,
        "summary": summary_path,
        "plot": output / "coefficient_summary.png",
        "compute_timing": timing_summary,
        **({"context_timing": context_timing_path} if context_timing_path is not None else {}),
        "manifest": result_path,
    }

def evaluate_online_gate(
    *,
    extraction_dir: str | Path,
    output_dir: str | Path,
    config: AdapterConfig,
    dataset: Any,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    eval_start_date: int | None = None,
    eval_end_date: int | None = None,
) -> dict[str, Path]:
    """Fit a causal shared soft gate and evaluate one query date at a time."""
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_started = perf_counter()
    config.validate()
    if config.method not in {
        "bayes_cov_shared_soft",
        "bayes_avgy_shared_soft",
        "catboost_cov_shared_soft",
        "catboost_avgy_shared_soft",
    }:
        raise ValueError(f"unknown online gate {config.method!r}")
    kind, candidate_name, _, _ = config.method.split("_", 3)
    arrays = open_extraction_arrays(extraction_dir, dataset=dataset)
    available_neighbors = int(arrays["neighbor_y"].shape[2])
    used_k = (
        int(config.used_k)
        if config.used_k is not None
        else max(int(value) for value in config.candidate_k_grid)
    )
    if used_k > available_neighbors:
        raise ValueError(
            f"gate requires used_k={used_k}, but extraction contains "
            f"max_k={available_neighbors}"
        )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    context_cache = (
        ContextForecastCache(
            arrays=arrays,
            output_dir=output,
            model=model,
            device=device or torch.device("cpu"),
        )
        if model is not None
        else None
    )
    retrieval_window_dates = np.asarray(
        arrays["retrieval_window_dates"], dtype=np.int64
    )
    extraction_config = ExtractionConfig(**load_array_manifest(extraction_dir)["config"])
    date_to_index = {
        int(date): index for index, date in enumerate(retrieval_window_dates)
    }
    n_dates, n_users, horizon = arrays["y"].shape
    scope_keys = (
        tuple(range(n_users))
        if config.fitting_scope == "same_user"
        else (0,)
    )
    metric_rows: list[dict[str, Any]] = []
    user_metric_rows: list[dict[str, Any]] = []
    cold_adaptation_seconds: float | None = None
    importance_rows: list[np.ndarray] = []
    fixed_indices_by_grid: dict[int, list[int]] = {}
    models_by_grid: dict[int, dict[int, Any | None]] = {}
    refit_by_grid: dict[int, dict[int, int]] = {}
    gate_row_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def cached_gate_row(date_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if date_index not in gate_row_cache:
            features, candidate = gate_features(
                arrays, date_index, candidate_name, used_k, context_cache
            )
            fitting_target = np.asarray(arrays["y"][date_index], dtype=np.float64)
            fitting_vanilla = np.asarray(arrays["vanilla"][date_index], dtype=np.float64)
            fitting_scale = np.maximum(
                np.asarray(arrays["query_std"][date_index], dtype=np.float64)[:, None],
                1e-8,
            )
            gate_row_cache[date_index] = (
                features,
                candidate,
                _advantage(
                    fitting_target,
                    fitting_vanilla,
                    candidate,
                    fitting_scale,
                    config.fit_loss,
                ),
            )
        return gate_row_cache[date_index]

    cold_setup_seconds = perf_counter() - evaluation_started
    for current_index, current_date_raw in enumerate(retrieval_window_dates):
        current_date = int(current_date_raw)
        if not bool(arrays["is_evaluation_query"][current_index]):
            continue
        if eval_start_date is not None and current_date < int(eval_start_date):
            continue
        if eval_end_date is not None and current_date > int(eval_end_date):
            continue

        batch_started = perf_counter()
        selected_dates = fitting_dates(
            current_date, config=extraction_config, n_fit=config.n_fit
        )
        selected_indices = [
            date_to_index[int(date)] for date in selected_dates if int(date) in date_to_index
        ]
        if len(selected_indices) < config.n_fit:
            continue
        grid_key = int(selected_dates[-1] % extraction_config.fit_stride)
        if config.fit_mode == "fixed":
            selected_indices = fixed_indices_by_grid.setdefault(grid_key, selected_indices)
        models = models_by_grid.setdefault(
            grid_key, {scope: None for scope in scope_keys}
        )
        since_refit = refit_by_grid.setdefault(
            grid_key,
            {scope: config.catboost_refit_stride for scope in scope_keys},
        )
        active = {scope: deque() for scope in scope_keys}
        advantage_sum = {scope: 0.0 for scope in scope_keys}
        advantage_square_sum = {scope: 0.0 for scope in scope_keys}
        for fitting_index in selected_indices:
            features, _, advantages = cached_gate_row(fitting_index)
            for user in range(n_users):
                scope = user if config.fitting_scope == "same_user" else 0
                advantage = float(advantages[user])
                active[scope].append((fitting_index, user, features[user], advantage))
                advantage_sum[scope] += advantage
                advantage_square_sum[scope] += advantage * advantage
        frozen = config.fit_mode == "fixed"

        advantage_means = {
            scope: advantage_sum[scope] / len(active[scope])
            for scope in scope_keys
        }
        target_scales = {
            scope: float(
                np.sqrt(
                    max(
                        advantage_square_sum[scope] / len(active[scope])
                        - advantage_means[scope] * advantage_means[scope],
                        0.0,
                    )
                )
            )
            for scope in scope_keys
        }
        if kind == "catboost":
            for scope in scope_keys:
                if models[scope] is None or (
                    not frozen
                    and since_refit[scope] >= config.catboost_refit_stride
                ):
                    fit_target = np.fromiter(
                        (row[3] for row in active[scope]),
                        dtype=np.float32,
                    )
                    fit_features = np.stack([row[2] for row in active[scope]])
                    models[scope] = _catboost(config)
                    models[scope].fit(fit_features, fit_target)
                    importance_rows.append(
                        np.asarray(
                            models[scope].get_feature_importance(),
                            dtype=np.float64,
                        )
                    )
                    since_refit[scope] = 0

        current_features, current_candidate, _ = cached_gate_row(current_index)
        if kind == "bayes":
            score = (
                np.asarray(
                    [advantage_means[user] for user in range(n_users)],
                    dtype=np.float64,
                )
                if config.fitting_scope == "same_user"
                else np.full(n_users, advantage_means[0], dtype=np.float64)
            )
        else:
            if config.fitting_scope == "same_user":
                score = np.asarray(
                    [
                        models[user].predict(current_features[user : user + 1])[0]
                        for user in range(n_users)
                    ],
                    dtype=np.float64,
                )
            else:
                assert models[0] is not None
                score = np.asarray(models[0].predict(current_features), dtype=np.float64)
        score_scales = (
            np.asarray(
                [target_scales[user] for user in range(n_users)],
                dtype=np.float64,
            )
            if config.fitting_scope == "same_user"
            else np.full(n_users, target_scales[0], dtype=np.float64)
        )
        weight = np.asarray(
            [
                _sigmoid_score(np.asarray([value]), float(scale))[0]
                for value, scale in zip(score, score_scales, strict=True)
            ]
        )[:, None]
        vanilla = np.asarray(arrays["vanilla"][current_index], dtype=np.float64)
        target = np.asarray(arrays["y"][current_index], dtype=np.float64)
        prediction = vanilla + weight * (current_candidate - vanilla)
        scale = np.maximum(
            np.asarray(arrays["query_std"][current_index], dtype=np.float64)[:, None],
            1e-8,
        )
        metric_rows.append(
            _date_metrics(current_date, config.method, prediction, target, vanilla, scale)
        )
        user_metric_rows.extend(
            _user_date_metrics(current_date, config.method, prediction, target, vanilla, scale)
        )
        if cold_adaptation_seconds is None:
            cold_adaptation_seconds = cold_setup_seconds + perf_counter() - batch_started
        for scope in scope_keys:
            since_refit[scope] += 1

    metrics_path = output / "metrics.json"
    metrics_path.write_text(
        json.dumps(_aggregate_user_metrics(user_metric_rows), indent=2), encoding="utf-8"
    )
    per_date_path = output / "per_date_metrics.csv"
    with per_date_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    per_user_date_path = output / "per_user_date_metrics.csv"
    with per_user_date_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(user_metric_rows[0]))
        writer.writeheader()
        writer.writerows(user_metric_rows)
    context_timing = context_cache.write_timing() if context_cache is not None else None

    files: dict[str, str] = {
        "metrics": metrics_path.name,
        "per_date_metrics": per_date_path.name,
        "per_user_date_metrics": per_user_date_path.name,
        "context_forecast_timing": (
            str(context_timing.relative_to(output)) if context_timing is not None else None
        ),
    }
    result: dict[str, Path] = {
        "metrics": metrics_path,
        "per_date": per_date_path,
        "per_user_date": per_user_date_path,
        **({"context_timing": context_timing} if context_timing is not None else {}),
    }
    if kind == "catboost":
        trajectory = np.stack(importance_rows)
        trajectory_path = output / "feature_importance_trajectory.npy"
        np.save(trajectory_path, trajectory)
        summary_path = output / "feature_importance_summary.csv"
        mean = trajectory.mean(axis=0)
        std = trajectory.std(axis=0)
        normalized = mean / max(float(mean.sum()), 1e-12)
        with summary_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("feature", "mean_importance", "importance_std", "importance"))
            writer.writerows(zip(GATE_FEATURE_NAMES, mean, std, normalized, strict=True))
        plot_path = output / "feature_importance_summary.png"
        plot_gate_importance(summary_path, plot_path)
        files.update(
            {
                "feature_importance_trajectory": trajectory_path.name,
                "feature_importance_summary": summary_path.name,
                "feature_importance_plot": plot_path.name,
            }
        )
        result.update(
            {"trajectory": trajectory_path, "summary": summary_path, "plot": plot_path}
        )
    if kind == "bayes":
        parameter_count = 2
        parameter_definition = "fitted advantage mean and dispersion"
    else:
        final_models = [
            model
            for values in models_by_grid.values()
            for model in values.values()
            if model is not None
        ]
        parameter_count = max(
            _catboost_adaptor_parameter_count(model) for model in final_models
        )
        parameter_definition = (
            "maximum fitted leaf values used per query user in the final CatBoost refit"
        )
    adaptation_seconds = perf_counter() - evaluation_started
    assert cold_adaptation_seconds is not None
    timing_path = write_compute_timing(
        output,
        extraction_timing=Path(extraction_dir) / "extraction_timing.json",
        adaptation_seconds=adaptation_seconds,
        evaluation_samples=len(user_metric_rows),
        cold_adaptation_seconds=cold_adaptation_seconds,
        method=config.method,
    )
    files["compute_timing"] = timing_path.name
    result["compute_timing"] = timing_path
    manifest_path = output / "result_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": RESULT_FORMAT,
                "method": config.method,
                "adapter_config": config.scientific_dict(),
                "parameters": adaptor_parameter_metadata(
                    parameter_count,
                    parameter_definition,
                ),
                "feature_names": list(GATE_FEATURE_NAMES),
                "evaluation": {
                    "first_query_date": int(metric_rows[0]["query_date"]),
                    "last_query_date": int(metric_rows[-1]["query_date"]),
                    "dates": len(metric_rows),
                },
                "files": files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result["manifest"] = manifest_path
    return result
