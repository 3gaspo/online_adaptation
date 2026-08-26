"""Single source of truth for publication profiles and ablation grids."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from src.proposal import (
    DEFAULT_ALPHA,
    DEFAULT_CANDIDATE_K_GRID,
    DEFAULT_MAX_K,
    DEFAULT_N_FIT,
    DEFAULT_N_STORE,
    DEFAULT_TSRAG_K,
)
from src.proposal.ridge import BASELINE_VARIABLES


LOGGER = logging.getLogger(__name__)


SMALL_DATASETS = ("Electricity", "Solar", "Traffic")
RAW_ETT_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
MIXED_QUANTITY_DATASETS = (*RAW_ETT_DATASETS, "Weather")
SOTA_DATASETS = (*RAW_ETT_DATASETS, "Weather", "Electricity", "exchange_rate")
RANGE_NAMES = ("short", "mid", "long")
RANGE_SETTINGS = {
    "hourly": {
        "short": (168, 24),
        "mid": (336, 48),
        "long": (504, 168),
    },
    "daily": {
        "short": (7, 1),
        "mid": (14, 2),
        "long": (30, 7),
    },
    "15min": {
        "short": (96, 4),
        "mid": (192, 8),
        "long": (672, 96),
    },
}
PERIOD_BY_FREQUENCY = {"hourly": 24, "daily": 7, "15min": 96}
DATASET_FREQUENCIES = {
    "electricity": "hourly",
    "traffic": "hourly",
    "solar": "hourly",
    "weather": "hourly",
    "etth1": "hourly",
    "etth2": "hourly",
    "ettm1": "15min",
    "ettm2": "15min",
    "exchange_rate": "daily",
}


def _time_datasets(data_root: Path) -> list[str]:
    catalog_path = data_root / "time" / "catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(
            f"full mode requires the prepared TIME catalog {catalog_path}"
        )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    return [f"time/{item['name']}" for item in catalog["datasets"]]


def datasets_for_mode(mode: str, data_root: str | Path) -> list[str]:
    if mode == "test":
        return ["Electricity"]
    if mode == "small":
        return list(SMALL_DATASETS)
    if mode == "full":
        return [
            *SMALL_DATASETS,
            *RAW_ETT_DATASETS,
            *_time_datasets(Path(data_root).expanduser().resolve()),
        ]
    raise ValueError("EXPERIMENT_MODE must be test, small, or full")


def _normalize_frequency(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "")
    aliases = {
        "h": "hourly",
        "1h": "hourly",
        "hour": "hourly",
        "hourly": "hourly",
        "d": "daily",
        "1d": "daily",
        "day": "daily",
        "daily": "daily",
        "15t": "15min",
        "15m": "15min",
        "15min": "15min",
        "15minute": "15min",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported dataset frequency {value!r}") from error


def _time_metadata(data_root: Path) -> dict[str, dict[str, Any]]:
    catalog_path = data_root / "time" / "catalog.json"
    if not catalog_path.is_file():
        return {}
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    return {f"time/{item['name']}": item for item in catalog["datasets"]}


def dataset_frequency(dataset: str, data_root: str | Path) -> str:
    known = DATASET_FREQUENCIES.get(str(dataset).lower())
    if known is not None:
        return known
    metadata = _time_metadata(Path(data_root).expanduser().resolve()).get(str(dataset))
    if metadata is None:
        raise KeyError(f"dataset {dataset!r} has no cadence metadata")
    frequency = metadata.get("configured_frequency", metadata.get("source_frequency"))
    if frequency is None:
        raise KeyError(f"TIME dataset {dataset!r} has no configured frequency")
    return _normalize_frequency(str(frequency))


def range_names_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "test":
        return ("long",)
    if mode in {"small", "full"}:
        return RANGE_NAMES
    raise ValueError("EXPERIMENT_MODE must be test, small, or full")


def settings_for_mode(
    mode: str,
    dataset: str,
    data_root: str | Path,
) -> list[tuple[int, int]]:
    frequency = dataset_frequency(dataset, data_root)
    return [RANGE_SETTINGS[frequency][name] for name in range_names_for_mode(mode)]


def homogeneous_datasets_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "test":
        return ("Weather",)
    if mode in {"small", "full"}:
        return MIXED_QUANTITY_DATASETS
    raise ValueError("EXPERIMENT_MODE must be test, small, or full")


def _base_task(dataset: str, setting: tuple[int, int]) -> dict[str, Any]:
    lookback, horizon = setting
    return {
        "dataset": dataset,
        "lookback": lookback,
        "horizon": horizon,
        "backbone": "chronos2",
        "retrieval_covariate_mode": "past_and_future",
        "method": "full_ridge_shared",
        "n_store": DEFAULT_N_STORE,
        "n_fit": DEFAULT_N_FIT,
        "fitting_scope": "same_user",
        "alpha": DEFAULT_ALPHA,
        "max_k": DEFAULT_MAX_K,
        "candidate_k_grid": DEFAULT_CANDIDATE_K_GRID,
        "used_k": None,
        "distance_space": "raw",
        "distance_metric": "euclidean",
        "retrieval_scope": "all",
        "store_mode": "rolling",
        "fit_stride": 0,
        "fit_mode": "rolling",
        "homogeneous_only": False,
        "fit_loss": "mse",
        "candidate": "cov",
        "tune_alpha": True,
    }


def _base_grid(mode: str, data_root: str | Path) -> list[dict[str, Any]]:
    return [
        _base_task(dataset, setting)
        for dataset in datasets_for_mode(mode, data_root)
        for setting in settings_for_mode(mode, dataset, data_root)
    ]


def _unfiltered_tasks_for_family(
    family: str,
    mode: str,
    data_root: str | Path,
) -> list[dict[str, Any]]:
    """Expand one family into explicit, independently manifested tasks."""
    if family == "homogeneous_ablation":
        return [
            {**_base_task(dataset, setting), "homogeneous_only": value}
            for dataset in homogeneous_datasets_for_mode(mode)
            for setting in settings_for_mode(mode, dataset, data_root)
            for value in (False, True)
        ]
    base = _base_grid(mode, data_root)
    if family == "main":
        tasks = list(base)
        for dataset in datasets_for_mode(mode, data_root):
            tasks.append(
                {
                    **_base_task(dataset, (512, 64)),
                    "backbone": "chronos_bolt",
                    "retrieval_covariate_mode": "none",
                    "method": "tsrag",
                    "distance_space": "tsrag",
                    "retrieval_scope": "same_user",
                    "max_k": DEFAULT_TSRAG_K,
                    "used_k": DEFAULT_TSRAG_K,
                    "tune_alpha": False,
                }
            )
        return tasks
    if family == "online_gates":
        return [
            {
                **task,
                "method": method,
                "used_k": DEFAULT_MAX_K,
                "tune_alpha": False,
            }
            for task in base
            for method in ("bayes_cov_shared_soft", "catboost_cov_shared_soft")
        ]
    if family == "n_store_ablation":
        values = (1_000, 2_000, 5_000) if mode == "test" else (10_000, 20_000, 30_000, 50_000)
        return [{**task, "n_store": value} for task in base for value in values]
    if family == "n_fit_ablation":
        values = (50, 100, 500) if mode == "test" else (50, 100, 500, 1_000)
        return [{**task, "n_fit": value} for task in base for value in values]
    if family == "fit_stride_ablation":
        return [
            {**task, "fit_stride": value}
            for task in base
            for value in (1, 0)
        ]
    if family == "alpha_ablation":
        return [
            {**task, "alpha": value, "tune_alpha": False}
            for task in base
            for value in (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
        ]
    if family == "k_ablation":
        return [
            {**task, "used_k": value}
            for task in base
            for value in (1, 3, 5, 10, 15, 20)
        ]
    if family == "l_ablation":
        datasets = datasets_for_mode(mode, data_root)
        return [
            _base_task(dataset, (lookback, 24))
            for dataset in datasets
            for lookback in (24, 96, 168, 336, 504, 512, 720)
        ]
    if family == "h_ablation":
        datasets = datasets_for_mode(mode, data_root)
        return [
            _base_task(dataset, (504, horizon))
            for dataset in datasets
            for horizon in (24, 48, 64, 96, 168, 336, 504)
        ]
    if family == "feature_design_ablation":
        return [
            {**task, "method": f"{design}_ridge_shared"}
            for task in base
            for design in BASELINE_VARIABLES
        ]
    if family == "formulation_ablation":
        methods = (
            "full_ridge_shared",
            "full_ridge_horizon",
            "full_delta_ridge_shared",
            "full_delta_ridge_horizon",
            "full_convex_shared",
            "full_convex_horizon",
        )
        return [{**task, "method": method} for task in base for method in methods]
    if family == "fixed_protocol_ablation":
        return [
            {**task, "store_mode": store_mode, "fit_mode": fit_mode}
            for task in base
            for store_mode in ("rolling", "fixed")
            for fit_mode in ("rolling", "fixed")
        ]
    if family == "general_scope_ablation":
        return [
            {
                **task,
                "retrieval_scope": retrieval_scope,
                "fitting_scope": fitting_scope,
            }
            for task in base
            for retrieval_scope in ("all", "same_user", "other_users")
            for fitting_scope in ("all", "same_user")
        ]
    if family == "backbone_ablation":
        datasets = datasets_for_mode(mode, data_root)
        return [
            {
                **_base_task(dataset, (512, 64)),
                "backbone": backbone,
                "retrieval_covariate_mode": "none",
                "method": "y_ridge_shared",
            }
            for dataset in datasets
            for backbone in (
                "chronos2",
                "chronos_bolt",
                "ts_icl",
                "tabpfn_ts",
                # "tirex2",  # Adapter-supported; excluded from launches for now.
            )
        ]
    if family == "sota_backbone_ablation":
        datasets = ["Electricity"] if mode == "test" else list(SOTA_DATASETS)
        return [
            {
                **_base_task(dataset, (512, 64)),
                "backbone": "chronos_bolt",
                "retrieval_covariate_mode": "none",
                "method": "y_ridge_shared",
            }
            for dataset in datasets
        ]
    raise ValueError(f"unknown experiment family {family!r}")


def tasks_for_family(
    family: str,
    mode: str,
    data_root: str | Path,
) -> list[dict[str, Any]]:
    tasks = _unfiltered_tasks_for_family(family, mode, data_root)
    metadata = _time_metadata(Path(data_root).expanduser().resolve())
    feasible: list[dict[str, Any]] = []
    for task in tasks:
        frequency = dataset_frequency(str(task["dataset"]), data_root)
        task["period"] = PERIOD_BY_FREQUENCY[frequency]
        task["store_stride"] = PERIOD_BY_FREQUENCY[frequency]
        if int(task["fit_stride"]) == 0:
            task["fit_stride"] = PERIOD_BY_FREQUENCY[frequency]
        values = metadata.get(str(task["dataset"]))
        if values is None:
            feasible.append(task)
            continue
        users = int(values["num_series"])
        dates = int(values["num_timestamps"])
        fitting_dates = int(task["n_fit"])
        if task["retrieval_scope"] == "same_user":
            retrieval_dates = int(task["max_k"])
        elif task["retrieval_scope"] == "all":
            retrieval_dates = math.ceil(int(task["max_k"]) / users)
        else:
            retrieval_dates = math.ceil(int(task["max_k"]) / (users - 1))
        required_dates = (
            int(task["lookback"])
            + 3 * int(task["horizon"])
            + retrieval_dates * int(task["store_stride"])
            + fitting_dates * int(task["fit_stride"])
            - 2
        )
        if dates >= required_dates:
            feasible.append(task)
        else:
            LOGGER.info(
                "skip causally infeasible TIME task dataset=%s L=%s H=%s dates=%s required=%s users=%s",
                task["dataset"],
                task["lookback"],
                task["horizon"],
                dates,
                required_dates,
                users,
            )
    return feasible
