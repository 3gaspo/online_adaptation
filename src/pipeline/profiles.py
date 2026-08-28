"""Single source of truth for publication profiles and ablation grids."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.proposal import (
    DEFAULT_ALPHA,
    DEFAULT_CANDIDATE_K_GRID,
    DEFAULT_MAX_K,
    DEFAULT_N_DATASTORE_DATES,
    DEFAULT_N_FIT,
    DEFAULT_TSRAG_K,
)
from src.proposal.contracts import ExtractionConfig
from src.proposal.date_planning import build_date_plan
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

DEADLINE_TIME_DATASETS = (
    "time/ne_china_wind_h",
    "time/coastal_t_s_h_part11",
    "time/sg_weather_d",
)


def _time_datasets(data_root: Path) -> list[str]:
    catalog_path = data_root / "time" / "catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(
            f"full mode requires the prepared TIME catalog {catalog_path}"
        )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    return [f"time/{item['name']}" for item in catalog["datasets"]]


def datasets_for_mode(
    mode: str,
    data_root: str | Path,
    selected: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    if selected is not None:
        return [str(dataset) for dataset in selected]
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


def range_names_for_mode(
    mode: str,
    selected: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if selected is not None:
        values = tuple(str(value) for value in selected)
        unknown = set(values) - set(RANGE_NAMES)
        if unknown:
            raise ValueError(f"unknown range names: {sorted(unknown)}")
        return values
    if mode == "test":
        return ("long",)
    if mode in {"small", "full"}:
        return RANGE_NAMES
    raise ValueError("EXPERIMENT_MODE must be test, small, or full")


def settings_for_mode(
    mode: str,
    dataset: str,
    data_root: str | Path,
    selected_ranges: list[str] | tuple[str, ...] | None = None,
) -> list[tuple[int, int]]:
    frequency = dataset_frequency(dataset, data_root)
    return [
        RANGE_SETTINGS[frequency][name]
        for name in range_names_for_mode(mode, selected_ranges)
    ]


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
        "n_datastore_dates": DEFAULT_N_DATASTORE_DATES,
        "n_store_windows": None,
        "n_fit": DEFAULT_N_FIT,
        "fitting_scope": "same_user",
        "alpha": DEFAULT_ALPHA,
        "max_k": DEFAULT_MAX_K,
        "candidate_k_grid": DEFAULT_CANDIDATE_K_GRID,
        "used_k": None,
        "distance_space": "raw",
        "distance_metric": "euclidean",
        "retrieval_scope": "all",
        "fixed_datastore": False,
        "fixed_training_set": False,
        "include_fitting_windows": True,
        "eval_start_date": None,
        "eval_end_date": None,
        "split_ratios": None,
        "align_period": True,
        "store_stride": 0,
        "fit_stride": 0,
        "homogeneous_only": False,
        "fit_loss": "mse",
        "candidate": "cov",
        "tune_alpha": True,
    }


def _base_grid(
    mode: str,
    data_root: str | Path,
    *,
    selected_datasets: list[str] | tuple[str, ...] | None = None,
    selected_ranges: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    return [
        _base_task(dataset, setting)
        for dataset in datasets_for_mode(mode, data_root, selected_datasets)
        for setting in settings_for_mode(
            mode, dataset, data_root, selected_ranges
        )
    ]


def _unfiltered_tasks_for_family(
    family: str,
    mode: str,
    data_root: str | Path,
    *,
    selected_datasets: list[str] | tuple[str, ...] | None = None,
    selected_ranges: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Expand one family into explicit, independently manifested tasks."""
    if family == "homogeneous_ablation":
        return [
            {**_base_task(dataset, setting), "homogeneous_only": value}
            for dataset in homogeneous_datasets_for_mode(mode)
            for setting in settings_for_mode(mode, dataset, data_root)
            for value in (False, True)
        ]
    base = _base_grid(
        mode,
        data_root,
        selected_datasets=selected_datasets,
        selected_ranges=selected_ranges,
    )
    if family == "deadline_fixed_protocol":
        return [
            {
                **task,
                "n_store_windows": 20_000,
                "n_fit": 0.5,
                "split_ratios": (0.3, 0.5, 0.2),
                "store_stride": 24,
                "fit_stride": 24,
                "align_period": True,
                "fixed_datastore": fixed_datastore,
                "fixed_training_set": fixed_training_set,
            }
            for task in base
            for fixed_datastore in (False, True)
            for fixed_training_set in (False, True)
        ]
    if family == "deadline_tsrag_comparison":
        metadata = _time_metadata(Path(data_root).expanduser().resolve())
        requested = datasets_for_mode(mode, data_root, selected_datasets)
        tasks: list[dict[str, Any]] = []
        for dataset in requested:
            if dataset not in DEADLINE_TIME_DATASETS:
                raise ValueError(
                    f"deadline TS-RAG dataset {dataset!r} is not in the "
                    f"verified subset {DEADLINE_TIME_DATASETS}"
                )
            values = metadata.get(dataset)
            if values is None:
                raise KeyError(f"prepared TIME catalog lacks {dataset!r}")
            users = int(values["num_series"])
            dates = int(values["num_timestamps"])
            ridge_store_dates = 20_000 // users
            ridge = {
                **_base_task(dataset, (512, 64)),
                "n_datastore_dates": ridge_store_dates,
                "n_store_windows": 20_000,
                "n_fit": 30,
                "store_stride": 1,
                "fit_stride": 24,
                "align_period": False,
                "eval_start_date": int(round(0.8 * dates)) - 1,
                "max_k": DEFAULT_MAX_K,
                "candidate_k_grid": DEFAULT_CANDIDATE_K_GRID,
                "used_k": None,
            }
            tasks.append(ridge)
            tasks.append(
                {
                    **ridge,
                    "backbone": "chronos_bolt",
                    "retrieval_covariate_mode": "none",
                    "method": "tsrag",
                    "distance_space": "tsrag",
                    "retrieval_scope": "same_user",
                    "n_datastore_dates": ridge_store_dates,
                    "n_store_windows": 20_000,
                    "split_ratios": (0.3, 0.5, 0.2),
                    "fixed_datastore": True,
                    "include_fitting_windows": False,
                    "max_k": DEFAULT_TSRAG_K,
                    "candidate_k_grid": (DEFAULT_TSRAG_K,),
                    "used_k": DEFAULT_TSRAG_K,
                    "tune_alpha": False,
                }
            )
        return tasks
    if family == "main":
        return [
            {
                **task,
                "method": method,
                "used_k": (
                    DEFAULT_MAX_K
                    if method in {"bayes_cov_shared_soft", "covariate_prediction"}
                    else None
                ),
                "tune_alpha": method not in {"bayes_cov_shared_soft", "covariate_prediction"},
            }
            for task in base
            for method in (
                "full_ridge_shared",
                "y_convex_shared",
                "bayes_cov_shared_soft",
                "covariate_prediction",
            )
        ]
    if family == "tsrag_comparison":
        tasks: list[dict[str, Any]] = []
        for dataset in datasets_for_mode(mode, data_root):
            datastore_ratio = 0.6 if dataset.split("/")[-1].lower().startswith("ett") else 0.7
            ridge = {
                **_base_task(dataset, (512, 64)),
                "retrieval_scope": "same_user",
                "n_datastore_dates": datastore_ratio,
                "store_stride": 1,
                "fit_stride": 1,
                "align_period": False,
                "eval_start_date": 0.8,
                "max_k": DEFAULT_TSRAG_K,
                "candidate_k_grid": (DEFAULT_TSRAG_K,),
                "used_k": DEFAULT_TSRAG_K,
            }
            tasks.append(ridge)
            tasks.append(
                {
                    **ridge,
                    "backbone": "chronos_bolt",
                    "retrieval_covariate_mode": "none",
                    "method": "tsrag",
                    "distance_space": "tsrag",
                    "fixed_datastore": True,
                    "include_fitting_windows": False,
                    "tune_alpha": False,
                }
            )
        return tasks
    if family == "n_datastore_dates_ablation":
        values = (25, 50, 100) if mode == "test" else (50, 100, 200, 500)
        return [
            {**task, "n_datastore_dates": value}
            for task in base
            for value in values
        ]
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
            {
                **task,
                "fixed_datastore": fixed_datastore,
                "fixed_training_set": fixed_training_set,
            }
            for task in base
            for fixed_datastore in (False, True)
            for fixed_training_set in (False, True)
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
                "chronos_t5",
                "ts_icl",
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
    *,
    selected_datasets: list[str] | tuple[str, ...] | None = None,
    selected_ranges: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    tasks = _unfiltered_tasks_for_family(
        family,
        mode,
        data_root,
        selected_datasets=selected_datasets,
        selected_ranges=selected_ranges,
    )
    metadata = _time_metadata(Path(data_root).expanduser().resolve())
    feasible: list[dict[str, Any]] = []
    for task in tasks:
        frequency = dataset_frequency(str(task["dataset"]), data_root)
        task["period"] = PERIOD_BY_FREQUENCY[frequency]
        if int(task["store_stride"]) == 0:
            task["store_stride"] = PERIOD_BY_FREQUENCY[frequency]
        if int(task["fit_stride"]) == 0:
            task["fit_stride"] = PERIOD_BY_FREQUENCY[frequency]
        values = metadata.get(str(task["dataset"]))
        if values is None:
            feasible.append(task)
            continue
        dates = int(values["num_timestamps"])
        extraction = ExtractionConfig(
            dataset=str(task["dataset"]),
            lookback=int(task["lookback"]),
            horizon=int(task["horizon"]),
            backbone=str(task["backbone"]),
            n_datastore_dates=task["n_datastore_dates"],
            n_store_windows=task.get("n_store_windows"),
            n_fit=task["n_fit"],
            max_k=int(task["max_k"]),
            retrieval_scope=str(task["retrieval_scope"]),
            fixed_datastore=bool(task["fixed_datastore"]),
            fixed_training_set=bool(task["fixed_training_set"]),
            include_fitting_windows=bool(task["include_fitting_windows"]),
            store_stride=int(task["store_stride"]),
            fit_stride=int(task["fit_stride"]),
            align_period=bool(task["align_period"]),
            period=int(task["period"]),
            eval_start_date=task["eval_start_date"],
            eval_end_date=task["eval_end_date"],
            split_ratios=task.get("split_ratios"),
        )
        try:
            build_date_plan(
                n_dates=dates,
                n_users=int(values["num_series"]),
                config=extraction,
            )
        except ValueError as error:
            LOGGER.info(
                "skip causally infeasible TIME task dataset=%s L=%s H=%s dates=%s reason=%s",
                task["dataset"],
                task["lookback"],
                task["horizon"],
                dates,
                error,
            )
        else:
            feasible.append(task)
    return feasible
