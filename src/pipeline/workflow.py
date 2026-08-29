"""Run reusable extraction, causal adaptation, and common-date tables."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shlex
from typing import Any

from omegaconf import DictConfig, OmegaConf
import torch

from src.data.load_dataset import load_csv_dataset, set_seed
from src.pipeline.runs import (
    allocate_run,
    identity_path,
    mark_status,
    pipeline_config_with_dependencies,
)
from src.pipeline.runtime import setup_logging
from src.model_loading.forecast import load_pretrained_model, resolve_device
from src.pipeline.contracts import (
    EXTRACTION_FORMAT,
    load_array_manifest,
)
from src.proposal.contracts import AdapterConfig, ExtractionConfig
from src.pipeline.adaptation import (
    evaluate_covariate_prediction,
    evaluate_online_gate,
    evaluate_online_linear,
)
from src.pipeline.extraction import extract_online_features
from src.pipeline.profiles import tasks_for_family
from src.results.reporting import build_online_tables, build_published_sota_table
from src.pipeline.tsrag import evaluate_online_tsrag


LOGGER = logging.getLogger(__name__)
PROJECT = "online_adaptation"
EXTRACTION_ORDER = (
    "retrieval_covariate_mode",
    "distance_space",
    "distance_metric",
    "max_k",
    "retrieval_scope",
    "n_store_windows",
    "n_datastore_dates",
    "n_fit",
    "fit_stride",
    "fixed_datastore",
    "fixed_training_set",
    "include_fitting_windows",
    "homogeneous",
)
ADAPTER_ORDER = (
    "method",
    "n_fit",
    "fitting_scope",
    "alpha",
    "tune_alpha",
    "validation_ratio",
    "alpha_grid",
    "candidate_k_grid",
    "used_k",
    "fixed_training_set",
    "fit_loss",
    "candidate",
)


def _stages(value: str) -> set[str]:
    stages = {part.strip() for part in str(value).split(",") if part.strip()}
    unknown = stages - {"extract", "adapt", "tables"}
    if unknown:
        raise ValueError(f"unknown workflow stages: {sorted(unknown)}")
    return stages


def _table_pipeline_filters() -> dict[str, Any]:
    filters: dict[str, Any] = {}
    for item in shlex.split(os.environ.get("TABLE_PIPELINE_CONFIGS", "")):
        if "=" not in item:
            raise ValueError(f"TABLE_PIPELINE_CONFIGS expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError("TABLE_PIPELINE_CONFIGS contains an empty key")
        parsed = OmegaConf.from_dotlist([f"value={value}"]).value
        filters[key] = (
            OmegaConf.to_container(parsed, resolve=True)
            if OmegaConf.is_config(parsed)
            else parsed
        )
    return filters


def _dataset_path(data_root: Path, dataset: str) -> Path:
    current = data_root
    for part in dataset.split("/"):
        candidate = current / part
        if candidate.exists():
            current = candidate
            continue
        matches = [
            child
            for child in current.iterdir()
            if child.is_dir() and child.name.casefold() == part.casefold()
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"cannot resolve dataset component {part!r} below {current}"
            )
        current = matches[0]
    return current


def _homogeneous_targets(dataset: str) -> list[str] | None:
    path = Path(__file__).resolve().parents[1] / "conf" / "homogeneous_channels.yaml"
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(config, dict)
    name = dataset.split("/", 1)[-1]
    values = config["homogeneous_targets"].get(name, None)  # type: ignore[index]
    if values is not None:
        return [str(value) for value in values]
    all_homogeneous = {str(value) for value in config["all_targets_are_homogeneous"]}  # type: ignore[index]
    if name in all_homogeneous or (dataset.startswith("time/") and "TIME" in all_homogeneous):
        return None
    raise KeyError(f"dataset {dataset!r} has no homogeneous-target declaration")


def _weight_path(weights_root: Path, backbone: str) -> Path:
    relative = {
        "chronos2": "chronos2",
        "chronos_bolt": "chronos-bolt-base",
        "chronos_t5": "chronos-t5-base",
        "ts_icl": "tsicl/tsicl-v1.ckpt",
    }[backbone]
    return weights_root / relative


def _load_backbone(
    task: dict[str, Any],
    *,
    weights_root: Path,
    device: torch.device,
    normalization: str,
) -> torch.nn.Module:
    backbone = str(task["backbone"])
    weight_path = _weight_path(weights_root, backbone)
    if not weight_path.exists():
        raise FileNotFoundError(f"missing {backbone} weights: {weight_path}")
    if backbone == "ts_icl":
        return load_pretrained_model(
            backbone,
            lags=int(task["lookback"]),
            horizon=int(task["horizon"]),
            normalization=normalization,
            pretrained_path=weight_path,
            device=device,
            model_kwargs={
                "device": str(device),
                "retrieval_covariate_mode": task["retrieval_covariate_mode"],
            },
        )
    return load_pretrained_model(
        backbone,
        lags=int(task["lookback"]),
        horizon=int(task["horizon"]),
        normalization=normalization,
        device=device,
        model_kwargs={
            "weights_path": str(weight_path),
            "device_map": str(device),
            "retrieval_covariate_mode": task["retrieval_covariate_mode"],
        },
    )


def _extraction_identity(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "retrieval_covariate_mode": task["retrieval_covariate_mode"],
        "distance_space": task["distance_space"],
        "distance_metric": task["distance_metric"],
        "max_k": task["max_k"],
        "retrieval_scope": task["retrieval_scope"],
        "n_store_windows": task.get("n_store_windows"),
        "n_datastore_dates": task["n_datastore_dates"],
        "n_fit": task["n_fit"],
        "fit_stride": task["fit_stride"],
        "fixed_datastore": task["fixed_datastore"],
        "fixed_training_set": task["fixed_training_set"],
        "include_fitting_windows": task["include_fitting_windows"],
        "homogeneous": "homogeneous" if task["homogeneous_only"] else "all",
    }


def _adapter_identity(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": task["method"],
        "n_fit": task["n_fit"],
        "fitting_scope": task["fitting_scope"],
        "alpha": task["alpha"],
        "tune_alpha": task["tune_alpha"],
        "validation_ratio": task["validation_ratio"],
        "alpha_grid": task["alpha_grid"],
        "candidate_k_grid": task["candidate_k_grid"],
        "used_k": task["used_k"],
        "fixed_training_set": task["fixed_training_set"],
        "fit_loss": task["fit_loss"],
        "candidate": task["candidate"],
    }


def _display_name(family: str, task: dict[str, Any]) -> str:
    backbone = str(task["backbone"])
    method = str(task["method"])
    if method == "tsrag":
        return "TS-RAG (chronos_bolt)"
    if family == "main":
        return {
            "full_ridge_shared": f"Online full ridge ({backbone})",
            "y_convex_shared": f"Convex mixture ({backbone})",
            "bayes_cov_shared_soft": f"Bayes covariate ({backbone})",
            "covariate_prediction": f"Covariate prediction ({backbone})",
        }[method]
    if family in {"tsrag_comparison", "deadline_tsrag_comparison"}:
        return f"Online full ridge ({backbone})"
    varying = {
        "n_datastore_dates_ablation": (
            f"N_datastore_dates={task['n_datastore_dates']}"
        ),
        "n_fit_ablation": f"N_fit={task['n_fit']}",
        "fit_stride_ablation": f"fit_stride={task['fit_stride']}",
        "alpha_ablation": f"alpha={task['alpha']}",
        "k_ablation": f"K={task['used_k']}",
        "fixed_protocol_ablation": (
            f"fixed_datastore={task['fixed_datastore']}, "
            f"fixed_training_set={task['fixed_training_set']}"
        ),
        "deadline_fixed_protocol": (
            f"fixed_datastore={task['fixed_datastore']}, "
            f"fixed_training_set={task['fixed_training_set']}, "
            f"fitting_scope={task['fitting_scope']}"
        ),
        "general_scope_ablation": (
            f"retrieval={task['retrieval_scope']}, fitting={task['fitting_scope']}"
        ),
        "homogeneous_ablation": "homogeneous" if task["homogeneous_only"] else "all variates",
        "backbone_ablation": backbone,
    }.get(family)
    return varying or method


def _allocate_extraction(
    cfg: DictConfig,
    task: dict[str, Any],
    output_root: Path,
):
    identity = _extraction_identity(task)
    root = identity_path(
        output_root / "online_extraction",
        task["dataset"],
        task["lookback"],
        task["horizon"],
        task["backbone"],
        EXTRACTION_ORDER,
        identity,
    )
    pipeline = {
        "artifact_schema": EXTRACTION_FORMAT,
        "query_stride": int(cfg.query_stride),
        "store_stride": int(task["store_stride"]),
        "fit_stride": int(task["fit_stride"]),
        "align_period": bool(task["align_period"]),
        "period": int(task["period"]),
        "eval_start_date": task["eval_start_date"],
        "eval_end_date": task["eval_end_date"],
        "split_ratios": task.get("split_ratios"),
        "n_store_windows": task.get("n_store_windows"),
        "normalization": str(cfg.normalization),
    }
    if task["dataset"].split("/")[-1].casefold() == "weather":
        pipeline["data.missing_values"] = str(cfg.missing_values or "zero")
    return allocate_run(
        root,
        project=PROJECT,
        workflow="online_extraction",
        dataset=task["dataset"],
        lookback=task["lookback"],
        horizon=task["horizon"],
        backbone=task["backbone"],
        model_config_order=EXTRACTION_ORDER,
        model_config=identity,
        pipeline_config=pipeline,
        runtime_config={"device": str(cfg.device)},
        seeds=(int(cfg.seed),),
        purpose=str(cfg.purpose),
        mode=str(cfg.mode),
        policy=str(cfg.conflict_policy),
        skip_completed=bool(cfg.skip_completed),
    )


def _run_extraction(
    cfg: DictConfig,
    task: dict[str, Any],
    allocation: Any,
    *,
    data_root: Path,
    weights_root: Path,
) -> None:
    if allocation.action == "skip":
        return
    mark_status(allocation.run_dir, "running")
    try:
        target_cols = _homogeneous_targets(task["dataset"]) if task["homogeneous_only"] else None
        if task["homogeneous_only"] and target_cols is None:
            LOGGER.info("dataset=%s declares every target homogeneous", task["dataset"])
        dataset = load_csv_dataset(
            _dataset_path(data_root, task["dataset"]),
            dataset_name=task["dataset"].split("/")[-1],
            target_cols=target_cols,
            drop_users=cfg.drop_users,
            missing_values=cfg.missing_values,
        )
        device = resolve_device(str(cfg.device))
        model = _load_backbone(
            task,
            weights_root=weights_root,
            device=device,
            normalization=str(cfg.normalization),
        )
        representation_model = None
        if task["distance_space"] == "tsrag":
            from src.external_models.tsrag.retriever import TSRAGRetriever

            representation_model = TSRAGRetriever(
                weights_root / "chronos-t5-base", device_map=str(cfg.device)
            )
        extraction = ExtractionConfig(
            dataset=task["dataset"],
            lookback=int(task["lookback"]),
            horizon=int(task["horizon"]),
            backbone=task["backbone"],
            n_datastore_dates=task["n_datastore_dates"],
            n_store_windows=task.get("n_store_windows"),
            n_fit=task["n_fit"],
            max_k=int(task["max_k"]),
            distance_space=task["distance_space"],
            distance_metric=task["distance_metric"],
            retrieval_scope=task["retrieval_scope"],
            fixed_datastore=bool(task["fixed_datastore"]),
            fixed_training_set=bool(task["fixed_training_set"]),
            include_fitting_windows=bool(task["include_fitting_windows"]),
            store_stride=int(task["store_stride"]),
            fit_stride=int(task["fit_stride"]),
            align_period=bool(task["align_period"]),
            period=int(task["period"]),
            query_stride=int(cfg.query_stride),
            eval_start_date=task["eval_start_date"],
            eval_end_date=task["eval_end_date"],
            split_ratios=task.get("split_ratios"),
            normalization=str(cfg.normalization),
            retrieval_covariate_mode=task["retrieval_covariate_mode"],
            homogeneous_only=bool(task["homogeneous_only"]),
            seed=int(cfg.seed),
        )
        extract_online_features(
            dataset=dataset,
            model=model,
            config=extraction,
            output_dir=allocation.run_dir,
            device=device,
            representation_model=representation_model,
            search_chunk_size=int(cfg.search_chunk_size),
            representation_batch_size=int(cfg.representation_batch_size),
        )
        extraction_manifest = load_array_manifest(allocation.run_dir)
        required = [
            "online_extraction_manifest.json",
            "extraction_timing.json",
            "setting_diagnostics.csv",
            "setting_diagnostics_samples.csv",
            "setting_diagnostics_sampling.json",
            "setting_diagnostics.png",
            "neighbor_diagnostics.csv",
            "neighbor_diagnostics_per_user.csv",
            "neighbor_diagnostics_all_samples.png",
            "neighbor_diagnostics_per_user.png",
            *[values["path"] for values in extraction_manifest["arrays"].values()],
        ]
        mark_status(allocation.run_dir, "completed", required_artifacts=required)
    except Exception:
        mark_status(allocation.run_dir, "interrupted")
        raise


def _adapter_config(
    cfg: DictConfig,
    task: dict[str, Any],
    *,
    resolved_n_fit: int | None = None,
) -> AdapterConfig:
    return AdapterConfig(
        method=task["method"],
        n_fit=int(task["n_fit"] if resolved_n_fit is None else resolved_n_fit),
        fitting_scope=task["fitting_scope"],
        alpha=float(task["alpha"]),
        tune_alpha=bool(task["tune_alpha"]),
        validation_ratio=float(task["validation_ratio"]),
        alpha_grid=tuple(float(value) for value in task["alpha_grid"]),
        candidate_k_grid=tuple(int(value) for value in task["candidate_k_grid"]),
        used_k=None if task["used_k"] is None else int(task["used_k"]),
        fixed_training_set=bool(task["fixed_training_set"]),
        fit_loss=task["fit_loss"],
        candidate=task["candidate"],
        catboost_iterations=int(cfg.catboost_iterations),
        catboost_depth=int(cfg.catboost_depth),
        catboost_learning_rate=float(cfg.catboost_learning_rate),
        catboost_refit_stride=int(cfg.catboost_refit_stride),
        seed=int(cfg.seed),
    )


def _run_adaptation(
    cfg: DictConfig,
    family: str,
    task: dict[str, Any],
    extraction_dir: Path,
    *,
    output_root: Path,
    data_root: Path,
    weights_root: Path,
) -> None:
    extraction_manifest = load_array_manifest(extraction_dir)
    resolved_n_fit = int(extraction_manifest["metadata"]["date_plan"]["n_fit"])
    identity = _adapter_identity(task)
    root = identity_path(
        output_root / "online_adaptation" / family,
        task["dataset"],
        task["lookback"],
        task["horizon"],
        task["backbone"],
        ADAPTER_ORDER,
        identity,
    )
    pipeline = pipeline_config_with_dependencies(
        _adapter_config(
            cfg, task, resolved_n_fit=resolved_n_fit
        ).scientific_dict(),
        {"online_extraction": extraction_dir},
    )
    if task["dataset"].split("/")[-1].casefold() == "weather":
        pipeline["data.missing_values"] = str(cfg.missing_values or "zero")
    allocation = allocate_run(
        root,
        project=PROJECT,
        workflow=f"online_adaptation/{family}",
        dataset=task["dataset"],
        lookback=task["lookback"],
        horizon=task["horizon"],
        backbone=task["backbone"],
        model_config_order=ADAPTER_ORDER,
        model_config=identity,
        pipeline_config=pipeline,
        runtime_config={"device": str(cfg.device)},
        seeds=(int(cfg.seed),),
        purpose=str(cfg.purpose),
        mode=str(cfg.mode),
        display_name=_display_name(family, task),
        policy=str(cfg.conflict_policy),
        skip_completed=bool(cfg.skip_completed),
    )
    if allocation.action == "skip":
        return
    mark_status(allocation.run_dir, "running")
    try:
        adapter = _adapter_config(
            cfg, task, resolved_n_fit=resolved_n_fit
        )
        target_cols = (
            _homogeneous_targets(task["dataset"])
            if task["homogeneous_only"]
            else None
        )
        dataset = load_csv_dataset(
            _dataset_path(data_root, task["dataset"]),
            dataset_name=task["dataset"].split("/")[-1],
            target_cols=target_cols,
            drop_users=cfg.drop_users,
            missing_values=cfg.missing_values,
        )
        if task["method"] == "tsrag":
            outputs = evaluate_online_tsrag(
                extraction_dir=extraction_dir,
                output_dir=allocation.run_dir,
                dataset=dataset,
                chronos_bolt_weights=weights_root / "chronos-bolt-base",
                tsrag_weights=weights_root / "ts-rag",
                device=str(cfg.device),
                batch_size=int(cfg.tsrag_batch_size),
            )
        elif task["method"] == "covariate_prediction":
            device = resolve_device(str(cfg.device))
            model = _load_backbone(
                task,
                weights_root=weights_root,
                device=device,
                normalization=str(cfg.normalization),
            )
            outputs = evaluate_covariate_prediction(
                extraction_dir=extraction_dir,
                output_dir=allocation.run_dir,
                config=adapter,
                dataset=dataset,
                model=model,
                device=device,
            )
        elif str(task["method"]).startswith(("bayes_", "catboost_")):
            device = resolve_device(str(cfg.device))
            model = _load_backbone(
                task,
                weights_root=weights_root,
                device=device,
                normalization=str(cfg.normalization),
            )
            outputs = evaluate_online_gate(
                extraction_dir=extraction_dir,
                output_dir=allocation.run_dir,
                config=adapter,
                dataset=dataset,
                model=model,
                device=device,
            )
        else:
            device = resolve_device(str(cfg.device))
            model = _load_backbone(
                task,
                weights_root=weights_root,
                device=device,
                normalization=str(cfg.normalization),
            )
            outputs = evaluate_online_linear(
                extraction_dir=extraction_dir,
                output_dir=allocation.run_dir,
                config=adapter,
                dataset=dataset,
                model=model,
                device=device,
            )
        required = [str(path.relative_to(allocation.run_dir)) for path in outputs.values()]
        mark_status(allocation.run_dir, "completed", required_artifacts=required)
    except Exception:
        mark_status(allocation.run_dir, "interrupted")
        raise


def _configured_tasks(
    cfg: DictConfig,
    data_root: Path,
    *,
    deadline_part: str | None = None,
) -> list[dict[str, Any]]:
    family = str(cfg.family)
    selected_datasets = (
        None if cfg.datasets is None else [str(value) for value in cfg.datasets]
    )
    selected_ranges = (
        None if cfg.ranges is None else [str(value) for value in cfg.ranges]
    )
    tasks = tasks_for_family(
        family,
        str(cfg.mode),
        data_root,
        selected_datasets=selected_datasets,
        selected_ranges=selected_ranges,
        deadline_part=(
            str(cfg.deadline_part) if deadline_part is None else str(deadline_part)
        ),
    )
    deadline_family = family in {
        "deadline_fixed_protocol",
        "deadline_tsrag_comparison",
    }
    for task in tasks:
        if (
            not deadline_family
            and family != "n_datastore_dates_ablation"
            and cfg.n_datastore_dates is not None
        ):
            value = cfg.n_datastore_dates
            task["n_datastore_dates"] = int(value) if isinstance(value, int) else float(value)
        if not deadline_family and family != "n_fit_ablation":
            task["n_fit"] = int(cfg.n_fit)
        if (
            not deadline_family
            and family != "fit_stride_ablation"
            and cfg.fit_stride is not None
        ):
            task["fit_stride"] = int(cfg.fit_stride)
        if not deadline_family and cfg.store_stride is not None:
            task["store_stride"] = int(cfg.store_stride)
        if not deadline_family and cfg.align_period is not None:
            task["align_period"] = bool(cfg.align_period)
        if not deadline_family and int(cfg.period):
            task["period"] = int(cfg.period)
        if not deadline_family and family != "fixed_protocol_ablation":
            if cfg.fixed_datastore is not None:
                task["fixed_datastore"] = bool(cfg.fixed_datastore)
            if cfg.fixed_training_set is not None:
                task["fixed_training_set"] = bool(cfg.fixed_training_set)
        if not deadline_family and cfg.eval_start_date is not None:
            task["eval_start_date"] = cfg.eval_start_date
        if not deadline_family and cfg.eval_end_date is not None:
            task["eval_end_date"] = cfg.eval_end_date
        if family not in {"general_scope_ablation", "deadline_fixed_protocol"}:
            task["fitting_scope"] = str(cfg.fitting_scope)
        if family != "alpha_ablation":
            task["alpha"] = float(cfg.alpha)
        if not deadline_family and cfg.retrieval_covariate_mode is not None:
            task["retrieval_covariate_mode"] = str(cfg.retrieval_covariate_mode)
        if family == "tsrag_comparison" and cfg.tsrag_k is not None:
            task["max_k"] = int(cfg.tsrag_k)
            task["used_k"] = int(cfg.tsrag_k)
            task["candidate_k_grid"] = (int(cfg.tsrag_k),)
        elif task["method"] == "tsrag":
            task["tune_alpha"] = False
        else:
            if not deadline_family and cfg.max_k is not None:
                task["max_k"] = int(cfg.max_k)
            if (
                not deadline_family
                and family != "k_ablation"
                and cfg.used_k is not None
            ):
                task["used_k"] = (
                    int(cfg.used_k)
                )
            if str(task["method"]).startswith(("bayes_", "catboost_")):
                task["used_k"] = (
                    int(task["max_k"])
                    if cfg.used_k is None
                    else int(cfg.used_k)
                )
        task["validation_ratio"] = (
            0.2 if deadline_family else float(cfg.ridge_validation_ratio)
        )
        task["alpha_grid"] = (
            (0.1, 0.01, 0.001)
            if deadline_family
            else tuple(float(value) for value in cfg.ridge_alpha_grid)
        )
        if not deadline_family and cfg.candidate_k_grid is not None:
            task["candidate_k_grid"] = tuple(
                int(value) for value in cfg.candidate_k_grid
            )
        if not deadline_family:
            task["tune_alpha"] = bool(task["tune_alpha"] and bool(cfg.tune_alpha))
        required_k = (
            int(task["used_k"])
            if task["used_k"] is not None
            else max(task["candidate_k_grid"])
        )
        if required_k > int(task["max_k"]):
            raise ValueError(
                f"task requires K={required_k}, but max_k={task['max_k']}"
            )
    return tasks


def run_workflow(
    cfg: DictConfig,
    *,
    stages_override: set[str] | None = None,
) -> None:
    setup_logging()
    stages = set(stages_override) if stages_override is not None else _stages(str(cfg.stages))
    project_root = Path.cwd().resolve()
    data_root = Path(cfg.data_root).expanduser()
    weights_root = Path(cfg.weights_root).expanduser()
    output_root = Path(cfg.outputs_root).expanduser()
    data_root = (project_root / data_root).resolve() if not data_root.is_absolute() else data_root.resolve()
    weights_root = (project_root / weights_root).resolve() if not weights_root.is_absolute() else weights_root.resolve()
    output_root = (project_root / output_root).resolve() if not output_root.is_absolute() else output_root.resolve()
    family = str(cfg.family)
    tasks = _configured_tasks(cfg, data_root)
    LOGGER.info("family=%s mode=%s tasks=%s stages=%s", family, cfg.mode, len(tasks), sorted(stages))
    set_seed(int(cfg.seed))
    for index, task in enumerate(tasks, start=1):
        LOGGER.info("task=%s/%s config=%s", index, len(tasks), task)
        try:
            extraction_allocation = _allocate_extraction(cfg, task, output_root)
            if "extract" in stages:
                _run_extraction(
                    cfg,
                    task,
                    extraction_allocation,
                    data_root=data_root,
                    weights_root=weights_root,
                )
            elif extraction_allocation.action != "skip":
                raise FileNotFoundError(
                    f"adapt stage requires a completed extraction: {extraction_allocation.run_dir}"
                )
            if "adapt" in stages:
                _run_adaptation(
                    cfg,
                    family,
                    task,
                    extraction_allocation.run_dir,
                    output_root=output_root,
                    data_root=data_root,
                    weights_root=weights_root,
                )
        except Exception:
            LOGGER.exception(
                "task=%s/%s completed status=failed config=%s",
                index,
                len(tasks),
                task,
            )
            raise
        LOGGER.info(
            "task=%s/%s completed status=success config=%s",
            index,
            len(tasks),
            task,
        )
    if "tables" in stages:
        report_tasks = (
            _configured_tasks(cfg, data_root, deadline_part="all")
            if bool(cfg.deadline_finalize)
            else tasks
        )
        report = build_online_tables(
            results_root=output_root / "online_adaptation" / family,
            output_dir=output_root / "reports" / family / str(cfg.mode),
            expected=[
                {
                    "dataset": task["dataset"],
                    "lookback": task["lookback"],
                    "horizon": task["horizon"],
                    "backbone": task["backbone"],
                    "model": _display_name(family, task),
                }
                for task in report_tasks
            ],
            pipeline_config=_table_pipeline_filters(),
            config_policy=os.environ.get(
                "TABLE_CONFIG_POLICY", str(cfg.table_config_policy)
            ),
            repeat_policy=os.environ.get(
                "TABLE_REPEAT_POLICY", str(cfg.table_repeat_policy)
            ),
            purposes=[os.environ.get("TABLE_PURPOSE", str(cfg.purpose))],
            seeds=[int(cfg.seed)],
            vanilla_source_policy=os.environ.get(
                "TABLE_VANILLA_SOURCE_POLICY", "strict"
            ),
        )
        if family == "sota_backbone_ablation":
            build_published_sota_table(
                detailed_csv=report["detailed_csv"],
                published_json=project_root / "PUBLISHED_BASELINES.json",
                output_dir=output_root / "reports" / family / str(cfg.mode),
            )
def run_stage(cfg: DictConfig, stage: str) -> None:
    """Run one explicit Slurm-facing workflow stage."""
    if stage not in {"extract", "adapt", "tables"}:
        raise ValueError(f"unknown workflow stage {stage!r}")
    run_workflow(cfg, stages_override={stage})
