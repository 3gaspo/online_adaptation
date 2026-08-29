"""Common-date detailed and equal-configuration online adaptation tables."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.pipeline.runs import (
    SelectedRun,
    load_manifest,
    manifest_is_selectable,
    select_identity_runs,
    write_report_manifest,
)


METRICS = (
    "mse",
    "nmse",
    "mae",
    "nmae",
    "relative_nmse_improvement_pct",
    "relative_mse_improvement_pct",
    "win_rate_pct",
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {
            key: (int(value) if key in {"query_date", "windows", "values"} else value if key == "method" else float(value))
            for key, value in row.items()
        }
        for row in csv.DictReader(path.open(encoding="utf-8"))
    ]


def _summarize(rows: list[dict[str, Any]], *, vanilla: bool = False) -> dict[str, float]:
    prefix = "vanilla_" if vanilla else ""
    result = {
        metric: float(np.mean([row[f"{prefix}{metric}"] for row in rows]))
        for metric in ("mse", "nmse", "mae", "nmae")
    }
    vanilla_mse = float(np.mean([row["vanilla_mse"] for row in rows]))
    vanilla_nmse = float(np.mean([row["vanilla_nmse"] for row in rows]))
    result["relative_mse_improvement_pct"] = (
        0.0 if vanilla else 100.0 * (vanilla_mse - result["mse"]) / max(vanilla_mse, 1e-12)
    )
    result["relative_nmse_improvement_pct"] = (
        0.0 if vanilla else 100.0 * (vanilla_nmse - result["nmse"]) / max(vanilla_nmse, 1e-12)
    )
    result["win_rate_pct"] = 0.0 if vanilla else 100.0 * float(
        np.mean([row["win_rate"] for row in rows])
    )
    return result


def _average_rows(row_sets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Average selected runs only when their evaluation dates are identical."""
    indexed = []
    for rows in row_sets:
        by_date = {int(row["query_date"]): row for row in rows}
        if len(by_date) != len(rows):
            raise ValueError("per-date metrics contain duplicate query dates")
        indexed.append(by_date)
    common_dates = set(indexed[0])
    if any(set(rows) != common_dates for rows in indexed[1:]):
        raise ValueError("selected runs do not use the same evaluation query dates")
    averaged: list[dict[str, Any]] = []
    for date in sorted(common_dates):
        rows = [values[date] for values in indexed]
        item: dict[str, Any] = {"query_date": date}
        for key in rows[0]:
            if key == "query_date":
                continue
            if key == "method":
                item[key] = rows[0][key]
            else:
                item[key] = float(np.mean([float(row[key]) for row in rows]))
        averaged.append(item)
    return averaged


def _require_equivalent_vanilla(
    row_sets: list[list[dict[str, Any]]],
    *,
    context: str,
) -> None:
    """Fail when runs advertised as one vanilla model contain different results."""
    fields = (
        "vanilla_mse",
        "vanilla_nmse",
        "vanilla_mae",
        "vanilla_nmae",
        "windows",
        "values",
    )
    reference = {int(row["query_date"]): row for row in row_sets[0]}
    for rows in row_sets[1:]:
        candidate = {int(row["query_date"]): row for row in rows}
        if set(candidate) != set(reference):
            raise ValueError(f"vanilla sources use different evaluation dates for {context}")
        for date, expected in reference.items():
            observed = candidate[date]
            if any(
                not np.isclose(
                    float(observed[field]),
                    float(expected[field]),
                    rtol=1e-10,
                    atol=1e-12,
                )
                for field in fields
            ):
                raise ValueError(
                    f"vanilla sources disagree for {context} at query_date={date}"
                )


def _require_matching_vanilla_support(
    row_sets: list[list[dict[str, Any]]],
    *,
    context: str,
) -> None:
    """Require identical dates and aggregation support before diagnostic reuse."""
    reference = {int(row["query_date"]): row for row in row_sets[0]}
    for rows in row_sets[1:]:
        candidate = {int(row["query_date"]): row for row in rows}
        if set(candidate) != set(reference):
            raise ValueError(f"vanilla sources use different evaluation dates for {context}")
        for date, expected in reference.items():
            observed = candidate[date]
            if any(
                int(observed[field]) != int(expected[field])
                for field in ("windows", "values")
            ):
                raise ValueError(
                    f"vanilla sources use different aggregation support for "
                    f"{context} at query_date={date}"
                )


def _vanilla_source_drift(
    row_sets: list[list[dict[str, Any]]],
) -> tuple[float, float]:
    """Return maximum absolute and relative metric drift from the first source."""
    fields = ("vanilla_mse", "vanilla_nmse", "vanilla_mae", "vanilla_nmae")
    reference = {int(row["query_date"]): row for row in row_sets[0]}
    max_absolute = 0.0
    max_relative = 0.0
    for rows in row_sets[1:]:
        candidate = {int(row["query_date"]): row for row in rows}
        for date, expected in reference.items():
            observed = candidate[date]
            for field in fields:
                expected_value = float(expected[field])
                absolute = abs(float(observed[field]) - expected_value)
                relative = absolute / max(abs(expected_value), 1e-12)
                max_absolute = max(max_absolute, absolute)
                max_relative = max(max_relative, relative)
    return max_absolute, max_relative


def _dependency_key(manifest: dict[str, Any]) -> str:
    pipeline = manifest.get("config", {}).get("pipeline", {})
    dependency = pipeline.get("dependency.online_extraction", {})
    encoded = json.dumps(dependency, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _latex(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    labels = {
        "dataset": "Dataset",
        "setting": "Setting",
        "model": "Model",
        "mse": "MSE",
        "nmse": "nMSE",
        "mae": "MAE",
        "nmae": "nMAE",
        "relative_nmse_improvement_pct": "$\\Delta$ nMSE (\\%)",
        "relative_mse_improvement_pct": "$\\Delta$ MSE (\\%)",
        "win_rate_pct": "Win rate (\\%)",
    }
    spec = "ll" + "r" * (len(columns) - 2) if len(columns) > 2 else "l" * len(columns)
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]
    lines.append(" & ".join(labels.get(column, column) for column in columns) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("_", "\\_"))
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_online_tables(
    *,
    results_root: str | Path,
    output_dir: str | Path,
    expected: list[dict[str, Any]] | None = None,
    pipeline_config: dict[str, Any] | None = None,
    config_policy: str = "distinct",
    repeat_policy: str = "selected",
    purposes: list[str] | None = None,
    seeds: list[int] | None = None,
    vanilla_source_policy: str = "strict",
) -> dict[str, Path]:
    """Build tables after intersecting dates across compared runs per setting."""
    if vanilla_source_policy not in {"strict", "first"}:
        raise ValueError("vanilla_source_policy must be strict or first")
    expected_keys = None
    if expected is not None:
        expected_keys = {
            (
                str(item["dataset"]),
                int(item["lookback"]),
                int(item["horizon"]),
                str(item["backbone"]),
                str(item["model"]),
            )
            for item in expected
        }
    root = Path(results_root).expanduser().resolve()
    active_launch = os.environ.get("EXPERIMENT_LAUNCH_ID")
    identity_roots = sorted(
        {
            path.parent.parent
            for path in root.rglob("manifest.json")
            if path.parent.name.startswith("run_")
            and "archive" not in path.relative_to(root).parts
        }
    )
    selected_runs: list[SelectedRun] = []
    raw_entries: list[dict[str, Any]] = []
    for identity_root in identity_roots:
        manifests = [load_manifest(path) for path in identity_root.glob("run_*/manifest.json")]
        selectable = [
            manifest
            for manifest in manifests
            if manifest_is_selectable(manifest, allow_ready_launch_id=active_launch)
        ]
        if not selectable:
            continue
        if expected_keys is not None:
            candidate_keys = {
                (
                    str(manifest["identity"]["dataset"]),
                    int(manifest["identity"]["lookback"]),
                    int(manifest["identity"]["horizon"]),
                    str(manifest["identity"]["backbone"]),
                    str(manifest.get("table", {}).get("display_name") or manifest["identity"]["backbone"]),
                )
                for manifest in selectable
            }
            if not candidate_keys & expected_keys:
                continue
        chosen = select_identity_runs(
            identity_root,
            requested_pipeline=pipeline_config,
            config_policy=config_policy,
            repeat_policy=repeat_policy,
            purposes=purposes,
            seeds=seeds,
            allow_ready_launch_id=active_launch,
        )
        for selected in chosen:
            manifest = dict(selected.manifest)
            identity = manifest["identity"]
            base_model = str(
                manifest.get("table", {}).get("display_name")
                or identity["backbone"]
            )
            key = (
                str(identity["dataset"]),
                int(identity["lookback"]),
                int(identity["horizon"]),
                str(identity["backbone"]),
                base_model,
            )
            if expected_keys is not None and key not in expected_keys:
                continue
            result_path = selected.run_dir / "result_manifest.json"
            per_date_path = selected.run_dir / "per_date_metrics.csv"
            if not result_path.is_file() or not per_date_path.is_file():
                raise FileNotFoundError(
                    f"selected run is missing online report artifacts: {selected.run_dir}"
                )
            raw_entries.append(
                {
                    "dataset": identity["dataset"],
                    "lookback": int(identity["lookback"]),
                    "horizon": int(identity["horizon"]),
                    "backbone": identity["backbone"],
                    "base_model": base_model,
                    "model": selected.label,
                    "dependency": _dependency_key(manifest),
                    "rows": _read_rows(per_date_path),
                }
            )
            selected_runs.append(selected)

    merged: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for entry in raw_entries:
        key = (
            entry["dataset"],
            entry["lookback"],
            entry["horizon"],
            entry["backbone"],
            entry["model"],
        )
        merged.setdefault(key, []).append(entry)
    entries = [
        {
            **items[0],
            "rows": _average_rows([item["rows"] for item in items]),
            "dependency_rows": [
                (str(item["dependency"]), item["rows"]) for item in items
            ],
        }
        for _, items in sorted(merged.items(), key=lambda item: tuple(str(value) for value in item[0]))
    ]
    if not entries:
        raise ValueError(f"no completed online result runs below {results_root}")
    if expected_keys is not None:
        actual_keys = {
            (
                str(entry["dataset"]),
                int(entry["lookback"]),
                int(entry["horizon"]),
                str(entry["backbone"]),
                str(entry["base_model"]),
            )
            for entry in entries
        }
        missing = sorted(expected_keys - actual_keys)
        if missing:
            raise ValueError(f"online report is missing expected completed runs: {missing}")

    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(
            (entry["dataset"], entry["lookback"], entry["horizon"]), []
        ).append(entry)
    detailed: list[dict[str, Any]] = []
    vanilla_source_groups: list[dict[str, Any]] = []
    for (dataset, lookback, horizon), group in sorted(groups.items()):
        date_sets = [{int(row["query_date"]) for row in entry["rows"]} for entry in group]
        common_dates = date_sets[0]
        if any(dates != common_dates for dates in date_sets[1:]):
            raise ValueError(
                f"models do not use identical evaluation dates for "
                f"{dataset} {lookback}:{horizon}"
            )
        filtered = [
            [row for row in entry["rows"] if int(row["query_date"]) in common_dates]
            for entry in group
        ]
        vanilla_groups: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {}
        for entry in group:
            for dependency, rows in entry["dependency_rows"]:
                vanilla_groups.setdefault(str(entry["backbone"]), []).append(
                    (
                        dependency,
                        [row for row in rows if int(row["query_date"]) in common_dates],
                    )
                )
        for backbone, sources in sorted(vanilla_groups.items()):
            sources = sorted(sources, key=lambda item: item[0])
            row_sets = [rows for _, rows in sources]
            context = f"{dataset} {lookback}:{horizon} backbone={backbone}"
            _require_matching_vanilla_support(row_sets, context=context)
            if vanilla_source_policy == "strict":
                _require_equivalent_vanilla(row_sets, context=context)
                vanilla_rows = _average_rows(row_sets)
                selected_dependency = None
            else:
                selected_dependency, vanilla_rows = sources[0]
            max_absolute_drift, max_relative_drift = _vanilla_source_drift(
                row_sets
            )
            label = f"Vanilla ({backbone})"
            detailed.append(
                {
                    "dataset": dataset,
                    "setting": f"{lookback}:{horizon}",
                    "model": label,
                    **_summarize(vanilla_rows, vanilla=True),
                    "dates": len(common_dates),
                }
            )
            vanilla_source_groups.append(
                {
                    "dataset": dataset,
                    "setting": f"{lookback}:{horizon}",
                    "backbone": backbone,
                    "model": label,
                    "dependency_signatures": sorted(
                        {dependency for dependency, _ in sources}
                    ),
                    "source_policy": vanilla_source_policy,
                    "selected_dependency_signature": selected_dependency,
                    "maximum_absolute_metric_drift": max_absolute_drift,
                    "maximum_relative_metric_drift": max_relative_drift,
                }
            )
        for entry, rows in zip(group, filtered, strict=True):
            detailed.append(
                {
                    "dataset": dataset,
                    "setting": f"{lookback}:{horizon}",
                    "model": entry["model"],
                    **_summarize(rows),
                    "dates": len(common_dates),
                }
            )

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    detailed_columns = ["dataset", "setting", "model", *METRICS, "dates"]
    detailed_path = output / "detailed_results.csv"
    with detailed_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=detailed_columns)
        writer.writeheader()
        writer.writerows(detailed)
    detailed_tex = output / "detailed_results.tex"
    _latex(detailed_tex, detailed, detailed_columns[:-1])

    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in detailed:
        by_model.setdefault(str(row["model"]), []).append(row)
    common_configurations = set.intersection(
        *(
            {(str(row["dataset"]), str(row["setting"])) for row in rows}
            for rows in by_model.values()
        )
    )
    if not common_configurations:
        raise ValueError("compared models have no common dataset and L:H settings")
    average = [
        {
            "model": model,
            **{
                metric: float(
                    np.mean(
                        [
                            float(row[metric])
                            for row in rows
                            if (str(row["dataset"]), str(row["setting"]))
                            in common_configurations
                        ]
                    )
                )
                for metric in METRICS
            },
            "configurations": len(common_configurations),
        }
        for model, rows in sorted(by_model.items())
    ]
    average_columns = ["model", *METRICS, "configurations"]
    average_path = output / "average_results.csv"
    with average_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=average_columns)
        writer.writeheader()
        writer.writerows(average)
    average_tex = output / "average_results.tex"
    _latex(average_tex, average, average_columns[:-1])
    report_manifest = write_report_manifest(
        output / "report_manifest.json",
        inputs=selected_runs,
        config_policy=config_policy,
        repeat_policy=repeat_policy,
        filters={
            "pipeline_config": dict(pipeline_config or {}),
            "purposes": list(purposes or []),
            "seeds": list(seeds or []),
            "expected": list(expected or []),
            "common_date_policy": "identical_dates_required_within_dataset_setting",
            "selected_run_average_policy": "equal_run_mean_on_identical_dates",
            "configuration_average_policy": (
                "equal_mean_on_dataset_setting_intersection_across_all_models"
            ),
            "vanilla_source_policy": vanilla_source_policy,
            "vanilla_source_groups": vanilla_source_groups,
            "files": {
                "detailed_csv": detailed_path.name,
                "detailed_tex": detailed_tex.name,
                "average_csv": average_path.name,
                "average_tex": average_tex.name,
            },
        },
    )
    return {
        "detailed_csv": detailed_path,
        "detailed_tex": detailed_tex,
        "average_csv": average_path,
        "average_tex": average_tex,
        "manifest": report_manifest,
    }


def build_published_sota_table(
    *,
    detailed_csv: str | Path,
    published_json: str | Path,
    output_dir: str | Path,
) -> Path:
    """Place causal online MSE beside immutable paper MSE with explicit protocols."""
    detailed = list(csv.DictReader(Path(detailed_csv).open(encoding="utf-8")))
    published = json.loads(Path(published_json).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    local_values: list[float] = []
    for row in detailed:
        if str(row["model"]).startswith("Vanilla"):
            continue
        value = float(row["mse"])
        local_values.append(value)
        rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "mse": value,
                "protocol": "causal online no-split; raw source scale",
            }
        )
    if local_values:
        rows.append(
            {
                "dataset": "Average",
                "model": "Online ridge (chronos_bolt)",
                "mse": float(np.mean(local_values)),
                "protocol": "equal-dataset causal online mean; raw source scale",
            }
        )
    for model, values in published["published_results"].items():
        for dataset, value in values.items():
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "mse": float(value),
                    "protocol": "published official test; train-split standardized",
                }
            )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "published_mse_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "model", "mse", "protocol"))
        writer.writeheader()
        writer.writerows(rows)
    (output / "published_mse_comparison_note.txt").write_text(
        str(published["comparability_note"]) + "\n", encoding="utf-8"
    )
    return path
