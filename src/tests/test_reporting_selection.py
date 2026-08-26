"""Focused online report selection and launcher-contract checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile

from src.pipeline.runs import allocate_run, mark_status
from src.results.reporting import build_online_tables


def _complete_run(
    identity: Path,
    *,
    query_stride: int,
    dates: tuple[int, ...],
    mse: float,
) -> Path:
    allocation = allocate_run(
        identity,
        project="online_adaptation",
        workflow="report_test",
        dataset="synthetic",
        lookback=4,
        horizon=2,
        backbone="chronos2",
        model_config_order=("method",),
        model_config={"method": "ridge"},
        pipeline_config={
            "dependency.online_extraction": {
                "schema_version": 1,
                "pipeline": {"query_stride": query_stride},
            }
        },
        seeds=(1,),
        purpose="publication",
        display_name="ridge",
    )
    run = allocation.run_dir
    (run / "result_manifest.json").write_text(
        json.dumps({"method": "ridge"}), encoding="utf-8"
    )
    rows = [
        {
            "query_date": date,
            "method": "ridge",
            "mse": mse,
            "mae": mse,
            "nmse": mse,
            "nmae": mse,
            "vanilla_mse": 1.0,
            "vanilla_mae": 1.0,
            "vanilla_nmse": 1.0,
            "vanilla_nmae": 1.0,
            "win_rate": 1.0,
            "windows": 1,
            "values": 2,
        }
        for date in dates
    ]
    with (run / "per_date_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mark_status(
        run,
        "completed",
        required_artifacts=("result_manifest.json", "per_date_metrics.csv"),
    )
    return run


def test_distinct_average_and_nested_filter_selection() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        identity = root / "results/ridge"
        _complete_run(identity, query_stride=1, dates=(1, 2, 3), mse=0.4)
        _complete_run(identity, query_stride=2, dates=(2, 3, 4), mse=0.6)
        expected = [
            {
                "dataset": "synthetic",
                "lookback": 4,
                "horizon": 2,
                "backbone": "chronos2",
                "model": "ridge",
            }
        ]

        distinct = build_online_tables(
            results_root=root / "results",
            output_dir=root / "distinct",
            expected=expected,
            purposes=["publication"],
            seeds=[1],
        )
        distinct_rows = list(
            csv.DictReader(distinct["detailed_csv"].open(encoding="utf-8"))
        )
        ridge_rows = [row for row in distinct_rows if not row["model"].startswith("Vanilla")]
        assert len(ridge_rows) == 2
        assert all("dependency.online_extraction.pipeline.query_stride" in row["model"] for row in ridge_rows)
        distinct_manifest = json.loads(distinct["manifest"].read_text(encoding="utf-8"))
        assert distinct_manifest["obtained"]["count"] == 2

        filtered = build_online_tables(
            results_root=root / "results",
            output_dir=root / "filtered",
            expected=expected,
            pipeline_config={
                "dependency.online_extraction": {"pipeline": {"query_stride": 2}}
            },
            purposes=["publication"],
            seeds=[1],
        )
        filtered_manifest = json.loads(filtered["manifest"].read_text(encoding="utf-8"))
        assert filtered_manifest["obtained"]["count"] == 1

        averaged = build_online_tables(
            results_root=root / "results",
            output_dir=root / "averaged",
            expected=expected,
            config_policy="average",
            purposes=["publication"],
            seeds=[1],
        )
        averaged_rows = list(
            csv.DictReader(averaged["detailed_csv"].open(encoding="utf-8"))
        )
        averaged_ridge = [row for row in averaged_rows if row["model"] == "ridge"]
        assert len(averaged_ridge) == 1
        assert float(averaged_ridge[0]["mse"]) == 0.5
        assert averaged_ridge[0]["dates"] == "2"


def test_profile_contract() -> None:
    project_root = Path(__file__).resolve().parents[2]
    runner = (project_root / "src/slurm/run_family.sh").read_text(encoding="utf-8")
    assert 'PROFILE_N_STORE="${N_STORE:-30000}"' in runner
    assert 'PROFILE_PURPOSE="${PURPOSE:-smoke}"' in runner
    assert 'PROFILE_PURPOSE="${PURPOSE:-publication}"' in runner


if __name__ == "__main__":
    test_distinct_average_and_nested_filter_selection()
    test_profile_contract()
