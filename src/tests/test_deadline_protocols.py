"""Focused contracts for the temporary deadline experiment fronts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.pipeline.profiles import DEADLINE_TIME_DATASETS, tasks_for_family
from src.proposal.contracts import ExtractionConfig
from src.proposal.date_planning import build_date_plan


ROOT = Path(__file__).resolve().parents[2]
TIME_METADATA = {
    "time/ne_china_wind_h": (8_765, 4, "H"),
    "time/coastal_t_s_h_part11": (8_784, 6, "H"),
    "time/sg_weather_d": (2_953, 24, "D"),
}


def _config(task: dict, *, query_stride: int = 127) -> ExtractionConfig:
    return ExtractionConfig(
        dataset=task["dataset"],
        lookback=task["lookback"],
        horizon=task["horizon"],
        backbone=task["backbone"],
        n_datastore_dates=task["n_datastore_dates"],
        n_store_windows=task["n_store_windows"],
        n_fit=task["n_fit"],
        max_k=task["max_k"],
        retrieval_scope=task["retrieval_scope"],
        fixed_datastore=task["fixed_datastore"],
        fixed_training_set=task["fixed_training_set"],
        include_fitting_windows=task["include_fitting_windows"],
        store_stride=task["store_stride"],
        fit_stride=task["fit_stride"],
        align_period=task["align_period"],
        period=task["period"],
        query_stride=query_stride,
        eval_start_date=task["eval_start_date"],
        eval_end_date=task["eval_end_date"],
        split_ratios=task["split_ratios"],
    )


class DeadlineProtocolTest(unittest.TestCase):
    def test_fixed_ablation_has_one_t3_grid_per_dataset_setting(self) -> None:
        tasks = tasks_for_family(
            "deadline_fixed_protocol",
            "small",
            ROOT / "datasets",
            selected_datasets=["Electricity", "Solar"],
            selected_ranges=["short", "long"],
        )
        self.assertEqual(len(tasks), 16)
        sizes = {"Electricity": (26_304, 312), "Solar": (8_760, 137)}
        grouped: dict[tuple[str, int, int], list[tuple[int, ...]]] = {}
        arms = set()
        for task in tasks:
            plan = build_date_plan(
                n_dates=sizes[task["dataset"]][0],
                n_users=sizes[task["dataset"]][1],
                config=_config(task),
            )
            grouped.setdefault(
                (task["dataset"], task["lookback"], task["horizon"]), []
            ).append(plan.evaluation_query_dates)
            arms.add((task["fixed_datastore"], task["fixed_training_set"]))
            self.assertEqual(plan.split_boundaries[-1], sizes[task["dataset"]][0])
            self.assertLessEqual(
                plan.n_datastore_dates * sizes[task["dataset"]][1], 20_000
            )
            self.assertEqual(task["n_store_windows"], 20_000)
            self.assertEqual(task["split_ratios"], (0.3, 0.5, 0.2))
        self.assertEqual(arms, {(False, False), (False, True), (True, False), (True, True)})
        self.assertEqual(len(grouped), 4)
        for grids in grouped.values():
            self.assertEqual(len(set(grids)), 1)

    def test_time_ridge_and_tsrag_share_exact_t3_dates(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            data_root = Path(directory)
            catalog_root = data_root / "time"
            catalog_root.mkdir()
            datasets = []
            for name, (dates, users, frequency) in TIME_METADATA.items():
                datasets.append(
                    {
                        "name": name.removeprefix("time/"),
                        "configured_frequency": frequency,
                        "num_series": users,
                        "num_timestamps": dates,
                    }
                )
            (catalog_root / "catalog.json").write_text(
                json.dumps({"datasets": datasets}), encoding="utf-8"
            )
            tasks = tasks_for_family(
                "deadline_tsrag_comparison",
                "full",
                data_root,
                selected_datasets=list(DEADLINE_TIME_DATASETS),
            )

        self.assertEqual(len(tasks), 6)
        for dataset, (dates, users, _) in TIME_METADATA.items():
            ridge, tsrag = [task for task in tasks if task["dataset"] == dataset]
            ridge_plan = build_date_plan(
                n_dates=dates, n_users=users, config=_config(ridge)
            )
            tsrag_plan = build_date_plan(
                n_dates=dates, n_users=users, config=_config(tsrag)
            )
            self.assertEqual(
                ridge_plan.evaluation_query_dates,
                tsrag_plan.evaluation_query_dates,
            )
            self.assertEqual(ridge["n_store_windows"], 20_000)
            self.assertEqual(ridge["n_datastore_dates"], 20_000 // users)
            self.assertEqual(ridge["n_fit"], 30)
            self.assertEqual(ridge["fit_stride"], 24)
            self.assertIsNone(ridge["used_k"])
            self.assertEqual(ridge["candidate_k_grid"], (1, 5, 10, 15))
            self.assertTrue(tsrag["fixed_datastore"])
            self.assertEqual(tsrag["n_store_windows"], 20_000)
            self.assertEqual(tsrag["retrieval_scope"], "same_user")
            self.assertEqual(tsrag["split_ratios"], (0.3, 0.5, 0.2))
            self.assertEqual(tsrag["used_k"], 5)


if __name__ == "__main__":
    unittest.main()
