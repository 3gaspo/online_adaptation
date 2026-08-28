"""Focused contracts for the temporary deadline experiment fronts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.pipeline.profiles import tasks_for_family
from src.proposal.contracts import ExtractionConfig
from src.proposal.date_planning import build_date_plan


ROOT = Path(__file__).resolve().parents[2]
TIME_METADATA = {
    "time/ne_china_wind_h": (8_765, 4, "H"),
    "time/coastal_t_s_h_part11": (8_784, 6, "H"),
    "time/sg_weather_d": (2_953, 24, "D"),
    "time/other_h": (8_760, 8, "H"),
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
        self.assertEqual(len(tasks), 32)
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
            arms.add(
                (
                    task["fixed_datastore"],
                    task["fixed_training_set"],
                    task["fitting_scope"],
                )
            )
            self.assertIsInstance(task["n_fit"], int)
            self.assertEqual(plan.split_boundaries[-1], sizes[task["dataset"]][0])
            self.assertLessEqual(
                plan.n_datastore_dates * sizes[task["dataset"]][1], 10_000
            )
            self.assertEqual(task["n_store_windows"], 10_000)
            self.assertEqual(task["n_fit"], 10)
            self.assertEqual(task["split_ratios"], (0.3, 0.5, 0.2))
        self.assertEqual(
            arms,
            {
                (fixed_datastore, fixed_training_set, fitting_scope)
                for fixed_datastore in (False, True)
                for fixed_training_set in (False, True)
                for fitting_scope in ("same_user", "all")
            },
        )
        self.assertEqual(len(grouped), 4)
        for grids in grouped.values():
            self.assertEqual(len(grids), 8)
            self.assertEqual(len(set(grids)), 1)

        priority = tasks_for_family(
            "deadline_fixed_protocol",
            "small",
            ROOT / "datasets",
            selected_datasets=["Electricity", "Solar"],
            selected_ranges=["short", "long"],
            deadline_part="priority",
        )
        remainder = tasks_for_family(
            "deadline_fixed_protocol",
            "small",
            ROOT / "datasets",
            selected_datasets=["Electricity", "Solar"],
            selected_ranges=["short", "long"],
            deadline_part="remainder",
        )
        online_per_user = tasks_for_family(
            "deadline_fixed_protocol",
            "small",
            ROOT / "datasets",
            selected_datasets=["Electricity", "Solar"],
            selected_ranges=["short", "long"],
            deadline_part="online_per_user",
        )
        fixed_shared = tasks_for_family(
            "deadline_fixed_protocol",
            "small",
            ROOT / "datasets",
            selected_datasets=["Electricity", "Solar"],
            selected_ranges=["short", "long"],
            deadline_part="fixed_shared",
        )
        self.assertEqual(len(priority), 8)
        self.assertEqual(len(remainder), 24)
        self.assertEqual(len(online_per_user), 4)
        self.assertEqual(len(fixed_shared), 4)
        self.assertEqual(
            {
                (
                    task["fixed_datastore"],
                    task["fixed_training_set"],
                    task["fitting_scope"],
                )
                for task in priority
            },
            {(False, False, "same_user"), (True, True, "all")},
        )

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
                selected_datasets=list(TIME_METADATA),
            )

            priority = tasks_for_family(
                "deadline_tsrag_comparison",
                "full",
                data_root,
                selected_datasets=list(TIME_METADATA),
                deadline_part="priority",
            )
            remainder = tasks_for_family(
                "deadline_tsrag_comparison",
                "full",
                data_root,
                selected_datasets=list(TIME_METADATA),
                deadline_part="remainder",
            )

        self.assertEqual(len(tasks), 8)
        self.assertEqual(len(priority), 6)
        self.assertEqual(len(remainder), 2)
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
            self.assertEqual(ridge["n_store_windows"], 10_000)
            self.assertLessEqual(ridge_plan.n_datastore_dates * users, 10_000)
            self.assertEqual(ridge["n_fit"], 10)
            self.assertEqual(ridge["fit_stride"], 24)
            self.assertTrue(ridge["align_period"])
            self.assertEqual(ridge["store_stride"], ridge["period"])
            fit_gap = (
                ((ridge["horizon"] + ridge["period"] - 1) // ridge["period"])
                * ridge["period"]
                if ridge["fit_stride"] % ridge["period"] == 0
                else ridge["horizon"]
            )
            self.assertEqual(
                ridge_plan.fitting_dates[-1] + fit_gap,
                ridge_plan.evaluation_start_date,
            )
            self.assertIsNone(ridge["used_k"])
            self.assertEqual(ridge["candidate_k_grid"], (1, 5, 10, 15))
            self.assertTrue(tsrag["fixed_datastore"])
            self.assertEqual(tsrag["n_store_windows"], 10_000)
            self.assertEqual(tsrag["retrieval_scope"], "same_user")
            self.assertEqual(tsrag["split_ratios"], (0.3, 0.5, 0.2))
            self.assertEqual(tsrag["used_k"], 5)
            self.assertFalse(tsrag["align_period"])
            self.assertEqual(tsrag["store_stride"], 1)


if __name__ == "__main__":
    unittest.main()
