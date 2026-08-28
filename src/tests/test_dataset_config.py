from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.load_dataset import load_csv_dataset


class DatasetConfigTest(unittest.TestCase):
    def test_scoped_and_run_drop_users_replace_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01", periods=3, freq="h"),
                    "a": [1.0, 2.0, 3.0],
                    "b": [4.0, 5.0, 6.0],
                    "c": [7.0, 8.0, 9.0],
                }
            ).to_csv(root / "tiny.csv", index=False)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "date_col": "date",
                        "drop_users": [0],
                        "online_adaptation": {"drop_users": [1]},
                    }
                ),
                encoding="utf-8",
            )

            inherited = load_csv_dataset(root, dataset_name="tiny")
            explicit_all = load_csv_dataset(root, dataset_name="tiny", drop_users=[])
            explicit = load_csv_dataset(root, dataset_name="tiny", drop_users=[2])

        self.assertEqual(inherited.user_names, ["a", "c"])
        self.assertEqual(explicit_all.user_names, ["a", "b", "c"])
        self.assertEqual(explicit.user_names, ["a", "b"])

    def test_missing_values_are_zero_filled_or_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01", periods=3, freq="h"),
                    "a": [1.0, None, 3.0],
                }
            ).to_csv(root / "tiny.csv", index=False)
            (root / "config.json").write_text(
                json.dumps({"date_col": "date", "missing_values": "zero"}),
                encoding="utf-8",
            )
            filled = load_csv_dataset(root, dataset_name="tiny")
            self.assertEqual(filled.missing_values_replaced, 1)
            self.assertEqual(float(filled.values[1, 0]), 0.0)
            with self.assertRaisesRegex(ValueError, "1 missing values"):
                load_csv_dataset(root, dataset_name="tiny", missing_values="error")


if __name__ == "__main__":
    unittest.main()
