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


if __name__ == "__main__":
    unittest.main()
