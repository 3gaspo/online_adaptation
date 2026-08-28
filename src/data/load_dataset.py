"""CSV dataset loading and shared experiment helpers."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


LOGGER = logging.getLogger(__name__)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _split_text(value: str) -> list[str]:
    return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return _split_text(value)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    return [value]


def _column_names(columns: Sequence[Any] | str | None) -> list[str]:
    return [str(item) for item in _as_list(columns)]


DATASET_CONFIG_KEYS = {
    "target_cols",
    "date_col",
    "drop_users",
    "rename_users",
    "aggr",
    "aggr_period",
    "missing_values",
}


def _dataset_config_path(
    path: str | Path,
    dataset_config: str | Path | None = None,
) -> tuple[Path, bool]:
    if dataset_config not in {None, ""}:
        config_path = Path(dataset_config).expanduser()
        return (config_path / "config.json" if config_path.is_dir() else config_path), True
    base = Path(path).expanduser()
    directory = base.parent if base.suffix.lower() == ".csv" else base
    return directory / "config.json", False


def _dataset_config_options(raw: Mapping[str, Any]) -> dict[str, Any]:
    options = {key: raw[key] for key in DATASET_CONFIG_KEYS if key in raw}
    scoped = raw.get("online_adaptation")
    if scoped is not None:
        if not isinstance(scoped, Mapping):
            raise ValueError("dataset config field 'online_adaptation' must be an object")
        if scoped.get("drop_users") is not None:
            options["drop_users"] = _as_list(scoped["drop_users"])
        options.update(
            {
                key: value
                for key, value in scoped.items()
                if key in DATASET_CONFIG_KEYS and key != "drop_users"
            }
        )
    return options


def load_dataset_config(
    path: str | Path,
    dataset_config: str | Path | None = None,
) -> dict[str, Any]:
    config_path, explicit = _dataset_config_path(path, dataset_config)
    if not config_path.exists():
        if explicit:
            raise FileNotFoundError(config_path)
        return {}
    if config_path.suffix.lower() != ".json":
        raise ValueError(f"dataset config must be JSON, got {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"dataset config must contain a JSON object: {config_path}")
    options = _dataset_config_options(raw)
    LOGGER.info("loaded dataset config path=%s keys=%s", config_path, sorted(options))
    return options


def _configured_value(explicit: Any, configured: Any, default: Any = None) -> Any:
    if explicit is not None:
        return explicit
    if configured is not None:
        return configured
    return default


def _drop_users(df: pd.DataFrame, drop_users: Sequence[Any] | str | None) -> pd.DataFrame:
    columns = []
    for item in _as_list(drop_users):
        item_text = str(item)
        if item in df.columns:
            columns.append(item)
        elif item_text in df.columns:
            columns.append(item_text)
        elif isinstance(item, int) or item_text.lstrip("-").isdigit():
            idx = int(item)
            if idx < 0 or idx >= len(df.columns):
                raise IndexError(f"drop user index out of range: {idx}")
            columns.append(df.columns[idx])
        else:
            columns.append(item_text)
    return df.drop(columns=columns) if columns else df


def _aggregate(df: pd.DataFrame, aggr: str | None, period: str) -> pd.DataFrame:
    if aggr is None or str(aggr).lower() in {"", "none"}:
        return df
    name = str(aggr).lower()
    if name == "sum":
        return df.resample(period).sum()
    if name == "mean":
        return df.resample(period).mean()
    if name == "last":
        return df.resample(period).last()
    if name == "first":
        return df.resample(period).first()
    if name == "asfreq":
        return df.asfreq(period)
    raise ValueError(f"unknown aggregation {aggr!r}")


def resolve_csv_path(path: str | Path, dataset_name: str | None = None) -> Path:
    base = Path(path).expanduser()
    if base.suffix.lower() == ".csv":
        if base.is_file():
            return base.resolve()
        if base.parent.is_dir():
            matches = [
                candidate
                for candidate in base.parent.iterdir()
                if candidate.is_file() and candidate.name.casefold() == base.name.casefold()
            ]
            if len(matches) == 1:
                return matches[0].resolve()
        raise FileNotFoundError(base)
    if not base.is_dir():
        raise FileNotFoundError(base)
    matches = sorted(
        (candidate for candidate in base.iterdir() if candidate.is_file() and candidate.suffix.casefold() == ".csv"),
        key=lambda candidate: candidate.name.casefold(),
    )
    if dataset_name:
        expected = f"{dataset_name}.csv".casefold()
        named = [candidate for candidate in matches if candidate.name.casefold() == expected]
        if len(named) == 1:
            return named[0].resolve()
        if len(matches) == 1:
            LOGGER.info(
                "dataset CSV name differs from dataset label label=%s path=%s",
                dataset_name,
                matches[0],
            )
            return matches[0].resolve()
        available = [candidate.name for candidate in matches]
        raise FileNotFoundError(
            f"no case-insensitive CSV match for dataset {dataset_name!r} in {base}; found {available}"
        )
    if len(matches) == 1:
        return matches[0].resolve()
    raise ValueError("pass a CSV file or a directory with dataset_name")


@dataclass
class CsvTimeSeries:
    """Date x user values for target-only forecasting experiments."""

    frame: pd.DataFrame
    missing_values_replaced: int = 0
    _values: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._values = self.frame.to_numpy(dtype=np.float32, copy=True)

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def datetimes(self) -> list[Any]:
        return list(self.frame.index)

    @property
    def user_names(self) -> list[str]:
        return [str(col) for col in self.frame.columns]

    @property
    def n_dates(self) -> int:
        return int(self.frame.shape[0])

    @property
    def n_users(self) -> int:
        return int(self.frame.shape[1])

    def validate_window(self, query_date: int, lags: int, horizon: int) -> None:
        """Validate ``X=(s-L,s]`` and ``Y=(s,s+H]`` for query date ``s``."""
        start = int(query_date) - int(lags) + 1
        stop = int(query_date) + int(horizon) + 1
        if start < 0 or stop > self.n_dates:
            raise ValueError(
                f"window X=({query_date}-{lags},{query_date}], "
                f"Y=({query_date},{query_date}+{horizon}] is outside "
                f"dataset with {self.n_dates} dates"
            )

    def window_tensor(
        self,
        query_date: int,
        lags: int,
        horizon: int,
        *,
        users: Sequence[int] | None = None,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``X=(s-L,s]`` and ``Y=(s,s+H]`` as ``(users, 1, time)``."""
        self.validate_window(query_date, lags, horizon)
        start = int(query_date) - int(lags) + 1
        stop = int(query_date) + int(horizon) + 1
        values = self.values[start:stop]
        if users is not None:
            values = values[:, list(users)]
        arr = torch.as_tensor(
            values.T.copy(),
            dtype=torch.float32,
            device=device,
        )
        x = arr[:, None, :lags]
        y = arr[:, None, lags:]
        return x, y


def load_csv_dataset(
    path: str | Path,
    *,
    dataset_name: str | None = None,
    target_cols: Sequence[Any] | str | None = None,
    date_col: str | None = None,
    drop_users: Sequence[Any] | str | None = None,
    rename_users: bool | None = None,
    aggr: str | None = None,
    aggr_period: str | None = None,
    missing_values: str | None = None,
    dataset_config: str | Path | None = None,
) -> CsvTimeSeries:
    config = load_dataset_config(path, dataset_config)
    target_cols = _configured_value(target_cols, config.get("target_cols"))
    date_col = _configured_value(date_col, config.get("date_col"))
    drop_users = (
        _as_list(config.get("drop_users"))
        if drop_users is None
        else _as_list(drop_users)
    )
    rename_users = bool(_configured_value(rename_users, config.get("rename_users"), False))
    aggr = _configured_value(aggr, config.get("aggr"))
    aggr_period = str(_configured_value(aggr_period, config.get("aggr_period"), "h"))
    missing_values = str(
        _configured_value(missing_values, config.get("missing_values"), "zero")
    ).lower()
    if missing_values not in {"zero", "error"}:
        raise ValueError("missing_values must be 'zero' or 'error'")

    csv_path = resolve_csv_path(path, dataset_name)
    if date_col:
        raw = pd.read_csv(csv_path, parse_dates=[date_col])
        raw = raw.set_index(date_col)
    else:
        raw = pd.read_csv(csv_path, index_col=0)
        try:
            raw.index = pd.to_datetime(raw.index)
        except Exception:
            pass

    raw = _aggregate(raw, aggr, aggr_period)
    missing_count = int(raw.isna().sum().sum())
    if missing_count and missing_values == "error":
        raise ValueError(f"dataset contains {missing_count} missing values")
    if missing_count:
        raw = raw.fillna(0.0)

    value_cols = list(raw.columns) if target_cols is None else _column_names(target_cols)
    missing = [col for col in value_cols if col not in raw.columns]
    if missing:
        raise KeyError(f"CSV target columns not found: {missing}")

    values = _drop_users(raw[value_cols].copy(), drop_users)
    if rename_users:
        values.columns = [f"user_{idx}" for idx in range(values.shape[1])]
    if values.empty:
        raise ValueError("dataset has no target columns after filtering")
    return CsvTimeSeries(values, missing_values_replaced=missing_count)


def load_json_kwargs(text_or_path: str | None) -> dict[str, Any]:
    if not text_or_path:
        return {}
    text = str(text_or_path)
    path = Path(text).expanduser()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(text)


def run_dir(output_dir: str | Path, save_name: str) -> Path:
    path = Path(output_dir).expanduser() / str(save_name)
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(exist_ok=True)
    return path.resolve()
