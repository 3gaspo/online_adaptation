"""Precomputed causal datastore, fitting, and evaluation date plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from src.proposal.contracts import ExtractionConfig


@dataclass(frozen=True)
class DatePlan:
    """Resolved T0, T1+T2, and T3 boundaries for one dataset setting."""

    n_datastore_dates: int
    n_fit: int
    datastore_start_date: int
    datastore_end_date: int
    first_retrieval_date: int
    fitting_dates: tuple[int, ...]
    evaluation_start_date: int
    evaluation_end_date: int
    evaluation_query_dates: tuple[int, ...]
    split_boundaries: tuple[int, int, int] | None = None

    def scientific_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evaluation_query_dates"] = list(self.evaluation_query_dates)
        payload["fitting_dates"] = list(self.fitting_dates)
        if self.split_boundaries is not None:
            payload["split_boundaries"] = list(self.split_boundaries)
        return payload


def resolve_relative_date(value: int | float | None, *, n_dates: int) -> int | None:
    """Resolve an absolute index or a fraction of the complete source timeline."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("date boundaries must be integer indices or ratios")
    if isinstance(value, int):
        return int(value)
    ratio = float(value)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("fractional date boundaries must lie in [0, 1]")
    return min(int(n_dates) - 1, int(math.floor(ratio * int(n_dates))))


def resolve_datastore_dates(
    value: int | float,
    *,
    available_datastore_dates: int,
) -> int:
    """Resolve a retained-origin count from an integer or eligible-grid ratio."""
    if isinstance(value, bool):
        raise ValueError("n_datastore_dates must be an integer count or a ratio")
    if isinstance(value, int):
        count = int(value)
    else:
        ratio = float(value)
        if not 0.0 < ratio <= 1.0:
            raise ValueError("ratio n_datastore_dates must lie in (0, 1]")
        count = max(1, int(math.floor(ratio * int(available_datastore_dates))))
    if count <= 0:
        raise ValueError("n_datastore_dates resolves to no dates")
    return count


def _causal_gap(horizon: int, *, align_period: bool, period: int) -> int:
    if not align_period:
        return int(horizon)
    return ((int(horizon) + int(period) - 1) // int(period)) * int(period)


def _store_date_limit(
    config: ExtractionConfig,
    *,
    n_users: int | None,
    available_dates: int,
) -> int:
    requested = resolve_datastore_dates(
        config.n_datastore_dates,
        available_datastore_dates=int(available_dates),
    )
    if config.n_store_windows is not None:
        if n_users is None or int(n_users) <= 0:
            raise ValueError("n_users is required when n_store_windows is specified")
        return min(requested, int(config.n_store_windows) // int(n_users))
    return requested


def _split_date_plan(
    *, n_dates: int, n_users: int | None, config: ExtractionConfig
) -> DatePlan:
    ratios = tuple(float(value) for value in config.split_ratios or ())
    t0_end = int(round(ratios[0] * int(n_dates)))
    t12_end = int(round((ratios[0] + ratios[1]) * int(n_dates)))
    boundaries = (t0_end, t12_end, int(n_dates))
    first_window = int(config.lookback) - 1
    last_window = int(n_dates) - int(config.horizon) - 1

    requested_evaluation_start = resolve_relative_date(
        config.eval_start_date,
        n_dates=n_dates,
    )
    evaluation_start = max(
        first_window,
        t12_end - 1
        if requested_evaluation_start is None
        else int(requested_evaluation_start),
    )
    evaluation_end = last_window
    evaluation = tuple(
        map(
            int,
            np.arange(
                evaluation_start,
                evaluation_end + 1,
                int(config.query_stride),
                dtype=np.int64,
            ),
        )
    )
    if not evaluation:
        raise ValueError("split protocol contains no T3 evaluation query dates")

    datastore_start = first_window
    datastore_end = t0_end - int(config.horizon) - 1
    if config.align_period:
        datastore_start += (
            int(evaluation_start) - int(datastore_start)
        ) % int(config.period)
        datastore_end -= (
            int(datastore_end) - int(datastore_start)
        ) % int(config.period)
    if datastore_end < datastore_start:
        raise ValueError("T0 contains no complete datastore windows")
    available_datastore_dates = 1 + (
        datastore_end - datastore_start
    ) // int(config.store_stride)
    datastore_dates = _store_date_limit(
        config,
        n_users=n_users,
        available_dates=available_datastore_dates,
    )
    if datastore_dates <= 0:
        raise ValueError("N_store cannot retain one complete date across all users")

    first_retrieval = max(first_window, t0_end - 1)
    fitting = ()
    if config.include_fitting_windows:
        available_fitting = tuple(
            map(
                int,
                np.arange(
                    first_retrieval,
                    t12_end - int(config.horizon),
                    int(config.fit_stride),
                    dtype=np.int64,
                ),
            )
        )
        if len(available_fitting) < int(config.n_fit):
            raise ValueError(
                f"T1+T2 contains {len(available_fitting)} complete fitting dates, "
                f"fewer than n_fit={config.n_fit}"
            )
        fitting = available_fitting[-int(config.n_fit) :]
    return DatePlan(
        n_datastore_dates=int(datastore_dates),
        n_fit=int(config.n_fit),
        datastore_start_date=int(datastore_start),
        datastore_end_date=int(datastore_end),
        first_retrieval_date=int(first_retrieval),
        fitting_dates=fitting,
        evaluation_start_date=int(evaluation_start),
        evaluation_end_date=int(evaluation_end),
        evaluation_query_dates=evaluation,
        split_boundaries=boundaries,
    )


def build_date_plan(
    *,
    n_dates: int,
    n_users: int | None = None,
    config: ExtractionConfig,
) -> DatePlan:
    """Resolve all date indexes before extraction or model evaluation begins."""
    config.validate()
    if config.split_ratios is not None:
        return _split_date_plan(n_dates=n_dates, n_users=n_users, config=config)
    first_window = int(config.lookback) - 1
    last_window = int(n_dates) - int(config.horizon) - 1
    available = last_window - first_window + 1
    if available <= 0:
        raise ValueError("dataset has no complete lookback-horizon windows")
    requested_start = resolve_relative_date(config.eval_start_date, n_dates=n_dates)
    store_gap = _causal_gap(
        config.horizon,
        align_period=config.align_period,
        period=config.period,
    )

    fitting: tuple[int, ...] = ()
    if requested_start is None:
        available_datastore_dates = available // int(config.store_stride)
        datastore_dates = _store_date_limit(
            config,
            n_users=n_users,
            available_dates=available_datastore_dates,
        )
        if datastore_dates <= 0:
            raise ValueError("N_store cannot retain one complete date across all users")
        datastore_start = first_window
        datastore_end = first_window + datastore_dates * int(config.store_stride) - 1
        if datastore_end > last_window:
            raise ValueError(
                f"n_datastore_dates={datastore_dates} with "
                f"store_stride={config.store_stride} does not fit in the dataset"
            )
        first_retrieval = datastore_end + store_gap
        if first_retrieval > last_window:
            raise ValueError(
                "dataset ends before the complete datastore becomes observable"
            )
        natural_evaluation_start = first_retrieval
        if config.include_fitting_windows:
            fitting = tuple(
                int(value)
                for value in first_retrieval
                + np.arange(int(config.n_fit), dtype=np.int64)
                * int(config.fit_stride)
            )
            fit_gap = _causal_gap(
                config.horizon,
                align_period=(
                    config.align_period
                    and int(config.fit_stride) % int(config.period) == 0
                ),
                period=config.period,
            )
            natural_evaluation_start = int(fitting[-1]) + fit_gap
        start = natural_evaluation_start
    else:
        start = int(requested_start)
        if start > last_window:
            raise ValueError("evaluation interval contains no complete query window")
        first_retrieval = start
        if config.include_fitting_windows:
            fit_gap = _causal_gap(
                config.horizon,
                align_period=(
                    config.align_period
                    and int(config.fit_stride) % int(config.period) == 0
                ),
                period=config.period,
            )
            last_fitting = start - fit_gap
            first_retrieval = (
                last_fitting - (int(config.n_fit) - 1) * int(config.fit_stride)
            )
            fitting = tuple(
                int(value)
                for value in first_retrieval
                + np.arange(int(config.n_fit), dtype=np.int64)
                * int(config.fit_stride)
            )

        datastore_start = first_window
        if config.align_period:
            datastore_start += (
                int(first_retrieval) - int(datastore_start)
            ) % int(config.period)
        datastore_end = int(first_retrieval) - store_gap
        available_datastore_dates = 1 + (
            datastore_end - datastore_start
        ) // int(config.store_stride)
        if available_datastore_dates <= 0:
            raise ValueError(
                "evaluation start leaves no room for a causal datastore and fitting grid"
            )
        datastore_dates = _store_date_limit(
            config,
            n_users=n_users,
            available_dates=available_datastore_dates,
        )
        if datastore_dates <= 0:
            raise ValueError("N_store cannot retain one complete date across all users")
    requested_end = resolve_relative_date(config.eval_end_date, n_dates=n_dates)
    end = last_window if requested_end is None else min(requested_end, last_window)
    if end < start:
        raise ValueError("evaluation interval contains no complete query window")
    evaluation = tuple(
        map(
            int,
            np.arange(start, end + 1, int(config.query_stride), dtype=np.int64),
        )
    )
    if not evaluation:
        raise ValueError("configuration contains no evaluation query dates")
    return DatePlan(
        n_datastore_dates=datastore_dates,
        n_fit=int(config.n_fit),
        datastore_start_date=first_window,
        datastore_end_date=datastore_end,
        first_retrieval_date=first_retrieval,
        fitting_dates=fitting,
        evaluation_start_date=start,
        evaluation_end_date=end,
        evaluation_query_dates=evaluation,
        split_boundaries=None,
    )
