"""Readable causal datastore and fitting-date selection rules."""

from __future__ import annotations

import numpy as np

from src.proposal.contracts import ExtractionConfig
from src.proposal.date_planning import DatePlan


def candidate_dates(
    query_t: int,
    *,
    config: ExtractionConfig,
    plan: DatePlan,
) -> np.ndarray:
    """Return aligned dates from one fixed T0 or a full-capacity rolling store."""
    first = int(plan.datastore_start_date)
    last = int(query_t) - int(config.horizon)
    if config.fixed_datastore:
        last = min(last, int(plan.datastore_end_date))
    if last < first:
        return np.empty(0, dtype=np.int64)
    if config.align_period:
        last -= (last - int(query_t)) % int(config.period)
    dates = np.arange(last, first - 1, -int(config.store_stride), dtype=np.int64)[::-1]
    if config.fixed_datastore:
        dates = dates[dates <= int(plan.datastore_end_date)]
    return dates[-int(plan.n_datastore_dates) :]


def fitting_dates(
    query_t: int,
    *,
    config: ExtractionConfig,
    plan: DatePlan,
    n_fit: int | None = None,
) -> np.ndarray:
    """Return the fixed T1+T2 grid or the latest complete online fitting grid."""
    count = int(plan.n_fit if n_fit is None else n_fit)
    if config.fixed_training_set:
        return np.asarray(plan.fitting_dates[:count], dtype=np.int64)
    first = int(plan.first_retrieval_date)
    last = int(query_t) - int(config.horizon)
    if last < first:
        return np.empty(0, dtype=np.int64)
    if config.align_period and int(config.fit_stride) % int(config.period) == 0:
        last -= (last - int(query_t)) % int(config.period)
    dates = np.arange(last, first - 1, -int(config.fit_stride), dtype=np.int64)[::-1]
    return dates[-count:]
