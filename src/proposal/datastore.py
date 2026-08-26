"""Readable causal datastore-size and date-selection rules."""

from __future__ import annotations

import numpy as np

from src.proposal.contracts import ExtractionConfig


def store_date_capacity(*, n_store: int, n_users: int, retrieval_scope: str) -> int:
    """Maximum complete causal dates retained for one query's datastore.

    Same-user retrieval interprets ``n_store`` per query user. Cross-user
    retrieval interprets it as a maximum global cardinality and retains only
    complete dates, so every user contributes the same number of windows.
    """
    n_store = int(n_store)
    n_users = int(n_users)
    if n_store <= 0 or n_users <= 0:
        raise ValueError("datastore size and user count must be positive")
    if retrieval_scope == "same_user":
        return n_store
    if retrieval_scope not in {"all", "other_users"}:
        raise ValueError(f"unknown retrieval scope {retrieval_scope!r}")
    if n_store < n_users:
        raise ValueError(
            f"cross-user n_store={n_store} must be at least the {n_users} users"
        )
    return n_store // n_users


def candidate_dates(
    query_t: int,
    *,
    config: ExtractionConfig,
    n_users: int,
) -> np.ndarray:
    """Return complete eligible window dates retained for a query datastore."""
    maximum_dates = store_date_capacity(
        n_store=config.n_store,
        n_users=n_users,
        retrieval_scope=config.retrieval_scope,
    )
    first = config.lookback - 1
    last = int(query_t) - config.horizon
    if last < first:
        return np.empty(0, dtype=np.int64)
    if config.align_period:
        last -= (last - int(query_t)) % int(config.period)
    dates = np.arange(last, first - 1, -config.store_stride, dtype=np.int64)[::-1]
    return dates[:maximum_dates] if config.store_mode == "fixed" else dates[-maximum_dates:]


def fitting_dates(
    query_t: int,
    *,
    config: ExtractionConfig,
    n_fit: int | None = None,
) -> np.ndarray:
    """Return fitting dates from their independent complete-date grid.

    ``N_fit`` always counts dates per user. Cross-user fitting therefore uses
    ``N_fit * n_users`` rows but still produces one shared fit for the date.
    """
    first = config.lookback - 1
    last = int(query_t) - config.horizon
    if last < first:
        return np.empty(0, dtype=np.int64)
    if (
        config.align_period
        and config.fit_stride % config.period == 0
    ):
        last -= (last - int(query_t)) % int(config.period)
    dates = np.arange(last, first - 1, -config.fit_stride, dtype=np.int64)[::-1]
    count = config.n_fit if n_fit is None else int(n_fit)
    return dates[-count:]


def first_retrieval_window_date(
    *,
    lookback: int,
    horizon: int,
    n_store: int,
    n_users: int,
    retrieval_scope: str = "all",
    neighbors: int = 1,
    store_stride: int = 1,
    align_period: bool = False,
    period: int = 1,
    store_mode: str = "rolling",
) -> int:
    """First window whose causal datastore can supply the requested neighbors."""
    capacity = store_date_capacity(
        n_store=n_store,
        n_users=n_users,
        retrieval_scope=retrieval_scope,
    )
    if store_mode == "fixed":
        required_dates = capacity
    elif retrieval_scope == "same_user":
        required_dates = int(neighbors)
    elif retrieval_scope == "all":
        required_dates = (int(neighbors) + int(n_users) - 1) // int(n_users)
    else:
        if int(n_users) < 2:
            raise ValueError("other-user retrieval requires at least two users")
        required_dates = (
            int(neighbors) + int(n_users) - 2
        ) // (int(n_users) - 1)
    causal_gap = int(horizon)
    if align_period:
        causal_gap = ((causal_gap + int(period) - 1) // int(period)) * int(period)
    return int(lookback) - 1 + causal_gap + (required_dates - 1) * int(store_stride)


def first_evaluation_query_date(
    *,
    first_retrieval_date: int,
    horizon: int,
    n_fit: int,
    fit_stride: int,
    align_period: bool,
    period: int,
) -> int:
    """First query whose complete independent fitting grid is extractable."""
    if int(n_fit) <= 0 or int(fit_stride) <= 0:
        raise ValueError("n_fit and fit_stride must be positive")
    causal_gap = int(horizon)
    if bool(align_period) and int(fit_stride) % int(period) == 0:
        causal_gap = (
            (causal_gap + int(period) - 1) // int(period)
        ) * int(period)
    return (
        int(first_retrieval_date)
        + causal_gap
        + (int(n_fit) - 1) * int(fit_stride)
    )
