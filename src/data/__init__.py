"""Dataset loading and neighbor retrieval utilities.

The package facade is lazy so the standalone TIME-preparation module does not
load the experiment-only tensor stack.
"""

from importlib import import_module

__all__ = [
    "CsvTimeSeries",
    "aligned_store_dates",
    "load_csv_dataset",
    "neighbor_to_query_scale",
    "period_eval_dates",
    "search_neighbors",
    "search_neighbors_other_users",
    "search_neighbors_same_user",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    owner = "load_dataset" if name in {"CsvTimeSeries", "load_csv_dataset"} else "neighbors"
    return getattr(import_module(f"{__name__}.{owner}"), name)
