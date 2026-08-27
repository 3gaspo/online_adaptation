"""Frozen third-party forecasting models kept separate from the proposal."""

from importlib import import_module

__all__ = [
    "Chronos2",
    "ChronosBolt",
    "ChronosT5",
    "TSICLForecaster",
]


_OWNERS = {
    "Chronos2": "chronos2",
    "ChronosBolt": "chronos_bolt",
    "ChronosT5": "chronos_t5",
    "TSICLForecaster": "ts_icl",
}


def __getattr__(name: str):
    try:
        owner = _OWNERS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    return getattr(import_module(f"{__name__}.{owner}"), name)
