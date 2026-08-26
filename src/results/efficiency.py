"""Shared experiment and cold-batch compute-time reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_compute_timing(
    output_dir: str | Path,
    *,
    extraction_timing: str | Path | Mapping[str, Any],
    adaptation_seconds: float,
    evaluation_samples: int,
    cold_adaptation_seconds: float,
    method: str,
) -> Path:
    """Report complete experiment cost, its amortized cost, and one cold batch."""
    if isinstance(extraction_timing, Mapping):
        extraction = dict(extraction_timing)
    else:
        extraction = json.loads(
            Path(extraction_timing).read_text(encoding="utf-8")
        )
    samples = int(evaluation_samples)
    if samples <= 0:
        raise ValueError("compute timing requires at least one evaluation sample")
    extraction_seconds = float(extraction["total_extraction_seconds"])
    adaptation_seconds = float(adaptation_seconds)
    total_seconds = extraction_seconds + adaptation_seconds
    cold = dict(extraction["cold_batch"])
    cold_extraction_seconds = float(cold["extraction_seconds"])
    cold_adaptation_seconds = float(cold_adaptation_seconds)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "compute_timing.json"
    path.write_text(
        json.dumps(
            {
                "method": str(method),
                "definition": (
                    "complete extraction plus adaptation/evaluation compute; cold batch "
                    "includes source-view reconstruction and starts with no pre-existing "
                    "computation or context cache"
                ),
                "evaluation_samples": samples,
                "extraction_seconds": extraction_seconds,
                "adaptation_seconds": adaptation_seconds,
                "total_experiment_seconds": total_seconds,
                "average_seconds_per_sample": total_seconds / samples,
                "cold_batch": {
                    "evaluation_query_date": int(cold["evaluation_query_date"]),
                    "samples": int(extraction["users"]),
                    "extraction_seconds": cold_extraction_seconds,
                    "adaptation_seconds": cold_adaptation_seconds,
                    "total_seconds": (
                        cold_extraction_seconds + cold_adaptation_seconds
                    ),
                    "seconds_per_sample": (
                        cold_extraction_seconds + cold_adaptation_seconds
                    )
                    / int(extraction["users"]),
                    "components": {
                        "metadata_and_statistics_seconds": float(
                            cold["metadata_and_statistics_seconds"]
                        ),
                        "retrieval_and_representation_seconds": float(
                            cold["retrieval_and_representation_seconds"]
                        ),
                        "forecast_seconds": float(cold["forecast_seconds"]),
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
