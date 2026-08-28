"""Project-specific TS-RAG evaluation on reusable causal retrieval features."""

from __future__ import annotations

from collections import OrderedDict
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from src.pipeline.contracts import (
    RESULT_FORMAT,
    adaptor_parameter_metadata,
    count_named_parameters,
    open_extraction_arrays,
    load_array_manifest,
)
from src.proposal.contracts import ExtractionConfig
from src.proposal.ridge import (
    _aggregate_user_metrics,
    _date_metrics,
    _user_date_metrics,
)
from src.results.efficiency import write_compute_timing


EXPECTED_LOOKBACK = 512
EXPECTED_HORIZON = 64
TSRAG_ADAPTOR_PREFIXES = ("encode_mlp.", "mha.", "ffn.", "gate_layer.")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted(path.rglob("best.pth")) if path.is_dir() else []
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one TS-RAG best.pth below {path}, found {len(matches)}"
        )
    return matches[0]


def _load_model(
    base_checkpoint: Path,
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    from transformers import AutoConfig

    from src.external_models.tsrag.arm import ChronosBoltModelForForecastingWithRetrieval

    config = AutoConfig.from_pretrained(str(base_checkpoint), local_files_only=True)
    model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(
        str(base_checkpoint),
        config=config,
        augment="moe",
        local_files_only=True,
    )
    state = _torch_load(checkpoint)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state dict: {checkpoint}")
    cleaned = OrderedDict(
        (str(key).removeprefix("module."), value) for key, value in state.items()
    )
    model.load_state_dict(cleaned, strict=True)
    return model.to(device).eval()


def evaluate_online_tsrag(
    *,
    extraction_dir: str | Path,
    output_dir: str | Path,
    dataset: Any,
    chronos_bolt_weights: str | Path,
    tsrag_weights: str | Path,
    device: str | torch.device = "cuda",
    batch_size: int = 256,
    eval_start_date: int | None = None,
    eval_end_date: int | None = None,
) -> dict[str, Path]:
    extraction_config = ExtractionConfig(
        **load_array_manifest(extraction_dir)["config"]
    )
    if extraction_config.lookback != EXPECTED_LOOKBACK:
        raise ValueError(f"TS-RAG requires L={EXPECTED_LOOKBACK}")
    if extraction_config.horizon != EXPECTED_HORIZON:
        raise ValueError(f"TS-RAG requires H={EXPECTED_HORIZON}")
    if not extraction_config.fixed_datastore:
        raise ValueError("native TS-RAG requires a fixed training datastore")
    if extraction_config.retrieval_scope != "same_user":
        raise ValueError("native TS-RAG requires same-user retrieval")
    if extraction_config.store_stride != 1 or extraction_config.align_period:
        raise ValueError("native TS-RAG requires unstrided training-set retrieval")
    torch_device = torch.device(device)
    checkpoint = _checkpoint_file(Path(tsrag_weights).expanduser().resolve())
    model = _load_model(
        Path(chronos_bolt_weights).expanduser().resolve(),
        checkpoint,
        torch_device,
    )
    parameter_count = count_named_parameters(
        model.named_parameters(),
        prefixes=TSRAG_ADAPTOR_PREFIXES,
    )
    if parameter_count <= 0:
        raise ValueError("TS-RAG checkpoint contains no recognized ARM parameters")
    quantiles = model.quantiles.detach().float().cpu()
    median_index = int(torch.abs(quantiles - 0.5).argmin().item())
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    evaluation_started = perf_counter()
    arrays = open_extraction_arrays(extraction_dir, dataset=dataset)
    if "x" not in arrays or "neighbor_x" not in arrays:
        raise ValueError("TS-RAG extraction must expose raw query and neighbor lookbacks")
    dates = np.asarray(arrays["retrieval_window_dates"], dtype=np.int64)
    selected = np.flatnonzero(
        np.asarray(arrays["is_evaluation_query"], dtype=bool)
        &
        (dates >= (dates[0] if eval_start_date is None else int(eval_start_date)))
        & (dates <= (dates[-1] if eval_end_date is None else int(eval_end_date)))
    )
    if not len(selected):
        raise ValueError("TS-RAG evaluation interval contains no query dates")
    rows: list[dict[str, Any]] = []
    user_rows: list[dict[str, Any]] = []
    n_users = int(arrays["y"].shape[1])
    cold_adaptation_seconds: float | None = None
    cold_setup_seconds = perf_counter() - evaluation_started

    for date_index in selected:
        batch_started = perf_counter()
        prediction_chunks: list[torch.Tensor] = []
        x = torch.as_tensor(np.asarray(arrays["x"][date_index]), dtype=torch.float32)
        neighbor_ids = np.asarray(
            arrays["neighbor_window_id"][date_index], dtype=np.int64
        )
        window_users = int(len(arrays["window_users"]))
        positions, users = np.divmod(neighbor_ids, window_users)
        retrieved = torch.as_tensor(
            np.concatenate(
                (
                    np.asarray(arrays["window_lookback"][positions, users]),
                    np.asarray(arrays["window_horizon"][positions, users]),
                ),
                axis=-1,
            ),
            dtype=torch.float32,
        )
        inference_started = perf_counter()
        with torch.inference_mode():
            for start in range(0, n_users, int(batch_size)):
                stop = min(start + int(batch_size), n_users)
                output = model(
                    context=x[start:stop].to(torch_device),
                    retrieved_seq=retrieved[start:stop].to(torch_device),
                )
                prediction_chunks.append(
                    output.quantile_preds[:, median_index, :].detach().cpu()
                )
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        model_seconds = perf_counter() - inference_started
        prediction = torch.cat(prediction_chunks).numpy().astype(np.float64)
        target = np.asarray(arrays["y"][date_index], dtype=np.float64)
        vanilla = np.asarray(arrays["vanilla"][date_index], dtype=np.float64)
        scale = np.maximum(
            np.asarray(arrays["query_std"][date_index], dtype=np.float64)[:, None],
            1e-8,
        )
        rows.append(
            _date_metrics(
                int(dates[date_index]), "tsrag", prediction, target, vanilla, scale
            )
        )
        user_rows.extend(
            _user_date_metrics(
                int(dates[date_index]), "tsrag", prediction, target, vanilla, scale
            )
        )
        if cold_adaptation_seconds is None:
            cold_adaptation_seconds = cold_setup_seconds + perf_counter() - batch_started

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / "metrics.json"
    metrics_path.write_text(
        json.dumps(_aggregate_user_metrics(user_rows), indent=2), encoding="utf-8"
    )
    per_date_path = root / "per_date_metrics.csv"
    with per_date_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_user_date_path = root / "per_user_date_metrics.csv"
    with per_user_date_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(user_rows[0]))
        writer.writeheader()
        writer.writerows(user_rows)
    adaptation_seconds = perf_counter() - evaluation_started
    assert cold_adaptation_seconds is not None
    timing_path = write_compute_timing(
        root,
        extraction_timing=Path(extraction_dir) / "extraction_timing.json",
        adaptation_seconds=adaptation_seconds,
        evaluation_samples=len(user_rows),
        cold_adaptation_seconds=cold_adaptation_seconds,
        method="tsrag",
    )
    manifest_path = root / "result_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": RESULT_FORMAT,
                "method": "tsrag",
                "parameters": adaptor_parameter_metadata(
                    parameter_count,
                    "TS-RAG ARM encode MLP, attention, FFN, and gate tensors",
                ),
                "evaluation": {
                    "first_query_date": int(rows[0]["query_date"]),
                    "last_query_date": int(rows[-1]["query_date"]),
                    "dates": len(rows),
                },
                "files": {
                    "metrics": metrics_path.name,
                    "per_date_metrics": per_date_path.name,
                    "per_user_date_metrics": per_user_date_path.name,
                    "compute_timing": timing_path.name,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "metrics": metrics_path,
        "per_date": per_date_path,
        "per_user_date": per_user_date_path,
        "compute_timing": timing_path,
        "manifest": manifest_path,
    }
