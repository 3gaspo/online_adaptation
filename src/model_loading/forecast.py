"""Minimal forecasting model wrapper for extraction experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange, repeat


FOUNDATION_MODEL_ALIASES = (
    "chronos2",
    "chronos_bolt",
    "chronos_t5",
    "ts_icl",
)


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameters, including wrapped pipelines.

    ForecastModel registers ordinary PyTorch backbones as ``base_model``. The
    Chronos pipeline, however, owns its PyTorch module under ``pipeline.model``
    and is not itself an ``nn.Module``, so it needs to be unwrapped explicitly.
    """
    target = getattr(model, "base_model", model)
    pipeline_model = getattr(getattr(target, "pipeline", None), "model", None)
    if isinstance(pipeline_model, nn.Module):
        target = pipeline_model
    parameters = list(target.parameters())
    return (
        sum(parameter.numel() for parameter in parameters),
        sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
    )


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    name = "auto" if device is None else str(device).lower()
    if name in {"auto", "gpu", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if name in {"gpu", "cuda"}:
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cpu")
    return torch.device(name)


class NoNormalization(nn.Module):
    name = "none"

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[()]]:
        return x, ()

    def inverse(self, y: torch.Tensor, state: tuple[()]) -> torch.Tensor:
        del state
        return y


class InstanceNormalization(nn.Module):
    name = "instance"

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mean = x.mean(dim=-1, keepdim=True).detach()
        std = x.std(dim=-1, keepdim=True, unbiased=False).detach()
        return (x - mean) / (std + self.eps), (mean, std)

    def inverse(
        self,
        y: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        mean, std = state
        return y * (std + self.eps) + mean


def build_normalization(name: str | None, **kwargs: Any) -> nn.Module:
    if name is None or str(name).lower() in {"", "none", "identity"}:
        return NoNormalization()
    if str(name).lower() == "instance":
        return InstanceNormalization(**kwargs)
    raise ValueError("lightweight extraction only supports normalization='none' or 'instance'")


class ForecastModel(nn.Module):
    """Compose one optional normalization with a base forecaster."""

    def __init__(self, base_model: nn.Module, normalization: nn.Module | None = None):
        super().__init__()
        self.base_model = base_model
        self.normalization = normalization or NoNormalization()
        self.lags = int(getattr(base_model, "lags"))
        self.dim = int(getattr(base_model, "dim", 1))
        self.horizon = int(getattr(base_model, "horizon"))
        self.supports_context = bool(getattr(base_model, "supports_context", True))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        past_covariates: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        x_norm, state = self.normalization.normalize(x)
        pred = self.base_model(
            x_norm,
            context=context,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            **kwargs,
        )
        return self.normalization.inverse(pred, state)

    @torch.no_grad()
    def representation(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        past_covariates: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not hasattr(self.base_model, "representation"):
            raise AttributeError(f"{self.base_model.__class__.__name__} has no representation()")
        x_norm, _ = self.normalization.normalize(x)
        return self.base_model.representation(
            x_norm,
            context=context,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            **kwargs,
        )


class RetrievalCovariateAdapter(nn.Module):
    """Translate retrieved windows into covariates without changing model identity."""

    MODES = {"none", "past", "past_and_future"}

    def __init__(self, base_model: nn.Module, *, mode: str) -> None:
        super().__init__()
        self.base_model = base_model
        self.mode = str(mode)
        if self.mode not in self.MODES:
            raise ValueError(
                "retrieval_covariate_mode must be none, past, or past_and_future"
            )
        self.lags = int(getattr(base_model, "lags"))
        self.dim = int(getattr(base_model, "dim", 1))
        self.horizon = int(getattr(base_model, "horizon"))
        self.supports_covariates = bool(
            getattr(base_model, "supports_covariates", False)
        )
        self.supports_context = self.mode != "none" and self.supports_covariates
        self.pipeline = getattr(base_model, "pipeline", None)

    @staticmethod
    def _join(
        explicit: torch.Tensor | None,
        retrieved: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if explicit is None:
            return retrieved
        if retrieved is None:
            return explicit
        if explicit.shape[0] != retrieved.shape[0] or explicit.shape[-1] != retrieved.shape[-1]:
            raise ValueError("explicit and retrieved covariates must share batch and time axes")
        return torch.cat([explicit, retrieved], dim=1)

    def _covariates(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None,
        covariates: dict[str, torch.Tensor | None] | None,
        past: torch.Tensor | None,
        future: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | None] | None:
        if covariates is not None:
            if past is not None or future is not None:
                raise ValueError("provide structured or named covariates, not both")
            past = covariates.get("past")
            future = covariates.get("future")
        supplied = context is not None or past is not None or future is not None
        if supplied and self.mode == "none":
            raise ValueError("covariates are disabled for this model configuration")
        if supplied and not self.supports_covariates:
            raise ValueError(
                f"{self.base_model.__class__.__name__} does not consume covariates"
            )
        if self.mode == "past" and future is not None:
            raise ValueError("future covariates require retrieval_covariate_mode=past_and_future")
        if context is not None:
            if context.ndim != 3 or context.shape[0] != x.shape[0]:
                raise ValueError(
                    "retrieval context must have shape (batch, covariates, time)"
                )
            if context.shape[-1] < self.lags:
                raise ValueError(f"retrieval context length must be at least {self.lags}")
            retrieved_past = context[..., : self.lags]
            retrieved_future = None
            if self.mode == "past_and_future":
                if context.shape[-1] < self.lags + self.horizon:
                    raise ValueError(
                        "past_and_future retrieval covariates require lags + horizon values"
                    )
                retrieved_future = context[
                    ..., self.lags : self.lags + self.horizon
                ]
            past = self._join(past, retrieved_past)
            future = self._join(future, retrieved_future)
        if past is None and future is None:
            return None
        return {"past": past, "future": future}

    def forward(
        self,
        x: torch.Tensor,
        covariates: dict[str, torch.Tensor | None] | None = None,
        context: torch.Tensor | None = None,
        *,
        past_covariates: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        structured = self._covariates(
            x,
            context,
            covariates,
            past_covariates,
            future_covariates,
        )
        return self.base_model(x, covariates=structured, **kwargs)

    @torch.no_grad()
    def representation(
        self,
        x: torch.Tensor,
        covariates: dict[str, torch.Tensor | None] | None = None,
        context: torch.Tensor | None = None,
        *,
        past_covariates: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not hasattr(self.base_model, "representation"):
            raise AttributeError(
                f"{self.base_model.__class__.__name__} has no representation()"
            )
        structured = self._covariates(
            x,
            context,
            covariates,
            past_covariates,
            future_covariates,
        )
        return self.base_model.representation(x, covariates=structured, **kwargs)


class Persistence(nn.Module):
    def __init__(self, lags: int, dim: int = 1, horizon: int | None = None, **kwargs: Any):
        super().__init__()
        del kwargs
        if horizon is None:
            raise ValueError("horizon is required")
        self.lags = int(lags)
        self.dim = int(dim)
        self.horizon = int(horizon)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        return repeat(x[..., -1], "batch dim -> batch dim horizon", horizon=self.horizon)


class Linear(nn.Module):
    def __init__(self, lags: int, dim: int = 1, horizon: int | None = None, **kwargs: Any):
        super().__init__()
        del kwargs
        if horizon is None:
            raise ValueError("horizon is required")
        self.lags = int(lags)
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.linear = nn.Linear(self.lags * self.dim, self.horizon * self.dim)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        y = self.linear(rearrange(x, "batch dim lags -> batch (dim lags)"))
        return rearrange(y, "batch (dim horizon) -> batch dim horizon", dim=self.dim)


def _state_dict_from_file(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path).expanduser(), map_location="cpu")
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model_state"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        return payload
    raise TypeError(f"expected a state-dict-like payload in {path}")


def _load_pretrained(model: ForecastModel, path: str | Path) -> None:
    state = _state_dict_from_file(path)
    try:
        model.load_state_dict(state)
        return
    except RuntimeError:
        pass
    model.base_model.load_state_dict(state)


def load_pretrained_model(
    name: str,
    *,
    lags: int,
    horizon: int,
    dim: int = 1,
    normalization: str | None = "none",
    pretrained_path: str | Path | None = None,
    device: str | torch.device | None = "auto",
    model_kwargs: dict[str, Any] | None = None,
) -> torch.nn.Module:
    model = load_model(
        name,
        lags=lags,
        dim=dim,
        horizon=horizon,
        normalization=normalization,
        pretrained_path=pretrained_path,
        **(model_kwargs or {}),
    )
    return model.to(resolve_device(device)).eval()


def load_model(
    name: str,
    *,
    lags: int,
    dim: int = 1,
    horizon: int,
    normalization: str | None = "none",
    pretrained_path: str | Path | None = None,
    normalization_kwargs: dict[str, Any] | None = None,
    retrieval_covariate_mode: str = "none",
    **kwargs: Any,
) -> ForecastModel:
    """Load a minimal extraction forecaster.

    Built-ins include ``persistence``, ``chronos2``, ``chronos_bolt``,
    ``chronos_t5``, and ``ts_icl``.
    """
    raw_name = str(name)
    key = raw_name.lower()
    if key in FOUNDATION_MODEL_ALIASES and raw_name != key:
        raise ValueError(f"foundation model aliases are case-sensitive: {raw_name!r}")
    registry = {
        "persistence": Persistence,
        "linear": Linear,
    }
    if key == "chronos2":
        from src.external_models.chronos2 import Chronos2

        registry[key] = Chronos2
    elif key == "chronos_bolt":
        from src.external_models.chronos_bolt import ChronosBolt

        registry[key] = ChronosBolt
    elif key == "chronos_t5":
        from src.external_models.chronos_t5 import ChronosT5

        registry[key] = ChronosT5
    elif key == "ts_icl":
        from src.external_models.ts_icl import TSICLForecaster

        registry[key] = TSICLForecaster
    if key not in registry:
        raise ValueError(f"unknown extraction model {name!r}")
    constructor_kwargs = dict(kwargs)
    if key in FOUNDATION_MODEL_ALIASES:
        constructor_kwargs["pretrained_path"] = pretrained_path
    base = registry[key](lags=lags, dim=dim, horizon=horizon, **constructor_kwargs)
    if key in FOUNDATION_MODEL_ALIASES:
        base = RetrievalCovariateAdapter(base, mode=retrieval_covariate_mode)
    model = ForecastModel(
        base,
        normalization=build_normalization(normalization, **(normalization_kwargs or {})),
    )
    if pretrained_path is not None and key not in FOUNDATION_MODEL_ALIASES:
        _load_pretrained(model, pretrained_path)
    return model
