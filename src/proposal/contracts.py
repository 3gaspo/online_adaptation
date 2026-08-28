"""Scientific configuration contracts for causal online adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

@dataclass(frozen=True)
class ExtractionConfig:
    dataset: str
    lookback: int
    horizon: int
    backbone: str = "chronos2"
    n_datastore_dates: int | float = 100
    n_store_windows: int | None = None
    n_fit: int | float = 100
    max_k: int = 20
    distance_space: str = "raw"
    distance_metric: str = "euclidean"
    retrieval_scope: str = "all"
    fixed_datastore: bool = False
    fixed_training_set: bool = False
    include_fitting_windows: bool = True
    store_stride: int = 24
    fit_stride: int = 0
    align_period: bool = True
    period: int = 24
    query_stride: int = 1
    eval_start_date: int | float | None = None
    eval_end_date: int | float | None = None
    split_ratios: tuple[float, float, float] | None = None
    normalization: str = "none"
    retrieval_covariate_mode: str = "past_and_future"
    homogeneous_only: bool = False
    seed: int = 1

    def __post_init__(self) -> None:
        if int(self.fit_stride) == 0:
            object.__setattr__(self, "fit_stride", int(self.period))

    def validate(self) -> None:
        for name in (
            "lookback",
            "horizon",
            "query_stride",
            "store_stride",
            "fit_stride",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.n_fit, bool):
            raise ValueError("n_fit must be an integer count or split ratio")
        if isinstance(self.n_fit, int):
            if self.n_fit <= 0:
                raise ValueError("integer n_fit must be positive")
        elif self.split_ratios is None or not 0.0 < float(self.n_fit) <= 1.0:
            raise ValueError(
                "fractional n_fit requires split_ratios and must lie in (0, 1]"
            )
        if isinstance(self.n_datastore_dates, bool):
            raise ValueError("n_datastore_dates must be an integer count or a ratio")
        if isinstance(self.n_datastore_dates, int):
            if self.n_datastore_dates <= 0:
                raise ValueError("integer n_datastore_dates must be positive")
        elif not 0.0 < float(self.n_datastore_dates) <= 1.0:
            raise ValueError("ratio n_datastore_dates must lie in (0, 1]")
        if self.n_store_windows is not None and int(self.n_store_windows) <= 0:
            raise ValueError("n_store_windows must be positive when specified")
        if self.split_ratios is not None:
            ratios = tuple(float(value) for value in self.split_ratios)
            if len(ratios) != 3 or any(value <= 0.0 for value in ratios):
                raise ValueError("split_ratios must contain three positive values")
            if not abs(sum(ratios) - 1.0) < 1e-9:
                raise ValueError("split_ratios must sum to one")
        if int(self.max_k) <= 0:
            raise ValueError("max_k must be positive")
        if self.retrieval_scope not in {"all", "same_user", "other_users"}:
            raise ValueError("retrieval_scope must be all, same_user, or other_users")
        if self.distance_space not in {"raw", "instance", "minmax", "fourier", "encoder", "tsrag"}:
            raise ValueError(f"unsupported distance space {self.distance_space!r}")
        if self.distance_metric not in {"euclidean", "cosine", "pearson"}:
            raise ValueError(f"unsupported distance metric {self.distance_metric!r}")
        if self.retrieval_covariate_mode not in {"none", "past", "past_and_future"}:
            raise ValueError(
                "retrieval_covariate_mode must be none, past, or past_and_future"
            )
        if self.align_period and self.period <= 0:
            raise ValueError("period must be positive when alignment is enabled")
        if self.align_period and self.store_stride % self.period:
            raise ValueError("store_stride must be a multiple of period when alignment is enabled")

    def scientific_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class AdapterConfig:
    method: str = "full_ridge_shared"
    n_fit: int = 100
    fitting_scope: str = "same_user"
    alpha: float = 1e-2
    tune_alpha: bool = True
    validation_ratio: float = 0.2
    alpha_grid: tuple[float, ...] = (1e-1, 1e-2, 1e-3)
    candidate_k_grid: tuple[int, ...] = (1, 5, 10, 15)
    used_k: int | None = None
    fixed_training_set: bool = False
    fit_loss: str = "mse"
    candidate: str = "cov"
    catboost_iterations: int = 300
    catboost_depth: int = 4
    catboost_learning_rate: float = 3e-2
    catboost_refit_stride: int = 1
    seed: int = 1

    def validate(self) -> None:
        if self.n_fit <= 0:
            raise ValueError("n_fit must be positive")
        if self.fitting_scope not in {"all", "same_user"}:
            raise ValueError("fitting_scope must be all or same_user")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError("validation_ratio must lie strictly between zero and one")
        if not self.alpha_grid or any(float(value) < 0 for value in self.alpha_grid):
            raise ValueError("alpha_grid must contain non-negative values")
        if not self.candidate_k_grid or any(
            int(value) <= 0 for value in self.candidate_k_grid
        ):
            raise ValueError("candidate_k_grid must contain positive values")
        if self.used_k is not None and int(self.used_k) <= 0:
            raise ValueError("used_k must be positive when specified")
        if len(set(self.alpha_grid)) != len(self.alpha_grid):
            raise ValueError("alpha_grid values must be unique")
        if len(set(self.candidate_k_grid)) != len(self.candidate_k_grid):
            raise ValueError("candidate_k_grid values must be unique")
        if (self.tune_alpha or self.used_k is None) and self.n_fit < 2:
            raise ValueError("hyperparameter selection requires at least two fitting rows")
        if self.fit_loss not in {"mse", "nmse"}:
            raise ValueError("fit_loss must be mse or nmse")
        if self.candidate not in {"cov", "avgy"}:
            raise ValueError("candidate must be cov or avgy")
        if self.catboost_refit_stride <= 0:
            raise ValueError("catboost_refit_stride must be positive")

    def scientific_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
