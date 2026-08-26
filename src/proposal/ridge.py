"""Exact ridge, delta-ridge, and convex online adaptor computations."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from src.proposal.extraction import ContextForecastCache


BASELINE_VARIABLES: dict[str, tuple[str, ...]] = {
    "cov": ("V", "C"),
    "avgy": ("V", "avgy"),
    "y": ("V", "Y"),
    "cov_y": ("V", "C", "Y"),
    "cov_avgy": ("V", "C", "avgy"),
    "residual": ("V", "Y", "N"),
    "full": ("V", "C", "Y", "N"),
}


def parse_method(method: str) -> tuple[str, str, str]:
    for formulation in ("delta_ridge", "ridge", "convex"):
        suffix_shared = f"_{formulation}_shared"
        suffix_horizon = f"_{formulation}_horizon"
        if method.endswith(suffix_shared):
            design = method[: -len(suffix_shared)]
            mode = "shared"
            break
        if method.endswith(suffix_horizon):
            design = method[: -len(suffix_horizon)]
            mode = "horizon"
            break
    else:
        raise ValueError(f"unknown online linear adaptor {method!r}")
    if design not in BASELINE_VARIABLES:
        raise ValueError(f"unknown feature design {design!r}")
    return design, formulation, mode


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)


def _avgy(
    arrays: dict[str, np.ndarray], date_index: int, neighbors: int | None = None
) -> np.ndarray:
    distance = np.asarray(arrays["distance"][date_index], dtype=np.float64)
    if neighbors is not None:
        distance = distance[:, :neighbors]
    normalized = (distance - distance.min(axis=-1, keepdims=True)) / np.maximum(
        distance.std(axis=-1, keepdims=True),
        1e-8,
    )
    weights = _softmax(-normalized, axis=-1)
    return np.einsum(
        "uk,ukh->uh",
        weights,
        np.asarray(arrays["neighbor_y"][date_index], dtype=np.float64)[:, :neighbors],
    )


def design_feature_names(
    design: str,
    formulation: str,
    neighbors: int,
) -> list[str]:
    """Describe a design without materializing a context forecast."""
    names: list[str] = []
    for signal in BASELINE_VARIABLES[design]:
        signal_names = {
            "V": ["V"],
            "C": ["C"],
            "avgy": ["avgy"],
            "Y": [f"Y_{index + 1}" for index in range(int(neighbors))],
            "N": [f"N_{index + 1}" for index in range(int(neighbors))],
        }[signal]
        if formulation == "delta_ridge":
            if signal == "V":
                continue
            signal_names = [f"{name}-V" for name in signal_names]
        names.extend(signal_names)
    return names


def design_tensor(
    arrays: dict[str, np.ndarray],
    date_index: int,
    *,
    design: str,
    formulation: str,
    neighbors: int | None = None,
    context_cache: ContextForecastCache | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vanilla = np.asarray(arrays["vanilla"][date_index], dtype=np.float64)
    parts: list[np.ndarray] = []
    names: list[str] = []
    signals = BASELINE_VARIABLES[design]
    for signal in signals:
        if signal == "V":
            value = vanilla[:, :, None]
            signal_names = ["V"]
        elif signal == "C":
            if context_cache is None:
                value = vanilla[:, :, None]
            else:
                if neighbors is None:
                    raise ValueError("K must be explicit for a context forecast")
                value = np.asarray(
                    context_cache.get(date_index, neighbors), dtype=np.float64
                )[:, :, None]
            signal_names = ["C"]
        elif signal == "avgy":
            value = _avgy(arrays, date_index, neighbors)[:, :, None]
            signal_names = ["avgy"]
        elif signal == "Y":
            value = np.moveaxis(
                np.asarray(arrays["neighbor_y"][date_index], dtype=np.float64)[:, :neighbors],
                1,
                2,
            )
            signal_names = [f"Y_{index + 1}" for index in range(value.shape[-1])]
        elif signal == "N":
            value = np.moveaxis(
                np.asarray(arrays["neighbor_pred"][date_index], dtype=np.float64)[:, :neighbors],
                1,
                2,
            )
            signal_names = [f"N_{index + 1}" for index in range(value.shape[-1])]
        else:  # pragma: no cover
            raise AssertionError(signal)
        if formulation == "delta_ridge":
            if signal == "V":
                continue
            value = value - vanilla[:, :, None]
            signal_names = [f"{name}-V" for name in signal_names]
        parts.append(value)
        names.extend(signal_names)
    target = np.asarray(arrays["y"][date_index], dtype=np.float64)
    if formulation != "convex":
        target = target - vanilla
    return np.concatenate(parts, axis=-1), target, names


class LinearStatistics:
    def __init__(self, *, horizon: int, features: int, mode: str, convex: bool) -> None:
        self.horizon = int(horizon)
        self.features = int(features)
        self.mode = mode
        self.convex = bool(convex)
        self.windows = 0
        self.y_sum_squares = (
            0.0 if mode == "shared" else np.zeros(horizon, dtype=np.float64)
        )
        if mode == "shared":
            self.sum_squares = np.zeros(features, dtype=np.float64)
            self.xtx = np.zeros((features, features), dtype=np.float64)
            self.xty = np.zeros(features, dtype=np.float64)
        else:
            self.sum_squares = np.zeros((horizon, features), dtype=np.float64)
            self.xtx = np.zeros((horizon, features, features), dtype=np.float64)
            self.xty = np.zeros((horizon, features), dtype=np.float64)

    def update(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        scale: float,
        sign: int,
    ) -> None:
        x = np.asarray(x, dtype=np.float64) / float(scale)
        y = np.asarray(y, dtype=np.float64) / float(scale)
        if self.mode == "shared":
            self.y_sum_squares += sign * float(np.einsum("h,h->", y, y))
            self.sum_squares += sign * np.einsum("hf,hf->f", x, x)
            self.xtx += sign * (x.T @ x)
            self.xty += sign * (x.T @ y)
        else:
            self.y_sum_squares += sign * np.square(y)
            self.sum_squares += sign * np.square(x)
            self.xtx += sign * np.einsum("hf,hg->hfg", x, x)
            self.xty += sign * x * y[:, None]
        self.windows += int(sign)
        if self.windows < 0:
            raise RuntimeError("rolling statistics removed more windows than they contain")

    def solve(self, alpha: float) -> np.ndarray:
        if self.windows <= 0:
            raise ValueError("cannot solve empty rolling statistics")
        observations = self.windows * self.horizon if self.mode == "shared" else self.windows
        if self.convex:
            if self.mode == "shared":
                return _solve_simplex(self.xtx / observations, self.xty / observations)
            return np.stack(
                [
                    _solve_simplex(
                        self.xtx[horizon] / observations,
                        self.xty[horizon] / observations,
                    )
                    for horizon in range(self.horizon)
                ]
            )
        if self.mode == "shared":
            rms = np.maximum(np.sqrt(np.maximum(self.sum_squares, 0.0) / observations), 1e-12)
            matrix = self.xtx / np.outer(rms, rms) / observations
            target = self.xty / rms / observations
            standardized = _solve(matrix + float(alpha) * np.eye(self.features), target)
            return standardized / rms
        coefficients = np.empty((self.horizon, self.features), dtype=np.float64)
        for horizon in range(self.horizon):
            rms = np.maximum(
                np.sqrt(np.maximum(self.sum_squares[horizon], 0.0) / observations),
                1e-12,
            )
            matrix = self.xtx[horizon] / np.outer(rms, rms) / observations
            target = self.xty[horizon] / rms / observations
            standardized = _solve(matrix + float(alpha) * np.eye(self.features), target)
            coefficients[horizon] = standardized / rms
        return coefficients

    def mean_squared_error(self, coefficients: np.ndarray) -> float:
        """Return the exact scaled loss represented by these statistics."""
        if self.windows <= 0:
            raise ValueError("cannot score empty rolling statistics")
        if self.mode == "shared":
            error = (
                float(self.y_sum_squares)
                - 2.0 * float(coefficients @ self.xty)
                + float(coefficients @ self.xtx @ coefficients)
            )
            observations = self.windows * self.horizon
        else:
            error = np.sum(
                self.y_sum_squares
                - 2.0 * coefficients * self.xty
                + np.einsum("hf,hfg,hg->h", coefficients, self.xtx, coefficients)
            )
            observations = self.windows * self.horizon
        return max(float(error) / observations, 0.0)


def _solve(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, target)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, target, rcond=None)[0]


def _solve_simplex(xtx: np.ndarray, xty: np.ndarray) -> np.ndarray:
    """Small projected-gradient simplex least-squares solve."""
    count = len(xty)
    weights = np.full(count, 1.0 / count, dtype=np.float64)
    largest = max(float(np.linalg.norm(xtx, ord=2)), 1e-8)
    step = 1.0 / largest
    for _ in range(2_000):
        candidate = _simplex_projection(weights - step * (xtx @ weights - xty))
        if np.max(np.abs(candidate - weights)) < 1e-10:
            return candidate
        weights = candidate
    return weights


def _simplex_projection(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, len(values) + 1)
    positive = sorted_values - cumulative / indices > 0
    rho = int(np.flatnonzero(positive)[-1])
    theta = cumulative[rho] / float(rho + 1)
    return np.maximum(values - theta, 0.0)


def _predict(
    vanilla: np.ndarray,
    design: np.ndarray,
    coefficients: np.ndarray,
    *,
    mode: str,
    convex: bool,
    fitting_scope: str,
) -> np.ndarray:
    if fitting_scope == "same_user":
        correction = (
            np.einsum("uhf,uf->uh", design, coefficients)
            if mode == "shared"
            else np.einsum("uhf,uhf->uh", design, coefficients)
        )
    else:
        correction = (
            np.einsum("uhf,f->uh", design, coefficients)
            if mode == "shared"
            else np.einsum("uhf,hf->uh", design, coefficients)
        )
    return correction if convex else vanilla + correction


def _date_metrics(
    query_date: int,
    method: str,
    prediction: np.ndarray,
    target: np.ndarray,
    vanilla: np.ndarray,
    scale: np.ndarray,
) -> dict[str, Any]:
    error = prediction - target
    vanilla_error = vanilla - target
    window_mse = np.mean(np.square(error), axis=-1)
    vanilla_window_mse = np.mean(np.square(vanilla_error), axis=-1)
    return {
        "query_date": int(query_date),
        "method": method,
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
        "nmse": float(np.mean((error / scale) ** 2)),
        "nmae": float(np.mean(np.abs(error) / scale)),
        "vanilla_mse": float(np.mean(np.square(vanilla_error))),
        "vanilla_mae": float(np.mean(np.abs(vanilla_error))),
        "vanilla_nmse": float(np.mean((vanilla_error / scale) ** 2)),
        "vanilla_nmae": float(np.mean(np.abs(vanilla_error) / scale)),
        "win_rate": float(np.mean(window_mse < vanilla_window_mse)),
        "windows": int(target.shape[0]),
        "values": int(target.size),
    }


def _user_date_metrics(
    query_date: int,
    method: str,
    prediction: np.ndarray,
    target: np.ndarray,
    vanilla: np.ndarray,
    scale: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for user in range(target.shape[0]):
        error = prediction[user] - target[user]
        reference_error = vanilla[user] - target[user]
        denominator = float(np.asarray(scale[user]).reshape(-1)[0])
        mse = float(np.mean(np.square(error)))
        reference_mse = float(np.mean(np.square(reference_error)))
        nmse = float(np.mean(np.square(error / denominator)))
        reference_nmse = float(np.mean(np.square(reference_error / denominator)))
        rows.append(
            {
                "query_date": int(query_date),
                "user_id": int(user),
                "method": method,
                "mse": mse,
                "nmse": nmse,
                "mae": float(np.mean(np.abs(error))),
                "nmae": float(np.mean(np.abs(error) / denominator)),
                "reference_mse": reference_mse,
                "reference_nmse": reference_nmse,
                "reference_mae": float(np.mean(np.abs(reference_error))),
                "reference_nmae": float(np.mean(np.abs(reference_error) / denominator)),
                "delta_mse": mse - reference_mse,
                "delta_nmse": nmse - reference_nmse,
                "win": int(mse < reference_mse),
            }
        )
    return rows


def _aggregate_user_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("online adaptor produced no per-user evaluation rows")
    result: dict[str, Any] = {
        "method": rows[0]["method"],
        "dates": len({int(row["query_date"]) for row in rows}),
        "users": len({int(row["user_id"]) for row in rows}),
        "samples": len(rows),
    }
    for field in (
        "mse", "nmse", "mae", "nmae",
        "reference_mse", "reference_nmse", "reference_mae", "reference_nmae",
    ):
        result[field] = float(np.mean([float(row[field]) for row in rows]))
    result.update(
        {
            "vanilla_mse": result["reference_mse"],
            "vanilla_nmse": result["reference_nmse"],
            "vanilla_mae": result["reference_mae"],
            "vanilla_nmae": result["reference_nmae"],
            "win_rate_pct": 100.0 * float(np.mean([int(row["win"]) for row in rows])),
        }
    )
    per_user: dict[int, dict[str, float]] = {}
    for user in sorted({int(row["user_id"]) for row in rows}):
        selected = [row for row in rows if int(row["user_id"]) == user]
        per_user[user] = {
            field: float(np.mean([float(row[field]) for row in selected]))
            for field in ("mse", "nmse", "reference_mse", "reference_nmse")
        }
    worst_count = max(1, int(np.ceil(0.1 * len(per_user))))
    result["w10_mse"] = float(
        np.mean(sorted(values["mse"] for values in per_user.values())[-worst_count:])
    )
    result["w10_nmse"] = float(
        np.mean(sorted(values["nmse"] for values in per_user.values())[-worst_count:])
    )
    result["per_user_mse_population_std"] = float(
        np.std([values["mse"] for values in per_user.values()], ddof=0)
    )
    result["per_user_nmse_population_std"] = float(
        np.std([values["nmse"] for values in per_user.values()], ddof=0)
    )
    result["users_improved_mse_pct"] = 100.0 * float(
        np.mean([values["mse"] < values["reference_mse"] for values in per_user.values()])
    )
    result["users_improved_nmse_pct"] = 100.0 * float(
        np.mean([values["nmse"] < values["reference_nmse"] for values in per_user.values()])
    )
    result["relative_mse_improvement_pct"] = 100.0 * (
        result["reference_mse"] - result["mse"]
    ) / max(result["reference_mse"], 1e-12)
    result["relative_nmse_improvement_pct"] = 100.0 * (
        result["reference_nmse"] - result["nmse"]
    ) / max(result["reference_nmse"], 1e-12)
    return result


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("online adaptor produced no evaluation rows")
    fields = (
        "mse",
        "mae",
        "nmse",
        "nmae",
        "vanilla_mse",
        "vanilla_mae",
        "vanilla_nmse",
        "vanilla_nmae",
        "win_rate",
    )
    result: dict[str, Any] = {"method": rows[0]["method"], "dates": len(rows)}
    for field in fields:
        result[field] = float(np.mean([float(row[field]) for row in rows]))
    result["relative_mse_improvement_pct"] = 100.0 * (
        result["vanilla_mse"] - result["mse"]
    ) / max(result["vanilla_mse"], 1e-12)
    result["relative_nmse_improvement_pct"] = 100.0 * (
        result["vanilla_nmse"] - result["nmse"]
    ) / max(result["vanilla_nmse"], 1e-12)
    result["win_rate_pct"] = 100.0 * result.pop("win_rate")
    return result


def _iter_rows(date_indices: Iterator[int], n_users: int) -> Iterator[tuple[int, int]]:
    for date_index in date_indices:
        for user in range(n_users):
            yield date_index, user
