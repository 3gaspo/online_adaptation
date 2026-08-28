"""Focused synthetic checks for exact causal online fitting."""

from __future__ import annotations

from pathlib import Path
import csv
import gc
import json
import sys
import tempfile
import types

import numpy as np
import torch

import src.pipeline.adaptation as adaptation_pipeline
from src.pipeline.runs import SCHEMA_VERSION, allocate_run, mark_status
from src.pipeline.contracts import (
    EXTRACTION_FORMAT,
    RESULT_FORMAT,
    count_named_parameters,
    load_array_manifest,
    open_extraction_arrays,
    write_array_manifest,
)
from src.proposal.contracts import AdapterConfig, ExtractionConfig
from src.proposal.datastore import candidate_dates, fitting_dates
from src.proposal.date_planning import build_date_plan
from src.proposal.gates import _catboost_adaptor_parameter_count
from src.pipeline.adaptation import evaluate_online_gate, evaluate_online_linear
from src.pipeline.extraction import extract_online_features
from src.proposal import (
    DEFAULT_N_DATASTORE_DATES,
    DEFAULT_N_FIT,
    DEFAULT_N_STORE_WINDOWS,
    DEFAULT_TSRAG_K,
)
from src.proposal.extraction import ContextForecastCache
from src.results.efficiency import write_compute_timing
from src.results.reporting import build_online_tables
from src.pipeline.profiles import RANGE_SETTINGS, dataset_frequency, tasks_for_family
from src.external_models.tsrag.retriever import TSRAGRetriever
from src.model_loading.forecast import (
    FOUNDATION_MODEL_ALIASES,
    RetrievalCovariateAdapter,
)


def _fake_extraction(
    root: Path,
    *,
    neighbors: int = 2,
    fixed_training_set: bool = False,
) -> types.SimpleNamespace:
    rng = np.random.default_rng(7)
    dates, users, lookback, horizon = 30, 3, 4, 2
    feature_root = root / "features"
    feature_root.mkdir(parents=True)
    values = rng.normal(size=(dates + lookback + horizon - 1, users)).astype(np.float32)
    values += np.arange(users, dtype=np.float32)[None, :] * 0.2
    windows = np.lib.stride_tricks.sliding_window_view(
        values, lookback + horizon, axis=0
    )
    x = windows[:, :, :lookback]
    y = windows[:, :, lookback:]
    vanilla = np.repeat(x[:, :, -1:], horizon, axis=-1).astype(np.float32)
    window_dates = np.arange(lookback - 1, lookback - 1 + dates, dtype=np.int64)
    retrieval_dates = window_dates[10:]
    neighbor_id = np.empty((len(retrieval_dates), users, neighbors), dtype=np.int64)
    for index, retrieval_date in enumerate(retrieval_dates):
        window_position = int(np.searchsorted(window_dates, retrieval_date))
        for user in range(users):
            for rank in range(neighbors):
                neighbor_id[index, user, rank] = max(0, window_position - rank - 1) * users + user
    forecast_ids = np.arange(dates * users, dtype=np.int64)
    arrays = {
        "window_dates": window_dates,
        "window_timestamps": np.asarray([str(date) for date in window_dates]),
        "window_users": np.asarray([str(user) for user in range(users)]),
        "window_mean": x.mean(axis=-1),
        "window_std": x.std(axis=-1),
        "window_constant": x.std(axis=-1) <= 1e-8,
        "forecast_window_id": forecast_ids,
        "forecast_value": vanilla.reshape(-1, horizon),
        "retrieval_window_dates": retrieval_dates,
        "is_evaluation_query": np.ones(len(retrieval_dates), dtype=np.bool_),
        "neighbor_window_id": neighbor_id,
        "distance": np.broadcast_to(
            np.linspace(0.1, 0.1 * neighbors, neighbors, dtype=np.float32),
            (len(retrieval_dates), users, neighbors),
        ).copy(),
        "neighbor_distance_raw": np.broadcast_to(
            np.linspace(0.3, 0.3 + 0.1 * (neighbors - 1), neighbors, dtype=np.float32),
            (len(retrieval_dates), users, neighbors),
        ).copy(),
        "neighbor_distance_instance_normalized": np.broadcast_to(
            np.linspace(0.5, 0.5 + 0.1 * (neighbors - 1), neighbors, dtype=np.float32),
            (len(retrieval_dates), users, neighbors),
        ).copy(),
        "candidate_count": np.full(len(retrieval_dates), 12, dtype=np.int64),
    }
    for name, value in arrays.items():
        np.save(feature_root / f"{name}.npy", value)
    config = ExtractionConfig(
        dataset="synthetic",
        lookback=lookback,
        horizon=horizon,
        n_datastore_dates=9 if fixed_training_set else 1,
        n_fit=5,
        max_k=neighbors,
        fixed_training_set=fixed_training_set,
        store_stride=1,
        fit_stride=1,
        align_period=False,
    )
    write_array_manifest(
        root,
        config=config.scientific_dict(),
        arrays=arrays,
        metadata={"users": users},
    )
    (root / "extraction_timing.json").write_text(
        json.dumps(
            {
                "users": users,
                "total_extraction_seconds": 0.2,
                "cold_batch": {
                    "evaluation_query_date": int(retrieval_dates[0]),
                    "retrieval_windows": 6,
                    "metadata_and_statistics_seconds": 0.01,
                    "retrieval_and_representation_seconds": 0.06,
                    "forecast_seconds": 0.02,
                    "extraction_seconds": 0.09,
                },
            }
        ),
        encoding="utf-8",
    )
    return types.SimpleNamespace(
        values=values,
        n_dates=len(values),
        n_users=users,
    )


def test_precomputed_date_plan_and_datastore_capacity() -> None:
    rolling = ExtractionConfig(
        dataset="synthetic",
        lookback=4,
        horizon=2,
        n_datastore_dates=5,
        n_fit=3,
        retrieval_scope="all",
        store_stride=1,
        fit_stride=1,
        align_period=False,
    )
    rolling_plan = build_date_plan(n_dates=100, config=rolling)
    assert rolling_plan.first_retrieval_date == 9
    assert rolling_plan.evaluation_start_date == 13
    assert candidate_dates(10, config=rolling, plan=rolling_plan).tolist() == [4, 5, 6, 7, 8]
    assert fitting_dates(20, config=rolling, plan=rolling_plan).tolist() == [16, 17, 18]

    evaluation_grids = set()
    for fixed_datastore in (False, True):
        for fixed_training_set in (False, True):
            config = ExtractionConfig(
                dataset="synthetic",
                lookback=4,
                horizon=2,
                n_datastore_dates=5,
                n_fit=3,
                fixed_datastore=fixed_datastore,
                fixed_training_set=fixed_training_set,
                store_stride=1,
                fit_stride=1,
                align_period=False,
                query_stride=7,
            )
            plan = build_date_plan(n_dates=100, config=config)
            evaluation_grids.add(plan.evaluation_query_dates)
            if fixed_training_set:
                assert fitting_dates(50, config=config, plan=plan).tolist() == [9, 10, 11]
    assert len(evaluation_grids) == 1

    aligned = ExtractionConfig(
        dataset="synthetic",
        lookback=4,
        horizon=2,
        n_datastore_dates=4,
        n_fit=3,
        retrieval_scope="all",
        store_stride=4,
        fit_stride=1,
        align_period=True,
        period=2,
    )
    aligned_plan = build_date_plan(n_dates=100, config=aligned)
    aligned_dates = candidate_dates(20, config=aligned, plan=aligned_plan)
    assert np.all((20 - aligned_dates) % 2 == 0)
    assert np.all(np.diff(aligned_dates) == 4)
    try:
        ExtractionConfig(
            dataset="synthetic",
            lookback=4,
            horizon=2,
            store_stride=3,
            align_period=True,
            period=2,
        ).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("period alignment must reject a non-multiple stride")
    assert aligned_dates.tolist() == [6, 10, 14, 18]

    ratio_config = ExtractionConfig(
        dataset="synthetic",
        lookback=4,
        horizon=2,
        n_datastore_dates=0.5,
        n_fit=3,
        store_stride=4,
        fit_stride=1,
        align_period=True,
        period=2,
    )
    ratio_plan = build_date_plan(n_dates=100, config=ratio_config)
    assert ratio_plan.n_datastore_dates == 11
    assert ratio_plan.datastore_end_date == 46


def test_tsrag_upstream_search_rule() -> None:
    class FakeIndexFlatL2:
        def __init__(self, dimension: int) -> None:
            self.dimension = dimension

        def add(self, candidates: np.ndarray) -> None:
            assert candidates.shape[1] == self.dimension
            self.candidates = candidates

        def search(
            self,
            query: np.ndarray,
            count: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            squared = ((query[:, None, :] - self.candidates[None, :, :]) ** 2).sum(-1)
            indices = np.argsort(squared, axis=1)[:, :count]
            return np.take_along_axis(squared, indices, axis=1), indices

    previous = sys.modules.get("faiss")
    sys.modules["faiss"] = types.SimpleNamespace(IndexFlatL2=FakeIndexFlatL2)
    try:
        retriever = object.__new__(TSRAGRetriever)
        distances, indices = retriever.search(
            np.asarray([[0.0, 0.0]], dtype=np.float32),
            np.asarray([[2.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            top_k=2,
        )
    finally:
        if previous is None:
            sys.modules.pop("faiss", None)
        else:
            sys.modules["faiss"] = previous
    assert indices.tolist() == [[1, 2]]
    assert distances.tolist() == [[0.0, 1.0]]

    class FakeInput:
        on_cpu = False

        def squeeze(self, dimension: int) -> "FakeInput":
            assert dimension == 1
            return self

        def cpu(self) -> "FakeInput":
            self.on_cpu = True
            return self

    class FakePipeline:
        @staticmethod
        def embed(value: FakeInput) -> tuple[torch.Tensor, None]:
            assert value.on_cpu
            return torch.zeros(1, 2, 3), None

    retriever = object.__new__(TSRAGRetriever)
    torch.nn.Module.__init__(retriever)
    retriever.pipeline = FakePipeline()
    assert retriever.representation(FakeInput()).shape == (1, 3)


def test_compact_extraction_and_independent_fitting_grid() -> None:
    class Dataset:
        def __init__(self) -> None:
            dates = np.arange(48, dtype=np.float32)
            self.values = np.stack((dates, np.sin(dates / 3.0)), axis=1)
            self.datetimes = list(range(len(dates)))
            self.user_names = ["trend", "wave"]
            self.n_dates = len(dates)
            self.n_users = 2

    class Model(torch.nn.Module):
        supports_context = True

        def __init__(self) -> None:
            super().__init__()
            self.context_shape = None

        def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if context is not None:
                self.context_shape = tuple(context.shape)
            return x[..., -1:].repeat(1, 1, 2)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        dataset = Dataset()
        model = Model()
        outputs = extract_online_features(
            dataset=dataset,
            model=model,
            config=ExtractionConfig(
                dataset="synthetic",
                lookback=4,
                horizon=2,
                n_datastore_dates=4,
                n_fit=3,
                max_k=2,
                store_stride=2,
                fit_stride=1,
                align_period=False,
                query_stride=4,
            ),
            output_dir=root,
            device=torch.device("cpu"),
        )
        assert all(path.is_file() for path in outputs.values())
        manifest = load_array_manifest(root)
        assert "window_lookback" not in manifest["arrays"]
        assert "window_horizon" not in manifest["arrays"]
        assert "window_forecast" not in manifest["arrays"]
        assert {"forecast_window_id", "forecast_value"} <= set(manifest["arrays"])
        arrays = open_extraction_arrays(root, dataset=dataset)
        assert arrays["x"].shape[-1] == 4
        assert arrays["y"].shape[-1] == 2
        assert np.any(~np.asarray(arrays["is_evaluation_query"], dtype=bool))
        context = ContextForecastCache(
            arrays=arrays,
            output_dir=root,
            model=model,
            device=torch.device("cpu"),
        )
        assert context.get(0, 2).shape == (2, 2)
        assert model.context_shape == (2, 2, 6)
        timing = json.loads((root / "extraction_timing.json").read_text())
        assert timing["cold_batch"]["extraction_seconds"] > 0.0
        del context
        del arrays
        gc.collect()


def test_retrieval_context_becomes_configured_covariates() -> None:
    class CovariateBase(torch.nn.Module):
        supports_covariates = True

        def __init__(self) -> None:
            super().__init__()
            self.lags = 4
            self.dim = 1
            self.horizon = 2
            self.received = None

        def forward(self, x: torch.Tensor, covariates=None) -> torch.Tensor:
            self.received = covariates
            return x[..., -1:].repeat(1, 1, self.horizon)

    x = torch.zeros(2, 1, 4)
    context = torch.arange(24, dtype=torch.float32).reshape(2, 2, 6)
    base = CovariateBase()
    model = RetrievalCovariateAdapter(base, mode="past_and_future")
    prediction = model(x, context=context)
    assert prediction.shape == (2, 1, 2)
    assert torch.equal(base.received["past"], context[..., :4])
    assert torch.equal(base.received["future"], context[..., 4:])

    disabled = RetrievalCovariateAdapter(CovariateBase(), mode="none")
    try:
        disabled(x, context=context)
    except ValueError as error:
        assert "disabled" in str(error)
    else:
        raise AssertionError("disabled retrieval covariates must be rejected")

    unsupported = CovariateBase()
    unsupported.supports_covariates = False
    rejected = RetrievalCovariateAdapter(unsupported, mode="past_and_future")
    try:
        rejected(x, context=context)
    except ValueError as error:
        assert "does not consume covariates" in str(error)
    else:
        raise AssertionError("unsupported retrieval covariates must be rejected")

    assert FOUNDATION_MODEL_ALIASES == (
        "chronos2",
        "chronos_bolt",
        "chronos_t5",
        "ts_icl",
    )


def test_rolling_ridge_and_bayes_outputs() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        extraction = root / "extraction"
        dataset = _fake_extraction(extraction)
        ridge = evaluate_online_linear(
            extraction_dir=extraction,
            output_dir=root / "ridge",
            config=AdapterConfig(
                n_fit=5, alpha=1e-2, tune_alpha=False, used_k=2
            ),
            dataset=dataset,
        )
        assert all(path.is_file() for path in ridge.values())
        ridge_manifest = json.loads(ridge["manifest"].read_text(encoding="utf-8"))
        assert ridge_manifest["format"] == RESULT_FORMAT
        assert ridge_manifest["parameters"] == {
            "adaptor": len(ridge_manifest["feature_names"]),
            "backbone_included": False,
            "definition": "maximum fitted linear coefficients used per evaluation query user",
        }
        timing = json.loads(ridge["compute_timing"].read_text(encoding="utf-8"))
        assert timing["average_seconds_per_sample"] > 0.0
        assert timing["cold_batch"]["total_seconds"] > 0.0
        metrics = json.loads(ridge["metrics"].read_text(encoding="utf-8"))
        assert {"w10_mse", "w10_nmse", "win_rate_pct"} <= set(metrics)
        assert list(csv.DictReader(ridge["per_user_date"].open(encoding="utf-8")))
        same_user_trajectory = np.load(ridge["trajectory"])
        assert same_user_trajectory.ndim == 3
        assert not np.allclose(
            same_user_trajectory[-1, 0],
            same_user_trajectory[-1, 2],
        )
        all_users = evaluate_online_linear(
            extraction_dir=extraction,
            output_dir=root / "all_users",
            config=AdapterConfig(
                n_fit=5,
                fitting_scope="all",
                alpha=1e-2,
                tune_alpha=False,
                used_k=2,
            ),
            dataset=dataset,
        )
        assert np.load(all_users["trajectory"]).ndim == 2
        assert "user_id" not in next(
            csv.DictReader(all_users["selection"].open(encoding="utf-8"))
        )
        for method in (
            "cov_ridge_shared",
            "avgy_ridge_shared",
            "y_ridge_shared",
            "cov_y_ridge_shared",
            "cov_avgy_ridge_shared",
            "residual_ridge_shared",
            "full_ridge_horizon",
            "full_delta_ridge_shared",
            "full_delta_ridge_horizon",
            "full_convex_shared",
            "full_convex_horizon",
        ):
            outputs = evaluate_online_linear(
                extraction_dir=extraction,
                output_dir=root / method,
                config=AdapterConfig(
                    method=method,
                    n_fit=5,
                    alpha=1e-2,
                    tune_alpha=False,
                    used_k=2,
                ),
                dataset=dataset,
            )
            assert outputs["metrics"].is_file()
        bayes = evaluate_online_gate(
            extraction_dir=extraction,
            output_dir=root / "bayes",
            config=AdapterConfig(
                method="bayes_cov_shared_soft", n_fit=5, used_k=2
            ),
            dataset=dataset,
        )
        assert all(path.is_file() for path in bayes.values())
        bayes_manifest = json.loads(bayes["manifest"].read_text(encoding="utf-8"))
        assert bayes_manifest["parameters"]["adaptor"] == 2
        assert bayes_manifest["parameters"]["backbone_included"] is False
        all_user_bayes = evaluate_online_gate(
            extraction_dir=extraction,
            output_dir=root / "all_user_bayes",
            config=AdapterConfig(
                method="bayes_cov_shared_soft",
                n_fit=5,
                fitting_scope="all",
                used_k=2,
            ),
            dataset=dataset,
        )
        assert all_user_bayes["metrics"].is_file()


def test_compute_timing_contract() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = write_compute_timing(
            temporary,
            extraction_timing={
                "users": 4,
                "total_extraction_seconds": 8.0,
                "cold_batch": {
                    "evaluation_query_date": 12,
                    "metadata_and_statistics_seconds": 1.0,
                    "retrieval_and_representation_seconds": 2.0,
                    "forecast_seconds": 3.0,
                    "extraction_seconds": 6.0,
                },
            },
            adaptation_seconds=4.0,
            evaluation_samples=8,
            cold_adaptation_seconds=7.0,
            method="external",
        )
        timing = json.loads(path.read_text(encoding="utf-8"))
        assert timing["total_experiment_seconds"] == 12.0
        assert timing["average_seconds_per_sample"] == 1.5
        assert timing["cold_batch"]["total_seconds"] == 13.0
        assert timing["cold_batch"]["seconds_per_sample"] == 3.25
        assert "source-view reconstruction" in timing["definition"]


def test_per_query_ridge_hyperparameter_selection() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        extraction = root / "extraction"
        dataset = _fake_extraction(extraction, neighbors=15)
        outputs = evaluate_online_linear(
            extraction_dir=extraction,
            output_dir=root / "tuned_ridge",
            config=AdapterConfig(
                n_fit=10,
                validation_ratio=0.2,
                alpha_grid=(1e-1, 1e-2, 1e-3),
                candidate_k_grid=(1, 5, 10, 15),
            ),
            dataset=dataset,
        )
        selected = list(csv.DictReader(outputs["selection"].open(encoding="utf-8")))
        assert selected
        assert {int(row["selected_k"]) for row in selected} <= {1, 5, 10, 15}
        assert {float(row["selected_alpha"]) for row in selected} <= {1e-1, 1e-2, 1e-3}
        manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
        assert manifest["format"] == RESULT_FORMAT
        assert manifest["hyperparameter_selection"]["enabled"] is True
        assert manifest["files"]["selected_hyperparameters"] == outputs["selection"].name

        strict_k = evaluate_online_linear(
            extraction_dir=extraction,
            output_dir=root / "strict_k",
            config=AdapterConfig(
                n_fit=10,
                used_k=5,
                alpha_grid=(1e-1, 1e-2),
            ),
            dataset=dataset,
        )
        strict_rows = list(
            csv.DictReader(strict_k["selection"].open(encoding="utf-8"))
        )
        assert {int(row["selected_k"]) for row in strict_rows} == {5}
        strict_manifest = json.loads(
            strict_k["manifest"].read_text(encoding="utf-8")
        )
        assert strict_manifest["hyperparameter_selection"]["alpha_enabled"] is True
        assert strict_manifest["hyperparameter_selection"]["k_enabled"] is False

        strict_alpha = evaluate_online_linear(
            extraction_dir=extraction,
            output_dir=root / "strict_alpha",
            config=AdapterConfig(
                n_fit=10,
                alpha=1e-2,
                tune_alpha=False,
                candidate_k_grid=(1, 5, 10, 15),
            ),
            dataset=dataset,
        )
        strict_alpha_manifest = json.loads(
            strict_alpha["manifest"].read_text(encoding="utf-8")
        )
        assert strict_alpha_manifest["hyperparameter_selection"]["alpha_enabled"] is False
        assert strict_alpha_manifest["hyperparameter_selection"]["k_enabled"] is True

        same_user_extraction = root / "same_user_extraction"
        same_user_dataset = _fake_extraction(same_user_extraction, neighbors=2)
        same_user = evaluate_online_linear(
            extraction_dir=same_user_extraction,
            output_dir=root / "same_user_tuned_ridge",
            config=AdapterConfig(
                n_fit=5,
                validation_ratio=0.2,
                alpha_grid=(1e-2, 1e-3),
                candidate_k_grid=(1, 2),
            ),
            dataset=same_user_dataset,
        )
        same_user_selected = list(
            csv.DictReader(same_user["selection"].open(encoding="utf-8"))
        )
        assert {row["user_id"] for row in same_user_selected} == {"0", "1", "2"}


def test_fixed_training_set_solves_one_ridge_per_user() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        extraction = root / "extraction"
        dataset = _fake_extraction(
            extraction,
            neighbors=2,
            fixed_training_set=True,
        )
        original_solve = adaptation_pipeline.LinearStatistics.solve
        solve_calls = 0

        def counted_solve(
            statistics: adaptation_pipeline.LinearStatistics,
            alpha: float,
        ) -> np.ndarray:
            nonlocal solve_calls
            solve_calls += 1
            return original_solve(statistics, alpha)

        adaptation_pipeline.LinearStatistics.solve = counted_solve
        try:
            outputs = evaluate_online_linear(
                extraction_dir=extraction,
                output_dir=root / "fixed_ridge",
                config=AdapterConfig(
                    method="y_ridge_shared",
                    n_fit=5,
                    alpha=1e-2,
                    tune_alpha=False,
                    used_k=2,
                    fixed_training_set=True,
                ),
                dataset=dataset,
            )
        finally:
            adaptation_pipeline.LinearStatistics.solve = original_solve
        assert solve_calls == dataset.n_users
        assert len(np.load(outputs["trajectory"])) > 1


def test_adaptor_parameter_helpers() -> None:
    class Parameter:
        def __init__(self, size: int) -> None:
            self.size = size

        def numel(self) -> int:
            return self.size

    named = [
        ("backbone.weight", Parameter(100)),
        ("mha.weight", Parameter(12)),
        ("gate_layer.bias", Parameter(3)),
    ]
    assert count_named_parameters(
        named,
        prefixes=("encode_mlp.", "mha.", "ffn.", "gate_layer."),
    ) == 15

    class CatBoostStub:
        @staticmethod
        def get_leaf_values() -> np.ndarray:
            return np.zeros(7)

    assert _catboost_adaptor_parameter_count(CatBoostStub()) == 7


def test_report_requires_identical_dates() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        results = root / "results"
        for index, dates in enumerate(((1, 2, 3), (1, 2, 3))):
            allocation = allocate_run(
                results / f"method_{index}",
                project="online_adaptation",
                workflow="test_report",
                dataset="synthetic",
                lookback=4,
                horizon=2,
                backbone="chronos2",
                model_config_order=("method",),
                model_config={"method": f"method_{index}"},
                pipeline_config={"query_stride": 1},
                seeds=(1,),
                purpose="publication",
                display_name=f"method {index}",
            )
            run = allocation.run_dir
            (run / "result_manifest.json").write_text(
                json.dumps({"method": f"method_{index}"}), encoding="utf-8"
            )
            rows = []
            for date in dates:
                rows.append(
                    {
                        "query_date": date,
                        "method": f"method_{index}",
                        "mse": 0.5,
                        "mae": 0.4,
                        "nmse": 0.3,
                        "nmae": 0.2,
                        "vanilla_mse": 1.0,
                        "vanilla_mae": 0.8,
                        "vanilla_nmse": 0.6,
                        "vanilla_nmae": 0.4,
                        "win_rate": 0.75,
                        "windows": 3,
                        "values": 6,
                    }
                )
            with (run / "per_date_metrics.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            mark_status(
                run,
                "completed",
                required_artifacts=("result_manifest.json", "per_date_metrics.csv"),
            )
        outputs = build_online_tables(results_root=results, output_dir=root / "tables")
        assert all(path.is_file() for path in outputs.values())
        detailed = list(csv.DictReader(outputs["detailed_csv"].open(encoding="utf-8")))
        assert {row["dates"] for row in detailed} == {"3"}
        report_manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
        assert report_manifest["requested"]["config_policy"] == "distinct"
        assert report_manifest["requested"]["repeat_policy"] == "selected"
        assert report_manifest["obtained"]["count"] == 2


def test_profile_contract() -> None:
    project_root = Path(__file__).resolve().parents[2]
    runner = (project_root / "src/slurm/run_family.sh").read_text(encoding="utf-8")
    config = (project_root / "src/conf/config.yaml").read_text(encoding="utf-8")
    slurm_root = project_root / "slurm"
    scope_front = slurm_root / "dgx/ablations/ablation_general_scope.slurm"
    assert DEFAULT_N_DATASTORE_DATES == 100
    assert DEFAULT_N_STORE_WINDOWS == 10_000
    assert DEFAULT_N_FIT == 10
    assert DEFAULT_TSRAG_K == 5
    assert ExtractionConfig(dataset="synthetic", lookback=4, horizon=2).n_fit == 10
    assert ExtractionConfig(
        dataset="daily", lookback=4, horizon=2, period=7, store_stride=7
    ).fit_stride == 7
    assert AdapterConfig().n_fit == 10
    assert runner.count('PROFILE_N_FIT="${N_FIT:-10}"') == 2
    assert 'srun --ntasks=1 python -m "$module"' in runner
    assert "n_fit: 10" in config
    assert "fit_stride: null" in config
    assert "retrieval_covariate_mode: null" in config
    assert '"fit_stride=${FIT_STRIDE:-null}"' in runner
    assert (
        '"retrieval_covariate_mode=${RETRIEVAL_COVARIATE_MODE:-null}"'
        in runner
    )
    assert "fitting_scope: same_user" in config
    assert '"fitting_scope=${FITTING_SCOPE:-same_user}"' in runner
    assert scope_front.is_file()
    assert not (
        slurm_root / "dgx/ablations/ablation_retrieval_scope.slurm"
    ).exists()
    assert "EXPERIMENT_FAMILY=general_scope_ablation" in scope_front.read_text(
        encoding="utf-8"
    )
    front_families = set()
    for front in slurm_root.glob("*/*/*.slurm"):
        assignments = [
            line
            for line in front.read_text(encoding="utf-8").splitlines()
            if line.startswith("EXPERIMENT_FAMILY=")
        ]
        assert len(assignments) == 1
        front_families.add(assignments[0].split("=", 1)[1])
    assert front_families == {
        "main",
        "tsrag_comparison",
        "deadline_fixed_protocol",
        "deadline_tsrag_comparison",
        "n_datastore_dates_ablation",
        "n_fit_ablation",
        "fit_stride_ablation",
        "alpha_ablation",
        "k_ablation",
        "l_ablation",
        "h_ablation",
        "feature_design_ablation",
        "formulation_ablation",
        "fixed_protocol_ablation",
        "sota_backbone_ablation",
        "general_scope_ablation",
        "homogeneous_ablation",
        "backbone_ablation",
    }
    assert all(
        tasks_for_family(family, "test", project_root)
        for family in front_families
        if not family.startswith("deadline_")
    )
    scope_tasks = tasks_for_family("general_scope_ablation", "test", project_root)
    assert {
        (task["retrieval_scope"], task["fitting_scope"])
        for task in scope_tasks
    } == {
        (retrieval_scope, fitting_scope)
        for retrieval_scope in ("all", "same_user", "other_users")
        for fitting_scope in ("all", "same_user")
    }
    assert {
        task["fit_stride"]
        for task in tasks_for_family("fit_stride_ablation", "test", project_root)
    } == {1, 24}
    homogeneous_test = tasks_for_family(
        "homogeneous_ablation", "test", project_root
    )
    assert {task["dataset"] for task in homogeneous_test} == {"Weather"}
    assert {
        (task["lookback"], task["horizon"]) for task in homogeneous_test
    } == {RANGE_SETTINGS["hourly"]["long"]}
    assert {task["homogeneous_only"] for task in homogeneous_test} == {False, True}
    standard_test_families = (
        "main",
        "n_datastore_dates_ablation",
        "n_fit_ablation",
        "fit_stride_ablation",
        "alpha_ablation",
        "k_ablation",
        "feature_design_ablation",
        "formulation_ablation",
        "fixed_protocol_ablation",
        "general_scope_ablation",
        "homogeneous_ablation",
    )
    for family in standard_test_families:
        tasks = tasks_for_family(family, "test", project_root)
        cadence_tasks = [task for task in tasks if task["method"] != "tsrag"]
        assert cadence_tasks
        assert all(
            (task["lookback"], task["horizon"])
            == RANGE_SETTINGS[dataset_frequency(task["dataset"], project_root)]["long"]
            for task in cadence_tasks
        )
    assert {
        task["method"]
        for task in tasks_for_family("main", "test", project_root)
    } == {
        "full_ridge_shared",
        "y_convex_shared",
        "bayes_cov_shared_soft",
        "covariate_prediction",
    }
    tsrag_fixed = [
        task
        for task in tasks_for_family("tsrag_comparison", "test", project_root)
        if task["method"] == "tsrag"
    ]
    assert {(task["lookback"], task["horizon"]) for task in tsrag_fixed} == {
        (512, 64)
    }
    assert {(task["max_k"], task["used_k"]) for task in tsrag_fixed} == {(5, 5)}
    assert {task["retrieval_scope"] for task in tsrag_fixed} == {"same_user"}
    assert {task["fixed_datastore"] for task in tsrag_fixed} == {True}
    assert {task["store_stride"] for task in tsrag_fixed} == {1}
    assert {task["align_period"] for task in tsrag_fixed} == {False}
    k_tasks = tasks_for_family("k_ablation", "test", project_root)
    assert {task["max_k"] for task in k_tasks} == {20}
    assert {task["used_k"] for task in k_tasks} == {1, 3, 5, 10, 15, 20}
    alpha_tasks = tasks_for_family("alpha_ablation", "test", project_root)
    assert {task["used_k"] for task in alpha_tasks} == {None}
    assert {task["tune_alpha"] for task in alpha_tasks} == {False}
    for family in (
        "n_datastore_dates_ablation",
        "n_fit_ablation",
        "fit_stride_ablation",
        "alpha_ablation",
        "l_ablation",
        "h_ablation",
        "feature_design_ablation",
        "formulation_ablation",
        "fixed_protocol_ablation",
        "sota_backbone_ablation",
        "general_scope_ablation",
        "homogeneous_ablation",
        "backbone_ablation",
    ):
        assert {
            task["used_k"]
            for task in tasks_for_family(family, "test", project_root)
        } == {None}
    for family in ("backbone_ablation", "sota_backbone_ablation"):
        assert {
            (task["lookback"], task["horizon"])
            for task in tasks_for_family(family, "test", project_root)
        } == {(512, 64)}
    assert {
        task["backbone"]
        for task in tasks_for_family("backbone_ablation", "test", project_root)
    } == {"chronos2", "chronos_bolt", "chronos_t5", "ts_icl"}
    assert {
        task["method"]
        for task in tasks_for_family("backbone_ablation", "test", project_root)
    } == {"y_ridge_shared"}
    assert {
        task["retrieval_covariate_mode"]
        for task in tasks_for_family("backbone_ablation", "test", project_root)
    } == {"none"}
    homogeneous_full = tasks_for_family(
        "homogeneous_ablation", "full", project_root
    )
    assert {task["dataset"] for task in homogeneous_full} == {
        "ETTh1",
        "ETTh2",
        "ETTm1",
        "ETTm2",
        "Weather",
    }
    assert {
        (task["lookback"], task["horizon"])
        for task in homogeneous_full
        if task["dataset"] in {"ETTm1", "ETTm2"}
    } == {(96, 4), (192, 8), (672, 96)}
    assert {
        (task["lookback"], task["horizon"])
        for task in homogeneous_full
        if task["dataset"] in {"ETTh1", "ETTh2", "Weather"}
    } == {(168, 24), (336, 48), (504, 168)}
    assert len(homogeneous_full) == 30
    assert EXTRACTION_FORMAT == "online_extraction_v1"
    assert RESULT_FORMAT == "online_adaptation_v1"
    assert SCHEMA_VERSION == 1

    with tempfile.TemporaryDirectory(dir=project_root / "outputs") as directory:
        data_root = Path(directory)
        catalog_root = data_root / "time"
        catalog_root.mkdir()
        catalog = {
            "datasets": [
                {
                    "name": "hourly_panel",
                    "configured_frequency": "H",
                    "num_series": 100,
                    "num_timestamps": 100_000,
                },
                {
                    "name": "daily_panel",
                    "configured_frequency": "D",
                    "num_series": 100,
                    "num_timestamps": 100_000,
                },
                {
                    "name": "quarter_hour_panel",
                    "configured_frequency": "15T",
                    "num_series": 100,
                    "num_timestamps": 100_000,
                },
            ]
        }
        (catalog_root / "catalog.json").write_text(
            json.dumps(catalog), encoding="utf-8"
        )
        test_tasks = tasks_for_family("main", "test", data_root)
        assert {
            (task["dataset"], task["lookback"], task["horizon"])
            for task in test_tasks
        } == {
            ("Electricity", *RANGE_SETTINGS["hourly"]["long"]),
        }
        full_tasks = tasks_for_family("main", "full", data_root)
        expected_settings = {
            "Electricity": set(RANGE_SETTINGS["hourly"].values()),
            "Solar": set(RANGE_SETTINGS["hourly"].values()),
            "Traffic": set(RANGE_SETTINGS["hourly"].values()),
            "ETTh1": set(RANGE_SETTINGS["hourly"].values()),
            "ETTh2": set(RANGE_SETTINGS["hourly"].values()),
            "ETTm1": set(RANGE_SETTINGS["15min"].values()),
            "ETTm2": set(RANGE_SETTINGS["15min"].values()),
            "time/hourly_panel": set(RANGE_SETTINGS["hourly"].values()),
            "time/daily_panel": set(RANGE_SETTINGS["daily"].values()),
            "time/quarter_hour_panel": set(RANGE_SETTINGS["15min"].values()),
        }
        for dataset, settings in expected_settings.items():
            assert {
                (task["lookback"], task["horizon"])
                for task in full_tasks
                if task["dataset"] == dataset
            } == settings
    assert "max_k: null" in config
    assert "candidate_k_grid: null" in config
    assert "used_k: null" in config
    assert "tune_alpha: true" in config
    assert "tsrag_k: null" in config
    assert "query_stride: 127" in config
    assert "ridge_validation_ratio: 0.2" in config
    assert '"ridge_validation_ratio=${RIDGE_VALIDATION_RATIO:-0.2}"' in runner
    assert '"max_k=${MAX_K:-null}"' in runner
    assert '"candidate_k_grid=${CANDIDATE_K_GRID:-null}"' in runner
    assert '"used_k=${USED_K:-null}"' in runner
    assert '"tsrag_k=${TSRAG_K:-null}"' in runner
    assert 'PROFILE_QUERY_STRIDE="${QUERY_STRIDE:-257}"' in runner
    assert 'PROFILE_QUERY_STRIDE="${QUERY_STRIDE:-127}"' in runner
    assert 'PROFILE_PURPOSE="${PURPOSE:-smoke}"' in runner
    assert 'PROFILE_PURPOSE="${PURPOSE:-publication}"' in runner
    assert "module=src.scripts.extract" in runner
    assert "module=src.scripts.adapt" in runner
    assert "module=src.scripts.tables" in runner
    for package in (
        "data",
        "model_loading",
        "external_models",
        "proposal",
        "scripts",
        "conf",
        "pipeline",
        "results",
        "visualization",
    ):
        assert (project_root / "src" / package).is_dir()
    assert not (project_root / "src" / "online").exists()
    assert not (project_root / "src" / "models").exists()
    assert not (project_root / "src" / "experiments").exists()


if __name__ == "__main__":
    test_precomputed_date_plan_and_datastore_capacity()
    test_tsrag_upstream_search_rule()
    test_compact_extraction_and_independent_fitting_grid()
    test_rolling_ridge_and_bayes_outputs()
    test_compute_timing_contract()
    test_per_query_ridge_hyperparameter_selection()
    test_fixed_training_set_solves_one_ridge_per_user()
    test_adaptor_parameter_helpers()
    test_report_requires_identical_dates()
    test_profile_contract()
