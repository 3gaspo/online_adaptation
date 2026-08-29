# Fully online retrieval adaptation

This project asks whether a frozen time-series foundation model can be improved
online using only labels available before each query. It compares rolling
ridge adaptation, causal gates, lightweight controls, and a source-adapted
TS-RAG reference on predeclared evaluation dates.

Datastore membership, retrieval, fitting dates, model selection, and timing are
causal at every query. The active project supersedes the archived fixed-split
Adaptation implementation and does not reuse its extraction artifacts.

## Documentation map

| Need | Document |
|---|---|
| Paper-ready causal formulation | [`latex/method_overview.pdf`](latex/method_overview.pdf) |
| Proposal, cache, adapter, and reporting flow | [`docs/architecture.md`](docs/architecture.md) |
| Main studies, deadline jobs, and every ablation | [`docs/experiment_catalog.md`](docs/experiment_catalog.md) |
| Finalized evidence and next required runs | [`docs/results_recap.md`](docs/results_recap.md) |
| Complete reproducibility specification | [`latex/experiment_guideline.pdf`](latex/experiment_guideline.pdf) |
| Full analyzed evidence record | [`latex/executive_summary.pdf`](latex/executive_summary.pdf) |

## Setup

Use the repository's prepared environment and run from the project root:

```bash
uv sync
export PYTHONPATH=.
```

Place wide CSV datasets under `datasets/`, foundation checkpoints under
`weights/`, and the one-line lowercaseable NNI at
`$HOME/codes/.secrets/nni` on both clusters. Full profiles also discover
eligible prepared TIME panels through `datasets/time/catalog.json`.
Every profile uses 10 fitting dates and caps each datastore at 10,000 total
cross-user windows. CSV gaps are zero-filled after aggregation by default;
set `missing_values=error` to reject them instead.

## Main executions

The primary causal study and matched external comparison are:

```bash
EXPERIMENT_MODE=test sbatch slurm/dgx/main/01_main_online_ridge.slurm
EXPERIMENT_MODE=full sbatch slurm/dgx/main/01_main_online_ridge.slurm
EXPERIMENT_MODE=full sbatch slurm/dgx/main/02_tsrag_comparison.slurm
```

The default `STAGES=extract,adapt,tables` performs reusable extraction,
adaptation/evaluation, then identical-date reporting. A stage subset is for
recovery. Standard DGX fronts have matching fronts under `slurm/selena/` with
the same science, for example:

```bash
EXPERIMENT_MODE=test sbatch slurm/selena/main/01_main_online_ridge_selena.slurm
```

Four deadline fronts are intentionally separate from the standard
cadence-aware profiles; the fixed-protocol fronts use only the short range:

```bash
sbatch slurm/selena/deadline/fixed_online_per_user_selena.slurm
sbatch slurm/selena/deadline/fixed_fixed_shared_selena.slurm
sbatch slurm/selena/deadline/fixed_ablation_remainder_selena.slurm
sbatch slurm/selena/deadline/tsrag_priority_t3_selena.slurm
```

The fixed remainder should use an `afterok` dependency on both fixed anchor
jobs. The priority TS-RAG front covers Electricity, Solar, all four ETT panels,
and three preselected TIME panels; broader TIME coverage remains in the main
full profile. Its table-only recovery temporarily selects the first
dependency-signature-sorted vanilla source after requiring identical dates and
aggregation support, and records the selected signature and cross-source drift
in the report manifest; every other report remains strict.

The [experiment catalog](docs/experiment_catalog.md) states what each main,
deadline, and ablation front varies. Exact causal sets, feature definitions,
selection rules, metrics, and profile axes remain in the experiment guideline.

## Outputs and cluster operations

- Reusable extraction: `outputs/extractions/`.
- Adaptation families: `outputs/online_adaptation/<family>/`.
- Publishable aggregates: `outputs/reports/<family>/<mode>/`.
- Per-run diagnostics and plots: the matching detailed output trees.
- Runtime streams: `logs/`; Selena streams and artifacts live under the
  project-specific scratch `logs_selena/` and `outputs_selena/` roots.

Preview and then mirror maintained code from DGX:

```bash
bash sync_code_to_selena.sh --dry-run
bash sync_code_to_selena.sh
```

`*deleting` identifies stale maintained code. Delayed deletion never touches
the excluded environment, dependency manifests, datasets, weights, outputs,
or logs.

Pull Selena artifacts from DGX with the smallest required tier:

```bash
bash sync_results_to_dgx.sh
bash sync_results_to_dgx.sh --size detailed
bash sync_results_to_dgx.sh --size full
```

The default retrieves logs and aggregate reports; `detailed` adds non-binary
run diagnostics and `full` adds recovery payloads. Use
`bash publish_job.sh <job-id>` for one terminal log pair or
`bash publish_job.sh` for complete logs plus aggregate reports.

## Documentation maintenance

```bash
PYTHONPATH=src python -m scripts.build_docs
PYTHONPATH=src python -m scripts.build_docs --render method
PYTHONPATH=src python -m scripts.build_docs --render all
```

The default validates the audience map and complete DGX-front coverage.
Formulation belongs in the method note, implementation mechanics in the
architecture page, planned comparisons in the catalog, and analyzed evidence
in the recap and executive summary.
