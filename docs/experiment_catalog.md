# Experiment catalog

This page maps submission fronts to scientific questions. Exact definitions,
metrics, and artifact rules remain in the
[`experiment guideline`](../latex/experiment_guideline.pdf).

## Standard profiles

| Mode | Scope |
|---|---|
| `test` | Electricity long range with accelerated smoke controls |
| `small` | Electricity, Solar, Traffic at short/mid/long ranges |
| `full` | Small plus ETT panels and causally feasible TIME datasets |

Hourly, daily, and 15-minute datasets use their own short/mid/long range maps.
External-comparison fronts fixed to `512:64` are intentionally outside that
map.

## Main and deadline fronts

| Front | Scientific question |
|---|---|
| `slurm/dgx/main/01_main_online_ridge.slurm` | Does rolling full ridge or a causal gate improve on vanilla Chronos-2? |
| `slurm/dgx/main/02_tsrag_comparison.slurm` | How does online full ridge compare with released TS-RAG on identical dates at `512:64`? |
| `slurm/dgx/deadline/fixed_ablation_30_50_20.slurm` | What is the effect of fixed versus rolling datastore and fitting sets on one fixed T3 grid? |
| `slurm/dgx/deadline/tsrag_time_t3.slurm` | How do ridge and TS-RAG compare on the selected TIME panels and common T3 dates? |

## Ablation fronts

| Front | Varied factor |
|---|---|
| `slurm/dgx/ablations/ablation_n_datastore_dates.slurm` | datastore capacity |
| `slurm/dgx/ablations/ablation_n_fit.slurm` | fitting dates per user |
| `slurm/dgx/ablations/ablation_fit_stride.slurm` | every-date versus cadence-aligned fitting |
| `slurm/dgx/ablations/ablation_alpha.slurm` | fixed ridge regularization |
| `slurm/dgx/ablations/ablation_k.slurm` | explicit neighbor count |
| `slurm/dgx/ablations/ablation_l.slurm` | lookback length |
| `slurm/dgx/ablations/ablation_h.slurm` | forecast horizon |
| `slurm/dgx/ablations/ablation_feature_design.slurm` | ridge feature set |
| `slurm/dgx/ablations/ablation_formulation.slurm` | ordinary, delta, and simplex formulations |
| `slurm/dgx/ablations/ablation_fixed_protocol.slurm` | fixed datastore and fixed fitting-set booleans |
| `slurm/dgx/ablations/ablation_general_scope.slurm` | retrieval scope x fitting scope |
| `slurm/dgx/ablations/ablation_homogeneous.slurm` | homogeneous-channel versus all-variate data |
| `slurm/dgx/ablations/ablation_sota_chronos_bolt.slurm` | causal Chronos-Bolt comparison with published contextual values |
| `slurm/dgx/ablations/ablation_backbones.slurm` | Chronos-2, Chronos-Bolt, Chronos-T5, TS-ICL compatibility |

Standard DGX fronts have matching Selena fronts with identical science.
Reports require identical evaluated dates within each comparison.
