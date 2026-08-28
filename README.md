# Fully online retrieval adaptation

This project evaluates retrieval and adaptor fitting on a date grid fixed before
extraction. At every evaluated query date, both the retrieval store and the
adaptor use only labeled windows already observable at that date. The primary
workflow compares rolling full shared ridge with matched lightweight baselines;
a separate `512:64` workflow compares it with released TS-RAG inference.

The project supersedes the read-only fixed-split project at
`../../archive/adaptation/`. It contains no architecture-tuning track and does
not read extraction artifacts produced by that archived project.

## Causal protocol

For query date `s`, lookback `L`, and horizon `H`,

```text
X_s = {z_(s-L+1), ..., z_s}
Y_s = {z_(s+1), ..., z_(s+H)}
```

A datastore window dated `r` is eligible only when `r + H <= s`; its
complete target is therefore known before prediction at `s`.
`N_datastore_dates` is either a positive integer number of retained window
origins or a ratio in `(0,1]` of the complete origins on the configured
datastore-stride grid. Retrieval scope then
determines whether each origin contributes one same-user candidate, all users,
or all other users. A rolling store does not start early with a partial history:
the first fitting window has the full configured date capacity. A fixed store
uses one immutable T0 interval for every fitting and evaluation query, with the
same interval subset by timestamp-period alignment when enabled. Retrieval
defaults to raw Euclidean distance and all users. Generic extraction caches
retain the nearest `max_k=20` candidates.
The adaptor either validates prefixes from `candidate_k_grid={1,5,10,15}` or
uses one explicitly requested `used_k` prefix.

All T0, T1+T2, and T3 indexes are resolved before extraction. Every fitting
window is admitted only after its own horizon is observed.
`N_fit` always counts dates per user. With the default
`fitting_scope=same_user`, each query user gets a separate fit over its latest
`N_fit` dates. With `fitting_scope=all`, one shared fit is made per query date
from all users over those same `N_fit` dates (`N_fit * U` rows).
`retrieval_scope` independently controls which neighbors are attached to each
window. Fitting dates are selected independently from the complete causal date
grid with `fit_stride`; they do not have to belong to the strided datastore.
`fixed_training_set=false` selects the latest complete grid separately for each
query. `fixed_training_set=true` fits once on the precomputed T1+T2 grid and
reuses that ridge for all T3 queries. `eval_start_date` and `eval_end_date`
accept absolute indexes or timeline ratios, and `query_stride` fixes the T3
grid. The natural T3 start reserves a complete T0 followed by all `N_fit`
dates, so all four fixed-protocol arms evaluate the identical T3 dates.
At every query the ridge uses the oldest 80% of its causal fitting
window to train candidates and the newest 20% to validate `(alpha, K)`, then
refits the selected pair on the complete window before forecasting.

Defaults are:

```text
N_datastore_dates = 100 retained window-origin dates
N_fit   = 100 dates per user
fit stride = one dataset period (24 hourly, 7 daily, 96 at 15 minutes)
fitting scope    = same_user
validation ratio = 0.2
max K extracted  = 20
candidate K grid = {1, 5, 10, 15}
used K           = unset (selected by validation)
alpha grid       = {1e-1, 1e-2, 1e-3}
method  = full_ridge_shared
```

Ridge sufficient statistics are accumulated in float64. Each feature is
scaled by its fitting-set root mean square before applying `alpha`; this
prevents signal units from determining regularization strength. Selection uses
the configured fitting loss (MSE by default), breaks exact ties toward smaller
`K` and then stronger regularization, and is repeated causally at every query.
Supplying `used_k` removes only K from validation. Consequently, the alpha
ablation fixes alpha while selecting K, the K ablation fixes `used_k` while
selecting alpha, and every other ridge ablation selects both.

## Reusable extraction

Extraction runs live below `outputs/online_extraction/` and are independently
manifested. Their identity includes dataset, `L:H`, backbone, retrieval space
  and metric, `max_k`, retrieval scope, `N_datastore_dates`, `N_fit`,
  `fit_stride`, fixed-datastore/fixed-training booleans, and the
  homogeneous-channel choice. `N_fit` is part
of extraction coverage because fitting windows need their own causal neighbor
rows. Compatible generic runs share one `max_k=20` cache, including the six
strict K-ablation runs at `{1,3,5,10,15,20}`. Validation candidates and an
explicit `used_k` consume nearest-`K` prefixes. Changes to fitting scope,
validation ratio or grids, linear
  design or formulation reuse a matching cache.

Each past fitting window reuses the neighbors extracted causally at that
window's own date. Before extraction, the workflow marks the unique union of
evaluation queries and their fitting windows, then fills each required
retrieval row exactly once. Because one extraction directory already identifies
the dataset, `L:H`, backbone, representation, strides, and scopes, these values
are not duplicated in table rows. A window ID is
`date_position * number_of_users + user_position`.

| Logical table | Key | Stored values |
|---|---|---|
| Window index | Window date and user position | Valid integer dates, timestamps, and user names. Lookbacks and horizons are not stored. |
| Window statistics | Window ID | Lookback mean, population standard deviation, and constant-window indicator for every valid window. |
| Retrieval representation | Window date and user | Expensive encoder, Fourier, min-max, or TS-RAG representation, only for dates used as retrieval rows or datastore candidates. Raw and instance retrieval omit this table. |
| Vanilla forecast | Window ID | One `H`-step backbone forecast for each required retrieval window and selected neighbor. Unselected datastore candidates are not forecast. |
| Retrieval neighbors | Retrieval-window date, query user, rank | Ranked neighbor window ID, configured retrieval distance, raw lookback RMS distance, instance-normalized lookback RMS distance, and the datastore candidate count. A per-date flag identifies evaluation queries. |
| Context forecast | Retrieval-window date and `K` | One all-user `U x H` contextual forecast, created only when a design requests that `K`, plus its elapsed and amortized time. |
| Ridge selection and trajectory | Query date, and user only for same-user fitting | Selected alpha, selected `K`, validation loss, adaptor parameter count, and the ordered coefficient trajectory. All-user fitting writes one selection row per date and does not use an `all` user sentinel. |
| Metrics | Query date, user, and method | Per-user/date errors, matching vanilla errors, deltas, and strict win; per-date aggregates; final global, per-user-dispersion, worst-10%-user, and improvement aggregates. |
| Compute timing | Method/run | Complete extraction, adaptation, total, average per date-user sample, and the cold all-user batch with component times. |

Lookbacks and horizons are sliced from the source view by window ID. Sparse
forecasts are joined by window ID instead of duplicated in retrieval rows.
Raw and instance-normalized retrieval have no representation table; instance
normalization uses the stored statistics on demand. Context is built lazily
for the requested candidate prefix and stored only for computed `(window,K)`
pairs. Fitting-date sets are deterministic functions of query date, `N_fit`,
and `fit_stride`; they are regenerated during adaptation rather than stored.

Extraction computes the unique required retrieval-window dates first. For each
such window `r`, it obtains `D_r` with `store_stride`, fills missing expensive
representations, searches and stores the nearest `K_max`, and marks only `r`
and its selected neighbors for vanilla forecasting. Forecasts are then
computed once per unique window ID. Adaptation independently reconstructs
`F_q` with `fit_stride`, updates rolling sufficient statistics, selects and
refits ridge, and builds only requested `K`-conditioned contexts.

Periodic alignment is enabled by default. Hourly, daily, and 15-minute panels
use periods and store strides of 24, 7, and 96 samples respectively. An
explicit stride must be a multiple of the period. Candidate dates satisfy
`query_date - window_date` being a period multiple and are separated by the
configured store stride.
The default fitting stride equals the same cadence period, while the
`fit_stride=1` ablation uses every causally available date independently of
datastore membership.

Every extraction also writes two diagnostic layers. Setting diagnostics are
specific to `(dataset, L:H)`: they sample complete `L+H` windows, compare the
first and last 10% of valid window dates for inter-date shift, and compare
randomly sampled user pairs at aligned window dates for inter-user shift. Neighbor
diagnostics are specific to the complete extraction configuration and report
same-user versus other-user selections, neighbor age in dataset time steps,
and retrieval-window-to-neighbor lookback distance over all retrieved samples and as
per-query-user averages. Both raw
and instance-normalized RMS distances are retained. Instance normalization
always standardizes each complete setting window with that window's own
`L`-step lookback statistics; neighbor lookback distances analogously
standardize each lookback independently.

The setting layer writes `setting_diagnostics.csv`, its sampled distances and
sampling record, and `setting_diagnostics.png`. The neighbor layer writes
`neighbor_diagnostics.csv`, `neighbor_diagnostics_per_user.csv`, and separate
all-sample and per-user PNG dashboards. Exact neighbor counts, means, and
standard deviations use the complete extraction; plotted distributions and
quantiles use a bounded deterministic sample. The extraction arrays retain
compact raw and instance-normalized neighbor lookback distances, not the raw
lookbacks themselves.

Adaptation runs live below `outputs/online_adaptation/<family>/`. Ridge writes
per-user/date, per-date, and aggregate metrics, every per-query selected alpha/K
and validation loss, its complete fitted-coefficient trajectory, a coefficient
mean/standard-deviation/importance CSV, and a PNG summary.
CatBoost writes the analogous rolling feature-importance trajectory, mean and
standard deviation. Every adaptation result manifest records an adaptor-only
parameter count: fitted coefficients for linear methods, the two fitted Bayes
moments, final CatBoost leaf values, or TS-RAG ARM tensors. Frozen forecasting
backbones and the retrieval encoder are excluded. Raw adapted predictions are
not retained. Aggregate metrics include overall MSE/nMSE/MAE/nMAE, strict
sample-level win rate, per-user population dispersion, the mean MSE and nMSE
of the independently selected worst 10% of users, and the percentage of users
whose mean MSE or nMSE improves over vanilla.

Every method, including TS-RAG and the other external-backbone paths, uses the
same compute-time writer. It reports complete extraction time, complete
adaptation/evaluation time, their sum, and the average seconds per evaluated
date-user sample. A single cold all-user batch starts from empty computation
and context caches, includes source-view/store reconstruction, and reports
metadata/statistics initialization, retrieval/representation,
vanilla-forecast, adaptation, and combined worst-case time.

## Methods and reports

The ridge feature designs are `cov`, `avgy`, `y`, `cov_y`, `cov_avgy`,
`residual`, and `full`. Each can use shared or horizon-specific coefficients;
the formulation ablation compares ordinary ridge, delta ridge, and
simplex-constrained convex fitting.

Retrieved neighbors are passed to capable backbones through the canonical
model's project wrapper. `retrieval_covariate_mode=past_and_future` supplies
both the rescaled neighbor lookbacks and their known horizons, `past` supplies
only lookback values, and `none` disables covariates without changing the model
alias. Any context-bearing method fails explicitly when the selected backbone
does not support covariates. The four-backbone compatibility profile therefore
uses the covariate-free `y_ridge_shared` design; the primary Chronos-2 ridge and
gate profiles exercise retrieved covariates.

`01_main_online_ridge.slurm` evaluates full shared ridge, the matched
simplex-constrained mixture of vanilla plus retrieved outcomes, the causal
shared soft Bayes covariate gate, the parameter-free covariate-informed
prediction, and vanilla Chronos-2. All use the cadence-specific adaptive `L:H`
grid. The Bayes candidate weight is the sigmoid of the fitted advantage divided
by its current fitting-set dispersion.

`02_tsrag_comparison.slurm` is restricted to `512:64`. It compares online full
ridge with released TS-RAG on identical evaluation dates. Native TS-RAG uses
Chronos-Bolt, Chronos-T5 EOS retrieval features, a fixed training-split
datastore, same-variable retrieval, unstrided origins, and `K=5`.

Every table stage requires identical evaluated query dates across methods for
each dataset and `L:H`; a mismatch fails instead of silently intersecting dates.
Overall averages use only the common dataset and `L:H` configurations shared by
every reported model. It writes:

- `detailed_results.{csv,tex}` with dataset, setting, model, MSE, nMSE, MAE,
  nMAE, relative nMSE/MSE improvement, win rate, and common-date count;
- `average_results.{csv,tex}` with one model per row and an equal-weight mean
  over dataset-setting configurations.

The Chronos-Bolt SOTA ablation also writes a side-by-side published MSE table
from `PUBLISHED_BASELINES.json`. Its note explicitly separates the causal
no-split raw-scale protocol from the papers' official-test, train-standardized
protocol; the values are not presented as paired estimates.

## Profiles

`EXPERIMENT_MODE` selects:

- `test`: Electricity at its cadence-specific `long` setting only
  (`504:168`); the Slurm smoke defaults are `N_datastore_dates=100`, `N_fit=100`,
  query stride 257, and two CatBoost trees;
- `small`: Electricity, Solar, and Traffic at their cadence-specific `short`,
  `mid`, and `long` settings;
- `full`: `small` plus raw ETTh1/ETTh2/ETTm1/ETTm2 and every causally feasible
  prepared dataset listed in `datasets/time/catalog.json`, at all three
  cadence-specific ranges. Infeasible dataset-setting pairs are logged from
  catalog sizes before extraction.

The range map is shared by every standard family: hourly data use `168:24`,
`336:48`, and `504:168`; daily data use `7:1`, `14:2`, and `30:7`; and
15-minute data use `96:4`, `192:8`, and `672:96`. Solar and Weather are
hourly after their configured aggregation. Prepared TIME datasets resolve
their cadence from `configured_frequency` in the catalog.

Explicit setting studies remain outside that standard range grid by design:
the `L` and `H` sensitivity fronts vary one axis, while TS-RAG, the
Chronos-Bolt SOTA comparison, and the cross-backbone compatibility study keep
the external checkpoint contract `512:64`. These fixed/custom configurations
are not labeled as short, mid, or long ranges.

All-user retrieval and same-user fitting are the adaptive-workflow defaults.
Both fixed-protocol booleans default to false. TS-RAG's separate native profile
uses the fixed same-user training datastore described above.

## Slurm entry points

Submit from this project root. Fronts default to the complete ordered
`STAGES=extract,adapt,tables` workflow and support recovery by overriding that
comma-separated value.

```bash
sbatch slurm/dgx/main/01_main_online_ridge.slurm
EXPERIMENT_MODE=small sbatch slurm/dgx/main/01_main_online_ridge.slurm
EXPERIMENT_MODE=full sbatch slurm/dgx/main/02_tsrag_comparison.slurm
```

Every standard DGX front has a matching `_selena.slurm` overflow front. The two
explicitly temporary deadline fronts exist only on DGX. For example:

```bash
sbatch slurm/selena/main/01_main_online_ridge_selena.slurm
EXPERIMENT_MODE=full sbatch slurm/selena/main/02_tsrag_comparison_selena.slurm
```

### Temporary deadline fronts

The deadline-only fixed ablation evaluates Electricity/Solar crossed with
short/long range and the four fixed-datastore/fixed-training-set combinations
in one job. It fixes the
source timeline before extraction to T0/T1+T2/T3 = 30%/50%/20%, uses a
chronological 80%/20% train/validation split inside the T1+T2 fitting dates,
uses fit stride 24, and evaluates every 127th valid T3 query date. Every
datastore retains at most 20,000 globally comparable windows. A fixed
datastore draws the most recent periodically aligned windows from T0; a rolling
datastore draws the most recent periodically aligned windows available to its
query. All four arms therefore have the same T3 date grid.

The deadline-only TS-RAG comparison is one job fixed to `512:64`. The verified
TIME panels are `time/ne_china_wind_h`,
`time/coastal_t_s_h_part11`, and `time/sg_weather_d`. In each shard, online
ridge and TS-RAG use the same T3 grid beginning at the 80% boundary, with query
stride 127. Ridge uses a rolling all-user datastore capped
at 20,000 windows, 30 fitting dates at fit stride 24, and selects both K from
`[1,5,10,15]` and alpha on the fitting split. TS-RAG uses the same global
20,000-window maximum, drawn from its fixed same-user T0 datastore with
unstrided origins, and fixed K=5.

The two complete jobs are independent and may be submitted concurrently:

```bash
sbatch slurm/dgx/deadline/fixed_ablation_30_50_20.slurm
sbatch slurm/dgx/deadline/tsrag_time_t3.slurm
```

The Selena variants keep the same family, profile, stages, and scientific
arguments while using partition `an`, QoS `an_preemptable`, an exclusive
allocation without disabling cluster requeue, WCKey `P12CU:DATASCIENCE`, distinct job names, and
`selena_`-prefixed launch IDs. They write Slurm streams and artifacts under
`/scratch/users/<lowercase-nni>/codes/online_adaptation/logs_selena/` and
`outputs_selena/`; DGX fronts continue to use project-local `logs/` and
`outputs/`. The shared `LOGS_ROOT` and `OUTPUTS_ROOT` variables remain
available for explicit runtime overrides; custom Slurm stream paths require
the corresponding `sbatch --output` and `--error` overrides.

The remaining DGX fronts are focused studies under
`slurm/dgx/ablations/`; their Selena counterparts are under
`slurm/selena/ablations/`:

- `ablation_{n_datastore_dates,n_fit,fit_stride,alpha,k,l,h}.slurm`;
- `ablation_feature_design.slurm` and `ablation_formulation.slurm`;
- `ablation_fixed_protocol.slurm`;
- `ablation_general_scope.slurm`, crossing retrieval scope (`all`,
  `same_user`, `other_users`) with fitting scope (`all`, `same_user`);
- `ablation_homogeneous.slurm`;
- `ablation_sota_chronos_bolt.slurm`;
- `ablation_backbones.slurm` (Chronos-2, Chronos-Bolt, Chronos-T5, and TS-ICL).

The publication `N_fit` sweep is `{50,100,500,1000}`. The fitting-stride
front holds `N_fit=100` and compares every causally available date
(`fit_stride=1`) with cadence-aligned fitting (`fit_stride=period`).

The homogeneous front has its own mixed-quantity dataset grid. `test` runs
Weather at its long range; `small` and `full` run ETTh1, ETTh2, ETTm1, ETTm2,
and Weather at all three cadence-specific ranges. The project-specific channel
lists are in `src/conf/homogeneous_channels.yaml`; dataset-local `config.json`
files remain responsible for general loading and preprocessing. Raw ETT
panels retain the six load channels and exclude `OT` in homogeneous-only runs;
Weather retains its four temperature channels (`T`, `Tpot`, `Tdew`, and
`Tlog`). The all-variate arm retains every non-date channel.

`DATA_ROOT`, `WEIGHTS_ROOT`, `DEVICE`, `SEED`, `PURPOSE`,
`N_DATASTORE_DATES`, `N_FIT`, `FIXED_DATASTORE`, `FIXED_TRAINING_SET`,
`EVAL_START_DATE`, `EVAL_END_DATE`, `QUERY_STRIDE`, `STORE_STRIDE`,
`ALIGN_PERIOD`, `FITTING_SCOPE`, `FIT_STRIDE`, `ALPHA`, `MAX_K`, `CANDIDATE_K_GRID`,
`USED_K`, `TUNE_ALPHA`, `RIDGE_VALIDATION_RATIO`, `RIDGE_ALPHA_GRID`,
`TSRAG_K`, `RETRIEVAL_COVARIATE_MODE`, `STAGES`,
`RUN_CONFLICT_POLICY`, and `SKIP_COMPLETE` are launcher overrides. Expected
weight locations are `chronos2/`, `chronos-bolt-base/`,
`chronos-t5-base/`, `ts-rag/`, and `tsicl/tsicl-v1.ckpt` below the resolved
weights root.
`QUERY_STRIDE` and `CATBOOST_ITERATIONS` may override the smoke controls;
`small` and `full` default to query stride 127.

Without explicit roots, launchers search nonempty `datasets/` and `weights/`
directories in the same order as the archived Adaptation workflow: project
local, immediate project parent, then the nested-workspace shared parent.

Reports use `TABLE_CONFIG_POLICY=distinct|latest|average` and
`TABLE_REPEAT_POLICY=selected|latest|distinct|average`, with the robust
defaults shown first. `TABLE_PIPELINE_CONFIGS` accepts whitespace-separated
`KEY=VALUE` filters, including dotted fields inside embedded upstream
dependencies, and `TABLE_PURPOSE` overrides the profile purpose filter.
Distinct scientific configurations receive labels for every differing nested
pipeline, dependency, or experiment field; exact repeats remain controlled by
`SELECTED_RUNS.txt`. Each report manifest records the policies, filters, and
exact input manifests selected.

## Code organization

Scientific, external-model, experiment-infrastructure, result, and plotting
responsibilities have distinct package owners:

- `src/proposal/`: causal datastore rules, normalized extraction,
  K-conditioned context caching, ridge, and gate computations, with no
  manifest, reporting, or plotting dependencies;
- `src/external_models/`: thin pinned-package adapters for frozen third-party
  backbones and a dedicated `tsrag/` package containing only the source-adapted
  ARM and upstream-faithful retriever;
- `src/model_loading/`: common normalization, construction, and checkpoint
  loading;
- `src/data/`: CSV loading, window construction, scale conversion, and nearest
  neighbor primitives;
- `src/scripts/`: explicit Hydra entry points for `extract`, `adapt`, and
  `tables`; `src/slurm/run_family.sh` invokes them sequentially according to
  `STAGES`;
- `src/pipeline/`: experiment matrices, artifact schemas, manifested run
  utilities, extraction/adaptation/TS-RAG evaluation, artifact writing, and
  orchestration;
- `src/results/`: diagnostics, common experiment/cold-batch compute timing,
  metrics, and tables;
- `src/visualization/`: plotting only;
- `src/conf/`: Hydra and project-specific experiment configuration.

The TS-RAG ARM follows the released MoE computation from
`UConn-DSIS/TS-RAG` at commit
`73ac807789d2e61b8a3dfc8514e3fc947fe185cc`, but is not a verbatim copy:
unrelated training/data paths and alternative augmentation modes are removed,
and the released MoE path is packaged over the local Chronos-Bolt base with
strict checkpoint loading. Its retriever preserves upstream Chronos-T5-base
bfloat16 embedding, final-token representation, float32
`faiss.IndexFlatL2` squared-L2 search, and `top_k + 1` search followed by
removal of the final result. The native profile preserves the released
`only_self_train` scope: each variable retrieves only from its own immutable
training interval. The retrieved
lookback--future sequences are passed to ARM on their original raw scale;
evaluation dates, metrics, and artifact writing are project-specific.

The `chronos2`, `chronos_bolt`, `chronos_t5`, and `ts_icl` keys are
the only foundation-model aliases. They use
the same byte-identical thin adapters and TIME input contract as TSFM
evaluation and TimeTensors. Chronos-2, Chronos-Bolt, and Chronos-T5 use the official
`chronos-forecasting==2.0.1` pipelines. The project-specific retrieval wrapper
translates context into past and optional future covariates on top of the
canonical adapter, so it does not introduce another model name. TS-ICL uses
a thin adapter over `tsicl>=0.2.1`; its architecture and native inference path
are not copied or reimplemented. The former TabPFN adapter source remains
unregistered without its dependency, and the retired TiREx-2 adapter is archived.

## Dataset configuration and manifests

The CSV loader automatically reads `config.json` beside a selected CSV.
Portable top-level values are overridden by an `online_adaptation` object, and
explicit run settings override both. For `drop_users`, null inherits, `[]`
keeps every CSV user, and a nonempty value replaces the dataset default. The
selected path and applied keys are logged.

The same `python -m src.scripts.prepare_time_csv` command used by the sibling
benchmarks prepares filtered TIME panels and `datasets/time/catalog.json`.
Full profiles discover those catalog entries automatically and select
cadence-specific short, mid, and long settings.

Run reuse is determined only by schema version, path identity, declared model,
pipeline and experiment parameters, and the run seed. Source files, datasets,
weights, logs, and output contents are not fingerprinted. Code or data changes
therefore require a deliberate rerun decision; use
`RUN_CONFLICT_POLICY=new` for another repeat of unchanged parameters.
The run-manifest, normalized extraction, and online-adaptation artifacts all
use their sole current version-1 contracts. No older experiment artifact
exists to preserve or migrate.

## Local check

Full extraction and backbone inference are cluster work. The focused local
Slurm contract and synthetic regression checks are:

```bash
PYTHONPATH=. python src/tests/test_slurm_workflow.py
PYTHONPATH=. python -m pytest src/tests/test_online_core.py src/tests/test_online_diagnostics.py src/tests/test_reporting_selection.py -q
```

It covers precomputed full-capacity datastore and fitting boundaries,
independent store/fitting strides, compact source-window joins, common compute timing, fixed/online ridge,
rolling Bayes, date-based cross-user fitting, per-user and W10 metrics,
coefficient artifacts, identical-date reporting, `L+H` setting-shift
sampling, neighbor-date summaries, and raw/instance-normalized distance plots.

## Synchronizing DGX and Selena

Keep `$HOME/codes/.secrets/nni` outside the project on both clusters. It
contains the NNI on one line; synchronization and Selena runtime helpers strip
whitespace and lowercase it for SSH accounts and scratch paths.

```bash
mkdir -p "$HOME/codes/.secrets"
printf '%s\n' 'YOUR_NNI' > "$HOME/codes/.secrets/nni"
chmod 600 "$HOME/codes/.secrets/nni"
```

After updating the DGX checkout, mirror its code to Selena with:

```bash
bash sync_code_to_selena.sh
```

The transfer derives the project directory name from the checkout, makes
Selena's code match DGX, and creates the scratch `outputs_selena/` and
`logs_selena/` directories. It preserves `.venv`, `.secrets`,
`pyproject.toml`, `uv.lock`, datasets, weights, and DGX-named outputs/logs in
the Selena checkout. Git metadata and dependency manifests are never
transferred.

After Selena jobs finish, run the result helper from the project checkout on
DGX. DGX initiates the SSH connection and pulls the lightweight artifacts, so
Selena needs no outbound SSH/SCP access:

```bash
bash sync_results_to_dgx.sh
bash sync_results_to_dgx.sh --size detailed
bash sync_results_to_dgx.sh --size full
```

The default lightweight tier transfers logs, manifests, aggregate metrics,
reports, and ordinary plots while omitting row-level window/user-date/sample
tables and per-run diagnostic plots. `detailed` adds those text and plotting
artifacts; `full` also retrieves cluster-only `.pt`, `.npy`, `.cbm`, and other
binary payloads for recovery or deep debugging. `--job-id ID` restricts the
logs to the exact standard job pair. Transfers never delete DGX files. Do not
run this helper on Selena; returned artifacts remain in the `_selena` trees.

## Maintenance

`PENDING_UPDATES.md` records focused checks, deferred cluster checks, and rerun
scope. `CLUSTER_STATUS.txt` records the latest submitted or analyzed workflow.
Brief daily triage compares stored fingerprints and updates those records only
for new source, artifact, or cluster state; unchanged blockers are carried
forward. Broad weekly maintenance verifies changed entries against the
implementation, runs complementary lightweight integration checks, reconciles
the README and LaTeX documents, and renders affected PDFs before resolving
entries.
After a terminal cluster job, `publish_job.sh <job-id>` is the manual artifact
publishing path; Slurm workflows never run Git commands. Running
`bash publish_job.sh` without a job ID uses the lightweight tier for `logs/`,
`outputs/`, and paired Selena trees; `--size detailed` adds the row-level and
per-run diagnostics omitted by default. Both tiers exclude `*.pt`, `*.npy`,
and `*.cbm`; those cluster-only payloads are available only through full sync.
A partial Selena namespace fails closed; numeric job-ID mode still selects
only the exact standard log pair.

Before staging, each selected non-excluded file larger than 100,000,000 bytes
is replaced for publication by `<original>.sample.txt`. Text samples contain
source metadata and the first 10% of content, capped at 10,000,000 bytes;
binary samples contain metadata only. The header retains the first UTC time
when the associated file became stale on Git because of its size. The original
is excluded literally from
both staging and commit selection. `PUBLISH_MAX_FILE_BYTES` and
`PUBLISH_SAMPLE_MAX_BYTES` override the positive byte limits, with the sample
limit required to remain smaller.
