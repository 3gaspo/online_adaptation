# Pending updates

## 2026-08-26 — Cluster launcher hierarchy

- Scope and behavior: moved the 16 DGX and 16 Selena submission fronts from
  the project root into `slurm/<cluster>/main/` and
  `slurm/<cluster>/ablations/` without changing their resources, families,
  artifact roots, or project-root resolution contract.
- Affected contracts: launcher locations and the focused architecture checks;
  submissions must still run from the project root so `SLURM_SUBMIT_DIR`
  resolves the repository correctly.
- Focused checks: the direct Slurm workflow contract ran 2 tests successfully
  and verified the complete 16-pair hierarchy. The prepared runtime lacks
  pytest, so the separate pytest-only profile check was not run.
- Deferred integration and documentation: update README, LaTeX, and cluster
  handoff commands in the planned documentation pass, then run the recursive
  DGX-to-Selena code sync to apply the same hierarchy remotely. No scientific
  rerun or artifact migration is required.

## 2026-08-26 — DGX-initiated Selena return and publication

- Scope and behavior: `sync_results_to_dgx.sh` now runs on DGX and pulls the
  isolated Selena trees. Unscoped publication includes paired `logs_selena/`
  and lightweight `outputs_selena/` under the existing heavy-payload
  exclusions; numeric job-ID mode remains standard-log-only.
- Affected contracts: result helper, publisher, focused workflow regression,
  README, shared guidance, and cluster handoff.
- Focused checks: Bash syntax passed for all 15 maintained scripts, all five
  publisher checks and both Online Adaptation workflow checks passed, and the
  nine publisher copies plus five suffix-result helpers are each byte-identical.
- Deferred integration: exercise one real pull and unscoped publication after
  a Selena test job. The README changed; the guideline's all-log/lightweight-
  output wording remains accurate, so LaTeX/PDF files are unchanged. No
  scientific rerun or artifact migration is required.

## 2026-08-26 — Selena overflow fronts and isolated artifacts

- Scope and behavior: added a `_selena.slurm` counterpart for all 16 DGX
  fronts. Every pair retains the same family, mode, stages, and scientific
  arguments; Selena selects partition `an`, exclusive non-requeued execution,
  the project WCKey, a distinct job name, and a `selena_` launch ID. The shared
  runner exposes overridable `OUTPUTS_ROOT` and `LOGS_ROOT`; they default to
  `outputs/` and `logs/`, while Selena selects `outputs_selena/` and
  `logs_selena/`.
- Affected files and contracts: 16 new fronts, shared runner, sync pair,
  ignored placeholders, focused workflow regression, README/guidance, cluster
  handoff, experiment guideline, and the shared requirement now governing all
  runnable experiment projects. The return sync preserves the Selena directory
  names on DGX and never merges into standard artifacts.
- Focused checks completed: Git Bash syntax passed across all 52 affected
  TSFM/Online shell files; both Online Adaptation workflow tests passed and
  verified all 16 DGX/Selena pairs; two clean LaTeX passes produced six pages;
  all six rendered pages were visually inspected with no clipping or overlap.
- Deferred integration and documentation: submit one Selena test front and
  exercise both sync directions on the real clusters. Scientific identity and
  DGX behavior are unchanged; no existing artifact, migration, or rerun is
  affected.

## 2026-08-26 — Shared resources and four-model backbone launch

- Scope and behavior: retained Adaptation's project-local, immediate-parent,
  and nested-workspace dataset/weight lookup order, removed the unrelated extra
  candidate, and added immediate-parent checkpoint discovery to every aligned
  foundation adapter. The backbone ablation now launches Chronos-2,
  Chronos-Bolt, TS-ICL, and TabPFN-TS; TiRex-2 remains implemented and
  alias-valid but visibly commented out.
- Affected files and contracts: shared Slurm resource resolver, backbone task
  profile/front, five aligned adapters, focused tests, README, guidance, and
  experiment-guideline source. Scientific behavior of the four active models
  and artifact identity are unchanged.
- Focused checks completed: all seven dependency-free foundation workflow
  guards passed, the focused task-profile call returned exactly the four active
  aliases, Git Bash syntax passed for the runner and backbone front, and
  SHA-256 checks confirmed byte-identical copies of all five adapters across
  Online Adaptation, TimeTensors, and TSFM.
- Deferred integration and documentation: resolve a real dataset and each
  active checkpoint from the shared parent on the cluster and run the backbone
  test profile. No checkpoint was loaded locally. Maintenance rebuilt the
  reconciled guideline PDF and visually inspected all six pages.
  No completed current artifact is invalidated; future backbone runs exclude
  TiRex-2.

## 2026-08-24 — Concise mathematical experiment guideline

- Scope and behavior: replaced the implementation-heavy guideline with a
  six-page problem, mathematical definition, measures, run list, code-object
  map, and direct extraction/adaptation pseudocode. Every symbol and reported
  quantity is defined. The text now states the implemented
  `supports_context=False` fallback, under which `C=V` for every profiled
  backbone and the current covariance-gate front predicts vanilla exactly.
- Affected files and contracts: documentation only
  (`latex/experiment_guideline.tex` and its rendered PDF), plus shared
  experiments-level writing guidance. No code, configuration, schema, artifact,
  or experiment setting changed.
- Focused checks completed: reconciled the equations and run tables with
  `src/proposal/`, `src/pipeline/`, `src/external_models/`, Hydra defaults, and
  every active Slurm family; two final pdfLaTeX passes produced a six-page PDF;
  all six rendered pages were visually inspected with no clipping, overlap,
  overflow, empty trailing page, or unreadable text.
- Deferred checks: none for this documentation-only rewrite.
- Documentation/LaTeX: the experiment guideline now follows the shared concise,
  formula-first structure. The README and executive summary are unchanged; no
  new experimental evidence was produced.
- Required reruns: none.

## 2026-08-24 — Shared max-K extraction and independent K selection

- Scope and behavior: replaced the overloaded neighbor-count contract with
  extraction-owned `max_k`, validation-owned `candidate_k_grid`, and optional
  adapter-owned `used_k`. Compatible generic experiments now share one
  `max_k=20` extraction. Supplying `used_k` disables only K selection; alpha
  selection remains independent. The alpha ablation fixes alpha and validates
  K, the K ablation shares one extraction across strict prefixes and validates
  alpha, and every other ridge ablation validates both. TS-RAG now uses its own
  Slurm-configured K=5. Publication and test query strides are 127 and 257.
- Affected files and contracts: the unreleased extraction and adaptation
  schemas remain version 1 but now persist `max_k`, `candidate_k_grid`,
  `used_k`, and independent alpha/K-selection metadata. Proposal contracts,
  extraction, ridge/gate adaptation, workflow identities, profiles, Hydra and
  Slurm configuration, tests, local guidance, README, and the rendered
  experiment guideline use only the new contract.
- Focused checks completed: `src.tests.test_online_core`,
  `src.tests.test_online_diagnostics`, and
  `src.tests.test_reporting_selection` passed in the thesis runtime; AST parsing
  passed for all eight changed Python modules; `bash -n
  src/slurm/run_family.sh` passed. The guideline compiled successfully twice to
  a six-page PDF, and all six pages were visually inspected without clipping,
  overflow, or stale protocol text.
- Deferred checks: Hydra workflow import/configuration expansion was not run
  because neither `uv` nor a prepared project environment is available locally
  and the shared thesis runtime lacks OmegaConf. Real-backbone, GPU-timing, and
  TS-RAG execution remain cluster checks.
- Documentation/LaTeX: README and the experiment guideline now distinguish
  extraction capacity, candidate prefixes, and used K; document K-ablation
  sharing and independent validation; record TS-RAG K=5; and state query
  strides 127/257. The executive summary is unchanged because no evidence was
  produced.
- Required reruns: run full `ablation_k.slurm` first to materialize compatible
  K=20 caches, then rerun every affected ridge/gate family against the new
  identities. Rerun the TS-RAG comparison because its K changed from 10 to 5.

## 2026-08-24 — Explicit persisted-table contract and algorithmic pipeline

- Scope and behavior: clarified every persisted extraction, adaptation, metric,
  and timing table by stating its logical key, exact stored values, and the
  values deliberately reconstructed or computed lazily. Recast the complete
  causal extraction and adaptation workflow as LaTeX algorithmic pseudocode.
- Affected files and contracts: documentation only (`README.md` and
  `latex/experiment_guideline.tex` plus its rendered PDF); no implementation,
  configuration, schema, or artifact behavior changed.
- Focused checks completed: the guideline compiled successfully twice to a
  six-page PDF, and all six rendered pages were visually inspected without
  clipping, overflow, or misplaced algorithms.
- Deferred checks: none for this documentation clarification. The existing
  real-backbone, FAISS, GPU-timing, and TS-RAG cluster smoke remains deferred
  under the implementation entry below.
- Documentation/LaTeX: README and the experiment guideline now use the same
  table names and exclusions. The executive summary is unchanged because no
  new experimental evidence was produced.
- Required reruns: none; this change only documents the current version-1
  contract and pipeline more explicitly.

## 2026-08-24 — Independent fitting grid, compact source views, and unified compute timing

- Scope and behavior: made fitting dates an independent causal-grid selection
  with `fit_stride`, defaulted to `N_fit=100` and cadence-period alignment,
  planned the unique union of query and fitting retrieval windows before
  extraction, and added a focused stride ablation. Replaced serialized
  lookback/horizon tensors with zero-copy dataset views, retained shared
  lookback statistics, and compacted representations and vanilla forecasts to
  only the window IDs that need them.
- Affected files and contracts: the unreleased extraction, adaptation, and
  result schemas remain version 1. Hydra settings, profiles, workflow
  signatures, Slurm fronts, retrieval/adaptation consumers, TS-RAG, shared
  timing output, diagnostics terminology, tests, local guidance, README, and
  the experiment guideline use the new contract only.
- Focused checks completed: the three focused modules
  `src.tests.test_online_core`, `src.tests.test_online_diagnostics`, and
  `src.tests.test_reporting_selection` passed in the thesis runtime; all Python
  sources parsed and `pyproject.toml` loaded successfully; and Git Bash parsed
  the shared runner and every root Slurm front. The core smoke exercised a
  synthetic compact extraction, independent fitting dates, sparse forecast
  joins, and the common compute-timing writer. The guideline compiled twice to
  a four-page PDF and every rendered page was visually inspected without
  clipping or layout defects.
- Deferred checks: in the prepared cluster environment, run one tiny
  real-backbone extraction/adaptation job and one TS-RAG job to validate model
  dependencies, FAISS retrieval, GPU timing synchronization, and the shared
  timing artifact with real devices; then run the fitting-stride test profile.
- Documentation/LaTeX: README and the rendered experiment guideline define the
  independent grids, compact v1 caches, timing quantities, terminology, and
  the complete algorithmic pipeline. The executive summary is unchanged
  because no result evidence was produced.
- Required reruns: all future extraction and adaptation artifacts must use the
  revised version-1 identity and layout; no released or reusable earlier
  artifact exists to migrate.

## 2026-08-24 — Normalized online caches, date-based fitting, and complete batch metrics

- Scope and behavior: replaced dense per-query neighbor payloads with shared
  window, window-statistics, sparse window-computation, and query-neighbor
  tables; moved context forecasts to lazy `(query window, K)` adaptation
  caches; made `N_fit` count dates per user under both fitting scopes; restored
  cadence-aware periodic alignment without a fitting stride; and added
  per-user/date metrics, worst-10%-user MSE/nMSE, strict global win rate, and
  complete all-user batch latency including reconstruction and cache fills.
- Affected files and contracts: extraction and adaptation schemas are now
  version 1; extraction identity includes fitting-date coverage; Hydra/profile
  cadence settings, Slurm overrides, ridge/gate/TS-RAG consumers, reporting
  artifacts, local guidance, README, focused tests, and experiment guideline
  use the normalized contract only. The small shared context-cache class lives
  in the existing extraction owner rather than a standalone module.
- Focused checks completed: all project sources passed Python bytecode
  compilation. Every zero-argument focused regression in
  `src/tests/test_online_core.py`, `src/tests/test_online_diagnostics.py`, and
  `src/tests/test_reporting_selection.py` passed in the thesis runtime using a
  no-op import stub for the unavailable
  `einops`; the exercised normalized-cache ridge/gate/reporting paths did not
  call that stub. A tiny two-user raw-retrieval extraction also completed with
  a pattern-equivalent `rearrange` stub and verified normalized window shapes,
  neighbor IDs, and sparse selected-neighbor forecasts. The runtime does not
  provide `pytest` or real `einops`.
- Deferred checks: in the prepared cluster environment, run
  `src/tests/test_online_core.py`, then a tiny test-mode extraction/adaptation
  smoke to exercise memmap joins, real backbone context batches, FAISS TS-RAG,
  and all-user versus same-user fitting trajectories.
- Documentation/LaTeX: README and experiment-guideline source describe the
  v1 tables, algorithm, alignment, metrics, and latency scope. The executive
  summary is unchanged because no result evidence was produced; render the
  guideline PDF during maintenance.
- Required reruns: all future online-adaptation extraction and result artifacts
  use version 1; no earlier artifact exists to preserve or migrate.

## 2026-08-23 — First extraction contract and upstream TS-RAG retrieval

- Scope and behavior: kept the new balanced-store and retrieval-timing behavior
  under `online_extraction_v1`, since no earlier extraction exists. Replaced
  TS-RAG's use of the proposal's generic neighbor search with the released
  Chronos-T5-base bfloat16 EOS representation and float32
  `faiss.IndexFlatL2` `top_k + 1` search/final-result removal. The experiment
  now supplies only causally accessible same-variable candidate dates, and ARM
  receives the retrieved raw lookback--future sequence without project-scale
  conversion.
- Affected files and contracts: extraction schema constant and assertion,
  TS-RAG retriever, causal extraction dispatch, TS-RAG dependency declaration,
  local provenance guidance, README, and experiment-guideline source. All run,
  extraction, and result schemas remain version 1.
- Focused checks completed: Python compilation passed for `src`; the focused
  `src/tests/test_online_core.py` regression passed in the shared thesis
  runtime, including the v1 assertion and an injected exact squared-L2 index
  check for upstream top-k ordering and final-result removal; and the updated
  `pyproject.toml` parsed successfully with `tomllib`.
- Deferred checks: use the project environment on the cluster to exercise the
  real FAISS package, Chronos-T5 checkpoint, raw retrieved sequences, strict
  TS-RAG ARM checkpoint loading, and end-to-end prediction parity.
- Documentation/LaTeX: README and experiment-guideline source now distinguish
  the unchanged retriever algorithm from experiment-owned causal index
  membership. The executive summary is unchanged because no evidence exists;
  render and inspect the guideline PDF during maintenance.
- Required reruns: none; no experiment or extraction artifact has yet been
  produced.

## 2026-08-23 — Balanced causal stores, deployment timing, and package separation

- Scope and behavior: made cross-user `N_store` a maximum cardinality formed
  from complete date-by-user panels, so the retained store contains
  `floor(N_store / n_users)` dates per user and never a partial oldest date.
  Made same-user `N_store` a per-user example cap, with all available causal
  dates retained when fewer exist, and rejected cross-user configurations with
  `N_store < n_users`. Extraction now records exact-neighbor search time and
  each fitted adaptor records naive row-wise and date-batched estimates of the
  retrieval-only cost of an uncached independent query.
- Architecture: separated data handling, model loading, external baselines,
  proposal logic, explicit experiment-stage scripts, pipeline/configuration
  infrastructure, result computation, and visualization. TS-RAG now lives
  exclusively under `src/external_models/tsrag/` and is documented as a
  pinned, source-adapted implementation rather than a verbatim upstream copy.
  Renamed the project-only homogeneous-channel catalog to
  `homogeneous_channels.yaml` to distinguish it from each dataset's general
  adjacent `config.json`.
- Affected contracts: the balanced-store rules, `extraction_timing.json`, and
  `deployment_retrieval_timing.json` belong to `online_extraction_v1`, the
  first and only extraction contract because no experiment has run. The
  result and run-manifest schemas likewise remain version 1.
- Focused checks completed: `src/tests/test_online_core.py`,
  `src/tests/test_online_diagnostics.py`, and
  `src/tests/test_reporting_selection.py` passed in the shared thesis runtime;
  `python -m compileall -q src` passed; and Git Bash syntax checks passed for
  `src/slurm/run_family.sh` and all 15 root Slurm fronts. The focused core test
  covers cross-user balancing, same-user caps, causal maturity, the minimum
  cross-user store size, timing estimates, schema v1, and package boundaries.
- Deferred checks: run a real-backbone extraction and ridge/gate adaptation on
  the cluster, validate measured timing artifacts and strict TS-RAG checkpoint
  loading, and exercise the CatBoost gate path. The current test profile's
  `QUERY_STRIDE=256` may yield fewer than the default `N_fit=1000` rows on
  Electricity; reconcile that smoke-profile sizing before its next submission.
- Documentation/LaTeX: updated the README, project guidance, workspace-wide
  architecture constraints, experiment-guideline source, and executive-summary
  source. Render and inspect both project PDFs during maintenance.
- Required reruns: none; no extraction or downstream experiment artifact has
  been produced under an earlier contract.

## 2026-08-21 — Cadence-aware L-H profiles and pre-experiment schema reset

- Scope and behavior: replaced the generic standard-setting grids with one
  per-dataset cadence map. Hourly, daily, and 15-minute configurations resolve
  `short`, `mid`, and `long` to `168:24`/`336:48`/`504:168`,
  `7:1`/`14:2`/`30:7`, and `96:4`/`192:8`/`672:96`, respectively. `test`
  selects only `long`; `small` and `full` select all three. Prepared TIME
  configurations take their cadence from catalog metadata. Explicit L/H
  sensitivity grids and external `512:64` checkpoint comparisons remain
  custom/fixed configurations and are not mislabeled as cadence ranges.
- Affected contracts: standard task expansion for every root Slurm front,
  homogeneous-profile expansion, TIME catalog consumption, and extraction,
  adaptation, and shared run manifest identifiers. Because no experiment has
  run, all three current schemas are version 1 with no compatibility reader.
- Focused checks completed: the thesis-project scaffold completed without a
  collision; Python compilation passed for the profiles, contracts, and focused
  test; `src/tests/test_online_core.py` passed in the shared thesis runtime;
  Git Bash syntax passed for the shared runner and all 15 root fronts; every
  front family expanded in test mode; synthetic TIME metadata confirmed
  long-only test and all-three full coverage at all three cadences; and a final
  scan found no schema identifier above version 1.
- Deferred checks: run the Electricity long-range ridge and gate smoke jobs on
  the cluster; inspect the first generated manifests and range-specific tables.
- Documentation/LaTeX: README and the experiment-guideline source describe the
  cadence map, profile coverage, and fixed/custom exceptions. Render and
  inspect the guideline PDF during maintenance; no evidence summary changed.
- Required reruns: none; the project has no completed experiment or reusable
  artifact under any prior schema.

## 2026-08-21 — Cadence-aware homogeneous-channel ablation

- Scope and behavior: restricted the homogeneous-channel ablation to the four
  original ETT panels and Weather, with Weather at its long range alone in
  `test` mode. `small` and `full` use hourly short/mid/long settings
  `168:24`, `336:48`, `504:168` for ETTh/Weather and 15-minute settings
  `96:4`, `192:8`, `672:96` for ETTm. The homogeneous Weather arm retains the
  four temperature channels while its comparison arm retains every variate.
- Affected contracts: `homogeneous_ablation` task expansion, the homogeneous
  channel catalog, its focused profile assertions, and the public Slurm profile
  documentation. The family no longer depends on the prepared TIME catalog in
  `full` mode.
- Focused checks completed: Python compilation passed for the profile and
  focused test modules. The current online-core profile check expanded exactly
  2 test tasks and 30 full tasks and confirmed both cadence grids; the earlier
  catalog check parsed the YAML and verified every selected Weather channel
  against the local CSV header.
- Deferred checks: run the Weather smoke on the cluster and inspect both
  channel-arm reports.
- Documentation/LaTeX: README now documents the family-specific datasets,
  cadence mapping, and channel subsets. Reconcile the project experiment
  guideline and render its PDF during maintenance; no evidence or executive
  conclusion changed.
- Required reruns: no completed project result exists. Any future or external
  homogeneous-ablation artifact produced with the former generic dataset or
  setting grid is outside the current protocol and must be rerun.

## 2026-08-21 — Configurable same-user fitting scope

- Scope and behavior: introduced `fitting_scope=all|same_user` for linear and
  gate adaptors, with `same_user` as the Python, Hydra, profile, and Slurm
  default. Same-user mode retains exactly `N_fit` matured rows independently
  for each query user and performs its chronological alpha/K selection within
  that user's window; all-user mode preserves the prior exact global row pool.
  TS-RAG evaluation alignment now applies the same maturity interpretation.
- Affected contracts: adaptation identity includes `fitting_scope` and result
  manifests use the current `online_adaptation_v1`. The former retrieval-scope front was
  replaced by `ablation_general_scope.slurm`, whose
  `general_scope_ablation` crosses retrieval scopes `all`, `same_user`, and
  `other_users` with fitting scopes `all` and `same_user`. Extraction identity
  remains unchanged, so fitting-scope variants reuse matching retrieval caches.
- Focused checks completed: the shared scientific runtime passed
  `src/tests/test_online_core.py`, `src/tests/test_reporting_selection.py`, and
  `src/tests/test_online_diagnostics.py`; coverage includes same/all-user ridge
  and Bayes paths, per-user chronological tuning, maturity alignment, and all
  six general-scope combinations. Python compilation passed for the affected
  modules and workflow; every test-mode family expanded successfully, with 42
  general-scope tasks; and Git Bash syntax passed for the shared runner and all
  15 root fronts.
- Deferred checks: the first cluster smoke must validate same-user fitting with
  real backbones, the real TS-RAG checkpoint, and CatBoost per-user refits and
  runtime. No heavy local inference or CatBoost fit was run.
- Documentation/LaTeX: README and both project documents define `N_fit` under
  both scopes, the default, cache reuse, the `FITTING_SCOPE` override, and the
  renamed general-scope front. Three pdfLaTeX passes per document succeeded;
  the final two-page guideline and one-page executive summary were rasterized
  and visually inspected without clipping, overlap, or table overflow.
- Required reruns: none because the project still has no completed adaptation
  result or pre-current-schema artifact requiring migration.

## 2026-08-21 — Causal per-query ridge hyperparameter selection

- Scope and behavior: made the primary ridge select alpha and K independently
  at every query. For each fitting scope, the oldest 80% of the current causal
  `N_fit` rows train the candidate grid and the newest 20% validate it; the
  winning pair is refitted on all rows before inference. Defaults are alpha
  `{1e-1, 1e-2, 1e-3}` and K `{1, 5, 10, 15}`, with exact ties resolved by
  smaller K and then stronger regularization. The ratio and grids are launcher
  overrides; fixed alpha/K ablations explicitly retain fixed settings.
- Affected contracts: tuned ridge extraction now retains 15 neighbors and
  candidate K values consume nearest-neighbor prefixes; adaptation identity
  includes the tuning flag, validation ratio and both grids; results use the
  current `online_adaptation_v1` and add `selected_hyperparameters.csv` plus selection
  counts in `result_manifest.json`.
- Focused checks completed: the shared scientific runtime passed
  `PYTHONPATH=. python src/tests/test_online_core.py`, including rolling
  same-user selection, exact 80/20 counts, fixed-hyperparameter paths and
  selection artifacts. Python compilation passed for the affected contracts,
  ridge, profiles, workflow and test modules. Git Bash syntax passed for the
  shared Slurm runner and every root front. The thesis-project scaffold ran
  without reporting a collision.
- Deferred checks: run `EXPERIMENT_MODE=test sbatch 01_main_online_ridge.slurm`
  on the cluster and inspect selection frequencies, validation losses, runtime
  and the 15-neighbor extraction footprint. The local shared runtime lacks
  Hydra, so launcher composition remains part of that integration check.
- Documentation/LaTeX: README and both LaTeX sources describe the causal
  internal validation protocol and sensitivity-only alpha/K ablations; PDF
  rendering is deferred to maintenance.
- Required reruns: no completed online result exists. A tuned main run cannot reuse
  a 10-neighbor main extraction because its manifested extraction width is 15.

## 2026-08-20 — Initial causal online adaptation project

- Scope and behavior: introduced persistent manifest-validated online extraction; exact global `N_store` and `N_fit` causal windows; standardized rolling/fixed ridge, delta ridge, convex fitting, Bayes and CatBoost gates; source-faithful TS-RAG inference; common-date reports; publication profiles and all requested ablation fronts.
- Affected contracts: `outputs/online_extraction/`, `outputs/online_adaptation/<family>/`, `outputs/reports/<family>/<mode>/`, Hydra configuration in `src/conf/`, and every root Slurm front.
- Focused checks completed: `PYTHONPATH=. python src/tests/test_online_core.py` passed; Python compilation of the online modules and workflow passed; Bash syntax validation of `src/slurm/run_family.sh` and all root Slurm fronts passed; profile expansion for every family in test mode passed.
- Deferred checks: run `01_main_online_ridge.slurm` in test mode on the cluster to validate actual Chronos-2, Chronos-T5, Chronos-Bolt, and TS-RAG checkpoints; run `02_online_gates.slurm` test mode to validate the installed CatBoost runtime and measured refit cost; inspect generated CSV/TeX tables and coefficient/importance plots.
- Documentation/LaTeX: README and both project LaTeX documents describe the new no-split contract; PDFs were rendered locally.
- Required reruns: this is a new project and has no reusable adaptation outputs. Its own extraction caches are reusable only when their manifests match; artifacts from `../adaptation/` are intentionally not consumed.

## 2026-08-21 — Root launcher repository fallback

- Scope and behavior: aligned every root Slurm front with the thesis launcher
  contract by resolving `PROJECT_ROOT` from `SLURM_SUBMIT_DIR` with the current
  working directory as the direct-shell fallback, then continuing to `cd` into
  that root before sourcing `src/slurm/run_family.sh`.
- Affected contracts: all 15 root `.slurm` fronts; no Python, scientific,
  manifest, artifact, or experiment-grid behavior changed.
- Focused checks completed: after the edit, Git Bash syntax passed for
  `src/slurm/run_family.sh` and all 15 fronts, and a contract assertion
  confirmed the standard fallback, one task per front, no arrays, and a
runner-recognized family for every front.
- Shared publisher validation also passed: Git Bash accepted all 10 project
  copies and their SHA-256 hashes were identical at
  `0A9E87E51517B9F5816BB92CDE726B9E383AB6B8A70DC251FEF429BF7B53B45C`.
- Deferred checks: the first real `sbatch` submission remains the external
  integration boundary.
- Documentation/LaTeX: no public command or protocol changed; README and both
  LaTeX sources require inspection only.
- Required reruns: none; no experiment has yet completed in this project.

Maintenance 2026-08-21: direct inspection confirmed the exact global-window,
rolling-store/fit, standardized ridge, gate, report-intersection, profile,
manifest, and artifact contracts. The three-test synthetic online core passed;
it was repeated because it is the project's sole dependency-light end-to-end
boundary. The post-fix Bash and front-contract checks above also passed. The
README and both LaTeX sources already match the implementation, so no source
rewrite was needed. Three pdfLaTeX passes produced clean two-page guideline and
one-page no-results summary PDFs; all three rendered pages passed visual
inspection. No executive-summary claim changed because no cluster artifact
exists. Remaining blockers are the documented online-ridge and gate test jobs,
checkpoint/runtime validation, generated-artifact inspection, and the first
real publisher handoff; no existing result requires a rerun.

## 2026-08-21 - 30,000-window smoke store and robust report selection

- Scope and behavior: made the test-profile `N_store` default 30,000 while
  retaining `N_fit=1,000`; assigned smoke/publication purposes by profile; and
  moved reports to the shared manifest selector with nested dependency filters,
  every config/repeat policy, purpose/seed eligibility, common-date averaging,
  dependency-specific vanilla rows, and standard report manifests.
- Affected contracts: `src/slurm/run_family.sh`, Hydra/report workflow,
  `src/online/report.py`, shared `src/experiment_runs.py`, focused tests, and
  README profile/selection documentation.
- Focused checks completed: dependency-aware report selection, filtering, and
  averaging passed in the shared scientific runtime; Python compilation and
  Git Bash syntax passed; the shared 13-test manifest suite passed in all five
  repositories that carry it.
- Deferred checks: the full online-core test requires the project environment
  because the shared runtime lacks `einops`; run the updated test smoke on the
  cluster to validate model checkpoints and the larger store.
- Documentation/LaTeX: README changed; no project result claim changed.
- Required reruns: none because no online result exists; every future test
  profile now uses the 30,000-window store unless explicitly overridden.

## 2026-08-21 — 1,000-row fit default and adaptor-only parameter counts

- Scope and behavior: standardized every Python, Hydra, and Slurm profile
  default at `N_store=30,000` and `N_fit=1,000`; retained the existing shared
  all-user fitting pool; and added adaptor-only parameter metadata to linear,
  Bayes, CatBoost, and TS-RAG result manifests.
- Affected contracts: adaptation results use the current `online_adaptation_v1` and
  require `parameters.adaptor`, `parameters.backbone_included=false`, and a
  method-specific definition. Linear methods count fitted coefficients, Bayes
  counts its fitted mean and dispersion, CatBoost counts final fitted leaf
  values, and TS-RAG counts only ARM encode-MLP, attention, FFN, and gate
  tensors. Frozen backbones and the retrieval encoder are excluded.
- Focused checks completed: `src/tests/test_online_core.py` and
  `src/tests/test_reporting_selection.py` passed in the shared thesis runtime;
  Python compilation of the affected online modules and workflow passed; Git
  Bash syntax validation of `src/slurm/run_family.sh` passed; and a final scan
  found no remaining version-1 result literals or 30,000-row fit defaults.
- Deferred checks: the first cluster smoke must validate the real TS-RAG
  checkpoint ARM count and CatBoost leaf count with the project dependencies;
  no local checkpoint inference or CatBoost fit was run.
- Documentation/LaTeX: README and both LaTeX documents now state the
  30,000/1,000 defaults and distinguish the all-user fitting pool from
  per-row retrieval scope. Three pdfLaTeX passes per changed document succeeded;
  the final two-page guideline and one-page executive summary were rasterized
  and visually inspected with no clipping, overlap, or table overflow.
- Required reruns: none because this project still has no completed adaptation
  result or pre-current-schema artifact requiring migration.

## 2026-08-21 — Extraction and neighbor distribution diagnostics

- Scope and behavior: added sampled `(dataset, L:H)` inter-date and aligned
  inter-user distances over complete `L+H` windows, plus extraction-specific
  same/other-user neighbor origins, ages, and lookback distances over all
  samples and as per-query-user averages. Raw and instance-normalized RMS
  versions are produced; complete windows are normalized only with their own
  `L`-step lookback statistics.
- Affected contracts: extraction artifacts use the current `online_extraction_v1`,
  include raw and instance-normalized neighbor-lookback distance arrays, and
  require setting/neighbor CSV summaries, sampling metadata, and three PNG
  diagnostic dashboards. The artifact schema is part of extraction run
  identity, so version-1 caches cannot satisfy the new workflow.
- Focused checks completed: `src/tests/test_online_core.py`,
  `src/tests/test_online_diagnostics.py`, and
  `src/tests/test_reporting_selection.py` passed in the shared thesis runtime;
  Python compilation passed for the contracts, extraction, diagnostics,
  workflow, and focused tests. The diagnostics test covered lookback-only
  normalization of full windows, level-invariant neighbor distances, sampled
  inter-date/inter-user comparisons, exact origin fractions, CSVs, and PNGs.
  All three synthetic PNG dashboards were rendered and visually inspected with
  readable titles, axes, legends, and no overlap or clipping.
- Deferred checks: run a real cluster extraction to measure diagnostic runtime
  and inspect the generated distributions on a multi-user dataset; reconcile
  `latex/experiment_guideline.tex` and render it during the next maintenance
  pass. Backbone inference was not run locally.
- Documentation/LaTeX: README documents the two diagnostic layers, distance
  definition, normalization, artifacts, and focused check. No result claim or
  executive-summary evidence changed.
- Required reruns: no completed project result or pre-current-schema extraction
  cache exists, so no migration or dependent rerun is required.

Maintenance 2026-08-23: direct inspection reconciled the current cadence-aware
profiles, same-user fitting default, causal ridge selection, adaptor-only
parameter metadata, manifest/report selection, and both diagnostic layers with
the implementation and required-artifact list. The shared-runtime
`src/tests/test_online_core.py` check passed; it was repeated because it is the
sole dependency-light end-to-end boundary spanning the updated defaults,
profiles, fitting scopes, manifests, adaptation artifacts, and report
intersection. The already-successful diagnostic and selector tests were not
repeated because they directly cover those focused surfaces. README was
current. `latex/experiment_guideline.tex` now specifies setting-shift sampling,
raw and lookback-normalized RMS distances, exact versus sampled summaries, and
the diagnostic artifact set. Three pdfLaTeX passes produced a clean three-page
PDF, and all pages passed visual inspection without clipping, overlap, overflow,
broken references, or unreadable text. The executive summary remains unchanged
because no cluster evidence exists. All pending entries remain blocked on the
first real ridge/gate/extraction cluster runs, checkpoint and CatBoost runtime
validation, generated-artifact inspection, live lifecycle observation, and the
manual publisher; no existing result requires rerunning.

Maintenance 2026-08-24: direct source, configuration, Slurm, test,
documentation, placeholder, and handoff inspection confirmed the version-1
Chronos-T5/FAISS retriever contract, causal same-user candidate filtering, raw
TS-RAG neighbor sequences, and absence of generated experiment artifacts. The
complementary diagnostics and report-selection tests passed in the shared
thesis runtime; the already-successful core retriever regression was not
repeated. The three-page experiment guideline and one-page no-results executive
summary were compiled and visually inspected without missing assets, clipping,
overflow, broken references, or unreadable typography. The guidance-only entry
is resolved. Package separation remains open because `proposal` still imports
pipeline, results, and visualization owners, while the TS-RAG external package
imports proposal metrics. The next local action is to move orchestration,
artifact writing, diagnostics, plotting, and shared metrics to their declared
outer owners, then rerun package-boundary and consumer checks. Real
FAISS/checkpoint/CatBoost execution, ridge/gate/extraction artifacts, lifecycle
observations, and the first publisher run remain external blockers.

## 2026-08-24 — Shared TIME and five-backbone evaluation surface

- Behavior and affected contracts: added the shared TIME preparation front and
  byte-identical Chronos-2, Chronos-Bolt, TS-ICL, TiRex-2, and TabPFN-TS
  adapters; added TabPFN-TS to the backbone profile and checkpoint routing;
  standardized adapter constructors; and kept the retrieval-aware chronos
  wrapper as a separate project specialization. Lazy package facades keep TIME
  preparation and individual adapters independent of unrelated dependencies.
  Moved extraction, adaptation, TS-RAG evaluation, artifacts, diagnostics, and
  plots to pipeline/result owners so proposal and external packages now depend
  only inward.
- Focused checks and outcomes: Python compilation, online-core adaptation,
  diagnostics, report selection, the dependency-free five-backbone and package
  boundary guard, TIME preparation, adapter imports, TOML parsing, and
  cross-project SHA-256 parity passed.
- Deferred integration: real Chronos/TS-ICL/TiRex/TabPFN checkpoints, FAISS,
  and CatBoost remain cluster checks; the prior proposal/package-boundary entry
  is resolved. No dependency was installed.
- README/LaTeX and reruns: README documents the shared surface and local
  Chronos specialization. Reconcile and render the guideline during
  maintenance. No completed adaptation artifact exists; run the backbone
  ablation test/full profiles under the new contract.

Maintenance 2026-08-25: direct package-boundary, profile, workflow, artifact,
README, guideline, summary, and handoff inspection confirmed the five shared
adapters, the project-specific retrieval wrapper, and the absence of experiment
payloads. The entire zero-argument `src/tests/test_online_core.py` suite passed
with a no-op `einops` import stub; this was repeated because it is the sole
dependency-light end-to-end boundary spanning extraction, fitting, adaptation,
and reporting, and no stubbed operation was called. The README and six-page
guideline were already current, and the one-page summary correctly makes no
result claim; both PDFs were rendered and visually inspected without rewrite.
The empty archive-only entry is removed. Real checkpoint, FAISS, CatBoost, and
cluster runs, artifact inspection, lifecycle observations, and the publisher
check remain pending.

## 2026-08-25 — Canonical foundation aliases and retrieval covariates

- Behavior and affected contracts: made `chronos2`, `chronos_bolt`, `ts_icl`,
  `tirex2`, and `tabpfn_ts` the sole foundation aliases; removed the separate
  retrieval-aware `chronos` model; layered
  `retrieval_covariate_mode=none|past|past_and_future` around the canonical
  shared adapter; and removed the silent unsupported-context fallback. Primary
  Chronos-2 profiles now send rescaled retrieved lookbacks and optional known
  horizons through the official covariate path. Covariate-free five-backbone
  and Chronos-Bolt ridge comparisons use `y_ridge_shared`; any explicit
  context sent to an unsupported backbone raises.
- Focused checks completed: the retrieval translation, disabled-mode error,
  unsupported-backbone error, profile identity, and five dependency-free
  foundation-workflow guards passed in the shared thesis runtime. Python AST
  parsing passed across all 148 source files in the three foundation projects;
  Bash syntax passed for every changed workflow; and SHA-256 comparison
  confirmed byte-identical copies of all five basic adapters. `pytest` could
  not start because it is absent from the prepared runtime, so the same focused
  tests were run through their direct and `unittest` entry points.
- Deferred integration: run one real Chronos-2 `past` and
  `past_and_future` contextual forecast and confirm Chronos-Bolt rejects a
  non-empty covariate batch in the prepared cluster environment. Foundation
  checkpoints, FAISS, and CatBoost were not loaded locally.
- README/LaTeX and reruns: README and the experiment-guideline source now
  specify canonical identity, layered retrieval covariates, and explicit
  failure. Re-render the guideline during maintenance; the executive summary
  is unchanged because no result was added. No completed current result exists;
  any externally retained run using the removed alias or vanilla context
  fallback is invalid and must be rerun.

Maintenance 2026-08-26: direct runner, profile, report-selection, README,
guideline, summary, placeholder, and handoff inspection confirmed the
single-task stage boundary and current canonical model/covariate contract. The
complementary `src/tests/test_reporting_selection.py` consumer passed. The
guideline was newer than its PDF and initially failed on an undefined local
`\code` command; adding the sibling-standard detokenizing macro and allowing
the long covariate setting to wrap produced a clean six-page PDF after final
pdfLaTeX passes. Every page passed visual inspection. The Slurm assertion and
LaTeX-defect entries are resolved. Real checkpoint, FAISS, CatBoost, cluster,
lifecycle, artifact, and publisher checks remain pending.
