# Finalized experiment recap

## Current evidence

No current online-adaptation experiment has completed and been analyzed, so
there is still no supported accuracy comparison between vanilla, ridge, a
causal gate, or TS-RAG.

The latest synchronized failures are diagnostic rather than scientific:

- fixed-protocol job 2964440 built a compact virtual-array context shorter
  than the required 168-step lookback;
- TS-RAG job 2964441 mixed CUDA token tensors with CPU bucket boundaries;
- the older online job 2964439 belongs to the superseded runtime contract and
  must not be used even if it reaches a terminal state;
- job 2955394 used an obsolete running-manifest contract, while job 2913487
  diagnosed the former quota-bound execution path.

The current code slices compact contexts correctly, performs tokenizer/bucket
work on CPU before device transfer, limits retrieval candidates to 10,000,
uses 10 causal fitting dates per query user, and zero-fills CSV NaNs by default.
The synchronized replacement sequence now shows:

- online-per-user anchor 2964505 and fixed-shared anchor 2964506 reached their
  final Solar adaptation with empty stderr, but their old logs contain no
  terminal workflow marker, so their final scheduler states remain uncertain;
- fixed-protocol remainder 2964508 completed all 12 adaptations and then
  failed while reporting because equivalent vanilla sources had distinct
  extraction-hash labels;
- priority TS-RAG replacement 2964507 completed all 16 extractions and its
  first ridge adaptation, then failed because the source-adapted ARM imported
  its forecasting class from the thin official Chronos-Bolt adapter.

The retained 24-row fixed-protocol result is valid report input, but no final
fixed or TS-RAG comparison has completed.

## Finalized conclusions

- The causal protocol, fixed evaluation grids, and artifact contracts remain
  the intended experiment.
- The failures above identify implementation/runtime defects; they provide no
  evidence about adaptation accuracy.
- No alpha, K, fitting scope, retrieval scope, or feature design is selected.

## Next evidence

Deploy the import, labeling, and terminal-record fixes. Resume the priority
TS-RAG workflow with only `adapt,tables`, then regenerate the fixed-protocol
comparison with only `tables`. The current workflow reuses completed exact
inputs. Accuracy claims remain blocked until both matched current-code reports
complete and are synchronized.

The formal evidence record remains
[`executive_summary.pdf`](../latex/executive_summary.pdf).
