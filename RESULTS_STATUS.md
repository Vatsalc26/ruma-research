# Results Status

This file is the repo's current reality check.

## What We Can Say Honestly

- The repo contains a broad and useful paper archive for architecture study.
- The sandbox demonstrates that a modular routed-memory toy model can be assembled and run.
- A first-class routed memory write/read path now exists in the sandbox.
- A toy update/retention demo exists.
- A first repeatable benchmark harness now exists with fixed evaluation sets and multiple baselines.
- A first real-text chunk benchmark now exists using `alice.txt`.
- A first real document-backed prototype now exists over the local markdown paper corpus.
- A first citation-first grounded answer layer now exists over retrieved document chunks.
- A first stale-guidance supersession and conflict-aware answering policy now exists on the system track.
- A first namespace-banded shard assignment and refresh policy now exists on the system track.
- An expanded curated versioned-manual corpus now exists for external-style update evaluation.
- The current planning stage now has a closed set of major architecture decisions.
- The project now has a clearer scientific doctrine and architecture framing.

## What We Cannot Yet Say Honestly

- that the project has solved catastrophic forgetting
- that the project is faster than strong dense baselines
- that the project has a real reasoning decoder
- that the current sandbox proves a new production architecture
- that the current memory mechanism is equivalent to external retrieval or knowledge freshness at scale

## Current Technical Truth

- Parts of the original sandbox were overclaiming relative to the evidence.
- `test_run.py` and `train.py` were previously broken because `DummyDataset` was missing; that scaffolding has now been repaired.
- The current C++ extension path depends on local toolchain availability.
- The previous "pond" implementation is best understood as a toy latent lookup, not a finished memory system.
- On the first synthetic factual-update benchmark in `sandbox/honest_benchmark.py`, the current memory-write path now reaches updated-fact exact match `1.0` with retrieval recall@1 `1.0` while preserving unaffected retention exact match `1.0`.
- On that same benchmark, naive fine-tuning also reaches updated-fact exact match `1.0`, but it damages unaffected retention badly, dropping exact match to `0.0`.
- The current memory path is much cheaper to update on that benchmark than naive fine-tuning (`~0.0017s` versus `~0.313s` in the local run), but it is still slower at inference and does not yet fully suppress stale parametric knowledge for replaced facts.
- On the repeated-update benchmark in `sandbox/supersession_benchmark.py`, explicit supersession of older memory records allows a second update to take effect at exact match `1.0` while unaffected retention stays at exact match `1.0` on the toy task.
- On the code-flavored synthetic benchmark in `sandbox/code_drift_benchmark.py`, the current memory-write path again reaches updated-fact exact match `1.0` with retrieval recall@1 `1.0`, while naive fine-tuning still damages retention more severely.
- On the harder documentation-chunk benchmark in `sandbox/doc_chunk_benchmark.py`, the current memory-write path improves update exact match from `0.0` to `0.75` while preserving retention better than naive fine-tuning, but it no longer solves the task perfectly, which is a healthier and more realistic result.
- On the first real-text benchmark in `sandbox/alice_chunk_benchmark.py`, the current memory-write path reaches update exact match `1.0` with retrieval recall@1 `1.0` on held-out `alice.txt` chunks, while naive fine-tuning still hurts retention much more strongly than the memory path.
- The new system-track prototype in `sandbox/real_doc_memory.py` and `sandbox/real_doc_demo.py` can ingest real markdown papers, preserve provenance, retrieve cited chunks, and accept a live append-style update. In the local run it indexed `17` markdown documents into `2382` chunk records across `17` namespaces, but it is still a retrieval layer rather than a full end-to-end language model answerer.
- The new answer path in `sandbox/real_doc_answerer.py` and `sandbox/real_doc_answer_demo.py` can turn retrieved evidence into citation-first extractive answers, but it is still grounded extraction rather than robust generative reasoning.
- The versioning and conflict path in `sandbox/real_doc_memory.py`, `sandbox/real_doc_answerer.py`, and `sandbox/real_doc_conflict_demo.py` can supersede same-lineage stale records and surface multi-source disagreement instead of silently blending it, but contradiction detection and source arbitration are still primitive.
- The shard-layout path in `sandbox/real_doc_memory.py` and `sandbox/real_doc_shard_demo.py` now supports namespace-banded deterministic shard placement plus refresh-based reassignment. In the local demo, widening shard bandwidth from `2` to `4` spread namespaces like `mamba` from `2` shards to `4` shards while preserving the same underlying records.
- On the expanded `12`-manual versioned-manual corpus benchmark in `sandbox/versioned_manual_benchmark.py`, the base-only system still fails completely on future-version facts (`no_update_future_eval = 0.0`), while explicit same-lineage supersession still reaches `superseded_update_eval = 1.0`, preserves unchanged guidance at `retention_eval = 1.0`, and now surfaces conflict cases at `conflict_eval = 0.8333`. The weaker append-without-supersession baseline remains poor at `naive_append_eval = 0.0833`, because answers still leak old and new guidance together.
- That same benchmark now also provides an updated paper-facing systems snapshot after the answer-layer change: base build time about `0.1612s`, superseded update ingest about `0.1324s`, conflict ingest about `0.0692s`, and average query latency still in the low-millisecond range on the local run. These numbers are still local prototype measurements, not production claims.
- On the new multi-namespace routing benchmark in `sandbox/multi_namespace_manual_benchmark.py`, unrestricted global search, simple keyword namespace routing, and oracle namespace routing all currently reach exact match `1.0` with namespace hit rate `1.0` on the current `12`-namespace manual corpus, while simple routing reduces average query latency from about `2.018ms` to about `1.358ms`. On the current corpus, routing quality is therefore still not the main bottleneck.
- On the refreshed scaling benchmark in `sandbox/retrieval_scaling_benchmark.py`, adding `1000` irrelevant distractor chunks increases unrestricted exact-search latency to about `3.213ms` while simple namespace-routed exact search stays around `1.435ms`, with both modes still at exact match `1.0`. This still supports ANN-style retrieval as the next borrowed infrastructure layer rather than a heavier router.
- The repo now also has an optional ANN backend adapter in `sandbox/ann_retrieval.py` plus `sandbox/ann_backend_benchmark.py`. In the current local environment, `exact`, `faiss_flat`, and `hnsw` all now run as real backends. On the current routed slices, `faiss_flat` is the strongest local backend at small-noise global search, while `hnsw` is active but not yet a clear latency win on the present benchmark.
- The repo now has a scaffolded external-corpus path in `external_corpora/python_ecosystem_changes/` plus `sandbox/external_corpus_benchmark.py`, so broader manually downloaded corpora can be plugged into the same benchmark flow without inventing a new harness.
- That external-corpus path is no longer only a scaffold. The expanded corpus now contains `10` curated official-source documents plus `2` active conflict notes across routed per-document namespaces, and `sandbox/external_corpus_benchmark.py` reaches `base_known_eval = 1.0`, `superseded_update_eval = 1.0`, `retention_eval = 1.0`, and `conflict_eval = 1.0`, while `no_update_future_eval = 0.0` and `naive_append_eval = 0.6`. This now meets the first-preprint external corpus freeze rule and becomes the main external evaluation set unless a specific weakness forces another corpus change.
- The new harder answer-quality pass in `sandbox/external_answer_quality_benchmark.py` now tests compositional updated-plus-retained answers and multi-source conflict synthesis on that frozen external corpus. On the current local run, the base-only system stays at `base_compositional_eval = 0.0`, the superseded update path reaches `superseded_compositional_eval = 1.0`, the naive append baseline reaches only `naive_append_compositional_eval = 0.4`, and conflict synthesis reaches `conflict_synthesis_eval = 0.5`. This is a healthier paper result than a full clean sweep because it shows a real remaining weakness in one conflict-heavy answer case.

## Repo Direction Going Forward

The repo is now organized around:

- `PROJECT_LENS.md`
- `ARCHITECTURE_SPEC.md`
- `RESEARCH_ROADMAP.md`
- `EVIDENCE_LEDGER.md`
- `RESULTS_STATUS.md`

These files should be treated as the source of truth for future work.
