# Evidence Ledger

This file tracks the real status of the project claims.

Statuses:

- `unsupported`: idea exists, no meaningful evidence yet
- `weak`: toy or narrow evidence only
- `mixed`: some evidence, but contradictory or incomplete
- `supported`: evidence is decent for the claimed scope
- `falsified`: current evidence contradicts the claim as stated

## Claim Table

| Claim ID | Claim | Status | Current evidence | Main gap |
| --- | --- | --- | --- | --- |
| C1 | New knowledge can be inserted without full retraining | mixed | `sandbox/test_memory_update.py` verifies the write/read path, `sandbox/honest_benchmark.py` shows cheap writes plus retrieval recall@1 of `1.0` and updated-fact exact match of `1.0` on the first synthetic benchmark, `sandbox/supersession_benchmark.py` shows a second memory update can supersede the first on that toy task, `sandbox/code_drift_benchmark.py` reproduces the update win on a code-flavored synthetic benchmark, `sandbox/doc_chunk_benchmark.py` shows longer retrieved chunks improve update exact match from `0.0` to `0.75` on a harder chunk-style prompt task, `sandbox/alice_chunk_benchmark.py` reaches update exact match `1.0` with retrieval recall@1 `1.0` on held-out real-text chunks from `alice.txt`, the expanded `12`-manual `sandbox/versioned_manual_benchmark.py` reaches `superseded_update_eval = 1.0` while the base-only system stays at `no_update_future_eval = 0.0`, and the expanded external corpus in `sandbox/external_corpus_benchmark.py` reaches `superseded_update_eval = 1.0` while `no_update_future_eval = 0.0` on `7` official-source documents plus `1` conflict note | Need broader external corpora and better handling of stale parametric knowledge under larger conflicting update sets |
| C2 | The architecture resists forgetting after updates | mixed | `sandbox/test_leak.py` shows a toy retention demo, `sandbox/honest_benchmark.py` shows memory writes preserving unaffected retention at exact match `1.0` while naive fine-tuning dropped unaffected retention to exact match `0.0`, `sandbox/supersession_benchmark.py` preserved unaffected retention through a second update, `sandbox/code_drift_benchmark.py` preserved code-flavored retention better than naive fine-tuning, `sandbox/doc_chunk_benchmark.py` preserved retention at exact match `0.75` versus naive fine-tuning exact match `0.0` on a harder chunk task, `sandbox/alice_chunk_benchmark.py` preserves real-text retention much better than naive fine-tuning on held-out `alice.txt` chunks, the expanded `12`-manual `sandbox/versioned_manual_benchmark.py` preserves retained manual guidance at `retention_eval = 1.0` while the naive append baseline collapses to mixed answers, and the now-frozen `10`-document external corpus in `sandbox/external_corpus_benchmark.py` reaches `retention_eval = 1.0` while the naive append baseline stays at `0.3` | Need broader retention benchmarks and more realistic long-horizon tasks |
| C3 | Sparse routing is computationally superior | falsified | Current `benchmark.py` run was slower than the simple dense baseline on this machine | Need better implementation and fair matched-quality comparison |
| C4 | The router avoids collapse | mixed | `test_pressure.py` shows load spread in the toy setup, `sandbox/honest_benchmark.py` shows retrieval recall@1 of `1.0` on the first synthetic update task after causal write-path alignment, `sandbox/alice_chunk_benchmark.py` reaches retrieval recall@1 of `1.0` on held-out real-text chunk updates, and `sandbox/multi_namespace_manual_benchmark.py` shows unrestricted global search, simple keyword namespace routing, and oracle namespace routing all reaching exact match `1.0` with namespace hit rate `1.0` on the current `12`-namespace manual corpus | Need broader semantic routing metrics, more ambiguous cross-namespace queries, and larger noisier corpora before claiming routing robustness at scale |
| C5 | The decoder performs meaningful internal reasoning before output | unsupported | Current confidence loop is simulated with randomness | Needs real learned or search-based mechanism |
| C6 | The C++ routing path is operational | mixed | Source exists, but local execution currently fails without toolchain dependencies | Need reproducible build path and working environment |
| C7 | The repo is now framed as hypotheses instead of proof | supported | `PROJECT_LENS.md`, `README.md`, and new spec docs reset the framing | Keep future edits disciplined |
| C8 | The repo has a repeatable baseline benchmark instead of only hand-picked demos | supported | `sandbox/honest_benchmark.py`, `sandbox/doc_chunk_benchmark.py`, `sandbox/code_drift_benchmark.py`, `sandbox/alice_chunk_benchmark.py`, the expanded `sandbox/versioned_manual_benchmark.py`, `sandbox/multi_namespace_manual_benchmark.py`, `sandbox/retrieval_scaling_benchmark.py`, `sandbox/ann_backend_benchmark.py`, and the expanded `sandbox/external_corpus_benchmark.py` now compare multiple update or retrieval paths on fixed evaluation sets, including a `12`-manual changing-document corpus and a `7`-document official-source external corpus plus systems, scaling, backend-availability, and update-policy measurements | Extend from small curated corpora to larger external tasks |

## Baseline Rules

Every new claim added here must include:

- exact baseline
- metric
- experimental scope
- date or commit reference when possible

## What Counts As Stronger Evidence

For this project, stronger evidence means:

- held-out evaluation instead of hand-picked examples
- comparison against at least one honest baseline
- ablations on the changed mechanism
- failure analysis, not only success examples
