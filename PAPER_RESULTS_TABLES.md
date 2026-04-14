# Paper Results Tables

This file is the paper-facing landing zone for benchmark tables and summary results.

These tables are still working tables, not final publication tables.

## Table 1: Current Main Results Snapshot

| Benchmark | Memory Update Path | Baseline / Comparator | Main Outcome |
| --- | --- | --- | --- |
| `sandbox/honest_benchmark.py` | updated-fact exact match `1.0`; unaffected retention exact match `1.0`; retrieval recall@1 `1.0` | no-update baseline: updated facts fail; naive fine-tuning learns updates but drops unaffected retention to `0.0` | memory updates beat no-update and preserve retained behavior better than naive fine-tuning on the toy task |
| `sandbox/supersession_benchmark.py` | second updates succeed with explicit supersession; unaffected retention stays intact | repeated-update without structured supersession is weaker | explicit same-lineage supersession matters |
| `sandbox/code_drift_benchmark.py` | updated behavior learned via memory updates with strong retrieval | naive fine-tuning harms retention more | update/retention pattern transfers to a code-flavored task |
| `sandbox/doc_chunk_benchmark.py` | update exact match improves from `0.0` to `0.75` | naive fine-tuning retains less cleanly | chunk-level evidence use is promising but not solved |
| `sandbox/alice_chunk_benchmark.py` | update exact match `1.0`; retrieval recall@1 `1.0` | no-update baseline fails on held-out updates | first real-text bridge benchmark succeeds |
| `sandbox/versioned_manual_benchmark.py` | on the expanded `12`-manual corpus with `6` conflict docs: `superseded_update_eval = 1.0`; `retention_eval = 1.0`; `conflict_eval = 0.8333` | `no_update_future_eval = 0.0`; `naive_append_eval = 0.0833` | version-aware supersession still clearly beats no-update and naive append on the current changing-doc corpus, but the refreshed answer layer now exposes one conflict-answer miss and a small amount of naive-append leakage |
| `sandbox/external_corpus_benchmark.py` | on the expanded external corpus with `10` official-source docs, `2` active conflict notes, and per-document routed namespaces: `base_known_eval = 1.0`; `superseded_update_eval = 1.0`; `retention_eval = 1.0`; `conflict_eval = 1.0` | `no_update_future_eval = 0.0`; `naive_append_eval = 0.6` | the same update/supersession pattern survives a broader external-doc bridge corpus, although the refreshed answer layer now allows some naive-append cases to partially pass |
| `sandbox/external_answer_quality_benchmark.py` | on the frozen external corpus, harder updated-plus-retained answers reach `superseded_compositional_eval = 1.0` | `base_compositional_eval = 0.0`; `naive_append_compositional_eval = 0.4`; `conflict_synthesis_eval = 0.5` | the answer layer now handles compositional updated answers well after superseded writes, but one conflict-heavy synthesis case still fails |
| `sandbox/multi_namespace_manual_benchmark.py` | `keyword_routed_eval = 1.0`; namespace hit rate `1.0`; avg latency `1.3576ms` | global search also reaches exact match `1.0` but at `2.0184ms`; oracle routing reaches the same exact match ceiling at `1.1988ms` | on the current corpus, simple namespace routing already matches global and oracle quality while reducing query cost |

## Table 2: Current Claim Boundary

| Claim | Current Status |
| --- | --- |
| low-cost versioned document updates are possible without full retraining | supported on current controlled corpora |
| retained guidance can survive updates better than naive fine-tuning or naive append | supported on current controlled corpora |
| same-lineage supersession is important | supported |
| conflict visibility is possible without silent blending | supported in current system-track demos |
| RUMA solves hallucination | not supported |
| RUMA solves sycophancy | not supported |
| RUMA is faster than strong dense baselines | not supported |
| RUMA is a universal replacement for transformer systems | not supported |

## Table 3: Current Systems Snapshot

| Metric | Current value | Scope |
| --- | --- | --- |
| base corpus documents | `12` | expanded versioned-manual corpus |
| base corpus chunks / records | `12` | one chunk per current manual in the starter corpus |
| active payload bytes | `5354` | in-memory payload bytes for the base corpus |
| base build time | `0.161169s` | `sandbox/versioned_manual_benchmark.py` current local run |
| superseded update ingest time | `0.132423s` | ingesting all `12` versioned updates with same-lineage supersession |
| naive append ingest time | `0.037093s` | ingesting all `12` updates without supersession |
| conflict ingest time | `0.069225s` | ingesting `6` active conflict docs |
| base known-query avg latency | `2.1137ms` | current local run |
| updated-query avg latency | `2.4839ms` | current local run |
| retention-query avg latency | `2.7196ms` | current local run |
| naive-append-query avg latency | `2.0401ms` | current local run |
| conflict-query avg latency | `3.2202ms` | current local run |

## Table 4: Routing Snapshot

| Mode | Exact match | Namespace hit rate | Avg query latency |
| --- | --- | --- | --- |
| unrestricted global search | `1.0` | `1.0` | `2.0184ms` |
| simple keyword namespace routing | `1.0` | `1.0` | `1.3576ms` |
| oracle namespace routing | `1.0` | `1.0` | `1.1988ms` |

## Table 5: Retrieval Scaling Snapshot

| Noise chunks | Total records | Global exact match | Global avg latency | Routed exact match | Routed avg latency |
| --- | --- | --- | --- | --- | --- |
| `0` | `30` | `1.0` | `1.4648ms` | `1.0` | `1.0236ms` |
| `100` | `130` | `1.0` | `3.1015ms` | `1.0` | `1.386ms` |
| `500` | `530` | `1.0` | `3.2711ms` | `1.0` | `1.3907ms` |
| `1000` | `1030` | `1.0` | `3.2134ms` | `1.0` | `1.4354ms` |

## Table 6: ANN Backend Snapshot

| Requested backend | Active backend in local run | Package available locally | Note |
| --- | --- | --- | --- |
| `exact` | `exact` | `yes` | current stable baseline |
| `faiss_flat` | `faiss_flat` | `yes` | active real ANN backend; strongest local global-search latency in the current small-noise comparison |
| `hnsw` | `hnsw` | `yes` | active real ANN backend; currently useful as a completed evidence path, though not yet a consistent latency win on the present routed slices |

## Table 7: Update Policy Ablation Snapshot

| Policy | Metric scope | Exact match | Avg query latency | Update time |
| --- | --- | --- | --- | --- |
| `no_update` | future queries | `0.0` | `1.3319ms` | `0.0s` |
| `superseded_update` | future queries | `1.0` | `2.4839ms` | `0.132423s` |
| `naive_append` | future queries | `0.0833` | `2.0401ms` | `0.037093s` |
| `superseded_update` | retention queries | `1.0` | `2.7196ms` | `0.132423s` |

## Table 8: Representative Failure Modes

| Condition | Representative case | Observed answer pattern | Failure mode |
| --- | --- | --- | --- |
| `no_update` | `orchid_updated_value` | returns stale `orchid sync` guidance | base corpus alone cannot answer future-version queries |
| `no_update` | `atlas_updated_value` | returns stale `summary` instead of `adaptive_summary` | stale same-lineage guidance remains active without a write |
| `naive_append` | `ruma_updated_value` | returns both `2` and `4` | old and new same-lineage guidance are blended together |
| `naive_append` | `tidal_updated_value` | returns both `daily` and `hourly_delta` | append-only updates create mixed answers instead of clean supersession |
| `external_naive_append` | `httpx_app_shortcut_updated_value` | returns both removed and pre-removal shortcut guidance | append-only updates create mixed guidance even on the first external corpus |
| `external_no_update` | `fastapi_pydantic_migration_updated_value` | returns the older mixed-support guidance | without a write, external base documents stay stale |

## Table 9: External Corpus Snapshot

| Metric | Current value | Scope |
| --- | --- | --- |
| external corpus documents | `10` | frozen official-source external corpus for the first narrow preprint |
| external corpus conflict docs | `2` | two active FastAPI/Pydantic migration conflict notes |
| base known exact match | `1.0` | `sandbox/external_corpus_benchmark.py` |
| no-update future exact match | `0.0` | external future-version queries |
| superseded update exact match | `1.0` | same-lineage update queries |
| retention exact match | `1.0` | unchanged external guidance queries |
| naive append exact match | `0.6` | append-without-supersession baseline |
| conflict exact match | `1.0` | external conflict query |
| base build time | `0.063783s` | local run |
| superseded update time | `0.018171s` | local run |
| naive append update time | `0.017945s` | local run |
| conflict ingest time | `0.003411s` | local run |

## Table 10: External Answer Quality Snapshot

| Condition | Exact match | Avg query latency | Avg citation count | Avg conflict count |
| --- | --- | --- | --- | --- |
| `base_compositional` | `0.0` | `1.2809ms` | `1.8` | `0.0` |
| `superseded_compositional` | `1.0` | `1.1716ms` | `2.0` | `0.0` |
| `naive_append_compositional` | `0.4` | `1.0595ms` | `2.0` | `0.8` |
| `conflict_synthesis` | `0.5` | `1.2814ms` | `2.5` | `1.0` |
| `overall_quality` | `0.9167` | `n/a` | `2.0833` | `0.1667` |

## Remaining Gaps Before Freeze

- one conflict-heavy answer-synthesis case still fails on the frozen external corpus
- public-release cleanup, tagged release prep, and Zenodo/preprint packaging still remain
