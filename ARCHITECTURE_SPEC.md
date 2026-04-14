# Architecture Spec: RUMA

`RUMA` stands for `Routed Updatable Memory Architecture`.

This is the new working research name for the project.

Legacy names:

- `LSER`
- `Pipeline & Pond`
- `Project Breaking`

Those names are preserved for historical context, but `RUMA` is the preferred technical name going forward because it centers the actual research target:

- routing
- explicit memory
- low-cost updates

## 1. Core Thesis

RUMA is based on a simple idea:

`A language system should not be forced to store all knowledge inside dense weights when some knowledge can be written to, routed through, and read from an explicitly updatable memory layer.`

The architecture is therefore organized around:

1. a small parametric core for contextual reasoning
2. a sparse router for conditional activation
3. updateable memory shards for storing knowledge
4. a fusion path that reintegrates retrieved memory into the current hidden state

## 2. What RUMA Is Not

RUMA is not currently:

- a proven replacement for Transformers
- a proof that catastrophic forgetting is solved
- a proof that sparse routing is faster at useful quality
- a proof of internal reasoning from confidence loops

Any document or script implying those stronger claims should be treated as legacy or hypothesis-level material.

## 3. Design Goals

The architecture should optimize for these goals in order:

1. explicit low-cost updates
2. retention after updates
3. semantic routing quality
4. controllable compute
5. practical systems behavior

If a future design improves speed but breaks updates or retention, it is not on target.

## 4. Main Components

### 4.1 Context Encoder

Purpose:

- read the current token sequence
- build a compact contextual representation
- form the query used for routing and retrieval

Current sandbox analogue:

- `ContextSynthesizer`

### 4.2 Router

Purpose:

- decide which memory shards or experts should be activated
- keep activation sparse and bounded
- preserve semantic locality

Current sandbox analogue:

- `LSHRouter`

Important caution:

The current fixed-random LSH router is only a placeholder baseline, not yet the final routing mechanism.

Current routing default:

- hybrid namespace preselection first
- retrieval performs fine-grained record selection inside chosen namespaces
- routing should stay sparse, top-k bounded, and easy to inspect
- learned routing is deferred until evidence shows the simpler path is the bottleneck

### 4.3 Memory Shards

Purpose:

- hold updateable records outside the main dense weights
- separate recent facts, domain memory, and other stored knowledge into queryable shards

A memory record should eventually carry:

- record id
- key
- payload / value
- shard
- namespace
- source / provenance
- timestamp
- version
- status
- optional confidence

Working layout:

- coarse human-legible namespaces
- finer shards inside each namespace
- sparse cross-namespace retrieval when a query needs multiple knowledge areas
- namespace-banded deterministic shard placement for the current system track
- refresh-based reassignment when shard policy changes or corpus composition shifts

Important constraint:

- memory should be append-first and provenance-preserving, not a blended knowledge soup

Current record granularity default:

- source-grounded chunk or passage records first
- summaries and fact tuples only as derived records later

Legacy term:

- `pond`

Preferred term:

- `memory shard`

### 4.4 Fusion Module

Purpose:

- combine current contextual state with retrieved memory
- avoid hard replacement of the live hidden state
- allow controllable contribution from memory

Current fusion default:

- metadata-aware reranking before fusion
- top-k bounded memory set
- single-hop gated residual fusion first
- multi-hop fusion deferred until later evidence justifies it

### 4.5 Decoder

Purpose:

- produce token logits from the fused state
- remain narrow enough that retrieved memory is doing real work

### 4.6 Updater

Purpose:

- write new memory records without broad retraining
- become the main answer to the "freshness" problem

This is a first-class architectural component, not an optional utility.

Current default:

- append-first writes
- provenance-preserving updates
- explicit supersession and uncertainty states
- no routine destructive overwrite
- same-lineage updates supersede prior active records
- cross-source disagreement is surfaced instead of silently erased
- routine consolidation of external memory into dense weights is deferred for this project stage

## 5. Interface-Level View

The architecture should be reasoned about through stable interfaces:

- `encode(input_ids) -> contextual_states`
- `route(contextual_states) -> shard_ids`
- `fetch(contextual_states, shard_ids) -> memory_states`
- `fuse(contextual_states, memory_states) -> fused_states`
- `decode(fused_states) -> logits`
- `update(new_records) -> memory_write_result`

Future agents should preserve this interface-level separation even if the internals change.

## 6. Current Sandbox Translation

The existing sandbox is best interpreted like this:

- `embedding + ContextSynthesizer`:
  weak context encoder
- `LSHRouter`:
  routing placeholder
- `NeuralPonds`:
  toy memory lookup stand-in
- `valve_out`:
  decoder head

What is missing or underdeveloped:

- first-class updater
- explicit memory metadata
- honest fusion mechanism
- evidence-backed routing evaluation
- non-toy retention testing

## 7. Required Benchmarks

RUMA should eventually be judged against three benchmark families.

### A. Update Benchmarks

Question:

- can the system absorb new facts without full retraining?

### B. Retention Benchmarks

Question:

- does adding new memory damage old behavior less than naive fine-tuning?

### C. Compute/Quality Benchmarks

Question:

- does sparse routing reduce cost without losing too much quality?

Current evaluation default:

- freshness benchmark
- retention benchmark
- retrieval/routing benchmark
- systems-cost benchmark
- comparison against at least one honest non-memory baseline

## 8. Open Research Questions

These are the live unknowns.

- Should routing be fixed, learned, hybrid, or retrieval-assisted?
- Should memory writes happen per token, per chunk, per document, or per concept?
- Should memory shards be domain-specific, geometry-derived, or dynamically balanced?
- How much parametric knowledge should remain in the core versus in memory?
- What fusion mechanism gives the best retention/update tradeoff?
- When does external memory beat fine-tuning, and when does it not?
- What supersession and conflict policy best preserves freshness without hiding disagreement?

## 9. Working Position

The current working position of the repo is:

`RUMA is a routed-memory research architecture aimed at freshness and retention first, not a general declaration that dense Transformer-era designs are obsolete.`

Current retrieval default:

- start with exact namespace-scoped vector search
- use normalized embeddings and inner-product similarity
- adopt HNSW as the first ANN upgrade path
- defer heavier PQ / IVF / GPU retrieval until scale pressure justifies them

Current update default:

- add new records rather than silently overwriting old ones
- preserve lineage for changed facts
- use record statuses such as `active`, `superseded`, `retracted`, and `uncertain`
- treat direct weight edits as baselines or special tools, not the default freshness mechanism
