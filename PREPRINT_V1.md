# RUMA: Routed Updatable Memory Architecture for Versioned Document Updates

**Author:** Dr. Vatsal Chavda  
**Affiliation:** Independent Researcher  
**ORCID:** https://orcid.org/0009-0005-8789-7301

## AI Assistance Disclosure

AI-assisted drafting and editing were used during development of this manuscript and associated research materials. Final architectural decisions, claim boundaries, experimental choices, and content selection remain under the author's control.

## Abstract

Keeping language systems current is expensive, and standard retrieval layers often do not distinguish cleanly between superseded guidance and genuinely conflicting active sources. This work presents RUMA, a routed updatable memory architecture that stores chunk-level records outside dense weights, applies append-first updates with lineage and provenance metadata, and answers through a citation-first grounded retrieval layer. The current implementation targets controlled changing-document settings rather than general-purpose generation.

Across synthetic update tasks, held-out real-text chunk tasks, an expanded versioned-manual corpus, a frozen external corpus built from official-source Python ecosystem documents, and new routing and retrieval benchmarks, explicit same-lineage supersession consistently outperforms no-update and naive append baselines while preserving retained guidance. On a harder answer-quality pass over the frozen external corpus, superseded updates reach compositional exact match `1.0`, while the base-only system remains at `0.0`, naive append reaches `0.4`, and one conflict-heavy synthesis case still fails. These results do not establish a universal replacement for dense language model architectures, but they do support a narrower claim: version-aware routed external memory is a promising path for low-cost document updates with inspectable evidence and controlled conflict handling on the current controlled corpora.

## 1. Introduction

Modern language systems are expensive to refresh. Once knowledge has been absorbed into dense weights, updating it often means retraining, fine-tuning, or relying on an external retrieval layer that may not encode whether a newer source should supersede an older one. In practice, this creates a messy combination of stale parametric knowledge, ad hoc retrieval, weak provenance, and limited visibility into whether disagreement reflects genuine conflict or obsolete instruction.

RUMA narrows that problem. It does not claim to solve hallucination, sycophancy, or general reasoning. Instead, it asks a smaller and more defensible question: can a routed external memory layer absorb versioned document updates more cleanly than naive alternatives while preserving retained guidance and surfacing conflicts instead of silently blending them away?

The first preprint positions RUMA as a memory-systems paper with controlled answering, not as a standalone frontier LLM. The contribution is the version-aware update path: chunk-level records, namespace-aware routing, append-first updates, explicit lineage tracking, supersession states, and citation-first grounded answers.

## 2. System Overview

RUMA currently consists of:

- a lightweight query/context path
- namespace-aware routing
- external memory records with provenance and lineage
- append-first updates with supersession states
- citation-first grounded answering

The current memory model is chunk-first, source-grounded, timestamped, lineage-aware, and status-tracked. Same-lineage replacements supersede older records. Cross-source disagreement remains visible rather than being silently flattened away.

## 3. Experimental Setting

The current evidence ladder includes:

- synthetic factual update benchmarks
- repeated supersession benchmarks
- code-flavored update benchmarks
- documentation chunk benchmarks
- real-text chunk benchmarks on `alice.txt`
- a versioned-manual corpus benchmark
- routing and retrieval scaling benchmarks
- a frozen external corpus built from official-source Python ecosystem documents
- a harder answer-quality benchmark over that external corpus

The external corpus used for the first preprint is now frozen at:

- `10` official-source base documents
- `2` active conflict notes

This freeze is intentionally narrow so the first claim stays reproducible and bounded.

## 4. Main Results

### 4.1 Core External-Corpus Update Results

| Benchmark | Base-known EM | No-update future EM | Superseded update EM | Retention EM | Naive append EM | Conflict EM |
| --- | --- | --- | --- | --- | --- | --- |
| `sandbox/external_corpus_benchmark.py` | `1.0` | `0.0` | `1.0` | `1.0` | `0.6` | `1.0` |

These results show the central pattern of the paper: once a same-lineage update is written with explicit supersession, future-version queries become answerable without destroying retained unchanged guidance. By contrast, leaving the corpus untouched fails on future-version queries, and naive append mixes old and new guidance.

### 4.2 Harder Answer-Quality Results

| Condition | Exact match | Avg. latency | Avg. citations | Avg. conflict count |
| --- | --- | --- | --- | --- |
| `base_compositional` | `0.0` | `1.2809ms` | `1.8` | `0.0` |
| `superseded_compositional` | `1.0` | `1.1716ms` | `2.0` | `0.0` |
| `naive_append_compositional` | `0.4` | `1.0595ms` | `2.0` | `0.8` |
| `conflict_synthesis` | `0.5` | `1.2814ms` | `2.5` | `1.0` |
| `overall_quality` | `0.9167` | `n/a` | `2.0833` | `0.1667` |

This benchmark is healthier than the earlier clean wins because it exposes a real remaining weakness: one conflict-heavy synthesis case still fails to express the full "dropped support" phrasing even while surfacing the relevant sources.

### 4.3 Routing and Retrieval Snapshot

| Evaluation | Main finding |
| --- | --- |
| `sandbox/multi_namespace_manual_benchmark.py` | global, keyword-routed, and oracle routing all reach exact match `1.0`; simple routing cuts latency from `2.0184ms` to `1.3576ms` |
| `sandbox/retrieval_scaling_benchmark.py` | global exact search rises from `1.4648ms` to `3.2134ms` as distractor noise grows from `0` to `1000`; routed exact search stays around `1.0236ms` to `1.4354ms` at the same exact-match ceiling |
| `sandbox/ann_backend_benchmark.py` | `exact`, `faiss_flat`, and `hnsw` are all active in the local runtime; ANN is real infrastructure now, but not yet the main scientific claim |

The current evidence therefore supports the project's present borrowing policy: retrieval infrastructure is the next borrowed layer, not a heavier learned router.

## 5. What This Paper Claims

This paper claims only the following:

- RUMA is a routed updatable memory architecture for versioned document updates.
- Same-lineage supersession works better than no-update and naive append on the current controlled corpora.
- Retained guidance survives updates on the current controlled corpora.
- Conflict visibility is inspectable, even though conflict synthesis is not fully solved.
- Version-aware routed external memory is a promising path for low-cost document updates with explicit provenance.

This paper does **not** claim:

- that RUMA solves hallucination
- that RUMA solves sycophancy
- that RUMA is already a standalone frontier LLM
- that RUMA is a universal replacement for transformer systems

## 6. Limitations

RUMA is still an early research system. The current results should be read narrowly.

First, the real-document path currently relies on citation-first grounded answering rather than a strong generative decoder. Second, the corpora are still controlled and relatively small compared with heterogeneous enterprise or open-web freshness settings. Third, the current latency and systems measurements are local prototype numbers, not production claims. Fourth, although both `faiss_flat` and `hnsw` are live in the local runtime, ANN is being treated as bounded infrastructure rather than as the main contribution. Finally, one conflict-heavy synthesis case still fails, and broader ablations on routing, retrieval, update policy, and answer behavior remain future work.

## 7. Artifact Availability

The first preprint should ship with:

- the frozen manuscript
- generated tables and figure sources in `paper_assets/`
- benchmark JSON artifacts in `sandbox/results/`
- a code release tagged to a stable public GitHub snapshot

The two figure sources already prepared in the repo are:

- `paper_assets/figure_ruma_architecture.mmd`
- `paper_assets/figure_benchmark_flow.mmd`

For the first preprint, clean vector/manual rendering is preferred. Mermaid source is sufficient as the publication source of truth.

## 8. Conclusion

RUMA is not yet a standalone foundation-model replacement. What it does offer, on the current controlled corpora, is a disciplined and inspectable update path for changing documents: explicit records, explicit lineage, explicit supersession, and citation-first answers. That narrower result is the appropriate claim for version 1 of the preprint.

## 9. Selected References

1. Lewis, P. et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. arXiv:2005.11401. https://arxiv.org/abs/2005.11401
2. Borgeaud, S. et al. *Improving language models by retrieving from trillions of tokens*. arXiv:2112.04426. https://arxiv.org/abs/2112.04426
3. Khandelwal, U. et al. *Generalization through Memorization: Nearest Neighbor Language Models*. arXiv:1911.00172. https://arxiv.org/abs/1911.00172
4. Wu, Y. et al. *Memorizing Transformers*. arXiv:2203.08913. https://arxiv.org/abs/2203.08913
5. Fedus, W., Zoph, B., and Shazeer, N. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. arXiv:2101.03961. https://arxiv.org/abs/2101.03961
6. Malkov, Y. A. and Yashunin, D. A. *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*. arXiv:1603.09320. https://arxiv.org/abs/1603.09320
7. Johnson, J., Douze, M., and Jegou, H. *Billion-scale similarity search with GPUs*. arXiv:1702.08734. https://arxiv.org/abs/1702.08734
8. Kirkpatrick, J. et al. *Overcoming catastrophic forgetting in neural networks*. PNAS, 2017. https://www.pnas.org/doi/10.1073/pnas.1611835114
9. Meng, K. et al. *Locating and Editing Factual Associations in GPT*. arXiv:2202.05262. https://arxiv.org/abs/2202.05262
10. Meng, K. et al. *MEMIT: Mass-Editing Memory in a Transformer*. arXiv:2210.07229. https://arxiv.org/abs/2210.07229
