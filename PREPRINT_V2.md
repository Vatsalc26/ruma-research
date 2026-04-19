# RUMA V2: Interleaved Routed Updatable Memory for Explicit Updates, Retained Guidance, and Grounded Evidence Control

**Author:** Dr. Vatsal Chavda  
**Affiliation:** Independent Researcher  
**ORCID:** https://orcid.org/0009-0005-8789-7301

## AI Assistance Disclosure

AI-assisted drafting and editing were used during development of this manuscript and the associated research materials. Final architectural decisions, experimental choices, claim boundaries, and release judgments remain under the author's control.

## Abstract

Large language systems still handle changing knowledge awkwardly. Retraining is expensive, naive retrieval mixes incompatible guidance, and lightweight override schemes often destroy selective retention. This paper studies `RUMA V2`, an interleaved routed updatable memory architecture designed for explicit updates, retained guidance, stale-memory suppression, and grounded evidence control. The public `V2` reference system interleaves `RUMA` blocks with a residual language-model backbone and equips each block with routed retrieval, fallback retrieval, reranking, lineage-aware filtering, trainable selectivity, sparse expert fusion, and residual merge.

Across controlled update benchmarks, the small standalone reference model reaches update exact match `1.0`, retention exact match `1.0`, and stale suppression `1.0` on bounded factual and code-update tasks, while the document-chunk update benchmark reaches the same first-update and retention targets. On official edit tasks, `RUMA supersession` reaches `CounterFact` canonical update `0.9922`, paraphrase `1.0`, retention `0.9938`, and `zsRE` canonical update `0.9609`, paraphrase `0.9609`, retention `1.0`. On grounded evidence tasks, the same architecture family reaches `FEVER` evidence recall@5 `0.9609` with answer hit `0.7656`, and `HotpotQA` full support-chain hit@8 `0.8359` with answer hit `0.5312`.

Broader standalone text generation remains more uneven. A comparison backbone based on selective state-space layers improves held-out small-model text retention and continuation, while the public reference backbone remains stronger on old-text suppression and grounded controller behavior. The resulting claim is narrower than a new-foundation-model claim but still substantive: `RUMA V2` is a credible interleaved architecture family for controlled knowledge change and grounded evidence use, and broader standalone natural-language scaling remains the main open challenge.

## 1. Introduction

The central problem behind `RUMA` is not generic retrieval. It is controlled knowledge change.

When guidance changes, a useful language system should be able to insert updated information, retain unaffected information, suppress stale superseded guidance, surface conflicts honestly, and answer with inspectable evidence. Existing low-cost alternatives do not satisfy these objectives at once. Append-only retrieval preserves history but mixes incompatible guidance. Subject-level overrides can improve update accuracy but often destroy selective retention. Narrow parameter-editing baselines can alter outputs, but they do not naturally provide explicit lineage, conflict visibility, or inspectable memory state [4, 5, 6, 7, 8].

`RUMA V1` established a bounded systems result for versioned document updates. `RUMA V2` asks a stronger question: can routed updatable memory become a real model-architecture component rather than only an external retrieval wrapper?

This paper answers that question at the architecture-family level rather than at the frontier-foundation-model level. The contribution is an interleaved memory architecture that combines explicit write-path supersession, routed retrieval, lineage-aware filtering, and grounded answer control inside a standalone-capable small-model scaffold.

The main contributions of the `V2` package are:

1. an interleaved routed updatable memory architecture with explicit write-path supersession, lineage-aware retrieval, trainable selectivity, sparse expert fusion, and grounded controller logic
2. a layered evaluation package spanning controlled update benchmarks, official edit tasks, grounded verification, multi-hop evidence recovery, and standalone small-model text maturity
3. an architecture-family result showing that explicit updatable memory can support a real standalone training path, while broader natural-language scaling remains the principal open problem

## 2. Positioning Relative to Prior Work

`RUMA V2` sits at the intersection of three research directions.

First, retrieval-augmented and memory-augmented systems improve factual grounding by consulting external evidence at inference time [4, 5, 6]. Those systems demonstrate the value of retrieval, but they do not by themselves define how changed guidance should supersede older guidance while preserving unaffected information. `RUMA` adopts retrieval as one mechanism inside a broader update-control architecture rather than treating retrieval as the whole story.

Second, parameter-editing methods target direct knowledge modification inside model weights [7, 8]. These methods are useful baselines for local factual change, but they do not naturally expose lineage, provenance, conflict state, or inspectable memory records. `RUMA` therefore treats parameter editing as an auxiliary comparison path rather than the main architectural answer.

Third, backbone innovations such as self-attention architectures and selective state-space architectures improve sequence modeling itself [1, 2, 3]. Those backbone choices matter inside `RUMA`, but they are not the core contribution of this paper. The main question here is whether explicit, routed, updatable memory can be made into a coherent interleaved model component. Backbone choice is treated as an internal tradeoff within that larger claim.

## 3. Architecture

The public `V2` reference system uses an interleaved residual backbone with:

- `12` total blocks
- `8` backbone blocks
- `4` interleaved `RUMA` blocks

Each `RUMA` block contains:

1. query projection
2. sparse routing
3. routed retrieval plus global fallback
4. reranking
5. lineage-, anchor-, and entity-aware filtering
6. trainable selectivity and controller logic
7. sparse expert fusion
8. residual merge back into the hidden state

The write path performs explicit same-lineage supersession and preserves provenance metadata. The answer/control path uses grounded retrieval and evidence gating rather than blind memory dominance. The public reference backbone follows the residual self-attention tradition, while the comparison configuration tests whether selective state-space modeling can improve broader standalone text behavior within the same `RUMA` scaffold [1, 2, 3].

The repo also contains a comparison configuration based on selective state-space layers for the broader standalone-text path [3]. In the `V2` package, that comparison system is reported as an internal architecture-family result rather than as the frozen public mainline system.

## 4. Experimental Protocol

The `V2` package is organized around the core claim that explicit updatable memory should support controlled knowledge change and grounded evidence use.

### 4.1 Controlled Update and Editing Tasks

The controlled update package includes:

- bounded factual update benchmark
- bounded code-update benchmark
- bounded document-chunk update benchmark
- `CounterFact` [7]
- `zsRE` [12]
- external versioned-document benchmark

These tasks test update success, retained guidance, stale-memory suppression, and second-update behavior under explicit change. The official editing tasks follow the `CounterFact` and `zsRE` evaluation tradition used in recent knowledge-editing work [7, 12].

### 4.2 Grounded Verification and Multi-hop Evidence Tasks

The grounded evaluation package includes:

- `FEVER` [9]
- bounded `BEIR` subsets (`FEVER`, `HotpotQA`, `NQ`, `SciFact`) [11]
- external versioned-document answer benchmark
- `HotpotQA` [10]

These tasks test whether the same architecture family transfers from explicit updates to evidence-aware verification and multi-hop support recovery. The core external benchmarks are drawn from `FEVER`, `HotpotQA`, and `BEIR`-style evaluation [9, 10, 11].

### 4.3 Standalone Small-Model Text Maturity

The broader standalone package evaluates held-out small-model continuation and retention behavior on two harder natural-language slices:

- a literary continuation slice
- a repository-markdown continuation slice

These held-out slices are not the main claim of `V2`, but they are important for judging whether interleaved `RUMA` can support a real standalone training path rather than only a controller-style wrapper.

### 4.4 Ablations

The first serious ablation package includes:

- reference-backbone versus selective-state-space-backbone comparison
- no-lineage ablation
- no-selectivity ablation
- no-sparse-expert ablation
- parameter-editing auxiliary path

## 5. Results

### 5.1 Controlled Update Stability

The controlled standalone stabilization suite shows that the final-form interleaved model retains the core updateability behavior:

- bounded factual update benchmark: first-update exact `1.0`, retention exact `1.0`, stale suppression `1.0`, second-update exact `1.0`
- bounded code-update benchmark: first-update exact `1.0`, retention exact `1.0`, stale suppression `1.0`, second-update exact `1.0`
- bounded document-chunk update benchmark: first-update exact `1.0`, retention exact `1.0`, stale suppression `1.0`, second-update exact `0.6667`

These results matter because they show that the interleaved architecture is not only a systems wrapper. It can be trained as a small standalone reference model while preserving update, retention, and stale-suppression objectives.

### 5.2 Official Edit Tasks

On the official edit harness:

- `CounterFact`: `RUMA supersession` canonical update `0.9922`, paraphrase `1.0`, retention `0.9938`
- `zsRE`: `RUMA supersession` canonical update `0.9609`, paraphrase `0.9609`, retention `1.0`

The relevant contrast is not only with no-update baselines. Cheap alternatives still fail important pieces of the objective:

- naive append collapses on update accuracy
- subject-level override collapses on selective retention
- parameter editing remains materially weaker than `RUMA` on retention-sensitive standalone slices

### 5.3 Grounded Verification and Multi-hop Answering

The broader controller-based package shows that the same architecture family transfers beyond controlled update tasks.

On `FEVER`, the interleaved controller path reaches:

- evidence recall@5 `0.9609`
- page recall@5 `0.9922`
- answer hit `0.7656`
- `NEI` abstain `0.4531`

On `HotpotQA`, the interleaved controller path reaches:

- supporting-fact recall@8 `0.8005`
- full support-chain hit@8 `0.8359`
- answer hit `0.5312`

These results do not prove that grounded answer control is solved. They do show that interleaved control substantially strengthens evidence use and multi-hop support recovery relative to earlier retrieval-only paths.

### 5.4 Standalone Small-Model Text Maturity

The broader standalone text package yields the following rolled-up macro result:

| System | Update teacher EM | Retention teacher EM | Update generation EM | Retention generation EM | Update generation token acc | Retention generation token acc | Old-text suppression |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `RUMA (reference self-attention backbone)` | `0.5` | `0.0` | `0.5625` | `0.0` | `0.6603` | `0.1384` | `1.0` |
| `RUMA (selective state-space comparison backbone)` | `0.5625` | `0.4167` | `0.625` | `0.5416` | `0.7712` | `0.686` | `0.75` |

Interpretation:

- the standalone interleaved story is now real beyond the earlier failure case on literary continuation
- the selective state-space comparison backbone is materially stronger on the current text-maturity suite
- that gain is not free, because stale and old-text suppression weaken relative to the public reference backbone

The broader standalone story is therefore no longer "not yet working." It is a meaningful but incomplete result: `RUMA` can support a standalone small-model path, but the best continuation-oriented backbone and the cleanest suppression-oriented backbone are not yet the same system.

## 6. Ablations and Interpretation

### 6.1 Lineage

Removing lineage hurts stale-memory suppression materially. On the current ablation summary, the reference system improves old-fact suppression by `+0.4167` over the no-lineage ablation. Under broader text pressure, the no-lineage path can still score well on update generation while collapsing on retention generation. Lineage is therefore not decorative metadata; it is a core mechanism.

### 6.2 Selectivity and Sparse Experts

The current exact-match ablations do not show large direct gains from selectivity and sparse experts on the controlled symbolic-style tasks alone. Their stronger justification comes from broader natural-language calibration and answer-control settings. The mechanisms belong in the architecture for principled reasons, but their full payoff is more visible in the broader standalone and answer-level path than in the narrowest update slices.

### 6.3 Backbone Tradeoff

The backbone comparison matters, but it is not the identity of the paper. On the controlled stabilization tasks, the reference self-attention backbone and the selective state-space comparison backbone are similarly strong. On the broader standalone text package, the selective state-space comparison becomes materially stronger on retention and continuation, while the reference backbone remains cleaner on old-text suppression and grounded controller behavior.

In practical terms, the `V2` release keeps the self-attention system as the public reference backbone and reports the selective state-space configuration as an internal comparison result within the same architecture family.

## 7. Claim Boundary and Limitations

This paper does **not** claim:

- that `RUMA V2` is already a broad standalone foundation model
- that `RUMA` replaces generic self-attention language-model systems
- that the current backbone comparison is permanently resolved
- that `RUMA` solves hallucination broadly

This paper **does** claim:

- `RUMA V2` is a serious interleaved architecture for explicit updates
- it is strong on retained guidance and stale-memory suppression over controlled structured update tasks
- it improves grounded verification and multi-hop evidence behavior in the broader controller-based evaluation package
- it supports a real small-model standalone training path, even though broader natural-language capability remains uneven

The main limitations follow directly from these boundaries. The current package is still a small-model research program rather than a large-scale deployment story. Standalone text behavior remains uneven across held-out text families. The grounded controller is strong on evidence use but still imperfect on abstention and replacement control. The strongest continuation-oriented backbone and the strongest suppression-oriented backbone are not yet the same configuration. Broader standalone natural-language calibration remains the main open challenge for the later flagship path.

## 8. Artifact Availability and Reproducibility

The release-facing `V2` package should ship with:

- this manuscript
- the public code release tagged to a stable GitHub snapshot
- the `sandbox/` codebase and result artifacts relevant to the `V2` claim
- generated paper assets under `paper_assets/`
- curated versioned-manual corpora
- curated external Python-ecosystem update corpus

Public code repository:

- `https://github.com/Vatsalc26/ruma-research`

The figure sources used by the manuscript are:

- `paper_assets/figure_ruma_architecture.mmd`
- `paper_assets/figure_benchmark_flow.mmd`

The practical release gate and package boundaries are defined in:

- [RUMA_V2_RELEASE_GATE.md](./RUMA_V2_RELEASE_GATE.md)
- [PUBLIC_RELEASE_CONTENTS.md](./PUBLIC_RELEASE_CONTENTS.md)
- [README_PUBLIC.md](./README_PUBLIC.md)

## 9. Conclusion

`RUMA V2` does not yet justify a new universal-backbone claim. It does justify a substantially stronger claim than the original narrow systems preprint: explicit routed updatable memory can function as a real interleaved architecture component, preserve selective retention, suppress stale guidance, and support stronger grounded evidence control across official edit tasks, verification, multi-hop behavior, and controlled standalone training.

These results support treating `RUMA V2` as a serious architecture-family result for controlled knowledge change and grounded evidence use.

## 10. Selected References

1. Vaswani, A. et al. *Attention Is All You Need*. NeurIPS 2017. https://arxiv.org/abs/1706.03762
2. Devlin, J. et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT 2019. https://arxiv.org/abs/1810.04805
3. Gu, A. and Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752. https://arxiv.org/abs/2312.00752
4. Lewis, P. et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. arXiv:2005.11401. https://arxiv.org/abs/2005.11401
5. Borgeaud, S. et al. *Improving language models by retrieving from trillions of tokens*. arXiv:2112.04426. https://arxiv.org/abs/2112.04426
6. Wu, Y. et al. *Memorizing Transformers*. arXiv:2203.08913. https://arxiv.org/abs/2203.08913
7. Meng, K. et al. *Locating and Editing Factual Associations in GPT*. arXiv:2202.05262. https://arxiv.org/abs/2202.05262
8. Meng, K. et al. *MEMIT: Mass-Editing Memory in a Transformer*. arXiv:2210.07229. https://arxiv.org/abs/2210.07229
9. Thorne, J. et al. *FEVER: a Large-scale Dataset for Fact Extraction and VERification*. NAACL-HLT 2018. https://aclanthology.org/N18-1074/
10. Yang, Z. et al. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. EMNLP 2018. https://aclanthology.org/D18-1259/
11. Thakur, N. et al. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. NeurIPS Datasets and Benchmarks 2021. https://arxiv.org/abs/2104.08663
12. Levy, O. et al. *Zero-Shot Relation Extraction via Reading Comprehension*. CoNLL 2017. https://aclanthology.org/K17-1034/
