# RUMA V2 Reverse Architecture Blueprint

This document answers the top-down design question directly:

`If we reverse-engineer the likely mature RUMA architecture from the current literature, what should we actually build next?`

The goal is not to rediscover everything from scratch.
The goal is to:

- borrow the parts the field already understands well
- preserve the parts that are uniquely RUMA
- avoid turning the project into architecture soup
- define a strong reference architecture that can later evolve toward a standalone model

## 1. Core Position

RUMA should still aim, long-term, at a standalone model architecture.

But the right near-term form is:

`a backbone model with repeated RUMA blocks interleaved into it`

That gives us:

- a realistic implementation path
- a clean way to compare against standard backbones
- a clear place where RUMA-specific mechanisms live

So the next serious move is **not**:

- replace the entire model with a brand new block family immediately

It is:

- define a strong reference backbone
- define the RUMA block
- interleave the two deliberately

## 2. What Is Uniquely RUMA

RUMA is **not** just "retrieval plus a base model."

Its distinct center is:

1. routed external memory
2. explicit updateability
3. lineage-aware supersession
4. retained guidance after edits
5. conflict visibility
6. grounded answer/evidence control

These are the pieces that should remain uniquely RUMA even while we borrow heavily elsewhere.

## 3. What We Borrow From Existing Paper Families

We already have enough papers in the repo to define the next architecture without waiting for a new reading sprint.

### 3A. Backbone / Local Sequence Modeling

Borrow from:

- `Attention Is All You Need`
- `BERT`
- `GPT-2`

Purpose:

- stable residual stream
- pre-norm residual block structure
- familiar training dynamics
- easiest serious baseline for the next implementation phase

### 3B. Efficient / Long-Context Backbone Variants

Borrow ideas from:

- `Mamba`
- `Mamba-2`
- `Jamba`
- `Hyena`
- `RetNet`
- `Longformer`
- `Linformer`
- `RWKV`

Purpose:

- future backbone swaps or hybrids
- memory-efficient local sequence processing
- long-context efficiency

Decision:

- these should influence the roadmap
- they should **not** replace the first serious reference backbone yet

### 3C. Retrieval and External Memory

Borrow from:

- `REALM`
- `RETRO`
- `Improving language models by retrieving`
- `Atlas`
- `FiD`-style retrieval-reader thinking
- `Memorizing Transformers`
- `Memory Networks`
- `End-to-End Memory Networks`
- `Neural Turing Machines`
- `Learning to Remember Rare Events`

Purpose:

- retrieval-augmented modeling
- explicit memory access
- memory-conditioned computation
- memory at inference without retraining everything

### 3D. Editing / Continual Learning

Borrow from:

- `ROME`
- `MEMIT`
- `MEND`
- `Locating and Editing Factual Associations in GPT`
- `Knowledge Editing for Large Language Models`
- `Learning without Forgetting`
- `Overcoming Catastrophic Forgetting`
- `Gradient Episodic Memory`

Purpose:

- clear baseline families for edits
- locality and generality thinking
- selective retention after updates
- explicit contrast between memory-based and parameter-editing paths

### 3E. Sparse Routing / Capacity Expansion

Borrow from:

- `Switch Transformers`
- `Switch Transformer`
- `GShard`
- `GLaM`
- `Mixtral`-style sparse activation thinking
- `Jamba`

Purpose:

- sparse routing
- expert specialization
- larger capacity without activating everything

Decision:

- sparse experts should enter **inside** the RUMA block first
- not as a full-model global default yet

## 4. The Chosen Reference Architecture

The first serious reverse-designed architecture should be:

`Transformer-backed RUMA with interleaved RUMA blocks`

Why this first:

- easiest to compare
- easiest to stabilize
- easiest to formalize
- easiest to attach to current local infrastructure
- still compatible with later Mamba/Jamba-style hybrids

## 5. Layer Families

The mature architecture should use **three** layer families.

### 5A. Backbone Layers

These are ordinary sequence-modeling blocks.

Job:

- maintain the token-level residual stream
- model local and medium-range context
- provide a stable hidden-state interface to memory

Reference first choice:

- Transformer pre-norm residual blocks

### 5B. RUMA Blocks

These are the main architecture contribution.

Job:

- decide when memory is needed
- route into memory
- retrieve and rerank evidence
- apply lineage/supersession logic
- fuse evidence back into the residual stream
- expose evidence sufficiency / conflict control

### 5C. Control Heads

These are lightweight but crucial.

Job:

- abstention
- evidence sufficiency
- conflict exposure
- calibration

These should stay explicit instead of being hidden inside a giant decoder stack.

## 6. Reference Layer Schedule

Start with a moderate reference stack, not an overgrown one.

Recommended first serious schedule:

- `12` total main blocks
- `8` backbone blocks
- `4` interleaved RUMA blocks

One practical arrangement:

1. Backbone
2. Backbone
3. RUMA
4. Backbone
5. Backbone
6. RUMA
7. Backbone
8. Backbone
9. RUMA
10. Backbone
11. Backbone
12. RUMA

This gives us:

- repeated memory interaction
- enough depth between memory accesses
- a clean ablation path versus a plain 12-block baseline

## 7. The RUMA Block

The first serious RUMA block should contain these sublayers:

1. `Query Projector`
   - maps the current residual state into a memory-query representation

2. `Sparse Namespace Router`
   - chooses a small subset of namespaces / memory regions
   - top-k sparse gating, not full dense search everywhere

3. `Hybrid Retriever`
   - dense + lexical retrieval
   - this is already justified by current benchmark evidence

4. `Reranker / Evidence Selector`
   - lightweight reranking before fusion
   - this is also already justified by evidence

5. `Lineage / Supersession Filter`
   - apply RUMA-specific rules:
   - prefer active same-lineage updates
   - surface cross-source disagreement
   - do not silently flatten conflicts

6. `Memory Fusion Layer`
   - inject retrieved evidence into the live residual state
   - first version should be gated and simple, not huge cross-attention everywhere

7. `Evidence Sufficiency / Conflict Head`
   - explicit score or head for:
   - enough evidence?
   - conflicting evidence?
   - abstain?

8. `Residual Merge`
   - merge the memory-conditioned signal back into the main hidden state

## 8. Where MoE Belongs

MoE should not be sprayed across the entire model first.

The most justified first MoE insertion points are:

1. `Router experts`
   - different experts specialize in domain / task routing

2. `Fusion experts`
   - different experts specialize in how evidence is combined

3. `Memory-type experts`
   - for example:
   - factual edits
   - code/doc updates
   - multi-hop evidence
   - verification/conflict-heavy cases

So the correct MoE stance is:

- `yes`, likely valuable later
- `no`, not global by default yet
- `yes`, inside RUMA blocks first

Recommended first sparse design if we add it:

- Switch-style top-1 or top-2 expert routing
- only on selected RUMA sublayers
- not on every layer of the full model

## 9. Why Not Go Full Jamba/Mamba Right Now

Because that would mix two different research questions too early:

1. what is the best local sequence backbone?
2. what is the right memory/update architecture?

If we change both at once, we learn less.

So the right order is:

1. prove the RUMA block on a stable backbone
2. then compare backbone variants later:
   - Transformer-backed RUMA
   - Mamba-backed RUMA
   - hybrid-backed RUMA

That is faster scientifically, even if it feels less flashy.

## 10. Namespace / Memory Shard Design

This should become **simpler and more principled**, not more manually complicated.

We should move away from vague human category piles like:

- math
- science
- code
- english language

Those are too coarse, and they also do not line up well with update lineage.

Instead use three cleaner axes:

### 10A. Domain Namespace

Examples:

- `wiki.biography`
- `docs.python.fastapi`
- `qa.hotpotqa`
- `factcheck.fever`
- `code.api`

### 10B. Topic / Entity Namespace

Examples:

- `entity.lionel_messi`
- `entity.fastapi`
- `topic.pydantic.v2`

### 10C. Lineage Namespace

Examples:

- `lineage.fastapi.python_support`
- `lineage.httpx.proxy_behavior`
- `lineage.fact.us_president`

The important design rule is:

- keep routing sparse
- keep namespace assignment semi-structured
- do not explode the system into a giant hand-maintained ontology

So yes, simplify the current mental model.

The mature namespace system should be:

- sparse
- compositional
- partly auto-derived
- lineage-aware

## 11. What We Remove Or Avoid

To keep the architecture serious and interpretable, avoid:

- giant hand-authored namespace taxonomies
- blind full-model MoE
- random attention variants without a measured reason
- parameter-editing as the primary update path
- silent fusion that destroys provenance
- overcomplicated first-pass fusion stacks

## 12. Phased Architecture Ladder

### Phase A: Reference Hybrid

`Transformer backbone + 4 RUMA blocks`

Features:

- hybrid retrieval
- reranking
- lineage/supersession
- explicit evidence control

This should be the next serious implementation target.

### Phase B: Sparse RUMA

Add:

- sparse expert routing inside RUMA blocks

Not yet:

- full-model MoE

### Phase C: Backbone Alternatives

Compare:

- Transformer-backed RUMA
- Mamba-backed RUMA
- hybrid Transformer/Mamba-backed RUMA

### Phase D: Standalone-Capable RUMA Family

Only after evidence earns it:

- stronger trainable fusion
- more direct memory-conditioned token modeling
- broader standalone architecture claim

## 13. The Mathematical Priority

The next math should formalize this architecture directly.

Not decorative math.

We need equations for:

- residual state
- query projection
- namespace gating
- hybrid retrieval
- reranking
- lineage/supersession filtering
- fusion
- evidence sufficiency / abstention
- update operator

MoE-specific math comes later, only once it is promoted into the architecture.

## 14. Immediate Build Order

The next serious build order should be:

1. keep the stronger hybrid + reranked retrieval stack
2. add a better evidence-sufficiency / calibration layer
3. define the first explicit RUMA block interface in code
4. interleave that block with a simple backbone path
5. compare against the current non-interleaved attachment baseline

## 15. Final Position

The project should now proceed with this mindset:

- use the literature in reverse
- do not rediscover solved infrastructure
- do not overfit to one famous paper
- keep the unique RUMA ideas central
- deepen architecture only where the evidence points

So the architectural answer is:

`RUMA should now become a Transformer-backed, interleaved memory architecture with repeated RUMA blocks, explicit evidence control, and later sparse experts inside the RUMA block.`

That is the cleanest serious next step toward a standalone architecture family.
