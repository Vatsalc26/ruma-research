# Project Lens / Research Doctrine

This document is the operating doctrine for Project Breaking and for any future agent or collaborator working on the architecture now referred to as `RUMA` (`Routed Updatable Memory Architecture`).

The short version:

- Critique the idea first.
- Borrow working components aggressively from existing architectures.
- Invent only where the bottleneck remains unsolved.
- Treat all current strong claims as hypotheses until revalidated.
- Build the project around three core claims:
  - updatable memory without full retraining
  - sparse routing with controllable compute
  - quality retention after updates

Working name:

- `RUMA` is the preferred name going forward.
- `LSER` and `Pipeline & Pond` are legacy names retained only for historical context.

## 1. What This Project Is

This project is not yet a proven successor to the Transformer.

It is a research program around a more modular language architecture:

- a small parametric reasoning core
- a sparse routing mechanism
- an explicitly updatable memory layer
- a lightweight fusion / decoding path

The project should be treated as a serious hypothesis generator and prototype sandbox.

## 2. What Problem We Are Actually Solving

We must not let the architecture become a vague "better AI" idea.

The main target problems are:

1. Knowledge freshness:
   adding new information without full retraining
2. Update efficiency:
   making updates cheaper than dense weight adaptation
3. Interference resistance:
   reducing damage to prior capabilities after updates
4. Controllable compute:
   activating only the memory or experts needed for a query

If a proposed module does not clearly help one of those, it is probably architectural drift.

## 3. First-Principles View

The project should be evaluated from first principles:

- Dense parametric weights are expensive to retrain.
- Dense attention and dense feed-forward execution spend compute on everything, even when only part of the system is relevant.
- New knowledge should ideally be inserted through a cheaper path than full model retraining.
- External or semi-external memory can, in principle, absorb freshness better than weights alone.
- Sparse activation is only useful if it preserves quality.

From this lens, the strongest form of the idea is not "magic ponds" or "millions of heads."

The strongest form is:

`small reasoning core + sparse routing + explicit updatable memory + lightweight fusion`

## 4. Architectural Interpretation

The original "pond" metaphor is useful for intuition, but the technical design must be more precise.

For this repo, a `pond` should be interpreted as a memory shard or routed memory pool.

A pond may contain some combination of:

- static knowledge blocks
- recently inserted facts
- domain-specific memory
- learned summaries or embeddings
- cached retrieval entries

But each stored item should eventually have:

- provenance
- timestamp or version
- routing key or address
- namespace or shard identity

Without those, "continuous leaking" becomes an uncontrolled metaphor instead of a system.

## 5. Invariants

These are the non-negotiable invariants for future work.

### Functional invariants

- New information must be insertable through an explicit update path.
- That update path must be cheaper than broad model retraining.
- Retrieval or memory access must materially improve outputs on held-out tasks.
- Old capabilities should not collapse after adding new knowledge.
- Routing must preserve semantics, not just distribute load.

### Scientific invariants

- A claim is not accepted without a baseline.
- A speed claim is not accepted without matched-quality comparison or matched-compute comparison.
- A forgetting claim is not accepted on toy examples alone.
- A reasoning claim is not accepted if the mechanism is simulated or random.
- A memory claim is not accepted if the system is only using hidden parametric memorization.

### Repo invariants

- Use the word `hypothesis` by default, not `proof`, unless evidence is strong.
- Separate idea documents from validated results.
- Keep a clear line between toy sandbox demos and architecture claims.

## 6. Evidence Standard

Every important claim needs:

1. A specific hypothesis
2. A metric
3. A baseline
4. An ablation
5. A failure analysis

Examples:

- Hypothesis: memory updates improve factual freshness without retraining
  - Metric: accuracy on inserted facts before and after memory insertion
  - Baseline: same model without memory insertion
  - Ablation: different routing policies, different shard layouts

- Hypothesis: sparse routing reduces compute without harming quality
  - Metric: latency, memory, throughput, accuracy/perplexity
  - Baseline: dense transformer or dense retrieval baseline at similar scale
  - Ablation: number of active shards, routing top-k, fusion depth

- Hypothesis: updates do not destroy prior capability
  - Metric: pre-update task score vs post-update task score
  - Baseline: fine-tuning baseline and frozen-model baseline
  - Ablation: update frequency, memory capacity, routing isolation

## 7. Failure Modes To Watch

This project is especially vulnerable to these traps:

- Toy-demo overinterpretation:
  memorizing two phrases is not continual learning
- Benchmark unfairness:
  comparing against a weak or badly optimized baseline
- Retrieval leakage:
  the system may look smart because the answer is already nearby
- Simulated mechanisms:
  random confidence loops are not evidence of reasoning
- Semantic collapse:
  routing may become random load balancing without useful specialization
- Architectural blur:
  mixing experts, retrieval, continual learning, and decoding into one vague module

Each future experiment should explicitly ask which trap it might be falling into.

## 8. Working Strategy

The default project strategy is:

1. Critique the idea first
2. Identify the exact bottleneck
3. Reuse the strongest known component for everything that is already solved
4. Invent only the missing mechanism
5. Test the missing mechanism in the smallest possible honest benchmark

This means:

- use known attention blocks if attention is not the research variable
- use known retrieval/indexing methods if memory freshness is the variable
- use known routing baselines before claiming a new router
- isolate one novelty per experiment where possible

## 9. Current Working Hypothesis

The current best hypothesis for the project is:

`A language system with routed external memory shards can absorb fresh information more cheaply than dense retraining, while preserving prior quality better than naive fine-tuning, if routing and fusion are well designed.`

This is a strong and testable hypothesis.

It is also narrower and more scientifically grounded than:

- "replace all transformers"
- "precalculated answers solve language"
- "millions of heads will fix everything"

## 10. Suggested Technical Translation

For now, future work should think in terms of these components:

- `Encoder / Context Synthesizer`
  - forms a compact query from the current context
- `Router`
  - selects memory shards or experts with sparse activation
- `Memory Shards / Ponds`
  - store updateable external memory, not just opaque magic reservoirs
- `Fusion Module`
  - integrates routed memory with the current hidden state
- `Decoder`
  - produces outputs from the fused representation
- `Updater`
  - inserts or edits memory without broad retraining

The updater is especially important. If the project wants freshness, the update path must be a first-class module, not an afterthought.

## 11. Short-Term Research Program

The first stage should not aim to beat frontier LLMs.

It should aim to validate three things:

1. Memory insertion works
   - add new facts into memory
   - measure whether the model can answer them
2. Old knowledge is retained
   - measure pre-update vs post-update performance
3. Sparse routing is useful
   - compare routed memory against dense access or no memory

If those three are not established, the rest of the architecture discussion is premature.

## 12. Rules For Future Agents

When working on this repo:

- Do not present toy sandbox success as architectural proof.
- Do not use metaphors where a mechanism is required.
- Do not claim efficiency without measured baselines.
- Do not claim continual learning from tiny memorization examples.
- Do not add speculative modules unless they map to a named bottleneck.
- Prefer a narrower, testable claim over a grand but blurry claim.

If forced to choose, choose clarity over hype and evidence over elegance.
