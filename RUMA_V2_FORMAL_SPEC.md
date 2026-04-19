# RUMA V2 Formal Specification

This document is the first serious mathematical framing for RUMA.

It is not meant to pretend that the current repo has already validated a full standalone foundation-model architecture.
It is meant to define the architecture precisely enough that future experiments, ablations, and model integrations can be judged against a stable technical reference.

## 1. Problem Setting

Let a user input sequence be:

`x = (x_1, x_2, ..., x_T)`

Let the external memory store be a set of records:

`M = {m_i}_{i=1}^N`

Each memory record is:

`m_i = (k_i, v_i, n_i, l_i, s_i, t_i, p_i, c_i)`

where:

- `k_i` is the retrieval key / embedding
- `v_i` is the payload representation
- `n_i` is the namespace
- `l_i` is the lineage id
- `s_i` is the status
- `t_i` is the timestamp or version marker
- `p_i` is provenance / source metadata
- `c_i` is an optional confidence or trust score

The goal of RUMA is to answer a query while:

- retrieving relevant records
- preferring same-lineage updated records over stale ones
- preserving unrelated behavior after updates
- surfacing genuine cross-source conflict rather than flattening it away

## 2. Core Architecture

RUMA is defined by six interface-level components:

1. `encode`
2. `route`
3. `fetch`
4. `rerank`
5. `fuse`
6. `update`

These components may be implemented with different internals later, but the interface separation is part of the architecture.

## 3. Encoding

Given input tokens `x`, the base model or encoder produces contextual states:

`H = E_theta(x) in R^(T x d)`

where:

- `E_theta` is the parametric encoder or base model
- `d` is the hidden dimension

Define a pooled query state:

`q = Pool(H) in R^d`

where `Pool` may be mean pooling, last-token pooling, or another stable query summarization rule.

## 4. Namespace Routing

Let the namespace set be:

`N = {nu_1, nu_2, ..., nu_J}`

RUMA first scores namespaces:

`r_j = g_phi(q, nu_j)`

where:

- `g_phi` is the routing function
- `r_j` is the routing score for namespace `nu_j`

The top-`K_n` namespaces are selected:

`N_top = TopK({r_j}, K_n)`

This is the coarse sparse routing stage.

Current V2 intent:

- keep namespace routing simple and inspectable first
- only move to heavier learned routing if routing quality becomes the measured bottleneck

## 5. Record Retrieval

Inside the selected namespaces, RUMA retrieves candidate records.

For each record `m_i` in the routed namespaces, compute a similarity score:

`a_i = sim(q, k_i)`

Candidate retrieval:

`C = TopK({a_i : n_i in N_top}, K_r)`

where:

- `sim` may be cosine similarity, dot product, or ANN-backed approximation
- `K_r` is the retrieval budget

Current V2 principle:

- retrieval should stay top-k bounded
- retrieval infrastructure can be borrowed aggressively
- RUMA's contribution is not "invent nearest-neighbor search again"

## 6. Metadata-Aware Reranking

RUMA should not rank candidates using semantic similarity alone.

Define a metadata-aware reranking score:

`u_i = alpha * a_i + beta * Fresh(i) + gamma * Trust(i) + delta * Status(i) + epsilon * Lineage(i)`

where:

- `Fresh(i)` rewards newer or less stale records
- `Trust(i)` uses source or provenance confidence
- `Status(i)` adjusts for states like `active`, `superseded`, `retracted`, `uncertain`
- `Lineage(i)` gives same-lineage update logic a controllable influence

Then:

`R = TopK({u_i : m_i in C}, K_f)`

where `K_f` is the final bounded evidence set used for fusion or answer synthesis.

## 7. Fusion

Let the final retrieved value representations be `{v_i}_{i in R}`.

Define attention-style evidence weights:

`alpha_i = softmax({u_i}_{i in R})`

Then define the evidence summary:

`c = Sum_{i in R} alpha_i * W_v v_i`

where `W_v` projects memory values into the fusion space.

Define a gating function:

`lambda_t = sigma(W_g [h_t ; c] + b_g)`

for each token state `h_t`.

Then fused states are:

`h_t_tilde = h_t + lambda_t odot c`

This is the single-hop gated residual fusion reference form for V2.

The point is not that this exact equation is final forever.
The point is that RUMA needs a real reference fusion definition instead of only narrative prose.

## 8. Decoding

The decoder or output head produces logits from fused states:

`p(x_(t+1) | x_<=t, M) = softmax(W_o h_t_tilde)`

In the more document-grounded system path, answer synthesis may be citation-first rather than purely generative.
Both are allowed as long as the experiment states clearly which answer mode is being used.

## 9. Update Operator

A new memory write is an incoming record:

`m_new = (k_new, v_new, n_new, l_new, s_new, t_new, p_new, c_new)`

The updater:

`M' = U(M, m_new)`

must preserve provenance rather than destructively blending records.

### 9A. Append-First Rule

New records are written as new explicit records:

`M' = M union {m_new}`

### 9B. Same-Lineage Supersession Rule

If there exists an active prior record `m_old` such that:

- `l_old = l_new`
- `n_old = n_new`
- `t_new > t_old`

then the prior record transitions:

`s_old := superseded`

while the new record becomes:

`s_new := active`

### 9C. Cross-Source Conflict Rule

If two active records disagree but do not share a replacement lineage relation, they should remain separately attributable.

RUMA should surface this as conflict, not silently erase one source.

## 10. Suggested Training Objective Family

RUMA V2 should eventually support a composite objective family rather than only a single next-token loss.

A useful reference form is:

`L = L_lm + lambda_ret * L_ret + lambda_sup * L_sup + lambda_conf * L_conf`

where:

- `L_lm` is language-model or answer-generation loss
- `L_ret` encourages useful retrieval/ranking
- `L_sup` encourages correct supersession behavior
- `L_conf` penalizes incorrect conflict handling or unsupported blending

This objective family is a V2 research target, not yet a claim that the current repo implements it fully.

## 11. Complexity View

The useful question is not the absolute cost of the whole base model.
It is the incremental cost added by RUMA relative to a comparable non-RUMA path.

### 11A. Routing

If namespace count is `J`, routing overhead is approximately:

`O(Jd)`

for a simple scorer.

### 11B. Retrieval

If the routed namespace contains `N_s` records:

- exact retrieval is approximately `O(N_s d)`
- ANN retrieval is approximately sublinear in practice, often closer to `O(log N_s)` query behavior depending on the backend and index structure

### 11C. Fusion

If the final retrieved set size is `K_f`, fusion overhead is approximately:

`O(K_f d)`

which is intentionally bounded by design.

### 11D. Update

Append-first writes should be much cheaper than broad retraining and scale primarily with:

- record encoding cost
- index insertion cost
- supersession bookkeeping cost

## 12. Ablation Targets

V2 should ablate at least:

- no routing vs namespace routing
- no lineage vs lineage-aware updates
- no supersession vs explicit supersession
- plain retrieval vs metadata-aware reranking
- no fusion or weak fusion vs gated fusion
- plain RAG vs RUMA

## 13. What This Formal Spec Does Not Yet Claim

This document does not claim:

- that the full objective above is already implemented
- that the current repo already has an optimal routing function
- that RUMA is already a validated backbone replacement

It provides the formal target that the next experiments should converge toward.
