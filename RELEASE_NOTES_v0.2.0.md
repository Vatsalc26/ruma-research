# Release Notes v0.2.0

`RUMA V2` is the first architecture-family release of the project.

It supersedes the narrower `V1` document-update systems package by adding:

- a frozen interleaved `RUMA` reference architecture
- a standalone small-model stabilization suite
- official edit benchmarks (`CounterFact`, `zsRE`)
- grounded verification and multi-hop benchmarks (`FEVER`, `HotpotQA`)
- a bounded `BEIR` subset bridge
- first serious ablations
- a broader standalone text-maturity package

## Main Release Framing

This release should be read as:

`RUMA V2 is an interleaved routed updatable memory architecture family for explicit updates, retained guidance, stale-memory suppression, and grounded evidence control.`

## Headline Results

- `CounterFact`: canonical `0.9922`, paraphrase `1.0`, retention `0.9938`
- `zsRE`: canonical `0.9609`, paraphrase `0.9609`, retention `1.0`
- bounded standalone stabilization:
  - bounded factual update benchmark: update `1.0`, retention `1.0`, stale suppression `1.0`
  - bounded code-update benchmark: update `1.0`, retention `1.0`, stale suppression `1.0`
  - bounded document-chunk update benchmark: update `1.0`, retention `1.0`, stale suppression `1.0`
- `FEVER` interleaved controller: evidence recall@5 `0.9609`, answer hit `0.7656`
- `HotpotQA` interleaved controller: full support-chain hit@8 `0.8359`, answer hit `0.5312`

## What Changed Relative To V1

- the project moved from a bounded external-memory systems note to a genuine interleaved architecture program
- the repo now contains a final-form small-model reference architecture
- the benchmark package now covers official edit, verification, retrieval, and multi-hop families
- the manuscript claim boundary is wider than `V1`, but still intentionally below a flagship standalone-model claim

## What This Release Still Does Not Claim

- a broad standalone foundation model
- a permanently settled backbone comparison
- a solution to hallucination in general
- a universal replacement for standard language-model backbones

## What Comes Next

A larger standalone follow-on branch remains active, but it should not be confused with the frozen `V2` release claim.
