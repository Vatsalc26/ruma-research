# Document Update Overview

This note explains the scope of the first public RUMA release.

## What This Release Focuses On

RUMA is presented here as a version-aware routed external memory architecture for changing-document updates.

The central problem is narrow:

- how to absorb new document guidance without full retraining
- how to supersede older same-lineage guidance cleanly
- how to preserve retained unchanged guidance
- how to surface active conflict instead of silently blending sources

## What This Release Is Not

This release should not be read as:

- a standalone frontier LLM
- a universal transformer replacement
- a complete solution to hallucination
- a complete solution to sycophancy

The first public claim is intentionally smaller and more defensible.

## Current System Shape

The released system combines:

- namespace-aware routing
- explicit external memory records
- append-first updates with lineage and status
- citation-first grounded answering

The update path is the main focus of the release.

## What The Benchmarks Support

On the current controlled corpora, the released benchmarks support:

- same-lineage supersession over no-update baselines
- retained guidance after updates
- visible conflict handling
- bounded routed retrieval behavior

They do not yet support broad claims about frontier-model performance.

## Why The Claim Is Narrow

The narrower claim is a strength, not a weakness.

It means the released evidence is aligned with the released story:

- changing documents
- controlled updates
- inspectable evidence
- explicit limitations

## Where To Look Next

For the public release, the most useful entry points are:

- `PREPRINT_V1.md`
- `ARCHITECTURE_SPEC.md`
- `PAPER_RESULTS_TABLES.md`
