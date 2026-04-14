# RUMA v0.1.0

First public research release for RUMA.

## What This Release Contains

- the frozen first-preprint manuscript source in `PREPRINT_V1.md`
- benchmark code under `sandbox/`
- result artifacts under `sandbox/results/`
- paper-facing assets under `paper_assets/`
- curated versioned-manual corpora
- the first frozen external Python-ecosystem changing-document corpus

## Narrow Claim Boundary

This release supports a narrow claim:

- explicit same-lineage supersession works better than no-update and naive append baselines on the current controlled corpora
- retained guidance survives updates on the current controlled corpora
- conflict visibility is inspectable through citation-first grounded answers

This release does **not** claim:

- a standalone frontier LLM
- a universal replacement for transformer systems
- a complete solution to hallucination or sycophancy

## Main Current Results

- external corpus benchmark:
  - `base_known_eval = 1.0`
  - `no_update_future_eval = 0.0`
  - `superseded_update_eval = 1.0`
  - `retention_eval = 1.0`
  - `naive_append_eval = 0.6`
  - `conflict_eval = 1.0`
- harder external answer quality:
  - `base_compositional_eval = 0.0`
  - `superseded_compositional_eval = 1.0`
  - `naive_append_compositional_eval = 0.4`
  - `conflict_synthesis_eval = 0.5`

## Artifacts

- public repo: `https://github.com/Vatsalc26/ruma-research`
- manuscript source: `PREPRINT_V1.md`
- software DOI: pending Zenodo archival release
- preprint DOI: pending PDF upload and Zenodo preprint release
