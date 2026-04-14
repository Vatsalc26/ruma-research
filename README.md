# RUMA

`RUMA` = `Routed Updatable Memory Architecture`

**License:** Apache 2.0  
**Status:** Experimental research release  
**Primary manuscript:** [PREPRINT_V1.md](./PREPRINT_V1.md)

## What This Release Is

RUMA is a version-aware routed external memory architecture for changing-document updates.

This release focuses on a narrow claim:

- explicit same-lineage supersession works better than no-update and naive append baselines on the current controlled corpora
- retained guidance can survive updates on the current controlled corpora
- conflict visibility can be surfaced through citation-first grounded answers

This release does **not** claim:

- a standalone frontier LLM
- a universal replacement for transformer systems
- a complete solution to hallucination or sycophancy

## What Is In This Release

- benchmark code under `sandbox/`
- curated versioned-manual corpora
- curated external Python-ecosystem update corpus
- paper-facing assets under `paper_assets/`
- the frozen first-preprint manuscript in `PREPRINT_V1.md`

## Read First

- [PREPRINT_V1.md](./PREPRINT_V1.md)
- [PROJECT_STATE_EXPLAINED.md](./PROJECT_STATE_EXPLAINED.md)
- [ARCHITECTURE_SPEC.md](./ARCHITECTURE_SPEC.md)
- [RESULTS_STATUS.md](./RESULTS_STATUS.md)
- [PAPER_RESULTS_TABLES.md](./PAPER_RESULTS_TABLES.md)

## Reproducibility

Key benchmark result artifacts live under:

- `sandbox/results/`

Key figure/table assets live under:

- `paper_assets/`

## Citation

If you use this release, please cite the software release and the accompanying preprint.

See:

- [CITATION.cff](./CITATION.cff)
