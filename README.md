# RUMA

`RUMA` = `Routed Updatable Memory Architecture`

<p align="center">
  <a href="https://github.com/Vatsalc26/ruma-research/releases/tag/v0.1.1"><img alt="Release: v0.1.1" src="https://img.shields.io/badge/release-v0.1.1-2563eb" /></a>
  <a href="https://doi.org/10.5281/zenodo.19563634"><img alt="Software DOI" src="https://img.shields.io/badge/software%20DOI-10.5281%2Fzenodo.19563634-0f766e" /></a>
  <a href="https://doi.org/10.5281/zenodo.19563611"><img alt="Preprint DOI" src="https://img.shields.io/badge/preprint%20DOI-10.5281%2Fzenodo.19563611-f59e0b" /></a>
  <a href="./LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-1f2937" /></a>
</p>

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
- [DOCUMENT_UPDATE_OVERVIEW.md](./DOCUMENT_UPDATE_OVERVIEW.md)
- [ARCHITECTURE_SPEC.md](./ARCHITECTURE_SPEC.md)
- [PAPER_RESULTS_TABLES.md](./PAPER_RESULTS_TABLES.md)
- [PREPRINT_FIGURE_PLAN.md](./PREPRINT_FIGURE_PLAN.md)

## Reproducibility

Key benchmark result artifacts live under:

- `sandbox/results/`

Key figure/table assets live under:

- `paper_assets/`

## Citation

If you use this release, please cite the software release and the accompanying preprint.

See:

- [CITATION.cff](./CITATION.cff)
- Software DOI: https://doi.org/10.5281/zenodo.19563634
- Preprint DOI: https://doi.org/10.5281/zenodo.19563611
