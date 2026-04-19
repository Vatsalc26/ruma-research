# RUMA V2

`RUMA` = `Routed Updatable Memory Architecture`

<p align="center">
  <a href="https://github.com/Vatsalc26/ruma-research/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/Vatsalc26/ruma-research?display_name=tag&color=2563eb" /></a>
  <a href="./LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-1f2937" /></a>
</p>

**License:** Apache 2.0  
**Status:** V2 architecture-family release

## What This Release Is

`RUMA V2` is an interleaved routed updatable memory architecture family for:

- explicit updates
- retained guidance
- stale-memory suppression
- grounded evidence control

This release is meant to be read as a serious small-model and architecture-family package, not as a frontier-model claim.

## What This Release Claims

- interleaved `RUMA` blocks can function as a real model-architecture component
- `RUMA` remains strong on official edit tasks and bounded standalone stabilization tasks
- the broader controller-based package is strong on `FEVER`, `HotpotQA`, external versioned documents, and the bounded `BEIR` subset bridge
- broader standalone text behavior is partially real, with a public reference backbone optimized for grounded controller behavior and a comparison backbone that improves small-model text retention

## What This Release Does Not Claim

- a broad standalone foundation model
- a permanently settled backbone comparison
- a general solution to hallucination
- a universal replacement for standard language-model backbones

## Headline Results

- `CounterFact`: `RUMA supersession` canonical `0.9922`, paraphrase `1.0`, retention `0.9938`
- `zsRE`: `RUMA supersession` canonical `0.9609`, paraphrase `0.9609`, retention `1.0`
- bounded standalone stabilization:
  - bounded factual update benchmark: update `1.0`, retention `1.0`, stale suppression `1.0`
  - bounded code-update benchmark: update `1.0`, retention `1.0`, stale suppression `1.0`
  - bounded document-chunk update benchmark: update `1.0`, retention `1.0`, stale suppression `1.0`
- `FEVER` interleaved controller: evidence recall@5 `0.9609`, answer hit `0.7656`, `NEI` abstain `0.4531`
- `HotpotQA` interleaved controller: supporting-fact recall@8 `0.8005`, full support-chain hit@8 `0.8359`, answer hit `0.5312`
- broader standalone text maturity:
  - `RUMA` with the public reference backbone: update-generation exact `0.5625`, retention-generation exact `0.0`, old-text suppression `1.0`
  - `RUMA` with the selective state-space comparison backbone: update-generation exact `0.625`, retention-generation exact `0.5416`, old-text suppression `0.75`

## Read First

- [PREPRINT_V2.md](./PREPRINT_V2.md)
- [RELEASE_NOTES_v0.2.1.md](./RELEASE_NOTES_v0.2.1.md)
- [sandbox/README.md](./sandbox/README.md)
- [paper_assets/v2_benchmark_package_snapshot.md](./paper_assets/v2_benchmark_package_snapshot.md)
- [RUMA_V2_FORMAL_SPEC.md](./RUMA_V2_FORMAL_SPEC.md)
- [RUMA_V2_REVERSE_ARCHITECTURE_BLUEPRINT.md](./RUMA_V2_REVERSE_ARCHITECTURE_BLUEPRINT.md)

## Package Layout

- `sandbox/`: core model code, benchmark harnesses, and result artifacts
- `paper_assets/`: release-facing CSV and markdown summary assets
- `versioned_manuals/`, `versioned_manual_updates/`, `versioned_manual_conflicts/`: curated bounded update corpora
- `external_corpora/python_ecosystem_changes/`: curated external update corpus built from official-source Python ecosystem documents

## Historical Note

The earlier narrow systems package remains available as the `V1` release path and manuscript in [PREPRINT_V1.md](./PREPRINT_V1.md). `V2` supersedes that package as the main architecture-family release.

Historical `V1` archival records:

- software DOI: `10.5281/zenodo.19563634`
- preprint DOI: `10.5281/zenodo.19563611`

<p>
  <a href="https://doi.org/10.5281/zenodo.19563634"><img alt="V1 software DOI" src="https://img.shields.io/badge/V1%20software%20DOI-10.5281%2Fzenodo.19563634-0f766e" /></a>
  <a href="https://doi.org/10.5281/zenodo.19563611"><img alt="V1 preprint DOI" src="https://img.shields.io/badge/V1%20preprint%20DOI-10.5281%2Fzenodo.19563611-f59e0b" /></a>
</p>

These `V1` records are retained for archival continuity. They should not be interpreted as the DOI records for the current `V2` release.

## Citation

Use [CITATION.cff](./CITATION.cff) and the release metadata for the public release you are actually citing. The earlier `V1` DOI-backed release remains a historical artifact and should not be mistaken for the `V2` package.
