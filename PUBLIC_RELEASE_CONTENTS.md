# Public Release Contents

This file defines the `V2` public-release package.

The working repo is still a research workshop. The public repo should be a clean architecture-family release artifact.

## Include

- `LICENSE`
- `NOTICE`
- `README_PUBLIC.md` (rename to `README.md` inside the public repo)
- `PREPRINT_V2.md`
- `PREPRINT_V1.md` as historical background only
- `RELEASE_NOTES_v0.2.0.md`
- `RUMA_V2_FORMAL_SPEC.md`
- `RUMA_V2_REVERSE_ARCHITECTURE_BLUEPRINT.md`
- `ARCHITECTURE_SPEC.md`
- `DOCUMENT_UPDATE_OVERVIEW.md`
- `sandbox/README.md`
- `sandbox/`
  - include the core model code
  - include V2 benchmark harnesses
  - include V2 result artifacts
  - exclude `v3_*` scripts and `results/v3_*`
- `paper_assets/`
  - include `v2_*`
  - include figure sources needed by the manuscript
  - include earlier release assets only when they still support the V2 story
- `versioned_manuals/`
- `versioned_manual_updates/`
- `versioned_manual_conflicts/`
- `external_corpora/python_ecosystem_changes/`
- `CITATION.cff`
- `.zenodo.json`

## Exclude

- third-party paper archives and local reference implementations
- raw benchmark downloads and non-release dataset caches
- local runtime and model folders
- internal rolling drafts and future-work branch files
- author-only operational notes and staging artifacts

## Why The Split Exists

The public repo should contain:

- the code required to understand and rerun the V2 claim
- the bounded corpora used in the V2 paper
- the release manuscript
- the release-facing result assets

It should not contain:

- raw third-party archives
- local accelerator environments
- internal operating notes
- future-work branches that are not part of the V2 claim
