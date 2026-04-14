# Python Ecosystem Changes Corpus

This folder is the first scaffold for a broader external changing-document corpus.

It is meant to contain curated markdown snippets from official public sources such as:

- release notes
- changelogs
- migration guides
- operator notes

The goal is not to mirror full documentation sites yet.

The goal is to create a small, inspectable, version-aware corpus with:

- one `base` snippet per topic
- one later `update` snippet per topic
- optional `conflict` snippets where two active sources should remain visible together

## Files

- `manifest.template.json`: fill this in and copy it to `manifest.json`
- `manifest.json`: first populated external corpus built from the downloaded official-source pages

Suggested subfolders once populated:

- `base/`
- `updates/`
- `conflicts/`

## Current Corpus

The first populated corpus currently focuses on:

- FastAPI higher-end Python support changes
- FastAPI Python support changes
- pytest higher-end Python support changes
- pytest Python support changes
- FastAPI Pydantic migration changes
- Typer Python support changes
- HTTPX app shortcut deprecation and removal
- HTTPX proxy cleanup changes
- Uvicorn Python support changes
- Pydantic AI Python support changes

It is still small and curated on purpose.

## Recommended Workflow

1. Start from `manifest.json` if you want to inspect or extend the current corpus.
2. Add matching markdown files under `base/`, `updates/`, and `conflicts/`.
3. Run:

```powershell
py -3 sandbox\external_corpus_benchmark.py --manifest external_corpora\python_ecosystem_changes\manifest.json
```

## Content Rule

Each base/update pair should contain:

- one changed fact
- one retained fact

Each file should start with short provenance metadata, for example:

```md
Source: FastAPI release notes
URL: https://fastapi.tiangolo.com/release-notes/
Version: 0.x.y
Captured: 2026-04-13
```
