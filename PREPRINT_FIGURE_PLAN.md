# Preprint Figure Plan

This file defines the minimal figure set for the first narrow RUMA preprint.

## Keep The Figure Set Small

For version 1, the paper does not need polished marketing-style visuals.

It needs:

1. one architecture diagram
2. one benchmark-flow diagram
3. tables for the main evidence

That is enough for the first preprint.

## Figure 1: Architecture Overview

Source of truth:

- `paper_assets/figure_ruma_architecture.mmd`

What it should show:

- query/context path
- namespace routing
- external memory records
- append-first updates
- supersession/conflict states
- citation-first answer layer

If Mermaid rendering looks poor in the final PDF, redraw it manually as a simple vector box-arrow diagram. The structure matters more than artistic polish.

## Figure 2: Benchmark Flow

Source of truth:

- `paper_assets/figure_benchmark_flow.mmd`

What it should show:

- base corpus
- update ingestion
- supersession versus naive append
- query evaluation
- retention/conflict/quality outputs

This figure should make the benchmark logic legible to readers without forcing them to parse the code.

## Tables Instead Of More Figures

The current first-preprint evidence is best represented by tables, not extra diagrams.

Recommended table set:

- main external-corpus results
- harder answer-quality results
- routing/retrieval snapshot
- claim boundary / limitations table if desired

## Claude / PDF Guidance

If you hand the manuscript to Claude for PDF conversion:

- keep Mermaid source as the input figure source
- tell Claude to either render Mermaid directly or redraw as simple vector boxes and arrows
- do not use AI-generated decorative imagery
- prefer clean technical diagrams and markdown tables
