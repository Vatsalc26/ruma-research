# Sandbox Code Map

This directory contains the executable research code for `RUMA`.

For the `V2` public release, this directory should be read as the code layer behind the frozen architecture-family claim.

## Core Architecture Files

- `ruma_v2_blocks.py`: interleaved `RUMA` block implementations, selectivity/control, and sparse expert fusion
- `ruma_v2_model.py`: main interleaved reference model
- `updater.py`: explicit write path with same-lineage supersession logic
- `router.py`: current routed-shard assignment layer
- `memory_shards.py`: in-memory shard store scaffold
- `dataset.py`: bounded dataset loaders and mixed-corpus helpers

## Main V2 Benchmark Entry Points

- `v2_final_form_ruma_benchmark.py`: standalone small-model final-form comparison
- `v2_final_form_ruma_stabilization_suite.py`: stabilization and ablation suite across fact, code, and document-style update slices
- `v2_official_edit_benchmark.py`: official `CounterFact` and `zsRE` edit benchmark harness
- `v2_external_baseline_benchmark.py`: external versioned-document baseline comparison
- `v2_fever_bridge_benchmark.py`: grounded verification benchmark
- `v2_hotpotqa_bridge_benchmark.py`: multi-hop bridge benchmark
- `v2_beir_subset_benchmark.py`: bounded official retrieval benchmark
- `v2_benchmark_eval_package.py`: unified V2 benchmark roll-up
- `v2_standalone_text_maturity_suite.py`: broader standalone text-maturity suite

## Main V2 Result Artifacts

- `results/v2_final_form_ruma_benchmark.json`
- `results/v2_final_form_ruma_stabilization_suite.json`
- `results/v2_official_edit_benchmark.json`
- `results/v2_external_baseline_benchmark.json`
- `results/v2_fever_bridge_benchmark.json`
- `results/v2_hotpotqa_bridge_benchmark.json`
- `results/v2_beir_subset_benchmark.json`
- `results/v2_benchmark_eval_package.json`
- `results/v2_standalone_text_maturity_suite.json`

## V2 Scope Notes

The `V2` release should be interpreted as an architecture-family package centered on:

- explicit updates
- retained guidance
- stale-memory suppression
- grounded evidence control

It should **not** be interpreted as a flagship standalone-foundation-model release.
