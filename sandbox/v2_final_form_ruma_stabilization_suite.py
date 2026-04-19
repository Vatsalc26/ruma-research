import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import (
    AliceChunkEditDataset,
    ToyCodeEditDataset,
    ToyDocChunkEditDataset,
    ToyFactEditDataset,
)
from ruma_v2_model import InterleavedRUMAModel


RUNTIME_CONFIG = {
    "torch_num_threads": 2,
    "torch_num_interop_threads": 1,
}

TRAINING_CONFIG = {
    "base_steps": 120,
    "controller_steps": 70,
    "second_controller_steps": 40,
    "parameter_edit_steps": 60,
    "batch_size": 8,
    "top_k": 4,
}

ABLATION_CONFIG = {
    "base_steps": 90,
    "controller_steps": 60,
    "second_controller_steps": 30,
    "batch_size": 8,
    "top_k": 4,
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def configure_runtime():
    torch.set_num_threads(RUNTIME_CONFIG["torch_num_threads"])
    try:
        torch.set_num_interop_threads(RUNTIME_CONFIG["torch_num_interop_threads"])
    except RuntimeError:
        pass


def evaluate_model(model, x, y, top_k=4, use_memory=False, causal=True):
    model.eval()
    with torch.no_grad():
        started = time.perf_counter()
        logits = model(x, top_k=top_k, use_memory=use_memory, causal=causal)
        elapsed = time.perf_counter() - started

    preds = torch.argmax(logits, dim=-1)
    token_accuracy = (preds == y).float().mean().item()
    exact_match = (preds == y).all(dim=1).float().mean().item()
    object_accuracy = (preds[:, -2] == y[:, -2]).float().mean().item() if y.size(1) >= 2 else exact_match

    return {
        "token_accuracy": round(float(token_accuracy), 4),
        "exact_match": round(float(exact_match), 4),
        "object_accuracy": round(float(object_accuracy), 4),
        "forward_seconds": round(float(elapsed), 6),
    }


def encode_for_retrieval(model, x):
    if hasattr(model, "encode_memory_queries"):
        return model.encode_memory_queries(x, causal=True)
    hidden_states, _, _, _, _, _ = model.encode_hidden(x, use_memory=False, causal=True)
    return hidden_states


def evaluate_memory_retrieval(model, x, inserted_records):
    model.eval()
    with torch.no_grad():
        contextual = encode_for_retrieval(model, x)
        summaries = contextual.mean(dim=1)
        routes = model.router(summaries.unsqueeze(1)).squeeze(1)

    expected_record_ids = {record.record_id for record in inserted_records}
    hits = 0
    for index in range(x.shape[0]):
        top_records = model.memory_store.top_records(
            shard_id=int(routes[index].item()),
            query=summaries[index],
            top_k=1,
        )
        if top_records and top_records[0][0].record_id in expected_record_ids:
            hits += 1

    return {
        "retrieval_recall_at_1": round(float(hits / max(1, x.shape[0])), 4),
    }


def train_model(model, dataset, *, steps, batch_size, lr, use_memory=False, top_k=4, batch_fn_name="base_train_batch"):
    criterion = nn.CrossEntropyLoss()
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr)

    model.train()
    started = time.perf_counter()
    for _ in range(steps):
        x, y = getattr(dataset, batch_fn_name)(batch_size=batch_size)
        optimizer.zero_grad()
        logits = model(x, top_k=top_k, use_memory=use_memory, causal=True)
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
    return time.perf_counter() - started


def _sample_fact_like_batch(dataset, batch_size, phase):
    updates = list(getattr(dataset, "update_facts", []))
    retention = list(getattr(dataset, "retention_facts", []))
    base = list(getattr(dataset, "base_facts", []))
    second_updates = list(getattr(dataset, "second_update_facts", []))

    chosen = []
    update_target = updates
    if phase == "second" and second_updates:
        update_target = second_updates

    update_count = max(1, batch_size // 2)
    chosen.extend(random.choice(update_target) for _ in range(update_count))

    if retention:
        retention_count = max(1, batch_size // 3)
        chosen.extend(random.choice(retention) for _ in range(retention_count))

    while len(chosen) < batch_size and base:
        chosen.append(random.choice(base))

    random.shuffle(chosen)
    return dataset.batch_from_facts(chosen)


def _sample_chunk_like_batch(dataset, batch_size):
    update_chunks = list(getattr(dataset, "update_chunks", []))
    retention_chunks = list(getattr(dataset, "retention_chunks", []))
    base_chunks = list(getattr(dataset, "base_chunks", []))

    chosen = []
    update_count = max(1, batch_size // 2)
    chosen.extend(random.choice(update_chunks) for _ in range(update_count))

    if retention_chunks:
        retention_count = max(1, batch_size // 3)
        chosen.extend(random.choice(retention_chunks) for _ in range(retention_count))

    while len(chosen) < batch_size and base_chunks:
        chosen.append(random.choice(base_chunks))

    random.shuffle(chosen)
    return dataset.batch_from_chunks(chosen)


def sample_stabilization_batch(spec, batch_size, phase):
    dataset = spec["dataset"]
    if hasattr(dataset, "batch_from_facts"):
        return _sample_fact_like_batch(dataset, batch_size=batch_size, phase=phase)
    if hasattr(dataset, "batch_from_chunks"):
        return _sample_chunk_like_batch(dataset, batch_size=batch_size)
    return dataset.base_train_batch(batch_size=batch_size)


def calibrate_memory_controller(model, spec, *, phase, steps=160, batch_size=16, lr=0.003, top_k=4):
    dataset = spec["dataset"]
    criterion = nn.CrossEntropyLoss()
    model.enable_controller_training()
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr)

    model.train()
    started = time.perf_counter()
    for step_index in range(steps):
        x, y = sample_stabilization_batch(spec, batch_size=batch_size, phase=phase)
        optimizer.zero_grad()
        logits = model(x, top_k=top_k, use_memory=True, causal=True)
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))

        if step_index >= steps // 2:
            loss = loss * 0.9

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
    return time.perf_counter() - started


def parameter_edit_finetune(model, dataset, steps=120, batch_size=16, lr=0.006):
    criterion = nn.CrossEntropyLoss()
    model.enable_parameter_edit_training()
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr)

    model.train()
    started = time.perf_counter()
    for _ in range(steps):
        x, y = dataset.update_train_batch(batch_size=batch_size)
        optimizer.zero_grad()
        logits = model(x, top_k=0, use_memory=False, causal=True)
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
    return time.perf_counter() - started


def build_dataset_spec(name, dataset, namespace, update_sources_prefix, memory_update_batch=None, second_update_batch=None):
    update_x, update_y = dataset.get_update_eval()
    memory_x = update_x
    memory_y = update_y
    if memory_update_batch is not None:
        memory_x, memory_y = memory_update_batch

    return {
        "name": name,
        "dataset": dataset,
        "namespace": namespace,
        "update_sources_prefix": update_sources_prefix,
        "retention": dataset.get_retention_eval(),
        "replaced": dataset.get_replaced_base_eval(),
        "update_eval": (update_x, update_y),
        "memory_update": (memory_x, memory_y),
        "second_update_eval": getattr(dataset, "get_second_update_eval", lambda: None)(),
        "second_memory_update": second_update_batch,
    }


def count_trainable_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def expert_usage_metrics(avg_fusion_expert_usage):
    values = [float(value) for value in avg_fusion_expert_usage.values()]
    total = max(1e-8, sum(values))
    probs = [value / total for value in values]
    entropy = -sum(prob * math.log(prob + 1e-8) for prob in probs)
    max_entropy = math.log(max(1, len(probs)))
    sharpness = 1.0 - (entropy / max(1e-8, max_entropy)) if max_entropy > 0 else 0.0
    dominant_expert = max(avg_fusion_expert_usage, key=avg_fusion_expert_usage.get)
    return {
        "dominant_expert": dominant_expert,
        "expert_entropy": round(float(entropy), 4),
        "expert_sharpness": round(float(sharpness), 4),
    }


def collect_aux_metrics(model, eval_x):
    _, aux = model(eval_x, top_k=4, return_aux=True, use_memory=True, causal=True)
    avg_expert_usage = {}
    if aux["ruma_blocks"]:
        expert_names = aux["ruma_blocks"][0]["fusion_expert_usage"].keys()
        for expert_name in expert_names:
            avg_expert_usage[expert_name] = round(
                float(
                    sum(block["fusion_expert_usage"][expert_name] for block in aux["ruma_blocks"])
                    / len(aux["ruma_blocks"])
                ),
                4,
            )
    return {
        "avg_ruma_sufficiency": aux["avg_ruma_sufficiency"],
        "avg_ruma_conflict": aux["avg_ruma_conflict"],
        "memory_stats": aux["memory_stats"],
        "avg_fusion_expert_usage": avg_expert_usage,
        "expert_specialization": expert_usage_metrics(avg_expert_usage) if avg_expert_usage else {},
        "ablation_config": aux.get("ablation_config", {}),
    }


def evaluate_ruma_path(
    model,
    spec,
    *,
    base_steps=TRAINING_CONFIG["base_steps"],
    controller_steps=TRAINING_CONFIG["controller_steps"],
    second_controller_steps=TRAINING_CONFIG["second_controller_steps"],
    batch_size=TRAINING_CONFIG["batch_size"],
    top_k=TRAINING_CONFIG["top_k"],
):
    dataset = spec["dataset"]
    retention_x, retention_y = spec["retention"]
    replaced_x, replaced_y = spec["replaced"]
    update_eval_x, update_eval_y = spec["update_eval"]
    memory_update_x, memory_update_y = spec["memory_update"]

    base_training_seconds = train_model(
        model,
        dataset,
        steps=base_steps,
        batch_size=batch_size,
        lr=0.02,
        use_memory=False,
        top_k=0,
        batch_fn_name="base_train_batch",
    )

    pre_update = {
        "retention_eval": evaluate_model(model, retention_x, retention_y, top_k=0, use_memory=False, causal=True),
        "replaced_old_fact_eval": evaluate_model(model, replaced_x, replaced_y, top_k=0, use_memory=False, causal=True),
        "update_eval": evaluate_model(model, update_eval_x, update_eval_y, top_k=0, use_memory=False, causal=True),
    }

    staged_model = copy.deepcopy(model)
    first_insert_started = time.perf_counter()
    first_inserted = staged_model.update_memory(
        memory_update_x,
        target_ids=memory_update_y,
        sources=[f"{spec['update_sources_prefix']}_first_{i}" for i in range(memory_update_x.shape[0])],
        namespaces=[spec["namespace"]] * memory_update_x.shape[0],
        timestamps=[f"{spec['name']}_first_t{i}" for i in range(memory_update_x.shape[0])],
        causal=True,
    )
    first_memory_update_seconds = time.perf_counter() - first_insert_started
    first_controller_seconds = calibrate_memory_controller(
        staged_model,
        spec,
        phase="first",
        steps=controller_steps,
        batch_size=batch_size,
        lr=0.003,
        top_k=top_k,
    )

    post_first_update = {
        "retention_eval": evaluate_model(staged_model, retention_x, retention_y, top_k=top_k, use_memory=True, causal=True),
        "replaced_old_fact_eval": evaluate_model(staged_model, replaced_x, replaced_y, top_k=top_k, use_memory=True, causal=True),
        "update_eval": evaluate_model(staged_model, update_eval_x, update_eval_y, top_k=top_k, use_memory=True, causal=True),
        "retrieval": evaluate_memory_retrieval(staged_model, memory_update_x, first_inserted),
    }
    post_first_aux = collect_aux_metrics(staged_model, update_eval_x)

    second_phase = None
    second_update_eval = spec["second_update_eval"]
    second_memory_update = spec["second_memory_update"]
    if second_update_eval is not None and second_memory_update is not None:
        second_eval_x, second_eval_y = second_update_eval
        second_memory_x, second_memory_y = second_memory_update
        second_insert_started = time.perf_counter()
        second_inserted = staged_model.update_memory(
            second_memory_x,
            target_ids=second_memory_y,
            sources=[f"{spec['update_sources_prefix']}_second_{i}" for i in range(second_memory_x.shape[0])],
            namespaces=[spec["namespace"]] * second_memory_x.shape[0],
            timestamps=[f"{spec['name']}_second_t{i}" for i in range(second_memory_x.shape[0])],
            causal=True,
        )
        second_memory_update_seconds = time.perf_counter() - second_insert_started
        second_controller_seconds = calibrate_memory_controller(
            staged_model,
            spec,
            phase="second",
            steps=second_controller_steps,
            batch_size=batch_size,
            lr=0.0025,
            top_k=top_k,
        )
        second_phase = {
            "second_memory_update_seconds": round(float(second_memory_update_seconds), 6),
            "second_controller_training_seconds": round(float(second_controller_seconds), 6),
            "second_update_eval": evaluate_model(staged_model, second_eval_x, second_eval_y, top_k=top_k, use_memory=True, causal=True),
            "prior_update_reproduction_eval": evaluate_model(staged_model, update_eval_x, update_eval_y, top_k=top_k, use_memory=True, causal=True),
            "second_retrieval": evaluate_memory_retrieval(staged_model, second_memory_x, second_inserted),
            "post_second_aux": collect_aux_metrics(staged_model, second_eval_x),
        }

    return {
        "backbone_type": staged_model.backbone_type,
        "base_training_seconds": round(float(base_training_seconds), 6),
        "first_memory_update_seconds": round(float(first_memory_update_seconds), 6),
        "first_controller_training_seconds": round(float(first_controller_seconds), 6),
        "controller_trainable_parameters": count_trainable_parameters(staged_model),
        "pre_update_base_eval": pre_update,
        "post_first_update_eval": post_first_update,
        "post_first_aux": post_first_aux,
        "second_phase": second_phase,
    }


def evaluate_parameter_edit_path(
    model,
    spec,
    *,
    base_steps=TRAINING_CONFIG["base_steps"],
    edit_steps=TRAINING_CONFIG["parameter_edit_steps"],
    batch_size=TRAINING_CONFIG["batch_size"],
):
    dataset = spec["dataset"]
    retention_x, retention_y = spec["retention"]
    replaced_x, replaced_y = spec["replaced"]
    update_eval_x, update_eval_y = spec["update_eval"]

    base_training_seconds = train_model(
        model,
        dataset,
        steps=base_steps,
        batch_size=batch_size,
        lr=0.02,
        use_memory=False,
        top_k=0,
        batch_fn_name="base_train_batch",
    )

    edit_model = copy.deepcopy(model)
    edit_training_seconds = parameter_edit_finetune(
        edit_model,
        dataset,
        steps=edit_steps,
        batch_size=batch_size,
        lr=0.006,
    )

    return {
        "backbone_type": edit_model.backbone_type,
        "base_training_seconds": round(float(base_training_seconds), 6),
        "edit_training_seconds": round(float(edit_training_seconds), 6),
        "edit_trainable_parameters": count_trainable_parameters(edit_model),
        "post_update_eval": {
            "retention_eval": evaluate_model(edit_model, retention_x, retention_y, top_k=0, use_memory=False, causal=True),
            "replaced_old_fact_eval": evaluate_model(edit_model, replaced_x, replaced_y, top_k=0, use_memory=False, causal=True),
            "update_eval": evaluate_model(edit_model, update_eval_x, update_eval_y, top_k=0, use_memory=False, causal=True),
        },
    }


def summarize_path(result):
    post = result.get("post_first_update_eval") or result.get("post_update_eval")
    if post is None:
        raise KeyError("Result payload is missing both post_first_update_eval and post_update_eval")
    summary = {
        "update_exact_match": post["update_eval"]["exact_match"],
        "retention_exact_match": post["retention_eval"]["exact_match"],
        "old_fact_suppression": round(float(1.0 - post["replaced_old_fact_eval"]["exact_match"]), 4),
    }
    second_phase = result.get("second_phase")
    if second_phase:
        summary["second_update_exact_match"] = second_phase["second_update_eval"]["exact_match"]
        summary["prior_update_suppression_after_second"] = round(
            float(1.0 - second_phase["prior_update_reproduction_eval"]["exact_match"]),
            4,
        )
    specialization = result.get("post_first_aux", {}).get("expert_specialization", {})
    if specialization:
        summary["expert_sharpness"] = specialization["expert_sharpness"]
    return summary


def build_summary(results):
    macro = {}
    comparison = {}
    systems = set()
    for dataset_results in results.values():
        systems.update(dataset_results.keys())

    systems = sorted(systems)
    for system_name in systems:
        aggregate = {}
        count = 0
        for dataset_results in results.values():
            if system_name not in dataset_results:
                continue
            summary = summarize_path(dataset_results[system_name])
            count += 1
            for key, value in summary.items():
                aggregate[key] = aggregate.get(key, 0.0) + float(value)
        if count > 0:
            macro[system_name] = {
                key: round(float(value / count), 4)
                for key, value in aggregate.items()
            }

    if "transformer_final_form_ruma" in systems:
        for system_name in systems:
            if system_name == "transformer_final_form_ruma":
                continue
            transformer_aggregate = {}
            system_aggregate = {}
            count = 0
            for dataset_results in results.values():
                if "transformer_final_form_ruma" not in dataset_results or system_name not in dataset_results:
                    continue
                transformer_summary = summarize_path(dataset_results["transformer_final_form_ruma"])
                system_summary = summarize_path(dataset_results[system_name])
                count += 1
                for key, value in transformer_summary.items():
                    transformer_aggregate[key] = transformer_aggregate.get(key, 0.0) + float(value)
                for key, value in system_summary.items():
                    system_aggregate[key] = system_aggregate.get(key, 0.0) + float(value)

            if count > 0:
                comparison[f"transformer_minus_{system_name}"] = {
                    key: round(
                        float((transformer_aggregate.get(key, 0.0) / count) - (system_aggregate.get(key, 0.0) / count)),
                        4,
                    )
                    for key in transformer_aggregate
                }

    return {
        "macro": macro,
        "comparison": comparison,
    }


def persist_partial_results(output_path, results):
    completed = {name: value for name, value in results.items() if name not in {"summary", "config"}}
    payload = dict(results)
    payload["summary"] = build_summary(completed) if completed else {"macro": {}, "comparison": {}}
    payload["config"] = {
        "runtime": RUNTIME_CONFIG,
        "training": TRAINING_CONFIG,
        "ablations": ABLATION_CONFIG,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def should_run_system(system_name, requested_systems):
    return requested_systems is None or system_name in requested_systems


def run_v2_final_form_ruma_stabilization_suite(datasets=None, systems=None, resume=True):
    configure_runtime()
    set_seed(53)

    fact_dataset = ToyFactEditDataset()
    code_dataset = ToyCodeEditDataset()
    doc_dataset = ToyDocChunkEditDataset()
    alice_dataset = AliceChunkEditDataset(file_path="alice.txt")

    specs = [
        build_dataset_spec(
            name="toy_facts",
            dataset=fact_dataset,
            namespace="toy_facts",
            update_sources_prefix="stabilized_fact_update",
            second_update_batch=fact_dataset.batch_from_facts(fact_dataset.second_update_facts),
        ),
        build_dataset_spec(
            name="toy_code",
            dataset=code_dataset,
            namespace="code_constraints",
            update_sources_prefix="stabilized_code_update",
            second_update_batch=code_dataset.batch_from_facts(code_dataset.second_update_facts),
        ),
        build_dataset_spec(
            name="toy_doc_chunks",
            dataset=doc_dataset,
            namespace="doc_chunks",
            update_sources_prefix="stabilized_doc_update",
            memory_update_batch=doc_dataset.get_update_memory_batch(),
        ),
        build_dataset_spec(
            name="alice_chunks",
            dataset=alice_dataset,
            namespace="alice_chunks",
            update_sources_prefix="stabilized_alice_update",
        ),
    ]

    if datasets:
        requested = {name.strip() for name in datasets if name.strip()}
        specs = [spec for spec in specs if spec["name"] in requested]
    requested_systems = None
    if systems:
        requested_systems = {name.strip() for name in systems if name.strip()}

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "v2_final_form_ruma_stabilization_suite.json"

    results = {}
    if resume and output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        results = {name: value for name, value in existing.items() if name not in {"summary", "config"}}

    for spec in specs:
        dataset = spec["dataset"]
        common_kwargs = {
            "vocab_size": dataset.vocab_size,
            "d_model": 64,
            "n_heads": 4,
            "num_shards": 6,
            "shard_capacity": 128,
            "num_backbone_blocks": 8,
            "num_ruma_blocks": 4,
            "top_k": 4,
        }

        dataset_results = dict(results.get(spec["name"], {}))

        if should_run_system("transformer_final_form_ruma", requested_systems) and (
            not resume or "transformer_final_form_ruma" not in dataset_results
        ):
            dataset_results["transformer_final_form_ruma"] = evaluate_ruma_path(
                model=InterleavedRUMAModel(backbone_type="transformer", **common_kwargs),
                spec=spec,
            )
            results[spec["name"]] = dataset_results
            persist_partial_results(output_path, results)

        if should_run_system("mamba_final_form_ruma", requested_systems) and (
            not resume or "mamba_final_form_ruma" not in dataset_results
        ):
            dataset_results["mamba_final_form_ruma"] = evaluate_ruma_path(
                model=InterleavedRUMAModel(backbone_type="mamba", **common_kwargs),
                spec=spec,
            )
            results[spec["name"]] = dataset_results
            persist_partial_results(output_path, results)

        if spec["name"] != "alice_chunks":
            if should_run_system("parameter_edit_auxiliary", requested_systems) and (
                not resume or "parameter_edit_auxiliary" not in dataset_results
            ):
                dataset_results["parameter_edit_auxiliary"] = evaluate_parameter_edit_path(
                model=InterleavedRUMAModel(backbone_type="transformer", **common_kwargs),
                spec=spec,
                )
                results[spec["name"]] = dataset_results
                persist_partial_results(output_path, results)

            if should_run_system("no_selectivity_ablation", requested_systems) and (
                not resume or "no_selectivity_ablation" not in dataset_results
            ):
                dataset_results["no_selectivity_ablation"] = evaluate_ruma_path(
                    model=InterleavedRUMAModel(
                        backbone_type="transformer",
                        use_selectivity=False,
                        **common_kwargs,
                    ),
                    spec=spec,
                    base_steps=ABLATION_CONFIG["base_steps"],
                    controller_steps=ABLATION_CONFIG["controller_steps"],
                    second_controller_steps=ABLATION_CONFIG["second_controller_steps"],
                    batch_size=ABLATION_CONFIG["batch_size"],
                    top_k=ABLATION_CONFIG["top_k"],
                )
                results[spec["name"]] = dataset_results
                persist_partial_results(output_path, results)

            if should_run_system("no_sparse_expert_ablation", requested_systems) and (
                not resume or "no_sparse_expert_ablation" not in dataset_results
            ):
                dataset_results["no_sparse_expert_ablation"] = evaluate_ruma_path(
                    model=InterleavedRUMAModel(
                        backbone_type="transformer",
                        use_sparse_experts=False,
                        **common_kwargs,
                    ),
                    spec=spec,
                    base_steps=ABLATION_CONFIG["base_steps"],
                    controller_steps=ABLATION_CONFIG["controller_steps"],
                    second_controller_steps=ABLATION_CONFIG["second_controller_steps"],
                    batch_size=ABLATION_CONFIG["batch_size"],
                    top_k=ABLATION_CONFIG["top_k"],
                )
                results[spec["name"]] = dataset_results
                persist_partial_results(output_path, results)

            if should_run_system("no_lineage_ablation", requested_systems) and (
                not resume or "no_lineage_ablation" not in dataset_results
            ):
                dataset_results["no_lineage_ablation"] = evaluate_ruma_path(
                    model=InterleavedRUMAModel(
                        backbone_type="transformer",
                        use_lineage_filtering=False,
                        **common_kwargs,
                    ),
                    spec=spec,
                    base_steps=ABLATION_CONFIG["base_steps"],
                    controller_steps=ABLATION_CONFIG["controller_steps"],
                    second_controller_steps=ABLATION_CONFIG["second_controller_steps"],
                    batch_size=ABLATION_CONFIG["batch_size"],
                    top_k=ABLATION_CONFIG["top_k"],
                )
                results[spec["name"]] = dataset_results
                persist_partial_results(output_path, results)

        results[spec["name"]] = dataset_results

    results["summary"] = build_summary({name: value for name, value in results.items() if name != "summary"})
    results["config"] = {
        "runtime": RUNTIME_CONFIG,
        "training": TRAINING_CONFIG,
        "ablations": ABLATION_CONFIG,
    }

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))
    print(f"\n[NOTE] Results written to: {output_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        run_v2_final_form_ruma_stabilization_suite(
            datasets=args.datasets,
            systems=args.systems,
            resume=not args.no_resume,
        )
    )
