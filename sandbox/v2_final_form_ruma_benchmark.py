import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import ToyCodeEditDataset, ToyDocChunkEditDataset, ToyFactEditDataset
from ruma_v2_model import InterleavedRUMAModel


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def evaluate_model(model, x, y, top_k=4, use_memory=False, causal=True):
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        logits = model(x, top_k=top_k, use_memory=use_memory, causal=causal)
        elapsed = time.perf_counter() - start

    preds = torch.argmax(logits, dim=-1)
    token_accuracy = (preds == y).float().mean().item()
    exact_match = (preds == y).all(dim=1).float().mean().item()
    object_accuracy = (preds[:, -2] == y[:, -2]).float().mean().item()

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
    for i in range(x.shape[0]):
        top_records = model.memory_store.top_records(
            shard_id=int(routes[i].item()),
            query=summaries[i],
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
    start = time.perf_counter()
    for _ in range(steps):
        x, y = getattr(dataset, batch_fn_name)(batch_size=batch_size)
        optimizer.zero_grad()
        logits = model(x, top_k=top_k, use_memory=use_memory, causal=True)
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
    return time.perf_counter() - start


def sample_calibration_batch(dataset, batch_size):
    update_count = max(1, batch_size // 2)
    retention_count = max(1, batch_size - update_count)
    chosen_facts = [random.choice(dataset.update_facts) for _ in range(update_count)]
    chosen_facts.extend(random.choice(dataset.retention_facts) for _ in range(retention_count))
    random.shuffle(chosen_facts)
    return dataset.batch_from_facts(chosen_facts)


def calibrate_memory_controller(model, spec, steps=160, batch_size=16, lr=0.003, top_k=4):
    dataset = spec["dataset"]
    criterion = nn.CrossEntropyLoss()
    model.enable_controller_training()
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr)

    model.train()
    start = time.perf_counter()
    for _ in range(steps):
        x, y = sample_calibration_batch(dataset, batch_size=batch_size)
        optimizer.zero_grad()
        logits = model(x, top_k=top_k, use_memory=True, causal=True)
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
    return time.perf_counter() - start


def parameter_edit_finetune(model, dataset, steps=120, batch_size=16, lr=0.006):
    criterion = nn.CrossEntropyLoss()
    model.enable_parameter_edit_training()
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr)

    model.train()
    start = time.perf_counter()
    for _ in range(steps):
        x, y = dataset.update_train_batch(batch_size=batch_size)
        optimizer.zero_grad()
        logits = model(x, top_k=0, use_memory=False, causal=True)
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
    return time.perf_counter() - start


def build_dataset_spec(name, dataset, namespace, update_sources_prefix, memory_update_batch=None):
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
    }


def count_trainable_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def evaluate_ruma_path(model, spec, *, base_steps=260, controller_steps=160):
    dataset = spec["dataset"]
    retention_x, retention_y = spec["retention"]
    replaced_x, replaced_y = spec["replaced"]
    update_eval_x, update_eval_y = spec["update_eval"]
    memory_update_x, memory_update_y = spec["memory_update"]

    base_training_seconds = train_model(
        model,
        dataset,
        steps=base_steps,
        batch_size=16,
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

    memory_model = copy.deepcopy(model)
    memory_start = time.perf_counter()
    inserted_records = memory_model.update_memory(
        memory_update_x,
        target_ids=memory_update_y,
        sources=[f"{spec['update_sources_prefix']}_{i}" for i in range(memory_update_x.shape[0])],
        namespaces=[spec["namespace"]] * memory_update_x.shape[0],
        timestamps=[f"{spec['name']}_t{i}" for i in range(memory_update_x.shape[0])],
        causal=True,
    )
    memory_update_seconds = time.perf_counter() - memory_start
    controller_training_seconds = calibrate_memory_controller(
        memory_model,
        spec,
        steps=controller_steps,
        batch_size=16,
        lr=0.003,
        top_k=4,
    )

    post_update = {
        "retention_eval": evaluate_model(memory_model, retention_x, retention_y, top_k=4, use_memory=True, causal=True),
        "replaced_old_fact_eval": evaluate_model(memory_model, replaced_x, replaced_y, top_k=4, use_memory=True, causal=True),
        "update_eval": evaluate_model(memory_model, update_eval_x, update_eval_y, top_k=4, use_memory=True, causal=True),
        "retrieval": evaluate_memory_retrieval(memory_model, memory_update_x, inserted_records),
    }

    _, aux = memory_model(update_eval_x, top_k=4, return_aux=True, use_memory=True, causal=True)
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
        "backbone_type": memory_model.backbone_type,
        "base_training_seconds": round(float(base_training_seconds), 6),
        "memory_update_seconds": round(float(memory_update_seconds), 6),
        "controller_training_seconds": round(float(controller_training_seconds), 6),
        "controller_trainable_parameters": count_trainable_parameters(memory_model),
        "pre_update_base_eval": pre_update,
        "post_update_eval": post_update,
        "post_update_aux": {
            "avg_ruma_sufficiency": aux["avg_ruma_sufficiency"],
            "avg_ruma_conflict": aux["avg_ruma_conflict"],
            "memory_stats": aux["memory_stats"],
            "avg_fusion_expert_usage": avg_expert_usage,
        },
    }


def evaluate_parameter_edit_path(model, spec, *, base_steps=260, edit_steps=120):
    dataset = spec["dataset"]
    retention_x, retention_y = spec["retention"]
    replaced_x, replaced_y = spec["replaced"]
    update_eval_x, update_eval_y = spec["update_eval"]

    base_training_seconds = train_model(
        model,
        dataset,
        steps=base_steps,
        batch_size=16,
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

    edit_model = copy.deepcopy(model)
    edit_training_seconds = parameter_edit_finetune(
        edit_model,
        dataset,
        steps=edit_steps,
        batch_size=16,
        lr=0.006,
    )

    post_update = {
        "retention_eval": evaluate_model(edit_model, retention_x, retention_y, top_k=0, use_memory=False, causal=True),
        "replaced_old_fact_eval": evaluate_model(edit_model, replaced_x, replaced_y, top_k=0, use_memory=False, causal=True),
        "update_eval": evaluate_model(edit_model, update_eval_x, update_eval_y, top_k=0, use_memory=False, causal=True),
    }

    return {
        "backbone_type": edit_model.backbone_type,
        "base_training_seconds": round(float(base_training_seconds), 6),
        "edit_training_seconds": round(float(edit_training_seconds), 6),
        "edit_trainable_parameters": count_trainable_parameters(edit_model),
        "pre_update_base_eval": pre_update,
        "post_update_eval": post_update,
    }


def build_summary(dataset_results):
    path_names = [
        "transformer_final_form_ruma",
        "mamba_final_form_ruma",
        "parameter_edit_auxiliary",
    ]
    metric_sums = {
        path_name: {
            "update_exact_match": 0.0,
            "update_object_accuracy": 0.0,
            "retention_exact_match": 0.0,
            "old_fact_reproduction_exact_match": 0.0,
            "old_fact_suppression": 0.0,
        }
        for path_name in path_names
    }

    per_dataset = {}
    dataset_count = 0
    for dataset_name, results in dataset_results.items():
        dataset_count += 1
        per_dataset[dataset_name] = {}
        for path_name in path_names:
            path_results = results[path_name]["post_update_eval"]
            replaced_exact = path_results["replaced_old_fact_eval"]["exact_match"]
            per_dataset[dataset_name][path_name] = {
                "update_exact_match": path_results["update_eval"]["exact_match"],
                "update_object_accuracy": path_results["update_eval"]["object_accuracy"],
                "retention_exact_match": path_results["retention_eval"]["exact_match"],
                "old_fact_reproduction_exact_match": replaced_exact,
                "old_fact_suppression": round(float(1.0 - replaced_exact), 4),
            }
            metric_sums[path_name]["update_exact_match"] += path_results["update_eval"]["exact_match"]
            metric_sums[path_name]["update_object_accuracy"] += path_results["update_eval"]["object_accuracy"]
            metric_sums[path_name]["retention_exact_match"] += path_results["retention_eval"]["exact_match"]
            metric_sums[path_name]["old_fact_reproduction_exact_match"] += replaced_exact
            metric_sums[path_name]["old_fact_suppression"] += 1.0 - replaced_exact

    macro = {}
    for path_name in path_names:
        macro[path_name] = {
            metric_name: round(float(metric_value / max(1, dataset_count)), 4)
            for metric_name, metric_value in metric_sums[path_name].items()
        }

    transformer = macro["transformer_final_form_ruma"]
    mamba = macro["mamba_final_form_ruma"]
    parameter_edit = macro["parameter_edit_auxiliary"]
    comparison = {
        "transformer_minus_mamba": {
            metric_name: round(float(transformer[metric_name] - mamba[metric_name]), 4)
            for metric_name in transformer
        },
        "transformer_minus_parameter_edit": {
            metric_name: round(float(transformer[metric_name] - parameter_edit[metric_name]), 4)
            for metric_name in transformer
        },
        "mamba_minus_parameter_edit": {
            metric_name: round(float(mamba[metric_name] - parameter_edit[metric_name]), 4)
            for metric_name in transformer
        },
    }

    return {
        "per_dataset": per_dataset,
        "macro": macro,
        "comparison": comparison,
    }


def run_v2_final_form_ruma_benchmark():
    set_seed(41)
    specs = [
        build_dataset_spec(
            name="toy_facts",
            dataset=ToyFactEditDataset(),
            namespace="toy_facts",
            update_sources_prefix="final_form_fact_update",
        ),
        build_dataset_spec(
            name="toy_code",
            dataset=ToyCodeEditDataset(),
            namespace="code_constraints",
            update_sources_prefix="final_form_code_update",
        ),
        build_dataset_spec(
            name="toy_doc_chunks",
            dataset=ToyDocChunkEditDataset(),
            namespace="doc_chunks",
            update_sources_prefix="final_form_doc_update",
            memory_update_batch=ToyDocChunkEditDataset().get_update_memory_batch(),
        ),
    ]

    results = {}
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
        results[spec["name"]] = {
            "transformer_final_form_ruma": evaluate_ruma_path(
                model=InterleavedRUMAModel(backbone_type="transformer", **common_kwargs),
                spec=spec,
            ),
            "mamba_final_form_ruma": evaluate_ruma_path(
                model=InterleavedRUMAModel(backbone_type="mamba", **common_kwargs),
                spec=spec,
            ),
            "parameter_edit_auxiliary": evaluate_parameter_edit_path(
                model=InterleavedRUMAModel(backbone_type="transformer", **common_kwargs),
                spec=spec,
            ),
        }

    results["summary"] = build_summary(
        {name: dataset_results for name, dataset_results in results.items() if name != "summary"}
    )

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "v2_final_form_ruma_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\n[NOTE] Results written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_v2_final_form_ruma_benchmark())
