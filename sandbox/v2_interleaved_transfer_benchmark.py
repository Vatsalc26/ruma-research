import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import ToyCodeEditDataset, ToyDocChunkEditDataset, ToyFactEditDataset
from ruma_model import RUMAModel
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


def train_base_model(model, dataset, steps=300, batch_size=16, lr=0.02):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    start = time.perf_counter()
    for _ in range(steps):
        x, y = dataset.base_train_batch(batch_size=batch_size)
        optimizer.zero_grad()
        logits = model(x, use_memory=False, causal=True)
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
    return time.perf_counter() - start


def encode_for_retrieval(model, x):
    if hasattr(model, "encode_memory_queries"):
        return model.encode_memory_queries(x, causal=True)
    if hasattr(model, "encode_context"):
        return model.encode_context(x, causal=True)
    hidden_states, _ = model.encode_hidden(x, use_memory=False, causal=True)
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


def run_model_path(model, spec):
    dataset = spec["dataset"]
    retention_x, retention_y = spec["retention"]
    replaced_x, replaced_y = spec["replaced"]
    update_eval_x, update_eval_y = spec["update_eval"]
    memory_update_x, memory_update_y = spec["memory_update"]

    training_seconds = train_base_model(model, dataset)

    pre_update = {
        "retention_eval": evaluate_model(model, retention_x, retention_y, use_memory=False, causal=True),
        "replaced_old_fact_eval": evaluate_model(model, replaced_x, replaced_y, use_memory=False, causal=True),
        "update_eval": evaluate_model(model, update_eval_x, update_eval_y, use_memory=False, causal=True),
    }

    no_update_model = copy.deepcopy(model)
    no_update_metrics = {
        "retention_eval": evaluate_model(no_update_model, retention_x, retention_y, use_memory=False, causal=True),
        "replaced_old_fact_eval": evaluate_model(no_update_model, replaced_x, replaced_y, use_memory=False, causal=True),
        "update_eval": evaluate_model(no_update_model, update_eval_x, update_eval_y, use_memory=False, causal=True),
        "update_seconds": 0.0,
    }

    memory_model = copy.deepcopy(model)
    start = time.perf_counter()
    inserted_records = memory_model.update_memory(
        memory_update_x,
        target_ids=memory_update_y,
        sources=[f"{spec['update_sources_prefix']}_{i}" for i in range(memory_update_x.shape[0])],
        namespaces=[spec["namespace"]] * memory_update_x.shape[0],
        timestamps=[f"{spec['name']}_t{i}" for i in range(memory_update_x.shape[0])],
        causal=True,
    )
    memory_update_seconds = time.perf_counter() - start
    memory_metrics = {
        "retention_eval": evaluate_model(memory_model, retention_x, retention_y, use_memory=True, causal=True),
        "replaced_old_fact_eval": evaluate_model(memory_model, replaced_x, replaced_y, use_memory=True, causal=True),
        "update_eval": evaluate_model(memory_model, update_eval_x, update_eval_y, top_k=4, use_memory=True, causal=True),
        "retrieval": evaluate_memory_retrieval(memory_model, memory_update_x, inserted_records),
        "update_seconds": round(float(memory_update_seconds), 6),
        "memory_stats": memory_model.memory_stats(),
    }

    return {
        "training_seconds": round(float(training_seconds), 6),
        "pre_update_base_eval": pre_update,
        "no_update_baseline": no_update_metrics,
        "memory_update_path": memory_metrics,
    }


def build_transfer_summary(results):
    per_dataset = {}
    attachment_retention_sum = 0.0
    interleaved_retention_sum = 0.0
    attachment_update_sum = 0.0
    interleaved_update_sum = 0.0
    attachment_update_object_sum = 0.0
    interleaved_update_object_sum = 0.0
    attachment_replaced_sum = 0.0
    interleaved_replaced_sum = 0.0

    for dataset_name, dataset_results in results.items():
        attachment = dataset_results["attachment_ruma"]["memory_update_path"]
        interleaved = dataset_results["interleaved_ruma"]["memory_update_path"]

        attachment_retention = attachment["retention_eval"]["exact_match"]
        interleaved_retention = interleaved["retention_eval"]["exact_match"]
        attachment_update = attachment["update_eval"]["exact_match"]
        interleaved_update = interleaved["update_eval"]["exact_match"]
        attachment_update_object = attachment["update_eval"]["object_accuracy"]
        interleaved_update_object = interleaved["update_eval"]["object_accuracy"]
        attachment_replaced = attachment["replaced_old_fact_eval"]["exact_match"]
        interleaved_replaced = interleaved["replaced_old_fact_eval"]["exact_match"]

        per_dataset[dataset_name] = {
            "update_exact_match_gap": round(float(interleaved_update - attachment_update), 4),
            "update_object_accuracy_gap": round(float(interleaved_update_object - attachment_update_object), 4),
            "retention_exact_match_gap": round(float(interleaved_retention - attachment_retention), 4),
            "replaced_old_fact_exact_match_gap": round(float(interleaved_replaced - attachment_replaced), 4),
            "retrieval_recall_gap": round(
                float(interleaved["retrieval"]["retrieval_recall_at_1"] - attachment["retrieval"]["retrieval_recall_at_1"]),
                4,
            ),
        }

        attachment_retention_sum += attachment_retention
        interleaved_retention_sum += interleaved_retention
        attachment_update_sum += attachment_update
        interleaved_update_sum += interleaved_update
        attachment_update_object_sum += attachment_update_object
        interleaved_update_object_sum += interleaved_update_object
        attachment_replaced_sum += attachment_replaced
        interleaved_replaced_sum += interleaved_replaced

    dataset_count = max(1, len(results))
    return {
        "per_dataset": per_dataset,
        "macro_attachment_update_exact_match": round(float(attachment_update_sum / dataset_count), 4),
        "macro_interleaved_update_exact_match": round(float(interleaved_update_sum / dataset_count), 4),
        "macro_attachment_update_object_accuracy": round(float(attachment_update_object_sum / dataset_count), 4),
        "macro_interleaved_update_object_accuracy": round(float(interleaved_update_object_sum / dataset_count), 4),
        "macro_attachment_retention_exact_match": round(float(attachment_retention_sum / dataset_count), 4),
        "macro_interleaved_retention_exact_match": round(float(interleaved_retention_sum / dataset_count), 4),
        "macro_attachment_replaced_old_fact_exact_match": round(float(attachment_replaced_sum / dataset_count), 4),
        "macro_interleaved_replaced_old_fact_exact_match": round(float(interleaved_replaced_sum / dataset_count), 4),
    }


def run_v2_interleaved_transfer_benchmark():
    set_seed(29)
    specs = [
        build_dataset_spec(
            name="toy_facts",
            dataset=ToyFactEditDataset(),
            namespace="toy_facts",
            update_sources_prefix="transfer_fact_update",
        ),
        build_dataset_spec(
            name="toy_code",
            dataset=ToyCodeEditDataset(),
            namespace="code_constraints",
            update_sources_prefix="transfer_code_update",
        ),
        build_dataset_spec(
            name="toy_doc_chunks",
            dataset=ToyDocChunkEditDataset(),
            namespace="doc_chunks",
            update_sources_prefix="transfer_doc_update",
            memory_update_batch=ToyDocChunkEditDataset().get_update_memory_batch(),
        ),
    ]

    results = {}
    for spec in specs:
        dataset = spec["dataset"]
        results[spec["name"]] = {
            "attachment_ruma": run_model_path(
                model=RUMAModel(
                    vocab_size=dataset.vocab_size,
                    d_model=64,
                    n_heads=4,
                    num_shards=6,
                    shard_capacity=128,
                ),
                spec=spec,
            ),
            "interleaved_ruma": run_model_path(
                model=InterleavedRUMAModel(
                    vocab_size=dataset.vocab_size,
                    d_model=64,
                    n_heads=4,
                    num_shards=6,
                    shard_capacity=128,
                    num_backbone_blocks=8,
                    num_ruma_blocks=4,
                    top_k=4,
                ),
                spec=spec,
            ),
        }

    results["summary"] = build_transfer_summary(
        {name: dataset_results for name, dataset_results in results.items() if name != "summary"}
    )

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "v2_interleaved_transfer_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\n[NOTE] Results written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_v2_interleaved_transfer_benchmark())
