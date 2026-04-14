import copy
import json
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import ToyFactEditDataset
from ruma_model import RUMAModel


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
    object_accuracy = (preds[:, 1] == y[:, 1]).float().mean().item()

    return {
        "token_accuracy": token_accuracy,
        "exact_match": exact_match,
        "object_accuracy": object_accuracy,
        "forward_seconds": elapsed,
    }


def train_base_model(model, dataset, steps=250, batch_size=16, lr=0.02):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for _ in range(steps):
        x, y = dataset.base_train_batch(batch_size=batch_size)
        optimizer.zero_grad()
        logits = model(x, use_memory=False, causal=True)
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()


def run_naive_finetune(model, dataset, steps=80, batch_size=8, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    start = time.perf_counter()
    for _ in range(steps):
        x, y = dataset.update_train_batch(batch_size=batch_size)
        optimizer.zero_grad()
        logits = model(x, use_memory=False, causal=True)
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
    return time.perf_counter() - start


def evaluate_memory_retrieval(model, x, inserted_records):
    model.eval()
    with torch.no_grad():
        contextual = model.encode_context(x, causal=True)
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
        "retrieval_recall_at_1": hits / max(1, x.shape[0]),
    }


def summarize_results(pre_base_metrics, no_update_metrics, memory_metrics, finetune_metrics):
    return {
        "pre_update_base_eval": pre_base_metrics,
        "no_update_baseline": no_update_metrics,
        "memory_update_path": memory_metrics,
        "naive_finetune_baseline": finetune_metrics,
    }


def run_honest_benchmark():
    print("\n=======================================================")
    print("      RUMA FIRST HONEST BENCHMARK HARNESS")
    print("=======================================================\n")

    set_seed(7)
    dataset = ToyFactEditDataset()
    retention_x, retention_y = dataset.get_retention_eval()
    replaced_x, replaced_y = dataset.get_replaced_base_eval()
    update_x, update_y = dataset.get_update_eval()

    print("[1/6] Training the shared base model on base facts...")
    base_model = RUMAModel(
        vocab_size=dataset.vocab_size,
        d_model=48,
        n_heads=2,
        num_shards=6,
        shard_capacity=128,
    )
    train_base_model(base_model, dataset)

    print("[2/6] Evaluating the base model before any updates...")
    pre_base_metrics = {
        "retention_eval": evaluate_model(base_model, retention_x, retention_y, use_memory=False, causal=True),
        "replaced_old_fact_eval": evaluate_model(base_model, replaced_x, replaced_y, use_memory=False, causal=True),
        "update_eval": evaluate_model(base_model, update_x, update_y, use_memory=False, causal=True),
    }

    print("[3/6] Building the no-update baseline...")
    no_update_model = copy.deepcopy(base_model)
    no_update_metrics = {
        "retention_eval": evaluate_model(no_update_model, retention_x, retention_y, use_memory=False, causal=True),
        "replaced_old_fact_eval": evaluate_model(no_update_model, replaced_x, replaced_y, use_memory=False, causal=True),
        "update_eval": evaluate_model(no_update_model, update_x, update_y, use_memory=False, causal=True),
        "update_seconds": 0.0,
    }

    print("[4/6] Running the external-memory update path...")
    memory_model = copy.deepcopy(base_model)
    start = time.perf_counter()
    inserted_records = memory_model.update_memory(
        update_x,
        target_ids=update_y,
        sources=[f"update_fact_{i}" for i in range(update_x.shape[0])],
        namespaces=["toy_facts"] * update_x.shape[0],
        timestamps=[f"t{i}" for i in range(update_x.shape[0])],
        causal=True,
    )
    memory_update_seconds = time.perf_counter() - start
    memory_metrics = {
        "retention_eval": evaluate_model(memory_model, retention_x, retention_y, use_memory=True, causal=True),
        "replaced_old_fact_eval": evaluate_model(memory_model, replaced_x, replaced_y, use_memory=True, causal=True),
        "update_eval": evaluate_model(memory_model, update_x, update_y, top_k=4, use_memory=True, causal=True),
        "retrieval": evaluate_memory_retrieval(memory_model, update_x, inserted_records),
        "update_seconds": memory_update_seconds,
        "memory_stats": memory_model.memory_stats(),
    }

    print("[5/6] Running the naive fine-tuning baseline...")
    finetune_model = copy.deepcopy(base_model)
    finetune_update_seconds = run_naive_finetune(finetune_model, dataset)
    finetune_metrics = {
        "retention_eval": evaluate_model(finetune_model, retention_x, retention_y, use_memory=False, causal=True),
        "replaced_old_fact_eval": evaluate_model(finetune_model, replaced_x, replaced_y, use_memory=False, causal=True),
        "update_eval": evaluate_model(finetune_model, update_x, update_y, use_memory=False, causal=True),
        "update_seconds": finetune_update_seconds,
    }

    print("[6/6] Summarizing results...")
    results = summarize_results(
        pre_base_metrics=pre_base_metrics,
        no_update_metrics=no_update_metrics,
        memory_metrics=memory_metrics,
        finetune_metrics=finetune_metrics,
    )
    print(json.dumps(results, indent=2))

    print("\n[NOTE] This harness is intentionally small and synthetic.")
    print("       It is still more honest than a hand-picked demo because it")
    print("       compares multiple update paths on fixed evaluation sets.")


if __name__ == "__main__":
    run_honest_benchmark()
