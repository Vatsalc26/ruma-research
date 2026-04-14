import copy
import json
import time

from dataset import AliceChunkEditDataset
from honest_benchmark import evaluate_model, evaluate_memory_retrieval, run_naive_finetune, set_seed, train_base_model
from ruma_model import RUMAModel


def run_alice_chunk_benchmark():
    print("\n=======================================================")
    print("      RUMA ALICE CHUNK BENCHMARK")
    print("=======================================================\n")

    set_seed(29)
    dataset = AliceChunkEditDataset()
    retention_x, retention_y = dataset.get_retention_eval()
    update_x, update_y = dataset.get_update_eval()

    print("[1/5] Training the base model on retained Alice chunks...")
    base_model = RUMAModel(
        vocab_size=dataset.vocab_size,
        d_model=48,
        n_heads=2,
        num_shards=6,
        shard_capacity=128,
    )
    train_base_model(base_model, dataset, steps=320, batch_size=12, lr=0.015)

    print("[2/5] Building the no-update baseline...")
    no_update_model = copy.deepcopy(base_model)
    no_update_metrics = {
        "retention_eval": evaluate_model(no_update_model, retention_x, retention_y, use_memory=False, causal=True),
        "update_eval": evaluate_model(no_update_model, update_x, update_y, use_memory=False, causal=True),
        "update_seconds": 0.0,
    }

    print("[3/5] Running the external-memory update path...")
    memory_model = copy.deepcopy(base_model)
    start = time.perf_counter()
    inserted_records = memory_model.update_memory(
        update_x,
        target_ids=update_y,
        sources=[f"alice_chunk_{i}" for i in range(update_x.shape[0])],
        namespaces=["alice_chunks"] * update_x.shape[0],
        timestamps=[f"a{i}" for i in range(update_x.shape[0])],
        causal=True,
    )
    memory_update_seconds = time.perf_counter() - start
    memory_metrics = {
        "retention_eval": evaluate_model(memory_model, retention_x, retention_y, use_memory=True, causal=True),
        "update_eval": evaluate_model(memory_model, update_x, update_y, use_memory=True, causal=True),
        "retrieval": evaluate_memory_retrieval(memory_model, update_x, inserted_records),
        "update_seconds": memory_update_seconds,
        "memory_stats": memory_model.memory_stats(),
    }

    print("[4/5] Running the naive fine-tuning baseline...")
    finetune_model = copy.deepcopy(base_model)
    finetune_update_seconds = run_naive_finetune(
        finetune_model,
        dataset,
        steps=120,
        batch_size=6,
        lr=0.008,
    )
    finetune_metrics = {
        "retention_eval": evaluate_model(finetune_model, retention_x, retention_y, use_memory=False, causal=True),
        "update_eval": evaluate_model(finetune_model, update_x, update_y, use_memory=False, causal=True),
        "update_seconds": finetune_update_seconds,
    }

    print("[5/5] Summarizing results...")
    results = {
        "no_update_baseline": no_update_metrics,
        "memory_update_path": memory_metrics,
        "naive_finetune_baseline": finetune_metrics,
    }
    print(json.dumps(results, indent=2))

    print("\n[NOTE] This is the first real-text benchmark in the sandbox.")
    print("       It still uses held-out chunk updates rather than a full external dataset,")
    print("       but it is no longer hand-authored symbolic content.")


if __name__ == "__main__":
    run_alice_chunk_benchmark()
