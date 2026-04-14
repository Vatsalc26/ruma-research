import copy
import json
import time

from dataset import ToyFactEditDataset
from honest_benchmark import evaluate_model, evaluate_memory_retrieval, set_seed, train_base_model
from ruma_model import RUMAModel


def run_supersession_benchmark():
    print("\n=======================================================")
    print("      RUMA SUPERSESSION BENCHMARK")
    print("=======================================================\n")

    set_seed(11)
    dataset = ToyFactEditDataset()
    retention_x, retention_y = dataset.get_retention_eval()
    update1_x, update1_y = dataset.get_update_eval()
    update2_x, update2_y = dataset.get_second_update_eval()

    print("[1/5] Training base model on original facts...")
    base_model = RUMAModel(
        vocab_size=dataset.vocab_size,
        d_model=48,
        n_heads=2,
        num_shards=6,
        shard_capacity=128,
    )
    train_base_model(base_model, dataset)

    print("[2/5] Writing the first update set into memory...")
    memory_model = copy.deepcopy(base_model)
    first_records = memory_model.update_memory(
        update1_x,
        target_ids=update1_y,
        sources=[f"first_update_{i}" for i in range(update1_x.shape[0])],
        namespaces=["toy_facts"] * update1_x.shape[0],
        timestamps=[f"u1_t{i}" for i in range(update1_x.shape[0])],
        causal=True,
    )
    first_metrics = {
        "retention_eval": evaluate_model(memory_model, retention_x, retention_y, use_memory=True, causal=True),
        "first_update_eval": evaluate_model(memory_model, update1_x, update1_y, use_memory=True, causal=True),
        "second_update_eval": evaluate_model(memory_model, update2_x, update2_y, use_memory=True, causal=True),
    }

    print("[3/5] Superseding the first memory records and writing the second update set...")
    superseded = memory_model.memory_store.update_status(
        [record.record_id for record in first_records],
        status="superseded",
    )
    start = time.perf_counter()
    second_records = memory_model.update_memory(
        update2_x,
        target_ids=update2_y,
        sources=[f"second_update_{i}" for i in range(update2_x.shape[0])],
        namespaces=["toy_facts"] * update2_x.shape[0],
        timestamps=[f"u2_t{i}" for i in range(update2_x.shape[0])],
        causal=True,
    )
    second_update_seconds = time.perf_counter() - start

    print("[4/5] Evaluating the post-supersession state...")
    second_metrics = {
        "retention_eval": evaluate_model(memory_model, retention_x, retention_y, use_memory=True, causal=True),
        "first_update_eval_after_supersession": evaluate_model(
            memory_model,
            update1_x,
            update1_y,
            use_memory=True,
            causal=True,
        ),
        "second_update_eval": evaluate_model(memory_model, update2_x, update2_y, use_memory=True, causal=True),
        "retrieval": evaluate_memory_retrieval(memory_model, update2_x, second_records),
        "superseded_records": superseded,
        "update_seconds": second_update_seconds,
        "memory_stats": memory_model.memory_stats(),
    }

    print("[5/5] Summarizing results...")
    results = {
        "after_first_update": first_metrics,
        "after_second_update": second_metrics,
    }
    print(json.dumps(results, indent=2))

    print("\n[NOTE] This benchmark tests append-first updates with explicit supersession.")
    print("       The key question is whether the latest memory update overrides earlier")
    print("       memory without damaging unaffected retained facts.")


if __name__ == "__main__":
    run_supersession_benchmark()
