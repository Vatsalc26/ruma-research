import copy
import json
import time

from dataset import ToyCodeEditDataset
from honest_benchmark import evaluate_model, evaluate_memory_retrieval, run_naive_finetune, set_seed, train_base_model
from ruma_model import RUMAModel


def run_code_drift_benchmark():
    print("\n=======================================================")
    print("      RUMA CODE-DRIFT BENCHMARK")
    print("=======================================================\n")

    set_seed(19)
    dataset = ToyCodeEditDataset()
    retention_x, retention_y = dataset.get_retention_eval()
    replaced_x, replaced_y = dataset.get_replaced_base_eval()
    update_x, update_y = dataset.get_update_eval()

    print("[1/6] Training the base model on project-constraint facts...")
    base_model = RUMAModel(
        vocab_size=dataset.vocab_size,
        d_model=48,
        n_heads=2,
        num_shards=6,
        shard_capacity=128,
    )
    train_base_model(base_model, dataset)

    print("[2/6] Evaluating the base model before any updates...")
    pre_metrics = {
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
        sources=[f"code_update_{i}" for i in range(update_x.shape[0])],
        namespaces=["code_constraints"] * update_x.shape[0],
        timestamps=[f"c{i}" for i in range(update_x.shape[0])],
        causal=True,
    )
    memory_update_seconds = time.perf_counter() - start
    memory_metrics = {
        "retention_eval": evaluate_model(memory_model, retention_x, retention_y, use_memory=True, causal=True),
        "replaced_old_fact_eval": evaluate_model(memory_model, replaced_x, replaced_y, use_memory=True, causal=True),
        "update_eval": evaluate_model(memory_model, update_x, update_y, use_memory=True, causal=True),
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
    results = {
        "pre_update_base_eval": pre_metrics,
        "no_update_baseline": no_update_metrics,
        "memory_update_path": memory_metrics,
        "naive_finetune_baseline": finetune_metrics,
    }
    print(json.dumps(results, indent=2))

    print("\n[NOTE] This is still synthetic, but it is closer to coding-session")
    print("       constraint updates than the original generic fact benchmark.")


if __name__ == "__main__":
    run_code_drift_benchmark()
