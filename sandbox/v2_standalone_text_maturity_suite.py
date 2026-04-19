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

from dataset import AliceChunkEditDataset, RepoMarkdownChunkDataset
from ruma_v2_model import InterleavedRUMAModel


RUNTIME_CONFIG = {
    "torch_num_threads": 2,
    "torch_num_interop_threads": 1,
}

TRAINING_CONFIG = {
    "base_steps": 150,
    "controller_steps": 80,
    "memory_text_steps": 120,
    "parameter_edit_steps": 80,
    "batch_size": 8,
    "top_k": 4,
    "prompt_ratio": 0.5,
}


def infer_model_device(model):
    return next(model.parameters()).device


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def configure_runtime():
    torch.set_num_threads(RUNTIME_CONFIG["torch_num_threads"])
    try:
        torch.set_num_interop_threads(RUNTIME_CONFIG["torch_num_interop_threads"])
    except RuntimeError:
        pass


def train_model(model, dataset, *, steps, batch_size, lr, use_memory=False, top_k=4, batch_fn_name="base_train_batch"):
    criterion = nn.CrossEntropyLoss()
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr)
    device = infer_model_device(model)

    model.train()
    started = time.perf_counter()
    for _ in range(steps):
        x, y = getattr(dataset, batch_fn_name)(batch_size=batch_size)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x, top_k=top_k, use_memory=use_memory, causal=True)
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
    return time.perf_counter() - started


def sample_text_batch(dataset, batch_size):
    chosen = []
    update_count = max(1, batch_size // 2)
    retention_count = max(1, batch_size // 4)
    base_count = max(0, batch_size - update_count - retention_count)

    chosen.extend(random.choice(dataset.update_chunks) for _ in range(update_count))
    chosen.extend(random.choice(dataset.retention_chunks) for _ in range(retention_count))
    chosen.extend(random.choice(dataset.base_chunks) for _ in range(base_count))
    random.shuffle(chosen)
    return dataset.batch_from_chunks(chosen)


def memory_conditioned_text_training(model, dataset, *, steps, batch_size, lr, top_k=4):
    criterion = nn.CrossEntropyLoss()
    model.enable_memory_conditioned_text_training()
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr)
    device = infer_model_device(model)

    model.train()
    started = time.perf_counter()
    for step_index in range(steps):
        x, y = sample_text_batch(dataset, batch_size=batch_size)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x, top_k=top_k, use_memory=True, causal=True)
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
        if step_index >= steps // 2:
            loss = loss * 0.9
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
    return time.perf_counter() - started


def controller_text_training(model, dataset, *, steps, batch_size, lr, top_k=4):
    criterion = nn.CrossEntropyLoss()
    model.enable_controller_training()
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr)
    device = infer_model_device(model)

    model.train()
    started = time.perf_counter()
    for step_index in range(steps):
        x, y = sample_text_batch(dataset, batch_size=batch_size)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x, top_k=top_k, use_memory=True, causal=True)
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
        if step_index >= steps // 2:
            loss = loss * 0.9
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
    return time.perf_counter() - started


def parameter_edit_finetune(model, dataset, *, steps, batch_size, lr):
    criterion = nn.CrossEntropyLoss()
    model.enable_parameter_edit_training()
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr)
    device = infer_model_device(model)

    model.train()
    started = time.perf_counter()
    for _ in range(steps):
        x, y = dataset.update_train_batch(batch_size=batch_size)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x, top_k=0, use_memory=False, causal=True)
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
    return time.perf_counter() - started


def evaluate_teacher_forced(model, x, y, *, top_k=4, use_memory=False):
    device = infer_model_device(model)
    x = x.to(device)
    y = y.to(device)
    model.eval()
    with torch.no_grad():
        started = time.perf_counter()
        logits = model(x, top_k=top_k, use_memory=use_memory, causal=True)
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


def evaluate_memory_retrieval(model, x, inserted_records):
    device = infer_model_device(model)
    x = x.to(device)
    model.eval()
    with torch.no_grad():
        contextual = model.encode_memory_queries(x, causal=True)
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
    return {"retrieval_recall_at_1": round(float(hits / max(1, x.shape[0])), 4)}


def generate_suffix(model, prompt_ids, target_len, *, top_k=4, use_memory=True):
    device = infer_model_device(model)
    generated = prompt_ids.clone().detach().to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(target_len):
            logits = model(generated.unsqueeze(0), top_k=top_k, use_memory=use_memory, causal=True)
            next_id = int(torch.argmax(logits[0, -1]).item())
            generated = torch.cat(
                [generated, torch.tensor([next_id], dtype=torch.long, device=device)],
                dim=0,
            )
    return generated[-target_len:].detach().cpu()


def evaluate_generation(model, dataset, chunks, *, prompt_ratio=0.5, top_k=4, use_memory=True):
    exact_hits = 0
    total_chars = 0
    char_hits = 0
    samples = []
    started = time.perf_counter()

    for chunk in chunks:
        sequence = dataset.encode(chunk)
        prompt_len = max(8, int(len(sequence) * prompt_ratio))
        prompt_len = min(prompt_len, len(sequence) - 4)
        prompt_ids = torch.tensor(sequence[:prompt_len], dtype=torch.long)
        target_ids = torch.tensor(sequence[prompt_len:], dtype=torch.long)
        generated_ids = generate_suffix(model, prompt_ids, len(target_ids), top_k=top_k, use_memory=use_memory)

        matches = (generated_ids == target_ids).float()
        char_hits += int(matches.sum().item())
        total_chars += int(target_ids.numel())
        exact_hits += int(bool(torch.all(generated_ids == target_ids).item()))

        if len(samples) < 3:
            samples.append(
                {
                    "prompt": dataset.decode(prompt_ids),
                    "target_suffix": dataset.decode(target_ids),
                    "generated_suffix": dataset.decode(generated_ids),
                }
            )

    elapsed = time.perf_counter() - started
    return {
        "token_accuracy": round(float(char_hits / max(1, total_chars)), 4),
        "exact_match": round(float(exact_hits / max(1, len(chunks))), 4),
        "generation_seconds": round(float(elapsed), 6),
        "samples": samples,
    }


def count_trainable_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def collect_aux_metrics(model, eval_x, *, top_k):
    device = infer_model_device(model)
    eval_x = eval_x.to(device)
    _, aux = model(eval_x, top_k=top_k, return_aux=True, use_memory=True, causal=True)
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
    values = [float(value) for value in avg_expert_usage.values()]
    total = max(1e-8, sum(values))
    probs = [value / total for value in values]
    entropy = -sum(prob * math.log(prob + 1e-8) for prob in probs)
    max_entropy = math.log(max(1, len(probs))) if probs else 0.0
    sharpness = 1.0 - (entropy / max(1e-8, max_entropy)) if max_entropy > 0 else 0.0
    dominant = max(avg_expert_usage, key=avg_expert_usage.get) if avg_expert_usage else None
    return {
        "avg_ruma_sufficiency": aux["avg_ruma_sufficiency"],
        "avg_ruma_conflict": aux["avg_ruma_conflict"],
        "memory_stats": aux["memory_stats"],
        "avg_fusion_expert_usage": avg_expert_usage,
        "expert_specialization": {
            "dominant_expert": dominant,
            "expert_entropy": round(float(entropy), 4),
            "expert_sharpness": round(float(sharpness), 4),
        } if avg_expert_usage else {},
        "ablation_config": aux.get("ablation_config", {}),
    }


def build_dataset_spec(name, dataset, namespace, update_sources_prefix):
    update_x, update_y = dataset.get_update_eval()
    retention_x, retention_y = dataset.get_retention_eval()
    replaced_x, replaced_y = dataset.get_replaced_base_eval()
    return {
        "name": name,
        "dataset": dataset,
        "namespace": namespace,
        "update_sources_prefix": update_sources_prefix,
        "update_eval_xy": (update_x, update_y),
        "retention_eval_xy": (retention_x, retention_y),
        "replaced_eval_xy": (replaced_x, replaced_y),
        "update_chunks": list(dataset.update_chunks),
        "retention_chunks": list(dataset.retention_chunks),
        "replaced_chunks": list(dataset.replaced_base_chunks),
    }


def evaluate_ruma_text_path(model, spec):
    dataset = spec["dataset"]
    update_x, update_y = spec["update_eval_xy"]
    retention_x, retention_y = spec["retention_eval_xy"]
    replaced_x, replaced_y = spec["replaced_eval_xy"]

    base_training_seconds = train_model(
        model,
        dataset,
        steps=TRAINING_CONFIG["base_steps"],
        batch_size=TRAINING_CONFIG["batch_size"],
        lr=0.018,
        use_memory=False,
        top_k=0,
        batch_fn_name="base_train_batch",
    )

    adapted = copy.deepcopy(model)
    memory_update_started = time.perf_counter()
    inserted_records = adapted.update_memory(
        update_x,
        target_ids=update_y,
        sources=[f"{spec['update_sources_prefix']}_{i}" for i in range(update_x.shape[0])],
        namespaces=[spec["namespace"]] * update_x.shape[0],
        timestamps=[f"{spec['name']}_t{i}" for i in range(update_x.shape[0])],
        causal=True,
    )
    memory_update_seconds = time.perf_counter() - memory_update_started

    controller_seconds = controller_text_training(
        adapted,
        dataset,
        steps=TRAINING_CONFIG["controller_steps"],
        batch_size=TRAINING_CONFIG["batch_size"],
        lr=0.003,
        top_k=TRAINING_CONFIG["top_k"],
    )

    text_adaptation_seconds = memory_conditioned_text_training(
        adapted,
        dataset,
        steps=TRAINING_CONFIG["memory_text_steps"],
        batch_size=TRAINING_CONFIG["batch_size"],
        lr=0.0025,
        top_k=TRAINING_CONFIG["top_k"],
    )

    post_update = {
        "update_teacher_forced": evaluate_teacher_forced(
            adapted, update_x, update_y, top_k=TRAINING_CONFIG["top_k"], use_memory=True
        ),
        "retention_teacher_forced": evaluate_teacher_forced(
            adapted, retention_x, retention_y, top_k=TRAINING_CONFIG["top_k"], use_memory=True
        ),
        "replaced_teacher_forced": evaluate_teacher_forced(
            adapted, replaced_x, replaced_y, top_k=TRAINING_CONFIG["top_k"], use_memory=True
        ),
        "update_generation": evaluate_generation(
            adapted,
            dataset,
            spec["update_chunks"],
            prompt_ratio=TRAINING_CONFIG["prompt_ratio"],
            top_k=TRAINING_CONFIG["top_k"],
            use_memory=True,
        ),
        "retention_generation": evaluate_generation(
            adapted,
            dataset,
            spec["retention_chunks"],
            prompt_ratio=TRAINING_CONFIG["prompt_ratio"],
            top_k=TRAINING_CONFIG["top_k"],
            use_memory=True,
        ),
        "replaced_generation": evaluate_generation(
            adapted,
            dataset,
            spec["replaced_chunks"],
            prompt_ratio=TRAINING_CONFIG["prompt_ratio"],
            top_k=TRAINING_CONFIG["top_k"],
            use_memory=True,
        ),
        "retrieval": evaluate_memory_retrieval(adapted, update_x, inserted_records),
        "aux": collect_aux_metrics(adapted, update_x, top_k=TRAINING_CONFIG["top_k"]),
    }

    return {
        "backbone_type": adapted.backbone_type,
        "base_training_seconds": round(float(base_training_seconds), 6),
        "memory_update_seconds": round(float(memory_update_seconds), 6),
        "controller_training_seconds": round(float(controller_seconds), 6),
        "memory_conditioned_text_training_seconds": round(float(text_adaptation_seconds), 6),
        "trainable_parameters": count_trainable_parameters(adapted),
        "post_update": post_update,
    }


def evaluate_parameter_edit_text_path(model, spec):
    dataset = spec["dataset"]
    update_x, update_y = spec["update_eval_xy"]
    retention_x, retention_y = spec["retention_eval_xy"]
    replaced_x, replaced_y = spec["replaced_eval_xy"]

    base_training_seconds = train_model(
        model,
        dataset,
        steps=TRAINING_CONFIG["base_steps"],
        batch_size=TRAINING_CONFIG["batch_size"],
        lr=0.018,
        use_memory=False,
        top_k=0,
        batch_fn_name="base_train_batch",
    )

    edited = copy.deepcopy(model)
    edit_training_seconds = parameter_edit_finetune(
        edited,
        dataset,
        steps=TRAINING_CONFIG["parameter_edit_steps"],
        batch_size=TRAINING_CONFIG["batch_size"],
        lr=0.005,
    )

    post_update = {
        "update_teacher_forced": evaluate_teacher_forced(edited, update_x, update_y, top_k=0, use_memory=False),
        "retention_teacher_forced": evaluate_teacher_forced(
            edited, retention_x, retention_y, top_k=0, use_memory=False
        ),
        "replaced_teacher_forced": evaluate_teacher_forced(edited, replaced_x, replaced_y, top_k=0, use_memory=False),
        "update_generation": evaluate_generation(
            edited, dataset, spec["update_chunks"], prompt_ratio=TRAINING_CONFIG["prompt_ratio"], top_k=0, use_memory=False
        ),
        "retention_generation": evaluate_generation(
            edited, dataset, spec["retention_chunks"], prompt_ratio=TRAINING_CONFIG["prompt_ratio"], top_k=0, use_memory=False
        ),
        "replaced_generation": evaluate_generation(
            edited, dataset, spec["replaced_chunks"], prompt_ratio=TRAINING_CONFIG["prompt_ratio"], top_k=0, use_memory=False
        ),
    }

    return {
        "backbone_type": edited.backbone_type,
        "base_training_seconds": round(float(base_training_seconds), 6),
        "edit_training_seconds": round(float(edit_training_seconds), 6),
        "trainable_parameters": count_trainable_parameters(edited),
        "post_update": post_update,
    }


def summarize_path(result):
    post = result["post_update"]
    summary = {
        "update_teacher_exact_match": post["update_teacher_forced"]["exact_match"],
        "retention_teacher_exact_match": post["retention_teacher_forced"]["exact_match"],
        "update_generation_exact_match": post["update_generation"]["exact_match"],
        "retention_generation_exact_match": post["retention_generation"]["exact_match"],
        "update_generation_token_accuracy": post["update_generation"]["token_accuracy"],
        "retention_generation_token_accuracy": post["retention_generation"]["token_accuracy"],
        "old_text_suppression": round(float(1.0 - post["replaced_generation"]["exact_match"]), 4),
    }
    aux = post.get("aux", {}).get("expert_specialization", {})
    if aux:
        summary["expert_sharpness"] = aux["expert_sharpness"]
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
            macro[system_name] = {key: round(float(value / count), 4) for key, value in aggregate.items()}

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

    return {"macro": macro, "comparison": comparison}


def persist_partial_results(output_path, results):
    completed = {name: value for name, value in results.items() if name not in {"summary", "config"}}
    payload = dict(results)
    payload["summary"] = build_summary(completed) if completed else {"macro": {}, "comparison": {}}
    payload["config"] = {"runtime": RUNTIME_CONFIG, "training": TRAINING_CONFIG}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def should_run_system(system_name, requested_systems):
    return requested_systems is None or system_name in requested_systems


def run_v2_standalone_text_maturity_suite(datasets=None, systems=None, resume=True):
    configure_runtime()
    set_seed(61)

    specs = [
        build_dataset_spec(
            name="alice_text_maturity",
            dataset=AliceChunkEditDataset(
                file_path="alice.txt",
                chunk_len=48,
                stride=52,
                num_base_chunks=32,
                num_update_chunks=8,
            ),
            namespace="alice_text_maturity",
            update_sources_prefix="alice_text_update",
        ),
        build_dataset_spec(
            name="repo_markdown_text",
            dataset=RepoMarkdownChunkDataset(
                chunk_len=56,
                stride=60,
                num_base_chunks=28,
                num_update_chunks=8,
            ),
            namespace="repo_markdown_text",
            update_sources_prefix="repo_markdown_update",
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
    output_path = results_dir / "v2_standalone_text_maturity_suite.json"

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
            "top_k": TRAINING_CONFIG["top_k"],
        }

        dataset_results = dict(results.get(spec["name"], {}))

        if should_run_system("transformer_final_form_ruma", requested_systems) and (
            not resume or "transformer_final_form_ruma" not in dataset_results
        ):
            dataset_results["transformer_final_form_ruma"] = evaluate_ruma_text_path(
                InterleavedRUMAModel(backbone_type="transformer", **common_kwargs),
                spec,
            )
            results[spec["name"]] = dataset_results
            persist_partial_results(output_path, results)

        if should_run_system("mamba_final_form_ruma", requested_systems) and (
            not resume or "mamba_final_form_ruma" not in dataset_results
        ):
            dataset_results["mamba_final_form_ruma"] = evaluate_ruma_text_path(
                InterleavedRUMAModel(backbone_type="mamba", **common_kwargs),
                spec,
            )
            results[spec["name"]] = dataset_results
            persist_partial_results(output_path, results)

        if should_run_system("parameter_edit_auxiliary", requested_systems) and (
            not resume or "parameter_edit_auxiliary" not in dataset_results
        ):
            dataset_results["parameter_edit_auxiliary"] = evaluate_parameter_edit_text_path(
                InterleavedRUMAModel(backbone_type="transformer", **common_kwargs),
                spec,
            )
            results[spec["name"]] = dataset_results
            persist_partial_results(output_path, results)

        if should_run_system("no_lineage_ablation", requested_systems) and (
            not resume or "no_lineage_ablation" not in dataset_results
        ):
            dataset_results["no_lineage_ablation"] = evaluate_ruma_text_path(
                InterleavedRUMAModel(backbone_type="transformer", use_lineage_filtering=False, **common_kwargs),
                spec,
            )
            results[spec["name"]] = dataset_results
            persist_partial_results(output_path, results)

        results[spec["name"]] = dataset_results

    results["summary"] = build_summary({name: value for name, value in results.items() if name != "summary"})
    results["config"] = {"runtime": RUNTIME_CONFIG, "training": TRAINING_CONFIG}
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
        run_v2_standalone_text_maturity_suite(
            datasets=args.datasets,
            systems=args.systems,
            resume=not args.no_resume,
        )
    )
