import argparse
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from official_benchmark_data import load_counterfact_records, load_zsre_records, validate_json_file
from real_doc_memory import TOKEN_RE, clean_text


def tokenize_prompt(text):
    return TOKEN_RE.findall(clean_text(text).lower())


class PromptTokenizer:
    def __init__(self):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.word2id = {self.pad_token: 0, self.unk_token: 1}
        self.id2word = {0: self.pad_token, 1: self.unk_token}

    def fit(self, texts):
        counts = Counter()
        for text in texts:
            counts.update(tokenize_prompt(text))
        for token, _ in counts.most_common():
            if token not in self.word2id:
                idx = len(self.word2id)
                self.word2id[token] = idx
                self.id2word[idx] = token

    def encode(self, text):
        tokens = tokenize_prompt(text)
        if not tokens:
            return [self.word2id[self.unk_token]]
        return [self.word2id.get(token, self.word2id[self.unk_token]) for token in tokens]

    @property
    def vocab_size(self):
        return len(self.word2id)


class LabelEncoder:
    def __init__(self):
        self.label2id = {}
        self.id2label = {}

    def fit(self, labels):
        for label in labels:
            if label not in self.label2id:
                idx = len(self.label2id)
                self.label2id[label] = idx
                self.id2label[idx] = label

    def encode(self, label):
        return self.label2id[label]

    def decode(self, index):
        return self.id2label[int(index)]

    @property
    def num_labels(self):
        return len(self.label2id)


class PromptClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels, d_model=128, hidden_dim=192):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.proj(pooled)


def pad_batch(sequences, pad_id=0):
    max_len = max(len(sequence) for sequence in sequences)
    input_ids = []
    attention_mask = []
    for sequence in sequences:
        pad_len = max_len - len(sequence)
        input_ids.append(sequence + [pad_id] * pad_len)
        attention_mask.append([1.0] * len(sequence) + [0.0] * pad_len)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.float32),
    )


def build_base_examples(records):
    examples = []
    for record in records:
        examples.append((record.canonical_prompt, record.target_true, "base_canonical"))
        for paraphrase in record.paraphrase_prompts:
            examples.append((paraphrase, record.target_true, "base_paraphrase"))
        for retention_case in record.retention_cases:
            examples.append((retention_case.prompt, retention_case.expected_answer, retention_case.case_kind))
    return examples


def build_update_examples(records):
    examples = []
    for record in records:
        examples.append((record.canonical_prompt, record.target_new, "update_canonical"))
        for paraphrase in record.paraphrase_prompts:
            examples.append((paraphrase, record.target_new, "update_paraphrase"))
    return examples


def build_eval_buckets(records):
    canonical = []
    paraphrase = []
    retention = []
    for record in records:
        canonical.append((record.canonical_prompt, record.target_new, record.case_id))
        for paraphrase_prompt in record.paraphrase_prompts:
            paraphrase.append((paraphrase_prompt, record.target_new, record.case_id))
        for retention_case in record.retention_cases:
            retention.append((retention_case.prompt, retention_case.expected_answer, record.case_id))
    return {
        "canonical": canonical,
        "paraphrase": paraphrase,
        "retention": retention,
    }


def batchify(examples, tokenizer, label_encoder, batch_size, shuffle, rng):
    ordered = list(examples)
    if shuffle:
        rng.shuffle(ordered)
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        prompts = [tokenizer.encode(prompt) for prompt, _, _ in batch]
        labels = [label_encoder.encode(answer) for _, answer, _ in batch]
        input_ids, attention_mask = pad_batch(prompts)
        yield input_ids, attention_mask, torch.tensor(labels, dtype=torch.long)


def train_epoch(model, optimizer, examples, tokenizer, label_encoder, batch_size, rng):
    model.train()
    total_loss = 0.0
    total_items = 0
    for input_ids, attention_mask, labels in batchify(
        examples, tokenizer, label_encoder, batch_size=batch_size, shuffle=True, rng=rng
    ):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        batch_items = labels.shape[0]
        total_loss += float(loss.item()) * batch_items
        total_items += batch_items
    return total_loss / max(1, total_items)


def evaluate_bucket(model, bucket_examples, tokenizer, label_encoder):
    if not bucket_examples:
        return {
            "exact_match": 0.0,
            "count": 0,
        }

    model.eval()
    passes = 0
    with torch.no_grad():
        for prompt, expected, _ in bucket_examples:
            input_ids, attention_mask = pad_batch([tokenizer.encode(prompt)])
            logits = model(input_ids, attention_mask)
            prediction = label_encoder.decode(int(torch.argmax(logits, dim=-1).item()))
            if clean_text(prediction).lower() == clean_text(expected).lower():
                passes += 1
    return {
        "exact_match": passes / max(1, len(bucket_examples)),
        "count": len(bucket_examples),
    }


def run_single_dataset(records, dataset_name, seed=0):
    rng = random.Random(seed)
    base_examples = build_base_examples(records)
    update_examples = build_update_examples(records)
    eval_buckets = build_eval_buckets(records)

    tokenizer = PromptTokenizer()
    tokenizer.fit([prompt for prompt, _, _ in base_examples + update_examples])

    label_encoder = LabelEncoder()
    label_encoder.fit([answer for _, answer, _ in base_examples + update_examples])

    model = PromptClassifier(vocab_size=tokenizer.vocab_size, num_labels=label_encoder.num_labels)

    base_started = time.perf_counter()
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    base_losses = []
    for _ in range(12):
        base_losses.append(
            train_epoch(
                model,
                base_optimizer,
                base_examples,
                tokenizer,
                label_encoder,
                batch_size=64,
                rng=rng,
            )
        )
    base_train_seconds = time.perf_counter() - base_started

    finetune_started = time.perf_counter()
    finetune_optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
    finetune_losses = []
    for _ in range(8):
        finetune_losses.append(
            train_epoch(
                model,
                finetune_optimizer,
                update_examples,
                tokenizer,
                label_encoder,
                batch_size=32,
                rng=rng,
            )
        )
    finetune_seconds = time.perf_counter() - finetune_started

    canonical_metrics = evaluate_bucket(model, eval_buckets["canonical"], tokenizer, label_encoder)
    paraphrase_metrics = evaluate_bucket(model, eval_buckets["paraphrase"], tokenizer, label_encoder)
    retention_metrics = evaluate_bucket(model, eval_buckets["retention"], tokenizer, label_encoder)

    return {
        "dataset": dataset_name,
        "record_count": len(records),
        "base_example_count": len(base_examples),
        "update_example_count": len(update_examples),
        "label_count": label_encoder.num_labels,
        "training": {
            "base_epochs": len(base_losses),
            "finetune_epochs": len(finetune_losses),
            "base_loss_start": round(base_losses[0], 6),
            "base_loss_end": round(base_losses[-1], 6),
            "finetune_loss_start": round(finetune_losses[0], 6),
            "finetune_loss_end": round(finetune_losses[-1], 6),
            "base_train_seconds": round(base_train_seconds, 6),
            "finetune_seconds": round(finetune_seconds, 6),
        },
        "metrics": {
            "canonical_update_exact_match": canonical_metrics["exact_match"],
            "paraphrase_exact_match": paraphrase_metrics["exact_match"],
            "retention_exact_match": retention_metrics["exact_match"],
        },
    }


def load_records_for_run(rome_dir, counterfact_limit, zsre_limit):
    rome_root = Path(rome_dir)
    datasets = {}

    counterfact_path = rome_root / "counterfact.json"
    if counterfact_path.exists() and validate_json_file(counterfact_path)["valid"]:
        datasets["counterfact"] = load_counterfact_records(counterfact_path, limit=counterfact_limit)

    zsre_path = rome_root / "zsre_mend_eval.json"
    if zsre_path.exists() and validate_json_file(zsre_path)["valid"]:
        datasets["zsre"] = load_zsre_records(zsre_path, limit=zsre_limit)

    return datasets


def run_v2_small_model_finetune_benchmark(rome_dir, counterfact_limit=128, zsre_limit=128, seed=0):
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    datasets = load_records_for_run(rome_dir, counterfact_limit=counterfact_limit, zsre_limit=zsre_limit)
    if not datasets:
        print(json.dumps({"error": "No valid official benchmark files found."}, indent=2))
        return 1

    started = time.perf_counter()
    results = {
        "rome_dir": str(Path(rome_dir).resolve()),
        "seed": seed,
        "datasets": {},
    }

    for dataset_name, records in datasets.items():
        results["datasets"][dataset_name] = run_single_dataset(records, dataset_name, seed=seed)

    results["total_runtime_seconds"] = round(time.perf_counter() - started, 6)
    output_path = results_dir / "v2_small_model_finetune_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\n[NOTE] Results written to: {output_path}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rome-dir", default="benchmark_data/rome_dsets")
    parser.add_argument("--counterfact-limit", type=int, default=128)
    parser.add_argument("--zsre-limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run_v2_small_model_finetune_benchmark(
            args.rome_dir,
            counterfact_limit=args.counterfact_limit,
            zsre_limit=args.zsre_limit,
            seed=args.seed,
        )
    )
