import torch
import random
import re
import json
from pathlib import Path

class DummyDataset:
    """
    A small synthetic next-token dataset used for sandbox wiring checks.
    Each sequence is an arithmetic progression modulo the vocabulary size so
    the model has a simple pattern it can learn during toy training.
    """
    def __init__(self, vocab_size, max_seq_len):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def generate_batch(self, batch_size):
        starts = torch.randint(0, self.vocab_size, (batch_size, 1))
        offsets = torch.arange(self.max_seq_len + 1).unsqueeze(0)
        seq = (starts + offsets) % self.vocab_size
        x = seq[:, :-1].long()
        y = seq[:, 1:].long()
        return x, y

class EnglishDummyDataset:
    def __init__(self):
        # A tiny hardcoded dictionary to test English generation
        self.words = ["<PAD>", "The", "brown", "cat", "jumped", "purple", "spaceship", "flew", ".", "Mars"]
        self.vocab_size = len(self.words)
        self.word2id = {w: i for i, w in enumerate(self.words)}
        self.id2word = {i: w for i, w in enumerate(self.words)}
        
    def encode(self, sentence):
        # Translates a string sentence into math IDs
        return [self.word2id[w] for w in sentence.split()]
        
    def decode(self, ids):
        # Translates math IDs back into an English string
        return " ".join([self.id2word[int(i)] for i in ids if int(i) != 0])
        
    def get_data_A(self):
        # The base knowledge string
        seq = self.encode("The brown cat jumped .")
        x = torch.tensor(seq[:-1]).unsqueeze(0)
        y = torch.tensor(seq[1:]).unsqueeze(0)
        return x, y
        
    def get_data_B(self):
        # The live leak knowledge string
        seq = self.encode("The purple spaceship flew .")
        x = torch.tensor(seq[:-1]).unsqueeze(0)
        y = torch.tensor(seq[1:]).unsqueeze(0)
        return x, y


class ToyFactEditDataset:
    """
    A tiny factual-update benchmark dataset for honest sandbox evaluation.
    It provides:
    - base facts for initial training
    - updated facts for post-training insertion or fine-tuning
    - fixed evaluation batches for freshness and retention measurement
    """

    def __init__(self):
        self.pad_token = "<PAD>"
        self.relation = "likes"
        self.period = "."
        self.subjects = [
            "Alice",
            "Bob",
            "Carol",
            "Dave",
            "Eve",
            "Finn",
            "Gina",
            "Hank",
        ]
        self.base_objects = [
            "apples",
            "bananas",
            "carrots",
            "dates",
            "eggs",
            "figs",
            "grapes",
            "honey",
        ]
        self.updated_objects = [
            "pears",
            "mangoes",
            "lettuce",
            "melons",
        ]
        self.second_updated_objects = [
            "plums",
            "papayas",
            "spinach",
            "kiwis",
        ]

        vocab = [self.pad_token] + self.subjects + [self.relation]
        vocab += self.base_objects + self.updated_objects + self.second_updated_objects + [self.period]
        self.words = vocab
        self.vocab_size = len(self.words)
        self.word2id = {word: idx for idx, word in enumerate(self.words)}
        self.id2word = {idx: word for idx, word in enumerate(self.words)}

        self.base_facts = [
            (subject, self.base_objects[i])
            for i, subject in enumerate(self.subjects)
        ]
        self.update_facts = [
            (self.subjects[i], self.updated_objects[i])
            for i in range(len(self.updated_objects))
        ]
        self.second_update_facts = [
            (self.subjects[i], self.second_updated_objects[i])
            for i in range(len(self.second_updated_objects))
        ]
        self.replaced_base_facts = self.base_facts[: len(self.update_facts)]
        self.retention_facts = self.base_facts[len(self.update_facts) :]

    def encode(self, sentence):
        return [self.word2id[word] for word in sentence.split()]

    def decode(self, ids):
        return " ".join(self.id2word[int(idx)] for idx in ids if int(idx) != 0)

    def fact_sentence(self, subject, obj):
        return f"{subject} {self.relation} {obj} {self.period}"

    def fact_to_xy(self, subject, obj):
        seq = self.encode(self.fact_sentence(subject, obj))
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

    def batch_from_facts(self, facts):
        xs = []
        ys = []
        for subject, obj in facts:
            x, y = self.fact_to_xy(subject, obj)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def sample_fact_batch(self, facts, batch_size, rng=None):
        rng = rng or random
        chosen = [rng.choice(facts) for _ in range(batch_size)]
        return self.batch_from_facts(chosen)

    def base_train_batch(self, batch_size, rng=None):
        return self.sample_fact_batch(self.base_facts, batch_size=batch_size, rng=rng)

    def update_train_batch(self, batch_size, rng=None):
        return self.sample_fact_batch(self.update_facts, batch_size=batch_size, rng=rng)

    def get_base_eval(self):
        return self.batch_from_facts(self.base_facts)

    def get_update_eval(self):
        return self.batch_from_facts(self.update_facts)

    def get_replaced_base_eval(self):
        return self.batch_from_facts(self.replaced_base_facts)

    def get_retention_eval(self):
        return self.batch_from_facts(self.retention_facts)

    def get_second_update_eval(self):
        return self.batch_from_facts(self.second_update_facts)


class ToyCodeEditDataset:
    """
    A code-flavored update benchmark with the same interfaces as ToyFactEditDataset.
    The goal is not realism in full, but to move one step closer to project-constraint
    style updates than generic food facts.
    """

    def __init__(self):
        self.pad_token = "<PAD>"
        self.relation = "uses"
        self.period = "."
        self.subjects = [
            "AuthModule",
            "DatabaseLayer",
            "BuildPipeline",
            "ApiServer",
            "CacheLayer",
            "TestRunner",
            "DeployStack",
            "RpcGateway",
        ]
        self.base_objects = [
            "jwt",
            "sqlite",
            "tsc",
            "json",
            "redis",
            "pytest",
            "docker",
            "rest",
        ]
        self.updated_objects = [
            "oauth",
            "postgres",
            "swc",
            "msgpack",
        ]
        self.second_updated_objects = [
            "session",
            "mysql",
            "esbuild",
            "yaml",
        ]

        vocab = [self.pad_token] + self.subjects + [self.relation]
        vocab += self.base_objects + self.updated_objects + self.second_updated_objects + [self.period]
        self.words = vocab
        self.vocab_size = len(self.words)
        self.word2id = {word: idx for idx, word in enumerate(self.words)}
        self.id2word = {idx: word for idx, word in enumerate(self.words)}

        self.base_facts = [
            (subject, self.base_objects[i])
            for i, subject in enumerate(self.subjects)
        ]
        self.update_facts = [
            (self.subjects[i], self.updated_objects[i])
            for i in range(len(self.updated_objects))
        ]
        self.second_update_facts = [
            (self.subjects[i], self.second_updated_objects[i])
            for i in range(len(self.second_updated_objects))
        ]
        self.replaced_base_facts = self.base_facts[: len(self.update_facts)]
        self.retention_facts = self.base_facts[len(self.update_facts) :]

    def encode(self, sentence):
        return [self.word2id[word] for word in sentence.split()]

    def decode(self, ids):
        return " ".join(self.id2word[int(idx)] for idx in ids if int(idx) != 0)

    def fact_sentence(self, subject, obj):
        return f"{subject} {self.relation} {obj} {self.period}"

    def fact_to_xy(self, subject, obj):
        seq = self.encode(self.fact_sentence(subject, obj))
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

    def batch_from_facts(self, facts):
        xs = []
        ys = []
        for subject, obj in facts:
            x, y = self.fact_to_xy(subject, obj)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def sample_fact_batch(self, facts, batch_size, rng=None):
        rng = rng or random
        chosen = [rng.choice(facts) for _ in range(batch_size)]
        return self.batch_from_facts(chosen)

    def base_train_batch(self, batch_size, rng=None):
        return self.sample_fact_batch(self.base_facts, batch_size=batch_size, rng=rng)

    def update_train_batch(self, batch_size, rng=None):
        return self.sample_fact_batch(self.update_facts, batch_size=batch_size, rng=rng)

    def get_base_eval(self):
        return self.batch_from_facts(self.base_facts)

    def get_update_eval(self):
        return self.batch_from_facts(self.update_facts)

    def get_replaced_base_eval(self):
        return self.batch_from_facts(self.replaced_base_facts)

    def get_retention_eval(self):
        return self.batch_from_facts(self.retention_facts)

    def get_second_update_eval(self):
        return self.batch_from_facts(self.second_update_facts)


class ToyDocChunkEditDataset:
    """
    A documentation-style update benchmark.
    Query/eval examples use a short factual sentence, while memory updates are written
    as longer chunk-like sentences that contain the relevant statement inside extra text.
    """

    def __init__(self):
        self.pad_token = "<PAD>"
        self.subjects = [
            "AuthModule",
            "DatabaseLayer",
            "BuildPipeline",
            "ApiServer",
            "CacheLayer",
            "TestRunner",
            "DeployStack",
            "RpcGateway",
        ]
        self.base_objects = [
            "jwt",
            "sqlite",
            "tsc",
            "json",
            "redis",
            "pytest",
            "docker",
            "rest",
        ]
        self.updated_objects = [
            "oauth",
            "postgres",
            "swc",
            "msgpack",
        ]

        template_words = [
            "docs",
            "state",
            "today",
            "that",
            "auth",
            "strategy",
            "uses",
            ".",
        ]

        vocab = [self.pad_token] + self.subjects + self.base_objects + self.updated_objects + template_words
        self.words = vocab
        self.vocab_size = len(self.words)
        self.word2id = {word: idx for idx, word in enumerate(self.words)}
        self.id2word = {idx: word for idx, word in enumerate(self.words)}

        self.base_facts = [
            (subject, self.base_objects[i])
            for i, subject in enumerate(self.subjects)
        ]
        self.update_facts = [
            (self.subjects[i], self.updated_objects[i])
            for i in range(len(self.updated_objects))
        ]
        self.replaced_base_facts = self.base_facts[: len(self.update_facts)]
        self.retention_facts = self.base_facts[len(self.update_facts) :]

    def encode(self, sentence):
        return [self.word2id[word] for word in sentence.split()]

    def decode(self, ids):
        return " ".join(self.id2word[int(idx)] for idx in ids if int(idx) != 0)

    def fact_sentence(self, subject, obj):
        return f"{subject} auth strategy uses {obj} ."

    def memory_chunk_sentence(self, subject, obj):
        return f"docs state today that {subject} auth strategy uses {obj} ."

    def sentence_to_xy(self, sentence):
        seq = self.encode(sentence)
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

    def fact_to_xy(self, subject, obj):
        return self.sentence_to_xy(self.fact_sentence(subject, obj))

    def memory_chunk_to_xy(self, subject, obj):
        return self.sentence_to_xy(self.memory_chunk_sentence(subject, obj))

    def batch_from_facts(self, facts):
        xs = []
        ys = []
        for subject, obj in facts:
            x, y = self.fact_to_xy(subject, obj)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def memory_batch_from_facts(self, facts):
        xs = []
        ys = []
        for subject, obj in facts:
            x, y = self.memory_chunk_to_xy(subject, obj)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def sample_fact_batch(self, facts, batch_size, rng=None):
        rng = rng or random
        chosen = [rng.choice(facts) for _ in range(batch_size)]
        return self.batch_from_facts(chosen)

    def base_train_batch(self, batch_size, rng=None):
        return self.sample_fact_batch(self.base_facts, batch_size=batch_size, rng=rng)

    def update_train_batch(self, batch_size, rng=None):
        return self.sample_fact_batch(self.update_facts, batch_size=batch_size, rng=rng)

    def get_update_memory_batch(self):
        return self.memory_batch_from_facts(self.update_facts)

    def get_retention_eval(self):
        return self.batch_from_facts(self.retention_facts)

    def get_replaced_base_eval(self):
        return self.batch_from_facts(self.replaced_base_facts)

    def get_update_eval(self):
        return self.batch_from_facts(self.update_facts)


def build_shared_char_vocab(chunks, shared_chars=None):
    pad_token = "\0"
    if shared_chars is None:
        chars = sorted(set("".join(chunks)))
    else:
        chars = list(shared_chars)
    vocab = [pad_token] + chars
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    return pad_token, vocab, stoi, itos


class AliceChunkEditDataset:
    """
    Real-text chunk benchmark built from alice.txt.
    It keeps the same train/update split pattern as the toy datasets, but uses
    held-out chunks from real book text instead of hand-authored symbolic facts.
    """

    def __init__(
        self,
        file_path="alice.txt",
        chunk_len=72,
        stride=110,
        num_base_chunks=8,
        num_update_chunks=4,
        shared_chars=None,
    ):
        with open(file_path, "r", encoding="utf-8") as handle:
            data = handle.read()

        normalized = re.sub(r"\s+", " ", data).strip()
        start = normalized.find("Alice was beginning")
        if start >= 0:
            normalized = normalized[start:]

        total_needed = num_base_chunks + num_update_chunks
        chunks = []
        cursor = 0
        while len(chunks) < total_needed and cursor + chunk_len + 1 < len(normalized):
            chunk = normalized[cursor : cursor + chunk_len]
            if len(chunk) == chunk_len:
                chunks.append(chunk)
            cursor += stride

        if len(chunks) < total_needed:
            raise ValueError("Not enough Alice text chunks for the requested split.")

        self.base_chunks = chunks[:num_base_chunks]
        self.update_chunks = chunks[num_base_chunks : num_base_chunks + num_update_chunks]
        self.replaced_base_chunks = self.base_chunks[: len(self.update_chunks)]
        self.retention_chunks = self.base_chunks[-min(4, len(self.base_chunks)) :]
        self.pad_token, self.vocab, self.stoi, self.itos = build_shared_char_vocab(chunks, shared_chars=shared_chars)
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[int(idx)] for idx in ids if int(idx) != 0)

    def chunk_to_xy(self, chunk):
        seq = self.encode(chunk)
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

    def batch_from_chunks(self, chunks):
        xs = []
        ys = []
        for chunk in chunks:
            x, y = self.chunk_to_xy(chunk)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def sample_chunk_batch(self, chunks, batch_size, rng=None):
        rng = rng or random
        chosen = [rng.choice(chunks) for _ in range(batch_size)]
        return self.batch_from_chunks(chosen)

    def base_train_batch(self, batch_size, rng=None):
        return self.sample_chunk_batch(self.base_chunks, batch_size=batch_size, rng=rng)

    def update_train_batch(self, batch_size, rng=None):
        return self.sample_chunk_batch(self.update_chunks, batch_size=batch_size, rng=rng)

    def get_retention_eval(self):
        return self.batch_from_chunks(self.retention_chunks)

    def get_replaced_base_eval(self):
        return self.batch_from_chunks(self.replaced_base_chunks)

    def get_update_eval(self):
        return self.batch_from_chunks(self.update_chunks)


class RepoMarkdownChunkDataset:
    """
    Local markdown-text continuation benchmark built from the repo's own docs.

    This gives the standalone final-form model a broader natural-language style
    target than the Alice-only slice while staying fully local and reproducible.
    """

    DEFAULT_FILES = [
        "README.md",
        "ARCHITECTURE_SPEC.md",
        "RUMA_V2_FORMAL_SPEC.md",
        "RUMA_V2_REVERSE_ARCHITECTURE_BLUEPRINT.md",
        "PREPRINT_V1.md",
        "PREPRINT_V2_ROLLING_DRAFT.md",
    ]

    def __init__(
        self,
        file_paths=None,
        chunk_len=64,
        stride=72,
        num_base_chunks=24,
        num_update_chunks=8,
        max_chars_per_file=4500,
        shared_chars=None,
    ):
        file_paths = file_paths or list(self.DEFAULT_FILES)
        documents = []
        for path in file_paths:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    text = handle.read()
            except FileNotFoundError:
                continue
            normalized = re.sub(r"\s+", " ", text).strip()
            if not normalized:
                continue
            if max_chars_per_file is not None:
                normalized = normalized[:max_chars_per_file]
            documents.append(f"[DOC {path}] {normalized}")

        if not documents:
            raise ValueError("No readable markdown documents were found for RepoMarkdownChunkDataset.")

        combined = " <DOCSEP> ".join(documents)
        total_needed = num_base_chunks + num_update_chunks
        chunks = []
        cursor = 0
        while len(chunks) < total_needed and cursor + chunk_len + 1 < len(combined):
            chunk = combined[cursor : cursor + chunk_len]
            if len(chunk) == chunk_len:
                chunks.append(chunk)
            cursor += stride

        if len(chunks) < total_needed:
            raise ValueError("Not enough markdown text chunks for the requested split.")

        self.base_chunks = chunks[:num_base_chunks]
        self.update_chunks = chunks[num_base_chunks : num_base_chunks + num_update_chunks]
        self.replaced_base_chunks = self.base_chunks[: len(self.update_chunks)]
        self.retention_chunks = self.base_chunks[-min(6, len(self.base_chunks)) :]
        self.file_paths = list(file_paths)
        self.pad_token, self.vocab, self.stoi, self.itos = build_shared_char_vocab(chunks, shared_chars=shared_chars)
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[int(idx)] for idx in ids if int(idx) != 0)

    def chunk_to_xy(self, chunk):
        seq = self.encode(chunk)
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

    def batch_from_chunks(self, chunks):
        xs = []
        ys = []
        for chunk in chunks:
            x, y = self.chunk_to_xy(chunk)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def sample_chunk_batch(self, chunks, batch_size, rng=None):
        rng = rng or random
        chosen = [rng.choice(chunks) for _ in range(batch_size)]
        return self.batch_from_chunks(chosen)

    def base_train_batch(self, batch_size, rng=None):
        return self.sample_chunk_batch(self.base_chunks, batch_size=batch_size, rng=rng)

    def update_train_batch(self, batch_size, rng=None):
        return self.sample_chunk_batch(self.update_chunks, batch_size=batch_size, rng=rng)

    def get_retention_eval(self):
        return self.batch_from_chunks(self.retention_chunks)

    def get_replaced_base_eval(self):
        return self.batch_from_chunks(self.replaced_base_chunks)

    def get_update_eval(self):
        return self.batch_from_chunks(self.update_chunks)


class DocumentChunkEditDataset:
    """
    Generic character-level chunk dataset built from a list of local documents.

    It mirrors the same base/update/replaced/retention interfaces used by the
    earlier standalone text datasets so the training/eval harnesses can reuse
    it directly.
    """

    def __init__(
        self,
        documents,
        chunk_len=64,
        stride=72,
        num_base_chunks=24,
        num_update_chunks=8,
        shared_chars=None,
    ):
        if not documents:
            raise ValueError("DocumentChunkEditDataset requires at least one document.")

        combined = " <DOCSEP> ".join(documents)
        total_needed = num_base_chunks + num_update_chunks
        chunks = []
        cursor = 0
        while len(chunks) < total_needed and cursor + chunk_len + 1 < len(combined):
            chunk = combined[cursor : cursor + chunk_len]
            if len(chunk) == chunk_len:
                chunks.append(chunk)
            cursor += stride

        if len(chunks) < total_needed:
            raise ValueError("Not enough document text chunks for the requested split.")

        self.base_chunks = chunks[:num_base_chunks]
        self.update_chunks = chunks[num_base_chunks : num_base_chunks + num_update_chunks]
        self.replaced_base_chunks = self.base_chunks[: len(self.update_chunks)]
        self.retention_chunks = self.base_chunks[-min(8, len(self.base_chunks)) :]
        self.pad_token, self.vocab, self.stoi, self.itos = build_shared_char_vocab(chunks, shared_chars=shared_chars)
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[int(idx)] for idx in ids if int(idx) != 0)

    def chunk_to_xy(self, chunk):
        seq = self.encode(chunk)
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

    def batch_from_chunks(self, chunks):
        xs = []
        ys = []
        for chunk in chunks:
            x, y = self.chunk_to_xy(chunk)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def sample_chunk_batch(self, chunks, batch_size, rng=None):
        rng = rng or random
        chosen = [rng.choice(chunks) for _ in range(batch_size)]
        return self.batch_from_chunks(chosen)

    def base_train_batch(self, batch_size, rng=None):
        return self.sample_chunk_batch(self.base_chunks, batch_size=batch_size, rng=rng)

    def update_train_batch(self, batch_size, rng=None):
        return self.sample_chunk_batch(self.update_chunks, batch_size=batch_size, rng=rng)

    def get_retention_eval(self):
        return self.batch_from_chunks(self.retention_chunks)

    def get_replaced_base_eval(self):
        return self.batch_from_chunks(self.replaced_base_chunks)

    def get_update_eval(self):
        return self.batch_from_chunks(self.update_chunks)


class NQOpenHoldoutChunkDataset(DocumentChunkEditDataset):
    DEFAULT_PATH = "benchmark_data/NaturalQuestions/natural-questions-master/nq_open/NQ-open.dev.jsonl"

    def __init__(
        self,
        jsonl_path=None,
        chunk_len=72,
        stride=84,
        num_base_chunks=24,
        num_update_chunks=8,
        max_examples=80,
        shared_chars=None,
    ):
        path = Path(jsonl_path or self.DEFAULT_PATH)
        documents = []
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if len(documents) >= max_examples:
                        break
                    record = json.loads(line)
                    question = re.sub(r"\s+", " ", record.get("question", "")).strip()
                    answers = record.get("answer") or []
                    answer = re.sub(r"\s+", " ", answers[0]).strip() if answers else ""
                    if question and answer:
                        documents.append(f"[NQ DEV] Question: {question} Answer: {answer}")
        if not documents:
            raise ValueError("No readable NQ-open holdout examples were found.")
        super().__init__(
            documents,
            chunk_len=chunk_len,
            stride=stride,
            num_base_chunks=num_base_chunks,
            num_update_chunks=num_update_chunks,
            shared_chars=shared_chars,
        )


class FeverHoldoutChunkDataset(DocumentChunkEditDataset):
    DEFAULT_PATH = "benchmark_data/FEVER/shared_task_dev.jsonl"

    def __init__(
        self,
        jsonl_path=None,
        chunk_len=72,
        stride=84,
        num_base_chunks=24,
        num_update_chunks=8,
        max_examples=80,
        shared_chars=None,
    ):
        path = Path(jsonl_path or self.DEFAULT_PATH)
        documents = []
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if len(documents) >= max_examples:
                        break
                    record = json.loads(line)
                    claim = re.sub(r"\s+", " ", record.get("claim", "")).strip()
                    label = re.sub(r"\s+", " ", record.get("label", "")).strip()
                    if claim and label:
                        documents.append(f"[FEVER DEV] Claim: {claim} Label: {label}")
        if not documents:
            raise ValueError("No readable FEVER holdout examples were found.")
        super().__init__(
            documents,
            chunk_len=chunk_len,
            stride=stride,
            num_base_chunks=num_base_chunks,
            num_update_chunks=num_update_chunks,
            shared_chars=shared_chars,
        )


class FlagshipMixedTextCorpusDataset:
    """
    Mixed local text-pretraining corpus for the next RUMA phase.

    It draws from already-downloaded benchmark/task text so the repo can run a
    broader standalone pretraining pass without network access and without
    pretending the flagship training story is still only Alice-style chunks.
    """

    DEFAULT_NQ_PATH = "benchmark_data/NaturalQuestions/natural-questions-master/nq_open/NQ-open.train.jsonl"
    DEFAULT_FEVER_PATH = "benchmark_data/FEVER/train.jsonl"
    DEFAULT_OFFICIAL_EDIT_PATH = "benchmark_data/prepared_official_edits/official_edit_train.jsonl"
    DEFAULT_SQUAD_PATH = "benchmark_data/SQuAD v2/train-v2.0.json"

    def __init__(
        self,
        nq_path=None,
        fever_path=None,
        official_edit_path=None,
        squad_path=None,
        chunk_len=72,
        stride=84,
        num_base_chunks=48,
        num_update_chunks=12,
        max_examples_per_source=64,
        shared_chars=None,
    ):
        self.repo_root = Path(__file__).resolve().parent.parent
        nq_path = self.repo_root / (nq_path or self.DEFAULT_NQ_PATH)
        fever_path = self.repo_root / (fever_path or self.DEFAULT_FEVER_PATH)
        official_edit_path = self.repo_root / (official_edit_path or self.DEFAULT_OFFICIAL_EDIT_PATH)
        squad_path = self.repo_root / (squad_path or self.DEFAULT_SQUAD_PATH)

        documents = []
        documents.extend(self._load_nq_documents(nq_path, limit=max_examples_per_source))
        documents.extend(self._load_fever_documents(fever_path, limit=max_examples_per_source))
        documents.extend(self._load_official_edit_documents(official_edit_path, limit=max_examples_per_source))
        documents.extend(self._load_squad_documents(squad_path, limit=max_examples_per_source))

        if not documents:
            raise ValueError("No readable flagship mixed-text documents were found.")

        combined = " <SRCSEP> ".join(documents)
        total_needed = num_base_chunks + num_update_chunks
        chunks = []
        cursor = 0
        while len(chunks) < total_needed and cursor + chunk_len + 1 < len(combined):
            chunk = combined[cursor : cursor + chunk_len]
            if len(chunk) == chunk_len:
                chunks.append(chunk)
            cursor += stride

        if len(chunks) < total_needed:
            raise ValueError("Not enough mixed-corpus text chunks for the requested split.")

        self.base_chunks = chunks[:num_base_chunks]
        self.update_chunks = chunks[num_base_chunks : num_base_chunks + num_update_chunks]
        self.replaced_base_chunks = self.base_chunks[: len(self.update_chunks)]
        self.retention_chunks = self.base_chunks[-min(8, len(self.base_chunks)) :]
        self.pad_token, self.vocab, self.stoi, self.itos = build_shared_char_vocab(chunks, shared_chars=shared_chars)
        self.vocab_size = len(self.vocab)
        self.source_count = len(documents)

    def _normalize_text(self, text):
        return re.sub(r"\s+", " ", text).strip()

    def _load_nq_documents(self, path, limit):
        documents = []
        if not path.exists():
            return documents
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(documents) >= limit:
                    break
                record = json.loads(line)
                question = self._normalize_text(record.get("question", ""))
                answers = record.get("answer") or []
                answer = self._normalize_text(answers[0]) if answers else ""
                if question and answer:
                    documents.append(f"[NQ] Question: {question} Answer: {answer}")
        return documents

    def _load_fever_documents(self, path, limit):
        documents = []
        if not path.exists():
            return documents
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(documents) >= limit:
                    break
                record = json.loads(line)
                claim = self._normalize_text(record.get("claim", ""))
                label = self._normalize_text(record.get("label", ""))
                if claim and label:
                    documents.append(f"[FEVER] Claim: {claim} Label: {label}")
        return documents

    def _load_official_edit_documents(self, path, limit):
        documents = []
        if not path.exists():
            return documents
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(documents) >= limit:
                    break
                record = json.loads(line)
                dataset_name = self._normalize_text(record.get("dataset", "edit"))
                split_role = self._normalize_text(record.get("split_role", "example"))
                input_text = self._normalize_text(record.get("input", ""))
                target = self._normalize_text(record.get("target", ""))
                if input_text and target:
                    documents.append(
                        f"[EDIT {dataset_name}] Role: {split_role} Prompt: {input_text} Target: {target}"
                    )
        return documents

    def _load_squad_documents(self, path, limit):
        documents = []
        if not path.exists():
            return documents
        payload = json.loads(path.read_text(encoding="utf-8"))
        for article in payload.get("data", []):
            for paragraph in article.get("paragraphs", []):
                if len(documents) >= limit:
                    return documents
                context = self._normalize_text(paragraph.get("context", ""))[:260]
                qas = paragraph.get("qas", [])
                if not context or not qas:
                    continue
                qa = qas[0]
                question = self._normalize_text(qa.get("question", ""))
                answers = qa.get("answers") or []
                answer = self._normalize_text(answers[0].get("text", "")) if answers else ""
                if question and answer:
                    documents.append(f"[SQuAD] Context: {context} Question: {question} Answer: {answer}")
        return documents

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[int(idx)] for idx in ids if int(idx) != 0)

    def chunk_to_xy(self, chunk):
        seq = self.encode(chunk)
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

    def batch_from_chunks(self, chunks):
        xs = []
        ys = []
        for chunk in chunks:
            x, y = self.chunk_to_xy(chunk)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def sample_chunk_batch(self, chunks, batch_size, rng=None):
        rng = rng or random
        chosen = [rng.choice(chunks) for _ in range(batch_size)]
        return self.batch_from_chunks(chosen)

    def base_train_batch(self, batch_size, rng=None):
        return self.sample_chunk_batch(self.base_chunks, batch_size=batch_size, rng=rng)

    def update_train_batch(self, batch_size, rng=None):
        return self.sample_chunk_batch(self.update_chunks, batch_size=batch_size, rng=rng)

    def get_retention_eval(self):
        return self.batch_from_chunks(self.retention_chunks)

    def get_replaced_base_eval(self):
        return self.batch_from_chunks(self.replaced_base_chunks)

    def get_update_eval(self):
        return self.batch_from_chunks(self.update_chunks)
