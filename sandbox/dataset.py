import torch
import random
import re

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

        chars = sorted(set("".join(chunks)))
        self.pad_token = "\0"
        self.vocab = [self.pad_token] + chars
        self.vocab_size = len(self.vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

        self.base_chunks = chunks[:num_base_chunks]
        self.update_chunks = chunks[num_base_chunks : num_base_chunks + num_update_chunks]
        self.retention_chunks = self.base_chunks[-min(4, len(self.base_chunks)) :]

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

    def get_update_eval(self):
        return self.batch_from_chunks(self.update_chunks)
