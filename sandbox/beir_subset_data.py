import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow.parquet as pq


@dataclass
class BeirQuery:
    query_id: str
    text: str
    title: str
    relevant_doc_ids: List[str]


@dataclass
class BeirDocument:
    doc_id: str
    title: str
    text: str


def _resolve_query_path(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "queries-00000-of-00001.parquet",
        dataset_root / "queries" / "queries-00000-of-00001.parquet",
        dataset_root / "Queries" / "queries-00000-of-00001.parquet",
        dataset_root / "Queires" / "queries-00000-of-00001.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find BEIR query parquet under {dataset_root}")


def _resolve_corpus_path(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "corpus-00000-of-00001.parquet",
        dataset_root / "corpus" / "corpus-00000-of-00001.parquet",
        dataset_root / "Corpus" / "corpus-00000-of-00001.parquet",
        dataset_root / "Courpus" / "corpus-00000-of-00001.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find BEIR corpus parquet under {dataset_root}")


def _resolve_qrels_path(dataset_root: Path, split_preference: str) -> Path:
    preferred = [
        dataset_root / f"{split_preference}.tsv",
        dataset_root / f"{split_preference} (1).tsv",
    ]
    fallback_patterns = [
        "*.tsv",
    ]

    for candidate in preferred:
        if candidate.exists():
            return candidate

    for pattern in fallback_patterns:
        matches = sorted(dataset_root.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Could not find BEIR qrels under {dataset_root}")


def _read_qrels(qrels_path: Path, query_limit: Optional[int]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    query_limit = None if query_limit is None else int(query_limit)

    with qrels_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query_id = str(row.get("query-id", "")).strip()
            doc_id = str(row.get("corpus-id", "")).strip()
            score = float(row.get("score", "0") or "0")
            if not query_id or not doc_id or score <= 0:
                continue

            if query_id not in mapping:
                if query_limit is not None and len(mapping) >= query_limit:
                    continue
                mapping[query_id] = []
            mapping[query_id].append(doc_id)

    return mapping


def _read_query_rows(query_path: Path, selected_query_ids: set) -> Dict[str, Dict[str, str]]:
    table = pq.read_table(query_path)
    rows = {}
    columns = table.to_pydict()
    query_ids = [str(item) for item in columns.get("_id", [])]
    titles = [str(item or "") for item in columns.get("title", [])]
    texts = [str(item or "") for item in columns.get("text", [])]

    for query_id, title, text in zip(query_ids, titles, texts):
        if query_id in selected_query_ids:
            rows[query_id] = {
                "title": title,
                "text": text,
            }

    return rows


def _scan_corpus_subset(
    corpus_path: Path,
    required_doc_ids: set,
    distractor_limit: int,
) -> List[BeirDocument]:
    parquet_file = pq.ParquetFile(corpus_path)
    selected_docs: Dict[str, BeirDocument] = {}
    distractors: List[BeirDocument] = []

    for batch in parquet_file.iter_batches(batch_size=4096):
        columns = batch.to_pydict()
        doc_ids = [str(item) for item in columns.get("_id", [])]
        titles = [str(item or "") for item in columns.get("title", [])]
        texts = [str(item or "") for item in columns.get("text", [])]

        for doc_id, title, text in zip(doc_ids, titles, texts):
            document = BeirDocument(doc_id=doc_id, title=title, text=text)
            if doc_id in required_doc_ids:
                selected_docs[doc_id] = document
            elif len(distractors) < distractor_limit:
                distractors.append(document)

        if len(selected_docs) >= len(required_doc_ids) and len(distractors) >= distractor_limit:
            break

    corpus = list(selected_docs.values())
    corpus.extend(distractors[:distractor_limit])
    return corpus


def load_beir_subset_records(
    dataset_root,
    query_limit=128,
    distractor_limit=4096,
    split_preference="test",
):
    dataset_root = Path(dataset_root)
    query_path = _resolve_query_path(dataset_root)
    corpus_path = _resolve_corpus_path(dataset_root)
    qrels_path = _resolve_qrels_path(dataset_root, split_preference=split_preference)

    qrels = _read_qrels(qrels_path, query_limit=query_limit)
    selected_query_ids = set(qrels.keys())
    query_rows = _read_query_rows(query_path, selected_query_ids)
    relevant_doc_ids = {
        doc_id
        for doc_ids in qrels.values()
        for doc_id in doc_ids
    }

    corpus = _scan_corpus_subset(
        corpus_path=corpus_path,
        required_doc_ids=relevant_doc_ids,
        distractor_limit=distractor_limit,
    )

    queries = []
    missing_query_ids = []
    for query_id, relevant_ids in qrels.items():
        query_row = query_rows.get(query_id)
        if query_row is None:
            missing_query_ids.append(query_id)
            continue
        queries.append(
            BeirQuery(
                query_id=query_id,
                text=query_row["text"],
                title=query_row["title"],
                relevant_doc_ids=relevant_ids,
            )
        )

    found_doc_ids = {document.doc_id for document in corpus}
    missing_doc_ids = sorted(relevant_doc_ids - found_doc_ids)

    return {
        "queries": queries,
        "corpus": corpus,
        "paths": {
            "query_path": str(query_path.resolve()),
            "corpus_path": str(corpus_path.resolve()),
            "qrels_path": str(qrels_path.resolve()),
        },
        "stats": {
            "query_count": len(queries),
            "requested_query_count": len(qrels),
            "relevant_doc_count": len(relevant_doc_ids),
            "corpus_doc_count": len(corpus),
            "distractor_count": max(0, len(corpus) - len(relevant_doc_ids)),
            "missing_query_count": len(missing_query_ids),
            "missing_relevant_doc_count": len(missing_doc_ids),
        },
    }
