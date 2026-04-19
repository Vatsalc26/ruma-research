from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from real_doc_memory import clean_text, slugify


@dataclass
class HotpotContextDoc:
    title: str
    namespace: str
    sentences: List[str] = field(default_factory=list)


@dataclass
class HotpotQARecord:
    question_id: str
    question: str
    answer: str
    question_type: str
    level: str
    supporting_facts: List[Tuple[str, int]] = field(default_factory=list)
    context_titles: List[str] = field(default_factory=list)
    context_namespaces: List[str] = field(default_factory=list)


def _coerce_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _coerce_mapping(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return {}


def _normalize_title(title):
    return clean_text(str(title or "")).strip()


def _normalize_sentence(sentence):
    return clean_text(str(sentence or "")).strip()


def _context_from_row(row):
    context = _coerce_mapping(row.get("context"))
    titles = [_normalize_title(item) for item in _coerce_list(context.get("title"))]
    sentence_groups = _coerce_list(context.get("sentences"))

    docs = []
    for index, title in enumerate(titles):
        raw_sentences = sentence_groups[index] if index < len(sentence_groups) else []
        sentences = [_normalize_sentence(item) for item in _coerce_list(raw_sentences)]
        sentences = [item for item in sentences if item]
        if not title or not sentences:
            continue
        docs.append(
            HotpotContextDoc(
                title=title,
                namespace=slugify(f"hotpotqa_{title}"),
                sentences=sentences,
            )
        )
    return docs


def _supporting_facts_from_row(row):
    supporting = _coerce_mapping(row.get("supporting_facts"))
    titles = [_normalize_title(item) for item in _coerce_list(supporting.get("title"))]
    sent_ids = [int(item) for item in _coerce_list(supporting.get("sent_id"))]

    facts = []
    for index, title in enumerate(titles):
        sent_id = sent_ids[index] if index < len(sent_ids) else None
        if title and sent_id is not None:
            facts.append((title, int(sent_id)))
    return facts


def load_hotpotqa_records(
    parquet_path,
    bridge_limit: Optional[int] = None,
    comparison_limit: Optional[int] = None,
):
    path = Path(parquet_path)
    frame = pd.read_parquet(path)

    bridge_limit = None if bridge_limit is None else int(bridge_limit)
    comparison_limit = None if comparison_limit is None else int(comparison_limit)

    bridge = []
    comparison = []
    corpus_by_title: Dict[str, HotpotContextDoc] = {}

    for _, row in frame.iterrows():
        question_type = clean_text(str(row.get("type", "")).lower())
        if question_type not in {"bridge", "comparison"}:
            continue

        if question_type == "bridge" and bridge_limit is not None and len(bridge) >= bridge_limit:
            continue
        if question_type == "comparison" and comparison_limit is not None and len(comparison) >= comparison_limit:
            continue

        context_docs = _context_from_row(row)
        supporting_facts = _supporting_facts_from_row(row)
        if not context_docs or not supporting_facts:
            continue

        context_titles = [doc.title for doc in context_docs]
        context_namespaces = [doc.namespace for doc in context_docs]
        record = HotpotQARecord(
            question_id=str(row.get("id", "")).strip(),
            question=clean_text(str(row.get("question", ""))),
            answer=clean_text(str(row.get("answer", ""))),
            question_type=question_type,
            level=clean_text(str(row.get("level", ""))),
            supporting_facts=supporting_facts,
            context_titles=context_titles,
            context_namespaces=context_namespaces,
        )

        for doc in context_docs:
            corpus_by_title.setdefault(doc.title, doc)

        if question_type == "bridge":
            bridge.append(record)
        else:
            comparison.append(record)

        if bridge_limit is not None and comparison_limit is not None:
            if len(bridge) >= bridge_limit and len(comparison) >= comparison_limit:
                break

    all_records = bridge + comparison
    supporting_titles = {title for record in all_records for title, _ in record.supporting_facts}

    return {
        "bridge": bridge,
        "comparison": comparison,
        "records": all_records,
        "corpus_docs": list(corpus_by_title.values()),
        "stats": {
            "bridge_count": len(bridge),
            "comparison_count": len(comparison),
            "question_count": len(all_records),
            "unique_context_doc_count": len(corpus_by_title),
            "unique_support_title_count": len(supporting_titles),
        },
    }
