import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from real_doc_memory import slugify


@dataclass
class FeverEvidenceSentence:
    page_id: str
    sentence_id: int
    text: str


@dataclass
class FeverRecord:
    claim_id: int
    claim: str
    label: str
    verifiable: bool
    evidence_keys: Set[Tuple[str, int]] = field(default_factory=set)
    evidence_pages: Set[str] = field(default_factory=set)


@dataclass
class FeverSentenceRecord:
    page_id: str
    sentence_id: int
    namespace: str
    text: str


def parse_fever_wiki_lines(lines_text):
    sentences = {}
    for raw_line in str(lines_text or "").splitlines():
        parts = raw_line.split("\t")
        if len(parts) < 2:
            continue
        sent_id_raw = parts[0].strip()
        sent_text = parts[1].strip()
        if not sent_id_raw.isdigit():
            continue
        if not sent_text:
            continue
        sentences[int(sent_id_raw)] = sent_text
    return sentences


def load_fever_records(
    path,
    support_limit: Optional[int] = None,
    refute_limit: Optional[int] = None,
    nei_limit: Optional[int] = None,
):
    support_limit = None if support_limit is None else int(support_limit)
    refute_limit = None if refute_limit is None else int(refute_limit)
    nei_limit = None if nei_limit is None else int(nei_limit)

    supports = []
    refutes = []
    nei = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            row = json.loads(raw_line)
            label = str(row.get("label", "")).strip().upper()
            claim = str(row.get("claim", "")).strip()
            if not claim:
                continue

            evidence_keys = set()
            evidence_pages = set()
            for evidence_group in row.get("evidence", []):
                for item in evidence_group:
                    if len(item) < 4:
                        continue
                    page_id = item[2]
                    sentence_id = item[3]
                    if not isinstance(page_id, str) or page_id is None:
                        continue
                    if sentence_id is None or not isinstance(sentence_id, int):
                        continue
                    evidence_keys.add((page_id, int(sentence_id)))
                    evidence_pages.add(page_id)

            record = FeverRecord(
                claim_id=int(row.get("id", 0)),
                claim=claim,
                label=label,
                verifiable=label in {"SUPPORTS", "REFUTES"},
                evidence_keys=evidence_keys,
                evidence_pages=evidence_pages,
            )

            if label == "SUPPORTS":
                if support_limit is None or len(supports) < support_limit:
                    supports.append(record)
            elif label == "REFUTES":
                if refute_limit is None or len(refutes) < refute_limit:
                    refutes.append(record)
            elif label == "NOT ENOUGH INFO":
                if nei_limit is None or len(nei) < nei_limit:
                    nei.append(record)

            if support_limit is not None and refute_limit is not None and nei_limit is not None:
                if len(supports) >= support_limit and len(refutes) >= refute_limit and len(nei) >= nei_limit:
                    break

    return {
        "supports": supports,
        "refutes": refutes,
        "nei": nei,
    }


def load_fever_wiki_subset(wiki_dir, required_page_ids, distractor_limit=256):
    wiki_dir = Path(wiki_dir)
    required_page_ids = set(required_page_ids)
    distractor_limit = int(distractor_limit)

    selected_pages: Dict[str, Dict[int, str]] = {}
    distractor_pages: Dict[str, Dict[int, str]] = {}

    for wiki_file in sorted(wiki_dir.glob("wiki-*.jsonl")):
        with wiki_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                row = json.loads(raw_line)
                page_id = str(row.get("id", "")).strip()
                if not page_id:
                    continue
                parsed_lines = parse_fever_wiki_lines(row.get("lines", ""))
                if not parsed_lines:
                    continue

                if page_id in required_page_ids:
                    selected_pages[page_id] = parsed_lines
                elif len(distractor_pages) < distractor_limit:
                    distractor_pages[page_id] = parsed_lines

        if required_page_ids.issubset(selected_pages.keys()) and len(distractor_pages) >= distractor_limit:
            break

    sentence_records = []
    for page_id, sentence_map in {**selected_pages, **distractor_pages}.items():
        namespace = slugify(f"fever_{page_id}")
        for sentence_id, sentence_text in sentence_map.items():
            sentence_records.append(
                FeverSentenceRecord(
                    page_id=page_id,
                    sentence_id=sentence_id,
                    namespace=namespace,
                    text=sentence_text,
                )
            )

    return {
        "selected_page_count": len(selected_pages),
        "distractor_page_count": len(distractor_pages),
        "sentence_records": sentence_records,
    }
