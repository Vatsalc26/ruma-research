import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from real_doc_memory import slugify


@dataclass
class SquadV2Question:
    question_id: str
    title: str
    namespace: str
    paragraph_id: str
    paragraph_index: int
    question: str
    answers: List[str] = field(default_factory=list)
    plausible_answers: List[str] = field(default_factory=list)
    is_impossible: bool = False
    context: str = ""


@dataclass
class SquadV2Paragraph:
    paragraph_id: str
    title: str
    namespace: str
    paragraph_index: int
    context: str
    question_ids: List[str] = field(default_factory=list)


def load_squad_v2_records(
    path,
    answerable_limit: Optional[int] = None,
    impossible_limit: Optional[int] = None,
    distractor_limit: int = 256,
):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("data", [])

    answerable = []
    impossible = []
    selected_paragraph_ids = set()
    selected_paragraphs = {}
    distractor_paragraphs = []

    answerable_limit = None if answerable_limit is None else int(answerable_limit)
    impossible_limit = None if impossible_limit is None else int(impossible_limit)
    distractor_limit = int(distractor_limit)

    for article in data:
        title = str(article.get("title", "")).strip() or "untitled"
        namespace = slugify(f"squad_v2_{title}")
        for paragraph_index, paragraph in enumerate(article.get("paragraphs", [])):
            context = str(paragraph.get("context", "")).strip()
            paragraph_id = f"{title}::{paragraph_index}"
            paragraph_record = SquadV2Paragraph(
                paragraph_id=paragraph_id,
                title=title,
                namespace=namespace,
                paragraph_index=paragraph_index,
                context=context,
            )

            paragraph_selected = False
            for qa in paragraph.get("qas", []):
                question = str(qa.get("question", "")).strip()
                if not question:
                    continue
                answers = [str(item.get("text", "")).strip() for item in qa.get("answers", []) if str(item.get("text", "")).strip()]
                plausible_answers = [
                    str(item.get("text", "")).strip()
                    for item in qa.get("plausible_answers", [])
                    if str(item.get("text", "")).strip()
                ]
                is_impossible = bool(qa.get("is_impossible", False))
                record = SquadV2Question(
                    question_id=str(qa.get("id", "")),
                    title=title,
                    namespace=namespace,
                    paragraph_id=paragraph_id,
                    paragraph_index=paragraph_index,
                    question=question,
                    answers=answers,
                    plausible_answers=plausible_answers,
                    is_impossible=is_impossible,
                    context=context,
                )

                if is_impossible:
                    if impossible_limit is None or len(impossible) < impossible_limit:
                        impossible.append(record)
                        paragraph_record.question_ids.append(record.question_id)
                        paragraph_selected = True
                else:
                    if answerable_limit is None or len(answerable) < answerable_limit:
                        answerable.append(record)
                        paragraph_record.question_ids.append(record.question_id)
                        paragraph_selected = True

            if paragraph_selected:
                selected_paragraph_ids.add(paragraph_id)
                selected_paragraphs[paragraph_id] = paragraph_record
            elif len(distractor_paragraphs) < distractor_limit and context:
                distractor_paragraphs.append(paragraph_record)

            if answerable_limit is not None and impossible_limit is not None:
                if len(answerable) >= answerable_limit and len(impossible) >= impossible_limit:
                    continue

    corpus_paragraphs = list(selected_paragraphs.values())
    for distractor in distractor_paragraphs[:distractor_limit]:
        if distractor.paragraph_id not in selected_paragraph_ids:
            corpus_paragraphs.append(distractor)

    return {
        "answerable": answerable,
        "impossible": impossible,
        "corpus_paragraphs": corpus_paragraphs,
        "stats": {
            "answerable_count": len(answerable),
            "impossible_count": len(impossible),
            "selected_paragraph_count": len(selected_paragraphs),
            "corpus_paragraph_count": len(corpus_paragraphs),
            "distractor_count": max(0, len(corpus_paragraphs) - len(selected_paragraphs)),
        },
    }
