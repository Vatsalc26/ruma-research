import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PromptCase:
    prompt: str
    expected_answer: str
    case_kind: str


@dataclass
class OfficialEditRecord:
    dataset_name: str
    case_id: str
    subject: str
    canonical_prompt: str
    target_true: str
    target_new: str
    paraphrase_prompts: List[str] = field(default_factory=list)
    retention_cases: List[PromptCase] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _load_json_array(path: Path):
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in {path}")
    return data


def validate_json_file(path):
    try:
        _load_json_array(Path(path))
    except Exception as exc:  # pragma: no cover - used for preflight
        return {"valid": False, "error": str(exc)}
    return {"valid": True, "error": None}


def load_counterfact_records(path, limit: Optional[int] = None):
    rows = _load_json_array(Path(path))
    if limit is not None:
        rows = rows[: int(limit)]

    records = []
    for index, row in enumerate(rows):
        if "requested_rewrite" in row:
            rewrite = row.get("requested_rewrite") or {}
            prompt_template = str(rewrite.get("prompt", "")).strip()
            subject = str(rewrite.get("subject", "")).strip()
            canonical_prompt = prompt_template.format(subject) if "{}" in prompt_template else prompt_template
            target_true = str((rewrite.get("target_true") or {}).get("str", "")).strip()
            target_new = str((rewrite.get("target_new") or {}).get("str", "")).strip()

            paraphrases = [str(item).strip() for item in row.get("paraphrase_prompts", []) if str(item).strip()]

            retention_cases = []
            for prompt in row.get("neighborhood_prompts", []):
                prompt = str(prompt).strip()
                if prompt and target_true:
                    retention_cases.append(
                        PromptCase(
                            prompt=prompt,
                            expected_answer=target_true,
                            case_kind="neighborhood",
                        )
                    )

            records.append(
                OfficialEditRecord(
                    dataset_name="counterfact",
                    case_id=str(row.get("case_id", index)),
                    subject=subject,
                    canonical_prompt=canonical_prompt,
                    target_true=target_true,
                    target_new=target_new,
                    paraphrase_prompts=paraphrases,
                    retention_cases=retention_cases,
                    metadata={
                        "raw_index": index,
                        "relation_id": str(rewrite.get("relation_id", "")).strip(),
                        "pararel_idx": row.get("pararel_idx"),
                        "attribute_prompt_count": len(row.get("attribute_prompts", [])),
                        "generation_prompt_count": len(row.get("generation_prompts", [])),
                    },
                )
            )
            continue

        true_answers = row.get("answers") or []
        target_true = str(true_answers[0]).strip() if true_answers else str(row.get("pred", "")).strip()
        target_new = str(row.get("alt", "")).strip()
        canonical_prompt = str(row.get("src", "")).strip()
        paraphrases = []
        rephrase = str(row.get("rephrase", "")).strip()
        if rephrase:
            paraphrases.append(rephrase)

        retention_cases = []
        loc_prompt = str(row.get("loc", "")).strip()
        loc_answer = str(row.get("loc_ans", "")).strip()
        if loc_prompt and loc_answer:
            retention_cases.append(
                PromptCase(
                    prompt=loc_prompt,
                    expected_answer=loc_answer,
                    case_kind="locality",
                )
            )

        records.append(
            OfficialEditRecord(
                dataset_name="counterfact",
                case_id=str(index),
                subject=str(row.get("subject", "")).strip(),
                canonical_prompt=canonical_prompt,
                target_true=target_true,
                target_new=target_new,
                paraphrase_prompts=paraphrases,
                retention_cases=retention_cases,
                metadata={
                    "raw_index": index,
                    "cond": str(row.get("cond", "")).strip(),
                    "pred": str(row.get("pred", "")).strip(),
                },
            )
        )
    return records


def load_zsre_records(path, limit: Optional[int] = None):
    rows = _load_json_array(Path(path))
    if limit is not None:
        rows = rows[: int(limit)]

    records = []
    for index, row in enumerate(rows):
        if "requested_rewrite" not in row:
            true_answers = row.get("answers") or []
            target_true = str(true_answers[0]).strip() if true_answers else str(row.get("pred", "")).strip()
            target_new = str(row.get("alt", "")).strip()
            canonical_prompt = str(row.get("src", "")).strip()
            paraphrases = []
            rephrase = str(row.get("rephrase", "")).strip()
            if rephrase:
                paraphrases.append(rephrase)

            retention_cases = []
            loc_prompt = str(row.get("loc", "")).strip()
            loc_answer = str(row.get("loc_ans", "")).strip()
            if loc_prompt and loc_answer:
                retention_cases.append(
                    PromptCase(
                        prompt=loc_prompt,
                        expected_answer=loc_answer,
                        case_kind="locality",
                    )
                )

            records.append(
                OfficialEditRecord(
                    dataset_name="zsre",
                    case_id=str(index),
                    subject=str(row.get("subject", "")).strip(),
                    canonical_prompt=canonical_prompt,
                    target_true=target_true,
                    target_new=target_new,
                    paraphrase_prompts=paraphrases,
                    retention_cases=retention_cases,
                    metadata={
                        "raw_index": index,
                        "cond": str(row.get("cond", "")).strip(),
                        "pred": str(row.get("pred", "")).strip(),
                    },
                )
            )
            continue

        rewrite = row.get("requested_rewrite") or {}
        prompt_template = str(rewrite.get("prompt", "")).strip()
        subject = str(rewrite.get("subject", "")).strip()
        canonical_prompt = prompt_template.format(subject) if "{}" in prompt_template else prompt_template
        target_true = str((rewrite.get("target_true") or {}).get("str", "")).strip()
        target_new = str((rewrite.get("target_new") or {}).get("str", "")).strip()

        paraphrases = [str(item).strip() for item in row.get("paraphrase_prompts", []) if str(item).strip()]

        retention_cases = []
        for prompt in row.get("neighborhood_prompts", []):
            prompt = str(prompt).strip()
            if prompt and target_true:
                retention_cases.append(
                    PromptCase(
                        prompt=prompt,
                        expected_answer=target_true,
                        case_kind="neighborhood",
                    )
                )

        records.append(
            OfficialEditRecord(
                dataset_name="zsre",
                case_id=str(row.get("case_id", index)),
                subject=subject,
                canonical_prompt=canonical_prompt,
                target_true=target_true,
                target_new=target_new,
                paraphrase_prompts=paraphrases,
                retention_cases=retention_cases,
                metadata={
                    "raw_index": index,
                    "relation_id": str(rewrite.get("relation_id", "")).strip(),
                    "pararel_idx": row.get("pararel_idx"),
                    "attribute_prompt_count": len(row.get("attribute_prompts", [])),
                },
            )
        )
    return records
