from dataclasses import dataclass

from real_doc_memory import clean_text


def normalize_text(text):
    return clean_text(text).lower()


@dataclass
class SubjectAnswerEntry:
    dataset_name: str
    subject: str
    normalized_subject: str
    answer: str
    timestamp: str
    case_id: str


class SubjectOverrideBaseline:
    """
    Simple entity-level override baseline for official edit benchmarks.

    This is intentionally stronger than naive append because it ignores prompt
    nuance and simply returns the latest stored answer for the matched subject.
    It is expected to solve many canonical/paraphrase updates cheaply while
    over-generalizing on retention prompts that mention the same subject.
    """

    def __init__(self):
        self.entries = []

    def insert(self, dataset_name, subject, answer, timestamp, case_id):
        normalized_subject = normalize_text(subject)
        if not normalized_subject or not answer:
            return
        self.entries.append(
            SubjectAnswerEntry(
                dataset_name=dataset_name,
                subject=subject,
                normalized_subject=normalized_subject,
                answer=answer,
                timestamp=timestamp,
                case_id=str(case_id),
            )
        )

    def answer(self, query):
        normalized_query = normalize_text(query)
        if not normalized_query:
            return {"answer": "", "hit": None}

        matches = [entry for entry in self.entries if entry.normalized_subject in normalized_query]
        if not matches:
            return {"answer": "", "hit": None}

        matches.sort(
            key=lambda entry: (
                1 if entry.timestamp == "update" else 0,
                len(entry.normalized_subject),
            ),
            reverse=True,
        )
        best = matches[0]
        return {
            "answer": best.answer,
            "hit": {
                "dataset_name": best.dataset_name,
                "subject": best.subject,
                "timestamp": best.timestamp,
                "case_id": best.case_id,
            },
        }
