import json
from pathlib import Path

from real_doc_answerer import CitationFirstAnswerer
from real_doc_memory import RealDocRUMAIndex


ANSWER_QUERIES = [
    "What is retrieval augmented generation?",
    "How do Mamba models help with long sequences?",
    "What is knowledge editing for large language models?",
]


def print_answer(packet):
    answer = packet["answer"].encode("ascii", errors="replace").decode("ascii")
    print(f"\nQuery: {packet['query']}")
    print(f"Answer: {answer}")
    if not packet["citations"]:
        print("Citations: none")
        return

    print("Citations:")
    for citation in packet["citations"]:
        print(
            f"  - {citation['source']}#chunk{citation['chunk_index']} "
            f"({citation['namespace']}, score={citation['score']:.3f})"
        )


def run_real_doc_answer_demo():
    repo_root = Path(__file__).resolve().parent.parent
    index = RealDocRUMAIndex(repo_root=repo_root)
    answerer = CitationFirstAnswerer(index)

    print("\n=======================================================")
    print("      RUMA REAL DOCUMENT ANSWER DEMO")
    print("=======================================================\n")

    print("[1/3] Building the paper corpus index...")
    index.build_from_paths()
    print(json.dumps(index.stats(), indent=2))

    print("\n[2/3] Answering grounded questions with citations...")
    for query in ANSWER_QUERIES:
        print_answer(answerer.answer(query, top_k=4, max_sentences=2))

    print("\n[3/3] Applying a live update and answering again...")
    update_text = (
        "RUMA now has a citation-first answer layer over the real document memory. "
        "The current phase uses extractive grounded answers with source citations instead "
        "of pretending to generate fully free-form reliable answers."
    )
    index.ingest_text_update(
        update_text,
        source="manual/ruma_answer_layer.md",
        namespace="ruma_updates",
        timestamp="phase2_answer_demo",
    )
    print_answer(answerer.answer("What answer layer does RUMA use now?", top_k=3, max_sentences=2))

    print("\n[NOTE] This is still a lightweight grounded answer path, not a final generative architecture.")


if __name__ == "__main__":
    run_real_doc_answer_demo()
