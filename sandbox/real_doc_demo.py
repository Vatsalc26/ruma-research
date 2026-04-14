import json
from pathlib import Path

from real_doc_memory import RealDocRUMAIndex


DEMO_QUERIES = [
    "How do retrieval augmented language models use external memory?",
    "What is sparse routing in mixture of experts models?",
    "How do state space models like Mamba handle long sequences?",
    "What is knowledge editing for large language models?",
]


def print_hits(query, hits):
    print(f"\nQuery: {query}")
    if not hits:
        print("  No hits.")
        return

    for hit in hits:
        print(
            f"  [{hit['score']:.3f}] {hit['namespace']} | "
            f"{hit['source']}#chunk{hit['chunk_index']}"
        )
        excerpt = hit["excerpt"][:220].encode("ascii", errors="replace").decode("ascii")
        print(f"    {excerpt}...")


def run_real_doc_demo():
    repo_root = Path(__file__).resolve().parent.parent
    index = RealDocRUMAIndex(repo_root=repo_root)

    print("\n=======================================================")
    print("      RUMA REAL DOCUMENT MEMORY DEMO")
    print("=======================================================\n")

    print("[1/3] Building a real document index from local paper markdown...")
    records = index.build_from_paths()
    print(json.dumps(index.stats(), indent=2))

    print("\n[2/3] Running retrieval over the indexed research corpus...")
    for query in DEMO_QUERIES:
        print_hits(query, index.search(query, top_k=3))

    print("\n[3/3] Demonstrating a live memory update on top of the paper corpus...")
    update_text = (
        "RUMA phase 2 now includes a real document memory prototype with provenance, "
        "chunk-level ingestion, and cited retrieval over local markdown papers."
    )
    index.ingest_text_update(
        update_text,
        source="manual/ruma_phase2_note.md",
        namespace="ruma_updates",
        timestamp="phase2_demo",
    )
    print_hits(
        "What does RUMA phase 2 include?",
        index.search("What does RUMA phase 2 include?", top_k=2),
    )

    print("\n[NOTE] This is a system-track prototype, not a paper-quality benchmark.")
    print("       It proves that RUMA can now ingest real local documents, preserve provenance,")
    print("       accept a live update, and retrieve cited chunks.")


if __name__ == "__main__":
    run_real_doc_demo()
