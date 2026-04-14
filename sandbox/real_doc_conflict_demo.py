import json
from pathlib import Path

from real_doc_answerer import CitationFirstAnswerer
from real_doc_memory import RealDocRUMAIndex


def print_packet(label, packet):
    answer = packet["answer"].encode("ascii", errors="replace").decode("ascii")
    print(f"\n{label}")
    print(f"Query: {packet['query']}")
    print(f"Answer: {answer}")
    print("Citations:")
    for citation in packet["citations"]:
        print(
            f"  - {citation['source']}#chunk{citation['chunk_index']} "
            f"({citation['namespace']}, score={citation['score']:.3f})"
        )
    if packet["conflicts"]:
        print("Conflict notices:")
        for conflict in packet["conflicts"]:
            if conflict["type"] == "multi_source_answer":
                print(f"  - {conflict['message']}")
                for source in conflict["sources"]:
                    print(
                        f"    * {source['source']}#chunk{source['chunk_index']} "
                        f"({source['namespace']}, score={source['score']:.3f})"
                    )
            else:
                print(
                    f"  - {conflict['source']}#chunk{conflict['chunk_index']} "
                    f"({conflict['namespace']}, score={conflict['score']:.3f})"
                )
    else:
        print("Conflict notices: none")


def run_real_doc_conflict_demo():
    repo_root = Path(__file__).resolve().parent.parent
    index = RealDocRUMAIndex(repo_root=repo_root)
    answerer = CitationFirstAnswerer(index)

    print("\n=======================================================")
    print("      RUMA VERSIONING AND CONFLICT DEMO")
    print("=======================================================\n")

    print("[1/4] Building the base paper corpus...")
    index.build_from_paths()
    print(json.dumps(index.stats(), indent=2))

    print("\n[2/4] Writing an original guidance record...")
    index.ingest_text_update(
        (
            "RUMA deployment notes say the default retrieval top_k is 2 for the current system track. "
            "This setting is part of the deployment guidance."
        ),
        source="manual/ruma_deploy_notes.md",
        namespace="ruma_updates",
        timestamp="v1",
        lineage="ruma_updates::deploy_notes",
        supersede_prior=False,
    )
    print_packet(
        "Before superseding:",
        answerer.answer(
            "What is the default retrieval top_k in RUMA deployment notes?",
            top_k=4,
            namespaces=["ruma_updates"],
        ),
    )

    print("\n[3/4] Writing a newer revision on the same lineage...")
    index.ingest_text_update(
        (
            "RUMA deployment notes now say the default retrieval top_k is 4 for the current system track. "
            "This newer note supersedes the older deployment guidance."
        ),
        source="manual/ruma_deploy_notes.md",
        namespace="ruma_updates",
        timestamp="v2",
        lineage="ruma_updates::deploy_notes",
        supersede_prior=True,
    )
    print_packet(
        "After superseding the old lineage:",
        answerer.answer(
            "What is the default retrieval top_k in RUMA deployment notes?",
            top_k=4,
            namespaces=["ruma_updates"],
        ),
    )

    print("\n[4/4] Adding a conflicting active source from another lineage...")
    index.ingest_text_update(
        (
            "An alternative operator guide says the default retrieval top_k in RUMA guidance is 6 for large retrieval jobs. "
            "This guide does not supersede the main deployment notes, but it is active guidance from another source."
        ),
        source="manual/ruma_operator_guide.md",
        namespace="ruma_updates",
        timestamp="ops_v1",
        lineage="ruma_updates::operator_guide",
        supersede_prior=False,
    )
    print_packet(
        "After introducing a conflicting active source:",
        answerer.answer(
            "What is the default retrieval top_k in RUMA guidance?",
            top_k=5,
            namespaces=["ruma_updates"],
        ),
    )

    active = len(index.store.records(statuses={"active"}))
    superseded = len(index.store.records(statuses={"superseded"}))
    print("\nStatus counts:")
    print(json.dumps({"active_records": active, "superseded_records": superseded}, indent=2))

    print("\n[NOTE] This demo shows the Phase 2 policy:")
    print("       same-lineage updates supersede older active guidance,")
    print("       but cross-source disagreement is surfaced instead of silently erased.")


if __name__ == "__main__":
    run_real_doc_conflict_demo()
