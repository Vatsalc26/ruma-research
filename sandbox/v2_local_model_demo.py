import argparse
import json
from pathlib import Path

from external_corpus_benchmark import apply_conflicts, apply_updates, build_base_index
from external_corpus_manifest import load_external_corpus_manifest, missing_files_for_manifest
from local_llama_model import LocalLlamaCppModel, build_grounded_prompt, postprocess_grounded_answer
from real_doc_answerer import CitationFirstAnswerer


def run_local_model_demo(manifest_path):
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    manifest = load_external_corpus_manifest(manifest_path)
    missing_files = missing_files_for_manifest(manifest)
    if missing_files:
        print("\n[ERROR] The external corpus manifest is present, but the markdown files are missing.")
        print("Populate the files below, then rerun the demo:\n")
        for path in missing_files:
            print(f"- {path}")
        return 1

    base_files = [
        {
            "path": str(manifest["corpus_root"] / entry["base_file"]),
            "namespace": entry["namespace"],
            "slug": entry["slug"],
        }
        for entry in manifest["documents"]
    ]
    update_files = [
        {
            "base_path": str(manifest["corpus_root"] / entry["base_file"]),
            "update_path": str(manifest["corpus_root"] / entry["update_file"]),
            "namespace": entry["namespace"],
            "slug": entry["slug"],
        }
        for entry in manifest["documents"]
    ]
    conflict_files = []
    for entry in manifest["documents"]:
        for conflict in entry["conflicts"]:
            conflict_files.append(
                {
                    "path": str(manifest["corpus_root"] / conflict["file"]),
                    "lineage": conflict["lineage"] or f"{entry['namespace']}::operator_guide::{entry['slug']}",
                    "namespace": entry["namespace"],
                }
            )

    index, _ = build_base_index(repo_root, base_files)
    apply_updates(index, update_files, supersede_prior=True)
    apply_conflicts(index, conflict_files, manifest["namespace"])
    answerer = CitationFirstAnswerer(index)
    local_model = LocalLlamaCppModel(repo_root=repo_root)

    if not local_model.is_available():
        status = local_model.availability()
        print(json.dumps({"local_model_available": False, **status}, indent=2))
        return 1

    selected_entries = []
    if manifest["documents"]:
        selected_entries.append(manifest["documents"][0])
    if len(manifest["documents"]) > 4:
        selected_entries.append(manifest["documents"][4])

    packets = []
    for entry in selected_entries:
        packet = answerer.answer(
            entry["update_query"],
            top_k=4,
            namespaces=[entry["namespace"]],
            max_sentences=2,
        )
        prompt = build_grounded_prompt(entry["update_query"], packet, max_evidence=3)
        model_answer = local_model.generate(
            prompt,
            max_tokens=48,
            temperature=0.0,
        )
        final_answer = postprocess_grounded_answer(
            entry["update_query"],
            model_answer,
            packet,
        )
        packets.append(
            {
                "name": entry["slug"],
                "query": entry["update_query"],
                "citation_first_answer": packet["answer"],
                "citations": packet["citations"],
                "conflicts": packet["conflicts"],
                "raw_local_model_answer": model_answer,
                "local_model_answer": final_answer,
            }
        )

    output = {
        "local_model_available": True,
        "model_status": local_model.availability(),
        "cases": packets,
    }
    output_path = results_dir / "v2_local_model_demo.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\n[NOTE] Results written to: {output_path}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="external_corpora/python_ecosystem_changes/manifest.json",
        help="Path to a filled external corpus manifest JSON file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_local_model_demo(parse_args().manifest))
