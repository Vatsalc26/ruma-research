import json
from pathlib import Path

from real_doc_memory import RealDocRUMAIndex


FOCUS_NAMESPACES = [
    "mamba",
    "retro",
    "ruma_updates",
]


def compact_layout(layout, namespaces):
    result = {}
    for namespace in namespaces:
        if namespace in layout:
            result[namespace] = layout[namespace]
    return result


def run_real_doc_shard_demo():
    repo_root = Path(__file__).resolve().parent.parent
    index = RealDocRUMAIndex(
        repo_root=repo_root,
        num_shards=24,
        namespace_bandwidth=2,
    )

    print("\n=======================================================")
    print("      RUMA SHARD ASSIGNMENT AND REFRESH DEMO")
    print("=======================================================\n")

    print("[1/4] Building the paper corpus with a narrow shard band...")
    index.build_from_paths()
    print(json.dumps(index.stats(), indent=2))
    print("\nNamespace layout snapshot:")
    print(json.dumps(compact_layout(index.namespace_layout(), FOCUS_NAMESPACES), indent=2))

    print("\n[2/4] Adding two update namespaces on top of the corpus...")
    index.ingest_text_update(
        (
            "RUMA deployment notes say shard refresh should happen after major namespace growth. "
            "This note belongs to the deployment lineage."
        ),
        source="manual/ruma_deploy_notes.md",
        namespace="ruma_updates",
        timestamp="shard_v1",
        lineage="ruma_updates::deploy_notes",
        supersede_prior=False,
    )
    index.ingest_text_update(
        (
            "An operator guide says shard refresh may be needed after noisy update ingestion. "
            "This note belongs to a separate operator lineage."
        ),
        source="manual/ruma_operator_guide.md",
        namespace="ruma_updates",
        timestamp="ops_v1",
        lineage="ruma_updates::operator_guide",
        supersede_prior=False,
    )
    print(json.dumps(compact_layout(index.namespace_layout(), FOCUS_NAMESPACES), indent=2))

    print("\n[3/4] Widening namespace shard bandwidth and refreshing assignments...")
    index.namespace_bandwidth = 4
    index.refresh_shard_assignments()
    print(json.dumps(index.stats(), indent=2))

    print("\n[4/4] Inspecting namespace layout after refresh...")
    print(json.dumps(compact_layout(index.namespace_layout(), FOCUS_NAMESPACES), indent=2))

    print("\n[NOTE] This demo shows the current shard policy:")
    print("       each namespace gets a stable shard band,")
    print("       chunks hash into that local band,")
    print("       and a refresh rebuild can rebalance assignments when the policy changes.")


if __name__ == "__main__":
    run_real_doc_shard_demo()
