import json
from pathlib import Path


def _slugify(value):
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_") or "default"


def _required_string(entry, key):
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest entry `{entry.get('slug', 'unknown')}` is missing `{key}`.")
    return value.strip()


def _normalize_conflicts(conflicts):
    normalized = []
    for index, conflict in enumerate(conflicts or []):
        if not isinstance(conflict, dict):
            raise ValueError(f"Conflict entry #{index} must be a JSON object.")
        conflict_file = _required_string(conflict, "file")
        query = _required_string(conflict, "query")
        must_contain = _required_string(conflict, "must_contain")
        lineage = conflict.get("lineage")
        normalized.append(
            {
                "file": conflict_file,
                "query": query,
                "must_contain": must_contain,
                "lineage": None if lineage is None else str(lineage).strip(),
            }
        )
    return normalized


def load_external_corpus_manifest(manifest_path):
    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest file not found: {manifest_file}. "
            "Copy the template to manifest.json and fill it first."
        )

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    corpus_root = manifest_file.parent
    namespace = str(manifest.get("namespace", "external_docs")).strip() or "external_docs"
    corpus_name = str(manifest.get("corpus_name", manifest_file.stem)).strip() or manifest_file.stem

    raw_documents = manifest.get("documents", [])
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError("Manifest must contain a non-empty `documents` list.")

    normalized_documents = []
    for entry in raw_documents:
        if not isinstance(entry, dict):
            raise ValueError("Each manifest document entry must be a JSON object.")

        slug = _required_string(entry, "slug")
        subject = _required_string(entry, "subject")
        base_file = _required_string(entry, "base_file")
        update_file = _required_string(entry, "update_file")
        update_query = _required_string(entry, "update_query")
        retention_query = _required_string(entry, "retention_query")
        base_value = _required_string(entry, "base_value")
        updated_value = _required_string(entry, "updated_value")
        retained_value = _required_string(entry, "retained_value")
        base_query = str(entry.get("base_query", update_query)).strip() or update_query

        normalized_documents.append(
            {
                "slug": slug,
                "subject": subject,
                "namespace": _slugify(f"{namespace}_{slug}"),
                "base_file": base_file,
                "update_file": update_file,
                "base_query": base_query,
                "update_query": update_query,
                "retention_query": retention_query,
                "base_value": base_value,
                "updated_value": updated_value,
                "retained_value": retained_value,
                "conflicts": _normalize_conflicts(entry.get("conflicts", [])),
            }
        )

    return {
        "manifest_path": manifest_file,
        "corpus_root": corpus_root,
        "corpus_name": corpus_name,
        "namespace": namespace,
        "documents": normalized_documents,
    }


def missing_files_for_manifest(manifest):
    corpus_root = manifest["corpus_root"]
    missing = []
    for entry in manifest["documents"]:
        for relative_path in [entry["base_file"], entry["update_file"]]:
            path = corpus_root / relative_path
            if not path.exists():
                missing.append(path)
        for conflict in entry["conflicts"]:
            path = corpus_root / conflict["file"]
            if not path.exists():
                missing.append(path)
    return missing


def build_external_corpus_cases(manifest):
    namespace = manifest["namespace"]
    corpus_root = manifest["corpus_root"]

    base_files = [
        {
            "path": str(corpus_root / entry["base_file"]),
            "namespace": entry["namespace"],
            "slug": entry["slug"],
        }
        for entry in manifest["documents"]
    ]
    update_files = [
        {
            "base_path": str(corpus_root / entry["base_file"]),
            "update_path": str(corpus_root / entry["update_file"]),
            "namespace": entry["namespace"],
            "slug": entry["slug"],
        }
        for entry in manifest["documents"]
    ]

    base_known_cases = []
    update_cases = []
    retention_cases = []
    conflict_files = []
    conflict_cases = []

    for entry in manifest["documents"]:
        base_path = entry["base_file"].replace("\\", "/")
        update_path = entry["update_file"].replace("\\", "/")
        base_known_cases.append(
            {
                "name": f"{entry['slug']}_base_value",
                "query": entry["base_query"],
                "must_contain": entry["base_value"],
                "expected_source_suffix": base_path,
                "namespaces": [entry["namespace"]],
            }
        )
        update_cases.append(
            {
                "name": f"{entry['slug']}_updated_value",
                "query": entry["update_query"],
                "must_contain": entry["updated_value"],
                "must_not_contain": entry["base_value"],
                "expected_source_suffix": update_path,
                "namespaces": [entry["namespace"]],
            }
        )
        retention_cases.append(
            {
                "name": f"{entry['slug']}_retained_value",
                "query": entry["retention_query"],
                "must_contain": entry["retained_value"],
                "namespaces": [entry["namespace"]],
            }
        )

        for conflict in entry["conflicts"]:
            conflict_path = conflict["file"].replace("\\", "/")
            conflict_lineage = conflict["lineage"] or f"{entry['namespace']}::operator_guide::{entry['slug']}"
            conflict_files.append(
                {
                    "path": str(corpus_root / conflict["file"]),
                    "lineage": conflict_lineage,
                    "namespace": entry["namespace"],
                }
            )
            conflict_cases.append(
                {
                    "name": f"{entry['slug']}_conflicting_guidance",
                    "query": conflict["query"],
                    "must_contain": conflict["must_contain"],
                    "expected_source_suffix": conflict_path,
                    "expect_conflict": True,
                    "top_k": 5,
                    "max_sentences": 2,
                    "namespaces": [entry["namespace"]],
                }
            )

    return {
        "namespace": namespace,
        "base_files": base_files,
        "update_files": update_files,
        "conflict_files": conflict_files,
        "base_known_cases": base_known_cases,
        "update_cases": update_cases,
        "retention_cases": retention_cases,
        "conflict_cases": conflict_cases,
    }
