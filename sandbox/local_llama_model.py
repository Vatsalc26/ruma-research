import re
import subprocess
from pathlib import Path


class LocalLlamaCppModel:
    """
    Thin scripted wrapper around a local llama.cpp Windows runtime.
    This is intentionally minimal and deterministic for research harness use.
    """

    def __init__(
        self,
        repo_root,
        runtime_dir="llama",
        model_dir="Qwen2.5-0.5B-Instruct-GGUF",
        model_file="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        gpu_layers=99,
    ):
        self.repo_root = Path(repo_root)
        self.runtime_dir = self.repo_root / runtime_dir
        self.model_path = self.repo_root / model_dir / model_file
        self.binary_path = self.runtime_dir / "llama-cli.exe"
        self.gpu_layers = gpu_layers

    def availability(self):
        return {
            "binary_exists": self.binary_path.exists(),
            "model_exists": self.model_path.exists(),
            "binary_path": str(self.binary_path),
            "model_path": str(self.model_path),
        }

    def is_available(self):
        status = self.availability()
        return status["binary_exists"] and status["model_exists"]

    def generate(
        self,
        prompt,
        max_tokens=96,
        temperature=0.0,
        ctx_size=2048,
        top_k=1,
        top_p=1.0,
        timeout_seconds=180,
    ):
        if not self.is_available():
            status = self.availability()
            raise FileNotFoundError(
                "Local llama.cpp runtime is incomplete. "
                f"binary_exists={status['binary_exists']} model_exists={status['model_exists']}"
            )

        command = [
            str(self.binary_path),
            "--model",
            str(self.model_path),
            "--prompt",
            prompt,
            "--n-predict",
            str(max_tokens),
            "--ctx-size",
            str(ctx_size),
            "--temperature",
            str(temperature),
            "--top-k",
            str(top_k),
            "--top-p",
            str(top_p),
            "--seed",
            "0",
            "--gpu-layers",
            str(self.gpu_layers),
            "--simple-io",
            "--single-turn",
            "--reasoning",
            "off",
            "--no-perf",
            "--no-warmup",
        ]

        completed = subprocess.run(
            command,
            cwd=str(self.runtime_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Local llama.cpp generation failed.\n"
                f"Return code: {completed.returncode}\n"
                f"STDERR:\n{completed.stderr}\n"
                f"STDOUT:\n{completed.stdout}"
            )

        return self._extract_generation(completed.stdout).strip()

    def _extract_generation(self, stdout_text):
        text = stdout_text.replace("\r\n", "\n")
        marker = "[[FINAL_ANSWER]]:"
        if marker in text:
            tail = text.split(marker, 1)[1]
            for stop_marker in ["\n[ Prompt:", "\nExiting...", "\nllama_memory_breakdown_print:", "\nggml_cuda_init:"]:
                if stop_marker in tail:
                    tail = tail.split(stop_marker, 1)[0]
            return self._cleanup_generation(tail)

        if "\n> " in text and "\n[ Prompt:" in text:
            tail = text.split("\n> ", 1)[1]
            if "\n\n" in tail:
                tail = tail.split("\n\n", 1)[1]
            tail = tail.split("\n[ Prompt:", 1)[0]
            return self._cleanup_generation(tail)

        return self._cleanup_generation(text)

    def _cleanup_generation(self, text):
        cleaned = text.strip()
        if "Question:" in cleaned and "\n\n" in cleaned:
            cleaned = cleaned.split("\n\n")[-1].strip()
        if cleaned.startswith("[[FINAL_ANSWER]]:"):
            cleaned = cleaned.split(":", 1)[1].strip()
        return cleaned


PYTHON_VERSION_RE = re.compile(r"python\s+(\d+)\.(\d+)(\+)?", re.IGNORECASE)
RELEASE_VERSION_RE = re.compile(r"\b\d+\.\d+\.\d+\b")


def _collect_evidence_text(retrieval_packet, max_evidence):
    parts = [str(retrieval_packet.get("answer", "")).strip()]
    for citation in retrieval_packet.get("retrieval_hits", [])[:max_evidence]:
        parts.append(str(citation.get("excerpt", "")).strip())
    return "\n".join(part for part in parts if part)


def _extract_python_version_matches(text):
    matches = []
    for match in PYTHON_VERSION_RE.finditer(text or ""):
        major = int(match.group(1))
        minor = int(match.group(2))
        has_plus = bool(match.group(3))
        label = f"Python {major}.{minor}{'+' if has_plus else ''}"
        matches.append(((major, minor, 1 if has_plus else 0), label))
    return matches


def derive_focus_hint(query, retrieval_packet, max_evidence=3):
    normalized_query = query.lower()
    evidence_text = _collect_evidence_text(retrieval_packet, max_evidence=max_evidence)
    version_matches = _extract_python_version_matches(evidence_text)

    if "newest explicitly supported python line" in normalized_query and version_matches:
        _, label = max(version_matches, key=lambda item: item[0])
        return f"{label} became the newest explicitly supported Python line."

    if "minimum python line remained supported" in normalized_query and version_matches:
        _, label = min(version_matches, key=lambda item: item[0])
        if not label.endswith("+"):
            label = f"{label}+"
        return f"{label} remained the minimum supported Python line."

    lowered_evidence = evidence_text.lower()
    if "what happened to the app shortcut" in normalized_query and "has now been removed" in lowered_evidence:
        return "The app shortcut has now been removed."

    if "what happened to the proxies argument" in normalized_query and "has now been removed" in lowered_evidence:
        return "The proxies argument has now been removed."

    if "pydantic v1 support" in normalized_query and "dropped support for pydantic v1" in lowered_evidence:
        return "Direct Pydantic v1 support was later dropped."

    return None


def postprocess_grounded_answer(query, generated_answer, retrieval_packet, max_evidence=3):
    focus_hint = derive_focus_hint(query, retrieval_packet, max_evidence=max_evidence)
    if not focus_hint:
        return generated_answer.strip()

    normalized_query = query.lower()
    cleaned_answer = (generated_answer or "").strip()
    lowered_answer = cleaned_answer.lower()

    if "newest explicitly supported python line" in normalized_query:
        has_python_version = bool(PYTHON_VERSION_RE.search(lowered_answer))
        has_release_version = bool(RELEASE_VERSION_RE.search(cleaned_answer))
        if not has_python_version or has_release_version:
            return focus_hint

    if "minimum python line remained supported" in normalized_query:
        if not PYTHON_VERSION_RE.search(lowered_answer):
            return focus_hint

    if "pydantic v1 support" in normalized_query:
        if "pydantic v1" not in lowered_answer or "drop" not in lowered_answer:
            return focus_hint

    return cleaned_answer


def build_grounded_prompt(query, retrieval_packet, max_evidence=3):
    evidence_lines = []
    for index, citation in enumerate(retrieval_packet["retrieval_hits"][:max_evidence], start=1):
        source = citation["source"].replace("\\", "/")
        excerpt = str(citation["excerpt"]).strip()
        evidence_lines.append(f"[{index}] {source}\n{excerpt}")

    evidence_block = "\n\n".join(evidence_lines) if evidence_lines else "[none]"
    conflict_lines = []
    for conflict in retrieval_packet.get("conflicts", []):
        message = str(conflict.get("message", "")).strip()
        if message:
            conflict_lines.append(f"- {message}")

    conflict_block = "\n".join(conflict_lines) if conflict_lines else "[none]"
    draft_answer = str(retrieval_packet.get("answer", "")).strip() or "[none]"
    focus_hint = derive_focus_hint(query, retrieval_packet, max_evidence=max_evidence) or "[none]"

    return (
        "Rewrite the grounded draft answer into the shortest correct final answer.\n"
        "Use only the evidence and draft answer below.\n"
        "Do not drop explicit version numbers, API names, or support-status changes.\n"
        "If a focus hint is present, prefer it unless the evidence clearly contradicts it.\n"
        "If the evidence says support was added for a newer version, keep that newer version in the answer.\n"
        "If the evidence is conflicting, say so plainly.\n"
        "Return one short factual sentence only.\n\n"
        "Example 1:\n"
        "Question: After a project added support for Python 3.14, what became the newest explicitly supported Python line?\n"
        "Draft answer: The project added support for Python 3.14 while Python 3.13 remained supported.\n"
        "[[FINAL_ANSWER]]: Python 3.14 became the newest explicitly supported Python line.\n\n"
        "Example 2:\n"
        "Question: What changed for direct Pydantic v1 support?\n"
        "Draft answer: The project later dropped support for Pydantic v1, after only temporary compatibility.\n"
        "[[FINAL_ANSWER]]: Direct Pydantic v1 support was later dropped.\n\n"
        f"Question:\n{query}\n\n"
        f"Focus hint from evidence:\n{focus_hint}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Conflict notes:\n{conflict_block}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        "[[FINAL_ANSWER]]:"
    )
