import json
from pathlib import Path

import torch

from ruma_v2_model import InterleavedRUMAModel


def run_v2_interleaved_ruma_demo():
    torch.manual_seed(0)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    model = InterleavedRUMAModel(
        vocab_size=128,
        d_model=64,
        n_heads=4,
        num_shards=6,
        shard_capacity=64,
        num_backbone_blocks=8,
        num_ruma_blocks=4,
        top_k=3,
    )
    model.eval()

    support_inputs = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ],
        dtype=torch.long,
    )
    support_targets = torch.tensor(
        [
            [14],
            [24],
            [34],
        ],
        dtype=torch.long,
    )
    model.update_memory(
        support_inputs,
        target_ids=support_targets,
        sources=["docs.fastapi", "docs.httpx", "docs.pydantic"],
        namespaces=["docs.python.fastapi", "docs.python.httpx", "docs.python.pydantic"],
        timestamps=["t1", "t1", "t1"],
    )

    query_inputs = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ],
        dtype=torch.long,
    )
    with torch.no_grad():
        logits, aux = model(query_inputs, top_k=3, return_aux=True, use_memory=True, causal=True)

    predicted_next_tokens = torch.argmax(logits[:, -1, :], dim=-1).detach().cpu().tolist()
    results = {
        "layer_schedule": aux["layer_schedule"],
        "predicted_next_tokens": predicted_next_tokens,
        "expected_next_tokens": support_targets.squeeze(-1).detach().cpu().tolist(),
        "avg_ruma_sufficiency": aux["avg_ruma_sufficiency"],
        "avg_ruma_conflict": aux["avg_ruma_conflict"],
        "ruma_blocks": aux["ruma_blocks"],
        "memory_stats": aux["memory_stats"],
    }

    output_path = results_dir / "v2_interleaved_ruma_demo.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\n[NOTE] Results written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_v2_interleaved_ruma_demo())
