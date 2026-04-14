import torch
import torch.nn as nn


class GatedMemoryFusion(nn.Module):
    """
    A small single-hop gated residual fusion block for combining live hidden
    state with retrieved memory state.
    This is the baseline fusion policy for the sandbox, not the final word on
    multi-hop or learned reranking behavior.
    """

    def __init__(self, d_model):
        super().__init__()
        self.candidate = nn.Linear(d_model * 2, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, context_state, memory_state):
        joined = torch.cat([context_state, memory_state], dim=-1)
        candidate = torch.tanh(self.candidate(joined))
        gate = torch.sigmoid(self.gate(joined))
        fused = context_state + gate * candidate
        return self.norm(fused)
