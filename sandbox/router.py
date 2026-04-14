import torch
import torch.nn as nn

class LSHRouter(nn.Module):
    """
    Fixed-hash sparse router placeholder for the sandbox.
    This is a toy routing baseline, not the intended final RUMA router.
    """
    def __init__(self, d_model, num_ponds=10):
        super().__init__()
        self.num_ponds = num_ponds
        # Fixed random hyperplanes simulate a cheap hash-style routing baseline.
        self.register_buffer("hyperplanes", torch.randn(d_model, num_ponds))
        
    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        projections = torch.matmul(x, self.hyperplanes)

        # Return the highest-scoring hash bucket as the routed shard id.
        assigned_ponds = torch.argmax(projections, dim=-1)

        return assigned_ponds
