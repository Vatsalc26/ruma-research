import torch
import torch.nn as nn

class ContextSynthesizer(nn.Module):
    """
    Component 1: Hybrid Attention Layer (Layer 1 Only)
    Responsible for establishing perfect initial context matching (solving the "bank" problem).
    """
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        # Standard Multi-head Attention
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def _build_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, x, causal=False):
        # x shape: [batch_size, seq_len, d_model]
        norm_x = self.norm(x)

        attn_mask = None
        if causal:
            attn_mask = self._build_causal_mask(x.size(1), x.device)

        # Self-attention reads the entire sentence to lock in definitions
        attn_out, _ = self.attention(norm_x, norm_x, norm_x, attn_mask=attn_mask)

        # Residual connection
        return x + attn_out
