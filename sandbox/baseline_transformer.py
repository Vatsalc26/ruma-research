import torch
import torch.nn as nn

class DenseTransformer(nn.Module):
    """
    Standard 'Dense' Transformer Baseline.
    Forces all tokens through a heavy Feed-Forward Network instead of routing to Ponds.
    This mimics the core structure of open-source models like Llama-3, scaled down for CPU testing.
    """
    def __init__(self, vocab_size, d_model=64, n_heads=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        
        # The traditional dense bottleneck block 
        # Every single token must pass through this massive matrix, causing N^2 lag
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def _build_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, x, causal=False):
        emb = self.embedding(x)

        attn_mask = None
        if causal:
            attn_mask = self._build_causal_mask(x.size(1), x.device)

        # Standard Attention
        attn_out, _ = self.attention(emb, emb, emb, attn_mask=attn_mask)
        x = self.norm1(emb + attn_out)

        # Heavy math block
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return self.out(x)
