import torch
import torch.nn as nn

class NeuralPonds(nn.Module):
    """
    Component 3: The 10 Ponds (Vector Latent Spaces)
    Precalculated 'water/logic' streams representing specific knowledge domains (Physics, Code, Nature, etc).
    """
    def __init__(self, num_ponds=10, d_model=256, pond_capacity=10000):
        super().__init__()
        self.d_model = d_model
        # ModuleList acts as our 10 distinct, disconnected reservoirs.
        # Each Embedding represents a Continuous Latent Space where logic mixes.
        self.ponds = nn.ModuleList([
            nn.Embedding(num_embeddings=pond_capacity, embedding_dim=d_model)
            for _ in range(num_ponds)
        ])
        
    def forward(self, context_vector, pond_assignments):
        """
        context_vector: [batch, seq_len, d_model]
        pond_assignments: [batch, seq_len] array of barcode indices mapped by the Router
        """
        batch_size, seq_len, d_model = context_vector.shape
        # This will hold the "water" fetched from the Ponds
        water_output = torch.zeros_like(context_vector)
        
        # Loop through the sequence to grab precalculated fluid
        # (In reality this is vectorized for speed, but written cleanly for Sandbox clarity)
        for b in range(batch_size):
            for s in range(seq_len):
                # 1. Look at the barcode to see which Pond to open
                pond_idx = pond_assignments[b, s].item()
                
                # 2. Extract a 'coordinate' flavor from the context vector
                lookup_flavor_idx = int(torch.abs(torch.sum(context_vector[b, s])).item()) % 10000
                
                # 3. Open the Tap: Pull the precalculated logic vector out of the chosen Pond
                # This perfectly mimics Retrieval (RAG/kNN) without heavy matrix math
                water_drop = self.ponds[pond_idx](
                    torch.tensor(lookup_flavor_idx, device=context_vector.device)
                )
                
                water_output[b, s] = water_drop
                
        return water_output
