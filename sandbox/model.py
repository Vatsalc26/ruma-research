import torch
import torch.nn as nn
from intake import ContextSynthesizer
from router import LSHRouter
from ponds import NeuralPonds

class PipelineArchitecture(nn.Module):
    """
    The Full Pipeline & Ponds Architecture
    """
    def __init__(self, vocab_size=5000, d_model=256, n_heads=8, num_ponds=10):
        super().__init__()
        
        # 1. Input Mapping
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Layer 1: Context Synthesizer (Jamba style Hybrid Attention)
        self.intake = ContextSynthesizer(d_model=d_model, n_heads=n_heads)
        
        # 3. Layer 2: Master Routing (LSH Barcode Scanner)
        self.router = LSHRouter(d_model=d_model, num_ponds=num_ponds)
        
        # 4. Layer 3: Knowledge Retrieval (The 10 Ponds)
        self.ponds = NeuralPonds(num_ponds=num_ponds, d_model=d_model)
        
        # 5. Layer 4: The Valve (Output Decoder / Simulation Engine)
        self.valve_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids):
        """
        input_ids: [batch_size, seq_len]
        """
        # Step 1: Embed words into vectors
        x = self.embedding(input_ids)
        
        # Step 2: Intake blends the context perfectly
        context_locked = self.intake(x)
        
        # Step 3: Router hashes the context to the correct Ponds
        routes = self.router(context_locked)
        
        # Step 4: The assigned Ponds release their precalculated flow
        water_stream = self.ponds(context_locked, routes)
        
        # Step 5: The Valve catches the math and translates it to specific words
        # (In a real scenario, the test-time reasoning loop would happen here)
        final_output = self.valve_out(water_stream)
        
        return final_output
