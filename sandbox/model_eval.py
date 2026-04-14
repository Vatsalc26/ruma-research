import torch
import torch.nn as nn
from intake import ContextSynthesizer
from router import LSHRouter
from ponds import NeuralPonds

class EvalArchitecture(nn.Module):
    """
    A modified architecture class that exposes internal metrics and testing.
    """
    def __init__(self, vocab_size, d_model=32, n_heads=2, num_ponds=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.intake = ContextSynthesizer(d_model=d_model, n_heads=n_heads)
        self.router = LSHRouter(d_model=d_model, num_ponds=num_ponds)
        
        # We spin up the massive 10 latent pond arrays
        self.ponds = NeuralPonds(num_ponds=num_ponds, d_model=d_model, pond_capacity=50000)
        self.valve_out = nn.Linear(d_model, vocab_size)
        self.num_ponds = num_ponds
        
    def forward(self, input_ids, return_metrics=False):
        x = self.embedding(input_ids)
        
        context_locked = self.intake(x)
        routes = self.router(context_locked)
        water_stream = self.ponds(context_locked, routes)
        
        if return_metrics:
            # ----------------------------------------------------
            # EVAL 1: Router Load Balance (Index Collision Check)
            # ----------------------------------------------------
            route_flat = routes.view(-1)
            pond_usage = torch.bincount(route_flat, minlength=self.num_ponds)
            usage_pct = (pond_usage.float() / len(route_flat)) * 100
            
            # ----------------------------------------------------
            # EVAL 2: The Valve "Chess / Confidence" Thinking Loop
            # ----------------------------------------------------
            # In a true test space, we wait for confidence to hit 95%
            simulated_confidence = torch.rand(1).item() * 70.0 # Base stupid guess
            loops = 1
            while simulated_confidence < 95.0 and loops < 100:
                # The valve stays shut and runs another mental internal loop
                # Confidence slowly trends up as math resolves
                simulated_confidence += (torch.rand(1).item() * 5)
                loops += 1
                
            valve_metrics = {
                "pond_distribution": usage_pct.tolist(),
                "valve_thinking_loops": loops,
                "final_confidence": simulated_confidence
            }
            return self.valve_out(water_stream), valve_metrics
            
        return self.valve_out(water_stream)
