import torch
from dataset_medium import CharDataset
from model_eval import EvalArchitecture
import os
import time

def run_pressure_test():
    if not os.path.exists('alice.txt'):
        print("[ERROR] Medium-capacity Text data not found. Please run fetch_data.py!")
        return
        
    print("\n=======================================================")
    print(" INITIATING MEDIUM-CAPACITY PRESSURE & LOAD-BALANCE TEST")
    print("=======================================================\n")
    
    # Load massive string text
    dataset = CharDataset('alice.txt', seq_len=64)
    print(f"[STATUS] Real Dataset Loaded. Unique Character Hash size: {dataset.vocab_size}")
    
    # Spool up the architecture
    print("[STATUS] Spooling up 10 Independent Vector Ponds...")
    model = EvalArchitecture(vocab_size=dataset.vocab_size, d_model=64, num_ponds=10)
    
    print("\n--- INITIATING FORWARD PASS LOAD TEST ---")
    x, y = dataset.get_batch(batch_size=32)
    
    start = time.time()
    # Force 32 sequences of 64 tokens (2,048 calculations) through the Router simultaneously
    logits, metrics = model(x, return_metrics=True)
    end = time.time()
    
    print(f"Time to process 2,048 tokens across 10 Ponds: {(end-start):.4f} seconds!")
    
    # -----------------------------------------------------------
    # EVAL 1: The Load Balance Proof
    # -----------------------------------------------------------
    print("\n--- METRIC 1: ROUTER LOAD-BALANCING EVAL ---")
    dist = metrics["pond_distribution"]
    print("Percentage of semantic input physically routed to each pond:")
    for i, pct in enumerate(dist):
        print(f"  Pond {i:02d}: {pct:.2f}%")
        
    zeros = sum(1 for p in dist if p == 0.0)
    if zeros == 0:
        print("\n[SUCCESS] In this toy load test, all 10 shards received some traffic.")
        print("[NOTE] This is only a shard-usage signal, not proof of semantic routing quality.")
    else:
        print(f"\n[FAIL] Index Collapse! {zeros} Ponds were entirely ignored. The Router is broken.")
        
    # -----------------------------------------------------------
    # EVAL 2: The Valve Proof
    # -----------------------------------------------------------
    print("\n--- METRIC 2: THE VALVE 'REASONING' TEST LOOP ---")
    print("Simulating the 'Think before you speak' threshold...")
    print(f"  Physics loops forced before release: {metrics['valve_thinking_loops']}")
    print(f"  Final Confidence Score achieved:     {metrics['final_confidence']:.2f}%\n")
    
    if metrics['final_confidence'] >= 95.0:
        print("[NOTE] The confidence loop is currently simulated, so this output is")
        print("       only a placeholder for a future real reasoning-time mechanism.")

if __name__ == "__main__":
    run_pressure_test()
