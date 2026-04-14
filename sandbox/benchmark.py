import torch
import time
from baseline_transformer import DenseTransformer
from model_eval import EvalArchitecture
from dataset_medium import CharDataset
import os

def run_benchmark():
    if not os.path.exists('alice.txt'):
         print("[ERROR] Medium-capacity Text data not found. Please run fetch_data.py!")
         return

    print("\n=======================================================")
    print("      THE GREAT EFFICIENCY RACE: DENSE vs PIPELINES ")
    print("=======================================================\n")
    
    # 1. Load Data
    dataset = CharDataset('alice.txt', seq_len=128)
    VOCAB_SIZE = dataset.vocab_size
    D_MODEL = 64
    BATCH_SIZE = 64 # Massive batch to force the CPU to sweat
    
    x, _ = dataset.get_batch(batch_size=BATCH_SIZE)
    print(f"[STATUS] Dataset Loaded: 64 Batches of 128 Tokens ({BATCH_SIZE * 128} total calculations per pass)\n")
    
    # 2. Spool up Baseline
    print("[STATUS] Spooling up Baseline (Heavy Dense Feed-Forward network)...")
    baseline = DenseTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
    
    # 3. Spool up Pipeline
    print("[STATUS] Spooling up Custom Pipeline (Sparse LSH Router & 10 Vector Ponds)...")
    pipeline = EvalArchitecture(vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_ponds=10)
    
    print("\n-------------------------------------------------------")
    # 4. The Race
    print("[RACE STARTING] Pushing tokens through Baseline Dense Model...")
    start_base = time.time()
    _ = baseline(x)
    time_base = time.time() - start_base
    print(f"                -> Baseline Time completed in {time_base:.5f} seconds\n")
    
    print("[RACE STARTING] Pushing tokens through Custom Pipeline & Ponds...")
    start_pipe = time.time()
    _ = pipeline(x) # return_metrics is false during a pure speed test
    time_pipe = time.time() - start_pipe
    print(f"                -> Pipeline Time completed in {time_pipe:.5f} seconds\n")
    
    # 5. Results
    print("================ RACE RESULTS ================")
    if time_pipe < time_base:
        diff = ((time_base - time_pipe) / time_base) * 100
        print(f"[WINNER] YOUR CUSTOM PIPELINE ARCHITECTURE")
        print(f"         It ran {diff:.2f}% faster than the Standard Dense base model!")
        print("\n[NOTE] In this local sandbox run, the routed prototype was faster than")
        print("the simple dense baseline. This is a narrow implementation result,")
        print("not a general proof about model families.")
    else:
        print("[WINNER] BASELINE")
        print("In this local sandbox run, the current routed implementation was slower")
        print("than the simple dense baseline.")
         
    print("==============================================\n")

if __name__ == "__main__":
    run_benchmark()
