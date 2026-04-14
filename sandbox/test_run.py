import torch
import time
from dataset import DummyDataset
from model import PipelineArchitecture

def validate_architecture():
    print("--------------------------------------------------")
    print("Initializing Pipeline & Ponds Architecture Sandbox")
    print("--------------------------------------------------\n")
    
    # 1. Setup Architecture params
    VOCAB_SIZE = 5000
    SEQ_LEN = 64
    BATCH_SIZE = 4
    D_MODEL = 256
    
    # Instantiate the model
    print("[1/3] Assembling mathematical components...")
    model = PipelineArchitecture(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
    print("      Model perfectly assembled!\n")
    
    # 2. Get Dummy Data
    print(f"[2/3] Generating random synthetic data...")
    print(f"      Batch Size: {BATCH_SIZE} | Sequence Length: {SEQ_LEN}\n")
    dataset = DummyDataset(vocab_size=VOCAB_SIZE, max_seq_len=SEQ_LEN)
    test_batch, _ = dataset.generate_batch(batch_size=BATCH_SIZE)
    
    print("[3/3] INITIATING FORWARD PASS...")
    start_time = time.time()
    
    # 3. Push data through the pipeline
    try:
        logits = model(test_batch)
        end_time = time.time()
        
        # Validate Shape
        print("\n================ VALIDATION RESULTS ================")
        print(f"Expected Output Matrix: [{BATCH_SIZE}, {SEQ_LEN}, {VOCAB_SIZE}]")
        print(f"Actual Output Matrix:   {list(logits.shape)}")
        
        if list(logits.shape) == [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]:
            print("\n[SUCCESS] The mathematical pieces fit flawlessly together!")
            print("[SUCCESS] Zero memory crashes detected.")
            print(f"Execution Time: {(end_time - start_time):.5f} seconds")
        else:
            print("\n[FAIL] The matrix shapes collapsed during the pipeline.")
            
    except Exception as e:
        print(f"\n[CRITICAL FAIL] The architecture crashed. Mathematical Error:")
        print(e)
    print("====================================================\n")

if __name__ == "__main__":
    validate_architecture()
