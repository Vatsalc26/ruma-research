import torch
import torch.nn as nn
import torch.optim as optim
from dataset import EnglishDummyDataset
from model import PipelineArchitecture

def run_leak_test():
    print("\n=======================================================")
    print(" INITIATING CONTINUOUS LEAK & ENGLISH GENERATION TEST ")
    print("=======================================================\n")
    
    dataset = EnglishDummyDataset()
    # Tiny model so it trains in half a second
    model = PipelineArchitecture(vocab_size=dataset.vocab_size, d_model=32, n_heads=2, num_ponds=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.05)
    
    # ---------------------------------------------------------
    # PHASE 1: Base Pretraining
    # ---------------------------------------------------------
    print("[Phase 1] Top-Down Training on base memory: 'The brown cat jumped .'")
    x_A, y_A = dataset.get_data_A()
    for _ in range(80): # Full training run
        optimizer.zero_grad()
        loss = criterion(model(x_A).view(-1, dataset.vocab_size), y_A.view(-1))
        loss.backward()
        optimizer.step()
    
    # Check if it remembers Data A
    pred_A = torch.argmax(model(x_A), dim=-1)[0]
    print(f"      -> Memory Output A: 'The {dataset.decode(pred_A)}'\n")
    
    
    # ---------------------------------------------------------
    # PHASE 2: The Leak
    # ---------------------------------------------------------
    print("[Phase 2] Simulating the Live 'Wikipedia Leak'...")
    print("          Running a tiny follow-up update on 'The purple spaceship flew .'.")
    x_B, y_B = dataset.get_data_B()
     
    # Note: This is still full-parameter updating on a toy example.
    for _ in range(15): 
        optimizer.zero_grad()
        loss = criterion(model(x_B).view(-1, dataset.vocab_size), y_B.view(-1))
        loss.backward()
        optimizer.step()
        
    # Check if The Leak took hold
    pred_B = torch.argmax(model(x_B), dim=-1)[0]
    print(f"      -> Memory Output B (The Leak): 'The {dataset.decode(pred_B)}'\n")
    
    
    # ---------------------------------------------------------
    # PHASE 3: The Ultimate Forgetting Test
    # ---------------------------------------------------------
    print("[Phase 3] Checking if the leak destroyed the old memory (Catastrophic Forgetting)...")
    pred_A_after = torch.argmax(model(x_A), dim=-1)[0]
    final_string_output = dataset.decode(pred_A_after)
    
    print(f"      -> Memory Output A (Rescanned): 'The {final_string_output}'\n")
    
    print("================ TEST RESULTS ================")
    if final_string_output == "brown cat jumped .":
        print("[SUCCESS] The toy example retained the original phrase after the follow-up update.")
        print("[NOTE] This is weak evidence only; it does not prove general continual learning.")
    else:
        print("[FAIL] The toy example lost the original phrase after the follow-up update.")
        print("[NOTE] This indicates interference in the current sandbox setup.")

if __name__ == "__main__":
    run_leak_test()
