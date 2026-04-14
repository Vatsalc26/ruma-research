import torch
import torch.nn as nn
import torch.optim as optim
import time
from dataset import DummyDataset
from model import PipelineArchitecture

def train_architecture():
    print("--------------------------------------------------")
    print("Initiating Pipeline & Ponds Training Sandbox v0.2 ")
    print("--------------------------------------------------\n")
    
    # 1. Setup Architecture params
    VOCAB_SIZE = 500
    SEQ_LEN = 16
    BATCH_SIZE = 8
    D_MODEL = 64
    EPOCHS = 500
    
    # Instantiate the model
    print("Assembling mathematical components...")
    # We use a small vocabulary and dimension so the CPU can train it blazingly fast
    model = PipelineArchitecture(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=2, num_ponds=4)
    model.train() # Set to training mode
    
    # 2. Setup the Trainer (Optimizer & Loss Function)
    criterion = nn.CrossEntropyLoss() # Grades the AI's guesses
    optimizer = optim.AdamW(model.parameters(), lr=0.01) # The Calculus engine that twists the dials
    
    # 3. Setup Dataset
    dataset = DummyDataset(vocab_size=VOCAB_SIZE, max_seq_len=SEQ_LEN)
    
    print("\n--- INITIATING BACKPROPAGATION ---")
    start_time = time.time()
    final_loss = None
    
    for epoch in range(EPOCHS + 1):
        # Grab a batch of predictable pattern data
        x, y_target = dataset.generate_batch(batch_size=BATCH_SIZE)
        
        # Reset the calculus gradients to zero
        optimizer.zero_grad()
        
        # Forward Pass: Ask the AI to guess the next word
        predictions = model(x) # Shape: [batch, seq_len, vocab_size]
        
        # Flatten the arrays to calculate the score properly
        predictions_flat = predictions.view(-1, VOCAB_SIZE)
        y_target_flat = y_target.contiguous().view(-1)
        
        # Calculate how stupid the guess was (The Loss)
        loss = criterion(predictions_flat, y_target_flat)
        final_loss = loss.item()
        
        # Backward Pass: Run the calculus backwards through the pipelines
        loss.backward()
        
        # Twist the dials (Update parameters)
        optimizer.step()
        
        # Print progress every 50 epochs
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss (Error Score): {loss.item():.4f}")
            
    end_time = time.time()
    print("====================================================")
    print(f"\n[SUCCESS] Training loop completed in {(end_time - start_time):.2f} seconds!")
    print(f"Final loss: {final_loss:.4f}")
    if final_loss is not None and final_loss < 1.0:
        print("The toy sandbox learned the synthetic pattern well.")
    else:
        print("The toy sandbox trained end to end, but this run did not converge to a strong low-loss solution.")
    print("====================================================\n")

if __name__ == "__main__":
    train_architecture()
