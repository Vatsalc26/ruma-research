import torch

class CharDataset:
    """
    A tokenizer capable of mapping individual English characters instead of words.
    This allows us to process absolute massive files seamlessly.
    """
    def __init__(self, file_path, seq_len):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        
        # Build the character mapping
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.data = data
        self.seq_len = seq_len
        
    def encode(self, s):
        return [self.stoi[c] for c in s]
        
    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])
        
    def get_batch(self, batch_size):
        # Generates a random massive batch of real text coordinates
        ix = torch.randint(len(self.data) - self.seq_len, (batch_size,))
        x = torch.stack([torch.tensor(self.encode(self.data[i:i+self.seq_len])) for i in ix])
        y = torch.stack([torch.tensor(self.encode(self.data[i+1:i+1+self.seq_len])) for i in ix])
        return x, y
