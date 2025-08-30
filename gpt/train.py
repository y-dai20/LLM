# MIT License
# Copyright (c) 2022 Andrej Karpathy

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

def train():
    with open(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/assets/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    print("length of dataset in characters: ", len(text))

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print("chars: " + ''.join(chars))
    print("vocab size: ", vocab_size)

    # create a mapping from characters to integers
    stoi = {ch : i for i, ch in enumerate(chars)}
    itos = {i : ch for i, ch in enumerate(chars)}
    encode = lambda s : [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l : ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    data = torch.tensor(encode(text), dtype=torch.long)
    print("data.shape, data.dtype: ", data.shape, data.dtype)

    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    batch_size = 4  # how many independent sequences will we process in parallel?
    block_size = 8  # what is the maximum context length for predictions?

    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y

    m = BigramLanguageModel(vocab_size)

    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    max_steps = 100
    batch_size = 32
    for steps in range(max_steps):  # increase number of steps for good results...
        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if steps % 10 == 0:
            print(f"Step: {steps}, Loss: {loss.item()}")

    print(loss.item())

    print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    train()
