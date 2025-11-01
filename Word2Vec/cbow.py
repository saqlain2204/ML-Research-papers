import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class CBOW(nn.Module):
    def __init__(self, embedding_size=100, vocab_size=-1):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)
    
    def forward(self, inputs):
        # inputs - batch_size x (context_size x 2)
        embeddings = self.embeddings(inputs).mean(1).squeeze(1) # batch_size x embedding_size
        return self.linear(embeddings)

sentences = [
    "we are building a cbow model",
    "this is a tiny dataset",
    "another simple sentence"
]

def build_vocab(sentences):
    words = set()
    for s in sentences:
        words.update(s.lower().split())
    idx2tok = ['<pad>', '<unk>'] + sorted(words)
    tok2idx = {t: i for i, t in enumerate(idx2tok)}
    return tok2idx, idx2tok

def make_pairs(sentences, tok2idx, window=2):
    pairs = []
    for s in sentences:
        tokens = s.lower().split()
        
        idxs = [tok2idx[t] for t in tokens]
        for i in range(len(idxs)):
            start = max(0, i - window)
            end = min(len(idxs), i + window + 1)
            context = []
            for j in range(start, end):
                if j != i:
                    context.append(idxs[j])
            pairs.append((context, idxs[i]))
    
    return pairs

tok2idx, idx2tok = build_vocab(sentences=sentences)

def collate(batch):
    # batch: list of (context_list, target_idx)
    contexts, targets = zip(*batch)
    maxlen = max(len(c) for c in contexts)
    ctx = torch.zeros((len(contexts), maxlen), dtype=torch.long)  # pad idx = 0
    for i, c in enumerate(contexts):
        ctx[i, :len(c)] = torch.tensor(c, dtype=torch.long)
    return ctx, torch.tensor(targets, dtype=torch.long)

def train_demo(epochs=100):
    tok2idx, idx2tok = build_vocab(sentences)
    pairs = make_pairs(sentences, tok2idx, window=4)
    loader = DataLoader(pairs, batch_size=4, shuffle=True, collate_fn=collate)

    model = CBOW(embedding_size=100, vocab_size=len(idx2tok))
    opt = optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total = 0.0
        for contexts, targets in loader:
            logits = model(contexts)
            loss = loss_fn(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch} loss {total/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        ctxt = ["building", "a"]
        inp = torch.tensor([[tok2idx[t] for t in ctxt]], dtype=torch.long)
        pred = model(inp).argmax(dim=1).item()
        print("predicted token:", idx2tok[pred])

if __name__ == "__main__":
    train_demo()



    
    
        
        
        