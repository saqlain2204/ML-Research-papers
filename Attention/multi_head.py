import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product(q, k, v, mask=None):
    d_k = k.size(-1)
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask==0, -float('inf'))
    
    attn_logits = F.softmax(attn_logits, dim=-1)
    result = torch.matmul(attn_logits, v)
    return result

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        assert embed_dim % n_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.h = n_heads
        self.d = embed_dim // n_heads
        
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, emebed_dim = q.size()
        
        q = self.wq(q).view(batch_size, seq_len, self.h, self.d).transpose(1, 2)
        k = self.wk(k).view(batch_size, seq_len, self.h, self.d).transpose(1, 2)
        v = self.wv(v).view(batch_size, seq_len, self.h, self.d).transpose(1, 2)
        
        y = scaled_dot_product(q, k, v)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, emebed_dim)
        return self.out(y)
    

if __name__ == "__main__":
    embed_dim = 8
    n_heads = 2
    seq_len = 4
    batch_size = 1
    
    q = torch.randn(batch_size, seq_len, embed_dim)
    k = torch.randn(batch_size, seq_len, embed_dim)
    v = torch.randn(batch_size, seq_len, embed_dim)
    
    mha = MultiHeadAttention(embed_dim, n_heads)
    out = mha(q, k, v, mask=torch.ones(batch_size, seq_len, seq_len))
    print(out)
    assert out.size() == torch.Size([batch_size, seq_len, embed_dim])