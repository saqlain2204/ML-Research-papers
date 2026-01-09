import torch
import torch.nn as nn
from .scaled_dot_product_attention import scaled_dot_product_attention as sdpa

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super().__init__()
        assert d_model % num_head == 0, "d_model must be divisble by num_head"
        self.num_head = num_head
        self.d_k = d_model // num_head
    
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def split_heads(self, x):
        b, t, _ = x.size()
        x = x.view(b, t, self.num_head, self.d_k)
        return x.transpose(1, 2)
    

        
        
        
    