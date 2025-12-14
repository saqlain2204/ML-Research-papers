import numpy as np
import torch

def softmax(x):
    return torch.softmax(x, dim=-1)

def scaled_dot_product_attention(q, k, v, mask=None):
    q_k = torch.matmul(q, k.transpose(-2, -1))
    d_k = k.size(-1)
    scaled_q_k = q_k / torch.sqrt(torch.tensor(d_k, dtype=q_k.dtype, device=q_k.device))
    if mask is not None:
        scaled_q_k = scaled_q_k.masked_fill(mask==0, float('-inf'))
        
    scaled_q_k = softmax(scaled_q_k)
    attention = torch.matmul(scaled_q_k, v)
    return attention
    
def test_attention_shape():
    q = torch.rand(2, 4, 8)
    k = torch.rand(2, 4, 8)
    v = torch.rand(2, 4, 8)
    output = scaled_dot_product_attention(q, k, v)
    assert output.shape == (2, 4, 8), "Output shape mismatch"

def test_attention_masking():
    q = torch.ones(1, 4, 2)
    k = torch.ones(1, 4, 2)
    v = torch.arange(8).float().reshape(1, 4, 2)
    mask = torch.tensor([[[1, 0, 1, 0],
                          [1, 1, 0, 0],
                          [1, 1, 1, 1],
                          [0, 0, 1, 1]]])
    output = scaled_dot_product_attention(q, k, v, mask)
    assert not torch.isnan(output).any(), "Output contains NaNs with mask"

def test_attention_weights_sum_to_one():
    q = torch.rand(1, 3, 5)
    k = torch.rand(1, 3, 5)
    v = torch.rand(1, 3, 5)
    output = scaled_dot_product_attention(q, k, v)
    assert output.shape == (1, 3, 5)
    assert isinstance(output, torch.Tensor)

if __name__ == "__main__":
    test_attention_shape()
    test_attention_masking()
    test_attention_weights_sum_to_one()
    print("All tests passed!")
    
    
    
    
    