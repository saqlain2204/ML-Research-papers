import math
import torch
import torch.nn.functional as F

def scaled_dot_product_attn(q, k, v, mask=None):
    # Let q, k, v.size() be [3, 2]
    d_k = k.size(-1) # Scaling value
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) # [3, 2] x [3, 2] -> [3, 2] x [2, 3] = [3, 3]
    attn_logits = attn_logits / math.sqrt(d_k) # Scaling step
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask==0, -float('inf')) # Take the mask make the zeros to True and replace them with -9e-15
    
    attn_logits = F.softmax(attn_logits, dim=-1)
    result = torch.matmul(attn_logits, v) # [3, 3] x [3, 2] = [3, 2]
    return result


if __name__ == "__main__":
    q = torch.randn(3, 2)
    k = torch.randn(3, 2)
    v = torch.randn(3, 2)
    
    attn = scaled_dot_product_attn(q, k, v, mask = torch.ones(3, 3))
    print(attn)
    assert attn.size() == torch.Size([3, 2])

    
        
            
    
    
    