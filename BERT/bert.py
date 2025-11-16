import torch
import torch.nn as nn

class BERTEmbedding(nn.Module):
    def __init__(self, 
                 vocab_size,
                 n_segments,
                 max_len,
                 embed_dim,
                 dropout):
        super().__init__()
        # Token embeddings: num_embeddings = vocab_size
        # Segment embeddings: num_embeddings = n_segments
        # Postition embeddings: num_embeddings = max_len
        self.tok_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.seg_embed = nn.Embedding(num_embeddings=n_segments, embedding_dim=embed_dim)
        self.pos_embed = nn.Embedding(num_embeddings=max_len, embedding_dim=embed_dim)
        
        self.drop = nn.Dropout(dropout)
        self.pos_inputs = torch.tensor([x for x in range(max_len)],)
    
    def forward(self, seq, seg):
        L = seg.size(0)
        p = self.pos_inputs[:L].to(seg.device)
        embed_val = self.tok_embed(seq) + self.seg_embed(seg) + self.pos_embed(p)
        return self.drop(embed_val)

class BERT(nn.Module):
    def __init__(self,
                 vocab_size, 
                 n_segments,
                 max_len,
                 embed_dim,
                 n_layers,
                 attn_heads,
                 dropout):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size=vocab_size, n_segments=n_segments, max_len=max_len, embed_dim=embed_dim, dropout=dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(embed_dim, attn_heads, embed_dim*4, batch_first=True)
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, n_layers)
        
    def forward(self, seq, seg):
        out = self.embedding(seq, seg)
        out = out.unsqueeze(0)
        out = self.encoder_block(out)
        return out.squeeze(0)
    

if __name__ == "__main__":
        VOCAB_SIZE = 30000
        N_SEGMENTS = 3
        MAX_LEN = 512
        EMBED_DIM = 768
        N_LAYERS = 12
        ATTN_HEADS = 12
        DROPOUT = 0.1
        
        sample_seq = torch.randint(high = VOCAB_SIZE, size=[MAX_LEN, ])
        sample_seg = torch.randint(high = N_SEGMENTS, size= [MAX_LEN, ])
        
        embedding = BERTEmbedding(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, DROPOUT)
        embedding_tensor = embedding(sample_seq, sample_seg)
        
        # print(embedding_tensor.size()) 
        
        bert = BERT(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, N_LAYERS, ATTN_HEADS, DROPOUT)
        out = bert(sample_seq, sample_seg)
        
        vocab = {"[PAD]":0, "[CLS]":1, "[SEP]":2}
        idx = 3
        for ch in "abcdefghijklmnopqrstuvwxyz ":
            vocab[ch] = idx
            idx += 1

        def encode(text, max_len):
            ids = [vocab.get(ch, 0) for ch in text.lower()]
            ids = ids[:max_len]
            pad = [0] * (max_len - len(ids))
            return torch.tensor(ids + pad)

        def make_segments(max_len):
            return torch.zeros(max_len, dtype=torch.long)

        text = "hello world"
        seq = encode(text, 512)
        seg = make_segments(512)

        out = bert(seq, seg)
        assert out.size() == torch.Size([MAX_LEN, EMBED_DIM])
        print(out)
        
        
            
        