import torch
import math
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.head_dim = emb_dim // num_heads

        # create query layer
        self.query_layer = nn.Linear(emb_dim, num_heads * self.head_dim)
        # create key layer
        self.key_layer = nn.Linear(self.emb_dim, num_heads * self.head_dim)
        # create value layer
        self.value_layer = nn.Linear(self.emb_dim, num_heads * self.head_dim)

        # create output linear layer
        self.output_layer = nn.Linear(num_heads * self.head_dim, self.emb_dim)
        
    def forward(self, query, key, value, mask=None):
        # query: (batch_size, query_length, emb_dim)
        # key, value: (batch_size, key_len, emb_dim)
        batch_size, query_len,_ = query.size()
        _, key_len,_ = key.size()

        # linear projection
        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = self.value_layer(value)

        # reshape: (B, L, D) ->  (B, L,  num_heads * head_dim) & transpose to split heads
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention scores
        # Q @ K^T: (B, H, L_q, L_k)
        key_out = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dim)

        # apply mask
        if mask is not None:
            key_out = key_out.masked_fill(mask == 0, -1e20)
        
        # softmax over key dimension
        attn_weights = torch.softmax(key_out, dim = -1)

        # weighted sum of values: (B, H, L_q, d_k)
        attn_out = torch.matmul(attn_weights, V)

        # concatenate heads: (B, H, L_q, d_k) -> (B, H, L_q, Heads * d_k)
        attn_out = (
            attn_out.transpose(1,2)
            .contiguous()
            .view(batch_size, query_len, self.num_heads * self.head_dim)
        )

        # final linear layer back to emb_dim
        out = self.output_layer(attn_out)
        return out   

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, forward_dim):
        super().__init__()
        self.attn = MultiHeadAttention(emb_dim, num_heads)

        # Normalization Layer
        self.norm1 = nn.LayerNorm(emb_dim, eps = 1e-6)
        self.norm2 = nn.LayerNorm(emb_dim, eps = 1e-6)

        # feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, emb_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        attn_out = self.attn(query, key, value, mask)
        
        # build a redidual connection and apply dropout
        x = self.dropout(attn_out) + query
        x = self.norm1(x)

        # FFN
        ffn_out = self.ffn(x)

        # residual connection and normalization after FFN
        x= x + self.dropout(ffn_out)
        x = self.norm2(x)

        return x

class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, forward_dim, dropout):
        super().__init__()
        
        self.norm = nn.LayerNorm(emb_dim, eps = 1e-6)
        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.TransformerBlock = TransformerBlock(
            emb_dim,
            num_heads,
            dropout,
            forward_dim
        )

    def forward(self, x, value, key, src_mask, tgt_mask):
        # masked self-attention
        self_attn_out = self.attn(x,x,x,tgt_mask)
        # build residual connection and apply dropout and normalization
        x = x + self.dropout(self_attn_out) 
        query = self.norm(x)
        # run through TransformerBlock
        cross_attn_out = self.TransformerBlock(query, key, value, src_mask)
        return cross_attn_out
        
def get_sinusoid_table(max_len, emb_dim):
    position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

    i = torch.arange(0, emb_dim, 2, dtype=torch.float)  # 0,2,4,6...(2i)
    div_term = 1 / (10000 ** (i / emb_dim))

    sinusoid_table = torch.zeros(max_len, emb_dim)

    sinusoid_table[:, 0::2] = torch.sin(position * div_term)
    sinusoid_table[:, 1::2] = torch.cos(position * div_term)

    return sinusoid_table