import torch
import torch.nn as nn
from modules import get_sinusoid_table, TransformerBlock, DecoderBlock

class Encoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            emb_dim,
            num_layers,
            num_heads,
            forward_dim,
            dropout,
            max_len,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        
        # create token-level Embeddings
        self.token_Embedding = nn.Embedding(vocab_size, emb_dim)

        # create position Embeddings
        self.max_len = max_len
        sinusoid_table = get_sinusoid_table(max_len, emb_dim)
        self.position_Embedding = nn.Embedding.from_pretrained(
            sinusoid_table,
            freeze = True
        )

        # dropout to decrease overfitting
        self.dropout = nn.Dropout(dropout)

        # TransformerBlock
        self.layers = nn.ModuleList([
            TransformerBlock(
                emb_dim,
                num_heads,
                dropout,
                forward_dim
            )
            for _ in range(num_layers)]
        )

    def forward(self, x, src_mask = None):
        batch_size, seq_length = x.shape
        device  = x.device

        if seq_length > self.max_len:
            raise ValueError(
                f"Source sequence length {seq_length} exceeds max_len {self.max_len}"
            )

        positions = torch.arange(seq_length, device = device).unsqueeze(0).expand(batch_size, seq_length)

        tok_emb = self.token_Embedding(x)
        pos_emb = self.position_Embedding(positions)
        x = tok_emb + pos_emb
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, x, x, src_mask)

        return x  

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_len = max_len

        # token embeddings
        self.tok_Embedding = nn.Embedding(vocab_size, emb_dim)
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        # num layers
        self.layers = nn.ModuleList([
            DecoderBlock(
                emb_dim, 
                num_heads, 
                forward_dim, 
                dropout)
            for _ in range(num_layers)
        ])
        # create positional embeddings
        sinusoid_table = get_sinusoid_table(max_len, emb_dim)
        self.position_embedding = nn.Embedding.from_pretrained(
            sinusoid_table,
            freeze = True
        )   

        # final projection to vocab
        self.output_layer = nn.Linear(emb_dim, vocab_size)


    def forward(self, x, encoder_out, src_mask, tgt_mask):
        batch_size, seq_len= x.shape
        device = x.device

        if seq_len > self.max_len:
            raise ValueError(
                f"Target sequence length {seq_len} exceeds max_len {self.max_len}"
            )

        tok_emb = self.tok_Embedding(x)

        positions = torch.arange(0, seq_len, device = device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.position_embedding(positions)

        x = tok_emb + pos_emb
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, tgt_mask)

        return self.output_layer(x)
    
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        emb_dim=512,
        num_layers=6,
        num_heads=8,
        forward_dim=2048,
        dropout=0.0,
        max_len=128,
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size,
            emb_dim,
            num_layers,
            num_heads,
            forward_dim,
            dropout,
            max_len
        )

        self.decoder = Decoder(
            tgt_vocab_size,
            emb_dim,
            num_layers,
            num_heads,
            forward_dim,
            dropout,
            max_len
        )

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx


    def create_src_mask(self, src):
        device = src.device
        # (batch_size, 1, 1, src_seq_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(device)
    

    def create_tgt_mask(self, tgt):
        _, tgt_len = tgt.shape

        tgt_padding_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt.device)
        ).bool()
        
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        return tgt_padding_mask & causal_mask

    def forward(self, src, tgt):
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)

        enc_out = self.encoder(src,src_mask)
        out = self.decoder(tgt, enc_out, src_mask, tgt_mask)

        return out

    @torch.no_grad()
    def generate(self, src, max_len, sos_idx, eos_idx = None):
        # src: (batch_size, src_len)
        self.eval()
        device = src.device
        batch_size = src.size(0)

        # encode the source once and reuse it at every decoding step
        src_mask = self.create_src_mask(src)
        enc_out = self.encoder(src, src_mask)

        # start every sequence with the <sos> token
        tgt = torch.full((batch_size, 1), sos_idx, dtype = torch.long, device = device)

        for _ in range(max_len - 1):
            tgt_mask = self.create_tgt_mask(tgt)
            # out: (batch_size, cur_len, tgt_vocab_size)
            out = self.decoder(tgt, enc_out, src_mask, tgt_mask)

            # take the logits at the last position and pick the most likely token
            next_token = out[:, -1, :].argmax(dim = -1, keepdim = True)
            tgt = torch.cat([tgt, next_token], dim = 1)

            # stop early once every sequence has produced <eos>
            if eos_idx is not None and (next_token == eos_idx).all():
                break

        return tgt