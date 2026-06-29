import torch
from model import Transformer

def main():
    # reproducibility
    torch.manual_seed(42)

    # demo hyperparameters
    src_vocab_size = 5000
    tgt_vocab_size = 6000
    emb_dim = 128
    num_heads = 8
    forward_dim = 512
    num_layers = 2
    dropout = 0.1
    max_len = 50

    batch_size = 4
    src_len = 10
    tgt_len = 8

    pad_token_id = 0

    # demo token ids
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))

    # randomly add some padding tokens to test whether the model can handle them
    src[0, -2:] = pad_token_id
    tgt[1, -1:] = pad_token_id

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_pad_idx=pad_token_id,
        tgt_pad_idx=pad_token_id,
        emb_dim=emb_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        forward_dim=forward_dim,
        dropout=dropout,
        max_len=max_len,
    )

    output = model(src, tgt)

    print("src shape:", src.shape)
    print("tgt shape:", tgt.shape)
    print("output shape:", output.shape)

    # expected: [batch_size, tgt_len, tgt_vocab_size]
    assert output.shape == (batch_size, tgt_len, tgt_vocab_size)
    print("Forward pass successful.")


if __name__ == "__main__":
    main()