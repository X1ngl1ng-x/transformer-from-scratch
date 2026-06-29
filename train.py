import torch
import torch.nn as nn
from model import Transformer

# reserved special token ids
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

def make_batch(batch_size, seq_len, vocab_size, device):
    # toy copy task: the model has to reproduce the source sequence
    # real tokens live in [3, vocab_size), ids 0/1/2 are reserved for pad/sos/eos
    tokens = torch.randint(3, vocab_size, (batch_size, seq_len), device = device)

    # encoder input
    src = tokens

    # decoder input is shifted right with a leading <sos>
    sos = torch.full((batch_size, 1), SOS_IDX, dtype = torch.long, device = device)
    tgt_in = torch.cat([sos, tokens], dim = 1)

    # labels are the sequence followed by <eos>
    eos = torch.full((batch_size, 1), EOS_IDX, dtype = torch.long, device = device)
    tgt_out = torch.cat([tokens, eos], dim = 1)

    return src, tgt_in, tgt_out

def evaluate(model, vocab_size, seq_len, device):
    # generate a fresh sequence and check whether the model copies it
    src, _, _ = make_batch(1, seq_len, vocab_size, device)
    generated = model.generate(
        src,
        max_len = seq_len + 2,
        sos_idx = SOS_IDX,
        eos_idx = EOS_IDX,
    )

    print("\nsource:   ", src[0].tolist())
    # drop the leading <sos> token for readability
    print("generated:", generated[0, 1:].tolist())

def main():
    # reproducibility
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # task / model hyperparameters
    vocab_size = 50
    emb_dim = 128
    num_heads = 8
    forward_dim = 512
    num_layers = 2
    dropout = 0.1
    max_len = 20

    # training hyperparameters
    batch_size = 32
    seq_len = 8
    num_steps = 800
    lr = 3e-4

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_pad_idx=PAD_IDX,
        tgt_pad_idx=PAD_IDX,
        emb_dim=emb_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        forward_dim=forward_dim,
        dropout=dropout,
        max_len=max_len,
    ).to(device)

    # cross entropy ignores padding positions
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    model.train()
    for step in range(1, num_steps + 1):
        src, tgt_in, tgt_out = make_batch(batch_size, seq_len, vocab_size, device)

        # logits: (batch_size, tgt_len, vocab_size)
        logits = model(src, tgt_in)

        # flatten to (batch_size * tgt_len, vocab_size) for cross entropy
        loss = criterion(
            logits.reshape(-1, vocab_size),
            tgt_out.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    # quick check: can the trained model copy an unseen sequence?
    evaluate(model, vocab_size, seq_len, device)

if __name__ == "__main__":
    main()
