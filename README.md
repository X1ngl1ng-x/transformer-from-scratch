# transformer-from-scratch

A PyTorch implementation of the vanilla Transformer encoder-decoder architecture based on Attention Is All You Need.

## Project structure

- `modules.py`: core building blocks such as multi-head attention and transformer blocks
- `model.py`: high-level encoder, decoder, and full Transformer model (with greedy `generate`)
- `demo_shapes.py`: minimal runnable demo for forward pass and output shape verification
- `train.py`: end-to-end training loop on a toy copy task, with autoregressive evaluation

## implementation elements:

    - Scaled dot-product attention
    - Multi-head attention
    - Masking
    - Positional Encoding
    - Encoder/Decoder Blocks
    - Residual connections + LayerNorm
    - Full encoder-decoder Transformer architecture

