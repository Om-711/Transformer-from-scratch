# 🚀 Transformer From Scratch (PyTorch)

A complete implementation of the **Transformer architecture from scratch using PyTorch**, inspired by the paper:

> **"Attention Is All You Need" — Vaswani et al., 2017**

This project builds the full Encoder-Decoder Transformer without using high-level transformer libraries like `torch.nn.Transformer`.

It is designed for deep understanding of:
- Self-Attention
- Multi-Head Attention
- Positional Encoding
- Masking
- Encoder-Decoder Architecture
- Greedy Decoding
- Training Pipeline

---

# 📂 Project Structure
Transformer-from-scratch/
│
├── model.py # Transformer model implementation
├── train.py # Training loop
├── config.py # Hyperparameters
├── utils.py # Helper functions
├── weights/ # Saved model checkpoints
├── runs/ # TensorBoard logs
└── README.md




---

# 🧠 Architecture Overview

## 🔹 Encoder

Each Encoder block contains:
- Multi-Head Self-Attention
- Feed Forward Network
- Residual Connections
- Layer Normalization

## 🔹 Decoder

Each Decoder block contains:
- Masked Multi-Head Self-Attention
- Cross-Attention (Encoder–Decoder)
- Feed Forward Network
- Residual Connections
- Layer Normalization

## 🔹 Positional Encoding

Sinusoidal positional encoding:
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))



---

# ⚙️ Configuration

Defined in `config.py`.

Example:

```python
{
    "batch_size": 8,
    "num_epochs": 20,
    "lr": 1e-4,
    "seq_len": 350,
    "d_model": 512,
    "lang_src": "en",
    "lang_tgt": "it",
    "model_folder": "weights",
    "model_basename": "tmodel_",
    "preload": None,
    "tokenizer_file": "tokenizer_{0}.json",
    "experiment_name": "runs/tmodel"
}
▶️ Training

Run:

python train.py

During training you will see:

Dataset loading

Tokenizer building

Training progress bar

Loss updates

Model checkpoint saving

Example output:

Processing epoch 00: 45%|████▌ | loss=2.314
💾 Model Checkpoints

Saved in:

weights/

Each epoch saves:

tmodel_00.pt
tmodel_01.pt
...
📊 TensorBoard Logging

Logs saved in:

runs/tmodel

Start TensorBoard:

tensorboard --logdir runs
🔍 Greedy Decoding (Inference)

Greedy decoding generates output token-by-token:

model_out = greedy_decode(
    model,
    encoder_input,
    encoder_mask,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device
)

It:

Starts with [SOS]

Predicts next token

Appends to sequence

Stops at [EOS] or max_len

🧪 Dataset

Uses HuggingFace datasets.

Example:

load_dataset("opus_books", "en-it", split="train")

⚠ Ensure the language pair is supported by the dataset.
