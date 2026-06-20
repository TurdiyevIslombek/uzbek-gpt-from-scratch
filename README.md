# Uzbek GPT — a language model from scratch

A ~103M-parameter, decoder-only GPT-style language model **pretrained from scratch on Uzbek (Latin script)** — with the transformer architecture implemented by hand (RMSNorm + RoPE + SwiGLU + multi-head causal attention), a custom Uzbek tokenizer, and a single-GPU pretraining run.

**Model weights:** [`IslombekT/uzbek-gpt-103m`](https://huggingface.co/IslombekT/uzbek-gpt-103m) on Hugging Face
**Tokenizer:** [`IslombekT/uzbek-bpe-16k`](https://huggingface.co/IslombekT/uzbek-bpe-16k) (16,384-vocab byte-level BPE)

This repository is the **code**: the architecture, the data pipeline, the training loop, and a generation script. It is meant to be readable end to end.

## Try it (no training, no GPU)

The trained weights are published on Hugging Face. To generate Uzbek text from them you only need this repo's code plus the weights file:

```bash
git clone https://github.com/TurdiyevIslombek/uzbek-gpt-from-scratch.git
cd uzbek-gpt-from-scratch
pip install -r requirements.txt

# download the trained weights (~425 MB) from the Hugging Face model
wget https://huggingface.co/IslombekT/uzbek-gpt-103m/resolve/main/model.safetensors

# generate
python generate.py --ckpt model.safetensors --prompt "Oʻzbekiston "
```

`generate.py` loads `model.safetensors` directly — no checkpoint of your own and no training required. The tokenizer downloads automatically from the Hub. A one-click runnable version is also available as a [Kaggle notebook](#).

## Why

Uzbek is a low-resource language with little dedicated open tooling. The goal was to build a usable Uzbek base model the hard way — understanding every component rather than fine-tuning an existing model — and release it openly as a foundation for further Uzbek NLP work.

## What's here

| File | What it does |
|---|---|
| `model.py` | The from-scratch transformer: RoPE, RMSNorm, SwiGLU, fused causal attention, the GPT module. |
| `tokenize_data.py` | Tokenizes the FineWeb-2 `uzn_Latn` split into `train.bin` / `val.bin` (uint16, `<|endoftext|>`-separated, 90/10 split). |
| `train.py` | The pretraining loop: bf16 mixed precision, gradient accumulation, cosine LR schedule, gradient clipping, `torch.compile`, checkpointing. |
| `generate.py` | Loads a checkpoint (`.pt` or `.safetensors`) and samples Uzbek text. |

## Architecture

| Component | Choice |
|---|---|
| Normalization | RMSNorm (pre-norm) |
| Positional encoding | Rotary Position Embeddings (RoPE) |
| Feed-forward | SwiGLU |
| Attention | Multi-head causal self-attention (scaled dot-product) |
| Layers / dim / heads | 12 / 768 / 12 |
| Vocab / context | 16,384 / 1,024 |
| Parameters | 103.06M |

## Training

- **Data:** Uzbek Latin split of [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) — ~1.23M documents → ~1.06B tokens (955M train / 106M val).
- **Hardware:** 1× RTX 4090 (24 GB).
- **Setup:** AdamW (0.9, 0.95; wd 0.1), warmup 400 → cosine decay (3e-4 → 3e-5), effective batch 192 sequences (~197K tokens/step), 9,700 steps (~2 epochs).
- **Result:** ~3.4 h wall-clock, final validation loss **3.059**.

## Quick start

```bash
pip install -r requirements.txt

# 1. (optional) rebuild the token files from FineWeb-2 — CPU, ~20-40 min
python tokenize_data.py

# 2. pretrain (single GPU)
python train.py

# 3. generate, from your checkpoint or straight from the HF release
python generate.py --ckpt ckpt_best.pt --prompt "Oʻzbekiston "
```

To generate from the published weights without training, download `model.safetensors` from the [Hugging Face model](https://huggingface.co/IslombekT/uzbek-gpt-103m) and pass it to `generate.py`.

## Limitations

A small base model on ~1B tokens: it produces fluent, grammatical Uzbek but is **not** an instruction/chat model, is not factually reliable, and can repeat without a repetition penalty. See the [model card](https://huggingface.co/IslombekT/uzbek-gpt-103m) for details and sample outputs.

## License

Apache-2.0.

## Acknowledgements

Built on the cleaned **FineWeb-2** corpus (Hugging Face). Architecture and training informed by the open from-scratch LM literature (GPT-2; the RoPE / RMSNorm / SwiGLU lines of work; and public educational implementations of transformer training).
