"""
Generate Uzbek text from a trained Uzbek GPT checkpoint.

Loads weights (either a training .pt checkpoint or model.safetensors from the
Hugging Face release) and samples with temperature, top-k, and a repetition
penalty (recommended for small base models, which otherwise loop).

  python generate.py --ckpt ckpt_best.pt --prompt "Oʻzbekiston "
"""

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import GPT

TOKENIZER = "IslombekT/uzbek-bpe-16k"


def load_model(ckpt_path, device):
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(ckpt_path)
        cfg = dict(vocab_size=16384, n_embd=768, block_size=1024, num_heads=12, n_layers=12)
    else:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["model"]
        c = ck["config"]
        cfg = dict(vocab_size=c["vocab_size"], n_embd=c["n_embd"],
                   block_size=c["block_size"], num_heads=c["n_head"], n_layers=c["n_layer"])
    model = GPT(**cfg).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, cfg["block_size"]


@torch.no_grad()
def generate(model, tok, block_size, prompt, device,
             max_new_tokens=120, temperature=0.8, top_k=50, rep_penalty=1.3):
    ids = tok.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        cond = x[:, -block_size:]
        logits = model(cond)[:, -1, :]
        for t in set(x[0].tolist()):
            logits[0, t] /= rep_penalty
        logits = logits / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits.float(), dim=-1)
        x = torch.cat([x, torch.multinomial(probs, num_samples=1)], dim=1)
    return tok.decode(x[0].tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    ap.add_argument("--prompt", default="Oʻzbekiston ")
    ap.add_argument("--tokens", type=int, default=120)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    model, block_size = load_model(args.ckpt, device)
    print(generate(model, tok, block_size, args.prompt, device, max_new_tokens=args.tokens))
