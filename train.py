"""
Pretraining loop for Uzbek GPT.

Reads pre-tokenized uint16 token IDs from train.bin / val.bin (memory-mapped),
trains the from-scratch GPT in bf16 with gradient accumulation, a cosine LR
schedule, gradient clipping, torch.compile, and periodic checkpointing.

Data prep (tokenize FineWeb-2 uzn_Latn into train.bin/val.bin) is in
tokenize_data.py. The tokenizer is published at IslombekT/uzbek-bpe-16k.
"""

import math
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT

# ===================== CONFIG =====================
DATA_DIR     = "."          # folder containing train.bin / val.bin
vocab_size   = 16384
n_embd       = 768
n_layer      = 12
n_head       = 12
block_size   = 1024
micro_batch  = 8            # raise if GPU memory allows; lower if OOM
grad_accum   = 24           # effective batch = micro_batch * grad_accum = 192
max_steps    = 9700         # ~2 passes over ~955M train tokens
warmup_steps = 400
peak_lr      = 3e-4
min_lr       = 3e-5
weight_decay = 0.1
grad_clip    = 1.0
eval_every   = 500
eval_iters   = 40
COMPILE      = True
ckpt_dir     = "."
# ==================================================

train_data = np.memmap(f"{DATA_DIR}/train.bin", dtype=np.uint16, mode="r")
val_data   = np.memmap(f"{DATA_DIR}/val.bin",   dtype=np.uint16, mode="r")


def get_batch(split, bs, bl, device):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - bl - 1, (bs,))
    x = torch.stack([torch.from_numpy(d[i:i + bl].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + bl + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def get_lr(step):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    p = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * p)) * (peak_lr - min_lr)


def main():
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    print(f"device={device} | bf16={use_bf16}")

    raw_model = GPT(vocab_size, n_embd, block_size, n_head, n_layer).to(device)
    print("parameters:", round(sum(p.numel() for p in raw_model.parameters()) / 1e6, 2), "M")
    model = torch.compile(raw_model) if COMPILE else raw_model

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=peak_lr,
                                  betas=(0.9, 0.95), weight_decay=weight_decay)
    cfg = dict(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer,
               n_head=n_head, block_size=block_size)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for sp in ["train", "val"]:
            losses = []
            for _ in range(eval_iters):
                xb, yb = get_batch(sp, micro_batch, block_size, device)
                with ctx:
                    lo = model(xb)
                    B, T, V = lo.shape
                    losses.append(F.cross_entropy(lo.view(B * T, V), yb.view(B * T)).item())
            out[sp] = sum(losses) / len(losses)
        model.train()
        return out

    best_val = float("inf")
    model.train()
    t0 = run_start = time.time()
    for step in range(max_steps):
        for g in optimizer.param_groups:
            g["lr"] = get_lr(step)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            xb, yb = get_batch("train", micro_batch, block_size, device)
            with ctx:
                lo = model(xb)
                B, T, V = lo.shape
                loss = F.cross_entropy(lo.view(B * T, V), yb.view(B * T)) / grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        optimizer.step()

        if step % eval_every == 0 or step == max_steps - 1:
            m = estimate_loss()
            dt = time.time() - t0
            t0 = time.time()
            elapsed = (time.time() - run_start) / 3600
            print(f"step {step:5d}/{max_steps} | train {m['train']:.4f} | "
                  f"val {m['val']:.4f} | {dt:.0f}s | {elapsed:.2f}h")
            torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "val": m["val"], "config": cfg}, f"{ckpt_dir}/ckpt_last.pt")
            if m["val"] < best_val:
                best_val = m["val"]
                torch.save({"model": raw_model.state_dict(), "step": step,
                            "val": m["val"], "config": cfg}, f"{ckpt_dir}/ckpt_best.pt")
    print(f"DONE | best val {round(best_val, 4)} | {(time.time() - run_start) / 3600:.2f}h")


if __name__ == "__main__":
    main()
