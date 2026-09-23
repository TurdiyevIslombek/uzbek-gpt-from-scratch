"""
================================================================================
ABLATION RE-SCORE  +  EVAL-SET DIAGNOSTIC
================================================================================
Run in the SAME Kaggle session as 02_clean_eval_and_score.py (it reuses the files it
wrote to /kaggle/working). Add your ablation_best.pt dataset as an Input first.

Two jobs:
  PART 1  diagnose WHY the clean eval set scores so differently from the old one
          — no GPU, ~5 seconds. The -0.081 "memorisation" number is confounded
          and this shows by how much.
  PART 2  score the 232M tokenizer-ablation model on the clean eval set, under
          both protocols, and bootstrap it against the from-scratch model.

~10 minutes. No retraining.
================================================================================
"""

import os, sys, glob, math, csv, json
import numpy as np
import torch

LN2  = math.log(2.0)
WORK = "/kaggle/working"
CLEAN_FILE = f"{WORK}/uz_heldout_dedup.txt"
OLD_FILE   = f"{WORK}/uz_heldout.txt"
for f in (CLEAN_FILE, OLD_FILE, f"{WORK}/dedup_chunks_from_scratch.csv"):
    if not os.path.exists(f):
        raise SystemExit(f"missing {f} — run 02_clean_eval_and_score.py in this session first")

TEXT     = open(CLEAN_FILE, encoding="utf-8").read()
OLD_TEXT = open(OLD_FILE,   encoding="utf-8").read()

CHUNK_TOKENS, SPAN_BYTES, CONTEXT_BYTES, MAX_TOKENS = 512, 800, 300, 512
RESAMPLES, SEED = 10_000, 20260913

# ============================== PART 1 ======================================
print("="*72); print("PART 1 — why do the two eval sets score so differently?"); print("="*72)

def profile(t, name):
    n = len(t)
    cyr  = sum(1 for c in t if '\u0400' <= c <= '\u04FF')
    lat  = sum(1 for c in t if ('a' <= c.lower() <= 'z') or c in "oʻgʻʼ‘’")
    dig  = sum(1 for c in t if c.isdigit())
    punc = sum(1 for c in t if not c.isalnum() and not c.isspace())
    words = t.split()
    print(f"  {name:<10} {n:>10,} chars | Cyrillic {cyr/n:6.2%} | Latin {lat/n:6.2%} "
          f"| digits {dig/n:5.2%} | punct {punc/n:5.2%} | mean word {np.mean([len(w) for w in words]):.2f}")
    return {"chars": n, "cyrillic_frac": cyr/n, "digit_frac": dig/n}

p_old   = profile(OLD_TEXT, "OLD")
p_clean = profile(TEXT,     "CLEAN")

print(f"""
  The from-scratch model's tokenizer was trained on Uzbek LATIN only. Cyrillic
  falls back to bytes, which is expensive for it and cheap for multilingual mGPT.
  If OLD carries more Cyrillic than CLEAN, that alone moves the two models by
  different amounts and the -0.081 'memorisation' figure is not memorisation.
""")

# mGPT as a control: it never trained on either set, so its shift between the two
# eval sets is pure text-difficulty. Subtract it to isolate anything model-specific.
FS_OLD, FS_CLEAN = 1.1078, 1.0264          # from v3 Protocol A
MG_OLD, MG_CLEAN = 1.1653, 1.1091          # 1.1653 = old npz, chunk-byte denominator
print(f"  from-scratch   OLD {FS_OLD:.4f} -> CLEAN {FS_CLEAN:.4f}   delta {FS_CLEAN-FS_OLD:+.4f}")
print(f"  mGPT base      OLD {MG_OLD:.4f} -> CLEAN {MG_CLEAN:.4f}   delta {MG_CLEAN-MG_OLD:+.4f}  <- control")
print(f"  difference-in-differences                  {(FS_CLEAN-FS_OLD)-(MG_CLEAN-MG_OLD):+.4f}")
print("  A NEGATIVE diff-in-diff means the from-scratch model did RELATIVELY BETTER")
print("  on text it never saw — the opposite of a memorisation benefit.\n")

# ============================== PART 2 ======================================
print("="*72); print("PART 2 — score the 232M tokenizer ablation on the clean set"); print("="*72)

from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

ck_paths = glob.glob("/kaggle/input/**/ablation_best.pt", recursive=True) or \
           glob.glob("/kaggle/input/**/ablation_last.pt", recursive=True)
if not ck_paths:
    raise SystemExit("ablation checkpoint not found. Add Input -> your ablation dataset.")
CKPT = ck_paths[0]
print(f"checkpoint: {CKPT} ({os.path.getsize(CKPT)/1e6:.0f} MB)")

mp = hf_hub_download("IslombekT/uzbek-gpt-103m", "model.py")
sys.path.insert(0, os.path.dirname(mp))
from model import GPT

ck  = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["config"]
print(f"config from checkpoint: {cfg}")
print(f"saved at step {ck.get('step')} | val {ck.get('val')}")

abl = GPT(cfg["vocab_size"], cfg["n_embd"], cfg["block_size"], cfg["n_head"], cfg["n_layer"])
abl.load_state_dict(ck["model"]); abl = abl.eval().cuda()
n_par = sum(p.numel() for p in abl.parameters())
print(f"ablation model: {n_par/1e6:.0f}M parameters\n")

tok = AutoTokenizer.from_pretrained("ai-forever/mGPT")   # the ablation's tokenizer
tok.model_max_length = 10**9

def logits_of(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else getattr(out, "logits", out)

@torch.no_grad()
def protocol_a(text, tag):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    n = len(ids)//CHUNK_TOKENS
    C = torch.tensor(ids[:n*CHUNK_TOKENS], dtype=torch.long).view(n, CHUNK_TOKENS)
    nats, byts, ntok = 0.0, 0, 0
    for i in range(n):
        x = C[i:i+1].cuda()
        lg = logits_of(abl, x); sl, st = lg[:, :-1, :].float(), x[:, 1:]
        nats += torch.nn.functional.cross_entropy(
            sl.reshape(-1, sl.size(-1)), st.reshape(-1), reduction="sum").item()
        byts += len(tok.decode(st[0].tolist()).encode("utf-8")); ntok += st.numel()
    print(f"  [{tag}] bpb={nats/(LN2*byts):.4f}  CE={nats/ntok:.4f}  chunks={n}")
    return nats/(LN2*byts)

def build_spans(text, span_bytes):
    spans, i, n, k = [], 0, len(text), 0
    while i < n:
        j, b = i, 0
        while j < n and b < span_bytes:
            b += len(text[j].encode("utf-8")); j += 1
        spans.append((f"span_{k:05d}", i, j)); i, k = j, k+1
    if spans and len(text[spans[-1][1]:spans[-1][2]].encode("utf-8")) < span_bytes//2:
        spans.pop()
    return spans

SPANS = build_spans(TEXT, SPAN_BYTES)

@torch.no_grad()
def protocol_b(tag):
    rows, over = [], 0
    for span_id, a, b in SPANS:
        n_bytes = len(TEXT[a:b].encode("utf-8"))
        c, got = a, 0
        while c > 0 and got < CONTEXT_BYTES:
            c -= 1; got += len(TEXT[c].encode("utf-8"))
        enc = tok(TEXT[c:b], return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        first = next((k for k, (s, e) in enumerate(offs) if s >= a-c), len(ids))
        if len(ids) < 2: continue
        if len(ids) > MAX_TOKENS:
            over += 1; drop = len(ids)-MAX_TOKENS
            ids = ids[drop:]; first = max(0, first-drop)
        x = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)
        lg = logits_of(abl, x)
        logits = lg[:, :-1, :].float(); target = x[:, 1:].clone()
        if first > 1: target[:, :first-1] = -100
        rows.append((span_id, torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1),
            ignore_index=-100, reduction="sum").item(), n_bytes))
    path = f"{WORK}/dedup_chunks_ablation.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["chunk_id","sum_nats","n_bytes"])
        w.writerows([(i, f"{v:.6f}", n) for i, v, n in rows])
    bpb = sum(r[1] for r in rows)/(LN2*sum(r[2] for r in rows))
    print(f"  [{tag}] bpb={bpb:.4f}  spans={len(rows)}" + (f"  ({over} truncated)" if over else ""))
    return bpb

a_clean = protocol_a(TEXT,     "ablation / CLEAN")
a_old   = protocol_a(OLD_TEXT, "ablation / OLD (contaminated)")
b_clean = protocol_b("ablation / spans")

# --------------------------- paired bootstrap -------------------------------
def load_csv(tag):
    d = {}
    with open(f"{WORK}/dedup_chunks_{tag}.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d[r["chunk_id"]] = (float(r["sum_nats"]), float(r["n_bytes"]))
    return d

def paired(ta, tb):
    a, b = load_csv(ta), load_csv(tb)
    assert set(a) == set(b), "span sets differ"
    ids = sorted(a)
    na = np.array([a[i][0] for i in ids]); ba = np.array([a[i][1] for i in ids])
    nb = np.array([b[i][0] for i in ids]); bb = np.array([b[i][1] for i in ids])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ids), size=(RESAMPLES, len(ids)))
    d = (nb[idx].sum(1)/(LN2*bb[idx].sum(1))) - (na[idx].sum(1)/(LN2*ba[idx].sum(1)))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": nb.sum()/(LN2*bb.sum()) - na.sum()/(LN2*ba.sum()),
            "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
            "p_le_zero": float((d <= 0).mean())}

bt = paired("from_scratch", "ablation")

print(f"""
{'='*72}
ABLATION RESULTS
{'='*72}
ablation model      {n_par/1e6:.0f}M params, mGPT tokenizer (vocab {cfg['vocab_size']:,})
                    trained on the same corpus text, same architecture

  ablation on OLD   (contaminated)   {a_old:.4f} bpb
  ablation on CLEAN (never seen)     {a_clean:.4f} bpb
  paper claimed                      1.1579 bpb (on contaminated text)

PROTOCOL A — 512-token chunks, CLEAN
  from-scratch 103M, own tokenizer   1.0264
  ablation     {n_par/1e6:.0f}M, mGPT tokenizer   {a_clean:.4f}

PROTOCOL B — byte-aligned spans, CLEAN  (the fair-context comparison)
  from-scratch 103M, own tokenizer   1.0281
  ablation     {n_par/1e6:.0f}M, mGPT tokenizer   {b_clean:.4f}

PAIRED BOOTSTRAP, from-scratch vs ablation (positive = dedicated tokenizer better)
  {bt['diff']:+.4f} bpb   95% CI [{bt['ci_lo']:+.4f}, {bt['ci_hi']:+.4f}]   excludes zero: {'YES' if bt['excludes_zero'] else 'NO'}   P(<=0)={bt['p_le_zero']:.4f}
{'='*72}
""")

json.dump({"eval_profile_old": p_old, "eval_profile_clean": p_clean,
           "ablation_params": int(n_par), "ablation_config": cfg,
           "ablation_a_clean": a_clean, "ablation_a_old": a_old,
           "ablation_b_clean": b_clean, "bootstrap_vs_from_scratch": bt},
          open(f"{WORK}/ablation_results.json", "w"), indent=2, default=float)
print(f"saved -> {WORK}/ablation_results.json")
