"""
================================================================================
FIGURE 2 — QLoRA ADAPTATION BUDGET SWEEP ON CLEAN DATA
================================================================================
Re-runs the sweep that produced the original Figure 2, this time on the
deduplicated held-out set and with an adaptation pool verified disjoint from it.
The original sweep (1.163 -> 1.121 -> 1.120) drew its adaptation pool from the
same stream position as its evaluation text, so it cannot be published.

Budgets: 0 (no adaptation), 1M, 10M tokens.
Scores each on the clean held-out set under both protocols, then plots Protocol B.

RUNTIME on a Kaggle T4, roughly:
    tokenize an 11M-token pool      ~15 min
    train 1M   (488 steps)          ~19 min
    train 10M  (4,882 steps)       ~190 min
    scoring, 3 models x 2 protocols ~20 min
                                   --------
                                   ~4 hours

RESTARTABLE. Each budget saves its adapter and its result. Re-running skips
anything already finished, so a session timeout costs you only the budget that
was in flight. If you have to split it across sessions, commit the notebook
("Save & Run All") so /kaggle/working persists.

REQUIREMENTS
    Run 02_clean_eval_and_score.py in this session first (or have its outputs in
    /kaggle/working): uz_heldout_dedup.txt and dedup_chunks_from_scratch.csv.
    !pip install -q transformers datasets peft bitsandbytes accelerate matplotlib
    Accelerator: GPU T4. Internet: ON.
================================================================================
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc, math, json, csv, time
import numpy as np
import torch

LN2  = math.log(2.0)
WORK = "/kaggle/working"
CLEAN_FILE = f"{WORK}/uz_heldout_dedup.txt"
if not os.path.exists(CLEAN_FILE):
    raise SystemExit(f"missing {CLEAN_FILE} — run 02_clean_eval_and_score.py in this session first")
TEXT = open(CLEAN_FILE, encoding="utf-8").read()

BASE      = "ai-forever/mGPT"
BUDGETS   = [0, 1_000_000, 10_000_000]
CTX_TRAIN = 512
PER_DEV, ACC = 2, 2
LR        = 2e-4
SEED      = 20260913

CHUNK_TOKENS, SPAN_BYTES, CONTEXT_BYTES, MAX_TOKENS = 512, 800, 300, 512
FROM_SCRATCH_B = 1.0281      # from results_v3.json, Protocol B
FROM_SCRATCH_A = 1.0264      # Protocol A

assert torch.cuda.is_available(), "Enable a GPU: Settings -> Accelerator -> GPU T4."
DEV  = "cuda"
BF16 = torch.cuda.get_device_capability(0)[0] >= 8
DT   = torch.bfloat16 if BF16 else torch.float16
print(f"GPU: {torch.cuda.get_device_name(0)} | bf16: {BF16}")

def free(): gc.collect(); torch.cuda.empty_cache()

from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, default_data_collator)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from torch.utils.data import Dataset

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.model_max_length = 10**9

# ---------------------- pool, sized for the largest budget -------------------
POOL_FILE = f"{WORK}/sweep_pool_{max(BUDGETS)}.npy"
if os.path.exists(POOL_FILE):
    POOL = np.load(POOL_FILE)
    print(f"reusing pool: {len(POOL)/1e6:.2f}M tokens")
else:
    need = int(max(BUDGETS) * 1.10)
    print(f"tokenizing ~{need/1e6:.1f}M adaptation tokens (streaming; ~15 min)...")
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="uzn_Latn",
                      split="train", streaming=True)
    ids, texts, t0 = [], [], time.time()
    for ex in ds:
        t = (ex.get("text") or "").strip()
        if not t: continue
        texts.append(t)
        ids.extend(tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
        if len(ids) >= need: break
        if len(ids) % 2_000_000 < 5000:
            print(f"  {len(ids)/1e6:.1f}M tokens, {(time.time()-t0)/60:.1f} min")
    POOL = np.asarray(ids[:need], dtype=np.int32)
    np.save(POOL_FILE, POOL)
    # the eval set comes from val.bin (end of corpus); the pool streams from the
    # start, so they should be disjoint. Verify at the text level anyway.
    pool_text = "\n".join(texts)
    probes = [TEXT[i:i+300] for i in np.linspace(0, len(TEXT)-400, 12).astype(int)]
    leak = sum(1 for p in probes if p in pool_text)
    print(f"tokenized {len(POOL)/1e6:.2f}M tokens | eval-text probes in pool: {leak}/12")
    if leak:
        raise SystemExit("ABORT — adaptation pool overlaps the eval set.")

class Blocks(Dataset):
    def __init__(s, a, c): s.a, s.c, s.n = a, c, len(a)//c
    def __len__(s): return s.n
    def __getitem__(s, i):
        x = s.a[i*s.c:(i+1)*s.c].astype(np.int64)
        return {"input_ids": torch.from_numpy(x), "labels": torch.from_numpy(x.copy())}

def bnb_cfg():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=DT, bnb_4bit_use_double_quant=True)

def train_budget(budget):
    adir = f"{WORK}/sweep_adapter_{budget}"
    if os.path.exists(f"{adir}/adapter_model.safetensors"):
        print(f"  reusing adapter for {budget:,}"); return adir
    steps = budget // (PER_DEV * ACC * CTX_TRAIN)
    print(f"  training {budget:,} tokens = {steps:,} steps...")
    m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb_cfg(),
                                             device_map={"": 0})
    m.config.use_cache = False
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                     task_type="CAUSAL_LM", target_modules="all-linear"))
    args = TrainingArguments(
        output_dir=adir, per_device_train_batch_size=PER_DEV,
        gradient_accumulation_steps=ACC, max_steps=steps, learning_rate=LR,
        lr_scheduler_type="cosine", warmup_steps=max(10, steps//20),
        logging_steps=max(50, steps//10), save_strategy="no", report_to=[],
        fp16=not BF16, bf16=BF16, gradient_checkpointing=True,
        optim="paged_adamw_8bit", seed=SEED)
    t0 = time.time()
    Trainer(model=m, args=args, train_dataset=Blocks(POOL[:int(budget*1.05)], CTX_TRAIN),
            data_collator=default_data_collator).train()
    print(f"  trained in {(time.time()-t0)/60:.1f} min")
    m.save_pretrained(adir); del m; free()
    return adir

def logits_of(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else getattr(out, "logits", out)

@torch.no_grad()
def protocol_a(model):
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    n = len(ids)//CHUNK_TOKENS
    C = torch.tensor(ids[:n*CHUNK_TOKENS], dtype=torch.long).view(n, CHUNK_TOKENS)
    nats, byts = 0.0, 0
    for i in range(n):
        x = C[i:i+1].to(DEV)
        lg = logits_of(model, x); sl, st = lg[:, :-1, :].float(), x[:, 1:]
        nats += torch.nn.functional.cross_entropy(
            sl.reshape(-1, sl.size(-1)), st.reshape(-1), reduction="sum").item()
        byts += len(tok.decode(st[0].tolist()).encode("utf-8"))
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
def protocol_b(model, tag):
    rows = []
    for span_id, a, b in SPANS:
        n_bytes = len(TEXT[a:b].encode("utf-8"))
        c, got = a, 0
        while c > 0 and got < CONTEXT_BYTES:
            c -= 1; got += len(TEXT[c].encode("utf-8"))
        enc = tok(TEXT[c:b], return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        first = next((k for k,(s,e) in enumerate(offs) if s >= a-c), len(ids))
        if len(ids) < 2: continue
        if len(ids) > MAX_TOKENS:
            drop = len(ids)-MAX_TOKENS; ids = ids[drop:]; first = max(0, first-drop)
        x = torch.tensor(ids, dtype=torch.long, device=DEV).unsqueeze(0)
        lg = logits_of(model, x)
        logits = lg[:, :-1, :].float(); target = x[:, 1:].clone()
        if first > 1: target[:, :first-1] = -100
        rows.append((span_id, torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1),
            ignore_index=-100, reduction="sum").item(), n_bytes))
    with open(f"{WORK}/sweep_chunks_{tag}.csv","w",newline="",encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["chunk_id","sum_nats","n_bytes"])
        w.writerows([(i, f"{v:.6f}", n) for i,v,n in rows])
    return sum(r[1] for r in rows)/(LN2*sum(r[2] for r in rows))

# ------------------------------- sweep --------------------------------------
RES_FILE = f"{WORK}/figure2_clean.json"
RES = json.load(open(RES_FILE)) if os.path.exists(RES_FILE) else {}

for budget in BUDGETS:
    key = str(budget)
    if key in RES:
        print(f"\n[{budget:,}] cached: B={RES[key]['bpb_b']:.4f}"); continue
    print(f"\n{'='*60}\nbudget {budget:,} tokens\n{'='*60}")
    adir = train_budget(budget) if budget > 0 else None
    m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb_cfg(),
                                             device_map={"": 0}).eval()
    if adir: m = PeftModel.from_pretrained(m, adir).eval()
    bb = protocol_b(m, key); ba = protocol_a(m)
    RES[key] = {"budget": budget, "bpb_b": bb, "bpb_a": ba}
    json.dump(RES, open(RES_FILE,"w"), indent=2)
    print(f"  Protocol B: {bb:.4f} | Protocol A: {ba:.4f}")
    del m; free()

# ------------------------------- plot ---------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xs = [RES[str(b)]["budget"] for b in BUDGETS]
ys = [RES[str(b)]["bpb_b"] for b in BUDGETS]
xpos = list(range(len(xs)))

fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=200)
ax.plot(xpos, ys, "o-", color="#1f4e79", lw=2, ms=7, label="mGPT-1.3B + QLoRA", zorder=3)
for x, y in zip(xpos, ys):
    ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=9, color="#1f4e79")
ax.axhline(FROM_SCRATCH_B, ls="--", color="#c0392b", lw=1.6,
           label=f"uzbek-gpt-103m (from scratch), {FROM_SCRATCH_B:.4f}", zorder=2)
ax.set_xticks(xpos)
ax.set_xticklabels(["0\n(zero-shot)", "1M", "10M"])
ax.set_xlabel("Uzbek adaptation tokens")
ax.set_ylabel("Bits-per-byte  (lower is better)")
ax.set_title("Adaptation budget vs. bits-per-byte, deduplicated held-out set",
             fontsize=11, pad=12)
lo = min(min(ys), FROM_SCRATCH_B); hi = max(max(ys), FROM_SCRATCH_B); pad = (hi-lo)*0.28 or 0.01
ax.set_ylim(lo-pad, hi+pad)
ax.grid(axis="y", alpha=0.25, ls=":")
ax.legend(frameon=False, fontsize=9, loc="center right")
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{WORK}/figure2_clean.png", bbox_inches="tight")
print(f"\nsaved -> {WORK}/figure2_clean.png")

# --------------------------- caption + prose --------------------------------
d_1m   = ys[1] - ys[0]
d_10m  = ys[2] - ys[1]
gap_10 = ys[2] - FROM_SCRATCH_B

print(f"""
{'='*72}
RESULTS
{'='*72}
  0 tokens   Protocol B {RES['0']['bpb_b']:.4f}   Protocol A {RES['0']['bpb_a']:.4f}
  1M tokens  Protocol B {RES[str(BUDGETS[1])]['bpb_b']:.4f}   Protocol A {RES[str(BUDGETS[1])]['bpb_a']:.4f}
  10M tokens Protocol B {RES[str(BUDGETS[2])]['bpb_b']:.4f}   Protocol A {RES[str(BUDGETS[2])]['bpb_a']:.4f}

  0 -> 1M   change {d_1m:+.4f} bpb
  1M -> 10M change {d_10m:+.4f} bpb   (tenfold data increase)
  gap to from-scratch at 10M: {gap_10:+.4f} bpb

DRAFT CAPTION (check the numbers read correctly before using):

  Figure 2. Bits-per-byte of the mGPT-1.3B + QLoRA baseline as a function of
  adaptation tokens (0, 1M, 10M), evaluated on the deduplicated held-out set
  under Protocol B. The adaptation pool was verified disjoint from the
  evaluation text. A tenfold increase in adaptation data (1M to 10M) changes
  bits-per-byte by {d_10m:.4f}, leaving the adapted baseline {gap_10:.3f} above the
  from-scratch model's {FROM_SCRATCH_B:.4f} (dashed line). Created by the author.
{'='*72}
""")
