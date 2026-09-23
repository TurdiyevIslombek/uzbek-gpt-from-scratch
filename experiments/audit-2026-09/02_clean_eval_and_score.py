"""
================================================================================
UZBEK-GPT PAPER — VERIFICATION RUN v3  (DEDUPLICATED EVAL SET)
================================================================================
Why v3 exists:

  v1  eval set was the literal opening of train.bin. Void.
  v2  eval set decoded from val.bin — but 6/16 probes still turned up in
      train.bin, at scattered positions (252M, 929M, 833M, ...). That is not a
      split error. FineWeb-2 itself contains the same passages in both halves.
      So val.bin is partly contaminated too, and no choice of slice fixes it.
  v3  filters DOCUMENT BY DOCUMENT. Builds a fingerprint index of every 24-token
      window in train.bin, then keeps only val documents with zero fingerprint
      overlap. What survives is text the from-scratch model provably never saw.

CONSEQUENCE YOU SHOULD ABSORB
  Your model's reported validation loss of 3.059 was measured on val.bin, which
  we now know is partly duplicated from train.bin. That number is optimistic too.
  This is a property of the corpus, not a mistake you made — but the paper has to
  say the eval set was deduplicated against the training split, because from now
  on it will have been.

STAGES
  1  fingerprint index of train.bin            (~3 min, one pass)
  2  walk val.bin, drop duplicated documents, build the clean eval set
  3  verify with exact search — must be 0/16
  4  rebuild the old contaminated set, for the memorisation delta
  5  adaptation pool + text-level leak check
  6  QLoRA fine-tune mGPT-1.3B, ctx 512, all-linear, ~1M tokens
  7  Protocol A (512-token chunks): 3 models on clean + from-scratch on old
  8  Protocol B (byte-aligned spans): 3 models on clean
  9  paired bootstrap
  10 RESULTS

RUNTIME ~70-90 min on a T4. Restartable; every stage caches to /kaggle/working.

FIRST, IN A SEPARATE CELL:
    !pip install -q transformers datasets peft bitsandbytes accelerate safetensors huggingface_hub
Accelerator: GPU T4. Internet: ON. Add Input: uzbek-fineweb2-tokens-16k.
================================================================================
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc, sys, math, json, csv, glob, time
import numpy as np
import torch

LN2  = math.log(2.0)
WORK = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)

# ------------------------------- config -------------------------------------
BASE          = "ai-forever/mGPT"
MY_MODEL_REPO = "IslombekT/uzbek-gpt-103m"
MY_TOK_REPO   = "IslombekT/uzbek-bpe-16k"
ADAPTER_DIR   = f"{WORK}/mgpt-uz-qlora-ctx512-dedup"

EOT_ID        = 0
KGRAM         = 24          # fingerprint window, ~18 Uzbek words
KEEP_MASK     = 31          # keep 1/32 of windows in the index
MAX_DUP_FRAC  = 0.0         # a document is rejected on ANY fingerprint match
MIN_DOC_WORDS = 30          # skip stubs, too short to fingerprint meaningfully
EVAL_WORDS    = 200_000

ADAPT_TOKENS  = 1_000_000
CTX_TRAIN     = 512
PER_DEV, ACC  = 2, 2
LR            = 2e-4

CHUNK_TOKENS  = 512
SPAN_BYTES    = 800
CONTEXT_BYTES = 300
MAX_TOKENS    = 512

N_PROBES      = 16
PROBE_TOK     = 32
RESAMPLES     = 10_000
SEED          = 20260913

assert torch.cuda.is_available(), "Enable a GPU: Settings -> Accelerator -> GPU T4."
DEV  = "cuda"
BF16 = torch.cuda.get_device_capability(0)[0] >= 8
DT   = torch.bfloat16 if BF16 else torch.float16
print(f"GPU: {torch.cuda.get_device_name(0)} | bf16: {BF16}")

def free(): gc.collect(); torch.cuda.empty_cache()
def stage(n, name): print(f"\n{'='*72}\nSTAGE {n} — {name}\n{'='*72}")

R = {"config": {"kgram": KGRAM, "eval_words": EVAL_WORDS, "adapt_tokens": ADAPT_TOKENS,
                "ctx_train": CTX_TRAIN, "span_bytes": SPAN_BYTES,
                "resamples": RESAMPLES, "seed": SEED}}

from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, default_data_collator)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.utils.data import Dataset

_B = np.uint64(1000003)

def kgram_hashes(arr, K=KGRAM, mask=KEEP_MASK, chunk=20_000_000):
    """Rolling polynomial hash of every K-gram, subsampled to 1/(mask+1)."""
    out, n = [], len(arr)
    if n < K:
        return np.zeros(0, dtype=np.uint64)
    for s in range(0, n - K + 1, chunk):
        e = min(s + chunk, n - K + 1); m = e - s
        h = np.zeros(m, dtype=np.uint64)
        for j in range(K):
            h = h * _B + arr[s+j : s+j+m].astype(np.uint64)
        out.append(h[(h & np.uint64(mask)) == 0])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.uint64)

def find_sequence(hay, needle):
    L = len(needle)
    if L == 0 or len(hay) < L: return []
    cand = np.flatnonzero(hay[:len(hay)-L+1] == needle[0])
    for k in range(1, L):
        if cand.size == 0: break
        cand = cand[hay[cand+k] == needle[k]]
    return cand.tolist()

# =============== STAGE 1 — fingerprint index of train.bin ===================
stage(1, "fingerprint every 24-token window of train.bin")
cands = glob.glob("/kaggle/input/**/train.bin", recursive=True)
if not cands:
    raise SystemExit("train.bin not found. Add Input -> uzbek-fineweb2-tokens-16k")
TRAIN_BIN = cands[0]
VAL_BIN   = os.path.join(os.path.dirname(TRAIN_BIN), "val.bin")
train_ids = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
val_ids   = np.memmap(VAL_BIN,   dtype=np.uint16, mode="r")
print(f"train.bin {len(train_ids):,} tokens | val.bin {len(val_ids):,} tokens")

IDX_FILE = f"{WORK}/train_fingerprints.npy"
if os.path.exists(IDX_FILE):
    IDX = np.load(IDX_FILE); print(f"reusing index: {len(IDX):,} fingerprints")
else:
    t0 = time.time()
    IDX = np.unique(kgram_hashes(np.asarray(train_ids)))
    np.save(IDX_FILE, IDX)
    print(f"built {len(IDX):,} fingerprints in {time.time()-t0:.0f}s ({IDX.nbytes/1e6:.0f} MB)")

def dup_fraction(tok_ids):
    h = kgram_hashes(np.asarray(tok_ids, dtype=np.uint16))
    if len(h) == 0: return 0.0, 0
    p = np.searchsorted(IDX, h); p[p >= len(IDX)] = 0
    return float((IDX[p] == h).mean()), len(h)

# ============ STAGE 2 — deduplicated eval set from val.bin ==================
stage(2, "build the eval set from val documents that are absent from train.bin")
tk = AutoTokenizer.from_pretrained(MY_TOK_REPO); tk.model_max_length = 10**9
CLEAN_FILE = f"{WORK}/uz_heldout_dedup.txt"

if os.path.exists(CLEAN_FILE):
    TEXT = open(CLEAN_FILE, encoding="utf-8").read()
    print(f"reusing {CLEAN_FILE}")
    R["docs_kept"] = R.get("docs_kept"); R["docs_rejected"] = R.get("docs_rejected")
else:
    kept, rejected, words, cur, i, shown = [], 0, 0, [], 0, 0
    while i < len(val_ids) and val_ids[i] != EOT_ID:   # start after a boundary
        i += 1
    i += 1
    while i < len(val_ids) and words < EVAL_WORDS:
        t = int(val_ids[i])
        if t == EOT_ID:
            if len(cur) >= MIN_DOC_WORDS:
                frac, n_fp = dup_fraction(cur)
                if n_fp > 0 and frac <= MAX_DUP_FRAC:
                    d = tk.decode(cur).strip()
                    if d:
                        kept.append(d); words += len(d.split())
                else:
                    rejected += 1
                    if shown < 3 and n_fp > 0 and frac > 0:
                        snip = tk.decode(cur[:40]).strip().replace("\n", " ")
                        print(f"  rejected ({frac:.0%} of windows in train): {snip[:110]}...")
                        shown += 1
            cur = []
        else:
            cur.append(t)
        i += 1
    TEXT = "\n".join(kept)
    open(CLEAN_FILE, "w", encoding="utf-8").write(TEXT)
    total = len(kept) + rejected
    print(f"\nkept {len(kept):,} documents, rejected {rejected:,} "
          f"({rejected/max(total,1):.1%} of val documents are duplicated in train)")
    R["docs_kept"], R["docs_rejected"] = len(kept), rejected

EVAL_BYTES = len(TEXT.encode("utf-8"))
print(f"eval set: {len(TEXT.split()):,} words, {EVAL_BYTES:,} bytes")
R["clean_eval_bytes"], R["clean_eval_words"] = EVAL_BYTES, len(TEXT.split())

# ==================== STAGE 3 — verify with exact search ====================
stage(3, "verify: exact search for eval passages inside train.bin")
eval_tok = np.asarray(tk(TEXT, add_special_tokens=False)["input_ids"], dtype=np.uint16)
pos = np.linspace(0, len(eval_tok)-PROBE_TOK-1, N_PROBES).astype(int)
hits = 0
for n, p in enumerate(pos, 1):
    h = find_sequence(train_ids, np.asarray(eval_tok[p:p+PROBE_TOK], dtype=np.uint16))
    hits += bool(h)
    print(f"  probe {n:>2}  eval tok {p:>9,}  " + (f"FOUND @{h[0]:,}" if h else "absent"))
print(f"\n{hits}/{N_PROBES} probes found in train.bin")
R["probes_in_train"] = f"{hits}/{N_PROBES}"
if hits > 0:
    raise SystemExit(f"ABORT — {hits}/{N_PROBES} still leaking. Lower KEEP_MASK to 15 "
                     f"(denser index) and rerun.")
print("eval set verified clean — proceeding")

# ============ STAGE 4 — the old contaminated set, for the delta =============
stage(4, "rebuild the OLD contaminated eval set (for the memorisation delta)")
OLD_FILE = f"{WORK}/uz_heldout.txt"
if os.path.exists(OLD_FILE):
    OLD_TEXT = open(OLD_FILE, encoding="utf-8").read(); print(f"reusing {OLD_FILE}")
else:
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="uzn_Latn", split="train", streaming=True)
    docs, w = [], 0
    for ex in ds:
        t = (ex.get("text") or "").strip()
        if not t: continue
        docs.append(t); w += len(t.split())
        if w >= EVAL_WORDS: break
    OLD_TEXT = "\n".join(docs); open(OLD_FILE, "w", encoding="utf-8").write(OLD_TEXT)
print(f"old eval set: {len(OLD_TEXT.split()):,} words")

# ================== STAGE 5 — adaptation pool + leak check ==================
stage(5, "adaptation pool")
POOL_FILE = f"{WORK}/adapt_dedup_{ADAPT_TOKENS}.npy"
tok_m = AutoTokenizer.from_pretrained(BASE)
if tok_m.pad_token is None: tok_m.pad_token = tok_m.eos_token
tok_m.model_max_length = 10**9

if os.path.exists(POOL_FILE):
    POOL = np.load(POOL_FILE); pool_text = ""
    print(f"reusing pool: {len(POOL)/1e6:.2f}M tokens")
else:
    need = int(ADAPT_TOKENS * 1.10)
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="uzn_Latn", split="train", streaming=True)
    ids, chunks = [], []
    for ex in ds:
        t = (ex.get("text") or "").strip()
        if not t: continue
        chunks.append(t)
        ids.extend(tok_m(t, add_special_tokens=False)["input_ids"] + [tok_m.eos_token_id])
        if len(ids) >= need: break
    POOL = np.asarray(ids[:need], dtype=np.int32); np.save(POOL_FILE, POOL)
    pool_text = "\n".join(chunks)
    print(f"tokenized {len(POOL)/1e6:.2f}M tokens from {len(chunks):,} docs")

if pool_text:
    probes = [TEXT[i:i+300] for i in np.linspace(0, max(len(TEXT)-400, 1), 12).astype(int)]
    leak = sum(1 for p in probes if p in pool_text)
    print(f"eval-text probes inside the adaptation pool: {leak}/{len(probes)}")
    R["adapt_pool_leak"] = f"{leak}/{len(probes)}"

# ======================= STAGE 6 — QLoRA fine-tune ==========================
stage(6, f"QLoRA mGPT-1.3B (ctx {CTX_TRAIN}, all-linear r=16, ~{ADAPT_TOKENS/1e6:.0f}M tokens)")
if os.path.exists(f"{ADAPTER_DIR}/adapter_model.safetensors"):
    print(f"reusing adapter: {ADAPTER_DIR}")
else:
    class Blocks(Dataset):
        def __init__(s, a, c): s.a, s.c, s.n = a, c, len(a)//c
        def __len__(s): return s.n
        def __getitem__(s, i):
            x = s.a[i*s.c:(i+1)*s.c].astype(np.int64)
            return {"input_ids": torch.from_numpy(x), "labels": torch.from_numpy(x.copy())}
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=DT, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
    m.config.use_cache = False
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                     task_type="CAUSAL_LM", target_modules="all-linear"))
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"trainable: {trainable:,}"); R["trainable_params"] = trainable
    steps = ADAPT_TOKENS // (PER_DEV * ACC * CTX_TRAIN)
    print(f"steps: {steps} x {PER_DEV*ACC*CTX_TRAIN} = {steps*PER_DEV*ACC*CTX_TRAIN:,} tokens")
    args = TrainingArguments(
        output_dir=ADAPTER_DIR, per_device_train_batch_size=PER_DEV,
        gradient_accumulation_steps=ACC, max_steps=steps, learning_rate=LR,
        lr_scheduler_type="cosine", warmup_steps=max(10, steps//20),
        logging_steps=50, save_strategy="no", report_to=[],
        fp16=not BF16, bf16=BF16, gradient_checkpointing=True,
        optim="paged_adamw_8bit", seed=SEED)
    t0 = time.time()
    Trainer(model=m, args=args, train_dataset=Blocks(POOL, CTX_TRAIN),
            data_collator=default_data_collator).train()
    print(f"trained in {(time.time()-t0)/60:.1f} min")
    m.save_pretrained(ADAPTER_DIR); del m; free()

# ============================ model loaders =================================
def load_from_scratch():
    mp = hf_hub_download(MY_MODEL_REPO, "model.py"); sys.path.insert(0, os.path.dirname(mp))
    from model import GPT
    g = GPT(vocab_size=16384, n_embd=768, block_size=1024, num_heads=12, n_layers=12)
    g.load_state_dict(load_file(hf_hub_download(MY_MODEL_REPO, "model.safetensors")), strict=False)
    return g.eval().to(DEV)

def load_mgpt(adapter=None):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=DT, bnb_4bit_use_double_quant=True)
    mm = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                              device_map={"": 0}).eval()
    return PeftModel.from_pretrained(mm, adapter).eval() if adapter else mm

def logits_of(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else getattr(out, "logits", out)

# ========================= STAGE 7 — Protocol A =============================
stage(7, "Protocol A — 512-token chunks")

@torch.no_grad()
def protocol_a(tok, model, text, tag):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    n = len(ids) // CHUNK_TOKENS
    C = torch.tensor(ids[:n*CHUNK_TOKENS], dtype=torch.long).view(n, CHUNK_TOKENS)
    nats, byts, ntok = 0.0, 0, 0
    for i in range(n):
        x = C[i:i+1].to(DEV)
        lg = logits_of(model, x); sl, st = lg[:, :-1, :].float(), x[:, 1:]
        nats += torch.nn.functional.cross_entropy(
            sl.reshape(-1, sl.size(-1)), st.reshape(-1), reduction="sum").item()
        byts += len(tok.decode(st[0].tolist()).encode("utf-8")); ntok += st.numel()
    out = {"bpb": nats/(LN2*byts), "ce": nats/ntok, "chunks": n, "bytes": byts}
    print(f"  [{tag}] bpb={out['bpb']:.4f}  CE={out['ce']:.4f}  chunks={n}")
    return out

# ========================= STAGE 8 — Protocol B =============================
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
def protocol_b(tok, model, tag):
    rows, over = [], 0
    for span_id, a, b in SPANS:
        n_bytes = len(TEXT[a:b].encode("utf-8"))
        c, got = a, 0
        while c > 0 and got < CONTEXT_BYTES:
            c -= 1; got += len(TEXT[c].encode("utf-8"))
        enc = tok(TEXT[c:b], return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        first = next((k for k, (s, e) in enumerate(offs) if s >= a - c), len(ids))
        if len(ids) < 2: continue
        if len(ids) > MAX_TOKENS:
            over += 1; drop = len(ids) - MAX_TOKENS
            ids = ids[drop:]; first = max(0, first - drop)
        x = torch.tensor(ids, dtype=torch.long, device=DEV).unsqueeze(0)
        lg = logits_of(model, x)
        logits = lg[:, :-1, :].float(); target = x[:, 1:].clone()
        if first > 1: target[:, :first-1] = -100
        nats = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1),
            ignore_index=-100, reduction="sum").item()
        rows.append((span_id, nats, n_bytes))
    path = f"{WORK}/dedup_chunks_{tag}.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["chunk_id","sum_nats","n_bytes"])
        w.writerows([(i, f"{v:.6f}", n) for i, v, n in rows])
    bpb = sum(r[1] for r in rows)/(LN2*sum(r[2] for r in rows))
    print(f"  [{tag}] bpb={bpb:.4f}  spans={len(rows)}" + (f"  ({over} truncated)" if over else ""))
    return bpb

A, B = {}, {}
g = load_from_scratch()
A["from_scratch_clean"] = protocol_a(tk, g, TEXT, "from_scratch / CLEAN")
A["from_scratch_old"]   = protocol_a(tk, g, OLD_TEXT, "from_scratch / OLD (contaminated)")
stage(8, f"Protocol B — {len(SPANS)} byte-aligned spans")
B["from_scratch"] = protocol_b(tk, g, "from_scratch")
del g; free()

base = load_mgpt()
A["mgpt_base"] = protocol_a(tok_m, base, TEXT, "mgpt_base / CLEAN")
B["mgpt_base"] = protocol_b(tok_m, base, "mgpt_base")
del base; free()

q = load_mgpt(adapter=ADAPTER_DIR)
A["mgpt_qlora"] = protocol_a(tok_m, q, TEXT, "mgpt_qlora / CLEAN")
B["mgpt_qlora"] = protocol_b(tok_m, q, "mgpt_qlora")
del q; free()

R["protocol_a"], R["protocol_b"] = A, B
DELTA = A["from_scratch_clean"]["bpb"] - A["from_scratch_old"]["bpb"]
R["contamination_delta_bpb"] = DELTA

# ======================= STAGE 9 — paired bootstrap =========================
stage(9, f"paired bootstrap, {RESAMPLES:,} resamples")

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
    assert np.allclose(ba, bb), "paired spans differ in bytes"
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ids), size=(RESAMPLES, len(ids)))
    d = (nb[idx].sum(1)/(LN2*bb[idx].sum(1))) - (na[idx].sum(1)/(LN2*ba[idx].sum(1)))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": nb.sum()/(LN2*bb.sum()) - na.sum()/(LN2*ba.sum()),
            "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
            "p_le_zero": float((d <= 0).mean()), "n_spans": len(ids)}

boots = {"vs_qlora": paired("from_scratch", "mgpt_qlora"),
         "vs_base":  paired("from_scratch", "mgpt_base")}
R["bootstrap"] = boots

# ============================ STAGE 10 — results ============================
stage(10, "RESULTS")
bq, bb_ = boots["vs_qlora"], boots["vs_base"]
print(f"""
eval set            val.bin documents with zero 24-token overlap with train.bin
                    {R['clean_eval_words']:,} words, {EVAL_BYTES:,} bytes
documents           kept {R.get('docs_kept','?')}, rejected {R.get('docs_rejected','?')} as duplicates
exact-search probes {R['probes_in_train']}  (0/{N_PROBES} required)
adaptation          {ADAPT_TOKENS:,} tokens, ctx {CTX_TRAIN}, all-linear r=16

MEMORISATION EFFECT (same model, same protocol, two eval sets)
  from-scratch on OLD   (trained on it)   {A['from_scratch_old']['bpb']:.4f} bpb
  from-scratch on CLEAN (never seen)      {A['from_scratch_clean']['bpb']:.4f} bpb
  memorisation was worth                  {DELTA:+.4f} bpb

PROTOCOL A — 512-token chunks, CLEAN eval set
  model                  bpb      per-token CE
  from-scratch 103M    {A['from_scratch_clean']['bpb']:.4f}      {A['from_scratch_clean']['ce']:.4f}
  mGPT-1.3B base       {A['mgpt_base']['bpb']:.4f}      {A['mgpt_base']['ce']:.4f}
  mGPT-1.3B + QLoRA    {A['mgpt_qlora']['bpb']:.4f}      {A['mgpt_qlora']['ce']:.4f}
  paper claimed        1.1050 / 1.1628 / 1.1214  (on contaminated text)

PROTOCOL B — {len(SPANS)} byte-aligned spans, CLEAN eval set
  from-scratch 103M    {B['from_scratch']:.4f}
  mGPT-1.3B base       {B['mgpt_base']:.4f}
  mGPT-1.3B + QLoRA    {B['mgpt_qlora']:.4f}

PAIRED BOOTSTRAP ({RESAMPLES:,} resamples, seed {SEED}; positive = from-scratch better)
  vs QLoRA baseline  {bq['diff']:+.4f} bpb  95% CI [{bq['ci_lo']:+.4f}, {bq['ci_hi']:+.4f}]  excludes zero: {'YES' if bq['excludes_zero'] else 'NO'}  P(<=0)={bq['p_le_zero']:.4f}
  vs zero-shot base  {bb_['diff']:+.4f} bpb  95% CI [{bb_['ci_lo']:+.4f}, {bb_['ci_hi']:+.4f}]  excludes zero: {'YES' if bb_['excludes_zero'] else 'NO'}  P(<=0)={bb_['p_le_zero']:.4f}
""")

json.dump(R, open(f"{WORK}/results_v3.json", "w"), indent=2, default=float)
print(f"saved -> {WORK}/results_v3.json, dedup_chunks_*.csv, {CLEAN_FILE}")
