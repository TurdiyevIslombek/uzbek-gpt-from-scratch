"""
================================================================================
CONTAMINATION CHECK — is the eval set inside the from-scratch model's TRAINING data?
================================================================================
Runs in ~2 minutes. No GPU needed. Run this before 02_clean_eval_and_score.py.

THE SUSPICION
  tokenize_data.py splits FineWeb-2 uzn_Latn 90/10 into train.bin / val.bin.
  A positional 90/10 split means train.bin = the FIRST 90% of the document stream.
  uz_heldout.txt was built by streaming FineWeb-2 from the START and taking the
  first ~200k words. If both are true, the "held-out" eval set sits inside
  train.bin, and uzbek-gpt-103m was trained on its own test set for ~2 epochs.

  That would make 1.105 bpb too low, in the direction that favours the paper's
  headline. It has to be checked before anything else is worth running.

WHAT IT DOES
  Takes several probe passages from the eval text, tokenizes them with
  uzbek-bpe-16k, and searches train.bin and val.bin for the exact token sequence.
  Exact-match search on token ids — no false positives.

ATTACH THE DATASET FIRST
  Notebook sidebar -> Add Input -> your dataset `uzbek-fineweb2-tokens-16k`.
  It mounts at /kaggle/input/uzbek-fineweb2-tokens-16k/
================================================================================
"""

import os, glob
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

TOK_REPO   = "IslombekT/uzbek-bpe-16k"
EVAL_FILE  = "/kaggle/working/uz_heldout.txt"
N_PROBES   = 12          # passages to test
PROBE_TOK  = 32          # tokens per probe; 32 is far beyond chance
EVAL_WORDS = 200_000

# ---- locate the bins -------------------------------------------------------
cands = glob.glob("/kaggle/input/**/train.bin", recursive=True)
if not cands:
    raise SystemExit("train.bin not found. Add Input -> uzbek-fineweb2-tokens-16k")
TRAIN_BIN = cands[0]
VAL_BIN   = os.path.join(os.path.dirname(TRAIN_BIN), "val.bin")
print(f"train.bin: {TRAIN_BIN}")
print(f"val.bin  : {VAL_BIN} ({'found' if os.path.exists(VAL_BIN) else 'MISSING'})")

train = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
val   = np.memmap(VAL_BIN, dtype=np.uint16, mode="r") if os.path.exists(VAL_BIN) else None
print(f"train tokens: {len(train):,}   val tokens: {len(val):,}" if val is not None
      else f"train tokens: {len(train):,}")

# ---- get the eval text -----------------------------------------------------
if os.path.exists(EVAL_FILE):
    TEXT = open(EVAL_FILE, encoding="utf-8").read()
    print(f"using existing {EVAL_FILE}")
else:
    print("rebuilding the eval text from the FineWeb-2 stream (same recipe as before)...")
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="uzn_Latn",
                      split="train", streaming=True)
    docs, w = [], 0
    for ex in ds:
        t = (ex.get("text") or "").strip()
        if not t:
            continue
        docs.append(t); w += len(t.split())
        if w >= EVAL_WORDS:
            break
    TEXT = "\n".join(docs)
print(f"eval text: {len(TEXT):,} chars, {len(TEXT.split()):,} words")

tok = AutoTokenizer.from_pretrained(TOK_REPO)
tok.model_max_length = 10**9
eval_ids = np.asarray(tok(TEXT, add_special_tokens=False)["input_ids"], dtype=np.uint16)
print(f"eval text -> {len(eval_ids):,} tokens under uzbek-bpe-16k\n")


def find_sequence(hay, needle):
    """Exact search for a token sequence. Vectorized progressive filter."""
    L = len(needle)
    if L == 0 or len(hay) < L:
        return []
    cand = np.flatnonzero(hay[:len(hay) - L + 1] == needle[0])
    for k in range(1, L):
        if cand.size == 0:
            break
        cand = cand[hay[cand + k] == needle[k]]
    return cand.tolist()


# ---- probe -----------------------------------------------------------------
positions = np.linspace(0, len(eval_ids) - PROBE_TOK - 1, N_PROBES).astype(int)
hits_train, hits_val = 0, 0

print(f"{'probe':>6}  {'eval token pos':>14}  {'in train.bin':>13}  {'in val.bin':>11}")
print("-" * 52)
for n, p in enumerate(positions, 1):
    needle = np.asarray(eval_ids[p:p + PROBE_TOK], dtype=np.uint16)
    t_hit = find_sequence(train, needle)
    v_hit = find_sequence(val, needle) if val is not None else []
    hits_train += bool(t_hit)
    hits_val += bool(v_hit)
    print(f"{n:>6}  {p:>14,}  {('YES @'+format(t_hit[0],',')) if t_hit else 'no':>13}  "
          f"{('YES @'+format(v_hit[0],',')) if v_hit else 'no':>11}")

print("-" * 52)
print(f"\n{hits_train}/{N_PROBES} probes found in train.bin")
print(f"{hits_val}/{N_PROBES} probes found in val.bin")

print()
if hits_train > 0:
    print("=" * 72)
    print("CONTAMINATED. The eval set is inside the from-scratch model's training")
    print("data. uzbek-gpt-103m saw this text for ~2 epochs, so 1.105 bpb is too")
    print("low, in the direction that flatters the paper's headline. Every bpb")
    print("number measured on this eval set is affected, including the ablation.")
    print("Do not evaluate against this eval set.")
    print("=" * 72)
elif hits_val > 0:
    print("Eval text sits in val.bin — genuinely held out from the from-scratch")
    print("model. The published numbers stand on this point.")
else:
    print("Not found in either bin. Either the probes crossed a document boundary")
    print("that got an <|endoftext|> inserted, or the eval text came from a")
    print("different stream position than these bins. Report this output as-is.")
