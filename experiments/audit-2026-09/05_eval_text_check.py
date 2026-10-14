"""
================================================================================
05 — EVAL-TEXT CHECK: does the tokenizer-normalized eval text flatter uzbek-gpt-103m?
================================================================================
THE QUESTION
  The clean held-out set was decoded from val.bin, which uzbek-bpe-16k had already
  normalized:  ’ -> '   ` -> '   o'/g' -> oʻ/gʻ   letter'letter -> ʼ
  So the eval text is written in *our tokenizer's* spelling. On the real web, 44% of
  oʻ/gʻ marks are ' ’ or ` (the rest are ‘ and ʻ). mGPT, the QLoRA baseline and the
  tokenizer-ablation model all learned from raw web text. If they handle ʻ worse than the
  marks they learned from, the paper's gaps are inflated.

WHAT THIS DOES
  1. Loads (or rebuilds) the 474 deduplicated held-out documents.
  2. Streams FineWeb-2 once, finds the ORIGINAL web text of each of those documents
     (exact match: normalizing the raw text must reproduce the eval text character for
     character), and builds the 1M-token QLoRA adaptation pool in the same pass.
  3. Builds two texts over the same documents: CLEAN (as in the paper) and RAW (as on the
     web). The normalizer maps one character to one character, so the two texts line up
     position by position, and the same spans can be scored in both.
  4. Re-trains the mGPT + QLoRA 1M baseline (same config as the paper).
  5. Scores all four models on both texts, Protocol B.
  6. Paired bootstrap: each gap on CLEAN, each gap on RAW, and the CHANGE between them.

HOW TO READ THE RESULT
  - CLEAN numbers should reproduce the paper (1.0281 / 1.0492 / 1.0774, QLoRA ~1.0764).
  - "change" = gap on RAW minus gap on CLEAN. If it is under 0.001 bits/byte (the paper's
    reported precision) or its 95% CI contains 0, the paper's numbers are robust to how
    the eval text is spelled. If it is clearly negative, the eval text was flattering
    uzbek-gpt-103m by that much. The pairing makes the CIs very tight, so the size
    threshold matters: a +0.0002 shift can be "significant" and still mean nothing.
  - bits/word is comparable ACROSS the two texts (same words); bits/byte is not (the
    marks have different byte lengths: ' 1 byte, ʻ 2 bytes, ’ 3 bytes).

RUNTIME ~80-100 min on a Kaggle T4. Every stage caches to /kaggle/working; a restart
       skips finished work.

SETTINGS  Accelerator GPU T4 (x2 is fine: the script uses one) · Internet ON · Add Input: uzbek-fineweb2-tokens-16k and
          ablation-best.
FIRST, IN A SEPARATE CELL:
    !pip install -q transformers datasets peft bitsandbytes accelerate safetensors huggingface_hub
================================================================================
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # Kaggle T4 x2: use ONE GPU (4-bit models cannot be split)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv, gc, glob, json, math, sys, time
from collections import defaultdict

import numpy as np

WORK = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)
LN2 = math.log(2.0)

BASE = "ai-forever/mGPT"
MY_MODEL_REPO = "IslombekT/uzbek-gpt-103m"
MY_TOK_REPO = "IslombekT/uzbek-bpe-16k"
ADAPTER_DIR = f"{WORK}/mgpt-uz-qlora-rawcheck"

# held-out construction: identical to 02_clean_eval_and_score.py
EOT_ID, KGRAM, KEEP_MASK, MIN_DOC_WORDS, EVAL_WORDS = 0, 24, 31, 30, 200_000
EXPECTED = {"kept": 474, "words": 200_818, "bytes": 1_646_167}

# scoring + QLoRA: identical to the paper
SPAN_BYTES, CONTEXT_BYTES, MAX_TOKENS = 800, 300, 512
ADAPT_TOKENS, CTX_TRAIN, PER_DEV, ACC, LR = 1_000_000, 512, 2, 2, 2e-4
RESAMPLES, SEED = 10_000, 20260913
NEGLIGIBLE = 0.001          # the paper reports gaps to 3 decimals
MODELS = ["from_scratch", "ablation", "mgpt_qlora", "mgpt_base"]
LABEL = {"from_scratch": "uzbek-gpt-103m", "ablation": "ablation (mGPT tok)",
         "mgpt_qlora": "mGPT-1.3B + QLoRA", "mgpt_base": "mGPT-1.3B base"}
PAPER_CLEAN = {"from_scratch": 1.0281, "ablation": 1.0492, "mgpt_base": 1.0774, "mgpt_qlora": 1.0764}

APOS = "'\u2018\u2019\u02BB\u02BC`\u00B4"


def stage(n, name):
    print(f"\n{'=' * 76}\nSTAGE {n} — {name}\n{'=' * 76}", flush=True)


# ============================ PURE FUNCTIONS ============================
_B = np.uint64(1000003)


def kgram_hashes(arr, K=KGRAM, mask=KEEP_MASK, chunk=20_000_000):
    out, n = [], len(arr)
    if n < K:
        return np.zeros(0, dtype=np.uint64)
    for s in range(0, n - K + 1, chunk):
        e = min(s + chunk, n - K + 1); m = e - s
        h = np.zeros(m, dtype=np.uint64)
        for j in range(K):
            h = h * _B + arr[s + j: s + j + m].astype(np.uint64)
        out.append(h[(h & np.uint64(mask)) == 0])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.uint64)


def build_clean_docs(train_ids, val_ids, decode):
    """The paper's deduplicated held-out set (same algorithm as build_heldout.py)."""
    idx = np.unique(kgram_hashes(np.asarray(train_ids)))

    def dup_fraction(tok_ids):
        h = kgram_hashes(np.asarray(tok_ids, dtype=np.uint16))
        if len(h) == 0:
            return 0.0, 0
        p = np.searchsorted(idx, h); p[p >= len(idx)] = 0
        return float((idx[p] == h).mean()), len(h)

    kept, words, cur, i = [], 0, [], 0
    while i < len(val_ids) and val_ids[i] != EOT_ID:
        i += 1
    i += 1
    while i < len(val_ids) and words < EVAL_WORDS:
        t = int(val_ids[i])
        if t == EOT_ID:
            if len(cur) >= MIN_DOC_WORDS:
                frac, n_fp = dup_fraction(cur)
                if n_fp > 0 and frac <= 0.0:
                    d = decode(cur).strip()
                    if d:
                        kept.append(d); words += len(d.split())
            cur = []
        else:
            cur.append(t)
        i += 1
    return kept


class RawMatcher:
    """Find the raw web original of each clean document.

    A raw document r is the original of clean document d when norm(r) == d exactly.
    Candidates are found by the first K characters (the normalizer only looks one
    character ahead, so norm(r[:K+50])[:K] == norm(r)[:K]). Duplicate clean documents
    are assigned in stream order.
    """
    K = 150
    _strip = {ord(c): None for c in APOS + " \t\r\n\u00a0"}

    def __init__(self, clean_docs, norm):
        self.docs, self.norm = clean_docs, norm
        self.raw = [None] * len(clean_docs)
        self.found = 0
        self.index, self.loose_index = defaultdict(list), defaultdict(list)
        for i, d in enumerate(clean_docs):
            self.index[d[:self.K]].append(i)
            self.loose_index[self._loose(d)[:100]].append(i)
        self.loose_hits = {}                      # i -> raw text, for diagnosis only

    def _loose(self, s):
        return s[:400].translate(self._strip)

    def offer(self, t):
        cands = self.index.get(self.norm(t[:self.K + 50])[:self.K])
        if cands:
            nt = self.norm(t)
            for i in cands:
                if self.raw[i] is None and self.docs[i] == nt:
                    self.raw[i] = t; self.found += 1
                    return True
        lc = self.loose_index.get(self._loose(t)[:100])
        if lc:
            for i in lc:
                if self.raw[i] is None and i not in self.loose_hits:
                    self.loose_hits[i] = t
                    break
        return False

    def done(self):
        return self.found == len(self.docs)


def build_spans(text, span_bytes=SPAN_BYTES, context_bytes=CONTEXT_BYTES):
    """Byte-sized spans with a run-up start c. Returns [(id, c, a, b)] in characters."""
    spans, i, n, k = [], 0, len(text), 0
    while i < n:
        j, b = i, 0
        while j < n and b < span_bytes:
            b += len(text[j].encode("utf-8")); j += 1
        spans.append((f"span_{k:05d}", i, j)); i, k = j, k + 1
    if spans and len(text[spans[-1][1]:spans[-1][2]].encode("utf-8")) < span_bytes // 2:
        spans.pop()
    out = []
    for sid, a, b in spans:
        c, got = a, 0
        while c > 0 and got < context_bytes:
            c -= 1; got += len(text[c].encode("utf-8"))
        out.append((sid, c, a, b))
    return out


def mark_census(text):
    counts = defaultdict(int)
    for a, b in zip(text, text[1:]):
        if a in "oOgG" and b in APOS:
            counts[b] += 1
    total = sum(counts.values()) or 1
    names = {"'": "'", "\u2018": "‘", "\u2019": "’", "\u02BB": "ʻ", "\u02BC": "ʼ", "`": "`", "\u00B4": "´"}
    return " ".join(f"{names[c]} {v / total:.0%}" for c, v in sorted(counts.items(), key=lambda x: -x[1]))


def paired_analysis(nats, byts, words, resamples=RESAMPLES, seed=SEED, ref="from_scratch"):
    """nats[model][text] and byts[text]: per-span arrays over the SAME spans.

    Returns point estimates and 95% CIs for bpb, gap to `ref`, and the change in gap
    between texts "raw" and "clean". One resample draw is shared by every model and both
    texts, so all differences are paired.
    """
    S = len(byts["clean"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, S, size=(resamples, S))
    bsum = {t: byts[t][idx].sum(1) for t in byts}
    boot, point = {}, {}
    for m in nats:
        for t in nats[m]:
            boot[(m, t)] = nats[m][t][idx].sum(1) / (LN2 * bsum[t])
            point[(m, t)] = nats[m][t].sum() / (LN2 * byts[t].sum())
    ci = lambda a: tuple(np.percentile(a, [2.5, 97.5]))
    res = {"bpb": {}, "bpw": {}, "gap": {}, "change": {}}
    for m in nats:
        for t in nats[m]:
            res["bpb"][(m, t)] = point[(m, t)]
            res["bpw"][(m, t)] = nats[m][t].sum() / LN2 / words[t]
    for m in nats:
        if m == ref:
            continue
        for t in ("clean", "raw"):
            g = boot[(m, t)] - boot[(ref, t)]
            res["gap"][(m, t)] = (point[(m, t)] - point[(ref, t)], *ci(g))
        ch = (boot[(m, "raw")] - boot[(ref, "raw")]) - (boot[(m, "clean")] - boot[(ref, "clean")])
        pt = (point[(m, "raw")] - point[(ref, "raw")]) - (point[(m, "clean")] - point[(ref, "clean")])
        res["change"][m] = (pt, *ci(ch))
    return res
def format_results(res, n_exact, n_total, n_used, changed, census_clean, census_raw):
    fmt = lambda g: f"{g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}]"
    lines = [
        "=" * 76, "RESULTS", "=" * 76,
        f"documents recovered   {n_exact}/{n_total} exact, {n_used} used",
        f"characters changed    {changed:,}  (CLEAN = our tokenizer's spelling, RAW = web spelling)",
        f"oʻ/gʻ marks           clean: {census_clean}",
        f"                      raw:   {census_raw}",
        "",
        f"{'model':<22}{'CLEAN bpb':>10}{'paper':>9}{'RAW bpb':>10}{'CLEAN bits/word':>17}{'RAW bits/word':>15}",
    ]
    for m in MODELS:
        lines.append(f"{LABEL[m]:<22}{res['bpb'][(m, 'clean')]:>10.4f}{PAPER_CLEAN[m]:>9.4f}"
                     f"{res['bpb'][(m, 'raw')]:>10.4f}{res['bpw'][(m, 'clean')]:>17.4f}"
                     f"{res['bpw'][(m, 'raw')]:>15.4f}")
    lines += ["", "GAP to uzbek-gpt-103m, bits/byte [95% CI]   (positive = uzbek-gpt-103m better)",
              f"{'':<22}{'CLEAN':>28}{'RAW':>28}{'CHANGE (raw - clean)':>30}"]
    for m in MODELS[1:]:
        lines.append(f"{LABEL[m]:<22}{fmt(res['gap'][(m, 'clean')]):>28}"
                     f"{fmt(res['gap'][(m, 'raw')]):>28}{fmt(res['change'][m]):>30}")
    lines += ["", f"VERDICT  (a change under {NEGLIGIBLE} bits/byte is below the paper's reported precision)"]
    for m in MODELS[1:]:
        pt, lo, hi = res["change"][m]
        clean_gap = res["gap"][(m, "clean")][0]
        rel = f"{abs(pt) / clean_gap:.0%} of the gap" if clean_gap else ""
        if (lo <= 0 <= hi) or abs(pt) < NEGLIGIBLE:
            v = "no meaningful change: robust to how the eval text is spelled"
        elif pt < 0:
            v = f"gap SHRINKS by {-pt:.4f} on web text ({rel}) — the normalized text flattered us"
        else:
            v = f"gap GROWS by {pt:.4f} on web text ({rel}) — the normalized text understated us"
        lines.append(f"  {LABEL[m]:<22}{v}")
    lines.append("=" * 76)
    return "\n".join(lines)


# ============================ END PURE FUNCTIONS ========================


def main():
    import torch
    from datasets import load_dataset
    from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                              TrainingArguments, Trainer, default_data_collator)
    from transformers.utils import logging as hf_logging
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from torch.utils.data import Dataset
    hf_logging.set_verbosity_error()

    assert torch.cuda.is_available(), "Enable a GPU: Settings -> Accelerator -> GPU T4."
    if torch.cuda.device_count() > 1:
        raise SystemExit(
            "This kernel still sees 2 GPUs, because PyTorch started before this script could pin GPU 0.\n"
            "Restart the kernel (Run -> Restart & clear cell outputs), then run the pip cell and this cell again.\n"
            "Finished stages are saved in /kaggle/working and will be reused.")
    DEV = "cuda"
    BF16 = torch.cuda.get_device_capability(0)[0] >= 8
    DT = torch.bfloat16 if BF16 else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)} | bf16: {BF16}")

    def free():
        gc.collect(); torch.cuda.empty_cache()

    tk = AutoTokenizer.from_pretrained(MY_TOK_REPO); tk.model_max_length = 10 ** 9
    NORM = tk.backend_tokenizer.normalizer.normalize_str
    tok_m = AutoTokenizer.from_pretrained(BASE); tok_m.model_max_length = 10 ** 9
    if tok_m.pad_token is None:
        tok_m.pad_token = tok_m.eos_token

    # ------------------------------------------------------------------ 1
    stage(1, "the 474 deduplicated held-out documents (as in the paper)")
    CLEAN_JSONL = f"{WORK}/uz_heldout_dedup.jsonl"
    if os.path.exists(CLEAN_JSONL):
        clean_docs = [json.loads(l)["text"] for l in open(CLEAN_JSONL, encoding="utf-8")]
        print(f"reusing {CLEAN_JSONL}")
    else:
        hits = glob.glob("/kaggle/input/**/train.bin", recursive=True)
        if not hits:
            raise SystemExit("train.bin not found. Add Input -> uzbek-fineweb2-tokens-16k")
        train_ids = np.memmap(hits[0], dtype=np.uint16, mode="r")
        val_ids = np.memmap(os.path.join(os.path.dirname(hits[0]), "val.bin"), dtype=np.uint16, mode="r")
        t0 = time.time()
        clean_docs = build_clean_docs(train_ids, val_ids, tk.decode)
        print(f"rebuilt in {time.time() - t0:.0f}s")
        with open(CLEAN_JSONL, "w", encoding="utf-8") as f:
            for d in clean_docs:
                f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
    joined = "\n".join(clean_docs)
    got = {"kept": len(clean_docs), "words": len(joined.split()), "bytes": len(joined.encode("utf-8"))}
    print(f"{got['kept']} documents | {got['words']:,} words | {got['bytes']:,} bytes")
    if got != EXPECTED:
        raise SystemExit(f"held-out set does not match the paper: {got} vs {EXPECTED}")
    print("matches the paper's held-out set exactly")

    # ------------------------------------------------------------------ 2
    stage(2, "stream FineWeb-2 once: recover the raw web text + build the adaptation pool")
    RAW_JSONL = f"{WORK}/uz_heldout_dedup_raw.jsonl"
    POOL_FILE = f"{WORK}/adapt_dedup_{ADAPT_TOKENS}.npy"
    need_pool = int(ADAPT_TOKENS * 1.10)
    pool_texts = []

    if os.path.exists(RAW_JSONL) and os.path.exists(POOL_FILE):
        rows = [json.loads(l) for l in open(RAW_JSONL, encoding="utf-8")]
        raw_docs = [r["raw"] for r in rows]
        POOL = np.load(POOL_FILE)
        print(f"reusing {RAW_JSONL} and {POOL_FILE}")
        loose_hits = {}
    else:
        matcher = RawMatcher(clean_docs, NORM)
        ds = load_dataset("HuggingFaceFW/fineweb-2", name="uzn_Latn", split="train", streaming=True)
        try:
            ds = ds.select_columns(["text"])
        except Exception:
            pass
        pool_ids, t0, seen = [], time.time(), 0
        for ex in ds:
            seen += 1
            t = (ex.get("text") or "").strip()
            if not t:
                continue
            if len(pool_ids) < need_pool:
                pool_texts.append(t)
                pool_ids.extend(tok_m(t, add_special_tokens=False)["input_ids"] + [tok_m.eos_token_id])
            matcher.offer(t)
            if seen % 100_000 == 0:
                print(f"  {seen:>9,} docs streamed | matched {matcher.found}/{len(clean_docs)} "
                      f"| {(time.time() - t0) / 60:.1f} min", flush=True)
            if matcher.done() and len(pool_ids) >= need_pool:
                break
        print(f"streamed {seen:,} documents in {(time.time() - t0) / 60:.1f} min")
        POOL = np.asarray(pool_ids[:need_pool], dtype=np.int32)
        np.save(POOL_FILE, POOL)
        raw_docs, loose_hits = matcher.raw, matcher.loose_hits
        with open(RAW_JSONL, "w", encoding="utf-8") as f:
            for i, r in enumerate(raw_docs):
                f.write(json.dumps({"doc": i, "raw": r}, ensure_ascii=False) + "\n")

    matched = [i for i, r in enumerate(raw_docs) if r is not None and len(r) == len(clean_docs[i])]
    n_exact = sum(r is not None for r in raw_docs)
    print(f"exact matches: {n_exact}/{len(clean_docs)} | character-aligned: {len(matched)}")
    missing = [i for i, r in enumerate(raw_docs) if r is None]
    if missing:
        print(f"not found: {len(missing)} documents; {len(loose_hits)} of them have a near-match")
        for i in list(loose_hits)[:3]:                       # show where they differ
            a, b = clean_docs[i], loose_hits[i]
            k = next((p for p in range(min(len(a), len(b))) if a[p] != b[p]), min(len(a), len(b)))
            print(f"  doc {i}: clean …{a[max(0, k - 30):k + 30]!r}")
            print(f"  {'':>{len(str(i)) + 4}}  raw   …{b[max(0, k - 30):k + 30]!r}")
    if len(matched) < 0.9 * len(clean_docs):
        raise SystemExit("Fewer than 90% of documents recovered. Check the near-match examples above before going on.")

    # ------------------------------------------------------------------ 3
    stage(3, "two texts over the same documents, and one set of spans for both")
    CLEAN = "\n".join(clean_docs[i] for i in matched)
    RAW = "\n".join(raw_docs[i] for i in matched)
    assert len(CLEAN) == len(RAW), "texts are not character-aligned"
    assert NORM(RAW) == CLEAN, "normalizing RAW does not reproduce CLEAN"
    SPANS = build_spans(CLEAN)
    TEXTS = {"clean": CLEAN, "raw": RAW}
    covered = slice(SPANS[0][2], SPANS[-1][3])
    WORDS = {t: len(TEXTS[t][covered].split()) for t in TEXTS}
    assert WORDS["clean"] == WORDS["raw"]
    changed = sum(1 for x, y in zip(CLEAN, RAW) if x != y)
    print(f"documents: {len(matched)} | spans: {len(SPANS)} | words: {WORDS['clean']:,}")
    print(f"bytes: clean {len(CLEAN.encode()):,} | raw {len(RAW.encode()):,}")
    print(f"characters that differ between the two texts: {changed:,}")
    print(f"oʻ/gʻ marks  clean: {mark_census(CLEAN)}")
    print(f"             raw:   {mark_census(RAW)}")
    if pool_texts:
        probes = [RAW[i:i + 300] for i in np.linspace(0, len(RAW) - 400, 12).astype(int)]
        pool_text = "\n".join(pool_texts)
        leak = sum(p in pool_text for p in probes)
        print(f"eval-text probes inside the adaptation pool: {leak}/12")
        if leak:
            raise SystemExit("ABORT — adaptation pool overlaps the eval text.")

    # ------------------------------------------------------------------ 4
    stage(4, f"QLoRA mGPT-1.3B (ctx {CTX_TRAIN}, all-linear r=16, ~{ADAPT_TOKENS / 1e6:.0f}M tokens) — as in the paper")

    def bnb():
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=DT, bnb_4bit_use_double_quant=True)

    if os.path.exists(f"{ADAPTER_DIR}/adapter_model.safetensors"):
        print(f"reusing adapter: {ADAPTER_DIR}")
    else:
        class Blocks(Dataset):
            def __init__(s, a, c): s.a, s.c, s.n = a, c, len(a) // c
            def __len__(s): return s.n
            def __getitem__(s, i):
                x = s.a[i * s.c:(i + 1) * s.c].astype(np.int64)
                return {"input_ids": torch.from_numpy(x), "labels": torch.from_numpy(x.copy())}
        m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb(), device_map={"": 0})
        m.config.use_cache = False
        m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)
        m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                         task_type="CAUSAL_LM", target_modules="all-linear"))
        steps = ADAPT_TOKENS // (PER_DEV * ACC * CTX_TRAIN)
        print(f"steps: {steps} x {PER_DEV * ACC * CTX_TRAIN} tokens")
        args = TrainingArguments(
            output_dir=ADAPTER_DIR, per_device_train_batch_size=PER_DEV,
            gradient_accumulation_steps=ACC, max_steps=steps, learning_rate=LR,
            lr_scheduler_type="cosine", warmup_steps=max(10, steps // 20),
            logging_steps=50, save_strategy="no", report_to=[],
            fp16=not BF16, bf16=BF16, gradient_checkpointing=True,
            optim="paged_adamw_8bit", seed=SEED)
        t0 = time.time()
        Trainer(model=m, args=args, train_dataset=Blocks(POOL, CTX_TRAIN),
                data_collator=default_data_collator).train()
        print(f"trained in {(time.time() - t0) / 60:.1f} min")
        m.save_pretrained(ADAPTER_DIR); del m; free()

    # ------------------------------------------------------------------ 5
    stage(5, "score all four models on CLEAN and RAW (Protocol B, identical spans)")

    def load_from_scratch():
        mp = hf_hub_download(MY_MODEL_REPO, "model.py"); sys.path.insert(0, os.path.dirname(mp))
        from model import GPT
        g = GPT(vocab_size=16384, n_embd=768, block_size=1024, num_heads=12, n_layers=12)
        g.load_state_dict(load_file(hf_hub_download(MY_MODEL_REPO, "model.safetensors")), strict=False)
        return g.eval().to(DEV)

    def load_ablation():
        hits = (glob.glob("/kaggle/input/**/ablation_best.pt", recursive=True)
                or glob.glob("/kaggle/input/**/ablation_last.pt", recursive=True))
        if not hits:
            raise SystemExit("ablation checkpoint not found. Add Input -> ablation-best")
        mp = hf_hub_download(MY_MODEL_REPO, "model.py"); sys.path.insert(0, os.path.dirname(mp))
        from model import GPT
        ck = torch.load(hits[0], map_location="cpu", weights_only=False)
        cfg = ck["config"]
        a = GPT(cfg["vocab_size"], cfg["n_embd"], cfg["block_size"], cfg["n_head"], cfg["n_layer"])
        a.load_state_dict(ck["model"])
        return a.eval().cuda()

    def load_mgpt(adapter=None):
        mm = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb(), device_map={"": 0}).eval()
        return PeftModel.from_pretrained(mm, adapter).eval() if adapter else mm

    LOADERS = {"from_scratch": (load_from_scratch, tk), "ablation": (load_ablation, tok_m),
               "mgpt_qlora": (lambda: load_mgpt(ADAPTER_DIR), tok_m), "mgpt_base": (load_mgpt, tok_m)}

    def logits_of(model, x):
        out = model(x)
        return out[0] if isinstance(out, tuple) else getattr(out, "logits", out)

    @torch.no_grad()
    def protocol_b(model, tok, text):
        rows, over = [], 0
        for sid, c, a, b in SPANS:
            n_bytes = len(text[a:b].encode("utf-8"))
            enc = tok(text[c:b], return_offsets_mapping=True, add_special_tokens=False)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            first = next((k for k, (s, e) in enumerate(offs) if s >= a - c), len(ids))
            if len(ids) < 2:
                continue
            if len(ids) > MAX_TOKENS:
                over += 1; drop = len(ids) - MAX_TOKENS
                ids = ids[drop:]; first = max(0, first - drop)
            x = torch.tensor(ids, dtype=torch.long, device=DEV).unsqueeze(0)
            lg = logits_of(model, x)
            logits = lg[:, :-1, :].float(); target = x[:, 1:].clone()
            if first > 1:
                target[:, :first - 1] = -100
            nats = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                ignore_index=-100, reduction="sum").item()
            rows.append((sid, nats, n_bytes))
        return rows, over

    csv_path = lambda m, t: f"{WORK}/rawcheck_{m}_{t}.csv"
    for m in MODELS:
        todo = [t for t in TEXTS if not os.path.exists(csv_path(m, t))]
        if not todo:
            print(f"  [{m}] cached"); continue
        load, tok = LOADERS[m]
        t0 = time.time()
        model = load()
        for t in todo:
            rows, over = protocol_b(model, tok, TEXTS[t])
            with open(csv_path(m, t), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["chunk_id", "sum_nats", "n_bytes"])
                w.writerows([(i, f"{v:.6f}", n) for i, v, n in rows])
            bpb = sum(r[1] for r in rows) / (LN2 * sum(r[2] for r in rows))
            print(f"  [{m:>12} / {t:5}] bpb={bpb:.4f}  spans={len(rows)}"
                  + (f"  ({over} truncated)" if over else ""), flush=True)
        del model; free()
        print(f"  [{m}] {(time.time() - t0) / 60:.1f} min")

    # ------------------------------------------------------------------ 6
    stage(6, f"paired bootstrap, {RESAMPLES:,} resamples")

    def load_csv(m, t):
        with open(csv_path(m, t), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return [r["chunk_id"] for r in rows], np.array([float(r["sum_nats"]) for r in rows]), \
            np.array([float(r["n_bytes"]) for r in rows])

    loaded = {(m, t): load_csv(m, t) for m in MODELS for t in TEXTS}
    common = set.intersection(*(set(v[0]) for v in loaded.values()))
    dropped = max(len(v[0]) for v in loaded.values()) - len(common)
    if dropped:
        print(f"note: {dropped} span(s) missing from some score files were left out of every comparison")
    keep = sorted(common)
    nats, byts = {m: {} for m in MODELS}, {}
    for (m, t), (ids, nt, bt) in loaded.items():
        pos = {sid: k for k, sid in enumerate(ids)}
        sel = np.array([pos[sid] for sid in keep])
        nats[m][t] = nt[sel]
        byts.setdefault(t, bt[sel])
    res = paired_analysis(nats, byts, WORDS)

    out = format_results(res, n_exact, len(clean_docs), len(matched), changed,
                         mark_census(CLEAN), mark_census(RAW))
    print("\n" + out)

    json.dump({
        "docs_exact": n_exact, "docs_used": len(matched), "chars_changed": changed, "spans": len(SPANS),
        "words": WORDS["clean"], "bytes": {t: int(byts[t].sum()) for t in byts},
        "marks": {"clean": mark_census(CLEAN), "raw": mark_census(RAW)},
        "bpb": {f"{m}/{t}": v for (m, t), v in res["bpb"].items()},
        "bits_per_word": {f"{m}/{t}": v for (m, t), v in res["bpw"].items()},
        "gap": {f"{m}/{t}": list(v) for (m, t), v in res["gap"].items()},
        "change": {m: list(v) for m, v in res["change"].items()},
        "resamples": RESAMPLES, "seed": SEED,
    }, open(f"{WORK}/rawcheck_results.json", "w"), indent=2, default=float)
    print(f"\nsaved -> {WORK}/rawcheck_results.json, rawcheck_*.csv, uz_heldout_dedup_raw.jsonl")


def _in_notebook():
    try:
        get_ipython  # noqa: F821
        return True
    except NameError:
        return False


if __name__ == "__main__" or _in_notebook():
    main()
