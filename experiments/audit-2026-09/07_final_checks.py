"""
================================================================================
07 — FINAL CHECKS BEFORE SUBMISSION
================================================================================
Three things a reviewer would ask about, answered in one run:

  A. TABLE 1, RE-MEASURED ON A DOCUMENTED SAMPLE
     Fertility (tokens per word) of every tokenizer on the 474 held-out documents in
     their original web spelling: text the tokenizer never trained on, a known size,
     and the same documents as every other result in the paper. mGPT's tokenizer is
     included this time.

  B. MODERN BASELINES
     mGPT is from 2022. Current open multilingual models (Qwen2.5, Qwen3) are scored
     zero-shot on the exact spans and text of 05, with a paired bootstrap against
     uzbek-gpt-103m. This result may be unfavourable. Report it either way.

  C. WHAT THE MODEL WRITES
     One continuation per prompt from uzbek-gpt-103m, with a fixed seed per prompt and
     no selection, so the samples in the paper are not cherry-picked.

INPUTS  Add Input -> Your Work -> the eval-text check notebook (05), with its output:
        uz_heldout_dedup.jsonl, uz_heldout_dedup_raw.jsonl, rawcheck_*.csv
SETTINGS  GPU T4 (x2 is fine: one is used) · Internet ON
FIRST, IN A SEPARATE CELL:
    !pip install -q transformers accelerate safetensors huggingface_hub tiktoken sentencepiece protobuf
RUNTIME  ~40-60 min. Every part caches to /kaggle/working; a restart skips finished work.
================================================================================
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv, gc, glob, json, math, sys, time
import numpy as np

WORK = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)
LN2 = math.log(2.0)
SPAN_BYTES, CONTEXT_BYTES, MAX_TOKENS = 800, 300, 512
RESAMPLES, SEED = 10_000, 20260913
MY_MODEL_REPO, MY_TOK_REPO = "IslombekT/uzbek-gpt-103m", "IslombekT/uzbek-bpe-16k"

TOKENIZERS = [   # (label, spec)  spec = hf:<repo> | tiktoken:<encoding>
    ("uzbek-bpe-16k (this study)", f"hf:{MY_TOK_REPO}"),
    ("XLM-RoBERTa", "hf:FacebookAI/xlm-roberta-base"),
    ("XGLM", "hf:facebook/xglm-564M"),
    ("mT5", "hf:google/mt5-base"),
    ("GPT-4o (o200k_base)", "tiktoken:o200k_base"),
    ("mBERT", "hf:google-bert/bert-base-multilingual-cased"),
    ("BLOOM", "hf:bigscience/bloom-560m"),
    ("Qwen2.5", "hf:Qwen/Qwen2.5-0.5B"),
    ("GPT-3.5/4 (cl100k_base)", "tiktoken:cl100k_base"),
    ("GPT-2", "tiktoken:gpt2"),
    ("mGPT", "hf:ai-forever/mGPT"),
]
MODERN = [       # open base models, no login needed; any that fail to load are skipped
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen3-1.7B-Base",
]
PROMPTS = [      # one sample each, seed fixed per prompt, nothing selected
    "Oʻzbekiston Respublikasi",
    "Toshkent shahrida",
    "Men oʻqishni yaxshi koʻraman, chunki",
    "Sunʼiy intellekt",
]
REFERENCE = {"uzbek-gpt-103m": 1.0380, "mGPT-1.3B + QLoRA (1M)": 1.0772, "mGPT-1.3B zero-shot": 1.0801}


def stage(n, name):
    print(f"\n{'=' * 76}\nPART {n} — {name}\n{'=' * 76}", flush=True)


# ============================ PURE FUNCTIONS ============================
def build_spans(text, span_bytes=SPAN_BYTES, context_bytes=CONTEXT_BYTES):
    """Identical to 05 and 06: byte-sized spans of the CLEAN text, [(id, c, a, b)]."""
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


def read_scores(path):
    with open(path, encoding="utf-8") as f:
        return {r["chunk_id"]: (float(r["sum_nats"]), float(r["n_bytes"])) for r in csv.DictReader(f)}


def bpb_of(scores):
    return sum(v[0] for v in scores.values()) / (LN2 * sum(v[1] for v in scores.values()))


def paired_gap(model_scores, ref_scores, resamples=RESAMPLES, seed=SEED):
    """bpb(model) - bpb(ref) over shared spans, paired 95% bootstrap CI."""
    ids = sorted(set(model_scores) & set(ref_scores))
    nm = np.array([model_scores[i][0] for i in ids]); nr = np.array([ref_scores[i][0] for i in ids])
    b = np.array([ref_scores[i][1] for i in ids])
    idx = np.random.default_rng(seed).integers(0, len(ids), size=(resamples, len(ids)))
    d = (nm[idx].sum(1) - nr[idx].sum(1)) / (LN2 * b[idx].sum(1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return (nm.sum() - nr.sum()) / (LN2 * b.sum()), lo, hi


def fertility_table(docs, counters):
    """counters: {label: (count_fn, vocab_size)} -> rows sorted by fertility."""
    words = sum(len(d.split()) for d in docs)
    nbytes = sum(len(d.encode("utf-8")) for d in docs)
    rows = []
    for label, (count, vocab) in counters.items():
        toks = sum(count(d) for d in docs)
        rows.append({"tokenizer": label, "vocab": vocab, "tokens": toks,
                     "fertility": toks / words, "bytes_per_token": nbytes / toks})
    return sorted(rows, key=lambda r: r["fertility"]), words, nbytes
# ============================ END PURE FUNCTIONS ========================


def find(pattern):
    for root in ("/kaggle/input", WORK):
        hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
        if hits:
            return hits[0]
    return None


def make_counter(spec):
    kind, name = spec.split(":", 1)
    if kind == "tiktoken":
        import tiktoken
        enc = tiktoken.get_encoding(name)
        return (lambda s: len(enc.encode_ordinary(s))), enc.n_vocab
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.model_max_length = 10 ** 9
    return (lambda s: len(tok.encode(s, add_special_tokens=False))), len(tok)


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.utils import logging as hf_logging
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    hf_logging.set_verbosity_error()

    assert torch.cuda.is_available(), "Enable a GPU: Settings -> Accelerator -> GPU T4."
    if torch.cuda.device_count() > 1:
        raise SystemExit("This kernel still sees 2 GPUs. Restart the kernel "
                         "(Run -> Restart & clear cell outputs), then run the pip cell and this cell again.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    def free():
        gc.collect(); torch.cuda.empty_cache()

    cj, rj = find("uz_heldout_dedup.jsonl"), find("uz_heldout_dedup_raw.jsonl")
    if not (cj and rj):
        raise SystemExit("Held-out files not found. Add Input -> the eval-text check notebook (05), with its output.")
    clean_docs = [json.loads(l)["text"] for l in open(cj, encoding="utf-8")]
    raw_docs = [json.loads(l)["raw"] for l in open(rj, encoding="utf-8")]
    use = [i for i, r in enumerate(raw_docs) if r is not None and len(r) == len(clean_docs[i])]
    RAW_DOCS = [raw_docs[i] for i in use]
    CLEAN = "\n".join(clean_docs[i] for i in use)
    RAW = "\n".join(RAW_DOCS)
    SPANS = build_spans(CLEAN)
    WORDS = len(RAW[SPANS[0][2]:SPANS[-1][3]].split())
    print(f"{len(use)} documents | {len(SPANS)} spans | {len(RAW.encode()):,} bytes of web-spelled text")

    ref_path = find("rawcheck_from_scratch_raw.csv")
    REF = read_scores(ref_path) if ref_path else None
    if REF:
        assert sorted(REF) == sorted(s[0] for s in SPANS), "05's spans do not match — wrong input notebook?"

    # ------------------------------------------------------------------ A
    stage("A", "Table 1 on the 474 held-out documents, web spelling")
    fert_file = f"{WORK}/table1_web.json"
    if os.path.exists(fert_file):
        T1 = json.load(open(fert_file)); print("cached")
    else:
        counters = {}
        for label, spec in TOKENIZERS:
            try:
                counters[label] = make_counter(spec)
            except Exception as e:
                print(f"  skipped {label}: {type(e).__name__}: {str(e)[:100]}")
        rows, words, nbytes = fertility_table(RAW_DOCS, counters)
        T1 = {"rows": rows, "words": words, "bytes": nbytes, "docs": len(RAW_DOCS)}
        json.dump(T1, open(fert_file, "w"), indent=2)
    for r in T1["rows"]:
        print(f"  {r['tokenizer']:<28} vocab {r['vocab']:>8,}   fertility {r['fertility']:.3f}   "
              f"bytes/token {r['bytes_per_token']:.3f}")

    # ------------------------------------------------------------------ B
    stage("B", "modern multilingual baselines, zero-shot, web spelling, the spans of 05")

    def logits_of(model, x):
        out = model(x)
        return out[0] if isinstance(out, tuple) else getattr(out, "logits", out)

    @torch.no_grad()
    def protocol_b(model, tok, text):
        rows, bad = [], 0
        for sid, c, a, b in SPANS:
            n_bytes = len(text[a:b].encode("utf-8"))
            enc = tok(text[c:b], return_offsets_mapping=True, add_special_tokens=False)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            first = next((k for k, (s, e) in enumerate(offs) if s >= a - c), len(ids))
            if len(ids) < 2:
                continue
            if len(ids) > MAX_TOKENS:
                drop = len(ids) - MAX_TOKENS; ids = ids[drop:]; first = max(0, first - drop)
            x = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)
            logits = logits_of(model, x)[:, :-1, :].float(); target = x[:, 1:].clone()
            if first > 1:
                target[:, :first - 1] = -100
            nats = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                ignore_index=-100, reduction="sum").item()
            if not math.isfinite(nats):
                bad += 1; continue
            rows.append((sid, nats, n_bytes))
        return rows, bad

    MOD = {}
    for name in MODERN:
        path = f"{WORK}/modern_{name.split('/')[-1]}_raw.csv"
        if not os.path.exists(path):
            try:
                t0 = time.time()
                tok = AutoTokenizer.from_pretrained(name); tok.model_max_length = 10 ** 9
                model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to("cuda").eval()
                n_par = sum(p.numel() for p in model.parameters())
                rows, bad = protocol_b(model, tok, RAW)
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f); w.writerow(["chunk_id", "sum_nats", "n_bytes"])
                    w.writerows([(i, f"{v:.6f}", n) for i, v, n in rows])
                json.dump({"params": n_par, "nonfinite_spans": bad}, open(path + ".meta.json", "w"))
                print(f"  {name:<24} {n_par / 1e9:.2f}B params | {(time.time() - t0) / 60:.1f} min"
                      + (f" | {bad} non-finite spans dropped" if bad else ""), flush=True)
                del model; free()
            except Exception as e:
                print(f"  skipped {name}: {type(e).__name__}: {str(e)[:120]}")
                free(); continue
        sc = read_scores(path)
        meta = json.load(open(path + ".meta.json"))
        entry = {"params": meta["params"], "bpb": bpb_of(sc),
                 "bits_per_word": sum(v[0] for v in sc.values()) / LN2 / WORDS, "spans": len(sc)}
        if REF:
            entry["gap"] = list(paired_gap(sc, REF))       # positive = uzbek-gpt-103m better
        MOD[name] = entry

    # ------------------------------------------------------------------ C
    stage("C", "what uzbek-gpt-103m writes (one sample per prompt, fixed seeds, no selection)")
    samples_file = f"{WORK}/samples.json"
    if os.path.exists(samples_file):
        SAMPLES = json.load(open(samples_file)); print("cached")
    else:
        mp = hf_hub_download(MY_MODEL_REPO, "model.py"); sys.path.insert(0, os.path.dirname(mp))
        from model import GPT
        g = GPT(vocab_size=16384, n_embd=768, block_size=1024, num_heads=12, n_layers=12)
        g.load_state_dict(load_file(hf_hub_download(MY_MODEL_REPO, "model.safetensors")), strict=False)
        g = g.eval().to("cuda")
        tk = AutoTokenizer.from_pretrained(MY_TOK_REPO)

        @torch.no_grad()
        def sample(prompt, seed, n_new=70, temp=0.8, top_p=0.9, rep=1.2):
            gen = torch.Generator(device="cuda").manual_seed(seed)
            ids = tk(prompt, add_special_tokens=False)["input_ids"]
            x = torch.tensor([ids], dtype=torch.long, device="cuda")
            for _ in range(n_new):
                lg = logits_of(g, x[:, -1024:])[0, -1, :].float()
                seen = torch.unique(x[0])
                lg[seen] = torch.where(lg[seen] > 0, lg[seen] / rep, lg[seen] * rep)
                probs = torch.softmax(lg / temp, dim=-1)
                sp, si = torch.sort(probs, descending=True)
                sp[(sp.cumsum(0) - sp) > top_p] = 0
                nxt = si[torch.multinomial(sp / sp.sum(), 1, generator=gen)]
                if int(nxt) == 0:                      # <|endoftext|>
                    break
                x = torch.cat([x, nxt.view(1, 1)], dim=1)
            return tk.decode(x[0, len(ids):].tolist())

        SAMPLES = [{"prompt": p, "seed": 1000 + k, "continuation": sample(p, 1000 + k)}
                   for k, p in enumerate(PROMPTS)]
        json.dump(SAMPLES, open(samples_file, "w"), ensure_ascii=False, indent=2)
        del g; free()

    # ------------------------------------------------------------------ results
    print(f"\n{'=' * 76}\nRESULTS\n{'=' * 76}")
    print(f"\nA. TABLE 1 — {T1['docs']} documents, {T1['words']:,} words, web spelling")
    for r in T1["rows"]:
        print(f"   {r['tokenizer']:<28}{r['vocab']:>9,}{r['fertility']:>9.3f}{r['bytes_per_token']:>9.3f}")
    print(f"\nB. BITS-PER-BYTE, web spelling, same spans (positive gap = uzbek-gpt-103m better)")
    for k, v in REFERENCE.items():
        print(f"   {k:<28}{'':>8}{v:>9.4f}")
    for name, e in MOD.items():
        gap = (f"{e['gap'][0]:+.4f} [{e['gap'][1]:+.4f}, {e['gap'][2]:+.4f}]" if "gap" in e else "")
        print(f"   {name:<28}{e['params'] / 1e9:>7.2f}B{e['bpb']:>9.4f}   {gap}")
    print("   (these models' training data may include some of the same web pages; that cannot be checked)")
    print(f"\nC. SAMPLES")
    for s in SAMPLES:
        print(f"   [{s['prompt']}] {s['continuation']}\n")
    json.dump({"table1": T1, "modern": MOD, "samples": SAMPLES, "reference": REFERENCE},
              open(f"{WORK}/final_checks.json", "w"), ensure_ascii=False, indent=2, default=float)
    print(f"saved -> {WORK}/final_checks.json, table1_web.json, samples.json, modern_*_raw.csv")


def _in_notebook():
    try:
        get_ipython  # noqa: F821
        return True
    except NameError:
        return False


if __name__ == "__main__" or _in_notebook():
    main()
