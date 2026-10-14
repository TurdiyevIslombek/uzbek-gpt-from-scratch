"""
================================================================================
06 — ADAPTATION SWEEP ON WEB-SPELLED TEXT (final Figure 2)
================================================================================
05_eval_text_check.py showed the paper's gaps shrink when the held-out text is
scored in its original web spelling instead of uzbek-bpe-16k's normalized spelling.
Figure 2 (the 0 / 1M / 10M adaptation sweep) was scored on the normalized text only.
This re-scores the sweep's adapters on BOTH spellings, with the exact spans of 05, so
Figure 2 and the §4.3 extrapolation can be reported on web-spelled text.

INPUTS  (Add Input -> Your Work -> Notebooks; tick "notebook output")
  1. the eval-text check notebook (05)  -> uz_heldout_dedup.jsonl,
     uz_heldout_dedup_raw.jsonl, rawcheck_*.csv                        REQUIRED
  2. the adaptation sweep notebook (04, "audit-2026-09-B")
     -> sweep_adapter_1000000/, sweep_adapter_10000000/                 OPTIONAL
     Found: ~40 min, nothing retrained, and the clean-text scores reproduce
            Figure 2 exactly (1.0757 / 1.0591) as a check.
     Missing: both adapters are retrained with the sweep's exact settings
            (~3.5 hours).
SETTINGS  GPU T4 (x2 is fine: one is used) · Internet ON
FIRST, IN A SEPARATE CELL:
    !pip install -q transformers datasets peft bitsandbytes accelerate safetensors huggingface_hub matplotlib
Every stage caches to /kaggle/working; a restart skips finished work.
================================================================================
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # Kaggle T4 x2: use ONE GPU (4-bit models cannot be split)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv, gc, glob, json, math, time
import numpy as np

WORK = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)
LN2 = math.log(2.0)
BASE = "ai-forever/mGPT"
BUDGETS = [1_000_000, 10_000_000]
SPAN_BYTES, CONTEXT_BYTES, MAX_TOKENS = 800, 300, 512
CTX_TRAIN, PER_DEV, ACC, LR, SEED, RESAMPLES = 512, 2, 2, 2e-4, 20260913, 10_000
FIG2_CLEAN = {1_000_000: 1.0757, 10_000_000: 1.0591}        # published Figure 2 (normalized text)
KNOWN = {("from_scratch", "clean"): 1.0281, ("from_scratch", "raw"): 1.0380,
         ("mgpt_base", "clean"): 1.0774, ("mgpt_base", "raw"): 1.0801}   # from 05


def stage(n, name):
    print(f"\n{'=' * 76}\nSTAGE {n} — {name}\n{'=' * 76}", flush=True)


# ============================ PURE FUNCTIONS ============================
def build_spans(text, span_bytes=SPAN_BYTES, context_bytes=CONTEXT_BYTES):
    """Identical to 05: byte-sized spans on the CLEAN text, [(id, c, a, b)] in characters."""
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
        rows = list(csv.DictReader(f))
    return {r["chunk_id"]: (float(r["sum_nats"]), float(r["n_bytes"])) for r in rows}


def bpb_of(scores):
    n = sum(v[0] for v in scores.values()); b = sum(v[1] for v in scores.values())
    return n / (LN2 * b)


def paired_gap(model_scores, ref_scores, resamples=RESAMPLES, seed=SEED):
    """bpb(model) - bpb(ref) over shared spans, with a paired 95% bootstrap CI."""
    ids = sorted(set(model_scores) & set(ref_scores))
    nm = np.array([model_scores[i][0] for i in ids]); nr = np.array([ref_scores[i][0] for i in ids])
    b = np.array([ref_scores[i][1] for i in ids])
    idx = np.random.default_rng(seed).integers(0, len(ids), size=(resamples, len(ids)))
    d = (nm[idx].sum(1) - nr[idx].sum(1)) / (LN2 * b[idx].sum(1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return (nm.sum() - nr.sum()) / (LN2 * b.sum()), lo, hi


def extrapolate(b1, b10, target):
    """Log-linear in tokens through the 1M and 10M points: tokens needed to reach `target`."""
    slope = b10 - b1                                   # bpb per decade (negative = improving)
    if slope >= 0:
        return slope, float("inf")
    return slope, 10 ** (7 + (b10 - target) / (-slope))
# ============================ END PURE FUNCTIONS ========================


def find(pattern):
    for root in ("/kaggle/input", WORK):
        hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
        if hits:
            return hits[0]
    return None


def main():
    import torch
    from datasets import load_dataset
    from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                              TrainingArguments, Trainer, default_data_collator)
    from transformers.utils import logging as hf_logging
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from torch.utils.data import Dataset
    hf_logging.set_verbosity_error()

    assert torch.cuda.is_available(), "Enable a GPU: Settings -> Accelerator -> GPU T4."
    if torch.cuda.device_count() > 1:
        raise SystemExit("This kernel still sees 2 GPUs. Restart the kernel "
                         "(Run -> Restart & clear cell outputs), then run the pip cell and this cell again.")
    BF16 = torch.cuda.get_device_capability(0)[0] >= 8
    DT = torch.bfloat16 if BF16 else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)} | bf16: {BF16}")

    def free():
        gc.collect(); torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(BASE); tok.model_max_length = 10 ** 9
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ------------------------------------------------------------------ 1
    stage(1, "the two texts from 05 (normalized = CLEAN, web spelling = RAW)")
    cj, rj = find("uz_heldout_dedup.jsonl"), find("uz_heldout_dedup_raw.jsonl")
    if not (cj and rj):
        raise SystemExit("uz_heldout_dedup.jsonl / uz_heldout_dedup_raw.jsonl not found.\n"
                         "Add Input -> Your Work -> the eval-text check notebook (05), with its output.")
    clean_docs = [json.loads(l)["text"] for l in open(cj, encoding="utf-8")]
    raw_docs = [json.loads(l)["raw"] for l in open(rj, encoding="utf-8")]
    use = [i for i, r in enumerate(raw_docs) if r is not None and len(r) == len(clean_docs[i])]
    CLEAN = "\n".join(clean_docs[i] for i in use)
    RAW = "\n".join(raw_docs[i] for i in use)
    assert len(CLEAN) == len(RAW)
    SPANS = build_spans(CLEAN)
    TEXTS = {"clean": CLEAN, "raw": RAW}
    print(f"{len(use)} documents | {len(SPANS)} spans | clean {len(CLEAN.encode()):,} bytes | raw {len(RAW.encode()):,} bytes")

    ref = {}
    for t in TEXTS:
        p = find(f"rawcheck_from_scratch_{t}.csv")
        ref[t] = read_scores(p) if p else None
    have_ref = all(ref.values())
    print("uzbek-gpt-103m span scores from 05: " + ("found — confidence intervals will be computed" if have_ref
                                                   else "not found — point estimates only"))
    if have_ref:
        ids = [s[0] for s in SPANS]
        assert sorted(ref["clean"]) == sorted(ids), "05's spans do not match — wrong input notebook?"

    # ------------------------------------------------------------------ 2
    stage(2, "adapters from the sweep (reuse if attached, otherwise retrain)")

    def bnb():
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=DT, bnb_4bit_use_double_quant=True)

    adapters, reused = {}, {}
    for bud in BUDGETS:
        p = find(f"sweep_adapter_{bud}/adapter_model.safetensors")
        if p:
            adapters[bud], reused[bud] = os.path.dirname(p), True
            print(f"  {bud:>10,} tokens: reusing {adapters[bud]}")

    missing = [b for b in BUDGETS if b not in adapters]
    if missing:
        print(f"  retraining {len(missing)} adapter(s) with the sweep's exact settings")
        POOL_FILE = f"{WORK}/sweep_pool_{max(BUDGETS)}.npy"
        if os.path.exists(POOL_FILE):
            POOL = np.load(POOL_FILE)
        else:
            need = int(max(BUDGETS) * 1.10)
            ds = load_dataset("HuggingFaceFW/fineweb-2", name="uzn_Latn", split="train", streaming=True)
            ids_, texts = [], []
            for ex in ds:
                t = (ex.get("text") or "").strip()
                if not t:
                    continue
                texts.append(t)
                ids_.extend(tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
                if len(ids_) >= need:
                    break
            POOL = np.asarray(ids_[:need], dtype=np.int32); np.save(POOL_FILE, POOL)
            pool_text = "\n".join(texts)
            probes = [RAW[i:i + 300] for i in np.linspace(0, len(RAW) - 400, 12).astype(int)]
            leak = sum(p in pool_text for p in probes)
            print(f"  pool {len(POOL) / 1e6:.2f}M tokens | eval-text probes in pool: {leak}/12")
            if leak:
                raise SystemExit("ABORT — adaptation pool overlaps the eval text.")

        class Blocks(Dataset):
            def __init__(s, a, c): s.a, s.c, s.n = a, c, len(a) // c
            def __len__(s): return s.n
            def __getitem__(s, i):
                x = s.a[i * s.c:(i + 1) * s.c].astype(np.int64)
                return {"input_ids": torch.from_numpy(x), "labels": torch.from_numpy(x.copy())}

        for bud in missing:
            adir = f"{WORK}/sweep_adapter_{bud}"
            if os.path.exists(f"{adir}/adapter_model.safetensors"):
                adapters[bud], reused[bud] = adir, False; continue
            steps = bud // (PER_DEV * ACC * CTX_TRAIN)
            print(f"  training {bud:,} tokens = {steps:,} steps")
            m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb(), device_map={"": 0})
            m.config.use_cache = False
            m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)
            m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules="all-linear"))
            args = TrainingArguments(
                output_dir=adir, per_device_train_batch_size=PER_DEV, gradient_accumulation_steps=ACC,
                max_steps=steps, learning_rate=LR, lr_scheduler_type="cosine",
                warmup_steps=max(10, steps // 20), logging_steps=max(50, steps // 10),
                save_strategy="no", report_to=[], fp16=not BF16, bf16=BF16,
                gradient_checkpointing=True, optim="paged_adamw_8bit", seed=SEED)
            t0 = time.time()
            Trainer(model=m, args=args, train_dataset=Blocks(POOL[:int(bud * 1.05)], CTX_TRAIN),
                    data_collator=default_data_collator).train()
            print(f"  trained in {(time.time() - t0) / 60:.1f} min")
            m.save_pretrained(adir); del m; free()
            adapters[bud], reused[bud] = adir, False

    # ------------------------------------------------------------------ 3
    stage(3, "score each adapter on CLEAN and RAW (the exact spans of 05)")

    def logits_of(model, x):
        out = model(x)
        return out[0] if isinstance(out, tuple) else getattr(out, "logits", out)

    @torch.no_grad()
    def protocol_b(model, text):
        rows = []
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
            lg = logits_of(model, x)
            logits = lg[:, :-1, :].float(); target = x[:, 1:].clone()
            if first > 1:
                target[:, :first - 1] = -100
            rows.append((sid, torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                ignore_index=-100, reduction="sum").item(), n_bytes))
        return rows

    path = lambda bud, t: f"{WORK}/sweep_{bud}_{t}.csv"
    for bud in BUDGETS:
        todo = [t for t in TEXTS if not os.path.exists(path(bud, t))]
        if not todo:
            print(f"  [{bud:>10,}] cached"); continue
        t0 = time.time()
        mm = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb(), device_map={"": 0}).eval()
        mm = PeftModel.from_pretrained(mm, adapters[bud]).eval()
        for t in todo:
            rows = protocol_b(mm, TEXTS[t])
            with open(path(bud, t), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["chunk_id", "sum_nats", "n_bytes"])
                w.writerows([(i, f"{v:.6f}", n) for i, v, n in rows])
            print(f"  [{bud:>10,} / {t:5}] bpb={bpb_of(read_scores(path(bud, t))):.4f}", flush=True)
        del mm; free()
        print(f"  [{bud:>10,}] {(time.time() - t0) / 60:.1f} min")

    # ------------------------------------------------------------------ 4
    stage(4, "results")
    fs = {t: (bpb_of(ref[t]) if have_ref else KNOWN[("from_scratch", t)]) for t in TEXTS}
    base = {}
    for t in TEXTS:
        p = find(f"rawcheck_mgpt_base_{t}.csv")
        base[t] = bpb_of(read_scores(p)) if p else KNOWN[("mgpt_base", t)]
    sw = {bud: {t: read_scores(path(bud, t)) for t in TEXTS} for bud in BUDGETS}
    curve = {t: [base[t]] + [bpb_of(sw[b][t]) for b in BUDGETS] for t in TEXTS}

    lines = ["=" * 76, "RESULTS", "=" * 76,
             "adapters: " + ", ".join(f"{b:,} {'reused' if reused[b] else 'retrained'}" for b in BUDGETS), "",
             f"{'adaptation tokens':<20}{'CLEAN bpb':>12}{'RAW bpb':>12}   gap to uzbek-gpt-103m on RAW [95% CI]"]
    for k, lab in enumerate(["0 (zero-shot)", "1M", "10M"]):
        if k == 0 or not have_ref:
            g = f"{curve['raw'][k] - fs['raw']:+.4f}"
        else:
            p, lo, hi = paired_gap(sw[BUDGETS[k - 1]]["raw"], ref["raw"])
            g = f"{p:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        lines.append(f"{lab:<20}{curve['clean'][k]:>12.4f}{curve['raw'][k]:>12.4f}   {g}")
    lines.append(f"{'uzbek-gpt-103m':<20}{fs['clean']:>12.4f}{fs['raw']:>12.4f}")
    if all(reused.values()):
        rep = all(abs(curve["clean"][k + 1] - FIG2_CLEAN[b]) < 0.00015 for k, b in enumerate(BUDGETS))
        lines.append(f"\nreproduction check: CLEAN column vs published Figure 2 (1.0757 / 1.0591): "
                     f"{'MATCH' if rep else 'DIFFERS — wrong adapters?'}")
    lines.append("")
    for t in ("raw", "clean"):
        slope, need = extrapolate(curve[t][1], curve[t][2], fs[t])
        lines.append(f"{t.upper():5}: 1M -> 10M {slope:+.4f} bpb per tenfold | log-linear close of the gap at "
                     f"~{need / 1e6:,.0f}M adaptation tokens")
    lines.append("(two points, one decade: an order of magnitude, not a prediction)")
    lines.append("=" * 76)
    print("\n" + "\n".join(lines))

    # ------------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [0, 1, 2]
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=200)
    ax.plot(x, curve["raw"], "o-", color="#1f4e79", lw=2, ms=7, zorder=3,
            label="mGPT-1.3B + QLoRA, web spelling")
    ax.plot(x, curve["clean"], "o--", color="#1f4e79", lw=1.2, ms=5, alpha=0.35, zorder=2,
            label="mGPT-1.3B + QLoRA, normalized spelling")
    for xi, y in zip(x, curve["raw"]):
        ax.annotate(f"{y:.4f}", (xi, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color="#1f4e79")
    ax.axhline(fs["raw"], ls="--", color="#c0392b", lw=1.6,
               label=f"uzbek-gpt-103m, web spelling ({fs['raw']:.4f})")
    ax.axhline(fs["clean"], ls=":", color="#c0392b", lw=1.2, alpha=0.4,
               label=f"uzbek-gpt-103m, normalized spelling ({fs['clean']:.4f})")
    ax.set_xticks(x); ax.set_xticklabels(["0\n(zero-shot)", "1M", "10M"])
    ax.set_xlabel("Uzbek adaptation tokens"); ax.set_ylabel("Bits-per-byte  (lower is better)")
    ax.set_title("Adaptation budget vs. bits-per-byte, deduplicated held-out set", fontsize=11, pad=12)
    lo = min(min(curve["raw"]), min(curve["clean"]), fs["raw"], fs["clean"])
    hi = max(max(curve["raw"]), max(curve["clean"]), fs["raw"], fs["clean"])
    ax.set_ylim(lo - (hi - lo) * 0.25, hi + (hi - lo) * 0.25)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{WORK}/figure2_final.png", bbox_inches="tight")

    out = {"curve_bpb": {t: dict(zip(["0", "1M", "10M"], curve[t])) for t in TEXTS},
           "from_scratch_bpb": fs, "adapters_reused": {str(b): reused[b] for b in BUDGETS}}
    for t in TEXTS:
        slope, need = extrapolate(curve[t][1], curve[t][2], fs[t])
        out[f"extrapolation_{t}"] = {"slope_per_decade": slope, "tokens_to_close": need}
    if have_ref:
        out["gap_raw_ci"] = {str(b): list(paired_gap(sw[b]["raw"], ref["raw"])) for b in BUDGETS}
        out["gap_clean_ci"] = {str(b): list(paired_gap(sw[b]["clean"], ref["clean"])) for b in BUDGETS}
    json.dump(out, open(f"{WORK}/sweep_raw_results.json", "w"), indent=2, default=float)
    print(f"\nsaved -> {WORK}/figure2_final.png, sweep_raw_results.json, sweep_*_clean.csv, sweep_*_raw.csv")


def _in_notebook():
    try:
        get_ipython  # noqa: F821
        return True
    except NameError:
        return False


if __name__ == "__main__" or _in_notebook():
    main()
