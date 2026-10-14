"""
09 — DOCUMENT-LEVEL (CLUSTER) BOOTSTRAP FOR EVERY REPORTED GAP
================================================================================
The earlier scripts resample the 2,058 scoring spans independently. Spans from
the same document are correlated, so that can make the intervals too narrow.
This script resamples the 474 DOCUMENTS instead: each bootstrap sample draws
474 documents with replacement and keeps all of their spans.

A span belongs to the document that contains its midpoint (spans are cut from
the documents joined by newlines, so a few spans cross a document boundary).

Inputs (all in results/): uz_heldout_dedup.jsonl, uz_heldout_dedup_raw.jsonl and
the per-span score files. CPU only, about a minute.

    python 09_document_bootstrap.py [results_dir]
================================================================================
"""
import bisect, csv, json, math, os, sys
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LN2, RESAMPLES, SEED = math.log(2.0), 10_000, 20260913
SPAN_BYTES, CONTEXT_BYTES = 800, 300


def build_spans(text):                       # identical to scripts 05-08k
    spans, i, n, k = [], 0, len(text), 0
    while i < n:
        j, b = i, 0
        while j < n and b < SPAN_BYTES:
            b += len(text[j].encode("utf-8")); j += 1
        spans.append((f"span_{k:05d}", i, j)); i, k = j, k + 1
    if spans and len(text[spans[-1][1]:spans[-1][2]].encode("utf-8")) < SPAN_BYTES // 2:
        spans.pop()
    return spans


def read(name):
    with open(os.path.join(R, name), encoding="utf-8") as f:
        return {r["chunk_id"]: (float(r["sum_nats"]), float(r["n_bytes"])) for r in csv.DictReader(f)}


clean = [json.loads(l)["text"] for l in open(os.path.join(R, "uz_heldout_dedup.jsonl"), encoding="utf-8")]
raw = [json.loads(l)["raw"] for l in open(os.path.join(R, "uz_heldout_dedup_raw.jsonl"), encoding="utf-8")]
use = [i for i, r in enumerate(raw) if r is not None and len(r) == len(clean[i])]
assert len(use) == 474, len(use)
text = "\n".join(clean[i] for i in use)
starts, pos = [], 0
for i in use:
    starts.append(pos); pos += len(clean[i]) + 1
spans = build_spans(text)
assert len(spans) == 2058, len(spans)
doc_of = {sid: bisect.bisect_right(starts, (a + b) // 2) - 1 for sid, a, b in spans}
D = len(use)
crossing = sum(1 for sid, a, b in spans if bisect.bisect_right(starts, a) != bisect.bisect_right(starts, b - 1))
print(f"documents {D} | spans {len(spans)} | spans crossing a document boundary {crossing} | "
      f"documents with at least one span {len(set(doc_of.values()))}")

rng = np.random.default_rng(SEED)
DRAWS = rng.integers(0, D, size=(RESAMPLES, D))
COUNTS = np.stack([np.bincount(d, minlength=D) for d in DRAWS])          # how often each doc is drawn
SPAN_IDX = np.random.default_rng(SEED).integers(0, len(spans), size=(RESAMPLES, len(spans)))


def gaps(model_file, ref_file):
    m, r = read(model_file), read(ref_file)
    ids = [s for s, _, _ in spans]
    nm = np.array([m[i][0] for i in ids]); nr = np.array([r[i][0] for i in ids]); b = np.array([r[i][1] for i in ids])
    point = (nm.sum() - nr.sum()) / (LN2 * b.sum())
    # span-level (as published before)
    ds = (nm[SPAN_IDX].sum(1) - nr[SPAN_IDX].sum(1)) / (LN2 * b[SPAN_IDX].sum(1))
    # document-level
    doc = np.array([doc_of[i] for i in ids])
    agg = lambda v: np.bincount(doc, weights=v, minlength=D)
    Nm, Nr, B = agg(nm), agg(nr), agg(b)
    dd = (COUNTS @ Nm - COUNTS @ Nr) / (LN2 * (COUNTS @ B))
    return point, np.percentile(ds, [2.5, 97.5]), np.percentile(dd, [2.5, 97.5]), nm.sum() / (LN2 * b.sum())


COMPARISONS = [
    ("mGPT-1.3B + QLoRA 1M", "web", "rawcheck_mgpt_qlora_raw.csv", "rawcheck_from_scratch_raw.csv"),
    ("mGPT-1.3B zero-shot", "web", "rawcheck_mgpt_base_raw.csv", "rawcheck_from_scratch_raw.csv"),
    ("mGPT-1.3B + QLoRA 10M", "web", "sweep_10000000_raw.csv", "rawcheck_from_scratch_raw.csv"),
    ("Ablation, clean retrain", "web", "rawcheck_ablation_clean_raw.csv", "rawcheck_from_scratch_raw.csv"),
    ("Qwen3-1.7B-Base", "web", "modern_Qwen3-1.7B-Base_raw.csv", "rawcheck_from_scratch_raw.csv"),
    ("Qwen3-0.6B-Base", "web", "modern_Qwen3-0.6B-Base_raw.csv", "rawcheck_from_scratch_raw.csv"),
    ("Qwen2.5-1.5B", "web", "modern_Qwen2.5-1.5B_raw.csv", "rawcheck_from_scratch_raw.csv"),
    ("Qwen2.5-0.5B", "web", "modern_Qwen2.5-0.5B_raw.csv", "rawcheck_from_scratch_raw.csv"),
    ("mGPT-1.3B + QLoRA 1M", "normalized", "rawcheck_mgpt_qlora_clean.csv", "rawcheck_from_scratch_clean.csv"),
    ("mGPT-1.3B zero-shot", "normalized", "rawcheck_mgpt_base_clean.csv", "rawcheck_from_scratch_clean.csv"),
    ("Ablation, clean retrain", "normalized", "rawcheck_ablation_clean_clean.csv", "rawcheck_from_scratch_clean.csv"),
]
print(f"\n{'gap to uzbek-gpt-103m (model minus mine)':<42}{'spelling':<12}{'bpb':>8}{'gap':>9}"
      f"   {'span-level 95% CI':<20}{'document-level 95% CI':<22}{'width x'}")
out = {}
for name, sp, mf, rf in COMPARISONS:
    p, s, d, bpb = gaps(mf, rf)
    w = (d[1] - d[0]) / (s[1] - s[0])
    print(f"{name:<42}{sp:<12}{bpb:>8.4f}{p:>+9.4f}   [{s[0]:+.4f}, {s[1]:+.4f}]   [{d[0]:+.4f}, {d[1]:+.4f}]   {w:.2f}")
    out[f"{name} / {sp}"] = {"bpb": bpb, "gap": p, "ci_span": list(s), "ci_document": list(d),
                              "document_ci_excludes_zero": bool(d[0] > 0 or d[1] < 0)}
json.dump(out, open(os.path.join(R, "document_bootstrap.json"), "w"), indent=2)
print(f"\nsaved -> {os.path.join(R, 'document_bootstrap.json')}")
