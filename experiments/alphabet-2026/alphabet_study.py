#!/usr/bin/env python3
"""
alphabet_study.py - what Uzbekistan's alphabet amendment (sh ch oʻ gʻ -> ş ç ö ğ) does
to real Uzbek text and to tokenizers. CPU only; a few minutes on the 474-document
deduplicated held-out set from the paper.

  pip install tiktoken transformers tokenizers sentencepiece protobuf
  python build_heldout.py                        # rebuilds the 474 documents, one per line
  python alphabet_study.py --heldout uz_heldout_dedup.jsonl --out results/
  python alphabet_study.py --heldout uz_heldout_dedup.jsonl --skip-tokenizers   # audit only

  Use the .jsonl file: it keeps each of the 474 documents separate. The paper's
  uz_heldout_dedup.txt joins documents with single newlines, so their boundaries are lost.

In a Jupyter / Kaggle notebook: save uzbek_imlo.py and this file with %%writefile, then run
  !python alphabet_study.py --heldout ... --out ...
or call main(["--heldout", "...", "--out", "..."]) after importing.

Writes to --out:

  1_apostrophes.md       which characters people actually type in oʻ / gʻ
  2_conversion_audit.md  how much text changes, and where plain find-and-replace and the
                         careful converter disagree (top words first)
  2_disagreements.csv    random sample of disagreements. Fill in the `verdict` column by
                         hand: naive_wrong / careful_wrong / both_ok / unclear. That turns
                         "naive replacement breaks things" into a measured error rate.
  3_tokenizers.md/.csv   fertility and bytes/token for every tokenizer on three versions
                         of the same text:
                           raw  = the web text as it is
                           norm = same alphabet, apostrophes made consistent
                           new  = the amended alphabet (careful converter applied to norm)
                         raw -> norm isolates the apostrophe mess; norm -> new isolates the
                         new letters.

Retraining (this is the part that tests "the tokenizer is the cheap lever"):
  1. Take a ~200 MB sample of training text (decode train.bin, or re-stream FineWeb-2).
  2. python uzbek_imlo.py -i sample.txt -o sample_norm.txt --mode normalize
     python uzbek_imlo.py -i sample.txt -o sample_new.txt  --mode normalize+careful
  3. With YOUR training script and identical settings (16,384 vocab), train one tokenizer
     on sample_norm.txt and one on sample_new.txt. Train both on the same sample - do not
     compare a new-alphabet tokenizer against the original uzbek-bpe-16k, which saw far
     more text, or data size becomes a confound.
  4. Re-run with --extra "retrained-old=file:tok_norm.json" "retrained-new=file:tok_new.json"

Notes:
  * Word counts are identical in all three versions (conversion never touches
    whitespace), so a fertility difference is purely a token-count difference.
  * Byte counts are NOT identical (oʻ is 2-4 bytes, ö is 2). If you go on to score
    models on converted text, don't compare bits-per-byte across spellings; use bits
    per word.
  * Make TOKENIZERS match the exact IDs of your paper's Table 1 benchmark.
"""

import argparse
import csv
import json
import os
import random
from collections import Counter

try:
    from uzbek_imlo import APOS_CHARS, careful_convert, naive_convert, normalize_apostrophes, to_old
except ModuleNotFoundError:
    # Pasted into a notebook cell after uzbek_imlo.py's code was run in an earlier cell:
    # the functions already exist in the notebook's namespace.
    _need = ("APOS_CHARS", "careful_convert", "naive_convert", "normalize_apostrophes", "to_old")
    if not all(n in globals() for n in _need):
        raise ModuleNotFoundError(
            "uzbek_imlo.py not found. Save it next to this script first "
            "(in a notebook: put %%writefile uzbek_imlo.py at the top of its cell and run it).")

TOKENIZERS = [  # (label, spec)  spec = hf:<repo> | tiktoken:<encoding> | file:<tokenizer.json>
                #                  prefix rev: = convert ş ç ö ğ back to sh ch oʻ gʻ before tokenizing
    ("uzbek-bpe-16k", "hf:IslombekT/uzbek-bpe-16k"),
    ("uzbek-bpe-16k + new→old map", "rev:hf:IslombekT/uzbek-bpe-16k"),
    ("XLM-RoBERTa", "hf:FacebookAI/xlm-roberta-base"),
    ("XGLM", "hf:facebook/xglm-564M"),
    ("mT5", "hf:google/mt5-base"),
    ("GPT-4o o200k_base", "tiktoken:o200k_base"),
    ("mBERT", "hf:google-bert/bert-base-multilingual-cased"),
    ("BLOOM", "hf:bigscience/bloom-560m"),
    ("Qwen2.5", "hf:Qwen/Qwen2.5-0.5B"),
    ("GPT-3.5/4 cl100k_base", "tiktoken:cl100k_base"),
    ("GPT-2", "tiktoken:gpt2"),
    ("mGPT", "hf:ai-forever/mGPT"),
]

APOS_LABELS = {
    "'": "U+0027 ' (ASCII apostrophe)",
    "\u2018": "U+2018 ‘ (left single quote)",
    "\u2019": "U+2019 ’ (right single quote)",
    "\u02BB": "U+02BB ʻ (official oʻ/gʻ mark)",
    "\u02BC": "U+02BC ʼ (official tutuq mark)",
    "`": "U+0060 ` (backtick)",
    "\u00B4": "U+00B4 ´ (acute accent)",
}


DEFAULT_HELDOUT = ["uz_heldout_dedup.jsonl", "/kaggle/working/uz_heldout_dedup.jsonl",
                   "uz_heldout_dedup.txt", "/kaggle/working/uz_heldout_dedup.txt"]


def find_heldout(path):
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        return path
    for p in DEFAULT_HELDOUT:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no held-out file found. Run build_heldout.py first, or pass --heldout.")


def load_passages(path, sep):
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line)["text"] for line in f if line.strip()]
    with open(path, encoding="utf-8") as f:
        return [p for p in f.read().split(sep) if p.strip()]


def apostrophe_census(passages):
    counts = Counter()
    for p in passages:
        for a, b in zip(p, p[1:]):
            if a in "oOgG" and b in APOS_CHARS:
                counts[b] += 1
    return counts


def conversion_audit(passages, seed):
    log, top, example = Counter(), Counter(), {}
    words = changed = passages_changed = passages_disagree = 0
    rows = []
    for pid, raw in enumerate(passages):
        careful = careful_convert(raw, log)
        naive = naive_convert(raw)
        r, c, n = raw.split(), careful.split(), naive.split()
        assert len(r) == len(c) == len(n), "conversion changed whitespace"
        words += len(r)
        any_changed = False
        for j, (rw, cw, nw) in enumerate(zip(r, c, n)):
            if cw != rw:
                changed += 1
                any_changed = True
            if cw != nw:
                top[rw] += 1
                example.setdefault(rw, (nw, cw))
                rows.append({"passage": pid, "original": rw, "naive": nw, "careful": cw,
                             "context": " ".join(r[max(0, j - 6): j + 7]), "verdict": ""})
        passages_changed += any_changed
        passages_disagree += careful != naive
    random.Random(seed).shuffle(rows)
    return dict(log=log, top=top, example=example, rows=rows, words=words, changed=changed,
                passages=len(passages), passages_changed=passages_changed,
                passages_disagree=passages_disagree)


def token_counter(spec):
    if spec.startswith("rev:"):
        inner = token_counter(spec[4:])
        return lambda s: inner(to_old(s))
    kind, name = spec.split(":", 1)
    if kind == "tiktoken":
        import tiktoken
        enc = tiktoken.get_encoding(name)
        return lambda s: len(enc.encode_ordinary(s))
    if kind == "file":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(name)
        return lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    if kind == "hf":
        try:
            from transformers import AutoTokenizer
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()        # silence "sequence too long" warnings
            tok = AutoTokenizer.from_pretrained(name)
            return lambda s: len(tok.encode(s, add_special_tokens=False))
        except Exception:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_pretrained(name)
            return lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    raise ValueError(f"unknown tokenizer spec: {spec}")


def pct(new, old):
    return f"{(new - old) / old * 100:+.1f}%"


def write_census(path, census):
    total = sum(census.values()) or 1
    lines = ["# Which mark do people type in oʻ / gʻ?", "",
             "Counts every o/O/g/G followed by an apostrophe-like character.", "",
             "| Character | Count | Share |", "|---|---:|---:|"]
    for ch, k in census.most_common():
        lines.append(f"| {APOS_LABELS.get(ch, repr(ch))} | {k:,} | {k / total:.1%} |")
    if census and not (census.get("'") or census.get("\u2019") or census.get("`")):
        lines += ["", "**Warning:** no ASCII apostrophes, right quotes or backticks at all. Real web "
                  "text always has them, so this text was probably normalized before it was saved "
                  "(for example, decoded from a tokenizer that maps them to ʻ). This table then shows "
                  "which marks survived normalization, not what people type."]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def write_audit(path_md, path_csv, au, n_sample):
    w = au["words"] or 1
    lines = [
        "# Conversion audit", "",
        f"- Passages: {au['passages']:,} · words: {au['words']:,}",
        f"- Words whose spelling changes (careful converter): {au['changed']:,} ({au['changed'] / w:.1%})",
        f"- Passages with at least one changed word: {au['passages_changed']:,} "
        f"({au['passages_changed'] / max(au['passages'], 1):.1%})",
        f"- Passages where naive find-and-replace and the careful converter disagree: "
        f"{au['passages_disagree']:,} ({au['passages_disagree'] / max(au['passages'], 1):.1%})",
        f"- Word-level disagreements: {len(au['rows']):,}", "",
        "## Careful converter decisions", "", "| Decision | Count |", "|---|---:|",
    ]
    lines += [f"| {k} | {v:,} |" for k, v in au["log"].most_common()]
    lines += ["", "## Top 40 disagreements", "", "| Original | Times | Naive | Careful |", "|---|---:|---|---|"]
    for word, k in au["top"].most_common(40):
        nw, cw = au["example"][word]
        lines.append(f"| `{word}` | {k:,} | `{nw}` | `{cw}` |")
    lines += ["", f"A random sample of {min(n_sample, len(au['rows']))} disagreements is in "
              f"`{os.path.basename(path_csv)}` — label the `verdict` column."]
    open(path_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    with open(path_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["passage", "original", "naive", "careful", "context", "verdict"])
        wr.writeheader()
        wr.writerows(au["rows"][:n_sample])


def tokenizer_table(passages, specs):
    versions = {"raw": passages}
    versions["norm"] = [normalize_apostrophes(p) for p in passages]
    versions["new"] = [careful_convert(p) for p in versions["norm"]]
    words = sum(len(p.split()) for p in passages)
    nbytes = {k: sum(len(p.encode("utf-8")) for p in v) for k, v in versions.items()}
    rows = []
    for label, spec in specs:
        try:
            count = token_counter(spec)
        except Exception as e:                      # keep going if one tokenizer fails
            print(f"  skipped {label}: {e}")
            continue
        toks = {k: sum(count(p) for p in v) for k, v in versions.items()}
        rows.append({"tokenizer": label,
                     **{f"fertility_{k}": toks[k] / words for k in versions},
                     **{f"bytes_per_token_{k}": nbytes[k] / toks[k] for k in versions}})
        print(f"  {label:<24} raw {toks['raw'] / words:.3f}   norm {toks['norm'] / words:.3f}   "
              f"new {toks['new'] / words:.3f}")
    return rows, words, nbytes


def write_tokenizers(path_md, path_csv, rows, words, nbytes):
    rows = sorted(rows, key=lambda r: r["fertility_new"])
    lines = ["# Tokenizers: current vs. amended alphabet", "",
             "Fertility = tokens per word (lower is better). Same words in every column.", "",
             "| Tokenizer | Raw | Norm | New | Raw→New | Norm→New | Bytes/token (new) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['tokenizer']} | {r['fertility_raw']:.3f} | {r['fertility_norm']:.3f} | "
                     f"{r['fertility_new']:.3f} | {pct(r['fertility_new'], r['fertility_raw'])} | "
                     f"{pct(r['fertility_new'], r['fertility_norm'])} | {r['bytes_per_token_new']:.3f} |")
    lines += ["", f"Words: {words:,}. Bytes: raw {nbytes['raw']:,} · norm {nbytes['norm']:,} · "
              f"new {nbytes['new']:,}.", "",
              "raw = web text as-is · norm = same alphabet, oʻ/gʻ on U+02BB and tutuq on U+02BC · "
              "new = norm converted to Ş Ç Ö Ğ.",
              "If a tokenizer applies its own apostrophe normalization before encoding, its "
              "published fertility corresponds to the Norm column.",
              "`+ new→old map` converts ş ç ö ğ back to sh ch oʻ gʻ before tokenizing: the "
              "zero-retraining way to keep a current-alphabet tokenizer and model working."]
    open(path_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    with open(path_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def parse_specs(items):
    return [tuple(s.split("=", 1)) for s in items]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="What the Uzbek alphabet amendment does to real text and to tokenizers.")
    ap.add_argument("--heldout", help=".jsonl with a 'text' field (preferred) or .txt split by --sep; "
                                      "default: uz_heldout_dedup.jsonl if present")
    ap.add_argument("--sep", default="\\n\\n", help="passage separator for .txt files (default: blank line)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sample", type=int, default=200, help="disagreements to export for hand-labelling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-tokenizers", action="store_true", help="audit only; no downloads")
    ap.add_argument("--only", nargs="*", help="replace the tokenizer list: label=spec ...")
    ap.add_argument("--extra", nargs="*", default=[], help="add tokenizers: label=spec ...")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    heldout = find_heldout(a.heldout)
    if heldout.endswith(".txt"):
        print("note: .txt input has no document boundaries; passages are split by --sep. "
              "Prefer the .jsonl from build_heldout.py.")
    passages = load_passages(heldout, a.sep.encode().decode("unicode_escape"))
    print(f"{len(passages):,} passages loaded from {heldout}")

    census = apostrophe_census(passages)
    write_census(os.path.join(a.out, "1_apostrophes.md"), census)
    total = sum(census.values()) or 1
    print("oʻ/gʻ marks: " + ", ".join(f"{APOS_LABELS.get(c, repr(c)).split(' (')[0]} {k / total:.0%}"
                                      for c, k in census.most_common()))
    if census and not (census.get("'") or census.get("\u2019") or census.get("`")):
        print("  warning: no ASCII/right-quote/backtick marks - this text looks pre-normalized")

    au = conversion_audit(passages, a.seed)
    write_audit(os.path.join(a.out, "2_conversion_audit.md"), os.path.join(a.out, "2_disagreements.csv"),
                au, a.sample)
    print(f"words changed by the amendment: {au['changed'] / max(au['words'], 1):.1%}; "
          f"naive and careful disagree in {au['passages_disagree']:,} of {au['passages']:,} passages")

    if not a.skip_tokenizers:
        specs = parse_specs(a.only) if a.only else TOKENIZERS
        specs += parse_specs(a.extra)
        print("tokenizers (fertility):")
        rows, words, nbytes = tokenizer_table(passages, specs)
        if rows:
            write_tokenizers(os.path.join(a.out, "3_tokenizers.md"), os.path.join(a.out, "3_tokenizers.csv"),
                             rows, words, nbytes)
    print(f"done -> {a.out}/")


def _in_notebook():
    """True inside Jupyter/Kaggle, where sys.argv belongs to the kernel, not to us."""
    try:
        get_ipython  # noqa: F821  (defined by IPython)
        return True
    except NameError:
        return False


if __name__ == "__main__" and not _in_notebook():
    main()
