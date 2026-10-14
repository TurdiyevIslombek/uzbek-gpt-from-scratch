#!/usr/bin/env python3
"""
build_heldout.py - rebuild the paper's deduplicated held-out set in a fresh session.

Same algorithm as stages 1-2 of 02_clean_eval_and_score.py:
  1. fingerprint every 24-token window of train.bin (1 in 32 kept)
  2. walk val.bin document by document; keep only documents with ZERO fingerprint
     overlap, until 200,000 words

CPU only. About 3-5 minutes and ~4 GB of RAM on Kaggle. No GPU needed.

Writes to --out (default /kaggle/working, or the current folder outside Kaggle):
  uz_heldout_dedup.jsonl   one document per line, {"text": ...}   <- use this one
  uz_heldout_dedup.txt     documents joined by "\\n", byte-identical to the paper's file

Then checks the result against the paper: 474 documents kept, 306 rejected,
200,818 words, 1,646,167 bytes. If any number differs, the files are saved as
uz_heldout_dedup.MISMATCH.* instead, so the study cannot use them by accident.

Needs the Kaggle dataset uzbek-fineweb2-tokens-16k (train.bin, val.bin) as an input,
or --train / --val paths.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

EOT_ID = 0
KGRAM = 24
KEEP_MASK = 31
MAX_DUP_FRAC = 0.0
MIN_DOC_WORDS = 30          # compared against the token count, exactly as in the paper's script
EVAL_WORDS = 200_000
TOK_REPO = "IslombekT/uzbek-bpe-16k"
EXPECTED = {"kept": 474, "rejected": 306, "words": 200_818, "bytes": 1_646_167}

_B = np.uint64(1000003)


def kgram_hashes(arr, K=KGRAM, mask=KEEP_MASK, chunk=20_000_000):
    """Rolling polynomial hash of every K-gram, subsampled to 1/(mask+1)."""
    out, n = [], len(arr)
    if n < K:
        return np.zeros(0, dtype=np.uint64)
    for s in range(0, n - K + 1, chunk):
        e = min(s + chunk, n - K + 1)
        m = e - s
        h = np.zeros(m, dtype=np.uint64)
        for j in range(K):
            h = h * _B + arr[s + j: s + j + m].astype(np.uint64)
        out.append(h[(h & np.uint64(mask)) == 0])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.uint64)


def build(train_ids, val_ids, decode, verbose=True):
    """Return (kept_documents, n_rejected). `decode` maps a list of token ids to text."""
    t0 = time.time()
    idx = np.unique(kgram_hashes(np.asarray(train_ids)))
    if verbose:
        print(f"fingerprints: {len(idx):,} in {time.time() - t0:.0f}s")

    def dup_fraction(tok_ids):
        h = kgram_hashes(np.asarray(tok_ids, dtype=np.uint16))
        if len(h) == 0:
            return 0.0, 0
        p = np.searchsorted(idx, h)
        p[p >= len(idx)] = 0
        return float((idx[p] == h).mean()), len(h)

    kept, rejected, words, cur, i = [], 0, 0, [], 0
    while i < len(val_ids) and val_ids[i] != EOT_ID:     # start after a document boundary
        i += 1
    i += 1
    while i < len(val_ids) and words < EVAL_WORDS:
        t = int(val_ids[i])
        if t == EOT_ID:
            if len(cur) >= MIN_DOC_WORDS:
                frac, n_fp = dup_fraction(cur)
                if n_fp > 0 and frac <= MAX_DUP_FRAC:
                    d = decode(cur).strip()
                    if d:
                        kept.append(d)
                        words += len(d.split())
                else:
                    rejected += 1
            cur = []
        else:
            cur.append(t)
        i += 1
    return kept, rejected


def find_bins(train, val):
    if train and val:
        return train, val
    hits = glob.glob("/kaggle/input/**/train.bin", recursive=True) or glob.glob("**/train.bin", recursive=True)
    if not hits:
        raise FileNotFoundError("train.bin not found. On Kaggle: Add Input -> uzbek-fineweb2-tokens-16k. "
                                "Elsewhere: pass --train and --val.")
    return hits[0], os.path.join(os.path.dirname(hits[0]), "val.bin")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rebuild the paper's deduplicated held-out set.")
    ap.add_argument("--train", help="path to train.bin (default: search /kaggle/input)")
    ap.add_argument("--val", help="path to val.bin (default: next to train.bin)")
    ap.add_argument("--out", default="/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
    a = ap.parse_args(argv)

    jsonl = os.path.join(a.out, "uz_heldout_dedup.jsonl")
    if os.path.exists(jsonl):
        n = sum(1 for _ in open(jsonl, encoding="utf-8"))
        print(f"{jsonl} already exists ({n} documents) - delete it to rebuild")
        return 0

    train_path, val_path = find_bins(a.train, a.val)
    train_ids = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_ids = np.memmap(val_path, dtype=np.uint16, mode="r")
    print(f"train.bin {len(train_ids):,} tokens | val.bin {len(val_ids):,} tokens")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_REPO)
    tok.model_max_length = 10 ** 9

    kept, rejected = build(train_ids, val_ids, tok.decode)

    text = "\n".join(kept)
    got = {"kept": len(kept), "rejected": rejected, "words": len(text.split()),
           "bytes": len(text.encode("utf-8"))}
    print(f"kept {got['kept']} documents, rejected {got['rejected']} | "
          f"{got['words']:,} words | {got['bytes']:,} bytes")
    bad = {k: (got[k], EXPECTED[k]) for k in EXPECTED if got[k] != EXPECTED[k]}

    # On a mismatch, save under a different name so alphabet_study.py cannot pick it up.
    suffix = ".MISMATCH" if bad else ""
    os.makedirs(a.out, exist_ok=True)
    out_jsonl = os.path.join(a.out, f"uz_heldout_dedup{suffix}.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for d in kept:
            f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, f"uz_heldout_dedup{suffix}.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    if bad:
        print("MISMATCH with the paper's held-out set (got, expected): " + str(bad))
        print(f"Saved as {out_jsonl} so the study will not use it by mistake.")
        return 1
    print("matches the paper's held-out set exactly")
    print(f"-> {jsonl}")
    return 0


def _in_notebook():
    try:
        get_ipython  # noqa: F821
        return True
    except NameError:
        return False


if __name__ == "__main__" and not _in_notebook():
    sys.exit(main())
