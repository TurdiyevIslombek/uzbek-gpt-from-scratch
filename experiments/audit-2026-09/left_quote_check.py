"""How much does the tokenizer's one known gap cost?

uzbek-bpe-16k normalizes the plain apostrophe ('), the right quote (’) and the
backtick (`) in oʻ / gʻ, but not the left quote (‘, U+2018). This script counts
how the mark is typed in the 474 test documents (web spelling) and how many
tokens the tokenizer would save if the left quote were treated like the plain
apostrophe. It changes nothing in the published tokenizer.

    python left_quote_check.py                      # downloads tokenizer.json
    python left_quote_check.py path/to/tokenizer.json

CPU only, a few seconds. Output: logs/left_quote_check.log
"""
import collections, json, os, sys
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(HERE, "results", "uz_heldout_dedup_raw.jsonl")

if len(sys.argv) > 1:
    tok_file = sys.argv[1]
else:
    from huggingface_hub import hf_hub_download
    tok_file = hf_hub_download("IslombekT/uzbek-bpe-16k", "tokenizer.json")
tok = Tokenizer.from_file(tok_file)

texts = [json.loads(line)["raw"] for line in open(TEST, encoding="utf-8")]
words = sum(len(t.split()) for t in texts)

marks = collections.Counter()
for t in texts:
    for i in range(1, len(t)):
        if t[i - 1] in "oOgG" and t[i] in "'‘’`ʻ´ʼ":
            marks[t[i]] += 1
total = sum(marks.values())

as_is = sum(len(tok.encode(t).ids) for t in texts)
fixed = sum(len(tok.encode(t.replace("‘", "'")).ids) for t in texts)

print(f"documents {len(texts)} | words {words:,}")
print(f"marks after o/g: {total:,}  " + ", ".join(
    f"U+{ord(k):04X} {k} {100 * v / total:.1f}%" for k, v in marks.most_common()))
print(f"tokens as published:              {as_is:,}  ({as_is / words:.4f} per word)")
print(f"tokens if ‘ were normalized too:  {fixed:,}  ({fixed / words:.4f} per word)")
print(f"saving: {100 * (1 - fixed / as_is):.2f}% of tokens")
