"""
Data preparation: tokenize the Uzbek Latin-script split of FineWeb-2 into
flat uint16 token-ID files (train.bin / val.bin) for fast memmapped training.

- One <|endoftext|> token is inserted between documents.
- uint16 storage: vocab 16,384 < 65,536, so the whole corpus is ~2GB, not 4GB.
- 90/10 train/val split.

Tokenizer: IslombekT/uzbek-bpe-16k (16,384-vocab byte-level BPE).
Run once (CPU-only, ~20-40 min for ~1B tokens).
"""

import array

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("IslombekT/uzbek-bpe-16k")
EOT = tok.convert_tokens_to_ids("<|endoftext|>")
assert isinstance(EOT, int) and 0 <= EOT < tok.vocab_size, "EOT token not found!"
print("vocab_size:", tok.vocab_size, "| EOT id:", EOT)

ds = load_dataset("HuggingFaceFW/fineweb-2", name="uzn_Latn",
                  split="train", streaming=True)

buf = array.array("H")  # uint16
docs = 0
for ex in ds:
    buf.extend(tok.encode(ex["text"], add_special_tokens=False))
    buf.append(EOT)
    docs += 1
    if docs % 100_000 == 0:
        print(f"{docs:,} docs | {len(buf) / 1e6:.0f}M tokens")

arr = np.frombuffer(buf, dtype=np.uint16)
split = int(0.9 * len(arr))
arr[:split].tofile("train.bin")
arr[split:].tofile("val.bin")
print(f"DONE — {docs:,} docs | total {len(arr) / 1e6:.0f}M tokens "
      f"| train {split / 1e6:.0f}M | val {(len(arr) - split) / 1e6:.0f}M")
