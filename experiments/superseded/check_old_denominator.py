"""Why re-scoring the old test set gives 1.1078, not the reported 1.1050.

perchunk_uzbekgpt.npz and perchunk_mgpt_base.npz are the original per-chunk
scores on the old test set (uz_heldout.txt): for every scored chunk, its total
bits and its byte count. The original notebook divided the total bits by the
byte count of the whole text file (EVAL_BYTES = 1,697,029), which includes the
bytes of the final partial chunk that was never scored. The corrected code
divides by the bytes it actually scored.

    python check_old_denominator.py
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WHOLE_FILE_BYTES = 1_697_029   # len(uz_heldout.txt.encode("utf-8")), printed by the original notebook

for name, reported in [("uzbekgpt", 1.1050), ("mgpt_base", 1.1628)]:
    d = np.load(os.path.join(HERE, f"perchunk_{name}.npz"))
    bits, scored = d["bits"].sum(), d["bytes"].sum()
    print(f"{name:10s} chunks {len(d['bits']):5d} | scored bytes {int(scored):,} | "
          f"bits / scored bytes = {bits / scored:.4f} | "
          f"bits / whole-file bytes = {bits / WHOLE_FILE_BYTES:.4f} | reported {reported}")
