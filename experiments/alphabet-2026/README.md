# Side study: the Uzbek alphabet amendment

In 2026 Uzbekistan's parliament approved an amendment to the Latin alphabet that replaces `sh`, `ch`, `oʻ`, `gʻ` with `ş`, `ç`, `ö`, `ğ`. This study asks what that does to real Uzbek text and to tokenizers. Check the signed text of the law on lex.uz before relying on the exact mapping.

| File | What it does |
|---|---|
| `uzbek_imlo.py` | Converts Uzbek text to the new alphabet. A careful converter (leaves links, handles, and foreign words alone), a naive find-and-replace baseline, apostrophe normalization, and the reverse mapping. `python uzbek_imlo.py --test` runs the self-tests. |
| `alphabet_study.py` | Apostrophe census, a conversion audit comparing careful and naive conversion, and tokenizer fertility on three versions of the same text. |
| `alphabet_study_kaggle.ipynb` | Runs everything on Kaggle (CPU only). Rebuilds the test set with `../audit-2026-09/build_heldout.py`. |
| `logs/alphabet_study.log` | Output of the run. |

## Findings

- About 28% of words change spelling under the amendment.
- `uzbek-bpe-16k` needs 38% more tokens on new-alphabet text (1.709 → 2.353 tokens per word, test set in normalized spelling). Multilingual tokenizers barely change: they already know these letters from Turkish.
- Converting the new letters back before tokenizing restores 1.709 exactly — no retraining of the tokenizer or the model.
- In a sample of 3,000 raw web documents, the most common way to type the mark in `oʻ`/`gʻ` is the left quote `‘` (34%); the official `ʻ` is used only 23% of the time.
