#!/usr/bin/env python3
"""
uzbek_imlo.py - convert Uzbek Latin text to the amended alphabet

    sh -> ş    ch -> ç    oʻ -> ö    gʻ -> ğ        (capitals: Ş Ç Ö Ğ)

Mapping as reported for the 2026 amendment approved by parliament. Verify it against
the final signed text (lex.uz) before publishing results. The tutuq belgisi (ʼ) is
left unchanged by this script.

Three functions:

  naive_convert(text)              plain find-and-replace. The baseline.
  careful_convert(text, log=None)  leaves URLs, e-mails, @handles, #hashtags and
                                   foreign-looking words alone; accepts every apostrophe
                                   character people actually type; reads a word-final o'
                                   as a closing quote; counts every decision in `log`.
  normalize_apostrophes(text)      same alphabet, consistent marks:
                                   oʻ/gʻ -> U+02BB, in-word tutuq -> U+02BC.
  to_old(text)                     the reverse: ş ç ö ğ -> sh ch oʻ gʻ.

Command line:

  python uzbek_imlo.py --test
  python uzbek_imlo.py "Men o'qishni yaxshi ko'raman"        # Men öqişni yaxşi köraman
  python uzbek_imlo.py -i in.txt -o out.txt --mode normalize+careful

  Streams line by line, so it works on a multi-GB corpus. A ~200 MB sample is
  plenty for training a 16k-vocabulary tokenizer.

In a Jupyter / Kaggle notebook, save this file first (%%writefile uzbek_imlo.py),
then either run it with !python uzbek_imlo.py --test, or import it:

  from uzbek_imlo import careful_convert, naive_convert, normalize_apostrophes
  main(["--test"])            # the CLI also accepts an argument list

Known limits (the log counts each one, so you can measure them instead of guessing):
  * the bare letter name "o'" standing alone is read as a closing quote
  * English words built only from letters Uzbek also uses ("fresh", "such") are only
    skipped if they are in FOREIGN_WORDS - extend the list from the audit's top words
  * a word containing c not followed by h is treated as foreign, so Uzbek words spelled
    with a nonstandard c ("licenziyalash" for "litsenziyalash") are left unconverted
  * a foreign name glued to an Uzbek suffix by an apostrophe ("Samsungʻning") is
    converted as if it were Uzbek
"""

import argparse
import re
import sys
from collections import Counter

# Every character people type between o/g and the next letter, in the wild.
APOS_CHARS = "'\u2018\u2019\u02BB\u02BC\u0060\u00B4"   # ' ‘ ’ ʻ ʼ ` ´
OKINA = "\u02BB"   # ʻ  official mark in oʻ / gʻ
TUTUQ = "\u02BC"   # ʼ  official tutuq belgisi
NEW = {"s": "ş", "S": "Ş", "c": "ç", "C": "Ç", "o": "ö", "O": "Ö", "g": "ğ", "G": "Ğ"}

_AP = re.escape(APOS_CHARS)
_A = "[" + _AP + "]"
_L = r"[^\W\d_]"                     # any Unicode letter

_PROTECT_RE = re.compile(
    r"(?:https?://|www\.)\S+"                                          # URLs
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"                                   # e-mails
    r"|(?<![\w" + _AP + r"])[@#][\w" + _AP + r"]+"                     # @handles, #hashtags
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:uz|com|ru|org|net|io|info|me|tv|gov|edu)\b(?:/\S*)?",
    re.IGNORECASE,
)
_WORD_RE = re.compile("(?:" + _L + "|" + _A + ")+")
_TARGET_RE = re.compile("[sScC][hH]|[oOgG]" + _A)
_HAS_APOS_RE = re.compile(_A)

# Letters or clusters native Uzbek Latin never uses: no w, no c outside "ch", etc.
# ("sch" and "tch" are deliberately absent: ruscha, forscha, nemischa are Uzbek.)
_FOREIGN_RE = re.compile(r"w|c(?!h)|th|ph|ck|ght", re.IGNORECASE)

# Foreign words the letter rules cannot catch. Extend from 2_conversion_audit.md.
FOREIGN_WORDS = {
    "english", "british", "spanish", "fresh", "flash", "cash", "push", "bash", "brush",
    "finish", "such", "much", "each", "rich", "teach", "reach", "beach", "match",
    "school", "machine", "dashboard", "hashtag", "share", "shop", "shopping", "shorts",
    "shell", "ssh", "sha", "chmod", "tech", "technology", "technical", "chrome",
    "chatgpt", "chevrolet", "chery", "changan", "chelsea", "michael", "charles",
    "chris", "christian", "christmas",
}


def _og_decision(word, i, quoted):
    """Is o/g + apostrophe at word[i] the letter oʻ/gʻ, or a quote mark?"""
    ch, n = word[i], len(word)
    prev = word[i - 1] if i > 0 else ""
    after = word[i + 2] if i + 2 < n else ""
    after2 = word[i + 3] if i + 3 < n else ""
    if ch == "O" and after.isupper() and after2.islower():
        return "skip_foreign_name"             # O'Reilly, O'Neil
    if after.isalpha() or (after and after in APOS_CHARS and after2.isalpha()):
        return "convert_o_g"                   # oʻzbek, moʻʼjiza, qoʻngʻiroq, boshlangʻich
    if ch in "gG" and prev in ("n", "N"):
        return "skip_ng_quote"                 # word-final: 'rang' - the mark closes a quote
    if ch in "gG" and not quoted:
        return "convert_o_g"                   # word-final gʻ: togʻ, bogʻ, sogʻ
    return "skip_final_quote"                  # 'dunyo' - the mark closes a quote


def _process_word(word, log, mode):
    """mode 'convert' = new letters; mode 'normalize' = same alphabet, canonical marks."""
    has_target = _TARGET_RE.search(word) is not None
    if mode == "convert" and not has_target:
        return word
    if mode == "normalize" and not _HAS_APOS_RE.search(word):
        return word
    if has_target:
        core = word.strip(APOS_CHARS)
        if core.lower() in FOREIGN_WORDS or _FOREIGN_RE.search(core):
            log["skip_foreign_word"] += 1
            return word

    out, i, n = [], 0, len(word)
    quoted = word[0] in APOS_CHARS
    while i < n:
        ch = word[i]
        nxt = word[i + 1] if i + 1 < n else ""
        if ch in "oOgG" and nxt and nxt in APOS_CHARS:
            decision = _og_decision(word, i, quoted)
            log[decision] += 1
            if decision == "convert_o_g":
                out.append(ch + OKINA if mode == "normalize" else NEW[ch])
                i += 2
                continue
        elif mode == "convert" and ch in "sScC" and nxt in ("h", "H"):
            log["convert_sh_ch"] += 1
            out.append(NEW[ch])
            i += 2
            continue
        elif mode == "normalize" and ch in APOS_CHARS and out and out[-1].isalpha() and nxt.isalpha():
            out.append(TUTUQ)                  # maʼlumot, moʻʼjiza
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _apply(text, fn, log):
    """Run fn on everything except protected spans (URLs, e-mails, handles, domains)."""
    out, pos = [], 0
    for m in _PROTECT_RE.finditer(text):
        out.append(fn(text[pos:m.start()]))
        if _TARGET_RE.search(m.group(0)):
            log["skip_protected"] += 1
        out.append(m.group(0))
        pos = m.end()
    out.append(fn(text[pos:]))
    return "".join(out)


def careful_convert(text, log=None):
    log = Counter() if log is None else log
    return _apply(text, lambda s: _WORD_RE.sub(lambda m: _process_word(m.group(0), log, "convert"), s), log)


def normalize_apostrophes(text, log=None):
    log = Counter() if log is None else log
    return _apply(text, lambda s: _WORD_RE.sub(lambda m: _process_word(m.group(0), log, "normalize"), s), log)


_TO_OLD = {"ö": "o" + OKINA, "Ö": "O" + OKINA, "ğ": "g" + OKINA, "Ğ": "G" + OKINA}


def to_old(text):
    """Amended alphabet -> current alphabet: ş->sh, ç->ch, ö->oʻ, ğ->gʻ.

    Lets a tokenizer and model trained on the current alphabet read new-alphabet text
    with no retraining. Capital Ş/Ç become SH/CH inside an all-caps word, Sh/Ch otherwise.
    """
    out, n = [], len(text)
    for i, ch in enumerate(text):
        if ch in "şç":
            out.append(ch.replace("ş", "sh").replace("ç", "ch"))
        elif ch in "ŞÇ":
            nxt = text[i + 1] if i + 1 < n else ""
            h = "H" if nxt.isalpha() and nxt.isupper() else "h"
            out.append(("S" if ch == "Ş" else "C") + h)
        else:
            out.append(_TO_OLD.get(ch, ch))
    return "".join(out)


_NAIVE = [(b + a, NEW[b]) for b in "oOgG" for a in APOS_CHARS] + [
    ("sh", "ş"), ("Sh", "Ş"), ("SH", "Ş"), ("sH", "ş"),
    ("ch", "ç"), ("Ch", "Ç"), ("CH", "Ç"), ("cH", "ç"),
]


def naive_convert(text):
    for old, new in _NAIVE:
        text = text.replace(old, new)
    return text


# ---------------------------------------------------------------------------- tests
_SAME = object()
_CASES = [  # input, careful, naive (_SAME = identical to careful; None = input unchanged)
    ("Men o'qishni yaxshi ko'raman", "Men öqişni yaxşi köraman", _SAME),
    ("O‘zbekiston, oʻzbek, o’zbek, o`zbek, O'ZBEKISTON", "Özbekiston, özbek, özbek, özbek, ÖZBEKISTON", _SAME),
    ("G'ijduvon, tog' va bog'", "Ğijduvon, toğ va boğ", _SAME),
    ("SHAHAR, Shahar, shahar", "ŞAHAR, Şahar, şahar", _SAME),
    ("uchun, chiroyli, MChJ, ShHT", "uçun, çiroyli, MÇJ, ŞHT", _SAME),
    ("ruscha, forscha", "rusça, forsça", _SAME),
    ("mo''jiza, ma'lumot, Is'hoq", "mö'jiza, ma'lumot, Is'hoq", _SAME),
    ("https://kun.uz/news/shahar-yangiliklari", None, "https://kun.uz/news/şahar-yangiliklari"),
    ("shahnoza@mail.uz", None, "şahnoza@mail.uz"),
    ("@shahar_uz #O'zbekiston", None, "@şahar_uz #Özbekiston"),
    ("ChatGPT, Chevrolet, English", None, "ÇatGPT, Çevrolet, Engliş"),
    ("U 'dunyo' dedi", None, "U 'dunyö dedi"),
    ("O'Reilly", None, "ÖReilly"),
    ("'rang'", None, "'ranğ"),
    ("qo'ng'iroq, boshlangʻich, yong‘in", "qönğiroq, boşlanğiç, yonğin", _SAME),   # ngʻ inside a word
    ("Qorongʻi va shoʻngʻin", "Qoronği va şönğin", _SAME),
]
_NORMALIZE_CASES = [
    ("o'zbek ma'lumot", "oʻzbek maʼlumot"),
    ("mo''jiza", "moʻʼjiza"),
    ("tog'", "togʻ"),
    ("'dunyo'", "'dunyo'"),
    ("oʻzbek maʼlumot", "oʻzbek maʼlumot"),          # already canonical: unchanged
]


def _selftest():
    failures = 0

    def check(label, got, want):
        nonlocal failures
        ok = got == want
        failures += not ok
        print(("ok    " if ok else "FAIL  ") + label + ("" if ok else f"\n      got:  {got!r}\n      want: {want!r}"))

    for text, careful, naive in _CASES:
        careful = text if careful is None else careful
        naive = careful if naive is _SAME else naive
        check(f"careful  {text}", careful_convert(text), careful)
        check(f"naive    {text}", naive_convert(text), naive)
        check(f"words    {text}", len(careful_convert(text).split()), len(text.split()))
    for text, want in _NORMALIZE_CASES:
        check(f"normal.  {text}", normalize_apostrophes(text), want)
    check("pipeline normalize+careful", careful_convert(normalize_apostrophes("Men o'qishni yaxshi ko'raman")),
          "Men öqişni yaxşi köraman")
    for text in ["Men o'qishni yaxshi ko'raman", "G'ijduvon, tog' va bog'", "SHAHAR va Choʻl", "qoʻngʻiroq chalindi"]:
        norm = normalize_apostrophes(text)
        check(f"round-trip  {text}", to_old(careful_convert(norm)), norm)
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


def main(argv=None):
    """Command-line entry point. Pass argv as a list to call it from a notebook."""
    ap = argparse.ArgumentParser(description="Uzbek Latin -> amended alphabet (sh ch oʻ gʻ -> ş ç ö ğ)")
    ap.add_argument("text", nargs="?", help="a string to convert")
    ap.add_argument("-i", "--input", help="input file (default: stdin)")
    ap.add_argument("-o", "--output", help="output file (default: stdout)")
    ap.add_argument("--mode", default="careful", choices=["careful", "naive", "normalize", "normalize+careful"])
    ap.add_argument("--test", action="store_true", help="run the self-tests")
    a = ap.parse_args(argv)
    if a.test:
        return _selftest()

    log = Counter()
    fn = {
        "careful": lambda s: careful_convert(s, log),
        "naive": naive_convert,
        "normalize": lambda s: normalize_apostrophes(s, log),
        "normalize+careful": lambda s: careful_convert(normalize_apostrophes(s), log),
    }[a.mode]

    if a.text is not None:
        print(fn(a.text))
        return 0
    src = open(a.input, encoding="utf-8") if a.input else sys.stdin
    dst = open(a.output, "w", encoding="utf-8") if a.output else sys.stdout
    for line in src:
        dst.write(fn(line))
    if src is not sys.stdin:
        src.close()
    if dst is not sys.stdout:
        dst.close()
    if log:
        print("decisions: " + ", ".join(f"{k}={v:,}" for k, v in log.most_common()), file=sys.stderr)
    return 0


def _in_notebook():
    """True inside Jupyter/Kaggle, where sys.argv belongs to the kernel, not to us."""
    try:
        get_ipython  # noqa: F821  (defined by IPython)
        return True
    except NameError:
        return False


if __name__ == "__main__" and not _in_notebook():
    sys.exit(main())
