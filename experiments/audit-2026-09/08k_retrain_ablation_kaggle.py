"""
================================================================================
08k — RETRAIN THE TOKENIZER ABLATION CLEANLY, FOR FREE ON KAGGLE
================================================================================
Same fix as 08_retrain_ablation.py (exact training documents, two full passes),
split across several free Kaggle sessions. Progress is saved to a PRIVATE Hugging
Face repo, so every session picks up where the last one stopped.

THREE MODES
  python 08k.py prepare          CPU session, once (~1-2 h, uses no GPU quota):
                                 builds the clean training file, checks it, uploads it
  torchrun ... 08k.py train      GPU T4 x2 sessions, repeated until done (~10.5 h each):
                                 trains, saves every 500 steps, stops cleanly before
                                 Kaggle's 12-hour limit
  python 08k.py score            GPU session, once at the end (~15 min):
                                 scores the finished model, prints RESULTS. Reads the
                                 final checkpoint from the attached train notebook's
                                 output first, so it works even if the Hugging Face
                                 upload failed.

DIFFERENCES FROM THE RUNPOD VERSION (report them in the paper; none should matter)
  - fp16 with loss scaling instead of bf16 (T4 GPUs have no bf16)
  - two GPUs, each 4 sequences x 24 accumulation steps: same 192-sequence batch
  - batch sampling uses a different random stream

SETUP (once)
  - Kaggle -> Add-ons -> Secrets: add HF_TOKEN = your Hugging Face WRITE token.
  - Add Input: dataset uzbek-fineweb2-tokens-16k, and your eval-text check
    notebook (05) with its output.
================================================================================
"""

import argparse, csv, glob, json, math, os, sys, threading, time
from datetime import timedelta

import numpy as np

WORK = os.environ.get("WORK", "/kaggle/working")
HF_REPO = "IslombekT/uzbek-ablation-clean-work"      # private; created automatically
LN2 = math.log(2.0)
BASE_TOK, MY_TOK, MY_MODEL_REPO = "ai-forever/mGPT", "IslombekT/uzbek-bpe-16k", "IslombekT/uzbek-gpt-103m"
DATA, META = "train_mgpt_clean.bin", "train_mgpt_clean.json"
LAST, FINAL = "ablation_clean_last.pt", "ablation_clean_final.pt"

n_embd, n_layer, n_head, block_size = 768, 12, 12, 1024
MICRO, GLOBAL_ACC = 4, 48                  # 4 x 48 x 1024 = 196,608 tokens/step, as before
peak_lr, min_lr, weight_decay, grad_clip = 3e-4, 3e-5, 0.1, 1.0
warmup_steps, EPOCHS, SEED = 400, 2, 1337
SAVE_EVERY, LOG_EVERY = 500, 50

SPAN_BYTES, CONTEXT_BYTES, MAX_TOKENS = 800, 300, 512
RESAMPLES, BOOT_SEED = 10_000, 20260913
KNOWN = {"from_scratch": {"clean": 1.0281, "raw": 1.0380}, "old_ablation": {"clean": 1.0492, "raw": 1.0477},
         "mgpt_qlora_1M": {"raw": 1.0772}, "mgpt_qlora_10M": {"raw": 1.0639}}


# ============================ PURE FUNCTIONS ============================
def same_documents(our_token_counts_sum, n_docs, last_eot_pos):
    return our_token_counts_sum + n_docs == last_eot_pos + 1


def find_sequence(hay, needle):
    L = len(needle)
    if L == 0 or len(hay) < L:
        return []
    cand = np.flatnonzero(hay[:len(hay) - L + 1] == needle[0])
    for k in range(1, L):
        if cand.size == 0:
            break
        cand = cand[hay[cand + k] == needle[k]]
    return cand.tolist()


def get_lr(step, max_steps):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    p = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * p)) * (peak_lr - min_lr)


def build_spans(text, span_bytes=SPAN_BYTES, context_bytes=CONTEXT_BYTES):
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


def paired_gap(model_scores, ref_scores, resamples=RESAMPLES, seed=BOOT_SEED):
    ids = sorted(set(model_scores) & set(ref_scores))
    nm = np.array([model_scores[i][0] for i in ids]); nr = np.array([ref_scores[i][0] for i in ids])
    b = np.array([ref_scores[i][1] for i in ids])
    idx = np.random.default_rng(seed).integers(0, len(ids), size=(resamples, len(ids)))
    d = (nm[idx].sum(1) - nr[idx].sum(1)) / (LN2 * b[idx].sum(1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return (nm.sum() - nr.sum()) / (LN2 * b.sum()), lo, hi


def build_data(docs, count_ours, encode_mgpt, eos_id, n_docs, out_path, batch=2000, log=print):
    ours_total, seen, written, buf = 0, 0, 0, []
    with open(out_path + ".partial", "wb") as f:
        def flush(b):
            nonlocal ours_total, written
            ours_total += sum(count_ours(b))
            for ids in encode_mgpt(b):
                arr = np.asarray(ids + [eos_id], dtype=np.int32)
                arr.tofile(f); written += len(arr)
        for text in docs:
            buf.append(text); seen += 1
            if len(buf) == batch:
                flush(buf); buf = []
            if seen % 100_000 == 0:
                log(f"  {seen:>9,} / {n_docs:,} documents | {written / 1e6:,.0f}M mGPT tokens")
            if seen == n_docs:
                break
        if buf:
            flush(buf)
    if seen != n_docs:
        raise SystemExit(f"stream ended after {seen:,} documents, expected {n_docs:,}")
    os.replace(out_path + ".partial", out_path)
    return ours_total, written


def sessions_left(steps_left, sec_per_step, hours_per_session):
    return math.ceil(steps_left * sec_per_step / (hours_per_session * 3600)) if sec_per_step > 0 else None
# ============================ END PURE FUNCTIONS ========================


def write_flag(path, value):
    with open(path + ".tmp", "w") as f:
        f.write(value)
    os.replace(path + ".tmp", path)          # atomic: the other GPU never sees a half-written flag


def log(msg, rank=0):
    if rank == 0:
        print(msg, flush=True)


def find_input(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)) + sorted(glob.glob(f"{WORK}/{pattern}"))
    return hits[0] if hits else None


def hf():
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Add it under Add-ons -> Secrets, then run the secrets cell.")
    api = HfApi(token=token)
    api.create_repo(HF_REPO, private=True, exist_ok=True)
    return api, token


def hf_has(api, name):
    return name in set(api.list_repo_files(HF_REPO))


def hf_get(api, token, name):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, name, token=token, local_dir=WORK)


# ----------------------------------------------------------------------------- prepare
def prepare():
    from transformers import AutoTokenizer
    from datasets import load_dataset
    api, token = hf()
    if hf_has(api, DATA) and hf_has(api, META):
        log(f"already prepared: {HF_REPO}/{DATA}"); return

    tb = find_input("train.bin")
    if not tb:
        raise SystemExit("train.bin not found. Add Input -> uzbek-fineweb2-tokens-16k")
    t = np.memmap(tb, dtype=np.uint16, mode="r")
    eot = np.flatnonzero(t == 0)
    n_docs, last_eot = len(eot), int(eot[-1])
    log(f"train.bin: {n_docs:,} documents, last <|endoftext|> at {last_eot:,}")
    del t, eot

    tk = AutoTokenizer.from_pretrained(MY_TOK); tk.model_max_length = 10 ** 9
    tok_m = AutoTokenizer.from_pretrained(BASE_TOK); tok_m.model_max_length = 10 ** 9
    eos = tok_m.eos_token_id if tok_m.eos_token_id is not None else 0
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="uzn_Latn", split="train", streaming=True)
    docs = ((ex.get("text") or "") for ex in ds)             # as tokenize_data.py: no strip, no skip
    t0 = time.time()
    ours, written = build_data(
        docs, lambda b: [len(x) for x in tk(b, add_special_tokens=False)["input_ids"]],
        lambda b: tok_m(b, add_special_tokens=False)["input_ids"], eos, n_docs, f"{WORK}/{DATA}")
    ok = same_documents(ours, n_docs, last_eot)
    log(f"built in {(time.time() - t0) / 60:.0f} min | same-documents check: {ours:,} + {n_docs:,} = "
        f"{ours + n_docs:,} vs {last_eot + 1:,} -> {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("The documents do not match train.bin. Send this output back before going on.")

    data = np.memmap(f"{WORK}/{DATA}", dtype=np.int32, mode="r")
    rj = find_input("uz_heldout_dedup_raw.jsonl")
    if not rj:
        raise SystemExit("uz_heldout_dedup_raw.jsonl not found. Add Input -> the eval-text check notebook (05).")
    raw = "\n".join(json.loads(l)["raw"] for l in open(rj, encoding="utf-8") if json.loads(l).get("raw"))
    probe = np.asarray(tok_m(raw, add_special_tokens=False)["input_ids"], dtype=np.int32)
    hits = sum(bool(find_sequence(data, probe[p:p + 32])) for p in np.linspace(0, len(probe) - 33, 16).astype(int))
    log(f"leak check: {hits}/16 test passages in the new training file")
    if hits:
        raise SystemExit("ABORT — test text is in the training data.")

    json.dump({"n_docs": n_docs, "mgpt_tokens": written, "our_tokens": ours, "leak_hits": hits},
              open(f"{WORK}/{META}", "w"))
    log(f"uploading {written * 4 / 1e9:.1f} GB to the private repo {HF_REPO} ...")
    api.upload_file(path_or_fileobj=f"{WORK}/{DATA}", path_in_repo=DATA, repo_id=HF_REPO)
    api.upload_file(path_or_fileobj=f"{WORK}/{META}", path_in_repo=META, repo_id=HF_REPO)
    log("PREPARE DONE. Next: a GPU T4 x2 session running the train cell.")


# ----------------------------------------------------------------------------- train
def train(hours):
    rank, world, local = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    flag = f"{WORK}/.ready_{os.environ.get('TORCHELASTIC_RUN_ID', 'single')}"
    t_start = time.time()

    # --- rank 0 fetches data + latest checkpoint; the others wait for it (before NCCL, so no timeout)
    if rank == 0:
        api, token = hf()
        if hf_has(api, FINAL):
            log("TRAINING ALREADY COMPLETE — run the score cell."); write_flag(flag, "done"); return
        if not os.path.exists(f"{WORK}/{DATA}"):
            log("downloading training data ..."); hf_get(api, token, DATA)
        if hf_has(api, LAST) and not os.path.exists(f"{WORK}/{LAST}"):
            log("downloading latest checkpoint ..."); hf_get(api, token, LAST)
        write_flag(flag, "ok")
    else:
        while not os.path.exists(flag):
            time.sleep(5)
        if open(flag).read() == "done":
            return

    import torch
    import torch.nn.functional as F
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download

    ddp = world > 1
    if ddp:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    acc_steps = GLOBAL_ACC // world
    assert acc_steps * world == GLOBAL_ACC, "use 1 or 2 GPUs"

    tok_m = AutoTokenizer.from_pretrained(BASE_TOK)
    VOCAB = tok_m.vocab_size
    data = np.memmap(f"{WORK}/{DATA}", dtype=np.int32, mode="r")
    max_steps = (len(data) * EPOCHS) // (MICRO * GLOBAL_ACC * block_size)

    mp = hf_hub_download(MY_MODEL_REPO, "model.py"); sys.path.insert(0, os.path.dirname(mp))
    from model import GPT
    torch.manual_seed(SEED + rank); torch.set_float32_matmul_precision("high")
    raw_model = GPT(vocab_size=VOCAB, n_embd=n_embd, block_size=block_size, num_heads=n_head, n_layers=n_layer).to(dev)
    opt = torch.optim.AdamW(raw_model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    start = 0
    if os.path.exists(f"{WORK}/{LAST}"):
        ck = torch.load(f"{WORK}/{LAST}", map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start = ck["step"] + 1
        del ck
    log(f"GPUs: {world} x {torch.cuda.get_device_name(local)} | {sum(p.numel() for p in raw_model.parameters()) / 1e6:.1f}M params | "
        f"steps {start:,} -> {max_steps:,} of {max_steps:,}", rank)

    gen = torch.Generator().manual_seed(SEED * 1000 + rank + start)
    model = raw_model
    try:                                            # compile is faster, but test it first
        compiled = torch.compile(raw_model)
        xt = torch.randint(0, VOCAB, (1, block_size), device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            compiled(xt).float().mean().backward()
        raw_model.zero_grad(set_to_none=True)
        model = compiled
        log("torch.compile: on", rank)
    except Exception as e:
        raw_model.zero_grad(set_to_none=True)
        log(f"torch.compile failed ({type(e).__name__}); continuing without it (slower, same result)", rank)
    if ddp:
        model = DDP(model, device_ids=[local])

    def get_batch():
        ix = torch.randint(len(data) - block_size - 1, (MICRO,), generator=gen)
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64)) for i in ix])
        return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)

    uploader = [None]

    def save(name, step, upload=True, wait=False):
        if rank != 0:
            return
        if uploader[0] is not None:
            uploader[0].join()
        torch.save({"model": raw_model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "config": dict(vocab_size=VOCAB, n_embd=n_embd, n_layer=n_layer,
                                                 n_head=n_head, block_size=block_size)},
                   f"{WORK}/{name}.tmp")
        os.replace(f"{WORK}/{name}.tmp", f"{WORK}/{name}")
        if upload:
            api, _ = hf()
            th = threading.Thread(target=lambda: api.upload_file(
                path_or_fileobj=f"{WORK}/{name}", path_in_repo=name, repo_id=HF_REPO,
                commit_message=f"{name} step {step}"))
            th.start(); uploader[0] = th
            if wait:
                th.join()

    budget = hours * 3600
    model.train()
    t0, n_since, stop = time.time(), 0, torch.zeros(1, device=dev)
    loss_acc = torch.zeros((), device=dev)
    step = start
    for step in range(start, max_steps):
        for g in opt.param_groups:
            g["lr"] = get_lr(step, max_steps)
        opt.zero_grad(set_to_none=True)
        for micro in range(acc_steps):
            if ddp:
                model.require_backward_grad_sync = (micro == acc_steps - 1)
            xb, yb = get_batch()
            with torch.autocast("cuda", dtype=torch.float16):
                lo = model(xb); B, T, V = lo.shape
                loss = F.cross_entropy(lo.reshape(B * T, V).float(), yb.reshape(B * T)) / acc_steps
            scaler.scale(loss).backward(); loss_acc += loss.detach()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        scaler.step(opt); scaler.update(); n_since += 1

        if step % LOG_EVERY == 0 or step == max_steps - 1:
            sps = (time.time() - t0) / n_since
            left = sessions_left(max_steps - step - 1, sps, hours)
            log(f"step {step:6,}/{max_steps:,} | loss {loss_acc.item() / n_since:.4f} | lr {get_lr(step, max_steps):.2e} | "
                f"{sps:.2f}s/step | this session {(time.time() - t_start) / 3600:.2f}h | "
                f"about {left} more session(s)", rank)
            loss_acc.zero_(); t0, n_since = time.time(), 0
        if (step + 1) % SAVE_EVERY == 0:
            save(LAST, step)

        stop.fill_(1.0 if (rank == 0 and time.time() - t_start > budget) else 0.0)
        if ddp:
            dist.broadcast(stop, src=0)
        if stop.item() > 0:
            log(f"time budget reached at step {step:,}: saving and uploading, then stopping", rank)
            save(LAST, step, wait=True)
            break
    else:
        save(FINAL, max_steps - 1, wait=True)
        log("TRAINING COMPLETE — run the score cell (it can run right now in this session).", rank)
    if ddp:
        dist.barrier(); dist.destroy_process_group()


# ----------------------------------------------------------------------------- score
def score():
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    path = find_input(FINAL)                       # attached train-notebook output, or this session's disk
    if path:
        log(f"final checkpoint: {path}")
    else:
        api, token = hf()
        if not hf_has(api, FINAL):
            log("No final checkpoint found. Add the ablation-train notebook (with its output) as an input, "
                "or run the train cell again if training is not finished."); return
        path = hf_get(api, token, FINAL)
    tok_m = AutoTokenizer.from_pretrained(BASE_TOK); tok_m.model_max_length = 10 ** 9
    mp = hf_hub_download(MY_MODEL_REPO, "model.py"); sys.path.insert(0, os.path.dirname(mp))
    from model import GPT
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["config"]
    m = GPT(vocab_size=c["vocab_size"], n_embd=c["n_embd"], block_size=c["block_size"],
            num_heads=c["n_head"], n_layers=c["n_layer"])
    m.load_state_dict(ck["model"]); m = m.eval().cuda()
    dp = find_input(DATA)
    if dp:
        expected = (len(np.memmap(dp, dtype=np.int32, mode="r")) * EPOCHS) // (MICRO * GLOBAL_ACC * c["block_size"])
        ok = ck["step"] == expected - 1
        log(f"checkpoint step {ck['step']:,} of {expected - 1:,} -> {'COMPLETE' if ok else 'NOT the final step'}")
        if not ok:
            log("This is not the finished model. Stop here and send me this log."); return
    else:
        log(f"checkpoint step {ck['step']:,} (training file not attached, so the step was not checked)")
    try:                                           # export the weights alone (small, safe format)
        from safetensors.torch import save_model
        save_model(m, f"{WORK}/ablation_clean_model.safetensors")
        log("saved weights -> ablation_clean_model.safetensors")
    except Exception as e:
        torch.save(m.state_dict(), f"{WORK}/ablation_clean_model.pt")
        log(f"saved weights -> ablation_clean_model.pt (safetensors failed: {type(e).__name__})")
    json.dump(c, open(f"{WORK}/ablation_clean_config.json", "w"), indent=2)

    cj, rj = find_input("uz_heldout_dedup.jsonl"), find_input("uz_heldout_dedup_raw.jsonl")
    missing = [n for n, f in [("uz_heldout_dedup.jsonl", cj), ("uz_heldout_dedup_raw.jsonl", rj),
                              ("rawcheck_from_scratch_clean.csv", find_input("rawcheck_from_scratch_clean.csv")),
                              ("rawcheck_from_scratch_raw.csv", find_input("rawcheck_from_scratch_raw.csv"))] if not f]
    if missing:
        log("Missing inputs: " + ", ".join(missing) + ". Add the eval-text check notebook (05) with its output as an input."); return
    clean_docs = [json.loads(l)["text"] for l in open(cj, encoding="utf-8")]
    raw_docs = [json.loads(l)["raw"] for l in open(rj, encoding="utf-8")]
    use = [i for i, r in enumerate(raw_docs) if r is not None and len(r) == len(clean_docs[i])]
    TEXTS = {"clean": "\n".join(clean_docs[i] for i in use), "raw": "\n".join(raw_docs[i] for i in use)}
    SPANS = build_spans(TEXTS["clean"])

    @torch.no_grad()
    def protocol_b(text):
        out = []
        for sid, cc, a, b in SPANS:
            n_bytes = len(text[a:b].encode("utf-8"))
            enc = tok_m(text[cc:b], return_offsets_mapping=True, add_special_tokens=False)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            first = next((k for k, (s, e) in enumerate(offs) if s >= a - cc), len(ids))
            if len(ids) < 2:
                continue
            if len(ids) > MAX_TOKENS:
                drop = len(ids) - MAX_TOKENS; ids = ids[drop:]; first = max(0, first - drop)
            x = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)
            logits = m(x)[:, :-1, :].float(); target = x[:, 1:].clone()
            if first > 1:
                target[:, :first - 1] = -100
            out.append((sid, F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                                             ignore_index=-100, reduction="sum").item(), n_bytes))
        return out

    res = {}
    for t, text in TEXTS.items():
        p = f"{WORK}/rawcheck_ablation_clean_{t}.csv"
        rows = protocol_b(text)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["chunk_id", "sum_nats", "n_bytes"])
            w.writerows([(i, f"{v:.6f}", n) for i, v, n in rows])
        sc = read_scores(p); ref = read_scores(find_input(f"rawcheck_from_scratch_{t}.csv"))
        words = len(text[SPANS[0][2]:SPANS[-1][3]].split())
        res[t] = {"bpb": sum(v[0] for v in sc.values()) / (LN2 * sum(v[1] for v in sc.values())),
                  "bits_per_word": sum(v[0] for v in sc.values()) / LN2 / words, "gap": list(paired_gap(sc, ref))}

    g = res["raw"]["gap"][0]
    lines = ["=" * 76, "RESULTS — clean tokenizer ablation (Kaggle, fp16, 2 x T4)", "=" * 76,
             f"trained to step {ck['step'] + 1:,}: two full passes over exactly uzbek-gpt-103m's training documents",
             "", f"{'':<30}{'web spelling':>16}{'normalized':>16}",
             f"{'uzbek-gpt-103m':<30}{KNOWN['from_scratch']['raw']:>16.4f}{KNOWN['from_scratch']['clean']:>16.4f}",
             f"{'ablation, OLD (flawed run)':<30}{KNOWN['old_ablation']['raw']:>16.4f}{KNOWN['old_ablation']['clean']:>16.4f}",
             f"{'ablation, NEW (clean run)':<30}{res['raw']['bpb']:>16.4f}{res['clean']['bpb']:>16.4f}", "",
             "gap, new ablation minus uzbek-gpt-103m (positive = dedicated tokenizer better) [95% CI]",
             f"  web spelling: {res['raw']['gap'][0]:+.4f} [{res['raw']['gap'][1]:+.4f}, {res['raw']['gap'][2]:+.4f}]",
             f"  normalized:   {res['clean']['gap'][0]:+.4f} [{res['clean']['gap'][1]:+.4f}, {res['clean']['gap'][2]:+.4f}]",
             f"bits per word, new ablation: web {res['raw']['bits_per_word']:.4f} | normalized {res['clean']['bits_per_word']:.4f}",
             "", "decomposition (web spelling):"]
    for name, key in [("vs mGPT + QLoRA 1M ", "mgpt_qlora_1M"), ("vs mGPT + QLoRA 10M", "mgpt_qlora_10M")]:
        total = KNOWN[key]["raw"] - KNOWN["from_scratch"]["raw"]
        lines.append(f"  {name}: total {total:.4f} = tokenizer {g:.4f} ({g / total:.0%}) + training data {total - g:.4f} ({1 - g / total:.0%})")
    lines.append("=" * 76)
    log("\n".join(lines))
    json.dump({"steps": ck["step"] + 1, "new_ablation": res, "known": KNOWN},
              open(f"{WORK}/retrain_results.json", "w"), indent=2, default=float)
    log(f"saved -> {WORK}/retrain_results.json, rawcheck_ablation_clean_*.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "train", "score"])
    ap.add_argument("--hours", type=float, default=10.5, help="training time per session before a clean stop")
    a = ap.parse_args()
    {"prepare": prepare, "train": lambda: train(a.hours), "score": score}[a.mode]()
