"""Benchmark: our ModernBERT vs Hugging Face ModernBertForMaskedLM on the same weights and inputs.

Run from the repo root:
    python -m evaluation.benchmark_vs_hf                 # released answerdotai/ModernBERT-base, real text
    python -m evaluation.benchmark_vs_hf --tiny          # tiny random model, no download (quick self-check)
    python -m evaluation.benchmark_vs_hf --device cuda   # on a Colab GPU (still fp32)

What it checks, in fp32 with dropout off:
  1. hidden states and logits on short sentences, a padded batch, and long sequences (> 129 tokens,
     so the local sliding window is actually exercised)
  2. prediction agreement: top-1 and top-5 at every real token, KL divergence of the distributions
  3. MLM loss and masked-token accuracy on 30%-masked text
  4. gradients of every parameter after one backward pass
  5. speed and (on GPU) peak memory - informational only, ours uses dense attention
"""

import argparse
import time

import torch
import torch.nn.functional as F

from modernbert import ModernBertPaperForMaskedLM
from modernbert.compare_with_hf import config_from_hf, hf_to_ours_state_dict

SENTENCES = [
    "The capital of France is Paris.",
    "ModernBERT is an encoder-only transformer trained on two trillion tokens.",
    "def add(a, b):\n    return a + b",
    "The quick brown fox jumps over the lazy dog.",
    "In 1969, Apollo 11 landed on the Moon and Neil Armstrong became the first person to walk on its surface.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "SELECT name, age FROM users WHERE age > 30 ORDER BY age DESC;",
    "She sells sea shells by the sea shore.",
]
LONG_PARAGRAPH = (
    "Transformers process sequences with attention, letting every token look at other tokens. "
    "Encoder models such as BERT read the whole sentence in both directions, which makes them strong "
    "for classification, retrieval and named entity recognition. ModernBERT keeps this design but adds "
    "rotary position embeddings, GeGLU layers, alternating local and global attention, and unpadding. "
    "Local layers only look at a window of nearby tokens, which saves compute on long documents, while "
    "every third layer is global and connects distant parts of the text. "
)

# Pass/fail thresholds for fp32. "rel" = max |ours - hf| / max |hf| for hidden states and logits.
TOL = {"rel": 1e-4, "top1_agree": 0.999, "loss_abs": 1e-4, "grad_rel": 1e-3}


# ----------------------------------------------------------------------------- loading
def load_pair(name: str | None, device: str, attn_impl: str):
    from transformers import AutoTokenizer, ModernBertConfig, ModernBertForMaskedLM

    if name:
        tok = AutoTokenizer.from_pretrained(name)
        hf = ModernBertForMaskedLM.from_pretrained(name, attn_implementation=attn_impl)
    else:  # tiny random model: no download, checks the benchmark logic itself
        tok = None
        torch.manual_seed(0)
        hf_cfg = ModernBertConfig(
            vocab_size=512, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, intermediate_size=96,
            local_attention=16, global_attn_every_n_layers=3, max_position_embeddings=1024,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2,
        )
        hf = ModernBertForMaskedLM(hf_cfg)
        hf.config._attn_implementation = attn_impl
    cfg = config_from_hf(hf.config)
    ours = ModernBertPaperForMaskedLM(cfg)
    result = ours.load_state_dict(hf_to_ours_state_dict(hf.state_dict(), cfg), strict=False)
    assert not result.missing_keys and not result.unexpected_keys, result
    return tok, hf.to(device).eval().float(), ours.to(device).eval().float(), cfg


# ----------------------------------------------------------------------------- inputs
def make_batches(tok, cfg, long_len: int):
    """Returns {name: (input_ids, attention_mask)}."""
    if tok is not None:
        enc = lambda texts: tok(texts, return_tensors="pt", padding=True)
        short = enc(SENTENCES[:1])
        padded = enc(SENTENCES)                                   # different lengths -> padding
        long_ids = tok(LONG_PARAGRAPH * 40, return_tensors="pt", truncation=True, max_length=long_len)
        long_pad = enc([LONG_PARAGRAPH * 40, LONG_PARAGRAPH * 3])  # long row + short row
        long_pad = {k: v[:, :long_len] for k, v in long_pad.items()}
        batches = {"1 sentence": short, f"{len(SENTENCES)} sentences, padded": padded,
                   f"long ({long_ids['input_ids'].shape[1]} tok)": long_ids,
                   "long + short, padded": long_pad}
        return {k: (v["input_ids"], v["attention_mask"]) for k, v in batches.items()}
    g = torch.Generator().manual_seed(1)
    rnd = lambda b, s: torch.randint(5, cfg.vocab_size, (b, s), generator=g)
    ids = rnd(4, 40); mask = torch.ones_like(ids); mask[1, 30:] = 0; mask[3, 12:] = 0; ids[mask == 0] = cfg.pad_token_id
    lids = rnd(2, long_len); lmask = torch.ones_like(lids); lmask[1, long_len // 3:] = 0; lids[lmask == 0] = cfg.pad_token_id
    return {"1 x 20": (rnd(1, 20), torch.ones(1, 20, dtype=torch.long)), "4 x 40, padded": (ids, mask),
            f"2 x {long_len}, padded": (lids, lmask)}


def mask_for_mlm(ids, attn, mask_id, special_ids, p=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    cand = attn.bool() & ~torch.isin(ids, torch.tensor(sorted(special_ids)))
    pick = (torch.rand(ids.shape, generator=g) < p) & cand
    labels = torch.where(pick, ids, torch.full_like(ids, -100))
    return torch.where(pick, torch.full_like(ids, mask_id), ids), labels


# ----------------------------------------------------------------------------- checks
@torch.no_grad()
def compare_outputs(hf, ours, ids, attn, device):
    ids, attn = ids.to(device), attn.to(device)
    real = attn.bool()
    h_hf = hf.model(input_ids=ids, attention_mask=attn).last_hidden_state
    h_ours = ours.model(ids, attention_mask=attn)
    l_hf = hf(input_ids=ids, attention_mask=attn).logits
    l_ours = ours(ids, attention_mask=attn)["logits"]
    a, b = l_ours[real], l_hf[real]
    top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    t5a, t5b = a.topk(5, -1).indices, b.topk(5, -1).indices
    top5 = torch.tensor([len(set(x.tolist()) & set(y.tolist())) / 5 for x, y in zip(t5a, t5b)]).mean().item()
    kl = F.kl_div(a.log_softmax(-1), b.log_softmax(-1), log_target=True, reduction="batchmean").item()
    return {
        "hidden_max_abs": (h_ours[real] - h_hf[real]).abs().max().item(),
        "hidden_rel": ((h_ours[real] - h_hf[real]).abs().max() / h_hf[real].abs().max()).item(),
        "logits_max_abs": (a - b).abs().max().item(),
        "logits_rel": ((a - b).abs().max() / b.abs().max()).item(),
        "top1_agree": top1, "top5_overlap": top5, "kl": max(kl, 0.0), "tokens": int(real.sum()),
    }


def compare_mlm(hf, ours, ids, attn, labels, device):
    ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
    with torch.no_grad():
        l_hf = hf(input_ids=ids, attention_mask=attn).logits
        loss_hf = F.cross_entropy(l_hf.reshape(-1, l_hf.size(-1)), labels.reshape(-1), ignore_index=-100)
        out = ours(ids, attention_mask=attn, labels=labels)
    sel = labels != -100
    acc = lambda logits: (logits.argmax(-1)[sel] == labels[sel]).float().mean().item()
    return {"loss_hf": loss_hf.item(), "loss_ours": out["loss"].item(),
            "loss_abs": abs(loss_hf.item() - out["loss"].item()),
            "acc_hf": acc(l_hf), "acc_ours": acc(out["logits"]), "masked": int(sel.sum())}


def compare_grads(hf, ours, ids, attn, labels, cfg, device):
    ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
    for m in (hf, ours):
        m.zero_grad(set_to_none=True)
    l_hf = hf(input_ids=ids, attention_mask=attn).logits
    F.cross_entropy(l_hf.reshape(-1, l_hf.size(-1)), labels.reshape(-1), ignore_index=-100).backward()
    ours(ids, attention_mask=attn, labels=labels)["loss"].backward()
    hf_grads = hf_to_ours_state_dict({n: p.grad for n, p in hf.named_parameters() if p.grad is not None}, cfg)
    worst, worst_name, n = 0.0, "", 0
    for name, p in ours.named_parameters():
        g_ref = hf_grads.get(name)
        if g_ref is None or p.grad is None:
            raise AssertionError(f"no gradient pair for {name}")
        rel = ((p.grad - g_ref).abs().max() / g_ref.abs().max().clamp_min(1e-12)).item()
        n += 1
        if rel > worst:
            worst, worst_name = rel, name
    return {"grad_rel": worst, "worst_param": worst_name, "params": n}


def timeit(fn, device, reps=5):
    fn()  # warm-up
    if device == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / reps * 1000
    mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")
    return ms, mem


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--tiny", action="store_true", help="tiny random model, no download")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn", default="eager", help="HF attention implementation: eager or sdpa")
    ap.add_argument("--long", type=int, default=512, help="length of the long-sequence test (> 129 tests windows)")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False   # keep GPU matmuls in true fp32
    torch.backends.cudnn.allow_tf32 = False

    tok, hf, ours, cfg = load_pair(None if args.tiny else args.model, args.device, args.attn)
    print(f"device={args.device}  hf_attn={args.attn}  layers={cfg.num_hidden_layers}  "
          f"params(ours)={sum(p.numel() for p in ours.parameters()) / 1e6:.1f}M")
    batches = make_batches(tok, cfg, args.long)
    ok = True

    print("\n[1-2] outputs on identical inputs (real tokens only)")
    print(f"{'input':<28}{'tokens':>7}{'hidden rel':>12}{'logits rel':>12}{'top-1':>8}{'top-5':>8}{'KL':>10}")
    for name, (ids, attn) in batches.items():
        r = compare_outputs(hf, ours, ids, attn, args.device)
        ok &= max(r["hidden_rel"], r["logits_rel"]) < TOL["rel"] and r["top1_agree"] >= TOL["top1_agree"]
        print(f"{name:<28}{r['tokens']:>7}{r['hidden_rel']:>12.2e}{r['logits_rel']:>12.2e}"
              f"{r['top1_agree']:>8.2%}{r['top5_overlap']:>8.2%}{r['kl']:>10.1e}")

    mask_id = tok.mask_token_id if tok is not None else 3
    special = set(tok.all_special_ids) if tok is not None else {0, 1, 2, 3}
    ids, attn = list(batches.values())[1]
    mids, labels = mask_for_mlm(ids, attn, mask_id, special)

    print("\n[3] MLM on 30%-masked text")
    r = compare_mlm(hf, ours, mids, attn, labels, args.device)
    ok &= r["loss_abs"] < TOL["loss_abs"]
    print(f"masked tokens={r['masked']}  loss hf={r['loss_hf']:.6f} ours={r['loss_ours']:.6f} "
          f"(diff {r['loss_abs']:.1e})  accuracy hf={r['acc_hf']:.2%} ours={r['acc_ours']:.2%}")

    print("\n[4] gradients after one backward pass")
    r = compare_grads(hf, ours, mids, attn, labels, cfg, args.device)
    ok &= r["grad_rel"] < TOL["grad_rel"]
    print(f"compared {r['params']} parameters, worst relative error {r['grad_rel']:.1e} ({r['worst_param']})")

    print("\n[5] speed (forward pass, informational)")
    for name, (ids, attn) in batches.items():
        ids, attn = ids.to(args.device), attn.to(args.device)
        with torch.no_grad():
            t_hf, m_hf = timeit(lambda: hf(input_ids=ids, attention_mask=attn), args.device)
            t_o, m_o = timeit(lambda: ours(ids, attention_mask=attn), args.device)
        mem = f"   peak MB hf={m_hf:.0f} ours={m_o:.0f}" if args.device == "cuda" else ""
        print(f"{name:<28} hf={t_hf:8.1f} ms   ours={t_o:8.1f} ms{mem}")

    print("\nRESULT:", "PASS - same results as Hugging Face" if ok else "FAIL - see rows above")


if __name__ == "__main__":
    main()
