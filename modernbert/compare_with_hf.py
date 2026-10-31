"""Numerical comparison against the Hugging Face ModernBERT implementation (plan step 7).

Requires `pip install transformers` (no GPU, no flash-attn). Two modes:

  python -m modernbert.compare_with_hf
      -> builds a TINY random HF ModernBERT, copies its weights into our model, compares outputs.
  python -m modernbert.compare_with_hf --pretrained answerdotai/ModernBERT-base
      -> same, with the released checkpoint (downloads ~600 MB from the Hugging Face Hub).
"""

import argparse

import torch

from .config import ModernBertPaperConfig
from .model import ModernBertPaperForMaskedLM


def config_from_hf(hf_cfg) -> ModernBertPaperConfig:
    rope = hf_cfg.rope_parameters
    return ModernBertPaperConfig(
        vocab_size=hf_cfg.vocab_size,
        hidden_size=hf_cfg.hidden_size,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        num_attention_heads=hf_cfg.num_attention_heads,
        intermediate_size=hf_cfg.intermediate_size,
        max_position_embeddings=hf_cfg.max_position_embeddings,
        layer_norm_eps=hf_cfg.norm_eps,
        norm_bias=hf_cfg.norm_bias,
        attn_qkv_bias=hf_cfg.attention_bias,
        attn_out_bias=hf_cfg.attention_bias,
        mlp_in_bias=hf_cfg.mlp_bias,
        mlp_out_bias=hf_cfg.mlp_bias,
        sliding_window=hf_cfg.local_attention,
        global_attn_every_n_layers=hf_cfg.global_attn_every_n_layers,
        rotary_emb_base=rope["full_attention"]["rope_theta"],
        local_attn_rotary_emb_base=rope["sliding_attention"]["rope_theta"],
        head_pred_bias=hf_cfg.classifier_bias,
        decoder_bias=hf_cfg.decoder_bias,
        pad_token_id=hf_cfg.pad_token_id,
    )


def hf_to_ours_state_dict(hf_sd: dict, cfg: ModernBertPaperConfig) -> dict:
    """Rename HF keys to ours. Wi is split into gate_proj (first half) and up_proj (second half)."""
    out, inter = {}, cfg.intermediate_size
    for k, v in hf_sd.items():
        k2 = k.replace("model.embeddings.tok_embeddings", "model.embeddings.token_embedding")
        k2 = k2.replace("model.final_norm", "model.norm")
        k2 = k2.replace(".attn_norm.", ".norm1.").replace(".mlp_norm.", ".norm2.")
        k2 = k2.replace(".attn.Wqkv.", ".attn.qkv_proj.").replace(".attn.Wo.", ".attn.out_proj.")
        k2 = k2.replace(".mlp.Wo.", ".mlp.down_proj.")
        k2 = k2.replace("head.dense.", "head_dense.").replace("head.norm.", "head_norm.")
        k2 = k2.replace("decoder.", "lm_head.")
        if ".mlp.Wi." in k2:
            out[k2.replace(".mlp.Wi.", ".mlp.gate_proj.")] = v[:inter]
            out[k2.replace(".mlp.Wi.", ".mlp.up_proj.")] = v[inter:]
            continue
        out[k2] = v
    return out


def report(name, a, b):
    diff = (a - b).abs()
    rel = diff.max() / b.abs().max().clamp_min(1e-12)
    ok = torch.allclose(a, b, atol=1e-4, rtol=1e-4)
    print(f"{name:<14} max_abs={diff.max():.3e}  mean_abs={diff.mean():.3e}  rel={rel:.3e}  allclose={ok}")
    return ok


def main(pretrained: str | None = None, seq_len: int = 40):
    from transformers import ModernBertConfig, ModernBertForMaskedLM

    torch.manual_seed(0)
    if pretrained:
        hf = ModernBertForMaskedLM.from_pretrained(pretrained, attn_implementation="eager")
    else:
        hf_cfg = ModernBertConfig(
            vocab_size=128, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            intermediate_size=96, local_attention=16, global_attn_every_n_layers=3,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2,
        )
        hf = ModernBertForMaskedLM(hf_cfg)
        hf.config._attn_implementation = "eager"
    hf.eval()

    cfg = config_from_hf(hf.config)
    ours = ModernBertPaperForMaskedLM(cfg).eval()
    missing, unexpected = ours.load_state_dict(hf_to_ours_state_dict(hf.state_dict(), cfg), strict=False)
    print("missing:", missing, "\nunexpected:", unexpected)

    ids = torch.randint(5, cfg.vocab_size, (2, seq_len))
    mask = torch.ones_like(ids)
    mask[1, seq_len - 7:] = 0          # right padding on the second row
    ids[1, seq_len - 7:] = cfg.pad_token_id
    real = mask.bool()

    with torch.no_grad():
        h_hf = hf.model(input_ids=ids, attention_mask=mask).last_hidden_state
        h_ours = ours.model(ids, attention_mask=mask)
        l_hf = hf(input_ids=ids, attention_mask=mask).logits
        l_ours = ours(ids, attention_mask=mask)["logits"]

    ok = report("hidden", h_ours[real], h_hf[real])
    ok &= report("logits", l_ours[real], l_hf[real])
    print("MATCH" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default=None)
    args = p.parse_args()
    main(args.pretrained)
