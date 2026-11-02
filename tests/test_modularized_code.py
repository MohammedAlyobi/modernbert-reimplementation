"""Staged tests for the modernbert package (audit plan, Stages 0-7).

Run from the modernbert-reimplementation folder:
    python -m pytest tests/test_modularized_code.py -v
or without pytest:
    python tests/test_modularized_code.py
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modernbert import (  # noqa: E402
    ModernBertPaperConfig,
    ModernBertPaperEmbeddings,
    ModernBertPaperForMaskedLM,
    ModernBertPaperMLP,
    ModernBertPaperModel,
    ModernBertPaperSelfAttention,
    ModernBertPaperTransformerBlock,
    RotaryEmbedding,
)

B, S = 2, 12


def tiny(**kw):
    return ModernBertPaperConfig.tiny(**kw)


def rand_ids(cfg, b=B, s=S):
    return torch.randint(1, cfg.vocab_size, (b, s))


def max_err(a, b):
    return (a - b).abs().max().item()


# ---------------------------------------------------------------- Stage 0: construction
def test_stage0_construction_and_tying():
    torch.manual_seed(0)
    m = ModernBertPaperForMaskedLM(tiny())
    named = list(m.named_parameters())
    assert len(m.state_dict()) == len(named) + 1, "tied weight should appear twice in state_dict only"
    assert m.lm_head.weight is m.model.embeddings.token_embedding.weight
    assert m.lm_head.bias is not None, "ModernBERT keeps a decoder bias"


def test_stage0_reference_parameter_count():
    # Paper: ModernBERT-base has 149M parameters. Building the full model on CPU takes a few seconds.
    m = ModernBertPaperForMaskedLM(ModernBertPaperConfig())
    n = sum(p.numel() for p in m.parameters())
    assert 149.0e6 < n < 150.0e6, n


def test_stage0_no_biases_except_decoder():
    m = ModernBertPaperForMaskedLM(tiny())
    with_bias = [n for n, _ in m.named_parameters() if n.endswith("bias")]
    assert with_bias == ["lm_head.bias"], with_bias


def test_stage0_first_layer_skips_prenorm():
    m = ModernBertPaperModel(tiny())
    assert isinstance(m.layers[0].norm1, torch.nn.Identity)
    assert isinstance(m.layers[1].norm1, torch.nn.LayerNorm)


# ---------------------------------------------------------------- Stage 1: primitives
def test_stage1_embeddings_shape():
    cfg = tiny()
    assert ModernBertPaperEmbeddings(cfg)(rand_ids(cfg)).shape == (B, S, cfg.hidden_size)


def test_stage1_rope_properties():
    torch.manual_seed(0)
    d = 8
    for interleaved in (False, True):
        rope = RotaryEmbedding(d, interleaved=interleaved)
        x = torch.randn(1, 1, 6, d)
        y = rope(x)
        assert y.shape == x.shape
        assert torch.allclose(y[..., 0, :], x[..., 0, :]), "position 0 must be the identity"
        assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5), "rotation preserves norm"
        # q.k depends only on the distance: same vectors at positions (1,3) and (2,4)
        q = torch.randn(d)
        k = torch.randn(d)
        seq = torch.zeros(1, 1, 5, d)
        rq = rope(seq + q)[0, 0]
        rk = rope(seq + k)[0, 0]
        assert math.isclose((rq[1] @ rk[3]).item(), (rq[2] @ rk[4]).item(), rel_tol=1e-4, abs_tol=1e-5)
    try:
        RotaryEmbedding(7)
        raise AssertionError("odd dim must raise")
    except ValueError:
        pass


def test_stage1_rope_layouts_equivalent_up_to_permutation():
    d = 8
    x = torch.randn(1, 1, 5, d)
    perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])  # interleaved -> half layout
    a = RotaryEmbedding(d, interleaved=True)(x)[..., perm]
    b = RotaryEmbedding(d, interleaved=False)(x[..., perm])
    assert torch.allclose(a, b, atol=1e-6)


def test_stage1_module_shapes():
    cfg = tiny()
    x = torch.randn(B, S, cfg.hidden_size)
    assert ModernBertPaperSelfAttention(cfg, 0)(x).shape == x.shape
    assert ModernBertPaperMLP(cfg)(x).shape == x.shape
    assert ModernBertPaperTransformerBlock(cfg, 1)(x).shape == x.shape


# ---------------------------------------------------------------- Stage 2-3: forward passes
def test_stage2_model_forward():
    cfg = tiny()
    h = ModernBertPaperModel(cfg)(rand_ids(cfg))
    assert h.shape == (B, S, cfg.hidden_size) and torch.isfinite(h).all()


def test_stage3_mlm_forward():
    cfg = tiny()
    m = ModernBertPaperForMaskedLM(cfg)
    ids = rand_ids(cfg)
    out = m(ids)
    assert out["logits"].shape == (B, S, cfg.vocab_size) and out["loss"] is None
    labels = torch.full_like(ids, -100)
    labels[:, ::3] = ids[:, ::3]
    out = m(ids, labels=labels)
    assert torch.isfinite(out["loss"])


def test_stage3_all_ignored_labels_give_nan():
    cfg = tiny()
    out = ModernBertPaperForMaskedLM(cfg)(rand_ids(cfg), labels=torch.full((B, S), -100))
    assert torch.isnan(out["loss"]), "documented behaviour: no masked tokens -> NaN loss"


def test_stage3_masked_prediction_same_loss():
    torch.manual_seed(0)
    cfg = tiny()
    m = ModernBertPaperForMaskedLM(cfg).eval()
    ids = rand_ids(cfg)
    labels = torch.full_like(ids, -100)
    labels[:, 1::4] = ids[:, 1::4]
    full = m(ids, labels=labels)["loss"]
    m.config.masked_prediction = True
    sparse = m(ids, labels=labels)
    assert sparse["logits"].shape == (int((labels != -100).sum()), cfg.vocab_size)
    assert torch.allclose(full, sparse["loss"], atol=1e-6)


# ---------------------------------------------------------------- Stage 4: gradients
def test_stage4_gradients():
    torch.manual_seed(0)
    cfg = tiny()
    m = ModernBertPaperForMaskedLM(cfg)
    ids = rand_ids(cfg)
    m(ids, labels=ids)["loss"].backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    # NOTE: do not assert the padding row gradient is zero - the tied decoder writes into it.


# ---------------------------------------------------------------- Stage 5: padding
def test_stage5_padding_invariance():
    torch.manual_seed(0)
    cfg = tiny()
    m = ModernBertPaperModel(cfg).eval()
    ids = rand_ids(cfg, 1, 10)
    pad = torch.full((1, 6), cfg.pad_token_id)
    ref = m(ids)
    right = m(torch.cat([ids, pad], 1), torch.cat([torch.ones(1, 10), torch.zeros(1, 6)], 1).long())[:, :10]
    left = m(torch.cat([pad, ids], 1), torch.cat([torch.zeros(1, 6), torch.ones(1, 10)], 1).long())[:, 6:]
    assert max_err(ref, right) < 1e-5
    assert max_err(ref, left) < 1e-5  # RoPE is relative, so shifting all positions changes nothing


# ---------------------------------------------------------------- Stage 6: attention pattern
def test_stage6_global_local_pattern():
    cfg = ModernBertPaperConfig.base(num_hidden_layers=22)
    glob = [i for i in range(22) if cfg.is_global_layer(i)]
    assert glob == [0, 3, 6, 9, 12, 15, 18, 21]
    assert cfg.rope_theta_for_layer(0) == 160_000.0 and cfg.rope_theta_for_layer(1) == 10_000.0


def test_stage6_window_mask_matches_flash_attention_definition():
    cfg = tiny(sliding_window=8)
    att = ModernBertPaperSelfAttention(cfg, layer_id=1)
    assert not att.is_global
    allowed = att.build_attention_mask(20, None, "cpu")[0, 0]
    assert allowed.sum(1).tolist() == [5, 6, 7, 8] + [9] * 12 + [8, 7, 6, 5]
    i = torch.arange(20).view(-1, 1)
    j = torch.arange(20).view(1, -1)
    assert torch.equal(allowed, (j >= i - 4) & (j <= i + 4))  # FA2 window_size=(4, 4)
    assert allowed.diagonal().all(), "every token can attend to itself"
    glob = ModernBertPaperSelfAttention(cfg, layer_id=0).build_attention_mask(20, None, "cpu")
    assert glob.all()


# ---------------------------------------------------------------- Stage 7: equivalent paths
def test_stage7_attention_matches_sdpa():
    torch.manual_seed(0)
    cfg = tiny(attn_out_dropout_prob=0.0)
    for layer_id in (0, 1):
        att = ModernBertPaperSelfAttention(cfg, layer_id).eval()
        x = torch.randn(B, S, cfg.hidden_size)
        mask = torch.ones(B, S, dtype=torch.long)
        mask[1, -3:] = 0
        ours = att(x, mask)
        q, k, v = att.qkv_proj(x).chunk(3, -1)
        q, k, v = (t.view(B, S, cfg.num_attention_heads, -1).transpose(1, 2) for t in (q, k, v))
        q, k = att.rotary(q), att.rotary(k)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=att.build_attention_mask(S, mask, "cpu"))
        ref = att.out_proj(o.transpose(1, 2).reshape(B, S, cfg.hidden_size))
        assert max_err(ours, ref) < 1e-5


def test_stage7_mlp_matches_packed_wi():
    cfg = tiny()
    mlp = ModernBertPaperMLP(cfg).eval()
    x = torch.randn(B, S, cfg.hidden_size)
    wi = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0)  # reference Wi
    inp, gate = F.linear(x, wi).chunk(2, -1)
    ref = mlp.down_proj(F.gelu(inp) * gate)
    assert max_err(mlp(x), ref) < 1e-6


def test_init_scales():
    cfg = ModernBertPaperConfig.tiny(num_hidden_layers=8, hidden_size=256, num_attention_heads=4, intermediate_size=512)
    m = ModernBertPaperForMaskedLM(cfg)
    w_in = m.model.layers[1].attn.qkv_proj.weight
    w_out = m.model.layers[1].mlp.down_proj.weight
    assert abs(w_in.std().item() / 0.02 - 0.88) < 0.05        # truncated at 2 std -> std ~ 0.88 * 0.02
    assert abs(w_out.std().item() / (0.02 / 4.0) - 0.88) < 0.05  # 0.02 / sqrt(2 * 8)
    assert w_in.abs().max() <= 2 * 0.02 + 1e-6


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"{len(tests)} tests passed")
