# modernbert — ModernBERT reference reimplementation

A readable, padded, dense-attention implementation of ModernBERT that is numerically identical
to the Hugging Face `ModernBertForMaskedLM` (max abs error ~1e-7 in fp32, see `compare_with_hf.py`).

Defaults reproduce **ModernBERT-base** (22 layers, hidden 768, GeGLU 2 x 1152, vocab 50368,
149.6M parameters). Use `ModernBertPaperConfig.tiny()` for experiments.

## What is ModernBERT-specific here

- sans-position embeddings: token embedding -> LayerNorm -> dropout (positions come from RoPE)
- no bias in any linear layer or LayerNorm, except the final decoder bias
- LayerNorm eps 1e-5
- pre-norm blocks, with the first block's attention LayerNorm removed (`skip_first_prenorm`)
- alternating attention: global every 3rd layer (0, 3, 6, ...), others local with +/- 64 tokens
- RoPE theta 160,000 on global layers and 10,000 on local layers, half-split ("non-interleaved") layout
- GeGLU MLP with `intermediate_size` = width of each branch
- dropout 0.1 after the attention output projection only
- MLM head: dense -> GELU -> LayerNorm -> tied vocabulary projection (+ bias)
- "full_megatron" init: truncated normal, output projections scaled by 1/sqrt(2 * num_layers)

## File map

| File | Contents |
|---|---|
| `config.py` | `ModernBertPaperConfig` + presets `base()`, `large()`, `pretraining_phase()`, `tiny()` |
| `embeddings.py` | `ModernBertPaperEmbeddings` |
| `attention.py` | `RotaryEmbedding` (both RoPE layouts) and `ModernBertPaperSelfAttention` (local/global mask) |
| `mlp.py` | `ModernBertPaperMLP` (GeGLU) |
| `block.py` | `ModernBertPaperTransformerBlock` (pre-norm, skip-first-prenorm) |
| `initialization.py` | `init_modernbert_weights` (full_megatron) |
| `model.py` | `ModernBertPaperModel` (encoder) and `ModernBertPaperForMaskedLM` (MLM head, optional masked_prediction) |
| `example.py` | `build_paper_reference_example()` smoke test |
| `compare_with_hf.py` | weight mapping + numerical comparison with Hugging Face ModernBERT |

## Running

From the `modernbert-reimplementation` folder:

```bash
python -m pytest tests/test_modularized_code.py -v          # staged tests (Stages 0-7)
```

Also from the `modernbert-reimplementation` folder (`pip install transformers` needed for the comparison):

```bash
python -c "from modernbert import build_paper_reference_example as b; c, m, o = b(); print(o['logits'].shape, float(o['loss']))"
python -m modernbert.compare_with_hf
python -m modernbert.compare_with_hf --pretrained answerdotai/ModernBERT-base
```

## Weight-name mapping to Hugging Face / reference

| Ours | HF `ModernBertForMaskedLM` |
|---|---|
| `model.embeddings.token_embedding` | `model.embeddings.tok_embeddings` |
| `model.layers.i.norm1` / `norm2` | `model.layers.i.attn_norm` / `mlp_norm` |
| `model.layers.i.attn.qkv_proj` / `out_proj` | `model.layers.i.attn.Wqkv` / `Wo` |
| `model.layers.i.mlp.gate_proj` + `up_proj` | `model.layers.i.mlp.Wi` (first half, second half) |
| `model.layers.i.mlp.down_proj` | `model.layers.i.mlp.Wo` |
| `model.norm` | `model.final_norm` |
| `head_dense` / `head_norm` | `head.dense` / `head.norm` |
| `lm_head` | `decoder` |

## Still simplified (on purpose)

- dense `[B, heads, S, S]` attention with an explicit mask (no FlashAttention, no unpadding, no packing)
- RoPE cos/sin recomputed every call
- the padded sequence path only; the reference trains unpadded with sequence packing
