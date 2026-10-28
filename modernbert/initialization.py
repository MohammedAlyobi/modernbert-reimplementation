"""ModernBERT weight initialization ("full_megatron" in the reference repo).

Every weight is drawn from a truncated normal N(0, std) cut at +/- init_cutoff_factor * std:

    module type   | layers                                  | std
    --------------|-----------------------------------------|-------------------------------
    "in"          | qkv_proj, gate_proj, up_proj, head_dense| initializer_range            (0.02)
    "out"         | out_proj, down_proj                     | 0.02 / sqrt(2 * num_layers)  (~0.003 for 22 layers)
    "emb"         | token_embedding                         | initializer_range
    "final_out"   | lm_head (only if NOT tied)              | hidden_size ** -0.5

Output projections write into the residual stream; shrinking them by 1/sqrt(2L) keeps the
variance of the residual stream from growing with depth (2 residual writes per layer).
Biases start at 0, LayerNorm weights at 1.
"""

import math
import torch.nn as nn
from .config import ModernBertPaperConfig

IN_MODULES = ("qkv_proj", "gate_proj", "up_proj", "head_dense")
OUT_MODULES = ("out_proj", "down_proj")


def _trunc_normal_(weight, std: float, cutoff: float):
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-cutoff * std, b=cutoff * std)


def init_modernbert_weights(model: nn.Module, config: ModernBertPaperConfig) -> None:
    stds = {
        "in": config.initializer_range,
        "out": config.initializer_range / math.sqrt(2.0 * config.num_hidden_layers),
        "emb": config.initializer_range,
        "final_out": config.hidden_size ** -0.5,
    }
    cutoff = config.init_cutoff_factor
    tied = config.tie_word_embeddings

    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(module, nn.Embedding):
            # Note: like the reference, this overwrites the padding row too (no practical effect:
            # padding_idx still blocks the lookup gradient, and pad keys are masked in attention).
            _trunc_normal_(module.weight, stds["emb"], cutoff)
        elif isinstance(module, nn.Linear):
            if leaf in IN_MODULES:
                _trunc_normal_(module.weight, stds["in"], cutoff)
            elif leaf in OUT_MODULES:
                _trunc_normal_(module.weight, stds["out"], cutoff)
            elif leaf == "lm_head":
                if not tied:  # tied decoder shares the embedding matrix, already initialized
                    _trunc_normal_(module.weight, stds["final_out"], cutoff)
            else:
                raise RuntimeError(f"No init rule for linear layer '{name}'")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.reset_parameters()  # weight = 1, bias = 0 (if present)
