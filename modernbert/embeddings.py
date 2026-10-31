# Embeddings give the model token representations. Attention + RoPE gives those representations positional awareness.
# this file is coherent for the current educational/reference path. The main thing to keep in mind is that 
# padding handling here is only the embedding side; actual padding masking/unpadding comes later.


import torch
import torch.nn as nn

from .config import ModernBertPaperConfig

class ModernBertPaperEmbeddings(nn.Module):
    """ModernBERT-style token embeddings without learned absolute positions."""
    # RoPE will enter when we implement attention, not here.

    def __init__(self, config: ModernBertPaperConfig):
        super().__init__()

        # lookup table with dim [30522, 768]
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=config.norm_bias)
        self.dropout = nn.Dropout(config.embed_dropout_prob)

    def forward(self, input_ids: torch.tensor) -> torch.Tensor:
        # This performs the lookup, dim [30522, 768] -> [batch_size, seq_len, hidden_size]
        x = self.token_embedding(input_ids)

        x = self.norm(x)    # change values not shape, dim [batch_size, seq_len, hidden_size]
        x = self.dropout(x) # after this go to transformer
        return x