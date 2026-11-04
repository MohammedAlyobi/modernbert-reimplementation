import torch
import torch.nn as nn

from .config import ModernBertPaperConfig

class ModernBertPaperEmbeddings(nn.Module):
    """ModernBERT-style token embeddings without learned absolute positions."""

    def __init__(self, config: ModernBertPaperConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=config.norm_bias)
        self.dropout = nn.Dropout(config.embed_dropout_prob)

    def forward(self, input_ids: torch.tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        x = self.norm(x)
        x = self.dropout(x) 
        return x