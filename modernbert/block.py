# The code is compact, but conceptually this is one of the most important files 
# because it defines the computation repeated num_hidden_layers times. the Pre-LayerNorm structure.

'''
config.py
   │
   ├──────────────┐
   ▼              ▼
attention.py    mlp.py
   │              │
   └──────┬───────┘
          ▼
       block.py
'''

import torch 
import torch.nn as nn

from .attention import ModernBertPaperSelfAttention
from .config import ModernBertPaperConfig
from .mlp import ModernBertPaperMLP


class ModernBertPaperTransformerBlock(nn.Module): # model.py will call block times = num_hidden_layers
    """Pre-LayerNorm block:  x1 = x0 + Attn(LN(x0));  x2 = x1 + MLP(LN(x1)).

    Layer 0 skips its attention LayerNorm (skip_first_prenorm) because the embedding output was
    just normalized - applying LN twice in a row would be redundant (paper section 2.1.1).
    """

    def __init__(self, config: ModernBertPaperConfig, layer_id: int = 0): # layer_id for global or local atten
        super().__init__()
        if config.skip_first_prenorm and layer_id == 0:
            self.norm1 = nn.Identity() # skip first prenorm, as in reference
        else:
            self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=config.norm_bias) # normalizes the hidden representation before attention.
                                                                                 # LayerNorm operates over the final dimension: [B, S, H] 
                                                                                 # normalize each token's H-dimensional representation
        
        self.attn = ModernBertPaperSelfAttention(config, layer_id=layer_id) # sub layer 1
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=config.norm_bias) # This is the LayerNorm before the MLP.
        self.mlp = ModernBertPaperMLP(config) # sub layer 2

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None):
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x



'''
18. Why the residual is outside the LayerNorm

Notice your code:

residual = x
x = self.norm1(x)
x = self.attn(x)
x = residual + x

rather than:

x = self.attn(x)
x = self.norm1(x)
x = residual + x

That difference matters.

Your implementation is:

$$ x + F(\operatorname{LN}(x)) $$

which is Pre-LN.

A Post-LN design would look more like:

$$ \operatorname{LN}(x + F(x)) $$

You're implementing the former.
'''