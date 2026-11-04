import torch
import torch.nn as nn
from torch.nn import functional as F # GELU is an activation function, so there is no trainable state associated with it.

from .config import ModernBertPaperConfig


class ModernBertPaperMLP(nn.Module): # This creates your MLP as a standard PyTorch module. cause inherits from nn.Module.
    """GeGLU MLP:  down( dropout( GELU(gate(x)) * up(x) ) ).

    intermediate_size is the width of EACH branch (1152 for base). The reference stores both
    branches in one matrix Wi of shape [2 * intermediate_size, hidden] and splits it with chunk(2):
        gate_proj.weight == Wi.weight[:intermediate_size]   (the half that goes through GELU)
        up_proj.weight   == Wi.weight[intermediate_size:]   (the linear "gate" half in the reference)
    """

    def __init__(self, config: ModernBertPaperConfig): # GeGLU(x)=GELU(Wg​x)⊙Wu​x "wg = gate projection, wu = up projection, op is elementwise mult"
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size,  bias=config.mlp_in_bias) # called gate cause after applying GELU, 
                                                                                              # it controls how much information from the other projection gets through.
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_in_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)
        self.dropout = nn.Dropout(config.mlp_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(hidden))           # dropout BEFORE down_proj, as in reference
