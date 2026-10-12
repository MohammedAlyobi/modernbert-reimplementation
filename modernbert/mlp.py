import torch
import torch.nn as nn
from torch.nn import functional as F # GELU is an activation function, so there is no trainable state associated with it.

from .config import ModernBertPaperConfig


class ModernBertPaperMLP(nn.Module): # This creates your MLP as a standard PyTorch module. cause inherits from nn.Module.
    """GeGLU-style MLP aligned with ModernBERT reference design. GELU Gated Linear Unit"""

    def __init__(self, config: ModernBertPaperConfig): # GeGLU(x)=GELU(Wg​x)⊙Wu​x "wg = gate projection, wu = up projection, op is elementwise mult"
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size,  bias=False) # called gate cause after applying GELU, 
                                                                                              # it controls how much information from the other projection gets through.
        '''
        This is the first important part.

        Suppose:

        hidden_size = 768
        intermediate_size = 3072

        Then this layer performs:

        $$ x \rightarrow W_gx $$

        with:

        $$ W_g \in \mathbb{R}^{3072 \times 768} $$

        So if:

        x: [B, S, 768]

        then:

        gate_proj(x): [B, S, 3072]

        The output will eventually become the gate.        
        '''
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        '''
        This is another projection from:

        $$ H \rightarrow I $$

        So:

        x:                 [B, S, H]
        up_proj(x):        [B, S, I]

        Notice that both projections receive the same input.

        But they have different learned weight matrices:

        $$ W_g \neq W_u $$

        So:

        $$ W_gx $$

        and

        $$ W_ux $$

        produce different representations.        
        '''
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        '''
        After the two branches are multiplied together, we're still in the intermediate dimension:

        [B, S, intermediate_size]

        The down projection brings it back:

        [B, S, intermediate_size]
                │
                ▼
            down_proj
                │
                ▼
        [B, S, hidden_size]

        Mathematically:

        $$ W_d \in \mathbb{R}^{H \times I} $$
        9. Why bias=False everywhere?

        You have:

        bias=False

        for all three Linear layers.

        This is intentional for your ModernBERT-style implementation.

        So instead of:

        $$ Wx+b $$

        you have:

        $$ Wx $$

        This is one of the architectural details you identified earlier as a ModernBERT design feature: bias-free linear layers.

        This also applies to your attention projections:

        qkv_proj
        out_proj

        which you implemented with bias=False.

        So your MLP and attention are consistent.        
        '''
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.gate_proj(x)) * self.up_proj(x)
        hidden = self.down_proj(hidden)
        return self.dropout(hidden)
