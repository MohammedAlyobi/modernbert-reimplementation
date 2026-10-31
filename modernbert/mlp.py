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
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_in_bias)
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

        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)
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
        self.dropout = nn.Dropout(config.mlp_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(hidden))           # dropout BEFORE down_proj, as in reference
