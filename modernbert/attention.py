import torch
import torch.nn as nn

from .config import ModernBertPaperConfig

class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) for query/key pairs."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even.")

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, presistent=False) # after model.to(cuda) -> buffer = stored with the module but not learned

    def forward(self, x: torch.Tensor, seq_len: int | None = None):
        device = x.device # This matters because tensors involved in the same operation need to be on compatible devices.
        if seq_len is None:
            seq_len = x.size(-2)

        """Your tensor is: [batch, heads, seq, head_dim]

        So: x.size(-2) means: sequence dimension

        For example:

        [8, 12, 128, 64]
                ↑
            -2
        giving: 128
        """

        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype) # a list of positions [0, 1, 2, ..., seq_len-1]
        freqs = torch.einsum("s,d->sd", positions, self.inv_freq.to(device))

        '''
        So conceptually:

        positions
            ↓
        [seq]
            ×
        inverse frequencies
            ↓
        [dim/2]
            ↓
        [seq, dim/2]

        For example: positions = [0,1,2], inv_freq = [a,b,c]

        result:

        [
            [0a, 0b, 0c],
            [1a, 1b, 1c],
            [2a, 2b, 2c]
        ]

        These values are the rotation angles.
        
        '''

        cos = freqs.cos().unsqueeze(0).unsqueeze(0) # [1, 1, seq, dim/2]
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)

        '''
        
        Because your Q/K tensors are: [batch, heads, seq, dim]

        and you want the positional values to broadcast across:

        batch
        heads

        So:

        cos
        [1, 1, seq, dim/2]

                ↓ broadcasting
        q
        [batch, heads, seq, dim/2]        
        
        '''

        # x = [x₀, x₁, x₂, x₃, x₄, x₅], forming pairs: (x₀, x₁), (x₂, x₃), (x₄, x₅)
        x_even = x[..., 0::2] # [x₀, x₂, x₄]
        x_odd = x[..., 1::2]  # [x₁, x₃, x₅]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        '''
        This is the actual RoPE operation.

        It's the standard 2D rotation: x1' = x1 * cos(θ) - x2 * sin(θ), x2' = x1 * sin(θ) + x2 * cos(θ)

        where θ is the rotation angle for that position and dimension.
        
        Every pair of dimensions gets rotated by a position-dependent angle.
        '''

        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

        '''
        You had:

        rotated_even
        [batch, heads, seq, dim/2]

        rotated_odd
        [batch, heads, seq, dim/2]

        stack(..., dim=-1) creates:

        [batch, heads, seq, dim/2, 2]

        So the pairs become:

        (even₀, odd₀)
        (even₁, odd₁)
        ...

        Then:

        .flatten(-2)

        returns it to:

        [batch, heads, seq, dim]

        So RoPE doesn't change the shape.

        That's an important invariant.        
        
        '''

class ModernBertPaperSelfAttention(nn.Module):
    """Reference attention with RoPE and ModernBERT-style local/global masking."""

    def __init__(self, config: ModernBertPaperConfig, layer_id: int = 0):
        super.__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self,self.hidden_size // self.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hideen_size must be divisble by num_attentions_heads")

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False) # This matches the bias-free linear-layer design to reproduce.
        # Instead of creating three separate layers for QKV, we can create a single linear layer 
        # that outputs all three concatenated together. This is more efficient and allows for 
        # better optimization during training. if hidden = 768, then 3*hidden = 2304, so the output of 
        # this layer will be [batch, seq_len, 2304], which we can then split into Q, K, V.
                
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False) 
        # This projection transforms [batch, seq, hidden_size] again.
        # attention result -> output projection -> hidden states
        
        self.rotary = RotaryEmbedding(self.head_dim, base=config.rotary_emb_base) # apply rotary to each head_dim
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob) # apply dropout to attention probabilities after softmax
        self.sliding_window = config.sliding_window # store local attention settings, tells attention how large the local window should be.
        self.is_global = (
            config.global_attn_every_n_layers > 0 and layer_id % config.global_attn_every_n_layers == 0 # alternate between local, global attention
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None): # input [bbatch, sequance, hidden] 
                                                                                    # optional attention mask 
                                                                                    # usually 1 -> real token 
                                                                                    # 0 -> padding
        batch_size, seq_len, _ = x.size()

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1) # Split the concatenated QKV tensor into three separate tensors along the last dimension. 
                                       # Each will have shape [batch, seq_len, hidden_size]. first column is Q, then K, then V

        # [batch, seq_len, hidden_size] -> view [batch, seq, heads, head_dim] -> transpose [batch, heads, seq, head_dim] 
        # "std shape for multi-head-attention"                                
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) 
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.rotary(q, seq_len=seq_len)
        k = self.rotary(k, seq_len=seq_len)
        '''
        This is the crucial ModernBERT positional operation.

        Notice:

        V is not rotated.

        Only Q and K receive RoPE.

        Why?

        Because positional information affects the calculation of attention relationships:

        $$ QK^T $$

        while V contains the information that gets aggregated.

        So:

        Q → RoPE
        K → RoPE
        V → unchanged    
        '''

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # calc att scores, 
                                                                               # scaled dot-product attention equation
                                                                               # Q: [batch, heads, seq, head_dim]
                                                                               # K transpose: [batch, heads, head_dim, seq]

        if not self.is_global and self.sliding_window > 0: # local attention
            q_idx = torch.arange(seq_len, device=x.device).view(1, 1, seq_len, 1) # reshape so it broadcasts against the attention-score tensor. varies down the rows
            k_idx = torch.arange(seq_len, device=x.device).view(1, 1, 1, seq_len) # Now the dimensions line up so you can compare: q pos vs k pos. varies across the columns
            radius = self.sliding_window // 2 # 128 // 2 = 64 before + itself + 64 after 
            valid = (torch.abs(k_idx - q_idx) <= radius).expand_as(scores)
            '''
            36. Create the local mask
            valid = (
                torch.abs(k_idx - q_idx) <= radius
            ).expand_as(scores)

            For each query/key pair:

            |key_position - query_position| <= radius

            means:

            allowed

            Otherwise:

            not allowed

            Conceptually:

                    keys →
                0 1 2 3 4 5 6
            q=0    ✓ ✓ ✓
            q=1    ✓ ✓ ✓ ✓
            q=2      ✓ ✓ ✓ ✓
            q=3        ✓ ✓ ✓ ✓
            ...

            This creates the local attention pattern.
            '''
            scores = scores.masked_fill(~valid, torch.info(scores.dtype).min)
            '''
            37. Mask invalid scores
            scores = scores.masked_fill(
                ~valid,
                torch.finfo(scores.dtype).min
            )

            For forbidden attention positions, you replace the score with a very negative number.

            Why?

            Because after softmax:

            softmax(very negative)
            ≈ 0

            So those connections effectively disappear.

            This is a standard attention masking technique.
            '''

        if attention_mask is not None: # If the caller supplied a mask, use it.
            key_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [batch, seq] -> [batch, 1, 1, seq], 
                                                                # allow mask to broadcast across heads, query positions 
                                                                # Apply the mask to the attention scores
            scores = scores.masked_fill(key_mask == 0, torch.finfo(scores.dtype).min) # So padded keys cannot be attended to.
                                                                                      # finfo(dtype).min rather than -inf because -inf can produce NaN 
                                                                                      # when an entire row is masked

        attn = torch.softmax(scores, dim=-1) # attention scores become probabilities. across all keys, sum = 1
        attn = self.dropout(attn) # During training, this can randomly drop some attention probabilities.
        out = torch.matmul(attn, v) # equation part when mult by V
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        '''
        42. Merge heads
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.hidden_size
            )
        )

        Before:

        [batch, heads, seq, head_dim]

        Transpose:

        [batch, seq, heads, head_dim]

        Then flatten:

        [batch, seq, hidden]

        because:

        heads × head_dim = hidden_size        
        '''
        return self.out_proj(out) # The merged representation passes through the output projection. output linear


'''
One important issue before we move on

There is one thing I would not build further on without keeping in mind:

Your current attention implementation is a reference attention implementation, not yet the final optimized ModernBERT attention.

That's actually appropriate for your roadmap.

You're deliberately doing:

Stage 0/1
reference padded attention
        ↓
verify correctness
        ↓
later:
FlashAttention
unpadding
sequence packing

So don't try to add FlashAttention here yet.

However, there is also a more substantive architectural detail to revisit later: the exact definition of the ModernBERT local/global attention pattern and window semantics should be checked against the reference implementation before you call this paper-faithful. Your current radius = sliding_window // 2 is a reasonable educational local-attention interpretation, but it shouldn't automatically be assumed to be the exact repository convention.

For now, that's not a blocker for learning the mechanics.

The key concepts you should now understand

You have essentially implemented:

Multi-head self-attention
QKV projection
head splitting/merging
scaled dot-product attention
RoPE
local attention masking
padding masking
bias-free attention projections
'''
