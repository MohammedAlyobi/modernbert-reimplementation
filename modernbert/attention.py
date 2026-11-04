import torch
import torch.nn as nn

from .config import ModernBertPaperConfig

class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) for query/key pairs."""

    def __init__(self, dim: int, base: float = 10000.0, interleaved: bool = False):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even.")
        self.dim = dim
        self.base = base
        self.interleaved = interleaved 
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False) # after model.to(cuda) -> buffer = stored with the module but not learned

    def cos_sin(self, seq_len: int, device: torch.device):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        return freqs.cos(), freqs.sin()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos, sin = self.cos_sin(x.size(-2), x.device) # [S, dim/2], broadcasts over [B, heads]

        xf = x.float() # compute rotation in fp32
        if self.interleaved:
            x1, x2 = xf[..., 0::2], xf[..., 1::2] # split into even and odd dimensions
            out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2) # apply rotation
        else:
            half = self.dim // 2
            x1, x2 = xf[..., :half], xf[..., half:] # split into first half and second half
            out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1) # apply rotation
        return out.type_as(x) # convert back to original dtype

    
class ModernBertPaperSelfAttention(nn.Module):
    """Dense reference attention with RoPE and ModernBERT local/global masking.

    Global layers: every token attends to every (non-padding) token.
    Local layers:  token i attends to key j only if |i - j| <= sliding_window // 2,
                   identical to FlashAttention-2 window_size=(64, 64) used by the reference.
    """

    def __init__(self, config: ModernBertPaperConfig, layer_id: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_id = layer_id

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=config.attn_qkv_bias) # bias-free linear-layer.
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attn_out_bias) 
        # This projection transforms [batch, seq, hidden_size] again.
        # attention result -> output projection -> hidden states
        
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.out_dropout = nn.Dropout(config.attn_out_dropout_prob)

        self.is_global = config.is_global_layer(layer_id)
        self.window_radius = -1 if self.is_global else config.sliding_window // 2
        self.rotary = RotaryEmbedding(
            self.head_dim,
            base=config.rope_theta_for_layer(layer_id),  # 160k for global, 10k for local layers
            interleaved=config.rotary_emb_interleaved,
        )


    def build_attention_mask(self, seq_len: int, attention_mask: torch.Tensor | None, device) -> torch.Tensor:
        """Boolean [B or 1, 1, S, S] mask: True = query may attend to key."""
        allowed = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=device)
        if self.window_radius >= 0:
            idx = torch.arange(seq_len, device=device)
            distance = (idx.view(-1, 1) - idx.view(1, -1)).abs()        # [S, S]
            allowed = allowed & (distance <= self.window_radius).view(1, 1, seq_len, seq_len)
        if attention_mask is not None:
            key_is_real = attention_mask.bool().view(attention_mask.size(0), 1, 1, seq_len)
            allowed = allowed & key_is_real                               # padded keys are invisible
        return allowed

    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None): # input [batch, sequance, hidden] 
                                                                                    # optional attention mask 
                                                                                    # usually 1 -> real token 
                                                                                    #         0 -> padding
        batch_size, seq_len, _ = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1) # Each will have shape [batch, seq_len, hidden_size].
                                                    # [batch, seq_len, hidden_size] -> view [batch, seq, heads, head_dim] 
                                                    #                               -> transpose [batch, heads, seq, head_dim] 
                                                    # "std shape for multi-head-attention"                                
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) 
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.rotary(q)
        k = self.rotary(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # calc att scores, 
                                                                               # scaled dot-product attention equation
                                                                               # Q: [batch, heads, seq, head_dim]
                                                                               # K transpose: [batch, heads, head_dim, seq]

        allowed = self.build_attention_mask(seq_len, attention_mask, x.device) # [B or 1, 1, S, S] mask: True = query may attend to key.
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)  

        attn = self.attn_dropout(torch.softmax(scores, dim=-1)) # During training, this can randomly drop some attention probabilities.
        out = torch.matmul(attn, v) # equation part when mult by V
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.out_dropout(self.out_proj(out)) # The merged representation passes through the output projection. output linear