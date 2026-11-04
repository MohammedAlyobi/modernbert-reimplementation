from dataclasses import asdict, dataclass


@dataclass # A field only becomes a dataclass field if it has a type annotation. 
class ModernBertPaperConfig:
    
    # --- sizes --------------------------------------------------------------------------
    vocab_size: int = 50368 # should be = len(tokenizer)
    hidden_size: int = 768  # constant hidden dimention through Transformer blocks
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    intermediate_size: int = 1152
    max_position_embeddings: int = 8192 # max sequance/context length supported by the model

    # --- normalization -------------------------------------------------------------------------- 
    layer_norm_eps: float = 1e-5
    norm_bias: bool = False
    skip_first_prenorm: bool = True # if True, skip the first prenorm in the first block. This is a ModernBERT design choice.
    
    # --- dropout (reference pretraining vals) -------------------------------------------------------------------------- 
    embed_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0   # dropout on softmax probabilities
    attn_out_dropout_prob: float = 0.1          # dropout after the attention output projection
    mlp_dropout_prob: float = 0.0               # dropout inside the MLP, before the down projection

    # --- bias switches (paper: no bias in any linear layer except the final decoder) ------
    attn_qkv_bias: bool = False
    attn_out_bias: bool = False
    mlp_in_bias: bool = False
    mlp_out_bias: bool = False

    # --- attention pattern ---------------------------------------------------------------
    sliding_window: int = 128            # TOTAL local window; each token sees +/- 64 neighbours
    global_attn_every_n_layers: int = 3  # layers 0, 3, 6, ... are global
    rotary_emb_base: float = 160_000.0   # RoPE theta for GLOBAL layers
    local_attn_rotary_emb_base: float = 10_000.0  # RoPE theta for LOCAL layers (-1 = same as global)
    rotary_emb_interleaved: bool = False # False = rotate first half vs second half (reference)

    # --- MLM head --------------------------------------------------------------------------
    head_pred_act: str = "gelu"
    head_pred_norm: bool = True
    head_pred_bias: bool = False         # bias of the head's dense layer
    decoder_bias: bool = True            # bias of the final vocabulary projection
    tie_word_embeddings: bool = True
    masked_prediction: bool = False      # True = run head/decoder only on positions with labels

    # --- initialization ("full_megatron" in the reference) --------------------------------
    initializer_range: float = 0.02
    init_cutoff_factor: float = 2.0      # truncate the normal at +/- 2 std
    
    # --- tokens ---------------------------------------------------------------------------------
    pad_token_id: int = 50283            # ModernBERT tokenizer [PAD]; set from tokenizer.pad_token_id

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if (self.hidden_size // self.num_attention_heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.global_attn_every_n_layers > 0 and self.sliding_window <= 0:
            raise ValueError("global_attn_every_n_layers requires sliding_window > 0")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be a valid token id")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def is_global_layer(self, layer_id: int) -> bool:
        """Global attention every N layers, starting at layer 0 (reference: layer_id % N == 0)."""
        if self.global_attn_every_n_layers <= 0:
            return self.sliding_window <= 0
        return layer_id % self.global_attn_every_n_layers == 0

    def rope_theta_for_layer(self, layer_id: int) -> float:
        if self.is_global_layer(layer_id) or self.local_attn_rotary_emb_base == -1:
            return self.rotary_emb_base
        return self.local_attn_rotary_emb_base

    # --- presets ----------------------------------------------------------------------------
    @classmethod
    def base(cls, **overrides):
        return cls(**overrides)

    @classmethod
    def large(cls, **overrides):
        kw = dict(hidden_size=1024, num_hidden_layers=28, num_attention_heads=16, intermediate_size=2624)
        kw.update(overrides)
        return cls(**kw)

    @classmethod
    def pretraining_phase(cls, **overrides):
        """First (1024-token) pretraining phase: theta 10k everywhere, per yamls/modernbert-*-pretrain."""
        kw = dict(max_position_embeddings=1024, rotary_emb_base=10_000.0, local_attn_rotary_emb_base=-1)
        kw.update(overrides)
        return cls(**kw)

    @classmethod
    def tiny(cls, **overrides):
        """Small config for unit tests and debugging (same architecture, tiny sizes)."""
        kw = dict(vocab_size=64, hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                  intermediate_size=48, max_position_embeddings=128, sliding_window=8,
                  global_attn_every_n_layers=2, pad_token_id=0)
        kw.update(overrides)
        return cls(**kw)

    def to_dict(self):
        return asdict(self)
