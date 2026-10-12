from dataclasses import asdict, dataclass


@dataclass  # Without it you'd hand-write thirty lines of self.vocab_size = vocab_size
            # 3 A field only becomes a dataclass field if it has a type annotation. 
            # vocab_size = 30522 with no : int would be ignored by the decorator and stay a plain class attribute.
class ModernBertPaperConfig:
    vocab_size: int = 30522 # should be = len(tokenizer)
    hidden_size: int = 768  # constant hidden dimention through Transformer blocks
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 8192 # max sequance/context length supported by the model
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    pad_token_id: int = 0
    embedding_layer: str = "sans_pos"
    attention_layer: str = "rope"
    sliding_window: int = 128
    global_attn_every_n_layers: int = 3
    rotary_emb_base: float = 10000.0
    head_pred_act: str = "gelu"
    head_pred_norm: bool = True
    head_pred_bias: bool = False

    attn_qkv_bias: bool = False
    attn_out_bias: bool = False
    mlp_in_bias: bool = False
    mlp_out_bias: bool = False

    def to_dict(self):
        return asdict(self)
