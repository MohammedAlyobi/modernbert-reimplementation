'''
                  INPUT
                    │
                    ▼
              input_ids [B,S]
                    │
                    ▼
               Embeddings
                    │
                    ▼
                [B,S,H]
                    │
                    ▼
        ┌───────────────────────┐
        │   Transformer Block   │
        │                       │
        │ Pre-LN → Attention    │
        │      + residual       │
        │ Pre-LN → GeGLU MLP    │
        │      + residual       │
        └───────────────────────┘
                    │
                  repeat
                    │
                    ▼
              Final LayerNorm
                    │
                    ▼
                [B,S,H]
                    │
                    ▼
              MLM head dense
                    │
                    ▼
                  GELU
                    │
                    ▼
               Head LayerNorm
                    │
                    ▼
             LM vocabulary head
                    │
                    ▼
              logits [B,S,V]
                    │
             ┌──────┴──────┐
             │             │
          no labels      labels
             │             │
             ▼             ▼
        return logits   CrossEntropy
                           │
                           ▼
                          loss
'''


import torch
import torch.nn as nn
import torch.nn.functional as F # functional operations -> cross_entropy

from .block import ModernBertPaperTransformerBlock
from .config import ModernBertPaperConfig
from .embeddings import ModernBertPaperEmbeddings
from .initialization import init_modernbert_weights

'''
ModernBertPaperModel

input IDs
   ↓
encoder
   ↓
hidden states
'''
class ModernBertPaperModel(nn.Module): # input IDs → contextual hidden representations
    """Compact encoder-only ModernBERT-style model."""

    def __init__(self, config: ModernBertPaperConfig):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertPaperEmbeddings(config) # [B, S] → [B, S, H]
        self.layers = nn.ModuleList( # These are neural-network submodules whose parameters need to be tracked, not only normal list
            # create n blocks, each with its own attention and MLP sublayers
            [ModernBertPaperTransformerBlock(config, layer_id = i) for i in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # After all Transformer blocks, you apply another normalization:
        
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=config.norm_bias)
        init_modernbert_weights(self, config) 

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None): # [B, S] → [B, S, H]
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids to be of shape [batch, sequence_length], but got {input_ids.shape}")
        if input_ids.size(1) > self.config.max_position_embeddings: # So here max_position_embeddings is effectively acting as your maximum supported sequence length/context limit. CAUSE WE USE Rope not absolute position embeddings, so we can support longer sequences than the model was trained on.
            raise ValueError(f"Sequence length {input_ids.size(1)} exceeds maximum position embeddings {self.config.max_position_embeddings}")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError(f"Expected attention_mask to be of shape {tuple(input_ids.shape)}, but got {tuple(attention_mask.shape)}")

        x = self.embeddings(input_ids) # [B, S] → [B, S, H]
        if attention_mask is None: # If the user doesn't provide a mask, you assume: Every token is valid.
            attention_mask = torch.ones_like(input_ids) # [1, 1] length = input_ids

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask) # pass through blocks

        return self.norm(x) # final normalization after all blocks, [B, S, H], inside block is embeddings

'''
ModernBertPaperForMaskedLM

input IDs
   ↓
encoder
   ↓
MLM head
   ↓
vocabulary logits
   ↓
cross entropy loss
'''
class ModernBertPaperForMaskedLM(nn.Module): # wraps the encoder with a prediction head. The pretraining model:
    """ModernBERT-style model with a masked language modeling head on top."""

    def __init__(self, config: ModernBertPaperConfig):
        super().__init__()
        self.config = config
        self.model = ModernBertPaperModel(config) # create the encoder
        self.head_dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.head_pred_bias) # H→H, no dim change like encoder
        self.head_act = nn.GELU() if config.head_pred_act.lower() == "gelu" else nn.Identity()
        self.head_norm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) 
            if config.head_pred_norm 
            else nn.Identity()
        )
        # Decoder bias: the paper keeps this ONE bias to soften the effect of weight tying.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias) # Vocab projection: This is where each hidden vector becomes a vocabulary prediction. [B,S,768] -> [B,S,30522]
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.token_embedding.weight # You're tying the input embedding weights and output vocabulary projection weights. So the same learned matrix is used for both.
        # Initialize the whole model (encoder again + head) with the ModernBERT scheme. 
        init_modernbert_weights(self, config) 

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.head_norm(self.head_act(self.head_dense(hidden))))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None
    ): 
        hidden= self.model(input_ids, attention_mask=attention_mask) # [B,S] -> [B,S,H]
        
        if self.config.masked_prediction and labels is not None:
            # Reference speed trick: only masked positions need vocabulary logits.
            # Returns logits of shape [num_masked, V] instead of [B, S, V]; the loss is identical.
            flat_labels = labels.reshape(-1) # [B*S]
            selected = flat_labels != -100 # [B*S] boolean mask: True = calculate loss here
            logits = self.head(hidden.reshape(-1, hidden.size(-1)))[selected] # [B*S, H] -> [num_masked, V]
            loss = F.cross_entropy(logits, flat_labels[selected]) # only calculate loss for masked positions
            return {"logits": logits, "loss": loss}

        logits = self.head(hidden) # [B,S,H] -> [B,S,V]
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100) 
        return {"logits": logits, "loss": loss}
