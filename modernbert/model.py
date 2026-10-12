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
        
        # apply() recursively visits the model and its submodules and calls:
        # self._init_weights(module), so it visits Embiddings, each Transformer block, and the final LayerNorm, initializing their weights according to the rules defined in _init_weights.
        self.apply(self._init_weights) 

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range) # init with normal distribution, mean=0, std=initializer_range=0.02
            if module.bias is not None: 
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range) # Same normal distribution is used for token embeddings.
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_() # This ensures the padding embedding remains zero.
        elif isinstance(module, nn.LayerNorm): # two learnable parameters:
            nn.init.ones_(module.weight) # init gamma to 1, so initially it doesn't scale the normalized output.
            nn.init.zeros_(module.bias) # init beta to 0, so initially it doesn't shift the normalized output.

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
        self.head_dense = nn.Linear(config.hidden_size, config.hidden_size) # H→H, no dim change like encoder
        self.head_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.head_pred_norm else nn.Identity()
        self.head_act = nn.GELU() if config.head_pred_act.lower() == "gelu" else nn.Identity()
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.head_pred_bias) # Vocab projection: This is where each hidden vector becomes a vocabulary prediction. [B,S,768] -> [B,S,30522]
        self.lm_head.weight = self.model.embeddings.token_embedding.weight # You're tying the input embedding weights and output vocabulary projection weights. So the same learned matrix is used for both.

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None
    ): 
        hidden= self.model(input_ids, attention_mask=attention_mask) # [B,S] -> [B,S,H]
        hidden = self.head_dense(hidden)
        hidden = self.head_act(hidden)
        hidden = self.head_norm(hidden)
        logits = self.lm_head(hidden) # [B,S,Hidden_size] -> [B,S,Vocab_size], Higher logit = model assigns a higher score to that token.
                                      # don't apply softmax
        '''
        F.cross_entropy() internally combines:

        $$ \operatorname{log\_softmax} $$

        and negative log likelihood.

        So passing already-softmaxed probabilities would actually be the wrong thing to do here.

        Your raw logits are correct.        
        '''
        loss = None
        if labels is not None:
            loss = F.cross_entropy( # cross entropy expects class dim in std form [N, C] not [B, S, V] 
                logits.reshape(-1, logits.size(-1)), # [B,S,V] into [B,S,V]
                labels.reshape(-1), # [B,S] into [B*S] to alight labels with logits, Each token position gets a target vocabulary ID.
                ignore_index = -100, # see down
            )

        return {"logits": logits, "loss": loss}
'''
This is essential for MLM.

Usually, you don't want to calculate loss for every token. You primarily want the positions selected for MLM prediction.

So labels can look conceptually like:

[-100, 2023, -100, -100, 3231, -100]

where:

-100 → don't calculate loss here

and actual token IDs mean:

calculate prediction loss here

Your cross_entropy call explicitly tells PyTorch:

ignore_index=-100

to ignore those positions.
'''                