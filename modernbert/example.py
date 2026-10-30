import torch
from .config import ModernBertPaperConfig
from .model import ModernBertPaperForMaskedLM


def build_paper_reference_example():
    """Small deterministic smoke test for the reference implmentation"""
    torch.manual_seed(42)

    config = ModernBertPaperConfig.tiny()
    model = ModernBertPaperForMaskedLM(config)
    model.eval()  # disable dropout for deterministic output

    input_ids = torch.randint(1, config.vocab_size, (2,12))
    attention_mask = torch.ones_like(input_ids)

    mask = torch.rand(input_ids.shape) < 0.25
    mask[:, 0] = True # guarantee at least one masked position per row (all -100 labels -> NaN loss)