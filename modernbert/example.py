import torch

from .config import ModernBertPaperConfig
from .model import ModernBertPaperForMaskedLM


def build_paper_reference_example():
    """Small deterministic smoke test for the reference implementation."""
    torch.manual_seed(42)

    config = ModernBertPaperConfig.tiny()
    model = ModernBertPaperForMaskedLM(config)
    model.eval()

    input_ids = torch.randint(1, config.vocab_size, (2, 12))
    attention_mask = torch.ones_like(input_ids)

    mask = torch.rand(input_ids.shape) < 0.25
    mask[:, 0] = True  # guarantee at least one masked position per row (all -100 labels -> NaN loss)
    labels = torch.full_like(input_ids, -100)
    masked_input = input_ids.clone()
    masked_input[mask] = config.pad_token_id  # stand-in for a [MASK] token in this toy example
    labels[mask] = input_ids[mask]

    out = model(input_ids=masked_input, attention_mask=attention_mask, labels=labels)

    assert out["logits"].shape == (2, 12, config.vocab_size)
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() > 0.0

    return config, model, out
