"""Actual HF parity gates: automatically skipped when the optional package is absent.

Random tiny configurations exercise architecture code, not pretrained semantics.
These gates must ALSO pass against each pinned real local checkpoint in M0.
"""
import pytest
import torch
from drift.runtime.decoder import FrozenDecoder

hf = pytest.importorskip("transformers", reason="optional HF adapter dependency not installed")


@pytest.mark.parametrize("family", ["qwen3", "llama", "llama3_rope"])
def test_stock_hf_forward_parity(family):
    if hf.__version__ != "4.56.2":
        pytest.fail("requalify this adapter before using a different Transformers version")
    common = dict(vocab_size=97, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=256, attention_dropout=0.0)
    if family == "qwen3":
        config = hf.Qwen3Config(head_dim=8, **common)
        model = hf.Qwen3ForCausalLM(config)
    else:
        if family == "llama3_rope":
            common["rope_scaling"] = {"rope_type": "llama3", "factor": 8.0,
                "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 128}
        config = hf.LlamaConfig(**common)
        model = hf.LlamaForCausalLM(config)
    model.config._attn_implementation = "eager"
    model.eval()
    ids = torch.tensor([1, 3, 5, 7, 9, 2])
    with torch.no_grad():
        stock = model(input_ids=ids[None], use_cache=False).logits[0]
        adapter = FrozenDecoder(model)
        ours = adapter.forward(ids).logits
        prefix = adapter.forward(ids[:4])
        transplanted = adapter.import_self_prefix(prefix.canonical_delta, torch.arange(4))
        continuation = adapter.forward(ids[4:], transplanted)
    torch.testing.assert_close(ours, stock, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(continuation.logits, stock[4:], atol=2e-5, rtol=2e-5)
