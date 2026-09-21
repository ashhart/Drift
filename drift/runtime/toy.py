"""Random small decoder for plumbing tests. It is NOT a trained language model."""
from types import SimpleNamespace
import torch
from torch import nn


class ToyRotary(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, x, position_ids):
        freq = 10000.0 ** (-torch.arange(0, self.dimension, 2,
                                       device=x.device).float() / self.dimension)
        phase = position_ids.float()[..., None] * freq
        phase = torch.cat((phase, phase), dim=-1)
        return phase.cos().to(x.dtype), phase.sin().to(x.dtype)


class ToyAttention(nn.Module):
    def __init__(self, width, qheads, kvheads, dim):
        super().__init__()
        self.q_proj = nn.Linear(width, qheads * dim, bias=False)
        self.k_proj = nn.Linear(width, kvheads * dim, bias=False)
        self.v_proj = nn.Linear(width, kvheads * dim, bias=False)
        self.o_proj = nn.Linear(qheads * dim, width, bias=False)
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.scaling = dim**-0.5


class ToyLayer(nn.Module):
    def __init__(self, width, qheads, kvheads, dim):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(width)
        self.self_attn = ToyAttention(width, qheads, kvheads, dim)
        self.post_attention_layernorm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(),
                                 nn.Linear(width * 2, width))


class ToyModel(nn.Module):
    def __init__(self, seed: int = 7, width: int = 24, qheads: int = 4,
                 kvheads: int = 2, dim: int = 6, layers: int = 3, vocab: int = 41):
        super().__init__()
        if dim % 2 or qheads % kvheads:
            raise ValueError("invalid toy dimensions")
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(vocab, width)
            self.model.layers = nn.ModuleList(
                ToyLayer(width, qheads, kvheads, dim) for _ in range(layers))
            self.model.norm = nn.LayerNorm(width)
            self.model.rotary_emb = ToyRotary(dim)
            self.lm_head = nn.Linear(width, vocab, bias=False)
        self.config = SimpleNamespace(model_type="drift_toy", hidden_size=width,
                                      num_attention_heads=qheads,
                                      num_key_value_heads=kvheads, head_dim=dim,
                                      num_hidden_layers=layers, vocab_size=vocab,
                                      max_position_embeddings=4096,
                                      rope_scaling=None, pretraining_tp=1)
