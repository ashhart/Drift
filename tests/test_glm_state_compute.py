"""A recurrent state built from per-token layer inputs chains exactly, lands in the running block, and fails whole."""
from __future__ import annotations
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from drift.serving.glm_state_compute import advance, build_state, layer_modules

HIDDEN, HEADS, DIM, WIDTH, KEPT = 16, 2, 4, 4, 6
LOCAL = HEADS * DIM


def reference_kernel(q, k, v, raw_g, beta, A_log, g_bias, initial_state, output_final_state, use_qk_l2norm_in_kernel,
                     safe_gate, lower_bound):
    """Sequential gated delta rule with the bounded gate, as a stand-in for vLLM's chunked kernel."""
    q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
    gate = lower_bound * torch.sigmoid(A_log.view(1, 1, -1, 1).exp() * (raw_g.float() + g_bias.view(1, 1, HEADS, DIM)))
    state = initial_state.clone().float()
    for t in range(q.shape[1]):
        decayed = state * gate[0, t].exp()[None, :, :, None]
        kt, vt, bt = k[0, t], v[0, t].float(), beta[0, t]
        delta = vt - torch.einsum("hkv,hk->hv", decayed[0], kt)
        state = decayed + (bt[:, None, None] * kt[:, :, None] * delta[:, None, :])[None]
    return None, state


class Linear:
    def __init__(self, inputs, outputs, seed):
        self.weight = torch.randn(outputs, inputs, generator=torch.Generator().manual_seed(seed)) / inputs ** 0.5

    def __call__(self, x):
        return (x @ self.weight.T,)


def layer(index, blocks=5):
    conv = lambda seed: NS(weight=torch.randn(LOCAL, 1, WIDTH, generator=torch.Generator().manual_seed(seed)), bias=None)
    return NS(prefix=f"language_model.model.layers.{index}.self_attn", hidden_size=HIDDEN, local_projection_size=LOCAL,
              local_num_heads=HEADS, head_dim=DIM, in_proj_qkvbfg_a=Linear(HIDDEN, 3 * LOCAL + HEADS + 2 * DIM, index),
              f_b_proj=Linear(DIM, LOCAL, 100 + index), q_conv1d=conv(1), k_conv1d=conv(2), v_conv1d=conv(3),
              A_log=torch.zeros(HEADS), dt_bias=torch.zeros(LOCAL), kda_safe_gate=True, kda_lower_bound=-5.0,
              _conv_state_dim_first=True, kv_cache=(torch.randn(blocks, 3 * LOCAL, KEPT), torch.randn(blocks, HEADS, DIM, DIM)), A=None)


def test_advancing_in_two_pieces_equals_advancing_at_once():
    kda = layer(0)
    inputs = torch.randn(9, HIDDEN)
    conv0, state0 = kda.kv_cache[0][2].clone(), kda.kv_cache[1][2].clone()
    whole = advance(kda, inputs, conv0, state0, reference_kernel)
    conv1, state1 = advance(kda, inputs[:4], conv0, state0, reference_kernel)
    pieces = advance(kda, inputs[4:], conv1, state1, reference_kernel)
    torch.testing.assert_close(pieces[1], whole[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(pieces[0], whole[0])
    projected = inputs @ kda.in_proj_qkvbfg_a.weight.T
    torch.testing.assert_close(whole[0][:, -KEPT:], projected[-KEPT:, :3 * LOCAL].T)            # the newest inputs, oldest first


def connector(layers):
    return NS(_kda_modules={int(k.prefix.split(".")[3]): k for k in layers}, _layer_to_group={k.prefix: 2 + i for i, k in enumerate(layers)},
              _state_groups={2: 64, 3: 64})


def test_the_state_lands_in_each_layer_s_running_block_only():
    layers = [layer(0), layer(1)]
    before = [tuple(t.clone() for t in k.kv_cache) for k in layers]
    step = NS(after=100, blocks=((0,), (1,), (3, 4), (0, 2)))                     # slot (100 - 1) // 64 = 1
    hidden = {0: np.random.default_rng(0).standard_normal((7, HIDDEN)).astype(np.float16),
              1: np.random.default_rng(1).standard_normal((7, HIDDEN)).astype(np.float16)}
    report = build_state(connector(layers), step, hidden, reference_kernel)
    assert report == {"layers": 2, "rows": 7}
    for k, (conv0, state0), block in zip(layers, before, (4, 2)):
        expected = advance(k, torch.as_tensor(hidden[int(k.prefix.split(".")[3])]).float(), conv0[block], state0[block], reference_kernel)
        torch.testing.assert_close(k.kv_cache[1][block], expected[1])
        others = [b for b in range(5) if b != block]
        assert torch.equal(k.kv_cache[1][others], state0[others]) and torch.equal(k.kv_cache[0][others], conv0[others])


@pytest.mark.parametrize("hidden,match", [
    ({0: np.zeros((3, HIDDEN), np.float16)}, "exactly"),
    ({0: np.zeros((3, HIDDEN), np.float16), 1: np.zeros((4, HIDDEN), np.float16)}, "same"),
    ({0: np.full((3, HIDDEN), np.inf, np.float16), 1: np.zeros((3, HIDDEN), np.float16)}, "finite"),
])
def test_a_bad_memory_changes_no_block(hidden, match):
    layers = [layer(0), layer(1)]
    before = [tuple(t.clone() for t in k.kv_cache) for k in layers]
    with pytest.raises(ValueError, match=match):
        build_state(connector(layers), NS(after=100, blocks=((0,), (1,), (3, 4), (0, 2))), hidden, reference_kernel)
    for k, (conv0, state0) in zip(layers, before):
        assert torch.equal(k.kv_cache[0], conv0) and torch.equal(k.kv_cache[1], state0)


def test_only_language_model_kda_layers_are_registered():
    kda, drafter = layer(0), layer(45)
    drafter.prefix = "model.layers.45.self_attn"
    registry = {kda.prefix: kda, drafter.prefix: drafter, "language_model.model.layers.3.self_attn.attn": NS()}
    assert layer_modules(registry) == {0: kda}
