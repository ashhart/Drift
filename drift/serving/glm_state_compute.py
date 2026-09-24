"""Build GLM's recurrent state from per-token layer inputs with GLM's own projections and kernels.

A memory can carry, for every recurrent (KDA) layer, the 4096-wide input each memory token would have given that
layer. Starting from the state after the prompt's head, each rank runs the layer's own input projection, short
convolution and chunked delta-rule kernel over those inputs on its own heads, and leaves the resulting state in
the request's running state block. With GLM's own captured inputs this must reproduce GLM's own state.
"""
import re

try:
    from live_publication import load_publication
except ImportError:
    from drift.serving.live_publication import load_publication

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn$")


def layer_modules(registry) -> dict:
    """KDA layers by index from vLLM's layer registry (the forward context's no_compile_layers)."""
    layers = {}
    for prefix, module in (registry or {}).items():
        match = _LAYER.search(prefix)
        if match and prefix.startswith("language_model.") and hasattr(module, "A_log") and hasattr(module, "in_proj_qkvbfg_a"):
            layers[int(match.group(1))] = module
    return layers


def _kernel():
    from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda_with_fused_gate
    return chunk_kda_with_fused_gate


def advance(layer, hidden, conv_state, state, kernel=None):
    """New (conv_state, state) for one layer after the inputs [n, hidden], in this rank's layout and dtypes."""
    import torch
    import torch.nn.functional as F
    n, local, heads, dim = hidden.shape[0], layer.local_projection_size, layer.local_num_heads, layer.head_dim
    qkv, beta, f_a, _ = layer.in_proj_qkvbfg_a(hidden)[0].split([3 * local, heads, dim, dim], dim=-1)
    g1 = layer.f_b_proj(f_a)[0].reshape(1, n, heads, dim)
    weight = torch.cat([m.weight.view(m.weight.size(0), m.weight.size(2)) for m in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d)])
    if getattr(layer.q_conv1d, "bias", None) is not None:
        raise ValueError("a biased short convolution is not supported")
    history = conv_state if layer._conv_state_dim_first else conv_state.transpose(0, 1)             # [channels, kept inputs]
    inputs = torch.cat([history.to(qkv.dtype), qkv.transpose(0, 1)], dim=1)                        # oldest first
    window = inputs[:, history.shape[1] - (weight.shape[1] - 1):]
    mixed = F.silu(F.conv1d(window[None].float(), weight[:, None, :].float(), groups=weight.shape[0])[0]).to(qkv.dtype)
    q, k, v = (part.reshape(1, n, heads, dim) for part in mixed.transpose(0, 1).split(local, dim=-1))
    _, final = (kernel or _kernel())(q=q, k=k, v=v, raw_g=g1, beta=torch.sigmoid(beta.float()).unsqueeze(0), A_log=layer.A_log,
                                     g_bias=layer.dt_bias, initial_state=state[None].float().contiguous(), output_final_state=True,
                                     use_qk_l2norm_in_kernel=True, safe_gate=layer.kda_safe_gate, lower_bound=layer.kda_lower_bound)
    kept = inputs[:, -history.shape[1]:].to(conv_state.dtype)
    return (kept if layer._conv_state_dim_first else kept.transpose(0, 1)), final[0].to(state.dtype)


def build_state(connector, step, hidden, kernel=None) -> dict:
    """Advance every KDA layer's running state block by the memory's inputs {layer: [n, hidden]}; all or nothing."""
    import torch
    layers = connector._kda_modules
    if set(hidden) != set(layers):
        raise ValueError("memory inputs do not cover exactly this server's recurrent layers")
    rows = {len(value) for value in hidden.values()}
    if len(rows) != 1 or not 0 < rows.pop() <= 4096:
        raise ValueError("memory inputs must hold the same 1..4096 rows for every layer")
    staged = []
    for index, layer in sorted(layers.items()):
        group = connector._layer_to_group[layer.prefix]
        slot = (step.after - 1) // connector._state_groups[group]
        if not 0 <= slot < len(step.blocks[group]):
            raise ValueError("running state block outside the request")
        block = step.blocks[group][slot]
        conv_view, state_view = layer.kv_cache
        source = torch.as_tensor(hidden[index]).to(device=state_view.device, dtype=layer.in_proj_qkvbfg_a.weight.dtype)
        conv, state = advance(layer, source, conv_view[block], state_view[block], kernel)
        if not (torch.isfinite(state).all() and torch.isfinite(conv.float()).all()):
            raise ValueError(f"layer {index}: the translated state is not finite")
        staged.append((conv_view, state_view, block, conv, state))
    for conv_view, state_view, block, conv, state in staged:
        conv_view[block].copy_(conv)
        state_view[block].copy_(state)
    return {"layers": len(staged), "rows": int(len(next(iter(hidden.values()))))}


def load_inputs(connector, step) -> dict:
    """The session's state.npz: h{layer} [n, hidden] for exactly this server's recurrent layers."""
    layers = connector._kda_modules
    if not layers:
        raise RuntimeError("no recurrent layer modules were registered from the forward context")
    width = next(iter(layers.values())).hidden_size
    entries = load_publication(connector._live_in / step.name / "state.npz", {f"h{i}": (width,) for i in layers})
    return {int(key[1:]): value for key, value in entries.items()}
