"""Cross-runtime fixtures: the tiny random torch models of `tests/test_adapters_next.py`
rebuilt in MLX (mlx-vlm 0.7.1) with the torch weights transferred.

The torch model is the reference. Its HF-style state dict is written to a temporary
safetensors file, loaded with `mx.load`, renamed to the checkpoint layout the MLX
`sanitize` expects, run through the upstream `sanitize` (which fuses `q_a_proj` +
`kv_a_proj_with_mqa` into `qkv_a_proj`, absorbs `kv_b_proj` into `embed_q` /
`unembed_out`, fuses GLM's linear-attention projections, splits Qwen's fused
`gate_up_proj` experts) and then loaded strictly. Every mapping decision is listed in
`docs/research/M4_MLX_NOTES.md`.

Only text models are built; the vision towers are never instantiated.
"""
from __future__ import annotations
import os
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping
import numpy as np
import torch

REQUIRED_MLX_VLM = "0.7.1"
FAMILIES = ("qwen4_exp", "glm5_next")


def _mx():
    import mlx.core as mx
    return mx


def require_mlx_vlm():
    import mlx_vlm
    if mlx_vlm.__version__ != REQUIRED_MLX_VLM:
        raise RuntimeError(f"MLX adapters are qualified on mlx-vlm=={REQUIRED_MLX_VLM}, found {mlx_vlm.__version__}")
    return mlx_vlm


# -- tensor conversion ---------------------------------------------------------

def to_mx(tensor: torch.Tensor):
    return _mx().array(tensor.detach().cpu().contiguous().numpy())


def to_torch(array) -> torch.Tensor:
    mx = _mx()
    mx = _mx()
    if array.dtype == mx.bfloat16:          # numpy has no bfloat16; real checkpoints produce it
        array = array.astype(mx.float32)
    return torch.from_numpy(np.array(array, copy=True))


# -- configs -------------------------------------------------------------------

def mlx_qwen_config(hf_config):
    """HF `Qwen4ExpTextConfig` -> mlx-vlm `qwen4_exp.TextConfig`.

    Field decisions: `output_gate_type` is None in HF (meaning "use hidden_act"); the
    MLX config validates it against {"sigmoid","silu"}, so it is resolved here the way
    both runtimes resolve it (`output_gate_type or hidden_act`). `rope_parameters` keeps
    HF's `rope_type` key; the MLX config renames it to `type` itself.
    """
    from mlx_vlm.models.qwen4_exp.config import TextConfig
    d = dict(hf_config.to_dict())
    d["output_gate_type"] = d.get("output_gate_type") or d.get("hidden_act", "silu")
    d["rope_parameters"] = dict(d["rope_parameters"])
    return TextConfig.from_dict(d)


def mlx_glm_config(hf_config):
    """HF `Glm5NextTextConfig` -> mlx-vlm `glm5_next.TextConfig` (field names coincide)."""
    from mlx_vlm.models.glm5_next.config import TextConfig
    return TextConfig.from_dict(dict(hf_config.to_dict()))


# -- weight transfer -----------------------------------------------------------

def _torch_state_to_mx(model: torch.nn.Module) -> dict[str, Any]:
    from safetensors.torch import save_file
    mx = _mx()
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        save_file(state, path)
        weights = mx.load(path)
    finally:
        os.unlink(path)
    return dict(weights)


def _strip(prefix: str, weights: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in weights.items():
        if not k.startswith(prefix):
            raise KeyError(f"unexpected key after sanitize: {k}")
        out[k[len(prefix):]] = v
    return out


def _load_strict(model, weights: dict[str, Any]) -> None:
    from mlx.utils import tree_flatten
    mx = _mx()
    expected = {k for k, _ in tree_flatten(model.parameters())}
    missing, extra = expected - set(weights), set(weights) - expected
    if missing or extra:
        raise KeyError(f"weight mapping mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    for k, v in weights.items():
        target = dict(tree_flatten(model.parameters()))[k]
        if tuple(v.shape) != tuple(target.shape):
            raise ValueError(f"shape mismatch for {k}: checkpoint {v.shape} vs model {target.shape}")
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())


def build_mlx_qwen(torch_model):
    """Qwen4ExpForCausalLM (tiny) -> mlx-vlm qwen4_exp LanguageModel with the same weights."""
    require_mlx_vlm()
    from mlx_vlm.models.qwen4_exp.language import LanguageModel
    from mlx_vlm.models.qwen4_exp.qwen4_exp import Model as Qwen4ExpVLM
    cfg = mlx_qwen_config(torch_model.config)
    # `LanguageModel.get_rope_index` (the public text-only prefill path) reads the VLM
    # wrapper config for the vision token ids and merge size even when no image is
    # present. A text-only shim carrying the upstream `ModelConfig` defaults keeps the
    # public stock forward runnable without instantiating the vision tower.
    model = LanguageModel(cfg, config=SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=248056, video_token_id=248057, vision_start_token_id=248053))
    raw = _torch_state_to_mx(torch_model)
    # HF causal-LM layout `model.X` / `lm_head.weight` -> checkpoint layout the mlx-vlm
    # VLM sanitizer expects (`model.language_model.X` / `lm_head.weight`).
    checkpoint = {}
    for k, v in raw.items():
        if k.startswith("model."):
            checkpoint["model.language_model." + k[len("model."):]] = v
        else:
            checkpoint[k] = v
    # The upstream sanitize only touches `self.config.text_config`; a shim avoids
    # instantiating the vision tower.
    shim = SimpleNamespace(config=SimpleNamespace(text_config=cfg))
    sanitized = Qwen4ExpVLM.sanitize(shim, checkpoint)
    _load_strict(model, _strip("language_model.", sanitized))
    model.eval()
    # MLX limitation: the fused gated-delta Metal kernel (qwen3_5/gated_delta.py) is
    # templated on head_dim/32 and fails to compile for the tiny config's head_dim 8
    # ("zero-length arrays are not permitted"). `Qwen3_5GatedDeltaNet` selects the
    # kernel with `use_kernel=not self.training`, so the reference ops path is chosen
    # by flagging only those module instances (not their children) as training.
    # Nothing else in the module reads the flag. Real head_dim 128 checkpoints do not
    # need this.
    for layer in model.layers:
        if layer.is_linear:
            layer.linear_attn._set_training_mode(True)
    return model


def build_mlx_glm(torch_model):
    """Glm5NextTextModel (tiny, headless) -> mlx-vlm glm5_next LanguageModel.

    The torch fixture has no LM head; the MLX LanguageModel always builds one, so it is
    loaded with zeros and never used (the adapter returns hidden states, as the torch
    adapter does)."""
    require_mlx_vlm()
    from mlx_vlm.models.glm5_next.language import LanguageModel
    mx = _mx()
    cfg = mlx_glm_config(torch_model.config)
    model = LanguageModel(cfg)
    raw = _torch_state_to_mx(torch_model)
    inter = cfg.moe_intermediate_size
    checkpoint = {}
    for k, v in raw.items():
        key = k
        key = key.replace(".self_attn.forget_gate.", ".self_attn.")
        if key.endswith(".self_attn.conv1d.weight"):
            # HF fused q|k|v depthwise conv [C,1,K] -> MLX Conv1d [C,K,1] on the fused
            # `qkv_conv`; the HF module concatenates q,k,v in that order too.
            key = key.replace(".self_attn.conv1d.weight", ".self_attn.qkv_conv.conv.weight")
            v = v.moveaxis(2, 1)
        if key.endswith(".mlp.experts.gate_up_proj"):
            # HF experts [E, 2I, H] (gate rows first) -> MLX SwitchGLU gate/up [E, I, H].
            base = "language_model.model." + key[: -len(".experts.gate_up_proj")]
            checkpoint[base + ".switch_mlp.gate_proj.weight"] = v[:, :inter, :]
            checkpoint[base + ".switch_mlp.up_proj.weight"] = v[:, inter:, :]
            continue
        if key.endswith(".mlp.experts.down_proj"):
            key = key[: -len(".experts.down_proj")] + ".switch_mlp.down_proj.weight"
        checkpoint["language_model.model." + key] = v
    checkpoint["language_model.lm_head.weight"] = mx.zeros((cfg.vocab_size, cfg.hidden_size), dtype=mx.float32)
    sanitized = model.sanitize(checkpoint)
    _load_strict(model, _strip("language_model.", sanitized))
    model.eval()
    return model


# -- families ------------------------------------------------------------------

def _torch_factories():
    from tests.test_adapters_next import make_glm, make_qwen
    return {"qwen4_exp": make_qwen, "glm5_next": make_glm}


def build_family(family: str):
    """(torch_model, torch_adapter, mlx_model, hf_config) for one family, same weights.

    The torch side is exactly `tests.test_adapters_next.make_qwen/make_glm`."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    torch_model, torch_adapter, _ = _torch_factories()[family]()
    mlx_model = build_mlx_qwen(torch_model) if family == "qwen4_exp" else build_mlx_glm(torch_model)
    return torch_model, torch_adapter, mlx_model, torch_model.config


def build_pair(family: str):
    """(torch_model, mlx_model, config): the pair the task asks for."""
    torch_model, _, mlx_model, config = build_family(family)
    return torch_model, mlx_model, config


# -- stock forwards ------------------------------------------------------------

def torch_stock_forward(family: str, torch_model, ids: torch.Tensor, past=None):
    """[T, X] output and the cache. Full prefill when `past` is None."""
    with torch.no_grad():
        result = torch_model(input_ids=ids[None], past_key_values=past, use_cache=True)
    output = result.logits if hasattr(result, "logits") else result.last_hidden_state
    return output[0], result.past_key_values


def mlx_stock_forward(family: str, mlx_model, ids: torch.Tensor, cache=None):
    """The MLX model's own public forward: `LanguageModel.__call__` for Qwen (logits, its
    own rope-index and batch-invariant decode machinery), `LanguageModel.model` for GLM
    (hidden states; the torch fixture is headless). Pass `cache` from `make_cache()` to
    continue; the cache is advanced in place as in the upstream generate loop."""
    mx = _mx()
    inputs = mx.array(ids.detach().cpu().numpy().astype(np.int32))[None]
    if family == "qwen4_exp":
        out = mlx_model(inputs, cache=cache).logits
    else:
        out = mlx_model.model(inputs, cache=cache)
    mx.eval(out)
    return to_torch(out[0]).float(), cache


def fixture_exchange(family: str, ids: torch.Tensor, split: int | None = None) -> dict[str, torch.Tensor]:
    """Run torch and MLX stock forwards on the same ids.

    Returns `torch_full`, `mlx_full` (prefill of all ids) and, when `split` is given,
    `torch_cont`, `mlx_cont` (prefill `ids[:split]` with a cache, then continue on
    `ids[split:]`)."""
    torch_model, _, mlx_model, _ = build_family(family)
    return exchange(family, torch_model, mlx_model, ids, split)


def exchange(family: str, torch_model, mlx_model, ids: torch.Tensor, split: int | None = None) -> dict[str, torch.Tensor]:
    out = {}
    out["torch_full"], _ = torch_stock_forward(family, torch_model, ids)
    out["mlx_full"], _ = mlx_stock_forward(family, mlx_model, ids)
    if split is not None:
        _, past = torch_stock_forward(family, torch_model, ids[:split])
        out["torch_cont"], _ = torch_stock_forward(family, torch_model, ids[split:], past)
        cache = mlx_model.make_cache()
        mlx_stock_forward(family, mlx_model, ids[:split], cache)
        out["mlx_cont"], _ = mlx_stock_forward(family, mlx_model, ids[split:], cache)
    return out


def max_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.double() - b.double()).abs()
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
            "max_rel": float((diff / b.double().abs().clamp_min(1e-12)).max()),
            "ref_max_abs": float(b.double().abs().max())}


# -- adapters ------------------------------------------------------------------

def build_adapters(family: str):
    """(torch_adapter, mlx_adapter, torch_model, mlx_model) sharing one set of weights."""
    from drift.adapters.mlx_glm5_next import MlxGlm5NextAdapter
    from drift.adapters.mlx_qwen4_exp import MlxQwen4ExpAdapter
    torch_model, torch_adapter, mlx_model, _ = build_family(family)
    mlx_adapter = (MlxQwen4ExpAdapter if family == "qwen4_exp" else MlxGlm5NextAdapter)(mlx_model)
    return torch_adapter, mlx_adapter, torch_model, mlx_model


# -- Metal precision ----------------------------------------------------------------

def gemm_precision() -> dict[str, float]:
    """Normalized max error of an fp32 [16,64]@[64,32] product on the default device
    against fp64 numpy, plus the same product against TF32-rounded inputs. mlx 0.32.2
    on Apple M5 (applegpu_g17) rounds fp32 GEMM operands to 10-bit mantissas unless
    `MLX_METAL_GPU_ARCH` names an older architecture (see M4_MLX_NOTES.md)."""
    mx = _mx()
    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 64)).astype(np.float32)
    b = rng.standard_normal((64, 32)).astype(np.float32)
    got = np.array(mx.matmul(mx.array(a), mx.array(b)), dtype=np.float64)
    exact = a.astype(np.float64) @ b.astype(np.float64)

    def tf32(x):
        return (x.view(np.uint32) & np.uint32(0xFFFFE000)).view(np.float32).astype(np.float64)

    scale = float(np.abs(exact).max())
    return {"vs_fp64": float(np.abs(got - exact).max() / scale),
            "vs_tf32_inputs": float(np.abs(got - tf32(a) @ tf32(b)).max() / scale),
            "architecture": str((mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()).get("architecture"))}


# -- sparse-selection tie diagnostics -------------------------------------------------

TIE_TOLERANCE = 1e-5


def _tied_at_cut(scores: np.ndarray, k: int) -> bool:
    """True when the k-th and (k+1)-th best candidates are (numerically) tied, so the
    top-k set depends on the runtime's tie-break (torch.topk vs mx.argpartition)."""
    if scores.shape[0] <= k:
        return False
    ordered = np.sort(scores)[::-1]
    return abs(ordered[k - 1] - ordered[k]) <= TIE_TOLERANCE * max(1.0, abs(ordered[k - 1]))


def selection_ties(family: str, mlx_adapter, ids: torch.Tensor) -> dict[int, list[int]]:
    """Query positions (per KV layer) of a full prefill of `ids` whose sparse-indexer
    top-k cut falls on a tie. Reproduces the upstream MLX scoring on the attention inputs
    the adapter recorded; a position after the first tie may legitimately differ between
    runtimes (the flipped selection changes that layer's output and every later layer)."""
    mx = _mx()
    mlx_adapter.forward(ids)
    inputs = {i: mlx_adapter._text.layers[i].self_attn._drift_last_input for i in mlx_adapter.descriptor.kv_layers}
    ties: dict[int, list[int]] = {}
    length = int(ids.numel())
    for layer, x in inputs.items():
        attention = mlx_adapter._text.layers[layer].self_attn
        positions = []
        if family == "qwen4_exp":
            ind = attention.indexer
            qk = ind.index_qk_proj(x).reshape(1, length, ind.n_heads + ind.kv_heads, ind.head_dim)
            q = ind._apply_rope(ind.q_layernorm(qk[:, :, :ind.n_heads]).transpose(0, 2, 1, 3), mx.arange(length)[None])
            raw = qk[:, :, ind.n_heads:].squeeze(2)
            blocks = length // ind.compress_ratio
            if blocks == 0:
                continue
            pooled = ind.k_layernorm(mx.mean(raw[:, :blocks * ind.compress_ratio].reshape(
                1, blocks, ind.compress_ratio, ind.head_dim).astype(mx.float32), axis=2))[:, None]
            pooled = ind._apply_rope(pooled, (mx.arange(blocks) * ind.compress_ratio)[None])
            scores = mx.sum(mx.maximum(q.astype(mx.float32) @ pooled.astype(mx.float32).transpose(0, 1, 3, 2), 0), axis=1)
            scores = np.array(scores[0] / (ind.head_dim ** 0.5), dtype=np.float64)          # [L, blocks]
            for i in range(length):
                complete = (i + 1) // ind.compress_ratio
                if complete > ind.block_topk and _tied_at_cut(scores[i, :complete], ind.block_topk):
                    positions.append(i)
        else:
            ind = attention.indexer
            if ind is None:
                continue
            from mlx_vlm.models.glm5_next.language import _score_index_keys
            from mlx_vlm.models.linear import linear
            q_a, _ = mx.split(linear(attention.qkv_a_proj, x), (attention.q_lora_rank,), axis=-1)
            q_resid = attention.q_a_layernorm(q_a)
            k, gate = ind._project_keys(x)
            packed = mx.concatenate([k, gate.astype(k.dtype), mx.ones((1, length, 1), dtype=k.dtype)], axis=-1)
            pool_keys, pool_indices, pool_valid, _ = ind._pooled_states(packed)
            q, w = ind._project_queries(x, q_resid)
            scores = np.array(_score_index_keys(q, pool_keys, w, ind.softmax_scale)[0], dtype=np.float64)   # [L, P]
            ends = np.array(pool_indices[0, :, -1]).astype(np.int64)
            valid = np.array(pool_valid[0]).astype(bool)
            select_k = min(ind.index_topk // ind.index_kpool, pool_keys.shape[1])
            for i in range(length):
                candidates = valid & (ends <= i)
                if candidates.sum() > select_k and _tied_at_cut(scores[i][candidates], select_k):
                    positions.append(i)
        if positions:
            ties[layer] = positions
    return ties


def first_tie(ties: Mapping[int, list[int]]) -> int | None:
    return min((p for ps in ties.values() for p in ps), default=None)
