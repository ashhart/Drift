"""Level-2 parity of an MLX adapter on a real local checkpoint (criteria: see the prereg file).

  PYTHONPATH=. .venv-next/bin/python scripts/studio_mlx_level2.py --family qwen4_exp \
      --checkpoint ~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP \
      --prereg configs/preregistration.studio-qwen38-level2.json --out local/studio/qwen38_level2.json
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=["qwen4_exp", "glm5_next"], required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--prereg", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
checkpoint = args.checkpoint.expanduser()
prereg = json.loads(args.prereg.read_text())
report: dict = {"family": args.family, "checkpoint": str(checkpoint), "preregistration_sha256": hashlib.sha256(args.prereg.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(), "stages": {}}
args.out.parent.mkdir(parents=True, exist_ok=True)


def flush(status: str) -> None:
    report["status"] = status
    args.out.write_text(json.dumps(report, indent=2) + "\n")


try:
    import mlx.core as mx
    from mlx_vlm.utils import load_model
    t0 = time.perf_counter()
    try:
        model = load_model(checkpoint)
        report["stages"]["load"] = {"strict": True}
    except Exception as error:                                   # e.g. MTP tensors rejected by a strict load
        report["stages"]["load"] = {"strict": False, "strict_error": f"{type(error).__name__}: {str(error)[:400]}"}
        model = load_model(checkpoint, strict=False)
    lm = model.language_model
    report["stages"]["load"].update({"seconds": time.perf_counter() - t0, "peak_gb": mx.get_peak_memory() / 2**30,
                                     "layers": len(lm.model.layers), "model_type": lm.args.model_type})
    flush("LOADED")
    vocab = int(lm.args.vocab_size)

    def to_mx(ids):
        return mx.array(ids.numpy().astype(np.int32))[None]

    def to_t(a):
        mx.eval(a)
        return torch.from_numpy(np.array(a.astype(mx.float32)))

    def decoder_forward(ids, cache, offset):
        """The decoder exactly as the adapter drives it, but with the STOCK attention classes."""
        position_ids = mx.arange(offset, offset + ids.shape[0], dtype=mx.int32)[None]
        hidden = lm.model(to_mx(ids), cache=cache, position_ids=position_ids) if args.family == "qwen4_exp" else lm.model(to_mx(ids), cache=cache)
        if args.family == "qwen4_exp":
            hidden = lm.model.embed_tokens.as_linear(hidden) if lm.args.tie_word_embeddings else lm.lm_head(hidden)
        return to_t(hidden[0])

    # ---- stock references, BEFORE the adapter substitutes anything
    stock = {}
    for n in prereg["lengths"]:
        g = torch.Generator().manual_seed(prereg["seeds"][0] + n)
        ids = torch.randint(3, vocab, (n + prereg["continuation_tokens"],), generator=g)
        full = decoder_forward(ids, None, 0)
        cache = lm.make_cache()
        pre = decoder_forward(ids[:n], cache, 0)
        cont = decoder_forward(ids[n:], cache, n)
        cache2 = lm.make_cache()
        decoder_forward(ids[:n], cache2, 0)
        steps = [decoder_forward(ids[n + j:n + j + 1], cache2, n + j) for j in range(prereg["continuation_tokens"])]
        stock[n] = {"ids": ids, "full": full, "cont": cont, "steps": torch.cat(steps)}
        if args.family == "qwen4_exp":
            public = lm(to_mx(ids)).logits
            stock[n]["public_full"] = to_t(public[0])
    report["stages"]["stock"] = {str(n): {"multi_vs_full_max_abs": float((s["cont"] - s["full"][n:]).abs().max()),
                                          "steps_vs_full_max_abs": float((s["steps"] - s["full"][n:]).abs().max()),
                                          **({"public_vs_decoder_full_max_abs": float((s["public_full"] - s["full"]).abs().max())} if "public_full" in s else {})}
                                 for n, s in stock.items()}
    flush("STOCK_DONE")

    # ---- adapter
    if args.family == "qwen4_exp":
        from drift.adapters.mlx_qwen4_exp import MlxQwen4ExpAdapter as Adapter
    else:
        from drift.adapters.mlx_glm5_next import MlxGlm5NextAdapter as Adapter
    from drift.adapters.base import ForeignEntries
    from drift.core.types import KV
    adapter = Adapter(lm)
    d = adapter.descriptor
    report["stages"]["adapter"] = {"kv_layers": list(d.kv_layers), "layout": d.layout, "kv_heads": d.kv_heads, "head_dim": d.head_dim}
    results = {}
    for n, s in stock.items():
        ids = s["ids"]
        full = adapter.forward(ids)
        prefix = adapter.forward(ids[:n])
        cont = adapter.forward(ids[n:], prefix.state)
        state, steps = prefix.state, []
        for j in range(prereg["continuation_tokens"]):
            out = adapter.forward(ids[n + j:n + j + 1], state)
            state, steps = out.state, steps + [out.output]
        steps = torch.cat(steps)
        tokens = 8
        if d.layout == "kv_split":
            hostile = ForeignEntries(torch.arange(tokens), {i: KV(torch.full((tokens, d.kv_heads, d.head_dim), 30.0), torch.full((tokens, d.kv_heads, d.head_dim), -30.0)) for i in d.kv_layers})
        else:
            hostile = ForeignEntries(torch.arange(tokens), {i: torch.full((tokens, d.head_dim), 30.0) for i in d.kv_layers})
        closed = adapter.forward(ids[n:], prefix.state, foreign=hostile, override=0.0)
        opened = adapter.forward(ids[n:], prefix.state, foreign=hostile, override=1.0)
        shapes = {str(i): (list(full.canonical[i].k.shape) if hasattr(full.canonical[i], "k") else list(full.canonical[i].shape)) for i in d.kv_layers}
        finite = all(bool(torch.isfinite(e.k if hasattr(e, "k") else e).all()) for e in full.canonical.values())
        masses = {str(i): [float(m.min()), float(m.max())] for i, m in opened.foreign_mass.items()}
        stock_drift = float((s["cont"] - s["full"][n:]).abs().max())
        adapter_drift = float((cont.output.float() - full.output.float()[n:]).abs().max())
        results[str(n)] = {
            "full_bit_identical": bool(torch.equal(full.output.float(), s["full"])),
            "full_max_abs": float((full.output.float() - s["full"]).abs().max()),
            "continuation_bit_identical": bool(torch.equal(cont.output.float(), s["cont"])),
            "continuation_max_abs": float((cont.output.float() - s["cont"]).abs().max()),
            "single_token_vs_stock_decoder_max_abs": float((steps.float() - s["steps"]).abs().max()),
            "hard_off_bit_identical": bool(torch.equal(closed.output, cont.output)),
            "foreign_changes_output": bool(not torch.allclose(opened.output.float(), cont.output.float())),
            "foreign_mass_min_max": masses, "canonical_shapes": shapes, "canonical_finite": finite,
            "stock_prefill_vs_incremental": stock_drift, "adapter_prefill_vs_incremental": adapter_drift,
            "reference_logit_abs_max": float(s["full"].abs().max()),
        }
        report["stages"]["parity"] = results
        flush("RUNNING")
    gates = []
    for r in results.values():
        gates += [r["full_bit_identical"], r["continuation_bit_identical"], r["hard_off_bit_identical"], r["foreign_changes_output"],
                  r["canonical_finite"], all(0.0 <= lo and hi <= 1.0 + 1e-6 for lo, hi in r["foreign_mass_min_max"].values()),
                  r["adapter_prefill_vs_incremental"] <= 1.10 * r["stock_prefill_vs_incremental"] + 1e-12]
    report["peak_gb"] = mx.get_peak_memory() / 2**30
    flush("PASSED" if all(gates) else "FAILED")
except Exception as error:
    report["error"] = f"{type(error).__name__}: {error}"
    report["traceback"] = traceback.format_exc()[-3000:]
    flush("ERROR")
    raise
