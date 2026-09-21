"""Reverse Drift over MCDMA, writer side (Studio, oMLX runtime). Resident; one JSON command per stdin line.
  prepare  Qwen reads a text, its cache is tapped, translated for GLM (frozen reverse translator), tiled and packed -> key, rows
  deliver  stage the prepared publication on every rank and release it on the head through the dedicated MCDMA targets
  confirm  wait for every rank's connector receipt and bind it to the same session, sequence and SHA-256
Opens no verbs objects; the MCDMA client library is plain userspace UDP."""
import argparse, ctypes, importlib.util, io, json, sys, time
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx
from tokenizers import Tokenizer
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from drift.serving.mcdma_reverse import ReversePublisher
from drift.serving.omlx_cache import Rope, kv_layer_indices, tap
from drift.translate.stacked import StackedReader

parser = argparse.ArgumentParser()
parser.add_argument("--reverse", type=Path, default=Path("local/live/stacked3_rev.npz"))
parser.add_argument("--gain-power", type=float, default=1.0)
parser.add_argument("--build", type=Path, default=Path("out/mcdma-target/build"))
args = parser.parse_args()
GL, QL = tuple(3 + 4 * i for i in range(11)), tuple(3 + 4 * i for i in range(12))
spec = importlib.util.spec_from_file_location("drift_mcdma", args.build / "mcdma.py"); mcdma = importlib.util.module_from_spec(spec); spec.loader.exec_module(mcdma)
mcdma._load = lambda: ctypes.CDLL(str(args.build / "libmcdma.dylib"))
publisher = ReversePublisher({"spark-a.invalid": mcdma.open("192.0.2.1", src="192.0.2.40"), "spark-b.invalid": mcdma.open("198.51.100.1", src="198.51.100.40")}, head="spark-a.invalid", timeout_s=60, split=32 << 20)   # fails here if a bridge is not up
ck = Path("~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()
cfg = json.loads((ck / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
from drift.serving.studio_guard import acquire
acquire("studio_mcdma_reverse.py", need_gb=150)
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
rev = StackedReader.load(args.reverse, QL, GL, kv_heads=0, head_dim=512)
prepared, delivered = {}, {}
print(json.dumps({"ready": True, "mailbox_sessions": {r: w.session for r, w in publisher.writers.items()}}), flush=True)
for line in sys.stdin:
    cmd = json.loads(line)
    try:
        if cmd["op"] == "prepare":
            ids = tok.encode(cmd["text"], add_special_tokens=False).ids
            t0 = time.perf_counter(); cache = lm.make_cache(); mx.eval(lm(mx.array(np.asarray(ids, dtype=np.int32))[None], cache=cache).logits); t_read = time.perf_counter() - t0
            t0 = time.perf_counter(); own = tap(cache, kv_layer_indices(cache), rope, 0, len(ids)); t_tap = time.perf_counter() - t0
            t0 = time.perf_counter()
            flat = {l: np.concatenate((own[l][0].reshape(len(ids), -1), own[l][1].reshape(len(ids), -1)), axis=1).astype(np.float32) for l in QL}
            latents = rev.read(flat, args.gain_power); t_translate = time.perf_counter() - t0
            t0 = time.perf_counter(); copies = int(cmd["copies"]); rows = len(ids) * copies
            sink = io.BytesIO(); np.savez(sink, **{f"l{l}": np.asarray(latents[l], dtype=np.float16) for l in GL}); body = sink.getvalue(); t_pack = time.perf_counter() - t0
            prepared[cmd["key"]] = (body, rows, copies)                         # ONE copy crosses the wire; each rank's bridge tiles it
            reply = {"ok": True, "rows": rows, "source_tokens": len(ids), "bytes": len(body), "seconds": {"qwen_reads_text": round(t_read, 4), "tap": round(t_tap, 4), "translate": round(t_translate, 4), "pack": round(t_pack, 4)}}
        elif cmd["op"] == "deliver":
            body, rows, copies = prepared[cmd["key"]]
            delivered[cmd["key"]] = publisher.deliver(cmd["session"], int(cmd["sequence"]), body, copies); reply = {"ok": True, **delivered[cmd["key"]]}
        elif cmd["op"] == "confirm":
            reply = {"ok": True, **publisher.confirm_applied(delivered[cmd["key"]], prepared[cmd["key"]][1])}
        elif cmd["op"] == "quit":
            break
        else:
            reply = {"ok": False, "error": "unknown op"}
    except Exception as error:
        from drift.serving.mcdma_mailbox import RETRIES
        reply = {"ok": False, "error": f"{type(error).__name__}: {error}"[:300], "transport_events": RETRIES["events"]}
    print(json.dumps(reply), flush=True)
for conn in publisher.conns.values():
    conn.close()
