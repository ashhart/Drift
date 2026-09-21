"""Persistent Studio worker with private cache state and validated foreign-memory append."""
import json, sys
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx
from tokenizers import Tokenizer
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from drift.serving.omlx_cache import Rope, append_entries, kv_layer_indices, tap_slots
from drift.serving.live_receiver_studio import append_memory
from drift.serving.worker_studio import continue_native, generate_native

configuration = None
if "--worker-session-config" in sys.argv:
    from drift.serving.worker_artifacts import verify_worker_manifests
    configuration = json.loads(Path(sys.argv[sys.argv.index("--worker-session-config") + 1]).read_text())
    verified = verify_worker_manifests(configuration)

ck = Path("~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()
if configuration is not None:
    from drift.serving.worker_contract import require
    require(verified["model"]["checkpoint_path"] == str(ck.resolve()), "CAPABILITY")
cfg = json.loads((ck / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
from drift.serving.studio_guard import acquire, idle_exit
acquire('studio_drift_worker.py', need_gb=150)        # one model process at a time, preflight memory check, hard MLX limits
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
STOP = {tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")}
probe = lm.make_cache(); mx.eval(lm(mx.array([[1, 2, 3, 4]]), cache=probe).logits)
KV, INDEX_DIM = kv_layer_indices(probe), int(probe[kv_layer_indices(probe)[0]].index_keys.shape[-1])
KV_LAYOUTS = {layer: (int(probe[layer].keys.shape[1]), int(probe[layer].keys.shape[-1])) for layer in KV}
S = {}


def run(ids):
    n = len(ids)
    positions = mx.array(np.arange(S["own_pos"], S["own_pos"] + n, dtype=np.int32))[None]
    logits = lm(mx.array(np.asarray(ids, dtype=np.int32))[None], cache=S["cache"], position_ids=positions).logits[0, -1]
    mx.eval(logits)
    slot0 = len(S["slot_pos"])
    S["slot_pos"] += list(range(S["own_pos"], S["own_pos"] + n)); S["own_slots"] += list(range(slot0, slot0 + n)); S["own_pos"] += n
    S["last"] = np.array(logits.astype(mx.float32))


def handle(cmd):
    op = cmd["op"]
    if op == "start":
        S.clear(); S.update(cache=lm.make_cache(), reserve=int(cmd.get("reserve", 4096)), own_pos=int(cmd.get("reserve", 4096)), foreign_pos=0, slot_pos=[], own_slots=[], last=None, done=False)
        return {"kv_layers": KV}
    if S.get("poisoned"):
        raise RuntimeError("live session is poisoned; start a fresh session")
    if op == "continue":
        return continue_native(S, cmd["ids"], run)
    if op == "extend":
        if "chat" in cmd:
            system = f"<|im_start|>system\n{cmd['system']}<|im_end|>\n" if cmd.get("system") else ""
            full = f"{system}<|im_start|>user\n{cmd['chat']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            if cmd.get("span") and "@@DRIFT@@" in full:
                head, tail = full.split("@@DRIFT@@", 1)
                run(tok.encode(head, add_special_tokens=False).ids)
                S["span"] = [S["own_pos"], S["own_pos"] + int(cmd["span"])]; S["span_next"] = S["own_pos"]; S["own_pos"] += int(cmd["span"])
                ids = tok.encode(tail, add_special_tokens=False).ids
                run(ids)
                return {"tokens": len(ids), "span": S["span"], "own_slots": len(S["own_slots"])}
            ids = tok.encode(full, add_special_tokens=False).ids
            run(ids)
            return {"tokens": len(ids), "system_tokens": len(tok.encode(system, add_special_tokens=False).ids) if system else 0, "own_slots": len(S["own_slots"])}
        else:
            ids = cmd.get("ids") or tok.encode(cmd["text"], add_special_tokens=False).ids
        run(ids)
        return {"tokens": len(ids), "own_slots": len(S["own_slots"])}
    if op == "append":
        return append_memory(S, cmd["memory"], KV_LAYOUTS, rope, INDEX_DIM, append_entries, mx.bfloat16, mx.eval)
    if op in ("generate", "generate_own"):
        return generate_native(S, int(cmd["tokens"]), STOP, run, tok.decode, include_ids=op == "generate_own")
    if op == "tap":
        slots = np.array(S["own_slots"][int(cmd.get("first", 0)):], dtype=np.int64)
        entries = tap_slots(S["cache"], KV, rope, slots, np.array([S["slot_pos"][s] for s in slots]))
        Path(cmd["out"]).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cmd["out"], **{f"k{l}": k.astype(np.float16) for l, (k, v) in entries.items()}, **{f"v{l}": v.astype(np.float16) for l, (k, v) in entries.items()})
        return {"tapped": int(len(slots)), "next_first": len(S["own_slots"])}
    raise ValueError(f"unknown op {op}")


touch = idle_exit(minutes=15)                                     # a dropped ssh link must not leave ~110 GB resident
if "--worker-session-config" in sys.argv:
    from drift.serving.worker_studio_backend import serve_studio
    from drift.serving.worker_cache_settle import settle_native_caches
    def settle_native():
        settle_native_caches(S.get('cache', []), mx.eval, mx.array)
    layouts = {f'{kind}{layer}': shape for layer, shape in KV_LAYOUTS.items() for kind in 'kv'}
    raise SystemExit(serve_studio(configuration, ck, handle, S, sys.stdin, sys.stdout, settle=settle_native, layouts=layouts))


print(json.dumps({"ready": True, "kv_layers": KV}), flush=True)
for line in sys.stdin:
    touch()
    line = line.strip()
    if not line:
        continue
    cmd = json.loads(line)
    if cmd.get("op") == "quit":
        break
    try:
        print(json.dumps({"ok": True, **handle(cmd)}), flush=True)
    except Exception as error:
        S["poisoned"] = True
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: live command failed; session poisoned"}), flush=True)
