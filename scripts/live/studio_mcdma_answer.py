"""Drift over MCDMA, reader side (Studio, oMLX runtime). For each job: handoffd RDMA-READs GLM's cache export from the Spark
into shared memory, this process parses it there, translates it on the GPU (K/V + selector keys), appends it to a fresh Qwen
cache and answers. No file crosses ssh for the memory payload. Every stage is timed. This process opens no verbs objects."""
import argparse, json, time
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx
from tokenizers import Tokenizer
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from drift.serving.glm53_handoff import read_latents
from drift.serving.handoffd_client import HandoffdClient
from drift.serving.omlx_cache import Rope, append_entries
from drift.translate.fanout import CorrectedFanoutReader, FanoutReader
from drift.translate.index_keys import IndexKeyReader
from drift.translate.mlx_reader import MlxForwardReader
from drift.translate.stacked import StackedReader

parser = argparse.ArgumentParser()
parser.add_argument("--jobs", type=Path, required=True, help="json list: {id, peer, remote, questions: [{question, answer}]}")
parser.add_argument("--artifacts", type=Path, default=Path("local/live"))
parser.add_argument("--socket", default="/tmp/handoffd.sock")
parser.add_argument("--gain-power", type=float, default=1.5)
parser.add_argument("--max-new", type=int, default=48)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
GL, QL = tuple(3 + 4 * i for i in range(11)), tuple(3 + 4 * i for i in range(12))
client = HandoffdClient(args.socket)
print("handoffd:", client.status(), flush=True)                                # fail before loading the model if the daemon is not up
ck = Path("~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()
cfg = json.loads((ck / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
from drift.serving.studio_guard import acquire
acquire("studio_mcdma_answer.py", need_gb=160)
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
STOP = {tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")}
base = StackedReader.load(args.artifacts / "stacked3.npz", GL, QL, kv_heads=2, head_dim=256)
corrected = CorrectedFanoutReader.load(args.artifacts / "v4_correction.safetensors", FanoutReader.load(args.artifacts / "fanout3.npz", base))
index = IndexKeyReader.load(args.artifacts / "index3.npz", QL, base.sha256)
reader = MlxForwardReader(corrected, index, args.gain_power)
reader.read({l: np.zeros((8, 512), np.float32) for l in GL})                   # warm the kernels


def step(ids, cache, start):
    positions = mx.arange(start, start + len(ids), dtype=mx.int32)[None]
    logits = lm(mx.array(np.asarray(ids, dtype=np.int32))[None], cache=cache, position_ids=positions).logits[0, -1]
    mx.eval(logits)
    return logits


def answer(question, cache, start):
    ids = tok.encode(f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
    t0 = time.time(); logits = step(ids, cache, start); first = time.time() - t0
    out, pos = [], start + len(ids)
    for _ in range(args.max_new):
        nxt = int(mx.argmax(logits).item())
        if nxt in STOP:
            break
        out.append(nxt); logits = step([nxt], cache, pos); pos += 1
    return tok.decode(out), first


results = []
for job in json.loads(args.jobs.read_text()):
    t0 = time.time(); pulled = client.pull(job["peer"], job["remote"], offset=0, unlink=True); t_pull = time.time() - t0
    t0 = time.time(); latents = read_latents(client.view(job["peer"], 0, pulled.bytes)); t_parse = time.time() - t0
    t0 = time.time(); entries, keys, order, _ = reader.read(latents); t_translate = time.time() - t0
    rows = []
    for q in job["questions"]:
        cache = lm.make_cache(); t0 = time.time()
        append_entries(cache, entries, rope, np.arange(len(order)), index.index_dim, dtype=mx.bfloat16, index_keys=keys)
        mx.eval([cache[l].keys for l in QL]); t_append = time.time() - t0
        text, first = answer(q["question"], cache, len(order))
        none, _ = answer(q["question"], lm.make_cache(), 0)
        rows.append({**q, "drift_over_mcdma": text, "no_memory": none, "correct": q["answer"].lower() in text.lower(), "correct_no_memory": q["answer"].lower() in none.lower(),
                     "append_s": round(t_append, 4), "first_token_s": round(first, 4)})
    row = {"id": job["id"], "writer_tokens": int(next(iter(latents.values())).shape[0]), "memory_rows": len(order),
           "transport": {"bytes": pulled.bytes, "rdma_loop_s": pulled.loop_ns / 1e9, "rdma_gbit_s": pulled.gbit_s, "job_s": pulled.job_ns / 1e9, "client_wall_s": round(t_pull, 4)},
           "seconds": {"parse_and_dequantize": round(t_parse, 4), "translate_on_gpu": round(t_translate, 4)}, "questions": rows}
    results.append(row); print(json.dumps({k: v for k, v in row.items() if k != "questions"} | {"correct": [r["correct"] for r in rows]}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps({"label": "EXPLORATORY: Drift over MCDMA", "results": results}, indent=2) + "\n")
