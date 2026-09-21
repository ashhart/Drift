"""Test A replicated over MCDMA, reader side (Studio, oMLX runtime). Same passages, questions, frozen translators and six
conditions as scripts/live/qa_v4.py; the ONLY change is transport: GLM's cache export is RDMA-READ from the Spark into shared
memory by the owner's handoffd and translated here on the GPU (v3 and v4). Stays resident: reads one chunk jobs-file path per
stdin line, appends rows to --out as jsonl, prints DONE. This process opens no verbs objects."""
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
parser.add_argument("--artifacts", type=Path, default=Path("local/live"))
parser.add_argument("--socket", default="/tmp/handoffd.sock")
parser.add_argument("--gain-power", type=float, default=1.5)
parser.add_argument("--max-new", type=int, default=160)
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
acquire("studio_mcdma_qa.py", need_gb=160)
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
STOP = {tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")}
base = StackedReader.load(args.artifacts / "stacked3.npz", GL, QL, kv_heads=2, head_dim=256)
corrected = CorrectedFanoutReader.load(args.artifacts / "v4_correction.safetensors", FanoutReader.load(args.artifacts / "fanout3.npz", base))
index = IndexKeyReader.load(args.artifacts / "index3.npz", QL, base.sha256)
fan = corrected.fan
plain = CorrectedFanoutReader(fan, np.zeros((base.input_mean.shape[0], 1), np.float32), np.zeros((1, 2 * 2 * 256 * len(QL)), np.float32), np.zeros((len(QL), 2), np.float32), "v3")
readers = {"v3": MlxForwardReader(plain, None, args.gain_power), "v4": MlxForwardReader(corrected, None, args.gain_power)}   # zero selector keys, as in Test A
for r in readers.values():
    r.read({l: np.zeros((8, 512), np.float32) for l in GL})                    # warm the kernels
from drift.serving.omlx_cache import tap


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


def remember(entries, rows):
    cache = lm.make_cache()
    append_entries(cache, entries, rope, np.arange(rows), index.index_dim, dtype=mx.bfloat16)
    return cache


import sys
args.out.parent.mkdir(parents=True, exist_ok=True)
previous = None                                                                # the previous passage's v4 memory is this passage's wrong memory
print("READY", flush=True)
for line in sys.stdin:
    if not line.strip():
        continue
    for job in json.loads(Path(line.strip()).read_text()):
        t0 = time.time(); pulled = client.pull(job["peer"], job["remote"], offset=0, unlink=True); t_pull = time.time() - t0
        latents = read_latents(client.view(job["peer"], 0, pulled.bytes))
        t0 = time.time(); memory = {name: r.read(latents) for name, r in readers.items()}; t_translate = time.time() - t0
        ids = tok.encode(job["text"], add_special_tokens=False).ids
        native = lm.make_cache(); step(ids, native, 0); own = tap(native, list(QL), rope, 0, len(ids)); del native
        wrong = previous if previous is not None else memory["v4"]
        for q in job["questions"]:
            answers = {"no_memory": answer(q["question"], lm.make_cache(), 0)[0],
                       "drift_v3": answer(q["question"], remember(memory["v3"][0], len(memory["v3"][2])), len(memory["v3"][2]))[0],
                       "drift_v4": answer(q["question"], remember(memory["v4"][0], len(memory["v4"][2])), len(memory["v4"][2]))[0],
                       "wrong_memory": answer(q["question"], remember(wrong[0], len(wrong[2])), len(wrong[2]))[0] if previous is not None else None,
                       "own_kv": answer(q["question"], remember(own, len(ids)), len(ids))[0],
                       "text_in_prompt": answer(f"{job['text']}\n\n{q['question']}", lm.make_cache(), 0)[0]}
            row = {"passage": job["id"], **q, **answers, "transport": {"bytes": pulled.bytes, "rdma_loop_s": pulled.loop_ns / 1e9, "rdma_gbit_s": pulled.gbit_s, "job_s": pulled.job_ns / 1e9, "translate_both_s": round(t_translate, 4)}}
            with args.out.open("a") as sink:
                sink.write(json.dumps(row) + "\n")
        previous = memory["v4"]
    print("DONE", line.strip(), flush=True)
