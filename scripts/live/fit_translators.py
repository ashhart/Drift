"""Found pool.v1 from real GLM-5.3 and Qwen3.8 taps and fit both members' translators (ridge).
Level l pairs GLM sparse-attention layer 3+4l with Qwen KV layer 3+4l (l = 0..10). Qwen's 12th KV
layer (47) enrolls a second reader against level 10 so every Qwen KV layer receives entries."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, torch
from drift.translate.pool import Layout, Translator, cross_error, fit_member, fit_pool_format, save_translator

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=Path, default=Path("local/live/corpus.json"))
parser.add_argument("--glm", type=Path, default=Path("local/live/taps_glm"))
parser.add_argument("--qwen", type=Path, default=Path("local/live/taps_qwen"))
parser.add_argument("--out", type=Path, default=Path("local/live/translators"))
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--ridge", type=float, default=1e-2)
args = parser.parse_args()
G_LAYERS, Q_LAYERS = [3 + 4 * i for i in range(11)], [3 + 4 * i for i in range(12)]
records = json.loads(args.corpus.read_text())["records"]
rows = {"train": {l: {"glm": [], "qwen": [], "qwen47": []} for l in range(11)}, "heldout": {l: {"glm": [], "qwen": [], "qwen47": []} for l in range(11)}}
for r in records:
    g, q = np.load(args.glm / f"{r['id']}.npz"), np.load(args.qwen / f"{r['id']}.npz")
    gi, qi = np.array([a[0] for a in r["aligned"]]), np.array([a[1] for a in r["aligned"]])
    for l in range(11):
        rows[r["split"]][l]["glm"].append(g[f"l{G_LAYERS[l]}"][gi].astype(np.float32))
        k, v = q[f"k{Q_LAYERS[l]}"][qi].astype(np.float32), q[f"v{Q_LAYERS[l]}"][qi].astype(np.float32)
        rows[r["split"]][l]["qwen"].append(np.concatenate((k.reshape(len(qi), -1), v.reshape(len(qi), -1)), axis=1))
        if l == 10:
            k, v = q[f"k47"][qi].astype(np.float32), q[f"v47"][qi].astype(np.float32)
            rows[r["split"]][l]["qwen47"].append(np.concatenate((k.reshape(len(qi), -1), v.reshape(len(qi), -1)), axis=1))
T = lambda parts: torch.from_numpy(np.concatenate(parts))
train = {l: {"glm": T(rows["train"][l]["glm"]), "qwen": T(rows["train"][l]["qwen"])} for l in range(11)}
held = {l: {"glm": T(rows["heldout"][l]["glm"]), "qwen": T(rows["heldout"][l]["qwen"])} for l in range(11)}
print("train rows per level:", train[0]["glm"].shape[0], "| heldout rows:", held[0]["glm"].shape[0])
fmt, pool_rows = fit_pool_format(train, width=args.width)
glm = Translator("glm", Layout("mla_latent", 1, 512), {G_LAYERS[l]: l for l in range(11)}, fmt)
qwen = Translator("qwen", Layout("kv_split", 2, 256), {Q_LAYERS[l]: l for l in range(11)}, fmt)
qwen_tail = Translator("qwen_tail", Layout("kv_split", 2, 256), {47: 10}, fmt)
rep_g = fit_member(glm, {l: train[l]["glm"] for l in range(11)}, pool_rows, ridge=args.ridge)
rep_q = fit_member(qwen, {l: train[l]["qwen"] for l in range(11)}, pool_rows, ridge=args.ridge)
rep_t = fit_member(qwen_tail, {10: T(rows["train"][10]["qwen47"])}, {10: pool_rows[10]}, ridge=args.ridge)
g2q = cross_error(glm, qwen, {l: held[l]["glm"] for l in range(11)}, {l: held[l]["qwen"] for l in range(11)})
q2g = cross_error(qwen, glm, {l: held[l]["qwen"] for l in range(11)}, {l: held[l]["glm"] for l in range(11)})
summary = {"pool": {"width": fmt.width, "levels": fmt.levels, "fingerprint": fmt.fingerprint}, "ridge": args.ridge,
           "train_rows": int(train[0]["glm"].shape[0]), "heldout_rows": int(held[0]["glm"].shape[0]),
           "heldout_cross_relative_rmse": {"glm_to_qwen": {l: round(e["cross_rmse"] / e["baseline_rmse"], 3) for l, e in sorted(g2q.items())},
                                           "qwen_to_glm": {l: round(e["cross_rmse"] / e["baseline_rmse"], 3) for l, e in sorted(q2g.items())}},
           "train_roundtrip_relative_rmse": {"glm": {l: round(v["roundtrip_rmse"] / v["baseline_rmse"], 3) for l, v in rep_g.items()},
                                             "qwen": {l: round(v["roundtrip_rmse"] / v["baseline_rmse"], 3) for l, v in rep_q.items()}}}
import shutil; shutil.rmtree(args.out, ignore_errors=True); args.out.mkdir(parents=True)
for t in (glm, qwen, qwen_tail):
    save_translator(t, args.out / t.member, {"corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(), "kind": "ridge", "semantic_transfer": "NOT_EVALUATED"})
(args.out / "pool.json").write_text(json.dumps(summary["pool"]))
(args.out / "fit_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=1))
