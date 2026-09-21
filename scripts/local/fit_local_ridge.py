"""Ridge initialisation of the local-pair translator (writer Qwen3.6-35B-A3B -> reader Qwen3.8-27B): every reader KV
layer from all writer KV layers at the token, streaming second moments, variance gain on validation rows. Same recipe as
scripts/live/fit_stacked_stream.py; the tokenizers are identical, so every token is an aligned row."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np, torch

parser = argparse.ArgumentParser()
parser.add_argument("--texts", type=Path, default=Path("local/local_pair/texts.jsonl"))
parser.add_argument("--writer", type=Path, default=Path("local/local_pair/taps_writer"))
parser.add_argument("--reader", type=Path, default=Path("local/local_pair/taps_reader"))
parser.add_argument("--out", type=Path, default=Path("local/local_pair/ridge.npz"))
parser.add_argument("--ridge", type=float, default=0.03)
args = parser.parse_args()
ids = [hashlib.sha256(json.loads(l)["text"].encode()).hexdigest()[:16] for l in args.texts.read_text().splitlines() if json.loads(l)["split"] == "calib"]
ids = [i for i in ids if (args.writer / f"{i}.npz").exists() and (args.reader / f"{i}.npz").exists()]
fit, val = [x for n, x in enumerate(ids) if n % 20], [x for n, x in enumerate(ids) if not n % 20]
stack = lambda z: np.concatenate([np.concatenate((z[k].reshape(len(z[k]), -1), z["v" + k[1:]].reshape(len(z[k]), -1)), 1) for k in sorted((f for f in z.files if f.startswith("k")), key=lambda f: int(f[1:]))], 1)
first_w, first_r = np.load(args.writer / f"{ids[0]}.npz"), np.load(args.reader / f"{ids[0]}.npz")
w_layers, r_layers = sorted(int(f[1:]) for f in first_w.files if f.startswith("k")), sorted(int(f[1:]) for f in first_r.files if f.startswith("k"))
d_in, d_out, n, t0 = stack(first_w).shape[1], stack(first_r).shape[1], 0, time.time()
sx, sy = torch.zeros(d_in, dtype=torch.float64), torch.zeros(d_out, dtype=torch.float64)
xx, xy = torch.zeros(d_in, d_in, dtype=torch.float64), torch.zeros(d_in, d_out, dtype=torch.float64)
load = lambda group: (torch.from_numpy(np.concatenate([stack(np.load(args.writer / f"{i}.npz")) for i in group]).astype(np.float64)), torch.from_numpy(np.concatenate([stack(np.load(args.reader / f"{i}.npz")) for i in group]).astype(np.float64)))
for start in range(0, len(fit), 64):
    X, Y = load(fit[start:start + 64])
    n += len(X); sx += X.sum(0); sy += Y.sum(0); xx += X.T @ X; xy += X.T @ Y
    print("accumulated", start + 64, "texts", n, "rows", round(time.time() - t0), "s", flush=True)
mx_, my_ = sx / n, sy / n
cxx, cxy = xx - n * torch.outer(mx_, mx_), xy - n * torch.outer(mx_, my_)
W = torch.linalg.solve(cxx + args.ridge * float(torch.diagonal(cxx).mean()) * torch.eye(d_in, dtype=torch.float64), cxy)
Xv, Yv = load(val); Xv, Yv = Xv - mx_, Yv - my_
pred = Xv @ W
width = d_out // len(r_layers)
rel = [float((pred[:, i * width:(i + 1) * width] - Yv[:, i * width:(i + 1) * width]).pow(2).mean().sqrt() / Yv[:, i * width:(i + 1) * width].pow(2).mean().sqrt()) for i in range(len(r_layers))]
gain = Yv.std(0) / pred.std(0).clamp_min(1e-8)
meta = {"writer_layers": w_layers, "reader_layers": r_layers, "reader_width": width, "fit_rows": n, "val_rows": int(len(Xv)), "val_rel_rmse": rel, "ridge": args.ridge, "median_gain": float(gain.median())}
np.savez(args.out, x_mean=mx_.float().numpy(), W=W.float().numpy(), b=my_.float().numpy(), gain=gain.float().numpy(), meta=np.array(json.dumps(meta)))
print(json.dumps(meta))
