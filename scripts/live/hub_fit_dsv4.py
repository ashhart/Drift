"""Add DeepSeek V4 to the shared space as a receiver: fit its decoder from pooled hub vectors to its compressed entries.

Inputs are DeepSeek's page captures (dsv4_capture_rows.py) and GLM's latents for the same passages, one npz per passage
under the same id. GLM's latents go into the hub through the anchor; each DeepSeek entry that the passage fills
completely takes GLM's hub vectors under its tokens (drift/translate/dsv4_member.py). Ridge maps
take that mean to the entry's content at ratio 4 (MLA values and indexer keys, every ratio-4 layer at once) and at
ratio 128, with each output's spread restored. Held-out passages give R-squared from GLM's hub vectors and, with
--val-qwen and --qwen-member, from Qwen's rows carried into the hub by Qwen's encoder: one decoder serves both senders.
With --translate, it writes each listed passage's translated content per sender for the DeepSeek gate.

  python3 scripts/live/hub_fit_dsv4.py --anchor out/hub/glm --captures out/dsv4_rows_train --glm out/hidden_train \\
      --val-captures out/dsv4_rows_val --val-glm out/hidden_val --val-qwen out/qtaps_val --qwen-member out/hub/qwen \\
      --out out/hub/dsv4 --translate 2119e4c9ed62a62b,bd55555fd5655e75
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from drift.translate import dsv4_member, hub
from drift.translate.context_pairs import layer_features
from drift.translate.spread import spread_gain


def capture(path: Path) -> tuple[np.ndarray, int, dict]:
    """A capture's DeepSeek token offsets, passage token count and decoded span content."""
    z = np.load(path)
    manifest = json.loads(bytes(z["manifest"]).decode())
    rows = {entry["name"]: (int(entry["ratio"]), z[f"r{i}"]) for i, entry in enumerate(manifest)}
    return z["offsets"], int(z["span"][2]), dsv4_member.content(rows)


def sender(kind: str, path: Path, anchor: hub.Member, qwen: hub.Member | None) -> tuple[np.ndarray, np.ndarray]:
    """A sender's hub vectors and token offsets for one passage: GLM through the anchor, Qwen through its encoder."""
    z = np.load(path)
    return (anchor.encode(layer_features(z)) if kind == "glm" else qwen.encode(z["x"].astype(np.float32))), z["offsets"]


def inputs(ratio: int, z: np.ndarray, source_offsets: np.ndarray, offsets: np.ndarray, count: int, total: int, whole: bool, width: int):
    """A passage's decoder inputs at one ratio and the entries they belong to: token slots at ratio 4, a mean at 128."""
    if ratio == 4 and width:
        x, kept = dsv4_member.slots(z, source_offsets, offsets, count, 4, total, width)
        return (x, kept) if not whole else (x[(kept + 1) * 4 <= count], kept[(kept + 1) * 4 <= count])
    return dsv4_member.pool(z, source_offsets, dsv4_member.spans(offsets, count, ratio, total, whole))


def pairs(captures: Path, source: Path, kind: str, anchor: hub.Member, qwen: hub.Member | None, whole: bool, width: int) -> dict:
    """Decoder inputs and targets for every entry of every passage both sides hold."""
    out = {"x4": [], "y4": [], "x128": [], "y128": [], "passages": 0}
    for path in sorted(Path(captures).glob("*.npz")):
        other = Path(source) / path.name
        if not other.exists():
            continue
        offsets, count, found = capture(path)
        z, source_offsets = sender(kind, other, anchor, qwen)
        for ratio, x, y, target in ((4, "x4", "y4", np.concatenate((found["mla4"], found["index4"]), axis=2)), (128, "x128", "y128", found["mla128"])):
            pooled, kept = inputs(ratio, z, source_offsets, offsets, count, len(target), whole, width)
            if len(kept):
                out[x].append(pooled); out[y].append(target[kept].reshape(len(kept), -1))
        out["passages"] += 1
    return {k: (np.concatenate(v) if isinstance(v, list) and v else v) for k, v in out.items()}


def r2(pred: np.ndarray, target: np.ndarray) -> float:
    return round(float(1 - ((pred - target) ** 2).sum() / ((target - target.mean(0)) ** 2).sum()), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--glm", type=Path, required=True)
    parser.add_argument("--val-captures", type=Path)
    parser.add_argument("--val-glm", type=Path)
    parser.add_argument("--val-qwen", type=Path)
    parser.add_argument("--qwen-member", type=Path)
    parser.add_argument("--slots", type=int, default=0, help="hub values per token slot at ratio 4; 0 averages the entry's tokens")
    parser.add_argument("--ridge4", type=float, default=0.3)
    parser.add_argument("--ridge128", type=float, default=10.0, help="stronger: one ratio-128 entry per 128 tokens gives few pairs")
    parser.add_argument("--translate", default="", help="comma-separated passage ids to translate for the gate")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    anchor, started = hub.load(args.anchor), time.time()
    qwen = hub.load(args.qwen_member) if args.qwen_member else None
    train = pairs(args.captures, args.glm, "glm", anchor, qwen, whole=True, width=args.slots)
    fits = {}
    for ratio, ridge in ((4, args.ridge4), (128, args.ridge128)):
        x, y = train[f"x{ratio}"], train[f"y{ratio}"]
        w, b = hub.fit_ridge(x, y, ridge)
        centre, gain = spread_gain(x @ w + b, y)
        fits[ratio] = (w, b, gain, centre)
    first = np.load(next(iter(sorted(Path(args.captures).glob("*.npz")))))
    found = dsv4_member.names({e["name"]: (int(e["ratio"]), None) for e in json.loads(bytes(first["manifest"]).decode())})
    decoder = dsv4_member.Decoder(np.array(sorted(found["mla4"])), np.array(sorted(found["mla128"])), *fits[4], *fits[128])
    report = {"passages": train["passages"], "entries4": int(len(train["x4"])), "entries128": int(len(train["x128"])), "slots": args.slots,
              "ridge4": args.ridge4, "ridge128": args.ridge128, "fit_seconds": round(time.time() - started, 1)}
    senders = [("glm", args.val_glm), ("qwen", args.val_qwen)] if args.val_captures else []
    for kind, folder in senders:
        if folder is None or (kind == "qwen" and qwen is None):
            continue
        val = pairs(args.val_captures, folder, kind, anchor, qwen, whole=True, width=args.slots)
        mla4, index4 = decoder.ratio4(val["x4"])
        y4 = val["y4"].reshape(len(val["y4"]), len(decoder.layers4), -1)
        report[f"held_out_{kind}"] = {"passages": val["passages"], "entries4": int(len(val["x4"])), "entries128": int(len(val["x128"])),
                                      "mla4_r2": r2(mla4.reshape(len(mla4), -1), y4[:, :, :448].reshape(len(y4), -1)),
                                      "index4_r2": r2(index4.reshape(len(index4), -1), y4[:, :, 448:].reshape(len(y4), -1)),
                                      "mla128_r2": r2(decoder.ratio128(val["x128"]).reshape(len(val["x128"]), -1), val["y128"]) if len(val["x128"]) else None}
    dsv4_member.save(decoder, args.out, report)
    for pid in filter(None, args.translate.split(",")):
        offsets, count, found = capture(Path(args.val_captures) / f"{pid}.npz")
        for kind, folder in senders:
            if folder is None or not (Path(folder) / f"{pid}.npz").exists() or (kind == "qwen" and qwen is None):
                continue
            z, source_offsets = sender(kind, Path(folder) / f"{pid}.npz", anchor, qwen)
            pooled4, kept4 = inputs(4, z, source_offsets, offsets, count, len(found["mla4"]), False, args.slots)
            pooled128, kept128 = inputs(128, z, source_offsets, offsets, count, len(found["mla128"]), False, args.slots)
            mla4, index4 = decoder.ratio4(pooled4)
            np.savez(args.out / f"translated_{kind}_{pid}.npz", mla4=mla4, index4=index4, kept4=kept4, mla128=decoder.ratio128(pooled128),
                     kept128=kept128, offsets=offsets, count=count)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
