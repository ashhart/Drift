"""Train a contextual cache reader on the Studio: GLM's latents over a context -> Qwen's own K/V rows for the same text.

Pairs come from GLM exports with token offsets (export_passages.py) and Qwen taps of the same texts in the drop-in's
layout (studio_tap_passages.py): where a GLM token and a Qwen token end on the same character, Qwen's own rows are the
target for that GLM position. The reader (drift/translate/context_reader.py) corrects a fixed per-token rows translator,
fitted by fit_state_translator.py on the same kind of data; the loss is the squared error on aligned positions in units
of each target dimension's spread. Validation R-squared on held-out contexts is reported beside the fixed translator's.
Squared error shrinks the rows toward their mean, which flattens Qwen's attention over them; at the end a per-dimension
gain fitted on training windows gives them Qwen's spread back (gain.npz, drift/translate/spread.py).
Options: --input full reads GLM's full latents instead of the rows map's principal directions; --contrast adds a loss
asking each predicted K and V row to pick out its own token's true row among the window's rows, and validation reports
that identity score either way; --init continues a trained reader with everything it was trained with. --direction
qwen-to-glm trains the reverse reader on the same windows: Qwen's rows over the context in, GLM's latents out at the
aligned positions, correcting a Qwen-to-GLM translator such as a shared-space pair (drift/translate/hub.py); its
identity score and per-layer R-squared are per GLM MLA layer.
Only the reader trains; both models stay frozen, and neither is loaded here.

  PYTHONPATH=. $OMLX_PY scripts/live/studio_train_context_reader.py --glm out/corpus_latents --qwen out/corpus_taps \\
      --val-glm out/code_glm_val --val-qwen out/code_qwen_val --rows out/rows_g2q_code.npz --out out/context_reader1
"""
import argparse, json, math, random, shutil, time
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from drift.serving.studio_guard import acquire
from drift.translate import ridge_map
from drift.translate.context_pairs import reverse_windows, windows
from drift.translate.context_reader import ContextReader, identity, identity_loss, row_blocks
from drift.translate.spread import restore, spread_gain

parser = argparse.ArgumentParser()
parser.add_argument("--glm", type=Path, required=True, help="GLM exports with offsets, one npz per context")
parser.add_argument("--qwen", type=Path, required=True, help="Qwen taps with offsets and x, same names")
parser.add_argument("--val-glm", type=Path, required=True)
parser.add_argument("--val-qwen", type=Path, required=True)
parser.add_argument("--rows", type=Path, required=True, help="the fixed per-token rows translator the reader corrects")
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--layers", type=int, default=4)
parser.add_argument("--heads", type=int, default=8)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--warmup", type=int, default=100)
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gain-windows", type=int, default=160, help="training windows the spread gain is fitted on")
parser.add_argument("--contrast", type=float, default=0.0,
                    help="weight of a loss asking each predicted K and V row to pick out its own token's true row among the window's rows")
parser.add_argument("--contrast-rows", type=int, default=384, help="aligned rows sampled per step for that loss")
parser.add_argument("--temperature", type=float, default=0.05, help="its softmax temperature, over cosine similarities")
parser.add_argument("--init", type=Path, help="continue a trained reader's folder: its weights, standardisation, input and shape, and its rows translator")
parser.add_argument("--input", choices=("reduced", "full"), default="reduced",
                    help="what the reader reads: the rows map's principal directions, or GLM's full latents, standardised")
parser.add_argument("--direction", choices=("glm-to-qwen", "qwen-to-glm"), default="glm-to-qwen")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
REVERSE = args.direction == "qwen-to-glm"
LAYER_WIDTH = 512 if REVERSE else 1024                                  # a GLM MLA layer's latent, or a Qwen layer's K and V
if args.init is not None:                                               # a continued reader keeps everything it was trained with
    started_from = json.loads((args.init / "config.json").read_text())
    args.input = started_from.get("input", "reduced")
    args.direction, REVERSE = started_from.get("direction", "glm-to-qwen"), started_from.get("direction") == "qwen-to-glm"
    LAYER_WIDTH = 512 if REVERSE else 1024
    args.width, args.layers, args.heads = (started_from["model"][k] for k in ("width", "layers", "heads"))
    args.rows = args.init / "rows.npz"
acquire("studio_train_context_reader.py", need_gb=24)                # caps MLX memory and its buffer cache: window lengths vary every step
args.out.mkdir(parents=True, exist_ok=True)
random.seed(args.seed); mx.random.seed(args.seed)
rows = ridge_map.load(args.rows)
(weight, bias, gain), = rows["layers"].values()
MEAN, BASIS = rows["mean"], rows["basis"]
W, B = mx.array(weight * gain[None, :]), mx.array(bias)
MEAN_MX, BASIS_MX = mx.array(MEAN), mx.array(BASIS)


def contexts(glm_dir: Path, qwen_dir: Path) -> list:
    """Per context: GLM's features as the reader reads them, float16 (reduced, or full with --input full), aligned GLM
    positions, and Qwen's own rows there, float16."""
    kept = (lambda f: f) if args.input == "full" else (lambda f: (f - MEAN) @ BASIS)
    pairs = reverse_windows(glm_dir, qwen_dir) if REVERSE else windows(glm_dir, qwen_dir)
    return [(kept(features).astype(np.float16), source, rows.astype(np.float16)) for _, features, source, rows in pairs]


started = time.time()
train, val = contexts(args.glm, args.qwen), contexts(args.val_glm, args.val_qwen)
print(json.dumps({"train_contexts": len(train), "train_rows": int(sum(len(s) for _, s, _ in train)), "val_contexts": len(val),
                  "val_rows": int(sum(len(s) for _, s, _ in val)), "load_s": round(time.time() - started, 1)}), flush=True)
sample = random.sample(train, min(64, len(train)))
if args.init is not None:
    scales = np.load(args.init / "scales.npz")
    component, target_scale = scales["component"], scales["target"]
    input_mean = scales["input_mean"] if "input_mean" in scales.files else np.zeros_like(component)
else:
    seen = np.concatenate([x.astype(np.float32) for x, _, _ in sample])
    input_mean = seen.mean(0).astype(np.float32) if args.input == "full" else np.zeros(seen.shape[1], np.float32)   # reduced features are centred already
    component = np.maximum(seen.std(0), 1e-6).astype(np.float32)
    target_scale = np.maximum(np.concatenate([t.astype(np.float32) for _, _, t in sample]).std(0), 1e-6).astype(np.float32)
    del seen
IM, CS, TS = mx.array(input_mean), mx.array(component), mx.array(target_scale)
model = ContextReader(len(component), W.shape[1], args.width, args.layers, args.heads)
if args.init is not None:
    model.load_weights(str(args.init / "reader.safetensors"))
mx.eval(model.parameters())


def predict(reader, x: mx.array, source: mx.array) -> tuple[mx.array, mx.array]:
    """The fixed translator's rows and the reader's rows at the aligned positions, from one window's stored features."""
    z = (x - MEAN_MX) @ BASIS_MX if args.input == "full" else x
    base = (z[source] @ W) + B
    return base, base + reader((x - IM) / CS)[source] * TS


BLOCKS = [(i * 512, (i + 1) * 512) for i in range(W.shape[1] // 512)] if REVERSE else row_blocks(W.shape[1])


def loss_fn(reader, x, source, target, rows):
    _, pred = predict(reader, x, source)
    loss = mx.mean(mx.square((pred - target) / TS))
    return loss + args.contrast * identity_loss(pred, target, rows, TS, args.temperature, BLOCKS) if args.contrast else loss


def evaluate() -> dict:
    """R-squared on held-out contexts, per full-attention layer and overall, for the fixed translator and the reader."""
    sse_base, sse_reader, total, sum_t, sum_tt, found_base, found_reader = 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0
    for x, source, target in val:
        x, source, target = mx.array(x.astype(np.float32)), mx.array(source), mx.array(target.astype(np.float32))
        base, pred = predict(model, x, source)
        found_base = found_base + identity(base, target, BLOCKS) * target.shape[0]
        found_reader = found_reader + identity(pred, target, BLOCKS) * target.shape[0]
        sse_base = sse_base + mx.sum(mx.square(base - target), axis=0)
        sse_reader = sse_reader + mx.sum(mx.square(pred - target), axis=0)
        sum_t, sum_tt, total = sum_t + mx.sum(target, axis=0), sum_tt + mx.sum(mx.square(target), axis=0), total + target.shape[0]
    sst = sum_tt - mx.square(sum_t) / total
    w = LAYER_WIDTH
    by_layer = lambda sse: [round(1 - float(mx.sum(sse[i * w:(i + 1) * w]) / mx.sum(sst[i * w:(i + 1) * w])), 4) for i in range(W.shape[1] // w)]
    kv = (lambda found: {"latent": round(float(mx.mean(found)) / total, 4)}) if REVERSE else \
        (lambda found: {"k": round(float(mx.mean(found[0::2])) / total, 4), "v": round(float(mx.mean(found[1::2])) / total, 4)})
    return {"fixed": round(1 - float(mx.sum(sse_base) / mx.sum(sst)), 4), "reader": round(1 - float(mx.sum(sse_reader) / mx.sum(sst)), 4),
            "identity_fixed": kv(found_base), "identity_reader": kv(found_reader),
            "reader_by_layer": by_layer(sse_reader), "fixed_by_layer": by_layer(sse_base)}


schedule = lambda step: args.lr * min(1.0, (step + 1) / args.warmup) * (0.1 + 0.45 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps)))
opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
step_fn = nn.value_and_grad(model, loss_fn)
history, losses = [{"step": 0, **evaluate()}], []
print(json.dumps(history[-1]), flush=True)
started = time.time()
for step in range(1, args.steps + 1):
    x, source, target = random.choice(train)
    opt.learning_rate = schedule(step)
    rows = mx.array(np.sort(np.random.default_rng(step).choice(len(source), min(args.contrast_rows, len(source)), replace=False)))
    loss, grads = step_fn(model, mx.array(x.astype(np.float32)), mx.array(source), mx.array(target.astype(np.float32)), rows)
    grads, _ = optim.clip_grad_norm(grads, 1.0)
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state, loss)
    losses.append(float(loss))
    if step % 50 == 0:
        print(json.dumps({"step": step, "loss": round(float(np.mean(losses[-50:])), 4), "seconds": round(time.time() - started, 1)}), flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        history.append({"step": step, "loss": round(float(np.mean(losses[-args.eval_every:])), 4), **evaluate()})
        print(json.dumps({k: v for k, v in history[-1].items() if not k.endswith("by_layer")}), flush=True)
        model.save_weights(str(args.out / "reader.safetensors"))
        (args.out / "history.json").write_text(json.dumps(history, indent=1))
rows_of = lambda windows_: [(np.array(predict(model, mx.array(x.astype(np.float32)), mx.array(s))[1]), t.astype(np.float32)) for x, s, t in windows_]
fitted = rows_of(random.Random(args.seed).sample(train, min(args.gain_windows, len(train))))
centre, gain = spread_gain(np.concatenate([p for p, _ in fitted]), np.concatenate([t for _, t in fitted]))
held = rows_of(val)
val_p, val_t = np.concatenate([p for p, _ in held]), np.concatenate([t for _, t in held])
r2 = lambda p: round(float(1 - ((p - val_t) ** 2).sum() / ((val_t - val_t.mean(0)) ** 2).sum()), 4)
history.append({"gain_windows": len(fitted), "val_r2_before_gain": r2(val_p), "val_r2_after_gain": r2(restore(val_p, centre, gain)),
                "gain_median": round(float(np.median(gain)), 4)})
print(json.dumps(history[-1]), flush=True)
(args.out / "history.json").write_text(json.dumps(history, indent=1))
np.savez(args.out / "gain.npz", centre=centre, gain=gain)                # squared error shrank the rows; the gain restores Qwen's spread
shutil.copyfile(args.rows, args.out / "rows.npz")
np.savez(args.out / "scales.npz", component=component, target=target_scale, input_mean=input_mean)
(args.out / "config.json").write_text(json.dumps({"model": {"in_dim": len(component), "out_dim": int(W.shape[1]), "width": args.width,
                                                            "layers": args.layers, "heads": args.heads}, "rows": str(args.rows), "input": args.input,
                                                  "direction": args.direction}, indent=1))
print(json.dumps({"steps": args.steps, "seconds": round(time.time() - started, 1)}))
