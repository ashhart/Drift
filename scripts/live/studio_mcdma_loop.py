"""Run exploratory two-way MCDMA epochs with cache-application receipts."""
import argparse, ctypes, importlib.util, io, json, sys, time
from pathlib import Path
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx.core as mx
from tokenizers import Tokenizer
from mlx_vlm.utils import load_model
from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
from drift.serving.mcdma_mailbox import Reader, Window
from drift.serving.mcdma_forward import Complete, ForwardMailbox, validate_translation
from drift.serving.mcdma_links import HEAD, parse_links
from drift.serving.mcdma_completion import wait_for_completion
from drift.serving.foreign_positions import ForeignPositionBank
from drift.serving.mcdma_reverse import ReversePublisher
from drift.serving.omlx_cache import Rope, append_entries, tap_slots
from drift.translate.fanout import CorrectedFanoutReader, FanoutReader
from drift.translate.index_keys import IndexKeyReader
from drift.translate.mlx_reader import MlxForwardReader
from drift.translate.stacked import StackedReader

parser = argparse.ArgumentParser()
parser.add_argument("--session", required=True)
parser.add_argument("--messages", type=Path, required=True)
parser.add_argument("--epoch-tokens", type=int, default=16)
parser.add_argument("--max-new", type=int, default=160)
parser.add_argument("--copies", type=int, default=3)
parser.add_argument("--prompt-copies", type=int, help="tiling for the FIRST publication (the prompt rows); default: --copies")
parser.add_argument("--publish-text", help="publish only the prompt tokens of this exact text (what the user told this model), not its instructions")
parser.add_argument("--followup", help="after the partner has finished and its taps are in the cache, ask this as a NEW user turn: the question is then computed AFTER the memory arrived")
parser.add_argument("--prompt-only", action="store_true", help="publish nothing after the first publication")
parser.add_argument("--glm-reserve", type=int, default=1024)
parser.add_argument("--own-start", type=int, default=2048, help="Qwen's own rotary position origin; foreign positions track source recency")
parser.add_argument("--foreign-row-cap", type=int, default=2048, help="maximum retained foreign rows, independent of the rotary position origin")
parser.add_argument("--sync-confirm", action="store_true", help="block every epoch until both ranks' caches applied the publication (the first loop measurement); default: confirm asynchronously")
parser.add_argument("--no-reverse", action="store_true", help="one-way control: Qwen publishes nothing; GLM's taps still arrive")
parser.add_argument("--no-forward", action="store_true", help="one-way control: GLM's taps are drained but NOT appended to Qwen's cache")
parser.add_argument("--no-link", action="store_true", help="control: same schedule, nothing published and nothing appended")
parser.add_argument("--artifacts", type=Path, default=Path("local/live"))
parser.add_argument("--build", type=Path, default=Path("out/mcdma-target/build"))
parser.add_argument("--links", default="", help="MCDMA legs to each rank as target/source, head first; required unless --no-link")
parser.add_argument("--state-artifact", type=Path, help="GLM-to-Qwen state translator: each appended tap also advances Qwen's linear-attention layers")
parser.add_argument("--state-gain-power", type=float, default=1.0, help="rescaling of the state translator's shrunk outputs")
parser.add_argument("--followup-block", action="store_true",
                    help="before the follow-up, open the user turn with a framing line, then give GLM's rows their own block of positions before the question")
parser.add_argument("--state-at", choices=["followup", "arrival"], default="followup",
                    help="advance Qwen's recurrent layers with GLM's words just before the follow-up question, or as each tap arrives")
parser.add_argument("--save-taps", type=Path, help="keep each applied forward tap's latents here, for offline replays of this session")
parser.add_argument("--rows-correction", type=Path, help="a trained rows correction from studio_train_memory_answer.py, added to each tap's translated rows")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
legs = {} if args.no_link else parse_links(args.links)
from drift.serving.studio_guard import acquire
acquire("studio_mcdma_loop.py", need_gb=160)
GL, QL = tuple(3 + 4 * i for i in range(11)), tuple(3 + 4 * i for i in range(12))
spec = importlib.util.spec_from_file_location("drift_mcdma", args.build / "mcdma.py"); mcdma = importlib.util.module_from_spec(spec); spec.loader.exec_module(mcdma)
mcdma._load = lambda: ctypes.CDLL(str((args.build / "libmcdma.dylib").resolve()))
HALF = 32 << 20
publisher = forward = None
if not args.no_link:
    forward = Reader(Window(mcdma.open(legs[HEAD][0], src=legs[HEAD][1]), HALF, HALF))      # stamp the forward mailbox BEFORE asking the head to send taps
    publisher = ReversePublisher({rank: mcdma.open(target, src=source) for rank, (target, source) in legs.items()}, head=HEAD, timeout_s=8, split=HALF)
    publisher.watch(args.session)
ck = Path("~/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()
cfg = json.loads((ck / "config.json").read_text()); t = cfg.get("text_config", cfg); rp = t.get("rope_parameters") or {}
rope = Rope(float(rp.get("rope_theta", 1e7)), int(t["head_dim"] * float(rp.get("partial_rotary_factor", 1.0))))
tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
with _force_qwen4_exp_sanitize_on_load(ck):
    model = load_model(ck)
lm = model.language_model
STOP = {tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")}
base = StackedReader.load(args.artifacts / "stacked3.npz", GL, QL, kv_heads=2, head_dim=256)
fwd = MlxForwardReader(CorrectedFanoutReader.load(args.artifacts / "v4_correction.safetensors", FanoutReader.load(args.artifacts / "fanout3.npz", base)),
                       IndexKeyReader.load(args.artifacts / "index3.npz", QL, base.sha256), 1.5)
rev = StackedReader.load(args.artifacts / "stacked3_rev.npz", QL, GL, kv_heads=0, head_dim=512)
fwd.read({l: np.zeros((8, 512), np.float32) for l in GL})
rows_fix = None
if args.rows_correction is not None:                                  # trained on answers: rows_fix(features)[order] is added to the stack's rows
    z = np.load(args.rows_correction)
    rows_fix = {key: z[key].astype(np.float32) for key in ("mean", "basis", "down", "up")}


def corrected(entries, order, latents):
    """The stack's rows plus the trained correction for each row, from the GLM token it translates."""
    reduced = (np.concatenate([latents[l] for l in sorted(latents)], axis=1) - rows_fix["mean"]) @ rows_fix["basis"]
    delta = (reduced[np.asarray(order)] @ rows_fix["down"]) @ rows_fix["up"]
    return {layer: (entries[layer][0] + delta[:, i * 1024:i * 1024 + 512].reshape(len(delta), 2, 256),
                    entries[layer][1] + delta[:, i * 1024 + 512:(i + 1) * 1024].reshape(len(delta), 2, 256)) for i, layer in enumerate(QL)}


state, pending_state = None, {}                                         # pending: translated inputs held for the follow-up
if args.state_artifact is not None:                                   # GLM latents -> inputs of Qwen's linear-attention layers
    z = np.load(args.state_artifact)
    state = {"mean": z["mean"].astype(np.float32), "basis": z["basis"].astype(np.float32),
             "layers": {int(l): (z[f"W{int(l)}"].astype(np.float32), z[f"b{int(l)}"], z[f"gain{int(l)}"] ** args.state_gain_power) for l in z["layers"]}}
INDEX_DIM = 128
messages = json.loads(args.messages.read_text())
prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
encoded = tok.encode(prompt, add_special_tokens=False); ids = encoded.ids
share = None
if args.publish_text:
    a = prompt.index(args.publish_text); b = a + len(args.publish_text)
    share = [i for i, (x, y) in enumerate(encoded.offsets) if x < b and y > a]
cache, own_slots, own_positions, fwd_used, glm_used = lm.make_cache(), [], [], 0, 0
foreign_bank = ForeignPositionBank(QL, args.foreign_row_cap)


def slots_now():
    return int(cache[QL[0]].offset)


position_gap = 0                                                        # own positions skipped to give GLM's rows their own block
FRAME = "My partner's message, from our shared memory:"               # carries no content; it only says where the block comes from


def step(tokens):
    start = args.own_start + len(own_positions) + position_gap; first = slots_now()
    foreign_bank.rephase(cache, rope, start)
    positions = mx.arange(start, start + len(tokens), dtype=mx.int32)[None]
    logits = lm(mx.array(np.asarray(tokens, dtype=np.int32))[None], cache=cache, position_ids=positions).logits[0, -1]
    mx.eval(logits)
    own_slots.extend(range(first, first + len(tokens))); own_positions.extend(range(start, start + len(tokens)))
    return logits


print(json.dumps({"ready": True, "prompt_tokens": len(ids)}), flush=True)
if sys.stdin.readline().strip() != "go":
    raise SystemExit("no go: session abandoned before generation")
T0 = time.perf_counter(); wall0_ns = time.time_ns()
logits = step(ids); t_prefill = time.perf_counter() - T0
generated, published_upto, epochs, taps, sequence, done = [], 0, [], [], 0, False


def publish_own(epoch):
    global published_upto, sequence, glm_used
    which = list(range(published_upto, len(own_slots)))
    if epoch == 0 and share is not None:
        which = share
    if epoch > 0 and args.prompt_only:
        which = []
    n = len(which)
    if n <= 0 or publisher is None or args.no_reverse:
        published_upto = len(own_slots)
        return None
    copies = args.prompt_copies if epoch == 0 and args.prompt_copies else args.copies
    rows = n * copies
    if glm_used + rows > args.glm_reserve:
        return {"skipped": "GLM's reserved span is full"}
    t0 = time.perf_counter(); own = tap_slots(cache, list(QL), rope, np.asarray([own_slots[i] for i in which]), np.asarray([own_positions[i] for i in which])); t_tap = time.perf_counter() - t0
    t0 = time.perf_counter(); latents = rev.read({l: np.concatenate((own[l][0].reshape(n, -1), own[l][1].reshape(n, -1)), axis=1).astype(np.float32) for l in QL}, 1.0); t_translate = time.perf_counter() - t0
    t0 = time.perf_counter(); sink = io.BytesIO(); np.savez(sink, **{f"l{l}": np.asarray(latents[l], dtype=np.float16) for l in GL}); body = sink.getvalue(); t_pack = time.perf_counter() - t0
    try:
        t0 = time.perf_counter(); delivered = publisher.deliver(args.session, sequence, body, copies); t_deliver = time.perf_counter() - t0
        t0 = time.perf_counter(); applied = publisher.confirm_applied(delivered, rows) if args.sync_confirm else None; t_applied = time.perf_counter() - t0
    except Exception as error:
        raise RuntimeError("reverse publication failed; session poisoned") from error
    record = {"session": args.session, "sequence": sequence, "source_sha256": delivered["sha256"], "own_tokens": [which[0], which[-1] + 1], "rows": rows, "bytes": len(body),
              "seconds": {"tap": round(t_tap, 5), "translate": round(t_translate, 5), "pack": round(t_pack, 5), "staged_both_ranks": delivered["staged_all_ranks_s"], "staged_and_released": round(t_deliver, 5), "wait_cache_applied_both_ranks": round(t_applied, 5)},
              "receipts": applied["receipts"] if applied else None}
    if applied is None:
        unconfirmed.append((delivered, rows, record))
    published_upto, sequence, glm_used = len(own_slots), sequence + 1, glm_used + rows
    return record


unconfirmed = []


def confirm_pending(block: bool = False):
    """Bind receipts for publications whose cache application was not awaited. Never skipped: at the end it blocks for the rest."""
    global publisher
    for item in list(unconfirmed):
        delivered, rows, record = item
        try:
            result = publisher.confirm_applied(delivered, rows) if block else publisher.poll_applied(delivered, rows)
        except Exception as error:
            raise RuntimeError("reverse confirmation failed; session poisoned") from error
        if result is not None:
            record["receipts"] = result["receipts"]; record["seconds"]["seen_applied_both_ranks_after_release"] = result["applied_all_ranks_s"]; unconfirmed.remove(item)


def advance_state(translated):
    """Call each linear-attention layer on its translated inputs with the live cache: its window and state move on."""
    for index, value in translated.items():
        lm.model.layers[index].linear_attn(mx.array(value[None].astype(np.float32)).astype(mx.bfloat16), mask=None, cache=cache[index])
    mx.eval([part for index in translated for part in (cache[index][0], cache[index][1]) if part is not None])


def drain_forward():
    global fwd_used, forward
    got = []
    if forward is not None and not isinstance(forward, ForwardMailbox):
        forward = ForwardMailbox(forward, args.session, GL)
    while forward is not None:
        t0 = time.perf_counter(); tap = forward.peek(); t_read = time.perf_counter() - t0
        if tap is None:
            return got
        try:
            if isinstance(tap, Complete):
                forward.acknowledge(tap)
                return got
            meta = tap.meta
            if args.no_forward:
                forward.acknowledge(tap)
                got.append({"session": args.session, "tap": meta["tap"], "dropped_by_control": True, "cache_applied": False})
                continue
            t0 = time.perf_counter(); entries, keys, order, _ = fwd.read(tap.latents)
            if rows_fix is not None:
                entries = corrected(entries, order, tap.latents)
            t_translate = time.perf_counter() - t0
            rows = validate_translation(entries, keys, order, tap.stop - tap.start, QL, INDEX_DIM)
            if fwd_used + rows > args.foreign_row_cap:
                raise RuntimeError("foreign row capacity exhausted; session poisoned")
            t0 = time.perf_counter()
            sources = tap.start + np.asarray(order, dtype=np.int64)
            positions = foreign_bank.positions(sources, args.own_start + len(own_positions))
            first_slot = append_entries(cache, entries, rope, positions, INDEX_DIM, dtype=mx.bfloat16, index_keys=keys)
            foreign_bank.remember(entries, sources, first_slot)
            mx.eval([value for layer in QL for value in (cache[layer].keys, cache[layer].values, getattr(cache[layer], "index_keys", None), getattr(cache[layer], "index_position_ids", None)) if value is not None])
            t_append = time.perf_counter() - t0
            t0 = time.perf_counter()
            if state is not None:                                     # GLM's tokens also pass through Qwen's recurrent layers, in GLM's order
                reduced = (np.concatenate([tap.latents[l] for l in sorted(GL)], axis=1) - state["mean"]) @ state["basis"]
                translated = {index: (reduced @ weight) * gain + bias for index, (weight, bias, gain) in state["layers"].items()}
                if args.state_at == "arrival":
                    advance_state(translated)
                else:
                    for index, value in translated.items():
                        pending_state.setdefault(index, []).append(value)
            t_state = time.perf_counter() - t0
            if args.save_taps is not None:                            # GLM's latents as they arrived, before the acknowledgement
                args.save_taps.mkdir(parents=True, exist_ok=True)
                np.savez(args.save_taps / f"{meta['tap']:06d}.npz", start=np.int64(tap.start), stop=np.int64(tap.stop),
                         **{f"l{layer}": tap.latents[layer].astype(np.float16) for layer in GL})
            forward.acknowledge(tap)
            got.append({"session": args.session, "tap": meta["tap"], "glm_positions": [tap.start, tap.stop], "file_mtime_ns": meta["file_mtime_ns"], "rows": rows, "bytes": meta["bytes"], "cache_applied": True, "qwen_own_tokens_when_appended": len(own_positions),
                        "qwen_generated_when_appended": len(generated), "seconds": {"mailbox_read_and_parse": round(t_read, 5), "translate_on_gpu": round(t_translate, 5), "append": round(t_append, 5),
                                                                         "state_advance": round(t_state, 5)}, "state_advanced": state is not None,
                        "studio_wall_ns": time.time_ns()})
            fwd_used += rows
        except Exception:
            forward.poisoned = True
            raise
    return got


epoch = 0
while True:
    t_epoch = time.perf_counter()
    made = 0
    if epoch > 0:
        t0 = time.perf_counter()
        while made < args.epoch_tokens and len(generated) < args.max_new and not done:
            nxt = int(mx.argmax(logits).item())
            if nxt in STOP:
                done = True; break
            generated.append(nxt); logits = step([nxt]); made += 1
            taps.extend(drain_forward())                                        # every token: GLM's newest entries are in the cache before the next token is computed
        t_generate = time.perf_counter() - t0
    else:
        t_generate = t_prefill
    reverse = publish_own(epoch)
    if publisher is not None:
        confirm_pending()
    appended = drain_forward()
    taps.extend(appended)
    epochs.append({"epoch": epoch, "tokens": made if epoch else len(ids), "since_start_s": round(time.perf_counter() - T0, 4), "generate_s": round(t_generate, 4), "reverse": reverse, "forward_taps_appended": [a.get("tap") for a in appended],
                   "epoch_overhead_s": round(time.perf_counter() - t_epoch - (t_generate if epoch else 0), 5)})
    epoch += 1
    if done or len(generated) >= args.max_new:
        break
if publisher is not None:
    confirm_pending(block=True)
taps.extend(wait_for_completion(sys.stdin, drain_forward, lambda: args.no_link or forward.finished))
followup_text, state_rows = None, 0
block = args.followup and args.followup_block and len(foreign_bank.sources) > 0
if block:                                                              # a framing line, then GLM's message as its own block, then the question
    step(tok.encode(f"<|im_end|>\n<|im_start|>user\n{FRAME}\n", add_special_tokens=False).ids)
    position_gap = int(foreign_bank.sources[-1] - foreign_bank.sources[0] + 1)
if args.followup and pending_state:                                    # GLM's whole message reaches the recurrent layers just before the question
    state_rows = int(sum(len(v) for v in next(iter(pending_state.values()))))
    advance_state({index: np.concatenate(values) for index, values in pending_state.items()})
if args.followup:
    opening = "\n\n" if block else "<|im_end|>\n<|im_start|>user\n"
    turn = tok.encode(f"{opening}{args.followup}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
    logits, reply = step(turn), []
    for _ in range(48):
        nxt = int(mx.argmax(logits).item())
        if nxt in STOP:
            break
        reply.append(nxt); logits = step([nxt])
    followup_text = tok.decode(reply)
report = {"session": args.session, "no_link": args.no_link, "followup_answer": followup_text, "state_at": args.state_at if state is not None else None,
          "state_rows_before_followup": state_rows, "followup_position_gap": position_gap, "text": tok.decode(generated), "generated_tokens": len(generated), "startup": {"prompt_tokens": len(ids), "prefill_s": round(t_prefill, 4)},
          "foreign_position_policy": "D16_source_age_per_step", "foreign_rephase_calls": foreign_bank.rephase_calls, "foreign_rephase_s": round(foreign_bank.rephase_seconds, 6),
          "peer_done": True, "forward_complete": False if args.no_link else forward.finished, "forward_tap_count": 0 if args.no_link else forward.expected,
          "studio_wall_start_ns": wall0_ns, "epochs": epochs, "forward_taps": taps, "glm_rows_used": glm_used, "qwen_virtual_rows_used": fwd_used, "peak_gb": round(mx.get_peak_memory() / 2**30, 1)}
args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=1))
print(json.dumps({"done": True, "text": report["text"][:400], "epochs": len(epochs), "forward_taps": len(taps)}), flush=True)
