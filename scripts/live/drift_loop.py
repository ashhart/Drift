"""Coordinate exploratory GLM and Qwen drift through bounded SSH file delivery."""
import hashlib, json, subprocess, sys, threading, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from livelib import GLM_LAYERS, OMLX_PY, QWEN_LAYERS, SPARK, SPARK_PEERS, STUDIO, load_reader
from drift.translate.stacked import StackedReader
from drift.eval.stream_repetition import RepetitionDetected, StreamingRepetition
from drift.serving.live_control import PendingMemory, interrupt_processes
from drift.serving.live_rank_delivery import RankDelivery
from drift.serving.live_options import parse_run_options
from drift.serving.live_forward import forward_publication
from drift.serving.live_publication import count, load_publication, wire_arrays

args, repetition_policy, parser = parse_run_options()
pending_memory = PendingMemory(args.max_pending_publications)
repetition = {who: StreamingRepetition(repetition_policy) for who in ("glm", "qwen")}
sc = json.loads(args.scenario.read_text())
args.out.mkdir(parents=True, exist_ok=True)
if any(args.out.iterdir()):
    parser.error("use a new empty output directory; live sessions cannot be resumed safely")
run = args.out.name
CM = ["-o", "ControlMaster=auto", "-o", f"ControlPath=/tmp/drift-cm-%C", "-o", "ControlPersist=180", "-o", "BatchMode=yes",
      "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2"]
ssh = lambda host, cmd, *, timeout=None: subprocess.run(["ssh", *CM, host, cmd], check=True, capture_output=True, text=True, timeout=args.io_timeout if timeout is None else min(args.io_timeout, timeout)).stdout
push = lambda src, host, dst: subprocess.run(["scp", *CM, "-q", str(src), f"{host}:{dst}"], check=True, timeout=args.io_timeout)
pull = lambda host, src, dst: subprocess.run(["rsync", "-a", "-e", "ssh " + " ".join(CM), f"{host}:{src}", str(dst)], check=True, timeout=args.io_timeout)
rank_delivery = None if args.no_link else RankDelivery((SPARK, *SPARK_PEERS), run, ssh, args.io_timeout, lambda: None)
fwd = rev = None
forward_recipe = None
if not args.no_link:
    fwd = load_reader(args.forward) if args.forward_recipe == "legacy" else load_reader(
        args.forward, recipe=args.forward_recipe, manifest=args.forward_manifest)
    if args.forward_recipe != "legacy":
        if args.gain_forward is not None and args.gain_forward != fwd.gain_power:
            parser.error("--gain-forward conflicts with the pinned recipe")
        args.gain_forward = fwd.gain_power
    forward_recipe = getattr(fwd, "metadata", {"kind": "legacy", "sha256": getattr(fwd, "sha256", None),
                                              "gain_power": args.gain_forward, "live_qualification": "BLOCKED"})
    rev = StackedReader.load(args.reverse, QWEN_LAYERS, GLM_LAYERS, kv_heads=0, head_dim=512)
    glm_layout = {f"l{l}": (rev.head_dim,) for l in GLM_LAYERS}
    qwen_layout = {f"{kind}{l}": (fwd.kv_heads, fwd.head_dim) for l in QWEN_LAYERS for kind in ("k", "v")}
T0, timeline, lock = time.time(), [], threading.Lock()


def log(who, kind, **more):
    with lock:
        timeline.append({"t": round(time.time() - T0, 2), "who": who, "kind": kind, **more})


for host in (STUDIO, SPARK, *SPARK_PEERS):
    ssh(host, "true")
subprocess.run(["rsync", "-a", "-e", "ssh " + " ".join(CM), "scripts", "drift", f"{STUDIO}:drift/"], check=True, timeout=args.io_timeout)
for f in ("scripts/live/spark_run.sh", "scripts/live/spark_live_session.py", "drift/serving/live_session_control.py"):
    push(f, SPARK, "/root/drift-live/")
ssh(SPARK, f"mkdir -p /root/drift-live/runs/{run}")
(args.out / "glm_messages.json").write_text(json.dumps([{"role": "system", "content": sc.get("glm_system", sc.get("system"))}, {"role": "user", "content": sc["glm_user"]}]))
push(args.out / "glm_messages.json", SPARK, f"/root/drift-live/runs/{run}/messages.json")
if not args.no_link:
    for peer in SPARK_PEERS:
        ssh(peer, f"test ! -e /dev/shm/glm53-handoff/tp-live-out/{run} && rm -rf /dev/shm/glm53-handoff/tp-live-in/{run} && mkdir -p /dev/shm/glm53-handoff/tp-live-in/{run}")
remote = f"local/studio/runs/{run}"
ssh(STUDIO, f"rm -rf drift/{remote} && mkdir -p drift/{remote}/in drift/{remote}/out")
worker = subprocess.Popen(["ssh", *CM, STUDIO, f"cd ~/drift && {OMLX_PY} scripts/live/studio_drift_worker.py 2>local/studio/drift_worker.log"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
stopping = threading.Event()
failures, processes, supervised = [], [worker], []


def fail(component, error):
    with lock:
        first = not failures
        if first:
            # Exceptions can contain model text or file contents: record only their type.
            failures.append({"component": component, "error_type": type(error).__name__})
        active = list(processes) if first else []
        graceful = list(supervised)
    stopping.set()
    interrupt_processes(active, graceful=graceful)


def check_failure():
    if failures:
        raise RuntimeError(f"INVALID: {failures[0]['component']} failed ({failures[0]['error_type']})")


if rank_delivery is not None:
    rank_delivery.check_failure, rank_delivery.pause = check_failure, stopping.wait


def observe_output(who, text, *, final=False):
    with lock:
        detected = repetition[who].feed(text, final=final)
    if detected:
        raise RepetitionDetected(f"{who} output exceeded the configured safety policy")


def repetition_report():
    with lock:
        return {who: monitor.report() for who, monitor in repetition.items()}


def expire():
    fail("deadline", TimeoutError())


def guarded(component, target):
    def run():
        try:
            target()
        except Exception as error:
            fail(component, error)
        finally:
            if component == "glm":
                state["glm_done"] = True
            elif component == "tap":
                state["tap_done"] = True
    return run


def call(**cmd):
    check_failure()
    worker.stdin.write(json.dumps(cmd) + "\n"); worker.stdin.flush()
    reply = json.loads(worker.stdout.readline())
    if not reply.get("ok", reply.get("ready")):
        raise RuntimeError(reply)
    return reply


state = {"glm_done": False, "qwen_done": False, "glm_text": "", "qwen_text": "", "reserved_used": 0, "forward_reserved_used": 0, "glm_first": None}


def glm_thread():
    link_flag = " --no-link" if args.no_link else ""
    proc = subprocess.Popen(["ssh", *CM, SPARK, f"/root/drift-live/spark_run.sh spark_live_session.py --session {run} --messages runs/{run}/messages.json --reserve {args.reserve} --max-new {args.max_new} --control-stdin{link_flag}"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    with lock:
        processes.append(proc)
        supervised.append(proc)
    if stopping.is_set():
        proc.terminate()
        return
    completed = False
    for line in proc.stdout:
        check_failure()
        event = json.loads(line)
        if event.get("done"):
            completed = True
        if "text" in event:
            observe_output("glm", event["text"])
            state["glm_text"] += event["text"]; log("glm", "says", text=event["text"])
        elif event.get("started"):
            if state["glm_first"] is not None:
                raise ValueError("duplicate GLM start event")
            state["glm_first"] = args.reserve + count(event["span_start"])
            log("glm", "started", own_prompt_tokens=event["own_prompt_tokens"])
    if proc.wait(timeout=args.io_timeout) != 0:
        raise RuntimeError("GLM process exited unsuccessfully")
    if not completed:
        raise RuntimeError("GLM stream ended without completion")
    observe_output("glm", "", final=True)
    log("glm", "finished")


def glm_tap_thread():
    """GLM's decode-time taps -> Qwen entries, queued for the Qwen loop to append."""
    seen = 0
    next_position = None
    (args.out / "glm_taps").mkdir(exist_ok=True)
    while not stopping.is_set():
        finished = state["glm_done"]
        rank_delivery.health()
        if next_position is None:
            next_position = state["glm_first"]
            if next_position is None:
                if finished:
                    raise ValueError("GLM completed without a source cursor")
                stopping.wait(0.05)
                continue
        try:
            pull(SPARK, f"/dev/shm/glm53-handoff/tp-live-out/{run}/", args.out / "glm_taps")
        except subprocess.CalledProcessError:
            if finished:
                raise  # A completed writer must have a retrievable final outbox.
            stopping.wait(0.15)
            continue
        if any((args.out / "glm_taps").glob("error.rank*")):
            raise RuntimeError("GLM connector reported a failed publication")
        while (args.out / "glm_taps" / f"{seen:06d}.npz").exists():
            check_failure()
            z = load_publication(args.out / "glm_taps" / f"{seen:06d}.npz", glm_layout,
                                 metadata=("start", "stop"), max_rows=args.max_publication_rows)
            rows = len(z[f"l{GLM_LAYERS[0]}"])
            if z["start"] != next_position or z["stop"] != next_position + rows:
                raise ValueError("GLM publication source cursor mismatch")
            payload, reader_rows = forward_publication(
                fwd, {l: z[f"l{l}"] for l in GLM_LAYERS}, rows, qwen_layout, args.gain_forward,
                max_rows=args.max_publication_rows, remaining=args.reserve - state["forward_reserved_used"])
            path = args.out / f"to_qwen_{seen:06d}.npz"
            np.savez(path, **payload)
            pending_memory.publish(f"{remote}/in/{seen:06d}.npz", reader_rows,
                                   lambda: push(path, STUDIO, f"drift/{remote}/in/{seen:06d}.npz"))
            state["forward_reserved_used"] += reader_rows
            log("link", "glm->qwen", positions=[int(z["start"]), int(z["stop"])], source_rows=rows, reader_rows=reader_rows)
            seen += 1
            next_position = z["stop"]
        if finished:
            names = {p.name for p in (args.out / "glm_taps").glob("*.npz")}
            if names != {f"{i:06d}.npz" for i in range(seen)}:
                raise ValueError("final GLM publication sequence has missing or unexpected files")
            return
        time.sleep(0.15)


def send_to_glm(seq, first):
    out = f"{remote}/out/{seq:06d}.npz"
    tapped = call(op="tap", first=first, out=out)
    n = count(tapped["tapped"])
    if count(tapped["next_first"]) != first + n or n > args.max_publication_rows:
        raise ValueError("Qwen publication source cursor mismatch")
    if not n:
        return seq, first
    pull(STUDIO, f"drift/{out}", args.out / f"qwen_tap_{seq:06d}.npz")
    z = load_publication(args.out / f"qwen_tap_{seq:06d}.npz", qwen_layout, max_rows=args.max_publication_rows)
    if len(z[f"k{QWEN_LAYERS[0]}"]) != n:
        raise ValueError("Qwen publication row count mismatch")
    flat = {l: np.concatenate((z[f"k{l}"].reshape(len(z[f"k{l}"]), -1), z[f"v{l}"].reshape(len(z[f"v{l}"]), -1)), axis=1).astype(np.float32) for l in QWEN_LAYERS}
    latents = rev.read(flat, args.gain_reverse)
    rows = tapped["tapped"] * args.copies
    if state["reserved_used"] + rows > args.reserve:
        raise RuntimeError("GLM reserve exhausted; refusing a silently dropped publication")
    index = np.arange(rows) % tapped["tapped"]
    path = args.out / f"to_glm_{seq:06d}.npz"
    validated = wire_arrays({f"l{l}": value for l, value in latents.items()}, glm_layout, n)
    np.savez(path, **{key: value[index] for key, value in validated.items()})
    for peer in SPARK_PEERS:                                        # other ranks first: the head's file is what schedules the write
        push(path, peer, f"/dev/shm/glm53-handoff/tp-live-in/{run}/{seq:06d}.npz")
    push(path, SPARK, f"/dev/shm/glm53-handoff/tp-live-in/{run}/.{seq:06d}.tmp.npz")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rank_delivery.admit(seq, digest)
    ssh(SPARK, f"mv /dev/shm/glm53-handoff/tp-live-in/{run}/.{seq:06d}.tmp.npz /dev/shm/glm53-handoff/tp-live-in/{run}/{seq:06d}.npz")
    rank_delivery.wait_applied(seq, rows, digest)
    state["reserved_used"] += rows
    log("link", "qwen->glm", tokens=tapped["tapped"], rows=rows)
    return seq + 1, tapped["next_first"]


threads = [threading.Thread(target=guarded("glm", glm_thread), daemon=True)]
if not args.no_link:
    threads.append(threading.Thread(target=guarded("tap", glm_tap_thread), daemon=True))
deadline = threading.Timer(args.max_seconds, expire)
deadline.daemon = True
started_threads = []
deadline.start()
try:
    if not json.loads(worker.stdout.readline()).get("ready"):
        raise RuntimeError("Studio worker did not become ready")
    call(op="start")
    for t in threads:
        t.start()
        started_threads.append(t)
    qsys = sc.get("qwen_system", sc.get("system")) or ""
    opened = call(op="extend", chat=sc["qwen_user"], system=qsys, span=args.reserve if "@@DRIFT@@" in qsys else None); log("qwen", "started")
    seq, first, generated = 0, int(opened.get("system_tokens", 0)), 0
    if not args.no_link:
        seq, first = send_to_glm(seq, first)
        waited = time.monotonic()
        while not pending_memory:
            check_failure()
            if state.get("tap_done") or time.monotonic() - waited >= args.io_timeout:
                raise RuntimeError("initial GLM memory did not arrive")
            stopping.wait(0.05)
        log("qwen", "barrier passed", waited=round(time.monotonic() - waited, 2))
    while not state["qwen_done"]:
        check_failure()
        if not args.no_link:
            while (delivery := pending_memory.take()) is not None:
                call(op="append", memory=delivery.path)
                pending_memory.complete(delivery)
                log("qwen", "memory appended", tokens=delivery.rows)
            seq, first = send_to_glm(seq, first)
        piece = call(op="generate", tokens=min(args.epoch_tokens, args.max_new - generated))
        check_failure()
        observe_output("qwen", piece["text"], final=piece["done"] or generated + piece["tokens"] >= args.max_new)
        if piece["tokens"] == 0 and not piece["done"]:
            raise RuntimeError("Studio generation made no progress")
        generated += piece["tokens"]; state["qwen_text"] += piece["text"]; log("qwen", "says", text=piece["text"])
        state["qwen_done"] = piece["done"] or generated >= args.max_new
    log("qwen", "finished")
    for t in started_threads:
        while t.is_alive():
            check_failure()
            t.join(timeout=0.2)
    if rank_delivery is not None:
        rank_delivery.health()
    if pending_memory.report()['pending']:
        raise RuntimeError('reader finished with undelivered memory publications')
    deadline.cancel()
    deadline.join(timeout=1)
    check_failure()
except Exception as error:
    fail("coordinator", error)
    (args.out / "failure.json").write_text(json.dumps({"verdict": "INVALID", "linked": not args.no_link,
        "failure": failures[0], "forward_recipe": forward_recipe, "pending_memory": pending_memory.report(), "repetition": repetition_report(), "seconds": round(time.time() - T0, 2)}, indent=2) + "\n")
    check_failure()
finally:
    stopping.set()
    deadline.cancel()
    try:
        worker.stdin.write('{"op":"quit"}\n'); worker.stdin.flush(); worker.stdin.close()
    except (OSError, ValueError):
        pass
    with lock:
        active = list(processes)
    for proc in active:
        try:
            if failures and proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    for t in started_threads:
        t.join(timeout=1)
check = lambda text: {k: (v.lower() in text.lower()) for k, v in sc.get("expect", {}).items()}
report = {"scenario": sc.get("name"), "forward_recipe": forward_recipe, "linked": not args.no_link, "glm_text": state["glm_text"], "qwen_text": state["qwen_text"], "glm_expect": check(state["glm_text"]), "qwen_expect": check(state["qwen_text"]),
          "seconds": round(time.time() - T0, 1), "timeline": sorted(timeline, key=lambda e: e["t"]), "pending_memory": pending_memory.report(), "repetition": repetition_report()}
(args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print("\nGLM :", state["glm_text"].replace("\n", " ")[:700]); print("\nQWEN:", state["qwen_text"].replace("\n", " ")[:700])
print("\nexpect GLM", report["glm_expect"], "| QWEN", report["qwen_expect"])
print("link events:", [(e["t"], e["kind"]) for e in report["timeline"] if e["who"] == "link"][:14])
