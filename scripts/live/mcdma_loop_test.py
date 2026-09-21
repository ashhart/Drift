"""Coordinate exploratory MCDMA epochs and verify their recorded causal schedule."""
import argparse, json, shlex, statistics, sys, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from livelib import OMLX_PY, SPARK, SPARK_PEERS, STUDIO
from loop_processes import LoopProcesses, SSH_OPTIONS

parser = argparse.ArgumentParser()
parser.add_argument("--glm-messages", type=Path, required=True)
parser.add_argument("--qwen-messages", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--epoch-tokens", type=int, default=16)
parser.add_argument("--qwen-max-new", type=int, default=160)
parser.add_argument("--glm-max-new", type=int, default=512)
parser.add_argument("--copies", type=int, default=3)
parser.add_argument("--reserve", type=int, default=1024)
parser.add_argument("--no-link", action="store_true")
parser.add_argument("--sync-confirm", action="store_true")
parser.add_argument("--no-reverse", action="store_true")
parser.add_argument("--prompt-copies", type=int)
parser.add_argument("--publish-text")
parser.add_argument("--prompt-only", action="store_true")
parser.add_argument("--followup")
parser.add_argument("--no-forward", action="store_true")
parser.add_argument("--wall-seconds", type=float, default=600)
parser.add_argument("--startup-seconds", type=float, default=60)
parser.add_argument("--reverse-artifact", type=Path, default=Path("local/live/stacked3_rev.npz"))
args = parser.parse_args()
if not args.reverse_artifact.is_file():
    parser.error("--reverse-artifact must name an existing translator file")
args.out.mkdir(parents=True, exist_ok=False); session = f"loop-{uuid.uuid4().hex}"; hosts = (SPARK, *SPARK_PEERS)
extra = (f"--publish-text {shlex.quote(args.publish_text)}" if args.publish_text else "") + (f" --followup {shlex.quote(args.followup)}" if args.followup else "")
with LoopProcesses(session, args.wall_seconds, args.startup_seconds) as owned:
    for host in hosts:
        owned.run("scp", "-q", *SSH_OPTIONS, "drift/serving/mcdma_mailbox.py", "scripts/mcdma_target/spark_bridge.py", "scripts/mcdma_target/npz_tile.py", "scripts/mcdma_target/spark_glm_session.py", "scripts/mcdma_target/glm_completion_sse.py", f"{host}:/root/drift-live/mcdma-target/")
    owned.run("rsync", "-a", "-e", shlex.join(["ssh", *SSH_OPTIONS]), "scripts", "drift", f"{STUDIO}:drift-frontier/")
    owned.run("rsync", "-a", "-e", shlex.join(["ssh", *SSH_OPTIONS]), str(args.reverse_artifact.resolve()), f"{STUDIO}:drift-frontier/local/live/stacked3_rev.npz")
    owned.run("ssh", *SSH_OPTIONS, SPARK, f"cat > /root/drift-live/mcdma-target/{session}.messages.json", input=args.glm_messages.read_text())
    owned.run("ssh", *SSH_OPTIONS, STUDIO, f"mkdir -p drift-frontier/out/{session} && cat > drift-frontier/out/{session}/messages.json", input=args.qwen_messages.read_text())
    bridges = []
    if not args.no_link:
        for rank, host in enumerate(hosts):
            bridge = owned.spawn(host, f"exec python3 /root/drift-live/mcdma-target/spark_bridge.py --rank {rank} {'--head' if rank == 0 else ''} --seconds {args.wall_seconds} --log /root/drift-live/mcdma-target/bridge-{session}.jsonl")
            bridges.append(bridge)
            event = json.loads(bridge.readline(owned.startup))
            if event.get("rank") != rank or not event.get("session"):
                raise RuntimeError("bridge did not acknowledge startup")
    qwen = owned.spawn(STUDIO, f"cd ~/drift-frontier && {OMLX_PY} scripts/live/studio_mcdma_loop.py --session {session} --messages out/{session}/messages.json --epoch-tokens {args.epoch_tokens} "
                             f"--max-new {args.qwen_max_new} --copies {args.copies} --glm-reserve {args.reserve} {'--no-link' if args.no_link else ''} {'--sync-confirm' if args.sync_confirm else ''} {'--no-reverse' if args.no_reverse else ''} {'--no-forward' if args.no_forward else ''} {'--prompt-only' if args.prompt_only else ''} {f'--prompt-copies {args.prompt_copies}' if args.prompt_copies else ''} {extra} --out out/{session}/qwen.json 2>out/{session}/worker.log")
    if json.loads(qwen.readline(owned.startup)).get("ready") is not True:
        raise RuntimeError("Qwen did not acknowledge startup")
    glm = owned.spawn(SPARK, f"exec /root/drift-live/spark_run.sh mcdma-target/spark_glm_session.py --session {session} --rows {args.reserve} --tap --max-new {args.glm_max_new} --messages /root/drift-live/mcdma-target/{session}.messages.json")
    ready = json.loads(glm.readline(owned.startup))
    if ready.get("ready") is not True:
        raise RuntimeError("GLM did not acknowledge startup")
    glm.release(); qwen.release()
    stream = json.loads(glm.readline(owned.remaining())); answer = json.loads(glm.readline(owned.remaining()))
    glm.finish(owned.remaining())
    terminal = answer.get("connector_finished")
    if type(terminal) is not dict or terminal.get("failed") != "":
        raise RuntimeError("GLM did not finish successfully")
    qwen.peer_done()
    qwen.finish(owned.remaining())
    report = json.loads(owned.run("ssh", *SSH_OPTIONS, STUDIO, f"cat drift-frontier/out/{session}/qwen.json"))
    bridge_log = [] if args.no_link else [json.loads(line) for line in owned.run("ssh", *SSH_OPTIONS, SPARK, f"cat /root/drift-live/mcdma-target/bridge-{session}.jsonl").splitlines()]
med = lambda v: round(statistics.median(v), 5) if v else None
reverse = [e["reverse"] for e in report["epochs"] if e["reverse"] and "seconds" in e["reverse"] and e["reverse"].get("receipts")]
taps = [t for t in report["forward_taps"] if "tap" in t and "seconds" in t]
tap_clock = sorted((t["file_mtime_ns"], t["glm_positions"][1]) for t in taps)                 # head clock: GLM had verified positions < stop at that time
schedule = []
for r in reverse:
    at = r["receipts"]["spark-a.invalid"]["receipt_mtime_ns"]
    before = [stop for ns, stop in tap_clock if ns <= at]; after = [stop for ns, stop in tap_clock if ns > at]
    schedule.append({"sequence": r["sequence"], "qwen_own_tokens": r["own_tokens"], "visible_to_glm_between_positions": [max(before) if before else None, min(after) if after else None]})
published = {e["tap"]: e for e in bridge_log if e.get("event") == "tap_published"}; acked = {e["tap"]: e for e in bridge_log if e.get("event") == "tap_acknowledged"}
spark_side = [(published[t]["ns"] - published[t]["file_mtime_ns"]) / 1e6 for t in published] ; spark_ack = [(acked[t]["ns"] - published[t]["ns"]) / 1e6 for t in acked if t in published]
released = {e["sequence"]: e["ns"] for e in bridge_log if e.get("event") == "released"}
relayed = {e["sequence"]: e for e in bridge_log if e.get("event") == "receipt_relayed"}
on_head = {"visible_in_inbox_to_connector_receipt_written_ms": med([(relayed[k]["receipt_mtime_ns"] - released[k]) / 1e6 for k in relayed if k in released and k > 0]),
           "receipt_written_to_relayed_into_region_ms": med([(relayed[k]["ns"] - relayed[k]["receipt_mtime_ns"]) / 1e6 for k in relayed if k > 0])}
chunks = stream["stream"]["chunks"]
glm_rate = {"chunks": len(chunks), "median_ms_between_chunks": med([(b[0] - a[0]) / 1e6 for a, b in zip(chunks, chunks[1:])]), "tokens_per_s_overall": None, "generated_tokens": answer["usage"]["completion_tokens"], "max_new_tokens_cap": args.glm_max_new}
summary = {"session": session, "reverse_on_head_clock": on_head, "glm_decode": glm_rate, "no_link": args.no_link, "glm": {"prompt": ready, "first_token_s": answer["first_token_s"], "total_s": answer["total_s"], "connector": answer["connector_finished"], "text": answer["text"], "finish_reason": answer["finish_reason"], "usage": answer["usage"], "stream_done": answer["stream_done"]},
           "qwen": {"followup_answer": report.get("followup_answer"), "text": report["text"], "generated_tokens": report["generated_tokens"], "startup": report["startup"]},
           "startup_reported_separately": {"glm_first_token_s": answer["first_token_s"], "qwen_prefill_s": report["startup"]["prefill_s"], "first_publication_wait_applied_s": reverse[0]["seconds"]["wait_cache_applied_both_ranks"] if reverse else None},
           "steady_state_reverse_ms": {k: med([r["seconds"][k] * 1e3 for r in reverse[1:]]) for k in (reverse[0]["seconds"] if reverse else {})}, "reverse_publications": len(reverse),
           "reverse_errors": [e["reverse"] for e in report["epochs"] if e["reverse"] and ("error" in e["reverse"] or "skipped" in e["reverse"] or not e["reverse"].get("receipts"))],
           "steady_state_forward_ms": {k: med([t["seconds"][k] * 1e3 for t in taps]) for k in (taps[0]["seconds"] if taps else {})}, "forward_taps": len(taps),
           "forward_on_spark_ms": {"tap_file_written_to_published_in_region": med(spark_side), "published_to_acknowledged_by_studio": med(spark_ack)},
           "epoch_overhead_ms_median": med([e["epoch_overhead_s"] * 1e3 for e in report["epochs"][1:]]), "qwen_epoch_generate_ms_median": med([e["generate_s"] * 1e3 for e in report["epochs"][1:]]),
           "causal_schedule": {"reverse": schedule, "forward": [{"tap": t["tap"], "glm_positions": t["glm_positions"], "qwen_generated_when_appended": t["qwen_generated_when_appended"]} for t in taps],
                               "forward_in_order": [t["tap"] for t in taps] == list(range(len(taps))), "reverse_in_order": [r["sequence"] for r in reverse] == list(range(len(reverse)))}}
(args.out / "report.json").write_text(json.dumps({"summary": summary, "qwen": report, "glm_stream": stream, "bridge_log": bridge_log}, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "causal_schedule"}, indent=1)); print(json.dumps(summary["causal_schedule"])[:1500])
