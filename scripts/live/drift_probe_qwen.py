"""EXPLORATORY: is memory that arrives AFTER the reader's own prompt (or mid-generation) still usable?
Uses memories already translated by a forward run (--memory-dir) and the Studio drift worker."""
import argparse, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from livelib import OMLX_PY, STUDIO, passage_id, sh

parser = argparse.ArgumentParser()
parser.add_argument("--cases", type=Path, default=Path("scripts/live/cases.json"))
parser.add_argument("--memory-run", default="demo1", help="forward run folder name on the Studio holding memory/<id>.npz")
args = parser.parse_args()
sh("rsync", "-a", "scripts", "drift", f"{STUDIO}:drift/")
worker = subprocess.Popen(["ssh", STUDIO, f"cd ~/drift && {OMLX_PY} scripts/live/studio_drift_worker.py 2>local/studio/drift_worker.log"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)


def call(**cmd):
    worker.stdin.write(json.dumps(cmd) + "\n"); worker.stdin.flush()
    reply = json.loads(worker.stdout.readline())
    if not reply.get("ok", reply.get("ready")):
        raise RuntimeError(reply)
    return reply


print(json.loads(worker.stdout.readline()))
LINK = ("You are linked to another AI model through a shared memory. What it has read appears directly in your memory, not in this chat, and may arrive at any moment. "
        "Before answering, recall what is in that shared memory and answer from it. Never say you lack context.")
for c in json.loads(args.cases.read_text()):
    memory = f"local/studio/runs/{args.memory_run}/memory/{passage_id(c['passage'])}.npz"
    print(f"\nQ: {c['question']}")
    call(op="start"); call(op="extend", chat=c["question"])
    print("  no memory            :", repr(call(op="generate", tokens=40)["text"][:150]))
    call(op="start"); call(op="append", memory=memory); call(op="extend", chat=c["question"])
    print("  memory BEFORE prompt :", repr(call(op="generate", tokens=60)["text"][:150]))
    call(op="start"); call(op="extend", chat=c["question"]); call(op="append", memory=memory)
    print("  memory AFTER prompt  :", repr(call(op="generate", tokens=60)["text"][:150]))
    call(op="start"); call(op="extend", chat=c["question"], system=LINK); call(op="append", memory=memory)
    print("  AFTER prompt + link  :", repr(call(op="generate", tokens=70)["text"][:170]))
    call(op="start"); call(op="extend", chat=c["question"], system=LINK); first = call(op="generate", tokens=8)["text"]; call(op="append", memory=memory)
    print("  MID-generation + link:", repr(first), "->", repr(call(op="generate", tokens=70)["text"][:170]))
    call(op="start"); call(op="extend", chat=c["question"], system=LINK)
    print("  link, NO memory      :", repr(call(op="generate", tokens=50)["text"][:150]))
worker.stdin.write('{"op":"quit"}\n'); worker.stdin.flush()
