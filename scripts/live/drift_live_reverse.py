"""Live exchange, reverse direction: Qwen3.8 (Studio, oMLX) reads passages; its K/V is translated into GLM-5.3
latents and written into the cache of the running vLLM server on the Sparks (DriftGlm53Connector), and GLM
answers questions it was never shown the text for.

  PYTHONPATH=. .venv/bin/python scripts/live/drift_live_reverse.py --cases scripts/live/cases.json --out local/live/rev1 \\
      --translator local/live/stacked2_rev.npz --gain-power 1.5 [--controls]
The memory occupies a span of placeholder positions sized in 64-token blocks (spark_inject_answer.plan); the
connector tiles the entries across it (GLM-5.3 attention is NoPE, so copies are position-free and act as a positive
log-prior of log(copies))."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizers import Tokenizer
from livelib import GLM_LAYERS, OMLX_PY, QWEN_LAYERS, SPARK, SPARK_PEERS, STUDIO, glm_read, passage_id, sh
from drift.translate.stacked import StackedReader

parser = argparse.ArgumentParser()
parser.add_argument("--cases", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--translator", type=Path, default=Path("local/live/stacked2_rev.npz"))
parser.add_argument("--gain-power", type=float, default=1.5)
parser.add_argument("--copies", type=float, default=1.0, help="copies of the memory across the placeholder span (a log(copies) attention prior)")
parser.add_argument("--placeholder-id", type=int, default=198)
parser.add_argument("--controls", action="store_true", help="also inject GLM's OWN latents of the passage (mechanism check) and paste the text (baseline)")
parser.add_argument("--max-new", type=int, default=80)
args = parser.parse_args()
cases = json.loads(args.cases.read_text())
for c in cases:
    c["id"] = passage_id(c["passage"])
args.out.mkdir(parents=True, exist_ok=True)
run, timing, t0 = args.out.name, {}, time.time()

# 1) writer: Qwen reads each passage on the Studio; canonical K/V tapped through the cache seam
tok = Tokenizer.from_file("local/tok/qwen/tokenizer.json")
(args.out / "qwen.ids.json").write_text(json.dumps({"records": [{"id": c["id"], "ids_qwen": tok.encode(c["passage"], add_special_tokens=False).ids} for c in cases]}))
remote = f"local/studio/runs/{run}"
sh("ssh", STUDIO, f"rm -rf drift/{remote} && mkdir -p drift/{remote}")
sh("rsync", "-a", "scripts", "drift", f"{STUDIO}:drift/")
sh("scp", "-q", str(args.out / "qwen.ids.json"), f"{STUDIO}:drift/{remote}/ids.json")
print("Qwen:", sh("ssh", STUDIO, f"cd ~/drift && {OMLX_PY} scripts/live/studio_tap_qwen.py --ids {remote}/ids.json --out {remote}/taps 2>/dev/null | tail -1").strip())
sh("rsync", "-a", f"{STUDIO}:drift/{remote}/taps/", str(args.out / "taps_qwen"))
timing["qwen_read_and_tap"] = round(time.time() - t0, 1); t1 = time.time()

# 2) sidecar: Qwen K/V -> GLM latents, tiled across one placeholder block
reader = StackedReader.load(args.translator, QWEN_LAYERS, GLM_LAYERS, kv_heads=0, head_dim=512) if args.translator.exists() else None   # None: controls only
(args.out / "inject").mkdir(exist_ok=True)


def tile(latents: dict, name: str) -> dict:
    """The connector tiles the entries across the memory span; the file holds one copy."""
    np.savez(args.out / "inject" / f"{name}.npz", **{f"l{l}": latents[l].astype(np.float16) for l in GLM_LAYERS})
    return {"name": name, "rows": int(next(iter(latents.values())).shape[0])}


jobs = []
for c in cases if reader else []:
    z = np.load(args.out / "taps_qwen" / f"{c['id']}.npz")
    flat = {l: np.concatenate((z[f"k{l}"].reshape(len(z[f"k{l}"]), -1), z[f"v{l}"].reshape(len(z[f"v{l}"]), -1)), axis=1).astype(np.float32) for l in QWEN_LAYERS}
    c["memory"] = tile(reader.read(flat, args.gain_power), f"{run}-{c['id']}-tp")
for i, c in enumerate(cases):
    memories = {"drift": c["memory"]} if reader else {}
    if reader and len(cases) > 1:
        memories["wrong_memory"] = cases[(i + 1) % len(cases)]["memory"]
    jobs.append({"id": c["id"], "question": c["question"], "answer": c.get("answer"), "memories": memories, **({"passage": c["passage"]} if args.controls else {})})
if args.controls:                                                  # GLM's own latents of the passage, written back the same way
    taps = glm_read({c["id"]: c["passage"] for c in cases}, args.out, run)
    for c, job in zip(cases, jobs):
        own = np.load(taps / f"{c['id']}.npz")
        job["memories"]["own_latents"] = tile({l: own[f"l{l}"].astype(np.float32) for l in GLM_LAYERS}, f"{run}-{c['id']}-own")
(args.out / "jobs.json").write_text(json.dumps(jobs))
timing["translate_and_tile"] = round(time.time() - t1, 1); t2 = time.time()

# 3) reader: memory files go to every tensor-parallel host (MLA latents are replicated), then GLM answers
for host in [SPARK, *SPARK_PEERS]:
    sh("ssh", host, "mkdir -p /dev/shm/glm53-handoff/tp-inject")
    sh("rsync", "-a", str(args.out / "inject") + "/", f"{host}:/dev/shm/glm53-handoff/tp-inject/")
sh("scp", "-q", "scripts/live/spark_run.sh", "scripts/live/spark_inject_answer.py", f"{SPARK}:/root/drift-live/")
sh("ssh", SPARK, f"mkdir -p /root/drift-live/runs/{run}")
sh("scp", "-q", str(args.out / "jobs.json"), f"{SPARK}:/root/drift-live/runs/{run}/jobs.json")
print("GLM:", sh("ssh", SPARK, f"/root/drift-live/spark_run.sh spark_inject_answer.py --jobs runs/{run}/jobs.json --out runs/{run}/answers.json "
                 f"--copies {args.copies} --placeholder-id {args.placeholder_id} --max-new {args.max_new} | tail -1").strip())
sh("scp", "-q", f"{SPARK}:/root/drift-live/runs/{run}/answers.json", str(args.out / "answers.json"))
for host in [SPARK, *SPARK_PEERS]:
    sh("ssh", host, f"rm -f /dev/shm/glm53-handoff/tp-inject/{run}-*")
timing["glm_answer"] = round(time.time() - t2, 1)
report = json.loads((args.out / "answers.json").read_text()); report["timing_seconds"] = timing
(args.out / "answers.json").write_text(json.dumps(report, indent=2) + "\n")
for r in report["results"]:
    print(f"\nQ: {r['question']}   (reference: {r['answer']})")
    for name in ("no_memory", "placeholders_only", "drift", "wrong_memory", "own_latents", "text_in_prompt"):
        if name in r:
            note = "" if "inject" not in r[name] else (" [INJECT FAILED: " + r[name]["inject"]["error"][:120] + "]" if "error" in r[name]["inject"] else f" [wrote {r[name]['inject']['layers']} layers]")
            print(f"  {name:>17}:{note} {r[name]['text'][:200]!r}")
print("\ntiming:", timing)
