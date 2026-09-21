"""EXPLORATORY: run the live cases under several translator variants (memories built on this host)."""
import json, hashlib, subprocess, sys, numpy as np
from pathlib import Path
cases = json.load(open("scripts/live/cases.json")); out = Path("local/live/variants"); out.mkdir(exist_ok=True)
S = np.load(sys.argv[1] if len(sys.argv) > 1 else "local/live/stacked.npz"); GL, QL = [3 + 4 * i for i in range(11)], [3 + 4 * i for i in range(12)]
R = "/Applications/oMLX.app/Contents/Resources"
sh = lambda *c: subprocess.run(c, check=True, capture_output=True, text=True).stdout
def save(name, cid, d):
    (out / name).mkdir(exist_ok=True); np.savez(out / name / f"{cid}.npz", **{k: v.astype(np.float16) for k, v in d.items()})
    return f"local/studio/variants/{name}/{cid}.npz"
jobs = []
for c in cases:
    cid = hashlib.sha256(c["passage"].encode()).hexdigest()[:16]
    g = np.load(f"local/live/run1/taps_glm/{cid}.npz"); own = np.load(f"local/live/tolerance/own/{cid}.npz")
    X = np.concatenate([g[f"l{l}"].astype(np.float32) for l in GL], 1) - S["g_mean"]
    mems = {}
    for name, power in (("stacked", 0.0), ("stacked_gain", 1.0), ("stacked_gain2", 2.0)):
        d = {}
        for l in QL:
            y = (X @ S[f"W{l}"]) * S[f"gain{l}"] ** power + S[f"b{l}"]
            d[f"k{l}"], d[f"v{l}"] = y[:, :512].reshape(-1, 2, 256), y[:, 512:].reshape(-1, 2, 256)
        mems[name] = save(name, cid, d)
    for name, (sk, sv) in (("own_shrinkK", (0.5, 1.0)), ("own_shrinkV", (1.0, 0.5))):
        d = {}
        for k in own.files:
            x = own[k].astype(np.float32); m = x.mean(0, keepdims=True); d[k] = m + (sk if k[0] == "k" else sv) * (x - m)
        mems[name] = save(name, cid, d)
    jobs.append({"id": cid, "question": c["question"], "answer": c["answer"], "memories": mems})
(out / "jobs.json").write_text(json.dumps(jobs))
sh("ssh", "studio", "rm -rf drift/local/studio/variants"); sh("rsync", "-a", str(out) + "/", "studio:drift/local/studio/variants/")
sh("ssh", "studio", f'cd ~/drift && PYTHONPATH="{R}/Python/framework-mlx-base/lib/python3.11/site-packages:{R}:." {R}/Python/cpython-3.11/bin/python3 scripts/live/studio_live_answer.py --jobs local/studio/variants/jobs.json --out local/studio/variants/answers.json --max-new 24 2>/dev/null | tail -1')
sh("rsync", "-a", "studio:drift/local/studio/variants/answers.json", str(out / "answers.json"))
for r in json.load(open(out / "answers.json"))["results"]:
    print("\nQ:", r["question"][:60], "| ref:", r["answer"])
    for k, v in r.items():
        if isinstance(v, dict): print(f"  {k:>14} [{v['answer_logprob']:7.2f}] {v['text'][:100]!r}")
