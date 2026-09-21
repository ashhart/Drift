"""EXPLORATORY probe: at the token where a fact's access code starts, is the OWNING thing / site linearly readable from
(a) GLM's stacked entries, (b) GLM's entries plus causal summaries, (c) Qwen's own K/V?  If (a) is poor and (c) is good, a
per-token translator cannot supply the binding the reader's keys carry, whatever its fit."""
import json, re, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "scripts/live")
from gen_multifact import THINGS, SITES
from tokenizers import Tokenizer
from drift.translate.context import causal_summaries
GL, QL = [3 + 4 * i for i in range(11)], [3 + 4 * i for i in range(12)]
glm, qwen = Tokenizer.from_file("local/tok/glm/tokenizer.json"), Tokenizer.from_file("local/tok/qwen/tokenizer.json")
records = json.loads(Path("local/live/multifact/corpus.json").read_text())["records"][:400]
P = np.load("local/live/frontier/ctx23m.npz")["P"]; mean = np.load("local/live/frontier/ctx23m.npz")["g_mean"]
things = sorted(THINGS, key=len, reverse=True)
feats = {"glm_token": [], "glm_with_summaries": [], "qwen_kv": []}; labels = {"thing": [], "site": []}; doc_of = []
for n, r in enumerate(records):
    text = r["text"]
    go, qo = glm.encode(text, add_special_tokens=False).offsets, qwen.encode(text, add_special_tokens=False).offsets
    g = np.concatenate([np.load(f"local/live/multifact/taps_glm/{r['id']}.npz")[f"l{l}"] for l in GL], 1).astype(np.float32) - mean
    z = np.load(f"local/live/multifact/taps_qwen/{r['id']}.npz")
    q = np.concatenate([np.concatenate((z[f"k{l}"].reshape(len(qo), -1), z[f"v{l}"].reshape(len(qo), -1)), 1) for l in (11, 23, 35)], 1).astype(np.float32)
    summ = causal_summaries(g @ P, (0.6, 0.9, 0.97))[0]
    for m in re.finditer(r"(?<![\d])\d{4}(?![\d])", text):
        before = text[:m.start()]
        site = max(SITES, key=lambda s: before.rfind(s)); thing = max(things, key=lambda t: before.rfind("the " + t))
        if before.rfind(site) < 0:
            continue
        end = m.end()                                                         # the LAST code token: where a reader's answer retrieval lands
        gi = max(i for i, (a, b) in enumerate(go) if b <= end); qi = max(i for i, (a, b) in enumerate(qo) if b <= end)
        feats["glm_token"].append(g[gi]); feats["glm_with_summaries"].append(np.concatenate((g[gi], summ[gi]))); feats["qwen_kv"].append(q[qi])
        labels["thing"].append(THINGS.index(thing)); labels["site"].append(SITES.index(site)); doc_of.append(n)
doc_of = np.array(doc_of); train, test = doc_of < 320, doc_of >= 320
print("code positions", len(doc_of), "train", int(train.sum()), "test", int(test.sum()))
for name, rows in feats.items():
    X = torch.from_numpy(np.stack(rows)).double(); X = (X - X[train].mean(0)) / (X[train].std(0) + 1e-6)
    out = {}
    for target, y in labels.items():
        y = torch.tensor(y); Y = torch.nn.functional.one_hot(y, int(y.max()) + 1).double()
        best = 0.0
        for lam in (1.0, 10.0, 100.0, 1000.0):                                # dual ridge: rows << features
            K = X[train] @ X[train].T
            alpha = torch.linalg.solve(K + lam * torch.eye(K.shape[0], dtype=torch.float64), Y[train] - Y[train].mean(0))
            pred = (X[test] @ X[train].T) @ alpha + Y[train].mean(0)
            best = max(best, float((pred.argmax(1) == y[test]).double().mean()))
        out[target] = round(best, 3)
    print(name, out, "chance thing", round(1 / len(THINGS), 3), "site", round(1 / len(SITES), 3))
