"""Export GLM's MLA latents for each passage's own tokens, read in the gate's framing. Runs on the head Spark via spark_run.sh.
Each npz also holds the tokens' character offsets and whether they map exactly onto the text (ASCII byte-level tokens)."""
import json, os, subprocess, sys, time, urllib.request, uuid
from pathlib import Path
import numpy as np
from glm53_handoff import read_latents
KEY, ROOT = os.environ["DRIFT_GLM_KEY"], Path("/dev/shm/glm53-handoff")
LINK = ("You are linked to another AI model through a shared memory. A document it has read is in your memory, not in this chat; "
        "you were never shown it as text. @@DRIFT@@ Answer the user's question from what you recall from that shared memory. "
        "Give the answer directly.")
items, out, worker = json.load(open(sys.argv[1])), Path(sys.argv[2]), sys.argv[3]
out.mkdir(exist_ok=True)


def post(path, body):
    req = urllib.request.Request("http://127.0.0.1:8888" + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


tok = post("/tokenize", {"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "system", "content": LINK}, {"role": "user", "content": "?"}], "add_generation_prompt": True, "return_token_strs": True, "chat_template_kwargs": {"enable_thinking": False}})
text, spans = "", []
for piece in tok["token_strs"]:
    spans.append((len(text), len(text) + len(piece))); text += piece
a = text.index("@@DRIFT@@")
head = tok["tokens"][:next(i for i, (x, y) in enumerate(spans) if y > a)]
for item in {i["passage"]: i for i in items}.values():
    pid = item["id"].split("-")[0]
    tokenized = post("/tokenize", {"model": "GLM-5.3-Flash-EXL3", "prompt": item["passage"], "add_special_tokens": False, "return_token_strs": True})
    ids, offsets, at = tokenized["tokens"], [], 0
    for piece in tokenized["token_strs"]:
        offsets.append((at, at + len(piece))); at += len(piece)
    name = f"fx-{uuid.uuid4().hex[:10]}"
    post("/v1/completions", {"model": "GLM-5.3-Flash-EXL3", "prompt": head + ids, "max_tokens": 1, "temperature": 0, "cache_salt": f"drift:{uuid.uuid4().hex}", "kv_transfer_params": {"glm53_handoff": True, "handoff_id": name}})
    t = time.time()
    while not (ROOT / name / "rank0.ready").exists() and time.time() - t < 60:
        time.sleep(0.05)
    ready = json.loads((ROOT / name / "rank0.ready").read_text())
    if ready.get("mode") != "arena":
        blob = (ROOT / name / "rank0.bin").read_bytes()
    else:                                                               # only the leased range of the daemon's arena
        with open(ROOT / "arena-local0.bin", "rb") as arena:
            arena.seek(int(ready["offset"])); blob = arena.read(int(ready["length"]))
    latents = read_latents(blob)
    np.savez(out / f"{pid}.npz", offsets=np.asarray(offsets, np.int32), aligned=np.bool_(at == len(item["passage"]) and item["passage"].isascii()),
             **{f"l{l}": v[len(head):len(head) + len(ids)].astype(np.float16) for l, v in latents.items()})
    subprocess.run(["rm", "-rf", str(ROOT / name)], check=True)
    subprocess.run(["ssh", "-o", "BatchMode=yes", worker, f"rm -rf {ROOT / name}"], check=True)
print(json.dumps({"passages": len(list(out.glob("*.npz")))}))
