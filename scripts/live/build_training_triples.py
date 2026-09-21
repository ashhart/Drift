"""Training / validation triples for answer-level training of the GLM -> Qwen translator (preregistration.v4-answer-level.json).
train: GLM-written passages of seeds 11/12 (train part) and 6262, plus programmatic passages; val: held-back seeds 11/12 and seed 5151."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from livelib import passage_id
from questions_v3 import QUESTION
PROGRAMMATIC = {"code": "What is the access code of the {thing} at {place}?", "weight": "What weight in kilograms is given for the {thing} at {place}?", "colour": "What colour was the casing of the {thing} at {place} repainted?",
                "person": "Who looks after the {thing} at {place}?", "day": "On which day of the week is the {thing} at {place} inspected?", "room": "In which room are the spare parts for the {thing} at {place} kept?"}
taps = Path("local/live/train_taps/taps_glm")


def written(path):
    out = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line); pid = passage_id(r["text"])
        if (taps / f"{pid}.npz").exists():
            out += [{"pid": pid, "passage": r["text"], "question": QUESTION[f["kind"]].format(genre=r["genre"], thing=r["thing"]), "answer": f["value"], "kind": f["kind"]} for f in r["facts"] if f["kind"] in QUESTION]
    return out


def programmatic(path, limit):
    out, seen = [], set()
    for it in json.loads(Path(path).read_text()):
        a = it["a"]
        if a["id"] in seen or not (taps / f"{a['id']}.npz").exists():
            continue
        seen.add(a["id"])
        out += [{"pid": a["id"], "passage": a["text"], "question": q.format(**a["facts"]), "answer": a["facts"][k], "kind": k} for k, q in PROGRAMMATIC.items()]
        if len(seen) >= limit:
            break
    return out


data = {"train": written("local/live/gen_train.jsonl") + written("local/live/gen_d1_eval.jsonl") + programmatic("local/local_pair/d1/train.json", 600),
        "val": written("local/live/gen_val.jsonl") + written("local/live/gen_d1_val.jsonl")}
Path("local/live/train_taps/triples.json").write_text(json.dumps(data))
print({k: len(v) for k, v in data.items()}, "| passages:", {k: len({x["pid"] for x in v}) for k, v in data.items()})
