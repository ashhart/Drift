"""Split-knowledge items exactly as D1 built them (scripts/live/d1_eval.py); `question` is D1's wording, `question_protocol` names the two sources the way the provenance training prompt does."""
import json, random, re
from pathlib import Path
from tokenizers import Tokenizer
from livelib import passage_id

NAMES_F = "Beatrix Cormac Thandiwe Anselm Rosalind Ezekiel Marisol Ulrich Saoirse Lorenzo Ottoline Bartholomew Yasmin Cedric Philippa Desmond Ximena Leopold Winifred Magnus".split()
NAMES_L = "Abernathy Villanueva Oduya Strickland Montague Lefebvre Underhill Calloway Drummond Esposito Fairweather Hargreaves Ingleby Juarez Kettering Lockwood Mwangi Nightingale Pemberton Rutherford".split()


def correct(text: str, item: dict) -> bool:
    has = lambda v: re.search(rf"(?<![\w]){re.escape(v.lower())}(?![\w])", text.lower().replace(",", "")) is not None
    return all(has(v) for v in item["must"]) and not any(has(v) for v in item["must_not"])


def build(passages: Path, per_type: int, seed: int = 99) -> tuple[list[dict], dict]:
    glm_tok, qwen_tok = Tokenizer.from_file("local/tok/glm/tokenizer.json"), Tokenizer.from_file("local/tok/qwen/tokenizer.json")
    usable = []
    for line in passages.read_text().splitlines():
        r = json.loads(line)
        facts = {f["kind"]: f["value"] for f in r["facts"]}
        if all(k in facts for k in ("code", "person", "weight", "colour")) and all(t.decode(t.encode(r["text"], add_special_tokens=False).ids) == r["text"] for t in (glm_tok, qwen_tok)):
            usable.append({**r, "facts": facts, "id": passage_id(r["text"])})
    rng, items, queue, counts = random.Random(seed), [], list(usable), {"conj": 0, "sum": 0, "hop": 0}
    while queue and any(v < per_type for v in counts.values()):
        kind = min((k for k in counts if counts[k] < per_type), key=lambda k: (counts[k], ["conj", "sum", "hop"].index(k)))
        a = queue.pop(0)
        if kind == "hop":
            others = set()
            while len(others) < 4:
                name = f"{rng.choice(NAMES_F)} {rng.choice(NAMES_L)}"
                if name != a["facts"]["person"]:
                    others.add(name)
            people = sorted(others | {a["facts"]["person"]}, key=lambda _: rng.random())
            ext = dict(zip(people, rng.sample(range(200, 990), 5)))
            note = "Staff directory. " + " ".join(f"{p} is on extension {e}." for p, e in ext.items())
            items.append({"kind": kind, "a": a, "note": note, "question": f"What is the phone extension of the person responsible for the {a['thing']}? That person is named in your shared memory; the directory is in your own note.",
                          "question_protocol": f"What is the phone extension of the person responsible for the {a['thing']}? That person is named in your partner's document; the directory is your own note.",
                          "must": [str(ext[a["facts"]["person"]])], "must_not": [str(e) for p, e in ext.items() if p != a["facts"]["person"]], "reference": f"{a['facts']['person']} -> {ext[a['facts']['person']]}"})
        else:
            j = next((n for n, b in enumerate(queue) if b["thing"] != a["thing"] and b["facts"]["colour"] != a["facts"]["colour"] and b["facts"]["code"] != a["facts"]["code"]), None)
            if j is None:
                break
            b = queue.pop(j)
            if kind == "conj":
                items.append({"kind": kind, "a": a, "note": b["text"], "question": f"What is the code mentioned in the {a['genre']} about the {a['thing']} (it is in your shared memory), and which colour is mentioned in your own note?",
                              "question_protocol": f"What is the code mentioned in your partner's document (the {a['genre']} about the {a['thing']}), and which colour is mentioned in your own note?",
                              "must": [a["facts"]["code"], b["facts"]["colour"]], "must_not": [b["facts"]["code"], a["facts"]["colour"]], "reference": f"{a['facts']['code']} and {b['facts']['colour']}"})
            else:
                total = int(a["facts"]["weight"]) + int(b["facts"]["weight"])
                items.append({"kind": kind, "a": a, "note": b["text"], "question": f"How many kilograms do the {a['thing']} (described in your shared memory) and the {b['thing']} (described in your own note) weigh together? Work it out and give the total.",
                              "question_protocol": f"How many kilograms do the {a['thing']} (partner's document) and the {b['thing']} (your own note) weigh together? Give the total.",
                              "must": [str(total)], "must_not": [], "reference": f"{a['facts']['weight']} + {b['facts']['weight']} = {total}"})
        counts[kind] += 1
    return items, counts
