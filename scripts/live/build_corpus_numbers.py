"""Number-dense calibration passages (programmatic) for the tokenisation-aware translators: GLM merges digits into
multi-digit tokens, Qwen writes single digits, so mid-number positions need their own training rows."""
import argparse, hashlib, json, random, re
from pathlib import Path
from tokenizers import Tokenizer
from drift.train.alignment import shared_causal_endpoints

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--n", type=int, default=1200)
parser.add_argument("--seed", type=int, default=31)
args = parser.parse_args()
glm, qwen = Tokenizer.from_file("local/tok/glm/tokenizer.json"), Tokenizer.from_file("local/tok/qwen/tokenizer.json")
rng = random.Random(args.seed)
text = Path("scripts/live/gen_corpus.py").read_text()
V = {name: eval(re.search(rf"^{name} = (.+)$", text, re.M).group(1)) for name in ("FIRST", "LAST", "COLOURS", "PLACES", "THINGS", "ANIMALS", "FOODS", "DAYS")}
person = lambda: f"{rng.choice(V['FIRST'])} {rng.choice(V['LAST'])}"
num = lambda lo, hi: str(rng.randint(lo, hi))
digits = lambda: num(10 ** (k := rng.choice([1, 2, 2, 3, 3, 3, 4])), 10 ** (k + 1) - 1)
S = [lambda: f"The access code for the {rng.choice(V['THINGS'])} is {num(1000, 9999)}.", lambda: f"It weighs {num(3, 990)} kilograms and sits in room {num(2, 99)}.",
     lambda: f"Invoice {num(10000, 99999)} covers {num(2, 980)} crates at {num(10, 999)} pounds each.", lambda: f"{person()} is on extension {num(100, 999)} until {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}.",
     lambda: f"The reading was {digits()} at {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d} and {digits()} an hour later.", lambda: f"Plot {num(2, 99)} measures {num(10, 400)} square metres; plot {num(2, 99)} measures {num(10, 400)}.",
     lambda: f"Serial number {digits()}-{digits()} was logged on {rng.choice(V['DAYS'])} by {person()}.", lambda: f"They counted {num(12, 9800)} items, of which {num(2, 99)} were {rng.choice(V['COLOURS'])}.",
     lambda: f"The {rng.choice(V['COLOURS'])} locker opens with {num(1000, 9999)}; the {rng.choice(V['COLOURS'])} one with {num(1000, 9999)}.", lambda: f"Bus {num(1, 399)} leaves {rng.choice(V['PLACES'])} at {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}.",
     lambda: f"The batch of {rng.choice(V['FOODS'])} came to {num(3, 990)} kilograms in {num(2, 60)} boxes.", lambda: f"Ticket {digits()} admits {num(2, 12)} people to {rng.choice(V['PLACES'])}.",
     lambda: f"In {num(1900, 2030)} the {rng.choice(V['THINGS'])} cost {num(100, 99999)} and lasted {num(2, 40)} years.", lambda: f"Call {person()} on {num(100, 999)} {num(1000, 9999)} about order {digits()}."]
records, seen = [], set()
while len(records) < args.n:
    passage = " ".join(f() for f in rng.sample(S, rng.randint(5, 8)))
    eg, eq = glm.encode(passage, add_special_tokens=False), qwen.encode(passage, add_special_tokens=False)
    pid = hashlib.sha256(passage.encode()).hexdigest()[:16]
    if pid in seen or glm.decode(eg.ids) != passage or qwen.decode(eq.ids) != passage or max(len(eg.ids), len(eq.ids)) > 320:
        continue
    seen.add(pid)
    aligned = shared_causal_endpoints(passage, list(eg.offsets), list(eq.offsets))
    records.append({"id": pid, "source": f"numbers:{len(records)}", "split": "train", "text": passage, "ids_glm": eg.ids, "ids_qwen": eq.ids, "aligned": [[g, q] for g, q, _ in aligned]})
args.out.write_text(json.dumps({"records": records}))
args.out.with_suffix(".ids.json").write_text(json.dumps({"records": [{k: r[k] for k in ("id", "split", "ids_glm", "ids_qwen")} for r in records]}))
print(json.dumps({"chunks": len(records), "glm_tokens": sum(len(r["ids_glm"]) for r in records), "qwen_tokens": sum(len(r["ids_qwen"]) for r in records),
                  "aligned": sum(len(r["aligned"]) for r in records), "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()[:16]}))
