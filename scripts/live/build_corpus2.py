"""Mixed calibration corpus for the live translators: open documentation prose found on this host
(vim help, perl pod), programmatic fact passages (names, digits, colours, times) and GLM-written
passages (gen_corpus.py). Tokenized for both members with causal-endpoint alignment (D8)."""
import argparse, glob, hashlib, json, random, re
from pathlib import Path
from tokenizers import Tokenizer
from drift.train.alignment import shared_causal_endpoints

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--generated", type=Path)
parser.add_argument("--docs", type=int, default=800)
parser.add_argument("--templated", type=int, default=700)
args = parser.parse_args()
glm, qwen = Tokenizer.from_file("local/tok/glm/tokenizer.json"), Tokenizer.from_file("local/tok/qwen/tokenizer.json")
rng = random.Random(23)
text = Path("scripts/live/gen_corpus.py").read_text()
lists = {name: re.search(rf'^{name} = (.+)$', text, re.M).group(1) for name in ("FIRST", "LAST", "COLOURS", "PLACES", "THINGS", "ANIMALS", "FOODS", "DAYS")}
V = {k: eval(v) for k, v in lists.items()}

# ---- 1) documentation prose
paras = []
for path in sorted(glob.glob("/usr/share/vim/vim*/doc/*.txt")) + sorted(glob.glob("/System/Library/Perl/**/*.pod", recursive=True)) + sorted(glob.glob("/usr/share/perl5/**/*.pod", recursive=True)):
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="strict")
    except Exception:
        continue
    raw = re.sub(r"[A-Z]<+\s?(.*?)\s?>+", r"\1", raw)                     # pod inline markup
    for para in re.split(r"\n\s*\n", raw):
        para = " ".join(para.split())
        if 400 <= len(para) <= 1100 and para.isascii() and para.count(" ") > 60 and not para.startswith(("=", "*", "|")):
            paras.append((f"doc:{Path(path).name}", para))
rng.shuffle(paras)
paras = paras[: args.docs * 2]

# ---- 2) programmatic fact passages
person = lambda: f"{rng.choice(V['FIRST'])} {rng.choice(V['LAST'])}"
clock = lambda: f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"
S = [lambda: f"The {rng.choice(V['THINGS'])} for {rng.choice(V['PLACES'])} arrives on {rng.choice(V['DAYS'])} at {clock()}.",
     lambda: f"The delivery code is {rng.randint(1000, 9999)} and it must be signed for by {person()}.",
     lambda: f"{person()} repainted the door in {rng.choice(V['COLOURS'])} rather than {rng.choice(V['COLOURS'])}.",
     lambda: f"Plot {rng.randint(2, 99)} was reassigned from {person()} to {person()}.",
     lambda: f"A {rng.choice(V['ANIMALS'])} had chewed through the {rng.choice(V['COLOURS'])} cable, so the {rng.choice(V['THINGS'])} stopped at {clock()}.",
     lambda: f"The secret ingredient is {rng.choice(V['FOODS'])}, and the batch weighed {rng.randint(3, 480)} kilograms.",
     lambda: f"If anything goes wrong, call {person()} on extension {rng.randint(100, 999)}.",
     lambda: f"There were {rng.randint(12, 980)} items on the pallet when {person()} counted them on {rng.choice(V['DAYS'])}.",
     lambda: f"Room {rng.randint(2, 99)} at {rng.choice(V['PLACES'])} is reserved for {person()} until {clock()}.",
     lambda: f"{person()} found the missing {rng.choice(V['THINGS'])} about {rng.randint(2, 40)} kilometres north of {rng.choice(V['PLACES'])}.",
     lambda: f"Last year's winner was {person()}, whose entry weighed {rng.randint(3, 480)} kilograms.",
     lambda: f"The {rng.choice(V['COLOURS'])} {rng.choice(V['THINGS'])} belongs to {person()}; the {rng.choice(V['COLOURS'])} one is on loan.",
     lambda: f"Invoice {rng.randint(10000, 99999)} covers {rng.randint(2, 60)} crates of {rng.choice(V['FOODS'])}.",
     lambda: f"The password for the {rng.choice(V['THINGS'])} is {rng.choice(V['ANIMALS'])}-{rng.randint(10, 99)}-{rng.choice(V['COLOURS'])}.",
     lambda: f"Nobody saw the {rng.choice(V['ANIMALS'])} again after {rng.choice(V['DAYS'])}, although {person()} left out some {rng.choice(V['FOODS'])}."]
templated = [(f"templated:{i}", " ".join(f() for f in rng.sample(S, rng.randint(5, 8)))) for i in range(args.templated)]

# ---- 3) GLM-written passages
generated, extra = [], {}
if args.generated:
    for line in args.generated.read_text().splitlines():
        r = json.loads(line)
        generated.append((f"generated:{r['seed']}:{r['n']}", r["text"]))
        extra[r["text"]] = {k: r[k] for k in ("genre", "place", "thing", "facts")}

records, skipped, docs = [], 0, 0
for source, passage in generated + templated + paras:
    if source.startswith("doc:") and docs >= args.docs:
        continue
    eg, eq = glm.encode(passage, add_special_tokens=False), qwen.encode(passage, add_special_tokens=False)
    if glm.decode(eg.ids, skip_special_tokens=False) != passage or qwen.decode(eq.ids, skip_special_tokens=False) != passage:
        skipped += 1; continue
    aligned = shared_causal_endpoints(passage, list(eg.offsets), list(eq.offsets))
    if len(aligned) < 20 or max(len(eg.ids), len(eq.ids)) > 320:
        skipped += 1; continue
    docs += source.startswith("doc:")
    records.append({"id": hashlib.sha256(passage.encode()).hexdigest()[:16], "source": source, "split": "train", "text": passage,
                    "ids_glm": eg.ids, "ids_qwen": eq.ids, "aligned": [[g, q] for g, q, _ in aligned], **extra.get(passage, {})})
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps({"records": records}))
args.out.with_suffix(".ids.json").write_text(json.dumps({"records": [{k: r[k] for k in ("id", "split", "ids_glm", "ids_qwen")} for r in records]}))
kinds = {k: sum(r["source"].startswith(k) for r in records) for k in ("generated", "templated", "doc")}
print(json.dumps({"chunks": len(records), "kinds": kinds, "skipped": skipped, "glm_tokens": sum(len(r["ids_glm"]) for r in records),
                  "qwen_tokens": sum(len(r["ids_qwen"]) for r in records), "aligned_rows": sum(len(r["aligned"]) for r in records),
                  "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}))
