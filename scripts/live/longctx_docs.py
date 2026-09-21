"""Long synthetic documents for the long-context evaluation (roadmap D3): unique thing/place facts, needle questions spread
over the whole document. Lengths are measured in READER tokens. Deterministic per (length, seed)."""
import argparse, json, random
from pathlib import Path
from tokenizers import Tokenizer

THINGS = "compressor,weather buoy,projector,espresso machine,forklift,telescope mirror,server rack,kiln,defibrillator,piano,beehive,water pump,printing press,drone,generator,freezer,microscope,loom,sailboat mast,tractor".split(",")
PLACES = [f"{a} {b}" for a in "Harbour,Lyngen,Riverside,Castlefield,Quarry,Pier,Clinic,Bakery,Ferry,Mill,Thistle,Saltmarsh,Pennywell,Gorse,Larkspur,Whinfell,Alder,Tamar,Kestrel,Dockside".split(",") for b in ("depot", "annex")]
NAMES = "Marta Ingrid Tomas Aisha Kenji Lucia Pavel Noor Emeka Sofia Callum Priya Mateo Hana Dmitri Amara Jonas Leila Rafael Yuki".split()
COLOURS = "red blue green yellow orange purple black white grey brown pink teal".split()
QUESTIONS = {"code": "What is the access code of the {thing} at the {place}? Answer with the code only.",
             "person": "Who looks after the {thing} at the {place}? Answer with the name only.",
             "colour": "What colour is the casing of the {thing} at the {place}? Answer with the colour only."}


def document(target_tokens, rng, tok):
    combos = [(a, b) for a in THINGS for b in PLACES]; rng.shuffle(combos)
    facts, parts, count = [], [], 0
    for thing, place in combos:
        f = {"thing": thing, "place": place, "code": str(rng.randint(1000, 9999)), "weight": str(rng.randint(12, 980)), "colour": rng.choice(COLOURS), "person": rng.choice(NAMES)}
        text = f"The {thing} at the {place} is looked after by {f['person']}. Its access code is {f['code']}. It weighs {f['weight']} kilograms and its casing is {f['colour']}. "
        parts.append(text); facts.append(f); count += len(tok.encode(text, add_special_tokens=False).ids)
        if count >= target_tokens:
            break
    return "".join(parts), facts


def writer_of_reader(text, writer_tok, reader_tok):
    """For every reader token, the writer token whose character span contains the reader token's last character."""
    w_end = [e for _, e in writer_tok.encode(text, add_special_tokens=False).offsets]
    out, t = [], 0
    for _, end in reader_tok.encode(text, add_special_tokens=False).offsets:
        while t < len(w_end) - 1 and w_end[t] < end:
            t += 1
        out.append(t)
    return out


def natural_documents(paths, sizes, per_size, questions, seed, tok, glm):
    """Long documents made of GLM-WRITTEN passages (free prose, several genres), one (genre, thing) pair at most once per document."""
    from questions_v3 import PRIORITY, QUESTION
    rng, pool, seen = random.Random(f"natural-{seed}"), [], set()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if r["text"] not in seen and r.get("facts") and all(t.decode(t.encode(r["text"], add_special_tokens=False).ids) == r["text"] for t in (tok, glm)):
                seen.add(r["text"]); pool.append(r)
    docs = []
    for size in sizes:
        for k in range(per_size):
            rng.shuffle(pool)
            chosen, combos = [], set()
            for r in pool:
                if (r["genre"], r["thing"]) not in combos and len(chosen) < size:
                    combos.add((r["genre"], r["thing"])); chosen.append(r)
            text = "\n\n".join(r["text"] for r in chosen)
            asked = [chosen[int(round(i * (len(chosen) - 1) / max(1, questions - 1)))] for i in range(questions)]
            qs = []
            for i, r in enumerate(asked):
                facts = {f["kind"]: f["value"] for f in r["facts"]}
                kinds = [x for x in PRIORITY if x in facts]
                kind = kinds[i % len(kinds)]
                qs.append({"kind": kind, "answer": facts[kind], "question": QUESTION[kind].format(genre=r["genre"], thing=r["thing"]) + " Answer with the value only.", "depth": round(chosen.index(r) / max(1, len(chosen) - 1), 3)})
            docs.append({"id": f"nat{seed}-{size}-{k}", "target": size, "text": text, "facts": len(chosen), "questions": qs, "writer_of_reader": writer_of_reader(text, glm, tok)})
    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[4000, 8000, 16000])
    parser.add_argument("--per-length", type=int, default=2)
    parser.add_argument("--questions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--passages", type=Path, nargs="*", help="jsonl files of GLM-written passages: build natural long documents instead of template ones")
    parser.add_argument("--sizes", type=int, nargs="*", default=[4, 8, 16, 32], help="passages per natural document")
    args = parser.parse_args()
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    tok, glm = Tokenizer.from_file("local/tok/qwen/tokenizer.json"), Tokenizer.from_file("local/tok/glm/tokenizer.json")
    docs = natural_documents(args.passages, args.sizes, args.per_length, args.questions, args.seed, tok, glm) if args.passages else []
    for length in ([] if args.passages else args.lengths):
        for k in range(args.per_length):
            rng = random.Random(f"{args.seed}-{length}-{k}")
            text, facts = document(length, rng, tok)
            kinds = list(QUESTIONS)
            picks = [facts[int(round(i * (len(facts) - 1) / (args.questions - 1)))] for i in range(args.questions)]
            qs = [{"kind": kinds[i % 3], "answer": f[kinds[i % 3]], "question": QUESTIONS[kinds[i % 3]].format(**f), "depth": round(facts.index(f) / max(1, len(facts) - 1), 3)} for i, f in enumerate(picks)]
            docs.append({"id": f"long{args.seed}-{length}-{k}", "target": length, "text": text, "facts": len(facts), "questions": qs, "writer_of_reader": writer_of_reader(text, glm, tok)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"docs": docs}, indent=1) + "\n")
    print(json.dumps({"docs": len(docs), "questions": sum(len(d["questions"]) for d in docs)}))
