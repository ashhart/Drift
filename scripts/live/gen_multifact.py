"""Distractor-rich calibration / training documents: many SIMILAR facts per document, so a translator has to carry which value
belongs to which thing. Vocabulary and phrasing are DISJOINT from the long-context evaluation documents (longctx_docs.py); the
existing calibration corpus stops at 320 tokens per chunk and one fact of each kind. Corpus format of build_corpus2.py plus QA items."""
import argparse, hashlib, json, random
from pathlib import Path
from tokenizers import Tokenizer
from drift.train.alignment import shared_causal_endpoints

THINGS = "lathe,oven,winch,turbine,centrifuge,harp,anvil,chiller,scanner,plotter,incubator,boiler,crane,sewing machine,seed drill,radio mast,ice maker,pottery wheel,bandsaw,autoclave,conveyor,hoist,cider press,spectrometer,milling machine,cement mixer,dishwasher,film projector,wind gauge,bottling line".split(",")
SITES = [f"{a} {b}" for a in "Ashby,Brackenridge,Coldwater,Dunmore,Eastfield,Foxhollow,Glenmoor,Hartwell,Ivybridge,Juniper,Kingsmead,Langdale,Marlow,Northgate,Oakhurst,Pinecrest,Queensway,Redcliffe,Stonebeck,Thornbury,Underwood,Valemount,Westbrook,Yarrow,Zennor".split(",") for b in ("workshop", "yard", "hall", "station")]
FIRST = "Agnes Bruno Celeste Dorian Elspeth Fabian Greta Hamish Imogen Jasper Katya Lionel Mirela Nikolai Odette Percival Quentin Rosalba Stellan Tamsin Ugo Verity Wilfred Xanthe Yorick Zelda".split()
COLOURS = "olive maroon navy cream scarlet amber violet turquoise beige crimson indigo silver bronze charcoal ivory lilac mustard russet".split()
PHRASES = {
    "person": ["The {thing} at the {site} is looked after by {person}.", "{person} is responsible for the {thing} at the {site}.", "At the {site}, the {thing} is in the care of {person}."],
    "code": ["Its access code is {code}.", "The access code for it is {code}.", "To unlock it, enter {code}."],
    "weight": ["It weighs {weight} kilograms.", "Its weight is {weight} kilograms."],
    "colour": ["Its casing is {colour}.", "The casing was painted {colour}.", "It has a {colour} casing."]}
QUESTIONS = {"code": "What is the access code of the {thing} at the {site}? Answer with the code only.", "person": "Who looks after the {thing} at the {site}? Answer with the name only.",
             "colour": "What colour is the casing of the {thing} at the {site}? Answer with the colour only.", "weight": "How many kilograms does the {thing} at the {site} weigh? Answer with the number only."}


def document(n_facts, rng):
    combos = rng.sample([(a, b) for a in THINGS for b in SITES], n_facts)
    facts, parts = [], []
    for thing, site in combos:
        f = {"thing": thing, "site": site, "person": rng.choice(FIRST), "code": str(rng.randint(1000, 9999)), "weight": str(rng.randint(12, 980)), "colour": rng.choice(COLOURS)}
        rest = ["code", "weight", "colour"]; rng.shuffle(rest)
        parts.append(" ".join(rng.choice(PHRASES[k]).format(**f) for k in ["person"] + rest)); facts.append(f)
    return " ".join(parts), facts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=600)
    parser.add_argument("--min-facts", type=int, default=3)
    parser.add_argument("--max-facts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    glm, qwen = Tokenizer.from_file("local/tok/glm/tokenizer.json"), Tokenizer.from_file("local/tok/qwen/tokenizer.json")
    rng, records, items = random.Random(args.seed), [], []
    while len(records) < args.docs:
        text, facts = document(rng.randint(args.min_facts, args.max_facts), rng)
        eg, eq = glm.encode(text, add_special_tokens=False), qwen.encode(text, add_special_tokens=False)
        if glm.decode(eg.ids, skip_special_tokens=False) != text or qwen.decode(eq.ids, skip_special_tokens=False) != text:
            continue
        rid = hashlib.sha256(text.encode()).hexdigest()[:16]
        aligned = shared_causal_endpoints(text, list(eg.offsets), list(eq.offsets))
        records.append({"id": rid, "source": f"multifact:{args.seed}", "split": "train", "text": text, "ids_glm": eg.ids, "ids_qwen": eq.ids, "aligned": [[g, q] for g, q, _ in aligned], "facts": len(facts)})
        for f in rng.sample(facts, min(4, len(facts))):
            kind = rng.choice(list(QUESTIONS))
            items.append({"pid": rid, "kind": kind, "question": QUESTIONS[kind].format(**f), "answer": f[kind], "facts": len(facts)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"records": records}))
    args.out.with_suffix(".ids.json").write_text(json.dumps({"records": [{k: r[k] for k in ("id", "split", "ids_glm", "ids_qwen")} for r in records]}))
    args.out.with_name(args.out.stem + ".items.json").write_text(json.dumps(items))
    print(json.dumps({"docs": len(records), "items": len(items), "glm_tokens": sum(len(r["ids_glm"]) for r in records), "qwen_tokens": sum(len(r["ids_qwen"]) for r in records),
                      "aligned_rows": sum(len(r["aligned"]) for r in records), "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}))
