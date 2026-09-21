"""Programmatic split-knowledge data for the local pair: passages with known facts, and D1-style items (conj / sum / hop)
built from pairs of them. Unlimited, so answer-level training is not starved; vocabulary for the test split is held out."""
import argparse, hashlib, json, random
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=Path("local/local_pair/d1"))
parser.add_argument("--train", type=int, default=900)
parser.add_argument("--val", type=int, default=150)
parser.add_argument("--test", type=int, default=240)
args = parser.parse_args()
FIRST = "Marta Ingrid Tomas Aisha Kenji Lucia Pavel Noor Emeka Sofia Callum Priya Mateo Hana Dmitri Amara Jonas Leila Rafael Yuki Oskar Fatima Bruno Mei Tariq Elena Kwame Freya Arjun Zofia".split()
LAST = "Okafor Petrenko Hale Lindqvist Moreau Tanaka Castillo Novak Adeyemi Brennan Kowalski Haddad Fischer Santos Virtanen Dubois Nakamura Ferrari Andersson Mbeki Rahman Sokolov Jensen Alvarez".split()
FIRST_T = "Beatrix Cormac Thandiwe Anselm Rosalind Ezekiel Marisol Ulrich Saoirse Lorenzo Ottoline Yasmin Cedric Philippa Desmond Ximena Leopold Winifred Magnus".split()
LAST_T = "Abernathy Villanueva Oduya Strickland Montague Lefebvre Underhill Calloway Drummond Esposito Fairweather Hargreaves Ingleby Juarez Kettering Lockwood Mwangi Nightingale Pemberton".split()
COLOURS = "red blue green yellow orange purple black white grey brown pink turquoise crimson navy olive maroon silver gold teal violet".split()
THINGS = "compressor,weather buoy,projector,espresso machine,forklift,telescope mirror,server rack,kiln,defibrillator,piano,beehive,water pump,printing press,drone,generator,freezer,microscope,loom,sailboat mast,tractor".split(",")
PLACES = "the Harbour Street depot,the Lyngen field hut,the Riverside allotments,Castlefield library,the north quarry,Pier 9 warehouse,Saint Anne's clinic,the Alder Lane bakery,the Tamar ferry terminal,the old mill studio".split(",")
PLACES_T = "the Thistle Wharf boatyard,Saltmarsh signal station,the Pennywell cider press,Gorse Hill cricket pavilion,the Larkspur Road laundry,Whinfell radio mast".split(",")
SENT = [lambda f: f"The {f['thing']} at {f['place']} is looked after by {f['person']}.", lambda f: f"Its access code is {f['code']}.", lambda f: f"It weighs {f['weight']} kilograms.",
        lambda f: f"The casing was repainted {f['colour']} last spring.", lambda f: f"Inspections happen every {f['day']}.", lambda f: f"Spare parts are kept in room {f['room']}."]
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()


def passage(rng, test):
    f = {"thing": rng.choice(THINGS), "place": rng.choice(PLACES_T if test else PLACES), "person": f"{rng.choice(FIRST_T if test else FIRST)} {rng.choice(LAST_T if test else LAST)}",
         "code": str(rng.randint(1000, 9999)), "weight": str(rng.randint(12, 980)), "colour": rng.choice(COLOURS), "day": rng.choice(DAYS), "room": str(rng.randint(2, 99))}
    order = SENT[:1] + rng.sample(SENT[1:], len(SENT) - 1)
    text = " ".join(s(f) for s in order)
    return {"id": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text, "facts": f}


def items(rng, n, test):
    out = []
    while len(out) < n:
        kind = ("conj", "sum", "hop")[len(out) % 3]
        a = passage(rng, test)
        if kind == "hop":
            names = {a["facts"]["person"]}
            while len(names) < 5:
                names.add(f"{rng.choice(FIRST_T if test else FIRST)} {rng.choice(LAST_T if test else LAST)}")
            people = sorted(names, key=lambda _: rng.random()); ext = dict(zip(people, rng.sample(range(200, 990), 5)))
            note = "Staff directory. " + " ".join(f"{p} is on extension {e}." for p, e in ext.items())
            out.append({"kind": kind, "a": a, "note": note, "question": f"What is the phone extension of the person who looks after the {a['facts']['thing']}? That person is named in your partner's document; the directory is your own note.",
                        "must": [str(ext[a["facts"]["person"]])], "must_not": [str(e) for p, e in ext.items() if p != a["facts"]["person"]]})
            continue
        b = passage(rng, test)
        if b["facts"]["thing"] == a["facts"]["thing"] or b["facts"]["colour"] == a["facts"]["colour"] or b["facts"]["code"] == a["facts"]["code"]:
            continue
        if kind == "conj":
            out.append({"kind": kind, "a": a, "note": b["text"], "question": f"What is the access code of the {a['facts']['thing']} in your partner's document, and what colour is the casing of the {b['facts']['thing']} in your own note?",
                        "must": [a["facts"]["code"], b["facts"]["colour"]], "must_not": [b["facts"]["code"], a["facts"]["colour"]]})
        else:
            total = int(a["facts"]["weight"]) + int(b["facts"]["weight"])
            out.append({"kind": kind, "a": a, "note": b["text"], "question": f"How many kilograms do the {a['facts']['thing']} (partner's document) and the {b['facts']['thing']} (your own note) weigh together? Give the total.",
                        "must": [str(total)], "must_not": []})
    return out


args.out.mkdir(parents=True, exist_ok=True)
texts = []
for split, n, seed in (("train", args.train, 1), ("val", args.val, 2), ("test", args.test, 3)):
    data = items(random.Random(seed), n, test=(split == "test"))
    (args.out / f"{split}.json").write_text(json.dumps(data))
    texts += [{"text": it["a"]["text"], "split": f"d1_{split}"} for it in data]
(args.out / "texts.jsonl").write_text("".join(json.dumps(t) + "\n" for t in texts))
print({"passages_to_tap": len(texts)}, json.dumps(json.loads((args.out / "val.json").read_text())[0])[:400])
