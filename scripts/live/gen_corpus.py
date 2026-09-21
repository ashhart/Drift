"""Have the live GLM server write short factual passages from randomised fact sheets (Spark host,
stdlib only). Each record keeps its fact sheet, so held-out records double as QA items with known
answers. Deterministic fact sheets per --seed; the prose is sampled."""
import argparse, json, os, random, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--heldout-vocab", action="store_true", help="names and places disjoint from the training lists")
parser.add_argument("--require", default="", help="comma-separated fact kinds every sheet must contain")
parser.add_argument("--url", default="http://127.0.0.1:8888/v1/chat/completions")
parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
args = parser.parse_args()
key = os.environ["DRIFT_GLM_KEY"]
FIRST = "Marta Ingrid Tomas Aisha Kenji Lucia Pavel Noor Emeka Sofia Callum Priya Mateo Hana Dmitri Amara Jonas Leila Rafael Yuki Oskar Fatima Bruno Mei Tariq Elena Kwame Freya Arjun Zofia Hugo Nia Stefan Carmen Idris Greta Luca Salma Viktor Ruth Andre Keiko Felix Deepa Omar Astrid Nikolai Wanjiru Pablo Signe".split()
LAST = "Okafor Petrenko Hale Lindqvist Moreau Tanaka Castillo Novak Adeyemi Brennan Kowalski Haddad Fischer Oyelaran Santos Virtanen Dubois Nakamura Ferrari Andersson Mbeki Rahman Sokolov Jensen Alvarez Kim Papadopoulos Gallagher Ivanova Chaudhry Bauer Nwosu Romano Holm Vasquez Yilmaz Marsh Kaplan Thorne Eze Laurent Bjork Medina Pretorius Quinn Sato Varga Whitlock Zamora Osei".split()
COLOURS = "red blue green yellow orange purple black white grey brown pink turquoise crimson navy olive maroon silver gold teal violet".split()
PLACES = "Harbour Street depot,Lyngen field hut,Riverside allotments,Castlefield library,the north quarry,Pier 9 warehouse,Saint Anne's clinic,the Alder Lane bakery,Tamar ferry terminal,the old mill studio,Kestrel Ridge observatory,the Dockside market,Hollow Oak primary school,the Verne Road garage,Blackwater lighthouse,the Moss Lane greenhouse,Juniper Court hotel,the Eastgate tram shed,Ferncliff museum,the Copperfield brewery".split(",")
THINGS = "compressor,weather buoy,projector,espresso machine,forklift,telescope mirror,server rack,kiln,defibrillator,piano,beehive,water pump,printing press,drone,generator,freezer,microscope,loom,sailboat mast,tractor".split(",")
ANIMALS = "seal fox heron badger otter crow goat owl squirrel stoat magpie hedgehog gull pike hare".split()
FOODS = "green apple,smoked paprika,roasted chestnut,pickled ginger,brown butter,dried apricot,black garlic,toasted sesame,fresh dill,candied lemon,juniper berries,miso paste,saffron,fennel seed,sour cherry".split(",")
GENRES = ["shipping note", "field report", "recipe card", "committee minutes", "maintenance log", "school newsletter item", "incident report", "hotel handover note", "museum label", "lab notebook entry",
          "ship's log entry", "club announcement", "inspection summary", "letter to a neighbour", "radio traffic bulletin", "theatre call sheet", "farm diary entry", "library notice", "race report", "expedition journal entry"]
if args.heldout_vocab:
    FIRST = "Beatrix Cormac Thandiwe Anselm Rosalind Ezekiel Marisol Ulrich Saoirse Lorenzo Ottoline Bartholomew Yasmin Cedric Philippa Desmond Ximena Leopold Winifred Magnus".split()
    LAST = "Abernathy Villanueva Oduya Strickland Montague Lefebvre Underhill Calloway Drummond Esposito Fairweather Hargreaves Ingleby Juarez Kettering Lockwood Mwangi Nightingale Pemberton Rutherford".split()
    PLACES = "the Thistle Wharf boatyard,Saltmarsh signal station,the Pennywell cider press,Gorse Hill cricket pavilion,the Larkspur Road laundry,Whinfell radio mast,the Quayside aquarium,Marrow Lane pottery".split(",")
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()


def sheet(rng):
    person = lambda: f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    pool = [("code", "the code", str(rng.randint(1000, 9999))), ("person", "the person responsible", person()), ("colour", "the colour", rng.choice(COLOURS)),
            ("time", "the time", f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"), ("day", "the day", rng.choice(DAYS)), ("count", "the number of items", str(rng.randint(12, 980))),
            ("animal", "the animal involved", rng.choice(ANIMALS)), ("ingredient", "the special ingredient", rng.choice(FOODS)), ("room", "the room or plot number", str(rng.randint(2, 99))),
            ("second_person", "the other person mentioned", person()), ("weight", "the weight in kilograms", str(rng.randint(3, 480)))]
    need = [k for k in args.require.split(",") if k]
    facts = [f for f in pool if f[0] in need] + rng.sample([f for f in pool if f[0] not in need], max(0, rng.randint(4, 6) - len(need)))
    rng.shuffle(facts)
    return {"genre": rng.choice(GENRES), "place": rng.choice(PLACES), "thing": rng.choice(THINGS), "facts": [{"kind": k, "label": l, "value": v} for k, l, v in facts]}


def write(i):
    rng = random.Random(args.seed * 1_000_003 + i)
    s = sheet(rng)
    lines = "\n".join(f"- {f['label']}: {f['value']}" for f in s["facts"])
    prompt = (f"Write a {s['genre']} of 70 to 120 words set at {s['place']}, concerning a {s['thing']}. Work every one of these facts into the prose exactly as written "
              f"(same spelling, same digits), each stated once in a natural sentence:\n{lines}\nPlain prose only: no title, no list, no markdown, no preamble.")
    body = json.dumps({"model": args.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 260, "temperature": 0.9, "top_p": 0.95,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                text = json.loads(resp.read())["choices"][0]["message"]["content"].strip()
            break
        except Exception:
            time.sleep(2 + 3 * attempt)
    else:
        return None
    s["facts"] = [f for f in s["facts"] if f["value"] in text]           # keep only facts that really made it into the prose
    return {"n": i, "seed": args.seed, "text": text, **s}


lock, done, started = threading.Lock(), 0, time.time()
args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("a") as sink, ThreadPoolExecutor(args.workers) as ex:
    for rec in ex.map(write, range(args.n)):
        if rec and len(rec["text"]) > 200:
            sink.write(json.dumps(rec) + "\n"); sink.flush(); done += 1
print(json.dumps({"written": done, "seconds": round(time.time() - started, 1)}))
