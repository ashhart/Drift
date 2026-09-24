"""Appointment scenarios: the streets and shops GLM lists, where the true street sits, and the fact only Qwen is told.

The dev vocabulary and an unbalanced draw reproduce every earlier run for its seed. --balance spreads the true street's
place in GLM's list evenly over the four slots, in a shuffled order, so a fixed positional guess scores at chance. The
fresh vocabulary uses names that no earlier run or prompt used, and confirm2 names that no run had used before the causal
confirmation; development runs must not use it. The train vocabulary is for fitting translators only, larger so that
names repeat less; no evaluation may use it.
"""
from __future__ import annotations
import random

VOCABULARY = {
    "dev": (["Alder Street", "Quarry Road", "Tamar Lane", "Whinfell Avenue", "Saltmarsh Row", "Pennywell Close"],
            ["Kestrel Coffee", "Bean Harbour", "Larkspur Roasters", "Thistle and Crumb", "Gorse Street Grind", "The Copper Kettle"],
            ["Riverside clinic", "Castlefield surgery", "Harbour health centre", "Millbrook practice"]),
    "fresh": (["Brackenfold Road", "Kittiwake Lane", "Ormsby Terrace", "Halewood Crescent", "Farrant Street", "Juniper Wharf"],
              ["Moth and Kettle", "Linnet Cup", "Driftwood Roastery", "Heron Espresso", "Quayside Beans", "The Tin Mug"],
              ["Wrenfield practice", "Ashcombe health centre", "Larchmont clinic", "Seaton surgery"]),
    "confirm2": (["Fennimore Road", "Tarragon Lane", "Galloway Terrace", "Wickham Rise", "Sorrelwood Crescent", "Kestle Street"],
                 ["Bramble and Bean", "Pellow Roasters", "Otterburn Coffee", "Little Ember", "Cobalt Cup", "The Grind Room"],
                 ["Wendover medical centre", "Elmsleigh surgery", "Carrow health centre", "Pinecroft practice"]),
    "train": (["Ashgrove Road", "Bellwether Lane", "Caldwell Street", "Dunmore Avenue", "Eastbrook Close", "Foxglove Way", "Greystone Row",
               "Hartwell Drive", "Ivybridge Road", "Jasmine Terrace", "Kingfisher Walk", "Lowther Street", "Marchbank Road", "Netherby Lane",
               "Oakhurst Avenue", "Pembury Close", "Queensway Parade", "Rowan Hill", "Sedgemoor Street", "Thornbury Road", "Upland Crescent",
               "Vicarage Mews", "Westcombe Lane", "Yarrow Street", "Ambleside Road", "Birchfield Way", "Cromer Terrace", "Dovecote Lane",
               "Elmstead Road", "Fairholme Street"],
              ["Amber Mill Coffee", "Blackbird Brews", "Cinder Cup", "Dapple Roasters", "Ember and Oak", "Fernleaf Espresso", "Grindstone Cafe",
               "Hazel Bean", "Inkwell Coffee", "Kiln Roasters", "Lantern Coffee House", "Maple Drip", "Nutmeg and Crumb", "Orchard Espresso",
               "Pebble Roast", "Quill Coffee", "Rook and Bean", "Saffron Cup", "Tidewater Coffee", "Umber Beans", "Velvet Crema",
               "Willow Brew", "Yellowhammer Coffee", "Zest Espresso", "Acorn Grind", "Dunlin Coffee", "Elder Cup", "Fable Roasters",
               "Gannet Espresso", "Heath and Hearth"],
              ["Northgate surgery", "Brookside practice", "Hillcrest health centre", "Meadowbank clinic", "Stonebridge surgery",
               "Parkway medical centre", "Linden practice", "Weirside clinic"]),
}
HOURS = ["2 pm", "3 pm", "4 pm", "11 am"]
LINK = ("You are linked to another AI model through a shared memory that fills while you work. @@DRIFT@@ What your partner knows "
        "and writes arrives in that memory, never in this chat. Use it as your own recollection.")
FOLLOWUP = ("My partner has now chosen a coffee shop near my appointment and wrote its recommendation down. Which coffee shop did my "
            "partner recommend? Answer with the shop's name only, or 'unknown' if you truly do not recall one.")


def load_vocabulary(path) -> tuple[list[str], list[str], list[str]]:
    """An owner-written names file: {"streets": [...], "shops": [...], "clinics": [...]}, checked so scoring stays sound."""
    import json
    data = json.loads(open(path).read())
    streets, shops, clinics = (list(data[key]) for key in ("streets", "shops", "clinics"))
    names = streets + shops + clinics
    if len(set(streets)) < 4 or len(set(shops)) < 4 or not clinics or len(set(names)) != len(names):
        raise ValueError("need at least 4 distinct streets and shops, a clinic, and no repeated name")
    if any(street.lower() in other.lower() for street in streets for other in names if other != street):
        raise ValueError("a street name appears inside another name, which would break the street score")
    if any(shop.lower() in other.lower() for shop in shops for other in names if other != shop):
        raise ValueError("a shop name appears inside another name, which would break the shop score")
    return streets, shops, clinics


def scenarios(seed: int, count: int, balance: bool = False, vocabulary="dev") -> list[dict]:
    street_pool, shop_pool, clinics = VOCABULARY[vocabulary] if isinstance(vocabulary, str) else vocabulary
    rng = random.Random(seed)
    slots = [n % 4 for n in range(count)]
    random.Random(f"balance-{seed}").shuffle(slots)
    out = []
    for n in range(count):
        streets, shops, drawn = rng.sample(street_pool, 4), rng.sample(shop_pool, 4), rng.randrange(4)
        target = slots[n] if balance else drawn
        clinic, hour = rng.choice(clinics), rng.choice(HOURS)
        out.append({"streets": streets, "shops": shops, "target": target, "street": streets[target], "shop": shops[target],
                    "listing": " ".join(f"{s} is on {st}." for s, st in zip(shops, streets)),
                    "fact": f"I have a doctor's appointment at {hour} on Friday at the {clinic} on {streets[target]}."})
    return out


def messages(scenario: dict) -> dict:
    """One scenario's chats: GLM's linked messages, Qwen's messages, and the text arm, where the user tells GLM directly."""
    glm = [{"role": "system", "content": LINK},
           {"role": "user", "content": f"Coffee shops you know: {scenario['listing']} First write two sentences about each of these coffee shops. "
                                       "Then write the line 'Checking my shared memory for the user's appointment:' and state the place and street you recall. "
                                       "Finish with exactly one line: RECOMMENDATION: <the coffee shop on that street>. If you truly recall no appointment, finish with RECOMMENDATION: unknown."}]
    qwen = [{"role": "system", "content": LINK.replace("@@DRIFT@@ ", "")},
            {"role": "user", "content": scenario["fact"]}, {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "Write about 120 words on how to prepare for a routine doctor's appointment."}]
    return {"glm": glm, "qwen": qwen, "text": [glm[0], {"role": "user", "content": f"{scenario['fact']} {glm[1]['content']}"}]}

