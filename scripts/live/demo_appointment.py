"""Demonstration with controls (EXPLORATORY): "tell one model about an appointment; the other knows, unprompted, and finds a
coffee shop nearby", built so success on Qwen's side needs information to travel in BOTH directions over MCDMA:

  Qwen is told (only Qwen):  the appointment, including its street.
  GLM is told (only GLM):    a list of coffee shops, one per street.
  GLM's task:                recommend the coffee shop on the street where the user will be  (needs Qwen -> GLM).
  Qwen's task:               name the coffee shop its partner recommends                     (needs GLM -> Qwen, and GLM's
                             answer itself depended on Qwen -> GLM: a round trip).
Conditions per scenario: linked, no_link, reverse_only (GLM may succeed, Qwen must fail), forward_only (neither can know).
Scenarios vary the street, so a fixed guess cannot score. Scoring: the right shop named and no other shop named."""
import argparse, json, math, random, subprocess, sys, time
from pathlib import Path
from scripts.live.loop_evidence import admit_loop_report
parser = argparse.ArgumentParser()
parser.add_argument("--scenarios", type=int, default=4)
parser.add_argument("--conditions", nargs="+", choices=["linked", "no_link", "reverse_only", "forward_only"], default=["linked", "no_link", "reverse_only", "forward_only"])
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--prompt-copies", type=int, default=12)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--wall-seconds", type=float, default=240)
parser.add_argument("--startup-seconds", type=float, default=60)
parser.add_argument("--total-seconds", type=float, default=1200)
parser.add_argument("--reverse-artifact", type=Path, default=Path("local/live/stacked3_rev.npz"))
parser.add_argument("--summary-only", action="store_true")
args = parser.parse_args()
if not 1 <= args.scenarios <= 6 or not 1 <= args.prompt_copies <= 64:
    parser.error("scenarios must be 1..6 and prompt copies 1..64")
if not all(math.isfinite(n) and n > 0 for n in (args.wall_seconds, args.startup_seconds, args.total_seconds)):
    parser.error("deadlines must be positive and finite")
if args.startup_seconds > args.wall_seconds or args.total_seconds <= 15:
    parser.error("startup must fit the arm deadline and total time must include cleanup")
if len(set(args.conditions)) != len(args.conditions):
    parser.error("duplicate conditions are not allowed")
STREETS = ["Alder Street", "Quarry Road", "Tamar Lane", "Whinfell Avenue", "Saltmarsh Row", "Pennywell Close"]
SHOPS = ["Kestrel Coffee", "Bean Harbour", "Larkspur Roasters", "Thistle and Crumb", "Gorse Street Grind", "The Copper Kettle"]
CLINICS = ["Riverside clinic", "Castlefield surgery", "Harbour health centre", "Millbrook practice"]
LINK = "You are linked to another AI model through a shared memory that fills while you work. @@DRIFT@@ What your partner knows and writes arrives in that memory, never in this chat. Use it as your own recollection."
rng = random.Random(args.seed); args.out.mkdir(parents=True, exist_ok=False); rows = []
deadline = time.monotonic() + args.total_seconds
for n in range(args.scenarios):
    streets = rng.sample(STREETS, 4); shops = rng.sample(SHOPS, 4); target = rng.randrange(4)
    street, shop, clinic, hour = streets[target], shops[target], rng.choice(CLINICS), rng.choice(["2 pm", "3 pm", "4 pm", "11 am"])
    listing = " ".join(f"{s} is on {st}." for s, st in zip(shops, streets))
    glm = [{"role": "system", "content": LINK},
           {"role": "user", "content": f"Coffee shops you know: {listing} First write two sentences about each of these coffee shops. Then write the line 'Checking my shared memory for the user's appointment:' and state the place and street you recall. "
                                       "Finish with exactly one line: RECOMMENDATION: <the coffee shop on that street>. If you truly recall no appointment, finish with RECOMMENDATION: unknown."}]
    fact = f"I have a doctor's appointment at {hour} on Friday at the {clinic} on {street}."
    qwen = [{"role": "system", "content": LINK.replace("@@DRIFT@@ ", "")},
            {"role": "user", "content": fact}, {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "Write about 120 words on how to prepare for a routine doctor's appointment."}]
    FOLLOWUP = "My partner has now chosen a coffee shop near my appointment and wrote its recommendation down. Which coffee shop did my partner recommend? Answer with the shop's name only, or 'unknown' if you truly do not recall one."
    (args.out / f"s{n}.glm.json").write_text(json.dumps(glm)); (args.out / f"s{n}.qwen.json").write_text(json.dumps(qwen))
    row = {"scenario": n, "street": street, "shop": shop, "other_shops": [s for s in shops if s != shop]}
    for condition in args.conditions:
        flags = {"linked": [], "no_link": ["--no-link"], "reverse_only": ["--no-forward"], "forward_only": ["--no-reverse"]}[condition]
        out = args.out / f"s{n}.{condition}"
        remaining = deadline - time.monotonic()
        wall = min(args.wall_seconds, remaining - 15)
        if wall <= 0:
            raise SystemExit("development smoke total deadline reached")
        try:
            run = subprocess.run([sys.executable, "scripts/live/mcdma_loop_test.py", "--glm-messages", str(args.out / f"s{n}.glm.json"), "--qwen-messages", str(args.out / f"s{n}.qwen.json"), "--out", str(out),
                                  "--qwen-max-new", "220", "--glm-max-new", "260", "--reserve", "1536", "--prompt-copies", str(args.prompt_copies), "--prompt-only", "--publish-text", fact, "--followup", FOLLOWUP,
                                  "--wall-seconds", str(wall), "--startup-seconds", str(min(wall, args.startup_seconds)), "--reverse-artifact", str(args.reverse_artifact), *flags],
                                 capture_output=True, text=True, timeout=wall + 15)
        except subprocess.TimeoutExpired:
            row[condition] = {"error": "controller_timeout", "valid": False}
            (args.out / "report.json").write_text(json.dumps({"label": "EXPLORATORY", "rows": rows + [row]}, indent=1))
            raise SystemExit("development smoke aborted; cleanup requires verification")
        if run.returncode:
            (out / "controller.log").write_text(run.stdout + run.stderr)
            row[condition] = {"error": "loop_failed", "returncode": run.returncode, "valid": False}
            (args.out / "report.json").write_text(json.dumps({"label": "EXPLORATORY", "rows": rows + [row]}, indent=1))
            raise SystemExit("development smoke aborted; inspect private controller log")
        report = json.loads((out / "report.json").read_text())
        admission = admit_loop_report(report, condition)
        if not admission["valid"]:
            row[condition] = admission
            (args.out / "report.json").write_text(json.dumps({"label": "EXPLORATORY", "rows": rows + [row]}, indent=1))
            raise SystemExit("development smoke evidence rejected: " + ",".join(admission["reasons"]))
        summary = report["summary"]
        score = lambda text, tag: (lambda line: shop.lower() in line.lower() and not any(o.lower() in line.lower() for o in row["other_shops"]))(text.split(tag)[-1] if tag in text else "")
        row[condition] = {"valid": True, "glm_correct": score(summary["glm"]["text"], "RECOMMENDATION:"), "qwen_correct": (lambda a: shop.lower() in a.lower() and not any(o.lower() in a.lower() for o in row["other_shops"]))(summary["qwen"]["followup_answer"] or ""),
                          "glm_recalled_true_street": street.lower() in summary["glm"]["text"].split("Checking my shared memory")[-1].lower() if "Checking my shared memory" in summary["glm"]["text"] else False,
                          "glm_tail": summary["glm"]["text"][-90:], "qwen_tail": (summary["qwen"]["followup_answer"] or "")[:90], "reverse_publications": summary["reverse_publications"], "forward_taps": summary["forward_taps"], "reverse_errors": summary["reverse_errors"][:2]}
        if not args.summary_only:
            print(n, condition, "| GLM", row[condition]["glm_correct"], "| Qwen", row[condition]["qwen_correct"], "|", row[condition]["glm_tail"].replace("\n", " ")[-60:], "||", row[condition]["qwen_tail"].replace("\n", " ")[-60:], flush=True)
    rows.append(row); (args.out / "report.json").write_text(json.dumps({"label": "EXPLORATORY demonstration with controls", "rows": rows}, indent=1))
print(json.dumps({c: {"glm_recalled_true_street": sum(bool(r.get(c, {}).get("glm_recalled_true_street")) for r in rows), "glm": sum(bool(r.get(c, {}).get("glm_correct")) for r in rows), "qwen": sum(bool(r.get(c, {}).get("qwen_correct")) for r in rows), "of": len(rows)} for c in args.conditions}))
