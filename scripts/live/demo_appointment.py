"""Demonstration with controls (EXPLORATORY): "tell one model about an appointment; the other knows, unprompted, and finds a
coffee shop nearby", built so success on Qwen's side needs information to travel in BOTH directions over MCDMA:

  Qwen is told (only Qwen):  the appointment, including its street.
  GLM is told (only GLM):    a list of coffee shops, one per street.
  GLM's task:                recommend the coffee shop on the street where the user will be  (needs Qwen -> GLM).
  Qwen's task:               name the coffee shop its partner recommends                     (needs GLM -> Qwen, and GLM's
                             answer itself depended on Qwen -> GLM: a round trip).
Conditions per scenario: linked, no_link, reverse_only (GLM may succeed, Qwen must fail), forward_only (neither can know).
Scenarios vary the street, so a fixed guess cannot score. Scoring: the right shop named and no other shop named.
--selection oracle sends only the rows of the appointment sentence, which the controller finds by its text. --selection all
sends every row Qwen holds, so no step of the transfer uses the answer."""
import argparse, json, math, subprocess, sys, time
from pathlib import Path
from scripts.live.appointment_scenarios import FOLLOWUP, load_vocabulary, messages, scenarios
from scripts.live.loop_evidence import admit_loop_report
parser = argparse.ArgumentParser()
parser.add_argument("--scenarios", type=int, default=4)
parser.add_argument("--conditions", nargs="+", choices=["linked", "no_link", "reverse_only", "forward_only", "text"], default=["linked", "no_link", "reverse_only", "forward_only"])
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--prompt-copies", type=int, default=12)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--wall-seconds", type=float, default=240)
parser.add_argument("--startup-seconds", type=float, default=60)
parser.add_argument("--total-seconds", type=float, default=1200)
parser.add_argument("--reverse-artifact", type=Path, default=Path("local/live/stacked3_rev.npz"))
parser.add_argument("--selection", choices=["oracle", "all"], default="oracle", help="oracle sends the appointment sentence's rows; all sends every row Qwen holds")
parser.add_argument("--reserve", type=int, default=1536, help="rows GLM reserves for incoming memory; must cover every published row times its copies")
parser.add_argument("--balance", action="store_true", help="put the true street in each list position equally often")
parser.add_argument("--vocabulary", choices=["dev", "fresh", "confirm2"], default="dev", help="fresh: names no earlier run or prompt used; confirm2: reserved for the causal confirmation")
parser.add_argument("--vocabulary-file", type=Path, help="owner-written names for a held-out run; replaces --vocabulary")
parser.add_argument("--seed-file", type=Path, help="read the seed from this file, so it never appears on a command line")
parser.add_argument("--summary-only", action="store_true")
parser.add_argument("--causal", action="store_true", help="GLM reads its own prompt after the first publication, with the Drift scheduler")
parser.add_argument("--forward-state", help="GLM-to-Qwen state translator on the Studio, relative to drift-frontier; linked arms advance Qwen's recurrent layers with each tap")
parser.add_argument("--save-taps", action="store_true", help="the Studio keeps GLM's forward taps of each linked session, for offline replays")
parser.add_argument("--forward-rows-correction", help="a trained rows correction on the Studio, relative to drift-frontier")
args = parser.parse_args()
if not 1 <= args.scenarios <= 48 or not 1 <= args.prompt_copies <= 64:
    parser.error("scenarios must be 1..48 and prompt copies 1..64")
if not all(math.isfinite(n) and n > 0 for n in (args.wall_seconds, args.startup_seconds, args.total_seconds)):
    parser.error("deadlines must be positive and finite")
if args.startup_seconds > args.wall_seconds or args.total_seconds <= 15:
    parser.error("startup must fit the arm deadline and total time must include cleanup")
if len(set(args.conditions)) != len(args.conditions):
    parser.error("duplicate conditions are not allowed")
if not 64 <= args.reserve <= 16384:
    parser.error("reserve must be 64..16384 rows")
try:
    names = load_vocabulary(args.vocabulary_file) if args.vocabulary_file else args.vocabulary
    if args.seed_file:
        args.seed = int(args.seed_file.read_text().strip())
except (OSError, ValueError, KeyError) as error:
    parser.error(f"held-out inputs: {error}")
held_out = bool(args.vocabulary_file or args.seed_file)
META = {"selection": args.selection, "prompt_copies": args.prompt_copies, "reserve": args.reserve, "causal": args.causal, "forward_state": args.forward_state, "forward_rows_correction": args.forward_rows_correction, "save_taps": args.save_taps, "seed": None if args.seed_file else args.seed,
        "balance": args.balance, "vocabulary": "owner file" if args.vocabulary_file else args.vocabulary, "held_out_inputs": held_out}
args.out.mkdir(parents=True, exist_ok=False); rows = []
deadline = time.monotonic() + args.total_seconds
for n, scenario in enumerate(scenarios(args.seed, args.scenarios, args.balance, names)):
    streets, shops, street, shop = (scenario[k] for k in ("streets", "shops", "street", "shop"))
    fact, chats = scenario["fact"], messages(scenario)
    (args.out / f"s{n}.glm.json").write_text(json.dumps(chats["glm"])); (args.out / f"s{n}.qwen.json").write_text(json.dumps(chats["qwen"]))
    (args.out / f"s{n}.text.glm.json").write_text(json.dumps(chats["text"]))             # the text arm: the user tells GLM directly
    row = {"scenario": n, "street": street, "shop": shop, "position": scenario["target"], "other_shops": [s for s in shops if s != shop],
           "other_streets": [s for s in streets if s != street]}
    selection = ["--publish-text", fact] if args.selection == "oracle" else []
    for condition in args.conditions:
        flags = {"linked": [], "no_link": ["--no-link"], "reverse_only": ["--no-forward"], "forward_only": ["--no-reverse"], "text": ["--no-link"]}[condition]
        out = args.out / f"s{n}.{condition}"
        remaining = deadline - time.monotonic()
        wall = min(args.wall_seconds, remaining - 15)
        if wall <= 0:
            raise SystemExit("development smoke total deadline reached")
        try:
            run = subprocess.run([sys.executable, "scripts/live/mcdma_loop_test.py", "--glm-messages", str(args.out / (f"s{n}.text.glm.json" if condition == "text" else f"s{n}.glm.json")), "--qwen-messages", str(args.out / f"s{n}.qwen.json"), "--out", str(out),
                                  "--qwen-max-new", "220", "--glm-max-new", "260", "--reserve", str(args.reserve), "--prompt-copies", str(args.prompt_copies), "--prompt-only", *selection, "--followup", FOLLOWUP,
                                  "--wall-seconds", str(wall), "--startup-seconds", str(min(wall, args.startup_seconds)), "--reverse-artifact", str(args.reverse_artifact), *flags, *(["--causal"] if args.causal else []), *(["--forward-state", args.forward_state] if args.forward_state else []), *(["--save-taps"] if args.save_taps and condition == "linked" else []),
                                  *(["--forward-rows-correction", args.forward_rows_correction] if args.forward_rows_correction else [])],
                                 capture_output=True, text=True, timeout=wall + 15)
        except subprocess.TimeoutExpired:
            row[condition] = {"error": "controller_timeout", "valid": False}
            (args.out / "report.json").write_text(json.dumps({"label": "EXPLORATORY", **META, "rows": rows + [row]}, indent=1))
            raise SystemExit("development smoke aborted; cleanup requires verification")
        if run.returncode:
            (out / "controller.log").write_text(run.stdout + run.stderr)
            row[condition] = {"error": "loop_failed", "returncode": run.returncode, "valid": False}
            (args.out / "report.json").write_text(json.dumps({"label": "EXPLORATORY", **META, "rows": rows + [row]}, indent=1))
            raise SystemExit("development smoke aborted; inspect private controller log")
        report = json.loads((out / "report.json").read_text())
        admission = admit_loop_report(report, "no_link" if condition == "text" else condition)     # the text arm runs unlinked
        if not admission["valid"]:
            row[condition] = admission
            (args.out / "report.json").write_text(json.dumps({"label": "EXPLORATORY", **META, "rows": rows + [row]}, indent=1))
            raise SystemExit("development smoke evidence rejected: " + ",".join(admission["reasons"]))
        summary = report["summary"]
        # A skipped publication sent nothing, so scoring its recall would read as a failed transfer.
        if condition in ("linked", "reverse_only") and any("skipped" in error for error in summary["reverse_errors"]):
            row[condition] = {"valid": False, "reasons": ["REVERSE_PUBLICATION_SKIPPED"]}
            (args.out / "report.json").write_text(json.dumps({"label": "EXPLORATORY", **META, "rows": rows + [row]}, indent=1))
            raise SystemExit("development smoke evidence rejected: REVERSE_PUBLICATION_SKIPPED; raise --reserve")
        recalled = summary["glm"]["text"].split("Checking my shared memory")[-1].lower() if "Checking my shared memory" in summary["glm"]["text"] else ""
        score = lambda text, tag: (lambda line: shop.lower() in line.lower() and not any(o.lower() in line.lower() for o in row["other_shops"]))(text.split(tag)[-1] if tag in text else "")
        row[condition] = {"valid": True, "glm_correct": score(summary["glm"]["text"], "RECOMMENDATION:"), "qwen_correct": (lambda a: shop.lower() in a.lower() and not any(o.lower() in a.lower() for o in row["other_shops"]))(summary["qwen"]["followup_answer"] or ""),
                          "glm_recalled_true_street": street.lower() in recalled,
                          "glm_named_only_true_street": street.lower() in recalled and not any(o.lower() in recalled for o in row["other_streets"]),
                          "glm_tail": summary["glm"]["text"][-90:], "qwen_tail": (summary["qwen"]["followup_answer"] or "")[:90], "reverse_publications": summary["reverse_publications"], "forward_taps": summary["forward_taps"], "reverse_errors": summary["reverse_errors"][:2]}
        if not args.summary_only:
            print(n, condition, "| GLM", row[condition]["glm_correct"], "| Qwen", row[condition]["qwen_correct"], "|", row[condition]["glm_tail"].replace("\n", " ")[-60:], "||", row[condition]["qwen_tail"].replace("\n", " ")[-60:], flush=True)
    rows.append(row); (args.out / "report.json").write_text(json.dumps({"label": "EXPLORATORY demonstration with controls", **META, "rows": rows}, indent=1))
print(json.dumps({c: {"glm_recalled_true_street": sum(bool(r.get(c, {}).get("glm_recalled_true_street")) for r in rows),
                      "glm_named_only_true_street": sum(bool(r.get(c, {}).get("glm_named_only_true_street")) for r in rows), "glm": sum(bool(r.get(c, {}).get("glm_correct")) for r in rows), "qwen": sum(bool(r.get(c, {}).get("qwen_correct")) for r in rows), "of": len(rows)} for c in args.conditions}))
