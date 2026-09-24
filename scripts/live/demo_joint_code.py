"""Joint code task over the live two-way loop: GLM writes code that needs a project detail only Qwen was told.

Arms per scenario: linked (the detail can only cross as Qwen's memory), no_link (GLM must guess) and text (the detail is in
GLM's prompt, the ceiling). Qwen publishes its whole context, so no step uses the answer. The driver stops at the first
invalid arm, as the appointment demo does, because a failed arm can leave remote processes that need checking.
"""
import argparse, json, math, subprocess, sys, time
from pathlib import Path
from scripts.live.joint_code_task import glm_messages, passes, qwen_messages, recalled, scenarios
from scripts.live.loop_evidence import admit_loop_report

parser = argparse.ArgumentParser()
parser.add_argument("--scenarios", type=int, default=16)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--arms", nargs="+", choices=["linked", "no_link", "text"], default=["linked", "no_link", "text"])
parser.add_argument("--prompt-copies", type=int, default=3)
parser.add_argument("--reserve", type=int, default=1536)
parser.add_argument("--wall-seconds", type=float, default=240)
parser.add_argument("--startup-seconds", type=float, default=60)
parser.add_argument("--total-seconds", type=float, default=3600)
parser.add_argument("--reverse-artifact", type=Path, default=Path("local/live/stacked3_rev.npz"))
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--causal", action="store_true", help="the linked arm's GLM reads its own prompt after the first publication")
parser.add_argument("--exploratory", action="store_true", help="label a development run so it cannot pass for the preregistered one")
args = parser.parse_args()
if not 1 <= args.scenarios <= 48 or not 1 <= args.prompt_copies <= 64 or not 64 <= args.reserve <= 16384:
    parser.error("scenarios must be 1..48, prompt copies 1..64 and reserve 64..16384")
if not all(math.isfinite(n) and n > 0 for n in (args.wall_seconds, args.startup_seconds, args.total_seconds)) or args.startup_seconds > args.wall_seconds:
    parser.error("deadlines must be positive and finite, and startup must fit the arm deadline")
if len(set(args.arms)) != len(args.arms):
    parser.error("duplicate arms are not allowed")
META = {"label": ("EXPLORATORY" if args.exploratory else "PREREGISTERED") + " joint code task", "seed": args.seed, "prompt_copies": args.prompt_copies,
        "reserve": args.reserve, "causal": args.causal}
args.out.mkdir(parents=True, exist_ok=False)
rows, deadline = [], time.monotonic() + args.total_seconds


def save():
    (args.out / "report.json").write_text(json.dumps({**META, "rows": rows}, indent=1))


def stop(row, arm, verdict, reason):
    row[arm] = verdict
    rows.append(row)
    save()
    raise SystemExit(f"joint task stopped: {reason}")


for n, scenario in enumerate(scenarios(args.seed, args.scenarios)):
    row = {"scenario": n, "variable": scenario["variable"]}
    (args.out / f"s{n}.qwen.json").write_text(json.dumps(qwen_messages(scenario)))
    for arm in args.arms:
        (args.out / f"s{n}.{arm}.glm.json").write_text(json.dumps(glm_messages(scenario, told=arm == "text")))
        out = args.out / f"s{n}.{arm}"
        wall = min(args.wall_seconds, deadline - time.monotonic() - 15)
        if wall <= 0:
            stop(row, arm, {"error": "total_deadline", "valid": False}, "total deadline reached")
        command = [sys.executable, "scripts/live/mcdma_loop_test.py", "--glm-messages", str(args.out / f"s{n}.{arm}.glm.json"),
                   "--qwen-messages", str(args.out / f"s{n}.qwen.json"), "--out", str(out), "--qwen-max-new", "220", "--glm-max-new", "320",
                   "--reserve", str(args.reserve), "--prompt-copies", str(args.prompt_copies), "--prompt-only", "--wall-seconds", str(wall),
                   "--startup-seconds", str(min(wall, args.startup_seconds)), "--reverse-artifact", str(args.reverse_artifact),
                   *([] if arm == "linked" else ["--no-link"]), *(["--causal"] if args.causal and arm == "linked" else [])]
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=wall + 15)
        except subprocess.TimeoutExpired:
            stop(row, arm, {"error": "controller_timeout", "valid": False}, "controller timeout; cleanup requires verification")
        if run.returncode:
            (out / "controller.log").write_text(run.stdout + run.stderr)
            stop(row, arm, {"error": "loop_failed", "returncode": run.returncode, "valid": False}, "loop failed; inspect the private controller log")
        report = json.loads((out / "report.json").read_text())
        admission = admit_loop_report(report, "linked" if arm == "linked" else "no_link")
        if not admission["valid"]:
            stop(row, arm, admission, "evidence rejected: " + ",".join(admission["reasons"]))
        summary = report["summary"]
        if arm == "linked" and any("skipped" in error for error in summary["reverse_errors"]):
            stop(row, arm, {"valid": False, "reasons": ["REVERSE_PUBLICATION_SKIPPED"]}, "publication skipped; raise --reserve")
        text = summary["glm"]["text"]
        row[arm] = {"valid": True, "passes": passes(text, scenario["variable"]), "recalled": recalled(text, scenario["variable"]),
                    "glm_tail": text[-200:], "reverse_publications": summary["reverse_publications"]}
        print(n, arm, "passes", row[arm]["passes"], "recalled", row[arm]["recalled"], flush=True)
    rows.append(row)
    save()
print(json.dumps({arm: {"passes": sum(bool(r.get(arm, {}).get("passes")) for r in rows),
                        "recalled": sum(bool(r.get(arm, {}).get("recalled")) for r in rows), "of": len(rows)} for arm in args.arms}))
