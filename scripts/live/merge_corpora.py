"""Merge corpus json files and build symlink folders so the streaming fits see one corpus and one folder per tap kind."""
import argparse, json, os
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=Path, nargs="+", required=True)
parser.add_argument("--taps", nargs="+", required=True, help="kind=folder[,folder...] e.g. glm=local/live/taps23_glm,local/live/multifact/taps_glm")
parser.add_argument("--out", type=Path, required=True, help="merged corpus json; symlink folders are created next to it as <stem>_<kind>")
args = parser.parse_args()
records, seen = [], set()
for path in args.corpus:
    for r in json.loads(path.read_text())["records"]:
        if r["id"] not in seen:
            seen.add(r["id"]); records.append(r)
for spec in args.taps:
    kind, folders = spec.split("=")
    target = args.out.with_name(f"{args.out.stem}_{kind}"); target.mkdir(parents=True, exist_ok=True)
    missing = 0
    for r in records:
        source = next((Path(f) / f"{r['id']}.npz" for f in folders.split(",") if (Path(f) / f"{r['id']}.npz").exists()), None)
        link = target / f"{r['id']}.npz"
        if source is None:
            missing += 1
        elif not link.exists():
            os.symlink(source.resolve(), link)
    print(kind, "linked", len(records) - missing, "missing", missing)
records = [r for r in records if all((args.out.with_name(f"{args.out.stem}_{s.split('=')[0]}") / f"{r['id']}.npz").exists() for s in args.taps)]
args.out.write_text(json.dumps({"records": records}))
print(json.dumps({"records": len(records), "train": sum(r["split"] == "train" for r in records)}))
