"""Build a local text corpus with causal-endpoint alignment and separate tap IDs."""
import argparse, hashlib, json, random, re
from pathlib import Path
from tokenizers import Tokenizer
from drift.train.alignment import shared_causal_endpoints

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--max-chunks", type=int, default=180)
parser.add_argument("--heldout", type=int, default=24)
parser.add_argument("--jsonl", type=Path, help="generated passages (gen_corpus.py) instead of the repository text")
args = parser.parse_args()
glm, qwen = Tokenizer.from_file("local/tok/glm/tokenizer.json"), Tokenizer.from_file("local/tok/qwen/tokenizer.json")
sources = [*sorted(Path("docs").rglob("*.md")), *sorted(Path("drift").rglob("*.py")), Path("README.md"), Path("AGENTS.md")]
chunks = []
for path in ([] if args.jsonl else sources):
    text = path.read_text(encoding="utf-8")
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if 350 <= len(para) <= 900 and para.isascii():
            chunks.append((str(path), para))
extra = {}
if args.jsonl:
    for line in args.jsonl.read_text().splitlines():
        r = json.loads(line)
        chunks.append((f"generated:{r['seed']}:{r['n']}", r["text"]))
        extra[r["text"]] = {k: r[k] for k in ("genre", "place", "thing", "facts")}
random.Random(7).shuffle(chunks)
records, skipped = [], 0
for source, text in chunks:
    eg, eq = glm.encode(text, add_special_tokens=False), qwen.encode(text, add_special_tokens=False)
    if glm.decode(eg.ids, skip_special_tokens=False) != text or qwen.decode(eq.ids, skip_special_tokens=False) != text:
        skipped += 1; continue
    aligned = shared_causal_endpoints(text, list(eg.offsets), list(eq.offsets))
    if len(aligned) < 20 or max(len(eg.ids), len(eq.ids)) > 320:
        skipped += 1; continue
    records.append({"id": hashlib.sha256(text.encode()).hexdigest()[:16], "source": source, "text": text,
                    "ids_glm": eg.ids, "ids_qwen": eq.ids, **extra.get(text, {}), "aligned": [[g, q] for g, q, _ in aligned]})
    if len(records) >= args.max_chunks:
        break
for i, r in enumerate(records):
    r["split"] = "heldout" if i < args.heldout else "train"
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps({"records": records}, indent=0))
ids_only = {"records": [{k: r[k] for k in ("id", "split", "ids_glm", "ids_qwen")} for r in records]}
args.out.with_suffix(".ids.json").write_text(json.dumps(ids_only))
tok_g, tok_q, rows = sum(len(r["ids_glm"]) for r in records), sum(len(r["ids_qwen"]) for r in records), sum(len(r["aligned"]) for r in records)
print(json.dumps({"chunks": len(records), "skipped": skipped, "train": sum(r["split"] == "train" for r in records), "heldout": args.heldout,
                  "glm_tokens": tok_g, "qwen_tokens": tok_q, "aligned_rows": rows, "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}))
