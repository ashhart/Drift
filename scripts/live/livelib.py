"""Shared SSH exchange helpers with environment-selected hosts."""
from __future__ import annotations
import hashlib, json, os, subprocess
from pathlib import Path
import numpy as np
from drift.translate.pack import ModelPack
from drift.translate.stacked import StackedReader

SPARK, STUDIO = os.environ.get("DRIFT_SPARK", "spark-a.invalid"), os.environ.get("DRIFT_STUDIO", "studio")
SPARK_PEERS = [h for h in os.environ.get("DRIFT_SPARK_PEERS", "spark-b.invalid").split(",") if h]      # other tensor-parallel hosts
MCDMA_LINKS = os.environ.get("DRIFT_MCDMA_LINKS", "")                                                  # Studio legs to each rank as target/source, head first
OMLX = "/Applications/oMLX.app/Contents/Resources"
OMLX_PY = f'PYTHONPATH="{OMLX}/Python/framework-mlx-base/lib/python3.11/site-packages:{OMLX}:." {OMLX}/Python/cpython-3.11/bin/python3'
GLM_LAYERS, QWEN_LAYERS = tuple(3 + 4 * i for i in range(11)), tuple(3 + 4 * i for i in range(12))


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def passage_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_reader(path: Path, *, recipe="legacy", manifest=None, kv_heads=2, head_dim=256):
    if recipe != "legacy":
        from drift.serving.live_recipe import load_recipe
        return load_recipe(manifest, recipe, GLM_LAYERS, QWEN_LAYERS, kv_heads, head_dim)
    if manifest is not None:
        raise ValueError("a manifest requires an explicit forward recipe")
    return StackedReader.load(path, GLM_LAYERS, QWEN_LAYERS, kv_heads=kv_heads, head_dim=head_dim)


def glm_read(passages: dict[str, str], out: Path, run: str) -> Path:
    """The writer reads each passage on the Sparks; the running vLLM server exports its cache latents."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("local/tok/glm/tokenizer.json")
    ids = {"records": [{"id": pid, "ids_glm": tok.encode(text, add_special_tokens=False).ids} for pid, text in passages.items()]}
    (out / "glm.ids.json").write_text(json.dumps(ids))
    remote = f"/root/drift-live/runs/{run}/glm_read"            # its own subfolder: wiping the run folder once destroyed 60 finished answers
    for peer in SPARK_PEERS:                                     # every rank exports a blob; only rank 0 is read, the rest must be swept
        sh("scp", "-q", "scripts/live/spark_janitor.sh", f"{peer}:/root/drift-janitor.sh")
        sh("ssh", peer, "nohup /root/drift-janitor.sh 7200 > /dev/null 2>&1 < /dev/null &")
    sh("ssh", SPARK, f"rm -rf {remote} && mkdir -p {remote}")
    sh("scp", "-q", "scripts/live/spark_run.sh", "scripts/live/spark_tap_glm.py", "drift/serving/glm53_handoff.py", f"{SPARK}:/root/drift-live/")
    sh("scp", "-q", str(out / "glm.ids.json"), f"{SPARK}:{remote}/ids.json")
    print("GLM:", sh("ssh", SPARK, f"/root/drift-live/spark_run.sh spark_tap_glm.py --ids {remote}/ids.json --out {remote}/taps").strip())
    sh("rsync", "-a", f"{SPARK}:{remote}/taps/", str(out / "taps_glm"))
    return out / "taps_glm"


def translate(taps: Path, ids: list[str], reader: StackedReader, gain_power: float, out: Path) -> Path:
    (out / "memory").mkdir(parents=True, exist_ok=True)
    for pid in ids:
        z = np.load(taps / f"{pid}.npz")
        entries = reader.read({l: z[f"l{l}"].astype(np.float32) for l in GLM_LAYERS}, gain_power)
        np.savez(out / "memory" / f"{pid}.npz", **{f"k{l}": k.astype(np.float16) for l, (k, v) in entries.items()}, **{f"v{l}": v.astype(np.float16) for l, (k, v) in entries.items()})
    return out / "memory"


def qwen_answer(jobs: list[dict], out: Path, run: str, max_new: int) -> dict:
    """The reader answers on the Studio. Job memory paths are relative to `out` here and rewritten for the remote run folder."""
    remote = f"local/studio/runs/{run}"
    fix = lambda p: f"{remote}/{p}" if p else p
    remote_jobs = [{**j, "memory": fix(j.get("memory")), "wrong_memory": fix(j.get("wrong_memory")), "memories": {k: fix(v) for k, v in (j.get("memories") or {}).items()}} for j in jobs]
    (out / "jobs.json").write_text(json.dumps(remote_jobs))
    sh("ssh", STUDIO, f"rm -rf drift/{remote} && mkdir -p drift/{remote}")
    sh("rsync", "-a", *[str(p) for p in out.iterdir() if p.is_dir() and p.name.startswith("memory")], str(out / "jobs.json"), f"{STUDIO}:drift/{remote}/")
    sh("rsync", "-a", "scripts", "drift", f"{STUDIO}:drift/")
    print("Qwen:", sh("ssh", STUDIO, f"cd ~/drift && {OMLX_PY} scripts/live/studio_live_answer.py --jobs {remote}/jobs.json --out {remote}/answers.json --max-new {max_new} 2>/dev/null | tail -1").strip())
    sh("rsync", "-a", f"{STUDIO}:drift/{remote}/answers.json", str(out / "answers.json"))
    return json.loads((out / "answers.json").read_text())


def load_packs(directory: Path) -> tuple[ModelPack, ModelPack]:
    return ModelPack.load(directory / "glm.pack.npz"), ModelPack.load(directory / "qwen.pack.npz", kv_heads=2)


def translate_packs(taps: Path, ids: list[str], writer: ModelPack, reader: ModelPack, gain_power: float, out: Path) -> Path:
    """GLM taps -> pool.v2 rows (the only thing that would cross the wire) -> Qwen entries."""
    (out / "memory").mkdir(parents=True, exist_ok=True)
    for pid in ids:
        z = np.load(taps / f"{pid}.npz")
        pool_rows = writer.write({l: z[f"l{l}"].astype(np.float32) for l in GLM_LAYERS})
        entries = reader.read(pool_rows, writer.pool_fingerprint, gain_power)
        np.savez(out / "memory" / f"{pid}.npz", **{f"k{l}": k.astype(np.float16) for l, (k, v) in entries.items()}, **{f"v{l}": v.astype(np.float16) for l, (k, v) in entries.items()})
    return out / "memory"
