"""Pin and prepare a complete flat connector bundle without remote mutation."""
import ast
import hashlib
import json
import re
from pathlib import Path
import sys
import tempfile

ROOT_SOURCE = "drift/serving/vllm_glm53_inject.py"
ENTRYPOINT = "drift_glm53_connector"
MAX_MODULES = 24                                                    # engineering bound on the flat bundle


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bundle_id(manifest):
    return digest(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("version") != 1 or manifest.get("entrypoint") != "drift_glm53_connector":
        raise ValueError("unsupported deployment manifest")
    files = manifest.get("files", [])
    mapping = {s["module"]: s["source"] for s in files}
    if len(mapping) != len(files) or mapping.get(ENTRYPOINT) != ROOT_SOURCE or len(files) > MAX_MODULES:
        raise ValueError("deployment must contain one complete flat dependency closure")
    for name, source in mapping.items():
        expected = ROOT_SOURCE if name == ENTRYPOINT else "drift/serving/" + name + ".py"
        if not name.isidentifier() or source != expected:
            raise ValueError("unsupported module/source mapping")
    targets = manifest.get("targets", [])
    if len(targets) != 2 or any(not isinstance(t, dict) for t in targets):
        raise ValueError("two pinned deployment targets are required")
    for target in targets:
        for key in ('ssh', 'container'):
            value = target.get(key)
            if type(value) is not str or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', value) is None:
                raise ValueError("invalid deployment target identity")
    if len({target['ssh'] for target in targets}) != 2:
        raise ValueError("two distinct deployment hosts are required")
    return manifest


def dependencies(source):
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                raise ValueError("relative dependency is not supported by the flat bundle")
            found.add(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__import__":
            if not node.args or not isinstance(node.args[0], ast.Constant):
                raise ValueError("dynamic dependency cannot be checked")
            found.add(node.args[0].value)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
            expected = 'importlib.import_module(os.environ.get("DRIFT_GLM53_BASE", "glm53_handoff_connector"))'
            if ast.dump(node) != ast.dump(ast.parse(expected, mode="eval").body):
                raise ValueError("undeclared dynamic dependency")
    return found


def source_closure(root):
    root = Path(root).resolve()
    found = {}
    pending = [ROOT_SOURCE, 'drift/serving/glm_prefill_scheduler.py']
    while pending:
        relative = pending.pop()
        name = ENTRYPOINT if relative == ROOT_SOURCE else Path(relative).stem
        if name in found:
            continue
        source = root / relative
        if source.is_symlink() or not source.resolve().is_relative_to(root) or source.stat().st_size > 1024 * 1024:
            raise ValueError("source path or size outside the bundle contract")
        found[name] = relative
        for imported in dependencies(source.read_text()):
            flat = imported.removeprefix("drift.serving.")
            local = root / "drift/serving" / (flat + ".py")
            if flat.isidentifier() and local.is_file():
                pending.append(str(local.relative_to(root)))
            elif imported.startswith("drift.") or imported.split(".")[0] not in sys.stdlib_module_names | {"numpy", "torch", "vllm"}:
                raise ValueError("undeclared dependency: " + imported)
        if len(found) > MAX_MODULES:
            raise ValueError("dependency closure exceeds the bundle limit")
    return found


def validate_sources(root, manifest):
    required = source_closure(root)
    declared = {s["module"]: s["source"] for s in manifest["files"]}
    if required != declared:
        raise ValueError("dependency closure differs from pinned files")
    for spec in manifest["files"]:
        if digest((Path(root) / spec["source"]).read_bytes()) != spec["sha256"]:
            raise ValueError("source hash mismatch: " + spec["module"])


def prepare(root, manifest, output):
    validate_sources(root, manifest)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".drift-bundle-") as temp:
        stage = Path(temp) / "candidate"
        stage.mkdir()
        for spec in manifest["files"]:
            data = (Path(root) / spec["source"]).read_bytes()
            if digest(data) != spec["sha256"]:
                raise ValueError("source hash changed during preparation")
            (stage / (spec["module"] + ".py")).write_bytes(data)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (stage / "rollback-inventory.json").write_text(json.dumps(manifest["targets"], indent=2) + "\n")
        stage.rename(output)
    return {"bundle_sha256": bundle_id(manifest), "output": str(output), "promotion_performed": False}


def pin_sources(root, manifest, commit):
    pinned = json.loads(json.dumps(manifest))
    pinned["source_commit"] = commit
    pinned["files"] = [{"module": name, "source": source, "sha256": digest((Path(root) / source).read_bytes())}
                       for name, source in sorted(source_closure(root).items())]
    validate_sources(root, pinned)
    return pinned
