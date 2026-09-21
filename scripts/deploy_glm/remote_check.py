"""Read-only payload run inside an existing Spark container."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def file_hash(path):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024 * 1024:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(request):
    target = request["target"]
    site = Path(target["site_packages"])
    directory = Path(request["candidate"]) if request["candidate"] else site
    if request["candidate"] and directory.resolve() == site.resolve():
        raise ValueError("staging directory aliases active site-packages")
    expected = {s["module"]: s["sha256"] for s in request["files"]}
    if request["candidate"]:
        allowed = {name + ".py" for name in expected} | {"manifest.json", "rollback-inventory.json"}
        if {path.name for path in directory.iterdir()} - allowed:
            raise ValueError("unexpected files could shadow staged imports")
    found = {name: file_hash(directory / (name + ".py")) for name in expected}
    active = {name: file_hash(site / (name + ".py")) for name in expected}
    base = file_hash(site / (target["base_module"] + ".py"))
    runtime = sys.version.split()[0] == target["python_version"] and sys.executable == target["python"]
    valid = runtime and found == expected and base == target["base_sha256"]
    result = {"host": target["ssh"], "valid": valid, "imports": False, "files": found,
              "active_files": active, "rollback_matches_manifest": active == target["rollback"],
              "base_sha256": base, "runtime_match": runtime}
    if request["candidate"] and valid:
        sys.dont_write_bytecode = True
        os.environ["DRIFT_GLM53_BASE"] = target["base_module"]
        sys.path.insert(0, str(directory))
        modules = {name: importlib.import_module(name) for name in expected}
        paths_match = all(Path(module.__file__).resolve() == (directory / (name + ".py")).resolve() for name, module in modules.items())
        unchanged = all(file_hash(directory / (name + ".py")) == value for name, value in expected.items())
        base_module = importlib.import_module(target["base_module"])
        base_matches = Path(base_module.__file__).resolve() == (site / (target["base_module"] + ".py")).resolve()
        connector = modules[request["entrypoint"]].DriftGlm53Connector
        result["imports"] = (paths_match and unchanged and base_matches
                             and file_hash(site / (target["base_module"] + ".py")) == target["base_sha256"]
                             and issubclass(connector, base_module.Glm53HandoffConnector))
        result["valid"] = bool(result["valid"] and result["imports"] and result["rollback_matches_manifest"])
    return result


if __name__ == "__main__":
    try:
        print("DRIFT_DEPLOY_CHECK=" + json.dumps(inspect(REQUEST), sort_keys=True))
    except Exception as error:
        print("DRIFT_DEPLOY_CHECK=" + json.dumps({"host": REQUEST["target"]["ssh"], "valid": False, "imports": False, "error": type(error).__name__}))
