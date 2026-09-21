from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from drift.taps.validation import validate_name


def package_name(adapter_id: str) -> str:
    family = adapter_id.split("/", 1)[0]
    return "drift_adapter_" + re.sub(r"[^a-z0-9_]", "_", family)


def scaffold_adapter(adapter_id: str, model_type: str, output: Path) -> dict:
    validate_name(adapter_id, "adapter")
    validate_name(model_type, "model type")
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    package = package_name(adapter_id)
    try:
        source = temporary / "src" / package
        source.mkdir(parents=True)
        (source / "__init__.py").write_text("from .adapter import load\n\n__all__ = [\"load\"]\n")
        (source / "adapter.py").write_text(
            "from __future__ import annotations\n\n\n"
            "def load(checkpoint_dir: str, device: str) -> object:\n"
            f"    raise RuntimeError(\"BLOCKED: implement the {model_type} adapter\")\n"
        )
        pyproject = (
            "[build-system]\n"
            "requires = [\"setuptools>=68\"]\n"
            "build-backend = \"setuptools.build_meta\"\n\n"
            "[project]\n"
            f"name = \"{package.replace('_', '-')}\"\n"
            "version = \"0.1.0\"\n"
            "requires-python = \">=3.11,<3.14\"\n"
            "dependencies = [\"drift-reference==0.2.0\"]\n\n"
            "[project.entry-points.\"drift.adapters\"]\n"
            f"\"{adapter_id}\" = \"{package}.adapter:load\"\n\n"
            "[tool.setuptools.packages.find]\n"
            "where = [\"src\"]\n"
        )
        (temporary / "pyproject.toml").write_text(pyproject)
        readme = {
            "adapter_id": adapter_id,
            "model_type": model_type,
            "next_gate": "Implement capture, foreign attention, rephasing, private state, and exact hard-off behavior, then run tiny and real-weight qualification.",
        }
        (temporary / "ADAPTER.json").write_text(json.dumps(readme, indent=2) + "\n")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return {"status": "PASSED", "adapter": adapter_id, "path": str(output.resolve())}
