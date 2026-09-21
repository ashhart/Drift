from __future__ import annotations
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_digest(model: torch.nn.Module) -> str:
    """Correctness helper. For 8B checkpoints prefer streaming on-disk shard hashes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        data = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(data.shape), data.dtype)).encode())
        digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def inventory(model_directory: Path | None = None) -> dict:
    packages = {}
    for package in ("torch", "numpy", "transformers", "mlx", "mlx-lm", "safetensors", "pytest"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": packages, "cuda_available": torch.cuda.is_available(),
              "mps_available": torch.backends.mps.is_available(),
              "model": None, "mcdma": "NOT_INSPECTED", "omp": "NOT_INSPECTED"}
    if model_directory is not None:
        config_path = model_directory / "config.json"
        config = json.loads(config_path.read_text())
        dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
        result["model"] = {"config_sha256": sha256_file(config_path),
                           "model_type": config["model_type"],
                           "layers": config["num_hidden_layers"],
                           "kv_heads": config["num_key_value_heads"], "head_dim": dim,
                           "rope_theta": config.get("rope_theta"),
                           "rope_scaling": config.get("rope_scaling"),
                           "tokenizer_files": {p.name: sha256_file(p) for p in model_directory.glob("tokenizer*") if p.is_file()},
                           "weight_shards": {p.name: sha256_file(p) for p in model_directory.glob("*.safetensors")},
                           "bytes_per_token_fp16_all_layers": 2 * 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * dim}
    return result
