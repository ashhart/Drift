"""Per-model translators into and out of a shared pool format (HIVE_MIND H1, H3, H9).

Every member owns one Translator: a writer per pool level (its canonical entry at the
mapped layer -> pool vector) and a reader per level (pool vector -> its canonical
entry). Members never map to each other directly. The pool format is fitted once by
the founding members, then frozen; later members enroll against it.

Layouts (docs/ADAPTERS.md descriptor): `kv_split` entries are KV [T,H,D] pairs and
flatten to width 2*H*D; `mla_latent` entries are [T,C] and are already flat.

Fitting is closed form here (centered ridge / PCA in float64). `kind="mlp"` adds a
trainable residual for later behavior training; it starts as the exact ridge map.
Nothing here claims semantic alignment; the round-trip and cross errors are numbers
to report, not evidence.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from drift.core.types import KV

POOL_VERSION = "pool.v1"


@dataclass(frozen=True)
class PoolFormat:
    version: str
    levels: int
    width: int
    fingerprint: str        # hash of the fitted format; members bind to it

    def check(self) -> None:
        if self.version != POOL_VERSION or self.levels <= 0 or self.width <= 0:
            raise ValueError("unsupported pool format")


@dataclass(frozen=True)
class Layout:
    kind: str               # "kv_split" | "mla_latent"
    heads: int              # 1 for mla_latent
    dim: int                # head_dim, or latent width for mla_latent

    @property
    def width(self) -> int:
        return 2 * self.heads * self.dim if self.kind == "kv_split" else self.dim

    def flatten(self, entry: KV | torch.Tensor) -> torch.Tensor:
        if self.kind == "kv_split":
            if not isinstance(entry, KV):
                raise ValueError("kv_split layout expects KV entries")
            entry.check()
            if entry.k.shape[1:] != (self.heads, self.dim):
                raise ValueError("entry shape does not match layout")
            return torch.cat((entry.k.flatten(1), entry.v.flatten(1)), dim=1)
        if not isinstance(entry, torch.Tensor) or entry.ndim != 2 or entry.shape[1] != self.dim:
            raise ValueError("mla_latent layout expects [T,C] tensors")
        if not torch.isfinite(entry).all():
            raise ValueError("nonfinite latent")
        return entry

    def unflatten(self, rows: torch.Tensor) -> KV | torch.Tensor:
        if rows.ndim != 2 or rows.shape[1] != self.width:
            raise ValueError("flat row width does not match layout")
        if self.kind == "kv_split":
            half = self.heads * self.dim
            return KV(rows[:, :half].reshape(-1, self.heads, self.dim),
                      rows[:, half:].reshape(-1, self.heads, self.dim))
        return rows


def _ridge(x: torch.Tensor, y: torch.Tensor, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Centered ridge in float64: returns (weight [out,in], bias [out])."""
    if ridge <= 0 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("need paired rows and positive ridge")
    a, b = x.detach().cpu().double(), y.detach().cpu().double()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("nonfinite fitting data")
    am, bm = a.mean(0), b.mean(0)
    ac, bc = a - am, b - bm
    lhs = ac.T @ ac + ridge * torch.eye(ac.shape[1], dtype=torch.float64)
    weight = torch.linalg.solve(lhs, ac.T @ bc)          # [in,out]
    return weight.T, bm - am @ weight


class _Map(nn.Module):
    """Linear map with an optional zero-initialized residual MLP (kind="mlp")."""
    def __init__(self, inputs: int, outputs: int, kind: str, rank: int = 64):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs)
        self.residual = None
        if kind == "mlp":
            self.residual = nn.Sequential(nn.LayerNorm(inputs), nn.Linear(inputs, rank), nn.SiLU(), nn.Linear(rank, outputs))
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        return y if self.residual is None else y + self.residual(x)

    @torch.no_grad()
    def set_affine(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        self.linear.weight.copy_(weight.to(self.linear.weight))
        self.linear.bias.copy_(bias.to(self.linear.bias))


class Translator(nn.Module):
    """One member's writer/reader pair for every pool level it participates in."""
    def __init__(self, member: str, layout: Layout, level_map: Mapping[int, int],
                 pool: PoolFormat, kind: str = "ridge", rank: int = 64):
        super().__init__()
        pool.check()
        if kind not in {"ridge", "mlp"}:
            raise ValueError("kind must be ridge or mlp")
        if not level_map or len(set(level_map.values())) != len(level_map):
            raise ValueError("each participating layer maps to exactly one level")
        if any(not 0 <= level < pool.levels for level in level_map.values()):
            raise ValueError("level outside the pool")
        self.member, self.layout, self.pool, self.kind, self.rank = member, layout, pool, kind, rank
        self.level_map = dict(level_map)                     # model layer -> pool level
        self.writers = nn.ModuleDict({str(level): _Map(layout.width, pool.width, kind, rank) for level in level_map.values()})
        self.readers = nn.ModuleDict({str(level): _Map(pool.width, layout.width, kind, rank) for level in level_map.values()})

    @property
    def layer_of(self) -> dict[int, int]:
        return {level: layer for layer, level in self.level_map.items()}

    def write(self, canonical: Mapping[int, KV | torch.Tensor]) -> dict[int, torch.Tensor]:
        """Canonical entries keyed by model layer -> pool vectors keyed by level, [T,width]."""
        out = {}
        for layer, level in self.level_map.items():
            if layer not in canonical:
                raise ValueError(f"missing canonical entry for layer {layer}")
            out[level] = self.writers[str(level)](self.layout.flatten(canonical[layer]))
        return out

    def read(self, pool_entries: Mapping[int, torch.Tensor]) -> dict[int, KV | torch.Tensor]:
        """Pool vectors keyed by level -> canonical entries keyed by this model's layer."""
        out = {}
        for layer, level in self.level_map.items():
            rows = pool_entries.get(level)
            if rows is None:
                raise ValueError(f"pool level {level} absent")
            if rows.ndim != 2 or rows.shape[1] != self.pool.width:
                raise ValueError("pool row width mismatch")
            out[layer] = self.layout.unflatten(self.readers[str(level)](rows))
        return out


# ----------------------------------------------------------------------------- fitting

def fit_pool_format(rows: Mapping[int, Mapping[str, torch.Tensor]], width: int) -> tuple[PoolFormat, dict[int, torch.Tensor]]:
    """Found the shared format from ALIGNED rows of the founding members.

    rows[level][member] is [N, member_width]; all members at a level share N (causal
    endpoints, D8). The shared vector is the top-`width` joint PCA of the concatenated
    member rows at each level. Returns the frozen format and the pool rows per level.
    """
    if width <= 0 or not rows:
        raise ValueError("positive width and at least one level required")
    pool_rows, digest = {}, hashlib.sha256()
    for level in sorted(rows):
        members = rows[level]
        if len(members) < 1:
            raise ValueError("a level needs at least one founding member")
        counts = {m.shape[0] for m in members.values()}
        if len(counts) != 1 or next(iter(counts)) <= width:
            raise ValueError("members must supply the same, sufficient number of aligned rows")
        joint = torch.cat([members[name].detach().cpu().double() for name in sorted(members)], dim=1)
        if not torch.isfinite(joint).all():
            raise ValueError("nonfinite founding rows")
        centered = joint - joint.mean(0)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        if vh.shape[0] < width:
            raise ValueError("joint width smaller than requested pool width")
        z = centered @ vh[:width].T                        # [N, width]
        pool_rows[level] = z
        digest.update(f"{level}:{tuple(joint.shape)}".encode())
        digest.update(vh[:width].contiguous().numpy().tobytes())
    fmt = PoolFormat(POOL_VERSION, max(rows) + 1, width, digest.hexdigest())
    return fmt, pool_rows


def fit_member(translator: Translator, member_rows: Mapping[int, torch.Tensor],
               pool_rows: Mapping[int, torch.Tensor], ridge: float = 1e-3) -> dict:
    """Fit one member's writers/readers against fixed pool rows (founding or enrollment).

    member_rows[level] is [N, member_width] aligned with pool_rows[level] [N, pool_width].
    Only this translator changes (H9). Returns per-level round-trip and fit errors.
    """
    report = {}
    for level in translator.writers:
        lvl = int(level)
        x, z = member_rows.get(lvl), pool_rows.get(lvl)
        if x is None or z is None:
            raise ValueError(f"missing rows for level {lvl}")
        if x.shape[1] != translator.layout.width or z.shape[1] != translator.pool.width:
            raise ValueError("row widths do not match the translator")
        w, b = _ridge(x, z, ridge)
        translator.writers[level].set_affine(w, b)
        w, b = _ridge(z, x, ridge)
        translator.readers[level].set_affine(w, b)
        with torch.no_grad():
            xf = x.detach().to(translator.writers[level].linear.weight)
            zf = z.detach().to(xf)
            written = translator.writers[level](xf)
            roundtrip = translator.readers[level](written)
            report[lvl] = {
                "write_rmse": float((written - zf).pow(2).mean().sqrt()),
                "roundtrip_rmse": float((roundtrip - xf).pow(2).mean().sqrt()),
                "baseline_rmse": float((xf - xf.mean(0)).pow(2).mean().sqrt()),
                "rows": int(x.shape[0]),
            }
    return report


def cross_error(writer: Translator, reader: Translator,
                writer_rows: Mapping[int, torch.Tensor], reader_rows: Mapping[int, torch.Tensor]) -> dict[int, dict]:
    """Held-out: reader.read(writer.write(x_writer)) vs x_reader, per shared level."""
    out = {}
    with torch.no_grad():
        for level in set(map(int, writer.writers)) & set(map(int, reader.readers)):
            x, y = writer_rows[level], reader_rows[level]
            z = writer.writers[str(level)](x.to(writer.writers[str(level)].linear.weight))
            y_hat = reader.readers[str(level)](z)
            yf = y.to(y_hat)
            out[level] = {"cross_rmse": float((y_hat - yf).pow(2).mean().sqrt()),
                          "baseline_rmse": float((yf - yf.mean(0)).pow(2).mean().sqrt())}
    return out


# ----------------------------------------------------------------------------- persistence

def save_translator(translator: Translator, directory: Path, provenance: dict) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    weights = directory / "translator.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in translator.state_dict().items()}, str(weights))
    manifest = {
        "format": 1, "member": translator.member, "kind": translator.kind, "rank": translator.rank,
        "layout": {"kind": translator.layout.kind, "heads": translator.layout.heads, "dim": translator.layout.dim},
        "level_map": {str(k): v for k, v in translator.level_map.items()},
        "pool": {"version": translator.pool.version, "levels": translator.pool.levels,
                 "width": translator.pool.width, "fingerprint": translator.pool.fingerprint},
        "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "provenance": provenance,
    }
    (directory / "translator.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_translator(directory: Path, pool: PoolFormat, device: str = "cpu") -> Translator:
    manifest = json.loads((directory / "translator.json").read_text())
    if manifest.get("format") != 1:
        raise ValueError("unknown translator artifact version")
    saved = manifest["pool"]
    if (saved["version"], saved["levels"], saved["width"], saved["fingerprint"]) != (pool.version, pool.levels, pool.width, pool.fingerprint):
        raise ValueError("translator was fitted against a different pool format; re-enroll")
    weights = directory / "translator.safetensors"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("translator checkpoint hash mismatch")
    layout = Layout(manifest["layout"]["kind"], int(manifest["layout"]["heads"]), int(manifest["layout"]["dim"]))
    level_map = {int(k): int(v) for k, v in manifest["level_map"].items()}
    translator = Translator(manifest["member"], layout, level_map, pool, manifest["kind"], int(manifest["rank"]))
    translator.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    return translator.to(device).eval()
