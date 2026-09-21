"""Tile the rows of every array in an .npz, stdlib only (the Sparks' host Python has no NumPy).

Input: np.savez output (stored zip of version-1.0 .npy members, C order, first axis = rows). Output: the same members with
their rows repeated `copies` times as whole blocks (row r of the output is row r % n of the input, what the reverse
publications use as an attention prior). Deterministic bytes: fixed timestamps, stored members, canonical headers, so every
rank produces the identical file and its SHA-256 can be compared across ranks."""
import ast, io, struct, zipfile

MAGIC = b"\x93NUMPY\x01\x00"


def _header(descr: str, shape: tuple) -> bytes:
    text = "{'descr': %r, 'fortran_order': False, 'shape': %r, }" % (descr, shape)
    pad = 64 - (len(MAGIC) + 2 + len(text) + 1) % 64
    text = text + " " * (pad % 64) + "\n"
    return MAGIC + struct.pack("<H", len(text)) + text.encode("latin1")


def tile(npz: bytes, copies: int, max_bytes: int = 128 << 20) -> tuple[bytes, int]:
    if type(copies) is not int or not 1 <= copies <= 64:
        raise ValueError("copies out of range")
    out, rows_seen = io.BytesIO(), set()
    with zipfile.ZipFile(io.BytesIO(npz)) as source, zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as sink:
        for name in sorted(source.namelist()):
            raw = source.read(name)
            if not name.endswith(".npy") or raw[:8] != MAGIC:
                raise ValueError("unsupported npz member")
            (length,) = struct.unpack("<H", raw[8:10])
            meta = ast.literal_eval(raw[10:10 + length].decode("latin1"))
            shape, data = tuple(meta["shape"]), raw[10 + length:]
            if meta.get("fortran_order") or len(shape) < 1 or shape[0] < 1 or len(data) % shape[0]:
                raise ValueError("unsupported array layout")
            if len(data) * copies > max_bytes:
                raise ValueError("tiled publication too large")
            rows_seen.add(shape[0])
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)); info.external_attr = 0o600 << 16
            sink.writestr(info, _header(meta["descr"], (shape[0] * copies,) + shape[1:]) + data * copies)
    if len(rows_seen) != 1:
        raise ValueError("members disagree on the row count")
    return out.getvalue(), rows_seen.pop() * copies
