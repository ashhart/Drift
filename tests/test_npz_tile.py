"""Spark-side tiling without NumPy gives exactly what the producer used to send, and identical bytes on every rank."""
import importlib.util, io
import numpy as np
import pytest
spec = importlib.util.spec_from_file_location("npz_tile", "scripts/mcdma_target/npz_tile.py"); npz_tile = importlib.util.module_from_spec(spec); spec.loader.exec_module(npz_tile)


def test_tiled_file_loads_as_the_block_repeated_arrays_and_passes_the_connectors_validator(tmp_path):
    from drift.serving.live_publication import load_publication
    rng = np.random.default_rng(0)
    arrays = {f"l{l}": rng.standard_normal((7, 512)).astype(np.float16) for l in (3, 7, 43)}
    sink = io.BytesIO(); np.savez(sink, **arrays)
    tiled, rows = npz_tile.tile(sink.getvalue(), 12)
    assert rows == 84 and npz_tile.tile(sink.getvalue(), 12)[0] == tiled                       # deterministic bytes
    loaded = np.load(io.BytesIO(tiled))
    index = np.arange(84) % 7
    for key, value in arrays.items():
        np.testing.assert_array_equal(loaded[key], value[index])
    path = tmp_path / "000000.npz"; path.write_bytes(tiled)
    checked = load_publication(path, {key: (512,) for key in arrays})
    assert next(iter(checked.values())).shape == (84, 512)
    with pytest.raises(ValueError):
        npz_tile.tile(sink.getvalue(), 0)
    with pytest.raises(ValueError):
        npz_tile.tile(sink.getvalue(), 12, max_bytes=1000)
