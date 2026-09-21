import io
from zipfile import ZipFile

import numpy as np
import pytest

from drift.serving.live_publication import count, load_publication, wire_arrays


def test_round_trip_and_integer_cursors(tmp_path):
    path = tmp_path / "tap.npz"
    np.savez(path, l3=np.ones((2, 4), np.float16), start=np.int64(1024), stop=np.int64(1026))
    result = load_publication(path, {"l3": (4,)}, metadata=("start", "stop"))
    assert result["start"] == 1024 and result["stop"] == 1026
    assert result["l3"].dtype == np.float32
    assert wire_arrays({"l3": result["l3"]}, {"l3": (4,)}, 2)["l3"].dtype == np.float16


@pytest.mark.parametrize("bad", [np.ones((0, 4)), np.ones((4097, 4)), np.ones((1, 3)),
                                np.ones((1, 4), int), np.ones((1, 4), object),
                                np.full((1, 4), np.nan), np.full((1, 4), np.inf),
                                np.full((1, 4), 1e300)])
def test_bad_arrays_rejected(tmp_path, bad):
    path = tmp_path / "tap.npz"
    np.savez(path, l3=bad)
    with pytest.raises(ValueError):
        load_publication(path, {"l3": (4,)})


def test_all_layers_required_with_consistent_rows_and_no_extra_fields(tmp_path):
    path = tmp_path / "tap.npz"
    layout = {"k3": (1, 2), "v3": (1, 2)}
    for fields in ({"k3": np.ones((1, 1, 2))},
                   {"k3": np.ones((1, 1, 2)), "v3": np.ones((2, 1, 2))},
                   {"k3": np.ones((1, 1, 2)), "v3": np.ones((1, 1, 2)), "ids": np.array([1])}):
        np.savez(path, **fields)
        with pytest.raises(ValueError):
            load_publication(path, layout)


def test_bogus_shape_rejected_before_allocation(tmp_path):
    path = tmp_path / "tap.npz"
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(buffer, {"shape": (1, 4), "fortran_order": False, "descr": "<f4"})
    with ZipFile(path, "w") as archive:
        archive.writestr("l3.npy", buffer.getvalue())  # claims 16 absent bytes
    with pytest.raises(ValueError, match="byte count"):
        load_publication(path, {"l3": (4,)})


@pytest.mark.parametrize("value", [True, -1, 1.5, "3", None])
def test_invalid_counts(value):
    with pytest.raises(ValueError):
        count(value)


@pytest.mark.parametrize("fields", [{"l3": np.full((1, 4), 1e10)},
                                    {"l3": np.full((1, 4), np.nan)},
                                    {"l3": np.ones((2, 4))}, {}])
def test_bad_translated_values_cannot_be_serialized(fields):
    with pytest.raises(ValueError):
        wire_arrays(fields, {"l3": (4,)}, 1)
