"""Storage key validation (SPEC 3.2/3.3)."""

from __future__ import annotations

import pytest

from screwit.storage.keys import KeyError as StorageKeyError
from screwit.storage.keys import validate_key


@pytest.mark.parametrize("key", [
    "scans/scn_8f2a/input/video.mov",
    "scans/scn_knob/cad/cad_07/model.stl",
    "scans/a/b",
])
def test_good_keys(key):
    assert validate_key(key) == key


@pytest.mark.parametrize("key", [
    "",
    "/etc/passwd",
    "scans/../etc/passwd",
    "scans/scn/../../x",
    "../scans/x/y",
    "notscans/x/y",
    "scans/x",                 # too few segments
    "scans//y",                # empty segment
    "scans/scn/..",            # traversal tail
    "scans\\scn\\input",       # backslash
    "scans/scn/input/\x00",    # null byte
])
def test_bad_keys_rejected(key):
    with pytest.raises(StorageKeyError):
        validate_key(key)
