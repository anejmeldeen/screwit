"""The AST pre-check rejects disallowed imports and dunder access (convenience filter, not boundary)."""

from __future__ import annotations

from screwit.codegen.astcheck import check_source


def test_valid_program_passes():
    src = (
        "import cadquery as cq\n"
        "import math\n"
        "def build(params):\n"
        "    return cq.Workplane('XY').box(10, 10, 10)\n"
    )
    assert check_source(src).ok


def test_socket_import_rejected():
    src = "import socket\ndef build(params):\n    return socket\n"
    res = check_source(src)
    assert not res.ok
    assert "socket" in res.reason


def test_os_from_import_rejected():
    src = "from os import system\ndef build(params):\n    return 1\n"
    res = check_source(src)
    assert not res.ok


def test_dunder_attribute_rejected():
    # The classic sandbox-escape shape: reach builtins through an object's __class__ chain.
    src = "def build(params):\n    return ().__class__.__bases__\n"
    res = check_source(src)
    assert not res.ok
    assert "dunder" in res.reason


def test_dunder_name_rejected():
    src = "def build(params):\n    return __import__('os')\n"
    res = check_source(src)
    assert not res.ok


def test_syntax_error_is_reported_not_raised():
    res = check_source("def build(:\n")
    assert not res.ok
    assert "parse" in res.reason
