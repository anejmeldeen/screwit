"""The running API must listen on 127.0.0.1 only, and refuse a non-loopback client (SPEC 3.3, AGENTS.md).

Verified two ways:
- the listening socket for the server's port is bound to 127.0.0.1 (read from `ss`, falling back to
  /proc/net/tcp), never 0.0.0.0;
- a request whose source address is a non-loopback local address is refused, when such an address can be
  bound on this host; otherwise that half skips with a reason.
"""

from __future__ import annotations

import ipaddress
import socket
import struct
import subprocess

import httpx
import pytest

from .conftest import Server


def _listening_addrs_for_port(port: int) -> list[str]:
    """Local addresses with a LISTEN socket on `port`, by `ss`, falling back to /proc/net/tcp."""
    addrs = _listen_addrs_via_ss(port)
    if addrs is not None:
        return addrs
    return _listen_addrs_via_proc(port)


def _listen_addrs_via_ss(port: int) -> list[str] | None:
    try:
        out = subprocess.run(["ss", "-ltnH"], capture_output=True, text=True, timeout=10).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    found: list[str] = []
    for line in out.splitlines():
        cols = line.split()
        if len(cols) < 4:
            continue
        local = cols[3]  # e.g. 127.0.0.1:8123 or [::1]:8123 or *:8123
        host, _, p = local.rpartition(":")
        if p != str(port):
            continue
        found.append(host.strip("[]"))
    return found


def _listen_addrs_via_proc(port: int) -> list[str]:
    found: list[str] = []
    for path, fmt in (("/proc/net/tcp", 4), ("/proc/net/tcp6", 16)):
        try:
            lines = open(path).read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            cols = line.split()
            if len(cols) < 4 or cols[3] != "0A":  # 0A = LISTEN
                continue
            hexaddr, hexport = cols[1].split(":")
            if int(hexport, 16) != port:
                continue
            raw = bytes.fromhex(hexaddr)
            if fmt == 4:
                ip = ipaddress.ip_address(struct.unpack("<I", raw)[0]).compressed
            else:
                words = struct.unpack("<4I", raw)
                ip = ipaddress.ip_address(b"".join(struct.pack(">I", w) for w in words)).compressed
            found.append(ip)
    return found


def test_listener_is_loopback_only(server: Server):
    addrs = _listening_addrs_for_port(server.port)
    assert addrs, f"no LISTEN socket found on port {server.port}"
    for a in addrs:
        assert a not in ("0.0.0.0", "::", "*"), f"server listening on wildcard address {a!r} (must be loopback)"
        assert ipaddress.ip_address(a).is_loopback, f"server listening on non-loopback {a!r}"


def _another_local_ipv4() -> str | None:
    """A non-loopback IPv4 address that is bindable on this host, or None if there is none."""
    for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        ip = info[4][0]
        addr = ipaddress.ip_address(ip)
        if not addr.is_loopback:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((ip, 0))
                    return ip
                except OSError:
                    continue
    return None


def test_non_loopback_client_refused(server: Server):
    src = _another_local_ipv4()
    if src is None:
        pytest.skip("no bindable non-loopback local address on this host")
    # Connecting from `src` to 127.0.0.1 would arrive on a loopback destination but a non-loopback
    # source; the loopback gate checks the client (source) address. A connection to the server's
    # loopback socket from a non-loopback source is itself typically unroutable, so treat either a
    # connect failure OR a 403 as "refused". A 2xx from a non-loopback source is the failure.
    transport = httpx.HTTPTransport(local_address=src)
    with httpx.Client(transport=transport, timeout=5.0) as c:
        try:
            r = c.get(f"{server.base_url}/api/demo-scans")
        except httpx.HTTPError:
            return  # connection refused / unreachable: the non-loopback source cannot reach it
        assert r.status_code == 403, (
            f"non-loopback client got {r.status_code}, expected 403 from the loopback gate"
        )
