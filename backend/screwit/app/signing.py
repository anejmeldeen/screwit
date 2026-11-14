"""Turn storage keys in API responses into signed URLs (SPEC 5: "Scan assets are returned as short-lived
signed URLs"). Every route that returns a contract carrying asset keys passes it through here, so the
browser never sees a bare key it cannot fetch.
"""

from __future__ import annotations

from screwit.contracts import CadResult, DemoScan, Part, Reconstruction

_RECON_FIELDS = ("splat_uri", "mesh_uri", "cameras_uri", "splat_mobile_uri", "mesh_mobile_uri")


def _url(storage, value: str | None) -> str | None:
    if value is None or value.startswith(("/", "http://", "https://")):
        return value  # already a URL (signed earlier, or absent)
    return storage.signed_url(value)


def reconstruction(storage, recon: Reconstruction) -> Reconstruction:
    update: dict = {f: _url(storage, getattr(recon, f)) for f in _RECON_FIELDS if getattr(recon, f)}
    update["keyframes"] = [_url(storage, k) for k in recon.keyframes]
    return recon.model_copy(update=update)


def part(storage, p: Part) -> Part:
    return p.model_copy(update={"mesh_uri": _url(storage, p.mesh_uri),
                                "thumbnail_uri": _url(storage, p.thumbnail_uri)})


def parts(storage, ps: list[Part]) -> list[Part]:
    return [part(storage, p) for p in ps]


def cad_result(storage, r: CadResult) -> CadResult:
    if r.files is None:
        return r
    files = r.files.model_copy(update={
        "stl": _url(storage, r.files.stl), "step": _url(storage, r.files.step),
        "source": _url(storage, r.files.source), "preview": _url(storage, r.files.preview)})
    return r.model_copy(update={"files": files})


def demo_scan(storage, d: DemoScan) -> DemoScan:
    return d.model_copy(update={"thumbnail_uri": _url(storage, d.thumbnail_uri)})
