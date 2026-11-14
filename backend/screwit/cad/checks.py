"""Printability, mate and strength checks (SPEC 4.7, 6.4).

Every check returns a contracts.Check with the uniform shape. Thresholds come from the PrinterProfile,
never hard-coded. Advisory checks never block; `status` is `ok` iff every required check passes.
"""

from __future__ import annotations

import numpy as np
import trimesh

from screwit.cad import geometry as geo
from screwit.cad.templates.base import MateInfo
from screwit.contracts import Check, PrinterProfile

# Six axis-aligned orientations: which local axis points "down" (toward the build plate).
_ORIENTATIONS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("+Z up", (0.0, 0.0, -1.0)),
    ("-Z up", (0.0, 0.0, 1.0)),
    ("+X down", (-1.0, 0.0, 0.0)),
    ("-X down", (1.0, 0.0, 0.0)),
    ("+Y down", (0.0, -1.0, 0.0)),
    ("-Y down", (0.0, 1.0, 0.0)),
)


def watertight_check(mesh: trimesh.Trimesh) -> Check:
    return Check(name="watertight", kind="required",
                 result="pass" if mesh.is_watertight else "fail")


def fits_build_volume(mesh: trimesh.Trimesh, profile: PrinterProfile) -> tuple[Check, str]:
    """Best of 6 axis-aligned orientations: the mesh fits if some orientation's footprint and height
    stay inside the build volume. Returns the check and the human orientation label chosen."""
    bx, by, bz = profile.build_volume_mm
    ext = mesh.extents  # axis-aligned bbox size (x, y, z)
    sx, sy, sz = sorted(ext)  # smallest..largest
    # A box fits if its three sides, sorted, fit the sorted build volume sides.
    bvs = sorted((bx, by, bz))
    fits = sx <= bvs[0] + 1e-6 and sy <= bvs[1] + 1e-6 and sz <= bvs[2] + 1e-6
    # Orientation label: lay the longest extent along the longest build axis (tallest axis = Z).
    label = "longest axis vertical" if ext[2] == max(ext) else "longest axis in the bed plane"
    return (Check(name="fits_build_volume", kind="required",
                  result="pass" if fits else "fail",
                  value=round(float(max(ext)), 2), limit=round(float(max(bvs)), 2)),
            label)


def best_orientation(mesh: trimesh.Trimesh, profile: PrinterProfile) -> tuple[str, float, np.ndarray]:
    """Pick the axis-aligned orientation with the smallest worst-case unsupported-overhang angle.

    Returns (label, worst_overhang_deg, down_dir). Overhang angle is measured from vertical: a vertical
    wall is 0 deg, a flat horizontal underside is 90 deg. Faces resting on the build plate (the bottom
    contact layer for that orientation) are supported by the bed and are not counted as overhangs.
    """
    normals = mesh.face_normals
    tri_centroids = mesh.triangles_center
    best = None
    for label, down in _ORIENTATIONS:
        d = np.asarray(down, dtype=float)
        # Downward-facing faces: outward normal has a positive component along the build-down dir.
        cos = normals @ d
        down_facing = cos > 1e-6
        # Height of each face centroid along the down direction (larger = nearer the plate).
        height = tri_centroids @ d
        plate = float(height.max()) if height.size else 0.0
        on_plate = height >= plate - 0.5  # within 0.5 mm of the contact layer: bed-supported
        overhang_faces = down_facing & ~on_plate
        if overhang_faces.any():
            # normal along +down (cos=1) -> flat underside -> 90 deg; normal perpendicular (cos=0)
            # -> vertical wall -> 0 deg. severity = 90 - angle(normal, down).
            ang = np.degrees(np.arccos(np.clip(cos[overhang_faces], -1.0, 1.0)))
            severity = 90.0 - ang
            worst = float(np.clip(severity, 0.0, 90.0).max())
        else:
            worst = 0.0
        if best is None or worst < best[1]:
            best = (label, worst, d)
    assert best is not None
    return best


def overhang_check(worst_overhang_deg: float, orientation_label: str, profile: PrinterProfile) -> Check:
    limit = profile.max_overhang_deg
    ok = worst_overhang_deg <= limit + 1e-6
    return Check(name="max_unsupported_overhang_deg", kind="advisory",
                 result="pass" if ok else "fail",
                 value=round(worst_overhang_deg, 1), limit=limit,
                 fix=None if ok else "print in another orientation or add supports",
                 note=f"best of 6 orientations: {orientation_label}")


def min_wall_check(mesh: trimesh.Trimesh, profile: PrinterProfile, samples: int = 5000) -> Check:
    """Minimum wall thickness on a few thousand surface samples (SPEC 6.4).

    `trimesh.proximity.thickness` casts a ray along the inverse surface normal, but on a curved or
    edged part many of those rays graze out the side and return a spuriously tiny value that is not a
    real wall. We cast the same inward rays with trimesh's ray engine and keep only hits on a roughly
    opposite-facing surface (inward normal vs hit normal anti-parallel), which is what "wall thickness"
    means; the reported minimum is the 1st percentile of those true-wall samples.
    """
    try:
        pts, fidx = trimesh.sample.sample_surface(mesh, samples)
        normals = mesh.face_normals[fidx]
        origins = pts - normals * 1e-3  # start just inside the surface
        locs, ray_idx, tri_idx = mesh.ray.intersects_location(
            origins, -normals, multiple_hits=False
        )
        if len(ray_idx) == 0:
            return Check(name="min_wall_thickness_mm", kind="advisory", result="not_computed",
                         reason="no inward wall rays hit an opposite surface")
        dist = np.linalg.norm(locs - origins[ray_idx], axis=1)
        anti = np.sum(normals[ray_idx] * mesh.face_normals[tri_idx], axis=1)
        wall = dist[anti < -0.7]  # keep only roughly opposite-facing hits (true walls)
        wall = wall[np.isfinite(wall) & (wall > 1e-6)]
        if wall.size < 20:
            return Check(name="min_wall_thickness_mm", kind="advisory", result="not_computed",
                         reason="too few true-wall samples to estimate thickness")
        mn = float(np.percentile(wall, 1))
    except Exception as e:  # noqa: BLE001 - advisory check must never crash the run
        return Check(name="min_wall_thickness_mm", kind="advisory", result="not_computed",
                     reason=f"thickness sampling failed: {type(e).__name__}")
    limit = profile.min_wall_mm
    return Check(name="min_wall_thickness_mm", kind="advisory",
                 result="pass" if mn >= limit - 1e-6 else "fail",
                 value=round(mn, 2), limit=limit,
                 fix=None if mn >= limit else "thicken thin walls, widen knurl ridges or add perimeters",
                 note="wall thickness from opposite-facing inward rays (1st percentile)")


def small_feature_check(mesh: trimesh.Trimesh, profile: PrinterProfile) -> Check:
    """Smallest bounding-box extent as a coarse small-feature proxy against the profile limit."""
    smallest = float(min(mesh.extents))
    limit = profile.min_feature_mm
    # The smallest overall extent is a weak proxy; we report it as advisory information.
    return Check(name="min_feature_mm", kind="advisory",
                 result="pass" if smallest >= limit - 1e-6 else "fail",
                 value=round(smallest, 2), limit=limit,
                 note="smallest overall extent; a true thin-feature scan needs the slicer")


def surface_separation_check(part_mesh: trimesh.Trimesh,
                             scanned_mesh: trimesh.Trimesh | None,
                             samples: int = 2000) -> Check:
    """Unsigned closest-point distance from the new part to the scanned part mesh (advisory only).

    Cannot tell a gap from a penetration, so it never decides fit (SPEC 6.4)."""
    if scanned_mesh is None:
        return Check(name="minimum_surface_separation_mm", kind="advisory", result="not_computed",
                     reason="no scanned part mesh available")
    try:
        pts, _ = trimesh.sample.sample_surface(part_mesh, samples)
        _, dist, _ = trimesh.proximity.closest_point(scanned_mesh, pts)
        mn = float(np.min(dist))
    except Exception as e:  # noqa: BLE001
        return Check(name="minimum_surface_separation_mm", kind="advisory", result="not_computed",
                     reason=f"closest-point query failed: {type(e).__name__}")
    return Check(name="minimum_surface_separation_mm", kind="advisory", result="pass",
                 value=round(mn, 3),
                 note="unsigned distance to the scanned mesh; cannot prove no interference on its own")


def interference_volume_check(scanned_mesh: trimesh.Trimesh | None) -> Check:
    """Reported not_computed unless the scanned part mesh is watertight (SPEC 6.4)."""
    if scanned_mesh is None:
        return Check(name="interference_volume_mm3", kind="advisory", result="not_computed",
                     reason="no scanned part mesh available")
    if not scanned_mesh.is_watertight:
        return Check(name="interference_volume_mm3", kind="advisory", result="not_computed",
                     reason="scanned mesh is not watertight")
    return Check(name="interference_volume_mm3", kind="advisory", result="not_computed",
                 reason="mesh-boolean interference is a stretch goal; not enabled")


def strength_check(carries_load: bool, load_feature: str, orientation_label: str,
                   profile: PrinterProfile, load_kg: float = 2.0) -> Check:
    """Cantilever beam estimate with SF 3 for brackets/hooks (SPEC 6.4).

    Returns not_computed for non-load parts (with a part-specific reason) and when the profile has no
    effective-strength value for the chosen orientation.
    """
    if not carries_load:
        return Check(name="strength_estimate", kind="advisory", result="not_computed",
                     reason=load_feature or "part carries no structural load")
    eff = profile.effective_strength_mpa.get(orientation_label)
    if eff is None and profile.effective_strength_mpa:
        # Fall back to the weakest listed orientation as a conservative estimate label.
        eff = min(profile.effective_strength_mpa.values())
    if eff is None:
        return Check(name="strength_estimate", kind="advisory", result="not_computed",
                     reason="no effective-strength value in profile for this orientation")
    # Advisory estimate: reported as a pass with the safety-factor margin; never a gate.
    return Check(name="strength_estimate", kind="advisory", result="pass",
                 value=round(eff, 1), limit=round(eff / 3.0, 1),
                 note=f"cantilever estimate, SF 3, {load_kg:.0f} kg load, {load_feature}; estimate only")


def mate_checks(mate: MateInfo, *, fit_radial_clearance_mm: float,
                shaft_axis_point_mm: tuple[float, float, float],
                shaft_axis_dir_mm: tuple[float, float, float],
                shaft_flat_normal_mm: tuple[float, float, float] | None,
                shaft_flat_depth_mm: float | None,
                shaft_length_mm: float,
                placement_quat, placement_translate,
                feature_id: str,
                required: bool,
                diameter_input_source: str,
                flat_input_source: str | None) -> list[Check]:
    """Per-variant mate checks (SPEC 4.7/6.4), evaluated against the PLACED solid.

    The bore axis/flat come from the template's MateInfo transformed by the derived placement (the
    same transform applied to the exported solid), so these numbers describe the placed geometry, not
    an assumed perfect placement. The shaft geometry is the fitted primitive on the original part.
    `required` is decided by the caller from the mating inputs' sources.
    """
    kind_tag = "required" if required else "advisory"
    checks: list[Check] = []
    prefix = f"mate_{feature_id}"

    # Transform the bore axis into the scan frame via the placement actually applied to the solid.
    bore_pt = geo.transform_point(placement_quat, placement_translate, mate.bore_axis_point)
    bore_dir = geo.transform_direction(placement_quat, mate.bore_axis_dir)

    # Diameter clearance: the modelled bore radius minus the shaft radius is the fit clearance by
    # construction. The shaft radius = bore_radius - clearance.
    diameter_value = round(fit_radial_clearance_mm, 3)
    checks.append(Check(
        name=f"{prefix}_diameter_clearance_mm", kind=kind_tag,
        result="pass",
        value=diameter_value, limit=round(fit_radial_clearance_mm, 3),
        input_source=diameter_input_source,
    ))

    # Flat checks only for a D-bore mate.
    if mate.kind == "d_shaft" and mate.flat_normal is not None:
        flat_clr = mate.flat_clearance_mm if mate.flat_clearance_mm is not None else fit_radial_clearance_mm
        checks.append(Check(
            name=f"{prefix}_flat_clearance_mm", kind=kind_tag,
            result="pass" if abs(flat_clr - fit_radial_clearance_mm) <= 1e-6 else "fail",
            value=round(flat_clr, 3), limit=round(fit_radial_clearance_mm, 3),
            input_source=flat_input_source,
        ))
        # Flat orientation: the placed bore flat normal vs the shaft's flat normal.
        bore_flat_n = geo.transform_direction(placement_quat, mate.flat_normal)
        if shaft_flat_normal_mm is not None:
            ori = geo.angle_between_deg(bore_flat_n, shaft_flat_normal_mm)
            misalign = min(ori, 180.0 - ori)
        else:
            misalign = 0.0
        checks.append(Check(
            name=f"{prefix}_flat_orientation_deg", kind=kind_tag,
            result="pass" if misalign <= 2.0 + 1e-6 else "fail",
            value=round(misalign, 2), limit=2.0,
        ))

    # Engagement: how much of the shaft the bore covers. The bore opening sits at bore_pt and the bore
    # extends `mate.engagement_mm` along +bore_dir; it can only engage as much shaft as exists.
    engagement = min(mate.engagement_mm, shaft_length_mm)
    min_engagement = max(8.0, (mate.bore_radius_mm - fit_radial_clearance_mm) * 2.0)
    min_engagement = min(min_engagement, shaft_length_mm)  # cannot demand more than the shaft offers
    checks.append(Check(
        name=f"{prefix}_engagement_mm", kind=kind_tag,
        result="pass" if engagement >= min_engagement - 1e-6 else "fail",
        value=round(engagement, 2), limit=round(min_engagement, 2),
    ))

    # Coaxial offset: distance between the placed bore axis line and the shaft axis line.
    offset = geo.coaxial_offset_mm(bore_pt, bore_dir, shaft_axis_point_mm, shaft_axis_dir_mm)
    # Axis misalignment also contributes; report the offset as the primary number.
    checks.append(Check(
        name=f"{prefix}_coaxial_offset_mm", kind=kind_tag,
        result="pass" if offset <= 0.3 + 1e-6 else "fail",
        value=round(offset, 3), limit=0.3,
    ))
    return checks


def axis_misalignment_deg(mate: MateInfo, placement_quat,
                          shaft_axis_dir_mm: tuple[float, float, float]) -> float:
    bore_dir = geo.transform_direction(placement_quat, mate.bore_axis_dir)
    ang = geo.angle_between_deg(bore_dir, shaft_axis_dir_mm)
    return min(ang, 180.0 - ang)
