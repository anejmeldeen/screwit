"""CAD service: validate -> resolve mates -> build -> place -> check -> export (SPEC 4.6/4.7/6.4).

Implements the CadService protocol (screwit.interfaces). Everything is scan-frame millimetres.
STL is written in millimetres (print), STEP for editing, model.py is the reproducible source, and a
headless CPU preview PNG. print_settings carries a volume-formula time/material estimate labelled as
such. codegen mode (a new shape) runs Claude-written CadQuery only inside the X7 sandbox container and
holds the result to the same printability checks.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import trimesh
from cadquery import exporters

from screwit.cad import checks as chk
from screwit.cad.preview import render_preview
from screwit.cad.source import render_model_source
from screwit.cad.templates import get_template
from screwit.cad.templates.base import MateInfo
from screwit.contracts import (
    CadFiles,
    CadRequest,
    CadResult,
    Check,
    DShaftFeature,
    InputRevisions,
    Mate,
    Part,
    PrinterProfile,
    PrintSettings,
)
from screwit.interfaces import Storage

# feature type -> which template kinds may mate with it (SPEC 4.6: a knob mates d_shaft or cylinder).
_COMPATIBLE: dict[str, set[str]] = {
    "cylinder": {"cylinder"},        # a round bore accepts a plain cylinder shaft
    "d_shaft": {"cylinder", "d_shaft"},  # a d_shaft accepts a round OR a D bore
}

# Measurements that define a fit, per feature type (data, keyed by geometry, not by object). Length
# and depth set engagement, not clearance. A type not listed here gates on all its measurements.
_FIT_FIELDS: dict[str, list[str]] = {
    "cylinder": ["radius"],
    "bore": ["radius"],
    "d_shaft": ["radius", "flat_depth"],
    "hole": ["radius"],
}


class CadError(Exception):
    """A request-level failure that yields status 'error' (not a stale-revision 409)."""


class StaleRequest(Exception):
    """Revisions in the request do not match the current parts (API returns 409)."""


def load_profile(name: str, profiles_dir: Path) -> PrinterProfile:
    path = profiles_dir / f"{name}.json"
    if not path.exists():
        raise CadError(f"unknown printer_profile {name!r}")
    return PrinterProfile.model_validate_json(path.read_text())


class CadRunner:
    """CadService implementation."""

    def __init__(self, profiles_dir: Path) -> None:
        self.profiles_dir = Path(profiles_dir)

    # -- CadService protocol ------------------------------------------------

    def templates(self) -> dict[str, dict]:
        from screwit.cad.templates import TEMPLATES

        return {name: t.json_schema() for name, t in TEMPLATES.items()}

    def run(self, req: CadRequest, parts: dict[str, Part], storage: Storage) -> CadResult:
        input_revisions = InputRevisions(
            reconstruction_revision=req.reconstruction_revision,
            part_measurement_revisions=dict(req.part_measurement_revisions),
        )

        if req.mode == "codegen":
            try:
                self._check_revisions(req, parts)
            except StaleRequest:
                raise
            try:
                return self._run_codegen(req, parts, storage, input_revisions)
            except CadError as e:
                return CadResult(
                    cad_id=req.cad_id, status="error", input_revisions=input_revisions, stale=False,
                    files=None, checks=[], print_settings=None,
                    notes=f"New-shape design failed: {e}", error=str(e),
                )

        try:
            self._check_revisions(req, parts)
        except StaleRequest:
            raise
        try:
            return self._run_template(req, parts, storage, input_revisions)
        except CadError as e:
            return CadResult(
                cad_id=req.cad_id, status="error", input_revisions=input_revisions, stale=False,
                files=None, checks=[], print_settings=None,
                notes=f"CAD build failed: {e}", error=str(e),
            )

    # -- internals ----------------------------------------------------------

    def _run_codegen(self, req: CadRequest, parts: dict[str, Part], storage: Storage,
                     input_revisions: InputRevisions) -> CadResult:
        """New shape: Claude writes CadQuery code that runs only in the X7 sandbox container; the
        winning STL is then held to the same printability checks as a template part.

        The generated code exposes no bore geometry, so mating fits cannot be checked analytically.
        Instead a required check passes only when every mating dimension came from calipers or the
        user (AGENTS.md: mating dims need caliper or user_confirmed), and the scanned-part surface
        separation is reported as advisory.
        """
        import shutil

        from screwit import codegen

        profile = load_profile(req.printer_profile, self.profiles_dir)
        mate_req: Mate | None = req.mates_with[0] if req.mates_with else None
        part = parts.get(mate_req.part_id) if mate_req is not None else None
        base_key = f"scans/{req.scan_id}/cad/{req.cad_id}"
        try:
            files_dict, rationale, attempts = codegen.design(
                req.codegen_prompt or "", part, dict(req.params), profile, storage,
                out_prefix=base_key)
        except Exception as e:  # noqa: BLE001 - LLM/docker failures become a clean error result
            raise CadError(f"{type(e).__name__}: {e}") from None
        if files_dict is None:
            raise CadError(rationale or "no candidate design built")

        scratch = Path(tempfile.mkdtemp(prefix="cad_codegen_"))
        try:
            stl_local = storage.local_path(files_dict["stl"], scratch)
            mesh = trimesh.load(str(stl_local), force="mesh")
            placement = req.placement
            if placement is None:
                from screwit.contracts import Placement

                placement = Placement(translate_mm=(0.0, 0.0, 0.0), rotation_quat=(0.0, 0.0, 0.0, 1.0))
            scanned_mesh = (self._load_scanned_mesh(req.scan_id, mate_req, storage, scratch)
                            if mate_req else None)

            checks: list[Check] = [chk.watertight_check(mesh)]
            fits, _ = chk.fits_build_volume(mesh, profile)
            checks.append(fits)
            if mate_req is not None:
                checks.append(self._codegen_mate_sources_check(req, parts))
            label, worst_overhang, _down = chk.best_orientation(mesh, profile)
            checks.append(chk.min_wall_check(mesh, profile))
            checks.append(chk.overhang_check(worst_overhang, label, profile))
            checks.append(chk.small_feature_check(mesh, profile))
            if scanned_mesh is not None:
                checks.append(chk.surface_separation_check(mesh, scanned_mesh))
            status = ("ok" if all(c.result == "pass" for c in checks if c.kind == "required")
                      else "failed_checks")
            print_settings = self._print_settings(mesh, None, profile, label)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

        tried = len(attempts)
        notes = f"New shape designed in a sandbox ({tried} candidates tried). {rationale}".strip()
        if status != "ok":
            notes += " Some required checks did not pass; confirm the dimensions before printing."
        return CadResult(
            cad_id=req.cad_id, status=status, input_revisions=input_revisions, stale=False,
            files=CadFiles(**files_dict), placement=placement, checks=checks,
            print_settings=print_settings, notes=notes, error=None,
        )

    @staticmethod
    def _codegen_mate_sources_check(req: CadRequest, parts: dict[str, Part]) -> Check:
        """The mated feature's fit-defining measurements must come from calipers or the user."""
        from screwit.contracts import Measurement

        mate = req.mates_with[0]
        part = parts.get(mate.part_id)
        feat = part.feature(mate.feature_id) if part is not None else None
        if feat is None:
            return Check(name="mating_dimensions_confirmed", kind="required", result="fail",
                         reason=f"feature {mate.feature_id} not found on part {mate.part_id}")
        names = _FIT_FIELDS.get(feat.type) or [
            n for n in type(feat).model_fields if isinstance(getattr(feat, n), Measurement)]
        unconfirmed = []
        for n in names:
            m = getattr(feat, n, None)
            if isinstance(m, Measurement) and m.caliper_mm is None and not m.user_confirmed:
                unconfirmed.append(n)
        if unconfirmed:
            return Check(name="mating_dimensions_confirmed", kind="required", result="fail",
                         reason=(f"{mate.feature_id} {', '.join(unconfirmed)} not measured with "
                                 f"calipers or confirmed"),
                         fix="Measure these with calipers (or confirm them) and design again.")
        return Check(name="mating_dimensions_confirmed", kind="required", result="pass",
                     note=f"{mate.feature_id} {', '.join(names)} from calipers or the user")

    def _check_revisions(self, req: CadRequest, parts: dict[str, Part]) -> None:
        for pid, rev in req.part_measurement_revisions.items():
            part = parts.get(pid)
            if part is None:
                raise StaleRequest(f"part {pid} is not present")
            if part.reconstruction_revision != req.reconstruction_revision:
                raise StaleRequest(
                    f"reconstruction_revision {req.reconstruction_revision} != part {pid} "
                    f"{part.reconstruction_revision}"
                )
            if part.measurement_revision != rev:
                raise StaleRequest(
                    f"part {pid} measurement_revision {part.measurement_revision} != requested {rev}"
                )

    def _run_template(self, req: CadRequest, parts: dict[str, Part], storage: Storage,
                      input_revisions: InputRevisions) -> CadResult:
        profile = load_profile(req.printer_profile, self.profiles_dir)
        template = get_template(req.template or "")

        # Resolve the mate (at most one for the current templates) before building.
        mate_req: Mate | None = req.mates_with[0] if req.mates_with else None
        mate_radial = None
        if mate_req is not None:
            self._validate_mate(req, template, mate_req, profile, parts)
            mate_radial = mate_req.radial_clearance_mm

        # Build the solid in the template-local frame (profile drives printable knurl defaults/limits).
        resolved = template.validate_params(req.params)
        try:
            build = template.build(resolved, mate_radial_clearance_mm=mate_radial, profile=profile)
        except ValueError as e:
            raise CadError(str(e)) from None
        local_solid = build.solid

        scratch = Path(tempfile.mkdtemp(prefix="cad_"))
        scanned_mesh = (
            self._load_scanned_mesh(req.scan_id, mate_req, storage, scratch) if mate_req else None
        )

        # Derive the scan-frame placement from the mated feature (never trust a requested placement
        # for a mating part); a non-mating part uses the requested placement or identity.
        placement, rot_matrix = self._derive_placement(
            req, build, mate_req, parts, scanned_mesh
        )

        # Place the solid in the scan frame, then export (viewer overlays STL/STEP directly).
        placed_solid = self._place_solid(local_solid, placement)

        stl_tmp = scratch / "model.stl"
        step_tmp = scratch / "model.step"
        exporters.export(placed_solid, str(stl_tmp))
        exporters.export(placed_solid, str(step_tmp))
        mesh = trimesh.load(str(stl_tmp), force="mesh")

        checks, orient_label = self._run_checks(
            req, template, build, mesh, profile, mate_req, parts, scanned_mesh, placement, rot_matrix
        )
        status = "ok" if all(c.result == "pass" for c in checks if c.kind == "required") else "failed_checks"

        # Export files under scans/<id>/cad/<cad_id>/.
        base_key = f"scans/{req.scan_id}/cad/{req.cad_id}"
        storage.put_file(f"{base_key}/model.stl", stl_tmp, "model/stl")
        storage.put_file(f"{base_key}/model.step", step_tmp, "application/step")

        source_text = render_model_source(req.template or "", resolved, mate_radial, placement)
        storage.put_bytes(f"{base_key}/model.py", source_text.encode("utf-8"), "text/x-python")

        preview_tmp = scratch / "preview.png"
        render_preview(mesh, preview_tmp)
        storage.put_file(f"{base_key}/preview.png", preview_tmp, "image/png")

        print_settings = self._print_settings(mesh, build, profile, orient_label)

        files = CadFiles(
            stl=f"{base_key}/model.stl",
            step=f"{base_key}/model.step",
            source=f"{base_key}/model.py",
            preview=f"{base_key}/preview.png",
        )
        notes = self._notes(req, mate_req, profile, status)
        return CadResult(
            cad_id=req.cad_id, status=status, input_revisions=input_revisions, stale=False,
            files=files, placement=placement, checks=checks, print_settings=print_settings,
            notes=notes, error=None,
        )

    def _derive_placement(self, req: CadRequest, build, mate_req: Mate | None,
                          parts: dict[str, Part], scanned_mesh):
        """Placement in the scan frame. For a mate, derive it from the feature (SPEC defect 1);
        otherwise use the requested placement, or identity if none was given.

        Returns (Placement, rotation_matrix | None). The rotation matrix is kept for the placed-solid
        mate checks.
        """
        import numpy as np

        from screwit.cad.placement import derive_mate_placement, shaft_free_end
        from screwit.contracts import Placement

        if mate_req is not None and build.mate is not None:
            part = parts[mate_req.part_id]
            feat = part.feature(mate_req.feature_id)
            assert feat is not None
            axis_point = getattr(feat, "axis_point_mm", (0.0, 0.0, 0.0))
            axis_dir = getattr(feat, "axis_dir", (0.0, 0.0, 1.0))
            flat_normal = feat.flat_normal if isinstance(feat, DShaftFeature) else None
            # Shaft free end: extreme of the real part mesh along +axis_dir, else from fitted length.
            shaft_len = getattr(feat, "length", None)
            fallback_len = shaft_len.value_mm if shaft_len is not None else build.mate.engagement_mm
            free_end = shaft_free_end(axis_point, axis_dir, scanned_mesh, fallback_len)
            # Engagement is the lesser of the bore depth and the available shaft length.
            engagement = min(build.mate.engagement_mm, fallback_len)
            quat, translate, rmat = derive_mate_placement(
                feature_axis_point=axis_point,
                feature_axis_dir=axis_dir,
                shaft_free_end_mm=free_end,
                engagement_mm=engagement,
                has_flat=build.mate.kind == "d_shaft" and build.mate.flat_normal is not None,
                feature_flat_normal=flat_normal,
                local_flat_inward=build.mate.flat_normal or (0.0, -1.0, 0.0),
            )
            return Placement(translate_mm=translate, rotation_quat=quat), rmat

        if req.placement is not None:
            from screwit.cad.geometry import quat_to_matrix

            return req.placement, quat_to_matrix(req.placement.rotation_quat)
        # No mate and no requested placement: identity.
        return Placement(translate_mm=(0.0, 0.0, 0.0), rotation_quat=(0.0, 0.0, 0.0, 1.0)), np.eye(3)

    def _place_solid(self, solid, placement):
        """Apply the placement (rotation then translation) to a CadQuery solid in the scan frame."""
        import math

        x, y, z, w = placement.rotation_quat
        angle = 2 * math.degrees(math.acos(max(-1.0, min(1.0, w))))
        axis_len = math.sqrt(x * x + y * y + z * z)
        placed = solid
        if axis_len > 1e-9 and abs(angle) > 1e-9:
            axis = (x / axis_len, y / axis_len, z / axis_len)
            placed = placed.rotate((0, 0, 0), axis, angle)
        tx, ty, tz = placement.translate_mm
        placed = placed.translate((tx, ty, tz))
        return placed

    def _validate_mate(self, req: CadRequest, template, mate_req: Mate, profile: PrinterProfile,
                       parts: dict[str, Part]) -> None:
        if not getattr(template, "mates", False):
            raise CadError(f"template {template.name!r} has no mating feature")
        part = parts.get(mate_req.part_id)
        if part is None:
            raise CadError(f"mate part {mate_req.part_id} not found")
        feat = part.feature(mate_req.feature_id)
        if feat is None:
            raise CadError(f"feature {mate_req.feature_id} not found on part {mate_req.part_id}")
        if feat.type not in _COMPATIBLE:
            raise CadError(
                f"feature {mate_req.feature_id} is a {feat.type}; a {template.name} mates only a "
                f"cylinder or d_shaft"
            )
        # The clearance must equal the profile's for the fit type; the agent does not choose it.
        profile_clr = profile.radial_clearance_mm.get(mate_req.fit)
        if profile_clr is None:
            raise CadError(f"profile has no clearance for fit {mate_req.fit!r}")
        if abs(mate_req.radial_clearance_mm - profile_clr) > 1e-6:
            raise CadError(
                f"radial_clearance_mm {mate_req.radial_clearance_mm} != profile {mate_req.fit} "
                f"clearance {profile_clr}; the clearance is copied from the profile, not chosen"
            )

    def _run_checks(self, req: CadRequest, template, build, mesh: trimesh.Trimesh,
                    profile: PrinterProfile, mate_req: Mate | None, parts: dict[str, Part],
                    scanned_mesh: trimesh.Trimesh | None, placement, rot_matrix) -> tuple[list[Check], str]:
        checks: list[Check] = [chk.watertight_check(mesh)]
        fits, _fit_label = chk.fits_build_volume(mesh, profile)
        checks.append(fits)

        # Mate checks (required only when every mating input source is caliper/user_confirmed),
        # computed from the PLACED solid against the feature.
        if mate_req is not None and build.mate is not None:
            checks.extend(self._mate_checks(req, build.mate, mate_req, parts, placement, rot_matrix))

        # Advisory printability, computed after choosing the best orientation.
        label, worst_overhang, _down = chk.best_orientation(mesh, profile)
        checks.append(chk.min_wall_check(mesh, profile))
        checks.append(chk.overhang_check(worst_overhang, label, profile))
        checks.append(chk.small_feature_check(mesh, profile))
        # Separation: the placed part mesh against the real scanned part mesh.
        checks.append(chk.surface_separation_check(mesh, scanned_mesh))
        checks.append(chk.interference_volume_check(scanned_mesh))
        checks.append(chk.strength_check(build.carries_load, build.load_feature, label, profile))
        return checks, label

    def _mate_checks(self, req: CadRequest, mate: MateInfo, mate_req: Mate,
                     parts: dict[str, Part], placement, rot_matrix) -> list[Check]:
        part = parts[mate_req.part_id]
        feat = part.feature(mate_req.feature_id)
        assert feat is not None

        # Mating inputs for this template: diameter always, flat for a D-bore.
        diameter_source = req.param_sources.get("shaft_diameter", "scan")
        flat_source = req.param_sources.get("shaft_flat_depth")

        is_d_bore = mate.kind == "d_shaft" and mate.flat_normal is not None
        sources = [diameter_source]
        if is_d_bore and flat_source is not None:
            sources.append(flat_source)
        required = all(s in ("caliper", "user_confirmed") for s in sources)

        shaft_axis_point = getattr(feat, "axis_point_mm", (0.0, 0.0, 0.0))
        shaft_axis_dir = getattr(feat, "axis_dir", (0.0, 0.0, 1.0))
        shaft_flat_normal = feat.flat_normal if isinstance(feat, DShaftFeature) else None
        shaft_flat_depth = feat.flat_depth.value_mm if isinstance(feat, DShaftFeature) else None
        shaft_length = feat.length.value_mm if hasattr(feat, "length") else mate.engagement_mm

        return chk.mate_checks(
            mate,
            fit_radial_clearance_mm=mate_req.radial_clearance_mm,
            shaft_axis_point_mm=shaft_axis_point,
            shaft_axis_dir_mm=shaft_axis_dir,
            shaft_flat_normal_mm=shaft_flat_normal,
            shaft_flat_depth_mm=shaft_flat_depth,
            shaft_length_mm=shaft_length,
            placement_quat=placement.rotation_quat,
            placement_translate=placement.translate_mm,
            feature_id=mate_req.feature_id,
            required=required,
            diameter_input_source=diameter_source,
            flat_input_source=flat_source,
        )

    def _load_scanned_mesh(self, scan_id: str, mate_req: Mate, storage: Storage,
                           scratch: Path) -> trimesh.Trimesh | None:
        """Load the scanned part's mesh for the advisory surface-separation check, if present.

        The part GLB lives at scans/<scan_id>/parts/<part_id>.glb (SPEC 3.2). Returns None if it is
        not in storage (fixtures often omit it), so the separation check reports not_computed.
        """
        key = f"scans/{scan_id}/parts/{mate_req.part_id}.glb"
        if storage.head(key) is None:
            return None
        try:
            local = storage.local_path(key, scratch)
            loaded = trimesh.load(str(local), force="mesh")
            if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) > 0:
                return loaded
        except Exception:  # noqa: BLE001 - advisory input; missing/broken mesh just means not_computed
            return None
        return None

    def _print_settings(self, mesh: trimesh.Trimesh, build, profile: PrinterProfile,
                        orient_label: str) -> PrintSettings:
        # Volume-formula estimate (labelled volume_formula, not a slicer).
        volume_mm3 = float(mesh.volume) if mesh.is_volume else float(abs(mesh.volume))
        infill = profile.infill_pct
        # Rough solid-equivalent: perimeters/top/bottom are near-solid, infill fills the core.
        shell_fraction = 0.35
        effective_fraction = shell_fraction + (1 - shell_fraction) * (infill / 100.0)
        material_mm3 = volume_mm3 * effective_fraction
        mass_g = material_mm3 / 1000.0 * profile.density_g_cm3
        extrude_time_s = material_mm3 / max(profile.volumetric_speed_mm3_s, 1e-6)
        est_time_min = max(1, int(round(extrude_time_s / 60.0 * 1.6)))  # 1.6x for travel/non-print moves
        orientation = orient_label or "best of 6 orientations"
        return PrintSettings(
            material=profile.material,
            layer_height_mm=profile.layer_height_mm,
            infill_pct=infill,
            orientation=orientation,
            supports=False,
            est_time_min=est_time_min,
            est_material_g=max(1, int(round(mass_g))),
            estimate_method="volume_formula",
        )

    def _notes(self, req: CadRequest, mate_req: Mate | None, profile: PrinterProfile,
               status: str) -> str:
        if mate_req is None:
            return "No mating feature; printability checks only."
        clr = mate_req.radial_clearance_mm
        base = (
            f"Bore has {clr} mm radial clearance from the coupon-calibrated profile; "
            f"test-fit and scale 101% if tight. The fit evidence is the caliper value plus the "
            f"tolerance coupon."
        )
        if status != "ok":
            base += " Some required checks did not pass; confirm the mating dimensions before printing."
        return base
