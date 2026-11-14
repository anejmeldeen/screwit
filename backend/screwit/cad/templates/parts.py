"""The parametric template library (SPEC 6.4, P5 workload 3).

Eight templates: knob, L-bracket, spacer/washer, pipe/hose adapter, cable clip, hook, cap/plug,
enclosure box. Each returns a solid in scan-frame millimetres; mating templates (knob, cap/plug,
adapter) attach a MateInfo describing the bore in their local frame.

Local-frame convention (see base.py): a mating bore is centred on the origin, its axis along +Z,
open on the -Z face that sits against the original part; a D-bore's flat is on the +X side, inward
normal -X.
"""

from __future__ import annotations

import math

import cadquery as cq

from screwit.cad.templates.base import BuildResult, MateInfo, ParamSpec, Template

_Z_AXIS = (0.0, 0.0, 1.0)


def _bore_radius(shaft_diameter: float, mate_radial_clearance_mm: float | None) -> float:
    clr = 0.0 if mate_radial_clearance_mm is None else mate_radial_clearance_mm
    return shaft_diameter / 2.0 + clr


def _cut_d_bore(solid: cq.Workplane, *, radius: float, depth: float, flat_depth: float,
                flat_clearance: float) -> cq.Workplane:
    """Cut a D-shaped bore from the -Z face upward along +Z.

    The round part of the bore has `radius`; one chord is cut back so the remaining flat-to-axis
    distance is (radius - flat_depth - flat_clearance), matching a shaft whose flat sits `flat_depth`
    in from its round surface, plus a modelled `flat_clearance` gap. The flat lies on the +Y side, so
    the bore's inward flat normal (from the flat toward the axis) is -Y. This local convention is
    chosen so the standard mate placement (Rx(90 deg), local +Y -> scan +Z) lines the bore flat up
    with a shaft whose flat_normal is +Z.
    """
    # Start with a cylindrical bore of the full radius, open on the -Z face.
    cut = (
        cq.Workplane("XY")
        .workplane(offset=-0.001)
        .circle(radius)
        .extrude(depth + 0.002)
    )
    # The flat on +Y: keep the bore material only on the -Y side of the plane y = flat_offset.
    flat_offset = radius - flat_depth - flat_clearance
    big = radius * 4 + 10
    keep = (
        cq.Workplane("XY")
        .workplane(offset=-0.001)
        .center(0, flat_offset - big / 2)
        .rect(big, big)
        .extrude(depth + 0.002)
    )
    d_cut = cut.intersect(keep)
    return solid.cut(d_cut)


class KnobTemplate(Template):
    name = "knob"
    mates = True
    # knurl_count / knurl_depth default to 0 here; the real printable defaults are derived from the
    # printer profile in resolve_knurl() (a fine knurl's ridges must be at least min_wall_mm wide).
    params = (
        ParamSpec("outer_diameter", 32.0, 8.0, 120.0, "grip outer diameter (mm)"),
        ParamSpec("height", 22.0, 6.0, 80.0, "overall height along the shaft axis (mm)"),
        ParamSpec("shaft_diameter", 6.0, 2.0, 25.0, "mating shaft diameter (mm)"),
        ParamSpec("shaft_flat_depth", 1.5, 0.0, 6.0, "D-shaft flat depth; 0 for a round shaft (mm)"),
        ParamSpec("bore_depth", 16.0, 4.0, 70.0, "how deep the bore goes into the knob (mm)"),
        ParamSpec("wall_mm", 3.0, 1.2, 12.0, "grip wall around the bore (mm)"),
        ParamSpec("knurl_count", 0.0, 0.0, 120.0, "number of knurl grooves; 0 = auto from profile"),
        ParamSpec("knurl_depth", 0.0, 0.0, 3.0, "knurl groove depth; 0 = auto from profile"),
    )

    def resolve_knurl(self, outer_diameter: float, count: float, depth: float,
                      profile=None) -> tuple[int, float]:
        """Return a printable (count, depth) for the knurl.

        A knurl is only printable when the land (flat ridge) left between two grooves is at least the
        profile's minimum wall thickness and each groove is at least one nozzle wide. When count/depth
        are 0 (auto) they are derived from the profile; when given they are validated and a bad combo
        raises ValueError so the range validation rejects an unprintable knurl.
        """
        min_wall = getattr(profile, "min_wall_mm", 1.2) if profile is not None else 1.2
        nozzle = getattr(profile, "nozzle_mm", 0.4) if profile is not None else 0.4
        circumference = math.pi * outer_diameter

        if count <= 0 or depth <= 0:
            # Auto: groove = 1.5 nozzle wide, land = 2 * min_wall; fit as many as the circumference
            # allows, and a shallow depth so the ridge roots stay solid.
            groove_w = 1.5 * nozzle
            land_w = 2.0 * min_wall
            n = int(circumference // (groove_w + land_w))
            n = max(0, min(n, 80))
            d = min(0.6, max(0.3, min_wall * 0.4))
            return n, d

        n = int(round(count))
        d = float(depth)
        if n == 0:
            return 0, 0.0
        groove_w = 1.5 * nozzle
        land_w = circumference / n - groove_w
        if land_w < min_wall - 1e-6:
            raise ValueError(
                f"knob knurl_count={n} leaves a {land_w:.2f} mm ridge land < min_wall {min_wall} mm "
                f"for a {outer_diameter} mm knob; reduce the count"
            )
        if d > outer_diameter * 0.1:
            raise ValueError(f"knob knurl_depth={d} is too deep for a {outer_diameter} mm knob")
        return n, d

    def build(self, params, *, mate_radial_clearance_mm=None, profile=None) -> BuildResult:
        p = self.validate_params(params)
        od = p["outer_diameter"]
        h = p["height"]
        bore_r = _bore_radius(p["shaft_diameter"], mate_radial_clearance_mm)
        bore_depth = min(p["bore_depth"], h - 1.0)
        flat_depth = p["shaft_flat_depth"]
        flat_clearance = 0.0 if mate_radial_clearance_mm is None else mate_radial_clearance_mm

        # Round grip body centred on the origin, growing along +Z.
        body = cq.Workplane("XY").circle(od / 2).extrude(h)

        # Knurled grip: printable groove count/depth derived from (or validated against) the profile.
        n, kd = self.resolve_knurl(od, p["knurl_count"], p["knurl_depth"], profile)
        if n > 0 and kd > 0:
            groove_w = max(1.5 * (getattr(profile, "nozzle_mm", 0.4) if profile else 0.4), 0.6)
            cutters = cq.Workplane("XY")
            for i in range(n):
                ang = 360.0 * i / n
                cutters = cutters.add(
                    cq.Workplane("XY")
                    .center(od / 2, 0)
                    .rect(2 * kd, groove_w)
                    .extrude(h)
                    .rotate((0, 0, 0), (0, 0, 1), ang)
                )
            body = body.cut(cutters)

        # Mating bore from the -Z face.
        if flat_depth > 0:
            body = _cut_d_bore(body, radius=bore_r, depth=bore_depth, flat_depth=flat_depth,
                               flat_clearance=flat_clearance)
            flat_normal: tuple[float, float, float] | None = (0.0, -1.0, 0.0)
            mate_kind = "d_shaft"
            fc: float | None = flat_clearance
        else:
            cutter = (cq.Workplane("XY").workplane(offset=-0.001)
                      .circle(bore_r).extrude(bore_depth + 0.001))
            body = body.cut(cutter)
            flat_normal = None
            mate_kind = "cylinder"
            fc = None

        mate = MateInfo(
            kind=mate_kind,
            bore_axis_point=(0.0, 0.0, 0.0),
            bore_axis_dir=_Z_AXIS,
            bore_radius_mm=bore_r,
            engagement_mm=bore_depth,
            flat_normal=flat_normal,
            flat_clearance_mm=fc,
        )
        return BuildResult(solid=body, mate=mate, carries_load=False,
                           load_feature="knobs carry no structural load")


class LBracketTemplate(Template):
    name = "l_bracket"
    params = (
        ParamSpec("leg_a_mm", 40.0, 10.0, 200.0, "first leg length (mm)"),
        ParamSpec("leg_b_mm", 40.0, 10.0, 200.0, "second leg length (mm)"),
        ParamSpec("width_mm", 30.0, 8.0, 150.0, "bracket width (mm)"),
        ParamSpec("thickness_mm", 4.0, 1.5, 12.0, "material thickness (mm)"),
        ParamSpec("hole_diameter_mm", 5.0, 2.0, 20.0, "mounting hole diameter (mm)"),
        ParamSpec("holes_per_leg", 2.0, 0.0, 4.0, "mounting holes per leg"),
    )

    def build(self, params, *, mate_radial_clearance_mm=None, profile=None) -> BuildResult:
        p = self.validate_params(params)
        a, b, w, t = p["leg_a_mm"], p["leg_b_mm"], p["width_mm"], p["thickness_mm"]
        hd = p["hole_diameter_mm"]
        n = int(round(p["holes_per_leg"]))

        # Horizontal leg lying in the XY plane (thickness t along Z), from x=0 to x=a.
        base = cq.Workplane("XY").box(a, w, t, centered=(False, True, False))
        # Vertical leg standing on the corner at x=0, rising in +Z from z=0 to z=b, thickness t along X.
        upright = cq.Workplane("XY").box(t, w, b, centered=(False, True, False))
        bracket = base.union(upright)

        if n > 0 and hd > 0:
            cutters = cq.Workplane("XY")
            for i in range(n):
                # Holes through the horizontal leg (axis +Z), spaced along x.
                x = a * (i + 1) / (n + 1)
                if x > t + hd:  # keep clear of the corner joint
                    cutters = cutters.add(
                        cq.Workplane("XY").center(x, 0).circle(hd / 2)
                        .extrude(t + 2).translate((0, 0, -1))
                    )
                # Holes through the vertical leg (axis +X), spaced along z.
                z = b * (i + 1) / (n + 1)
                if z > t + hd:
                    cutters = cutters.add(
                        cq.Workplane("YZ").workplane(offset=-1).center(0, z).circle(hd / 2)
                        .extrude(t + 2)
                    )
            try:
                bracket = bracket.cut(cutters)
            except Exception:  # noqa: BLE001 - if a hole falls on an edge, ship the solid bracket
                pass

        return BuildResult(solid=bracket, mate=None, carries_load=True,
                           load_feature="L-bracket loaded as a cantilever")


class SpacerTemplate(Template):
    name = "spacer"
    params = (
        ParamSpec("outer_diameter_mm", 18.0, 4.0, 120.0, "outer diameter (mm)"),
        ParamSpec("inner_diameter_mm", 8.2, 1.0, 100.0, "through-hole diameter (mm)"),
        ParamSpec("height_mm", 5.0, 0.5, 60.0, "height / thickness (mm)"),
    )

    def build(self, params, *, mate_radial_clearance_mm=None, profile=None) -> BuildResult:
        p = self.validate_params(params)
        od, idd, h = p["outer_diameter_mm"], p["inner_diameter_mm"], p["height_mm"]
        if idd >= od:
            raise ValueError("spacer inner_diameter_mm must be smaller than outer_diameter_mm")
        ring = cq.Workplane("XY").circle(od / 2).circle(idd / 2).extrude(h)
        return BuildResult(solid=ring, mate=None, carries_load=False,
                           load_feature="washer/spacer carries compressive load only")


class AdapterTemplate(Template):
    name = "pipe_adapter"
    mates = True
    params = (
        ParamSpec("inlet_diameter_mm", 25.0, 4.0, 120.0, "inlet bore diameter (mates a shaft/pipe) (mm)"),
        ParamSpec("outlet_diameter_mm", 18.0, 4.0, 120.0, "outlet bore diameter (mm)"),
        ParamSpec("inlet_depth_mm", 15.0, 3.0, 80.0, "inlet socket depth (mm)"),
        ParamSpec("outlet_depth_mm", 15.0, 3.0, 80.0, "outlet socket depth (mm)"),
        ParamSpec("wall_mm", 3.0, 1.2, 12.0, "wall thickness around each socket (mm)"),
    )

    def build(self, params, *, mate_radial_clearance_mm=None, profile=None) -> BuildResult:
        p = self.validate_params(params)
        in_r = _bore_radius(p["inlet_diameter_mm"], mate_radial_clearance_mm)
        out_r = p["outlet_diameter_mm"] / 2
        in_d, out_d, wall = p["inlet_depth_mm"], p["outlet_depth_mm"], p["wall_mm"]
        total = in_d + out_d
        in_outer = in_r + wall
        out_outer = out_r + wall
        # Two coaxial tubes stacked along +Z; inlet socket opens on the -Z face.
        body = (
            cq.Workplane("XY").circle(in_outer).extrude(in_d)
            .faces(">Z").workplane().circle(out_outer).extrude(out_d)
        )
        # Inlet bore from -Z.
        body = body.cut(cq.Workplane("XY").workplane(offset=-0.001).circle(in_r).extrude(in_d + 0.001))
        # Outlet bore from +Z.
        body = body.cut(
            cq.Workplane("XY").workplane(offset=total + 0.001).circle(out_r).extrude(-out_d - 0.001)
        )
        # A small through-passage so it is one hollow part, not two blind sockets.
        thru_r = min(in_r, out_r) * 0.6
        body = body.cut(cq.Workplane("XY").workplane(offset=-0.001).circle(thru_r).extrude(total + 0.002))
        mate = MateInfo(
            kind="cylinder",
            bore_axis_point=(0.0, 0.0, 0.0),
            bore_axis_dir=_Z_AXIS,
            bore_radius_mm=in_r,
            engagement_mm=in_d,
            flat_normal=None,
            flat_clearance_mm=None,
        )
        return BuildResult(solid=body, mate=mate, carries_load=False,
                           load_feature="pipe/hose adapter carries line pressure only")


class CableClipTemplate(Template):
    name = "cable_clip"
    params = (
        ParamSpec("cable_diameter_mm", 8.0, 2.0, 40.0, "cable diameter the clip grips (mm)"),
        ParamSpec("wall_mm", 2.4, 1.2, 6.0, "clip wall thickness (mm)"),
        ParamSpec("depth_mm", 10.0, 4.0, 40.0, "clip depth along the cable (mm)"),
        ParamSpec("opening_fraction", 0.65, 0.3, 0.95, "mouth width as a fraction of the cable diameter"),
        ParamSpec("base_mm", 3.0, 1.5, 10.0, "mounting base thickness (mm)"),
    )

    def build(self, params, *, mate_radial_clearance_mm=None, profile=None) -> BuildResult:
        p = self.validate_params(params)
        r = p["cable_diameter_mm"] / 2
        wall, depth = p["wall_mm"], p["depth_mm"]
        base = p["base_mm"]
        outer_r = r + wall
        # Base slab under the C-ring.
        slab = cq.Workplane("XY").box(outer_r * 2, depth, base, centered=(True, True, False))
        ring = (
            cq.Workplane("XY").workplane(offset=base)
            .center(0, 0).circle(outer_r).circle(r).extrude(depth)
            .rotate((0, 0, 0), (1, 0, 0), -90)
            .translate((0, depth / 2, base + outer_r))
        )
        # C opening: cut a mouth on the +Z top.
        mouth_w = p["opening_fraction"] * p["cable_diameter_mm"]
        mouth = cq.Workplane("XY").box(mouth_w, depth + 2, outer_r, centered=(True, True, False)) \
            .translate((0, 0, base + outer_r))
        clip = slab.union(ring).cut(mouth)
        return BuildResult(solid=clip, mate=None, carries_load=False,
                           load_feature="cable clip grips only")


class HookTemplate(Template):
    name = "hook"
    params = (
        ParamSpec("shank_length_mm", 40.0, 10.0, 160.0, "straight shank length before the curve (mm)"),
        ParamSpec("shank_thickness_mm", 6.0, 2.0, 20.0, "shank square cross-section (mm)"),
        ParamSpec("hook_radius_mm", 14.0, 4.0, 60.0, "inner radius of the hook curve (mm)"),
        ParamSpec("base_width_mm", 24.0, 10.0, 80.0, "mounting base width (mm)"),
        ParamSpec("base_thickness_mm", 4.0, 2.0, 12.0, "mounting base thickness (mm)"),
        ParamSpec("hole_diameter_mm", 5.0, 2.0, 16.0, "mounting hole diameter (mm)"),
    )

    def build(self, params, *, mate_radial_clearance_mm=None, profile=None) -> BuildResult:
        p = self.validate_params(params)
        st = p["shank_thickness_mm"]
        sl = p["shank_length_mm"]
        hr = p["hook_radius_mm"]
        bw, bt = p["base_width_mm"], p["base_thickness_mm"]
        hd = p["hole_diameter_mm"]

        # Mounting base: a plate in the X-Z plane (its thickness bt along +Y), standing in the X-Z wall.
        base = cq.Workplane("XZ").box(bw, bw, bt, centered=(True, True, False))
        base = base.cut(
            cq.Workplane("XZ").workplane(offset=-1).circle(hd / 2).extrude(bt + 2)
        )
        # Shank: a square bar rising in +Z from the base, standing off the wall in +Y.
        shank = cq.Workplane("XY").box(st, st, sl, centered=(True, False, False)).translate((0, bt, 0))

        # Hook curve: sweep the square section along a quarter-circle arc in the Y-Z plane that turns
        # the tip from pointing +Z over to pointing +Y, centred at (y=bt+hr, z=sl).
        path = (
            cq.Workplane("YZ")
            .moveTo(bt, sl)
            .threePointArc((bt + hr - hr * 0.707, sl + hr * 0.707), (bt + hr, sl))
        )
        profile = cq.Workplane("XY").rect(st, st).translate((0, bt, sl))
        try:
            curve = profile.sweep(path)
            hook = base.union(shank).union(curve)
        except Exception:  # noqa: BLE001 - if the sweep is degenerate, ship the straight hook (shank only)
            hook = base.union(shank)

        return BuildResult(solid=hook, mate=None, carries_load=True,
                           load_feature="hook loaded as a cantilever at the tip")


class CapTemplate(Template):
    name = "cap"
    mates = True
    params = (
        ParamSpec("bore_diameter_mm", 20.0, 4.0, 120.0, "diameter of the tube/pipe the cap covers (mm)"),
        ParamSpec("depth_mm", 14.0, 4.0, 80.0, "how far the cap slips over the tube (mm)"),
        ParamSpec("wall_mm", 2.4, 1.2, 10.0, "cap wall thickness (mm)"),
        ParamSpec("top_mm", 3.0, 1.2, 12.0, "closed top thickness (mm)"),
    )

    def build(self, params, *, mate_radial_clearance_mm=None, profile=None) -> BuildResult:
        p = self.validate_params(params)
        in_r = _bore_radius(p["bore_diameter_mm"], mate_radial_clearance_mm)
        depth, wall, top = p["depth_mm"], p["wall_mm"], p["top_mm"]
        outer_r = in_r + wall
        total = depth + top
        body = cq.Workplane("XY").circle(outer_r).extrude(total)
        # Open socket from -Z up to `depth`, leaving a closed `top` at +Z.
        body = body.cut(cq.Workplane("XY").workplane(offset=-0.001).circle(in_r).extrude(depth + 0.001))
        mate = MateInfo(
            kind="cylinder",
            bore_axis_point=(0.0, 0.0, 0.0),
            bore_axis_dir=_Z_AXIS,
            bore_radius_mm=in_r,
            engagement_mm=depth,
            flat_normal=None,
            flat_clearance_mm=None,
        )
        return BuildResult(solid=body, mate=mate, carries_load=False,
                           load_feature="cap/plug carries no structural load")


class EnclosureTemplate(Template):
    name = "enclosure"
    params = (
        ParamSpec("inner_x_mm", 60.0, 10.0, 220.0, "internal length X (mm)"),
        ParamSpec("inner_y_mm", 40.0, 10.0, 180.0, "internal width Y (mm)"),
        ParamSpec("inner_z_mm", 25.0, 5.0, 180.0, "internal height Z (mm)"),
        ParamSpec("wall_mm", 2.4, 1.2, 8.0, "wall thickness (mm)"),
        ParamSpec("corner_radius_mm", 2.0, 0.0, 20.0, "outer corner fillet radius (mm)"),
    )

    def build(self, params, *, mate_radial_clearance_mm=None, profile=None) -> BuildResult:
        p = self.validate_params(params)
        ix, iy, iz = p["inner_x_mm"], p["inner_y_mm"], p["inner_z_mm"]
        wall = p["wall_mm"]
        r = p["corner_radius_mm"]
        ox, oy, oz = ix + 2 * wall, iy + 2 * wall, iz + wall  # open-topped box
        box = cq.Workplane("XY").box(ox, oy, oz, centered=(True, True, False))
        if r > 0:
            try:
                box = box.edges("|Z").fillet(min(r, min(ox, oy) / 2 - 0.1))
            except Exception:
                pass
        # Hollow it from +Z, leaving a wall floor.
        cavity = cq.Workplane("XY").workplane(offset=wall).box(ix, iy, iz + 1, centered=(True, True, False))
        box = box.cut(cavity)
        return BuildResult(solid=box, mate=None, carries_load=False,
                           load_feature="enclosure carries no structural load")
