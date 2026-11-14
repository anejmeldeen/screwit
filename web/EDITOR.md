# 3D component editor

## Controls

- Each Parts card has **Hide/Show** and **Isolate…**. **Show all** restores hidden components.
- Isolate opens a separate modal viewing window with its own orbit controls and measurement state. Close or Escape returns to the assembly. Isolation does not change assembly visibility or source data.
- Explode separates parts around the assembly center and resolves overlapping bounding spheres. It preserves each part's internal geometry. The camera reframes the exploded assembly. The slider at zero restores the original placement exactly.
- Hiding parts, exploding, or measuring uses the component meshes even on a phone. Whole-object mobile meshes and splats do not carry selectable component identities. Show all components before returning to Splat.

## Measurements

Click **Measure**, then select a measurement type:

1. **Point to point**: click any two positions on the mesh surface. Markers and a segment identify the selected positions. No predefined vertex points are required.
2. **Edge length (including curves)**: hover near an outlined edge and click the highlighted chain. Shows the polyline length, total absolute X/Y/Z travel along the chain, and signed endpoint displacement. A closed rim has nonzero travel but zero endpoint displacement.
3. **Edge to edge**: select two highlighted edge chains. Reports the shortest distance between the finite polylines, with a segment between their closest points.
4. **Face to face**: select two highlighted faces. Reports the shortest distance between the finite face patches, rather than between infinite planes.

Click Clear to start again. After a completed two-feature measurement, clicking starts a new pair. Distances and components are scan-frame millimetres (Z up), independent of the web camera. Starting Measure or Set scale resets explode; explode is disabled during measurement. Measurements always use assembled geometry. Set scale remains available in the assembly window.

These are measurements of the reconstructed triangle mesh, not exact analytic CAD geometry or caliper measurements. Feature edges are boundaries and creases of at least 30 degrees; smooth segments are joined across shallow turns, while corners/junctions break chains. Coplanar adjacent triangles form a face; curved surfaces remain faceted. Rotate the model or isolate a component to reach obscured surfaces. Mesh resolution and scan calibration limit measurement accuracy.

## Phone/HQ investigation (existing behavior retained)

`ViewerPanel.tsx` stores the selected quality and uses `splatUriFor` in `lib/device.ts` to choose `splat_uri` (HQ) or `splat_mobile_uri` (Phone). The original is used if no mobile splat exists.

The mesh path does not read the selected quality. After merging the upstream editor redesign, all devices use per-component `Part.mesh_uri` assets so editing and explosion remain available on phones. The `meshUriFor` quality helper exists but is not wired into the viewer. Therefore toggling Phone/HQ in desktop Mesh view does not change the mesh. In Splat view the resolution/asset size can change, but similar fixture assets may look alike. This switch is not a change of model units or dimensions. Its handlers and quality selection are unchanged by this editor work.

## STEP/STP feasibility (no export changes made)

Generated CAD already has STEP and STL output: see `backend/screwit/cad/service.py` (CadQuery exports) and `components/cards/Cards.tsx` (download actions). `.step` and `.stp` denote the same family of exchange files.

The viewer's scanned components are GLB triangle meshes, and the splat is a visual point-based representation. Exporting those as a faceted STEP representation is possible with additional conversion code, but does not recover smooth, editable CAD features. A useful editable STEP solid needs mesh repair plus surface fitting/reconstruction (and assembly/unit handling). For printing a scanned object, a repaired watertight STL/3MF is the more direct path; STEP itself does not make a scan watertight or printable.

## Verification

From `web/`: `npm test`, `npm run typecheck`, `npm run lint`, `npm run build`.

`tests/editorGeometry.test.ts` checks boundary/crease extraction, curved rim lengths and axis travel, segment and finite-face distances, intersecting faces, separation of coaxial parts, independent cloned materials, and feature extraction from every bundled component GLB. The existing frontend tests also run. Full browser interaction verification is pending; screen-based checks were stopped after the user questioned the OS permission prompt.
