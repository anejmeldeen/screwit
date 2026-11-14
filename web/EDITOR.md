# 3D component editor

## Controls

- Each Parts card has **Hide/Show** and **Isolate…**. **Show all** restores hidden components.
- Isolate opens a separate modal viewing window with its own orbit controls and measurement state. Close or Escape returns to the assembly. Isolation does not change assembly visibility or source data.
- Explode separates parts around the assembly center and resolves overlapping bounding spheres. It preserves each part's internal geometry. Separating parts preserves the current camera angle, target, and zoom. The slider at zero restores the original placement exactly.
- Hiding parts, exploding, or measuring uses the component meshes even on a phone. Whole-object mobile meshes and splats do not carry selectable component identities. Show all components before returning to Splat.

## Measurements

Click **Measure**, then select a measurement type:

1. **Point to point**: click any two positions on the mesh surface. Markers and a segment identify the selected positions. No predefined vertex points are required.
2. **Edge length (including curves)**: hover near an outlined edge and click the highlighted chain. Shows the polyline length, total absolute X/Y/Z travel along the chain, and signed endpoint displacement. A closed rim has nonzero travel but zero endpoint displacement.
3. **Edge to edge**: select two highlighted edge chains. Reports the shortest distance between the finite polylines, with a segment between their closest points.
4. **Face to face**: select two highlighted faces. Reports the shortest distance between the finite face patches, rather than between infinite planes.

Click Clear to start again. After a completed two-feature measurement, clicking starts a new pair. Distances and components are scan-frame millimetres (Z up), independent of the web camera. Starting Measure or Set scale resets explode; explode is disabled during measurement. Measurements always use assembled geometry. Set scale remains available in the assembly window.

These are measurements of the reconstructed triangle mesh, not exact analytic CAD geometry or caliper measurements. Feature edges are boundaries and creases of at least 30 degrees; smooth segments are joined across shallow turns, while corners/junctions break chains. Coplanar adjacent triangles form a face; curved surfaces remain faceted. Rotate the model or isolate a component to reach obscured surfaces. Mesh resolution and scan calibration limit measurement accuracy.

## Theme and panel sizing

The editor uses the homepage yellow (#f1d45d) on warm charcoal, with brighter grid lines and bold dark button labels. Measurement tables wrap labels and stay within their cards.

Drag the yellow divider handles to resize Parts and Assistant, or focus a handle and use arrow keys (Home/End select the minimum/maximum). Wide screens resize panel widths; narrower layouts resize stacked panel heights. Panels can expand into the available window space, leaving a 220px minimum viewer area; stacked panels can grow to at least 1600px tall.

The HQ/Phone switch has been removed. All devices use editable per-component meshes; splat quality is selected automatically from device capability.

## Part downloads

The Parts panel offers **Export whole object**. Select a part for its own download buttons, or click **Isolate…**, then **Download STEP (.stp)** or **Download STL**. Downloads use the current component mesh, including combined parts and GLB node transforms, in scan-frame millimetres. They do not include viewer-only explosion offsets or camera transforms.

STEP exports faceted solids from closed meshes; it does not recover smooth analytic CAD surfaces. Open meshes return a repair message. STL is also available for slicers and mesh-repair tools. Exports do not certify printing fit or mesh quality.

Whole objects use `GET /api/scans/{scan_id}/export?format=stp|stl` and combine all current saved part meshes. Hidden parts remain included; viewer-only visibility is not a geometry edit.

The part API route is `GET /api/scans/{scan_id}/parts/{part_id}/export?format=stp|stl`. The server needs the existing `cad` optional dependencies (`pip install -e './backend[cad]'`). Conversion runs in a separate process with a 60-second deadline, a 64 MB input limit, and a 20,000-triangle STEP limit. Larger scans can use STL. No contract changes or model calls are required.

## Verification

From `web/`: `npm test`, `npm run typecheck`, `npm run lint`, `npm run build`.

Backend: `backend/.venv/bin/python -m pytest backend/tests/app -q`.

`tests/editorGeometry.test.ts` checks boundary/crease extraction, curved rim lengths and axis travel, segment and finite-face distances, intersecting faces, separation of coaxial parts, independent cloned materials, and feature extraction from every bundled component GLB.

`backend/tests/app/test_part_export.py` reimports STEP output to verify solid validity, dimensions, volume, disconnected components and node transforms; it also covers STL downloads, missing parts, unsupported formats and open-mesh errors.

Browser checks cover the narrow layout, desktop panel resizing with the keyboard, measurement-card containment, and isolation download controls. File conversion and download responses are verified by the backend tests.
