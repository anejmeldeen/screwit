"""Export the actual component mesh in scan-frame millimetres, without fitting new geometry."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

MAX_MESH_BYTES = 64 * 1024 * 1024
MAX_STEP_FACES = 20_000


def export_part_mesh(data: bytes, file_format: Literal["stp", "stl"]) -> bytes:
    if len(data) > MAX_MESH_BYTES:
        raise ValueError("This mesh is too large to export (64 MB limit).")
    # CAD conversion has a deadline and cannot tie up an API thread indefinitely.
    with TemporaryDirectory(prefix="screwit-export-") as directory:
        source = Path(directory) / "part.glb"
        target = Path(directory) / f"part.{file_format}"
        source.write_bytes(data)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "screwit.cad.part_export", str(source), str(target)],
                capture_output=True, text=True, timeout=60, check=False,
            )
        except subprocess.TimeoutExpired:
            raise ValueError("Export took too long. Try STL or a lower-resolution scan.") from None
        if result.returncode:
            message = result.stderr.strip().splitlines()
            raise ValueError(message[-1] if message else "Could not convert this component.")
        return target.read_bytes()


def convert(source: Path, target: Path) -> None:
    import numpy as np
    import trimesh

    # to_geometry applies every GLB node transform, including combined-part placements.
    scene = trimesh.load(source, file_type="glb", force="scene")
    mesh = scene.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("This component does not contain a valid triangle mesh.")
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces):
        raise ValueError("This component contains no usable faces.")
    if target.suffix == ".stl":
        target.write_bytes(mesh.export(file_type="stl"))
        return
    if len(mesh.faces) > MAX_STEP_FACES:
        raise ValueError("STEP export supports up to 20,000 triangles. Download STL for this detailed scan.")
    if not mesh.is_watertight:
        raise ValueError("This scan has open edges. Download STL and repair the mesh before creating a STEP solid.")

    import cadquery as cq

    solids = []
    for component in mesh.split(only_watertight=False):
        component.fix_normals()
        faces = [cq.Face.makeFromWires(cq.Wire.makePolygon(
            [cq.Vector(*point) for point in triangle], close=True,
        )) for triangle in component.triangles]
        solid = cq.Solid.makeSolid(cq.Shell.makeShell(faces))
        if not solid.isValid() or solid.Volume() <= 0:
            raise ValueError("This mesh could not form a valid STEP solid. Download STL to repair it.")
        solids.append(solid)
    # Explicit STEP type allows the requested .stp extension. Units are millimetres.
    cq.exporters.export(cq.Compound.makeCompound(solids), str(target), exportType="STEP")


if __name__ == "__main__":
    try:
        convert(Path(sys.argv[1]), Path(sys.argv[2]))
    except ImportError:
        print("STEP export needs the server's CAD dependencies installed.", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("Could not export this mesh. Try STL or repair the scan geometry.", file=sys.stderr)
        sys.exit(1)
