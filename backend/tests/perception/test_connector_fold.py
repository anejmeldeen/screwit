"""Connector fold (part_atoms.fold_connectors): a small part wedged against its neighbours is not
its own part, it belongs to the neighbour it is attached to (the headphone "hinge" that should be
part of an ear cup or the headband). The rule is pure geometry -- small area plus a large share of
the surface sitting against other parts -- so it treats a hinge, a bracket or any wedged connector
the same way and never looks at a label or object category (AGENTS.md).

A free-standing small part (a knob held off the body) is mostly exposed, so it is kept.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from screwit.perception import part_atoms as pa


def _patches_from_labels(mesh: trimesh.Trimesh, labels: np.ndarray) -> pa.Patches:
    """A Patches whose patch == group, one patch per label value, over the given working mesh."""
    labels = np.asarray(labels, dtype=np.int64)
    return pa.Patches(mesh, labels, int(labels.max()) + 1)


def _union(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    try:
        u = trimesh.boolean.union(parts)
    except Exception:  # noqa: BLE001 - no CSG backend on this host
        pytest.skip("no boolean/CSG backend available for union")
    if not getattr(u, "is_watertight", False):
        pytest.skip("union did not produce a single watertight surface")
    return u


def test_wedged_connector_is_folded_into_a_neighbour():
    """A dumbbell: two large spheres joined by a thin neck. The neck is small and buried against
    both spheres, so it folds into one of them -- two parts, not three. No label is consulted."""
    b1 = trimesh.creation.icosphere(subdivisions=3, radius=22.0)
    b1.apply_translation([-25.0, 0.0, 0.0])
    b2 = trimesh.creation.icosphere(subdivisions=3, radius=22.0)
    b2.apply_translation([25.0, 0.0, 0.0])
    neck = trimesh.creation.cylinder(radius=7.0, height=40.0)
    neck.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    u = _union([b1, b2, neck])

    fc = u.triangles_center
    labels = np.where(fc[:, 0] < -9.0, 0, np.where(fc[:, 0] > 9.0, 1, 2)).astype(np.int64)
    p = _patches_from_labels(u, labels)
    assert p.n == 3

    folded = pa.fold_connectors(p, [0, 1, 2])
    groups = sorted(set(folded))
    assert len(groups) == 2, f"neck should have folded away, got groups {groups}"
    # The two large spheres survive; the neck (patch 2) joined whichever sphere it shares the most
    # boundary with, so it carries that sphere's group id, not its own.
    assert folded[2] in (folded[0], folded[1])


def test_free_standing_small_part_is_kept():
    """A small sphere held well off a split body on a thin stalk is mostly exposed (little of its
    surface sits against the body), so it is NOT a connector and stays its own part."""
    b1 = trimesh.creation.box([50.0, 50.0, 30.0])
    b1.apply_translation([-45.0, 0.0, 0.0])
    b2 = trimesh.creation.box([50.0, 50.0, 30.0])
    b2.apply_translation([45.0, 0.0, 0.0])
    knob = trimesh.creation.icosphere(subdivisions=3, radius=12.0)
    knob.apply_translation([-45.0, 0.0, 55.0])
    stalk = trimesh.creation.cylinder(radius=2.5, height=30.0)
    stalk.apply_translation([-45.0, 0.0, 30.0])
    u = _union([b1, b2, knob, stalk])

    fc = u.triangles_center
    labels = np.where(fc[:, 2] > 40.0, 2, np.where(fc[:, 0] > 0.0, 1, 0)).astype(np.int64)
    p = _patches_from_labels(u, labels)
    assert p.n == 3

    folded = pa.fold_connectors(p, [0, 1, 2])
    assert len(set(folded)) == 3, "an exposed small part must be kept, not folded"


def test_fold_never_drops_below_min_groups():
    """Even if every group looks small, the fold keeps at least ``min_groups`` parts."""
    b1 = trimesh.creation.icosphere(subdivisions=3, radius=22.0)
    b1.apply_translation([-25.0, 0.0, 0.0])
    b2 = trimesh.creation.icosphere(subdivisions=3, radius=22.0)
    b2.apply_translation([25.0, 0.0, 0.0])
    neck = trimesh.creation.cylinder(radius=7.0, height=40.0)
    neck.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    u = _union([b1, b2, neck])
    fc = u.triangles_center
    labels = np.where(fc[:, 0] < -9.0, 0, np.where(fc[:, 0] > 9.0, 1, 2)).astype(np.int64)
    p = _patches_from_labels(u, labels)

    folded = pa.fold_connectors(p, [0, 1, 2], min_groups=3)
    assert len(set(folded)) == 3


def test_owner_relabelled_contiguous_largest_first():
    """The returned owner map is relabelled to contiguous ids with group 0 the largest by area."""
    b1 = trimesh.creation.icosphere(subdivisions=3, radius=26.0)   # largest
    b1.apply_translation([-28.0, 0.0, 0.0])
    b2 = trimesh.creation.icosphere(subdivisions=3, radius=18.0)
    b2.apply_translation([28.0, 0.0, 0.0])
    neck = trimesh.creation.cylinder(radius=7.0, height=42.0)
    neck.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    u = _union([b1, b2, neck])
    fc = u.triangles_center
    labels = np.where(fc[:, 0] < -9.0, 0, np.where(fc[:, 0] > 9.0, 1, 2)).astype(np.int64)
    p = _patches_from_labels(u, labels)

    folded = pa.fold_connectors(p, [0, 1, 2])
    ids = sorted(set(folded))
    assert ids == list(range(len(ids)))     # contiguous from 0
    area = p.work.area_faces
    gface = np.asarray(folded)[p.labels]
    areas = [area[gface == g].sum() for g in ids]
    assert areas[0] == max(areas)           # group 0 is the largest
