import { BufferGeometry, Group, Mesh, Object3D, Triangle, Vector3 } from "three";

export type MeasureMode = "point" | "edge-length" | "edge-distance" | "face-distance";
export interface MeshFeature {
  id: string;
  kind: "edge" | "face";
  points: Vector3[];
  triangles: Triangle[];
}
export interface FeaturePick { partId: string; feature: MeshFeature }
export interface MeshTopology { edges: MeshFeature[]; faces: MeshFeature[]; triangleFaces: MeshFeature[] }

// Weld duplicated GLTF vertices before finding boundary/crease edges. Coplanar
// triangle diagonals are deliberately excluded; smooth rim segments form chains.
export function meshTopology(geometry: BufferGeometry): MeshTopology {
  const pos = geometry.getAttribute("position");
  const vertices: Vector3[] = [];
  const weld = new Map<string, number>();
  const ids: number[] = [];
  for (let i = 0; i < pos.count; i++) {
    const p = new Vector3().fromBufferAttribute(pos, i);
    const key = p.toArray().map(v => Math.round(v * 1e5)).join(",");
    let id = weld.get(key);
    if (id === undefined) { id = vertices.length; weld.set(key, id); vertices.push(p); }
    ids.push(id);
  }
  const count = geometry.index?.count ?? pos.count;
  const triangles: Triangle[] = [];
  const normals: Vector3[] = [];
  const edgeMap = new Map<string, { a: number; b: number; faces: number[] }>();
  const neighbors: number[][] = [];
  for (let i = 0; i < count; i += 3) {
    const v = [0, 1, 2].map(j => ids[geometry.index ? geometry.index.getX(i + j) : i + j]);
    const t = new Triangle(...v.map(id => vertices[id]) as [Vector3, Vector3, Vector3]);
    const fi = triangles.length;
    triangles.push(t); normals.push(t.getNormal(new Vector3())); neighbors.push([]);
    for (let j = 0; j < 3; j++) {
      const a = v[j], b = v[(j + 1) % 3];
      const key = a < b ? `${a}:${b}` : `${b}:${a}`;
      const edge = edgeMap.get(key) ?? { a, b, faces: [] };
      edge.faces.push(fi); edgeMap.set(key, edge);
    }
  }
  const crease: { a: number; b: number }[] = [];
  for (const e of edgeMap.values()) {
    if (e.faces.length === 2) {
      const [a, b] = e.faces;
      if (normals[a].dot(normals[b]) > 1 - 1e-6) {
        neighbors[a].push(b); neighbors[b].push(a);
      }
    }
    if (e.faces.length !== 2 || normals[e.faces[0]].dot(normals[e.faces[1]]) < Math.cos(Math.PI / 6)) crease.push(e);
  }
  const faces: MeshFeature[] = [], triangleFaces: MeshFeature[] = [];
  for (let i = 0; i < triangles.length; i++) {
    if (triangleFaces[i]) continue;
    const f: MeshFeature = { id: `f${i}`, kind: "face", points: [], triangles: [] };
    const queue = [i]; triangleFaces[i] = f;
    while (queue.length) {
      const j = queue.pop()!; f.triangles.push(triangles[j]);
      for (const n of neighbors[j]) if (!triangleFaces[n]) { triangleFaces[n] = f; queue.push(n); }
    }
    faces.push(f);
  }
  const adjacency = new Map<number, number[]>();
  crease.forEach((e, i) => { for (const v of [e.a, e.b]) adjacency.set(v, [...(adjacency.get(v) ?? []), i]); });
  const used = new Set<number>();
  const edges: MeshFeature[] = [];
  function walk(start: number, first: number) {
    const points = [vertices[start]];
    let current = start, ei = first;
    while (!used.has(ei)) {
      used.add(ei); const e = crease[ei];
      current = e.a === current ? e.b : e.a; points.push(vertices[current]);
      const next = adjacency.get(current)!;
      // Corners and junctions separate edges; shallow turns preserve curves.
      if (next.length !== 2 || current === start) break;
      const ni = next.find(n => !used.has(n)); if (ni === undefined) break;
      const ne = crease[ni], nv = ne.a === current ? ne.b : ne.a;
      const incoming = points[points.length - 1].clone().sub(points[points.length - 2]).normalize();
      const outgoing = vertices[nv].clone().sub(vertices[current]).normalize();
      if (incoming.dot(outgoing) < Math.cos(Math.PI / 4)) break;
      ei = ni;
    }
    edges.push({ id: `e${edges.length}`, kind: "edge", points, triangles: [] });
  }
  for (const [v, es] of adjacency) if (es.length !== 2) for (const i of es) if (!used.has(i)) walk(v, i);
  crease.forEach((e, i) => { if (!used.has(i)) walk(e.a, i); });
  return { edges, faces, triangleFaces };
}

export function nearestEdge(edges: MeshFeature[], point: Vector3): MeshFeature | null {
  let best: MeshFeature | null = null, distance = Infinity;
  for (const edge of edges) for (let i = 1; i < edge.points.length; i++) {
    const [p, q] = closestSegments(point, point, edge.points[i - 1], edge.points[i]);
    const d = p.distanceToSquared(q);
    if (d < distance) { best = edge; distance = d; }
  }
  return best;
}

export function edgeStats(f: MeshFeature) {
  let length = 0, x = 0, y = 0, z = 0;
  for (let i = 1; i < f.points.length; i++) {
    const d = f.points[i].clone().sub(f.points[i - 1]);
    length += d.length(); x += Math.abs(d.x); y += Math.abs(d.y); z += Math.abs(d.z);
  }
  const delta = f.points.at(-1)!.clone().sub(f.points[0]);
  return { length, x, y, z, delta };
}

// Closest points on finite segments (including zero-length point segments).
export function closestSegments(p: Vector3, q: Vector3, a: Vector3, b: Vector3): [Vector3, Vector3] {
  const u = q.clone().sub(p), v = b.clone().sub(a), w = p.clone().sub(a);
  const aa = u.dot(u), bb = u.dot(v), cc = v.dot(v), dd = u.dot(w), ee = v.dot(w);
  let s = 0, t = 0;
  if (aa < 1e-16 && cc < 1e-16) return [p.clone(), a.clone()];
  if (aa < 1e-16) t = Math.max(0, Math.min(1, ee / cc));
  else if (cc < 1e-16) s = Math.max(0, Math.min(1, -dd / aa));
  else {
    const denom = aa * cc - bb * bb;
    s = denom > 1e-16 ? Math.max(0, Math.min(1, (bb * ee - cc * dd) / denom)) : 0;
    t = (bb * s + ee) / cc;
    if (t < 0) { t = 0; s = Math.max(0, Math.min(1, -dd / aa)); }
    else if (t > 1) { t = 1; s = Math.max(0, Math.min(1, (bb - dd) / aa)); }
  }
  return [p.clone().addScaledVector(u, s), a.clone().addScaledVector(v, t)];
}
const triangleEdges = (t: Triangle): [Vector3, Vector3][] => [[t.a, t.b], [t.b, t.c], [t.c, t.a]];
function intersectTriangle(p: Vector3, q: Vector3, t: Triangle): Vector3 | null {
  const normal = t.getNormal(new Vector3()), dir = q.clone().sub(p);
  const denominator = normal.dot(dir);
  if (Math.abs(denominator) < 1e-12) return null;
  const s = normal.dot(t.a.clone().sub(p)) / denominator;
  if (s < 0 || s > 1) return null;
  const hit = p.clone().addScaledVector(dir, s);
  return t.containsPoint(hit) ? hit : null;
}
export function featureDistance(a: MeshFeature, b: MeshFeature): [Vector3, Vector3] {
  let best: [Vector3, Vector3] = [new Vector3(), new Vector3()], min = Infinity;
  const consider = (pair: [Vector3, Vector3]) => {
    const d = pair[0].distanceToSquared(pair[1]); if (d < min) { min = d; best = pair; }
  };
  if (a.kind === "edge" && b.kind === "edge") {
    for (let i = 1; i < a.points.length; i++) for (let j = 1; j < b.points.length; j++)
      consider(closestSegments(a.points[i - 1], a.points[i], b.points[j - 1], b.points[j]));
  } else {
    for (const ta of a.triangles) for (const tb of b.triangles) {
      for (const p of [ta.a, ta.b, ta.c]) consider([p.clone(), tb.closestPointToPoint(p, new Vector3())]);
      for (const p of [tb.a, tb.b, tb.c]) consider([ta.closestPointToPoint(p, new Vector3()), p.clone()]);
      for (const [p, q] of triangleEdges(ta)) {
        const hit = intersectTriangle(p, q, tb); if (hit) return [hit, hit.clone()];
        for (const [r, s] of triangleEdges(tb)) consider(closestSegments(p, q, r, s));
      }
      for (const [p, q] of triangleEdges(tb)) { const hit = intersectTriangle(p, q, ta); if (hit) return [hit, hit.clone()]; }
    }
  }
  return best;
}

// Center on the assembly, then separate bounding boxes along deterministic
// radial directions. Relax collisions so coaxial/nested parts cannot travel as
// a single cluster. Returned offsets are full-slider scan-frame millimetres.
export function explodeOffsets(parts: { part_id: string; centroid_mm: unknown[]; obb: { extent: unknown[] } }[]): Map<string, Vector3> {
  if (parts.length === 1) return new Map([[parts[0].part_id, new Vector3()]]);
  const center = new Vector3();
  parts.forEach(p => center.add(new Vector3(...p.centroid_mm as [number, number, number])));
  center.divideScalar(parts.length || 1);
  const radius = Math.max(20, ...parts.map(p => new Vector3(...p.obb.extent as [number, number, number]).length() / 2));
  const centers = parts.map(p => new Vector3(...p.centroid_mm as [number, number, number]));
  const offsets = parts.map((p, i) => {
    const dir = centers[i].clone().sub(center);
    const angle = i * Math.PI * (3 - Math.sqrt(5));
    dir.add(new Vector3(Math.cos(angle), Math.sin(angle), (i % 2 ? -1 : 1) * 0.3).multiplyScalar(radius * 0.35));
    return dir.normalize().multiplyScalar(radius * 1.5);
  });
  for (let pass = 0; pass < 32; pass++) for (let i = 0; i < parts.length; i++) for (let j = i + 1; j < parts.length; j++) {
    const d = centers[j].clone().add(offsets[j]).sub(centers[i]).sub(offsets[i]);
    const ri = new Vector3(...parts[i].obb.extent as [number, number, number]).length() / 2;
    const rj = new Vector3(...parts[j].obb.extent as [number, number, number]).length() / 2;
    const gap = ri + rj + radius * 0.15 - d.length();
    if (gap > 0) { if (d.lengthSq() < 1e-12) d.set(1, 0, 0); d.normalize().multiplyScalar(gap * 0.5); offsets[i].sub(d); offsets[j].add(d); }
  }
  return new Map(parts.map((p, i) => [p.part_id, offsets[i]]));
}

export function clonePartScene(scene: Group) {
  const clone = scene.clone(true);
  clone.traverse(o => { if ((o as Mesh).isMesh) { const m = o as Mesh; m.material = Array.isArray(m.material) ? m.material.map(mat => mat.clone()) : m.material.clone(); } });
  return clone;
}
