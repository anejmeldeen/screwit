import { describe, expect, it } from "vitest";
import { BoxGeometry, CylinderGeometry, Group, Mesh, MeshStandardMaterial, Triangle, Vector3 } from "three";
import { clonePartScene, closestSegments, edgeStats, explodeOffsets, featureDistance, meshTopology, type MeshFeature } from "../lib/editorGeometry";
const v = (x: number, y = 0, z = 0) => new Vector3(x, y, z);
const edge = (points: Vector3[]): MeshFeature => ({ id: "e", kind: "edge", points, triangles: [] });
const face = (triangles: Triangle[]): MeshFeature => ({ id: "f", kind: "face", triangles, points: [] });

describe("mesh measurements", () => {
  it("welds box vertices and excludes triangulation diagonals", () => {
    const topology = meshTopology(new BoxGeometry(10, 20, 30));
    expect(topology.edges).toHaveLength(12);
    expect(topology.faces).toHaveLength(6);
    expect(topology.faces.every(f => f.triangles.length === 2)).toBe(true);
    expect(topology.edges.map(e => edgeStats(e).length).sort((a,b) => a-b)).toEqual([10,10,10,10,20,20,20,20,30,30,30,30]);
  });
  it("joins a curved cylinder rim and measures arc and axis travel, not its chord", () => {
    const topology = meshTopology(new CylinderGeometry(10, 10, 20, 64));
    expect(topology.edges).toHaveLength(2);
    for (const rim of topology.edges) {
      const s = edgeStats(rim);
      expect(s.length).toBeCloseTo(2 * Math.PI * 10, 1);
      expect(s.x).toBeCloseTo(40, 4); expect(s.z).toBeCloseTo(40, 4);
      expect(s.y).toBe(0); expect(s.delta.length()).toBe(0);
    }
  });
  it("finds a gap between segment interiors and handles point segments", () => {
    const [a,b] = featureDistance(edge([v(-10),v(10)]), edge([v(0,-10,5),v(0,10,5)]));
    expect(a.distanceTo(b)).toBe(5);
    const [c,d] = closestSegments(v(4,5),v(4,5),v(0),v(10));
    expect(c.distanceTo(d)).toBe(5); expect(d.x).toBe(4);
  });
  it("measures finite faces rather than infinite planes", () => {
    const a = face([new Triangle(v(0),v(10),v(0,10))]);
    const b = face([new Triangle(v(20,0,5),v(30,0,5),v(20,10,5))]);
    const [p,q] = featureDistance(a,b);
    expect(p.distanceTo(q)).toBeCloseTo(Math.sqrt(125));
  });
  it("returns zero for crossing faces even when all vertices lie outside", () => {
    const a = face([new Triangle(v(-10,-10),v(10,-10),v(0,10))]);
    const b = face([new Triangle(v(0,0,-10),v(0,0,10),v(20,0,0))]);
    const [p,q] = featureDistance(a,b); expect(p.distanceTo(q)).toBe(0);
  });
});
describe("assembly editing", () => {
  it("separates nested coaxial parts and is independent of the origin", () => {
    const parts = [0,8,80].map((z,i) => ({ part_id: `p${i}`, centroid_mm: [0,0,z], obb: { extent: [60,60,30] } }));
    const result = explodeOffsets(parts);
    const translated = explodeOffsets(parts.map(p => ({...p, centroid_mm: p.centroid_mm.map(n => n+1000)})));
    for (const p of parts) expect(result.get(p.part_id)!.distanceTo(translated.get(p.part_id)!)).toBeLessThan(1e-8);
    for (let i = 0; i < parts.length; i++) for (let j = i+1; j < parts.length; j++) {
      const a = new Vector3(...parts[i].centroid_mm as [number,number,number]).add(result.get(parts[i].part_id)!);
      const b = new Vector3(...parts[j].centroid_mm as [number,number,number]).add(result.get(parts[j].part_id)!);
      expect(a.distanceTo(b)).toBeGreaterThan(89);
    }
  });
  it("keeps an isolated single component at its original position", () => {
    expect(explodeOffsets([{ part_id: "only", centroid_mm: [10,20,30], obb: { extent: [20,20,20] } }]).get("only")!.length()).toBe(0);
  });
  it("does not share mutable materials with other component windows", () => {
    const source = new Group(); source.add(new Mesh(new BoxGeometry(),new MeshStandardMaterial()));
    const a = clonePartScene(source), b = clonePartScene(source);
    expect((a.children[0] as Mesh).material).not.toBe((b.children[0] as Mesh).material);
    expect((a.children[0] as Mesh).geometry).toBe((b.children[0] as Mesh).geometry);
  });
});

describe("bundled GLB geometry", () => {
  it("extracts selectable edges and faces from the actual demo components", async () => {
    const { readFileSync } = await import("node:fs");
    const { GLTFLoader } = await import("three/examples/jsm/loaders/GLTFLoader.js");
    for (const scan of ["scn_trap", "scn_knob"]) {
      const parts = JSON.parse(readFileSync(`../fixtures/scans/${scan}/parts.json`, "utf8"));
      for (const part of parts) {
        const bytes = readFileSync(`../fixtures/${part.mesh_uri}`);
        const gltf = await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), "");
        let edges = 0, faces = 0;
        gltf.scene.traverse(o => {
          if (!(o as Mesh).isMesh) return;
          const topology = meshTopology((o as Mesh).geometry);
          edges += topology.edges.length; faces += topology.faces.length;
          for (const e of topology.edges) expect(edgeStats(e).length).toBeGreaterThan(0);
        });
        expect(edges, `${scan}/${part.part_id} edges`).toBeGreaterThan(0);
        expect(faces, `${scan}/${part.part_id} faces`).toBeGreaterThan(0);
      }
    }
  });
});
