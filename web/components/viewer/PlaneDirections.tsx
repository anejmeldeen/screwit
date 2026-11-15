"use client";

// Direction labels at the ends of the ground grid (web frame: XZ plane, Y up): +X, −X, +Y and −Y of
// the scan frame. +X is the arrow printed on the scale board and +Y runs up the board's page
// (SPEC 3.1), so the labels say which way the object sat on the table, and they are the axes every
// measurement uses. Plain three sprites with canvas-text textures (no drei Text) keep the mesh-path
// bundle small; they keep a constant on-screen size, so they read at the far end of the grid too.

import { useEffect, useMemo } from "react";
import { useThree } from "@react-three/fiber";
import { CanvasTexture, Group, Matrix4, SRGBColorSpace, Sprite, SpriteMaterial, Vector3 } from "three";

const X_COLOR = "#ff7a6b";
const Y_COLOR = "#7ad98f";
const AXES: { label: string; scan: [number, number, number]; color: string }[] = [
  { label: "+X", scan: [1, 0, 0], color: X_COLOR },
  { label: "−X", scan: [-1, 0, 0], color: X_COLOR },
  { label: "+Y", scan: [0, 1, 0], color: Y_COLOR },
  { label: "−Y", scan: [0, -1, 0], color: Y_COLOR },
];
// On-screen label height as a share of the viewer's height (sizeAttenuation off).
const LABEL_HEIGHT = 0.045;

// Where each label goes: the scan-frame axis mapped through the one scan_to_web transform (so a
// different contract matrix still labels the right way), at the grid's edge on the plane.
export function gridEndLabels(scanToWeb: Matrix4, halfSize: number) {
  return AXES.flatMap((a) => {
    const dir = new Vector3(...a.scan).transformDirection(scanToWeb);
    dir.y = 0;
    if (dir.lengthSq() <= 1e-9) return [];
    return [{ label: a.label, color: a.color, position: dir.normalize().multiplyScalar(halfSize) }];
  });
}

function labelTexture(text: string, color: string): { texture: CanvasTexture; aspect: number } | null {
  if (typeof document === "undefined") return null;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const font = "600 64px system-ui, -apple-system, 'Segoe UI', sans-serif";
  ctx.font = font;
  const width = Math.ceil(ctx.measureText(text).width) + 28;
  const height = 88;
  canvas.width = width;
  canvas.height = height;
  ctx.font = font; // resizing the canvas resets the context
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  ctx.lineWidth = 10;
  ctx.strokeStyle = "rgba(14, 15, 14, 0.9)";
  ctx.strokeText(text, width / 2, height / 2 + 2);
  ctx.fillStyle = color;
  ctx.fillText(text, width / 2, height / 2 + 2);
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  return { texture, aspect: width / height };
}

export function PlaneDirections({ scanToWeb, halfSize }: { scanToWeb: Matrix4; halfSize: number }) {
  const invalidate = useThree((s) => s.invalidate);
  const built = useMemo(() => {
    const group = new Group();
    const owned: { dispose: () => void }[] = [];
    for (const l of gridEndLabels(scanToWeb, halfSize)) {
      const tex = labelTexture(l.label, l.color);
      if (!tex) continue;
      const material = new SpriteMaterial({
        map: tex.texture, transparent: true, depthWrite: false, sizeAttenuation: false,
      });
      const sprite = new Sprite(material);
      sprite.center.set(0.5, 0); // stands on the grid's edge
      sprite.position.copy(l.position);
      sprite.scale.set(LABEL_HEIGHT * tex.aspect, LABEL_HEIGHT, 1);
      group.add(sprite);
      owned.push(tex.texture, material);
    }
    return { group, dispose: () => owned.forEach((o) => o.dispose()) };
  }, [scanToWeb, halfSize]);
  useEffect(() => {
    invalidate(); // frameloop="demand": draw the new labels
    return () => built.dispose();
  }, [built, invalidate]);
  return <primitive object={built.group} />;
}
