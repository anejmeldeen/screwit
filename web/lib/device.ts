// Device-capability heuristic for picking phone vs desktop 3D assets
// (AGENTS.md "Performance budgets"). Phone assets (splat_mobile_uri /
// mesh_mobile_uri) are smaller and meshopt-compressed; desktop keeps the full
// assets. The client picks by capability and the viewer offers a toggle.

import type { Reconstruction } from "./contracts.gen";

export type AssetQuality = "mobile" | "desktop";

// Is this a phone-class device? Coarse pointer OR a narrow viewport (the lead's
// rule). Phones start in MESH view and never auto-fetch the splat. The
// Playwright mobile preset (iPhone 12: coarse pointer, 390 px) matches this.
export function isPhone(): boolean {
  if (typeof window === "undefined") return false;
  const coarse =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse)").matches;
  const narrow = Math.min(window.innerWidth, window.innerHeight) <= 820;
  return coarse || narrow;
}

// Rough "is this a low-power / small device" test. Errs toward mobile on touch
// phones and low core counts; desktop otherwise.
export function detectQuality(): AssetQuality {
  if (typeof navigator === "undefined" || typeof window === "undefined") {
    return "desktop";
  }
  const coarsePointer =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse)").matches;
  const smallScreen = Math.min(window.innerWidth, window.innerHeight) <= 820;
  const fewCores = (navigator.hardwareConcurrency ?? 8) <= 4;
  const lowMem =
    (navigator as unknown as { deviceMemory?: number }).deviceMemory != null &&
    (navigator as unknown as { deviceMemory?: number }).deviceMemory! <= 4;
  return coarsePointer && (smallScreen || fewCores || lowMem)
    ? "mobile"
    : "desktop";
}

// Resolve the splat URI for a quality, falling back to the full asset when the
// mobile variant is absent.
export function splatUriFor(
  recon: Reconstruction,
  quality: AssetQuality,
): string | null {
  if (quality === "mobile" && recon.splat_mobile_uri) {
    return recon.splat_mobile_uri;
  }
  return recon.splat_uri ?? null;
}

// Resolve the mesh-source: the full mesh is per-part GLBs (Part.mesh_uri). The
// reconstruction's mesh_mobile_uri / mesh_uri are the whole-object mesh; we use
// the mobile whole-object mesh as a lightweight fallback when present.
export function meshUriFor(
  recon: Reconstruction,
  quality: AssetQuality,
): string | null {
  if (quality === "mobile" && recon.mesh_mobile_uri) {
    return recon.mesh_mobile_uri;
  }
  return recon.mesh_uri ?? null;
}

export function hasMobileAssets(recon: Reconstruction | null): boolean {
  return !!(recon && (recon.splat_mobile_uri || recon.mesh_mobile_uri));
}
