// Distinct per-part colours, shared by the parts panel and the 3D viewer.
// Kept free of three.js imports so the workspace shell can use it without pulling the viewer bundle.
const PALETTE = [
  "#60a5fa",
  "#f472b6",
  "#34d399",
  "#fbbf24",
  "#a78bfa",
  "#f87171",
  "#22d3ee",
  "#a3e635",
];

export function partColor(index: number): string {
  return PALETTE[index % PALETTE.length];
}
