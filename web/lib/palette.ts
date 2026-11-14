// Shared with scripts/recolor_fixture_previews.py and the fixture generator.
// Keep this module free of three.js so cards do not load the viewer bundle.
import palette from "./partPalette.json";

export function partColor(index: number, partId?: string): string {
  let slot = index;
  if (partId) {
    const numbered = /^p([1-9]\d*)$/.exec(partId);
    // Original component IDs match their baked preview palette slots. Other
    // IDs get a stable hash so reorder, hide and isolation cannot recolor them.
    slot = numbered ? Number(numbered[1]) - 1 : Array.from(partId).reduce(
      (hash, char) => (Math.imul(hash, 31) + char.charCodeAt(0)) >>> 0, 0,
    );
  }
  return palette[((slot % palette.length) + palette.length) % palette.length];
}
