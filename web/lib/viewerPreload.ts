// Server-only: resolve the JS chunks for the mesh-path viewer dynamic imports
// from the build manifest, so the workspace route can emit low-priority
// <link rel="prefetch"> for them in the initial HTML. The browser then fills
// idle bandwidth with three.js/r3f during the shell load instead of starting
// at hydration. (modulepreload was measured and is worse on slow 4G: at high
// priority it starved the shell, 2.4 s -> 4.0 s.)
//
// Next does NOT emit modulepreload for `ssr:false` dynamic chunks, but it
// writes .next/react-loadable-manifest.json mapping each dynamic import to its
// files. We read the entries for ViewerPanel's ./Viewer and ./WholeMesh imports
// (three.js + r3f + the mesh loader) and deliberately EXCLUDE the SplatLayer /
// Spark and CAD entries. Read once and cache; on any failure return [] and log
// once so the page never breaks.

// NOTE: server-only module (reads the build manifest from disk). It is only
// imported by the workspace route's server component; never import it into a
// client component.
import { readFile } from "node:fs/promises";
import { join } from "node:path";

// Only these dynamic-import keys are on the mesh-visible critical path. Spark
// (SplatLayer) and CAD (CadOverlay) are intentionally omitted.
const MESH_PATH_KEYS = [
  "components/viewer/ViewerPanel.tsx -> ./Viewer",
  "components/viewer/ViewerPanel.tsx -> ./WholeMesh",
];

// Guard: never preload a Spark / splat chunk even if the manifest shape shifts.
const SPLAT_CHUNK_DENY = /spark|splat/i;

let cached: string[] | null = null;
let loggedFailure = false;

interface LoadableEntry {
  id?: string | number;
  files?: string[];
}

function logOnce(msg: string): void {
  if (loggedFailure) return;
  loggedFailure = true;
  // eslint-disable-next-line no-console
  console.warn(`[viewerPreload] ${msg}; skipping viewer prefetch`);
}

// Returns the list of chunk paths (rooted at /_next/) to prefetch for the
// mesh path. Empty when the manifest is missing or the lookup fails.
export async function meshViewerPreloadHrefs(): Promise<string[]> {
  if (cached) return cached;
  try {
    const manifestPath = join(
      process.cwd(),
      ".next",
      "react-loadable-manifest.json",
    );
    const raw = await readFile(manifestPath, "utf8");
    const manifest = JSON.parse(raw) as Record<string, LoadableEntry>;

    const files = new Set<string>();
    for (const key of MESH_PATH_KEYS) {
      const entry = manifest[key];
      if (!entry || !Array.isArray(entry.files)) continue;
      for (const f of entry.files) {
        if (!f.endsWith(".js")) continue;
        if (SPLAT_CHUNK_DENY.test(f)) continue;
        files.add(f.startsWith("/") ? f : `/_next/${f}`);
      }
    }

    if (files.size === 0) {
      logOnce("no mesh-path chunks found in react-loadable-manifest.json");
      cached = [];
      return cached;
    }
    cached = Array.from(files);
    return cached;
  } catch (err) {
    logOnce(`could not read react-loadable-manifest.json: ${String(err)}`);
    cached = [];
    return cached;
  }
}
