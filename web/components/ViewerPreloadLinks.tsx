// Server component: emits low-priority <link rel="prefetch"> for the mesh-path
// viewer chunks (three.js + r3f + the mesh loader) so the browser fetches them
// from the initial HTML (~first paint) instead of waiting for hydration to trigger
// the dynamic import. Spark/splat and CAD chunks are deliberately excluded.
// Renders nothing when the manifest lookup fails.
import { meshViewerPreloadHrefs } from "@/lib/viewerPreload";

export async function ViewerPreloadLinks() {
  const hrefs = await meshViewerPreloadHrefs();
  if (!hrefs.length) return null;
  return (
    <>
      {hrefs.map((href) => (
        // prefetch (low priority) warms the cache without competing with the
        // render-blocking shell scripts; modulepreload (high priority) starved
        // the shell on slow 4G and made first-mesh worse.
        <link key={href} rel="prefetch" as="script" href={href} />
      ))}
    </>
  );
}
