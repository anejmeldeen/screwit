// Performance marks the X6 perf team's Lighthouse/Playwright tests read
// (AGENTS.md "Performance budgets"). Each mark is emitted at most once per page
// load so the tests see a single, meaningful timestamp.

const emitted = new Set<string>();

export function markOnce(name: string): void {
  if (typeof performance === "undefined" || !performance.mark) return;
  if (emitted.has(name)) return;
  emitted.add(name);
  try {
    performance.mark(name);
  } catch {
    /* marks are best-effort */
  }
}

export const PERF_MARKS = {
  shellReady: "shell:ready",
  firstMesh: "viewer:first-mesh",
  firstSplat: "viewer:first-splat",
  chatFirstToken: "chat:first-token",
} as const;

// Reset (test/dev only): lets a page re-mark after a soft reload.
export function resetMarks(): void {
  emitted.clear();
}

// Run non-critical work after first paint to keep the main thread free during
// load (lowers Total Blocking Time). Falls back to setTimeout where
// requestIdleCallback is unavailable (Safari/iOS).
export function afterPaint(fn: () => void): void {
  if (typeof window === "undefined") return;
  const ric = (
    window as unknown as {
      requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => number;
    }
  ).requestIdleCallback;
  if (ric) {
    ric(() => fn(), { timeout: 1500 });
  } else {
    setTimeout(fn, 0);
  }
}
