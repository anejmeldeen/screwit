"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { api } from "./api";
import { subscribeScanEvents } from "./sse";
import type { Part, Reconstruction, Scan } from "./contracts.gen";

// A number that persists across reloads in localStorage. Reads synchronously before the first paint
// (useLayoutEffect) so a resizable panel renders at its saved size immediately, with no visible snap
// from the default to the stored value. SSR-safe: starts from the default, hydrates before paint.
export function usePersistentNumber(key: string, initial: number): [number, (v: number) => void] {
  const [value, setValue] = useState(initial);
  useLayoutEffect(() => {
    try {
      const raw = window.localStorage.getItem(key);
      if (raw !== null) {
        const n = parseFloat(raw);
        if (!Number.isNaN(n)) setValue(n);
      }
    } catch {
      /* localStorage unavailable (private mode) -> keep the default */
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  const set = useCallback(
    (v: number) => {
      setValue(v);
      try {
        window.localStorage.setItem(key, String(v));
      } catch {
        /* ignore persistence failure */
      }
    },
    [key],
  );
  return [value, set];
}

export function stageState(scan: Scan | null, name: string) {
  return scan?.stages.find((s) => s.name === name)?.state;
}

export function isStageDone(scan: Scan | null, name: string) {
  return stageState(scan, name) === "done";
}

// Subscribe to live scan progress over SSE, seeded by an initial GET.
export function useScan(scanId: string) {
  const [scan, setScan] = useState<Scan | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!scanId) return;
    let alive = true;
    api
      .getScan(scanId)
      .then((s) => alive && setScan(s))
      .catch((e) => alive && setError(String(e?.message ?? e)));

    const unsub = subscribeScanEvents(
      scanId,
      (s) => alive && setScan(s),
      () => {
        /* backoff/reconnect handled inside subscribeScanEvents */
      },
    );
    return () => {
      alive = false;
      unsub();
    };
  }, [scanId]);

  return { scan, error };
}

// Reconstruction.json is written by the pipeline only AFTER the mesh stage (it carries the mesh and
// splat URIs). Fetching on scaleDone was too early: scale finishes mid-pipeline, before
// reconstruction.json exists, so the viewer got a 404/stale result and -- since the revision never
// moved -- never re-fetched, leaving the model empty until a manual reload. Fetch once the mesh
// stage is done (reconstruction.json now exists), and re-fetch when reconstruction_revision moves
// (a /scale rescale) or the scan reaches a terminal state.
export function useReconstruction(scanId: string, scan: Scan | null) {
  const [recon, setRecon] = useState<Reconstruction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const lastRev = useRef<number | null>(null);
  const meshDone = isStageDone(scan, "mesh");
  const terminal = scan?.status === "ready" || scan?.status === "failed";
  const rev = scan?.reconstruction_revision ?? null;

  const refresh = useCallback(() => {
    if (!scanId) return;
    api
      .getReconstruction(scanId)
      .then(setRecon)
      .catch((e) => setError(String(e?.message ?? e)));
  }, [scanId]);

  useEffect(() => {
    if (!meshDone && !terminal) return;
    if (lastRev.current === rev && recon !== null) return;
    lastRev.current = rev;
    refresh();
  }, [meshDone, terminal, rev, refresh, recon]);

  return { recon, error, refresh };
}

// Parts load when the parts stage is done; reload when any revision moves.
export function useParts(scanId: string, scan: Scan | null) {
  const [parts, setParts] = useState<Part[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const partsDone = isStageDone(scan, "parts");
  const rev = scan?.reconstruction_revision ?? null;

  const refresh = useCallback(() => {
    if (!scanId) return;
    api
      .getParts(scanId)
      .then(setParts)
      .catch((e) => setError(String(e?.message ?? e)));
  }, [scanId]);

  useEffect(() => {
    if (!partsDone) return;
    refresh();
  }, [partsDone, rev, refresh]);

  return { parts, error, refresh, setParts };
}
