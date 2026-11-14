"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { subscribeScanEvents } from "./sse";
import type { Part, Reconstruction, Scan } from "./contracts.gen";

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
