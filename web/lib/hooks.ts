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

// Reconstruction is fetched once the scale stage is done (it carries
// scan_to_web and the asset URIs). Re-fetched when reconstruction_revision moves.
export function useReconstruction(scanId: string, scan: Scan | null) {
  const [recon, setRecon] = useState<Reconstruction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const lastRev = useRef<number | null>(null);
  const scaleDone = isStageDone(scan, "scale");
  const rev = scan?.reconstruction_revision ?? null;

  const refresh = useCallback(() => {
    api
      .getReconstruction(scanId)
      .then(setRecon)
      .catch((e) => setError(String(e?.message ?? e)));
  }, [scanId]);

  useEffect(() => {
    if (!scaleDone) return;
    if (lastRev.current === rev) return;
    lastRev.current = rev;
    refresh();
  }, [scaleDone, rev, refresh]);

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
