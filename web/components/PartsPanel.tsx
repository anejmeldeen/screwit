"use client";

import { useEffect, useState } from "react";
import type { Part } from "@/lib/contracts.gen";
import {
  editableMeasurements,
  resolveAllDimensions,
  ResolvedDimension,
} from "@/lib/parts";
import { api, ApiError } from "@/lib/api";
import { assetUrl } from "@/lib/storage";
import { partColor } from "@/lib/palette";
import { EmptyState, Skeleton } from "@/components/ui";

function fmt(v: number | null): string {
  return v == null ? "—" : v.toFixed(1);
}

function DimensionRow({ d }: { d: ResolvedDimension }) {
  const cell = (
    label: string,
    val: number | null,
    source: ResolvedDimension["preferred"],
  ) => (
    <td
      className={`px-2 py-1 text-right ${
        d.preferred === source ? "font-semibold text-accent" : "text-muted"
      }`}
    >
      {fmt(val)}
    </td>
  );
  return (
    <tr className="border-t border-line">
      <td className="px-2 py-1 text-left text-ink">{d.name.replace(/_/g, " ")}</td>
      {cell("scan", d.scanMm, "scan")}
      {cell("snapped", d.snappedMm, "snapped")}
      {cell("caliper", d.caliperMm, "caliper")}
    </tr>
  );
}

function PartCard({
  part,
  index,
  selected,
  onChanged,
  scanId,
  onCombine,
  hidden, onToggleHidden, onIsolate,
}: {
  hidden: boolean;
  onToggleHidden: (id: string) => void;
  onIsolate: (id: string) => void;
  part: Part;
  index: number;
  selected: boolean;
  onChanged: () => void;
  scanId: string;
  onCombine: () => void;
}) {
  const dims = resolveAllDimensions(part);
  const editable = editableMeasurements(part);
  const [renaming, setRenaming] = useState(false);
  const [label, setLabel] = useState(part.label);
  const [err, setErr] = useState<string | null>(null);
  const [caliperBusy, setCaliperBusy] = useState(false);


  async function saveLabel() {
    setErr(null);
    try {
      await api.patchPart(scanId, part.part_id, { label } as never);
      setRenaming(false);
      onChanged();
    } catch (e) {
      setErr(e instanceof ApiError ? `${e.message} (${e.status})` : String(e));
    }
  }

  async function saveCaliper(
    featureId: string,
    field: string,
    caliperMm: number,
  ) {
    setCaliperBusy(true);
    setErr(null);
    try {
      await api.patchPart(scanId, part.part_id, {
        measurements: [{ feature_id: featureId, field, caliper_mm: caliperMm }],
      } as never);
      onChanged();
    } catch (e) {
      setErr(e instanceof ApiError ? `${e.message} (${e.status})` : String(e));
    } finally {
      setCaliperBusy(false);
    }
  }

  return (
    <div
      data-testid={`part-row-${part.part_id}`}
      className={`part-card rounded-lg border bg-panel p-3 ${
        selected ? "border-accent" : "border-line"
      }`}
    >
      <header className="flex items-center gap-3">
        <span
          className="mt-1 h-3 w-3 shrink-0 rounded-sm"
          style={{ backgroundColor: partColor(index, part.part_id) }}
        />
        <span className="h-12 w-12 shrink-0 overflow-hidden rounded bg-wash">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={assetUrl(part.thumbnail_uri)}
            alt=""
            loading="lazy"
            decoding="async"
            className="h-full w-full object-cover"
            onError={(e) =>
              ((e.currentTarget as HTMLImageElement).style.visibility = "hidden")
            }
          />
        </span>
        <h3 className="min-w-0 break-words font-bold">{part.label}</h3>
      </header>


      {/* Actions */}
      <details className="component-section"><summary>Actions</summary><div className="component-actions">
        <button aria-label={`${hidden ? "Show" : "Hide"} ${part.label}`} aria-pressed={hidden}
          onClick={() => onToggleHidden(part.part_id)}
          className="rounded border border-line px-1.5 py-0.5 hover:border-accent">
          {hidden ? "Show" : "Hide"}
        </button>
        <button aria-label={`Isolate ${part.label}`} onClick={() => onIsolate(part.part_id)}
          className="rounded border border-line px-1.5 py-0.5 hover:border-accent">Isolate</button>
        {renaming ? (
          <span className="rename-controls">
            <input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              className="w-32 rounded border border-line bg-paper px-1.5 py-0.5"
            />
            <button onClick={saveLabel} className="text-accent">
              Save
            </button>
            <button onClick={() => setRenaming(false)} className="text-muted">
              Cancel
            </button>
          </span>
        ) : (
          <button
            onClick={() => setRenaming(true)}
            className="rounded border border-line px-1.5 py-0.5 hover:border-line"
          >
            Rename
          </button>
        )}
        <button onClick={onCombine}>Combine</button>
      </div>

      </details>
      <details className="part-details"><summary>Measurements</summary>
      {/* Dimensions table */}
      {dims.length ? (
        <div className="dimension-table-wrap"><table className="dimension-table">
          <thead className="text-muted">
            <tr>
              <th className="px-2 py-1 text-left font-bold">Size (mm)</th>
              <th className="px-2 py-1 text-right font-bold">Measured</th>
              <th className="px-2 py-1 text-right font-bold">Standard</th>
              <th className="px-2 py-1 text-right font-bold">Your value</th>
            </tr>
          </thead>
          <tbody>
            {dims.map((d) => (
              <DimensionRow key={d.name} d={d} />
            ))}
          </tbody>
        </table></div>
      ) : null}

      {/* Caliper entry per feature measurement */}
      {editable.length ? (
        <details className="mt-2 text-xs">
          <summary>Caliper values (mm)</summary>
          <div className="mt-2 space-y-1">
            {editable.map((e) => (
              <div
                key={`${e.featureId}.${e.field}`}
                className="caliper-row"
              >
                <span className="min-w-0 break-words text-muted">
                  {e.featureId}.{e.field}
                </span>
                <input
                  type="number"
                  step="0.01"
                  placeholder={
                    e.measurement.scan_mm != null
                      ? `scan ${e.measurement.scan_mm}`
                      : "mm"
                  }
                  disabled={caliperBusy}
                  aria-label={`${e.featureId} ${e.field} in millimetres`}
                  className="w-full min-w-0 rounded border border-line bg-paper px-1.5 py-0.5"
                  onKeyDown={(ev) => {
                    if (ev.key === "Enter") {
                      const v = parseFloat(
                        (ev.target as HTMLInputElement).value,
                      );
                      if (!Number.isNaN(v))
                        saveCaliper(e.featureId, e.field, v);
                    }
                  }}
                />
              </div>
            ))}
          </div>
        </details>
      ) : null}

      </details>
      {err ? <p className="mt-2 text-xs text-rose-800">{err}</p> : null}
    </div>
  );
}

export function PartsPanel({
  hiddenPartIds = [], onToggleHidden = () => {}, onIsolate = () => {},
  scanId,
  parts,
  partsReady,
  selectedId, onCombine,
  onChanged,
}: {
  hiddenPartIds?: string[];
  onToggleHidden?: (id: string) => void;
  onIsolate?: (id: string) => void;
  scanId: string;
  parts: Part[] | null;
  partsReady: boolean;
  selectedId: string | null;
  onCombine: () => void;
  onChanged: () => void;
}) {
  const [history, setHistory] = useState({can_undo: false, can_restore_example: false});
  const [busy, setBusy] = useState(false);
  const [editError, setEditError] = useState("");
  useEffect(() => {
    let alive = true;
    api.partHistory(scanId).then((h) => {if (alive) setHistory(h);}).catch(() => {});
    return () => { alive = false; };
  }, [scanId, parts]);
  async function recover(reset = false) {
    if (reset && !window.confirm("Restore the original example parts? This removes your part edits to this example.")) return;
    setBusy(true); setEditError("");
    try {
      if (reset) await api.restoreExample(scanId); else await api.undoMerge(scanId);
      onChanged();
    } catch (e) { setEditError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  if (!partsReady) {
    return (
      <div data-testid="parts-panel">
        <EmptyState
          title="Parts not ready yet"
          hint="They appear when the Parts stage finishes."
        />
      </div>
    );
  }
  if (parts === null) {
    return (
      <div data-testid="parts-panel" className="space-y-3">
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} className="h-24" />
        ))}
      </div>
    );
  }
  if (parts.length === 0) {
    return (
      <div data-testid="parts-panel">
        <EmptyState title="No parts detected" />
      </div>
    );
  }

  return (
    <div data-testid="parts-panel" className="space-y-3">
      {!selectedId && <details className="assembly-history"><summary>Assembly history</summary><div className="part-history"><button disabled={busy || !history.can_undo} onClick={() => recover()}>↶ Undo combine</button>{history.can_restore_example ? <button disabled={busy} onClick={() => recover(true)}>Restore original example</button> : null}</div></details>}
      {editError ? <p role="alert">{editError}</p> : null}
      {parts.map((p, i) => selectedId === p.part_id && (
        <PartCard
          key={p.part_id}
          part={p}
          hidden={hiddenPartIds.includes(p.part_id)} onToggleHidden={onToggleHidden} onIsolate={onIsolate}
          index={i}
          selected={selectedId === p.part_id}
          onChanged={onChanged}
          scanId={scanId}
          onCombine={onCombine}
        />
      ))}
    </div>
  );
}
