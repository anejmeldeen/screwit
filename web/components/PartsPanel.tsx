"use client";

import { useState } from "react";
import type { Part } from "@/lib/contracts.gen";
import {
  editableMeasurements,
  resolveAllDimensions,
  ResolvedDimension,
} from "@/lib/parts";
import { api, ApiError } from "@/lib/api";
import { assetUrl } from "@/lib/storage";
import { partColor } from "@/lib/palette";
import { Badge, EmptyState, Skeleton } from "@/components/ui";

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
      <td className="px-2 py-1 text-left text-ink">{d.name}</td>
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
  onSelect,
  onChanged,
  scanId,
  mergeState,
}: {
  part: Part;
  index: number;
  selected: boolean;
  onSelect: (id: string) => void;
  onChanged: () => void;
  scanId: string;
  mergeState: {
    pick: string | null;
    setPick: (id: string | null) => void;
  };
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

  async function doMerge(otherId: string) {
    setErr(null);
    try {
      await api.mergeParts(scanId, {
        part_ids: [part.part_id, otherId],
      } as never);
      mergeState.setPick(null);
      onChanged();
    } catch (e) {
      setErr(e instanceof ApiError ? `${e.message} (${e.status})` : String(e));
    }
  }

  return (
    <div
      data-testid={`part-row-${part.part_id}`}
      className={`rounded-lg border bg-panel p-3 ${
        selected ? "border-accent" : "border-line"
      }`}
    >
      <button
        className="flex w-full items-start gap-3 text-left"
        onClick={() => onSelect(part.part_id)}
      >
        <span
          className="mt-1 h-3 w-3 shrink-0 rounded-sm"
          style={{ backgroundColor: partColor(index) }}
        />
        <span className="h-12 w-12 shrink-0 overflow-hidden rounded bg-wash">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={assetUrl(part.thumbnail_uri)}
            alt={part.label}
            loading="lazy"
            decoding="async"
            className="h-full w-full object-cover"
            onError={(e) =>
              ((e.currentTarget as HTMLImageElement).style.visibility = "hidden")
            }
          />
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-2">
            <span className="truncate font-medium">{part.label}</span>
            <Badge tone="slate">
              {(part.label_confidence * 100).toFixed(0)}%
            </Badge>
          </span>
          {part.description ? (
            <span className="mt-0.5 block truncate text-xs text-muted">
              {part.description}
            </span>
          ) : null}
        </span>
      </button>

      {/* Actions */}
      <div className="mt-2 flex flex-wrap gap-2 text-xs">
        {renaming ? (
          <span className="flex items-center gap-1">
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
        {mergeState.pick === null ? (
          <button
            onClick={() => mergeState.setPick(part.part_id)}
            className="rounded border border-line px-1.5 py-0.5 hover:border-line"
          >
            Merge…
          </button>
        ) : mergeState.pick === part.part_id ? (
          <span className="text-muted">pick another part to merge into</span>
        ) : (
          <button
            onClick={() => doMerge(mergeState.pick!)}
            className="rounded border border-amber-600 px-1.5 py-0.5 text-amber-800"
          >
            Merge with {mergeState.pick}
          </button>
        )}
      </div>

      {/* Dimensions table */}
      {dims.length ? (
        <table className="mt-3 w-full text-xs">
          <thead className="text-muted">
            <tr>
              <th className="px-2 py-1 text-left font-normal">dimension (mm)</th>
              <th className="px-2 py-1 text-right font-normal">scan</th>
              <th className="px-2 py-1 text-right font-normal">snapped</th>
              <th className="px-2 py-1 text-right font-normal">caliper</th>
            </tr>
          </thead>
          <tbody>
            {dims.map((d) => (
              <DimensionRow key={d.name} d={d} />
            ))}
          </tbody>
        </table>
      ) : null}

      {/* Caliper entry per feature measurement */}
      {editable.length ? (
        <details className="mt-2 text-xs">
          <summary className="cursor-pointer text-muted">
            Enter caliper value
          </summary>
          <div className="mt-2 space-y-1">
            {editable.map((e) => (
              <div
                key={`${e.featureId}.${e.field}`}
                className="flex items-center gap-2"
              >
                <span className="w-40 text-muted">
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
                  className="w-24 rounded border border-line bg-paper px-1.5 py-0.5"
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

      {err ? <p className="mt-2 text-xs text-rose-800">{err}</p> : null}
    </div>
  );
}

export function PartsPanel({
  scanId,
  parts,
  partsReady,
  selectedId,
  onSelect,
  onChanged,
}: {
  scanId: string;
  parts: Part[] | null;
  partsReady: boolean;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onChanged: () => void;
}) {
  const [mergePick, setMergePick] = useState<string | null>(null);

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
      {parts.map((p, i) => (
        <PartCard
          key={p.part_id}
          part={p}
          index={i}
          selected={selectedId === p.part_id}
          onSelect={onSelect}
          onChanged={onChanged}
          scanId={scanId}
          mergeState={{ pick: mergePick, setPick: setMergePick }}
        />
      ))}
    </div>
  );
}
