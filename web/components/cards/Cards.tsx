"use client";

import { useState } from "react";
import { assetUrl } from "@/lib/storage";
import { Badge } from "@/components/ui";
import { PartText } from "./PartText";

// Card payloads. The SSE wire delivers a card as {type, payload}; these render
// each payload shape (SPEC 4.5 repair_plan, 4.7 cad_result, parts_to_buy BOM).
// `onPartChip` wires [[part:pX]] references to the viewer.

type AnyRecord = Record<string, unknown>;

export function RepairPlanCard({
  payload,
  onPartChip,
}: {
  payload: AnyRecord;
  onPartChip: (id: string) => void;
}) {
  const steps = (payload.steps as AnyRecord[]) ?? [];
  const safety = (payload.safety as string[]) ?? [];
  const tools = (payload.tools as string[]) ?? [];
  const diagnosis = payload.diagnosis as AnyRecord | undefined;

  return (
    <div
      data-testid="chat-card-repair-plan"
      className="rounded-lg border border-line bg-wash p-3"
    >
      <div className="mb-2 flex items-center gap-2">
        <Badge tone="sky">Repair plan</Badge>
        {diagnosis?.confidence != null ? (
          <span className="text-xs text-muted">
            confidence {(Number(diagnosis.confidence) * 100).toFixed(0)}%
          </span>
        ) : null}
      </div>
      {diagnosis?.summary ? (
        <p className="mb-2 text-sm text-ink">{String(diagnosis.summary)}</p>
      ) : null}

      {safety.length ? (
        <div className="mb-2 rounded border border-amber-600/40 bg-amber-100 p-2 text-xs text-amber-800">
          <p className="font-semibold">Safety</p>
          <ul className="list-disc pl-4">
            {safety.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {tools.length ? (
        <p className="mb-2 text-xs text-muted">Tools: {tools.join(", ")}</p>
      ) : null}

      <ol className="space-y-1.5">
        {steps.map((step, i) => (
          <li
            key={i}
            data-testid="repair-step"
            className="flex items-start gap-2 text-sm"
          >
            <input type="checkbox" className="mt-1" />
            <span>
              <PartText
                text={String(step.text ?? "")}
                onPartChip={onPartChip}
              />
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export function PartsToBuyCard({ payload }: { payload: AnyRecord[] }) {
  const items = payload ?? [];
  const total = items.reduce(
    (sum, it) =>
      sum + Number(it.est_price_usd ?? 0) * Number(it.qty ?? 1),
    0,
  );
  return (
    <div
      data-testid="chat-card-parts-to-buy"
      className="rounded-lg border border-line bg-wash p-3"
    >
      <div className="mb-2">
        <Badge tone="green">Parts to buy</Badge>
      </div>
      <ul className="space-y-2">
        {items.map((it, i) => {
          const links = (it.buy_links as AnyRecord[]) ?? [];
          return (
            <li key={i} className="text-sm">
              <div className="flex justify-between">
                <span className="font-medium">
                  {String(it.name)} ×{Number(it.qty ?? 1)}
                </span>
                <span className="text-ink">
                  ${Number(it.est_price_usd ?? 0).toFixed(2)}
                </span>
              </div>
              {it.spec ? (
                <p className="text-xs text-muted">{String(it.spec)}</p>
              ) : null}
              {links.length ? (
                <div className="mt-1 flex flex-wrap gap-2">
                  {links.map((l, j) => (
                    <a
                      key={j}
                      href={String(l.url)}
                      target="_blank"
                      rel="noreferrer"
                      className="text-xs text-accent underline"
                    >
                      {String(l.vendor ?? "Buy")}
                    </a>
                  ))}
                </div>
              ) : null}
            </li>
          );
        })}
      </ul>
      <p className="mt-2 border-t border-line pt-2 text-right text-sm font-semibold">
        Total: ${total.toFixed(2)}
      </p>
    </div>
  );
}

export function CadResultCard({
  payload,
  onShowInViewer,
}: {
  payload: AnyRecord;
  onShowInViewer: (cad: AnyRecord) => void;
}) {
  const checks = (payload.checks as AnyRecord[]) ?? [];
  const required = checks.filter((c) => c.kind === "required");
  const advisory = checks.filter((c) => c.kind === "advisory");
  const files = (payload.files as AnyRecord) ?? {};
  const print = (payload.print_settings as AnyRecord) ?? {};
  const stale = !!payload.stale;

  return (
    <div
      data-testid="chat-card-cad-result"
      className="rounded-lg border border-orange-700/60 bg-wash p-3"
    >
      <div className="mb-2 flex items-center gap-2">
        <Badge tone="amber">CAD result</Badge>
        <span
          className={`text-xs ${
            payload.status === "ok" ? "text-emerald-800" : "text-rose-800"
          }`}
        >
          {String(payload.status)}
        </span>
        {stale ? (
          <span data-testid="cad-stale-banner">
            <Badge tone="red">STALE — re-run before printing</Badge>
          </span>
        ) : null}
      </div>

      {files.preview ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={assetUrl(String(files.preview))}
          alt="CAD preview"
          className="mb-2 max-h-40 rounded border border-line bg-wash object-contain"
          onError={(e) =>
            ((e.currentTarget as HTMLImageElement).style.display = "none")
          }
        />
      ) : null}

      {/* Required checks: pass/fail */}
      <div className="mb-2">
        <p className="text-xs font-semibold text-ink">Required checks</p>
        <ul className="mt-1 space-y-0.5 text-xs">
          {required.map((c, i) => (
            <li
              key={i}
              data-testid={`cad-check-${String(c.name)}`}
              data-result={String(c.result)}
              className="flex items-center justify-between gap-2"
            >
              <span className="text-ink">{String(c.name)}</span>
              <span className="flex items-center gap-2">
                {c.input_source ? (
                  <span className="text-muted">
                    {String(c.input_source)}
                  </span>
                ) : null}
                {c.value != null ? (
                  <span className="text-muted">
                    {Number(c.value)}
                    {c.limit != null ? ` / ${Number(c.limit)}` : ""}
                  </span>
                ) : null}
                <Badge tone={c.result === "pass" ? "green" : "red"}>
                  {String(c.result)}
                </Badge>
              </span>
            </li>
          ))}
        </ul>
      </div>

      {/* Advisory checks: values, incl not_computed reasons */}
      {advisory.length ? (
        <details className="mb-2 text-xs">
          <summary className="cursor-pointer text-muted">
            Advisory checks
          </summary>
          <ul className="mt-1 space-y-0.5">
            {advisory.map((c, i) => (
              <li
                key={i}
                data-testid={`cad-check-${String(c.name)}`}
                data-result={String(c.result)}
                className="flex items-center justify-between gap-2"
              >
                <span className="text-muted">{String(c.name)}</span>
                <span className="text-muted">
                  {c.result === "not_computed"
                    ? `not computed — ${String(c.reason ?? "")}`
                    : `${c.value != null ? Number(c.value) : ""}${
                        c.limit != null ? ` / ${Number(c.limit)}` : ""
                      } (${String(c.result)})`}
                </span>
              </li>
            ))}
          </ul>
        </details>
      ) : null}

      {/* Print settings */}
      <div className="mb-2 text-xs text-muted">
        {print.material ? <span>{String(print.material)} · </span> : null}
        {print.layer_height_mm ? (
          <span>{String(print.layer_height_mm)} mm layers · </span>
        ) : null}
        {print.infill_pct ? <span>{String(print.infill_pct)}% infill · </span> : null}
        {print.est_time_min ? <span>~{String(print.est_time_min)} min</span> : null}
      </div>

      <div className="flex flex-wrap gap-2 text-xs">
        {files.stl ? (
          <a
            href={assetUrl(String(files.stl))}
            download
            className="rounded border border-line px-2 py-1 hover:border-line"
          >
            Download STL
          </a>
        ) : null}
        {files.step ? (
          <a
            href={assetUrl(String(files.step))}
            download
            className="rounded border border-line px-2 py-1 hover:border-line"
          >
            Download STEP
          </a>
        ) : null}
        <button
          onClick={() => onShowInViewer(payload)}
          className="rounded border border-orange-600 px-2 py-1 text-orange-800 hover:bg-orange-100"
        >
          Show in viewer
        </button>
      </div>
      {payload.notes ? (
        <p className="mt-2 text-xs text-muted">{String(payload.notes)}</p>
      ) : null}
    </div>
  );
}

// Render text with [[part:pX]] chips as clickable buttons.
// (Implementation lives in ./PartText and is imported above.)
