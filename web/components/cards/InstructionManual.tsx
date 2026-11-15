"use client";

import { useContext, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { Part, RepairPlan } from "@/lib/contracts.gen";
import { assetUrl } from "@/lib/storage";
import { ManualContext } from "./ManualContext";

function PartFigure({ part }: { part: Part }) {
  const [failed, setFailed] = useState(false);
  return <figure>
    {part.thumbnail_uri && !failed
      // Scan thumbnails can be signed URLs on external storage.
      // eslint-disable-next-line @next/next/no-img-element
      ? <img src={assetUrl(part.thumbnail_uri)} alt={part.label} onError={() => setFailed(true)} />
      : <span>Preview unavailable</span>}
    <figcaption>{part.label}</figcaption>
  </figure>;
}

export function InstructionManual({ plan }: { plan: RepairPlan }) {
  const { name, parts } = useContext(ManualContext);
  const [open, setOpen] = useState(false);
  const [views, setViews] = useState<Map<string, string> | null>(null);
  const selections = useMemo(() => plan.steps.map(step => [...new Set([
    ...(step.part_ids ?? []), ...Array.from(step.text.matchAll(/\[\[part:([^\]]+)\]\]/g), m => m[1]),
  ])].sort()), [plan.steps]);
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setViews(null);
    import("./manualIllustrations").then(m => m.renderManualViews(parts, selections, controller.signal))
      .then(images => { if (!controller.signal.aborted) setViews(images); })
      .catch(() => { if (!controller.signal.aborted) setViews(new Map()); });
    return () => controller.abort();
  }, [open, parts, selections]);
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (open) dialog.current?.showModal();
  }, [open]);
  const byId = new Map(parts.map(p => [p.part_id, p]));
  const readable = (text: string) => text.replace(/\s*\[\[part:([^\]]+)\]\]/g, (match, id, offset) => {
    const label = byId.get(id)?.label ?? id;
    return text.slice(0, offset).toLowerCase().endsWith(label.toLowerCase()) ? "" : match.replace(/\[\[part:[^\]]+\]\]/, label);
  });
  return <>
    <button className="manual-launch" onClick={() => setOpen(true)}>Open illustrated manual</button>
    {open && createPortal(<dialog ref={dialog} className="instruction-manual" aria-label="Instruction manual"
      onClose={() => setOpen(false)}>
      <nav><button onClick={() => dialog.current?.close()}>Back to workspace</button>
        <button onClick={() => window.print()}>Print / Save PDF</button></nav>
      <article>
        <header><small>{plan.intent} manual</small><h1>{name}</h1><p>{readable(plan.diagnosis.summary)}</p></header>
        {!!plan.safety?.length && <section><h2>Before you start</h2><ul>{plan.safety.map((s,i) => <li key={i}>{readable(s)}</li>)}</ul></section>}
        {!!plan.tools?.length && <section><h2>Tools</h2><p>{plan.tools.join(" · ")}</p></section>}
        <section><h2>Parts and materials</h2>{plan.bom?.length
          ? <ul>{plan.bom.map((item,i) => <li key={i}><b>{item.qty} × {item.name}</b> {item.spec}</li>)}</ul>
          : <p>No purchase items identified in this plan.</p>}
          {!!plan.printable?.length && <p>Printable designs: {plan.printable.join(", ")}</p>}</section>
        {plan.steps.map((step, index) => {
          const ids = new Set(selections[index]);
          const image = views?.get(JSON.stringify(selections[index]));
          const shown = ids.size ? parts.filter(p => ids.has(p.part_id)) : parts;
          return <section className="manual-step" key={step.n}>
            <h2>Step {step.n}</h2><p>{readable(step.text)}</p>
            {image && <figure className="manual-assembly">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={image} alt={ids.size ? "Assembly with step components highlighted" : "Assembly overview"} />
              <figcaption>{ids.size ? shown.map(p => p.label).join(" · ") : "Assembly overview"}</figcaption>
            </figure>}
            {!image && !!shown.length && <div className="manual-figures" aria-label={ids.size ? "Parts for this step" : "Assembly parts overview"}>
              {shown.map(part => <PartFigure key={part.part_id} part={part} />)}
            </div>}
          </section>;
        })}
      </article>
    </dialog>, document.body)}
  </>;
}
