"use client";

import { useRef } from "react";

export function ResizeHandle({ label, orientation, value, min, max, reverse = false, onChange, className }: {
  label: string;
  orientation: "horizontal" | "vertical";
  value: number;
  min: number;
  max: number;
  reverse?: boolean;
  onChange: (value: number) => void;
  className: string;
}) {
  const drag = useRef<{ position: number; value: number } | null>(null);
  const change = (next: number) => onChange(Math.round(Math.min(max, Math.max(min, next))));
  return <div role="separator" tabIndex={0} aria-label={label} aria-orientation={orientation}
    aria-valuemin={min} aria-valuemax={max} aria-valuenow={Math.round(value)}
    aria-valuetext={`${Math.round(value)} pixels`} title={`${label} · drag or use arrow keys`}
    className={`resize-handle ${className}`} data-orientation={orientation}
    onPointerDown={e => {
      if (e.button !== 0) return;
      e.preventDefault();
      e.currentTarget.focus();
      e.currentTarget.setPointerCapture(e.pointerId);
      drag.current = { position: orientation === "vertical" ? e.clientX : e.clientY, value };
    }}
    onPointerMove={e => {
      if (!drag.current) return;
      const position = orientation === "vertical" ? e.clientX : e.clientY;
      change(drag.current.value + (position - drag.current.position) * (reverse ? -1 : 1));
    }}
    onPointerUp={e => { drag.current = null; e.currentTarget.releasePointerCapture(e.pointerId); }}
    onLostPointerCapture={() => { drag.current = null; }}
    onPointerCancel={() => { drag.current = null; }}
    onKeyDown={e => {
      const backward = orientation === "vertical" ? "ArrowLeft" : "ArrowUp";
      const forward = orientation === "vertical" ? "ArrowRight" : "ArrowDown";
      if (![backward, forward, "Home", "End"].includes(e.key)) return;
      e.preventDefault();
      if (e.key === "Home") change(min);
      else if (e.key === "End") change(max);
      else change(value + (e.key === forward ? 20 : -20) * (reverse ? -1 : 1));
    }}><span aria-hidden="true" /></div>;
}
