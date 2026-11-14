"use client";

import { ReactNode } from "react";

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} />;
}

export function EmptyState({
  title,
  hint,
}: {
  title: string;
  hint?: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-line p-8 text-center text-muted">
      <p className="font-medium text-ink">{title}</p>
      {hint ? <p className="mt-1 text-sm">{hint}</p> : null}
    </div>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="rounded-md border border-amber-600/40 bg-amber-100 p-3 text-sm text-amber-800">
      {message}
    </div>
  );
}

export function Badge({
  children,
  tone = "slate",
}: {
  children: ReactNode;
  tone?: "slate" | "green" | "red" | "amber" | "sky";
}) {
  const tones: Record<string, string> = {
    slate: "bg-wash text-ink",
    green: "bg-emerald-100 text-emerald-800",
    red: "bg-rose-100 text-rose-800",
    amber: "bg-amber-100 text-amber-800",
    sky: "bg-sky-100 text-sky-800",
  };
  return (
    <span
      className={`inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-muted">
      <span className="h-3 w-3 animate-spin rounded-full border-2 border-line border-t-accent" />
      {label}
    </div>
  );
}
