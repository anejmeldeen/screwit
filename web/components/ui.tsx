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
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-slate-700 p-8 text-center text-slate-400">
      <p className="font-medium text-slate-200">{title}</p>
      {hint ? <p className="mt-1 text-sm">{hint}</p> : null}
    </div>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="rounded-md border border-amber-600/40 bg-amber-900/20 p-3 text-sm text-amber-200">
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
    slate: "bg-slate-700 text-slate-200",
    green: "bg-emerald-800/60 text-emerald-200",
    red: "bg-rose-800/60 text-rose-200",
    amber: "bg-amber-800/60 text-amber-200",
    sky: "bg-sky-800/60 text-sky-200",
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
    <div className="flex items-center gap-2 text-sm text-slate-400">
      <span className="h-3 w-3 animate-spin rounded-full border-2 border-slate-500 border-t-accent" />
      {label}
    </div>
  );
}
