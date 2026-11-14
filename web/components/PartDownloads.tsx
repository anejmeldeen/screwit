"use client";

import { useState } from "react";
import { API_BASE } from "@/lib/env";
import type { Part } from "@/lib/contracts.gen";

export function PartDownloads({ scanId, part }: { scanId: string; part?: Part }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState("");
  async function download(format: "stp" | "stl") {
    setBusy(format); setError("");
    try {
      const response = await fetch(`${API_BASE}/scans/${encodeURIComponent(scanId)}${part ? `/parts/${encodeURIComponent(part.part_id)}` : ""}/export?format=${format}`);
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || "Download failed. Please try again.");
      }
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = `${(part?.label ?? scanId).replace(/[^a-z0-9_-]+/gi, "-") || "part"}.${format}`;
      document.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) { setError(e instanceof Error ? e.message : "Download failed."); }
    finally { setBusy(null); }
  }
  return <div className="part-downloads">
    <p className="export-heading">{part ? `Export ${part.label}` : "Export whole object"}</p>
    <div className="flex flex-wrap gap-2">
      <button disabled={!!busy} onClick={() => download("stp")}>{busy === "stp" ? "Preparing STEP…" : "Download STEP (.stp)"}</button>
      <button disabled={!!busy} onClick={() => download("stl")}>{busy === "stl" ? "Preparing STL…" : "Download STL"}</button>
    </div>
    {error ? <p role="alert">{error}</p> : null}
  </div>;
}
