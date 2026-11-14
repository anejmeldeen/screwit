"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { assetUrl } from "@/lib/storage";
import type { DemoScan } from "@/lib/contracts.gen";
import { EmptyState, ErrorNote, Skeleton } from "@/components/ui";

export default function LandingPage() {
  const [demos, setDemos] = useState<DemoScan[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .demoScans()
      .then((d) => alive && setDemos(d))
      .catch((e) => alive && setError(String(e?.message ?? e)));
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div className="mx-auto max-w-7xl px-4 py-12">
      <section className="mb-12 max-w-2xl">
        <h1 className="text-4xl font-bold tracking-tight">
          Something broke. Let&apos;s understand it.
        </h1>
        <p className="mt-3 text-lg text-slate-300">
          Scan a broken object with your phone. Screw It reconstructs it in 3D,
          names the parts, measures them, and walks you through a fix — or prints
          a replacement.
        </p>
        <Link
          href="/scan/new"
          className="mt-6 inline-flex rounded-md bg-accent px-5 py-2.5 font-medium text-ink hover:bg-sky-300"
        >
          Start a scan
        </Link>
      </section>

      <section>
        <h2 className="mb-4 text-xl font-semibold">Demo gallery</h2>
        {/* The shell renders even when the API is down. */}
        {error ? (
          <ErrorNote
            message={`Could not load demo scans (${error}). The page still works; try a scan or check the API server.`}
          />
        ) : demos === null ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-48" />
            ))}
          </div>
        ) : demos.length === 0 ? (
          <EmptyState
            title="No demo scans yet"
            hint="Process a hero scan to populate the gallery."
          />
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {demos.map((d) => (
              <Link
                key={d.scan_id}
                href={`/scan/${d.scan_id}`}
                className="group overflow-hidden rounded-lg border border-slate-800 bg-panel transition hover:border-accent"
              >
                <div className="aspect-video w-full overflow-hidden bg-slate-800">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={assetUrl(d.thumbnail_uri)}
                    alt={d.title}
                    className="h-full w-full object-cover opacity-90 transition group-hover:opacity-100"
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).style.visibility =
                        "hidden";
                    }}
                  />
                </div>
                <div className="flex items-center justify-between p-3">
                  <span className="font-medium">{d.title}</span>
                  <span className="rounded bg-slate-700 px-1.5 py-0.5 text-xs text-slate-300">
                    Scenario {d.scenario}
                  </span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
