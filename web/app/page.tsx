"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type BuildSummary } from "@/lib/api";
import { ExamplePreview } from "@/components/ExamplePreview";
import { Skeleton } from "@/components/ui";
import { ScrewMascot } from "@/components/ScrewMascot";

export default function LandingPage() {
  const [demos, setDemos] = useState<BuildSummary[] | null>(null);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let alive = true;
    setError(false);
    api.builds().then((d) => { if (alive) setDemos(d); })
      .catch(() => { if (alive) setError(true); });
    return () => { alive = false; };
  }, [attempt]);

  return (
    <div className="home-page">
      <section className="home-hero" aria-labelledby="home-title">
        <div className="hero-copy">
          <h1 id="home-title">Screw It.</h1>
          <p className="hero-subtitle">Let&apos;s build.</p>
          <div className="hero-actions">
            <Link href="/scan/new" className="button-primary">Scan something <span aria-hidden="true">↗</span></Link>
            <a href="#builds" className="button-secondary">Your builds <span aria-hidden="true">↓</span></a>
          </div>
        </div>
        <ScrewMascot />
      </section>

      <section id="builds" className="examples-section" aria-labelledby="builds-title">
        <div className="section-heading"><h2 id="builds-title">Your builds.</h2><span>Your ideas, taking shape.</span></div>
        {error ? (
          <div className="examples-empty" role="status"><p>Your builds aren&apos;t available right now.</p><button type="button" className="text-link" onClick={() => setAttempt((n) => n + 1)}>Try again ↗</button></div>
        ) : demos === null ? (
          <div className="example-grid" aria-label="Loading builds" aria-busy="true">{[0, 1].map((i) => <Skeleton key={i} className="h-48" />)}</div>
        ) : demos.length === 0 ? (
          <div className="examples-empty"><p>A fresh start. Make your first build.</p><Link href="/scan/new" className="text-link">Scan something ↗</Link></div>
        ) : (
          <div className="example-grid">{demos.map((demo) => (
            <Link key={demo.scan_id} href={`/scan/${demo.scan_id}`} className="example-card">
              <div className="example-image">
                {demo.is_example ? <ExamplePreview scanId={demo.scan_id} /> : <span className="build-placeholder" aria-hidden="true">↗</span>}
              </div>
              <div className="example-title"><h3>{demo.title}</h3><span aria-hidden="true">↗</span></div>
              <div className="build-meta"><span>{demo.is_example ? "Example build" : "Your scan"}</span><span>{demo.status}</span></div>
            </Link>
          ))}</div>
        )}
      </section>
      <footer className="site-footer"><span>HackUMass 2026</span></footer>
    </div>
  );
}
