"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { assetUrl } from "@/lib/storage";
import type { DemoScan } from "@/lib/contracts.gen";
import { Skeleton } from "@/components/ui";
import { ScrewMascot } from "@/components/ScrewMascot";

export default function LandingPage() {
  const [demos, setDemos] = useState<DemoScan[] | null>(null);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let alive = true;
    setError(false);
    api.demoScans().then((d) => { if (alive) setDemos(d); })
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
            <a href="#examples" className="button-secondary">Try an example <span aria-hidden="true">↓</span></a>
          </div>
        </div>
        <ScrewMascot />
        <div className="hero-caption">Scan an object. Explore its parts. Figure it out.</div>
      </section>

      <section id="examples" className="examples-section" aria-labelledby="examples-title">
        <div className="section-heading"><h2 id="examples-title">Something to start with.</h2><span>Example scans</span></div>
        {error ? (
          <div className="examples-empty" role="status"><p>Examples aren&apos;t available right now.</p><button type="button" className="text-link" onClick={() => setAttempt((n) => n + 1)}>Try again ↗</button></div>
        ) : demos === null ? (
          <div className="example-grid" aria-label="Loading examples" aria-busy="true">{[0, 1].map((i) => <Skeleton key={i} className="h-48" />)}</div>
        ) : demos.length === 0 ? (
          <div className="examples-empty"><p>No examples yet.</p><Link href="/scan/new" className="text-link">Start your own scan ↗</Link></div>
        ) : (
          <div className="example-grid">{demos.map((demo) => (
            <Link key={demo.scan_id} href={`/scan/${demo.scan_id}`} className="example-card">
              <div className="example-image">
                <span className="thumbnail-fallback" aria-hidden="true">↗</span>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={assetUrl(demo.thumbnail_uri)} alt="" loading="lazy" onError={(e) => { e.currentTarget.style.visibility = "hidden"; }} />
              </div>
              <div className="example-title"><h3>{demo.title}</h3><span aria-hidden="true">↗</span></div>
            </Link>
          ))}</div>
        )}
      </section>
      <footer className="site-footer"><span>HackUMass 2026</span></footer>
    </div>
  );
}
