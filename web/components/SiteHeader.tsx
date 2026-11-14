"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function SiteHeader() {
  const path = usePathname();
  return (
    <header className={`site-header ${path.startsWith("/scan/") && path !== "/scan/new" ? "site-header-workbench" : ""}`}>
      <Link href="/" className="wordmark" aria-label="Screw It home">Screw It<span aria-hidden="true">.</span></Link>
      <nav aria-label="Main navigation">
        <Link href="/" aria-current={path === "/" ? "page" : undefined}>Home</Link>
        <Link href="/#examples">Examples</Link>
        <Link href="/scan/new" aria-current={path === "/scan/new" ? "page" : undefined}>New scan <span aria-hidden="true">↗</span></Link>
      </nav>
    </header>
  );
}
