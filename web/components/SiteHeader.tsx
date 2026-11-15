"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { api } from "@/lib/api";
import { notifyAccountChange, useAccount } from "./AccountProvider";

export function SiteHeader() {
  const path = usePathname();
  const account = useAccount();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  if (path === "/login") return null;
  async function logout() {
    setBusy(true);
    setError("");
    try {
      await api.logout();
      notifyAccountChange();
      window.location.assign("/login");
    } catch {
      setError("Couldn't sign out. Try again.");
      setBusy(false);
    }
  }
  const accountControls = <div className="account-controls">
    <span className="account-name">{account?.username}</span>
    <button type="button" onClick={logout} disabled={busy}>{busy ? "Signing out…" : "Sign out"}</button>
    {error && <span role="alert">{error}</span>}
  </div>;
  if (path.startsWith("/scan/")) return null;
  return (
    <header className={`site-header ${path.startsWith("/scan/") && path !== "/scan/new" ? "site-header-workbench" : ""}`}>
      <Link href="/" className="wordmark" aria-label="Screw It home">Screw It<span aria-hidden="true">.</span></Link>
      <nav aria-label="Main navigation">
        <Link href="/#builds" aria-current={path === "/" ? "page" : undefined}>My builds</Link>
        <Link href="/scan/new" aria-current={path === "/scan/new" ? "page" : undefined}>New scan <span aria-hidden="true">↗</span></Link>
      </nav>
      {accountControls}
    </header>
  );
}
