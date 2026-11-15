"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { api } from "@/lib/api";
import { DashboardPreferences } from "@/lib/dashboardPreferences";
import { notifyAccountChange, useAccount } from "./AccountProvider";
import { DashboardIcon, type DashboardIconName } from "./DashboardIcon";
import { ScrewMascot } from "./ScrewMascot";
import { SiteHeader } from "./SiteHeader";

const sections: { href: string; title: string; icon: DashboardIconName }[] = [
  { href: "/", title: "Workspaces", icon: "dashboard" },
  { href: "/settings", title: "Settings", icon: "settings" },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  if (!sections.some((section) => section.href === path)) {
    return <><SiteHeader /><main id="main-content">{children}</main></>;
  }
  return <DashboardPreferences><DashboardShell>{children}</DashboardShell></DashboardPreferences>;
}

function DashboardShell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const account = useAccount();
  const [menuOpen, setMenuOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function logout() {
    setBusy(true);
    try {
      await api.logout();
      notifyAccountChange();
      window.location.assign("/login");
    } catch { setError("Couldn't sign out. Please try again."); setBusy(false); }
  }
  return (
    <div className="dashboard-shell">
      <aside className="dashboard-sidebar" aria-label="Workspace navigation">
        <div className="sidebar-brand">
          <Link href="/" className="sidebar-wordmark" aria-label="Screw It workspaces">Screw It<span>.</span>
            <svg className="sidebar-screw" viewBox="0 0 124 16" aria-hidden="true">
              <path fill="currentColor" d="M3 1h5l5 4h97l12 3-12 3H13l-5 4H3Q1 15 1 13V3q0-2 2-2Z" />
              <path stroke="#27281f" strokeWidth="1.5" d="M5 4v8" />
              {Array.from({ length: 14 }, (_, i) => <path key={i} d={`M${24 + i * 6} 4l-3 8`} stroke="currentColor" strokeWidth="2" />)}
              {Array.from({ length: 14 }, (_, i) => <path key={i} d={`M${26 + i * 6} 5l-2.5 6`} stroke="#27281f" strokeWidth="1.4" />)}
            </svg>
          </Link>
          <button className="sidebar-menu" type="button" aria-label="Toggle navigation" aria-expanded={menuOpen} aria-controls="sidebar-navigation" onClick={() => setMenuOpen(!menuOpen)}><DashboardIcon name="menu" /></button>
        </div>
        <div id="sidebar-navigation" className={`sidebar-content ${menuOpen ? "is-open" : ""}`}>
          <nav aria-label="Main navigation">{sections.map((section) => (
            <Link key={section.href} href={section.href} aria-current={path === section.href ? "page" : undefined} onClick={() => setMenuOpen(false)}>
              <DashboardIcon name={section.icon} />{section.title}{path === section.href && <span className="sidebar-active-dot" />}
            </Link>
          ))}</nav>
          <div className="sidebar-bottom">
            <div className="sidebar-mascot"><ScrewMascot /></div>
            <Link href="/settings" className="sidebar-account" onClick={() => setMenuOpen(false)}>
              <span className="account-avatar">{account?.username.slice(0, 1).toUpperCase()}</span>
              <span><strong>{account?.username}</strong><small>Personal workspace</small></span>
              <DashboardIcon name="settings" />
            </Link>
            <button type="button" className="sidebar-signout" onClick={logout} disabled={busy}><DashboardIcon name="logout" />{busy ? "Signing out…" : "Sign out"}</button>
            {error && <p className="sidebar-error" role="alert">{error}</p>}
          </div>
        </div>
      </aside>
      <main id="main-content" className="dashboard-main">
        {children}
      </main>
    </div>
  );
}
