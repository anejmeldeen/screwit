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
          <Link href="/" className="sidebar-wordmark" aria-label="Screw It workspaces">Screw It<span>.</span></Link>
          <button className="sidebar-menu" type="button" aria-label="Toggle navigation" aria-expanded={menuOpen} aria-controls="sidebar-navigation" onClick={() => setMenuOpen(!menuOpen)}><DashboardIcon name="menu" /></button>
        </div>
        <div id="sidebar-navigation" className={`sidebar-content ${menuOpen ? "is-open" : ""}`}>
          <p className="sidebar-label">Your workshop</p>
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
        {path !== "/" && <div className="dashboard-topline"><span>YOUR WORKSHOP / {sections.find((section) => section.href === path)?.title.toUpperCase()}</span><span className="workspace-owner">{account?.username}&apos;s workspace</span></div>}
        {children}
        {path !== "/" && <footer className="dashboard-footer"><span>Screw It. Let&apos;s build.</span><span>HackUMass 2026</span></footer>}
      </main>
    </div>
  );
}
