"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type BuildSummary } from "@/lib/api";
import { useDashboardPreferences, type WorkspaceSort } from "@/lib/dashboardPreferences";
import { DashboardIcon } from "./DashboardIcon";
import { useAccount } from "./AccountProvider";
import { Skeleton } from "./ui";

function modifiedLabel(value: string) {
  const date = new Date(value);
  const minutes = Math.max(0, Math.floor((Date.now() - date.getTime()) / 60000));
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 1440) return `${Math.floor(minutes / 60)} hr ago`;
  if (minutes < 2880) return "Yesterday";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

export function WorkspaceDashboard({ examplesOnly = false }: { examplesOnly?: boolean }) {
  const account = useAccount();
  const { preferences, save, error: preferenceError } = useDashboardPreferences();
  const [builds, setBuilds] = useState<BuildSummary[] | null>(null);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [search, setSearch] = useState("");
  useEffect(() => {
    let alive = true;
    const refresh = () => api.builds().then((items) => {
      if (alive) { setBuilds(items); setError(false); }
    }).catch(() => { if (alive) setError(true); });
    refresh();
    window.addEventListener("focus", refresh);
    return () => { alive = false; window.removeEventListener("focus", refresh); };
  }, [attempt]);

  const [menu, setMenu] = useState<{ build: BuildSummary; x: number; y: number } | null>(null);
  const [actionError, setActionError] = useState("");
  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(null);
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") close(); };
    window.addEventListener("click", close);
    window.addEventListener("scroll", close, true);
    window.addEventListener("keydown", escape);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("keydown", escape);
    };
  }, [menu]);

  async function act(build: BuildSummary, action: "Rename" | "Delete") {
    setMenu(null);
    setActionError("");
    try {
      if (action === "Rename") {
        const title = window.prompt("Workspace name", build.title)?.trim();
        if (title === undefined || title === build.title) return;
        if (!title || title.length > 120) throw new Error("Enter a name between 1 and 120 characters.");
        await api.renameBuild(build.scan_id, title);
      } else {
        if (!window.confirm(`Delete “${build.title}” from your workspaces?`)) return;
        await api.deleteBuild(build.scan_id);
      }
      setAttempt(n => n + 1);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Could not update the workspace.");
    }
  }

  const buttons = (build: BuildSummary) => (["Rename", "Delete"] as const).map(action =>
    <button type="button" key={action} aria-label={`${action} ${build.title}`} onClick={() => act(build, action)}>{action}</button>);

  const workspaces = (builds || []).filter((build) => !examplesOnly || build.is_example);
  const visible = workspaces.filter((build) => build.title.toLowerCase().includes(search.trim().toLowerCase())).sort((a, b) => {
    if (preferences.sort === "name") return a.title.localeCompare(b.title);
    const field = preferences.sort === "created" ? "created_at" : "updated_at";
    return new Date(b[field]).getTime() - new Date(a[field]).getTime();
  });

  return <>
    <header className="dashboard-heading">
      <div><p className="dashboard-eyebrow">{examplesOnly ? "A little inspiration" : `Welcome back, ${account?.username || "builder"}`}</p>
        <h1>{examplesOnly ? "Examples." : "Dashboard"}</h1>
        <p>{examplesOnly ? "A few starting points for your next big idea." : "Open a saved scan or create a new workspace."}</p></div>
      <Link href="/scan/new" className="button-primary dashboard-new"><DashboardIcon name="plus" />New workspace</Link>
    </header>
    <section className="workspace-section" aria-labelledby="workspaces-title">
      <div className="workspace-section-heading"><h2 id="workspaces-title">{examplesOnly ? "Example workspaces" : "Your workspaces"}<span>{builds === null ? "—" : workspaces.length}</span></h2><span className="workspace-section-note">Saved scans and models</span></div>
      <div className="workspace-toolbar">
        <label className="workspace-search"><DashboardIcon name="search" /><span className="sr-only">Search workspaces</span><input type="search" placeholder="Find a workspace…" value={search} onChange={(event) => setSearch(event.target.value)} /></label>
        <label className="workspace-sort"><span>Sort by</span><select aria-label="Sort workspaces" value={preferences.sort} onChange={(event) => save({ sort: event.target.value as WorkspaceSort })}><option value="modified">Last modified</option><option value="name">Name</option><option value="created">Date created</option></select></label>
      </div>
      {actionError && <p role="alert">{actionError}</p>}
      {menu && <div className="workspace-context" aria-label="Workspace actions" role="group" style={{ left: menu.x, top: menu.y }}>{buttons(menu.build)}</div>}
      {preferenceError && <p className="settings-caption" role="status">{preferenceError}</p>}
      <div className={`workspace-list ${preferences.compact ? "is-compact" : ""}`}>
        <div className="workspace-columns" aria-hidden="true"><span>Workspace</span><span>Actions</span><span>Last modified</span><span /></div>
        {error ? <div className="workspace-empty" role="alert"><DashboardIcon name="box" /><h3>We couldn&apos;t load your workspaces.</h3><p>Give it another try in a moment.</p><button type="button" className="text-link" onClick={() => { setError(false); setAttempt((n) => n + 1); }}>Try again ↗</button></div>
          : builds === null ? <div className="workspace-skeletons" role="status" aria-label="Loading workspaces">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-20" />)}</div>
          : visible.length === 0 ? <div className="workspace-empty"><DashboardIcon name={search ? "search" : "box"} /><h3>{search ? "No workspaces found." : examplesOnly ? "No examples in this account yet." : "Your next idea starts here."}</h3><p>{search ? "Try a different name or clear your search." : "Scan an object to make your first workspace."}</p>{search ? <button className="text-link" onClick={() => setSearch("")}>Clear search</button> : <Link href="/scan/new" className="button-primary"><DashboardIcon name="plus" />New workspace</Link>}</div>
          : <ul aria-label="Workspaces">{visible.map((build) => <li key={build.scan_id} className="workspace-row" onContextMenu={event => {
            event.preventDefault();
            setMenu({ build, x: Math.max(0, Math.min(event.clientX, window.innerWidth - 148)), y: Math.max(0, Math.min(event.clientY, window.innerHeight - 96)) });
          }}>
              <Link href={`/scan/${build.scan_id}`} className="workspace-identity workspace-row-link"><span className={`workspace-icon ${build.is_example ? "is-example" : ""}`}><DashboardIcon name={build.is_example || build.input_type === "mesh" ? "box" : "camera"} /></span><span><strong>{build.title}</strong><small>{build.is_example ? "Example workspace" : `${build.input_type.charAt(0).toUpperCase() + build.input_type.slice(1)} scan`}</small></span></Link>
              <span className="workspace-actions">{buttons(build)}</span>
              <time dateTime={build.updated_at} title={new Date(build.updated_at).toLocaleString()}><span className="mobile-modified-label">Modified </span>{modifiedLabel(build.updated_at)}</time>
              <DashboardIcon name="arrow" className="workspace-open" />
          </li>)}</ul>}
      </div>
      {builds !== null && !error && visible.length > 0 && <p className="workspace-list-footer">{visible.length} workspace{visible.length === 1 ? "" : "s"}{search ? ` matching “${search}”` : ""}</p>}
    </section>
  </>;
}
