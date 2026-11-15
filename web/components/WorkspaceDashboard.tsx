"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type BuildSummary } from "@/lib/api";
import { useDashboardPreferences, type WorkspaceSort } from "@/lib/dashboardPreferences";
import { DashboardIcon } from "./DashboardIcon";
import { Skeleton } from "./ui";

const WorkspaceSpin = dynamic(() => import("./WorkspaceSpin"), { ssr: false });

function WorkspacePreview({ build, spinning }: { build: BuildSummary; spinning: boolean }) {
  const [failed, setFailed] = useState(false);
  if (failed || (!build.is_example && build.status !== "ready")) return null;
  const id = encodeURIComponent(build.scan_id);
  const src = build.is_example ? `/examples/${id}.webp` : `/api/builds/${id}/preview.png`;
  return <div className="workspace-preview-frame">
    {/* eslint-disable-next-line @next/next/no-img-element */}
    <img className="workspace-preview" src={src} alt={`3D preview of ${build.title}`}
    loading="lazy" width={480} height={288} onError={() => setFailed(true)} />
    {spinning && <WorkspaceSpin id={build.scan_id} />}
  </div>;
}

function openedLabel(value: string) {
  const date = new Date(value);
  const minutes = Math.max(0, Math.floor((Date.now() - date.getTime()) / 60000));
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 1440) return `${Math.floor(minutes / 60)} hr ago`;
  if (minutes < 2880) return "Yesterday";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

export function WorkspaceDashboard({ examplesOnly = false }: { examplesOnly?: boolean }) {
  const { preferences, save, error: preferenceError } = useDashboardPreferences();
  const [builds, setBuilds] = useState<BuildSummary[] | null>(null);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [hovered, setHovered] = useState<string | null>(null);
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

  const [editingId, setEditingId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  async function saveName(build: BuildSummary, form: HTMLFormElement) {
    const title = String(new FormData(form).get("title") ?? "").trim();
    if (!title) {
      setActionError("Enter a workspace name.");
      form.querySelector("input")?.focus();
      return;
    }
    setSaving(true);
    setActionError("");
    try {
      if (title !== build.title) await api.renameBuild(build.scan_id, title);
      setBuilds(items => items?.map(item => item.scan_id === build.scan_id ? { ...item, title } : item) ?? null);
      setEditingId(null);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Could not rename the workspace.");
    } finally { setSaving(false); }
  }

  async function act(build: BuildSummary, action: "Rename" | "Delete") {
    setMenu(null);
    setActionError("");
    try {
      if (action === "Rename") {
        setEditingId(build.scan_id);
        return;
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
    <button type="button" key={action} disabled={saving} aria-label={`${action} ${build.title}`} onClick={() => act(build, action)}>{action}</button>);

  const workspaces = (builds || []).filter((build) => !examplesOnly || build.is_example);
  const visible = workspaces.filter((build) => build.title.toLowerCase().includes(search.trim().toLowerCase())).sort((a, b) => {
    if (preferences.sort === "name") return a.title.localeCompare(b.title);
    if (preferences.sort === "opened" && (a.opened_at || b.opened_at)) {
      return (b.opened_at ? Date.parse(b.opened_at) : 0) - (a.opened_at ? Date.parse(a.opened_at) : 0);
    }
    return Date.parse(b.created_at) - Date.parse(a.created_at);
  });

  return <>
    <section className="workspace-section" aria-labelledby="workspaces-title">
      <div className="workspace-section-heading">
        <h1 id="workspaces-title">{examplesOnly ? "Example workspaces" : "Your Workspaces"}<span>{builds === null ? "—" : workspaces.length}</span></h1>
        <Link href="/scan/new" className="button-primary dashboard-new"><DashboardIcon name="plus" />New workspace</Link>
      </div>
      <div className="workspace-toolbar">
        <label className="workspace-search"><DashboardIcon name="search" /><span className="sr-only">Search workspaces</span><input type="search" placeholder="Find a workspace…" value={search} onChange={(event) => setSearch(event.target.value)} /></label>
        <label className="workspace-sort"><span>Sort by</span><select aria-label="Sort workspaces" value={preferences.sort} onChange={(event) => save({ sort: event.target.value as WorkspaceSort })}><option value="opened">Last opened</option><option value="name">Name</option><option value="created">Date created</option></select></label>
      </div>
      {actionError && <p role="alert">{actionError}</p>}
      {menu && <div className="workspace-context" aria-label="Workspace actions" role="group" style={{ left: menu.x, top: menu.y }}>{buttons(menu.build)}</div>}
      {preferenceError && <p className="settings-caption" role="status">{preferenceError}</p>}
      <div className={`workspace-list ${preferences.compact ? "is-compact" : ""}`}>
        {error ? <div className="workspace-empty" role="alert"><DashboardIcon name="box" /><h3>We couldn&apos;t load your workspaces.</h3><p>Give it another try in a moment.</p><button type="button" className="text-link" onClick={() => { setError(false); setAttempt((n) => n + 1); }}>Try again ↗</button></div>
          : builds === null ? <div className="workspace-skeletons" role="status" aria-label="Loading workspaces">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-20" />)}</div>
          : visible.length === 0 ? <div className="workspace-empty"><DashboardIcon name={search ? "search" : "box"} /><h3>{search ? "No workspaces found." : examplesOnly ? "No examples in this account yet." : "Your next idea starts here."}</h3><p>{search ? "Try a different name or clear your search." : "Scan an object to make your first workspace."}</p>{search && <button className="text-link" onClick={() => setSearch("")}>Clear search</button>}</div>
          : <ul className="workspace-cards" aria-label="Workspaces">{visible.map((build) => <li key={build.scan_id} className="workspace-card" onPointerEnter={event => {
            if (event.pointerType !== "touch" && window.matchMedia("(prefers-reduced-motion: no-preference)").matches) setHovered(build.scan_id);
          }} onPointerLeave={() => setHovered(null)} onContextMenu={event => {
            if (editingId === build.scan_id) return;
            event.preventDefault();
            setMenu({ build, x: Math.max(0, Math.min(event.clientX, window.innerWidth - 148)), y: Math.max(0, Math.min(event.clientY, window.innerHeight - 96)) });
          }}>
              <WorkspacePreview build={build} spinning={hovered === build.scan_id && editingId !== build.scan_id} />
              <div><span className="workspace-card-label">Workspace</span>
              <div className="workspace-identity"><div className="workspace-name">
                {editingId === build.scan_id ? <form id={`rename-${build.scan_id}`} onSubmit={event => {
                  event.preventDefault();
                  void saveName(build, event.currentTarget);
                }}>
                  <input name="title" aria-label="Workspace name" defaultValue={build.title} maxLength={120} required
                    autoFocus onFocus={event => event.currentTarget.select()} readOnly={saving}
                    onKeyDown={event => { if (event.key === "Escape" && !saving) { setEditingId(null); setActionError(""); } }} />
                </form> : <Link href={`/scan/${build.scan_id}`} className="workspace-row-link"><strong>{build.title}</strong></Link>}
                {!build.is_example && <small>{build.input_type.charAt(0).toUpperCase() + build.input_type.slice(1)} scan</small>}</div></div></div>
              <div><span className="workspace-card-label">Actions</span><span className="workspace-actions">{editingId === build.scan_id ? <>
                <button type="submit" form={`rename-${build.scan_id}`} disabled={saving}>Save</button>
                <button type="button" disabled={saving} onClick={() => { setEditingId(null); setActionError(""); }}>Cancel</button>
              </> : buttons(build)}</span></div>
              <div><span className="workspace-card-label">Last opened</span><time dateTime={build.opened_at ?? undefined} title={build.opened_at ? new Date(build.opened_at).toLocaleString() : undefined}>{build.opened_at ? openedLabel(build.opened_at) : "No recorded open"}</time></div>
          </li>)}</ul>}
      </div>
    </section>
  </>;
}
