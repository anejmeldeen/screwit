"use client";

import dynamic from "next/dynamic";
import { workspaceSlugs } from "@/lib/workspacePaths";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { api, type BuildSummary } from "@/lib/api";
import { useDashboardPreferences } from "@/lib/dashboardPreferences";
import { DashboardIcon } from "./DashboardIcon";
import { Skeleton } from "./ui";

const sortLabels = { opened: "Last opened", name: "Name", created: "Date created" } as const;

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
  const [sortOpen, setSortOpen] = useState(false);
  const sortControl = useRef<HTMLDivElement>(null);
  const sortButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!sortOpen) return;
    function outside(event: PointerEvent) {
      if (!sortControl.current?.contains(event.target as Node)) setSortOpen(false);
    }
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [sortOpen]);
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
  const renameCancelled = useRef(false);   // Escape was pressed: the blur that follows must not save

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
    if (action === "Rename") {
      renameCancelled.current = false;
      setEditingId(build.scan_id);
      return;
    }
    // Delete at once, no confirmation: the card goes now, then the server is told; if that fails
    // the reload below brings the workspace back.
    setBuilds(items => items?.filter(item => item.scan_id !== build.scan_id) ?? null);
    try {
      await api.deleteBuild(build.scan_id);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Could not delete the workspace.");
    }
    setAttempt(n => n + 1);
  }

  const actionIcons = { Rename: "pencil", Delete: "trash" } as const;
  // Icon buttons on the cards; the right-click menu also shows the words.
  const buttons = (build: BuildSummary, withLabels = false) => (["Rename", "Delete"] as const).map(action =>
    <button type="button" key={action} disabled={saving} aria-label={`${action} ${build.title}`} title={action}
      className={withLabels ? "workspace-menu-button" : "workspace-icon-button"} onClick={() => act(build, action)}>
      <DashboardIcon name={actionIcons[action]} />{withLabels ? action : null}
    </button>);

  // Bulk select: "Select" turns every card into a checkbox, and the bar above the cards deletes or
  // renames all the selected workspaces at once (several get the name numbered 1, 2, 3…).
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [bulkRenaming, setBulkRenaming] = useState(false);
  function stopSelecting() {
    setSelecting(false);
    setSelected(new Set());
    setBulkRenaming(false);
  }
  function toggle(id: string) {
    setSelected(current => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }
  useEffect(() => {
    if (!selecting || bulkRenaming) return;
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") { setSelecting(false); setSelected(new Set()); }
    };
    window.addEventListener("keydown", escape);
    return () => window.removeEventListener("keydown", escape);
  }, [selecting, bulkRenaming]);

  const slugs = workspaceSlugs(builds ?? []);
  const workspaces = (builds || []).filter((build) => !examplesOnly || build.is_example);
  const visible = workspaces.filter((build) => build.title.toLowerCase().includes(search.trim().toLowerCase())).sort((a, b) => {
    if (preferences.sort === "name") return a.title.localeCompare(b.title);
    if (preferences.sort === "opened" && (a.opened_at || b.opened_at)) {
      return (b.opened_at ? Date.parse(b.opened_at) : 0) - (a.opened_at ? Date.parse(a.opened_at) : 0);
    }
    return Date.parse(b.created_at) - Date.parse(a.created_at);
  });
  // Bulk actions apply to what is selected AND shown, never to cards a search has hidden.
  const chosen = visible.filter(build => selected.has(build.scan_id));
  const allChosen = visible.length > 0 && chosen.length === visible.length;

  async function bulkDelete() {
    const ids = new Set(chosen.map(build => build.scan_id));
    if (!ids.size) return;
    setActionError("");
    setBuilds(items => items?.filter(item => !ids.has(item.scan_id)) ?? null);
    stopSelecting();
    const results = await Promise.allSettled([...ids].map(id => api.deleteBuild(id)));
    const failed = results.filter(result => result.status === "rejected").length;
    if (failed) setActionError(`Could not delete ${failed} of ${ids.size} workspaces.`);
    setAttempt(n => n + 1);
  }

  async function bulkRename(form: HTMLFormElement) {
    const base = String(new FormData(form).get("title") ?? "").trim();
    if (!base) {
      setActionError("Enter a workspace name.");
      return;
    }
    const names = new Map(chosen.map((build, i) => [build.scan_id, chosen.length === 1 ? base : `${base} ${i + 1}`]));
    setSaving(true);
    setActionError("");
    const results = await Promise.allSettled([...names].map(([id, title]) => api.renameBuild(id, title)));
    setSaving(false);
    setBuilds(items => items?.map(item => names.has(item.scan_id) ? { ...item, title: names.get(item.scan_id)! } : item) ?? null);
    const failed = results.filter(result => result.status === "rejected").length;
    if (failed) {
      setActionError(`Could not rename ${failed} of ${names.size} workspaces.`);
      setAttempt(n => n + 1);
    }
    stopSelecting();
  }

  return <>
    <section className="workspace-section" aria-labelledby="workspaces-title">
      <div className="workspace-section-heading">
        <h1 id="workspaces-title">{examplesOnly ? "Example workspaces" : "Your Workspaces"}<span>{builds === null ? "—" : workspaces.length}</span></h1>
      </div>
      <div className="workspace-toolbar">
        <label className="workspace-search"><DashboardIcon name="search" /><span className="sr-only">Search workspaces</span><input type="search" placeholder="Find a workspace…" value={search} onChange={(event) => setSearch(event.target.value)} /></label>
        <div className="workspace-sort" ref={sortControl}
          onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget)) setSortOpen(false); }}
          onKeyDown={event => {
            if (event.key === "Escape") { setSortOpen(false); sortButton.current?.focus(); }
          }}>
          <button type="button" ref={sortButton} aria-expanded={sortOpen} aria-controls="workspace-sort-options"
            onClick={() => setSortOpen(open => !open)}>Sort by · {sortLabels[preferences.sort]}</button>
          <div id="workspace-sort-options" className="workspace-sort-options" role="group" aria-label="Workspace order" hidden={!sortOpen}>
            {(Object.keys(sortLabels) as (keyof typeof sortLabels)[]).map(value =>
              <button key={value} type="button" aria-pressed={preferences.sort === value} onClick={() => {
                save({ sort: value }); setSortOpen(false); sortButton.current?.focus();
              }}><span>{sortLabels[value]}</span><span aria-hidden="true">{preferences.sort === value ? "✓" : ""}</span></button>
            )}
          </div>
        </div>
        <button type="button" className="workspace-select-toggle" aria-pressed={selecting}
          disabled={!selecting && visible.length === 0}
          onClick={() => { if (selecting) stopSelecting(); else { setEditingId(null); setMenu(null); setSelecting(true); } }}>
          {selecting ? "Done" : "Select"}
        </button>
        <Link href="/scan/new" className="button-primary dashboard-new"><DashboardIcon name="plus" />New workspace</Link>
      </div>
      {selecting && <div className="workspace-bulk" role="toolbar" aria-label="Selected workspaces">
        {bulkRenaming ? <form className="workspace-bulk-rename" onSubmit={event => {
          event.preventDefault();
          void bulkRename(event.currentTarget);
        }}>
          {/* Enter renames them all; Escape or clicking away cancels. */}
          <input name="title" aria-label="New name for the selected workspaces" required maxLength={112} autoFocus
            readOnly={saving} placeholder={chosen.length > 1
              ? `New name for all ${chosen.length}, numbered 1 to ${chosen.length} · Enter to rename`
              : "New name · Enter to rename"}
            onKeyDown={event => { if (event.key === "Escape" && !saving) setBulkRenaming(false); }}
            onBlur={() => { if (!saving) setBulkRenaming(false); }} />
        </form> : <>
          <span className="workspace-bulk-count" aria-live="polite">{chosen.length} selected</span>
          <button type="button" onClick={() => setSelected(allChosen ? new Set() : new Set(visible.map(build => build.scan_id)))}>
            {allChosen ? "Clear" : "Select all"}
          </button>
          <button type="button" disabled={!chosen.length || saving} onClick={() => setBulkRenaming(true)}>
            <DashboardIcon name="pencil" />Rename
          </button>
          <button type="button" disabled={!chosen.length || saving} onClick={() => void bulkDelete()}>
            <DashboardIcon name="trash" />Delete
          </button>
        </>}
      </div>}
      {actionError && <p role="alert">{actionError}</p>}
      {menu && <div className="workspace-context" aria-label="Workspace actions" role="group" style={{ left: menu.x, top: menu.y }}>{buttons(menu.build, true)}</div>}
      {preferenceError && <p className="settings-caption" role="status">{preferenceError}</p>}
      <div className={`workspace-list ${preferences.compact ? "is-compact" : ""}`}>
        {error ? <div className="workspace-empty" role="alert"><DashboardIcon name="box" /><h3>We couldn&apos;t load your workspaces.</h3><p>Give it another try in a moment.</p><button type="button" className="text-link" onClick={() => { setError(false); setAttempt((n) => n + 1); }}>Try again ↗</button></div>
          : builds === null ? <div className="workspace-skeletons" role="status" aria-label="Loading workspaces">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-20" />)}</div>
          : visible.length === 0 ? <div className="workspace-empty"><DashboardIcon name={search ? "search" : "box"} /><h3>{search ? "No workspaces found." : examplesOnly ? "No examples in this account yet." : "Your next idea starts here."}</h3><p>{search ? "Try a different name or clear your search." : "Scan an object to make your first workspace."}</p>{search && <button className="text-link" onClick={() => setSearch("")}>Clear search</button>}</div>
          : <ul className="workspace-cards" aria-label="Workspaces">{visible.map((build) => <li key={build.scan_id}
            className={`workspace-card${selecting ? " is-selecting" : ""}${selected.has(build.scan_id) ? " is-selected" : ""}`}
            onClick={selecting ? () => toggle(build.scan_id) : undefined} onPointerEnter={event => {
            if (event.pointerType !== "touch" && window.matchMedia("(prefers-reduced-motion: no-preference)").matches) setHovered(build.scan_id);
          }} onPointerLeave={() => setHovered(null)} onContextMenu={event => {
            if (selecting || editingId === build.scan_id) return;
            event.preventDefault();
            setMenu({ build, x: Math.max(0, Math.min(event.clientX, window.innerWidth - 148)), y: Math.max(0, Math.min(event.clientY, window.innerHeight - 96)) });
          }}>
              <WorkspacePreview build={build} spinning={hovered === build.scan_id && editingId !== build.scan_id} />
              <div className="workspace-identity">
                {selecting && <input type="checkbox" className="workspace-check" aria-label={`Select ${build.title}`}
                  checked={selected.has(build.scan_id)} onClick={event => event.stopPropagation()}
                  onChange={() => toggle(build.scan_id)} />}
                <div className="workspace-name">
                {editingId === build.scan_id ? <form id={`rename-${build.scan_id}`} onSubmit={event => {
                  event.preventDefault();
                  void saveName(build, event.currentTarget);
                }}>
                  {/* Enter saves, Escape cancels, clicking away saves. */}
                  <input name="title" aria-label="Workspace name" defaultValue={build.title} maxLength={120} required
                    autoFocus onFocus={event => event.currentTarget.select()} readOnly={saving}
                    onKeyDown={event => {
                      if (event.key === "Escape" && !saving) { renameCancelled.current = true; setEditingId(null); setActionError(""); }
                    }}
                    onBlur={event => {
                      if (renameCancelled.current || saving) return;
                      const value = event.currentTarget.value.trim();
                      if (!value || value === build.title) { setEditingId(null); setActionError(""); return; }
                      event.currentTarget.form?.requestSubmit();
                    }} />
                </form> : selecting ? <strong title={build.title}>{build.title}</strong>
                  : <Link href={`/scan/${encodeURIComponent(slugs.get(build.scan_id)!)}`} className="workspace-row-link"><strong title={build.title}>{build.title}</strong></Link>}
                <small><time dateTime={build.opened_at ?? undefined} title={build.opened_at ? `Last opened ${new Date(build.opened_at).toLocaleString()}` : undefined}>{build.opened_at ? openedLabel(build.opened_at) : "Not opened yet"}</time></small></div>
              {!selecting && editingId !== build.scan_id && <span className="workspace-actions">{buttons(build)}</span>}</div>
          </li>)}</ul>}
      </div>
    </section>
  </>;
}
