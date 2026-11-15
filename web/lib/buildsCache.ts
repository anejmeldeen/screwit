// The signed-in account's workspace list, kept so the dashboard draws it at once and then refreshes
// it from the API (stale-while-revalidate). A module variable covers moves inside the app; a copy in
// sessionStorage covers a reload of this tab and disappears with the tab. Nothing goes to
// localStorage, so the list never outlives the browser session on disk. Signing in or out, renaming
// the account and an expired session all clear it (see AccountProvider).

import type { BuildSummary } from "./api";

const KEY = "screwit:builds";
const VERSION = 1;
interface Entry { v: number; username: string; builds: BuildSummary[] }
let memory: Entry | null = null;

function isBuild(value: unknown): value is BuildSummary {
  if (!value || typeof value !== "object") return false;
  const b = value as Record<string, unknown>;
  return typeof b.scan_id === "string" && typeof b.title === "string" && typeof b.status === "string"
    && typeof b.created_at === "string" && typeof b.updated_at === "string"
    && (b.opened_at === null || typeof b.opened_at === "string")
    && typeof b.input_type === "string" && typeof b.is_example === "boolean";
}

function save(entry: Entry) {
  if (typeof window === "undefined") return;   // never on the server, where module state is shared by every request
  memory = entry;
  try { sessionStorage.setItem(KEY, JSON.stringify(entry)); } catch { /* The in-memory copy still works. */ }
}

/** The last list loaded for this account in this tab, or null when there is none. */
export function readCachedBuilds(username: string): BuildSummary[] | null {
  if (typeof window === "undefined" || !username) return null;
  if (memory) return memory.username === username ? memory.builds : null;
  try {
    const entry = JSON.parse(sessionStorage.getItem(KEY) ?? "null") as Partial<Entry> | null;
    if (entry?.v !== VERSION || entry.username !== username || !Array.isArray(entry.builds) || !entry.builds.every(isBuild)) {
      return null;
    }
    memory = entry as Entry;
    return memory.builds;
  } catch {
    return null;
  }
}

export function writeCachedBuilds(username: string, builds: BuildSummary[]) {
  if (username) save({ v: VERSION, username, builds });
}

/** Opening a workspace moves it to the top of "Last opened" before the dashboard next refreshes. */
export function markBuildOpened(username: string, scanId: string) {
  const builds = readCachedBuilds(username);
  if (!builds?.some(build => build.scan_id === scanId)) return;
  const now = new Date().toISOString();
  writeCachedBuilds(username, builds.map(build => build.scan_id === scanId ? { ...build, opened_at: now } : build));
}

export function clearCachedBuilds() {
  memory = null;
  try { sessionStorage.removeItem(KEY); } catch { /* Nothing was stored. */ }
}
