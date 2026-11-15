// The API client. Every REST call to the FastAPI backend goes through here.
//
// AGENTS.md decision 1 overrides SPEC 3.3: there is no bearer/demo token. The
// browser speaks to its own origin and Next rewrites /api/* to the loopback
// FastAPI server. This client therefore adds NO Authorization header. Asset
// bytes (splats, meshes, thumbnails) are loaded by the storage client / three
// loaders from signed URLs and never pass through here.

import { API_BASE } from "./env";
import type {
  CadResult,
  CreateScanRequest,
  CreateScanResponse,
  DemoScan,
  MeasureRequest,
  MeasureResponse,
  MergeRequest,
  Part,
  PartPatch,
  Reconstruction,
  ScaleRequest,
  Scan,
  SceneGraph,
} from "./contracts.gen";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly url: string,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface Account { username: string }
export interface BuildSummary {
  scan_id: string;
  title: string;
  status: string;
  created_at: string;
  updated_at: string;
  input_type: string;
  is_example: boolean;
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    cache: "no-store",
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      const message = body && (body.detail || body.message);
      detail = typeof message === "string" ? message : "Please check your username and password.";
    } catch {
      /* non-JSON error body; keep statusText */
    }
    if (res.status === 401 && !path.startsWith("/auth/") && typeof window !== "undefined") {
      window.dispatchEvent(new Event("screwit:session-expired"));
    }
    throw new ApiError(res.status, url, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  me: () => request<Account>("/auth/me"),
  login: (username: string, password: string) => request<Account>("/auth/login", {
    method: "POST", body: JSON.stringify({ username, password }),
  }),
  register: (username: string, password: string) => request<Account>("/auth/register", {
    method: "POST", body: JSON.stringify({ username, password }),
  }),
  logout: () => request<void>("/auth/logout", { method: "POST" }),
  builds: () => request<BuildSummary[]>("/builds"),
  // Landing / gallery
  demoScans: () => request<DemoScan[]>("/demo-scans"),

  // Scan lifecycle
  createScan: (body: CreateScanRequest) =>
    request<CreateScanResponse>("/scans", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  startScan: (scanId: string) =>
    request<Scan>(`/scans/${scanId}/start`, { method: "POST" }),
  getScan: (scanId: string) => request<Scan>(`/scans/${scanId}`),

  // Reconstruction + scale
  getReconstruction: (scanId: string) =>
    request<Reconstruction>(`/scans/${scanId}/reconstruction`),
  scale: (scanId: string, body: ScaleRequest) =>
    request<Reconstruction>(`/scans/${scanId}/scale`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // Parts
  getParts: (scanId: string) => request<Part[]>(`/scans/${scanId}/parts`),
  patchPart: (scanId: string, partId: string, body: PartPatch) =>
    request<Part>(`/scans/${scanId}/parts/${partId}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  mergeParts: (scanId: string, body: MergeRequest) =>
    request<Part>(`/scans/${scanId}/parts/merge`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  partHistory: (scanId: string) => request<{can_undo: boolean; can_restore_example: boolean}>(`/scans/${scanId}/parts/history`),
  undoMerge: (scanId: string) => request<Part[]>(`/scans/${scanId}/parts/undo`, {method: "POST"}),
  restoreExample: (scanId: string) => request<Part[]>(`/scans/${scanId}/parts/restore-example`, {method: "POST"}),

  // Scene graph + measurement
  getSceneGraph: (scanId: string) =>
    request<SceneGraph>(`/scans/${scanId}/scene-graph`),
  measure: (scanId: string, body: MeasureRequest) =>
    request<MeasureResponse>(`/scans/${scanId}/measure`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // CAD
  getCad: (scanId: string, cadId: string) =>
    request<CadResult>(`/scans/${scanId}/cad/${cadId}`),
};

export type Api = typeof api;
