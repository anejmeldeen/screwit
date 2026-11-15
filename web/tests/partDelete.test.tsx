// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { act, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "../lib/api";
import { Workspace } from "../components/Workspace";
import type { Part } from "../lib/contracts.gen";

const sinkParts: Part[] = JSON.parse(readFileSync("../fixtures/scans/scn_sink/parts.json", "utf8"));
const refreshParts = vi.fn();

vi.mock("../lib/hooks", () => ({
  useScan: () => ({ scan: { status: "ready" } }),
  useReconstruction: () => ({ recon: null }),
  // Stateful, so the workspace's own setParts after a delete shows in the hierarchy.
  useParts: () => {
    const [parts, setParts] = useState(sinkParts);
    return { parts, setParts, refresh: refreshParts };
  },
  usePersistentNumber: (_key: string, initial: number) => [initial, vi.fn()],
  isStageDone: () => true,
}));
vi.mock("../lib/api", async () => {
  const actual = await vi.importActual<typeof import("../lib/api")>("../lib/api");
  return {
    ApiError: actual.ApiError,
    api: {
      deletePart: vi.fn(), partHistory: vi.fn(), undoMerge: vi.fn(), restoreExample: vi.fn(),
      mergeParts: vi.fn(), builds: async () => [], openBuild: async () => {},
    },
  };
});
vi.mock("../components/ChatPanel", () => ({ ChatPanel: () => <div>Assistant conversation</div> }));
vi.mock("../components/viewer/ViewerPanel", () => ({ ViewerPanel: () => <div>3D view</div> }));

let host: HTMLDivElement;
let root: Root;
const inspector = () => host.querySelector<HTMLElement>('[aria-label="Inspector"]')!;
const hierarchy = () => [...host.querySelectorAll('[aria-label="Hierarchy"] .scene-tree button')].map(b => b.textContent);
const buttonNamed = (text: string) => [...host.querySelectorAll("button")].find(b => b.textContent === text);
const deleteButton = () => host.querySelector<HTMLButtonElement>("button.part-delete")!;
// Let the mocked requests resolve and React commit what follows from them.
const settle = () => act(async () => { for (let i = 0; i < 5; i++) await Promise.resolve(); });

async function click(button: HTMLButtonElement | undefined) {
  expect(button).toBeDefined();
  await act(async () => button!.click());
  await settle();
}

beforeEach(async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.clearAllMocks();
  vi.mocked(api.deletePart).mockResolvedValue(undefined);
  vi.mocked(api.partHistory).mockResolvedValue({ can_undo: false, can_restore_example: false });
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  await act(async () => root.render(<Workspace scanId="scn_test" title="Kitchen" />));
  await click(buttonNamed("Sink basin"));
  expect(inspector().querySelector('[data-testid="part-row-p1"]')).toBeTruthy();
});

afterEach(async () => {
  vi.useRealTimers();
  await act(async () => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
});

it("deletes a part on the second click, clears the selection and lights up Undo", async () => {
  expect(deleteButton().textContent).toBe("Delete part");
  expect(deleteButton().getAttribute("aria-label")).toBe("Delete part Sink basin");
  await click(deleteButton());
  expect(deleteButton().textContent).toBe("Are you sure?");
  expect(deleteButton().classList.contains("is-confirming")).toBe(true);
  expect(api.deletePart).not.toHaveBeenCalled();

  vi.mocked(api.partHistory).mockResolvedValue({ can_undo: true, can_restore_example: false });
  const historyCalls = vi.mocked(api.partHistory).mock.calls.length;
  await click(deleteButton());
  expect(api.deletePart).toHaveBeenCalledTimes(1);
  expect(api.deletePart).toHaveBeenCalledWith("scn_test", "p1");
  expect(hierarchy()).not.toContain("Sink basin");
  expect(hierarchy()).toHaveLength(sinkParts.length - 1);
  expect(inspector().querySelector('[data-testid^="part-row-"]')).toBeNull();
  expect(inspector().textContent).toContain("Assembly history");
  expect(refreshParts).toHaveBeenCalled();
  expect(vi.mocked(api.partHistory).mock.calls.length).toBeGreaterThan(historyCalls);
  expect(buttonNamed("↶ Undo")?.disabled).toBe(false);
});

it("falls back to Delete part on Escape and after four seconds", async () => {
  await click(deleteButton());
  await act(async () => window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" })));
  expect(deleteButton().textContent).toBe("Delete part");

  vi.useFakeTimers();
  await click(deleteButton());
  expect(deleteButton().textContent).toBe("Are you sure?");
  await act(async () => vi.advanceTimersByTime(3900));
  expect(deleteButton().textContent).toBe("Are you sure?");
  await act(async () => vi.advanceTimersByTime(200));
  expect(deleteButton().textContent).toBe("Delete part");
  await click(deleteButton());  // a click after the fallback arms again; it does not delete
  expect(api.deletePart).not.toHaveBeenCalled();
});

it("shows the error and keeps the part when the delete fails", async () => {
  vi.mocked(api.deletePart).mockRejectedValue(
    new ApiError(409, "/api/scans/scn_test/parts/p1", "An example workspace must keep at least one part."));
  await click(deleteButton());
  await click(deleteButton());
  expect(inspector().textContent).toContain("An example workspace must keep at least one part. (409)");
  expect(inspector().querySelector('[data-testid="part-row-p1"]')).toBeTruthy();
  expect(hierarchy()).toContain("Sink basin");
  expect(deleteButton().textContent).toBe("Delete part");
  expect(deleteButton().disabled).toBe(false);
});
