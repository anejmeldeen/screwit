// @vitest-environment jsdom
import { readFileSync } from "node:fs";
const sinkParts = JSON.parse(readFileSync("../fixtures/scans/scn_sink/parts.json", "utf8"));
import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { Workspace } from "../components/Workspace";

vi.mock("../lib/hooks", () => ({
  useScan: () => ({ scan: { status: "ready" } }),
  useReconstruction: () => ({ recon: null }),
  useParts: () => ({ parts: sinkParts, refresh: vi.fn() }),
  usePersistentNumber: (_key: string, initial: number) => [initial, vi.fn()],
  isStageDone: () => true,
}));
vi.mock("../lib/api", () => ({ api: {
  builds: async () => [], openBuild: async () => {}, partHistory: async () => ({}),
} }));
vi.mock("../components/ChatPanel", () => ({ ChatPanel: () => <div>Assistant conversation</div> }));
vi.mock("../components/viewer/ViewerPanel", () => ({
  ViewerPanel: ({ onSelect, focusPartId }: { focusPartId: string | null; onSelect: (id: string) => void }) =>
    <button data-camera-focus={focusPartId ?? ""} onClick={() => onSelect("p2")}>Pick faucet in scene</button>,
}));
it("inspects hierarchy and scene selections and returns to assembly export", async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  async function click(label: string) {
    const button = [...host.querySelectorAll("button")].find(b => b.textContent === label);
    expect(button).toBeDefined();
    await act(async () => button!.click());
  }
  try {
    await act(async () => root.render(<Workspace scanId="scn_test" title="Kitchen" />));
    const inspector = host.querySelector<HTMLElement>('[aria-label="Inspector"]')!;
    await click("Sink basin");
    expect(inspector.querySelector('[data-testid="part-row-p1"]')).toBeTruthy();
    const hierarchy = host.querySelector('[aria-label="Hierarchy"]')!;
    const divider = hierarchy.nextElementSibling!;
    expect(divider.getAttribute("aria-label")).toBe("Resize hierarchy height");
    expect(divider.nextElementSibling).toBe(inspector);
    const assistant = host.querySelector<HTMLElement>('[aria-label="Assistant"]')!;
    expect(assistant.hidden).toBe(false);
    await click("Pick faucet in scene");
    expect(host.querySelector("[data-camera-focus]")?.getAttribute("data-camera-focus")).toBe("");
    expect(inspector.hidden).toBe(false);
    expect(inspector.querySelector('[data-testid="part-row-p2"]')).toBeTruthy();
    expect(inspector.querySelector('[data-testid="part-row-p1"]')).toBeNull();
    await click("Kitchen");
    expect(inspector.textContent).toContain("Export assembly");
  } finally {
    await act(async () => root.unmount());
    host.remove();
    vi.unstubAllGlobals();
  }
});
