// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ScanSetupForm } from "../components/ScanSetupForm";
import { api } from "../lib/api";
import { uploadToPresignedPost } from "../lib/storage";

vi.mock("../lib/api", () => ({ api: { createScan: vi.fn(), startScan: vi.fn() }, ApiError: class extends Error {} }));
vi.mock("../lib/storage", () => ({ uploadToPresignedPost: vi.fn() }));
let root: Root;
let container: HTMLDivElement;
const onStarted = vi.fn();
function render(title: string) {
  root.render(createElement(ScanSetupForm, { title, onTitleChange: vi.fn(), onStarted }));
}
function choosePhotos() {
  const input = container.querySelector('input[multiple]')!;
  Object.defineProperty(input, "files", { configurable: true, value: [
    new File(["a"], "front.jpg", { type: "image/jpeg" }), new File(["b"], "back.jpg", { type: "image/jpeg" }),
  ] });
  input.dispatchEvent(new Event("change", { bubbles: true }));
}
const submit = () => container.querySelector("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
beforeEach(() => {
  vi.clearAllMocks();
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  container = document.createElement("div"); document.body.append(container); root = createRoot(container);
  vi.mocked(api.createScan).mockResolvedValue({ scan_id: "scn_new", uploads: [{ key: "a" }, { key: "b" }] } as never);
  vi.mocked(api.startScan).mockResolvedValue({} as never);
  vi.mocked(uploadToPresignedPost).mockResolvedValue(undefined);
});
afterEach(async () => { await act(async () => root.unmount()); container.remove(); });

describe("workspace setup inside the editor", () => {
  it("waits for a title and explicit submission, then uploads every photo", async () => {
    await act(async () => render(""));
    await act(async () => choosePhotos());
    expect(api.createScan).not.toHaveBeenCalled();
    expect((container.querySelector('button[type="submit"]') as HTMLButtonElement).disabled).toBe(true);
    await act(async () => render("  Chair repair  "));
    await act(async () => { submit(); });
    expect(api.createScan).toHaveBeenCalledWith({ title: "Chair repair", input_type: "photos", photo_count: 2 });
    expect(uploadToPresignedPost).toHaveBeenCalledTimes(2);
    expect(api.startScan).toHaveBeenCalledWith("scn_new");
    expect(onStarted).toHaveBeenCalledWith("scn_new");
  });
  it("keeps the form on upload failure and retries the existing workspace", async () => {
    vi.mocked(uploadToPresignedPost).mockRejectedValueOnce(new Error("Connection lost"));
    await act(async () => render("Chair"));
    await act(async () => choosePhotos());
    await act(async () => { submit(); });
    expect(container.textContent).toContain("Connection lost");
    expect(onStarted).not.toHaveBeenCalled();
    await act(async () => { submit(); });
    expect(api.createScan).toHaveBeenCalledTimes(1);
    expect(onStarted).toHaveBeenCalledWith("scn_new");
  });
});
