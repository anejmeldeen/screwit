// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { MarkdownMessage } from "../components/MarkdownMessage";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

it("turns part references into chips inside bold text, headings and tables", async () => {
  const onPartChip = vi.fn();
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  const text = [
    "## Parts of [[part:p3]]",
    "",
    "**[[part:p1]] Headband** (180 mm long)",
    "",
    "| Part | Note |", "| --- | --- |", "| [[part:p2]] | *see [[part:p4]]* |",
  ].join("\n");
  await act(async () => root.render(<MarkdownMessage text={text} onPartChip={onPartChip} />));
  expect(host.textContent).not.toContain("[[part:");
  const chips = Array.from(host.querySelectorAll("button")).map(b => b.textContent);
  expect(chips).toEqual(["p3", "p1", "p2", "p4"]);
  await act(async () => (host.querySelectorAll("button")[1] as HTMLButtonElement).click());
  expect(onPartChip).toHaveBeenCalledWith("p1");
  await act(async () => root.unmount());
});
