// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { ManualContext } from "../components/cards/ManualContext";
import { InstructionManual } from "../components/cards/InstructionManual";
import type { RepairPlan } from "../lib/contracts.gen";

vi.mock("../components/cards/manualIllustrations", () => ({
  renderManualViews: async () => new Map([['["p4"]', "data:image/png;base64,AA=="]]),
}));

it("opens a printable plan and maps the selected step to the sink scan", async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  const parts = JSON.parse(readFileSync("../fixtures/scans/scn_sink/parts.json", "utf8"));
  const plan: RepairPlan = {
    plan_id:"test", intent:"repair", diagnosis:{ summary:"Inspect the trap", confidence:.7, part_ids:["p4"] },
    steps:[{n:1, text:"Inspect the P-trap [[part:p4]].", part_ids:["p4"]}],
    tools:["Bucket"], safety:[], bom:[{name:"Washer",qty:2,spec:"Match existing size",est_price_usd:0,buy_links:[]}],
    printable:[], followups:[],
  };
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  const print = vi.spyOn(window, "print").mockImplementation(() => {});
  try {
    await act(async () => root.render(<ManualContext.Provider value={{name:"Kitchen sink",parts}}>
      <InstructionManual plan={plan}/></ManualContext.Provider>));
    await act(async () => host.querySelector("button")!.click());
    const manual = document.querySelector("dialog[open]")!;
    expect(manual.querySelector("h1")?.textContent).toBe("Kitchen sink");
    expect(manual.textContent).toContain("2 × Washer");
    expect(manual.querySelector(".manual-step p")?.textContent).toBe("Inspect the P-trap.");
    expect(manual.querySelector("figcaption")?.textContent).toBe("P-trap");
    (manual.querySelectorAll("button")[1] as HTMLButtonElement).click();
    expect(print).toHaveBeenCalledOnce();
  } finally {
    await act(async () => root.unmount());
    print.mockRestore();
    host.remove();
  }
});
