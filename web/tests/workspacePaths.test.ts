import { expect, it } from "vitest";
import { workspaceSlugs } from "../lib/workspacePaths";

it("creates readable, distinct names independent of dashboard order", () => {
  const titles = ["Kitchen Sink", "New", "Tap!", "tap", "新 模型", "???"];
  const builds = titles.map((title, i) => ({ title, scan_id: "scn_" + i }));
  expect([...workspaceSlugs(builds).values()]).toEqual(["kitchen-sink", "new~scn_1", "tap~scn_2", "tap~scn_3", "新-模型", "workspace"]);
  expect(workspaceSlugs([...builds].reverse())).toEqual(workspaceSlugs(builds));
});
