// Playwright performance runner (brief step 3).
//
// Mobile emulation (iPhone 12-class viewport + UA), 4x CPU throttling and a slow-4G network profile,
// both applied through CDP. Measures, on the hero scan workspace:
//   - time to the workspace shell (performance.mark 'shell:ready', or DOM fallback: status bar visible)
//   - time to first mesh / first splat drawn (marks 'viewer:first-mesh' / 'viewer:first-splat')
//   - frame times while orbiting for 5 s (rAF deltas collected in page; p50 fps and 5th-pctile fps)
//   - chat first-token latency (mark 'chat:first-token', or DOM fallback: first assistant text)
// Then a desktop, unthrottled run for comparison.
//
// NEVER waits for networkidle: the workspace keeps an SSE progress stream open, so the network is
// never idle and a networkidle wait would hang. Navigation uses waitUntil:"commit" and every signal
// is a mark or a DOM-testid wait with its own bounded timeout.
//
// This host has NO GPU: Chromium renders WebGL in software (SwiftShader), so orbit fps is NOT
// representative of a real phone. It is reported with that caveat; the real-phone /perf page
// (DEVICE_TEST.md) is the authority for the 30 fps budget.
//
// Usage: node playwright/run.mjs <baseUrl> <outJsonPath>
// Always exits 0; writes {"skipped": true, "reason": ...} when the web app / browser is unavailable.

import { writeFileSync } from "node:fs";

const baseUrl = process.argv[2] || "http://127.0.0.1:3000";
const outPath = process.argv[3] || "out/playwright.json";
const HERO = "scn_trap";

// performance.mark names emitted by the web app (web/lib/perf.ts). 'shell:ready' is being added by
// P1; until it lands the runner falls back to the status-bar testid becoming visible.
const MARKS = {
  shell: "shell:ready",
  firstMesh: "viewer:first-mesh",
  firstSplat: "viewer:first-splat",
  chatFirstToken: "chat:first-token",
};

// DOM hooks the web app exposes (verified in web/: status-bar, viewer-canvas, chat-input, chat-send).
const SEL = {
  statusBar: '[data-testid="status-bar"]',
  viewerCanvas: '[data-testid="viewer-canvas"]',
  chatInput: '[data-testid="chat-input"]',
  chatSend: '[data-testid="chat-send"]',
  // No stable assistant-message testid yet; the assistant bubble renders markdown in a .prose block.
  assistantText: ".prose",
};

function bail(reason) {
  writeFileSync(outPath, JSON.stringify({ skipped: true, reason, marks_needed: MARKS }, null, 2));
  console.error(`playwright: SKIP — ${reason}`);
  process.exit(0);
}

let chromium, devices;
try {
  ({ chromium, devices } = await import("playwright"));
} catch (e) {
  bail(`playwright not installed (run 'npm ci' in perf/): ${e.message}`);
}

// Read a mark's startTime (ms from navigation) if present, else null.
async function markTime(page, name) {
  return page.evaluate((n) => {
    const e = performance.getEntriesByName(n, "mark");
    return e.length ? e[e.length - 1].startTime : null;
  }, name);
}

// Wait up to timeoutMs for a mark to appear; returns its startTime or null. Polls cheaply; never
// blocks on the network.
async function waitForMark(page, name, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const t = await markTime(page, name);
    if (t != null) return t;
    await page.waitForTimeout(100);
  }
  return null;
}

// The WebGL renderer string, to flag software (SwiftShader) rendering.
async function webglRenderer(page) {
  return page.evaluate(() => {
    try {
      const c = document.createElement("canvas");
      const gl = c.getContext("webgl2") || c.getContext("webgl");
      if (!gl) return null;
      const dbg = gl.getExtension("WEBGL_debug_renderer_info");
      return dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    } catch {
      return null;
    }
  });
}

function isSoftware(renderer) {
  return !!renderer && /swiftshader|software|llvmpipe|angle \(google/i.test(renderer);
}

// Collect rAF deltas for durationMs while dragging a pointer across the viewer canvas to orbit.
async function measureOrbitFps(page, durationMs) {
  const canvas = page.locator(SEL.viewerCanvas).first();
  if (!(await canvas.count())) {
    return { p50_fps: null, worst_fps: null, frames: 0, detail: `no ${SEL.viewerCanvas} to orbit` };
  }
  const box = await canvas.boundingBox();
  if (!box) return { p50_fps: null, worst_fps: null, frames: 0, detail: "viewer canvas has no box" };

  await page.evaluate((ms) => {
    window.__perfFrames = [];
    window.__perfDone = false;
    let last = performance.now();
    const stop = last + ms;
    function tick(now) {
      window.__perfFrames.push(now - last);
      last = now;
      if (now < stop) requestAnimationFrame(tick);
      else window.__perfDone = true;
    }
    requestAnimationFrame(tick);
  }, durationMs);

  // Pointer drag across the canvas centre to orbit (r3f frameloop renders on pointer input).
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  const steps = Math.ceil(durationMs / 50);
  for (let i = 0; i < steps; i++) {
    const dx = Math.sin(i / 3) * (box.width * 0.2);
    const dy = Math.cos(i / 5) * (box.height * 0.1);
    await page.mouse.move(cx + dx, cy + dy);
    await page.waitForTimeout(50);
  }
  await page.mouse.up();
  await page.waitForFunction(() => window.__perfDone === true, null, { timeout: durationMs + 5000 }).catch(() => {});

  const deltas = await page.evaluate(() => (window.__perfFrames || []).filter((d) => d > 0));
  if (!deltas.length) return { p50_fps: null, worst_fps: null, frames: 0, detail: "no rAF frames collected" };
  const fps = deltas.map((d) => 1000 / d).sort((a, b) => a - b);
  const pct = (p) => fps[Math.min(fps.length - 1, Math.max(0, Math.round((p / 100) * fps.length) - 1))];
  return {
    p50_fps: Math.round(pct(50) * 10) / 10,
    worst_fps: Math.round(pct(5) * 10) / 10, // 5th-percentile (slowest) fps = a floor
    frames: deltas.length,
  };
}

// Send a chat message and time the first token: the 'chat:first-token' mark if emitted, else the
// first non-empty assistant text node. Uses chat-input + chat-send, never networkidle.
async function measureChatFirstToken(page) {
  const input = page.locator(SEL.chatInput).first();
  if (!(await input.count())) return { first_token_ms: null, source: null, detail: `no ${SEL.chatInput}` };

  // Clear any mark from a prior interaction isn't needed (first load); record wall-clock start.
  const start = Date.now();
  await input.fill("why is it leaking?");
  const send = page.locator(SEL.chatSend).first();
  if (await send.count()) {
    await send.click();
  } else {
    await input.press("Enter");
  }

  // Race the mark against the DOM fallback, bounded.
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    const mark = await markTime(page, MARKS.chatFirstToken);
    if (mark != null) {
      // Mark time is ms-from-navigation; report latency from send using wall clock for consistency.
      return { first_token_ms: Date.now() - start, source: "mark:chat:first-token" };
    }
    const hasText = await page.evaluate((sel) => {
      const nodes = Array.from(document.querySelectorAll(sel));
      return nodes.some((n) => n.textContent && n.textContent.trim().length > 0);
    }, SEL.assistantText);
    if (hasText) return { first_token_ms: Date.now() - start, source: "dom:assistant-text" };
    await page.waitForTimeout(100);
  }
  return { first_token_ms: null, source: null, detail: "no chat:first-token mark and no assistant text in 15 s" };
}

// Wait for the shell: the 'shell:ready' mark, else the status bar becoming visible. Returns
// { ms, source } where ms is wall-clock from navigation start (nav0) and source names the signal.
async function waitForShell(page, nav0, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const mark = await markTime(page, MARKS.shell);
    if (mark != null) return { ms: mark, source: "mark:shell:ready" };
    const visible = await page.locator(SEL.statusBar).first().isVisible().catch(() => false);
    if (visible) return { ms: Date.now() - nav0, source: "dom:status-bar-visible" };
    await page.waitForTimeout(100);
  }
  return { ms: null, source: null };
}

async function runWorkspace(page, nav0, pageErrors) {
  const shell = await waitForShell(page, nav0, 15000);

  // The hero scan defaults to splat view when a splat is ready, so the mesh layer (which fires
  // viewer:first-mesh) is not mounted. The budget is "mesh visible under 5 s", so switch to the
  // Mesh view once the toggle appears, then wait for the mark.
  let meshToggleClicked = false;
  try {
    const meshBtn = page.getByRole("button", { name: /^mesh$/i }).first();
    await meshBtn.waitFor({ state: "visible", timeout: 8000 });
    await meshBtn.click({ timeout: 2000 });
    meshToggleClicked = true;
  } catch {
    /* toggle absent (no splat, so mesh is the only view) or not shown in time */
  }

  // First mesh needs the viewer canvas mounted; the mark fires when the mesh is drawn. r3f runs
  // frameloop="demand", so nudge a render (tiny pointer move over the canvas) to force a draw after
  // the GLB loads. Allow 30 s on throttled mobile (4x CPU + slow-4G loading four part GLBs).
  const canvasForNudge = page.locator(SEL.viewerCanvas).first();
  const nudge = async () => {
    try {
      const b = await canvasForNudge.boundingBox();
      if (b) {
        await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
        await page.mouse.move(b.x + b.width / 2 + 2, b.y + b.height / 2 + 2);
      }
    } catch {
      /* canvas not ready yet */
    }
  };
  let firstMeshMs = null;
  {
    const deadline = Date.now() + 30000;
    while (Date.now() < deadline) {
      firstMeshMs = await markTime(page, MARKS.firstMesh);
      if (firstMeshMs != null) break;
      await nudge();
      await page.waitForTimeout(250);
    }
  }
  const firstSplatMs = await markTime(page, MARKS.firstSplat);

  let fps = { p50_fps: null, worst_fps: null, frames: 0, detail: "first mesh never drawn" };
  if (firstMeshMs != null) fps = await measureOrbitFps(page, 5000);

  const chat = await measureChatFirstToken(page);
  const renderer = await webglRenderer(page);
  return {
    shell_ms: shell.ms,
    shell_source: shell.source,
    mesh_toggle_clicked: meshToggleClicked,
    first_mesh_ms: firstMeshMs,
    first_splat_ms: firstSplatMs,
    orbit_fps: fps,
    webgl_renderer: renderer,
    fps_representative: !isSoftware(renderer),
    fps_note: isSoftware(renderer)
      ? "software renderer (SwiftShader), not representative — see /perf on a real phone"
      : "hardware renderer",
    chat,
    page_errors: pageErrors.slice(0, 10),
  };
}

// Attach console/network diagnostics to a page before navigation. Returns the collecting array.
function attachDiagnostics(page) {
  const pageErrors = [];
  page.on("console", (m) => {
    if (m.type() === "error") pageErrors.push(`console.error: ${m.text()}`.slice(0, 300));
  });
  page.on("pageerror", (e) => pageErrors.push(`pageerror: ${e.message}`.slice(0, 300)));
  page.on("requestfailed", (r) => {
    const u = r.url();
    if (/\.glb|\.spz|\.ply|assets/.test(u)) {
      pageErrors.push(`requestfailed: ${u} (${r.failure()?.errorText})`.slice(0, 300));
    }
  });
  return pageErrors;
}

async function runMobile(browser) {
  const iphone = devices["iPhone 12"];
  const context = await browser.newContext({ ...iphone });
  const page = await context.newPage();
  const pageErrors = attachDiagnostics(page);
  const client = await context.newCDPSession(page);
  await client.send("Emulation.setCPUThrottlingRate", { rate: 4 });
  await client.send("Network.enable");
  // Lighthouse's mobile "slow 4G" profile (150 ms RTT, 1.6 Mbps down, 750 kbps up) expressed as
  // DevTools packet-level throttling, using Lighthouse's own adjustment factors for this method
  // (RTT x 3.75, throughput x 0.9): 562.5 ms, 1474.56 kbps down, 675 kbps up. The earlier
  // 400 kbps / 400 ms was about four times slower than the profile AGENTS.md names.
  await client.send("Network.emulateNetworkConditions", {
    offline: false,
    downloadThroughput: (1474.56 * 1024) / 8,
    uploadThroughput: (675 * 1024) / 8,
    latency: 562.5,
  });
  const nav0 = Date.now();
  await page.goto(`${baseUrl}/scan/${HERO}`, { waitUntil: "commit" });
  const r = await runWorkspace(page, nav0, pageErrors);
  await context.close();
  return { emulation: "iPhone 12, 4x CPU, slow-4G (CDP)", ...r };
}

async function runDesktop(browser) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const pageErrors = attachDiagnostics(page);
  const nav0 = Date.now();
  await page.goto(`${baseUrl}/scan/${HERO}`, { waitUntil: "commit" });
  const r = await runWorkspace(page, nav0, pageErrors);
  await context.close();
  return { emulation: "desktop 1440x900, unthrottled", ...r };
}

async function main() {
  try {
    const resp = await fetch(`${baseUrl}/scan/${HERO}`, { method: "GET" });
    if (resp.status >= 500) bail(`web server at ${baseUrl} returned HTTP ${resp.status}`);
  } catch (e) {
    bail(`web server not reachable at ${baseUrl}: ${e.message}`);
  }

  let browser;
  try {
    browser = await chromium.launch({ headless: true, args: ["--no-sandbox", "--disable-dev-shm-usage"] });
  } catch (e) {
    bail(`could not launch Chromium (run 'npx playwright install chromium'): ${e.message}`);
  }
  try {
    const mobile = await runMobile(browser);
    const desktop = await runDesktop(browser);
    writeFileSync(outPath, JSON.stringify({ skipped: false, baseUrl, marks_needed: MARKS, mobile, desktop }, null, 2));
    console.error(`playwright: wrote ${outPath}`);
  } catch (e) {
    bail(`playwright run failed: ${e.message}`);
  } finally {
    await browser.close();
  }
}

await main();
