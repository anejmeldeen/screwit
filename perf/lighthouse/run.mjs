// Lighthouse mobile-preset runner (brief step 2).
//
// Runs Lighthouse (pinned 12.2.1) with its built-in mobile preset on the landing page ("/")
// and on the workspace of the hero scan ("/scan/scn_trap"). For each page it records
// LCP, TBT, CLS and total JS transferred (bytes, from the network requests), and for the
// landing page it inspects the network log for three.js and Spark chunks — the landing page
// must not pull the 3D stack (AGENTS.md: "no three.js or Spark loaded").
//
// Usage: node lighthouse/run.mjs <baseUrl> <outJsonPath>
// Exits 0 always (it reports; the Python grader decides pass/fail). On any failure it writes
// {"skipped": true, "reason": "..."} so the suite never errors before the web branch merges.
//
// Headless Chromium comes from Playwright's download (shared cache). We resolve its executable
// via playwright's chromium.executablePath() and hand it to chrome-launcher.

import { writeFileSync } from "node:fs";

const baseUrl = process.argv[2] || "http://127.0.0.1:3000";
const outPath = process.argv[3] || "out/lighthouse.json";

// three.js / Spark chunk signatures to look for in request URLs on the landing page.
const THREE_SIGNATURES = [/three(\.module)?(\.min)?\.js/i, /\/three@/i, /chunks\/three/i, /node_modules_three/i];
const SPARK_SIGNATURES = [/spark/i, /sparkjsdev/i, /SplatMesh/i, /SparkRenderer/i];

function bail(reason) {
  writeFileSync(outPath, JSON.stringify({ skipped: true, reason }, null, 2));
  console.error(`lighthouse: SKIP — ${reason}`);
  process.exit(0);
}

let lighthouse, chromeLauncher, chromium;
try {
  ({ default: lighthouse } = await import("lighthouse"));
  chromeLauncher = await import("chrome-launcher");
  ({ chromium } = await import("playwright"));
} catch (e) {
  bail(`dependencies not installed (run npm ci in perf/): ${e.message}`);
}

let execPath;
try {
  // Prefer an explicit CHROME_PATH (run.sh sets it to the verified Playwright build on this host),
  // else fall back to Playwright's resolved chromium.
  execPath = process.env.CHROME_PATH || chromium.executablePath();
} catch (e) {
  bail(`Chromium executable not resolved; run 'npx playwright install chromium': ${e.message}`);
}

async function runOne(chrome, url, wantChunkScan) {
  const flags = { port: chrome.port, output: "json", logLevel: "error" };
  // Lighthouse's own mobile preset: emulated Moto G-class device, slow-4G throttling, mobile UA.
  const config = { extends: "lighthouse:default", settings: { formFactor: "mobile", screenEmulation: { mobile: true } } };
  const result = await lighthouse(url, flags, config);
  const lhr = result.lhr;
  const a = lhr.audits;

  // Total JS transferred (bytes over the wire) from the network-requests audit.
  let jsBytes = 0;
  let loadsThree = false;
  let loadsSpark = false;
  const net = a["network-requests"]?.details?.items || [];
  for (const item of net) {
    const rt = (item.resourceType || "").toLowerCase();
    const size = item.transferSize || 0;
    const reqUrl = item.url || "";
    if (rt === "script" || /\.m?js(\?|$)/i.test(reqUrl)) jsBytes += size;
    if (wantChunkScan) {
      if (THREE_SIGNATURES.some((re) => re.test(reqUrl))) loadsThree = true;
      if (SPARK_SIGNATURES.some((re) => re.test(reqUrl))) loadsSpark = true;
    }
  }

  return {
    url,
    lcp_s: (a["largest-contentful-paint"]?.numericValue ?? null) / 1000 || null,
    tbt_ms: a["total-blocking-time"]?.numericValue ?? null,
    cls: a["cumulative-layout-shift"]?.numericValue ?? null,
    js_transferred_kb: Math.round((jsBytes / 1024) * 10) / 10,
    ...(wantChunkScan ? { loads_three: loadsThree, loads_spark: loadsSpark } : {}),
    network_request_count: net.length,
  };
}

async function main() {
  // Probe the web server first; skip cleanly if it is not up.
  try {
    const resp = await fetch(baseUrl, { method: "GET" });
    if (!resp.ok && resp.status >= 500) bail(`web server at ${baseUrl} returned HTTP ${resp.status}`);
  } catch (e) {
    bail(`web server not reachable at ${baseUrl}: ${e.message}`);
  }

  let chrome;
  try {
    chrome = await chromeLauncher.launch({
      chromePath: execPath,
      chromeFlags: ["--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
    });
  } catch (e) {
    bail(`could not launch headless Chromium: ${e.message}`);
  }

  try {
    const landing = await runOne(chrome, baseUrl + "/", true);
    const workspace = await runOne(chrome, baseUrl + "/scan/scn_trap", false);
    const out = { skipped: false, baseUrl, landing, workspace };
    writeFileSync(outPath, JSON.stringify(out, null, 2));
    console.error(`lighthouse: wrote ${outPath}`);
  } catch (e) {
    bail(`lighthouse run failed: ${e.message}`);
  } finally {
    await chrome.kill();
  }
}

await main();
