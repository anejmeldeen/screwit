// Runtime configuration for the web app.
//
// The browser always talks to its OWN origin: Next.js rewrites /api/:path* to
// the FastAPI server (see next.config.mjs), so there is a single origin and no
// CORS. Per AGENTS.md decision 1 there is NO bearer/demo token anywhere; the
// API gate is loopback-only. Asset (signed) URLs are fetched as given.

// API calls are same-origin, rooted at /api.
export const API_BASE = "/api";
