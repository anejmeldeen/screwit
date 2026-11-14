// Same-origin proxy from the website to the FastAPI server (AGENTS.md decision 1).
//
// A next.config rewrite would bake the API address into the build, so a built site could only ever
// talk to 127.0.0.1:8000. This handler reads SCREWIT_API_ORIGIN on every request instead, so the same
// build works for dev, tests and the offline demo laptop. Bodies are streamed both ways, which keeps
// the chat and progress SSE streams flowing token by token and lets uploads pass without buffering.

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "host",
  "content-length",
]);

function apiOrigin(): string {
  const origin = process.env.SCREWIT_API_ORIGIN || "http://127.0.0.1:8000";
  const url = new URL(origin);
  // The API must stay on this machine; refuse to forward anywhere else.
  if (!["127.0.0.1", "localhost", "[::1]", "::1"].includes(url.hostname)) {
    throw new Error(`SCREWIT_API_ORIGIN must be a loopback address, got ${url.hostname}`);
  }
  return url.origin;
}

async function proxy(req: Request, ctx: { params: Promise<{ path: string[] }> }): Promise<Response> {
  const { path } = await ctx.params;
  const incoming = new URL(req.url);
  const target = `${apiOrigin()}/api/${path.map(encodeURIComponent).join("/")}${incoming.search}`;

  const headers = new Headers();
  req.headers.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) headers.set(key, value);
  });

  const hasBody = !["GET", "HEAD"].includes(req.method);
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: req.method,
      headers,
      body: hasBody ? req.body : undefined,
      redirect: "manual",
      cache: "no-store",
      signal: req.signal,
      // Required by Node's fetch when streaming a request body.
      ...(hasBody ? { duplex: "half" } : {}),
    } as RequestInit);
  } catch {
    return Response.json({ detail: "API server unreachable" }, { status: 502 });
  }

  const out = new Headers();
  upstream.headers.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) out.set(key, value);
  });
  if ((upstream.headers.get("content-type") || "").startsWith("text/event-stream")) {
    // no-transform stops response compression from buffering the stream.
    out.set("cache-control", "no-cache, no-transform");
    out.set("x-accel-buffering", "no");
  }
  return new Response(upstream.body, { status: upstream.status, headers: out });
}

export const GET = proxy;
export const POST = proxy;
export const PATCH = proxy;
export const PUT = proxy;
export const DELETE = proxy;
export const HEAD = proxy;
