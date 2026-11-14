# Deploy (Screw It)

Files here describe how Screw It runs. P5 owns them; **nothing in this directory is deployed by the
build** — the hackathon operator runs these by hand. Security posture follows SPEC 3.3 and AGENTS.md
decision 1 (no self-minted token; loopback-only servers; SSH/Tunnels for remote access).

- `docker-compose.yml` — Redis (RQ queue) and Postgres (records); no published ports.
- `.env.example` — every secret the stack needs; copy to `.env` (gitignored) and fill it in.
- `Dockerfile.gpu-worker` — the GPU reconstruction/perception worker image (built on the GPU host).
- `Dockerfile.media` — the media-validation containment image (ffmpeg + Pillow); see below.
- `s3-bucket.yaml` — CloudFormation for the private upload/asset bucket.

## Run locally

```bash
cd deploy
cp .env.example .env        # fill in strong random passwords; generate the signing key:
python -c "import secrets; print(secrets.token_hex(32))"   # paste into SCREWIT_STORAGE_SIGNING_KEY
docker compose up -d        # starts redis + postgres on the private compose network only
docker compose ps           # both healthy
docker compose config       # validate the compose file (run in CI too)
```

In dev, storage is the local `./data` disk store (AGENTS.md decision 3), not S3, so no AWS account is
needed. The API (P4) and web app (P1) bind `127.0.0.1` and refuse a non-loopback bind; the Next.js
server proxies `/api/*` to FastAPI so the browser sees one origin.

Neither Redis nor Postgres publishes a port. To inspect one from your laptop, SSH local-forward to
this host rather than publishing a port:

```bash
# from your laptop (does not change the compose file or open any public port):
ssh -L 15432:127.0.0.1:5432 user@host     # then connect psql to localhost:15432
```

For this to reach the container you would run compose with a `127.0.0.1:5432:5432` publish on a
throwaway debug override — never a bare `5432:5432`, which binds `0.0.0.0` and bypasses the host
firewall. Prefer `docker compose exec postgres psql ...` and avoid publishing at all.

## Remote access (SSH local-forward and Amazon Tunnels)

There is no bearer/demo token (AGENTS.md decision 1). The servers are loopback-only; you reach the
demo from elsewhere by **sharing the loopback**, not by widening a bind:

- **SSH local port-forward** to the presenter's laptop or the GPU host:
  `ssh -L 3000:127.0.0.1:3000 -L 8000:127.0.0.1:8000 user@host`, then open `http://localhost:3000`.
  The forward is point-to-point and authenticated by SSH; nothing is published publicly.
- **Amazon Tunnels share** when hosted on a Cloud Desktop: a Midway-authenticated HTTPS URL scoped to
  your team's posix group (https://w.amazon.com/bin/view/Tunnels/). This is the right choice for a
  devbox-hosted demo; it is Midway-gated, so only authorised Amazonians reach it.
- The FastAPI gate (`screwit.app.auth`, P4) admits a request only if it arrived on a loopback socket.
  A `midway` mode exists as a **fail-closed stub** that rejects every request with 401 until a real
  Midway integration is wired — it never admits anyone.

When the web app is hosted on a public HTTPS origin, put **Caddy** (automatic Let's Encrypt) or a
**Cloudflare Tunnel** in front of the API on the GPU host (SPEC 3.3): the camera API and mixed-content
rules both require HTTPS. Set `WebOrigin` in `s3-bucket.yaml` to that HTTPS origin so S3 CORS matches.

## Offline demo build (presenter laptop)

The judged demo must survive a dead venue network (SPEC 8/9). Build the frontend plus the mock-mode
API to run entirely on the presenter laptop at `localhost`:

1. Build the web app with every JS/font dependency bundled (no CDN fetches) — P1's `next build` with
   self-hosted fonts.
2. Run the API in **mock mode** (P4's env flag) with `FixtureStorage` over `fixtures/` and the hero
   assets copied into a local `./data` store, so every route serves fixtures with no external call.
3. Start both before judging and rehearse with Wi-Fi **off**. `localhost` is a secure context, so the
   camera and HTTPS rules do not block anything. The presenter switches the browser tab, not a server.

No S3, Redis, Postgres, GPU or LLM is needed for the offline build — the cached hero scans and the
recorded hero conversation carry the demo.

## S3 bucket (when deploying with cloud storage)

```bash
aws cloudformation deploy \
  --template-file s3-bucket.yaml \
  --stack-name screwit-storage \
  --parameter-overrides BucketName=screwit-uploads-<account>-<region> \
                        WebOrigin=https://<your-web-origin> \
  --region <region>
# validate first:
cfn-lint s3-bucket.yaml
```

The bucket is private with all four Block Public Access settings on, SSE, versioning, TLS-only access,
a `scans/` lifecycle expiry, and CORS limited to the exact web origin for POST/GET/HEAD. The browser
only ever receives short-lived pre-signed POST (upload) and signed GET (asset) URLs minted by the API;
it never gets any standing access to the bucket.

## GPU worker image

Built on the GPU host, not in dev (no local GPU). It bakes Depth Anything 3 (+ pinned gsplat kernels),
SAM 2 and Grounded SAM 2 and all weights at build time so nothing downloads at demo time. Fill in the
upstream repo commits and the licence-approved geometry checkpoint (SPEC 1.5) before building:

```bash
docker build -f deploy/Dockerfile.gpu-worker -t screwit-gpu-worker:latest .
```

Record the **approved geometry checkpoint** here once the licence gate (SPEC 1.5) is cleared with the
organisers: `DA3NESTED-GIANT-LARGE-1.1` (CC BY-NC 4.0, splat path) if accepted for this event, else
`facebook/map-anything-apache` (Apache-2.0, mesh-only, no splat). ___approved checkpoint: TBD___

## Media-validation containment image

Every ffmpeg probe/decode and photo-header read of an untrusted upload runs contained
(`screwit.storage`). On a host where Docker is available, the validator prefers a per-job container:

```bash
docker build -f deploy/Dockerfile.media -t screwit-media:latest .
```

The validator then runs `docker run --rm --network none --read-only --cap-drop ALL --security-opt
no-new-privileges --user 65534:65534 --pids-limit N --memory M --cpus C` with only the per-job
directory bind-mounted writable. Set `SCREWIT_MEDIA_IMAGE` to override the tag. Where Docker is not
available it falls back to the rlimits mode (clean env, scratch-only cwd, CPU/AS/NPROC/FSIZE rlimits,
wall-clock timeout, plus `unshare -rn`/`bwrap` for no-network when those work unprivileged). The
validator records which mechanism ran in `last_containment`.

## Security checklist (verify before any hosting)

- [ ] `docker compose config` passes; no service has a bare `host:port` publish. Any publish is
      `127.0.0.1:...` only. Redis and Postgres publish nothing.
- [ ] `deploy/.env` exists, holds strong random secrets, and is gitignored. No credential value is
      committed anywhere (grep the repo).
- [ ] Redis requires a password; Postgres keeps its own auth (no `trust`, no empty password).
- [ ] API and web servers bind `127.0.0.1` and refuse a non-loopback bind; `midway` mode stub is
      fail-closed (401 for everyone) until real Midway is wired.
- [ ] S3: all four Block Public Access on, SSE on, TLS-only policy, CORS origin == the real web
      origin, lifecycle expiry set. No public ACLs, no public bucket policy.
- [ ] GPU worker is not internet-reachable; only the API is public, and only during judging, behind
      Caddy/Cloudflare HTTPS.
- [ ] `grep -RnE '0\.0\.0\.0|:::|0\.0\.0\.0/0' deploy/` returns nothing (no wildcard binds/ingress).
- [ ] Demo link uses SSH local-forward or a Midway-gated Tunnels share; there is no bearer token.
