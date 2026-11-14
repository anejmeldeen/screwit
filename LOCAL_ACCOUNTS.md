# Local accounts

Start the app normally with `./run.sh start`, then log in with **admin / password123**.
Use **Create an account** on the login page for another person. New accounts start with an empty
build gallery; new scans are automatically saved to the signed-in account. Sign out from the header
to switch accounts.

Accounts, salted password hashes, sessions and build ownership use the existing local SQLite
database. No Supabase or external authentication service is needed. `run.sh` uses
`data-live/screwit.db`; direct backend launches default to `data/screwit.db`. Keep using the same
data directory to retain the same accounts and builds. The database and scan files should be backed
up together.

On startup, the app creates admin if it does not exist and assigns the existing examples and any
previously unowned scans to it. It does not reset passwords or change existing ownership on restart.
Sessions last 30 days, use an HttpOnly, SameSite=Strict cookie, and are revoked on sign out. The
loopback-only server restriction remains in place. This is a local account system; email, password
reset and cloud sync are not included.

Build ownership is enforced on every scan API route, including chat, edits, exports and streams,
and on local asset downloads and uploads. Admin owns the initial examples but does not get access
to other people's builds. Account-dependent responses are not cached.

## Verification

- `cd web && npm run typecheck && npm run lint && npm test && npm run build`
- `cd backend && .venv/bin/ruff check screwit/app tests/app`
- `cd backend && SCREWIT_MODE=mock .venv/bin/python -m pytest tests/app -q`

Account tests cover registration, login failures, duplicate usernames, session expiry/revocation,
restart persistence, existing-build migration, and isolation of reads, mutations, uploads, assets
and CAD records. The integration and browser hero-flow fixtures sign in as admin before accessing
examples.

Verified locally: 83 app tests (the original 80-test run plus the final 28-test account suite),
20 API integration tests, and 51 frontend tests; Python lint, TypeScript, ESLint and the production
build pass. Browser checks cover the login layout, admin sign-in, both gallery examples, opening
the knob workspace, and sign-out. Production first-load JavaScript is 106 kB for login and 109 kB
for the gallery. GPU processing and the full throttled phone performance suite were not run for
this account change. No dependencies were installed and no shared contract changes were needed.
