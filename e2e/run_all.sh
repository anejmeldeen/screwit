#!/usr/bin/env bash
# X3 end-to-end runner: runs the backend integration suite and the Playwright browser suite,
# then prints a one-line pass/skip/fail summary for each.
#
# Both suites skip cleanly (never error) when the other teams' branches are not yet merged:
#   - integration needs `python -m screwit.app` (P4).
#   - e2e additionally needs a built Next.js app in WEB_DIR (P1, default <main>/web).
#
# Servers started by the suites bind 127.0.0.1 only and are torn down when the suite ends.
#
# Env:
#   SCREWIT_MAIN_TREE  main checkout holding fixtures + shared venv (default resolved below)
#   WEB_DIR            Next.js app dir for the e2e suite (default <main>/web)
#   PLAYWRIGHT_BROWSERS_PATH  where chromium is installed (default $KIROCREW_SCRATCH or the cache)
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # .../e2e/e2e
WT="$(cd "$HERE/.." && pwd)"                                     # worktree root (screw-it-wt/e2e)
MAIN_TREE="${SCREWIT_MAIN_TREE:-/local/home/brancli/.kiro/crew/workspace/screw-it}"
export SCREWIT_MAIN_TREE="$MAIN_TREE"
VENV="$MAIN_TREE/backend/.venv"
PY="$VENV/bin/python"
BACKEND="$WT/backend"

if [[ ! -x "$PY" ]]; then
  echo "FATAL: venv python not found at $PY (set SCREWIT_MAIN_TREE)" >&2
  exit 2
fi

# Install chromium under a persistent, writable path if Playwright is present.
if [[ -z "${PLAYWRIGHT_BROWSERS_PATH:-}" ]]; then
  export PLAYWRIGHT_BROWSERS_PATH="${KIROCREW_SCRATCH:-$HOME/.cache}/ms-playwright"
fi

summarize() {   # $1 = label, $2 = pytest exit code, $3 = junit xml
  local label="$1" code="$2" xml="$3" line=""
  if [[ -f "$xml" ]]; then
    line="$("$PY" - "$xml" <<'PY'
import sys, xml.etree.ElementTree as ET
r = ET.parse(sys.argv[1]).getroot()
s = r if r.tag == "testsuite" else r.find("testsuite")
if s is None:
    print("no results"); raise SystemExit
t = int(s.get("tests", 0)); f = int(s.get("failures", 0))
e = int(s.get("errors", 0)); k = int(s.get("skipped", 0))
p = t - f - e - k
print(f"{p} passed, {k} skipped, {f} failed, {e} errors (of {t})")
PY
)"
  fi
  case "$code" in
    0) echo "  $label: PASS   ${line:-(see log)}" ;;
    5) echo "  $label: NO TESTS COLLECTED" ;;
    *) echo "  $label: EXIT $code   ${line:-(see log)}" ;;
  esac
}

INT_XML="$(mktemp --suffix=.xml)"; E2E_XML="$(mktemp --suffix=.xml)"

echo "== backend integration suite =="
PYTHONPATH="$BACKEND" "$PY" -m pytest "$BACKEND/tests/integration" -q --color=no --junit-xml="$INT_XML"
INT_CODE=$?

echo
echo "== Playwright e2e suite =="
if "$PY" -c "import playwright" >/dev/null 2>&1; then
  PYTHONPATH="$BACKEND" "$PY" -m pytest -c "$HERE/pytest.ini" "$HERE" -q --color=no --junit-xml="$E2E_XML"
  E2E_CODE=$?
else
  echo "playwright not installed; skipping (install: uv pip install -p $PY -r $HERE/requirements.txt)"
  E2E_CODE=0
fi

echo
echo "==================== SUMMARY ===================="
summarize "integration" "$INT_CODE" "$INT_XML"
summarize "e2e        " "$E2E_CODE" "$E2E_XML"
echo "================================================="

rm -f "$INT_XML" "$E2E_XML"
# Non-zero only on a real failure/error (not on skips or no-collection).
if [[ "$INT_CODE" != 0 && "$INT_CODE" != 5 ]] || [[ "$E2E_CODE" != 0 && "$E2E_CODE" != 5 ]]; then
  exit 1
fi
exit 0
