"""Put perf/ on sys.path so tests import budgets / report / asset_sizes / api_latency directly."""

import sys
from pathlib import Path

PERF_DIR = Path(__file__).resolve().parents[1]
if str(PERF_DIR) not in sys.path:
    sys.path.insert(0, str(PERF_DIR))
