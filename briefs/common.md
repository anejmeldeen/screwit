# Brief rules for every builder

- Main tree (read-only for you): /local/home/brancli/.kiro/crew/workspace/screw-it
- Your worktree is the ONLY place you write: /local/home/brancli/.kiro/crew/workspace/screw-it-wt/<name> (branch ws/<name>).
- Read AGENTS.md first: ownership table, toolchain, and the decisions that OVERRIDE SPEC.md (no bearer/demo token; loopback-only servers; midway fail-closed stub; LLM provider switch; local storage in dev).
- Edit only paths you own. Never run git checkout/reset/clean/stash in the main tree. Do not push.
- Python: shared venv. Run tests as
  `PYTHONPATH=<worktree>/backend /local/home/brancli/.kiro/crew/workspace/screw-it/backend/.venv/bin/python -m pytest <worktree>/backend/tests/<area> -q`
  and lint with `<main>/backend/.venv/bin/ruff check <your paths> --config <worktree>/backend/pyproject.toml`.
- New Python package: `~/.local/bin/uv pip install -p <main>/backend/.venv/bin/python <pkg>` and list it in your report.
- No GPU on this machine. Real model calls (Depth Anything 3, MapAnything, SAM 2, Grounded SAM 2) go behind adapters that lazy-import torch; their tests carry @pytest.mark.gpu. Every stage also needs a CPU fake so it is testable end to end. Verify adapter API names against the projects' official READMEs/docs; do not guess.
- ffmpeg: use `imageio_ffmpeg.get_ffmpeg_exe()` (no ffprobe; parse `ffmpeg -i`).
- Contracts in backend/screwit/contracts/ are the source of truth. Validate every input and output with them. If you need a contract change, do not edit; work around it locally and request it in your report.
- Never bind anything to 0.0.0.0. Never write a credential to a file.
- Do not claim anything you did not run. Commit with `git -c user.name=Kiro -c user.email=brancli@amazon.com commit`.
- STOP when your tests pass, lint is clean and everything is committed. REPORT: what you built, exact verification commands and results, what you could not verify and why, packages installed, contract/interface changes requested.
