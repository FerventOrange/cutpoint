# Contributing

Thanks for helping. Bug reports, fixes, docs and focused features are welcome.

## Set up

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), ffmpeg. A GPU is not needed for the
tests.

```powershell
git clone https://github.com/FerventOrange/cutpoint.git
cd cutpoint
uv sync --extra dev
```

Add `--extra ski --extra events` when working on those modules (list every extra you use;
`uv sync` removes the others).

## Check your change

```powershell
uv run --extra dev pytest
uvx ruff@0.17.1 check src tests
uvx ruff@0.17.1 format --check src tests
```

CI runs the same lint and format checks, and the tests on Windows and Linux. Optionally install
the hooks with `uv run --extra dev pre-commit install`; they run ruff on commit. Ruff settings
(line length 100, rule set) are in `pyproject.toml`.

## Writing tests

Add tests only for cases that deserve them: logic that is valuable and easy to get wrong. Tests
must not need models, a GPU or real footage; build small synthetic inputs instead. Tests that need
ffmpeg skip themselves when it is not on `PATH`.

- Test behaviour the pipeline (analyze -> clips -> assemble, mix -> director -> qa -> repair ->
  render, stitch) or an agent depends on: outputs, `--json` shapes, exit codes, file layout. Not
  internal structure.
- Every real bug gets one regression test naming the bug in its docstring. No variants.
- Always test data safety (never overwrite/delete user files, partial-file cleanup, backups,
  SDK/NDA guards). One test per guarantee.
- Test algorithm decisions (director, QA, crash, mix, stabilization) with small synthetic inputs;
  assert the decision, not exact floats.
- Don't test constants, dataclass defaults, one-line wrappers, log wording, or inputs
  argparse/config validation/our callers already reject.
- One table-driven test per rule, at most 3-4 cases: valid, a typical invalid, the boundary that
  matters.
- Mock only at process boundaries (ffmpeg, SDK, Ollama) and assert outcomes, not just that a mock
  was called.
- Before adding a test, check whether an existing test already covers the path; extend it instead.

## Commits and pull requests

- Branch from `main`; do not commit to `main` directly.
- Commit format: `<type>/<scope> : <description>`, for example
  `fix/clips : keep audio fade when extending` or `docs/readme : add quickstart`. Types used in
  this repository include `feat`, `fix`, `docs`, `chore`, `refactor`, `test`.
- Keep a pull request to one topic. Describe what changed and how you checked it.
- Update the docs in `docs/` when behaviour changes. Only document what the code does.

## Code conventions

- Source videos are never modified; outputs go to the project folder.
- Import heavy optional dependencies lazily and mark the import (see existing `noqa: PLC0415`).
- Do not commit media files, model weights, caches or `config.json`.
- Never commit the Insta360 SDK or paste its output or documentation into issues, pull requests
  or docs. See [docs/insta360-sdk.md](docs/insta360-sdk.md).

## Layout

See [docs/architecture.md](docs/architecture.md).
