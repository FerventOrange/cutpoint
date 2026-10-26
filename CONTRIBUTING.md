# Contributing

Thanks for helping. Bug reports, fixes, docs and focused features are welcome.

## Set up

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), ffmpeg. A GPU is not needed for the
tests.

```powershell
git clone https://github.com/FerventOrange/cutpoint.git
cd cutpoint
uv sync
```

The dev tools (pytest, ruff, pre-commit) are a default dependency group, so no flag is needed.
Add `--extra ski --extra events` when working on those modules. `uv sync` removes the extras
you leave out; see [Optional extras](docs/getting-started.md#optional-extras).

## Check your change

```powershell
uv run pytest
uvx ruff@0.17.1 check src tests
uvx ruff@0.17.1 format --check src tests
```

`uv run pytest` is the full suite as CI runs it; it deselects only tests marked `hardware`. For a
quicker local run, skip the slow and ffmpeg tests (about a third of the time):

```powershell
uv run pytest -m "not slow and not ffmpeg"
uv run pytest -m hardware
```

The second command runs the hardware tests (real GPU, such as the VAAPI encode), which CI lacks.
A `-m` on the command line replaces the default one. Markers (declared in `pyproject.toml`):

- `slow`: spawns real subprocesses or interpreters, or runs longer than about 2 s. Add
  `@pytest.mark.slow`.
- `ffmpeg`: runs real ffmpeg or ffprobe. Use `needs_ffmpeg` from `conftest` (it also skips
  without ffmpeg); otherwise add `@pytest.mark.ffmpeg`.
- `hardware`: needs GPU hardware that CI lacks.

CI runs the same lint and format checks, and the tests on Windows and Linux. Optionally install
the hooks with `uv run pre-commit install`; they run ruff on commit. Ruff settings
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

## Releasing

1. Bump `version` in `pyproject.toml` and add a `CHANGELOG.md` entry for it.
2. Merge the pull request.
3. Create a GitHub release with tag `X.Y.Z` (no `v` prefix) and the CHANGELOG notes.

Publishing the release runs `.github/workflows/release.yml`, which builds the package, checks
that its version equals the tag, and publishes it to PyPI with Trusted Publishing.

To test the release pipeline without releasing, run the Release workflow manually (Actions tab,
"Run workflow"). It builds `<pyproject version>.dev<run number>` and publishes it to TestPyPI
only, never to PyPI. One-time setup first: add a pending trusted publisher on test.pypi.org for
project `cutpoint` (owner `FerventOrange`, repository `cutpoint`, workflow `release.yml`,
environment `testpypi`), and create a `testpypi` environment in the repository settings.

## Layout

See [docs/architecture.md](docs/architecture.md).
