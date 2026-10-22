# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `cutpoint --version`.
- Linux: `cutpoint setup` downloads the official llama.cpp Linux build (the oldest CUDA build
  with cudart in the newest llama.cpp release, 12.8 today, when an NVIDIA driver is detected;
  otherwise Vulkan, otherwise CPU). `llama-server`
  is found via `CUTPOINT_LLAMA_SERVER`, the bundled build, then `PATH`.
- `./cutpoint` POSIX shell wrapper that runs the repo `.venv`, like `cutpoint.cmd`.

### Changed

- `cutpoint setup` downloads the PANNs checkpoint from a Hugging Face mirror first (much faster than
  Zenodo) and falls back to Zenodo. It verifies size and sha256, and moves to the next source when
  one fails or starts too slowly. `CUTPOINT_PANNS_URL` is tried first when set (#30).
- **Upgrade note:** the CUDA libraries for GPU Whisper moved to the `gpu` extra; the base install
  is CPU-only and about 3.6 GB smaller. NVIDIA users: add `gpu` (`uv sync --extra gpu ...`,
  `uv tool install "cutpoint[gpu]" --reinstall`); `all` includes it. Without it, Whisper uses the
  CPU and logs one INFO line with the install command (#27).
- `cutpoint setup` logs which host each download comes from ("downloading <file> from <host>", or
  "resuming ..."), so a fallback to a slower mirror is visible. Mirror warnings name only the host
  (#52).
- Project paths can be written `projects/<name>/...` on every OS; docs use forward slashes.
- Path matching (`source_remap`, `stage` keys) is case-sensitive on Linux and stays
  case-insensitive on Windows.
- Linux: stage index keys changed; run `cutpoint stage --clear` (without inputs) and stage again.
  Ski features are recomputed once.

### Fixed

- `stage --exclude` matches the original path of footage read through `source_remap`, as
  `analyze --exclude` does, instead of the remapped location. Files are still copied from where
  `analyze` reads them (#93).
- `clips`, `extend`, `360 analyze`, `360 director`, `360 stitch` and `360 status` now read footage
  whose raw folder moved through `source_remap`. Previously ffmpeg failed with "No such file", and
  `360 analyze` skipped every moved clip with "no _00_.insv for the IMU". `360 status` finds the
  Studio export next to both the original and the moved camera folder. The project keeps recording
  the original paths (#92).
- From an editable clone install on Windows with a CPU-only torch, the install command for a
  missing `events` or `all` extra now notes that the `uv pip install ... -e` form needs
  `--reinstall-package torch --reinstall-package torchvision` with an NVIDIA GPU, as uv keeps the
  installed CPU torch otherwise. `uv sync` already installs the locked CUDA build (#96).
- `analyze` records its inputs in `highlights.json` `settings.inputs` as absolute paths, keeping
  your spelling (no symlink or mapped-drive resolution). Relative inputs were stored as given, so
  `crashes --reanalyze` or ski `events` run from another directory wrongly reported the footage as
  moved, or re-analyzed whatever matched the relative path there. Projects analyzed earlier with
  relative inputs keep them until `analyze` runs again (#78).
- With `--recursive`, a sub-folder that matches `--exclude` (for example `*/that/*`) is skipped
  entirely, so an unreadable excluded folder no longer logs a warning. The "Input not readable"
  error no longer depends on Python version behaviour (it uses `stat`) (#94).
- `crashes` and `events` skip footage that is not reachable (offline drive, moved file) with a
  warning and continue, and fail before writing anything when none is reachable. With a source
  skipped, `crashes` and ski `events` leave `highlights.json` unchanged instead of re-running the
  analysis; other `events` styles re-rank with the events found (#57).
- `crashes` (default `--reanalyze`) and ski `events` no longer exit 1 with "Input not found" after
  writing results, or silently drop footage from `highlights.json`, when an input or source is not
  at the path `analyze` recorded (moved, or only reachable through `source_remap`). They warn that
  `highlights.json` is left unchanged and exit 0 (#67). Remapped footage is now re-analyzed (#80).
- `analyze` with `source_remap` active records the original path, not the remapped copy, in
  `highlights.json`, the `transcript.txt` headers, read errors, `telemetry.json` and ski feature
  metadata, while still reading the files through the remap. Once the copy is removed and the
  footage is back at its original path, `crashes`, `events` and `clips` find it instead of skipping
  it as "footage not reachable". Caches are reused. Projects analyzed with a remap before this fix
  keep the copy paths until `analyze` runs again; rerun `crashes` and `events` afterwards so
  their results match the new paths (#77).
- `analyze` on an input whose original path is gone but which `source_remap` maps to an existing
  location reads from that location instead of failing with `Input not found`, and records the
  original spelling; for folders, `--recursive` and `--exclude` patterns match the original paths.
  `crashes --reanalyze` and ski `events` re-analyze such remapped projects instead of skipping
  them. They still warn and leave `highlights.json` unchanged only when re-analysis would fail,
  drop a source or re-key one, for example a file moved out of a folder input that still exists,
  or a 0.1.0 project that recorded the copy path (rerun `analyze` for those) (#80).
- Windows: a `source_remap` key and an input on a mapped network drive match whether each is
  written with the drive letter (`Z:\Footage`) or in UNC form (`\\server\share\Footage`), also
  while the drive is disconnected. `analyze` records such a remapped input in UNC form, like a
  reachable one, so a project no longer mixes both spellings of the same footage, and the
  analysis cache and `stage` keys of a remapped copy match the original's (#95).
- `extend` reads each clip's source through `source_remap` like `clips` does, instead of the
  remapped copy recorded in the manifest's `read_from`. Once the copy is removed and the footage
  is back at its original path, `extend` finds it instead of failing on the deleted copy;
  `read_from` is used only when the source cannot be found and the copy still exists. The new
  manifest records where each clip was read from, and `--reuse` still copies unchanged clips when
  only the read location changed (#91).
- Missing-extra messages now give the install command that matches the install type:
  `uv tool install "cutpoint[...]" --reinstall` for tool installs (keeping the extras, source and
  package index already in use, with a CUDA PyTorch index note on Windows, and on Linux for
  drivers too old for CUDA 13), `pip install "cutpoint[...]"` for PyPI installs, and the
  `uv sync --extra` form only in a clone. The pip commands can be pasted into PowerShell.
- The missing-extra install hint no longer shows the CUDA PyTorch index note (or the Linux driver
  note) for tool installs from a clone directory, an editable install or a git URL, which already
  get CUDA torch from Cutpoint's own `[tool.uv.sources]`; it appears only for wheel, sdist, URL
  and PyPI installs (#61).
- The missing-extra hint for an editable install (`uv pip install -e <clone>`) into a venv other
  than the clone's `.venv` no longer suggests `uv sync --extra X`, which installs into the clone's
  `.venv` and not the running environment. It suggests `uv sync` only when Cutpoint runs from the
  clone's `.venv`; otherwise it gives `uv pip install --python "<python>" -e "<clone path>[X]"`
  (#83).
- `cutpoint events` checks for the `events` extra before unloading Ollama models (part of #23).
- The Ollama default URL is `http://127.0.0.1:11434`; `localhost` could hang on some Windows
  machines (#15).
- The Ollama check reports "not reachable" instead of raising on a non-JSON reply (#15).
- `mix`, `profile`, `crashes` and other commands that read cached visual data skip offline footage with a warning instead of failing (#15).
- `360 director --preview` no longer crashes for a clip without a low-res `.lrv` proxy (#15).
- `360 director` and other readers of cached voice data skip offline footage with a "footage not reachable" warning instead of crashing (#46).
- `cutpoint mix` with nothing selected (no candidate fits `--target`, or every candidate is excluded) fails with `ERROR Nothing selected in <project>: ...` and exit code 1, writes no files, and no longer crashes with `ZeroDivisionError` (#47).
- The Ollama availability check reports "not reachable" instead of raising when `/api/tags` returns JSON that is not an object (#49).
- Missing-extra install hints leave out explicit uv indexes (with a note to keep your uv config), show the CUDA PyTorch note also with a mirror or CPU index configured without repeating `--index-strategy`, and, when CPU-only torch is installed, add `--reinstall-package torch --reinstall-package torchvision` (uv pip) or `--force-reinstall` (pip) (#50).
- `360 director` no longer creates folders for a project that does not exist (#15).
- `extend` keeps a backup of a director path file before re-planning over it and never overwrites an earlier one: repeated re-plans keep `<file>.bak`, `<file>.1.bak`, and so on (#15).
- `360 director` keeps a hand-edited path file as `<file>.bak` (then `<file>.1.bak`, and so on) before re-planning over it, like `extend` does, and never overwrites an earlier backup (#48).
- An input folder or file that cannot be read (no permission, stale network mount, name too long) gives a clear `Input not readable` error instead of a traceback. Unreadable subfolders are skipped with a warning during a folder scan, a `source_remap` target that cannot be checked falls back to the original path, and a backup is not written when its name cannot be checked (#79).
- Backups of 360 path files and of `mix_segments.json` are written atomically (temp file, then rename), so a full disk or a failed write never leaves a truncated backup or a truncated `mix_segments.json` (#81).
- `cutpoint 360 director --out-name` keeps the path file of a segment that is no longer selected as `<file>.bak` instead of deleting it (#58).
- `cutpoint 360 director --out-name` retries removing a deselected segment's files on the next run if they could not be removed (for example, a preview open in a player on Windows), says so accurately in the log, and no longer makes a needless "re-planned" backup when a renumbered segment's old name is reused (#69).
- The missing-extra hint for `uv pip install` from a clone directory or git URL no longer asks for the PyTorch index flags on Windows, because uv applies Cutpoint's own CUDA index; it only adds `--reinstall-package torch --reinstall-package torchvision` when CPU-only torch is installed. Wheel and sdist installs and `pip` still get the flags (#71).
- `mix --target` must be greater than 0 (#15).
- `360 repair` no longer leaves a `mix_segments.bak.json` behind when the re-run mix fails (for example "Nothing selected"); the backup is written only after the new mix succeeds (#60).
- `cutpoint 360 repair` no longer overwrites an earlier `mix_segments.bak.json`; it uses the next free name (`mix_segments.1.bak.json`, ...), and an `--out` that names the segments file by another path (relative, `..`) still gets a backup (#68).
- A malformed `CUTPOINT_PANNS_URL` (e.g. `http://[foo`) is now treated as a failed mirror: setup logs a warning naming the bad value (without any `user:password@`) and continues with the Hugging Face and Zenodo mirrors (#59).
- `cutpoint setup` no longer logs credentials, paths or query strings from `CUTPOINT_PANNS_URL`, including values without `//` (such as `user:token@host/x`) and in error messages; only the host is shown (#70).
- `cutpoint setup` verifies llama.cpp downloads against the sha256 digest GitHub publishes: a
  corrupt cached archive is downloaded again, and a mismatch aborts and keeps the existing
  install (#35).
- `cutpoint setup` installs llama.cpp into a staging folder and swaps it in, so switching
  builds or using `--force` no longer leaves stale files in `tools/llama.cpp`. A failed or
  interrupted install keeps the previous one, and setup refuses to replace a `tools/llama.cpp` it
  did not install (no `VERSION.txt`). If a running `llama-server` keeps the previous build from
  being deleted (Windows), setup warns and the next `cutpoint setup --only llama` removes
  `tools/llama.cpp.old` once the server is stopped (#36).
- `cutpoint events` and `cutpoint 360 analyze` check that the model files and tools the selected
  layers need are installed before unloading Ollama models, importing torch or starting a long
  analyze. One error lists every missing file and the `cutpoint setup --only ...` command (#23,
  #31). Runs whose results are cached now also need the model files.
- `cutpoint profile` and `cutpoint crashes` only unload Ollama models when the CLIP steps can run
  (#32).
- `cutpoint profile --context` logs a warning when the context could not be applied (Ollama model
  unavailable, no usable answer), and falls back to a library profile when the context names
  exactly one, for example "mtb" (#24).
- `cutpoint.cmd` and `cutpoint.ps1` header comments now say the `.venv` from `uv sync` is all
  they need.
- GPU Whisper on Linux: the pip-installed CUDA libraries are preloaded.
- Remote-mount detection works on Linux.
- A failed `cutpoint analyze` no longer leaves an empty project folder behind: it removes only a
  folder that run created and that is still empty (#26).
- The Hugging Face "unauthenticated requests" warning is printed once, in Cutpoint's log format,
  instead of twice (#25).
- The missing-extra hint keeps the install's source: a local folder or git commit instead of the
  PyPI release. For `uv tool install` it keeps the package index flags (`--default-index` when that
  was used) and the pinned `--python`, and pip installs on Windows get the CUDA PyTorch index note
  for `events` (#34).
- Projects created on Windows (sources like `D:\Videos\x.mp4`) show correct source file names
  on Linux. Reading the files still needs a `source_remap` entry.
- Clearer ffmpeg-not-found hint on Linux.
- Caption font is chosen with `fc-match sans-serif` on Linux.
- Insta360 X3 file lookup is case-insensitive on Linux.
- Insta360 SDK runs get the needed `LD_LIBRARY_PATH` on Linux.
- `llama-server` is cleaned up when the parent process dies on Linux.
- Project and JSON files respect the umask instead of fixed permissions.
- `cutpoint crashes --force` logs a source without readable GoPro telemetry at INFO instead of
  DEBUG only (#37).
- Insta360 X3: the `_00_` IMU file is reported with its on-disk spelling on every OS; Windows and
  macOS showed the queried name when the case differed (#33).
- `cutpoint 360 analyze` no longer crashes when the footage becomes unreachable before the events re-rank. It warns, leaves `highlights.json` as written before the events, exits 0, and asks you to rerun once the footage is back (#82).

## [0.1.0] - 2026-10-21

First public release.

### Added

- Agent-driven command-line interface (`cutpoint`) with `AGENTS.md` instructions, so a coding
  agent can run the edit and read the result files (`highlights.json`, `manifest.json`,
  `events.json`, `transcript.txt`).
- `cutpoint guide` prints the agent guide (AGENTS.md); it ships in the package for PyPI installs.
- Core pipeline: `analyze` (transcription and scene scoring), `clips` (trimmed, numbered clips,
  optional vertical 9:16 crop) and `assemble` (`final.mp4`, DaVinci Resolve timelines).
- Game-stream events with HUD OCR (`cutpoint events`).
- Action-cam telemetry (GoPro GPMF GPS/IMU) and crash detection, with activity profiles
  (`--context` on `profile` and `crashes`, `cutpoint profile`).
- `cutpoint mix`: variety selection of an action day's moments (runs, jumps, crashes, lifts,
  scenery, social), played in arc order by `cutpoint clips --mix`.
- 360 module for Insta360 footage:
  - stitching through the Insta360 Media SDK, which you obtain yourself (see
    `docs/insta360-sdk.md`);
  - a virtual-camera director with subject framing and a wider framing option;
  - per-frame stabilization with a 70% horizon lock;
  - QA and repair of rendered clips.
- Local models only: Ollama (Qwen vision) for frame rating, plus optional downloaded audio,
  OCR and visual models. No API credits are spent on video.
- Windows and Linux support.

### Known limitations

Planned and not yet available: an agent skill, `cutpoint edit` (one-command editing),
`cutpoint init` (first-run setup), JSON output on stdout for every command, SRT transcripts,
and activity profiles for layers other than crash detection.
