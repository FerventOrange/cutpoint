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

- Project paths can be written `projects/<name>/...` on every OS; docs use forward slashes.
- Path matching (`source_remap`, `stage` keys) is case-sensitive on Linux and stays
  case-insensitive on Windows.
- Linux: stage index keys changed; run `cutpoint stage --clear` (without inputs) and stage again.
  Ski features are recomputed once.

### Fixed

- Missing-extra messages now give the install command that matches the install type:
  `uv tool install "cutpoint[...]" --reinstall` for tool installs (keeping the extras, source and
  package index already in use, with a CUDA PyTorch index note on Windows, and on Linux for
  drivers too old for CUDA 13), `pip install "cutpoint[...]"` for PyPI installs, and the
  `uv sync --extra` form only in a clone. The pip commands can be pasted into PowerShell.
- `cutpoint events` checks for the `events` extra before unloading Ollama models (part of #23).
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
- Projects created on Windows (sources like `D:\Videos\x.mp4`) show correct source file names
  on Linux. Reading the files still needs a `source_remap` entry.
- Clearer ffmpeg-not-found hint on Linux.
- Caption font is chosen with `fc-match sans-serif` on Linux.
- Insta360 X3 file lookup is case-insensitive on Linux.
- Insta360 SDK runs get the needed `LD_LIBRARY_PATH` on Linux.
- `llama-server` is cleaned up when the parent process dies on Linux.
- Project and JSON files respect the umask instead of fixed permissions.

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
