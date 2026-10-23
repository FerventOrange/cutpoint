# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `cutpoint ui` New Project "Choose video…" and "Choose folder…" buttons open the operating
  system's own file or folder dialog on the computer running the server (Windows: system dialog
  through tkinter; Linux, with a display: zenity, falling back to kdialog; macOS: osascript)
  through `POST /api/pick`. Where no dialog is available (headless, Docker, or an SSH session on
  Windows and macOS) the buttons are hidden;
  the Source field always accepts a typed or pasted full path.
- `cutpoint ui` Settings (gear button in the header) sets the projects folder, saved as
  `projects_dir` in `config.json` through `GET`/`PUT /api/settings`. The folder must exist or only
  its last part may be missing (it is created), and it must be writable. Existing projects are not
  moved. The setting is read-only when `CUTPOINT_PROJECTS_DIR` is set and cannot change while a job
  is running.
- `cutpoint ui` sequence rows show a thumbnail per clip, served by
  `GET /api/projects/{pid}/sources/{sid}/frame` and cached in `<project>/ui/frames/` (at most 500
  per project, least recently used evicted, stale stills removed). Rows support inline rename,
  remove with a confirm popover, multi-select with batch remove, up/down buttons and drag reorder.
- `cutpoint ui` Moments empty state for a project that is not analyzed yet: the number of videos
  found, "No moments analyzed yet", an "Analyze footage" button and a "Show folder" button.
  After analysis, "Show folder" sits next to "Analyze again" in the column header.
  "Show folder" opens the folder of the project's videos in the system file manager through
  `POST /api/projects/{pid}/reveal` (token-protected; it only opens the project's own source
  folder). The button is hidden where no file manager is available (for example Docker, or Linux
  without a display); `/api/session` reports `can_reveal`.
- `cutpoint ui` "Show folder" buttons next to "Export clips" (opens the latest export folder,
  `clips_ui_*`) and next to "Assemble" (opens the project folder holding the latest final video
  and its `.fcpxml` and `.otio` timelines). Each appears only once that folder exists and
  where a file manager is available. `POST /api/projects/{pid}/reveal` takes
  `?target=sources|clips|outputs` (default `sources`); the server resolves every path itself.
- Resizable side column in the UI (drag, keyboard, double-click to reset); width is remembered per
  browser.

### Changed

- The UI's New Project Source field no longer has the in-browser "Browse" button and folder
  browser; the `/api/browse` endpoint is removed.
- The UI's "Remove from sequence" button moved above the video preview.
- The UI's Moments column header no longer shows the "Moments" title (kept for screen readers);
  after analysis it shows the moment count, "Analyze again" and "Show folder".
- The `dev` extra is replaced by a `dev` dependency group that `uv sync` and `uv run` install by
  default, so `uv run pytest` works in a fresh clone; `--extra dev` is no longer accepted (#122).

### Fixed

- On Windows, `clips --force`, `extend` in place and `extend --out ... --force` refuse a volume
  mounted into an NTFS folder (for example `C:\mnt\vol`) with the mount-point message instead of
  calling it a drive root. A real drive root (`C:\`, a UNC share root) keeps the drive-root
  message (#185).
- On Windows, a source read through `source_remap` from a disconnected mapped drive is now
  recorded in its network form (`\\server\share\...`), the same as `analyze` records it, so
  exclusions and carried segments match it (#181); older `Z:\...` entries still match. Clip
  manifests (`clips/manifest.json` `source`) now record `\\server\share\...` instead of
  `Z:\...` for a remapped mapped drive.
- `cutpoint mix` and `cutpoint 360 repair` match carried segments, `telemetry.json` start times
  and ordering to candidates, and `360 repair` matches crash impacts to segments, by the resolved,
  normalised source path (symlinks, `a/../b` spellings); `360 repair` reads a relative source in
  the segments file against that file's folder, so a re-mix keeps carried clips and their ids. On
  Windows a drive-relative source such as `D:a.insv` in `mix_exclude.json` is ignored with a
  WARNING instead of silently never matching, and the rest of the file loads (`360 repair` keeps
  the entry in the file when it rewrites it). `360 repair` stores one exclusion for a span already
  stored under another spelling of its source (#181).
- `cutpoint ui`: saving the review or exporting clips in a project made on the command line (no
  `ui` folder yet) no longer fails with a 500 error; the `ui` folder is created first. The project
  list no longer shows a folder that cannot be opened (for example one that is a symlink or
  junction to outside the projects folder). A job whose result could not be recorded (see #168)
  shows the reason as its error, without a `409:` prefix or a traceback in the server log.
- `cutpoint mix` and `cutpoint 360 repair` match an exclusion in `mix_exclude.json` to a segment
  by the resolved, normalised path of its source, so symlinks and spellings such as `a/../b`
  match in both commands and in every repair step. On Windows, `clips --segments` rejects a
  drive-relative source such as `D:foo.mp4` with the same message as `360 repair`.
  A relative `source` in `mix_exclude.json` (or the file given to `mix --exclude-file`) is now read
  against that file's folder instead of never matching, so `mix` honours exclusions written by older
  `360 repair` versions (#175).
- `cutpoint ui` on Windows: cancelling a job now sends `CTRL_BREAK_EVENT` to its process group
  first and waits up to 5 seconds, so the command removes its partial files and exits 130; only
  then does `taskkill /T /F` run (also the fallback when the UI has no console, for example under
  pythonw). The UI no longer treats a hidden leftover `.<name>.part.mp4` as an exported clip when
  deciding whether there is something to assemble (#125).
- Windows: a JSON result file (`highlights.json`, `manifest.json`, cache files, ...) that stays
  locked now gives the same "open in another program" error naming the file as a locked clip,
  instead of a bare `PermissionError`. For "Access is denied" (error 5) the hint also says the
  file or its folder may be read-only. llama-server and `360 stitch` exit codes are shown signed
  with their hex form, e.g. `-1073741819 (0xC0000005)`, like ffmpeg's. A failed `360 director`
  run, or a failed 360 `clips`/`extend` run, no longer leaves an empty `pano_paths/<name>` folder
  it created (#146).
- `360 repair --out` into a folder that does not exist creates it instead of failing with
  FileNotFoundError; a file in the way is an error naming it, and `--out` pointing at an existing
  folder says `<path> is a folder, not a segments file` before any work. Exclusions written by
  earlier versions with a relative source (as spelt in a hand-written segments file) are taken
  against the folder of the `mix_exclude.json` holding them, and written back absolute with new
  ones. A `cutpoint mix` segment whose source goes through a symlink now matches its rendered clip
  instead of being skipped as stale. On Windows, a drive-relative source (`D:clip.insv`) is an
  error asking for the full path (#151).
- Windows: Ctrl+Break (or `CTRL_BREAK_EVENT` sent to the process group) now stops a command
  like Ctrl+C: partial files are removed and the exit code is 130 instead of `0xC000013A`.
  AGENTS.md notes which ways of stopping Cutpoint on Windows give 130 and which end it without
  cleanup (exit 1, or -1 from Stop-Process) (#148).
- A running `cutpoint ui` (or any long-running process) now picks up edits to `config.json`
  without a restart. Valid edits apply; a config broken after startup is reported (HTTP 409 in the
  UI) until it is fixed. A changed `video_encoder` applies to the next encode (#137).
- `cutpoint ui` changing the projects folder (`PUT /api/settings`) and starting a job are now
  mutually exclusive: a job started from a second tab while the folder changes can no longer run
  against the old folder; the settings change is refused with 409 instead, and starting a job
  is refused with 409 if the folder changed after the project was opened (#143).
- `clips --segments`, `360 director --segments`, `360 stitch --segments` and
  `360 repair --segments` pointing at an existing folder now say `<path> is a folder, not a
  segments file`. `360 repair --segments` also says "Segments file not found" for a missing path
  instead of "not a segments list" (#150).
- `360 repair` now says `<path> is not valid JSON: not UTF-8 text (save it as UTF-8)` for a
  UTF-16 `--segments` or `--qa` file (and `<path> is not valid JSON: ...` for broken JSON)
  instead of "not a segments list" or "QA report not found". `--qa <folder>` says `<path> is a
  folder, not a QA report`, and a `--qa` file that is not a JSON object says `<path>: not a QA
  report` instead of crashing (follow-up to #150).
- A `--segments` file saved with a UTF-8 BOM (Windows PowerShell 5.1) now loads for `clips`,
  `360 director` and `360 stitch`; a UTF-16 file gives `<path> is not valid JSON` naming the file
  (#150).
- A wrongly typed `config.json` value (for example `"projects_dir": 5`) now gives one
  `ERROR` line, `<config path>: <key> must be <type>, not <found>`, and exit code 1 instead of a
  traceback, in the CLI and in `cutpoint ui` jobs; the UI's 409 response shows the same message
  (#138, #140). `source_remap_exclude` given as a single string is now rejected instead of being
  split per character.
- `config.json` saved with a UTF-8 byte order mark (Windows PowerShell 5.1
  `Set-Content -Encoding UTF8`) is now accepted; a UTF-16 file gives the
  "Invalid config file <path>" error.
- JSON files saved with a UTF-8 BOM (Windows Notepad, PowerShell) are read correctly, one in
  another encoding (UTF-16, ANSI) counts as unreadable instead of failing with a traceback, and
  `cutpoint ui` leaves out a learned style that fails to load, with a WARNING naming its file,
  instead of failing the request (#126).
- `extend <clips_dir> --out DIR --force` into a non-empty `DIR` now exports into a hidden staging
  folder and swaps the result in only when every clip succeeded (#147), like in-place `extend` and
  `clips --force`: a failure or Ctrl+C leaves `DIR` unchanged. Files in `DIR` that the run does not
  write stay, and a WARNING names the `.mp4` clips left there that were not re-exported (`assemble`
  includes them). Same-named files are overwritten as before. `--out` pointing at a non-empty drive root or mount point with `--force` is refused before any export.
- `clips --force` into a non-empty folder and in-place `extend`, where the clips folder is a
  drive root, a mount point or a Linux bind mount, are now refused before any export with one
  ERROR line naming the folder and saying to use `--out` with a folder on that drive. A drive
  root crashed with an "empty name" error, and a mount point or bind mount failed with a
  permission error or a cross-device rollback only after every clip was exported.
  `extend --out DIR --force` into such a non-empty `DIR` gives the same message, and a
  bind-mounted `DIR` is caught too. `clips` into an empty drive root or mount point exports
  straight into it, as `extend --out` does (a failed run there may keep the finished clips);
  every other folder is still staged (#160).
- A `source_remap` key spelled the Windows way (for example `D:\Videos` -> `/mnt/d/Videos`) now
  applies on Linux and macOS (#149). It matches Windows-style sources case-insensitively with
  either slash, so a Windows project (or `clips --segments` with `D:\` sources) opened on Linux
  finds its footage. Stage keys for Windows-form sources are built as on Windows (no
  current-folder prefix, case-insensitive: `d:\videos\t\x.mp4`), so `stage` and `stage --clear`
  accept any spelling of such a path. Reading a staged copy accepts either slash and any case of
  the `source_remap` prefix; below the prefix the case must match the staged file. The analysis cache key becomes
  `D:\Videos\T\x.mp4` (was `D:\Videos/T/x.mp4`), so those files are analyzed once more and
  staged copies under such a key must be staged once more. Old copies stay until
  `cutpoint stage --clear` with no targets; `--clear <target>` does not match old keys.
- `cutpoint stage` on Linux and macOS keeps files that differ only in case below a Windows-form
  `source_remap` prefix (`T/x.mp4` and `T/X.MP4`) apart: each gets its own copy and index entry,
  the second is no longer reported as "Already staged" or replaces the first's entry, and reading
  one never returns the other's copy. Index entries now record the exact path below the prefix
  (`"rel"`); existing entries keep working without a new copy and get it the next time `stage`
  sees them. A running process (`cutpoint ui`) now picks up copies staged by another process and
  an edited `stage_dir` or `source_remap` without a restart, and stops using a copy that was
  deleted or truncated by hand (#163).
- `cutpoint ui` with an invalid `config.json` (not JSON, a rejected value or a wrongly typed one)
  no longer fails with a 500 and an empty project list (#133). Project routes answer 409 with the
  config file and the error, the banner shows it, and Settings stays usable to fix the projects
  folder. A rejected `video_encoder` now names where it came from: `config.json` or
  `CUTPOINT_VIDEO_ENCODER`.
- `cutpoint ui` project picker shows "Projects unavailable" instead of staying empty when the
  project list cannot be loaded (for example an invalid `config.json`), and "New project" is
  disabled until a later load succeeds. A new project that cannot be opened after creation shows
  the error on the banner (#152).
- `clips --segments` with a missing file now says what a relative path was resolved against (#127);
  a Windows path starting with `\` or `/` is shown as typed, without that note.
- Segment ids: a generated `sNNN` id that a user id already uses gets a letter suffix (`s002b`),
  and duplicate user ids are an error naming both segments, in `clips` and `360 repair` (#120).
- `360 repair` resolves relative `source` paths in a hand-written segments file against the
  file's folder, as `clips` does, and writes exclusions with the absolute path. A written-back
  segments file in the same folder keeps the spelling you used; with `--out` in another folder
  the path is absolute (#119).
- Exports no longer leave a broken file under the real name (#109). Clips, `extend`, 360 renders
  and `final.mp4` are written to a hidden `.<name>.part.mp4` and renamed when complete. A failed
  or cancelled export (Ctrl+C, or the UI cancel on Linux and macOS) leaves no partial file and
  removes the temp file. A hard kill (SIGKILL, or the UI cancel on Windows, which uses
  `taskkill /F`) can leave a hidden `.part.mp4`; every command ignores it and `clips --force`
  removes it.
- `clips --force` and `extend --out <clips_dir> --force` replace the folder all or nothing
  (#109). The new clips are exported into a hidden `.<folder>.cutpoint-tmp` next to the folder
  and swapped in only when every clip succeeded. Before, `clips --force` deleted the old clips
  and `manifest.json` first, so a failed or cancelled run left an empty folder, and an in-place
  `extend` stopped by a locked clip left a mix of new and old clips and a `clips.tmp` folder.
  Now a failure (a GPU encode error, Ctrl+C, a clip locked by another program) leaves the old
  clips and manifest untouched and no temp folder. `clips --force` replaces only what an earlier
  cutpoint run made: the clips listed in the old `manifest.json`, leftover `.part.mp4` files and
  the names it writes, plus `manifest.json`. Before, it deleted every `.mp4` in the folder, so
  with `--out` pointing at the folder of the footage it deleted the source video it had just
  read. Now a file the run reads is never replaced (a new clip name that is a source stops the
  run with an error before any export); any other `.mp4` (a clip renamed during review or added
  by hand) stays and is assembled, with a WARNING naming it; with
  no readable old manifest only the new names are replaced. After a hard kill mid-swap, the
  next `clips` or `extend` on that folder puts the old clips back from `.<folder>.cutpoint-bak` (if its record is unreadable, it stops with an error and leaves the backup for you to restore by hand).
  A failed fresh `clips` run now leaves no partial clips, so it can be re-run without `--force`.
  A leftover `.<folder>.cutpoint-bak` or `.cutpoint-tmp` that cannot be removed (a file in it
  still open, on Windows) now stops the next run with an error that names the folder and says
  to close the program or delete the folder, instead of a raw `[WinError 32]`.
- A failed run of `clips`, `extend` or `assemble` no longer leaves behind an empty output folder
  that it created. When a file stays locked on Windows (open in a player, an editor or a virus
  scanner), the error now names the file and says to close it and re-run. ffmpeg crash codes
  show signed with their hex value, for example `exit -1073741819 (0xC0000005)` (#109).
- With `video_encoder` `auto`, if a GPU encode (NVENC, AMF or VAAPI) fails on real footage, that
  clip, 360 render or final video is redone once on the CPU (libx264) with one WARNING, and the
  rest of the run uses the CPU. A forced GPU encoder is not retried; its error now says to set
  `CUTPOINT_VIDEO_ENCODER=cpu` or `"video_encoder": "cpu"` in `config.json` (#109).
- A failed GPU encoder probe (`cutpoint capabilities`, or a video encoder forced with
  `CUTPOINT_VIDEO_ENCODER`) now reports ffmpeg's first real error, for example "Cannot load
  libcuda.so.1", instead of the trailing "Nothing was written into output file" or "Conversion
  failed!" lines. Forcing `vaapi` on Windows or macOS now says "VAAPI is Linux-only" (#113).
- `cutpoint 360 repair --segments` works on a hand-written segments file (the
  `[{source, start, end, label}]` format of `clips --segments`) instead of crashing with
  `KeyError: 'id'`. For a file not written by `cutpoint mix`, affected segments are trimmed to
  their longest clean part or dropped; nothing is refilled, your segments (and ids, if any) are
  kept, and only `mix_exclude.json` is updated. A segment missing `source`, `start` or `end`
  (or with non-numeric times) gives an `ERROR` line instead of a traceback (#110).
- UI: removing the staged clip with the stage button and clicking again no longer removes a
  different clip; it re-adds the same one, keeping its name.
- `cutpoint ui` binds the port before printing the URL; on a busy port it exits 1 with an error
  naming the port and suggesting `--port`.

### Security

- `cutpoint ui` ignores an `exports.json` entry whose export folder (`dir`) or final video
  (`final`) is `.`, `..` or does not start with a letter, digit or `_`, so a hand-edited or corrupted file can no longer point
  outside the project folder (#141).
- `cutpoint ui` also ignores an `exports.json` export folder or final video (and its
  `.fcpxml`/`.otio` timelines) that a symlink or Windows junction inside the project resolves
  to outside the project folder, so neither is shown, served nor opened in the file manager
  (an export whose `manifest.json` resolves outside is ignored too); a broken symlink named like the next
  `final_*.mp4` or `clips_ui_*` output counts as taken, so no output is written through it
  (#155).
- `cutpoint ui` treats a project metadata file (`highlights.json`, and `meta.json`,
  `review.json` and `exports.json` in the project's `ui` folder) that a symlink or Windows
  junction inside the project resolves to outside the project folder as missing, so it is
  neither read nor makes the folder a project; saving the review, exporting clips (its
  `segments-*.json` file and `exports.json`), recording an assembled final and analyzing
  (`highlights.json`) refuse with a 409 error naming the file instead of writing through the link.
  The preview and thumbnail caches (`ui/proxy`, `ui/frames`) are not used either when they, or
  the `ui` folder, resolve outside the project: no proxy or still is served, built or written
  there (preparing a preview or loading a thumbnail gives a 409 error) and stale stills there are
  never deleted (#168).
- Docker image (#112): uv is pinned by version and digest instead of an unpinned `curl | sh`
  install; the container runs as the non-root user `cutpoint` (uid 1000, build arguments
  `CUTPOINT_UID` / `CUTPOINT_GID`) owning `/data`; `git` and `curl` are no longer installed; and
  `CUTPOINT_LLAMA_BACKEND=cpu` is set. **Behavior change:** the 0.2.0 image ran as root, so an
  existing `cutpoint-data` volume is root-owned and fails with `Permission denied`; fix it once
  with `docker run --rm --user 0 --entrypoint chown --mount source=cutpoint-data,target=/data
  cutpoint -R 1000:1000 /data` (see the Docker section of `docs/getting-started.md`).

## [0.2.0] - 2026-10-22

### Added

- Optional `ui` extra and `cutpoint ui`: a loopback-only browser workspace for standard-video
  analysis, moment preview/trim/reordering, clip export and final assembly. Includes saved
  review sequences, seekable media/proxies, isolated cancellable jobs, and non-overwriting
  exports with MP4/FCPXML/OTIO downloads.
- Runtime-probed AMD video encoding (VAAPI/AMF) alongside NVENC, with software fallback,
  `video_encoder` / `CUTPOINT_VIDEO_ENCODER` overrides, and `cutpoint capabilities [--json]`
  reporting hardware presence separately from usable encoding and transcription support.
- Windows llama.cpp Vulkan/CPU asset selection alongside CUDA, and `CUTPOINT_LLAMA_BACKEND`
  overrides on Windows and Linux. AMD faster-whisper transcription remains CPU-only.
- `cutpoint --version`.
- Linux: `cutpoint setup` downloads the official llama.cpp Linux build (the oldest CUDA build
  with cudart in the newest llama.cpp release, 12.8 today, when an NVIDIA driver is detected;
  otherwise Vulkan, otherwise CPU). `llama-server`
  is found via `CUTPOINT_LLAMA_SERVER`, the bundled build, then `PATH`.
- `./cutpoint` POSIX shell wrapper that runs the repo `.venv`, like `cutpoint.cmd`.
- Linux Docker recipe (`Dockerfile`, `.dockerignore` build-context exclusions) for the CPU browser
  workspace, with an optional `INSTALL_EVENTS` build argument; see the getting-started guide.

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
- **Upgrade note:** CLIP-derived caches change. On existing projects, run `cutpoint events` again
  (and `cutpoint 360 analyze` for 360 footage) before `crashes`, `profile`, `mix` or
  `refmatch --taste`; until then those commands find no visual data. `crashes` then recomputes its
  dense scores, and text embeddings are recomputed on first use. Taste heads trained with
  `refmatch --taste` before 0.2.0 were fit on the old (GELU) CLIP embeddings; re-train them.
- **Upgrade note:** `video_encoder` `auto` now selects AMF (Windows, AMD GPUs) or VAAPI (Linux,
  AMD and Intel GPUs) when it probes as usable, where 0.1.0 used libx264; set
  `"video_encoder": "cpu"` in `config.json` or `CUTPOINT_VIDEO_ENCODER=cpu` to keep software
  encoding.

### Fixed

- `crashes`: camera knocks while boarding a lift no longer count as a crash. On-lift damping now
  also uses the CLIP lift rides that `mix` uses for 360 footage without GPS (Insta360 X3), and it
  reaches 5 s past the event, so contact in the lift line just before the ride is damped too (#15).
- `stage --clear <inputs>` selects exactly the files `stage <inputs>` would copy: it lists the
  original folder as `stage` does (read through `source_remap`) and honours `--recursive` and
  `--exclude`. Previously it listed only the remapped folder and ignored sub-folders, so some
  staged copies were not removed (#103).
- `stage --clear <inputs>` clears each input on its own. An input that no longer exists (nor its
  `source_remap` target) clears the copies recorded under its path in the stage index, honouring
  `--recursive` and `--exclude`; with none recorded it logs a warning. Previously such a copy could
  only be removed by a bare `--clear`, and one missing input stopped the other inputs' copies from
  being cleared. `stage --help` now describes `--exclude` and `--recursive`.
- The CLIP model for `events` and `crashes` uses `ViT-L-14-quickgelu` to match the OpenAI
  weights instead of building them with GELU. The checkpoint is unchanged; cached visual
  features (including 360 views), text embeddings and dense crash scores are recomputed (see the
  upgrade note under Changed).
- Install instructions load `$HOME/.local/bin/env` after the uv installer so `uv` works in a fresh
  shell.
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
- From a clone installed with plain pip on Windows, the install command for a missing `events` or
  `all` extra now notes `--extra-index-url https://download.pytorch.org/whl/cu126` for an NVIDIA
  GPU (plus `--force-reinstall` when a CPU-only torch is installed), as the non-editable pip hint
  does. pip ignores the project's CUDA torch index and would install the CPU build (#104).
- The install command for a missing extra puts its notes in one parenthetical joined with "; "
  instead of up to three in a row, e.g. `... -e ".[events]" (from the repository; with an NVIDIA
  GPU add ... so torch uses CUDA; PowerShell: in cmd drop the leading &)`.
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
