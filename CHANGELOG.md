# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `--json` on `crashes` and `profile`. `crashes` reports the `crashes.json` path, the crash
  count, each crash with its `crashes.json` keys (`id`, `source`, `t0`, `t1`, `confidence`,
  `confirmed`, `origin`, `clip`, `sheet`), the `activity.json` path and the profile name; a
  missing CLIP or torch is `missing_extra`. New warning codes: `footage_unreachable`,
  `no_telemetry`, `no_visual_layer`, `highlights_not_updated` and `context_not_applied`. An
  unknown `--activity` is `bad_args` (still exit 1) (#4).
- Docs for the `--json` result envelope: a "JSON output" section in `docs/commands.md` (envelope,
  error and warning codes, `outputs` keys per command), `--json` in the options of each command
  that has it, and `AGENTS.md`, the skill and the FAQ no longer call it planned; `events`,
  `crashes` and `profile` are still planned (#4).
- `--json` result envelope, starting with `cutpoint paths --json`: the command writes exactly one
  JSON object to stdout, `{"ok", "command", "schema": 1, "project", "outputs", "warnings",
  "error"}`, with `error.code` one of `bad_args`, `not_found`, `ffmpeg_missing`,
  `missing_extra`, `output_not_empty`, `interrupted` or `error`. Logs, stray output and the
  output of child processes go to stderr; exit codes are unchanged. Other commands follow;
  the existing `--json` of `profile` is unchanged (#4).
- `--json` on `analyze`, `clips`, `assemble`, `edit` and `extend`: `outputs` lists the files
  written (absolute paths) and their key facts, such as the candidate count, each clip with its
  subtitles, source and times, and the final video's duration. `edit` writes one envelope with
  the outputs and warnings of all its steps. Clips kept in a clips folder give a `kept_clips`
  warning, and a vision rating that could not run (Ollama or the model not available) gives a
  `vision_unavailable` warning; a bad `--crop`, `--size` or `--clip-override`, unknown `--ids`,
  or `extend` with nothing to do, is `bad_args` (still exit 1). `extend`'s "Kept" warning now names the folder by its absolute
  path, and an in-place `extend` gives it for clips without a manifest entry instead of saying
  they were skipped (#4).
- `--json` on `capabilities`, `mix`, `stage`, `setup` and every `360` subcommand (`scan`,
  `analyze`, `status`, `director`, `stitch`, `check`, `qa`, `repair`). `outputs` holds the files
  written (absolute paths) and their key numbers: `mix` the segments, report and sheet files and
  each moment with its duration; `stage` each file with its source, staged copy, size and
  whether it was `copied` or `skipped` (`--list`: the staged copies, `--clear`: the count
  removed); `setup` per component each file with its size and whether it was `downloaded`,
  `found` or is `missing`, and for `pip` the missing modules; the `360` commands their
  `pano.json`, path JSONs, previews, sheets, `qa.json` or stitched files, never the Insta360
  SDK's own output. New warning codes: `torch_no_cuda`, `sdk_needs_nvidia`,
  `exclusions_ignored`, `content_qa_failed`, `nothing_staged_under_input`,
  `events_extra_missing`, `events_not_in_highlights`, `not_a_360_clip`,
  `vl_check_unavailable`, `stitch_failed`, `qa_clips_skipped` and `qa_clips_stale`. `stage`
  without inputs, a bad `mix --quotas` and a bad `360 director --out-name` are `bad_args`
  (still exit 1) (#4).
- `cutpoint skill install [--dir DIR] [--force]` installs a Claude Code skill
  (`~/.claude/skills/cutpoint/SKILL.md` by default) so an agent finds Cutpoint for requests such
  as "edit this video" or "find the best moments" and runs the edit loop; it points to
  `cutpoint guide` for details. A differing installed skill is kept unless `--force` is given.
  `cutpoint skill print` prints the skill to stdout (#1).
- `cutpoint edit <video...> --kind highlights|montage|clips|funny-moments` runs `analyze`,
  `clips` and `assemble` in one command and copies the result to `<video>-final.mp4` and
  `<video>-final.srt` in the project folder. Each kind sets the style, selection, clip order,
  transition, loudness normalisation and captions; `--style`, `--top`, `--max-total`,
  `--vertical`, `--force` and `--events` override it. `--vision` / `--no-vision` work as on
  `analyze` (default: Ollama rating if reachable) (#2).
- SRT subtitles: `clips` writes `001_<label>.srt` next to each clip and `assemble` writes
  `final.srt` (`final_vertical.srt`, ...) next to the final video, from the cached transcripts
  of `analyze`, timed to the clip and to the joined timeline including crossfades and padding.
  No transcription runs for them; without a cached transcript there is no `.srt` and one INFO
  line. `clips --force` replaces only the `.srt` files its old `manifest.json` lists (#5).
- `cutpoint capabilities` reports torch (installed, version, CUDA or CPU-only build, whether it
  reaches the GPU). On a machine with a usable NVIDIA driver, a CPU-only torch gives a warning
  with the exact reinstall command for the install type (clone, tool install, pip), and a CUDA
  build that cannot reach the GPU gives driver advice. It also reports the Insta360 Media SDK
  (found or not) and warns that the SDK needs an NVIDIA GPU when none is found; `360 stitch`
  logs the same warning before it runs. New JSON keys: `torch` and `insta360_sdk` (#22).
- `cutpoint ui` preview player has a mute button and a volume slider next to Play, shared by
  moments, clips and the final video and remembered per browser (default unmuted at 100%). `M`
  toggles mute (#142).
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
- `360 qa` counts the bad frames (blocked or low information) that `360 director --vl-check`
  stored in the path JSON: a clip fails with more than 1 inside its span; clips without a `vl`
  record are judged as before and `qa` does not call Ollama. `qa.json` gets a per-clip `vl`
  field and the edit totals `vl_checked` and `vl_bad`, and the log table a `vl` column (#16).
- `final.fcpxml` and `final.otio` hold a cross-dissolve at each join when `assemble` uses
  `--transition fade` (and for `edit` kinds that crossfade). It starts on the edit, and the
  clips and total length match `final.mp4`; with cuts the timelines are unchanged. On the OTIO
  audio track a clip without audio now leaves a gap, so later audio stays in sync (#246).
- HUD presets can define event types with an `events` list (`name`, `when` matchers, `score`,
  `max`, `payoff`, `tail_s`, `seed`) and the vision prompt's `title`, `view` and `hype`: custom
  events add to the highlight boost, extend clip ends and set the priority of vision windows.
  Preset types are checked before the built-in ones and can replace them (#8).

### Changed

- The HUD presets folder in `CUTPOINT_HOME` (`cutpoint paths`) is now trusted like the bundled
  folder: `"builtin": true` is honoured there, and only project `hud_presets` folders ignore it
  (with a WARNING). User and bundled presets never get a separate VLM cache key; a project preset
  gets one only when its prompt differs from the user (or bundled) preset of the same game key.
  After adding a user preset with `title`, `view` or `hype`, or editing those fields in one, run `cutpoint events <project> --force vlm`.
- The Ollama vision rating is opt-in for every style: `analyze` and `edit` use it only with
  `--vision`, and `cutpoint ui` only when "Rate frames with a local vision model (Ollama)" is
  ticked next to "Analyze footage" or in the dialog "Analyze again" now opens (there it starts as
  the previous analysis asked). Before, it ran whenever Ollama was reachable (except for `ski`),
  loading about 7 GB into GPU memory unasked. `--no-vision` is still accepted and changes
  nothing. The `vision_unavailable` warning now only comes with `--vision`. `highlights.json`
  records the request as `settings.vision_requested`; re-analyzing (`cutpoint events`) uses
  vision only if the earlier run asked for it, so a project analyzed before this change (no
  record) is re-analyzed without it. The style setting `vision_default` is gone. Building an activity profile
  (`profile`, `crashes`) and `360 director --vl-check` also unload the vision model when they
  are done with it (`--vl-check` also when it fails or is interrupted). With CLIP installed,
  building a profile now unloads only that model, not every model loaded in Ollama.
- `profile --json` writes the result envelope: the `activity.json` record it printed before is
  under `outputs` with the same keys, plus `activity_file` (its path) (#4).
- `cutpoint capabilities --json` writes the result envelope: the report it printed before is now
  its `outputs`, with the same keys (`platform`, `gpu_vendors_present`, `video_encoders`,
  `active_video_encoder`, `transcription`, `llama_cpp`, `torch`, `insta360_sdk`); read
  `outputs.torch` instead of `torch`. The torch and Insta360 SDK warnings are also in
  `warnings` (#4).
- `cutpoint 360 repair --json` writes the result envelope: the report it printed before is now
  its `outputs`, with the same keys (`segments`, `written`, `findings`, `skipped`, `trimmed`,
  `dropped`, `added`, `capped`, `exclusions`, `rendered_s`, `content_qa`) plus `exclude_file`;
  the repair summary is now logged to stderr with `--json` too (#4).
- `cutpoint 360 qa --json` is the result envelope. `--json-only` (skip the contact sheets) is
  unchanged; `--json` no longer abbreviates it (#4).
- `cutpoint ui` runs its jobs (analyze, clips, assemble, ...) and proxy encodes at below-normal
  priority (Windows `BELOW_NORMAL_PRIORITY_CLASS`; elsewhere `nice -n 10` when available), and
  the ffmpeg they start inherits it, so the desktop and the browser stay responsive. Cancelling a
  job works as before.
- `cutpoint ui` preview proxies have a keyframe every 30 frames and at most 30 fps, so seeking in
  a proxy is about as quick as in the source (the encoder default allowed up to ~0.4 s per seek).
- `cutpoint ui` encodes preview proxies with NVENC (`h264_nvenc`) when it works on the machine and
  `video_encoder` is `auto` or `nvenc`, with a bitrate cap that keeps the file near libx264's
  size; otherwise, and when an NVENC encode fails (it is retried with libx264, which is then used
  for the rest of the run), with libx264 as before. On a 1080p60 recording this took about half
  the CPU time of libx264.
- `cutpoint edit` takes `analyze`'s `--vision-top`, `--ollama-url` and `--ollama-model` with the
  same defaults and checks (`--vision-top` must be greater than 0) and passes them to the analysis.
- The UI's New Project Source field no longer has the in-browser "Browse" button and folder
  browser; the `/api/browse` endpoint is removed.
- The UI's "Remove from sequence" button moved above the video preview.
- `cutpoint ui`: the preview is re-laid out and fits the window without scrolling. At the top,
  the moment's name and source on one row with "Add to sequence" (the Preview title is kept for
  screen readers only); the video letterboxed in a full-width panel that takes the height the
  other rows leave (up to 16:9 of the column width, at least 180 px; a very short window scrolls
  the column), above one transport bar (small clock on the left, a play/pause icon button in the
  middle, mute and volume on the right); a full-width trim rail with tick divisions and a gold
  selection; then Clip Start, the transcript in a rounded box and Clip End side by side, each
  field with "-0.5 s", "Use playhead" and "+0.5 s" in one row; the score and signals are a small
  line under them. There is no keyboard hints line; the shortcuts (Space, M, `[` and `]`) work
  as before, Play's tooltip names Space and the "Use playhead" tooltips name `"["` and `"]"`. In
  a narrow column Clip Start and Clip End go above the transcript, then stack.
- The UI's Moments column header no longer shows the "Moments" title (kept for screen readers);
  after analysis it shows the moment count, "Analyze again" and "Show folder".
- The `dev` extra is replaced by a `dev` dependency group that `uv sync` and `uv run` install by
  default, so `uv run pytest` works in a fresh clone; `--extra dev` is no longer accepted (#122).
- `cutpoint 360 qa` on a missing folder says "clips folder not found", like `360 check` and
  `360 stitch --from-clips` (#200).
- `cutpoint 360 qa` checks `--size` when the command line is parsed: a bad value is a usage
  error (exit 2, like `clips --size`), also with `--paths`.
- The Docker base image is pinned by digest (Dependabot proposes updates), and CI builds the
  image and runs `--version`, `capabilities` and a non-root `paths`; a weekly workflow also
  builds the `INSTALL_EVENTS=true` image and imports `librosa` and `torch` (#121).
- `cutpoint 360 director` paths are evaluated with a monotone cubic (PCHIP) between keys instead
  of linear interpolation: speed no longer jumps at keys and a value never overshoots its keys.
  A key interval falls back to exact linear where the cubic would pan faster than the nearby
  linear speed, or go over a `360 qa` pan limit that linear stays under (for example a move
  between two stops or the fastest stretch), so the peak pan speed never rises above linear and
  qa pan gates are not made worse. Yaw across +-180 degrees stays smooth, shots with 2 keys are
  unchanged and repeated key times still act as a step. Existing `360 director --preview` videos
  are reused and keep the old motion until deleted; paths are not re-planned (#14).

### Removed

- The bundled game HUD presets (`r6.json`, `wt.json`) and the game-name
  hints that picked them. Detection uses each preset's `match` field. To keep a game preset, copy
  yours to the user presets folder (`cutpoint paths`), where `"builtin": true` is honoured, or let
  region discovery write one (#8).

### Fixed

- `analyze` (and `edit`) unloads the Ollama vision model right after the vision pass, also when
  it fails or is interrupted, instead of leaving it in GPU memory (about 7 GB for qwen3-vl:8b)
  for Ollama's default 5 minutes. Only the model the run used is unloaded; if unloading fails,
  the run goes on (a DEBUG line).
- The vision retry for the `ski` style now asks about skiing (jumps, crashes, chairlifts) like
  the first attempt, not about menus and scoreboards.
- In generic (verb-style) kill feeds the player's own deaths no longer count
  as player kills in the highlight boost, so moments with deaths rank lower. The
  teamkill check for "REVERSE FRIENDLY FIRE" now matches the start of a banner, like payoff
  banners.
- The player's own death in the built-in coloured-box kill feed no longer adds the kill boost:
  a new default event type `player_deaths` is checked after `teamkills` and before `player_kills`.
  The death is still a tail payoff and a VLM seed, so the reaction after it stays; a self-kill
  still counts as a kill. This needs the player's name in `player_tags` or `--player`. The VLM
  hint now says "the PLAYER died" or "the PLAYER's kill" for built-in feed kills too.
- `--style ski`: a stretch of footage at one score (a 15-30 s `cutpoint events` VLM window, a
  long fast section) gives one clip instead of several back-to-back clips that filled the top
  10. Each pick takes the run around its peak that stays above 70% of it, centred on a flat top,
  and a lower peak with no dip between it and an earlier pick counts as part of that moment;
  peaks separated by a dip, such as two jumps on one run, are still picked. On the Day-1 test
  footage without events the top 10 went from 6 to 10 distinct moments. `cutpoint mix` picks
  its ski candidates the same way, so a long stretch now gives it fewer candidates (#282).
- `--style ski`: friends in view no longer lift the lift/stop mask, and the VLM `friends_faces`
  tag counts less as action: scaled by 1 - idle/10, so not at all in a window the VLM tags as
  fully `chairlift`, `standing_talking` or `lodge`, nor on a lift or while stopped. Standing and
  chatting at the top or unloading from a chairlift no longer gets the top VLM score; friends
  while riding still raise a moment. A learned preset's `mask_friends` above 0 now means a hard
  mask (0 is still the soft mask), and `cutpoint fit` and `cutpoint refmatch` fit the soft and
  the hard mask once each instead of six thresholds that now give the same result. Saved
  features still count friends in `vlm` until `analyze --style ski` is re-run; refit learned
  presets after that (#283).
- `cutpoint ui` no longer logs a `ConnectionResetError` (`[WinError 10054]`) ERROR traceback each
  time the browser cancels a video request, such as on a seek or when switching sources.
- `assemble --loudnorm`, and every `edit` kind that normalises loudness, no longer fails with
  "Requested resampling engine is unavailable" on ffmpeg builds without libsoxr (such as the
  gyan.dev "essentials" build). The limiter uses soxr only when the build has it, else ffmpeg's
  default resampler.
- `cutpoint ui` shows one banner when the server restarted or the link's session token is stale
  ("The Cutpoint server restarted or this link expired. Open the new link printed in the
  terminal.") and stops polling, prepare retries and timers, instead of failing repeatedly.
  Other 403 errors (host or origin guard) are unchanged.
- `cutpoint ui` applies edits to `source_remap` and `source_remap_exclude` in `config.json` on
  the next request; the cached source list used to keep the old mapping for up to 60 s.
- `cutpoint ui` no longer probes every source with ffprobe when a project opens (200 sources took
  about 12 s, now about 0.1 s). Each source is probed when the player first loads it.
- `cutpoint ui` shows the real error for an unanalyzed project whose source folder is missing or
  unreadable, with how to fix it (reconnect the drive or set a source remap), instead of "No
  videos found".
- `cutpoint ui` lists an unanalyzed project's sources from the `source_remap` target when the
  source folder has moved.
- `cutpoint ui` caches each project's source list, so preview and media requests no longer re-list
  the source folder or re-read `highlights.json` on every request, which was slow on network
  shares. Changes are picked up on the next request: the cache is checked against
  `highlights.json`, `ui/meta.json` and the picked folder's modification time, with a 60 s
  backstop (#129).
- `highlights.json` errors now read "has no sources list" / "has no candidates list", and
  `{"candidates": null}` gives this error instead of a crash.
- `extend --force` (and in-place `extend`) also replaces leftover `.part` files from a hard-killed
  run. Its "Kept" warning now matches `clips`, including the wording when there is no manifest.
- Re-analyzing a project (`cutpoint events`) keeps `vision_top` and the Ollama URL and model; the `cutpoint ui` Analyze action keeps them too.
  `analyze` now stores them in the `highlights.json` settings; an automatic `vision_top` stays
  automatic.
- `cutpoint ui`: a clip removed from the Sequence and added back returns to its previous position
  (clamped to the current length) instead of going to the end. Removed clips that are not ranked
  moments are listed under "Removed clips" with an "Add back" button until the page reloads (#128).
- `cutpoint ui`: the outputs panel also offers each final's `.srt` subtitles (`final.srt`,
  `final_ui_*.srt`, `<name>-final.srt`) for download, served as plain text.
- A `highlights.json` without a `sources` list now gives an error naming the file and saying to
  re-run `cutpoint analyze`, instead of a `KeyError` traceback, in `events`, `crashes`, `refmatch`
  and `profile`/activity.
- A corrupt, unreadable or non-object `highlights.json` now gives an error naming the file and
  saying to re-run `cutpoint analyze`, in `clips`, `events`, the commands that read highlights
  through them, and re-analysis. Before, it was reported as missing or crashed with a traceback;
  a missing file still gives the old message (#228).
- `cutpoint ui`: the assemble job also picks a fresh final name when that final's `.srt` (for
  example `final_ui_1.srt`) already exists, so it no longer overwrites it (#224).
- `cutpoint ui`: a cancelled job shows "stopping" (status `cancelling` in `/api/jobs`) until its
  process exits, and Cancel is disabled meanwhile (#225).
- `cutpoint ui`: Cancel gives the job about 12 s (2 x the 5 s soft stop plus a margin) to remove
  its partial files before ending the process tree. Before, it was 5 s, which could end the job
  while it was still stopping ffmpeg or MediaSDKTest and leave a large partial file. The Cancel
  request returns at once and the job shows "stopping" until it exits. Preview proxies are still
  stopped after 5 s (#257).
- `cutpoint ui` has a favicon (#226).
- `cutpoint ui`: the outputs panel lists every final video in the project folder (`final.mp4`,
  `final_*.mp4`, `<name>-final.mp4`, ...), newest first with the export's own final first, each
  with its `.fcpxml` and `.otio`. Until a new export is assembled, the preview of an earlier
  final is titled "Earlier final" (#227).
- `assemble` with hard cuts and added audio fades (clips without a baked fade) trims each
  clip's audio to where the next clip starts (its container duration), as the video and
  `final.srt` already do. Clips whose audio ran past their video drifted: 3 such clips gave
  6.6 s of video and 6.0 s of audio, now both 6.6 s. Clips cutpoint exports change by under
  half a frame (#235).
- `assemble --fade-dur` must be greater than 0 (exit 2), and a crossfade lasts at least one
  frame (shorter fades are raised to one frame, INFO). `0` or a fade under one frame used to
  cut the video short: two 2 s clips at 25 fps gave 2.04 s of video. A clip too short for a
  one-frame crossfade makes the join use cuts (WARNING) (#238).
- `assemble --transition fade` pads or trims each clip's audio to its video before the
  crossfade, as the video and `final.srt` already follow the video. A clip whose audio ran past
  its video pushed the later audio late: a 2 s clip with 2.3 s of audio crossfaded with a 2 s
  clip gave 3.76 s of video and 4.06 s of audio, now both 3.76 s.
  The fix also covers clips whose audio is shorter than the clip or missing: such audio no longer
  pulls the later clips' audio early or leaves extra audio.
- `extend` into a folder with a manifest (in place, or `--out` with `--force`) removes the `.srt`
  the old manifest lists for a clip deleted during review, as `clips --force` does. Before, the
  new manifest dropped the clip's entry and kept its `.srt`, which was then unlisted and never
  replaced or removed by a later run. An `.srt` the old manifest does not list stays (#239).
- `extend --out DIR --force` removes a clip that `DIR`'s manifest lists but this run does not
  re-export (deleted from the input folder during review), with its `.srt`, as `clips --force`
  does. Before, both stayed in `DIR` but not in the new manifest, so the clip was reported as
  added by hand, `assemble` included it, and no later run replaced or removed the `.srt`.
  Clips and `.srt` files the old manifest does not list still stay.
- `clips --force` replaces only the clips and `.srt` files whose name is spelled on disk as
  the old `manifest.json` lists it, as `extend` does. On a case-insensitive file system a clip
  renamed in case only during review (listed `001_joke.mp4`, renamed `001_Joke.mp4`) and its
  `.srt` were removed without a warning when the new run did not write that name; they now
  stay, and the WARNING about kept clips names the clip. A listed `.srt` renamed in case only
  still goes, under its new spelling, when `clips --force` or `extend` writes or replaces its
  clip, so no `.srt` with the old cues stays unlisted when the new run has no transcript.
- `assemble --transition fade` rounds the crossfade to whole frames (of the first clip's
  rate) and uses that length for the video, the audio and the cut offsets in `final.srt`.
  xfade used whole frames while the audio and subtitles used the exact fade: 0.25 s at
  29.97 fps over 2 joins gave 6.039 s of video and 6.000 s of audio. Crossfade times are also
  passed to ffmpeg to the microsecond instead of the millisecond, and each xfade offset half a
  frame early (the fade starts on the same frame), so rounded timestamps of earlier joins no
  longer add a frame from the third clip on (#243).
- `extend` writes each re-exported clip's `.srt` for its new range and records it in the
  manifest (`srt`), in place and with `--out`. In place, the `.srt` the old manifest lists is
  replaced (removed when the source has no cached transcript), so no subtitles with the old
  timing stay beside the clip; an `.srt` the manifest does not list is never overwritten, as
  with `clips --force` (follow-up to #213).
- `clips --force` and `extend --out DIR --force` no longer replace a hand-added or renamed
  `.mp4` that has the name of a new clip but is not in the old `manifest.json` (without a
  readable manifest, any existing `.mp4` with a new clip's name). The run stops with exit 1
  before exporting anything, and an ERROR names the files and says to move or rename them
  (follow-up to #213).
- `final.srt` with cut transitions no longer drifts when a clip's audio runs past its video.
  Each clip is placed where the cut join really starts it: at the end of the previous file (its
  container duration), not of its video. Crossfades are unchanged (follow-up to #213).
- `cutpoint analyze .` or a drive root (`analyze D:\`) without `--out` no longer gets a
  project with an empty name, and a folder named with dots (`trip.2024`) is no longer cut at the
  last dot: the default project is named as for `edit` (a file's name without extension, the
  folder's full name, `project` for a drive root). A project made from such a folder by an
  earlier version keeps its old name (`trip`); pass `--out trip` to keep using it.
- `clips --top` and `--max-total`, and `360 director --top` and `360 stitch --top` /
  `--max-total`, reject 0 and negative values as usage errors (exit 2); `clips --top -1` used to
  select all but one moment.
- `analyze --vision-top` and `--max-candidates` reject 0 and negative values, and
  `--must-include-crash` values outside 0 to 1 (0 still turns it off), as usage errors (exit 2);
  `--vision-top -3` used to send all but the last 3 candidates to the vision model.
- Windows with the `events` and `gpu` extras: importing torch after whisper's GPU check (for
  example `360 analyze` with cached views but no cached event layers) no longer fails with
  `WinError 127` on `cudnn_cnn64_9.dll`. On Windows the GPU check imports torch first when it is
  installed, so torch loads its own cuDNN before the `gpu` extra's newer one (#215).
- `cutpoint ui`: when `<project>/ui/proxy` or `<project>/ui/frames` is a file instead of a folder,
  preparing a preview fails with a WARNING "<project>/ui/proxy is not a folder; remove or rename
  it" and a thumbnail request gives a 409 error with the same message for `ui/frames`, like
  `<project>/ui` itself (#184), instead of a logged traceback or a 500 error; nothing is left
  behind (#198).
- Re-running the analysis recorded in `highlights.json` (as `crashes --reanalyze` and `events` on
  a ski project do) reports an existing but empty `highlights.json` (`{}`), one without its style
  and sources, or one that holds a list, text or a number instead of an object, as "records no
  analysis; run 'cutpoint analyze' again" instead of "not found" or a `KeyError` or
  `AttributeError` traceback (#200).
- A failed or interrupted audio extraction (`analyze`, `assemble --captions`) now removes its
  temporary `.tmp.wav` file in the cache folder or next to the final video.
- On Ctrl+C or Ctrl+Break, Cutpoint now waits for ffmpeg to exit (at most 5 s, then stops it and
  waits again) before removing a partial `.part` export or `.tmp.wav` file, so on Windows the file
  is no longer left behind while ffmpeg still holds it open (#199). The exit code stays 130.
- `cutpoint 360 stitch` now stops MediaSDKTest like ffmpeg: on Ctrl+C or Ctrl+Break (or the UI's
  cancel) it gets at most 5 s to exit before it is killed; on `pano.stitch_timeout_s` it is killed
  at once. Either way Cutpoint then waits at most 5 s for it, instead of up to 30 s after the kill (#229).
- On Windows, `cutpoint 360 repair` retries writing its segments backup (`<name>.bak.json`) for a
  moment while another program (a virus scanner) briefly holds the file, like its other writes.
- `cutpoint ui` project metadata edge cases (#184): a preview proxy is not built or moved into
  place when `ui/proxy` became a link outside the project while queued or encoding (its temporary
  file is removed); one unreadable entry in the projects folder is skipped instead of failing the
  project list with a 500 error; saving the review when `<project>/ui` is a file or a broken link
  inside the project gives a 409 error "<project>/ui is not a folder; remove or rename it" instead
  of a 500 error (a clips export gives it if `ui` changes during the request), and nothing is
  created through the link (preparing a preview then fails with a WARNING naming it, without a
  traceback); a clips export refused because another job is running or the
  projects folder changed, or whose command could not be started, no longer leaves a
  `segments-*.json` file behind.
- On Windows, `clips --force`, `extend` in place and `extend --out ... --force` refuse a volume
  mounted into an NTFS folder (for example `C:\mnt\vol`) with the mount-point message instead of
  calling it a drive root. A real drive root (`C:\`, a UNC share root) keeps the drive-root
  message (#185).
- On Windows, a source read through `source_remap` from a disconnected mapped drive is now
  recorded in its network form (`\\server\share\...`), the same as `analyze` records it, so
  exclusions and carried segments match it (#181); older `Z:\...` entries still match. Clip
  manifests (`clips/manifest.json` `source`) now record `\\server\share\...` instead of
  `Z:\...` for a remapped mapped drive.
- On Windows, a `source_remap_exclude` glob written for a mapped drive (`Z:\...`) now also
  excludes the same file recorded in its network form (`\\server\share\...`), and the reverse, so
  an excluded source is no longer read from its remap copy (#193). Everything excluded before
  stays excluded; a drive whose mapping cannot be read is compared as before.
- A failed or interrupted `cutpoint 360 repair --out <new folder>` removes the folders it created
  for `--out`, only while they are empty; existing folders and files are kept (#189).
- A failed or interrupted (Ctrl+C) `analyze`, `stage`, refmatch check-frames run, `cutpoint ui`
  proxy build or `cutpoint ui` new-project request now removes the output folders that run
  created, only while they are empty; pre-existing folders are never removed. The same holds for
  `cutpoint 360 qa`, the 360 paths QA, `cutpoint 360 stitch` (before the first SDK run; a Ctrl+C
  during the SDK keeps the stitched folder because it holds the clip's log) and `events` region
  discovery (on a first run no `discovery/<game>/` folder or empty `ocr.jsonl` is left)
  (#194).
- A failed `--out` such as `a/new/../x` left the new empty folder `a/new` behind (`assemble`,
  `clips`, `extend` and the pano commands, through `output_dir`); it is now removed, and the
  existing `a` is kept (#194, #189).
- On Windows, a staged copy is now used when its source is read through a `subst` drive letter
  and was staged under the folder that drive maps to, or the reverse (for example a
  `source_remap` prefix spelled `Q:\Footage` with `Q:` a `subst` of `D:\Footage`). Before, the
  run read the slow source again. Index keys and copy names are unchanged, so nothing has to be
  staged again (#186).
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
