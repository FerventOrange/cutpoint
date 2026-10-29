# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `install.sh`: one-step Linux installer (`curl -LsSf .../install.sh | sh`). It installs uv and
  ffmpeg when missing, picks CUDA torch (`cutpoint[all]`, `--torch-backend cu126`) with an NVIDIA
  GPU or CPU torch (`cutpoint[events,ui,keyring,ski]`) without, and `--lite` installs
  `cutpoint[ui,keyring,ski]` (no torch). Flags:
  `--yes --dry-run --cpu --cuda --source pypi|git[@ref]|PATH --no-ffmpeg`; full mode can ask to
  run `cutpoint setup` and pull the Ollama model. Running it again upgrades or switches mode. CI
  lints it and checks the planned commands.
- `install.ps1`: the same one-step installer for Windows (`irm .../install.ps1 | iex`; Windows
  PowerShell 5.1 and PowerShell 7). It installs uv and, with winget, ffmpeg when missing, then
  Cutpoint with the same extras and torch choice as `install.sh` (CUDA torch with an NVIDIA GPU,
  CPU torch without, `-Lite` for no torch). Parameters:
  `-Yes -DryRun -Cpu -Cuda -Source pypi|git[@ref]|PATH -NoFfmpeg`, each also a `CUTPOINT_*`
  environment variable for `irm | iex`. CI lints it, checks the planned commands under both
  PowerShells and runs a real lite install on Windows.
- `cutpoint ui`: "Export clips" of a 360 project first stitches the recordings it needs with the
  Insta360 Media SDK when the SDK runs (`360 stitch`, skipping recordings already stitched), then
  renders from the stitched equirects. The job log shows the disk estimate (about 54 GB per hour
  of footage); too little free space, a missing SDK, no NVIDIA GPU or a failed stitch exports from the preview
  with a WARNING, and Export says "Exported from the 360 preview (not stitched)". The new
  `pano.auto_stitch` key (default `true`) turns it off; the command-line `clips` is unchanged.
- Docs: how to install Ollama and the models for `--vision` and AI Chat.
- `cutpoint ui` Settings > AI assistant: choose a Provider preset instead of typing a base URL:
  None (local Ollama, with a model name), Claude (Anthropic), Codex (OpenAI) or Grok (xAI, the new
  `assistant.provider` `xai` at `https://api.x.ai/v1`). Grok's key comes from the new
  `XAI_API_KEY`, used only for `api.x.ai`, kept from jobs and redacted from logs like the others.
  The Model field lists the provider's models (Ollama's pulled ones; Anthropic, OpenAI and xAI
  only with a key and after consent; non-chat and `*-codex` models left out), cached 10 minutes, with
  Refresh list; any name can still be typed. A `url` in `config.json` that no preset has shows as
  "Custom (from config.json)" and keeps working; its model can still be changed.
- `cutpoint ui` AI Chat: bring your own AI. `assistant.provider` can be `openai` (OpenAI, or
  LM Studio, vLLM and llama-server at `assistant.url`) or `anthropic` (default model
  `claude-sonnet-5-5`, `claude-opus-5-5` selectable), chosen in Settings with the model and base
  URL and checked when saved. The API key comes from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or
  `CUTPOINT_ASSISTANT_API_KEY`, or the system keyring with the new optional `keyring` extra
  (Settings shows only "set (env)", "set (keyring)" or "not set"; `all` includes it). A key
  belongs to the base URL's host: `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` go only to
  `api.openai.com` and `api.anthropic.com`, and a key is sent only over `https://` unless the
  server is on this machine. A key is never written to
  `config.json`, project files, logs or the page, and jobs never get it. Before its first
  message a hosted provider (any not on this machine, a remote Ollama too) needs a one-time
  consent, kept in `config.json`; it gets moment labels, transcript snippets, file names (never
  paths) and times. The header names the provider, model and host (for example `OpenAI-compatible · server's
  default model · openrouter.ai`); a remote Ollama counts as hosted, not local (#346).
- `cutpoint ui`: the backend of the AI Chat tab below (Ollama). It is set by a new
  `assistant` section in `config.json` (`provider`, `model`, `url`, `context_tokens`, `consent`;
  no API key), and the UI's API (token header only) keeps its chat and history in
  `ui/assistant/chat.json` and `history.json` in the project (#346).
- `cutpoint ui` AI Chat tab: ask a local model to find, add, remove, reorder or trim moments.
  Each change is previewed in the Sequence tab (marked rows, moments and trim edges) before
  Apply; Undo and Redo (Ctrl+Z, Ctrl+Y) cover manual and AI edits, and an applied change can be
  reverted from its history (#346).
- `cutpoint ui`: Settings > Theme switches between Blue (the default, unchanged), Light and
  Dark. The choice is saved in the browser and applied before the page first paints; Light and
  Dark keep text, controls and focus outlines at WCAG AA contrast, and Blue keeps today's colours
  (#445). The Dark theme's panels are lighter, and Light uses the same yellow for buttons and
  fills as the other themes (#466).
- `assemble --audio-transition cut|crossfade` sets the audio join apart from `--transition`; by
  default it follows it. `--transition cut --audio-transition crossfade` is refused (exit 2) (#466).
- `clips` and `edit --color auto|underwater` and `--color-strength 0-1`: automatic color
  correction per clip (levels, white balance, vibrance; `underwater` also restores red, with
  per-channel levels), measured on each clip and vetoed when a simulation predicts clipping, a
  worse cast, or too much or too little color.
  Clips that look good, log footage (use `--lut`) and 360 clips are left as they are. Not with
  `--lut`. The `color` key in `config.json` sets a default (`off`), never applied to `gaming`
  footage. `manifest.json` records each clip's `color` entry and `extend` re-applies it. With
  `--color` off (the default) the clips are unchanged (#290).
- `--json`: a `grade_dropped` warning when `clips --force` replaces graded clips with ungraded
  ones (#290).
- `--json`: a `color_on_gaming` warning when `--color` is given for `gaming` footage (#494).
- `cutpoint ui` Export tab: a Color slider (Off, Auto, Underwater) under Compress Audio Peaks,
  with a Strength slider (0-100%, shown with Auto or Underwater), passed to Export clips and
  One-Click Edit as `--color` and `--color-strength` (left out at 100%). Color is remembered per
  project (Strength for all); a project with no choice starts on the `color`
  key in `config.json` (`GET /api/settings` returns `color`), else Off, and at Off for gaming
  projects (the project view's new `gaming`); a grade picked there is still sent. The jobs API
  refuses an unknown `color`, a `color_strength` outside 0-1 or without a grade, and `color`
  with `lut` (#461).
- `cutpoint ui`: the Moments list has a sort order, in the Sort and filter menu below: Score (the default) or
  Position in video, which sorts by source name (natural order, so `clip2` comes before `clip10`)
  and then start time. The rank badge on each card keeps its score rank in either order. The
  choice is remembered per browser (#469).
- `cutpoint ui`: each moment card shows its candidate id from `highlights.json` (`h001`), the id
  that Candidate ids in the Export tab and `clips --ids` take, and the moments filter matches it
  (#487).
- `cutpoint ui`: the Moments list's search field has a funnel button beside it that opens Sort and
  filter: the sort order and quick filters (In sequence, Not in sequence or All; Ready only; a
  source file when the project has several). They combine with the search text, a dot
  on the button shows while a filter or a sort other than Score is on, and with nothing left the
  list reads "No moments match these filters." The sort and filters are remembered per browser,
  the source file only while the project is open.

- `cutpoint ui` Export tab: "Clips to Export" exports the sequence (default) or automatic picks
  (Top moments, Max total, Candidate ids, Order), and "Cut Timing" sets the speech, tail, event
  tail and extend limits; a blank field keeps the CLI default. "Extend export" runs `extend` on
  the latest export into a new `clips_ui_<stamp>` folder.
- `cutpoint ui` One-Click Edit: Highlights, Montage, Clips and Funny moments presets run
  `cutpoint edit --kind` in the project's style and make the result the current export. A second
  run asks before replacing earlier clips. The presets are off, with the reason shown, for 360
  projects and for projects analyzed with options `edit` cannot repeat.
- `cutpoint ui`: the trim rail shows a filmstrip of stills (keyframes; exact ones when zoomed in)
  behind the clip markers, and the project's still cache keeps up to 1500 stills (#261).
- Editions: Cutpoint is full (every feature) or lite (no torch: no `events` or `gpu` extra).
  `cutpoint capabilities` prints the edition first, and its `--json` has the new key `edition`
  (`"full"` or `"lite"`; the schema stays 1) (#502).

### Changed

- A feature that needs a missing extra now says it is "not available in Cutpoint lite" and how
  to add it: for a tool install, the installer again without `--lite` / `-Lite` and with
  `CUTPOINT_LITE` unset (`curl -LsSf
  https://raw.githubusercontent.com/FerventOrange/cutpoint/main/install.sh | sh` or `irm
  https://raw.githubusercontent.com/FerventOrange/cutpoint/main/install.ps1 | iex`); in a clone,
  `uv sync` without `--no-group full`. A tool installed from git or a local path gets both
  commands with its source (`... | sh -s -- --source git@main`, `$env:CUTPOINT_SOURCE =
  'git@main'; irm ... | iex`); one from a fork or a wheel or sdist URL gets `uv tool install
  "cutpoint[...] @ <url>" --torch-backend cpu|cu126 --python 3.12 --reinstall --no-sources`,
  since the installers only install from PyPI, github.com/FerventOrange/cutpoint or a local
  path. The CUDA torch fix in `capabilities` says the
  same. The hints no longer print a `uv tool install` command without `--torch-backend`, which
  can pull CUDA torch on a machine without an NVIDIA GPU. Help, messages and
  docs no longer call `ui`, `keyring` and `ski` optional extras (#502).

- `--color underwater` grades only clips that look underwater (mean red below 0.77 of green
  and 0.85 of blue, green above 0.79 of blue); the others are skipped as `not_underwater`, and
  plain `auto`'s INFO hint uses the same rule. Water bluer than that, such as open ocean, is
  skipped too. A washed-out underwater grade is now vetoed by its saturation (below 0.63 of
  its value: `veto_faded`) instead of its colorfulness, and red-only highlight clipping, up
  to 3 % of the frame, no longer counts toward its clipping veto. `--color underwater` on
  `gaming` footage logs a stronger WARNING (it still grades). Cached measurements are redone
  (color version 3); `extend` keeps re-applying older records as they were (#494).

- Packaging: the `ui`, `keyring` and `ski` packages (fastapi, uvicorn, keyring, telemetry-parser)
  are core dependencies, so a plain `uv tool install cutpoint` or `pip install cutpoint` runs
  `cutpoint ui` and the 360 and telemetry commands. The extras `ui`, `keyring` and `ski` stay as
  empty aliases; `all` is now `events` and `gpu`. `gpu` and `events` are unchanged (torch is still
  opt-in for installs from PyPI). In a clone, `uv sync` installs everything through the new default
  dependency group `full`; `uv sync --no-group full` gives the lite environment. CI and the
  Dockerfile skip the group. `uv run` also syncs `full`, so an existing clone downloads torch and
  CUDA (about 7 GB or more) on its next `uv run`; use `--no-group full` or `UV_NO_GROUP=full` to
  stay lite.

- `cutpoint ui` Export tab: "The sequence (hand-picked)" reads "Sequence" and "Max total (s)"
  reads "Max video length" (still `--max-total`, the cap on all clips together). Top
  moments, Candidate ids, the Cut Timing fields other than Event tails, the "Latest export ..." facts
  (not its warnings) and Extend export are hidden unless the new Settings > "Show advanced export options" (off by
  default, kept in the browser) is ticked; hidden fields keep their values and are still sent,
  and "Advanced options in use: ..." names any that are. The Compress Audio Peaks, Color and
  Strength sliders are centred with equal margins, and each tick label (Light, Auto) sits
  centred under its stop.
- `cutpoint ui`: tighter moment cards: Add/Remove sits top right beside the rank and title, then
  the candidate id and the times ("h001 0:00.00 to 0:07.70 · 7.7 s"), the file name on its own line only when the project
  has several sources, the strip and the transcript, with the preview readiness at the bottom
  right.
- `cutpoint ui`: Smart edges is split into Speech-aware cuts and Smart tails (both on by
  default; a saved Smart edges choice carries over to both). "Analyze again" keeps a recorded
  `--recover`.
- Faster `assemble --transition fade` (the highlights and montage kinds): only the video around
  each fade is re-encoded and the rest of every clip is stream-copied. The video has the same
  number of frames at the same times, and the audio, `final.srt` and timelines are the same; the
  copied video skips one encode generation.
  Clip exports now carry one forced keyframe 0.5 s from each end (none with AMF), which covers
  fades up to 0.5 s (the default is 0.25 s) and makes them about 5% larger; longer fades use
  the clips' own keyframes.
  `CUTPOINT_FADE_COPY=0` re-encodes the whole video as before (#463).
- `cutpoint ui`: the sequence column has Sequence and Export tabs. Export holds one "Export &
  Assembly Options" panel (vertical, captions, Smart edges, loudness, linked transitions, video
  and audio transition, Compress Audio Peaks; remembered per browser) and one "Show Exports
  Folder" button. The padding field is removed; `--extend` and `extend` remain. New defaults:
  Smart edges (`--speech-aware --smart-tail`) and Normalize loudness are on, and Compress Audio
  Peaks is Light unless `compress_peaks` is set in config.json; a saved browser choice wins (#466).
- Faster exports with a GPU encoder: `clips` and `edit` export 2 clips at a time with NVENC,
  VAAPI or AMF, and still 1 with libx264. The new `export_workers` setting (`auto`, or 1 to 3;
  `CUTPOINT_EXPORT_WORKERS` overrides it) changes this. The clips, their names and manifest.json
  are the same as before, and a failure or Ctrl+C still stops every export and removes its
  partial file (#464).
- `cutpoint ui`: each moment card shows where the moment lies in its source (the moment in gold
  on a strip of the whole source, read-only) instead of the score bar; the score is in the strip's
  hover title and in Stats for Nerds (#467).
- Faster runs: a source is probed with ffprobe once per run instead of 3 to 5 times (about 0.1 s
  each). The result is reused until the file, or its staged copy, changes; results are unchanged
  (#372).
- HUD region discovery is about 22 % faster on a first run (1424 s to 1115 s over 5 test clips,
  CPU OCR): it recognises the text boxes of a frame one at a time instead of in padded batches.
  Extra notice regions at the screen edges may come out slightly different, and so may their
  notice text; on the 5 test clips, kills, kill names, menu frames and highlight candidates were
  unchanged. Frames already in a discovery cache keep their earlier reading; to get the new
  regions, delete `projects/<name>/discovery/<game>/ocr.jsonl` (or use a fresh project) and run
  discovery again. The OCR layer reads as before, so raw OCR caches stay valid (#443).
- `clips --force` and `edit --force` log a WARNING when they replace a clips folder whose
  manifest recorded a grade (`lut`/`color`) and the new run has none. The output is unchanged.

### Fixed

- `cutpoint ui`: Extend export is disabled, with the reason in its tooltip, while Extend start
  and Extend end are blank or 0 and Speech-aware cuts is off. Before, it ran
  `extend --no-speech-aware` with nothing to do, which failed.
- `cutpoint ui`: the Play sequence button sits under Undo and Redo, as wide as the pair, and
  keeps its size and place when it turns into Stop. It used to shrink and jump up beside Redo.
- `cutpoint ui`: the Sequence, AI Chat and Export tabs share the sequence column's full width
  evenly at any width the column is dragged to; a label too long for a narrow tab ends in an
  ellipsis. Before, they kept their labels' widths at the left of the column.
- `cutpoint ui`: a 360 project with no moments yet reads "360 footage: 5 recordings found"
  instead of "5 360 clips found", which looked like 5,360, and "No 360 recordings found" when
  there are none.
- `cutpoint ui` Export tab: after an export with Color on, the export state says how many clips
  the grade changed and why it left the others as is, from `color.skip` in `manifest.json` (for
  example "Color: no clips graded (too much highlight clipping)."). An export whose clips the
  safety checks all skipped looked as if Color had done nothing, with no reason given. The note
  shows with the advanced export options hidden too.
- `cutpoint ui`: the final video in the Export tab no longer plays on under Play sequence. It
  kept playing while its tab was hidden, so its audio ran over every sequence clip. Starting
  the preview or leaving the Export tab now pauses it, and playing it stops Play sequence. Play sequence also no longer
  cuts a clip short, or stalls, when it moves to a clip whose start is inside the preview part
  already playing but whose end is not: it loads a part that covers the whole clip. A clip, or a
  moment, of another source whose times fall inside the preview part already playing no longer
  plays that source's picture and sound in its place.
- `assemble`: the audio and container of `final.mp4` now end with the video, to within one audio
  frame. Cut joins ended about 24 ms late and started the video about 21 ms late, and
  `--loudnorm` or the limiter padded the audio by about 0.1 s. The `.fcpxml` and `.otio` of cut
  joins now end where `final.mp4` ends. Plain cut joins now re-encode the audio once (192k AAC)
  instead of stream-copying it; the video is still stream-copied (#473).

- `cutpoint ui`: stills of BT.709 and HDR sources now show the same colours as their exports.
  A JPEG is always decoded as BT.601 full range, so these stills were about 7 codes off (up to
  32 on saturated colours). They are now converted to BT.601 before encoding, which rebuilds them
  once under a new name. Previews of full-range BT.601 sources are encoded in TV range like
  exports, keeping their BT.601 tags. BT.601 stills and every other preview are unchanged (#441).
- HLG (HDR) exports, `cutpoint ui` previews and stills are brighter: they came out about 16 %
  darker than a reference tone map (libplacebo bt.2390), and now are within about 3 % on the
  measured phone clip. HLG is tone-mapped with `mobius` at 203-nit diffuse white instead of
  `hable`; previews and stills of HLG sources are rebuilt once. PQ and every other source are
  unchanged (#449).
- 360 renders no longer tilt the horizon by 20 to 30 deg when the camera turns over in a clip
  (a fall, or the camera being handled). The view follows 30 % of the camera's lean, and that
  lean had no limit; it is now capped at `pano.max_follow_deg` (default 30 deg), so at most
  9 deg shows. On the measured Insta360 footage no 10 s window is over the 12 deg QA cap (55
  were), and riding footage is unchanged (#442).
- HUD region discovery finds icon kill feeds (`name [icon] name`, such as Rainbow Six Siege) at a
  screen edge as `feed` regions, and the generic parser reads their lines as kills, using the name
  boxes' colours and the gap between them. It reads icon lines (exactly two names) only with a
  preset whose `feed` has `row_notices` (a discovered icon feed), and drops icon kills read on menu
  frames; an icon kill first read as a still screen opens is not taken as feed activity. Other rows of such a feed ("X has found the bombs")
  become notices, and a kill takes both names from its most complete read that repeats, or from the most complete read when none repeats (#433). A game that already has a preset keeps it: move the old preset away and run `analyze` or `events` again, or run `events --discover-regions` and copy the new `proposal.json`'s `feed` region and `"feed": {"row_notices": true}` into the preset by hand; the new region needs new OCR reads.
- `clips --force` and `extend` on macOS (and on case-insensitive drives elsewhere, such as exFAT
  or FAT drives on Linux, NTFS only when mounted case-insensitive) no longer fail when the old
  manifest.json names a clip in another case than the folder (`001_A.mp4` listed, `001_a.mp4` on
  disk): the swap moved the same file twice. Names now match case-insensitively where the folder
  is case-insensitive, checked once per folder without writing to it; Windows is unchanged and
  case-sensitive folders keep `001_A` and `001_a` apart. A listed `.srt` renamed in case only is
  now replaced with its clip there too, as on Windows (#247).
- Highlight candidates in game menus and lobbies are left out. `analyze`, and `events` when it
  re-ranks `highlights.json`, drop a candidate with half or more of its time in the menu frames
  the HUD OCR found, and trim one that starts or ends in a menu to its gameplay part when at least
  the style's minimum clip length is left. The menu frames are recorded as
  `stats.ocr.<source>.menu_spans` in `events.json`; HUD OCR results are re-interpreted once from
  the raw OCR cache. Footage without menu frames gets the same candidates as before (#440).
- HUD discovery on a short source proposes the same regions when it runs again from its OCR
  cache. Its bursts overlap on a source under about 3 min, and two frames at the same time
  (to 0.1 s) shared one id, so the cached run read them back as frames of one burst. Such a
  frame now has its burst and frame number in its id; an existing `ocr.jsonl` is kept, and only
  the frames that shared an id are OCR'd again, once. On such a source the first run can change
  too, because frames at the same time now count as separate frames. Sources where no frames
  share a time are unchanged (#444).
- Parallel exports with NVENC (`video_encoder` `auto` or `nvenc`) no longer switch to libx264,
  or fail, when the driver has no free encode session (older GeForce drivers while ShadowPlay or
  OBS holds one): the rest of the run encodes one clip at a time on the GPU (one INFO line) and
  the clip is encoded on the GPU again; only a clip that still finds no session alone is redone
  on the CPU, as before. A Ctrl+C or a failed export now also stops the clips' ffprobe runs and
  audio track checks at once instead of waiting for them, and ffprobe calls time out after 300 s
  instead of never, except the one that lists every frame of a clip for `assemble` (#470).

## [0.3.0] - 2026-10-27

### Highlights

- `cutpoint edit <video...> --kind highlights|montage|clips|funny-moments` goes from footage to a
  finished video with `.srt` subtitles in one command.
- Agent support: a `--json` result envelope on stdout for the main commands (`analyze`, `clips`, `assemble`,
  `edit` and more; see Added), and `cutpoint skill install` / `skill print` for Claude Code.
- Color: HDR (HLG, PQ) and BT.2020 footage is tone mapped to SDR Rec.709 instead of coming out
  washed out, all exported clips are tagged BT.709, and `--lut` applies a 3D LUT you supply.
- Audio: `--compress-peaks` evens out loud peaks, and a recording that writes the same audio to
  several tracks (OBS) is no longer summed into a clipped, too loud clip.
- Hardware: Linux AMD GPUs decode H.264 with VAAPI, decoders follow the GPUs detected, and
  `CUTPOINT_VIDEO_DECODER=cpu` is the escape hatch.
- Analysis: `analyze` and `edit` now run the reactions, keyframes and crashes steps themselves;
  game HUD reading ignores menu and lobby screens, learns more event text and reads faster; the
  Ollama vision rating is opt-in.
- Safer exports and cancelling: exports are all or nothing, Ctrl+C and Ctrl+Break remove partial
  files, and a lock stops two runs on one clips folder. Security: `cutpoint ui` and the Docker
  image are hardened.
- `cutpoint ui` is a fuller workspace: trim handles, whole-source scrubbing, fast moment previews,
  Play sequence, Insta360 360 projects and Settings.

### Added

- `cutpoint edit <video...> --kind highlights|montage|clips|funny-moments` runs `analyze`,
  `clips` and `assemble` in one command and copies the result to `<video>-final.mp4` and
  `<video>-final.srt` in the project folder. Each kind sets the style, selection, clip order,
  transition, loudness normalisation and captions; `--style`, `--top`, `--max-total`,
  `--vertical`, `--force` and `--events` override it, and `--vision-top`, `--ollama-url` and
  `--ollama-model` work as on `analyze` (#2).
- SRT subtitles: `clips` writes `001_<label>.srt` next to each clip and `assemble` writes
  `final.srt` (`final_vertical.srt`, ...) next to the final video, from the cached transcripts
  of `analyze`, timed to the clip and to the joined timeline. No transcription runs for them;
  without a cached transcript there is no `.srt` and one INFO line. `clips --force` replaces only
  the `.srt` files its old `manifest.json` lists (#5).
- `--json` result envelope: the command writes exactly one JSON object to stdout, `{"ok",
  "command", "schema": 1, "project", "outputs", "warnings", "error"}`, with `error.code` one of
  `bad_args`, `not_found`, `ffmpeg_missing`, `missing_extra`, `output_not_empty`, `interrupted` or
  `error`. Logs and child process output go to stderr; exit codes are unchanged. It is available
  on `paths`, `analyze`, `clips`, `assemble`, `edit`, `extend`, `events`, `crashes`, `profile`,
  `capabilities`, `mix`, `stage`, `setup` and every `360` subcommand. `outputs` lists the files
  written (absolute paths) and their key facts, and `warnings` carry codes such as `kept_clips`,
  `kept_sources`, `step_failed`, `vision_unavailable`, `footage_unreachable`, `no_telemetry`,
  `no_visual_layer`, `highlights_not_updated` and `context_not_applied`. Documented in the "JSON
  output" section of `docs/commands.md` (#4).
- `cutpoint skill install [--dir DIR] [--force]` installs a Claude Code skill
  (`~/.claude/skills/cutpoint/SKILL.md` by default) so an agent finds Cutpoint for requests such
  as "edit this video" and runs the edit loop; `cutpoint skill print` prints it to stdout (#1).
- `--lut FILE` (a `.cube` or `.3dl` 3D LUT) and `--lut-strength 0-1` (default 1) on `clips` and
  `edit` grade every clip, `360` renders included, after the BT.709 conversion. Use the LUT made
  for the footage's log or flat profile; Cutpoint ships none. `clips/manifest.json` records `lut`
  and `lut_strength` per clip, and `extend` grades its re-exports the same way. Without `--lut`
  the output is unchanged (#290).
- `--compress-peaks off|light|strong` on `clips`, `assemble` and `edit` (and the `compress_peaks`
  key in `config.json`; default `off`) pulls sudden loud peaks such as shouts, gunshots and mic
  bumps down toward the speech level, evens out loudness between clips, and keeps the set at its
  overall loudness with a makeup gain and a true-peak limiter. `clips` bakes it into every clip
  and records it in `manifest.json` (`compress_peaks`, `compress_peaks_makeup_db`), so `assemble`
  does not compress those clips again. On an R6 montage with commentary and wind, the loudest
  moment went from 12.2 dB above speech to 7.5 dB (light) or 5.2 dB (strong). The UI has a
  matching "Compress peaks" select (#326).
- `analyze` and `edit` run the reactions (PANNs sound events, and the voice layer when crashes
  run), keyframes (HUD OCR with discovery for `gaming` and `funny`, the CLIP visual layer for
  `action`, `ski` and `sport` with CUDA) and crashes (action-cam) steps, one after another, then
  rank. Without the `events` extra or the model files a step is skipped with an INFO line;
  nothing is downloaded. `--skip reactions|keyframes|crashes` (and `analysis.skip` in
  `config.json`) leaves steps out, `CUTPOINT_PROGRESS=1` prints `PROGRESS {...}` step lines on
  stderr, and a failed step gives a `step_failed` warning and the run continues without it.
- Linux AMD GPUs decode limited-range 8-bit 4:2:0 H.264 with VAAPI for analysis, HUD OCR, 360
  view sampling and 360 renders. HEVC, 10-bit, 4:4:4 and full-range sources still decode on the
  CPU, and a failed GPU decode falls back to the CPU (#22). `CUTPOINT_VIDEO_DECODER=cpu` decodes
  everything on the CPU, an escape hatch for a GPU decoder that misbehaves on real footage.
- `cutpoint capabilities` reports torch (installed, version, CUDA or CPU-only build, whether it
  reaches the GPU) with the exact reinstall command when a CPU-only torch meets an NVIDIA driver,
  and the Insta360 Media SDK (found or not; `360 stitch` warns that it needs an NVIDIA GPU). New
  JSON keys: `torch` and `insta360_sdk` (#22).
- `cutpoint ui` trim and preview: the preview plays and scrubs the whole source, and the clip
  start and end are handles on a trim rail that you drag, or focus and move with the arrow keys
  (0.1 s, Shift 1 s), PageUp/PageDown and Home/End, with zoom, an overview strip and screen
  reader sliders. "Use playhead" and `[` / `]` set the start or end anywhere in the source
  (#324, #352).
- `cutpoint ui` previews sources the browser cannot play within seconds: the server first
  encodes a short proxy of the selected moment while the whole-source proxy builds, and HDR
  proxies are converted to SDR. "Play sequence" previews the cut back to back, starting a clip as
  soon as a short proxy covers it. Rows show whether a preview is ready, encoding or queued, and
  the trim rail hatches the parts that cannot play yet. `POST
  /api/projects/{id}/sources/{sid}/prepare` takes an optional `{lo, hi}` body (#323, #325).
- `cutpoint ui` makes projects from Insta360 X3 footage (`.insv` and `.lrv` files are recognised,
  and a folder mixing 360 and regular videos is refused). "Analyze footage" runs `cutpoint 360
  analyze --scan`, which scans and analyzes in one command. While an analysis runs, the Moments
  column lists each step with its state, and `GET /api/jobs` returns them as `progress` (#320,
  #341).
- `cutpoint ui` workspace additions: a Settings dialog for the projects folder (`projects_dir`
  in `config.json`, `GET`/`PUT /api/settings`) and the vision default; a "Stats for Nerds" overlay
  (off by default); a mute button and volume slider (`M` toggles mute); native file and folder
  dialogs for New Project (`POST /api/pick`); "Show folder" buttons for sources, clip exports
  and finals (`POST /api/projects/{pid}/reveal`); sequence thumbnails with inline rename,
  multi-select, remove and drag reorder; and a resizable side column (#142, #353).
- The UI transcript marks words Whisper was unsure of (confidence below 50%). Transcripts now store
  each word's `probability`, and `highlights.json` candidates list those words in
  `transcript_unsure`. A source transcribed before this keeps its cached transcript until it is
  re-transcribed (delete its `cache/<source>/transcript-*.json`, or analyze with another `--model`)
  (#327).
- HUD presets can define event types with an `events` list (`name`, `when` matchers, `score`,
  `max`, `payoff`, `tail_s`, `seed`, plus `reaction` for 0-12 points once per candidate and
  `styles`) and the vision prompt's `title`, `view` and `hype`. `player_deaths` now defaults to
  `reaction` 8 for `funny`. Preset types are checked before the built-in ones and can replace
  them (#8).
- HUD region discovery proposes up to 2 extra `notice` regions (`extra_notice`, `extra_notice2`)
  where text keeps showing outside every discovered region. A new preset gets them, marked
  `"origin": "outside"`; for a game that already has a preset, `--discover-regions` writes them only
  to `suggested_regions` in `proposal.json`. `events --json` reports their number as `suggested`
  (#314).
- The OCR layer keeps a raw read of each source in `cache/<source>/ocr-raw-*`. When an update
  changes only how HUD text is interpreted, the next run replays it in seconds, with no OCR, and an
  interrupted read resumes near where it stopped. The first run after this update still reads the
  footage once. The raw read holds OCR text, so it is git-ignored and never written inside a git
  checkout (#315).
- `final.fcpxml` and `final.otio` hold a cross-dissolve at each join when `assemble` uses
  `--transition fade`, matching `final.mp4`; on the OTIO audio track a clip without audio now
  leaves a gap, so later audio stays in sync (#246).
- `360 qa` counts the bad frames that `360 director --vl-check` stored in the path JSON (a clip
  fails with more than 1 inside its span); `qa.json` gets a per-clip `vl` field and the totals
  `vl_checked` and `vl_bad` (#16).

### Changed

- **BREAKING:** `clips --force` and `extend` refuse a `manifest.json` Cutpoint did not write
  (another `generator`, or no `clips` list of entries with a `file`) and one they cannot read,
  stopping with an `ERROR` before exporting; move or rename it. An unreadable manifest used to
  be replaced (#145).
- Exported clips are now always tagged BT.709. BT.601 SDR sources (`smpte170m`, `bt470bg`,
  `smpte240m`, and SD without a matrix tag) are converted to BT.709 with ffmpeg's `colorspace`
  filter; HD without a matrix tag is tagged with no pixel change. This fixes hue shifts when
  `assemble --transition cut` joins clips from mixed sources. `assemble` re-encodes and
  `--captions` output are tagged the same way, and so are clips graded with `--lut` on older ffmpeg
  (#289, #290).
- The Ollama vision rating is opt-in for every style: `analyze` and `edit` use it only with
  `--vision`, or when `"vision": true` in `config.json` (the UI Settings checkbox) turns it on by
  default, which `--no-vision` then skips. Before, it ran whenever Ollama was reachable (except for
  `ski`), loading about 7 GB into GPU memory unasked. The rating now runs after the events
  ranking, and the Ollama model is unloaded after the pass, also when it fails or is interrupted.
  The style setting `vision_default` is gone. `highlights.json` records `settings.vision_requested`,
  `vision_top` and the Ollama URL and model, so re-analysis (`cutpoint events`, the UI Analyze
  action) keeps them; a project analyzed before this change is re-analyzed without vision.
- Vision ratings are cached per source in `cache/<source>/vision-ratings.json` (keyed by the model
  and its digest, prompts, options, frame size and frame times) and saved as they arrive. A rerun
  asks Ollama only about candidates it has not rated, and with Ollama down `analyze --vision` uses
  the cached ratings; the `vision_unavailable` warning gives the counts. Ratings cached before
  this have no digest and are rated again once (#377, #413).
- `analyze` writes `events.json` and `events.txt` (and `crashes.json` for action footage) and
  ranks with them. `highlights.json`, `transcript.txt` and these files are written together at
  the end, so a cancelled run leaves the previous files as they were. The analysis score stays in
  each candidate's `score_pre_events`; `highlights.pre-events.json` is written only when the
  analysis was not ranked with events yet (#408).
- Decoding follows the GPUs detected: NVDEC is probed only with an NVIDIA GPU and VAAPI only on AMD
  render nodes, so an AMD machine no longer tries CUDA first, and the chosen decoder is logged once
  (`Video decoder: ...`). `events` no longer retries a failed CPU decode as a GPU one (#22, #429).
- `cutpoint ui` preview proxies use the GPU encoder exports use (NVENC or VAAPI, per
  `video_encoder`), have a keyframe every 30 frames and at most 30 fps so seeking is quick, and the
  UI's jobs and proxy encodes run at below-normal priority so the desktop and browser stay
  responsive (#22, #430).
- `analyze` on `gaming` and `funny` footage starts the HUD OCR in the background while
  transcription, scene detection and the reactions step use the GPU. It needs Whisper on CUDA, a
  HUD preset for every game and enough free VRAM; the results are unchanged.
  `CUTPOINT_ANALYZE_OVERLAP=0` turns it off. Experimental and off by default:
  `CUTPOINT_ANALYZE_SHARED_DECODE=split` shares one CUDA decode between scene detection and the
  OCR, spooling frames to the source's cache folder (at most 8 spool files, about 2 GB) (#375,
  #402).
- Less CPU time in analysis: the HUD OCR runs on at most 4 threads and, on a machine with at least
  4 CPUs, splits them over 2 OCR engines (a 300 s game clip took 286 s instead of 345 s). The voice
  layer reuses the PANNs model, and `cutpoint 360 analyze` loads CLIP once and recomputes only the
  selfie view in its second pass. Results are unchanged; `vram_mib.<layer>.baseline_mib` in
  `events.json` is now the monitor's first reading during the layer (#372, #374, #376, #403).
- A self-kill (the player is both killer and victim in a kill feed) no longer counts as a player
  kill: a new default event type `self_kills` adds 3 points each in every style and no longer counts
  toward DOUBLE/TRIPLE KILL streaks. The player's own death in a kill feed no longer adds the kill
  boost either (new default event type `player_deaths`). Both need `player_tags` or `--player`; run
  `cutpoint events` to rebuild streaks.
- Re-running `analyze` keeps the `--game` of the last `cutpoint events` run, and its `--player`
  tags when `player_tags` in `config.json` is empty.
- The HUD presets folder in `CUTPOINT_HOME` (`cutpoint paths`) is now trusted like the bundled
  folder: `"builtin": true` is honoured there, and only project `hud_presets` folders ignore it
  (with a WARNING). After adding or editing `title`, `view` or `hype` in a user preset, run
  `cutpoint events <project> --force vlm`.
- `--json` of `profile`, `capabilities`, `360 repair` and `360 qa` now writes the result envelope. For
  `profile`, `capabilities` and `360 repair` the report each printed before is now under `outputs`
  with the same keys (read `outputs.torch` instead of `torch`), and `360 qa --json` is now the
  envelope instead of abbreviating `--json-only` (#4).
- `360 qa` measures the horizon tilt of each clip from the v360 commands its render emitted, so it
  catches render roll errors the old model prediction could not. Each `.cmd` file now starts with a
  `# cutpoint-v360 ...` line; clips rendered before this fall back to the model until re-rendered
  (#12).
- `cutpoint 360 director` paths are evaluated with a monotone cubic (PCHIP) between keys instead of
  linear interpolation, so speed no longer jumps at keys and a value never overshoots. Existing
  `360 director --preview` videos keep the old motion until deleted (#14).
- The `all` extra now includes `ui`, so `cutpoint[all]` and `uv sync --extra all` also install the
  browser workspace. The `dev` extra is replaced by a `dev` dependency group that `uv sync` and
  `uv run` install by default; `--extra dev` is no longer accepted (#122).
- `cutpoint ui` layout: the preview is re-laid out to fit the window without scrolling (name and
  "Add to sequence" on one row, a full-width video and one transport bar, then Clip Start, the
  transcript and Clip End), the Moments header shows the count, "Analyze again" and "Show folder",
  and "Remove from sequence" moved above the preview. The shortcuts (Space, M, `[`, `]`) work as
  before.

### Removed

- The built-in Rainbow Six Siege HUD preset and the bundled War Thunder preset (`wt.json`), and the
  game-name hints that picked them. Footage of these games now has no preset: the first `analyze`
  (gaming or funny) or `events` run discovers the HUD regions and installs a preset (several minutes
  on CPU OCR), and kill detection can differ from 0.2.0. A preset you wrote yourself keeps working
  from the user presets folder (`cutpoint paths`), where `"builtin": true` keeps the original R6
  interpretation. Presets are matched by their `match` field (#8).
- The in-browser "Browse" button and folder browser in the UI's New Project Source field, and the
  `/api/browse` endpoint.

### Fixed

- HDR footage (HLG or PQ) and BT.2020 footage came out washed out, because it was cut down to 8
  bits with its HDR tags kept. `clips`, `extend` and `360` renders now convert it to SDR Rec.709
  with `zscale` and `tonemap` (hable) and tag it bt709, and UI thumbnails are converted the same
  way. This costs CPU time (about 30 fps at 1080p on 8 threads). Without `zscale` in the ffmpeg
  build a WARNING says so and the clip is exported as before; SDR sources get the same ffmpeg
  command as before. 10-bit footage that looks log or flat gets a WARNING suggesting the camera's
  LUT. `probe` now reads the four color tags (#289, #412).
- Clips of a recording with the same audio on several tracks (OBS writes its mix to every enabled
  track) came out 20·log10(N) dB too loud and clipped, because every track was summed. `clips`,
  `extend` and `edit` now leave out a track that copies an earlier one (checked once per source,
  cached in `audio-tracks.json`, and again over each clip's span), with an INFO line naming the
  tracks kept. Distinct tracks such as game and mic are still summed, and single-track sources are
  unchanged (#365).
- Audio and subtitle sync in `assemble`: with hard cuts, with `--transition fade` and in
  `final.srt`, a clip whose audio ran past (or fell short of) its video no longer drifts later
  audio; crossfades are rounded to whole frames; `--fade-dur` must be greater than 0 (a fade under
  one frame used to cut the video short); and a clip whose source audio ends before the clip ends
  its fade-out where the audio ends (#235, #238, #243, #250).
- `assemble --loudnorm`, and every `edit` kind that normalises loudness, no longer fails with
  "Requested resampling engine is unavailable" on ffmpeg builds without libsoxr (such as the
  gyan.dev "essentials" build).
- Exports are all or nothing. Clips, `extend`, 360 renders and `final.mp4` are written to a hidden
  `.<name>.part.mp4` and renamed when complete, and `clips --force` and `extend --out <clips_dir>
  --force` export into a hidden `.<folder>.cutpoint-tmp` and swap it in only when every clip
  succeeded. A failure, GPU encode error or Ctrl+C leaves the old clips and manifest untouched, and
  a failed fresh `clips` run leaves no partial clips or empty output folder. After a hard kill the
  next run puts the old clips back from `.<folder>.cutpoint-bak`. When `clips --force` or `extend`
  would replace a folder that is a drive root, mount point or bind mount, the run is refused before
  any export with a message to use `--out` (#109, #147, #160, #185).
- `clips --force` and `extend` replace only what Cutpoint made: the clips and `.srt` files listed in
  the old `manifest.json`, spelled as on disk. Before, `clips --force` deleted every `.mp4` in the
  folder, which with `--out` pointing at the footage's folder deleted the source video. A file the
  run reads is never replaced, hand-added or renamed clips stay (with a WARNING) and are assembled,
  a hand-added `.mp4` with a new clip's name stops the run before exporting, and `extend` writes the
  `.srt` of each re-exported clip and removes those of deleted clips (#145, #213, #239).
- `clips`, `extend` and `assemble` hold a lock file `.<folder>.cutpoint-lock`; a second run on the
  folder stops with an `ERROR` naming the process that holds it, and a lock left by a killed run
  is taken over. `manifest.json` records each clip's size and `"generator": "cutpoint"`; older
  manifests still work, matched by name. `assemble` restores or removes the staging and backup
  folders an interrupted run left, and `kept_sources` now comes from `clips` and `extend` (#145).
- With `video_encoder` `auto`, if a GPU encode (NVENC, AMF or VAAPI) fails on real footage, that
  clip, 360 render or final video is redone once on the CPU with one WARNING and the rest of the run
  uses the CPU. A forced GPU encoder is not retried, and a failed encoder probe reports ffmpeg's
  first real error (#109, #113).
- Cancelling and Ctrl+C: Cutpoint waits for ffmpeg to exit (at most 5 s) before removing a partial
  `.part` export or `.tmp.wav` file, `360 stitch` stops MediaSDKTest the same way, and failed or
  interrupted runs of `analyze`, `stage`, `360 qa`, `360 stitch`, `360 repair`, the UI and `events`
  discovery remove the output folders they created while empty. On Windows, Ctrl+Break stops a
  command like Ctrl+C (partial files removed, exit 130). In `cutpoint ui`, Cancel sends
  `CTRL_BREAK_EVENT` first and gives the job about 12 s to clean up, and the job shows "stopping"
  until it exits (#125, #148, #189, #194, #199, #225, #229, #257).
- HUD OCR menu detection: lobbies, operator and map select, result screens and other menus (also
  ones with little text, which hold still) no longer produce OCR events, while kills and banners in
  still gameplay (a scoped view) are kept. A frame counts as a menu from several lines of new text
  that stay on screen, or from a screen that holds still; the first run with a JSON preset
  re-reads the footage (replayed from the raw OCR cache when it has one). New stats:
  `stats.ocr.<source>.menu_frames`, `static_frames`, `text_frames`, `static_area` (#312, #319, #409).
- HUD region discovery: a busy HUD is no longer dropped as a menu, a column of score popups is no
  longer taken for the kill feed (it becomes a `popup` proposal with no region), a discovered kill
  feed grows by rows seen in fewer frames and ignores one stray box at its edge, and the
  `keyframes` step shows a percentage during discovery. Existing presets are not changed: run
  `cutpoint events <project> --discover-regions` and copy the new `proposal.json`, or delete a
  discovered preset to discover it again. Discovery params record `menu_min_boxes`, `menu_cover`
  and `menu_jump` instead of `menu_boxes` (#410).
- HUD phrase matching and mined `candidate_phrases` (profile versions 3 to 7): a multi-word phrase
  matches only whole words, mining finds mixed-case texts and phrases inside longer lines, and it
  leaves out gamertag slots, hint and tutorial sentences and control prompts. No command writes
  candidate phrases yet, so analysis results do not change. Built-in presets keep their OCR cache:
  run `cutpoint events <project> --force ocr` for the new matching (#313).
- `--vision`: the Ollama rating no longer outranks the analysis. It now scales the candidate's score
  (5/10 leaves it as is, 10/10 lifts it and 0/10 lowers it by the style's `vision_weight`, capped
  at 100), and the model's label replaces the clip's only from a rating of 7; every rated candidate
  keeps it in `vision_label` (#284).
- `analyze --language` no longer reuses a transcript cached for another language: a forced
  language has its own cache file `transcript-<model>-<language>.json`. A transcript made with
  `--language` before this fix is cached under the plain name; delete
  `cache/<source>/transcript-*.json` to transcribe again (#371).
- `crashes` and the `events` crash layer no longer miss a rider who fell from a high speed to a
  stop (a drop from at least 40 km/h to 4 km/h or less within 10 s, off a lift, now gets the dense
  check).
- `--style ski`: a stretch of footage at one score gives one clip instead of several back-to-back
  clips (the top 10 went from 6 to 10 distinct moments on the test footage; `cutpoint mix` picks the
  same way), and friends in view no longer lift the lift/stop mask or give the top VLM score to
  standing and chatting at a chairlift. A learned preset's `mask_friends` above 0 now means a hard mask (0 is still the soft mask). Saved
  features count friends in `vlm` until `analyze --style ski` is re-run; refit learned presets
  after that. `analyze --style ski` also scores a
  cached visual layer without the `events` extra (#282, #283).
- An ffmpeg step that reports progress (audio extract, scene proxy) could wait forever after ffmpeg
  had exited, so an analysis never finished; it now ends when ffmpeg exits (#338).
- `cutpoint ui` previews and projects: stale preview proxies are deleted (also when a source drive
  stays offline, after 30 days), a project's source list is cached and sources are probed only when
  the player loads them (200 sources took about 12 s to open, now about 0.1 s), edits to
  `source_remap` apply on the next request, and a clip removed from the Sequence and added back
  returns to its position (#128, #129, #411).
- `cutpoint ui` errors and banners: one banner when the server restarted or the link expired; the
  real error for an unanalyzed project whose source folder is missing; "Projects unavailable" and
  409 responses with the config file and error for an invalid `config.json`, while Settings stays
  usable; a port in use exits 1 with a hint; and no `ConnectionResetError` traceback when the
  browser cancels a video request (#133, #152).
- `cutpoint ui` page and outputs: the page no longer scrolls past the Jobs bar on a wide window and
  its files are revalidated on each load (#354, #415), a running analysis no longer looks frozen
  with reduced motion on (#322), the outputs panel lists every final with its `.fcpxml`, `.otio`
  and `.srt` (newest first), the assemble job no longer overwrites an existing final's `.srt`, and
  the UI has a favicon (#224, #226, #227).
- `cutpoint ui` project folder edge cases: when `ui`, `ui/proxy` or `ui/frames` is a file or a
  broken link, the UI gives a 409 error naming it instead of a 500 or traceback; an unreadable
  entry in the projects folder is skipped; saving the review in a command-line project creates the
  `ui` folder; and a failed export leaves no `segments-*.json` behind (#184, #198).
- New Project in the UI says which file types it found in a folder with no supported videos
  (#321).
- Windows paths and `source_remap`: a remapped mapped drive is recorded in its network form
  (`\\server\share\...`) and `source_remap_exclude` matches both spellings (clip manifests now record
  `\\server\share\...` instead of `Z:\...`); a staged copy is used when its source is read through
  a `subst` drive letter; and `cutpoint mix` and `360 repair` match exclusions and carried segments
  by the resolved path, read a relative `source` against its file's folder and reject a
  drive-relative `D:a.insv` (#119, #175, #181, #186, #193).
- A `source_remap` key spelled the Windows way (`D:\Videos` -> `/mnt/d/Videos`) now applies on Linux
  and macOS, case-insensitively with either slash, and `stage` keeps files that differ only in case
  apart. The analysis cache key becomes `D:\Videos\T\x.mp4`, so those files are analyzed and staged
  once more; old staged copies stay until `cutpoint stage --clear` with no targets (#149, #163).
- Config and JSON files: a wrongly typed `config.json` value gives one `ERROR` line and exit 1
  instead of a traceback (`source_remap_exclude` given as a single string is rejected); a running
  `cutpoint ui` picks up edits to `config.json` without a restart; JSON files saved with a UTF-8 BOM
  are read and other encodings give a clear error; a corrupt or incomplete `highlights.json` says
  to re-run `cutpoint analyze`; and `--segments` errors name the problem (a folder, a missing
  file, invalid JSON), with `null` for `tail` and `extend` keys now meaning unset (#126, #137, #138,
  #140, #150, #200, #228).
- Argument validation: `clips --top` and `--max-total`, `360 director --top`, `360 stitch --top`
  and `--max-total`, `analyze --vision-top` and `--max-candidates` reject 0 and negative values, and
  `--must-include-crash` values outside 0 to 1, as usage errors (exit 2); `360 qa --size` is checked
  when the command line is parsed. `clips --top -1` used to select all but one moment.
- `cutpoint analyze .` or a drive root without `--out` no longer gets a project with an empty name,
  and a folder named with dots (`trip.2024`) is no longer cut at the last dot. A project made from
  such a folder by an earlier version keeps its old name; pass `--out trip` to keep using it.
- `cutpoint 360 repair --segments` works on a hand-written segments file instead of crashing with
  `KeyError: 'id'`, `360 repair --out` into a new folder creates it, duplicate segment ids are an
  error, and `cutpoint mix` matches a segment whose source goes through a symlink (#110, #120, #151).
- Windows with the `events` and `gpu` extras: importing torch after whisper's GPU check no longer
  fails with `WinError 127` on `cudnn_cnn64_9.dll` (#215). A JSON result file that stays locked now
  gives the same "open in another program" error as a locked clip, and ffmpeg, llama-server and
  `360 stitch` crash codes show signed with their hex form (#146).

### Security

- `cutpoint ui` ignores an `exports.json` entry whose export folder (`dir`) or final video
  (`final`) is `.`, `..`, does not start with a letter, digit or `_`, or that a symlink or Windows
  junction resolves to outside the project folder, so a hand-edited or corrupted file can no longer
  point outside the project: such an export is neither shown, served nor opened in the file manager,
  and no output is written through a link (#141, #155).
- `cutpoint ui` treats a project metadata file (`highlights.json`, and `meta.json`, `review.json`
  and `exports.json` in the project's `ui` folder) that a symlink or Windows junction resolves to
  outside the project folder as missing, and refuses to save the review, export clips, record a final
  or analyze through it (409 error naming the file). The preview and thumbnail caches (`ui/proxy`,
  `ui/frames`) are not used either when they resolve outside the project (#168).
- Docker image (#112): uv is pinned by version and digest instead of an unpinned `curl | sh`
  install; the container runs as the non-root user `cutpoint` (uid 1000, build arguments
  `CUTPOINT_UID` / `CUTPOINT_GID`) owning `/data`; `git` and `curl` are no longer installed; and
  `CUTPOINT_LLAMA_BACKEND=cpu` is set. **Behavior change:** the 0.2.0 image ran as root, so an
  existing `cutpoint-data` volume is root-owned and fails with `Permission denied`; fix it once
  with `docker run --rm --user 0 --entrypoint chown --mount source=cutpoint-data,target=/data
  cutpoint -R 1000:1000 /data` (see the Docker section of `docs/getting-started.md`).

## [0.2.0] - 2026-10-22

### Highlights

- Optional `ui` extra and `cutpoint ui`: a browser workspace for analysis, moment preview, trim and
  reordering, clip export and final assembly.
- AMD GPU video encoding (VAAPI/AMF) next to NVENC, probed at run time with software fallback, and
  `cutpoint capabilities` to report what is usable.
- Linux support: llama.cpp downloaded by `cutpoint setup`, a `./cutpoint` wrapper and a Docker
  recipe for the browser workspace.
- Upgrade notes: the CUDA libraries for GPU Whisper moved to the `gpu` extra, `video_encoder`
  `auto` can now pick AMD encoders, and CLIP-derived caches must be rebuilt.
- Footage on drives moved or remapped with `source_remap`, and offline footage, is handled across
  `analyze`, `crashes`, `events`, `clips`, `extend` and the `360` commands.
- `cutpoint setup` downloads faster and verifies what it downloads, and missing-extra messages give
  the install command that matches the install type.

### Added

- Optional `ui` extra and `cutpoint ui`: a loopback-only browser workspace for standard-video
  analysis, moment preview/trim/reordering, clip export and final assembly. Includes saved
  review sequences, seekable media/proxies, isolated cancellable jobs, and non-overwriting
  exports with MP4/FCPXML/OTIO downloads.
- Runtime-probed AMD video encoding (VAAPI/AMF) alongside NVENC, with software fallback,
  `video_encoder` / `CUTPOINT_VIDEO_ENCODER` overrides, and `cutpoint capabilities [--json]`
  reporting hardware presence separately from usable encoding and transcription support.
- llama.cpp backends: Windows Vulkan/CPU assets alongside CUDA, and on Linux `cutpoint setup`
  downloads the official llama.cpp build (CUDA when an NVIDIA driver is detected, otherwise Vulkan,
  otherwise CPU). `CUTPOINT_LLAMA_BACKEND` overrides the choice on both, and `llama-server` is
  found via `CUTPOINT_LLAMA_SERVER`, the bundled build, then `PATH`. AMD faster-whisper
  transcription remains CPU-only.
- Linux Docker recipe (`Dockerfile`, `.dockerignore`) for the CPU browser workspace, with an
  optional `INSTALL_EVENTS` build argument; see the getting-started guide.
- `cutpoint --version`, and a `./cutpoint` POSIX shell wrapper that runs the repo `.venv`, like
  `cutpoint.cmd`.

### Changed

- **Upgrade note:** the CUDA libraries for GPU Whisper moved to the `gpu` extra; the base install
  is CPU-only and about 3.6 GB smaller. NVIDIA users: add `gpu` (`uv sync --extra gpu ...`,
  `uv tool install "cutpoint[gpu]" --reinstall`); `all` includes it. Without it, Whisper uses the
  CPU and logs one INFO line with the install command (#27).
- **Upgrade note:** CLIP-derived caches change, because the CLIP model for `events` and `crashes`
  now uses `ViT-L-14-quickgelu` to match the OpenAI weights (the checkpoint is unchanged). On
  existing projects, run `cutpoint events` again (and `cutpoint 360 analyze` for 360 footage)
  before `crashes`, `profile`, `mix` or `refmatch --taste`; until then those commands find no
  visual data. Taste heads trained with `refmatch --taste` before 0.2.0 must be re-trained.
- **Upgrade note:** `video_encoder` `auto` now selects AMF (Windows, AMD GPUs) or VAAPI (Linux,
  AMD and Intel GPUs) when it probes as usable, where 0.1.0 used libx264; set
  `"video_encoder": "cpu"` in `config.json` or `CUTPOINT_VIDEO_ENCODER=cpu` to keep software
  encoding.
- Project paths can be written `projects/<name>/...` on every OS. Path matching (`source_remap`,
  `stage` keys) is case-sensitive on Linux and stays case-insensitive on Windows; on Linux the stage
  index keys changed, so run `cutpoint stage --clear` (without inputs) and stage again. Ski features
  are recomputed once.
- `cutpoint setup` downloads the PANNs checkpoint from a Hugging Face mirror first (much faster than
  Zenodo) and falls back to Zenodo, verifies size and sha256, and logs which host each download
  comes from. `CUTPOINT_PANNS_URL` is tried first when set (#30, #52).

### Fixed

- `source_remap` and moved footage: `analyze` records the original path (absolute, in your
  spelling) instead of the remapped copy and reads from the remap when the original is gone;
  `clips`, `extend`, `360 analyze`, `360 director`, `360 stitch` and `360 status` read footage whose
  folder moved through `source_remap`; `crashes --reanalyze` and ski `events` re-analyze remapped
  projects and no longer exit 1 with "Input not found". Projects analyzed with a remap or relative
  inputs before this keep the old paths until `analyze` runs again; rerun `crashes` and `events` afterwards so their
  results match the new paths (#77, #78, #80, #91, #92).
- `stage --clear <inputs>` selects exactly the files `stage <inputs>` would copy (honouring
  `--recursive` and `--exclude`) and clears each input on its own, also one that no longer exists,
  and `stage --exclude` matches the original path of footage read through `source_remap` (#93,
  #103).
- Windows: a `source_remap` key and an input on a mapped network drive match whether written with
  the drive letter (`Z:\Footage`) or in UNC form (`\\server\share\Footage`), also while the drive is
  disconnected, and `analyze` records it in UNC form (#95).
- `crashes`, `events`, `mix`, `profile`, `360 director` and other commands that read cached data
  skip footage that is not reachable (offline drive, moved file) with a warning instead of failing,
  and `crashes` and ski `events` leave `highlights.json` unchanged when a source was skipped. They
  fail before writing anything when none is reachable (#15, #46, #57, #67, #82).
- Missing-extra messages now give the install command that matches the install type:
  `uv tool install "cutpoint[...]" --reinstall` for tool installs (keeping the extras, source and
  package index in use), `pip install "cutpoint[...]"` for PyPI installs, and the `uv sync --extra`
  form only in a clone, with the CUDA PyTorch index notes shown only where they apply (#34, #50, #61,
  #71, #83, #96, #104).
- `cutpoint events` and `cutpoint 360 analyze` check that the model files and tools the selected
  layers need are installed before unloading Ollama models, importing torch or starting a long
  analyze. One error lists every missing file and the `cutpoint setup --only ...` command (#23,
  #31). Runs whose results are cached now also need the model files.
- `cutpoint setup` verifies llama.cpp downloads against the sha256 digest GitHub publishes and
  installs into a staging folder that is swapped in, so a corrupt download or a failed install keeps
  the previous one (#35, #36). It no longer logs credentials, paths or query strings from
  `CUTPOINT_PANNS_URL`, and a malformed value is treated as a failed mirror (#59, #70).
- `crashes`: camera knocks while boarding a lift no longer count as a crash (#15).
- `cutpoint 360 director`, `extend` and `360 repair` keep a backup of a path file or
  `mix_segments.json` before re-planning over it and never overwrite an earlier one
  (`<file>.bak`, `<file>.1.bak`, ...), backups are written atomically, and `360 repair` leaves no
  backup behind when the re-run mix fails. `360 director --out-name` keeps the path file of a
  deselected segment as a backup (#15, #48, #58, #60, #68, #69, #81).
- `cutpoint mix` with nothing selected fails with `ERROR Nothing selected in <project>: ...` and
  exit code 1 instead of crashing, and `mix --target` must be greater than 0 (#15, #47).
- The Ollama default URL is `http://127.0.0.1:11434` (`localhost` could hang on some Windows
  machines), and the Ollama check reports "not reachable" instead of raising on a non-JSON reply
  (#15, #49).
- `cutpoint profile --context` logs a warning when the context could not be applied and falls back
  to a library profile when the context names exactly one, for example "mtb"; `profile` and
  `crashes` only unload Ollama models when the CLIP steps can run (#24, #32).
- An input folder or file that cannot be read gives a clear `Input not readable` error instead of a
  traceback, and a failed `cutpoint analyze` no longer leaves an empty project folder behind (#26,
  #79).
- Linux: GPU Whisper preloads the pip-installed CUDA libraries, remote-mount detection works,
  Insta360 X3 file lookup is case-insensitive, SDK runs get the needed `LD_LIBRARY_PATH`,
  `llama-server` is cleaned up when the parent process dies, project and JSON files respect the
  umask, the caption font is chosen with `fc-match sans-serif`, and projects created on Windows show
  correct source file names (reading the files still needs a `source_remap` entry) (#33).

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
