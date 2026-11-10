# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `settings.vision` in `highlights.json` records what the vision step cost (ratings, cache hits, seconds and tokens per phase, and the analyze time before it), with one DEBUG line per request and an INFO summary, and `scripts/vision_eval.py` compares projects by that cost and by the moments they keep (#909). Ranking is unchanged.
- The UI header shows what the footage was detected as ("Detected: Gameplay", or "Not sure: General style"), with the reason and the sources that disagree, and changes the style in one step: choosing one analyzes the project again in it (#897, #880).

### Fixed

- UI "Analyze again" on a project made on the command line with `--style auto` keeps `--style auto` instead of the style it picked.

### Changed

- The UI New project dialog defaults to Auto (action-cam stays preselected with camera telemetry).
- **BREAKING:** `POST /projects` without `style` creates an `auto` project (it was `gaming`).
- **BREAKING:** Vision no longer renames moments or clips; its description is kept in `vision_label` (#913). Re-analyze a project to get the original labels back.
- **BREAKING:** the action-cam vision tags are sport-neutral: `powder_spray` is `spray`, `fast_groomer` is `fast_open`, `chairlift` is `transit` and `lodge` is `indoor_break`. Old keys are ignored, so an existing project's tags read 0 until you run `cutpoint events` again.
- **BREAKING:** the timing key `ski_s` is `action_cam_s`, and `skiing_s` in `telemetry.json` is `riding_s`.
- The visual detector finds people only (the skis and snowboard classes are dropped), and the AI Chat action words include biking, surfing, skating and trail riding.
- **BREAKING:** crash detection without a model no longer saturates: a profile without normal-footage captions (`incident.neg`, such as the no-model fallback) takes those of `generic`, so its visual and dense crash scores, and the confidences in `crashes.json`, drop from the old 1.0 visual score. Crash dense caches are rebuilt once (#932).
- `crashes` with a discovered profile never scores a frame below `--activity generic`: the visual and dense checks take the larger of the profile's crash score and that of `generic` (#932).
- Discovered activity profiles are cached per footage (the files, the model, `--context` and the prompts): the same footage in a new project or run gets the same profile without a new synthesis, and an identical discovered profile reuses its library file instead of adding a `<name>-<hash>` copy. `--refresh-profile` also ignores that cache and replaces it (with `--context`, or while the footage matches no library profile; delete a matching one with `cutpoint profile delete NAME` to start over) (#932).

### Removed

- **BREAKING:** the `ski` extra is gone (`ui`, `keyring` stay as aliases), and `--style ski` is now an ordinary unknown style with no special message.

## [0.5.0] - 2026-11-10

### Highlights

- Footage detection: `cutpoint detect` and `--style auto` tell gameplay, action cam, talk and general footage apart, ask the vision model about unsure footage and use `gaming` for a gameplay guess (#513, #896, #898).
- Activity profiles are discovered from the footage; only `base` and `generic` ship, and `cutpoint profile list|show|create|validate|delete` manages the library (#906).
- Gaming streams: a dialogue reader marks funny or heated talk, detected events become moments, and the HUD reader learns announcements, score readouts and explosions (#779, #772, #800, #801).
- Resume and memory safety: `cutpoint resume`, resumable clip exports, `--batch-size` and a memory guard with a `low_memory` error (#109).
- Custom scoring styles with `cutpoint style`, and crashes become moments for every style except 360 (#843, #486).
- `cutpoint ui` is redesigned (Graphite theme, filmstrip trim, Cut mode with per-cut transitions) and has a Models tab for the Chat and Vision models, including subscriptions.

### Added

- `cutpoint profile list|show|create|validate|delete` manage the activity profile library in `<CUTPOINT_HOME>/profiles`, with `--json` (#906).
- Activity profile discovery also runs for action-cam footage, and in `analyze` and `edit` whenever a local Ollama answers, without `--vision` (#906).
- `--style auto` and `cutpoint detect` show unsure footage to the vision model; `detect` takes `--vision`/`--no-vision`, `--ollama-url` and `--ollama-model`, and results gain `decided_by` (#513, #896, #898).
- Footage category detection: `cutpoint detect <inputs> [--json]` and opt-in `--style auto` on `analyze` and `edit`, with a HUD text check for unsure sources (#513).
- `cutpoint style list|show|create|validate|delete|features` manages your own scoring styles, used with `--style NAME` (#843).
- Crashes in `crashes.json` become moments for every style except 360, with a Crashes filter and crash badge in `cutpoint ui` (#486).
- A `dialogue` analyze step for `gaming`: a text model marks funny or heated talk, and each stretch becomes a moment titled by what it is about (#779, #782).
- In `gaming`, detected events can create moments of their own, and the candidate list has no fixed cap (#772).
- Learned on-screen announcements, on-screen score readouts and confirmed explosions in the audio become events (#800, #801).
- HUD region discovery labels regions with the vision model and ignores stream chat (#787, #776).
- `--ocr-det-skip` / `--no-ocr-det-skip` (and `"ocr_det_skip"` in `config.json`) for faster gameplay HUD reading, off by default (#807).
- `--visual-index` / `--no-visual-index` (and `"visual_index"` in `config.json`) write the frame embeddings for the AI Chat's visual search, off by default (#789).
- `cutpoint resume <project> [--dry-run] [--json]` re-runs an interrupted or failed run from `progress.json`; `cutpoint ui` offers Resume too (#109).
- Resumable `clips` and `extend` exports keep finished clips for the next run (#109).
- `--batch-size N` on `analyze`, `events`, `crashes` and `360 analyze` unloads models between sources on long runs (#109, #713).
- Memory guard: Cutpoint checks free memory before a model loads and stops with the `low_memory` error when short (`memory_guard` in `config.json`) (#109).
- `cutpoint setup --only whisper` downloads the Whisper model ahead of time; runs load Whisper from the local cache first (#651).
- `assemble <project>` and `edit --no-keep-clips` make the final video without keeping clips (#519).
- `assemble --joins FILE` sets single joins to `cut` or `fade`; Cut mode in `cutpoint ui` sets them per cut (#774).
- Visual candidates: `action-cam`, `action` and `sport` analyses also list scenic views, group shots and cruising runs (#645).
- AI Chat can search the timeline by a text query (`look_for`), set Color, watch a moment and see what analysis saw (#529, #640, #634).
- A Models tab in `cutpoint ui` sets the Chat and Vision models; Vision can use Ollama, the Anthropic, OpenAI and xAI APIs or a subscription (#575, #616).
- `cutpoint proxy`, `cutpoint login` and `cutpoint logout` for Claude Code, Codex and Antigravity subscriptions; `cutpoint ui` starts the proxy itself (#831).
- `cutpoint ui`: Delete this project, Select all in the sequence, multi-source projects, Analysis options, Export picture and audio options, and a single-key shortcuts switch (#746, #745, #546, #549, #550, #748).
- `install.sh` and `install.ps1` offer to install Ollama in full mode; releases attach the installers and `SHA256SUMS` (#525, #531, #696).

### Changed

- **BREAKING:** Cutpoint ships only the `base` and `generic` activity profiles; `hike`, `moto`, `mtb`, `skate`, `ski`, `snowboard` and `surf` are removed, and `--activity` takes only a library name (#906).
- **BREAKING:** the `ski` style is renamed `action-cam`, with no alias; `--style ski` is refused; a `config.json`, project or learned preset that says `ski` stops with an error: analyse again.
- **BREAKING:** action-cam projects keep their features in `action_cam_features/` (was `ski_features/`) (#610).
- **BREAKING:** the `events` VLM layer uses the shared vision model (Ollama `qwen3.5:9b` or the hosted Vision model); the llama-server + Qwen2.5-Omni stack is removed, with the `events` options `--ctx`, `--image-max-tokens`, `--frame-height` and `--no-audio` and `setup --force`, and `cutpoint setup` downloads about 2 GB instead of 8 GB.
- **BREAKING:** the AI Chat consent is version 3, as whole transcripts now go to a hosted chat model during `analyze`; users of one are asked again.
- When `--style auto` is still unsure it uses `gaming` for a gameplay guess, else the General style (`action`) (#896, #898).
- The default vision model is `qwen3.5:9b` instead of `qwen3-vl:8b` (#475).
- `--vision` rates a moment's frames together with what was said in it (#770).
- `assemble` and `edit` have `--format mp4|fcpxml|otio` (#519).
- `--must-include-crash` and the default `must_include_crash` (0.6) apply to every non-360 style (#486).
- Scenic view moments are 5-7 s instead of 3-5 s (#738).
- The `PROGRESS` steps of `analyze` are now 7 and of `edit` 9 (new `dialogue` step).
- The default kill words are generic only.
- `cutpoint ui` has a new look, the Graphite theme, as the default, with a rebuilt center stage, moments grid and side column (#747, #750).
- Faster analysis: styles that do not weigh scenes skip full scene detection, scene detection stays on the GPU, and HUD discovery spends less time on the vision model (#730, #845).
- `assemble --loudnorm` applies one gain to the whole edit, then the limiter, and warns when the edit stays below -14 LUFS (#523).

### Removed

- **BREAKING:** `cutpoint setup --torch-cuda` is gone.
- **BREAKING:** `cutpoint events --preset-scope package` is gone; use `user` or leave it out.
- **BREAKING:** `cutpoint refmatch` and `cutpoint fit` are gone; they are now the development script `scripts/refmatch.py` (`uv run python scripts/refmatch.py match|fit`); learned styles still load.
- **BREAKING:** the personal taste head and its `taste` feature are gone; learned styles rank without it.
- **BREAKING:** `cutpoint 360 check` is gone; use `360 qa <clips_dir>`.
- **BREAKING:** `assemble` with `--format mp4` no longer writes `final.fcpxml` and `final.otio`.

### Fixed

- Busy gameplay beside a kill feed, or with many notices on screen, is no longer taken for a menu (#858, #891).
- A recording that is not 16:9 no longer skips all HUD text reading (#873).
- Stream chat, clock-stamped logs and menu banners no longer over-flag kills and moments (#776, #777, #790).
- Whisper can write laughter markers, and laughter is counted once and more strictly (#791, #792).
- `clips` without `--top`, `--ids` or `--max-total` no longer takes one long moment as the whole edit (#786).
- A vision rating no longer counts once a moment is widened or trimmed (#802).
- An explicit `--ollama-url` or `--ollama-model` now wins over the hosted models and reaches the events and crash steps; a bad port in `OLLAMA_HOST` or `ollama_url` no longer fails a run (#816, #818, #839, #840).
- The VLM layer of `events` no longer invents kills or enemies in games without combat (#596).
- A source that ends loud no longer has its last moment picked again and again (#771).
- `cutpoint detect` and `--style auto` no longer take camera footage with a news lower third for gameplay (#701).
- Vision no longer sends frames to an Ollama on another machine without the user's consent (#631).
- Requests to Ollama no longer follow redirects (#719).
- AI Chat proposes sequences reliably with a local model, reaches length targets and understands more edit phrasings (#733, #737, #739, #743, #864).
- `clips` and `assemble` fill gaps in a source's audio timestamps, so the audio ends with the video (#545).
- HLG exports, previews and stills keep their real saturation (#548).
- 360 clips filmed upright or handheld keep a steady heading and no longer spin (#547).
- Recordings with copied audio tracks are analysed at the level of one track (#685).
- 10-bit sources decode on the GPU in crash detection, the visual index and 360 QA (#870).

### Security

- The PANNs audio checkpoint loads as weights only, so a tampered file cannot run code (#598).
- `CUTPOINT_ASSISTANT_API_KEY` is never sent to `api.anthropic.com`, `api.openai.com` or `api.x.ai` (#504).

## [0.4.0] - 2026-10-29

### Highlights

- One-step installers: `install.sh` for Linux and `install.ps1` for Windows.
- Editions: Cutpoint is full (every feature) or lite (no torch) (#502).
- Automatic color correction per clip with `--color auto|underwater` (#290, #494).
- `cutpoint ui` AI Chat: ask a model to find, add, remove, reorder or trim moments, with Ollama or your own provider (#346).
- Faster exports and runs: parallel NVENC exports, faster fade joins and fewer ffprobe calls (#464, #463, #372).

### Added

- `install.sh`: one-step Linux installer that installs uv and ffmpeg when missing and picks CUDA or CPU torch; `--lite` installs without torch.
- `install.ps1`: the same one-step installer for Windows.
- Editions: Cutpoint is full (every feature) or lite (no torch: no `events` or `gpu` extra) (#502).
- `clips` and `edit --color auto|underwater` and `--color-strength 0-1`: automatic color correction per clip (#290).
- `cutpoint ui` AI Chat: find, add, remove, reorder or trim moments with a local model or your own AI provider (#346).
- `assemble --audio-transition cut|crossfade` sets the audio join apart from `--transition` (#466).
- `cutpoint ui`: Light and Dark themes, a Color slider in Export, a sorted and filtered Moments list, candidate ids on moment cards and a filmstrip on the trim rail (#445, #461, #469, #487, #261).
- `cutpoint ui` One-Click Edit: Highlights, Montage, Clips and Funny moments presets.
- `cutpoint ui`: "Export clips" of a 360 project first stitches the recordings it needs.

### Changed

- A feature that needs a missing extra says it is "not available in Cutpoint lite" and how to install it (#502).
- `--color underwater` grades only clips that look underwater (#494).
- Packaging: the `ui`, `keyring` and `ski` packages are core dependencies, so every install runs `cutpoint ui` and the 360 and telemetry commands.
- `cutpoint ui`: Smart edges is split into Speech-aware cuts and Smart tails.
- Faster `assemble --transition fade`: only the video around each join is re-encoded (#463).
- Faster exports with a GPU encoder: `clips` and `edit` export 2 clips at a time with NVENC (#464).
- Faster runs: a source is probed with ffprobe once per run (#372).
- HUD region discovery is about 22 % faster on a first run (#443).

### Fixed

- `assemble`: the audio and container of `final.mp4` end with the video (#473).
- `cutpoint ui` stills of BT.709 and HDR sources show the same colours as their exports (#441).
- HLG (HDR) exports, `cutpoint ui` previews and stills are brighter, as they came out too dark (#449).
- 360 renders no longer tilt the horizon when the camera turns over in a clip (#442).
- HUD region discovery finds icon kill feeds (#433).
- `clips --force` and `extend` work on macOS and other case-insensitive drives (#247).
- Highlight candidates in game menus and lobbies are left out (#440).
- Parallel exports with NVENC no longer switch to libx264 (#470).

## [0.3.0] - 2026-10-27

### Highlights

- `cutpoint edit <video...> --kind highlights|montage|clips|funny-moments` goes from footage to a finished video with `.srt` subtitles in one command.
- Agent support: a `--json` result envelope on stdout for the main commands, and `cutpoint skill install` / `skill print` for Claude Code.
- Color and audio: HDR and BT.2020 footage is tone mapped to SDR Rec.709, `--lut` applies your 3D LUT, and `--compress-peaks` evens out loud peaks.
- Analysis: `analyze` and `edit` run the reactions, keyframes and crashes steps themselves; game HUD reading ignores menus and the Ollama vision rating is opt-in.
- Safer exports and cancelling: exports are all or nothing, Ctrl+C removes partial files, and a lock stops two runs on one clips folder.
- `cutpoint ui` is a fuller workspace: trim handles, whole-source scrubbing, fast previews, Play sequence, Insta360 360 projects and Settings.

### Added

- `cutpoint edit <video...> --kind highlights|montage|clips|funny-moments` runs analyze, clips and assemble in one command (#2).
- SRT subtitles next to each clip and the final video (#5).
- `--json` result envelope on stdout for the main commands (#4).
- `cutpoint skill install [--dir DIR] [--force]` installs a Claude Code skill; `skill print` prints it (#1).
- `--lut FILE` and `--lut-strength 0-1` on `clips` and `edit` (#290).
- `--compress-peaks off|light|strong` on `clips`, `assemble` and `edit` (#326).
- `analyze` and `edit` run the reactions, keyframes and crashes steps.
- Linux AMD GPUs decode H.264 with VAAPI (#22).
- `cutpoint capabilities` reports torch and the usable decoders and encoders (#22).
- `cutpoint ui`: whole-source preview and trim, fast previews of sources the browser cannot play, Insta360 X3 projects and a Settings dialog (#324, #352, #323, #325, #320, #341, #142, #353).
- HUD presets can define event types with an `events` list (#8).

### Changed

- **BREAKING:** `clips --force` and `extend` refuse a `manifest.json` Cutpoint did not write (#145).
- Exported clips are always tagged BT.709 (#289, #290).
- The Ollama vision rating is opt-in for every style (`--vision`).
- `analyze` writes `events.json` and `events.txt` (and `crashes.json` for action footage) (#408).
- Decoding follows the GPUs detected (#22, #429).
- Less CPU time in analysis: the HUD OCR runs in the background and on fewer threads (#372, #374, #375, #376, #402, #403).
- The `all` extra includes `ui` (#122).

### Removed

- The built-in Rainbow Six Siege HUD preset and the bundled War Thunder preset (`wt.json`) (#8).
- The in-browser "Browse" button and folder browser in the UI's New Project Source field.

### Fixed

- HDR (HLG or PQ) and BT.2020 footage no longer comes out washed out (#289, #412).
- Clips of a recording with the same audio on several tracks are no longer too loud and clipped (#365).
- Audio and subtitle sync in `assemble` with hard cuts and fades (#235, #238, #243, #250).
- `assemble --loudnorm` no longer fails on ffmpeg builds without libsoxr.
- Exports are all or nothing: partial files no longer stay behind (#109, #147, #160, #185).
- `clips --force` and `extend` replace only what Cutpoint made (#145, #213, #239).
- With `video_encoder` `auto`, a GPU encode that fails is redone on the CPU (#109, #113).
- Cancelling and Ctrl+C wait for ffmpeg and remove partial files (#125, #148, #189, #194, #199, #225, #229, #257).
- HUD OCR menu detection leaves out lobbies, map select, result screens and other menus (#312, #319, #409).
- `--vision`: the Ollama rating no longer outranks the analysis (#284).
- `analyze --language` no longer reuses a transcript cached for another language (#371).
- An ffmpeg step that reports progress no longer waits forever (#338).
- Windows paths and `source_remap` on mapped drives, also from Linux (#119, #149, #163, #175, #181, #186, #193).
- A wrongly typed `config.json` value gives one `ERROR` line and exit 1 (#126, #137, #138, #140, #150, #200, #228).
- Windows with the `events` and `gpu` extras no longer fails with `WinError 127` when importing torch (#215).

### Security

- `cutpoint ui` ignores an `exports.json` entry whose export folder or final video points outside the project (#141, #155).
- `cutpoint ui` treats a project metadata file that a link resolves to outside the project as missing (#168).
- Docker image: uv is pinned by version and digest (#112).

## [0.2.0] - 2026-10-22

### Highlights

- Optional `ui` extra and `cutpoint ui`: a browser workspace for analysis, moment preview, trim and reordering, clip export and final assembly.
- AMD GPU video encoding (VAAPI/AMF) next to NVENC, with software fallback, and `cutpoint capabilities` to report what is usable.
- Linux support: llama.cpp downloaded by `cutpoint setup`, a `./cutpoint` wrapper and a Docker recipe for the browser workspace.
- Footage on moved or remapped drives (`source_remap`) and offline footage is handled across the commands.
- `cutpoint setup` downloads faster and verifies what it downloads.

### Added

- Optional `ui` extra and `cutpoint ui`: a loopback-only browser workspace for standard-video projects.
- AMD video encoding (VAAPI/AMF) alongside NVENC, probed at run time, with software fallback.
- llama.cpp backends: Windows Vulkan/CPU assets alongside CUDA, and Linux downloads by `cutpoint setup`.
- Linux Docker recipe for the CPU browser workspace.
- `cutpoint --version`, and a `./cutpoint` POSIX shell wrapper.

### Changed

- **Upgrade note:** the CUDA libraries for GPU Whisper moved to the `gpu` extra; NVIDIA users: add the `gpu` extra (#27).
- **Upgrade note:** CLIP-derived caches change, because the CLIP model for `events` and `crashes` now uses `ViT-L-14-quickgelu`; run `cutpoint events` again before `crashes`, `profile` or `mix`.
- **Upgrade note:** `video_encoder` `auto` now selects AMF (Windows, AMD) or VAAPI (Linux, AMD and Intel); set `"video_encoder": "cpu"` to keep software encoding.
- Project paths can be written `projects/<name>/...` on every OS.
- `cutpoint setup` downloads the PANNs checkpoint from a Hugging Face mirror first (#30, #52).

### Fixed

- `source_remap` and moved footage work across `analyze`, `crashes`, `events`, `clips`, `extend` and the `360` commands (#77, #78, #80, #91, #92, #95).
- Commands that read cached data no longer fail on offline footage (#15, #46, #57, #67, #82).
- Missing-extra messages give the install command that matches the install type (#34, #50, #61, #71, #83, #96, #104).
- `cutpoint events` and `cutpoint 360 analyze` check their model files and tools before they start (#23, #31).
- `cutpoint setup` verifies llama.cpp downloads against their sha256 digest (#35, #36, #59, #70).
- `crashes`: camera knocks while boarding a lift no longer count as a crash (#15).
- `cutpoint 360 director`, `extend` and `360 repair` keep a backup of a file they replace (#15, #48, #58, #60, #68, #69, #81).
- The Ollama default URL is `http://127.0.0.1:11434`, as `localhost` could hang on some Windows machines (#15, #49).
- Linux: GPU Whisper finds the pip-installed CUDA libraries (#33).

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

[Unreleased]: https://github.com/FerventOrange/cutpoint/compare/0.5.0...HEAD
[0.5.0]: https://github.com/FerventOrange/cutpoint/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/FerventOrange/cutpoint/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/FerventOrange/cutpoint/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/FerventOrange/cutpoint/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/FerventOrange/cutpoint/releases/tag/0.1.0
