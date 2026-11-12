# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Cutpoint lite, which has no OCR engine, gives a source the frame probe leaves unsure between gameplay and general a screen-text cue in place of the HUD text check: word-sized shapes with edges that stay in the same place at the edge of the picture in most of 8 frames. At least 3 of them make the source ambiguous gameplay (never sure; reason `screen text anchors in N places (no OCR)`), so `--style auto` leans to `gaming` for thin-HUD recordings it called general or talk before. Cached as `category-anchors-v1.json`; full installs, whose HUD text check reads the text, are unchanged (#904).
- `style_base` in `highlights.json` (next to `style` and `style_kind`) records the built-in style the project's style is based on (its own name for a built-in style), so `scripts/vision_prior_eval.py` groups a style of yours without the style library. The event layers (nominating events, scoring, the vision windows) read it too; after the style is deleted from the library, scoring, nominating events and the vision windows still match the events of that base. Older files without it fall back to the library.
- `settings.features_by_source` in `highlights.json` records, per source, the features its scores were divided by (an action-cam source without telemetry, or with IMU-only telemetry (no GPS rows, no wind reading), divides by fewer than `settings.features_available`, which stays the install's list). Older files without it still load (#960).
- AI Chat word splices: a `find_words` tool finds where a word or phrase is said, to the word, from the cached transcript words (short words such as "a" too). An `add_range` op adds an exact source range, so the chat can join words said in different places ("make her say ..."). Exact rows are cut at their own times (no tails, speech-aware moves or padding, a 10 ms fade) and joined with hard cuts, and `cutpoint extend` leaves them as they are. A word timed under 0.1 s is widened to 0.1 s around its middle. A chat trim to either edge of a word `find_words` found makes that clip exact as well, so a cut asked at a word stays there. `exact` is a new segment key for `clips --segments`.
- Footage category Action: a sport or fast physical activity filmed by a camera that rides along or follows the person. `--style auto` finds it when the vision footage question answers `action` (asked for unsure sources and for camera-tagged sources without telemetry, #958) and uses the `action` style (`funny` for `funny-moments`). A `camera` answer stays General. The action-cam style still needs camera telemetry. Old projects with category `general` still load.
- `scripts/bench.py`, a development benchmark across footage categories: runs `analyze --style auto` on labelled footage listed in a manifest, scores the detection per bucket (right, ambiguous, sure-wrong) and the ranking with `moment_eval.py`, keeps holdout footage hidden unless `--holdout` (pinned by a hash), compares variants or two results files with an advisory gate, samples Ollama and GPU use per run, and writes a results JSON and `report.md` (#900). See [docs/workflows/benchmark.md](docs/workflows/benchmark.md).
- `settings.vision` in `highlights.json` records what the vision step cost (ratings, cache hits, seconds and tokens per phase, and the analyze time before it), with one DEBUG line per request and an INFO summary, and `scripts/vision_eval.py` compares projects by that cost (`--cold-pre` for a cold analyze time) and by the moments they keep (#909). Ranking is unchanged.
- `vision_time_share` in `config.json` caps the vision rating at a share of the analyze time before it (default 0.3). It rates the moments nearest the cut first, takes cached ratings after the cap, makes at least one call and can run over by one call (#909).
- The UI header shows what the footage was detected as ("Detected: Gameplay", or "Not sure: General style"), with the reason and the sources that disagree, and changes the style in one step: choosing one analyzes the project again in it (#897, #880).

### Fixed

- Crash detection skips a source whose telemetry, IMU or cached layers cannot be read, with a WARNING naming the file and a `step_failed` warning (`files` names it), instead of failing the crashes step for every source of the project (#985); when every source fails, a WARNING names the last file and its error, and the command ends with one ERROR line. A GoPro telemetry payload that telemetry-parser hands over as a nested list (the likely cause of "'list' object cannot be interpreted as an integer") is read as if flat; one it cannot use is skipped.
- Vision rating: a Vision model that gives no usable reply (empty, cut off, unparseable, or a failed request) for 3 moments in a row stops the rating with one WARNING naming the provider and model (`vision_unavailable` in `--json`) instead of being asked about every moment; an OpenAI-compatible thinking model's rating is read from `reasoning_content`/`reasoning` when its content is empty (#986).
- The shared decode of a 10-bit source on NVDEC keeps the format NVDEC hands over instead of converting every frame (the conversion cost about 20 % on 4K HEVC). This applies only when NVDEC decodes the codec and format (8/10-bit 4:2:0 HEVC, VP9 and AV1, 8-bit 4:2:0 H.264); other sources, such as 10-bit H.264 that ffmpeg decodes in software, keep their own format.
- Scene cuts land where the clips are cut on every ffmpeg version: the full scene pass, the split NVDEC/CPU pass and the edge windows of a style without scene weight all count from the file's start. A video that starts after the audio (common in replay-buffer and remuxed recordings) or an MPEG-TS file (`.ts`, `.mts`, `.m2ts`) could move them by a quarter second or more, so clip edges snapped next to the cut. MPEG-TS and other containers that do not seek exactly get the full pass instead of edge windows, and `CUTPOINT_SCENE_DECODE=hybrid` no longer splits them. The scene caches are now `scenes-v2.json` and version 2 of `scenes-windows.json`, so each source's scene cuts are detected again once. Until `analyze` runs again, `cutpoint events` on a project analysed before scores its sources without scene cuts in a style that weighs them (an INFO line names each source) (#976).
- The AI Chat no longer ends with "no answer after 8 rounds" when the moments cannot answer a request (a montage of something the footage does not show): the last round, and the round after the model hit its limit of watched moments, offer only `propose_edit` and ask for the answer, so the model says what it could not find, what the moments show and what it can do instead. A model that still looks something up gets that round once more; then the turn ends with an earlier proposal, its first answer or a short answer naming the searches it made (#988).
- A project analyzed with a user style you later deleted keeps working in `cutpoint clips`, the browser workspace and the AI Chat: they use the built-in base `highlights.json` records (`style_base`), as `events` and re-analysis already do. An older file without the base gets one clear error from `clips` (analyze again with another `--style`); the page shows the style's options as unknown and the AI Chat plans edges as gaming (with a warning). Analyze again and the one-click presets of the page run in that base (not the deleted name) and the project's chosen style becomes the base; with no recorded base the page answers "pick another style". A `--style NAME` you give on the command line must still exist.
- The other Ollama probes (`/api/ps` of the vision rater and the memory guard, the AI Chat model list) wait 1 s to connect, not 3 or 5, and the AI Chat's "not reachable" message gives the host and port and a short reason ("connection refused") instead of the full URL and exception text, so no `user:password@` reaches a log line (#928).
- A moment marked `vision_skipped` loses the mark when a re-rank (`cutpoint events`, or the rescore after the vision step) puts it in the default edit of the ranking without vision, so rated moments can no longer pass a moment the edit now uses (#949).
- A vision-rated moment held at an unrated neighbour's score gets the same score on every re-rank instead of moving by 0.1 between the first and the second; candidates tied on score and grid score are listed by source order and time in `highlights.json` (#949).
- **BREAKING:** `analyze` and `edit` with `--style action-cam` (or a style over it) on footage without camera telemetry no longer end with no moments and no warning: when no source has GoPro or Insta360 X3 telemetry the `action` style runs instead, recorded as `style` `action` with `settings.style_requested` and `settings.style_fallback` in `highlights.json`, and an `action_cam_no_telemetry` warning (WARNING line and `--json`). A project where only some sources have telemetry, or whose telemetry could not be read, stays action-cam and the warning names those sources. `360 analyze` never falls back (#939).
- **BREAKING:** in an action-cam project where only some sources have camera telemetry, the sources without it (a phone or DJI clip next to a GoPro) give moments again: their score leaves the telemetry features (airtime, crash, speed, carve, faces, stopped, short clip) out of its divisor and their pick floor is scaled to match, so their best moments rank with the others and reach the default edit. Sources with telemetry score exactly as before, but mixed projects re-rank (#950).
- **BREAKING:** in an action-cam project, a source whose telemetry has no GPS rows and no wind-meter reading (an Insta360 X3 file: IMU only) leaves speed out of its score's divisor and its pick floor is scaled to match, so its jumps, crashes and talk rank with the moments of full-telemetry sources instead of scoring low. GoPro sources (GPS rows or wind readings) and sources without telemetry score exactly as before, but mixed projects with such sources re-rank (#959).
- UI "Analyze again" on a project made on the command line with `--style auto` keeps `--style auto` instead of the style it picked.
- `cutpoint events` rebuilding the crash moments lists candidates tied on score by source order and time, as `analyze` does, instead of in the order they were saved (#961).
- A hosted Vision model (OpenAI, xAI, OpenAI-compatible servers, subscriptions) asked the footage question of `--style auto` gets the answer schema as structured output, as Ollama does (Anthropic still only by the prompt). A reply missing a key the schema requires is unusable, from a hosted model or Ollama, as one that is not JSON (#920).
- `memory_guard` in `config.json` rejects `NaN` and `Infinity` with the same error as a negative value, as `vision_time_share` does (#944).
- `--style auto` and `cutpoint detect` find Action footage without telemetry that was detected as General: a source that only its camera tags made General (a DJI file, for example) is now shown to the vision model too, and only an `action` answer changes it (any other answer, or no vision, keeps it General by its camera tags, so phone clips and re-exports stay General); the vision question now calls fast first-person footage, such as FPV drone flight, action (FPV drone clips can still be answered `camera`; that part of #958 is still open). The answers are cached in `category-vision-v3.json`, so sources are asked again once; camera-tagged sources add one vision request each (#958).
- Saving a discovered activity profile never overwrites a library file that is not a readable profile (a corrupt or foreign JSON file with the same name): the profile is saved as `<name>-<hash>` (or the next free `-2`, `-3`, ...) and a WARNING names the file left in place.
- An OpenAI-compatible server (not OpenAI's own Responses API) that answers HTTP 400 naming `response_format` or `json_schema` is asked once again without it, and later calls of that provider leave it out; the reply is still checked for a JSON object with the required keys. Other 400s, timeouts and key errors are not retried (#920).
- A Vision reply that is not a JSON object is dropped with a DEBUG line naming the model and the JSON type, no longer silently.
- `cutpoint events` and the re-analysis of an action-cam project no longer end with `Unknown style` after the project's style was deleted with `cutpoint style delete`: the style degrades to the built-in base `highlights.json` records (`style_base`), with an INFO line naming both, so the crash moments are rebuilt and the events nominate and rescore as that base; the re-analysis records the base as the project's style. An older file without `style_base` ends with an ERROR line that names the missing style and asks for `cutpoint analyze` with another `--style`. A project of a deleted style over `action` or `sport` also keeps its visual pool when `highlights.json` has no `settings.visual_pool` record.

### Changed

- The vision rating prior (`RATING_PRIOR`, the mean a prompt kind is shrunk toward when fewer than 5 moments of it are rated) is now kept per style group, action-like (action, sport, action-cam and styles over them) and other. Measured with the new `scripts/vision_prior_eval.py` (#948), the frames prior of action-like styles is 5.1 instead of 5.6, which shifts the centre of frame ratings when fewer than 5 moments of a project are rated; the other priors keep their values.
- Profile discovery, the footage category probes and the AI Chat's timeline sheet grab their frames up to 4 at a time (each frame is its own ffmpeg), as the vision rating does; frames, order and skipped grabs are unchanged (#946).
- Activity profile discovery gives the same profile for the same footage: the describe and synthesis calls to local Ollama use temperature 0 and seed 42 (discovery only), and the describe call has an answer schema. Discovering the same footage again, also with `--refresh-profile`, overwrites that footage's own discovered library profile instead of adding a `<name>-<hash>` copy, and renames the file when the new answer names a different activity; the file stores a `footage` field. After upgrading, discoveries cached by earlier versions are not reused, and footage whose old discovered file has no `footage` field gets one more new file once (#947).
- **BREAKING:** the `detection` block of `highlights.json` is version 4 (it adds the category `action`). The frame probe cache stays `category-v3.json`, so sources are not probed again. `scripts/style_auto_eval.py` label files now need `action` entries (its flip gate requires every category to be labelled), so footage without telemetry that was labelled `general` (sport, for example) should be relabelled `action`.
- The UI header style menu lists Auto, Gameplay, Action, Action cam (only with telemetry) and Talk. General is no longer a menu item but is still a detection label. The crash step runs for Action footage.
- Discovered activity profiles are named after the activity alone: crash words of the floor and of the profile are dropped from the name and the `detect` captions (`ski-crash` is `ski`), and the prompts ask for the activity, never an incident. The vision model sees the most different keyframes, skipping dark or flat ones (a cut to black, white-out snow), and crash captions that fit more than 10% of the footage's own keyframes are dropped (#916).
- `activity.json` records `discovery` (`describe_s`, `attempts`, per attempt `synth_s`, `answered` and `timed_out`, `failed` with the reason when a failure is remembered, and `cached`), and `crashes.json` `timings.profile_s`. The synthesis retry is skipped when the first attempt took nearly the whole timeout (#916).
- **BREAKING:** without a model, the fallback profile is always `generic` (it was named after CLIP's best activity term); CLIP's three best terms are kept in `description` (#916).
- **BREAKING:** discovered profiles no longer reword the VLM prompts: they keep the generic wording (and its VLM window cache) (#916).
- **BREAKING:** crash phrases written by the LLM are capped at 0.75 (was 0.9), so one alone no longer makes a crash candidate (#916).
- **BREAKING:** a discovery the model answered unusably or that timed out is remembered per footage: the model is not asked again until `--refresh-profile`, another model or a new discovery version; a connection error is not remembered, so the next run asks again (#916). Profiles discovered by earlier versions stay in the library and keep matching their footage, with their old name, VLM wording and 0.9 phrase caps; to discover one again, run `cutpoint profile delete NAME`, then the command with `--refresh-profile`.
- The vision rating grabs the frames of a moment in parallel (up to 4 ffmpeg at once): the frame grab takes about a quarter less time on 4K footage, with the same frames (#909).
- The UI New project dialog defaults to Auto (action-cam stays preselected with camera telemetry).
- **BREAKING:** `--vision-top N` caps the total number of moments rated, closest to the cut first. By default the vision rating rates the default edit's moments (not must-include ones) plus max(2, 30% of the edit's moments), at most 24, below its cut, instead of one moment per 2 minutes of footage (10-60) from the top, so a rating can drop a weak pick from the edit (#909).
- **BREAKING:** the vision ratings of each prompt kind (with speech, frames only) are centred on their mean in the run, with a prior when fewer than 5 are rated, in place of the fixed 1.9 taken off a rating with speech. A rated moment's score stays between those of its nearest unrated neighbours, so a rating never moves it past an unrated moment, except moments the default edit skipped above its cut when vision ran (too long or overlapping a pick); they are marked `vision_skipped`. Scores change when a project is analyzed again; `cutpoint events` on an existing project applies the clamp to its ratings as they are (#909).
- **BREAKING:** `POST /projects` without `style` creates an `auto` project (it was `gaming`).
- **BREAKING:** Vision no longer renames moments or clips; its description is kept in `vision_label` (#913). Re-analyze a project to get the original labels back.
- **BREAKING:** the action-cam vision tags are sport-neutral: `powder_spray` is `spray`, `fast_groomer` is `fast_open`, `chairlift` is `transit` and `lodge` is `indoor_break`. Old keys are ignored, so an existing project's tags read 0 until you run `cutpoint events` again.
- **BREAKING:** the timing key `ski_s` is `action_cam_s`, and `skiing_s` in `telemetry.json` is `riding_s`.
- The visual detector finds people only (the skis and snowboard classes are dropped), and the AI Chat action words include biking, surfing, skating and trail riding.
- **BREAKING:** crash detection without a model no longer saturates: a profile without normal-footage captions (`incident.neg`, such as the no-model fallback) takes those of `generic`, so its visual and dense crash scores, and the confidences in `crashes.json`, drop from the old 1.0 visual score. Crash dense caches are rebuilt once (#932).
- `crashes` with a discovered profile never scores a frame below `--activity generic`: the visual and dense checks take the larger of the profile's crash score and that of `generic` (#932).
- Discovered activity profiles are cached per footage (the files, the model, `--context` and the prompts): the same footage in a new project or run gets the same profile without a new synthesis, and an identical discovered profile reuses its library file instead of adding a `<name>-<hash>` copy. `--refresh-profile` also ignores that cache and replaces it (with `--context`, or while the footage matches no library profile; delete a matching one with `cutpoint profile delete NAME` to start over) (#932).
- The full scene detection pass of `analyze` is about 1/3 faster on machines with an NVIDIA GPU when nothing else uses the CPU: NVDEC and the CPU decode the two parts of an 8-bit H.264 source (MP4, MOV, M4V or MKV) of 2 to 30 minutes at once, with nearly the same scene cuts (a rare cut differs where the CPU and NVDEC frames differ slightly). It does not run while the background HUD OCR reads a source or Whisper runs on the CPU, and needs at least 6 logical CPUs. `CUTPOINT_SCENE_DECODE` (`auto`, `gpu`, `cpu`, `hybrid`) chooses the decode; `gpu` keeps the single decode of before (#730). See [docs/configuration.md](docs/configuration.md).

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
