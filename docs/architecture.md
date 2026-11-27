# Architecture

For contributors. The package is `src/cutpoint/`; `cli.py` is the entry point for `cutpoint`.

## Pipeline

```text
sources --> analyze --> highlights.json, transcript.txt
                |
                +--> events (optional) --> events.json, events.txt, re-ranked highlights.json
                |
           clips  --> clips/NNN_label.mp4 + manifest.json
                |
           extend (optional) --> clips_ext<S>s/
                |
          assemble --> final.mp4, or final.fcpxml / final.otio (--format); + final.srt

clips --vertical --> clips_vertical/  --assemble--> final_vertical.mp4
assemble <project> --> .assemble-tmp-<id>/ (temporary clips, removed) --> final.mp4
edit = analyze + clips + assemble (--no-keep-clips: the temporary path above)
```

Each stage reads the previous stage's files in the project folder, so any stage can be re-run
or replaced by hand-edited input (`clips --segments`). Per-source results are cached in
`<CUTPOINT_HOME>\cache`, keyed by path, size and modification time. [Pipeline](pipeline.md) has
a detailed diagram of each analyze step, the background OCR and the `edit --kind` presets.

## Modules

| Module | Role |
|---|---|
| `cli.py`, `__main__.py` | argparse entry point and error handling (exit codes 0, 1, 2, 130); `python -m cutpoint` |
| `edit.py` | `cutpoint edit`: analyze, clips and assemble in one run |
| `kinds.py` | `edit --kind` presets (`KINDS`), import-free for the parser |
| `jsonout.py`, `progress.py` | `--json` result envelope; `CUTPOINT_PROGRESS` lines |
| `capabilities.py` | `cutpoint capabilities` report |
| `skill/` | `cutpoint skill` and the packaged `SKILL.md` |
| `ui/` (package: `app.py` builds the app; `routes/` holds the API routers by group; helpers in submodules), `web/` | Loopback browser workspace; static ES modules, project/review API, scoped media/proxies, and isolated CLI subprocess jobs |
| `cache.py`, `state.py`, `utils.py` | JSON cache files and atomic writes, `retry_locked` for files another program holds; project state; shared helpers |
| `errors.py` | Shared exceptions: `FFmpegError`, `FFmpegNotFoundError`, `EncoderUnavailableError`, `FolderBusyError` |
| `procs.py` | Child processes: isolated start, low priority, soft stop and tree kill, Windows job object so children end with Cutpoint, liveness check |
| `folders.py` | Output folders: temporary part files, atomic outputs, staged swap of a clips folder with recovery, source safety checks, folder locks |
| `paths.py`, `config.py` | `CUTPOINT_HOME` and every data location; `config.json`, projects folder, source remap |
| `projectfiles.py` | Names of the files Cutpoint writes into a project folder, and the `highlights.json` loader; stdlib only at import |
| `projectdelete.py` | The own files and folders of a project, for Settings > Delete this project in `cutpoint ui`; nothing the user added is removed |
| `runstate.py` | `progress.json` and the project run lock of the long project commands; read by `resume` |
| `intervals.py` | Time intervals `(start, end)`: merge, overlap (half-open) and the free stretch around a time; stdlib only |
| `segments.py` | One rule set for a review's segment list, shared by the review UI and the AI Chat ops |
| `memguard.py`, `winvol.py` | Memory guard: a model loads only when its memory is free; Windows drive and volume queries |
| `hub.py`, `imageio.py` | One copy of the Hugging Face Hub warnings on stderr; picture files read and written through bytes (non-ASCII paths on Windows) |
| `analyze.py`, `sources.py`, `scenes.py`, `vision_step.py`, `crash_moments.py`, `reanalyze.py` | Signals, peak picking, merging, snapping, `highlights.json`; cached audio and transcripts per source; scene cuts (`scenes.py`); the vision step of the ranking (`vision_step.py`); crash moments outside action-cam (`crash_moments.py`); re-analysis and the candidate ids it keeps (`reanalyze.py`) |
| `features.py`, `scorer.py`, `dsp.py`, `audio_tracks.py` | Named scoring features and the weighted scorer; signal processing; audio track selection |
| `vision_record.py` | The vision rating's record on a candidate: the blend into its score, and whether the rating (and the default edit's skip mark) still holds for its span; stdlib only, so `scorer.py` and the event layers import it without a cycle through `analyze.py` |
| `pool_record.py` | The visual pool's mark on a candidate (`POOL`, `is_pool`); stdlib only, so `scorer.py` and the event scoring import it without a cycle through `actioncam.py` |
| `default_fit.py`, `repeats.py`, `selfcheck.py`, `visual_pool.py` | The default moment selection that fits a style's `target_total`; repeat damping of keys a source says or shows often; self-checks of a finished analysis (`settings.self_checks`); visual candidates (`v001`...) for `action-cam`, `action` and `sport` |
| `category.py` | Footage category detection: `--style auto` and `cutpoint detect` |
| `rerank.py` | `cutpoint rerank`: re-rank an analysed project with another style from its caches |
| `tsrepair.py`, `envelope.py` | Joined `.ts` and `.mpg` recordings whose timestamps jump (read through a stream copy in the cache); audio peak envelope for the UI's trim rail |
| `cacheprune.py` | `cutpoint cache --prune`: remove source cache folders not used for a number of days |
| `srt.py` | Subtitles of clips and finals |
| `activity.py` | Activity profiles (`cutpoint profile`, `activity.json`) |
| `transcribe.py`, `vision.py`, `backends.py` | faster-whisper wrapper; Ollama client; Ollama URL handling |
| `assistant/` | Backend of the `cutpoint ui` assistant: `providers.py` (Ollama, OpenAI-compatible, Anthropic, xAI and the `claude-code`, `codex` and `antigravity` subscription providers), `keys.py` (API keys), `cliproxy.py`, `_proxy_guardian.py`, `proxy_credentials.py`, `proxy_tls.py`, `private_files.py` (the `cutpoint proxy` daemon, its supervisor, keyring credentials and recovery journal, loopback TLS identity, private files), `session.py` (turns), `prompting.py`, `toolcalls.py`, `proposing.py` (a turn's prompt, read tools and proposal checks), `context.py`, `ranking.py` (the ranking preview tool), `ops.py`, `edges.py`, `intent.py`, `pacing.py`, `topup.py` (clip ends, message intent, mix pacing, filling a short proposal), `lookfor.py`, `watch.py`, `words.py` (the `look_for`, `view_moment` and `find_words` tools), `errtext.py` (error text without folders), `store.py` (`ui/assistant/chat.json`, `history.json`); the page side is the AI Chat tab in `web/chat.js` ([AI Chat](commands.md#ai-chat)) |
| `styles.py` | Built-in styles and learned styles |
| `stylefiles.py` | The user's own style files in the styles folder (`cutpoint style`) |
| `library.py` | Shared rules of the profile and style libraries: name rule, reserved names, case-clash check, finding and deleting library files, "name or file" targets (used by `activity.py` and `stylefiles.py`) |
| `planning.py`, `clips.py`, `encode.py` | Moment selection and padding; ordering, export and manifest; format choice and frame-accurate encode; `scripts/moment_eval.py` is the ranking benchmark against the owner's labels (measure every ranking change with it); `scripts/bench.py` runs `analyze --style auto` on labelled footage of every category and reports detection and ranking per bucket, A vs B ([Benchmark](workflows/benchmark.md)) |
| `clipresume.py` | Finished clips of an interrupted `clips` or `extend` export, taken over by the next run |
| `color.py` | `--color auto` / `underwater`: per-clip measurement, grade estimate, numpy simulation and veto, per-source cache; `scripts/color_eval.py` is its development harness |
| `tail.py`, `boundaries.py`, `crop.py` | Smart tails, speech-aware in/out points, cropping |
| `extend.py` | Re-export a clips folder with padding |
| `assemble.py`, `timeline.py` | Concat or crossfade, loudness, captions; FCPXML 1.10 and OTIO writers |
| `stage.py` | Local copies of slow sources; staged-read lookup used by ffmpeg calls |
| `ffmpeg_utils.py` | ffmpeg discovery, probing, loudness, encoder choice, read-error detection |
| `telemetry.py`, `actioncam.py`, `crash.py`, `mix.py` | Action-cam module; `scripts/refmatch.py` is the action-cam scorer's development harness (fits learned styles to a reference edit; your own styles come from `cutpoint style`, `stylefiles.py`) |
| `insta360.py` | Insta360 X3 file names and the `_00_.insv` IMU trailer, below `pano/` so the core reads X3 telemetry without importing it |
| `extras.py` | Maps missing optional modules to the extra and install command |
| `events/` | `cutpoint events`: layers for audio (`audio_events.py`), HUD OCR (`killfeed.py` entry point, frame reading and segments; `ocr_engine.py` OCR backend and shared lanes; `ocr_read.py` region reads to events; `feed_parser.py` kill feed rows and player tags; `tracker.py` event tracker and kill, banner, replay and streak rules; `score_readout.py` score readouts; `menu_rules.py` menu-screen rules; `regions.py`, `discover.py`, `discover_run.py` (discovery of one OCR layer), `game_fit.py` (HUD fit check), `hud_map.py` and `feed_labels.py` (vision HUD map and region labels), `menu_spans.py`, `text.py`, `ocr_prefetch.py`, `ocr_raw.py`), voice (`voice.py`), dialogue reader (`dialogue.py`), confirmed explosions (`explosions.py`), visual (`visual.py`), model windows (`vlm_windows.py`, through the shared vision client in `vision.py`), `pipeline.py` to run them, `layers.py` (layer names, import-free for the parser), `scoring.py` and `tails.py` to feed results back, `report.py` and `sheets.py` (`events.txt`, contact sheets), `common.py`, `eventtypes.py`, `paths.py`, `cli.py`; `setup.py` downloads models |
| `pano/` | `cutpoint 360`: `x3.py` (clip discovery), `geometry.py` (IMU levelling), `boxes.py` (box overlap), `views.py`, `director.py`, `render.py`, `stitch.py` (SDK wrapper), `qa.py`, `qa_compare.py`, `qa_horizon.py`, `qa_paths.py`, `vlcheck.py`, `candidates.py` (camera paths of the candidates), `repair.py`, `project.py` (`pano.json` and the `pano` config section), `cli.py`. Not optional code: `telemetry.py`, `analyze.py`, `mix.py` and `actioncam.py` import `pano.x3` or `pano.views`; only the SDK stitch is optional |

See [360 director](workflows/360-director.md) for the 360 design.

## Conventions

- Optional heavy dependencies (torch, open_clip, RapidOCR) are imported lazily inside functions.
- Source videos are read-only. Outputs go to the project folder.
- Long GPU work runs in layers so only one model is loaded at a time.
- The browser workspace shares the pipeline's result files. It stores review choices and export
  history under `<project>/ui/`, and runs one heavy CLI job at a time rather than keeping models
  inside the HTTP server: no models stay loaded while idle (the AI Chat's visual search loads the
  CLIP text model there and drops it after 10 minutes without a search). Each UI export gets a
  fresh destination.
- Tests are CPU-only and do not need models or footage. See [CONTRIBUTING.md](../CONTRIBUTING.md).
