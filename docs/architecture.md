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
          assemble --> final.mp4 + final.fcpxml + final.otio
```

Each stage reads the previous stage's files in the project folder, so any stage can be re-run
or replaced by hand-edited input (`clips --segments`). Per-source results are cached in
`<CUTPOINT_HOME>\cache`, keyed by path, size and modification time. [Pipeline](pipeline.md) has
a detailed diagram of each analyze step, the background OCR and the `edit --kind` presets.

## Modules

| Module | Role |
|---|---|
| `cli.py` | argparse entry point and error handling (exit codes 0, 1, 130) |
| `ui.py`, `web/` | Optional loopback browser workspace; static ES modules, project/review API, scoped media/proxies, and isolated CLI subprocess jobs |
| `paths.py`, `config.py` | `CUTPOINT_HOME` and every data location; `config.json`, projects folder, source remap |
| `analyze.py`, `sources.py` | Signals, peak picking, merging, snapping, `highlights.json`; cached audio and transcripts per source |
| `transcribe.py`, `vision.py`, `backends.py` | faster-whisper wrapper; Ollama client; Ollama URL handling |
| `assistant/` | Backend of the `cutpoint ui` assistant: `providers.py` (Ollama), `session.py` (turns), `context.py`, `ops.py`, `store.py` (`ui/assistant/chat.json`, `history.json`); no page UI yet |
| `styles.py` | Built-in styles and learned styles |
| `planning.py`, `clips.py`, `encode.py` | Moment selection and padding; ordering, export and manifest; format choice and frame-accurate encode |
| `tail.py`, `boundaries.py`, `crop.py` | Smart tails, speech-aware in/out points, cropping |
| `extend.py` | Re-export a clips folder with padding |
| `assemble.py`, `timeline.py` | Concat or crossfade, loudness, captions; FCPXML 1.10 and OTIO writers |
| `stage.py` | Local copies of slow sources; staged-read lookup used by ffmpeg calls |
| `ffmpeg_utils.py` | ffmpeg discovery, probing, loudness, encoder choice, read-error detection |
| `telemetry.py`, `ski.py`, `crash.py`, `refmatch.py`, `mix.py` | Action-cam module |
| `extras.py` | Maps missing optional modules to the extra and install command |
| `events/` | `cutpoint events`: layers for audio (`audio_events.py`), HUD OCR (`killfeed.py`, `regions.py`, `discover.py`), voice (`voice.py`), visual (`visual.py`), model windows (`vlm_windows.py`, `llama_server.py`), `pipeline.py` to run them, `scoring.py` to feed results back; `setup.py` downloads models; `presets/` is the bundled HUD presets folder (no game presets ship) |
| `pano/` | `cutpoint 360`: `x3.py` (clip discovery), `geometry.py` (IMU levelling), `views.py`, `director.py`, `render.py`, `stitch.py` (SDK wrapper), `qa.py`, `repair.py`, `project.py` (`pano.json` and the `pano` config section) |

See [360 director](workflows/360-director.md) for the 360 design.

## Conventions

- Optional heavy dependencies (torch, open_clip, RapidOCR) are imported lazily inside functions.
- Source videos are read-only. Outputs go to the project folder.
- Long GPU work runs in layers so only one model is loaded at a time.
- The browser workspace shares the pipeline's result files. It stores review choices and export
  history under `<project>/ui/`, and runs one heavy CLI job at a time rather than keeping models
  inside the HTTP server. Each UI export gets a fresh destination.
- Tests are CPU-only and do not need models or footage. See [CONTRIBUTING.md](../CONTRIBUTING.md).
