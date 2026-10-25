# Command reference

Run `cutpoint <command> --help` for the full option list of your installed version. This page
explains what each command does and its most useful options. In the examples, `cutpoint` means
`uv run cutpoint` (or the installed tool).

A project argument that is a bare name (`my-edit`) or starts with `projects/` is placed in the
projects folder (see [Configuration](configuration.md#projects_dir)). Absolute paths, and
relative paths that exist in the current folder, are used as given.

Global option: `-v` / `--verbose` / `--debug` (before the command) turns on debug logging and
prints tracebacks of errors.

| Command | Purpose |
|---|---|
| [`ui`](#ui) | Open a local browser editing workspace for standard video |
| [`capabilities`](#capabilities) | Report GPU vendors, working encoders, transcription, torch and SDK support |
| [`edit`](#edit) | Analyze, export clips and assemble in one command |
| [`analyze`](#analyze) | Find and rank highlight candidates |
| [`events`](#events) | Detect sounds, HUD events and describe windows with a model |
| [`clips`](#clips) | Export moments as numbered clips |
| [`extend`](#extend) | Re-export a clips folder with more padding |
| [`assemble`](#assemble) | Join clips into one video |
| [`stage`](#stage) | Copy slow-storage sources to a local disk |
| [`crashes`](#crashes), [`mix`](#mix), [`refmatch`](#refmatch), [`fit`](#fit) | Action-cam module |
| [`360 ...`](workflows/360.md) | Insta360 X3 module |
| [`setup`](#setup), [`paths`](#paths), [`guide`](#guide), [`skill`](#skill) | Setup, data, the agent guide and skill |

## JSON output

`--json` makes a command write its result as exactly one JSON object on stdout. It is available
on `paths`, `analyze`, `clips`, `assemble`, `edit`, `extend`, `capabilities`, `mix`, `stage`,
`setup`, `crashes`, `profile` and every `360` subcommand (`scan`, `analyze`, `status`,
`director`, `stitch`, `check`, `qa`, `repair`). Not yet on `events` and the other commands. Logs and errors stay on stderr, and
output of child processes such as ffmpeg goes to stderr too, so stdout can be parsed as is. Exit
codes are unchanged.

```json
{"ok": true, "command": "clips", "schema": 1, "project": "D:\\cutpoint\\projects\\clip",
 "outputs": {"clips_dir": "D:\\cutpoint\\projects\\clip\\clips", "count": 8, "...": "..."},
 "warnings": [{"code": "kept_clips", "message": "Kept 005_x.mp4 in ...", "files": ["..."]}],
 "error": null}
```

| Key | Content |
|---|---|
| `ok` | `true` unless the command failed |
| `command` | The command as typed, such as `analyze` or `360 qa` |
| `schema` | `1` |
| `project` | Absolute path of the project folder, or `null` when the command has none (`paths`, `capabilities`, `setup`, `stage`, and `360 status` without a project) |
| `outputs` | Object: the files written (absolute path strings) and key facts. The keys per command are below |
| `warnings` | List of `{"code", "message", "files"}`; `files` is a list of absolute paths and may be empty |
| `error` | `null`, or `{"code", "message"}`. `message` is the `ERROR` line printed on stderr |

With a failure, `ok` is `false` and `outputs` holds whatever was recorded before it. An
argument error (`bad_args` from the parser) also writes an envelope, then exits 2. A command that
checks an option itself (a bad `--crop`, an odd or non-positive `--size`, `--clip-override`, `--ids`, `mix --quotas`,
`360 director --out-name`, `stage` with no inputs, an unknown `--activity`) reports `bad_args` and exits 1.

| `error.code` | Exit | Cause |
|---|---|---|
| `bad_args` | 2 (1 for the checks above) | The command line or an option value is invalid |
| `not_found` | 1 | An input, project or file is missing |
| `ffmpeg_missing` | 1 | ffmpeg or ffprobe is not found |
| `missing_extra` | 1 | An optional extra is not installed |
| `output_not_empty` | 1 | A file or folder is in the way; see `--force` |
| `interrupted` | 130 | Ctrl+C, Ctrl+Break or SIGTERM |
| `error` | 1 | Anything else |

| `warnings[].code` | Command | Meaning |
|---|---|---|
| `kept_clips` | `clips`, `extend` (so `edit`) | `.mp4` files the old manifest does not list stayed in the folder; `assemble` includes them |
| `kept_sources` | `clips` | A file the run reads as a source was not replaced |
| `skipped_clips` | `extend` | Clips without a manifest entry were not re-exported |
| `vision_unavailable` | `analyze` | `--vision` was passed but Ollama or the model is not available, so no vision rating ran |
| `torch_no_cuda`, `sdk_needs_nvidia` | `capabilities`, `360 stitch` (the second only) | The warnings of the report, see [GPU troubleshooting](troubleshooting/ffmpeg-gpu.md#torch-and-the-insta360-sdk-in-capabilities) |
| `exclusions_ignored` | `mix` | Exclusions whose source is relative to the current folder of a drive were ignored |
| `content_qa_failed` | `mix` | The content checks of the mix failed |
| `nothing_staged_under_input` | `stage --clear` | An input that no longer exists has no staged copy recorded |
| `events_extra_missing` | `setup` | The `pip` component found modules for `events` missing |
| `events_not_in_highlights` | `360 analyze` | A source was not reachable, so `highlights.json` lacks the events |
| `not_a_360_clip` | `360 director` | A selected moment is not a 360 clip and was skipped |
| `vl_check_unavailable` | `360 director` | `--vl-check` could not run or stopped answering |
| `stitch_failed` | `360 stitch` | A clip's stitch failed; `files` names its log |
| `qa_clips_skipped` | `360 qa` | Clips that could not be checked |
| `qa_clips_stale` | `360 repair` | QA clips with no matching segment, or newer than the QA |
| `footage_unreachable` | `crashes` | A source is not reachable (offline drive, moved file) and was skipped |
| `no_telemetry` | `crashes` | A source has no IMU telemetry; its crashes come from the transcript only |
| `no_visual_layer` | `crashes` | A source has no visual layer; run `events` first so what the camera sees counts |
| `highlights_not_updated` | `crashes` | The reanalyze step was skipped (footage not reachable), so `highlights.json` is unchanged |
| `context_not_applied` | `crashes`, `profile` | `--context` was not applied (Ollama or the LLM unavailable, or no usable profile from it) |

`outputs` keys per command (`project` as in the table above):

| Command | `outputs` |
|---|---|
| `paths` | One key per folder printed (`home`, `config`, `cache`, `models`, `tools`, `llama.cpp`, `stage`, `styles`, `presets`, `profiles`, `projects`, `bundled`) and `missing`, the keys whose path does not exist |
| `analyze` | `highlights`, `transcript`, `candidates` (count), `suggested_top` (when set), `style` |
| `clips` | `clips_dir`, `manifest`, `clips` (each `file`, `srt`, `label`, `source`, `start`, `end`, `score`), `count`, `total_seconds` |
| `extend` | As `clips`, and `seconds` (`{"start", "end"}`) |
| `assemble` | `final`, `fcpxml`, `otio`, `srt`, `duration`, `clip_count` |
| `edit` | `kind`, the outputs of `analyze`, `clips` and `assemble`, and `copy` and `copy_srt` (the `<name>-final` files; `copy_srt` is `null` if none was written). One envelope holds the warnings of all steps |
| `capabilities` | The report's own keys (as printed by `capabilities` without `--json`, including `torch` and `insta360_sdk`) |
| `mix` | `segments`, `report`, `sheet`, `moments` (each `id`, `source`, `start`, `end`, `duration`, `category`), `count`, `total_seconds`, `target_seconds`, `categories` (per category `n` and `seconds`), `content_qa_pass` |
| `stage` | `stage_dir`. Staging: `files`, `copied`, `skipped`, `bytes`, `read_s`, `mb_per_s`. `--list`: `files` (each `source`, `staged`, `bytes`, `read_s`). `--clear`: `removed` |
| `setup` | `components`: one key per component (all by default, or those given to `--only`): `files` (each `path`, `bytes`, `status` of `downloaded`, `found` or `missing`), and `version` for `llama`; for `pip`, `installed` and `missing` |
| `360 scan` | `pano`, `timings`, `clips` (each `id`, `mode`, `duration`, `frames`, `insv`, `insv_bytes`, `lrv`), `count`, `total_seconds` |
| `360 analyze` | `timings`, `views` (`cached`, `computed`); unless `--views-only`, also the outputs of `analyze` (`highlights`, `transcript`, `candidates`, `suggested_top`, `style`); unless `--views-only` or `--no-events`, also `events` |
| `360 status` | `sdk` (`exe`, `origin`, `found`, `runs`, `version`); with a project, `clips` (each `id`, `mode`, `duration`, `views`, `render`, `render_file`) |
| `360 director` | `paths_dir`, `paths` (each `path`, `preview`), `count`, `vl_check` |
| `360 stitch` | `clips` (each `clip`, `output`, `status`, `seconds`, `log_file`, `estimated_gb`), `dry_run`. Never the SDK's own output |
| `360 check` | `check_dir`, `sheets` |
| `360 qa` | `qa_dir`, `report`, `mode`, `sheets`, `clips` (each `file`, `score`, `pass`, `fails`), `count`, `passed`, `score` |
| `360 repair` | The repair report's own keys, and `exclude_file` (`null` if there are no exclusions) |
| `crashes` | `crashes` (the `crashes.json` path), `count`, `candidates` (strongest first, each with the `crashes.json` keys `id`, `source`, `t0` and `t1` (impact span, s), `confidence` (0-1), `confirmed` (dense check ran), `origin` (`telemetry` or `words`), `clip` (`start`, `end`, `until`, and `reaction` when planned) and `sheet` (when one was written)), `activity_file` (the `activity.json` path), `profile_name` |
| `profile` | The `activity.json` record's own keys (as the older `profile --json` printed them: `version`, `request`, `name`, `source` (`flag`, `library`, `synthesized`, `context` or `fallback`), `profile`, and when set `scores`, `description`, `described_by`, `model`), and `activity_file` (its path) |

`360 qa --json-only` skips the contact sheets; it is a different option from `--json`.

## ui

```text
cutpoint ui [--port 8765]
```

Requires the `ui` extra: from a clone, `uv run --extra ui cutpoint ui`; for a tool install,
`uv tool install "cutpoint[ui]" --reinstall`, then `cutpoint ui`. Include any other extras
you already use when installing or running.

Open the complete printed loopback URL, including `#t=…`; keep it private because it grants access
to local footage. The workspace can create/open projects, analyze standard video,
preview and trim candidates, save and reorder a sequence, export clips, and assemble a final
video with optional fades, loudness normalization and captions. Nothing is uploaded. It does not expose action-cam or 360 commands.

New Project's Source field takes a full path that you type or paste; it always accepts one.
"Choose video…" and "Choose folder…" open the operating system's own file or folder dialog on the
computer running `cutpoint ui` (Windows: the system dialog through tkinter; Linux, with a display
(DISPLAY or WAYLAND_DISPLAY): zenity, falling back to kdialog; macOS: osascript). Where no dialog is
available (headless, Docker, no display, or an SSH session on Windows and macOS), those buttons are
hidden and you type or paste the path.

The gear button in the header opens Settings, which sets the projects folder (saved as
`projects_dir` in `config.json`). The folder must exist, or only its last part may be missing, in
which case it is created; it must be writable. Existing projects are not moved. The setting is
read-only when `CUTPOINT_PROJECTS_DIR` is set, and it cannot be changed while a job is running.
Settings also has "Rate frames with a local vision model (Ollama)" (off by default; saved as
[`vision`](configuration.md#vision) in `config.json`): the default of the vision checkboxes below,
and of `analyze` and `edit` on the command line.
The UI uses `POST /api/pick` for the dialogs and `GET`/`PUT /api/settings` for Settings.

Before a project is analyzed, the Moments column offers "Analyze footage", with a "Rate frames
with a local vision model (Ollama)" checkbox (it starts from the Settings default, which is off
unless you turn it on; ticked it passes `--vision`, unticked `--no-vision`), and "Show folder", which
opens the folder with the project's videos in the system file manager (hidden where that is not
possible, such as Docker). After analysis, "Show folder" stays available next to "Analyze again"
in the column header. "Analyze again" asks for confirmation in a dialog with the same vision
checkbox, ticked only if the previous analysis asked for vision (a project not analyzed yet:
the Settings default). "Show folder" also appears next to "Export clips" (the latest export
folder, `clips_ui_*`) and next to "Assemble" (the project folder, which holds the final videos and
their `.srt` subtitles and `.fcpxml` and `.otio` timelines), once that folder exists.

The sequence panel shows a thumbnail per clip and supports inline rename (pencil or double-click),
remove, multi-select with batch remove, up/down buttons and drag reorder. Thumbnails are cached as
JPEGs in `<project>/ui/frames/` (at most 500 per project, least recently used evicted). Drag the
splitter to resize the side column (arrow keys, Shift, Home/End; double-click resets); the width is
remembered per browser. If the port is busy, `cutpoint ui` exits 1 with an `ERROR` naming the port;
pass another `--port`.

Jobs run one at a time with logs and cancellation. Results remain the usual Cutpoint files;
review choices live under `<project>/ui/`. Exports use fresh `clips_ui_*` folders and finals use
fresh `final_ui_*` names, so earlier exports are not overwritten (a final name is also skipped
when its `.srt` exists). The outputs panel lists every final video in the project folder, newest
first with the export's own final first; until a new export is assembled, the preview of an
earlier final is titled "Earlier final". A cancelled job shows "stopping" (status `cancelling`
in `/api/jobs`) until its process exits; Cancel is disabled meanwhile. Cancel returns at once;
the job has about 12 s to remove its partial files before its process tree is ended (preview
proxies: 5 s). Assembly uses the latest successful UI export, not any subsequent unexported sequence changes. Failed/cancelled exports
may leave partial output folders, which are not offered for assembly.

For codecs the browser cannot preview directly, selecting a source starts a local H.264/AAC proxy
under `<project>/ui/proxy/`. Multiple audio tracks are mixed for preview, matching clip export.
The preview transport has a mute button and a volume slider next to Play. One setting covers every
preview (moments, clips and final) and is remembered per browser; the default is unmuted at 100%.
`M` toggles mute, and Space plays or pauses even when the volume controls have focus.
Original footage and export quality are unchanged. The server binds only to `127.0.0.1`;
remote access is not supported.

## capabilities

```text
cutpoint capabilities [--json]
```

Reports GPU vendors present separately from usable video encoders, the selected encoder,
faster-whisper's transcription device, and the backend order `setup` would use for llama.cpp.
It also reports torch (installed, CUDA or CPU-only build, whether it reaches the GPU; with the
reinstall command when it is CPU-only on a machine with a usable NVIDIA driver) and the Insta360
Media SDK (found or not, and a warning when no NVIDIA GPU is found). `360 stitch` logs that
warning before it runs. The JSON keys are `torch` and `insta360_sdk`; see
[GPU troubleshooting](troubleshooting/ffmpeg-gpu.md#torch-and-the-insta360-sdk-in-capabilities).
Encoder checks actually encode a short clip; an encoder listed by ffmpeg is not necessarily
usable. This command downloads nothing. `--json` writes the report, one key per section, in the
[result envelope](#json-output) under `outputs`.

Use `video_encoder` in `config.json` or `CUTPOINT_VIDEO_ENCODER` to select `auto`, `nvenc`,
`amf`, `vaapi` or `cpu`. Auto falls back to software when no hardware probe succeeds, and redoes an
export once on the CPU if a GPU encode fails on real footage (one WARNING); an
explicit unavailable GPU encoder fails rather than silently selecting another backend.
AMD transcription remains CPU-only with the current faster-whisper runtime.
See [GPU troubleshooting](troubleshooting/ffmpeg-gpu.md) for drivers and device permissions.

## edit

```text
cutpoint edit <file|folder> [<file|folder> ...] [--kind KIND] [--style STYLE] [--out DIR]
              [--top N] [--max-total S] [--vertical] [--force] [--events]
              [--vision | --no-vision] [--vision-top N] [--ollama-url URL]
              [--ollama-model MODEL] [--json]
```

Runs [`analyze`](#analyze), [`clips`](#clips) and [`assemble`](#assemble) in one go with the
settings of a kind, then copies the final video and its subtitles to `<name>-final.mp4` and
`.srt` in the project folder (a hard link where the file system allows it). `<name>` is the
first input's name: a file's name without extension, or the folder's name. Nothing is written next to the source videos: a project folder that holds
one of them is refused.

| Kind | Style | Selection | Order | Transition | Other |
|---|---|---|---|---|---|
| `highlights` (default) | `action` | up to 180 s | chronological | fade | loudnorm |
| `montage` | `action` | up to 60 s | by score | fade | loudnorm |
| `clips` | `action` | top 10 | by score | cut | loudnorm; the clips are the deliverable |
| `funny-moments` | `funny` | top 8 | chronological | cut | loudnorm, burned-in captions |

| Option | Meaning |
|---|---|
| `--style`, `--top N`, `--max-total S` | Replace the kind's value; the others stay (`--kind montage --top 5`: the 5 best that fit in 60 s) |
| `--out DIR` | Project folder. Default: `<projects>\<name>` (the first input's name, as above) |
| `--vertical` | 9:16 clips in `clips_vertical`, `final_vertical.mp4`, `<name>-final_vertical.mp4` |
| `--force` | Replace the clips of an earlier run, as `clips --force`. Without it a non-empty clips folder stops the run before the analysis |
| `--events` | Also run [`events`](#events) with its defaults to re-rank the moments. Needs the `events` extra and `cutpoint setup`; checked before the analysis |
| `--vision` / `--no-vision` | As on `analyze`: rate top candidates with Ollama, or not. Default: [`vision`](configuration.md#vision) in `config.json`, else off |
| `--vision-top N`, `--ollama-url`, `--ollama-model` | As on [`analyze`](#analyze): candidates sent to vision, and the Ollama server and model |
| `--json` | One [result envelope](#json-output) on stdout for the whole run |

Re-runs reuse the cached analysis. `edit` exits like the other commands (`130` on Ctrl+C, partial
files removed) and logs the final video, its subtitles and the clips folder last. To change the
cut afterwards, delete or rename clips (rename a clip's `.srt` with it) and run `assemble`;
that updates `final.mp4` and `final.srt`, not the `<name>-final` copies.

## analyze

```text
cutpoint analyze <file|folder> [<file|folder> ...] --style STYLE [--out DIR] [--json] [options]
```

Transcribes the audio, scores it on a 0.1 s grid, and writes `highlights.json` and
`transcript.txt` to the project folder. Inputs may be any mix of files and folders; they are
de-duplicated and kept in the order given. Each source has its own cache, so adding one file to
an analysed set processes only the new file.

| Option | Meaning |
|---|---|
| `--style` | `gaming`, `action`, `sport`, `funny`, `commentary`, `ski`, or a learned preset |
| `--out DIR` | Project folder. Default: `<projects>\<name>`: a file's name without extension, or the folder's full name (`.` resolved; `project` for a drive root); for several inputs, their common parent folder's name |
| `--exclude GLOB` | Skip files whose name or path matches (case-insensitive, repeatable) |
| `--recursive` / `--no-recursive` | Search sub-folders of folder inputs (default: no). A folder whose full path matches `--exclude` (`*/Captures/*` or `*/Captures/`) is skipped completely |
| `--vision` / `--no-vision` | Rate top candidates with a local Ollama vision model, or not. Default: [`vision`](configuration.md#vision) in `config.json`, else off for every style, so Ollama is never needed unless you ask; `--no-vision` skips it when `config.json` turns it on. The model is unloaded from GPU memory right after the rating |
| `--vision-top N` | Candidates sent to vision. Default: one per 2 minutes of footage, between 10 and 60 |
| `--ollama-url`, `--ollama-model` | Defaults: `OLLAMA_HOST` / `ollama_url` / `http://127.0.0.1:11434`, and `qwen3-vl:8b` |
| `--model NAME` | Whisper model. Default `medium` on CUDA, `small` on CPU |
| `--language en` | Force the transcription language (default: auto-detect) |
| `--no-scenes` | Skip scene detection |
| `--max-candidates N` | Candidates per source before merging |
| `--recover` | After a read error, resume analysis past the bad region |
| `--must-include-crash CONF` | Action-cam: always clip crashes with at least this confidence, 0 to 1 (0 disables) |
| `--json` | Result as one [JSON envelope](#json-output) on stdout |

Signals combined: loudness (EBU R128 against the clip's own baseline), speech keywords from the
transcript, laughter, scene-cut density and speech continuity. Likely Whisper hallucinations
(a short phrase repeated, very low confidence) are dropped first. Boundaries are snapped to
scene cuts and never cut a word.

Each candidate in `highlights.json` has `id`, `source`, `start`, `end`, `score` (0-100),
`signals`, `label` and a `transcript` snippet. `transcript.txt` has one line per pause or
sentence: `[Round1.mp4 00:12:31.4-00:12:38.9] text`.

If ffmpeg hits read errors (a damaged file, a failing drive), `analyze` warns loudly, records
`read_errors` on that source in `highlights.json`, and stops analysing that source at the
failure time. `--recover` then probes past the bad region and continues from the first
readable point.

## events

```text
cutpoint events <project> [--layers audio voice visual crash ocr vlm] [--no-vlm] [options]
```

Optional; needs the `events` extra and `cutpoint setup`. Run it after `analyze`. It works in
layers, one GPU model at a time, and writes `events.json` and `events.txt`, then re-ranks
`highlights.json` (the original is kept as `highlights.pre-events.json`).

Footage that is not reachable (offline drive, moved file) is skipped with a `footage not reachable` warning; if none is reachable the command fails before writing anything. With a source skipped, `events.json` is rewritten without that source's events the ski re-rank is skipped, and other styles re-rank with the events found; rerun once the footage is back (caches make it quick).

| Layer | What |
|---|---|
| `audio` | PANNs sound events: laughter, screams, cheering, gunfire, explosions |
| `ocr` | Game HUD text: kill feeds, round banners (see [game streams](workflows/game-streams.md)) |
| `voice` | Wind-robust speech and laughter (Silero VAD + PANNs) |
| `visual` | CLIP frame embeddings, person detection, beauty and group scores |
| `crash` | Runs `cutpoint crashes` (action-cam projects) |
| `vlm` | Qwen2.5-Omni-7B through a managed llama-server describes candidate windows from frames and audio |

Default layers: `audio ocr vlm`; action-cam projects: `audio voice visual crash vlm`.

| Option | Meaning |
|---|---|
| `--layers`, `--no-vlm` | Choose layers. Layers left out still contribute cached results |
| `--force LAYER ...` | Recompute those layers |
| `--only GLOB` | Only sources whose file name matches |
| `--avoid "GLOB@START-END"` | Never decode this range (repeatable). Recorded read errors are always avoided |
| `--game KEY`, `--player TAG` | HUD preset (`auto` or a preset key) and your in-game name(s) |
| `--discover-regions` / `--no-discover` | Force or disable HUD region discovery |
| `--preset-scope {user,project}` | Where a discovered preset is saved |
| `--max-windows N`, `--window S`, `--pre S`, `--post S`, `--window-at "GLOB@START-END"` | Model window count and geometry |
| `--sheets` | Write contact sheets to `<project>\visual_check` |
| `--no-rescore` | Do not touch `highlights.json` |

Every layer is cached per source, so reruns are cheap.

## clips

```text
cutpoint clips <project> [--top N | --ids h001 h004 ... | --segments FILE | --mix] [options]
```

Exports moments as `001_<label>.mp4`, `002_<label>.mp4`, ... in playback order, plus
`manifest.json`. Cuts are re-encoded for frame accuracy with a working NVIDIA/AMD hardware
encoder, or libx264 when no hardware encoder passes its probe. Every clip gets the same
resolution, frame rate, pixel format and audio, so clips join losslessly. The command refuses
to write into a non-empty folder without `--force`. With `--force`, the new clips are exported
into a hidden `.<folder>.cutpoint-tmp` next to the folder and swapped in only when every clip
succeeded; a failed or cancelled run leaves the old clips as they were. `--force` replaces only
what an earlier run made: the clips listed in the old `manifest.json`, leftover `.part.mp4`
files, the names this run writes and `manifest.json`. A file the run reads as a source is never
replaced; if a new clip name is one, the run stops with an error before exporting. Every other
file stays, including `.mp4` files the old manifest does not list (renamed or added by hand); a
WARNING names them, since `assemble` includes every `.mp4` in the folder; move them out before
re-running `clips --force` if they are not part of the edit. If such a file has the name of a
new clip, the run stops with an error naming it before exporting; move or rename it. Without a
readable old manifest, every existing `.mp4` counts as such a file. Because the staging folder sits next to the output
folder, `clips --force` refuses before any export a non-empty output folder that is a drive
root, a mount point or a bind mount; use `--out` with a folder on that drive. An empty drive
root or mount point is exported into directly, each clip under a temporary name until it is
complete; a failed run there may keep the finished clips.

| Option | Meaning |
|---|---|
| `--top N` | Top N by score (default: the style's target length) |
| `--ids ...` | Explicit candidate ids |
| `--segments FILE` | Export exactly the listed segments in the listed order |
| `--max-total S` | Keep the best-scoring moments that fit in S seconds |
| `--order` | `chronological`, `score`, `story` (default; opens with the second-best moment, ends with the best) or `arc` |
| `--vertical` | Centre-crop to 9:16, 1080x1920. Default folder `clips_vertical` |
| `--size WxH` | Output resolution (even numbers). Ignored with `--vertical` |
| `--crop auto\|none\|W:H:X:Y` | Crop borders before scaling |
| `--smart-tail` / `--no-smart-tail`, `--tail-min`, `--tail-max` | Extend clips that end on speech until the reaction settles |
| `--event-tails` / `--no-event-tails`, `--event-window S` | Extend to cover a payoff event (needs `events.json`) |
| `--extend S`, `--extend-start S`, `--extend-end S` | Padding ("handles") for trimming in an editor |
| `--speech-aware` / `--no-speech-aware`, `--speech-max-extend`, `--speech-max-lead` | Move in/out points to utterance boundaries |
| `--audio-fade S` | Audio fade baked into each clip (default 0.25, 0 = off) |
| `--out DIR`, `--force` | Output folder; replace the clips an earlier run made in a non-empty folder |
| `--mix` | Action-cam: export `mix_segments.json` in arc order |
| `--pano-source`, `--pano-cmd-every` | 360 clips only |
| `--json` | Result as one [JSON envelope](#json-output) on stdout |

Segments file (`--segments`). Relative `source` paths resolve against the JSON file's folder
(on Windows, a drive-relative path such as `D:foo.mp4` is an error; give the full path):

```json
[
  {"source": "D:/Videos/stream.mp4", "start": 3782.9, "end": 3789.7, "label": "banter"},
  {"source": "D:/Videos/stream.mp4", "start": 4464.5, "end": 4507.0, "label": "fight", "tail": 2.5},
  {"source": "D:/Videos/stream.mp4", "start": 1866.0, "end": 1891.9, "label": "boxed", "crop": "1364:750:276:164"}
]
```

Optional per-segment keys:

| Key | Meaning |
|---|---|
| `tail` | Extend the end by exactly this many seconds (no detection) |
| `smart_tail` | `false` disables smart and event tails for the segment |
| `speech_aware` | `false` keeps both boundaries exactly as given |
| `crop` | `auto`, `none` or `W:H:X:Y`; overrides `--crop` |
| `extend`, `extend_start`, `extend_end` | Padding for this segment |
| `speech_max_extend`, `speech_max_lead` | Speech-aware caps for this segment |
| `min_start`, `max_end` | Source times the clip never extends past |

Notes:

- **Tails.** A clip that ends on speech is extended through laughter or a reaction, up to
  `--tail-max` after the last word. With `events.json` present, a payoff (laughter, a kill,
  a round banner, a model-rated funny or hype moment) starting within `--event-window` seconds
  after the speech extends the clip to it.
- **Speech-aware boundaries** run last, on the final in/out points, and on by default. Clips of the
  same source never overlap; overlapping padding is split in the longest pause between them. To
  keep an exact cut (for example to exclude a word at the boundary), set `"speech_aware": false`
  on the segment.
- **Cropping.** `--crop auto` runs ffmpeg `cropdetect` on 30 frames at each of 6 points and
  caches the result. It only finds black borders. A project can set a per-source crop in
  `<project>\project.json`: `{"crop": {"Stream*.mp4": "auto", "Round4*.mp4": "none"}}`.
  Precedence: segment `crop`, then `--crop`, then `project.json`, then none.
- **Mixed aspect ratios.** The output format follows the tallest source. When 16:9 and 8:7
  sources mix, pass `--size 1920x1080` and a `crop` on the 8:7 segments.
- `manifest.json` records each clip's source, start, end, score, label, tail, crop, padding and
  speech moves. `extend` and `assemble` read it.
- **Subtitles.** Each clip gets `001_<label>.srt` next to it, from the source's cached
  transcript (the one `analyze` made; no transcription runs), timed to the clip's padded start.
  A source without a cached transcript gets no `.srt` (one INFO line); a transcript without
  speech in the clip gives an empty one. The manifest records the file (`srt`), and `--force`
  replaces only the `.srt` files the old manifest lists.
- Each export is checked for silent truncation (an ffmpeg read error, or a clip more than
  0.3 s short). The read is retried once; a clip still short is re-encoded to end cleanly, with a
  `read_error` record in the manifest.

## extend

```text
cutpoint extend <clips_dir> [--seconds S] [--start S] [--end S] [--out DIR] [--reuse DIR] [--json] [options]
```

Re-exports an existing clips folder with more padding, for when you are trimming in an editor
and want more room. Only `.mp4` files still in the folder are re-exported, with their exact
names and order, so deletions and renames from review are kept.

```powershell
cutpoint extend projects/my-edit/clips --seconds 2          # projects/my-edit/clips_ext2s
cutpoint extend projects/my-edit/clips --start 1 --end 3    # clips_ext1-3s
cutpoint extend projects/my-edit/clips                      # speech boundaries only, clips_speech
```

- `--clip-override GLOB KEY=S` (repeatable) sets per-clip `speech_max_extend`, `speech_max_lead`,
  `extend`, `extend_start` or `extend_end` for matching file names, for example
  `--clip-override 005_* speech_max_extend=45`.
- `--json` writes the result as one [JSON envelope](#json-output) on stdout.
- `--reuse DIR` copies a clip from an earlier output instead of re-encoding when source, in/out
  points, crop, format and fade are identical.
- The original folder is never overwritten unless you pass `--out <clips_dir> --force`. Then
  the clips are swapped in only when all of them were re-exported: a failed or cancelled run,
  or a clip locked by another program, leaves the original clips as they were. The same holds for
  `--out DIR --force` into a non-empty `DIR`: `DIR` is unchanged unless every clip was exported.
  As with `clips --force`, only the clips and `.srt` files `DIR`'s `manifest.json` lists, and
  leftover `.part.mp4` files, are replaced, and listed ones this run does not re-export (deleted
  from the input folder during review) are removed with their `.srt`. Other files in `DIR` stay; a WARNING names `.mp4`
  clips left there, because `assemble` will include them. An `.mp4` in `DIR` that `DIR`'s
  `manifest.json` does not list and that has the name of a clip to re-export stops the run
  before any export, as with `clips --force`. `--out DIR --force` is refused
  before any export when a non-empty `DIR` is a drive root, a mount point or a bind mount, and
  so is in-place `extend` of such a clips folder; use `--out` with a folder on that drive.
- **Subtitles.** Each re-exported clip gets its `.srt` for the new range, as `clips` writes it,
  and the manifest records it (`srt`). An `.srt` the target folder's manifest lists for a
  re-exported clip is replaced (removed if the source has no cached transcript); one it does
  not list (made or renamed by hand) is never overwritten, and that clip gets no new `.srt`
  (one WARNING). The `.srt` it lists for a clip no longer in the folder (deleted during review)
  is removed, as with `clips --force`.

## assemble

```text
cutpoint assemble <clips_dir> [--out final.mp4] [--transition cut|fade] [--fade-dur 0.25]
                  [--audio-fade S] [--loudnorm] [--captions] [--model MODEL] [--json]
```

Joins the `.mp4` files currently in the folder, in natural filename order (`2_x` before
`10_x`).

- `cut` (default) uses stream copy when all clips share parameters, else normalises and
  re-encodes. `fade` crossfades video and audio over `--fade-dur` seconds (default 0.25,
  greater than 0), rounded to whole frames of the first clip's frame rate (0.25 s at 29.97 fps
  is 7 frames, 0.2336 s) so video, audio and `final.srt` use the same length: at least one
  frame and at most 45% of the shortest clip. When one frame is more than that, the clips are
  joined with cuts (WARNING).
- `--loudnorm` runs two-pass EBU R128 normalisation toward -14 LUFS with a true-peak limiter,
  then measures the result and redoes the pass with a lower ceiling if the true peak is above
  -1.5 dBTP. Very peaky material can land up to about 1 LU below target.
- `--captions` transcribes the result and burns in bold lower-third captions in 2-4 word
  chunks. The ASS file is kept next to the output. `--model` picks the Whisper model.
- `--json` writes the result as one [JSON envelope](#json-output) on stdout.
- Default output: `<project>\final.mp4`; `final_vertical.mp4` for `clips_vertical`;
  `final_<name>.mp4` for other folders. A `.fcpxml` and a `.otio` timeline are always written
  next to it. With `--transition fade` (and the `edit` kinds that crossfade) they hold a
  cross-dissolve at each join, starting on the edit, with the clips and total length of
  `final.mp4`; with cuts they are hard cuts. On the OTIO audio track a clip without audio
  leaves a gap, so later audio stays in sync.
- Subtitles `final.srt` (named after the output) are written next to it from the clips'
  cached source transcripts, placed on the joined timeline (crossfades included) by each
  clip's `manifest.json` start and end. A clip renamed during review has no manifest entry;
  its own `.srt` (renamed with it) is used instead. Clips with neither, or whose source has no
  cached transcript, get none; one INFO line names them, and with none at all no `.srt` is
  written.

## stage

```text
cutpoint stage [inputs ...] [--exclude GLOB] [--recursive] [--list] [--clear] [--json]
```

Copies sources on slow or network storage to a local folder once; every later ffmpeg or ffprobe
read of those files uses the copy. Projects, caches, manifests and timelines keep the original
path. A copy that would leave less than 5 GB free is refused. `--list` shows what is staged,
`--clear` deletes the copies (all, or those `stage` would copy for the given inputs with the same
`--recursive` and `--exclude`). Each input is cleared on its own: one that no longer exists, even
through `source_remap`, clears the copies recorded under its path (the file itself, or the files
directly inside the folder, at any depth with `--recursive`, minus `--exclude` matches); one with
none recorded gives a warning and the other inputs are still cleared. A running `cutpoint ui`
uses new copies, and a changed `stage_dir` or `source_remap`, without a restart. See
[slow storage](troubleshooting/storage.md).

## crashes

```text
cutpoint crashes <project> [--dense | --no-dense] [--dense-top N] [--sheets N] [--lead S]
                 [--max-after S] [--max-clip S] [--force] [--reanalyze | --no-reanalyze]
                 [--activity NAME] [--context TEXT] [--refresh-profile] [--json]
```

Action-cam module (extras `ski` and `events`). Finds crashes and wipeouts in an analysed
project from full-rate IMU data, visual, audio and transcript evidence, and writes
`crashes.json` plus contact sheets in `crash_check\`. Confident crashes are always exported by
`clips`. The visual prompts and crash words come from the project's activity profile (see
`profile`; the three profile options are the same). See
[action-cam workflow](workflows/action-cam.md). `--json` writes the result as one
[JSON envelope](#json-output) on stdout.

Footage that is not reachable (offline drive, moved file) is skipped with a `footage not reachable` warning; if none is reachable the command fails before writing anything. With a source skipped, the default reanalyze step is skipped and `highlights.json` is left unchanged; rerun once the footage is back.

## profile

```text
cutpoint profile <project> [--activity NAME] [--context TEXT] [--refresh-profile] [--json]
```

Picks or builds the project's activity profile (what a crash looks and sounds like in this
footage) and writes `activity.json` in the project. `crashes` and the `crash` layer of `events`
reuse it.

| Option | Meaning |
|---|---|
| `--activity NAME` | Use a library profile: `ski`, `snowboard`, `mtb`, `moto`, `skate`, `surf`, `hike`, `generic`, or one learned earlier. Default: detected from the footage |
| `--context TEXT` | What the video is, in the user's words. The local LLM turns it into extra crash captions and slang on top of the profile. A changed text rebuilds the profile |
| `--refresh-profile` | Ignore `activity.json` and detect or build the profile again |
| `--json` | One [result envelope](#json-output) on stdout; `outputs` holds the `activity.json` record (`name`, `source` (`flag`, `library`, `synthesized`, `context` or `fallback`), `scores`, `description`, `profile`) and `activity_file`. Output of the local model never reaches stdout |

Needs the visual layer of `events` for detection, and Ollama with a Qwen model
(`CUTPOINT_VISION_MODEL`) for new profiles and `--context`. Without them it falls back to the
nearest library profile or `generic`, and tries again on the next run.

## mix

```text
cutpoint mix <project> [--target S] [--quotas CAT=SHARE,...] [--out FILE] [--exclude-file FILE] [--json]
```

Action-cam module. Picks a varied set of moments (runs, jumps, crashes with reactions, lift
rides, views, social) under per-category quotas, and writes `mix_segments.json`,
`mix_report.json` and `mix_sheet.jpg`. Default target 240 s. `--json` writes the result as one
[JSON envelope](#json-output) on stdout.

If nothing is selected (no candidate fits `--target`, or every candidate is excluded), `mix`
fails with `ERROR Nothing selected in <project>: ...`, exit code 1, and writes no files. Raise
`--target` or loosen the exclusions.

## refmatch

```text
cutpoint refmatch <project> <reference_video> [--name NAME] [--no-fit] [--taste] [--check N]
```

Action-cam module. Finds the shots of a finished edit in the project's sources and fits style
weights to what you kept, saved as a learned style in `<CUTPOINT_HOME>\styles\<name>.json`.
It assumes the edit keeps the source framing; mirrored, heavily zoomed or picture-in-picture
shots will not match.

## fit

```text
cutpoint fit <project> [<project> ...] [--name NAME] [--taste | --no-taste] [--ablate] [--no-save]
```

Fits one learned style on several projects that already have `refmatch.json` (run `refmatch
--no-fit` on each first), with held-out cross-validation. `--ablate` refits on feature subsets;
`--no-save` reports only.

## setup

```text
cutpoint setup [--only pip panns llama omni visual voice] [--force] [--json]
```

Downloads the models and tools for `events`. On Windows and Linux, auto selection prefers CUDA
when a usable NVIDIA driver is detected, otherwise Vulkan, then CPU, depending on available
release assets. Windows CUDA uses the CUDA 12.4 distribution; Linux uses the oldest CUDA
distribution with cudart in the newest matching release. Set `CUTPOINT_LLAMA_BACKEND` to
`auto`, `cuda`, `vulkan` or `cpu` to control selection; an explicit backend does not silently
fall back. `llama-server` is found via `CUTPOINT_LLAMA_SERVER`, then the bundled build, then
`PATH`. `--json` writes the result as one [JSON envelope](#json-output) on stdout. See [Getting started](getting-started.md#download-models-only-for-cutpoint-events).

## paths

```text
cutpoint paths [--json]
```

Prints the resolved data folders: `CUTPOINT_HOME`, config, cache, models, tools, stage, styles,
activity profiles and projects. `--json` writes them as one [JSON envelope](#json-output).

## guide

```text
cutpoint guide
```

Prints the guide for AI agents driving Cutpoint ([AGENTS.md](../AGENTS.md)) to stdout. Use it
when Cutpoint is installed from PyPI and there is no repository to read.

## skill

```text
cutpoint skill install [--dir DIR] [--force]
cutpoint skill print
```

`skill install` copies the Claude Code skill for agents driving Cutpoint to
`<DIR>/cutpoint/SKILL.md`. The default `DIR` is `~/.claude/skills`, the Claude Code user skills
folder (`$CLAUDE_CONFIG_DIR/skills` when `CLAUDE_CONFIG_DIR` is set); use `--dir` for a project's `.claude/skills` or another agent's skills folder. An
installed skill with the same content is left as is. One with different content (an older
version or your own edits) is an error naming the file; `--force` replaces it. The skill is short
and points the agent to `cutpoint guide` for details.

`skill print` prints the skill to stdout, for agents that do not use a skills folder.
