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
`setup`, `events`, `crashes`, `profile` and every `360` subcommand (`scan`, `analyze`, `status`,
`director`, `stitch`, `check`, `qa`, `repair`). Not yet on the other commands. Logs and errors stay on stderr, and
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
checks an option itself (a bad `--crop`, an odd or non-positive `--size`, `--clip-override`, `--ids`, a `--lut` file that is not `.cube`/`.3dl`, unreadable or that ffmpeg cannot load, `--lut-strength` without `--lut`, `--color` or `--color-strength` with `--lut`, `--color-strength` when no `--color` correction runs, `mix --quotas`,
`360 director --out-name`, `stage` with no inputs, an unknown `--activity`) reports `bad_args` and exits 1.

| `error.code` | Exit | Cause |
|---|---|---|
| `bad_args` | 2 (1 for the checks above) | The command line or an option value is invalid |
| `not_found` | 1 | An input, project or file is missing |
| `ffmpeg_missing` | 1 | ffmpeg or ffprobe is not found |
| `missing_extra` | 1 | An optional extra is not installed (the feature is not available in Cutpoint lite) |
| `output_not_empty` | 1 | A file or folder is in the way; see `--force` |
| `folder_busy` | 1 | Another cutpoint run holds the folder's lock (`.<folder>.cutpoint-lock`); wait and retry |
| `interrupted` | 130 | Ctrl+C, Ctrl+Break or SIGTERM |
| `error` | 1 | Anything else |

| `warnings[].code` | Command | Meaning |
|---|---|---|
| `kept_clips` | `clips`, `extend` (so `edit`) | `.mp4` files the old manifest does not list stayed in the folder (the message names clips renamed during review); `assemble` includes them |
| `kept_sources` | `clips`, `extend` | A file the run reads, or the clips were cut from, was not replaced |
| `skipped_clips` | `extend` | Files without a manifest entry were not re-exported; the message says why for each (a source, renamed during review, or added by hand) |
| `vision_unavailable` | `analyze` | Vision rating was requested (`--vision` or `"vision": true` in `config.json`) but Ollama or the model is not available; if some of the top candidates have cached ratings those are used (the message gives the counts), otherwise no vision rating ran |
| `grade_dropped` | `clips` (so `edit`) | `--force` replaced clips whose manifest recorded a grade (`lut` or `color`) and this run grades none; give `--lut` or `--color` again to keep it |
| `color_on_gaming` | `clips` (so `edit`) | `--color` was given for `gaming` footage: game graphics are graded too (`underwater` is made for footage shot underwater; `--color auto` or `--lut` fits games better) |
| `step_failed` | `analyze` (so `edit`) | An analysis step (reactions, keyframes or crashes) failed with an `ERROR` line; the run went on without it |
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
| `footage_unreachable` | `crashes`, `events` (so `360 analyze`) | A source is not reachable (offline drive, moved file) and was skipped |
| `no_telemetry` | `crashes`, `events` (crash layer) | A source has no IMU telemetry; its crashes come from the transcript only |
| `no_visual_layer` | `crashes`, `events` (crash layer) | A source has no visual layer; run `events` first so what the camera sees counts |
| `highlights_not_updated` | `crashes`, `events` | The reanalyze step (`events`: the ski re-rank) was skipped (a source could not be found at its recorded or remapped path, or was not reachable), so `highlights.json` is unchanged |
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
| `capabilities` | The report's own keys (as printed by `capabilities` without `--json`, including `edition`, `torch` and `insta360_sdk`) |
| `mix` | `segments`, `report`, `sheet`, `moments` (each `id`, `source`, `start`, `end`, `duration`, `category`), `count`, `total_seconds`, `target_seconds`, `categories` (per category `n` and `seconds`), `content_qa_pass` |
| `stage` | `stage_dir`. Staging: `files`, `copied`, `skipped`, `bytes`, `read_s`, `mb_per_s`. `--list`: `files` (each `source`, `staged`, `bytes`, `read_s`). `--clear`: `removed` |
| `setup` | `components`: one key per component (all by default, or those given to `--only`): `files` (each `path`, `bytes`, `status` of `downloaded`, `found` or `missing`), and `version` for `llama`; for `pip`, `installed` and `missing` |
| `360 scan` | `pano`, `timings`, `clips` (each `id`, `mode`, `duration`, `frames`, `insv`, `insv_bytes`, `lrv`), `count`, `total_seconds` |
| `360 analyze` | `timings`, `views` (`cached`, `computed`); with `--scan`, also those of `360 scan`; unless `--views-only`, also the outputs of `analyze` (`highlights`, `transcript`, `candidates`, `suggested_top`, `style`); unless `--views-only` or `--no-events`, also `events` |
| `360 status` | `sdk` (`exe`, `origin`, `found`, `runs`, `version`); with a project, `clips` (each `id`, `mode`, `duration`, `views`, `render`, `render_file`) |
| `360 director` | `paths_dir`, `paths` (each `path`, `preview`), `count`, `vl_check` |
| `360 stitch` | `clips` (each `clip`, `output`, `status`, `seconds`, `log_file`, `estimated_gb`), `dry_run`. Never the SDK's own output |
| `360 check` | `check_dir`, `sheets` |
| `360 qa` | `qa_dir`, `report`, `mode`, `sheets`, `clips` (each `file`, `score`, `pass`, `fails`), `count`, `passed`, `score` |
| `360 repair` | The repair report's own keys, and `exclude_file` (`null` if there are no exclusions) |
| `crashes` | `crashes` (the `crashes.json` path), `count`, `candidates` (strongest first, each with the `crashes.json` keys `id`, `source`, `t0` and `t1` (impact span, s), `confidence` (0-1), `confirmed` (dense check ran), `origin` (`telemetry` or `words`), `clip` (`start`, `end`, `until`, and `reaction` when planned) and `sheet` (when one was written)), `activity_file` (the `activity.json` path), `profile_name` |
| `events` | `events` and `events_txt` (the `events.json` and `events.txt` paths), `layers` (the layers in `events.json`), `count` (events), `windows` (model windows), `highlights` (its path once re-ranked, `null` with `--no-rescore` or when the ski re-rank was skipped), `highlights_backup` (`highlights.pre-events.json`, when this run created it); when written, `crashes` (the `crashes.json` path, from the `crash` layer), `sheets` (contact sheet paths, `--sheets`) and `discovery` (per game `proposal`, `overlay`, `preset`, `installed` and `suggested`, the number of extra notice regions listed under `suggested_regions` in `proposal.json` for a game that kept its preset, else 0; `preset` is `built-in` when the kept preset is marked `"builtin": true`, in the user or bundled presets folder). A ski re-rank re-runs `analyze`, so its outputs (`transcript`, `candidates`, `suggested_top`, `style`) are added. A missing `events` extra (torch and the other layer modules) is `missing_extra`; missing model files or llama-server are `not_found` and name the `cutpoint setup` command |
| `profile` | The `activity.json` record's own keys (as the older `profile --json` printed them: `version`, `request`, `name`, `source` (`flag`, `library`, `synthesized`, `context` or `fallback`), `profile`, and when set `scores`, `description`, `described_by`, `model`), and `activity_file` (its path) |

`360 qa --json-only` skips the contact sheets; it is a different option from `--json`.

## Progress output

With `CUTPOINT_PROGRESS=1` in the environment, `analyze`, `edit` and `360 analyze` print one line per state change on stderr:

```text
PROGRESS {"step":"reactions","index":2,"total":6,"label":"Reactions","state":"running","pct":40,"detail":null}
```

| Field | Content |
|---|---|
| `step` | Step name, from the table below. A stable API |
| `index`, `total` | Position from 1, and the number of steps of the command |
| `label` | Display name |
| `state` | `running`, `done`, `skipped` or `failed` |
| `pct` | 0 to 100, or `null` when there is no cheap measure |
| `detail` | Short text (a skip reason, an error) or `null` |

Running updates come at most 2 per second. Steps run in table order. The `pct` of `transcribe` covers each source's audio extract (its first 10%) and then Whisper.

| `step` | `analyze` | `edit` |
|---|---|---|
| `transcribe` | 1 | 1 |
| `reactions` | 2 | 2 |
| `keyframes` | 3 | 3 |
| `crashes` | 4 | 4 |
| `ranking` | 5 | 5 |
| `vision` | 6 | 6 |
| `clips` | | 7 |
| `assemble` | | 8 |

`360 analyze` has its own 4 steps: `scan` (Read 360 clips; skipped without `--scan`), `views`
(360 views), `analyze` (Transcription and ranking) and `events` (Events and re-rank; skipped with
`--no-events`). `scan` and `views` report `pct` and the clip as `detail`.

On Ctrl+C the current step ends `failed` with detail `cancelled`. After a cancel or a fatal
error, steps after the failed one are not emitted. A first-time HUD discovery shows
`first run for <game>: discovering HUD regions` as the detail of `keyframes`.

## ui

```text
cutpoint ui [--port 8765]
```

Part of every install, lite included: run `cutpoint ui` (`uv run cutpoint ui` from a clone).

Open the complete printed loopback URL, including `#t=…`; keep it private because it grants access
to local footage. The workspace can create/open projects, analyze standard video,
preview and trim candidates, save and reorder a sequence, export clips, and assemble a final
video with optional fades (video and audio set separately), loudness normalization, [peak compression](#peak-compression) and
captions. Nothing is uploaded. A camera
folder or file of Insta360 X3 footage makes a 360 project, analyzed with `360 analyze --scan`
(see [360 video](workflows/360.md#in-the-ui)). It does not expose the action-cam commands or the
other `360` commands.

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
[`vision`](configuration.md#vision) in `config.json`): every analysis the UI starts, a first
one or "Analyze again", uses it (passed as `--vision` or `--no-vision`; 360 analyses never rate frames), and it is the default of
`analyze` and `edit` on the command line. The UI has no per-run vision checkbox. "Analyze again"
keeps the Ollama model, URL and vision-top the last analysis recorded.
"Stats for Nerds" (off by default) shows the open moment's score, rank, analyzed times, signals
and keywords over the top-left corner of the preview. It shows as soon as you tick it; Save keeps
it, in the browser (local storage), not in `config.json`, and Cancel or Escape undoes it.
Theme picks the colours: Blue (the default, today's colours), Light or Dark (both keep text,
controls and focus outlines at WCAG AA contrast). Dark has lighter panels; Light uses the same
yellow for buttons and fills as the other themes, with darker amber for text, focus rings, the selected tab, checkboxes and sliders. It recolours the UI at once and, like Stats for
Nerds, Save keeps it in the browser and Cancel or Escape undoes it.
"Show advanced export options" (off by default) shows the Export tab's advanced controls (marked
"advanced" below). Hidden, they keep their values and are still sent with each job, and
"Advanced options in use: ..." under the options names any that are; a hidden value out of range
shows them so the browser can point at it. Like Stats for Nerds, it shows at once, Save keeps it
in the browser, and Cancel or Escape undoes it.
The UI uses `POST /api/pick` for the dialogs and `GET`/`PUT /api/settings` for Settings.

Before a project is analyzed, the Moments column offers "Analyze footage" and "Show folder", which
opens the folder with the project's videos in the system file manager (hidden where that is not
possible, such as Docker). After analysis, "Show folder" stays available next to "Analyze again"
in the column header. "Analyze again" asks for confirmation first. The Export tab has one "Show Exports
Folder" button: it opens the latest export's `clips_ui_*` folder, or the project folder (final
videos, `.srt`, `.fcpxml`, `.otio`) once that export is assembled.

After analysis, the Moments list has a search field (label, words spoken, file name or candidate id such as `h001`) and a
funnel button beside it that opens Sort and filter. Sort by is Score (the default; rank #1
first) or Position in video (by source name in natural order, then start time); each card's
rank badge keeps its score rank either way. The quick filters are Sequence (All, In sequence,
Not in sequence), Ready only (the moment's preview plays now) and Source file when
the project has several sources. They combine with the search text. A dot on the button shows
while a filter or a sort other than Score is on, and when nothing passes the list reads "No
moments match these filters." Escape closes the menu. The sort and filters are remembered per
browser (local storage); the source file only while the project stays open.
Each card shows the rank and title with Add or Remove on the right, the candidate id, the times
and length, the file name when the project has several sources, a strip of the moment's place in its source, the
transcript, and the preview readiness (Ready, Encoding, Queued) at the bottom right.

While an analysis runs, the Moments column lists its steps (Transcription, Reactions, Keyframes,
Crashes, Ranking, Vision rating; for a 360 project those of `360 analyze`) under "Analyze footage", or under the column header for "Analyze
again": each with its state (waiting, running with a progress bar and a percentage when the step
reports one, done, skipped with the reason, failed with the error). A step still running when
the job fails reads "interrupted", one stopped by Cancel reads "cancelled", and the steps that
never started read "not run". The list stays after a failed or cancelled analysis
until another job runs, and gives way to the moments once the analysis completes. The UI asks for
the steps by starting its jobs with `CUTPOINT_PROGRESS=1`; `GET /api/jobs` returns them per job as
`progress` (`total` and `steps`, ordered by step; for a 360 analysis also `plan`, the steps the
job will report as `step` and `label`). A Cutpoint version that sends no steps shows
the log in Jobs only, as before.

The sequence column has three tabs: Sequence, AI Chat (see [AI Chat](#ai-chat) below) and Export.
The Export tab holds one "Export &
Assembly Options" panel; each option has a hover description, and the choices are remembered per
browser:

| Option | Effect | Default |
|---|---|---|
| Vertical 9:16 | Exports and assembles vertical clips | off |
| Burn in captions | `assemble --captions` | off |
| Speech-aware cuts | `--speech-aware` | on |
| Smart tails | `--smart-tail` | on |
| Normalize loudness | `--loudnorm` | on |
| Link Transitions | On: the audio transition follows the video (Cut gives Cut, Fade gives Crossfade) and the audio select is disabled | on |
| Video transition | Cut or Fade (`--transition`) | Cut |
| Audio transition | Cut or Crossfade (`--audio-transition`); Crossfade is unavailable under a video Cut | follows the video |
| Compress Audio Peaks | None, Light or Max (`off`, `light`, `strong`) | Light |
| Color | Off, Auto or Underwater (`--color`), in Export clips and One-Click Edit; Extend export keeps each clip's grade | `color` in `config.json`, else Off; Off for gaming projects |
| Strength | 0-100% of the Color correction (`--color-strength`, left out at 100%); shown with Auto or Underwater | 100% |

The padding field is gone from the UI; `clips --extend` and [`extend`](#extend) still work. A saved
"Smart edges" choice from an earlier version carries over to both Speech-aware cuts and Smart tails.

The Export tab has more controls above and below that panel. A blank Cut Timing field keeps the
CLI default.

**Clips to Export.** By default the export is the Sequence, as hand-picked. Automatic picks
export from the ranked moments instead:

| Control | CLI flag |
|---|---|
| Top moments (advanced) | `--top` |
| Max video length: the cap on all clips together | `--max-total` |
| Candidate ids (advanced; replaces Top moments): the ids on the moment cards, such as `h001` | `--ids` |
| Order: Style, Story, Best first or Chronological | `--order` |

**Cut Timing.**

| Control | CLI flag |
|---|---|
| Max end move (advanced) | `--speech-max-extend` |
| Max start move (advanced) | `--speech-max-lead` |
| Tail min / Tail max (advanced) | `--tail-min` / `--tail-max` |
| Event tails: Auto, On or Off | (Auto: no flag) / `--event-tails` / `--no-event-tails` |
| Event window (advanced) | `--event-window` |
| Extend start / Extend end (advanced) | `--extend-start` / `--extend-end` |

The "Latest export ..." facts under the options are advanced too; warnings and guidance there
(such as "Exported from the 360 preview (not stitched)") always show.

**Extend export** (advanced) runs [`extend`](#extend) on the latest export into a new `clips_ui_<stamp>`
folder. The result becomes the current export. It is disabled while Extend start and Extend end
are blank or 0 and Speech-aware cuts is off (extend would have nothing to do); the tooltip says so.

**One-Click Edit** has the presets Highlights, Montage, Clips and Funny moments. Make runs
`cutpoint edit --kind <preset>` in the project's own style, with the Vertical and Compress Audio
Peaks choices and the vision choice from Settings. Clips go to `clips` and the video to
`final.mp4` (`clips_vertical` and `final_vertical.mp4` when vertical), with the `<name>-final`
copy next to it; that export becomes the current one. If earlier clips exist,
a second run asks before replacing them and passes `--force` only when you confirm. The presets
are off, with the reason shown, for 360 projects and for projects analyzed with `--exclude` or
`--recursive`, `--no-scenes`, a non-default whisper `--model`, `--language`, `--max-candidates`
or `--recover`.

"Analyze again" keeps a recorded `--recover`.

API: the clips job accepts the new fields, `POST /api/projects/<id>/jobs` takes the new `extend` and `edit` actions,
and the project view reports `preset_problem` (why the presets are off) and `preset_taken` (the
earlier clips a second Make would replace).

The sequence panel shows a thumbnail per clip and supports inline rename (pencil or double-click),
remove, multi-select with batch remove, up/down buttons and drag reorder. Thumbnails are cached as
JPEGs in `<project>/ui/frames/` (at most 1500 per project, least recently used evicted). Drag the
splitter to resize the side column (arrow keys, Shift, Home/End; double-click resets); the width is
remembered per browser. A second splitter between the Moments column and the preview works the same
way (Shift for bigger steps); its width is remembered per browser too, stays between 260 and 640 px
and at most half the window, and never leaves the preview narrower than 360 px. If the port is busy, `cutpoint ui` exits 1 with an `ERROR` naming the port;
pass another `--port`.

Undo and Redo in the Sequence header (Ctrl+Z; Ctrl+Y or Ctrl+Shift+Z) step back and forward
through sequence edits: add, remove, reorder, rename, trim (a run of nudges to one edge within a
second is one step), reset, and applied AI Chat changes. They keep the last 50 steps and are
cleared when another project opens. In a text field the keys edit the text instead.

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

The "Compress Audio Peaks" slider (None, Light or Max) is passed as
`--compress-peaks` to both the clips export and the assembly, so an export bakes it into its clips
and the assembly compresses only the clips exported without it. It starts on
[`compress_peaks`](configuration.md#compress_peaks) in `config.json` when that sets it, else on
Light (`GET /api/settings` returns `compress_peaks` and `compress_peaks_set`), and a job sent without it uses that value; other
values are refused with 400. Previews do not apply it.

The "Color" slider (Off, Auto or Underwater) and its Strength are sent as `color` and
`color_strength` (0-1) with the clips and edit jobs and passed as `--color` and
`--color-strength`. A job sent without `color` passes no flag, so the CLI applies
[`color`](configuration.md#color) in `config.json` itself (never to gaming footage). A project
opens with the choice this browser saved for that project (Color is remembered per project,
Strength for all), else `color` in `config.json` (`GET /api/settings` returns `color`), else Off;
a gaming project (the project view's `gaming`) opens at Off, and a grade
picked there is still sent (the CLI warns). An unknown `color`, a `color_strength` outside 0-1 or
without `auto` or `underwater`, and `color` together with `lut` are refused with 400. The UI has
no LUT option. Previews do not apply the grade.

For codecs the browser cannot preview directly, selecting a source starts a local H.264/AAC proxy
under `<project>/ui/proxy/`. Multiple audio tracks are mixed for preview, all of them (clip export leaves out tracks that
copy another one).
While that whole-source proxy builds, a short proxy of the selected moment (10 s before it to
20 s after it) is encoded first, so the moment can be previewed and trimmed within seconds.
Scrubbing or playing past it shows "Preparing this part of the source…" and prepares that part
the same way; playback prepares the next part about 20 s ahead, and when it still has to wait
there it plays on by itself once the part is ready (a pause or a scrub meanwhile cancels that).
The whole-source proxy takes over once it is ready. Moment and sequence rows say
whether their preview is ready, encoding or queued, the trim rail hatches the parts that cannot
play yet, and "Full preview 42%" shows the whole-source proxy's progress. Play sequence starts a
clip as soon as a short proxy covers all of it. HDR and BT.2020 sources are previewed converted
to SDR, as in exports.
The preview transport has a mute button and a volume slider next to Play. One setting covers every
preview (moments, clips and final) and is remembered per browser; the default is unmuted at 100%.
`M` toggles mute, and Space plays or pauses even when the volume controls have focus.
Playback and scrubbing cover the whole source: playback runs on past the clip end, and Play at
the end of the source starts the clip again. The trim rail under the video is a zoomable view of
the source with the clip (gold) and the analyzed range (dashed) as markers; it opens on the clip
plus half its length (at least 10 s) either side. Behind the markers, a filmstrip shows at most
one still per 78 px (one every 78 to 156 px): the keyframe at or before each point, so a still
can be a little early; zoomed in to under 1 s per still, exact stills instead. The stills load a
few at a time once the view stops moving, nearest the playhead first, and are cached with the
thumbnails; a still that fails leaves the rail's ticks showing, and after 3 failures a source
gets no filmstrip until the page reloads. With the browser's data saver on
(`prefers-reduced-data` or Save-Data), the rail keeps its ticks only. "Zoom out", "Zoom in" and
"Fit clip" change the view, the overview strip next to them shows the view in the whole source
and pans it (drag or arrow keys), and the wheel over the rail pans it (Ctrl+wheel zooms). The
rail follows the playhead during playback. On the rail, the arrow keys move the playhead by 0.5%
of the view (at least 0.05 s), and PageUp and PageDown by a tenth of the view. "Use playhead" (`[` and `]`) sets Clip Start or Clip End anywhere in the source, and the
rail widens to show it. The gold start and end edges on the rail are handles: drag them (a drag held at
the rail's edge pans a zoomed view), or Tab to one and move it with the arrow keys (0.1 s;
Shift+arrow 1 s), PageUp and PageDown (5 s), Home and End (as far as it can go). A handle keeps
the clip at least 0.1 s long and inside the source, and the preview seeks to it. Space still
plays and pauses while a handle has focus. Clips longer than 10 minutes get a note under the trim fields.
"Play sequence" in the Sequence header previews the cut without exporting: it plays each clip
from its current start to its end, in sequence order, from the sources (or their proxies), with
hard cuts. The clock shows the clip number and the time in the sequence, the playing row is
marked, and trims made meanwhile apply at once. Clicking a row jumps to that clip, Space pauses
and resumes, and Escape or "Stop" ends it; removing the playing clip also stops it. A clip whose
proxy is still being built is waited for, and one whose preview cannot be built is skipped.
Original footage and export quality are unchanged. The server binds only to `127.0.0.1`;
remote access is not supported.

### AI Chat

The AI Chat tab talks to the model set in the [`assistant`](configuration.md#assistant) section of
`config.json`, chosen under Settings > AI assistant as a Provider preset with a model; no base
URL is typed:

- **None (local Ollama)**, the default: the configured Ollama (`OLLAMA_HOST` or `ollama_url`).
  Choose a pulled model, or Other model… for any id; Default uses `qwen3.5:9b`, else the vision model.
- **Claude (Anthropic)**: `https://api.anthropic.com`; `claude-sonnet-5-5` (default) or
  `claude-opus-5-5`.
- **Codex (OpenAI)**: `https://api.openai.com/v1`; choose an OpenAI chat model
  (`*-codex` models need OpenAI's Responses API and are not supported).
- **Grok (xAI)**: `https://api.x.ai/v1` (OpenAI-compatible); choose an xAI model.

A base URL set in `config.json` that no preset has (LM Studio, vLLM, llama-server, OpenRouter, an
Ollama on another machine) shows as "Custom (from config.json)" and keeps working: its base URL
is read-only there, and its model can still be changed
([`assistant`](configuration.md#assistant)).

The Model dropdown offers the models the provider lists. Ollama
lists its pulled models (`/api/tags`); Anthropic, OpenAI and xAI list theirs (`/v1/models`, with
the key) only once the key is set and you agreed to that provider and host in the AI Chat tab
(see below); until then nothing is asked and the field says why. A preset other than the saved
provider uses only its own key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY` or its
keyring entry), never `CUTPOINT_ASSISTANT_API_KEY`, which may be meant for the saved host.
Models that do not chat here are left out by name: ids starting with `text-embedding-`, `tts-`,
`whisper-`, `dall-e-`, `omni-moderation-`, `text-moderation-`, `gpt-image-`, `sora-`,
`davinci-` or `babbage-`, or containing `-tts`, `-transcribe`, `-realtime`, `-audio`, `-image`,
`imagine` or `codex` (OpenAI and xAI). A list is
kept for 10 minutes; Refresh list asks again. The Model dropdown offers the listed models; when
the list cannot be loaded it offers the preset's own models (Claude's two, Ollama's `qwen3.5:9b`) and the saved model,
and "Other model…" shows a Model id field to type any other id.

Saving there checks that the provider answers (reachable, a model chosen and, at
`api.openai.com`, `api.x.ai` and `api.anthropic.com`, the key accepted) and says why when it
cannot. Another
host (OpenRouter, a LAN server) may list its models without a key, so its key is checked when you
send the first message. The API key is read from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`XAI_API_KEY` or `CUTPOINT_ASSISTANT_API_KEY` when `cutpoint ui` starts, else from the system
keyring: Settings has Save key and Clear key. `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY` and `XAI_API_KEY` go only to `api.anthropic.com`, `api.openai.com` and
`api.x.ai`; a key for another host (OpenRouter, a LAN
server) comes from `CUTPOINT_ASSISTANT_API_KEY` or is saved for that base URL, and goes only over
`https://` ([keys](configuration.md#assistant)). Settings only says "set (env)", "set
(keyring)" or "not set"; the page never receives a key. To install Ollama and its models, see
[getting started](getting-started.md#optional-ollama-for-frame-rating-and-ai-chat).

A provider whose host is not this machine (`localhost` or a loopback address; an Ollama on
another machine counts too) is hosted: before the first message the tab asks once, per provider
and host, to send the project's moment labels, transcript snippets, file names (not full paths),
times and your messages there. The answer is kept in `config.json` (`assistant.consent`); until
then nothing is sent. A hosted reply is not streamed, so Cancel takes effect when it arrives.

Its header shows the model and provider (local, or the hosted host), with a note
that a small local model can miss moments or misread a request. Type a request and press Enter
or Send (Shift+Enter starts a new line); the reply shows the steps the model took ("Searched
for …") and can be stopped with Cancel. The chat needs an analyzed project.

- Moments the model points to can be highlighted in the Moments list ("Highlight in list") or
  the list filtered by the model's search words.
- A proposed change opens in the Sequence tab as a preview of the new sequence: each row is
  marked Added, Moved (old to new position), Trimmed (old to new times) or Renamed, removed clips are
  listed as Removed, the moments are marked in the Moments list, and the trim rail shows the new
  start and end as dashed edges. Apply replaces the working sequence (Save keeps it, Undo takes
  it back); Discard drops the preview. Up to three alternative suggestions are previewed the same
  way. If the sequence changed since the proposal was made (a note says so; an applied
  proposal's button reads "Review again"), Apply makes the change on the current sequence and
  says so, or tells you to discard it and ask again when it no longer fits. While Apply waits,
  Discard is off and a new proposal from the chat is previewed after it.
- "Applied changes" lists the changes applied in this project; "Revert this change" previews a
  change that puts the sequence back as it was before, applied like any other. If the sequence
  changes after that preview, Apply previews the revert again on it instead.
- "Clear chat" deletes the conversation only; the sequence and the applied changes stay.

Nothing is applied without Apply. API: `GET /api/projects/<id>/assistant`, `POST .../turns` and
`GET .../turns/<turn>?after=<n>`, `POST .../turns/<turn>/cancel`,
`POST .../proposals/<proposal>/applied`, `DELETE .../chat`, `GET .../history` and
`POST .../history/<change>/revert` (which returns a `revert-<change>` proposal with a `base_key`
and writes nothing; applying it with another sequence or without that key is refused with 409).
`GET .../assistant` also returns `consent` (`needed`, `key`, `host`, and `missing_key`: how to add a missing API key, or null); `POST .../turns` answers
409 while `needed` is true. `POST /api/assistant/consent` with that `key` records the consent;
`POST /api/assistant/key` (`provider`, `key`) and `POST /api/assistant/key/clear` (`provider`)
save or remove a keyring key for the host of the saved base URL and return only where each key
comes from (`keys`) and the host it is for (`key_hosts`). `GET /api/assistant/models?provider=`
(`ollama`, `anthropic`, `openai`, `xai` or `custom`; `&refresh=1` skips the cache) returns
`state` (`ok`, `key_needed`, `consent_needed`, `unsupported` or `error`), `models` (`id`,
`name`) and `message`; never a key.

## capabilities

```text
cutpoint capabilities [--json]
```

Reports the edition first: `full`, or `lite` when torch and the rest of the `events` extra are
not installed. Then GPU vendors present separately from usable video encoders, the selected encoder,
faster-whisper's transcription device, and the backend order `setup` would use for llama.cpp.
It also reports torch (installed, CUDA or CPU-only build, whether it reaches the GPU; with the
reinstall command when it is CPU-only on a machine with a usable NVIDIA driver) and the Insta360
Media SDK (found or not, and a warning when no NVIDIA GPU is found). `360 stitch` logs that
warning before it runs. The JSON keys are `edition` (`"full"` or `"lite"`), `torch` and
`insta360_sdk`; see
[GPU troubleshooting](troubleshooting/ffmpeg-gpu.md#torch-and-the-insta360-sdk-in-capabilities).
Encoder checks actually encode a short clip; an encoder listed by ffmpeg is not necessarily
usable. This command downloads nothing. `--json` writes the report, one key per section, in the
[result envelope](#json-output) under `outputs`.

Use `video_encoder` in `config.json` or `CUTPOINT_VIDEO_ENCODER` to select `auto`, `nvenc`,
`amf`, `vaapi` or `cpu`. Auto falls back to software when no hardware probe succeeds, and redoes an
export once on the CPU if a GPU encode fails on real footage (one WARNING); an
explicit unavailable GPU encoder fails rather than silently selecting another backend.
`CUTPOINT_VIDEO_DECODER=cpu` keeps decoding on the CPU instead of NVDEC or VAAPI.
AMD transcription remains CPU-only with the current faster-whisper runtime.
See [GPU troubleshooting](troubleshooting/ffmpeg-gpu.md) for drivers and device permissions.

## edit

```text
cutpoint edit <file|folder> [<file|folder> ...] [--kind KIND] [--style STYLE] [--out DIR]
              [--top N] [--max-total S] [--vertical] [--force] [--events]
              [--vision | --no-vision] [--vision-top N] [--ollama-url URL]
              [--ollama-model MODEL] [--skip STEP [STEP ...]]
              [--compress-peaks off|light|strong] [--lut FILE [--lut-strength 0-1]]
              [--color off|auto|underwater [--color-strength 0-1]] [--json]
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
| `--skip STEP ...` | As on [`analyze`](#analyze): leave out `reactions`, `keyframes` and/or `crashes`. Give it after the input paths |
| `--compress-peaks off\|light\|strong` | [Peak compression](#peak-compression), baked into the clips (so the final has it too). Default: `compress_peaks` in `config.json`, else off |
| `--lut FILE`, `--lut-strength 0-1` | Grade the clips with a 3D LUT, as on [`clips`](#clips) (so the final has it too). Checked before the analysis |
| `--color off\|auto\|underwater`, `--color-strength 0-1` | [Automatic color correction](#clips) of the clips, as on `clips` (so the final has it too). Not with `--lut`; checked before the analysis |
| `--json` | One [result envelope](#json-output) on stdout for the whole run |

With `CUTPOINT_PROGRESS=1`, `edit` reports 8 [steps](#progress-output): the 6 of `analyze`, then `clips` and `assemble`. `--kind clips` still assembles. With `--events`, `keyframes` is reported again after `vision`, for the VLM layer.

Re-runs reuse the cached analysis. `edit` exits like the other commands (`130` on Ctrl+C, partial
files removed) and logs the final video, its subtitles and the clips folder last. To change the
cut afterwards, delete or rename clips (rename a clip's `.srt` with it) and run `assemble`;
that updates `final.mp4` and `final.srt`, not the `<name>-final` copies.

## analyze

```text
cutpoint analyze <file|folder> [<file|folder> ...] --style STYLE [--out DIR] [--json] [options]
```

Transcribes the audio, scores it on a 0.1 s grid, runs the analysis steps below, and writes
`highlights.json` and `transcript.txt` to the project folder. Inputs may be any mix of files and folders; they are
de-duplicated and kept in the order given. Each source has its own cache, so adding one file to
an analysed set processes only the new file.

| Option | Meaning |
|---|---|
| `--style` | `gaming`, `action`, `sport`, `funny`, `commentary`, `ski`, or a learned preset |
| `--out DIR` | Project folder. Default: `<projects>\<name>`: a file's name without extension, or the folder's full name (`.` resolved; `project` for a drive root); for several inputs, their common parent folder's name |
| `--exclude GLOB` | Skip files whose name or path matches (case-insensitive, repeatable) |
| `--recursive` / `--no-recursive` | Search sub-folders of folder inputs (default: no). A folder whose full path matches `--exclude` (`*/Captures/*` or `*/Captures/`) is skipped completely |
| `--vision` / `--no-vision` | Rate top candidates with a local Ollama vision model, or not. Default: [`vision`](configuration.md#vision) in `config.json`, else off for every style, so Ollama is never needed unless you ask; `--no-vision` skips it when `config.json` turns it on. The 0-10 rating scales a candidate's score (5 leaves it as is, 10 lifts it and 0 lowers it by a share set per style, about 20-45%, capped at 100); candidates not rated keep theirs, and the model's label names the clip only from a rating of 7. The model is unloaded from GPU memory right after the rating. Ratings are cached per source (`vision-ratings.json`), so a rerun asks Ollama only about candidates it has not rated and repeats the same scores. The cache also records the Ollama model's digest: a model pulled again under the same tag is rated again, and with Ollama down the cached ratings for the tag are used |
| `--vision-top N` | Candidates sent to vision. Default: one per 2 minutes of footage, between 10 and 60 |
| `--ollama-url`, `--ollama-model` | Defaults: `OLLAMA_HOST` / `ollama_url` / `http://127.0.0.1:11434`, and `qwen3-vl:8b` |
| `--model NAME` | Whisper model. Default `medium` on CUDA, `small` on CPU |
| `--language en` | Force the transcription language (default: auto-detect). A forced language is transcribed and cached separately (`transcript-<model>-<language>.json`), and `clips`, `extend` and `assemble` subtitle the project from that transcript |
| `--skip STEP ...` | Leave out `reactions`, `keyframes` and/or `crashes` (give it after the input paths). Default: [`analysis.skip`](configuration.md#analysis) in `config.json`, else none |
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
`signals`, `label` and a `transcript` snippet; `transcript_unsure` lists the snippet's words
Whisper heard with confidence below 0.5 as `[index, confidence]`, the index counting words in
`transcript.split(" ")` (omitted when none is unsure or the transcript has no confidences). With
`--vision`, a rated candidate also has the model's own label in `vision_label` (`label` takes it
only from a rating of 7).
`transcript.txt` has one line per pause or sentence: `[Round1.mp4 00:12:31.4-00:12:38.9] text`.

### Analysis steps

The steps run one after another, and only one model is loaded at a time. One exception: with a
CUDA GPU, the HUD OCR of `keyframes` starts in the background while `transcribe` and `reactions`
use the GPU, and `keyframes` takes its results in its turn. It runs only for `gaming` and `funny`
when Whisper runs on CUDA, every game has a HUD preset (discovery stays in its turn), a source
read without errors, and the GPU has at least 2 GB free and 8 GB in all (unknown counts as
enough). The results, `events.json`, `highlights.json`, the OCR caches and the progress steps
are the same as with it off, and so are the log lines and their order, apart from rare `-v`
debug lines of helper threads and a `resuming the OCR read` line when a background read is
handed back to its turn; only timings differ. Set
`CUTPOINT_ANALYZE_OVERLAP=0` to turn it off.

Experimental, off by default: `CUTPOINT_ANALYZE_SHARED_DECODE=split` also moves the scene
detection proxy of `transcribe` into the background, for a source of up to 30 minutes with one
video stream (no cover art) whose scene cuts are not cached yet. It makes the proxy in the same
GPU decode (CPU without one) as the background HUD OCR (one pass instead of two) and spools the OCR frames to
the source's cache folder, in files of about 256 MB removed as they are read. The spool grows
by as much as the OCR read lags behind the decode, up to 8 files (about 2 GB; 4 GB without the
cap on a 446 s 1080p clip): then the decode, and the proxy with it, waits for the OCR read to
catch up. It is skipped when the drive has less free space than that cap (or every OCR frame of
a shorter source) plus 1 GB. `0`, unset or any other value: as described above. When the
background proxy has not started by the time scene detection needs it, or fails, scene
detection decodes as usual; the scene cuts, `events.json`, `highlights.json` and the OCR
caches are the same as with it off. Each source then logs an INFO line with the scene
detection time, the wait for the background proxy and the spool peak, and one with the wait for
the background OCR, for measuring; a source with cover art or more than one video stream, or a
drive without room for the spool, logs why at once. When Cutpoint is killed or crashes during
the split, its ffmpeg stops at once on Windows and with ffmpeg 7 and newer; on Linux and macOS
with ffmpeg 6.x it finishes encoding the proxy first. Either way the spool and
`proxy.tmp.mp4` stay in the cache folder until the next `analyze` that reads the source's HUD
OCR in the background removes them.

| Step | What |
|---|---|
| `transcribe` | Whisper transcription, scoring and scene detection |
| `reactions` | PANNs sound events (laughter, screams, cheering). Also the voice layer when `crashes` will run |
| `keyframes` | Game HUD text (OCR). For `gaming` and `funny`, HUD region discovery runs when no preset matches the game. For `action`, `ski` and `sport`, the CLIP visual layer, only with a CUDA GPU |
| `crashes` | Action-cam footage only (telemetry found, or style `ski`, `action` or `sport`) |
| `ranking` | Re-ranks the candidates with the events, after leaving out those in HUD menu time ([game streams](workflows/game-streams.md)); `highlights.json`, `events.json`, `events.txt` and the other result files are written after the last step |
| `vision` | Ollama frame rating, only when vision is on |

Nothing new is required. Without the `events` extra or the model files, a step is skipped with an
`INFO` line and nothing is downloaded; `cutpoint setup` and the extra add them. A step that fails
logs an `ERROR`, adds a `step_failed` warning and the run continues without it. The llama-server
VLM layer is not part of `analyze`; it stays in [`events`](#events) and `edit --events`.

`analyze` also writes `events.json` and `events.txt` (and `crashes.json` for action footage).
`highlights.json` is written once, atomically, at the end: a cancelled run leaves the previous
file, and a cancelled first run leaves none. The other result files are written with it, just
before it, each atomically: `transcript.txt`, `events.json`, `events.txt`, `crashes.json`, the
`crash_check\` sheets, `activity.json` and, for `ski`, `telemetry.json` and `ski_features\`
(older sheets and features of the project are removed at the same time). A failed or cancelled
run, in any step, leaves all of them as they were, and a cancelled first run leaves no project
folder, unless HUD discovery wrote its proposal (`discovery\`, below). A run stopped after the transcription therefore has no `transcript.txt` yet; the rerun
reads the cached transcription. What stays on purpose: the per-source caches outside the
project (audio, transcript, scene cuts, event layers, dense crash checks), so a rerun resumes
where it stopped; a HUD preset found by discovery, installed in the presets folder at once,
with its proposal in the project's `discovery\` folder; and an activity profile written by the
local LLM to the profile library. `analyze` never writes `visual_check\`; only
[`events --sheets`](#events) does. The analysis score stays in each candidate's
`score_pre_events`, and `highlights.pre-events.json` is written only when the analysis was not
ranked with events yet.

If ffmpeg hits read errors (a damaged file, a failing drive), `analyze` warns loudly, records
`read_errors` on that source in `highlights.json`, and stops analysing that source at the
failure time. `--recover` then probes past the bad region and continues from the first
readable point.

#### Typical cost

Measured on 1080p gaming footage (RTX 3080 10 GB), rounded:

| Run | Total | Breakdown | GPU memory peak |
|---|---|---|---|
| GPU, one 7.4 min file, first run | about 8.5 min | `transcribe` 76 s, `reactions` 5 s, `keyframes` 7 min (about 4 min of it HUD discovery), `ranking` under 1 s | about 4 GB |
| GPU, 2 x 10 min files, first run | about 17 min | `transcribe` 2.7 min, `reactions` 26 s, `keyframes` 14 min (about 7 min of it discovery) | about 3.7 GB |
| Cached re-run | about 5 s | | |
| CPU only, one 10 min file, discovery done | about 5.7 min | `transcribe` 2 min, `reactions` 16 s, `keyframes` 3.4 min (OCR on the CPU) | |
| CPU only, 60 s action clip | 25 s | `keyframes` skipped with "no CUDA GPU (CLIP on the CPU: about an hour per hour)" | |

`keyframes` dominates first runs. HUD discovery runs once per game and fills the first half of the
`keyframes` bar ("discovering HUD: n/N frames"). The CLIP layer is skipped without a CUDA GPU because on the
CPU it would take about an hour per hour of footage.

To force a CPU run, set `CUDA_VISIBLE_DEVICES=-1`; an empty value does not hide the GPU on
Windows. No analysis cache key includes the device, so the `reactions`, `keyframes` and
`crashes` results are reused across devices (after a GPU run, a CPU run reuses them, and the
other way round). The transcript is cached per Whisper model (`--model`; the default is `medium`
on CUDA and `small` on CPU), so it is reused only when both runs use the same `--model`.

## events

```text
cutpoint events <project> [--layers audio voice visual crash ocr vlm] [--no-vlm] [options]
```

Optional; needs the `events` extra and `cutpoint setup`. [`analyze`](#analysis-steps) already runs the `audio`, `ocr`, `voice`, `visual` and `crash` layers it can; use `events` for the `vlm` layer, to force layers (`--force`) or to rerun them with other options. It works in
layers, one GPU model at a time, and writes `events.json` and `events.txt`, then re-ranks
`highlights.json` (the original is kept as `highlights.pre-events.json`). The re-rank drops or
trims candidates in HUD menu time, as `analyze` does ([game streams](workflows/game-streams.md)).

Footage that is not reachable (offline drive, moved file) is skipped with a `footage not reachable` warning; if none is reachable the command fails before writing anything. With a source skipped, `events.json` is rewritten without that source's events; the ski re-rank is skipped and other styles re-rank with the events found; rerun once the footage is back (caches make it quick).

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
| `--json` | One [result envelope](#json-output) on stdout for the whole run |

Every layer is cached per source, so reruns are cheap. The model devices in the `stats` of
`events.json` (`audio_device`, `voice.device`, `visual.device`) name where a model ran in this
run, so a layer that only read its cache leaves the key out.

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
readable old manifest, every existing `.mp4` counts as such a file. A clip renamed during
review is recognised by its size (manifest `size`) when its exported name is gone and no
other unlisted clip or missing entry has that size; the WARNING then names it as a clip cutpoint exported,
with its exported name, and it still stays. A `manifest.json` that cutpoint may not have
written (unreadable, another `generator`, or no `clips` list of entries with a `file`) is
never replaced: the run stops with an error before exporting; move or rename it. Because the staging folder sits next to the output
folder, `clips --force` refuses before any export a non-empty output folder that is a drive
root, a mount point or a bind mount; use `--out` with a folder on that drive. An empty drive
root or mount point is exported into directly, each clip under a temporary name until it is
complete; a failed run there may keep the finished clips. With `export_workers` above 1 these can include clips after the failed one, and some of them may have no `.srt`.

One run at a time works on a folder: `clips`, `extend` and `assemble` hold the lock file
`.<folder>.cutpoint-lock` next to it while they run, so a second run never removes the first
one's staging folder. A second run stops with an `ERROR` naming the process that holds the
lock. The file is removed when the run ends, also on Ctrl+C. One left by a run that was
killed (`taskkill /F`) is taken over once its process is gone; one from a run on another
computer (a network folder) is not, so delete it by hand if no run is active.

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
| `--compress-peaks off\|light\|strong` | [Peak compression](#peak-compression) baked into each clip, then a true-peak limiter (default: `compress_peaks` in `config.json`, else off) |
| `--lut FILE` | Grade every clip with a 3D LUT (`.cube` or `.3dl`), see **LUT grade** below. Default: no grade |
| `--lut-strength 0-1` | How much of the `--lut` grade to apply, blended with the ungraded picture (default 1; needs `--lut`) |
| `--color off\|auto\|underwater` | Automatic color correction per clip, see **Automatic color** below. Default: `color` in `config.json` (never for `gaming` footage), else off. Not with `--lut` |
| `--color-strength 0-1` | How much of the `--color` correction to apply (default 1); not with `--lut` |
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
- **Color.** Every clip from YUV footage (nearly all video) is BT.709 (limited range, tagged
  bt709), so clips from different sources join without a color shift; RGB footage (`gbr`) is
  exported as before. HDR (HLG, PQ) and BT.2020 footage is tone-mapped to SDR (needs `zscale`
  in the ffmpeg build; without it a WARNING says so and the clip keeps its colors). BT.601
  footage is converted to BT.709, with one INFO line per source: footage with a BT.601 matrix
  tag (`smpte170m`, `bt470bg` or `smpte240m`, such as NVIDIA DVR captures), and SD footage
  (below 720 lines and 1280 columns) without a matrix tag. HD footage without a matrix tag is
  taken as BT.709 and only tagged; its pixels do not change, nor do those of BT.709 footage.
  `360` renders do the same.
- **LUT grade.** `--lut FILE` applies a 3D LUT (`.cube` or `.3dl`; Cutpoint ships none) to
  every clip, `360` renders included, after the BT.709 conversion above: the frames go to RGB
  with the BT.709 matrix, through ffmpeg `lut3d` (tetrahedral) and back to BT.709 limited
  range. Use the LUT made for the footage's log or flat profile and for Rec.709 output (for
  example GoPro Flat or GP-Log to Rec.709); a LUT on footage it was not made for distorts the
  colors, and Cutpoint cannot tell. `--lut-strength` below 1 blends the graded picture over the
  ungraded one. The file is checked before any work: a missing file is `not_found`; another
  extension, a LUT ffmpeg cannot load or a strength outside 0-1 is `bad_args`. Each clip
  records the LUT's absolute path and strength (`lut`, `lut_strength`), and `extend` grades the
  clips it re-exports the same way (it stops before exporting if the LUT has moved); a
  `clips --force` re-run needs `--lut` again. Without it, `clips --force` logs a WARNING when it replaces clips whose manifest recorded a grade. `assemble` leaves graded clips as they are. On a
  1080p30 export from 4K GoPro footage the grade added about 0.2 s per second of clip.
- **Automatic color.** `--color auto` measures each clip on up to 8 keyframes of its inner
  80 %, through the same crop and BT.709 conversion as its export, and grades it with ffmpeg
  filters only: a levels stretch (black at most 0.06, white at least 0.88), a white balance
  from the near-neutral mid-tones (60 % of the cast, at most ±12 % per channel) and a vibrance
  boost for dull clips. `--color underwater` also borrows red from green, takes the black and
  white points per channel (stretching the red the water took) and lets the red gain reach
  +60 %; it is only used when asked for, and only grades clips that look underwater: mean red
  below 0.77 of the mean green and 0.85 of the mean blue, and green above 0.79 of the blue
  (`not_underwater` for the others; water bluer than that, such as open ocean, does not pass
  either). `--color-strength` below 1 makes every
  correction milder. A clip gets no grade, with the reason in its manifest entry, when it
  already looks good (`good`), is log or flat 10-bit footage (`log`: use `--lut` with the
  camera's LUT; an INFO line says so), is a `360` clip (`360`), or when a simulation on the
  measured frames predicts more clipping than the levels intend (`veto_clipped`; underwater,
  pixels whose red alone clips high, up to 3 % of the frame, do not count), a worse cast
  (`veto_cast`: the gray-world cast rises by more than 0.005 of the luma range, or to more than
  1.5 times its value and by more than 0.002), a picture over 25 % more colorful (`veto_m3`) or
  a washed-out one: with `auto`, one that loses more than 15 % of its colorfulness
  (`veto_m3_drop`); underwater, one whose median saturation falls below 0.63 of its value, such
  as flat sand turned grey (`veto_faded`), or when its frames could not be decoded
  (`no_frames`). Plain `auto` logs an INFO line naming clips that look underwater (by the same
  rule); it does not change them. `gaming` footage is graded
  only with an explicit `--color` (with a WARNING, a stronger one for `underwater`, and a
  `color_on_gaming` `--json` warning): `color` in `config.json` never applies to
  it. `--color` cannot be combined with `--lut`. Each clip records a `color` entry (method,
  version, strength, parameters, skip reason, metrics before and predicted), and `extend`
  grades its re-exports with those parameters without measuring again. Measurements are cached
  per source (`color/` in its cache folder).
- `manifest.json` records each clip's source, start, end, score, label, tail, crop, grade (`lut` or `color`), padding,
  speech moves and size in bytes (`size`), and `"generator": "cutpoint"`. `extend` and
  `assemble` read it. Manifests without `size` or `generator` (written by earlier versions)
  still work; their clips are matched by name only.
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
- A clip exported with `clips --lut` is graded again with the LUT and strength its manifest
  entry records; if that file is gone, the run stops before exporting (`not_found`). A clip
  exported with `clips --color` is graded with the parameters its `color` entry records, not
  measured again for the new range; an unreadable `color` entry, or one from a newer Cutpoint,
  stops the run before exporting.
- `--reuse DIR` copies a clip from an earlier output instead of re-encoding when source, in/out
  points, crop, grade (`lut` or `color` parameters), format and fade are identical.
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
  As with `clips --force`, a `manifest.json` in `DIR` that cutpoint may not have written stops
  the run before any export.
- An `.mp4` in the folder without a manifest entry is not re-exported; the WARNING says
  whether it is a source of the clips, a clip renamed during review (recognised by its size,
  naming its exported name), or added by hand.
- **Subtitles.** Each re-exported clip gets its `.srt` for the new range, as `clips` writes it,
  and the manifest records it (`srt`). An `.srt` the target folder's manifest lists for a
  re-exported clip is replaced (removed if the source has no cached transcript); one it does
  not list (made or renamed by hand) is never overwritten, and that clip gets no new `.srt`
  (one WARNING). The `.srt` it lists for a clip no longer in the folder (deleted during review)
  is removed, as with `clips --force`.

## assemble

```text
cutpoint assemble <clips_dir> [--out final.mp4] [--transition cut|fade] [--fade-dur 0.25]
                  [--audio-transition cut|crossfade] [--audio-fade S] [--loudnorm] [--compress-peaks off|light|strong]
                  [--captions] [--model MODEL] [--json]
```

Joins the `.mp4` files currently in the folder, in natural filename order (`2_x` before
`10_x`). Like `clips`, it first puts back or removes what an interrupted `clips --force` or
`extend` left next to the folder (`.<folder>.cutpoint-tmp`, `.<folder>.cutpoint-bak`), and it
holds the folder's [lock](#clips) while it runs. The audio fade and peak compression baked
into a clip renamed during review are read from its manifest entry, matched by size.

- `cut` (default) stream-copies the video when all clips share parameters, else normalises and
  re-encodes it. The audio is always re-encoded once (192k AAC), so `final.mp4`'s audio and
  container end with the video to within one audio frame, and the `.fcpxml` and `.otio` end
  where `final.mp4` ends. `fade` crossfades video and audio over `--fade-dur` seconds (default 0.25,
  greater than 0), rounded to whole frames of the first clip's frame rate (0.25 s at 29.97 fps
  is 7 frames, 0.2336 s) so video, audio and `final.srt` use the same length: at least one
  frame and at most 45% of the shortest clip. When one frame is more than that, the clips are
  joined with cuts (WARNING).
  A fade re-encodes only the video around each join and stream-copies the rest of every clip,
  from keyframe to keyframe. The audio goes through the same filters as a full re-encode, so
  `final.mp4` has the same number of frames at the same times, and the same audio, and
  `final.srt` and the timelines are unchanged. `clips` and `extend` force one keyframe 0.5 s
  from each end of every clip (about 5% larger clips), so a fade of up to 0.5 s (the default
  is 0.25 s) re-encodes little more than the fade itself; longer fades use the clips' own
  keyframes. Clips exported before that, AMF exports or clips added by hand use the keyframes
  they have, so more of them is re-encoded. The whole video is re-encoded as before, with one
  INFO line giving the reason, when the clips differ in codec, size, frame rate, colour tags
  or encoder settings, are not constant frame rate, or would need more than 60% re-encoded
  anyway.
  `CUTPOINT_FADE_COPY=0` always re-encodes the whole video.
- `--audio-transition cut|crossfade` sets the audio join separately. Default: it follows
  `--transition` (`cut` gives `cut`, `fade` gives `crossfade`). `--transition cut
  --audio-transition crossfade` is refused (`bad_args`, exit 2): a crossfade under a hard cut
  cannot stay in sync without source audio. `--transition fade --audio-transition cut` cuts the
  audio in the middle of the video dissolve; the `.fcpxml` then has no audio crossfade and
  leaves the audio cut point to the editor (probably the start of the dissolve), while the
  `.otio` matches `final.mp4` exactly.
- `--loudnorm` runs two-pass EBU R128 normalisation toward -14 LUFS with a true-peak limiter,
  then measures the result and redoes the pass with a lower ceiling if the true peak is above
  -1.5 dBTP. Very peaky material can land up to about 1 LU below target. The audio ends
  with the video, without padding.
- `--compress-peaks light|strong` applies [peak compression](#peak-compression) to the clips
  that were not exported with it (their `manifest.json` entry has no `compress_peaks`), in the
  join, before the edge fade. The final audio then always ends with the true-peak limiter
  and its -1.5 dBTP check, with or without `--loudnorm` (without it, the level stays where the
  speech was). Default: `compress_peaks` in `config.json`, else off.
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

### Peak compression

`--compress-peaks` (`clips`, `assemble`, `edit`; `compress_peaks` in `config.json`) pulls
sudden loud moments such as shouts, laughs into the mic, gunshots and mic bumps down toward
the speech level, and evens out loudness between clips. It is off by default, and with
`off` the output is unchanged.

1. Every clip is measured (momentary loudness). The typical level is the median of all the
   clips' audio, ignoring silence and quiet background.
2. One RMS compressor (attack 5 ms, release 200 ms) starts a fixed amount above that level
   for every clip (set in LUFS; the compressor measures each channel's RMS, so on stereo it
   acts about 2-3 dB later than the table says). So a burst is pulled toward the speech of the whole edit, not toward a
   clip's own quiet background.
3. Each clip gets a gain, before the compressor, that moves its compressed loudness toward
   the mean of the clips. The gain is capped, because a boost also raises background noise.
4. Each clip is measured again through its gain and compressor (audio only, about as long as
   the first measurement), and one makeup gain for the whole set goes after the compressor.
   It brings the set's integrated loudness (all clips together, so weighted by duration) back
   to what it was, so the edit is not quieter than with `off`. A clip's gain plus its makeup
   is at most the preset's max boost, so a near-silent clip that is already boosted gets
   little or no makeup. The makeup goes before the limiter, so the ceiling still holds.

| Preset | Compressor starts | Ratio | Leveling | Max boost / cut |
|---|---|---|---|---|
| `light` | 8 dB above the typical level | 3:1 | half way to the mean | 6 / 12 dB |
| `strong` | 4 dB above it | 6:1 | all the way | 10 / 20 dB |

Measured on 12 clips (an R6 montage with gunfire and shouting, voice-chat commentary and
GoPro wind), exported and assembled with `--loudnorm`:

| | off | light | strong |
|---|---|---|---|
| Loudest moment above speech | 12.2 dB | 7.5 dB | 5.2 dB |
| 99th-percentile loudness above speech | 10.0 dB | 5.9 dB | 2.8 dB |
| Loudness spread between clips | 9.9 LU | 4.3 LU | 4.7 LU |
| True peak of `final.mp4` | -2.2 dBTP | -2.2 dBTP | -2.0 dBTP |
| Quiet parts of the commentary clips | | +5.5 dB | +9.9 dB |

The quiet parts rise because those clips are boosted to match the others. Use `light` when
background noise matters more than even loudness.

Without `--loudnorm` the makeup keeps the level. Measured on six 10 s clips of an R6 round
(gunfire) and six of another gameplay recording, exported with `clips`:

| Set | off | light, before the makeup | light | strong, before the makeup | strong |
|---|---|---|---|---|---|
| R6: loudness / true peak / range | -20.9 LUFS / +0.2 dBTP / 15.3 LU | -24.1 / -5.2 / 8.9 | -20.9 / -3.0 / 8.9 | -27.1 / -9.4 / 5.7 | -20.8 / -3.0 / 7.0 |
| Both (12 clips) | -19.3 / +0.2 / 18.0 | -23.9 / -4.9 / 13.2 | -19.3 / -3.0 / 14.6 | -27.3 / -9.4 / 15.1 | -19.4 / -2.9 / 13.9 |
| Other gameplay | -16.1 / -2.1 / 23.0 | -23.7 / -9.3 / 18.7 | -18.0 / -3.2 / 19.0 | -27.3 / -12.4 / 21.2 | -18.9 / -3.1 / 16.6 |

On the other gameplay set alone the makeup reaches its cap (6 dB light, 10 dB strong), so it ends
1.9 and 2.8 LU below `off`. With `--loudnorm`, `assemble` sets the final level anyway: the
result is the same within 0.1 LU (-14.0 LUFS, -2.2 dBTP).

`clips` bakes it into each clip, followed by a true-peak limiter (clips stay below about
-1.8 dBTP), records `compress_peaks` and the clip's makeup (`compress_peaks_makeup_db`, dB) in
`manifest.json` and tags the file (container comment
`cutpoint:compress_peaks`). `assemble` does not compress those clips again, also after they
are renamed. `extend` re-exports clips without compression (one WARNING). 360 clips are not
compressed.

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

Action-cam module (its visual check needs the `events` extra, which Cutpoint lite leaves out).
Finds crashes and wipeouts in an analysed
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
(`CUTPOINT_VISION_MODEL`;
[install](getting-started.md#optional-ollama-for-frame-rating-and-ai-chat)) for new profiles
and `--context`. Without them it falls back to the nearest library profile or `generic`, and
tries again on the next run.

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
