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
| [`ui`](#ui) | Open the local browser editing workspace |
| [`capabilities`](#capabilities) | Report GPU vendors, working encoders, transcription, torch and SDK support |
| [`edit`](#edit) | Analyze, export clips and assemble in one command |
| [`analyze`](#analyze) | Find and rank highlight candidates |
| [`detect`](#detect) | Detect the footage category and the style `--style auto` picks |
| [`events`](#events) | Detect sounds, HUD events and describe windows with a model |
| [`clips`](#clips) | Export moments as numbered clips |
| [`extend`](#extend) | Re-export a clips folder with more padding |
| [`assemble`](#assemble) | Join clips into one video |
| [`stage`](#stage) | Copy slow-storage sources to a local disk |
| [`crashes`](#crashes), [`mix`](#mix) | Action-cam module |
| [`resume`](#resume) | Re-run an interrupted or failed run from its `progress.json` |
| [`profile`](#profile) | Pick or discover the project's activity profile; list, show, create, validate and delete the profiles of your library |
| [`style`](#style) | List, show, create, validate and delete your own scoring styles |
| [`360 ...`](workflows/360.md) | Insta360 X3 module |
| [`setup`](#setup), [`paths`](#paths), [`guide`](#guide), [`skill`](#skill) | Setup, data, the agent guide and skill |
| [`proxy`](#proxy), [`login`, `logout`](#login) | Subscription logins for the hosted models of `cutpoint ui` |

## JSON output

`--json` makes a command write its result as exactly one JSON object on stdout. It is available
on `paths`, `analyze`, `detect`, `clips`, `assemble`, `edit`, `extend`, `capabilities`, `mix`, `stage`,
`setup`, `events`, `crashes`, `profile`, `style` (every verb), `resume` and every `360` subcommand (`scan`, `analyze`, `status`,
`director`, `stitch`, `qa`, `repair`). `ui`, `guide`, `skill`, `proxy`, `login` and
`logout` have no `--json`. Logs and errors stay on stderr, and
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
| `command` | The command as typed, such as `analyze` or `360 qa` (`resume`: the command it replays, see [resume](#resume)) |
| `schema` | `1` |
| `project` | Absolute path of the project folder, or `null` when the command has none (`paths`, `capabilities`, `setup`, `stage`, and `360 status` without a project) |
| `outputs` | Object: the files written (absolute path strings) and key facts. The keys per command are below |
| `warnings` | List of `{"code", "message", "files"}`; `files` is a list of absolute paths and may be empty |
| `error` | `null`, or `{"code", "message"}`. `message` is the `ERROR` line printed on stderr |

With a failure, `ok` is `false` and `outputs` holds whatever was recorded before it. An
argument error (`bad_args` from the parser) also writes an envelope, then exits 2. A command that
checks an option itself (a bad `--crop`, an odd or non-positive `--size`, `--clip-override`, `--ids`, a `--lut` file that is not `.cube`/`.3dl`, unreadable or that ffmpeg cannot load, `--lut-strength` without `--lut`, `--color` or `--color-strength` with `--lut`, `--color-strength` when no `--color` correction runs, `mix --quotas`,
`360 director --out-name`, `stage` with no inputs, an unknown `--activity`, a `profile create` file that is not valid JSON) reports `bad_args` and exits 1.
So do these combinations:

- `extend` with no padding and `--no-speech-aware`
- `360 qa --compare` with `--paths`
- `assemble <project> --force` with `--format mp4`
- `assemble <clips_dir>` with an option that only works from a project
- `edit --format fcpxml|otio` with `--no-keep-clips`

| `error.code` | Exit | Cause |
|---|---|---|
| `bad_args` | 2 (1 for the checks above) | The command line or an option value is invalid |
| `not_found` | 1 | An input, project or file is missing |
| `ffmpeg_missing` | 1 | ffmpeg or ffprobe is not found |
| `missing_extra` | 1 | An optional extra is not installed (the feature is not available in Cutpoint lite) |
| `output_not_empty` | 1 | A file or folder is in the way; see `--force` |
| `folder_busy` | 1 | Another cutpoint run holds the folder's lock (`.<folder>.cutpoint-lock`); wait and retry. `analyze`, `edit`, `events`, `crashes`, `profile`, `360 analyze` and `clips` hold `<project>/.analysis.run.cutpoint-lock` (see [Run lock](#run-lock-and-progressjson)) |
| `low_memory` | 1 | Too little free memory to load a model, also after Cutpoint freed cached models and unloaded Ollama's; close other programs, then run the command the `ERROR` line names (`cutpoint resume <project>`, or the same command again when no `progress.json` was written yet). See [Memory guard](#memory-guard) |
| `interrupted` | 130 | Ctrl+C, Ctrl+Break or SIGTERM |
| `error` | 1 | Anything else |

| `warnings[].code` | Command | Meaning |
|---|---|---|
| `resume_sources_changed` | `resume` | Sources of the recorded run are missing now; `files` names them |
| `resume_args_changed` | `analyze`, `edit`, `events`, `crashes`, `profile`, `360 analyze`, `clips` | The run replaced the `progress.json` of an interrupted or `low_memory` run with a different command or options; `files` names `<project>/progress.json` |
| `kept_clips` | `clips`, `extend` (so `edit`) | `.mp4` files the old manifest does not list stayed in the folder (the message names clips renamed during review); `assemble` includes them |
| `kept_sources` | `clips`, `extend` | A file the run reads, or the clips were cut from, was not replaced |
| `join_unmatched` | `assemble` with `--joins` | A pair in the joins file is not two clips next to each other in this order; `files` names the two clips. That join uses `--transition` |
| `join_no_room` | `assemble` with `--joins` | A Fade join has no room for one frame next to a clip (more than 45% of the shorter clip); `files` names the two clips. It becomes a cut |
| `temporary_clips` | `assemble <project>`, `edit --no-keep-clips` | The clips were exported into a temporary folder and removed after the join; `outputs` lists no clips. Export them with `clips` to keep them |
| `temporary_clips_left` | `assemble <project>`, `edit --no-keep-clips` | A temporary clips folder could not be removed (a file in it is open in another program); `files` names it. Delete it, or the next such run in the project removes it |
| `skipped_clips` | `extend` | Files without a manifest entry were not re-exported; the message says why for each (a source, renamed during review, or added by hand) |
| `vision_unavailable` | `analyze` | Vision rating was requested (`--vision` or `"vision": true` in `config.json`) but Ollama or the model is not available, or the port of `OLLAMA_HOST` or `ollama_url` is invalid (the step is skipped, the rest of the run goes on); if some of the moments to rate have cached ratings those are used (the message gives the counts), otherwise no vision rating ran |
| `vision_consent_needed` | `analyze`, `detect`, `events`, `360 director` (so `edit`, `360 analyze`), `profile`, `crashes` | The Vision role's provider needs the Vision consent and it was not given, so nothing was sent to it: a hosted provider (an API key or a subscription login), or an Ollama on another machine (`--ollama-url`, `vision_assistant.url`, `OLLAMA_HOST` or `ollama_url` is not `localhost` or a loopback address, or is on the Cutpoint proxy's port; `profile` and `crashes` use only that Ollama). Vision was skipped, or the activity profile fell back without the model; allow vision requests in the Models tab of `cutpoint ui` (Vision role), or add the key the message names (`vision-v2:<provider>@<host>[:port]`, such as `vision-v2:ollama@10.0.0.5:11434` or `vision-v2:openai@api.openai.com`) to the [`vision_assistant.consent`](configuration.md#vision_assistant) list in `config.json`. For the footage check of `--style auto` and `detect`, only when vision was asked for (`--vision` or `"vision": true`); by default the check is skipped with an INFO line |
| `chat_consent_needed` | `analyze` (so `edit`) | The dialogue step found no usable text model and the AI Chat model is hosted (an API key, a subscription login, or an Ollama on another machine) without the AI Chat consent, so no transcript was sent. The step was skipped; allow it in the AI Chat tab of `cutpoint ui`, or add the key the message names (`assistant-v3:<provider>@<host>`) to [`assistant.consent`](configuration.md#assistant) in `config.json`. The Vision consent does not count. It is not given when a local Ollama answers instead |
| `vlm_unavailable` | `events` (so `edit --events`) | The VLM layer ran but the vision model (Ollama or the hosted Vision role) is not available, or the port of `OLLAMA_HOST` or `ollama_url` is invalid (only the VLM layer is skipped, with a WARNING); windows not described yet stayed undescribed and cached descriptions were used (the message gives the count) |
| `grade_dropped` | `clips` (so `edit`) | `--force` replaced clips whose manifest recorded a grade (`lut` or `color`) and this run grades none; give `--lut` or `--color` again to keep it |
| `pano_mixed_sources` | `clips` (so `edit`), `assemble <project>` | The run's 360 clips are not rendered like-for-like: stitched and dual fisheye sources mixed, or an equirect rendered without the IMU levelling (FlowState, no sidecar, Studio export); `files` names the 360 clips (empty for a temporary export: `assemble <project>`, `edit --no-keep-clips`). Export with `--pano-source dfisheye` to compare their `360 qa` with the previews (see [360 video](workflows/360.md#render-sources)) |
| `color_on_gaming` | `clips` (so `edit`) | `--color` was given for `gaming` footage: game graphics are graded too (`underwater` is made for footage shot underwater; `--color auto` or `--lut` fits games better) |
| `loudness_below_target` | `assemble` (so `edit`) | `--loudnorm` left the final more than 1 LU below -14 LUFS: the limiter took the peaks the gain pushed over the ceiling; `--compress-peaks` leaves it more room |
| `step_failed` | `analyze` (so `edit`) | An analysis step (reactions, keyframes or crashes) failed with an `ERROR` line; the run went on without it. The dialogue step adds it too when there was not enough free memory for its text model (`Dialogue skipped: ...`). A failed HUD OCR (keyframes for `gaming` and `funny`) is a `WARNING` instead, whose message names the effect (menu filtering and HUD events skipped) and the source it failed on; `files` lists the sources it did not read. Out of memory says so, with what to try |
| `transcript_fallback` | `analyze` (so `edit`) | Without `--model`, Whisper ran (or an earlier run's transcript made so is reused) with `small` on the CPU instead of `medium` on the GPU, which was short of memory or failed; `files` names the sources. Speech-based moments can be weaker. A later run without `--model` transcribes them again with `medium` once it fits on the GPU (close the programs using it); after a CUDA failure the `small` transcript is kept (`--model medium` transcribes it again) |
| `action_cam_no_telemetry` | `analyze` (so `edit`, `360 analyze`) | An action-cam style (`action-cam`, or a style of yours over it) on footage without GoPro or Insta360 X3 telemetry. When no source has any, the run used the `action` style instead (not in `360 analyze`): `highlights.json` has `style` `action`, `settings.style_requested` the style given and `settings.style_fallback` (`from`, `reason`); `files` lists every source; give `--style action` (or `auto`) to choose it. When some sources have telemetry, or a source's telemetry could not be read, the style stays and `files` names the sources scored from sound, speech and visual features only (no speed, airtime or masking) |
| `style_ambiguous` | `analyze`, `detect` (so `edit`) | `--style auto` could not tell the footage category apart (the message gives the evidence), so the run used its best guess's style when that is gameplay (`gaming`), else the General style (`action`), for `analyze` and `edit` alike (the message names the style used; `funny` for `funny-moments` either way); `files` lists the sources that were not clear. Give `--style` to choose one |
| `unknown_profile` | `style create`, `style validate` | The style's `profile` or an `applies_to` profile is not in the [activity profile library](#profile); it is stored as written |
| `feature_not_computed` | `style create`, `style validate` | A weight names a feature the style's base never measures (for example a telemetry feature over a generic base), so it has no effect |
| `invalid_style` | `style list` | A file in the styles folder is not a usable style; the message gives the reason. Run `style validate NAME` |
| `style_sources_disagree` | `analyze`, `detect` (so `edit`) | With `--style auto`, sources making up more than 10% of the footage look like another category than the project's; the whole project still gets one style. `files` names them: analyze them apart, or give `--style` |
| `torch_no_cuda`, `sdk_needs_nvidia` | `capabilities`, `360 stitch` (the second only) | The warnings of the report, see [GPU troubleshooting](troubleshooting/ffmpeg-gpu.md#torch-and-the-insta360-sdk-in-capabilities) |
| `exclusions_ignored` | `mix` | Exclusions whose source is relative to the current folder of a drive were ignored |
| `content_qa_failed` | `mix` | The content checks of the mix failed |
| `nothing_staged_under_input` | `stage --clear` | An input that no longer exists has no staged copy recorded |
| `events_extra_missing` | `setup` | The `pip` component found modules for `events` missing |
| `stale_timings_kept` | `360 analyze` | The run failed after `highlights.json` changed and the earlier run's `pano_timings.json` could not be removed (held open by another program); `files` names it. It does not describe the analysis in the project: delete it, or rerun `360 analyze` |
| `events_not_in_highlights` | `360 analyze` | A source was not reachable, so `highlights.json` lacks the events |
| `not_a_360_clip` | `360 director` | A selected moment is not a 360 clip and was skipped |
| `vl_check_unavailable` | `360 director` | `--vl-check` could not run or stopped answering |
| `stitch_failed` | `360 stitch` | A clip's stitch failed; `files` names its log |
| `qa_clips_skipped` | `360 qa` | Clips that could not be checked |
| `qa_clips_unmatched` | `360 qa --compare` | Clips in only one of the two `qa.json` files |
| `qa_source_mismatch` | `360 qa --compare` | A `qa.json` was written for another folder than `<dir>` |
| `qa_fps_mismatch` | `360 qa --compare` | The two `qa.json` were checked at different `--fps`; frames without a partner are `unpaired` |
| `qa_clips_stale` | `360 repair` | QA clips with no matching segment, or newer than the QA |
| `footage_unreachable` | `crashes`, `events` (so `360 analyze`) | A source is not reachable (offline drive, moved file) and was skipped |
| `no_telemetry` | `crashes`, `events` (crash layer) | A source has no IMU telemetry; its crashes come from the transcript only |
| `no_visual_layer` | `crashes`, `events` (crash layer) | A source has no visual layer; run `events` first so what the camera sees counts |
| `highlights_not_updated` | `crashes`, `events` | The reanalyze step (`events`: the action-cam re-rank) was skipped (a source could not be found at its recorded or remapped path, or was not reachable), so `highlights.json` is unchanged |
| `context_not_applied` | `crashes`, `profile` | `--context` was not applied (Ollama or the LLM unavailable, or no usable profile from it) |

`outputs` keys per command (`project` as in the table above):

| Command | `outputs` |
|---|---|
| `paths` | One key per folder printed (`home`, `config`, `cache`, `models`, `tools`, `stage`, `styles`, `presets`, `profiles`, `projects`, `bundled`) and `missing`, the keys whose path does not exist |
| `resume` | `resume` (`argv`, `status_before`, `done_before`), `resumed`, and the replayed command's outputs |
| `analyze` | `highlights`, `transcript`, `candidates` (count, visual candidates included), `visual_candidates` (count; action-cam, action and sport), `suggested_top` (when set; the number of moments the default `clips` edit picks), `style`; with `--style auto`, `detection` (the `highlights.json` block, see [detect](#detect)) |
| `detect` | `category` (`gameplay`, `action-cam`, `action`, `talk` or `general`), `ambiguous`, `reason`, `scores` (share of the footage per category), `suggest` (`commentary` for talky gameplay, else `null`), `decided_by` (the evidence that decided the sure sources, most footage first: `telemetry`, `static HUD`, `HUD text`, `camera tags`, `360 frame`, `speech` or `vision`; empty when ambiguous), `sources` (each `path`, `duration`, `category`, `ambiguous`, `reason`, `decided_by` (`null` when ambiguous), `gameplay` (score 0-1, `null` with telemetry) and `speech` (share of the source covered by words, `null` before transcription)), `pending` (talk not checked yet: a long source has no transcript), `styles` (`analyze` and `funny-moments`: the style each would use) |
| `clips` | `clips_dir`, `manifest`, `clips` (each `file`, `srt`, `label`, `source`, `start`, `end`, `score`), `count`, `total_seconds`, `taken_over` (finished clips taken over from a stopped export's `.<folder>.cutpoint-resume`, not encoded again; 0 when none, missing for temporary clips) |
| `extend` | As `clips` (`taken_over` included), and `seconds` (`{"start", "end"}`) |
| `assemble` | `format` (`mp4`, `fcpxml` or `otio`); the file written: `final` (the video, `mp4`), or `fcpxml` or `otio` (the timeline); `srt` (`null` if none was written), `duration` (of the video or the timeline), `clip_count`, and `joins` with `--joins` (one `{from, to, transition}` per pair in the file that matched, with the transition used: `cut` or `fade`). From a project with `--format fcpxml\|otio`, also the outputs of `clips` for the kept clips; with `mp4`, no clip paths (warning `temporary_clips`) |
| `edit` | `kind`, the outputs of `analyze` (`visual_candidates` only for action-cam, action and sport), `clips` (none with `--no-keep-clips`) and `assemble`, and `copy` and `copy_srt` (the `<name>-final` files: the video, or the timeline with `--format fcpxml\|otio`; `copy_srt` is `null` if none was written). One envelope holds the warnings of all steps |
| `capabilities` | The report's own keys (as printed by `capabilities` without `--json`, including `edition`, `torch` and `insta360_sdk`) |
| `mix` | `segments`, `report`, `sheet`, `moments` (each `id`, `source`, `start`, `end`, `duration`, `category`), `count`, `total_seconds`, `target_seconds`, `categories` (per category `n` and `seconds`), `content_qa_pass` |
| `stage` | `stage_dir`. Staging: `files`, `copied`, `skipped`, `bytes`, `read_s`, `mb_per_s`. `--list`: `files` (each `source`, `staged`, `bytes`, `read_s`). `--clear`: `removed` |
| `setup` | `components`: one key per component (all by default, or those given to `--only`): `files` (each `path`, `bytes`, `status` of `downloaded`, `found` or `missing`); for `pip`, `installed` and `missing`; for `whisper`, `model` (its name), `path` (its folder in the Hugging Face cache) and `status` (`downloaded`, `found`, or `missing` with `model` and `path` `null` when the step did not finish) |
| `360 scan` | `pano`, `timings`, `clips` (each `id`, `mode`, `duration`, `frames`, `insv`, `insv_bytes`, `lrv`), `count`, `total_seconds` |
| `360 analyze` | `timings` (`pano_timings.json`: this run's `views`, `analyze_s`, `events_s`, `paths_s`, `scan_s` with `--scan` and `total_s`; with `--views-only` the earlier file's keys stay, with this run's `views` and `views_only_s`; `analyze_total_s` was removed, older files may still have it; see [360 video](workflows/360.md#commands); `null` in the envelope of a run that failed after `highlights.json` changed, as the earlier run's file was removed, or kept with a `stale_timings_kept` warning), `views` (`cached`, `computed`); with `--scan`, also those of `360 scan`; unless `--views-only`, also the outputs of `analyze` (`highlights`, `transcript`, `candidates`, `visual_candidates`, `suggested_top`, `style`); unless `--views-only` or `--no-events`, also `events`; unless `--views-only`, also `candidate_paths` (`dir`, `candidates`, `computed`, `cached`, `skipped`, `failed`, `seconds`, `flags`: candidates per flag code; see [360 video](workflows/360.md#camera-paths-of-the-candidates)) |
| `360 status` | `sdk` (`exe`, `origin`, `found`, `runs`, `version`); with a project, `clips` (each `id`, `mode`, `duration`, `views`, `render`, `render_file`) |
| `360 director` | `paths_dir`, `paths` (each `path`, `preview`), `count`, `vl_check` |
| `360 stitch` | `clips` (each `clip`, `output`, `status`, `seconds`, `log_file`, `estimated_gb`), `dry_run`. Never the SDK's own output |
| `360 qa` | `qa_dir`, `report`, `mode`, `sheets`, `clips` (each `file`, `score`, `pass`, `fails`), `count`, `passed`, `score`, `by_mode` and `by_self_method` (the `qa.json` edit's `edge_cut` breakdowns: per render `mode` or `self_method`, its `clips`, `frames`, `duration` and `edge_cut` share; `{}` without a checked clip). With `--compare`: `edge_gap` (its path), `mode` (`compare`), `rendered` and `paths` (the two `qa.json`), `clips` (each `file`, `rendered_only`, `causes`, `other_detail`), `count`, `rendered_only`, `causes`, `other_detail`, `by_mode` and `by_self_method` (`edge_gap.json`'s frame counts per rendered `mode` or `self_method`: `clips`, `frames`, `rendered_edge`, `paths_edge`, `rendered_only`; `unknown` for an older `qa.json`) (see [360 video](workflows/360.md#where-the-two-qas-disagree)) |
| `360 repair` | The repair report's own keys, and `exclude_file` (`null` if there are no exclusions) |
| `crashes` | `crashes` (the `crashes.json` path), `count`, `candidates` (strongest first, each with the `crashes.json` keys `id`, `source`, `t0` and `t1` (impact span, s), `confidence` (0-1), `confirmed` (dense check ran), `origin` (`telemetry` or `words`), `clip` (`start`, `end`, `until`, and `reaction` when planned) and `sheet` (when one was written)), `activity_file` (the `activity.json` path), `profile_name` |
| `events` | See [events outputs](#events-outputs) below |
| `profile` | For `profile <project>`: the `activity.json` record's own keys (as the older `profile --json` printed them: `version`, `request`, `name`, `source` (`flag`, `library`, `synthesized`, `context` or `fallback`), `profile`, and when set `scores`, `description`, `described_by`, `model`), and `activity_file` (its path) |
| `profile list\|show\|create\|validate\|delete` | `list`: `profiles_dir`, `profiles` (each `name`, `type` (`built-in` or `user`), `source`, `floor`, `valid`, `problems`, `path`, `description`; an unusable file also gives an `invalid_profile` warning). `show`: `name`, `type`, `source`, `floor`, `path`, `description`, `profile`. `create`: `name`, `path`, `replaced`, `profile` (a `profile_entry_ignored` warning per entry dropped). `validate`: the check result (`valid`, `problems`, `notes`) and its warnings. `delete`: `name`, `path`. The command names are `profile list` and so on |
| `style` | `style list`: `styles_dir`, `styles` (each `name`, `type` (`built-in` or `user`), `legacy`, `valid`, `problems`, `warnings` (each `code`, `message`), `source`, `base`, `description`, `profile`, `applies_to` (`categories`, `profiles`), `path`). `style show`: `name`, `type`, `legacy`, `source`, `base`, `kind`, `description`, `profile`, `applies_to`, `path`, `hash`, `mask_friends`, `must_include_crash`, `weights` (each `feature`, `weight`, `from`: `base` or `style`). `style create`: `name`, `path`, `replaced`, `style` (as `style show`). `style validate`: `name`, `path`, `type`, `legacy`, `valid`, `problems`, `notes`. `style delete`: `name`, `path`. `style features`: `features` (each `name`, `layer`, `description`, `extra`, `available`) |

### events outputs

`events` and `events_txt` (the `events.json` and `events.txt` paths), `layers` (the layers in `events.json`), `count` (events), `windows` (model windows), `highlights` (its path once re-ranked, `null` with `--no-rescore` or when the action-cam re-rank was skipped), `highlights_backup` (`highlights.pre-events.json`, when this run created it); when written, `crashes` (the `crashes.json` path, from the `crash` layer), `sheets` (contact sheet paths, `--sheets`) and `discovery` (per game `proposal`, `overlay`, `preset`, `installed` and `suggested`, the number of extra notice and score regions listed under `suggested_regions` in `proposal.json` for a game that kept its preset, else 0). An action-cam re-rank re-runs `analyze`, so its outputs (`transcript`, `candidates`, `visual_candidates`, `suggested_top`, `style`) are added. A missing `events` extra (torch and the other layer modules) is `missing_extra`; missing model files are `not_found` and name the `cutpoint setup` command. The VLM layer needs no model files: when its vision model does not answer, a WARNING says so and cached window descriptions are used

`360 qa --json-only` skips the contact sheets; it is a different option from `--json`.

## Stopping a run

Ctrl+C (and Ctrl+Break on Windows) exits `130` and removes partial files. From a program on
Windows, start Cutpoint with `CREATE_NEW_PROCESS_GROUP` and send `CTRL_BREAK_EVENT`
(`CTRL_C_EVENT` to a process id is not delivered). To get `130`, run `.venv\Scripts\cutpoint.exe`
(or the tool-installed `cutpoint`) directly: under `uv run`, Cutpoint still cleans up but uv
exits `0xC000013A`. `taskkill /F`, `Stop-Process` and `Popen.terminate()`/`kill()` stop it
without cleanup (exit `1`; `-1` from `Stop-Process`).

## Progress output

With `CUTPOINT_PROGRESS=1` in the environment, `analyze`, `edit` and `360 analyze` print one line per state change on stderr:

```text
PROGRESS {"step":"reactions","index":2,"total":7,"label":"Reactions","state":"running","pct":40,"detail":null}
```

| Field | Content |
|---|---|
| `step` | Step name, from the table below. A stable API |
| `index`, `total` | Position from 1, and the number of steps of the command |
| `label` | Display name |
| `state` | `running`, `done`, `skipped` or `failed` |
| `pct` | 0 to 100, or `null` when there is no cheap measure |
| `detail` | Short text (a skip reason, an error) or `null` |

The `dialogue` step is new: `analyze` now has 7 steps and `edit` 9 (it was 6 and 8), and `ranking`, `vision`, `clips` and `assemble` moved down one. `dialogue` reports `skipped` with a reason for every style but `gaming`. Running updates come at most 2 per second. Steps run in table order. The `pct` of `transcribe` covers each source's audio extract (its first 10%) and then Whisper.

| `step` | `analyze` | `edit` |
|---|---|---|
| `transcribe` | 1 | 1 |
| `reactions` | 2 | 2 |
| `keyframes` | 3 | 3 |
| `crashes` | 4 | 4 |
| `dialogue` | 5 | 5 |
| `ranking` | 6 | 6 |
| `vision` | 7 | 7 |
| `clips` | | 8 |
| `assemble` | | 9 |

`360 analyze` has its own 4 steps: `scan` (Read 360 clips; skipped without `--scan`), `views`
(360 views), `analyze` (Transcription and ranking) and `events` (Events and re-rank; skipped with
`--no-events`). `scan` and `views` report `pct` and the clip as `detail`.

When a step starts with part of its work cached (a rerun after an interruption, or a `--batch-size` boundary), it logs `<name>: N/M done, continuing` as an INFO line and sends the same text as the `detail`. `events` reports its layers (`audio`, `voice`, `visual`, `ocr`, `crash`, `vlm`, `write`) only in `progress.json` (no `PROGRESS` lines), so `resume --dry-run` lists them; so does `crashes` with its steps `profile` (activity profile; skipped when nothing resolves it), `inputs`, `dense` (the dense check, with its windows as `done`/`total`, updated at each `--batch-size` boundary) and `result` (contact sheets and `crashes.json`). After Ctrl+C or a failure, `progress.json` shows the step the run was in as `failed`, with `status` giving the reason; after a hard kill both stay `running` (see [Run lock and progress.json](#run-lock-and-progressjson)).

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
captions. Video files are not uploaded (a hosted model you enable after consent receives only what its role needs, such as frames or text). A camera
folder or file of Insta360 X3 footage makes a 360 project, analyzed with `360 analyze --scan`
(see [360 video](workflows/360.md#in-the-ui)); its moments preview through the director's camera
path (the one the analysis cached when it is current), with badges for poor paths
([Director preview](workflows/360.md#director-preview-and-camera-path-flags)). It does not expose
the action-cam commands or the other `360` commands; for standard footage, Highlight style starts on "Auto (detect from footage)" (`--style auto`; the project is
analyzed with it) and also lists the
`action-cam` style, which the dialog preselects instead when an input has GoPro or Insta360 telemetry (a
stream-list probe of the first 12 files `analyze` would find) unless you chose another style.

The header reads "cutpoint / <project name>": the name opens the project menu, which lists the projects
(the open one marked) and ends, after a divider, with "New project…". The browser tab shows the same title.

After the first analysis a style button follows the project name (not for 360 projects). For an
Auto project it reads "Detected: Gameplay" (the category `highlights.json` records in
`detection`), or "Not sure: General style" when the detection was unsure and fell back
(`gaming` when it leaned to gameplay, else `action`); a project in a style you chose reads
"Style: Talk (your choice)". Its menu gives the recorded reason, why an unsure detection used its
style, one "Disagree: b.mp4 looks like Talk (…)" line per source of another category when they are
more than 10% of the footage, and a suggested style. Below are Auto (detect from footage), the
categories with their style (Gameplay: gaming, Action: action, Talk: commentary, and Action cam:
action-cam when a source has camera telemetry or the project already uses it) and Other styles
(the rest, your custom styles included). Choosing one opens the "Analyze again as Talk?" dialog
with the Analysis options filled in; confirming runs `analyze --style <style>` (cached
transcripts and per-source results are reused) and makes it the project's style. Nothing is
deleted: the sequence, clips, final videos and AI Chat stay. After you leave Auto, the menu keeps
showing "Detected earlier: Gameplay" from a copy in `ui/meta.json`. The choices wait while a job
runs, as "Analyze again" does.

New Project's Source field takes the full paths of one or more videos and folders, one per line,
that you type or paste; it always accepts them. A leading `~` is the home folder of the computer
running `cutpoint ui` (`~/Videos`), here and in the Settings projects folder. They are the project's `analyze` inputs, as on the
command line. "Include sub-folders" adds `--recursive` and "Skip files matching" takes `--exclude`
patterns, one per line (matched against each file's name or full path, such as `*DVR*` or
`*/Captures/*`); "Analyze again" keeps them. The project is refused when no video is left to
analyze. Insta360 X3 footage still makes a 360 project of its own: its one folder or file, without
the two options, and a folder that mixes 360 and regular videos is refused. A project made with
either option has no one-click presets (`edit` has no `--recursive` or `--exclude`). "Choose video…" and "Choose folder…" add a line; they open the operating system's own file or folder dialog on the
computer running `cutpoint ui` (Windows: the system dialog through tkinter; Linux, with a display
(DISPLAY or WAYLAND_DISPLAY): zenity, falling back to kdialog; macOS: osascript). Where no dialog is
available (headless, Docker, no display, or an SSH session on Windows and macOS), those buttons are
hidden and you type or paste the path.

The gear button in the header opens Settings, which sets the projects folder (saved as
`projects_dir` in `config.json`). The folder must exist, or only its last part may be missing, in
which case it is created; it must be writable. Existing projects are not moved. The setting is
read-only when `CUTPOINT_PROJECTS_DIR` is set, and it cannot be changed while a job is running.
Settings > Analysis also turns on [`visual_index`](configuration.md#visual_index); every analysis
or One-Click Edit the page starts passes `--visual-index` or `--no-visual-index` from it.
The Models tab (next to AI Chat) sets the model for each role: Chat (the
[`assistant`](configuration.md#assistant) of AI Chat) and Vision
([`vision_assistant`](configuration.md#vision_assistant), the model that rates frames), each
with its own provider and model; both roles offer the same providers. Provider names say how
they sign in: Anthropic API, OpenAI API and xAI API use an API key (saved per provider and host, so
one key serves both roles); Claude Code subscription, Codex subscription and Antigravity subscription
use a login ("Connected accounts"). The UI starts the subscription [proxy](#proxy) itself, so
you do not need `cutpoint proxy` in a terminal to sign in. Log in (and Retry) on a Connected
accounts row always starts it when it is not running (the row shows "Starting the proxy…" and
shows an error there if it fails). A chat turn with a subscription model, a model-list refresh and
opening Connected accounts start it only when a subscription login is saved; nothing starts when
the UI starts. It also starts before an analyze or edit job (or a resumed one) whose Vision role or
gaming dialogue step uses a subscription provider, when a login is saved; if the start fails, the job
log has a WARNING and the job still runs. 360 jobs do not start it. A proxy the UI started stops when the UI exits (Ctrl+C, closing the console or a
hard kill); a proxy you started with `cutpoint proxy` is left running. With the proxy running and
consent granted for the role, the tab lists each account's advertised models (Refresh catalog asks
again); Other model… takes any id.
The tab works without a project, and
one Save models saves both roles. "Rate frames with vision model"
(off by default; saved as [`vision`](configuration.md#vision) in `config.json`) is there too:
every analysis the UI starts, a first one or "Analyze again", uses it (ticked saves
`vision: true`, passed as `--vision`; unticked removes the key, so no flag is passed and the
footage check of `--style auto` runs; a hand-set `false` passes `--no-vision`; 360 analyses never rate frames), and it is the default of `analyze` and `edit` on
the command line. A hosted Vision provider gets frames and the speech in each moment only after "Allow vision requests" in
the tab; saving never grants it. The UI has no per-run vision checkbox. "Analyze again" keeps
the Ollama model, URL and vision-top the last analysis recorded.

Unsaved role selections and API-key drafts survive tab switches; reload the page to discard them
and read the configuration again. A custom Vision Ollama URL uses that server's model catalog.

"Stats for Nerds" (off by default) shows the open moment's score, rank, analyzed times, signals
and keywords over the top-left corner of the preview. It shows as soon as you tick it; Save keeps
it, in the browser (local storage), not in `config.json`, and Cancel or Escape undoes it.
Theme picks the colours: Graphite (the default: neutral greys with amber for what is chosen),
Light or Blue (the colours of before Graphite). Graphite and Light keep text, controls and focus
outlines at WCAG AA contrast; Light uses the same yellow for buttons and fills as the other
themes, with darker amber for text, focus rings, the selected tab, checkboxes and sliders. It
recolours the UI at once and, like Stats for Nerds, Save keeps it in the browser and Cancel or
Escape undoes it. The Dark theme of earlier versions is gone: a browser that saved Dark shows
Graphite. The UI uses Inter and JetBrains Mono when the machine has them, else the system fonts;
nothing is downloaded.
"Show advanced export options" (off by default) shows the Export panel's advanced controls (marked
"advanced" below). Hidden, they keep their values and are still sent with each job, and
"Advanced options in use: ..." under the options names any that are; a hidden value out of range
shows them so the browser can point at it. Like Stats for Nerds, it shows at once, Save keeps it
in the browser, and Cancel or Escape undoes it.
"Verbose moment labels" (off by default) shows each tile's source video name, id (`h001`) and start
to end time in the Moments tab; off, the tile keeps its score, kind, label and length, and the
search still finds a moment by the hidden name and id. It works like Stats for Nerds: at once, Save
keeps it in the browser, Cancel or Escape undoes it.
"Single-key shortcuts" (on by default) lets the keys M (mute), I and [ (trim start) and O and ] (trim end,
at the playhead) act on the page. Turn it off, as WCAG 2.1.4 asks for speech-input and stray-key
users, and none of them does anything and their hints disappear; Space, Escape, arrow keys, keys
inside text fields and controls, and Ctrl or Cmd shortcuts such as Ctrl+Z keep working. It works like
Stats for Nerds: at once, Save keeps it in the browser, Cancel or Escape undoes it.
The UI uses `POST /api/pick` for the dialogs and `GET`/`PUT /api/settings` for Settings.

With a project open, Settings also has "Delete this project". The dialog names the project and the
size of what will be deleted, lists the files that stay, and has Cancel focused. Only what
Cutpoint wrote is deleted: the result files (`highlights.json`, `transcript.txt`, `events.*`,
`progress.json`, `activity.json`, `crashes.json`, `telemetry.json`, `mix_*`, the finals with their
`.srt` and timelines), the `ui/` folder and the other folders Cutpoint made in the project,
hidden temporary and resume folders, and in each clips folder the clips its `manifest.json` lists
with their `.srt` and the manifest. Everything else stays: staged footage copies, clips you added
by hand, `hud_presets` (you may have calibrated them), unknown files and folders, and anything that
is or holds a source video of any project; the project folder is removed only once it is empty. A
symlink or junction is never followed, and your source videos are never touched. The files go to
the Recycle Bin / Trash. On Windows Cutpoint first checks that the Recycle Bin of the drive will
keep them (a fixed drive, not set to delete at once, big enough for the largest item); in a
container there is no trash. If the trash cannot be used, the page asks a second time before
deleting them for good. It is refused (409) while a job, an analysis or export, a preview
render or an AI Chat turn is running on the project. After deleting, the page shows the start
screen. API: `GET /api/projects/<id>/delete-plan` (`name`, `size`, `files`, `kept`, `kept_count`)
and `DELETE /api/projects/<id>` with `{"permanent": true}` for the second case; a trash that cannot
take the files answers 409 with `"code": "no_trash"`.

Before a project is analyzed, the Moments column offers "Analyze footage" and "Show folder", which
opens the folder with the project's videos in the system file manager (hidden where that is not
possible, such as Docker). After analysis, "Analyze again" and "Show folder" are in the project
menu (the project's name in the header), above a line with the moment count, the style and how
long the analysis took; the Moments head shows "Moments" and the total count. "Analyze again"
waits while another job runs and asks for confirmation first. The Export panel has one "Show Exports
Folder" button: it opens the latest export's `clips_ui_*` folder, or the project folder (final
videos and `.srt`, and a `.fcpxml` or `.otio` timeline only when assembled with `--format
fcpxml|otio`, the Output format) once that export is assembled. A timeline assembled without a
video is listed and downloadable as that export's final; the `.assemble-tmp-<id>` folders of
`assemble <project>` are never listed.

After analysis, the Moments list has a search field (label, words spoken, file name or candidate id such as `h001`) and a
funnel button beside it that opens Sort and filter. Sort by is Score (the default; rank #1
first) or Position in video (by source name in natural order, then start time); each card's
rank badge keeps its score rank either way. The quick filters are Sequence (All, In sequence,
Not in sequence), Ready only (the moment's preview plays now) and Source file when
the project has several sources. They combine with the search text. A dot on the button shows
while a filter or a sort other than Score is on, and when nothing passes the list reads "No
moments match these filters." Escape closes the menu. The sort and filters are remembered per
browser (local storage); the source file only while the project stays open. A project with a
long dialogue bit (a whole dialogue stretch over 90 s) also offers Long bits, which shows only those; they
have a "Bit" badge with the length (m:ss), and the tile's hover text adds what happens in the bit.
Long bits is not remembered; it lasts until the page is reloaded. A crash moment has a "CRASH 0.91" badge with its confidence: solid when it is always kept, dashed when it is ranked like any moment.
The moments are tiles, two across in the default column width. A kind filter above them (All,
Action, Scenic, Funny, Dialogue, Crashes) shows one kind; a chip is hidden when no moment has that kind, and the filter falls back to All; it is a view of the list and is not kept. The
kind is worked out in the page for display, as AI Chat names it. A moment a talk stretch created
(`signals.events.nominated` holds `dialogue-<kind>`) is always Dialogue, titled or not. For any
other moment, the words of its labels
(not a generic `action-moment`) and its `kind` decide first (Scenic for a `view` or a word such as
scenic or beauty, Funny for laugh or funny, Action for `crew`, `cruise` or a word such as ski, bike, surf or fast
or crash, Dialogue for talk or banter), then the strongest signal of the moment that rates 5 or
more out of 10 (crash, speed, airtime, wipeout or carve: Action; beauty or scenic: Scenic; laughter:
Funny; the page does not have the cues of the events VLM windows), then a transcript of four words
or more (Dialogue), else Action. Each tile shows the moment's still (the one its
sequence row uses; the server cuts it once, the first time the tile scrolls into view, and caches
it) with a kind dot, the length and an Add button over its corner (a dark + on amber; a white − on
red once the moment is in the sequence); under it the rank and title, "kind · score" followed by
the preview readiness as an icon (a spinner while encoding, a check when ready, a clock when
queued; hover it for the word), the candidate id, the times, the file name when the project has
several sources, and a strip of the moment's place in its source. The tile has no transcript text:
hover it for the full title and the transcript, which the search still finds. A dialogue moment
(one a talk stretch created or widened) also shows what the reader says happens in it, two lines at
most, and in the hover text. A moment a stretch only widened keeps its own type (worked out from the label it had), but takes
the stretch's title as its name and the title's slug as its label.
A moment with only a generic label (`action-moment`, `loud-moment`) is titled with what the
vision model saw: its vision rating label, else the label of its `events` VLM window (the search
finds both). `highlights.json`, the label a moment takes into the sequence and clip file names
keep the analysis label. A moment of the visual pool (beauty shots, friends, cruising runs)
carries a "visual" badge with its kind, such as "visual · view".

While an analysis runs, the Moments column lists its steps (Transcription, Reactions, Keyframes,
Crashes, Dialogue, Ranking, Vision rating; for a 360 project those of `360 analyze`) under "Analyze footage", or under the column header for "Analyze
again": each with its state (waiting, running with a progress bar and a percentage when the step
reports one, done, skipped with the reason, failed with the error). A step still running when
the job fails reads "interrupted", one stopped by Cancel reads "cancelled", and the steps that
never started read "not run". The list stays after a failed or cancelled analysis
until another job runs, and gives way to the moments once the analysis completes. The UI asks for
the steps by starting its jobs with `CUTPOINT_PROGRESS=1`; `GET /api/jobs` returns them per job as
`progress` (`total` and `steps`, ordered by step; for a 360 analysis also `plan`, the steps the
job will report as `step` and `label`). A Cutpoint version that sends no steps shows
the log in Jobs only, as before.

During the first analysis the illustration, the count of videos found, "Analysis options…" and the "Analyzing…" button give way to
"Show folder", the steps and a yellow "Pause analysis" button that stays at the top of the column. Pause cancels the job as Cancel does, and the column then
offers Resume, which continues from the last transcribed video. Before the first video is
transcribed Pause asks first, because there is nothing to resume and the analysis starts over.

The side column has three tabs: Sequence, AI Chat (see [AI Chat](#ai-chat) below) and Models.
The header's Export button, next to Save sequence, opens the Export panel in the side column, with
no tab selected; Back (or Escape) returns to the previous tab, and running jobs keep running. A
sequence row shows its
kind as a colour bar and a name, and its exported length at the right, with "cut N s + tail"
under it when a tail runs the export past the cut.
The Export panel holds one "Export &
Assembly Options" panel; each option has a hover description, and the choices are remembered per
browser:

| Option | Effect | Default |
|---|---|---|
| Vertical 9:16 | Exports and assembles vertical clips | off |
| Burn in captions | `assemble --captions`; off with a timeline Output format | off |
| Speech-aware cuts | `--speech-aware` | on |
| Smart tails | `--smart-tail` | on |
| Normalize loudness | `--loudnorm`; off with a timeline Output format | on |
| Output format | What Assemble writes (`assemble --format`): Video (mp4), Final Cut (fcpxml) or OpenTimelineIO (otio). A timeline uses the current export's clips in their reviewed order and is written in about a second, with no audio processing or burned-in captions: Normalize loudness, Burn in captions, Captions model and Assembly audio fade are off with it (the server refuses the first two) | Video (mp4) |
| Link Transitions | On: the audio transition follows the video (Cut gives Cut, Fade gives Crossfade) and the audio select is disabled. A join set to Fade in Cut mode still crossfades its audio under a Cut video transition | on |
| Video transition | Cut or Fade (`--transition`); the default of every cut. In Cut mode the Cuts strip sets single cuts to Cut, Fade or Default (follows this control), passed as `--joins` | Cut |
| Audio transition | Cut or Crossfade (`--audio-transition`); Crossfade is unavailable under a video Cut | follows the video |
| Compress Audio Peaks | None, Light or Max (`off`, `light`, `strong`) | Light |
| Color | Off, Auto or Underwater (`--color`), in Export clips and One-Click Edit; Extend export keeps each clip's grade | `color` in `config.json`, else Off; Off for gaming projects |
| Strength | 0-100% of the Color correction (`--color-strength`, left out at 100%); shown with Auto or Underwater | 100% |

The padding field is gone from the UI; `clips --extend` and [`extend`](#extend) still work. A saved
"Smart edges" choice from an earlier version carries over to both Speech-aware cuts and Smart tails.

The Export panel has more controls above and below that panel. A blank Cut Timing field keeps the
CLI default.

**Clips to Export.** By default the export is the Sequence, as hand-picked. Automatic picks
export from the ranked moments instead:

| Control | CLI flag |
|---|---|
| Top moments (advanced) | `--top` |
| Max video length: the cap on all clips together | `--max-total` |
| Candidate ids (advanced; replaces Top moments): the ids on the moment cards (shown with Settings > Verbose moment labels), such as `h001` | `--ids` |
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

**Clip Picture & Audio** (Export clips) and the assembly fields under the transitions
(Assemble). Default or a blank field sends no flag, so the CLI's default applies:

| Control | CLI flag |
|---|---|
| Crop (off for 360 projects): Default, Auto, None or Custom (a Crop box `W:H:X:Y` in source pixels) | `clips --crop` (Default: no flag) |
| Output size (advanced; off with Vertical 9:16 and for 360 projects): `WxH` in even pixels, such as `1280x720` | `clips --size` |
| Clip audio fade (advanced; 0-10 s, 0 = off) | `clips --audio-fade` |
| Fade length (advanced; with the Fade video transition, or a join set to Fade in Cut mode) | `assemble --fade-dur` |
| Assembly audio fade (advanced; off with the Crossfade audio transition, except at joins that come out as cuts in a mixed video): for clips exported without a fade | `assemble --audio-fade` |
| Captions model (advanced; with Burn in captions): Default, `large-v3`, `medium` or `small` | `assemble --model` |

The "Latest export ..." facts under the options are advanced too; warnings and guidance there
(such as "Exported from the 360 preview (not stitched)") always show. When a change in the Cuts
strip or the Video transition, which is the Default of the other cuts, would make a cut come out
differently from the final video, "Assemble again to apply the joins." shows until Assemble
finishes, the change is undone, or a new export replaces the clips. It stays away for a final whose
joins are not known (a One-Click Edit's, or one assembled by an earlier version).

**Extend export** (advanced) runs [`extend`](#extend) on the latest export into a new `clips_ui_<stamp>`
folder. The result becomes the current export. It is disabled while Extend start and Extend end
are blank or 0 and Speech-aware cuts is off (extend would have nothing to do); the tooltip says so.

**One-Click Edit** (marked Experimental: it picks and joins moments without your review) has the presets Highlights, Montage, Clips and Funny moments. Make runs
`cutpoint edit --kind <preset>` in the project's own style, with the Vertical and Compress Audio
Peaks choices and the vision choice from the Models tab. Clips go to `clips` and the video to
`final.mp4` (`clips_vertical` and `final_vertical.mp4` when vertical), with the `<name>-final`
copy next to it; that export becomes the current one. If earlier clips exist,
a second run asks before replacing them and passes `--force` only when you confirm. The presets
are off, with the reason shown, for 360 projects and for projects analyzed with `--exclude` or
`--recursive`, `--no-scenes`, a non-default whisper `--model`, `--language`, `--max-candidates`,
`--must-include-crash` with a value other than the style's default (0.6) or `--recover`. A Skip steps choice is passed on as `edit --skip`.

**Analysis options** are in the "Analyze again" dialog, and in the dialog that "Analysis
options…" opens before the first analysis ("Analyze footage" alone uses none of them). Each one
is one [`analyze`](#analyze) flag; a blank field or unticked box passes none, so the CLI default
applies:

| Control | CLI flag |
|---|---|
| Transcription model | `--model` |
| Language | `--language` |
| Max candidates | `--max-candidates` |
| Crashes become moments at | `--must-include-crash` |
| Skip scene detection | `--no-scenes` |
| Recover after read errors | `--recover` |
| Skip steps: Reactions, Keyframes, Crashes, Dialogue | `--skip reactions keyframes crashes dialogue` |

The dialog starts with what the project's analysis used, as `highlights.json` records it (the
model only when it is not the default), so "Analyze again" repeats it unless you change it. It
keeps a recorded `--recover` the same way. `highlights.json` does not record `--skip`, so the UI
keeps it in `ui/analysis.json` for that analysis only; after a command-line re-run it starts
unticked. 360 projects have no Analysis options (`360 analyze` takes none of these flags).

**Refine** is a block of the same dialog, below the options. Each control starts its own job,
not an analysis:

| Control | CLI command |
|---|---|
| Describe the footage + Apply | [`crashes`](#crashes) `<project> --context=<text>` |
| Look closer with vision | [`events`](#events) `<project>` |

Both are off until the project has been analyzed, and in Cutpoint lite (the reason is shown).
Describe the footage shows only where the analysis runs the crash step: an action, sport or
action-cam style, footage that Auto detection chose as action-cam, action or general, or sources with
camera telemetry (the analysis wrote `crashes.json`); before the first analysis, no style or
Auto. It is filled with the
description `activity.json` holds and needs text to apply (there is no reset). Look closer with
vision is off for 360 projects. A `vision_consent_needed`, `vision_unavailable`, `vlm_unavailable` or
`context_not_applied` (the description was not used) warning of the run is shown as a note in the job's log. A cancelled run is offered as **Resume crash detection**
or **Resume closer look**.

API: the clips job accepts the new fields (also `crop`, `size` and `audio_fade`), the assemble
job `fade_dur`, `audio_fade` and `caption_model` (`fade_dur` is refused with the Cut transition
and `caption_model` without `captions`), `POST /api/projects/<id>/jobs` takes the new `extend` and `edit` actions,
and the project view reports `preset_problem` (why the presets are off) and `preset_taken` (the
earlier clips a second Make would replace). The analyze job takes `model`, `language`,
`max_candidates`, `must_include_crash`, `no_scenes`, `recover` and `skip` (a list); a field it
does not send keeps what the analysis used, which the project view reports as `analyze_options`
(with `must_include_crash_default`, the style's value, `null` for 360 projects).

The sequence panel shows a thumbnail per clip and supports inline rename (pencil or double-click),
remove, multi-select with batch remove (a "Select all" checkbox above the list ticks every clip, or clears
them all when every clip is ticked; Ctrl+A (Cmd+A) in the list ticks every clip), up/down buttons and drag reorder (the row actions show on hover or keyboard focus, and the whole row drags). A footer shows the total length; click the total to set a target length (such as `4:00` or seconds; empty removes it). The target only guides the progress bar (and is shown after the Storyboard's total, as "26 clips · 3:28 of ~4:00"), is saved with the project (in `ui/review.json`, so it follows the project and goes with it when deleted, whatever the browser or port) and cuts nothing. Thumbnails are cached as
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
when its `.srt` exists). The outputs panel lists every final video, or timeline assembled without a
video, in the project folder, newest first with the export's own final first; until a new
export is assembled as a video, the preview of an earlier final is titled "Earlier final". A cancelled job shows "stopping" (status `cancelling`
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
`M` toggles mute (unless Single-key shortcuts is off in Settings > Keyboard), and Space plays or pauses even when the volume controls have focus.
Playback and scrubbing cover the whole source: playback runs on past the clip end, and Play at
the end of the source starts the clip again. The trim rail under the video is a zoomable view of
the source with the clip (gold) and the analyzed range (dashed) as markers; it opens on the clip
plus half its length (at least 10 s) either side. Behind the markers, a filmstrip shows at most
one still per 78 px (one every 78 to 156 px): the keyframe at or before each point, so a still
can be a little early; zoomed in to under 1 s per still, exact stills instead. The stills load a
few at a time once the view stops moving, nearest the playhead first, and are cached with the
thumbnails; a still that fails leaves the rail's ticks showing, and after 3 failures a source
gets no filmstrip until the page reloads. With the browser's data saver on
(`prefers-reduced-data` or Save-Data), the rail keeps its ticks only. The zoom levels "10 s", "30 s" and "2 min" set the width of the view around the playhead
(the pressed one shows the current width; a level longer than the source is off); pressing the pressed
level again fits the view to the clip, and Ctrl+wheel over the rail zooms. The source's audio level is drawn in a band under the filmstrip (`GET
/api/projects/<id>/sources/<sid>/envelope`: the peak per bucket of 0.05 s, merged to at most 6000
values for a long source, 0 to 1, `peaks` empty without audio; all audio tracks mixed; decoded once with ffmpeg at low
priority, one source at a time, and cached in `<project>/ui/envelope/`, keyed on the source's path,
size and modification time; the call answers `202` with `{"pending": true}` while the decode runs,
and the page asks again every 5 s). A time ruler runs under the rail. Below it, the events `analyze` and `events`
found (`GET /api/projects/<id>/events`, from `events.json`: sounds such as laughter, on-screen
text such as kill feeds, and camera crashes, with their time and confidence; the model's
descriptions are left out) are marked in the view with a short label, at least 80 px apart, the
most confident first; hover one for its kind, time and confidence. Talk (dialogue) stretches show as "Talk"
ranges labelled with the bit's title. Crashes from `crashes.json` (the `crashes` step) are marked
when their confidence is at least max(0.2, 0.6 x the project's best crash); one within 2 s of a
telemetry crash is not marked twice. A source with no events (also
in a project that has an `events.json`) marks the other moments of the same source that start in the
view instead, with their names, coloured by kind. The thin overview strip under the rail shows the view in the whole source
and pans it (drag, or the arrow keys by a tenth of the view, PageUp and PageDown by a whole view,
Home and End to the ends), and the wheel over the rail pans it. The
rail follows the playhead during playback. On the rail, the arrow keys move the playhead by 0.5%
of the view (at least 0.05 s), and PageUp and PageDown by a tenth of the view. `I` and `O` (or `[` and `]`) set the clip's in and out points at the playhead (unless Single-key shortcuts is off in Settings > Keyboard), anywhere in the source,
and the rail widens to show it; they work whenever no form field (sliders excepted), menu, dialog or column splitter has the focus. The gold start and end edges on the rail are handles: drag them (a drag held at
the rail's edge pans a zoomed view), or Tab to one and move it with the arrow keys (0.1 s;
Shift+arrow 1 s), PageUp and PageDown (5 s), Home and End (as far as it can go). A handle keeps
the clip at least 0.1 s long and inside the source, and the preview seeks to it. Space still
plays and pauses while a handle has focus. Outside the clip the rail is dimmed, and each handle is a slider that reads its time aloud. The handles edit the raw range. When the export cuts a clip
elsewhere (an event tail, a smart tail, a speech-aware move or Extend), the rail and the overview show
the difference as a shaded, dashed strip beyond the markers, and the line under the video adds it, for
example "Cut 0:10.0 → 0:16.0 | 6.00 s +10.22 s event tail exports as 16.22 s". That line shows the cut in, the cut out, the
length, any tail the export adds and the exported length; there are no start and end fields, so the handles and
`I` and `O` are the way to trim. "Reset to moment" on the line puts the in and out back to the analyzed times
(it is off when they already match). A trim the clip cannot take (shorter than 0.1 s, or outside the
source) is refused with a message under the line. Clips longer than 10 minutes get a
note there too.
"Play sequence" in the Sequence header previews the cut without exporting: it plays each clip
from its current start to its end, in sequence order, from the sources (or their proxies), with
hard cuts. The clock shows the clip number and the time in the sequence, the playing row is
marked, and trims made meanwhile apply at once. Clicking a row jumps to that clip, Space pauses
and resumes, and Escape or "Stop" ends it; removing the playing clip also stops it. A clip whose
proxy is still being built is waited for, and one whose preview cannot be built is skipped.
The "Clip", "Cut" and "Sequence" buttons in the bar over the video switch what the stage shows: Clip is the
selected clip with its trim, Cut is the join of two neighbouring clips (see below), Sequence is
"Play sequence" (the same, so Escape and Stop end it, and the switch follows the playing state).
Under the rail, the Storyboard shows the
sequence as one block per clip, as wide as the clip's exported length and coloured by kind. Click a
block to open that clip for trimming, drag it onto another or press Alt+Left or Alt+Right to
reorder (undoable, like the Sequence list), and double-click it or press Shift+Enter to play the
sequence from that clip. Arrow keys move between the blocks. The stage title shows the open clip's number in the sequence, and for a clip in the sequence
two buttons beside it, named for the clip before and the clip after ("‹ Clip 1", "Clip 3 ›"; "Previous" and "Next" while there is none), open them.

Cut mode needs a sequence of at least two clips; with fewer, pressing "Cut" says so under the
stage. It shows the cut after the selected clip (the last clip shows the cut before it) in place of the
player, the trim rail and the Storyboard: the last frame of clip 1 and the first frame of clip 2
with their out and in times, tails included. "Previous cut", "Next cut" and the Cuts strip move
between all cuts; in the strip, the arrow keys, Home and End move the focus and Enter opens a
cut. Opening a moment outside the sequence, or leaving fewer than two clips, returns to Clip mode.
The `I`, `O`, `[` and `]` keys do nothing here.

"Play across the cut" (Space) plays the last 3 s of clip 1, then the first 3 s of clip 2, each as
it exports, tails included; a shorter clip plays whole. The player takes the place of the two frames
while it runs, and Escape or "Stop" ends it. It follows this cut's transition: a Fade is previewed
as a crossfade, a Cut as a hard cut. The Export panel's Video transition applies to the whole video
and is the default of every cut; each cut's own choice (Cut, Fade or Default) is set in the Cuts
strip and kept per clip pair, so it stays when the clips are reordered and returns when the pair is
next to each other again. Default follows the Export panel's Video transition. At export the UI
passes the choices to `assemble --joins`.

"Roll the cut" moves clip 1's out and clip 2's in by the same amount, so the sequence keeps its
length, also for clips from different places. Drag the join (up to 3 s each way; the waveforms show
3 s either side of the cut) or Tab to it: the arrow keys move it 0.1 s, Shift+arrow 1 s,
PageUp and PageDown 5 s, Home and End to the limit. The roll stops when either clip would
be shorter than 0.1 s, when clip 1 reaches the end of its source or clip 2 the start of its
source, and a line under the track says which. It does not stop at other clips: when clip 1 or 2
runs into a third clip of the same source, the same line says so ("Clip 1 repeats part of clip
3"). A drag is one undo step, a run of key presses within a second too, and a roll needs saving
like any trim. Tails are worked out again after a roll, so the exported length can change slightly.

Original footage and export quality are unchanged. The server binds only to `127.0.0.1`;
remote access is not supported.

### AI Chat

Lengths: the model sees each clip's length as it will export, with its room in the source, the
sequence total and the length you asked for in your message ("~4 min", "240 s", "1:30"). A
proposal more than 15% over or under that length is sent back to the model up to 3 times (a summary
that claims a change its ops do not make is sent back along with it, once; otherwise Cutpoint
rebuilds the summary from the changes). A sequence length the summary claims ("now about 34 s", "a
total of 90 s") is checked against the exported length, with or without a length in your message:
a claim more than 15% off goes back with a proposal sent back for its length, else the summary is
rebuilt. The closest valid one that is not more than 15% over is offered if the retries miss, and a
proposal more than 15% over is never shown; the chat shows such an attempt as a muted line.
A trim that lengthens a clip ends it at the end of the phrase spoken there.

Auto-fill: when the proposal you are shown would still be more than 15% short of the length you
asked for, Cutpoint fills it in code. It lengthens clips that end mid-speech or are under 5 s
(at most +10 s per end, to the end of the phrase), then alternates between adding one moment from
each of the assistant's searches in turn and lengthening the next clip, until the length is close
enough and never more than 15% over. The added ops are marked "Auto-filled" in the preview and
the summary says how far it got. Without search results it adds the best-scoring unused moments, one in turn from each kind
(view, speed, laugh, ...) already in the sequence. If the assistant ends its turn with only
text while the sequence is still short of the length you asked for, Cutpoint builds this fill
from the current sequence itself and shows it next to the text, labelled "the assistant didn't
propose an edit" (not for a question). A proposal that
removes half or more of the clips when you did not ask to remove any is sent back and never used.

Mixing kinds: when your message asks to break up, mix in or intersperse kinds ("an occasional
funny moment to break up the riding", "sprinkle in", "now and then", "alternate", "vary the
pace"; an audio mix is none), Cutpoint reorders the final proposal in code. Each row is riding
(fast, speed, crash, cruise, crew), scenic (view, beauty), funny (laugh) or dialogue (spoken
words over a clip that is not riding); the scenic, funny and dialogue clips are spread evenly
between the riding runs, each kind keeps its order and a scenic first clip stays first. If you also
name dialogue (dialogue, conversation, banter, commentary, talk or chat moments) and the sequence has
none, up to 3 dialogue moments are added (the assistant's searches first, then the best unused
ones), within 15% of the length you asked for. When one does not fit, it replaces the
lowest-scoring riding clip of a similar length among those the auto-fill added this turn; clips you
or the assistant picked are never removed, and with none to swap no dialogue is added. Negated
wording ("don't break up the riding", "no dialogue") and a fade or transition between clips do not
count. These are `move`, `add` and `remove` ops marked
`auto` (one `order` op when a sequence needs more moves than the 100-op cap allows); the preview marks the rows "Auto-spread" and the summary says how many clips of each kind
were spread. Without such wording the order is the assistant's. `move` and `add` positions
written as `3.0` or `"3"` count as 3.

The AI Chat tab talks to the model set in the [`assistant`](configuration.md#assistant) section of
`config.json`, chosen in the Models tab (Chat role) as a Provider with a model; no base
URL is typed:

- **None (local Ollama)**, the default: the configured Ollama (`OLLAMA_HOST` or `ollama_url`).
  Choose a pulled model, or Other model… for any id; Default uses the first pulled of `qwen3.5:9b`, the vision model and `qwen3-vl:8b`, else `qwen3.5:9b`.
- **Anthropic API**: `https://api.anthropic.com`, with an API key; `claude-sonnet-5-5` (default)
  or `claude-opus-5-5`. Not the Claude Code subscription below.
- **OpenAI API**: `https://api.openai.com/v1`, with an API key; choose an OpenAI model. Cutpoint
  asks `api.openai.com` through OpenAI's Responses API (`/v1/responses`), so every current model
  (`gpt-6-astra`, `gpt-6.1-sol`, `gpt-6.1-terra`, `gpt-6-luna`, `*-codex`) makes tool calls. Any
  other base URL (OpenRouter, LM Studio, vLLM, llama-server, Ollama's `/v1`) is asked through
  `/v1/chat/completions`. Not the Codex subscription below.
- **Grok (xAI)**: `https://api.x.ai/v1` (OpenAI-compatible); choose an xAI model.
- **Claude Code subscription**, **Codex subscription** and **Antigravity subscription**: your
  login (`cutpoint login NAME`, or Connected accounts in the Models tab), no API key. **Use at your
  own risk**. Each lists the models the account itself offers, once logged in and after you agree
  to that provider in the tab; nothing is requested before consent.

All hosted API and subscription providers display **Use at your own risk**. Ollama/Local
does not show this warning.

A base URL set in `config.json` that no preset has (LM Studio, vLLM, llama-server, OpenRouter, an
Ollama on another machine) shows as "Custom (from config.json)" and keeps working: its base URL
is read-only there, and its model can still be changed
([`assistant`](configuration.md#assistant)).

The Model dropdown offers the models the provider lists. Ollama
lists its pulled models (`/api/tags`); Anthropic, OpenAI and xAI list theirs (`/v1/models`, with
the key) only once the key is set and you agreed to that provider and host in the AI Chat tab
(see below); until then nothing is asked and the field says why. A preset other than the saved
provider uses only its own key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY` or its
keyring entry), never `CUTPOINT_ASSISTANT_API_KEY`, which is only for custom base URLs.
Models that do not chat here are left out by name: ids starting with `text-embedding-`, `tts-`,
`whisper-`, `dall-e-`, `omni-moderation-`, `text-moderation-`, `gpt-image-`, `sora-`,
`davinci-` or `babbage-`, or containing `-tts`, `-transcribe`, `-realtime`, `-audio`, `-image`,
`imagine` or `codex` (OpenAI and xAI; `codex` ones stay at `api.openai.com`). A list is
kept for 10 minutes; Refresh list asks again. The Model dropdown offers the listed models; when
the list cannot be loaded it offers the preset's own models (Claude's two, OpenAI's four and xAI's five current ones, Ollama's `qwen3.5:9b`) and the saved model,
and "Other model…" shows a Model id field to type any other id.

Saving there checks that the provider answers (reachable, a model chosen and, at
`api.openai.com`, `api.x.ai` and `api.anthropic.com`, the key accepted) and says why when it
cannot. A provider whose host is not this machine is not contacted before you agree to it in
the AI Chat tab, with or without a key: until then Save sends nothing, says only whether a key
is missing, and notes that the check waits; the check
runs when you save again after agreeing, and the first message checks it too. Another
host (OpenRouter, a LAN server) may list its models without a key, so its key is checked when you
send the first message. The API key is read from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or
`XAI_API_KEY` when `cutpoint ui` starts, then from the system keyring (the Models tab has Save key and
Clear key), then, only for a custom base URL, from `CUTPOINT_ASSISTANT_API_KEY`.
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `XAI_API_KEY` go only to `api.anthropic.com`,
`api.openai.com` and `api.x.ai`, and `CUTPOINT_ASSISTANT_API_KEY` never goes to any of those
three; a key for another host (OpenRouter, a LAN
server) is saved for that base URL or comes from `CUTPOINT_ASSISTANT_API_KEY`, and goes only over
`https://` ([keys](configuration.md#assistant)). The Models tab only says "set (env)", "set
(keyring)" or "not set"; for Vision, a key that comes only from an environment variable shows
"env (CLI only)": analyses started from `cutpoint ui` get no key variables and use the key saved in
the Models tab, so "Allow vision requests" waits for a key saved there. The page never receives a key. To install Ollama and its models, see
[getting started](getting-started.md#optional-ollama-for-frame-rating-and-ai-chat).

A provider whose host is not this machine (`localhost` or a loopback address; an Ollama on
another machine counts too), or whose URL is on this machine on the port of the Cutpoint
[proxy](#proxy), which forwards to hosted models (the running proxy's port, else `CUTPOINT_CLIPROXY_PORT`, the port saved at its last start, or 18317), is hosted: before the first message the tab asks once, per provider
and host (with its port when that is not the default), to send the project's analysis results (moment labels, scores and detection signals, vision
descriptions, transcript snippets, file names (not full paths) and times) and your messages, never video
frames unless you allow them for a project (below), and never audio. The answer is kept in
`config.json` (`assistant.consent`, as `assistant-v3:<provider>@<host>`); it also covers the whole transcripts `analyze` sends to a hosted chat model for its [dialogue reader](#dialogue-reader). It never covers the
Vision role, which asks separately. Until then nothing is sent. A hosted reply is not streamed,
so Cancel takes effect when it arrives.

**Watching a moment.** With a model that takes images, the assistant can look at a moment when
the labels, transcripts and scores cannot answer (for example "show me the fish" or "which of
these has the best view"); it is told never to watch every moment. Each watched moment sends one
image, a timeline sheet: 6 frames spread across the clip (9 for a clip longer than 12 s), three
per row, each stamped with its number and source time, about 1.5k image tokens on Claude, with
the moment's transcript lines and the sounds heard in it (from `events.json`, such as laughter
or a scream) as text. Under 24576 tokens of context a sheet is half-size (about 750 image tokens
on Claude): a message can watch at most 1 moment with a context of 8192 tokens or less, 2 under
24576 tokens, and 4 with a larger one; the steps show "Watched … (6 frames)". Sheets are
cached in the project's `ui/assistant/sheets`, per moment, clip window, sheet size and source file.

- Which models: an Ollama model when Ollama lists `vision` for it (`qwen3.5:9b` does), every
  Claude model, and any other model unless its name has `gpt-3.5`, `o1-mini`, `o3-mini`,
  `grok-3`, `grok-code` or `grok-build`, or it is a `grok-2` model without `vision` or `image`.
  Whether a model of another server (OpenRouter, LM
  Studio, vLLM, a subscription) takes images is not known in advance: it is offered the tool,
  and if its server refuses the image, the request is sent again without it and that model is
  not offered the tool again until `cutpoint ui` restarts.
- Consent: an Ollama on this machine (`localhost` or a loopback address, except on the
  Cutpoint proxy's port) needs none. Any other
  provider gets frames only once you allow that provider and host for that project. When the
  model wants to watch without it, nothing is sent, the model is told to answer from the text,
  and the chat shows a box: "Allow for this project", or "Allow and ask again", which also sends
  your message again. The answer is kept in `config.json` (`assistant.frames_consent`, per
  project folder, as `frames-v1:<provider>@<host>`); the AI Chat consent above is still needed,
  and neither it nor the Vision consent covers frames. To take it back, remove the entry.

Its header shows the model and provider (local, or the hosted host), with a note
that a small local model can miss moments or misread a request. Type a request and press Enter
or Send (Shift+Enter starts a new line); the reply shows the steps the model took ("Searched
for …") and can be stopped with Cancel. The chat needs an analyzed project.

The model reads the moments as text: label (a generic one shown as what the vision model saw,
as in the Moments list), score, times, transcript snippet and, when analysis has them, `kind`,
`seen` (the vision rating label and the description of the `events` VLM window over the moment),
`heard` (laughter, from `signals.events.laughter` of 0.12 or more, or screams detected) and `cues`: visual, telemetry and vision-tag strengths on
one 0-10 scale (`beauty`, `people`, `group`, `speed`, `carve`, `airtime`, `crash`, `wipeout`,
`stopped`, the action-cam tags such as `transit` or `scenic_vista`, and more; the strongest six
per moment in the list, all of them in `get_moment`). Its `find_moments` tool searches labels,
transcripts, `seen` text, heard sounds and the names of cues of 5 or more (`air` finds `airtime`
and `jump_air`). The instructions name only the cues the project has, and requests such as
"beauty shots", "casual skiing" or "bloopers" work without moment ids only when it has the cues
for them: action-cam footage with telemetry (`--style action-cam`), the `events` vision layer
(window descriptions and tags) or visual-pool moments. Other styles have the vision rating
label (with `--vision`), `loudness`, `laughter` and the laughter or screams heard, so the model
finds their moments by those, the label and the transcript. Only these results are sent; video
frames go only through view_moment (above), once allowed for the project.

`look_for` searches the footage itself, with every chat provider, local or hosted. It compares a
short English description ("powder spray", "sunset over the peaks") with the cached frame
embeddings of every source of the project (the `events` visual layer, about one frame per second,
the same ones the visual candidates read; written by `analyze` for `action`, `sport` and `action-cam`
footage, and for the other styles only with [`--visual-index`](#analyze)) using local CLIP, so it also finds footage the scorer
never proposed. A project none of whose sources has them is not offered the tool, and a call that
still fails shows the page "Visual search isn't available for this project: its analysis has no
frame index (analyze --visual-index writes it)". It decodes nothing and downloads nothing, and no frame leaves the machine: the
model gets text only, the best windows as moment rows with `match` 0-10 and `confidence`:
`strong` from a CLIP cosine of 0.222, `weak` from 0.202 (a few at most, so a query the footage
may answer, such as "fish" in a reef, still lists its best guesses, while one it cannot answer
lists nothing or a few weak ones). A window that a moment or a row of the sequence covers is that
moment. A clear window nothing covers (cosine 0.26 or more) comes back as a new moment (`new:
true`, labeled with the query, an id of the same kind as an analysis moment) that the proposal can
add and that is saved with the proposal. Any other uncovered window is `unconfirmed`: it has an id
so it can be watched, but it is never a new moment by itself, and `find_moments` and the moment
list do not include it. The instructions tell the model to confirm a weak or unconfirmed window
with `view_moment` when the chat model takes images, and otherwise to tell you it is uncertain,
never to present it as confirmed. `propose_edit` refuses to add an unconfirmed window (also in a
suggestion) until `view_moment` has shown it; the chat lists each one as "possible" with an
"Add to sequence" button, and your click is the confirmation (it is a normal edit that Ctrl+Z
undoes). The instructions say similarity is
not certainty: a match is likely or possible, never sure. It needs the full edition (Cutpoint lite
says "not available in Cutpoint lite" with the install command), the CLIP weights (`cutpoint
setup --only visual`) and a cached visual layer (`analyze` writes it for the `action-cam`,
`action` and `sport` styles); without one it says so and the model falls back on `find_moments`.

- Moments the model points to can be highlighted in the Moments list ("Highlight in list") or
  the list filtered by the model's search words (offered only when the list's search, which also
  matches vision labels, finds one of them).
- The chat shows what the assistant did as one compact summary line (for example "Searched 3 times
  · 30 moments found") with a Details toggle, and a proposed change as a card: its summary, one chip
  per change, a strip of the sequence after it, and the buttons Apply, Preview and Discard. Apply works
  straight from the card, without opening the preview first; if the sequence changed since the
  proposal was made, it opens the preview again instead. Discard on a card that is not the previewed
  one only marks that card Discarded. Preview opens the change in the Sequence tab (below).
- A new proposal is previewed in the Sequence tab as soon as it arrives; Preview shows it again: each row is
  marked Added, Moved (old to new position), Trimmed (old to new times) or Renamed, removed clips are
  listed as Removed, the moments are marked in the Moments list, and the trim rail shows the new
  start and end as dashed edges. Apply replaces the working sequence (Save keeps it, Undo takes
  it back); Discard drops the preview. Up to three alternative suggestions are previewed the same
  way. If the sequence changed since the proposal was made (a note says so; an applied
  proposal's button reads "Review again"), Apply makes the change on the current sequence and
  says so, or tells you to discard it and ask again when it no longer fits. While Apply waits,
  Discard is off and a new proposal from the chat is previewed after it.
- A proposal can also set Color (the Color slider of Export clips and One-Click Edit, #496):
  `off`, `auto` or `underwater` with an optional strength. The preview names it ("Color: Auto,
  strength 80% (now Off)"); Apply moves the sliders, which are saved for the project as if you had
  moved them; Undo and Redo put the sliders back and forward with the sequence, and a change that
  did not set Color never moves them. The instructions say `underwater` only grades underwater
  footage (#495), so the assistant proposes it only when the footage is underwater. Applied
  changes that set Color list it, and Revert puts the earlier Color back.
- "Applied changes" lists the changes applied in this project; "Revert this change" previews a
  change that puts the sequence back as it was before, applied like any other. If the sequence
  changes after that preview, Apply previews the revert again on it instead.
- "Clear chat" deletes the conversation only; the sequence and the applied changes stay.

Nothing is applied without Apply. API: `GET /api/projects/<id>/assistant`, `POST .../turns` and
`GET .../turns/<turn>?after=<n>`, `POST .../turns/<turn>/cancel`,
`POST .../proposals/<proposal>/applied` (the page sends its `color`, `{mode, strength}`, so a
proposal's `color` op is recorded as `before` and `after`), `DELETE .../chat`, `GET .../history` and
`POST .../history/<change>/revert` (which returns a `revert-<change>` proposal with a `base_key`
and writes nothing; applying it with another sequence or without that key is refused with 409).
`GET .../assistant` also returns `consent` (`needed`, `key`, `host`, `missing_key`: how to add a missing API key or log in to a subscription, or null, and `missing_kind`: `"login"` for a subscription provider, `"key"` otherwise, null when nothing is missing); `POST .../turns` answers
409 while `needed` is true. `POST /api/assistant/consent` with that `key` records the consent.
`GET .../assistant` also returns `frames` (`needed`, `key`: the `frames-v1:` key, null for a
provider on this machine, `host`); `POST .../assistant/frames-consent` with that `key` allows
frames of the project (409 when the provider changed, is on this machine, or the AI Chat
consent is missing). A turn's events can include `frames_consent` (`id`, `key`, `host`) when the
model asked to watch without it, `tool` events named `view_moment` (`id`, `frames`,
`cached`, or `error`: `limit` when the per-message limit was reached) and `look_for` (`query`,
`found`, `new`, `weak`, `unconfirmed`, `possible` (the unconfirmed windows: `id`, `source`,
`start`, `end`, `label`, `confidence`), or `error`: why it could not search, with `reason`, a short
sentence for the page, when there is one); a `proposal` event can carry `color`
(`mode`, `strength` or null) and `moments` (the `look_for` moments its ops add);
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
faster-whisper's transcription device.
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
              [--top N] [--max-total S] [--vertical] [--force] [--no-keep-clips]
              [--format mp4|fcpxml|otio] [--events]
              [--vision | --no-vision] [--vision-top N] [--ollama-url URL]
              [--ollama-model MODEL] [--skip STEP [STEP ...]]
              [--visual-index | --no-visual-index] [--ocr-det-skip | --no-ocr-det-skip]
              [--compress-peaks off|light|strong] [--lut FILE [--lut-strength 0-1]]
              [--color off|auto|underwater [--color-strength 0-1]] [--json]
```

Runs [`analyze`](#analyze), [`clips`](#clips) and [`assemble`](#assemble) in one go with the
settings of a kind, then copies the final video and its subtitles to `<name>-final.mp4` and
`.srt` in the project folder (a hard link where the file system allows it). The clips stay in
the clips folder unless `--no-keep-clips` is given. `<name>` is the
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
| `--style`, `--top N`, `--max-total S` | Replace the kind's value; the others stay (`--kind montage --top 5`: the 5 best that fit in 60 s). `--style auto` picks the style from the footage as [`detect`](#detect) does (the funny style for `funny-moments`), before the `--events` check; unclear footage gets `gaming` when it leans to gameplay, else the General style (`action`; `funny` for `funny-moments` either way) |
| `--out DIR` | Project folder. Default: `<projects>\<name>` (the first input's name, as above) |
| `--vertical` | 9:16 clips in `clips_vertical`, `final_vertical.mp4`, `<name>-final_vertical.mp4` |
| `--force` | Replace the clips of an earlier run, as `clips --force`. Without it a non-empty clips folder stops the run before the analysis |
| `--keep-clips` / `--no-keep-clips` | Keep the clips in `clips` (default). `--no-keep-clips` exports them into a temporary folder in the project, as [`assemble <project>`](#assemble-from-a-project) does, and removes it after the join (after a failure or Ctrl+C too): the project gets `final.mp4`, `final.srt` and the `<name>-final` copies, and an existing clips folder is not touched. Refused (exit 2) with `--kind clips`, a timeline `--format` or `--force` |
| `--format mp4\|fcpxml\|otio` | As on [`assemble`](#assemble): `mp4` (default) makes the final video; `fcpxml` or `otio` writes only that timeline, referencing the kept clips, and copies it to `<name>-final.fcpxml` or `.otio` |
| `--events` | Also run [`events`](#events) with its defaults, without the layers of the `--skip` steps, to re-rank the moments. Needs the `events` extra and `cutpoint setup`; checked before the analysis |
| `--vision` / `--no-vision` | As on `analyze`: rate the default edit and the moments below its cut with Ollama, or not. Default: [`vision`](configuration.md#vision) in `config.json`, else off. With `--style auto` they also govern the footage check of unsure sources (see [detect](#detect)): on unless `--no-vision` or `"vision": false`; `--vision` also warns of a missing consent |
| `--vision-top N`, `--ollama-url`, `--ollama-model` | As on [`analyze`](#analyze): the most moments sent to vision, and the Ollama server and model (also for the VLM layer of `--events`); an invalid port in `--ollama-url` exits 2 (`bad_args`) |
| `--skip STEP ...` | As on [`analyze`](#analyze): leave out `reactions`, `keyframes`, `crashes` and/or `dialogue`. Give it after the input paths |
| `--visual-index` / `--no-visual-index` | As on `analyze`: also write the frame embeddings of the AI Chat's visual search for `gaming`, `funny` and `commentary` footage, or not. Default: [`visual_index`](configuration.md#visual_index) in `config.json`, else off |
| `--ocr-det-skip` / `--no-ocr-det-skip` | As on `analyze`, also for the OCR layer of `--events`. Default: [`ocr_det_skip`](configuration.md#ocr_det_skip) in `config.json`, else off |
| `--compress-peaks off\|light\|strong` | [Peak compression](#peak-compression), baked into the clips (so the final has it too). Default: `compress_peaks` in `config.json`, else off |
| `--lut FILE`, `--lut-strength 0-1` | Grade the clips with a 3D LUT, as on [`clips`](#clips) (so the final has it too). Checked before the analysis |
| `--color off\|auto\|underwater`, `--color-strength 0-1` | [Automatic color correction](#clips) of the clips, as on `clips` (so the final has it too). Not with `--lut`; checked before the analysis |
| `--json` | One [result envelope](#json-output) on stdout for the whole run |

With `CUTPOINT_PROGRESS=1`, `edit` reports 9 [steps](#progress-output): the 7 of `analyze`, then `clips` and `assemble`. `--kind clips` still assembles. With `--events`, `keyframes` is reported again after `vision`, for the VLM layer.

Re-runs reuse the cached analysis. `edit` exits like the other commands (`130` on Ctrl+C, partial
files removed) and logs the final video (or timeline), its subtitles and the clips folder (or
`none (temporary, removed)`) last. To change the
cut afterwards, delete or rename clips (rename a clip's `.srt` with it) and run `assemble`;
that updates `final.mp4` and `final.srt`, not the `<name>-final` copies.

## analyze

```text
cutpoint analyze <file|folder> [<file|folder> ...] [--style STYLE] [--out DIR] [--json] [options]
```

Transcribes the audio, scores it on a 0.1 s grid, runs the analysis steps below, and writes
`highlights.json` and `transcript.txt` to the project folder. Inputs may be any mix of files and folders; they are
de-duplicated and kept in the order given. Each source has its own cache, so adding one file to
an analysed set processes only the new file.

| Option | Meaning |
|---|---|
| `--style` | `gaming` (default), `action`, `sport`, `funny`, `commentary`, `action-cam` (`ski` is not a style), one of your own styles ([`cutpoint style`](#style)) or a learned preset, or `auto`: pick one from the footage, as [`detect`](#detect) does. `auto` never picks `sport` or `funny`; unclear footage gets `gaming` when it leans to gameplay, else `action`, with a `style_ambiguous` warning. `highlights.json` keeps the style it picked in `style`, `settings.style_requested` is `auto` and `detection` holds the evidence, so `events` and re-runs keep it without detecting again. `action-cam` (or a style of yours over it) on footage where no source has camera telemetry runs as `action`: `settings.style_requested` names the style given and `settings.style_fallback` is `{"from": <style>, "reason": "no telemetry"}`, with an `action_cam_no_telemetry` warning; a re-run asks for the given style again |
| `--out DIR` | Project folder. Default: `<projects>\<name>`: a file's name without extension, or the folder's full name (`.` resolved; `project` for a drive root); for several inputs, their common parent folder's name |
| `--exclude GLOB` | Skip files whose name or path matches (case-insensitive, repeatable) |
| `--recursive` / `--no-recursive` | Search sub-folders of folder inputs (default: no). A folder whose full path matches `--exclude` (`*/Captures/*` or `*/Captures/`) is skipped completely |
| `--vision` / `--no-vision` | Rate the default edit and the moments below its cut with the [`vision_assistant`](configuration.md#vision_assistant) model (Ollama by default), on their frames and, except for `action-cam`, the speech in them, or not. Default: [`vision`](configuration.md#vision) in `config.json`, else off for every style; `--no-vision` skips it when `config.json` turns it on. With `--style auto` they also govern the footage check of unsure sources (see [detect](#detect)): on unless `--no-vision` or `"vision": false`; `--vision` also warns of a missing consent. See **Vision rating** below |
| `--vision-top N` | The most moments sent to vision, those closest to the cut first. Default: the edit's moments (not must-include ones) plus max(2, 30% of its moments) below its cut, at most 24 |
| `--ollama-url`, `--ollama-model` | Defaults: `vision_assistant.url` (Vision role `ollama`) / `OLLAMA_HOST` / `ollama_url` / `http://127.0.0.1:11434`, and `CUTPOINT_VISION_MODEL` / `vision_assistant.model` (Vision role `ollama`) / `qwen3.5:9b` (`qwen3-vl:8b`, the earlier default, still works with `--ollama-model qwen3-vl:8b`). With a [`vision_assistant`](configuration.md#vision_assistant) of any provider other than `ollama` in `config.json`, the frames (here and for `360 director --vl-check`) go to that provider instead: an API (`anthropic`, `openai`, `xai`, with your API key) or a subscription (`claude-code`, `codex`, `antigravity`), once you agreed to it in the Models tab ("Allow vision requests"), unless you give `--ollama-url` and/or `--ollama-model`: then that Ollama is used instead (a flag left out takes its default) and nothing goes to the hosted provider. An Ollama URL that is not this machine (`localhost` or a loopback address, except on the Cutpoint proxy's port) needs that Vision consent too; without it vision is skipped with a WARNING (`vision_consent_needed`). An `--ollama-url` with an invalid port (such as `myhost:99999`) exits 2 with `bad_args`; a bad port in `OLLAMA_HOST`, `ollama_url`, the Models-tab Vision URL (`vision_assistant.url`) or the AI Chat Ollama URL only skips the step |
| `--model NAME` | Whisper model. Default `medium` on CUDA, `small` on CPU |
| `--language en` | Force the transcription language (default: auto-detect). A forced language is transcribed and cached separately (`transcript-<model>-<language>.json`), and `clips`, `extend` and `assemble` subtitle the project from that transcript |
| `--skip STEP ...` | Leave out `reactions`, `keyframes`, `crashes` and/or `dialogue` (give it after the input paths). Default: [`analysis.skip`](configuration.md#analysis) in `config.json`, else none |
| `--visual-index` / `--no-visual-index` | Also write the frame embeddings (the CLIP visual layer, in the `keyframes` step, before the HUD text for `gaming` and `funny`) that the AI Chat's [`look_for`](#ai-chat) searches, for `gaming`, `funny` and `commentary` footage; `action`, `sport` and `action-cam` footage always has them. They change no ranking. Sources with telemetry: the crashes step also reads them, so `crashes.json` can differ; the ranking does not. Like that layer for the other styles it needs the `events` extra, its model files and a CUDA GPU, and is left out with an INFO line otherwise (the HUD text of `gaming` and `funny` still runs). Cached per source like the rest: a later run with the option on computes only what is missing, and one with it off leaves what exists. Default: [`visual_index`](configuration.md#visual_index) in `config.json`, else off |
| `--ocr-det-skip` / `--no-ocr-det-skip` | Gameplay HUD text (the `keyframes` step): skip the text detection of a HUD region that has not changed since detection last ran there, and keep that result. Faster on footage with a still HUD; on two test clips it saved a third and a tenth of the detection calls and lost no event. A read with it is cached apart from one without it, so switching reads the footage again once. Default: [`ocr_det_skip`](configuration.md#ocr_det_skip) in `config.json`, else off |
| `--no-scenes` | Skip scene detection |
| `--max-candidates N` | Candidates per source before merging. Default: enough to cover about 3 times the style's target length, at least 30. `gaming` has an open list instead: see [Candidates](#candidates) |
| `--recover` | After a read error, resume analysis past the bad region |
| `--must-include-crash CONF` | Crashes in `crashes.json` with at least this confidence, 0 to 1, become must-include moments with their reaction: they replace the moments they overlap and rank first. Every style; default 0.6; 0 forces none. Weaker crashes at or above the project floor (max(0.2, 0.6 x the strongest crash)) are ranked like any moment, and one overlapping a moment only marks it. See [Crash moments](#crash-moments) |
| `--batch-size N` | Close each model (Whisper, the step models, the background HUD OCR engine and the `--vision` rating's Ollama model, which then rates the candidates of N sources at a time) after N sources and load it again for the next ones, to give memory back on long runs; the results are the same (once the vision time cap is reached, which moments near the cut get rated can differ). Default 0: keep it loaded. Also on `360 analyze` (its view layer, analysis and events) |
| `--json` | Result as one [JSON envelope](#json-output) on stdout |

**Vision rating.** The model rates the moments of the default edit (what `clips` takes
without options) and the k moments just below its cut in the ranking before vision, k = 30%
of the edit's moments, rounded up, at least 2 and at most 24, so a rating can drop a weak pick
from the edit or bring in one from below. A moment above the cut that the edit skipped for its
length or because it overlaps a pick, must-include moments and visual candidates are not rated. The moments closest to the
cut are rated first: the last pick, the first moment below the cut, the pick before it, the
second below, and so on. The ratings of each prompt kind (with
speech, frames only) are centred on their mean in this run, so a rating means better or worse
than the other moments rated; with fewer than 5 ratings of a kind the mean is pulled toward a
fixed prior. The centred 0-10 rating scales a candidate's score (5 leaves it as is, 10 lifts it
and 0 lowers it by a share set per style, about 20-45%, capped at 100); candidates not rated keep
theirs. A rated moment never passes an unrated one, except moments the default edit skipped
above its cut when vision ran (too long or overlapping a pick): its final score,
event points included, stays between the scores of its nearest unrated neighbours in the
ranking without vision (a tie that would sort it past the neighbour becomes 0.1 inside instead,
unless the two neighbours tie), so rated moments can only change places among themselves and
with those skipped moments; with the whole edit rated, a weak pick can fall to just above the
first unrated moment under the band. A skipped moment is marked `vision_skipped` (its `start`
and `end` then) and counts as a neighbour again once `events` widens or trims it. The step stops
asking the model once it has taken [`vision_time_share`](configuration.md#vision_time_share)
of the analyze time before it (default 30%; a re-rank measures against the first analysis).
It makes at least one call, the call under way finishes (so it can run over by one call), and
moments with a cached rating are still taken; one INFO line says how many moments near the cut
were not rated.
The model's description goes to `vision_label` and never changes the label or the clip name. A local model is unloaded
from GPU memory right after rating (with `--batch-size N` also after the candidates of every N
sources). Ollama ratings are cached per source
(`vision-ratings.json`), so a rerun asks only about candidates not rated and repeats the same
scores. The cache also records the model's digest: a model pulled again under the same tag is
rated again, and with Ollama down the cached ratings for the tag are used.

With the generic styles (all but `action-cam`), a candidate with speech is rated on its frames
together with what was said in it (the first 240 characters of its `transcript`), on local and hosted
models alike, so speech that fits what is on screen can raise the rating. A window with no speech
is rated on its frames alone, as before. The speech prompt rates higher than the frames-only
one, so each kind is centred on its own mean before it counts (see above), and talk does not
beat silent gameplay just for being talk. `signals.vision` keeps the raw rating. `action-cam` is unchanged. A rating
with speech has its own cache key: those windows are rated again once, then cached.

Signals combined: loudness (EBU R128 against the clip's own baseline), speech keywords from the
transcript, laughter, scene-cut density and speech continuity. The `laughter` signal counts
laughter written in the transcript: Whisper's non-speech markers (`[laughter]`, `(laughs)`,
`*laughing*`, in brackets, parentheses or asterisks) and interjections such as `haha` and `lol`;
the spoken word "laugh" or "laughter" does not count. Laughter heard in the audio is
`signals.events.laughter`, a separate value. Whisper may write such a marker (the default token
suppression that blocked `[`, `(` and `*` is off); the marker is taken out of the transcript, the
subtitles and the captions and only counts here. Existing transcripts have none. Likely Whisper hallucinations
(a short phrase repeated, very low confidence) are dropped first. Boundaries are snapped to
scene cuts (at most 1 s away) and never cut a word. A style whose `scene` weight is 0 (`funny`,
`commentary`, or one of your styles that sets it to 0) uses scene cuts only for that snapping, so
`analyze` does not decode the whole video for them: once the candidates are picked, it detects
cuts only in windows of 2.5 s either side of each candidate edge, cached per source as
`scenes-windows.json` next to `scenes.json` (later runs decode only the windows not covered yet).
The windows always decode on the CPU, also when a GPU decoder (NVDEC, VAAPI) is in use: each is a
short ffmpeg run, and a GPU decoder's start-up costs more than it saves there. A source over 30
minutes keeps the full pass, which reads keyframes only there and is faster than the windows.
A full `scenes.json` already in the cache is used instead, and a style with a scene weight runs
the full pass as before, also on a project analysed with a style without one. The `scene` signal of such a style's candidates is 0. Each signal is a named feature, and a style's weights name
these features. The `score` is a candidate's weighted peak divided by the weights of the
features this install can compute. A feature that needs a missing extra (the `events` extra
in Cutpoint lite) is left out, so the built-in generic styles (gaming, action, sport, funny,
commentary) score the same on lite and full installs. Action-cam styles are renormalised the same way:
on lite their `events` features (`vlm`, `voice_laughter`, `beauty`, `scenic`, `people`, `group`,
`voice`) are left out, and a renormalised score is capped at 100. `social` counts on lite too:
there it takes the laughter written in the transcript; full installs also take the laughter,
screams and cheering heard (`events`).
The features it could compute are in `settings.features_available`. Action-cam candidates' `signals`
name the action-cam features; `voice_laughter` (laughter heard over the wind noise) was `laughter`
before #514 phase 1b.

Each candidate in `highlights.json` has `id`, `source`, `start`, `end`, `score` (0-100),
`signals`, `label` and a `transcript` snippet; `transcript_unsure` lists the snippet's words
Whisper heard with confidence below 0.5 as `[index, confidence]`, the index counting words in
`transcript.split(" ")` (omitted when none is unsure or the transcript has no confidences). With
`--vision`, a rated candidate also has the model's own description in `vision_label` and
`vision_rated` (the `start` and `end` it rated, the score before and after the rating, `base`
and `blended`, the prompt `kind`, `speech` or `frames`, and its `centre`). Vision never changes a moment's `label`, so its clip gets the same name with and
without `--vision` (the rating can still change which moments are picked and their order). The rating counts only
while the candidate keeps that span: once `events` widens or a menu trim cuts it, the rating
moves to `vision_stale` (`score`, `label`, `window`), `signals.vision` is `null` and `vision_label`
goes; the span coming back restores them. A moment the default edit skipped above its cut when
vision ran has `vision_skipped` (its `start` and `end` then; see **Vision rating**). A candidate the dialogue reader widened also has `widened_from` (its
earlier `start`, `end`, `transcript` and `transcript_unsure`, and `absorbed`, the full candidates
merged into it), and `dialogue_description` (the stretch's description) when no transcript
snippet could be rebuilt for its new span; where the reader ran, `signals.events.dialogue` is
the strength of the stretch that boosted it. `timings` holds the seconds each step took and `total_s`, the whole
analysis; a re-rank of an earlier analysis (`events` on action-cam projects, `360 analyze`,
`crashes --reanalyze`) keeps that `total_s` and records its own time as `rerank_s`.
`timings.sources` breaks the same run down per source (one entry per recovered part, with its
`offset`): `source` (the path as in `sources[].path`), `probe_s`, `audio_extract_s`,
`loudness_s`, `transcribe_s`, `scenes_s`, `ocr_wait_s` (seconds the analysis waited for the
background HUD text read of that source; 0 without one) and `cached` (its audio and transcript
were reused), `overlap` (`true` when its scene cuts ran alongside Whisper with
`CUTPOINT_ANALYZE_SCENE_OVERLAP=1`; omitted otherwise) and `scene_windows` (`true` when its
scene cuts were detected only around candidate edges, see above; `scenes_s` is then the time of
those windows; omitted otherwise). The sums above are unchanged; the log also marks the start of each
source's scene cuts. After a re-rank, `sources`, like the step timings, describes the re-rank's
own pass, usually all `cached`.
`transcript.txt` has one line per pause or sentence: `[Round1.mp4 00:12:31.4-00:12:38.9] text`.

Visual candidates (`action-cam`, `action` and `sport` styles, and learned styles based on
them). The scorer ranks scenery and steady riding low, so `analyze` also lists a few of them
after the scored candidates, from the cached visual layer, action-cam grids and GPS speed
(no extra decode): entries with `"pool": "visual"`, ids `v001`, `v002`, ... and a `kind`:
`view` (a beauty peak of at least 0.5, 5-7 s, label "scenic view / beauty shot"), `crew`
(people close to the camera, 4-6 s, "group / friends shot") or `cruise` (8 s of a steady
speed of at least 8 km/h with no jump, crash or wipeout, "casual cruising run"; only with GPS
speed). None is on a lift, standing still or within a `crashes.json` crash; per source at most
2 of a kind and 5 in all, and none overlapping a scored candidate by half of the shorter clip.
Their `score` is the peak times 10 (0-10, display only; the peak is `signals.peak`, 0-1);
they are not in `suggested_top`, not re-ranked by `events` and stay last.
`settings.visual_pool` has their `count` and `kinds`.

When the `vision` step ran, `settings.vision` holds `requested`, `failed` and `uncached` (when
offline) and what it cost, summed over the run (no list per moment): `rated` (moments with a
rating, `cached` of them from the cache), `calls` (model requests, retries included; a failed
request counts as a call with its wall time), `vision_s` (the step's seconds), `pre_vision_s` (the
analyze time before it), `grab_s` (frame grabs), `load_s`, `prompt_s` and `reply_s` (Ollama's own
timings of model load, prompt reading and reply), `prompt_tokens`, `reply_tokens`, `model` and
`provider`. A re-rank of an earlier analysis adds `"rerank": true`: its `pre_vision_s` is only
the cached re-run, not the full analysis. The band and the cap add `cut` (the default edit's
cut, 1 + the rank of its last pick before vision), `band_k` (moments below the cut),
`rated_by_kind` (`speech` and `frames` counts), `centre` (`speech` and `frames`, `null` for a
kind not rated), `share` (`vision_time_share`), `budget_s`, `budget_base` (`pre_vision_s`, or
`earlier_total_s` for a re-rank), `capped` (the cap stopped the calls), `skipped_by_cap`
(moments it left unrated) and `clamped` (rated moments whose score the clamp changed; each
`events` rescore updates it). `requested` is the band's size (the edit's moments (not must-include ones) plus
`band_k` below, fewer at the end of the list) after `--vision-top`. A hosted provider reports no timings: its wall time is
in `reply_s`. `-v` logs one DEBUG line per request. `scripts/vision_eval.py` compares such projects (see the script's docstring).

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

Opt-in, off by default: `CUTPOINT_ANALYZE_SCENE_OVERLAP=1` detects a source's scene cuts at the
same time as Whisper transcribes it, instead of after it. It starts only for a source whose
style weighs scene cuts (not one that detects them around candidate edges only), that is not skipped, whose scene cuts are not cached and whose
transcript is about to be made; with the transcript cached, the cuts are detected in their turn
as usual. It helps on long sources where Whisper and scene detection take about as long (a long
talk or screen recording): the step takes about the longer of the two, but Whisper runs slower
meanwhile, since both compete for the machine. Where one of them is much shorter than the
other, there is little to gain. The scene cuts, `highlights.json` (apart from `timings`) and
the caches are the same as with it off. An error in scene detection still stops the run after
the transcription; when the transcription fails or is stopped, the scene detection is stopped
too (its ffmpeg killed or its detection stopped within about 0.1 s plus a frame decode, no `scenes.json` written). In
`timings.sources`, `scenes_s` is then the time of the scene thread, mostly inside
`transcribe_s`, and the source has `"overlap": true`. The overlap does not start for a source whose scene proxy comes from a shared decode
(`CUTPOINT_ANALYZE_SHARED_DECODE=split`). The INFO line "detecting scene cuts" appears when
the transcription starts. Any other value, or unset: as described above.

| Step | What |
|---|---|
| `transcribe` | Whisper transcription, scoring and scene detection |
| `reactions` | PANNs sound events (laughter, screams, cheering). Also the voice layer when `crashes` will run |
| `keyframes` | Game HUD text (OCR). For `gaming` and `funny`, HUD region discovery runs when no preset matches the game; the vision model, when it answers, labels the regions it proposes. For `action`, `action-cam` and `sport`, the CLIP visual layer, only with a CUDA GPU; with `--visual-index` that layer runs for `gaming`, `funny` and `commentary` too (frame embeddings for the AI Chat's visual search, no effect on ranking) |
| `crashes` | Action-cam footage only (telemetry found, or style `action-cam`, `action` or `sport`) |
| `dialogue` | A text model reads the whole transcript and marks talk worth keeping. Open-list styles only (`gaming`). See [Dialogue reader](#dialogue-reader) |
| `ranking` | Re-ranks the candidates with the events, after leaving out those in HUD menu time ([game streams](workflows/game-streams.md)); `highlights.json`, `events.json`, `events.txt` and the other result files are written after the last step |
| `vision` | Ollama frame rating, only when vision is on. The model sees frames only, so a moment with laughter heard in it (`signals.events.laughter` of 0.12 or more) gets a sentence in the prompt saying so |

Nothing new is required. Without the `events` extra or the model files, a step is skipped with an
`INFO` line and nothing is downloaded; `cutpoint setup` and the extra add them. A step that fails
logs an `ERROR`, adds a `step_failed` warning and the run continues without it. A failed HUD
OCR logs a `WARNING` instead, naming the sources left without menu filtering and HUD events
(for out of memory: close other programs, set `CUTPOINT_ANALYZE_OVERLAP=0` or pass
`--skip keyframes`); the sources read before it keep theirs. The VLM
layer is not part of `analyze`; it stays in [`events`](#events) and `edit --events`.

#### Dialogue reader

The `dialogue` step runs for open-list styles (`gaming`) after the reactions and events layers, and is skipped with a reason for the others and for footage with no speech. `--skip dialogue` (or `analysis.skip`) leaves it out. A text model reads the whole transcript of each source in overlapping pieces (8 minutes each, 1 minute shared) and marks stretches of talk worth keeping: `funny`, `out_of_pocket`, `ragebait`, `back_and_forth` or `story`, from the setup to the payoff. Each becomes an event in `events.json` with `kind` `text` and `type` `dialogue` (`label` is the talk kind, `title` 3 to 6 words naming who or what the bit is about ("" when the model gave none), `description` what happens, `details` the model's rating, backend and model). The candidate the stretch nominates, or the one it widens (the most confident stretch's title), gets the title as `dialogue_title` and its slug (without apostrophes, cut at a word) as `label`, so it is part of the clip file name (`dialogue-<kind>` for a nominee without a title, and a widened candidate keeps its label). In `cutpoint ui` the moment is named by the title, in the moments list and in AI Chat. Each stretch also gets a `DIALOGUE` line in `events.txt`. A later `events` run, or an `analyze --skip dialogue`, keeps the dialogue events an earlier run wrote.

The model is chosen once per run:

| Model | Ranked by |
|---|---|
| An explicit `--ollama-url` and/or `--ollama-model` on `analyze` or `edit` (wins over the AI Chat model, hosted or Ollama; a flag left out takes its default; an Ollama on another machine needs the AI Chat consent, else the step is skipped with a `chat_consent_needed` warning and nothing goes to a hosted model) | As the Ollama row below |
| The AI Chat model, when it is a hosted API or subscription and you gave the [AI Chat consent](configuration.md#assistant) (`assistant-v3:`) | The model's own 1 to 10 rating |
| A local Ollama (the AI Chat model when that is an Ollama, else `qwen3.5:9b`), or an Ollama on another machine you gave the AI Chat consent for | Reaction sounds (laughter, screams, cheering) per minute inside the stretch; a stretch with none still counts, at a low baseline |
| A hosted AI Chat model without the consent, and no local model | The step is skipped with a `chat_consent_needed` warning |
| No model | The step is skipped with an INFO line |
| Not enough free memory for the model | The step is skipped with a `step_failed` warning |

The transcript is sent whole to a hosted model, which is why the AI Chat consent changed to version 3 (see [configuration](configuration.md#assistant)). Each source's reading is cached with its transcript, model and prompt, so a re-analysis with the same ones asks nothing.

The stretches feed the candidates:

- A stretch no candidate overlaps (or one too long to widen them) nominates a candidate of its own that keeps the whole stretch (starting 1 s before it), not limited by the style's maximum length.
- A candidate a stretch overlaps is widened to cover the whole stretch, but only when the result (the stretch and every candidate it overlaps) stays within 90 s. Several candidates one stretch overlaps become one, the best scoring; its id and signals stay. **The others stay in `highlights.json` under `widened_from.absorbed` but are no longer candidates, so `--ids` rejects their ids**; a moment you noted by id may be inside another candidate now. A longer stretch is a candidate of its own, over the candidates it overlaps, and those stay as they are. A widened candidate keeps only the part of the stretch that leads into its original moment: where the transcript has no speech for 8 s or more before the moment, it starts after that silence, and it ends before such a silence after the moment (the 90 s limit is checked before this). A re-run widens again from the original bounds.
- A candidate a stretch overlaps gets up to 3 points on top of its other event points, scaled by the stretch's strength.
- A pick skips a candidate that mostly overlaps one already picked from the same source (not with `--ids`), so a clip inside a long stretch is not picked twice.
- `--max-total` (and the `montage` kind) skips a dialogue candidate that is longer than what is left of the cap; it is not trimmed.

`analyze` also writes `events.json` and `events.txt` (and `crashes.json` for action footage).
`highlights.json` is written once, atomically, at the end: a cancelled run leaves the previous
file, and a cancelled first run leaves none. The other result files are written with it, just
before it, each atomically: `transcript.txt`, `events.json`, `events.txt`, `crashes.json`, the
`crash_check\` sheets, `activity.json` and, for `action-cam`, `telemetry.json` and `action_cam_features\`
(older sheets and features of the project are removed at the same time). A failed or cancelled
run, in any step, leaves all of them as they were, and a first run cancelled before its first source is transcribed leaves no project
folder (after that the folder keeps `progress.json`, see [`resume`](#resume)), unless HUD discovery wrote its proposal (`discovery\`, below). A run stopped after the transcription therefore has no `transcript.txt` yet; the rerun
reads the cached transcription. What stays on purpose: the per-source caches outside the
project (audio, transcript, scene cuts, event layers, dense crash checks), so a rerun resumes
where it stopped; a HUD preset found by discovery, installed in the presets folder at once,
with its proposal in the project's `discovery\` folder; and an activity profile written by the
local LLM to the profile library. `analyze` never writes `visual_check\`; only
[`events --sheets`](#events) does. The analysis score stays in each candidate's
`score_pre_events`, and `highlights.pre-events.json` is written only when the analysis was not
ranked with events yet.

### Crash moments

After the crash step, crashes in `crashes.json` become moments for every style except 360 (`360 analyze` makes none; `settings.crash_moments` in `highlights.json` is false there). A crash with a confidence of at least `--must-include-crash` (default 0.6) becomes a must-include moment with its reaction: it replaces the moments it overlaps and ranks first. A weaker crash at or above the project floor, max(0.2, 0.6 x the strongest crash), is ranked like any moment; if it overlaps an existing moment, it marks that moment and changes nothing else. New crash moments have `kind` `crash` and the labels `crash` and `crash-reaction`. Crash candidates carry `crash`, `crash_confidence` and `crash_forced`, and `settings.must_include_crash` records the threshold. `analyze` and a re-analysis build them; a `cutpoint events` re-run (and "Look closer with vision" in `cutpoint ui`) rebuilds them from the current `crashes.json`, as a re-analysis would: the same moments and marks, a moment whose crash is unchanged stays as it is (id, vision rating), new ones take the next ids after `settings.last_candidate_id`, and `--no-rescore`, 360 projects and `settings.crash_moments` false leave them alone (action-cam keeps its re-analysis). A candidate that a must-include crash replaced does not come back when that crash goes away: re-analyze for that. AI Chat names them as the `crash` kind (`find_moments` finds them by `crash`, and its pacing treats them as footage of the activity, not as a breaker). In `gaming`, dialogue widening and menu-time dropping leave must-include moments alone.

### Candidates

Candidates come from the audio and transcript grids: per source, the strongest peaks, at most
the default count above (or `--max-candidates`), each at least a quarter of the strongest peak.
`gaming` has an open list: it has no fixed cap, so a long video can have more moments than the
usual count, but beyond that count a moment must reach 0.5 of the strongest peak. The default
sequence stays about 90 s. `--max-candidates` sets the cap for every style, as before.

In styles with an open list (`gaming`), detected events can also create candidates of their own
("nominated" moments), so a moment no longer has to be found by the grids first. Other styles
create none and behave as before. When events are scored (`analyze` with its event
steps, or `events`), each event that can score and that no candidate covers gets a window
(a few seconds before it, and after it as its type's tail or the style's post-roll); events
close together make one candidate. A laughter sound heard with a confidence below 0.12 (the detector's own threshold) nominates a
candidate, and so labels it `laughter`, only with backup within 5 s: another laughter, scream or
cheering sound, a laughter word or marker in the transcript, or loud speech. A stronger laugh
needs none. A nominated candidate has `signals.events.nominated` (the
event names) and keeps the score its window had from the grids as `grid_score`; its `score` is at
least 0.6 of the best other candidate's of that source, so the events decide its rank. A dialogue stretch over 90 s has exactly 0.6 of it, and its laughter and on-screen events count per minute. Ids of
nominees continue after the highest id ever given (`settings.last_candidate_id`). Menu time is
left out. Run again, the
same events give the same nominees. A source whose caches are gone gets none.

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
| GPU, 72 min 1080p60 gameplay, `--visual-index` | | visual index about 373 s per hour of video, about 25 MB per hour of cache | |
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

## detect

```text
cutpoint detect <file|folder> [<file|folder> ...] [--exclude GLOB] [--recursive] [--vision | --no-vision] [--ollama-url URL] [--ollama-model NAME] [--json]
```

Tells which category the footage is and which style `--style auto` would use, without analyzing
it. It reads the container tags, the camera telemetry and 40 small frames per source (24 frames
spread over the source, read at 640x360, plus 4 pairs of 4 consecutive frames; a few seconds each;
cached as `category-v3.json` in the source's cache folder), plus the HUD text check and the vision
check (below) for unsure sources, and writes nothing to a project. An older cache
(`category-v2.json`) is not read: every source is probed once more after updating.

| Category | Evidence | `analyze --style auto` | `edit --kind funny-moments --style auto` |
|---|---|---|---|
| Action cam | Camera telemetry (GoPro GPMF, Insta360 X3 IMU) in at least half of the footage | `action-cam` | `funny` |
| Gameplay | A static HUD: overlay pixels that stay put while the picture moves, or HUD text that stays in place (the HUD text check below); screen-recorder tags help | `gaming` | `funny` |
| Talk | Not gameplay, and speech covers at least 55% of a source of 8 minutes or more | `commentary` | `funny` |
| Action | A sport or fast physical activity filmed by a camera that rides along or follows the person, without telemetry. Only the vision answer `action` tells | `action` | `funny` |
| General | Anything else. Sure only with camera tags (a phone or camera make), a 2:1 frame of a 360 camera, or the vision answer | `action` | `funny` |

The other kinds of `edit` (`highlights`, `montage`, `clips`) use the `analyze` column.

The project gets one category, by the share of the footage duration; a source is only talk once
it has a transcript (`detect` uses one from an earlier `analyze`; `analyze` decides talk after
transcribing). A category with weak evidence (no HUD, only screen-recorder tags, which re-encoded
camera footage also has, or speech coverage between 40% and 55%) is ambiguous: `--style auto`
then asks the vision model (below); if that does not decide, it uses `gaming` when the best guess is gameplay, else the General style (`action`), with a `style_ambiguous` warning that names the style (`funny` for `funny-moments` either way). Talk is not followed yet: an unsure talk guess gets the General style (#914). Gameplay with a lot of speech stays
gameplay; an INFO line suggests `--style commentary`. An opaque box across most of the bottom
of the picture (a news lower third or ticker) is not a HUD: its text counts for neither the
static HUD nor the HUD text check, so camera footage with one is not called gameplay. A game whose HUD is such a panel (a
Doom-style status bar, an RTS console) is taken for a box too and can become ambiguous.

**Vision check.** A source the frame probe, the recorder tags and the HUD text check leave unsure
is shown to the vision model: 6 frames at 640 px, to Ollama with `qwen3.5:9b`, or to the hosted
Vision provider the user allowed in the Models tab of `cutpoint ui` (a hosted provider, or an
Ollama on another machine, needs the Vision consent). It is on by default and runs for unsure
sources only; `--no-vision` or `"vision": false` in `config.json` skips it, and `--vision` (also
on `analyze` and `edit`) asks for it explicitly (a missing consent is then a `vision_consent_needed` warning). `--ollama-url` and `--ollama-model` choose the Ollama and win
over a hosted Vision provider, as for other vision use. It is skipped in Cutpoint lite, without
Ollama running, and without consent: by default quietly (one INFO line), and the source stays
unsure; with `--vision` or `"vision": true` given explicitly, a skip for lack of consent is a
`vision_consent_needed` warning. An `other` answer leaves the source unsure; a `camera` answer is General, an `action` answer is Action. Answers are cached per
model and host as `category-vision-v2.json` (an older `category-vision-v1.json` is not read: unsure sources are asked once more). `decided_by` in the result (and the `detect` output)
names the evidence that decided: `vision` for these sources.

**HUD text check.** A thin HUD (a few lines of text, as in War Thunder, Rainbow Six or
Battlefield) is too small to show in the small frames. So a source those leave unsure between
gameplay and general, with no camera tags and not a still picture, also gets the HUD text check:
the HUD OCR engine of `analyze` finds the text on 8 frames at 1280 px and reads each text box that
stays in the same place in most of them. Text alone is not enough to make a source sure: at least 3 such boxes count as HUD text, and
when the gameplay score without that text is below 0.4 the source stays unsure and goes to the
vision check. Clocks, dates, speeds, GPS positions (a camera's or dashcam's burned-in overlay) and the subtitle
band at the bottom centre do not count. It takes a few seconds per unsure source, is cached as
`category-text-v1.json` (read again when the OCR engine changes) and runs only for `detect` and
`--style auto`. In Cutpoint lite, which has no OCR engine, it is skipped and the source stays
ambiguous.

A 2:1 frame counts as 360 footage only at 3840 px wide or more; `decided_by` is then `360 frame`.

`highlights.json` of an `auto` analysis has a `detection` block: `version` (4), `category`,
`emphasis` (`hype`, or `funny` for `funny-moments`), `style`, `reason`, `ambiguous`, `scores`,
`fallback` (the style used when the category was unclear, else `null`), `suggest`, `decided_by` and
`sources` (as in the [`--json` outputs](#json-output)).

## events

```text
cutpoint events <project> [--layers audio voice visual crash ocr vlm] [--no-vlm] [options]
```

Optional; needs the `events` extra and `cutpoint setup`. [`analyze`](#analysis-steps) already runs the `audio`, `ocr`, `voice`, `visual` and `crash` layers it can; use `events` for the `vlm` layer, to force layers (`--force`) or to rerun them with other options. It works in
layers, one GPU model at a time, and writes `events.json` and `events.txt`, then re-ranks
`highlights.json` (the original is kept as `highlights.pre-events.json`). The re-rank drops or
trims candidates in HUD menu time, as `analyze` does ([game streams](workflows/game-streams.md)), and,
in `gaming`, nominates candidates for events no candidate covers (see [Candidates](#candidates)).

Footage that is not reachable (offline drive, moved file) is skipped with a `footage not reachable` warning; if none is reachable the command fails before writing anything. With a source skipped, `events.json` is rewritten without that source's events; the action-cam re-rank is skipped and other styles re-rank with the events found; rerun once the footage is back (caches make it quick).

| Layer | What |
|---|---|
| `audio` | PANNs sound events: laughter, screams, cheering |
| `ocr` | Game HUD text: kill feeds, round banners (see [game streams](workflows/game-streams.md)) |
| `voice` | Wind-robust speech and laughter (Silero VAD + PANNs) |
| `visual` | CLIP frame embeddings, person detection, beauty and group scores |
| `crash` | Runs `cutpoint crashes` (action-cam projects) |
| `vlm` | The shared vision model describes candidate windows from frames, the transcript and the other layers' hints (no audio is sent): Ollama with `qwen3.5:9b` (`CUTPOINT_VISION_MODEL` changes it), or the hosted (API or subscription) Vision model of the Models tab in `cutpoint ui` ([`vision_assistant`](configuration.md#vision_assistant)) once you allowed vision requests there. If the model does not answer, the layer keeps its cached descriptions; without the Vision consent for the provider it uses none of them and warns (`vision_consent_needed`) |

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
| `--batch-size N` | Close each layer's model after N sources (the vision model at the next source boundary after at least N `vlm` windows, never within one source's windows; the crash layer's CLIP after N dense windows) and load it again; the results are the same. Default 0: keep it loaded |
| `--json` | One [result envelope](#json-output) on stdout for the whole run |

Tuning options, rarely needed:

| Option | Meaning |
|---|---|
| `--audio-thr GROUP=P`, `--audio-floor GROUP=P` | PANNs absolute threshold and relative-rule floor per sound group, for example `Laughter=0.1` |
| `--ocr-fps F` | Frames per second read by the HUD OCR (default 3.0) |
| `--no-hwaccel` | Decode on the CPU for the OCR |
| `--ocr-det-skip` / `--no-ocr-det-skip` | Skip the text detection of a HUD region that has not changed since detection last ran there, as on [`analyze`](#analyze). Without the flag, `events` reads as the analysis did, so it reuses that read: on when `analyze --ocr-det-skip` recorded `"ocr_det_skip": true` in the `settings` of `highlights.json`, off when it did not (older projects too). Only without a recorded analysis: [`ocr_det_skip`](configuration.md#ocr_det_skip) in `config.json`, else off |
| `--ocr-debug DIR` | Save region crops of OCR events to `DIR` |
| `--discover-bursts N` | Frame bursts sampled by HUD region discovery, 4 frames each (default 40) |
| `--frames N` | Frames per model window (default 14) |
| `--no-transcript` | Do not send the transcript excerpt to the model |

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
to write into a non-empty folder without `--force` (see [Replacing clips](#replacing-clips---force)).

### Replacing clips (--force)

With `--force`, the new clips are exported
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

### Resuming an export

`clips` and `extend` (in place, or with `--out` a new, empty or, with `--force`, non-empty
folder) export into the hidden staging folder and record each finished clip in its
`resume.json`. When the run fails, is interrupted (Ctrl+C) or is killed hard, the staging
folder is kept as the hidden `.<folder>.cutpoint-resume` next to the clips folder, with the
finished clips only; the clips folder itself is left as it was, so `assemble` never sees a new
or partial clip. The next `clips` or `extend` run into that folder (the same command again, or
`cutpoint resume <project>`) takes over every finished clip whose name, source, in and out
points, format, crop, grade, audio fade and peak compression are unchanged, encodes only the
others and removes the folder; a run with other clips removes it and encodes them all. 360
clips and clips trimmed by a read error are always encoded again. Temporary clips (`assemble <project>`, `edit --no-keep-clips`)
and an export straight into an empty drive root or mount point keep no resume folder. Delete
the folder by hand to start over; it only holds clips Cutpoint exported.

### Folder lock

One run at a time works on a folder: `clips`, `extend` and `assemble` hold the lock file
`.<folder>.cutpoint-lock` next to it while they run, so a second run never removes the first
one's staging folder. A second run stops with an `ERROR` naming the process that holds the
lock. The file is removed when the run ends, also on Ctrl+C. One left by a run that was
killed (`taskkill /F`) is taken over once its process is gone; one from a run on another
computer (a network folder) is not, so delete it by hand if no run is active.

### Clips options

| Option | Meaning |
|---|---|
| `--top N` | Top N by score (default: the best moments that fit the style's target length, a moment that does not fit in the room left is skipped and the next one tried; a must-include crash is always kept, and if none fits the best one is taken). An automatic pick skips a candidate that mostly overlaps one already picked from the same source; `--ids` does not. |
| `--ids ...` | Explicit candidate ids |
| `--segments FILE` | Export exactly the listed segments in the listed order |
| `--max-total S` | Keep the best-scoring moments that fit in S seconds. An automatic pick skips a candidate that mostly overlaps one already picked from the same source; `--ids` does not. |
| `--order` | `chronological`, `score`, `story` (default, except `commentary`: `chronological`; opens with the second-best moment, ends with the best) or `arc` (the default with `--mix`) |
| `--vertical` | Centre-crop to 9:16, 1080x1920. Default folder `clips_vertical` |
| `--size WxH` | Output resolution (even numbers). Default: the largest source's, capped at 1080p. Ignored with `--vertical` |
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

Reserved slots for visual candidates. An automatic choice (`--top`, the default
edit and `--max-total`, so also `edit` and `assemble <project>`) in a project of
an `action-cam`, `action` or `sport` style keeps about one pick in eight for the
[visual candidates](#analyze): none below 5 picks, one from 5, two from 16 (at most 2; with
`--max-total`, counted from the moments that fit). They go by peak, and only one with a peak
of at least 0.6 that overlaps no picked moment takes a slot, in place of the lowest-scoring
pick that is not a must-include crash, a crash or a jump (a crash moment, an action-cam jump or
wipeout moment, or a moment overlapping a `crashes.json` crash of at least 0.2 confidence; with
`--max-total` only when the total still fits). A slot no visual candidate qualifies for keeps
its scored moment, and other styles choose as before. `--ids v001` exports a visual candidate in any project.

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
  1.5 times its value and by more than 0.002), a picture over 25 % more colorful (`veto_m3`, a safety net: with the current levels
  stretch and vibrance caps no grade reaches it) or
  a washed-out one: with `auto`, one that loses more than 15 % of its colorfulness
  (`veto_m3_drop`); underwater, one whose median saturation falls below 0.63 of its value, such
  as flat sand turned grey (`veto_faded`), or when its frames could not be decoded
  (`no_frames`). An INFO line names each clip not graded with its reason, as recorded
  (`not_underwater` adds that `--color auto` may suit it; `log` that `--lut` fits it). Plain
  `auto` logs an INFO line naming clips that look underwater (by the same rule); it does not
  change them. `gaming` footage is graded
  only with an explicit `--color` (with a WARNING, a stronger one for `underwater`, and a
  `color_on_gaming` `--json` warning): `color` in `config.json` never applies to
  it. `--color` cannot be combined with `--lut`. Each clip records a `color` entry (method,
  version, strength, parameters, skip reason, metrics before and predicted), and `extend`
  grades its re-exports with those parameters without measuring again. Measurements are cached
  per source (`color/` in its cache folder).
- `manifest.json` records each clip's source, start, end, score, label, tail, crop, grade (`lut` or `color`), padding,
  speech moves and size in bytes (`size`), and `"generator": "cutpoint"`. Its header's
  `smart_tail.enabled` and `speech_aware.enabled` are `true` only when at least one clip got
  them (a sequence whose every segment sets `"smart_tail": false` or a `tail`, or
  `"speech_aware": false`, records `false`). `extend` and
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

- `--speech-aware` (default on) moves in/out points that fall mid-utterance to the utterance's
  start or end, at most `--speech-max-extend` (default 8 s) later and `--speech-max-lead`
  (default 5 s) earlier, and keeps clips of one source from overlapping. With no padding it is
  the only change (folder `<clips_dir>_speech`); `--no-speech-aware` with no padding is
  refused (`bad_args`). `--audio-fade S` (default 0.25, 0 = off) is baked into every clip;
  `--pano-source` and `--pano-cmd-every` apply to 360 clips only.
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
- The clips are exported into a hidden staging folder next to the output folder and moved in
  only when all of them were exported, also into a new or empty `--out` folder: a failed or
  cancelled run makes no new folder and keeps its finished clips for the next run (see
  [Resuming an export](#resuming-an-export)). An empty drive root or mount point is exported
  into directly, as with `clips`.
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
cutpoint assemble <clips_dir> [--out final.mp4] [--format mp4|fcpxml|otio] [--transition cut|fade]
                  [--fade-dur 0.25] [--audio-transition cut|crossfade] [--audio-fade S] [--joins FILE]
                  [--loudnorm] [--compress-peaks off|light|strong] [--captions] [--model MODEL] [--json]
cutpoint assemble <project> [--top N | --ids h001 h004 ... | --segments FILE] [--max-total S]
                  [the options above] [the export options of clips]
```

Joins the `.mp4` files currently in the folder, in natural filename order (`2_x` before
`10_x`). Given a project instead, it exports the moments first (see
[Assemble from a project](#assemble-from-a-project)). Like `clips`, it first puts back or removes what an interrupted `clips --force` or
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
- `--joins FILE` sets single joins from a JSON file; `--transition` stays the default of every
  other join. It works on a clips folder and on `assemble <project>`, not on `edit`.

  ```json
  {"joins": [{"from": "001_a.mp4", "to": "002_b.mp4", "transition": "cut"}]}
  ```

  `from` and `to` are the file names of two clips next to each other, in filename order;
  `transition` is `cut` or `fade`. A pair that does not match gets a WARNING (`join_unmatched`)
  and uses `--transition`; a Fade with no room for a frame becomes a cut (`join_no_room`).
  The audio follows each join's video: a Cut join is an audio cut, a Fade join uses
  `--audio-transition`, which is a crossfade by default, also under `--transition cut`. So a
  Fade join crossfades its audio even under a global cut. For hard audio on the fades, use
  `--transition fade --audio-transition cut` and mark the other joins as cuts in the file.
  There is one fade length (`--fade-dur`) for all fades.
- `--loudnorm` measures the joined audio (EBU R128) and applies one linear gain toward
  -14 LUFS, so the levels between and inside clips stay as they are, followed by a true-peak
  limiter; it then measures the result and redoes the pass with a lower ceiling if the true
  peak is above -1.5 dBTP. The limiter takes the peaks the gain pushes over the ceiling, so
  peaky material lands below target: a WARNING names the level when it is more than 1 LU
  below (GoPro wind: -16.6 LUFS without `--compress-peaks`, -14.0 with `strong`). The audio
  ends with the video, without padding.
- `--compress-peaks light|strong` applies [peak compression](#peak-compression) to the clips
  that were not exported with it (their `manifest.json` entry has no `compress_peaks`), in the
  join, before the edge fade. The final audio then always ends with the true-peak limiter
  and its -1.5 dBTP check, with or without `--loudnorm` (without it, the level stays where the
  speech was). Default: `compress_peaks` in `config.json`, else off.
- `--captions` transcribes the result and burns in bold lower-third captions in 2-4 word
  chunks. The ASS file is kept next to the output. `--model` picks the Whisper model.
- `--json` writes the result as one [JSON envelope](#json-output) on stdout.
- Default output: `<project>\final.mp4`; `final_vertical.mp4` for `clips_vertical`;
  `final_<name>.mp4` for other folders.
- `--format mp4|fcpxml|otio` (default `mp4`) picks what is written. `mp4` writes the video and
  no timelines (0.4.0 and earlier always wrote a `.fcpxml` and a `.otio` next to it; scripts
  that read them must now ask for one). `fcpxml` (Final Cut Pro, DaVinci Resolve) or `otio`
  (DaVinci Resolve 18.5+, OpenTimelineIO tools) writes only that timeline, named after
  `--out` with the format's extension (`final.fcpxml`), plus `final.srt`. It references the
  clips in the folder by absolute path, so keep them; nothing is joined or encoded, so it
  takes about a second. `--loudnorm`, `--captions` and `--model` are refused with it (exit 2);
  `--audio-fade` and `--compress-peaks` do not apply. With `--transition fade` (and the
  `edit` kinds that crossfade) the timeline holds a cross-dissolve at each join, starting on
  the edit, with the clips and total length `final.mp4` would have; with cuts it has hard
  cuts. On the OTIO audio track a clip without audio leaves a gap, so later audio stays in
  sync. Other formats of the same name left by an earlier run are not removed; a WARNING
  names them.
- Subtitles `final.srt` (named after the output) are written next to it from the clips'
  cached source transcripts, placed on the joined timeline (crossfades included) by each
  clip's `manifest.json` start and end. A clip renamed during review has no manifest entry;
  its own `.srt` (renamed with it) is used instead. Clips with neither, or whose source has no
  cached transcript, get none; one INFO line names them, and with none at all no `.srt` is
  written.

### Assemble from a project

`cutpoint assemble <project>` makes the final video without keeping clips. The positional is
read as a project when it holds `highlights.json` (a bare name is also looked up in the
projects folder, as for `clips`) or when `--segments` is given (then it is the project to
write to, as for `clips --segments`). Everything else is a clips folder, as before; the
options below are then refused (`bad_args`, exit 1).

- The moments are chosen as `clips` chooses them: `--top N`, `--ids`, `--segments FILE`,
  `--max-total S` and `--order` (default: the best moments that fit the style's target length, as `clips`; `suggested_top` of `highlights.json` is their count). `--mix` is
  not offered; pass `--segments <project>/mix_segments.json --order arc`.
- They are exported with the export options of [`clips`](#clips): `--vertical`, `--size`,
  `--crop`, `--smart-tail`, `--tail-min`, `--tail-max`, `--event-tails`, `--event-window`,
  `--extend`, `--extend-start`, `--extend-end`, `--speech-aware`, `--speech-max-extend`,
  `--speech-max-lead`, `--pano-source`, `--pano-cmd-every`, `--lut`, `--lut-strength`,
  `--color` and `--color-strength`. `--audio-fade` and `--compress-peaks` are baked into the
  clips (and so not applied twice in the join), as in `edit`.
- With `--format mp4` (default) the clips go into a temporary folder in the project,
  `.assemble-tmp-<id>`, under the same per-folder [lock](#clips) `clips` takes. The join and
  `final.srt` (from the clips' subtitles) follow, then the folder is removed: after success,
  after a failure and on Ctrl+C or Cancel (exit 130) alike, with its staging folder and, for
  360 clips, its camera paths in `pano_paths\.assemble-tmp-<id>`. Only `final.mp4`
  (`final_vertical.mp4` with `--vertical`, or `--out`) and `final.srt` are left; an existing
  `clips` folder is never touched. Camera paths edited for a `clips` export are not reused by
  a temporary export. The run holds its own lock (`..assemble-tmp-<id>.run.cutpoint-lock`)
  while it lasts, so a later run removes the folders left by one that was killed hard (a UI
  Cancel included) and keeps those of a run still going. `--force` is refused (nothing is
  replaced). Before the export, the free space of the project's drive is checked, as
  `360 stitch` does: it must hold the clips, the final video and its working copy (estimated
  at 20 Mb/s, the typical rate at 1080p30, scaled by the frame size and by the frame rate
  above 30 fps) plus 1 GB, or the run stops before exporting. With `--json`, `outputs` has no
  clip paths and a `temporary_clips` warning says so; a folder that could not be removed (open
  in another program) is named by a `temporary_clips_left` warning instead.
- With `--format fcpxml` or `otio` the timeline references the clips, so they are kept: they
  are exported into the clips folder `clips` uses (`clips`, or `clips_vertical`), with the
  same rules (a non-empty folder needs `--force`), and the timeline is written as from that
  folder. This is the same as `clips` followed by `assemble <clips_dir> --format`.
- To review, rename or delete clips before the join, use `clips` and `assemble <clips_dir>`
  instead.

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
GoPro wind), exported and assembled with `--loudnorm` before #523 (dynamic loudnorm):

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
1.9 and 2.8 LU below `off`. With `--loudnorm`, `assemble` applies one gain to the whole edit
toward -14 LUFS.

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
                 [--batch-size N] [--activity NAME] [--context TEXT] [--refresh-profile]
                 [--ollama-url URL] [--ollama-model NAME] [--json]
```

Action-cam module (its visual check needs the `events` extra, which Cutpoint lite leaves out).
Finds crashes and wipeouts in an analysed
project from full-rate IMU data, visual, audio and transcript evidence, and writes
`crashes.json` plus contact sheets in `crash_check\`. Confident crashes are always exported by
`clips`. The visual prompts and crash words come from the project's activity profile; a frame's
crash score is never below that of `--activity generic` (see
`profile <project>`; `--activity`, `--context`, `--refresh-profile`, `--ollama-url` and `--ollama-model` work as on [`profile`](#profile)). See
[action-cam workflow](workflows/action-cam.md). `--batch-size N` closes the dense confirm's
CLIP model after N dense windows and loads it again for the next ones, to give memory back on
long runs (as the crash layer of [`events`](#events) does); `crashes.json` is the same. Default
0: keep it loaded. `--json` writes the result as one [JSON envelope](#json-output) on stdout.

Footage that is not reachable (offline drive, moved file) is skipped with a `footage not reachable` warning; if none is reachable the command fails before writing anything. With a source skipped, the default reanalyze step is skipped and `highlights.json` is left unchanged; rerun once the footage is back.

## resume

```text
cutpoint resume <project> [--dry-run] [--json]
```

Re-runs the last `analyze`, `edit`, `events`, `crashes`, `profile`, `360 analyze` or `clips` of a
project that was interrupted or failed, from the command line recorded in its `progress.json`
(see [Run lock and progress.json](#run-lock-and-progressjson)). `<project>` is resolved like
the project of other commands (a bare name is in the projects folder). The recorded command
runs in the recorded working directory with its recorded options, a `--force` among them;
`resume` has no `--force` (it is a `bad_args` error, exit 2). Per-source caches make the rerun
continue where the run stopped. `edit` keeps its clips when the `clips` step was done and
`clips/manifest.json` is unchanged since; an `edit` or `clips` stopped while exporting encodes
only the clips it had not finished (see [Resuming an export](#resuming-an-export)).

Statuses that are resumed: `interrupted` (also `running` with a dead process), `failed` and
`low_memory`. Status `done` logs `nothing to resume` and exits 0. Sources of the record that
are missing now log a WARNING (`resume_sources_changed`); the run continues.

`--dry-run` runs nothing and prints:

```text
cutpoint <argv>
  in <cwd>, last run <status>
  <step>: <state>[, <done>/<total> done]
```

| Exit | Cause |
|---|---|
| the replayed command's own | The run was resumed |
| 0 | Status `done`, or `--dry-run` |
| 1 | No or unreadable `progress.json` (`not_found`); a record without a usable command (`error`); a live run holds the project (`folder_busy`) |
| 2 | Bad arguments, such as `--force` |

With `--json`, `outputs.resume` is `{argv, status_before, done_before}` (`done_before`: per
step `{state, done, total}`) and `outputs.resumed` is `true` when the command was replayed,
`false` for `done`, `--dry-run` or an error before the replay starts (`true` when the replayed command itself fails). Resume stops with `error` (exit 1) when `progress.json` records a run on another project (for example a copied project folder). The replayed command's own `outputs` and
`warnings` are in the same envelope. `command` is the recorded command (`edit`, `analyze`, ...)
once a valid `progress.json` was read (replay, done, `--dry-run`, `folder_busy`); it is
`resume` when no record could be read and for bad arguments.

### Memory guard

Before a model loads (Whisper, PANNs, the visual models, CLIP, the OCR engine, and Ollama's
vision model before the request that loads it), Cutpoint checks that the model's estimated need
plus `memory_guard` GB of RAM (`config.json`, default `2`; `0` turns the check off) are free;
a model on the GPU needs its VRAM instead, and Ollama's model the size Ollama reports, in RAM
and VRAM together. When memory is short it frees cached models and unloads Ollama's, checks
again and, still short, stops with `low_memory` (exit 1, `progress.json` status `low_memory`)
and an `ERROR` naming `cutpoint resume <project>` (or, before `progress.json` is written, the
same command again). Whisper, PANNs and the crash check's CLIP short of VRAM run on the CPU instead (a `WARNING`); Whisper without `--model` then runs the CPU default, `small`, as after a CUDA failure, when it is downloaded already (else `medium`, with a `WARNING` naming the `cutpoint setup` command for `small`); an explicit `--model` is kept. A later run without `--model` transcribes that source again with `medium` when it fits on the GPU then (not after a CUDA failure: `--model medium` transcribes it again), else reuses the `small` transcript (an INFO line names it), never one an explicit `--model small` made; the fallback run and each run reusing its transcript give a `transcript_fallback` warning. Free RAM is read on Linux and Windows only
(elsewhere there is no check). Free VRAM is that of the GPU the models use: the first entry of
`CUDA_VISIBLE_DEVICES` (an index or a GPU UUID), else GPU 0; when it hides every GPU (empty or
`-1`) or names one `nvidia-smi` does not list, VRAM is unknown and not checked. An index is
`nvidia-smi`'s numbering, which matches CUDA's with `CUDA_DEVICE_ORDER=PCI_BUS_ID`; a UUID always
matches. A vision server on this machine that is not Ollama gets only the
RAM check and a `WARNING`, as Cutpoint cannot unload it: llama-server, LM Studio and other
OpenAI-compatible servers have no unload request in that API, so `--batch-size` does not free
their memory either. A hosted Vision provider has nothing on this machine to free and gets no
extra request.

Ollama reloads a model asked for another context size (`num_ctx`), about 7 s for `qwen3.5:9b`.
AI Chat asks for 50000 (`context_tokens`); vision requests need 8192 (16384 for the VLM layer
and dialogue reading, 12288 for a rating's retry). Before each request, vision asks a local
Ollama which context the model is loaded with (`/api/ps`) and, when that is larger, uses it, so
chatting while an analysis rates frames loads the model once. A request never asks for less
than it needs; an Ollama on another machine, or one that does not answer `/api/ps`, gets what
the request needs. The memory guard checks the vision model again before a request with
another context than its last one, as Ollama reloads it then. The larger context costs VRAM:
measured on an RTX 3080, `qwen3.5:9b` takes 5.46 GiB at 8192 and 6.67 GiB at 50000.

### Run lock and progress.json

`analyze`, `edit`, `events`, `crashes`, `profile`, `360 analyze` and `clips` hold the lock
file `<project>/.analysis.run.cutpoint-lock` while they run and record themselves in
`<project>/progress.json` (`clips` only in a project with a `highlights.json` or `pano.json`,
and never over the record of an interrupted run of another command, which `cutpoint resume`
then still continues). A second run on the project stops with `folder_busy` and an `ERROR`
naming the process, as for the [folder lock](#folder-lock) of `clips`: wait, or delete the
lock file if no run is active. `events`, `crashes` and `profile` treat a folder with only a
`progress.json` as no project: they write nothing and take no lock. A first `analyze` or
`edit` writes `progress.json` only after its first source is transcribed. After Ctrl+C or a
failure, the step the run was in reads `failed` (the `PROGRESS` states) and `status` says why.
After a hard kill the file keeps `running`; `cutpoint resume` reads such a record, whose
process is gone on this computer, as `interrupted`.

| Key | Content |
|---|---|
| `version`, `generator` | Format version and the writer |
| `command`, `argv`, `cwd` | The command, its full argument list and its working directory |
| `options_key` | Identifies the command and options |
| `status` | `running`, `done`, `interrupted`, `failed` or `low_memory` |
| `pid`, `host`, `started`, `updated` | The process, its computer and times |
| `exit_code`, `error` | Exit code and `{code, message}` of a finished run |
| `sources` | The sources of the run |
| `steps` | Per step name `{state, done, total, pct}`; `edit` adds `steps.clips.manifest` (`{path, stamp}`) |

## profile

```text
cutpoint profile <project> [--activity NAME] [--context TEXT] [--refresh-profile]
                 [--ollama-url URL] [--ollama-model NAME] [--json]
cutpoint profile list [--json]
cutpoint profile show NAME [--json]
cutpoint profile create NAME --from FILE [--force] [--json]
cutpoint profile validate NAME|FILE [--json]
cutpoint profile delete NAME [--json]
```

Picks or discovers the project's activity profile (what a crash looks and sounds like in this
footage) and writes `activity.json` in the project. `crashes` and the `crash` layer of `events`
reuse it. Cutpoint ships only `base` (the floor every profile inherits) and `generic`. Every
other profile is in your library, `<CUTPOINT_HOME>/profiles` (see `cutpoint paths`): discovered
from your footage, or made by you or your AI agent. A project named `list`, `show`, `create`,
`validate` or `delete` is given as `projects/<name>`.

Discovery: the vision model describes a few frames chosen to differ as much as possible, the
text model writes a profile, named after the activity alone (crash words are dropped from its
name and recognition captions), and Cutpoint drops each crash caption that fits much of the
footage itself (more than 10% of its keyframes). The VLM prompts keep the generic wording, and
the model's crash phrases are capped at 0.75, so a phrase of its own never makes a crash
candidate without other evidence. The profile is
saved to the library with `"source": "discovered"` (a file with the same content is reused),
so the next similar video finds it. The profile is also cached per footage (the files, the
model, `--context` and the prompts), so the same footage in a new project gets the same profile
without asking the model again. A discovery the model answered unusably or that timed out is
remembered the same way: the model is not asked again for that footage until
`--refresh-profile`, another model or a new Cutpoint version (an INFO line says so); a
connection error is not, so the next run asks again. `activity.json` records the time it took in
`discovery` (`describe_s`, `attempts`, per attempt `synth_s`, `answered` (a JSON reply came
back) and `timed_out`, `failed` with the reason when the failure is remembered; `cached: true` when an earlier
result was reused), and `crashes.json` the profile step in `timings.profile_s`. The shorter
retry is skipped when the first attempt took nearly the whole timeout. In
`analyze` and `edit` it runs whenever a local Ollama answers, without `--vision`, and is quiet:
a missing consent, a bad Ollama port, too little free memory or a fallback is one INFO line.
`profile`, `crashes` and `events` still warn.

| Option | Meaning |
|---|---|
| `--activity NAME` | Use `generic` or a profile of your library (`cutpoint profile list`). An unknown name is `bad_args`. Default: found in your library or discovered from the footage |
| `--context TEXT` | What the video is, in the user's words. The local LLM turns it into extra crash captions and slang on top of the profile. A changed text rebuilds the profile |
| `--refresh-profile` | Ignore `activity.json` and the profile (or failed discovery) cached for this footage, and detect or build the profile again, and replace the cached one. Footage that matches a library profile keeps it unless you give `--context`; to start over, run `cutpoint profile delete NAME` first |
| `--ollama-url`, `--ollama-model` | The Ollama server and model that build the profile, as on [`analyze`](#analyze) (a flag left out takes its default). A new profile needs no deleting of `activity.json` to use them: add `--refresh-profile` |
| `--json` | One [result envelope](#json-output) on stdout; `outputs` holds the `activity.json` record (`name`, `source` (`flag`, `library`, `synthesized`, `context` or `fallback`), `scores`, `description`, `profile`) and `activity_file`. Output of the local model never reaches stdout |

Needs the visual layer of `events` for detection, and Ollama with a Qwen model
(`CUTPOINT_VISION_MODEL`;
[install](getting-started.md#optional-ollama-for-frame-rating-and-ai-chat)) for discovery
and `--context`. Without a model it uses the nearest library profile, else `generic`
(`source: fallback`; CLIP's three best activity-vocabulary terms with their shares go in
`description`, `described_by: clip`), and tries again on the next run. A project that already has an
`activity.json` keeps its recorded profile, also one no longer in the library. An Ollama on another machine (not `localhost` or a loopback
address) gets frames and the `--context` text only once the Vision consent for it was given in
the Models tab of `cutpoint ui`; until then it is not used, with a WARNING
(`vision_consent_needed`).

Profile library verbs (they read no project):

| Verb | Meaning |
|---|---|
| `list` | `base` and `generic`, then your profiles, each marked built-in, discovered or hand, with `source` and whether it is valid. A file that is not usable is listed with a warning (`invalid_profile`) |
| `show NAME` | A profile's captions, words and wording, and its file |
| `create NAME` | Save a profile of your own (`source: hand`) from the JSON of `--from FILE` (`-` reads standard input). `NAME` replaces the file's own name. Letters, digits, `-` and `_`; `base` and `generic` are refused. An existing name needs `--force` |
| `validate NAME\|FILE` | Check a profile by name or a `.json` file by path and list every problem; exit 1 if invalid |
| `delete NAME` | Delete one of your profiles. `base` and `generic` cannot be deleted. There is no confirmation |

`cutpoint profile show generic` prints the format.

## style

```text
cutpoint style list [--json]
cutpoint style show NAME [--json]
cutpoint style create NAME (--base STYLE | --from FILE) [--weight FEATURE=W ...] [--description TEXT]
                      [--applies-to KIND:NAME ...] [--profile NAME] [--force] [--json]
cutpoint style validate NAME|FILE [--json]
cutpoint style delete NAME [--json]
cutpoint style features [--json]
```

Manages your own scoring styles: JSON files in `<CUTPOINT_HOME>/styles` (`CUTPOINT_STYLES_DIR`;
format in [Configuration](configuration.md#custom-styles)). A style is partial weight overrides over
a built-in style and is used with `--style NAME` on `analyze` and `edit`. Nothing here reads
footage or a project.

| Verb | Meaning |
|---|---|
| `list` | The built-in styles and yours, with `source`, `base`, `applies_to` and whether each is valid. A file that is not usable is listed with a warning (`invalid_style`) |
| `show NAME` | The effective weights of a style (its base's, then its own) and where each comes from (`base` or `style`), with the style `hash` |
| `create NAME` | Save a style (`source: hand`). Needs `--base`, or a `--from` file that has one |
| `validate NAME\|FILE` | Check a style by name or a `.json` file by path and list every problem; exit 1 if invalid. A legacy preset is valid and noted as legacy |
| `delete NAME` | Delete one of your styles. Built-in styles cannot be deleted. There is no confirmation |
| `features` | The feature names weights can use, their layer, and whether this install has them (`available`; `extra` names the extra a feature needs) |

`create` options:

| Option | Meaning |
|---|---|
| `NAME` | Letters, digits, `-` and `_`; not a built-in style name and not `auto` |
| `--base STYLE` | The built-in style it builds on (not another style of yours) |
| `--from FILE` | Start from this style JSON (a legacy preset also works). The other options replace its values and `--weight` adds to its weights; its `source` is kept. `create NAME --from FILE --force` converts a legacy preset to schema 1 |
| `--weight FEATURE=W ...` | A feature weight, 0 or more, over the base's. Names are those of `style features` |
| `--description TEXT` | What the style is for |
| `--applies-to KIND:NAME ...` | `category:NAME` (`action-cam`, `gameplay`, `action`, `talk`, `general`) or `profile:NAME`. Stored only: `--style auto` does not use it yet |
| `--profile NAME` | An activity profile name. Stored only |
| `--force` | Replace your style of this name |
| `--json` | One [result envelope](#json-output) on stdout; `outputs` per verb in [the table above](#json-output). The command names are `style list`, `style show` and so on |

An existing name without `--force` is `output_not_empty`; a missing style or file is `not_found`;
a rejected value (unknown feature, negative weight, bad name) is `bad_args` with exit 1, and a
command-line syntax error is exit 2. `create` and `validate` warn with `unknown_profile` and
`feature_not_computed`; a legacy preset is never rewritten automatically.

`analyze` records `settings.style_hash` in `highlights.json`: the first 12 hex characters of the
SHA-256 of the style's canonical JSON, so a change to a style file shows in the project.

```powershell
uv run cutpoint style features
uv run cutpoint style create bmx --base action --weight scene=0.4 --weight speech=0.1 --description "BMX park"
uv run cutpoint style show bmx
uv run cutpoint analyze D:\Videos\ride.mp4 --style bmx --out ride
```

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

## setup

```text
cutpoint setup [--only pip whisper panns visual voice] [--whisper-model NAME] [--json]
```

Downloads the Whisper model and the models for `events` (about 2.5 GB, 3.5 GB with an NVIDIA
GPU): Whisper, PANNs, CLIP ViT-L/14, the Faster R-CNN detector, the aesthetic head and Silero
VAD. `whisper` fetches the model `analyze` uses without `--model` (`medium` with CUDA, about
1.5 GB; `small` without, about 0.5 GB) into the Hugging Face cache; `--whisper-model NAME` fetches
another faster-whisper name (such as `large-v3`) or Hugging Face repo id instead (with
`whisper` in `--only`; without it, exit 2). It needs no
extra, so `setup --only whisper` works in Cutpoint lite (the `visual` step of a full `setup`
needs the `events` extra). Without it, a run downloads the model it needs on first use; once the
model is cached, runs load it without a Hugging Face Hub request. The VLM layer uses the shared vision model instead (Ollama with
`qwen3.5:9b`, or a hosted Vision model), which `setup` does not download. `--json` writes the result as one [JSON envelope](#json-output) on stdout. See [Getting started](getting-started.md#download-models-only-for-cutpoint-events).

## paths

```text
cutpoint paths [--json]
```

Prints the resolved data folders: `CUTPOINT_HOME`, config, cache, models, tools,
stage, styles, HUD presets, activity profiles, projects and the bundled presets. `--json` writes them as one [JSON envelope](#json-output).

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

## proxy

```text
cutpoint proxy [--port PORT]
```

`proxy` runs the local authenticated [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI)
daemon in the foreground. **Use at your own risk**. Subscriptions (Claude Code, ChatGPT Codex,
Google Antigravity) route their OAuth authorization, token refresh and inference requests
through this loopback service (`127.0.0.1`). Available in both full and lite installs without
requiring local models or PyTorch. `cutpoint ui` starts this proxy itself when it needs it (see
[`ui`](#ui)) and stops it on exit, so `cutpoint proxy` is no longer needed for sign-in there. Run it
in a dedicated terminal to keep the proxy running across UI restarts, for `cutpoint login`, and for
command-line runs (`analyze`, `edit`, `events`) whose Vision or AI Chat model is a subscription. A
proxy started this way is never stopped by the UI.

On first run, `cutpoint proxy` lazily downloads the checksum-verified ~24 MB binary (pinned version
8.0.47) to `<CUTPOINT_HOME>/tools/cliproxy/` and starts it on port 18317 (`CUTPOINT_CLIPROXY_PORT` or
`--port` overrides the port; `CUTPOINT_CLIPROXY_BINARY` overrides the executable path).
Subscription credentials use Cutpoint's keyring integration with a native OS backend:
Secret Service/KWallet, macOS Keychain or Windows Credential Manager. File-based keyrings
are rejected. A chained backend selects one native store and never falls through to another
backend after an operation fails. A missing, locked or failing keyring stops startup or
credential persistence; Cutpoint does not fall back to permanent plaintext storage.

CLIProxyAPI needs private plaintext working files in `<CUTPOINT_HOME>/cliproxy/auths/`.
An independent supervisor restores them only after acquiring the daemon lock, checkpoints
refreshes and removals during the run, and stops and reaps the daemon before final persistence
and cleanup. It scans the working files every 5 seconds and reads or writes the keyring and
re-checks file permissions only when they changed since the last checkpoint (and always at
start and at the final persistence on stop, so a refresh just before Ctrl+C is still saved).
If a credential disappears or a read observes incomplete JSON or UTF-8 during a
periodic scan, the whole snapshot is deferred until the next scan; no partial account map is
committed. Unsafe paths and permission errors still stop persistence. The supervisor survives
a foreground CLI hard kill on every supported
OS. Login and logout report success only after a keyring checkpoint.

If final keyring persistence fails, the supervisor uses its cached recovery key to encrypt the
latest working files into `<CUTPOINT_HOME>/cliproxy/credential-recovery.enc` before removing
the plaintext files. The recovery key is stored only in the native OS keyring. The command
still reports failure; the next start authenticates and recovers the journal before restoring
older keyring data. Do not delete the journal if you need the latest refreshed credentials.
Corrupt or incomplete recovery data fails closed rather than replacing the committed snapshot.

Killing the supervisor too, losing the host, or failing to write both the keyring and encrypted
recovery can still leave private plaintext files. On restart, surviving working files are
recovered before older keyring data. An interrupted daemon write cannot be made atomic with
the OS keyring, and removing a file is not secure erasure of filesystem snapshots or backups.
Use encrypted OS storage, including swap, for protection against whole-host failure or
offline access. POSIX directories/files use `0700`/`0600`; Windows uses a verified, protected
current-user-only DACL instead of inherited grants. Existing owners must match the current user
or the current token's default owner; the latter is normalized to the current user and verified.
Other owners and unsupported permissions fail closed.
Configuration, the TLS private key, and separate random local client/management tokens remain
private files under `<CUTPOINT_HOME>/cliproxy/`, including `keys.json`; these are not
subscription credentials. Prompt cloaking and prompt/error logging are disabled.

The port is persisted in `keys.json`; startup selects `--port`, then
`CUTPOINT_CLIPROXY_PORT`, then the saved port (or the default). While the owned daemon is
running, management and inference use its saved runtime port even if a client's environment
specifies another port. Each explicit start creates a fresh TLS identity and local
client/management tokens. Management and inference
use HTTPS and verify the listener against that private local trust file on the actual request
connection, before sending credentials or request bodies. There is no HTTP or unverified-TLS
fallback. Startup also refuses occupied ports and requires the installation's running-daemon
lock. This does not protect against same-user malware or administrators able to read the
keyring or replace private files. Connection failures are distinguished from HTTP errors.
Startup waits up to 120 seconds for the daemon to answer while the supervisor runs (an INFO
line after a few seconds says it is still waiting, for example during a keyring unlock prompt)
and fails at once if the supervisor exits first.
Ctrl+C stops and reaps the daemon, completes credential cleanup, and exits with code `130`.

TLS and encrypted recovery use the core `cryptography` package in both full and lite installs.
Where upstream does not publish a wheel, a source build needs its
[Rust/OpenSSL build prerequisites](https://cryptography.io/en/latest/installation/).

The child runs from the private `<CUTPOINT_HOME>/cliproxy/run/` directory, not the caller's directory.
It ignores the caller's `.env` and strips CLIProxyAPI's remote-store, Home, deployment and
`MANAGEMENT_PASSWORD` overrides, keeping the generated private management key exclusive. Network
proxy and TLS/CA settings are preserved for the child's upstream connections; Cutpoint's local
HTTPS client trusts only its generated certificate. Startup refuses an existing `.env` in the
private run directory rather than loading or deleting it. The cross-platform supervisor owns
the daemon lock through child termination and credential cleanup, so a restart cannot race
the previous supervisor's finalization.

## login

```text
cutpoint login claude-code|codex|antigravity [--no-browser]
cutpoint logout claude-code|codex|antigravity
```

`login` starts an OAuth sign-in to a Claude Code, Codex (ChatGPT) or Antigravity subscription
through the running `cutpoint proxy` daemon. **Use at your own risk**. It opens the provider's
authorization URL in your browser and completes via local callback; `--no-browser` logs the URL
to stderr without opening a browser window. Subscription credentials are persisted in the OS
keyring, never in `config.json`; CLIProxyAPI uses private runtime files while the proxy runs
(see the recovery and platform limits under [`proxy`](#proxy)).

An unsuccessful login (including polling errors, timeout or Ctrl+C) attempts to cancel the pending
authorization; cancellation is best-effort if the proxy is unreachable. Successful logins are not
cancelled. Ctrl+C exits `130`, including at the Antigravity confirmation prompt.

`logout` removes the provider's subscription sessions from the running proxy and checkpoints
their removal in the OS keyring before reporting success.

Set `assistant.provider` (Chat) or `vision_assistant.provider` (Vision) to `claude-code`, `codex`
or `antigravity` to use a subscription provider; the Models tab in `cutpoint ui` reflects the same
login status and starts login flows under Connected accounts.

An enabled, nonexpired account remains connected during a temporary provider error or rate-limit
cooldown. Requests then report the proxy's availability condition instead of requiring fresh login.

Activity detection for `profile`, `crashes` and the crash step of `analyze` stays on Ollama,
regardless of the selected Vision provider. An Ollama on another machine needs the Vision consent
for its URL here too; without it nothing is sent and the profile falls back (`vision_consent_needed`).

Logging in to Antigravity upstream may auto-provision a free-tier companion project if your account
does not already have one. Cutpoint discloses this during sign-in and in the UI.
Requests to subscription providers go to the vendor's hosted service and require explicit vendor
consent in AI Chat or the Models tab, as with any hosted provider.
