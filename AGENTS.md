# Cutpoint for AI agents

You are an AI agent asked to edit video with Cutpoint. This file is what you need. Read this any
time with `cutpoint guide`. If you are instead changing this repository's code, read
[CONTRIBUTING.md](https://github.com/FerventOrange/cutpoint/blob/main/CONTRIBUTING.md).

Cutpoint finds the best moments in long recordings, exports them as numbered clips and
assembles a finished video. It runs locally, never modifies source videos, and writes
everything into a project folder.

## Status: what you can rely on

| Available now | Planned (do not call; it does not exist yet) |
|---|---|
| CLI `cutpoint` with the commands below | `--context` on `edit`, clip order chosen by a local model |
| `cutpoint edit`: analyze, clips and assemble in one command | |
| `cutpoint resume <project>`: re-runs an interrupted or failed run from its `progress.json` | |
| `cutpoint analyze` includes the reaction, keyframe and crash steps, and for `gaming` a `dialogue` step (a text model marks funny or heated talk); crashes in `crashes.json` become moments (not for 360) | |
| Result files in the project folder (JSON and text), `.srt` subtitles next to clips and finals | Profiles for the other layers and for games |
| Activity profiles for crash detection: `cutpoint profile`, `--context` on `profile` and `crashes` | |
| Footage category detection: `cutpoint detect`, and `--style auto` on `analyze` and `edit` (opt-in; unclear footage gets the default style) | |
| Frame index for the AI Chat's visual search: `--visual-index` on `analyze` and `edit` (off by default; `"visual_index": true` in `config.json`) also writes the CLIP frame embeddings for `gaming`, `funny` and `commentary` footage; `action`, `sport` and `action-cam` always have them | |
| VLM layer of `cutpoint events` through the shared vision model: Ollama with `qwen3.5:9b`, or a hosted Vision model the user allowed in the Models tab of `cutpoint ui` | |
| `--json` result envelope on stdout: `paths`, `analyze`, `detect`, `clips`, `assemble`, `edit`, `extend`, `capabilities`, `mix`, `stage`, `setup`, `crashes`, `profile`, `style ...`, `events`, `resume`, `360 ...` | |
| Custom styles: `cutpoint style list\|show\|create\|validate\|delete\|features` (weights over a built-in style, used with `--style NAME`; `applies_to` is stored only, `--style auto` ignores it) | |
| Agent skill: `cutpoint skill install` (Claude Code), `cutpoint skill print` | |
| Color grade with a 3D LUT the user supplies: `--lut FILE` and `--lut-strength` on `clips` and `edit` | Color matching between clips |
| Automatic color correction per clip: `--color auto\|underwater` and `--color-strength` on `clips` and `edit` (not with `--lut`) | |
| Output format on `assemble` and `edit`: `--format mp4` (default: the final video, no timelines) or `--format fcpxml\|otio` (only that timeline, referencing the kept clips, no encode) | |
| Final video without keeping clips: `assemble <project>` (with `--top`, `--ids`, `--segments`, `--max-total` and the `clips` export options) and `edit --no-keep-clips` | |
| Single joins on `assemble`: `--joins FILE` (`{"joins": [{"from": "001_a.mp4", "to": "002_b.mp4", "transition": "cut"\|"fade"}]}`), `--transition` for the others; not on `edit` | |
| Action-cam and 360 modules: `crashes`, `mix`, `360 ...` (see the workflow pages) | |
| Commands the user runs, not you: `ui` (browser workspace), `proxy`, `login`, `logout` (subscription logins for the hosted models of `ui`). They have no `--json` | |

If the user asks for "edit this video and make it a montage" in one step, use
[One command](#one-command); to choose the moments yourself, use the loop below it.

To let Claude Code find Cutpoint on its own, the user can install the skill with
`cutpoint skill install` (ask first: it writes to their Claude Code settings folder); see
[docs/agents.md](https://github.com/FerventOrange/cutpoint/blob/main/docs/agents.md#install-the-skill-claude-code).

## Calling the CLI

- If installed with the installer (see
  [docs/getting-started.md](https://github.com/FerventOrange/cutpoint/blob/main/docs/getting-started.md#install)), run `cutpoint <command> ...`; from a
  clone, `uv run cutpoint <command> ...`. The examples below use `uv run cutpoint`; with a tool
  install, drop `uv run`. First-time setup is in
  [docs/getting-started.md](https://github.com/FerventOrange/cutpoint/blob/main/docs/getting-started.md).
  Check it works with `cutpoint paths` (or `uv run cutpoint paths`).
- Exit codes: `0` success, `1` error, `130` interrupted, `2` bad arguments. To stop a run from a
  program on Windows and keep the cleanup, see
  [docs/commands.md](https://github.com/FerventOrange/cutpoint/blob/main/docs/commands.md#stopping-a-run).
- Logs and errors go to stderr as `HH:MM:SS LEVEL message`; errors are `ERROR` lines. Use
  `cutpoint -v <command>` (flag before the command) for debug output and tracebacks.
- Paths inside a project must start with `projects/<name>/...` or be absolute; a bare name
  means the project itself. Forward slashes work on every OS; do not use backslashes in bash.
- Results are files. Read the JSON files below instead of parsing logs. `--json` (on the commands in
  the Status table) also writes one JSON object to stdout, with the file paths and warnings:
  `ok`, `command`, `schema`, `project`, `outputs`, `warnings` (`code`, `message`, `files`) and
  `error` (`code`, `message`, or `null`). `error.code` is `bad_args` (exit 2 for the command
  line, 1 when a command rejects an option value itself, such as a `--lut` file or
  `--clip-override`), `not_found`,
  `ffmpeg_missing`, `missing_extra`, `output_not_empty`, `folder_busy` (another run holds the folder's lock: wait or retry), `low_memory` (too little free
  memory for a model: tell the user to close other programs, then run the command the ERROR line names (`cutpoint resume <project>`, or the same command again when no progress.json was written yet)),
  `interrupted` (130) or `error`.
  Read `warnings` such as `kept_clips`, `kept_sources`, `step_failed` and `chat_consent_needed` (the dialogue step was skipped: the hosted AI Chat model has no consent) and tell the user.
  With `CUTPOINT_PROGRESS=1`, stderr also has `PROGRESS {...}` lines per step (`analyze` has 7 steps, `edit` 9). Keys of both:
  [docs/commands.md#json-output](https://github.com/FerventOrange/cutpoint/blob/main/docs/commands.md#json-output)
  and [#progress-output](https://github.com/FerventOrange/cutpoint/blob/main/docs/commands.md#progress-output).
- Custom styles: run `style list`, `style show` and `style features` before `style create`; never invent feature names (`style features` lists them). `style delete` asks nothing: only delete a style you created.
- Commands are re-runnable. Per-source results are cached (path, size, modification time).
- `clips` refuses a non-empty output folder without `--force`. `--force` replaces only the clips
  its `manifest.json` lists, once every new clip is exported; every other file stays, sources
  included. Clips renamed or added by hand stay and are assembled: a WARNING names them. Tell
  the user; move them out only if you exported them, and never delete a file you did not
  export. If such a clip has the name of a new clip, the run stops before exporting with an
  `ERROR` naming it; tell the user to move or rename it.
- `analyze`, `edit`, `events`, `crashes`, `profile`, `360 analyze` and `clips` hold
  `<project>/.analysis.run.cutpoint-lock` while they run. A second run gets `folder_busy`: wait
  or retry; delete the lock file only if you are sure no run is active.
- A `clips` or `extend` run that fails or is interrupted keeps its finished clips in the
  hidden `.<folder>.cutpoint-resume` next to the clips folder (never in it, so `assemble` sees
  none of them). The next run into that folder with the same clips encodes only the missing ones
  (`cutpoint resume <project>` replays an `edit` or `clips` run); a run with other clips removes
  the folder. Leave it alone; it is not a clips folder.
- One run at a time works on a clips folder (lock file `.<folder>.cutpoint-lock`). If `clips`,
  `extend` or `assemble` stops with an `ERROR` that another run holds the lock, wait for it;
  delete the lock file only if you are sure no run is active. If `clips --force` or `extend`
  refuses a `manifest.json` Cutpoint did not write (or cannot read), tell the user; never delete
  or overwrite their manifest.
- Transcription, `analyze` steps, `events` and export can take minutes to hours. Run them in the background or
  with a long timeout.
- Never modify or delete the user's source videos. You may delete or rename clips in a project's
  `clips` folder during review.
- Use the real options: `cutpoint <command> --help` is authoritative;
  [docs/commands.md](https://github.com/FerventOrange/cutpoint/blob/main/docs/commands.md)
  explains them.

## One command

```powershell
uv run cutpoint edit D:\Videos\clip.mp4 --kind funny-moments
```

- `--kind` is `highlights` (default; up to 3 min, chronological, crossfades), `montage` (up to
  60 s, best first, crossfades), `clips` (the 10 best; the clips are the deliverable) or
  `funny-moments` (funny style, 8 moments, chronological, burned-in captions).
- Options: `--style`, `--top`, `--max-total`, `--vertical`, `--force` and `--events` override
  the kind. `--no-keep-clips` exports the clips into a temporary folder that is removed after
  the join (not with `--kind clips`). `--format fcpxml|otio` writes that timeline instead of
  the final video (clips kept). `--vision` adds the optional frame rating, which also reads what was said in each moment
  except for `action-cam` (Ollama by default;
  the Vision role in the `cutpoint ui` Models tab can route it to a hosted provider). It is off
  for every style unless `"vision": true` in the user's `config.json` turns it on; `--no-vision`
  then skips it. An explicit `--ollama-url` or `--ollama-model` wins over a hosted Vision
  provider and over the AI Chat model for dialogue reading, and also applies to the
  crash/profile step. An `--ollama-url` on another machine, or a hosted Vision provider in the
  Models tab, needs the user's Vision consent (Models tab); without it vision is skipped with a
  `vision_consent_needed` warning: tell the user. An invalid port in `--ollama-url` exits 2 (`bad_args`).
  `--vision-top`, `--ollama-url`, `--ollama-model` and `--skip` work as on
  `analyze`; `--visual-index` and `--no-visual-index` as on `analyze` (the frame embeddings for
  the visual search of `cutpoint ui`, off by default; they change no ranking);
  `--ocr-det-skip` and `--no-ocr-det-skip` as on `analyze` (faster gameplay HUD reading, off
  by default; `"ocr_det_skip": true` in `config.json`);
  `--compress-peaks`, `--lut`, `--lut-strength`, `--color` and `--color-strength` as on `clips`.
- Outputs: the project `<projects>/clip` (`--out NAME` to change it) with the clips,
  `final.mp4` and `clip-final.mp4` plus `.srt` subtitles. The last log lines name the final
  video, its subtitles and the clips folder (`none (temporary, removed)` with
  `--no-keep-clips`). To change the cut, review and re-assemble as in steps 5-7 below; that
  updates `final.mp4` and `final.srt`, not the `clip-final` copies.
- A non-empty clips folder needs `--force` (ask the user first, as for `clips --force`); with
  `--no-keep-clips` it is left alone.

## The full edit

Given a request such as "edit D:\Videos\clip.mp4 into a funny-moments montage":

1. **Pick a style** from the request: `funny`, `gaming`, `action`, `sport`, `commentary`
   (long talk), `action-cam` (helmet-, chest- or bike-mounted action-cam footage, best with
   GoPro or Insta360 X3 telemetry; formerly `ski`, which is no longer accepted).
   Footage that is not any of these: use `action` or `gaming`.
2. **Analyze.**

   ```powershell
   uv run cutpoint analyze D:\Videos\clip.mp4 --style funny --out clip-edit
   ```

   On Linux: `uv run cutpoint analyze ~/Videos/clip.mp4 --style funny --out clip-edit`.

   Writes `highlights.json` and `transcript.txt` into the project folder
   (`<projects>/clip-edit`, see `cutpoint paths`). It also runs the reaction, keyframe and crash
   steps it can and writes `events.json` and `events.txt`; a step is skipped with an INFO line
   when the extra or model files are missing, and `--skip reactions keyframes crashes dialogue`,
   given after the paths, leaves them out. For `gaming` it also runs `dialogue`: a text model reads
   the whole transcript and marks funny or heated talk (`DIALOGUE` lines in `events.txt`), each
   stretch a candidate that keeps its whole length and is labelled with a short title of who or
   what the bit is about (`dialogue_title`; its slug is the label and clip file name). A candidate such a stretch overlaps is widened
   to it and candidates it joins become one, but only when the result stays within 90 s (a longer
   stretch is a candidate of its own and the others stay as they are), so a candidate id from an
   earlier run can be absorbed:
   it stays in `highlights.json` under `widened_from.absorbed` but is no longer a candidate, and
   `--ids` rejects it. Read the file again. A widened candidate drops a silence of 8 s or more
   (no speech in the transcript) before or after its original moment. A hosted AI Chat model needs the user's chat
   consent for it (`chat_consent_needed` warning: tell the user); with `--max-total` such a
   candidate longer than the cap is dropped, not trimmed. A stretch over 90 s overlaps the candidates
   inside it: `--top` and `--max-total` skip the overlapping one, but `--ids` does not, so do not
   name both.
3. **Optional, for the VLM layer or to force layers:**
   `uv run cutpoint events clip-edit` (needs the `events` extra and `cutpoint setup`; the VLM
   layer needs Ollama with `qwen3.5:9b`, or a hosted Vision model the user allowed in the
   Models tab of `cutpoint ui`, and is skipped with a WARNING when neither answers). It adds
   the VLM layer, re-ranks `highlights.json` and rewrites `events.json` and `events.txt`. If it
   reports a missing module, tell the user how to add the extra (see
   [When something fails](#when-something-fails)).
4. **Choose moments.** Read `highlights.json` (`candidates` with `id`, `source`, `start`, `end`,
   `score` 0-100, `label`, `transcript`; `suggested_top`) and, for banter and funny moments,
   `transcript.txt` (`[file.mp4 00:12:31.4-00:12:38.9] text`). `suggested_top` is how many
   moments plain `clips` takes: the best that fit the style's target length, skipping longer
   ones (if none fits, the best one). `--top` with that number is not the same pick (it takes
   the N best whatever their length), so for the default edit run `clips` without `--top`.
   Then either:
   - take the top N: `uv run cutpoint clips clip-edit --top 8` (or `--max-total 60` for a
     length cap, `--ids h001 h004`), or
   - export your own picks with a segments file:

     ```json
     [
       {"source": "D:/Videos/clip.mp4", "start": 312.4, "end": 330.0, "label": "intro"},
       {"source": "D:/Videos/clip.mp4", "start": 1204.0, "end": 1219.5, "label": "joke"}
     ]
     ```

     `uv run cutpoint clips clip-edit --segments segments.json`
5. **Review.** Clips are `clips\001_<label>.mp4`, `002_...`, and `clips\manifest.json` maps each
   to its source and times. Tell the user what was exported. Delete clips the user rejects and
   rename to reorder (for example `005_x.mp4` to `001b_x.mp4`, and its `005_x.srt` to
   `001b_x.srt`, so `final.srt` keeps its subtitles).
6. **Assemble.**

   ```powershell
   uv run cutpoint assemble projects/clip-edit/clips --transition fade --loudnorm
   ```

   Writes `final.mp4` and `final.srt` in the project folder, and no timelines. If the user
   wants to keep editing in DaVinci Resolve or Final Cut Pro, add `--format fcpxml` (or
   `--format otio`): it writes only `final.fcpxml` (or `final.otio`) and `final.srt`,
   referencing the clips, without making `final.mp4`.
7. **Report** the paths of `final.mp4` (or the timeline) and the clips folder.

If the user wants only the final video and does not want to review clips, skip steps 4-6:
`uv run cutpoint assemble clip-edit --top 8 --transition fade --loudnorm` (also `--ids`,
`--segments`, `--max-total` and the `clips` export options such as `--vertical`) exports the
moments into a temporary folder, writes `final.mp4` and `final.srt`, and removes the temporary
clips, also when it fails or is interrupted; an existing `clips` folder is not touched. With
`--json` a `temporary_clips` warning says the clips were not kept.

Variants:

| Request | Do |
|---|---|
| Short clip or reel for social media | `clips <project> --vertical --max-total 60`, then `assemble projects/<project>/clips_vertical --captions` |
| Highlights of a specific length | `clips <project> --max-total <seconds>` |
| "Extra" instruction such as "only funny moments" or "keep the speech intact" | Choose `--style`, filter candidates by `transcript`/`label`, use `--segments`. The CLI does not parse these instructions |
| The user says what the footage is ("downhill mtb, chest mount, muddy lens") | Before `crashes`, run `profile <project> --context "<the user's words>"`; check the choice with `profile <project> --json` (`outputs.name` and `outputs.source`; a `context_not_applied` warning means the user's words were not used; with `vision_consent_needed` the user's Ollama is on another machine and needs their Vision consent in the Models tab: tell them). Use `--activity NAME` only when the user names a sport that has a profile. Do not edit profile files |
| A run was interrupted or failed | `resume <project>` (`--dry-run` shows the recorded command and step counts). It replays the recorded command line, including a recorded `--force`, and takes no `--force` itself. Status `done`: nothing to resume |
| Room to trim in an editor | `clips ... --extend 1.5` or `extend <clips_dir> --seconds 2` |
| The user asks to fix dull, washed-out or tinted colors | `clips <project> --color auto` (or on `edit`); `--color underwater` only for underwater footage or when an INFO line suggests it. Clips that look good, log footage, 360 clips and clips a safety check vetoes stay as they are: tell the user which clips were not graded and why, from `color.skip` of each clip in `clips/manifest.json` (codes: [docs/commands.md](https://github.com/FerventOrange/cutpoint/blob/main/docs/commands.md#clips), **Automatic color**). `assemble <project>` (mp4) and `edit --no-keep-clips` keep no manifest: use the per-clip INFO lines of the run instead |
| The user gives a LUT for log or flat footage ("use my GoPro Flat LUT") | `clips <project> --lut <file>.cube` (or on `edit`); `--lut-strength 0.5` for half the grade. Only with a LUT the user names; never pick one for them |
| 360 (Insta360 X3) footage | [docs/workflows/360.md](https://github.com/FerventOrange/cutpoint/blob/main/docs/workflows/360.md) |
| Action-cam with telemetry | `analyze <input> --style action-cam`, then [docs/workflows/action-cam.md](https://github.com/FerventOrange/cutpoint/blob/main/docs/workflows/action-cam.md) |
| Gameplay with kill feeds | [docs/workflows/game-streams.md](https://github.com/FerventOrange/cutpoint/blob/main/docs/workflows/game-streams.md) |

## Project files

| File | Written by | Content |
|---|---|---|
| `highlights.json` | `analyze` (written once, at the end), `events` | Ranked candidates (`h001`...) and settings (`settings.last_candidate_id`: the highest id given; `settings.must_include_crash`); crashes from `crashes.json` are candidates too (not for 360): `crash`, `crash_confidence`, `crash_forced` (confidence at least `must_include_crash`, default 0.6: kept and ranked first), new ones with `kind` `crash`; in `gaming`, events can add candidates of their own: `signals.events.nominated` names the events, `grid_score` is the score before the events; `gaming` has no fixed cap on candidates; action, sport and action-cam projects also list visual candidates last (`v001`..., `"pool": "visual"`, `kind` `view`, `crew` or `cruise`, `score` 0-10) |
| `transcript.txt` | `analyze` | Time-stamped transcript |
| `events.json`, `events.txt` | `analyze`, `events` | Detected sounds, HUD events, model descriptions, talk stretches (`kind` `text`, `type` `dialogue`) |
| `clips/manifest.json` | `clips`, `extend` | Per clip: source, start, end, score, label, tail, crop, lut or color, srt |
| `clips/001_<label>.srt`, ... | `clips`, `extend` | Subtitles of each clip, from the cached transcript |
| `final.mp4`, `final.srt` | `assemble` (`--format mp4`, the default) | Result and subtitles |
| `final.fcpxml` or `final.otio`, `final.srt` | `assemble --format fcpxml` or `--format otio` | Timeline referencing the clips (no `final.mp4`) and subtitles |
| `<name>-final.mp4` (or `.fcpxml` / `.otio`), `<name>-final.srt` | `edit` | Copy of the final video (or timeline) and its subtitles; `<name>` is the first input's name (a file's name without extension, or the folder's name) |
| `.assemble-tmp-<id>/` | `assemble <project>`, `edit --no-keep-clips` | Temporary clips, removed when the run ends; never edit them |
| `crashes.json`, `telemetry.json`, `mix_segments.json` | action-cam commands; `crashes.json` also `analyze` (action footage) | See the action-cam page |
| `progress.json` | `analyze`, `edit`, `events`, `crashes`, `profile`, `360 analyze`, `clips` | Last run: `command`, `argv`, `cwd`, `status`, `steps`. A first `analyze` or `edit` writes it only after its first source is transcribed. Read by `resume` |
| `.<folder>.cutpoint-resume/` | `clips`, `extend` (after a failed or interrupted export) | Finished clips kept for the next run; not a clips folder, leave it alone |
| `activity.json` | `profile`, `crashes` | Activity profile in use: `name`, `source`, `profile` |

## When something fails

Read the `ERROR` line, then
[docs/troubleshooting/README.md](https://github.com/FerventOrange/cutpoint/blob/main/docs/troubleshooting/README.md).
The usual causes: ffmpeg not on `PATH`, a missing optional extra, Ollama not reachable (only
affects `--vision`), a non-empty output folder.

Whisper runs on the CPU unless the `gpu` extra is installed; if the user has an NVIDIA GPU and
`transcribe` logs an INFO line about missing CUDA libraries, add `gpu` with the install command
for their install (next list).

Cutpoint comes as full (every feature) or lite (no torch: no `events` and `gpu` extras).
`cutpoint capabilities --json` names it in `outputs.edition`. A feature that needs a missing extra
says it is "not available in Cutpoint lite" and prints the command for the install. Tell the user the
command for theirs:

- Tool install (the installer): run the installer again without `--lite` (`-Lite` for
  `install.ps1`) and with `CUTPOINT_LITE` unset, or it installs lite again. It picks the torch
  build for the machine and is also how to upgrade (`uv tool upgrade` drops the torch build).
  `curl -LsSf https://raw.githubusercontent.com/FerventOrange/cutpoint/main/install.sh | sh`
  (Linux, macOS) or
  `irm https://raw.githubusercontent.com/FerventOrange/cutpoint/main/install.ps1 | iex`
  (Windows PowerShell). A tool installed from git or a local path needs its source again; the
  hint prints both commands with it, for example
  `curl -LsSf .../install.sh | sh -s -- --source git@main` and
  `$env:CUTPOINT_SOURCE = 'git@main'; irm .../install.ps1 | iex`. The installers only install
  from github.com/FerventOrange/cutpoint; for a fork the hint prints a `uv tool install` command
  with `--torch-backend` instead, as for a wheel or sdist URL.
- From a clone: `uv sync`. It installs every extra (a large download with torch and CUDA); a
  lite clone was synced with `--no-group full` or `UV_NO_GROUP=full`, so drop those.

`ui`, `keyring` and `ski` are not extras any more: the browser workspace, the keyring and
action-cam telemetry are in every install.

## Rules

- Ask the user before running `cutpoint setup` (about 2.5 GB of downloads, 3.5 GB with an
  NVIDIA GPU), `cutpoint stage` (copies footage to local disk), `clips --force` or `edit --force`
  over an existing folder or `cutpoint skill install` (writes to their Claude Code settings
  folder).
- Do not paste Insta360 SDK files, documentation or output into any file or message, other than
  `error:` lines.
- Do not invent options. Check `--help`.
