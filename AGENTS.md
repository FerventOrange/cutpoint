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
| CLI `cutpoint` with the commands below | `cutpoint edit`, `cutpoint init` |
| Result files in the project folder (JSON and text) | JSON on stdout for every command |
| Activity profiles for crash detection: `cutpoint profile`, `--context` on `profile` and `crashes` | Agent skill, profiles for the other layers and for games |

If the user asks for "edit this video and make it a montage" in one step, do it with the loop
below.

## Calling the CLI

- If installed as a tool (`uv tool install cutpoint`), run `cutpoint <command> ...`; from a
  clone, `uv run cutpoint <command> ...`. The examples below use `uv run cutpoint`; with a tool
  install, drop `uv run`. First-time setup is in
  [docs/getting-started.md](https://github.com/FerventOrange/cutpoint/blob/main/docs/getting-started.md).
  Check it works with `cutpoint paths` (or `uv run cutpoint paths`).
- Exit codes: `0` success, `1` error, `130` interrupted, `2` bad arguments.
- Logs and errors go to stderr as `HH:MM:SS LEVEL message`; errors are `ERROR` lines. Use
  `cutpoint -v <command>` (flag before the command) for debug output and tracebacks.
- Paths inside a project must start with `projects/<name>/...` (forward slashes work on every OS; do not use backslashes in bash) or be absolute; a bare name means
  the project itself.
- Results are files, not stdout. Read the JSON files below instead of parsing logs.
- Commands are re-runnable. Per-source results are cached (path, size, modification time).
  `clips` refuses a non-empty output folder without `--force`, which replaces only the clips its
  `manifest.json` lists, once every new clip is exported (other files, sources included, stay;
  clips renamed or added by hand stay and are assembled; a WARNING names them, so tell the
  user, and move them out only if they are clips you exported, never delete a file you did not
  export).
- Transcription, `events` and export can take minutes to hours. Run them in the background or
  with a long timeout.
- Never modify or delete the user's source videos. You may delete or rename clips in a project's
  `clips` folder during review.
- Use the real options: `cutpoint <command> --help` is authoritative;
  [docs/commands.md](https://github.com/FerventOrange/cutpoint/blob/main/docs/commands.md)
  explains them.

## The full edit

Given a request such as "edit D:\Videos\clip.mp4 into a funny-moments montage":

1. **Pick a style** from the request: `funny`, `gaming`, `action`, `sport`, `commentary`
   (long talk), `ski` (action-cam with telemetry). Footage that is not any of these: use `action`
   or `gaming`.
2. **Analyze.**

   ```powershell
   uv run cutpoint analyze D:\Videos\clip.mp4 --style funny --out clip-edit
   ```

   On Linux: `uv run cutpoint analyze ~/Videos/clip.mp4 --style funny --out clip-edit`.

   Writes `highlights.json` and `transcript.txt` into the project folder (`<projects>\clip-edit`, see
   `cutpoint paths`).
3. **Optional, game footage or when a model-based pass is requested:**
   `uv run cutpoint events clip-edit` (needs the `events` extra and `cutpoint setup`). It
   re-ranks `highlights.json` and writes `events.json` and `events.txt`. If it reports a missing
   module, tell the user how to add the extra (see [When something fails](#when-something-fails)).
4. **Choose moments.** Read `highlights.json` (`candidates` with `id`, `source`, `start`, `end`,
   `score` 0-100, `label`, `transcript`; `suggested_top`) and, for banter and funny moments,
   `transcript.txt` (`[file.mp4 00:12:31.4-00:12:38.9] text`). Then either:
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
   rename to reorder (for example `005_x.mp4` to `001b_x.mp4`).
6. **Assemble.**

   ```powershell
   uv run cutpoint assemble projects/clip-edit/clips --transition fade --loudnorm
   ```

   Writes `final.mp4`, `final.fcpxml` and `final.otio` in the project folder.
7. **Report** the paths of `final.mp4` and the clips folder.

Variants:

| Request | Do |
|---|---|
| Short clip or reel for social media | `clips <project> --vertical --max-total 60`, then `assemble projects/<project>/clips_vertical --captions` |
| Highlights of a specific length | `clips <project> --max-total <seconds>` |
| "Extra" instruction such as "only funny moments" or "keep the speech intact" | Choose `--style`, filter candidates by `transcript`/`label`, use `--segments`. The CLI does not parse these instructions |
| The user says what the footage is ("downhill mtb, chest mount, muddy lens") | Before `crashes`, run `profile <project> --context "<the user's words>"`; check the choice with `profile <project> --json`. Use `--activity NAME` only when the user names a sport that has a profile. Do not edit profile files |
| Room to trim in an editor | `clips ... --extend 1.5` or `extend <clips_dir> --seconds 2` |
| 360 (Insta360 X3) footage | [docs/workflows/360.md](https://github.com/FerventOrange/cutpoint/blob/main/docs/workflows/360.md) |
| Action-cam with telemetry | [docs/workflows/action-cam.md](https://github.com/FerventOrange/cutpoint/blob/main/docs/workflows/action-cam.md) |
| Gameplay with kill feeds | [docs/workflows/game-streams.md](https://github.com/FerventOrange/cutpoint/blob/main/docs/workflows/game-streams.md) |

## Project files

| File | Written by | Content |
|---|---|---|
| `highlights.json` | `analyze`, `events` | Ranked candidates and settings |
| `transcript.txt` | `analyze` | Time-stamped transcript |
| `events.json`, `events.txt` | `events` | Detected sounds, HUD events, model descriptions |
| `clips/manifest.json` | `clips`, `extend` | Per clip: source, start, end, score, label, tail, crop |
| `final.mp4`, `final.fcpxml`, `final.otio` | `assemble` | Result and timelines |
| `crashes.json`, `telemetry.json`, `mix_segments.json` | action-cam commands | See the action-cam page |
| `activity.json` | `profile`, `crashes` | Activity profile in use: `name`, `source`, `profile` |

## When something fails

Read the `ERROR` line, then
[docs/troubleshooting/README.md](https://github.com/FerventOrange/cutpoint/blob/main/docs/troubleshooting/README.md).
The usual causes: ffmpeg not on `PATH`, a missing optional extra, Ollama not reachable (only
affects `--vision`), a non-empty output folder.

Whisper runs on the CPU unless the `gpu` extra is installed; if the user has an NVIDIA GPU and
`transcribe` logs an INFO line about missing CUDA libraries, add `gpu` the same way as below.

A missing extra `X` (for example `events`) is added differently per install. Tell the user the
command for theirs:

- From a clone: `uv sync --extra X`, listing every other extra in use too (the error prints this
  form).
- Tool install: `uv tool install "cutpoint[X]" --reinstall`, listing every extra in use. For a
  PyPI or wheel install (not one from a clone directory or git URL, with `uv tool` or `uv pip`) with an NVIDIA GPU and the
  `events` extra, add
  `--index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match` so torch
  uses CUDA ([getting started](https://github.com/FerventOrange/cutpoint/blob/main/docs/getting-started.md#install-as-a-tool-optional)).

## Rules

- Ask the user before running `cutpoint setup` (about 8 GB of downloads), `cutpoint stage`
  (copies footage to local disk) or `clips --force` over an existing folder.
- Do not paste Insta360 SDK files, documentation or output into any file or message, other than
  `error:` lines.
- Do not invent options. Check `--help`.
