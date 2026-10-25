---
name: cutpoint
description: Edit videos locally with the Cutpoint CLI. Use when the user asks to edit this video, make highlights, a montage or funny moments, cut clips from this recording, find the best moments in footage, or assemble clips into a finished video. Not for converting, compressing or re-encoding a single video, or general questions about video formats, codecs or players.
---

# Cutpoint

Cutpoint finds the best moments in long recordings, exports them as numbered clips and
assembles a finished video. It runs locally and writes everything into a project folder.

This skill is the short version. For anything beyond the basics (variants such as vertical
reels, 360 and action-cam footage, project files, failures), run `cutpoint guide` and follow
it: it is the full reference.

## First run

1. Check the CLI: `cutpoint paths`. From a clone of the repository, use `uv run cutpoint paths`
   and put `uv run` before every command below.
2. Check that ffmpeg is present: `ffmpeg -version`. If it is not on `PATH`, stop and tell the
   user to install it.
3. Run `cutpoint capabilities` once (GPU, working encoders, what transcription runs on).
4. Tell the user where projects are written: the `projects` line of `cutpoint paths`. They can
   change it in the `cutpoint ui` settings, with the `CUTPOINT_PROJECTS_DIR` environment
   variable or with `projects_dir` in config.json.

## Choose the path

Style (`--style` on `cutpoint edit` and `cutpoint analyze`): go by what the user says about the
footage, not the file name alone. Pick `funny`, `gaming`, `action`, `sport`, `commentary` (long
talk) or `ski` (action-cam with telemetry); anything else: `action` or `gaming`.

One step ("edit this into a montage"):

```text
cutpoint edit <video> --kind highlights|montage|clips|funny-moments
```

It writes the project with the clips, `final.mp4` and `<video>-final.mp4` plus `.srt`
subtitles; the last log lines name them. `--out NAME` names the project.

`--vision` (on `edit` and `analyze`) adds an optional frame rating by a local Ollama vision
model. It is off by default: slower and uses several GB of GPU memory, so pass it only when the
user asks for it. If the user's `config.json` has `"vision": true`, it is on by default and
`--no-vision` skips it.

Choosing the moments yourself:

1. `cutpoint analyze <video> --style <style> --out <project>`. It also runs the reaction, keyframe
   and crash steps it can (skipped with an INFO line when the extra or models are missing;
   `--skip reactions keyframes crashes` leaves them out) and writes `events.json`.
2. Read `highlights.json` (ranked `candidates` with `id`, `start`, `end`, `score`, `label`,
   `transcript`) and `transcript.txt` in the project folder.
3. Export: `cutpoint clips <project> --top 8`, `cutpoint clips <project> --ids h001 h004` or
   `cutpoint clips <project> --segments segments.json` (the format is in `cutpoint guide`).
4. Review: tell the user which clips were exported. Delete clips they reject; rename to reorder
   (`005_x.mp4` to `001b_x.mp4`, and its `.srt` the same way).
5. Assemble: `cutpoint assemble projects/<project>/clips --transition fade --loudnorm`.

## Rules

- Never modify or delete the user's source videos.
- Ask the user before `cutpoint setup` (about 8 GB of downloads), `cutpoint stage` (copies
  footage to local disk), and before `cutpoint clips --force` or `cutpoint edit --force` over a
  non-empty folder.
- Transcription, `cutpoint analyze`, `cutpoint events` and export can take minutes to hours: run them in the
  background or with a long timeout.
- Results are files: read the JSON in the project folder instead of parsing logs. Logs and
  errors go to stderr as `HH:MM:SS LEVEL message`; exit code `0` is success.
- `--json` prints the result as one JSON object on stdout (logs stay on stderr): `ok`, `command`,
  `schema`, `project`, `outputs` (absolute paths and key facts), `warnings` (`code`, `message`,
  `files`) and `error` (`code`, `message`, or `null`). It works on `paths`, `analyze`, `clips`,
  `assemble`, `edit`, `extend`, `capabilities`, `mix`, `stage`, `setup`, `events`, `crashes`,
  `profile` and every `360` subcommand.
- Check `cutpoint <command> --help`; never invent options.
- When done, report the absolute paths of `final.mp4` (and the `<name>-final.mp4` copy after
  `cutpoint edit`), its `.srt` and the clips folder, exactly as the last log lines print them;
  do not shorten them. If a WARNING starts with "Kept" (clips renamed or added by hand), tell
  the user which files it names.
- A missing optional extra (an `ERROR` naming a module): tell the user the install command for
  their install; `cutpoint guide` lists both forms (clone and tool install).

## Example prompts

- "Edit D:\Videos\stream.mp4 into a funny-moments video."
- "Make a 60-second montage from ~/Videos/ride.mp4."
- "Find the best moments in this recording and cut them into clips."
- "Cut the intro and the joke at 20 minutes from this talk and join them with fades."
