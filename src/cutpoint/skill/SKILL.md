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

Style (`--style` on `cutpoint edit` and `cutpoint analyze`): `--style auto` is the default, so
give none and it picks from the footage. `cutpoint detect <video> --json` shows its pick (`outputs.styles`); a
`style_ambiguous` warning means it could not tell and used `gaming` (best guess gameplay) or `action`; the warning names the style, and `detect` shows `decided_by`. Unsure footage, and camera footage without telemetry, is shown to the vision model first (`--no-vision` skips it). Name a style only when the user says
what the footage is. Pick `funny`, `gaming`, `action`, `sport`, `commentary` (long talk) or
`action-cam` (action-cam footage with GoPro or Insta360 X3 telemetry). Without any telemetry
`action-cam` runs as `action` with an `action_cam_no_telemetry` warning: tell the user.

One step ("edit this into a montage"):

```text
cutpoint edit <video> --kind highlights|montage|clips|funny-moments
```

It writes the project with the clips, `final.mp4` and `<name>-final.mp4` plus `.srt`
subtitles; the last log lines name them. `--out NAME` names the project. `--no-keep-clips`
removes the clips after the join when the user wants only the final video.

`--vision` (on `edit` and `analyze`) adds an optional frame rating by the vision model (Ollama
by default; the Vision role in the `cutpoint ui` Models tab can route it to a hosted
provider). It is off by default: slower and uses several GB of GPU memory, so pass it only when the
user asks for it. If the user's `config.json` has `"vision": true`, it is on by default and
`--no-vision` skips it.

Choosing the moments yourself:

1. `cutpoint analyze <video> --out <project>` (add `--style <style>` only to override the detected one). It also runs the reaction, keyframe
   and crash steps it can (skipped with an INFO line when the extra or models are missing;
   `--skip reactions keyframes crashes dialogue` leaves them out) and writes `events.json`.
2. Read `highlights.json` (ranked `candidates` with `id`, `start`, `end`, `score`, `label`,
   `transcript`) and `transcript.txt` in the project folder.
3. Export: `cutpoint clips <project> --top 8`, `cutpoint clips <project> --ids h001 h004` or
   `cutpoint clips <project> --segments segments.json` (the format is in `cutpoint guide`).
4. Review: tell the user which clips were exported. Delete clips they reject; rename to reorder
   (`005_x.mp4` to `001b_x.mp4`, and its `.srt` the same way).
5. Assemble: `cutpoint assemble projects/<project>/clips --transition fade --loudnorm`.
   If the user wants only the final video and no clip review, skip steps 3-4:
   `cutpoint assemble <project> --top 8 --transition fade --loudnorm` exports the moments into a
   temporary folder, joins them and removes the clips.

Another style on an analysed project ("more talk", "the funny bits"): `cutpoint rerank
<project> --style NAME --dry-run` shows the new ranking in seconds from the analysis's caches;
without `--dry-run` it saves it (the file before is kept as `highlights.before-rerank.json`;
unchanged moments keep their ids). `error.code` `needs_analyze` means a feature the style needs
is missing: tell the user, and with their OK run the command the ERROR line names
(`rerank ... --analyze`, which analyzes again: minutes to hours). `events` and
`rerank --analyze` keep the re-ranked style; running `analyze` or `edit` again without
`--style` detects the footage again: pass `--style NAME` to keep it.

## Rules

- Never modify or delete the user's source videos.
- Own scoring styles: `cutpoint style list`, `show` and `features` before `create`; never invent feature
  names (`cutpoint style features` lists them). Use a style with `--style NAME`.
- Ask the user before `cutpoint setup` (about 2.5 GB of downloads, 3.5 GB with an NVIDIA GPU),
  `cutpoint stage` (copies footage to local disk), and before `cutpoint clips --force` or
  `cutpoint edit --force` over a non-empty folder.
- Transcription, `cutpoint analyze`, `cutpoint events` and export can take minutes to hours:
  run them in the background or with a long timeout.
- Results are files: read the JSON in the project folder instead of parsing logs. Logs and
  errors go to stderr as `HH:MM:SS LEVEL message`; exit code `0` is success.
- `--json` prints the result as one JSON object on stdout (logs stay on stderr): `ok`, `command`,
  `schema`, `project`, `outputs` (absolute paths and key facts), `warnings` (`code`, `message`,
  `files`) and `error` (`code`, `message`, or `null`). It works on `paths`, `analyze`, `detect`, `clips`,
  `assemble`, `edit`, `extend`, `capabilities`, `mix`, `stage`, `setup`, `events`, `crashes`,
  `profile`, `resume`, `rerank` and every `360` subcommand.
- Cutpoint ships only the `base` and `generic` activity profiles. Others are discovered or made by
  the user or you: `cutpoint profile list|show|create|validate|delete` (`create NAME --from FILE`).
  `--activity NAME` takes a name from `profile list`.
- After an interrupted or failed `analyze`, `edit`, `events`, `crashes`, `profile`,
  `360 analyze` or `clips`, `cutpoint resume <project>` replays the recorded command (`--dry-run` shows it).
  On `low_memory` (too little free memory for a model), ask the user to close other programs,
  then run the command the ERROR line names (`cutpoint resume <project>`, or the same command
  again when no progress.json was written yet).
- If `clips`, `extend` or `assemble` stops with an `ERROR` that another run holds the folder's
  lock (`.<folder>.cutpoint-lock`; with `--json`, `error.code` is `folder_busy`), wait or
  retry; delete the lock file only if no run is active. If `clips --force` or `extend` refuses
  a `manifest.json` Cutpoint did not write, tell the user; never delete it.
- `analyze`, `edit`, `events`, `crashes`, `profile`, `360 analyze`, `clips` and a saving
  `rerank` hold `<project>/.analysis.run.cutpoint-lock`; on `folder_busy` wait or retry, and delete the lock
  only if no run is active.
- Check `cutpoint <command> --help`; never invent options.
- When done, report the absolute paths of `final.mp4` (and the `<name>-final.mp4` copy after
  `cutpoint edit`), its `.srt` and the clips folder, exactly as the last log lines print them;
  do not shorten them. If a WARNING starts with "Kept" (clips renamed or added by hand), tell
  the user which files it names.
- A missing optional extra (an `ERROR` naming a module, "not available in Cutpoint lite"): tell
  the user the install command for their install; `cutpoint guide` lists both forms (installer
  and clone).

## Example prompts

- "Edit D:\Videos\stream.mp4 into a funny-moments video."
- "Make a 60-second montage from ~/Videos/ride.mp4."
- "Find the best moments in this recording and cut them into clips."
- "Cut the intro and the joke at 20 minutes from this talk and join them with fades."
