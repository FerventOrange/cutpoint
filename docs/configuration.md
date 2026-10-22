# Configuration

Cutpoint needs no configuration for a first run. Everything below is optional.

## Data folder: CUTPOINT_HOME

`cutpoint paths` prints the resolved locations. The default root is `%LOCALAPPDATA%\cutpoint`
on Windows and `~/.local/share/cutpoint` (or `$XDG_DATA_HOME/cutpoint`) elsewhere. Set the
`CUTPOINT_HOME` environment variable to change it.

```text
CUTPOINT_HOME/
  config.json      settings (optional)
  cache/           per-source analysis cache; safe to delete
  models/          PANNs, CLIP, detector, aesthetic head, Silero VAD, Qwen2.5-Omni
  tools/           llama.cpp build (for `cutpoint events`)
  stage/           local copies made by `cutpoint stage`; disposable
  styles/          learned styles from `refmatch` / `fit` (JSON, editable)
  presets/         HUD presets written by region discovery
  profiles/        activity profiles built by `cutpoint profile` / `crashes` (JSON)
  projects/        project folders, unless `projects_dir` says otherwise
  sdk/insta360/    optional: your own Insta360 Media SDK (see docs/insta360-sdk.md)
  360-stitched/    optional: SDK stitch output (large)
  360-studio/      optional: Insta360 Studio equirect exports
```

Folders are created when a command needs them.

## Environment variables

All optional.

| Variable | Purpose |
|---|---|
| `CUTPOINT_HOME` | Data root |
| `CUTPOINT_CONFIG` | Alternative path to `config.json` |
| `CUTPOINT_CACHE_DIR` | Cache location |
| `CUTPOINT_MODELS_DIR`, `CUTPOINT_TOOLS_DIR` | Model weights, llama.cpp |
| `CUTPOINT_PROJECTS_DIR` | Projects folder (overrides `projects_dir`) |
| `CUTPOINT_STAGE_DIR` | Folder for `cutpoint stage` copies (overrides `stage_dir`) |
| `CUTPOINT_STYLES_DIR` | Learned styles |
| `CUTPOINT_FFMPEG`, `CUTPOINT_FFPROBE` | Full paths to ffmpeg and ffprobe |
| `CUTPOINT_LLAMA_SERVER` | Full path to `llama-server` |
| `CUTPOINT_PANNS_URL` | URL for the PANNs checkpoint, tried before the built-in mirrors (see [models troubleshooting](troubleshooting/models.md#cutpoint-setup-fails-or-is-incomplete)) |
| `CUTPOINT_VISION_MODEL` | Ollama model for `analyze --vision` and for building activity profiles (default `qwen3-vl:8b`) |
| `CUTPOINT_INSTA360_SDK` | Full path to the SDK's `MediaSDKTest` executable (overrides `pano.sdk_exe`) |
| `OLLAMA_HOST` | Ollama URL; overrides `ollama_url` (a bare `host:port` gets `http://`) |

## config.json

Put `config.json` in `CUTPOINT_HOME` (or point `CUTPOINT_CONFIG` at it). Copy
[`config.example.json`](../config.example.json) and keep only the keys you need. Keys starting
with `_` are ignored.

```json
{
  "projects_dir": "D:\\Videos\\cutpoint-projects",
  "source_remap": {
    "D:\\Videos": "E:\\Videos-backup"
  },
  "source_remap_exclude": ["*unverified*"],
  "stage_dir": "C:\\Cutpoint\\stage",
  "player_tags": ["YourGameName"],
  "ollama_url": "http://127.0.0.1:11434"
}
```

JSON needs doubled backslashes in Windows paths, or use forward slashes.

### projects_dir

Where project folders live. Default `<CUTPOINT_HOME>\projects`. A project argument that is a bare
name (`my-edit`) or starts with `projects/` is placed here. That applies to `analyze --out`,
`clips <project>`, `events <project>`, and, with a `projects/` prefix, to `clips --out`,
`clips --segments`, `assemble <clips_dir>` and `assemble --out`. Absolute paths, and relative
paths that exist in the current folder, are used as given.

### source_remap and source_remap_exclude

`{"old prefix": "new prefix"}`. A source under an old prefix is read from the new prefix when
the file exists there. Use it when footage moves to another drive or you work from a backup
copy. Projects keep the original paths in their records, so nothing has to be re-analysed;
only reads are redirected. `analyze` also accepts an input whose original path is gone when
the remap maps it to an existing location, and still records the original spelling. The analysis cache key uses the canonical (original) path, size and
modification time, so a copy that preserves the modification time keeps its cache.
`source_remap_exclude` lists file-name or path globs that are never remapped (for example a
copy you have not verified).

Projects analyzed with a remap by cutpoint 0.1.0 recorded the copy paths. Run `analyze` again to
record the original paths, then run `crashes` and `events` again: their existing results still
name sources by the copy path, so event boosts and ski crash must-includes stop matching until
they are regenerated.

On Windows, a prefix on a mapped network drive matches in either form: `Z:\Footage` and
`\\server\share\Footage` are the same folder, also while `Z:` is disconnected. Projects name
sources on a mapped drive in UNC form (`settings.inputs` keeps your spelling).

Prefix and key matching is case-sensitive on Linux (`source_remap` prefixes, `stage` keys) and
stays case-insensitive on Windows.

### stage_dir

Local folder for `cutpoint stage` copies. Default `<CUTPOINT_HOME>\stage`.

### player_tags

Your in-game name(s) for game-HUD OCR in `cutpoint events`, so kills and deaths are attributed
to you. Matching is fuzzy and ignores `_ . -` and spaces. `cutpoint events --player NAME`
overrides it per run. Default: none.

### ollama_url

Ollama base URL for `analyze --vision` and the model layers of `events`. `OLLAMA_HOST` overrides
it. Default `http://127.0.0.1:11434` (not `localhost`, which can hang on some Windows machines; see
[troubleshooting](troubleshooting/models.md)).

### pano

Settings for the optional 360 module. Every key is optional.

| Key | Meaning |
|---|---|
| `sdk_exe` | Path of the Insta360 Media SDK executable. `CUTPOINT_INSTA360_SDK` overrides it. Default: auto-detected under `<CUTPOINT_HOME>\sdk\insta360\`. See [insta360-sdk.md](insta360-sdk.md) |
| `stitched_dir` | Where `cutpoint 360 stitch` writes equirects. Default `<CUTPOINT_HOME>\360-stitched`. About 54 GB per hour of footage |
| `stitch_timeout_s` | Kill one clip's stitch after this many seconds. Default: no limit |
| `stitch_args` | SDK arguments placed after the input and output. Replaces the defaults. See [insta360-sdk.md](insta360-sdk.md) |
| `studio_dirs` | Folders searched for Insta360 Studio equirect exports. Default `<CUTPOINT_HOME>\360-studio`; `_stitched\`, `export\` and `Export\` next to the camera files are always searched |
| `default_style` | `--style` of `cutpoint 360 analyze` when none is given. Default `ski` |
| `cmd_every` | Frames between virtual-camera updates when rendering. Default 1. 2 renders faster but leaves camera shake on every other frame. `--pano-cmd-every` overrides it |
| `level_strength` | Horizon lock of the stabilised view, 0 to 1. 1 keeps the horizon level; 0 lets the view lean with the camera's smoothed tilt. Default 0.7 |
| `tilt_follow_s` | Smoothing in seconds of the tilt the view follows when `level_strength` is below 1. Default 0.5 |
| `imu_offset_s` | Time offset in seconds between the motion sensor and the video, if a clip's correction looks early or late. Default 0 |
| `equirect_yaw_offset` | Degrees added to the director's yaw for Studio exports whose yaw 0 is not the camera's forward direction. Default 0 |
| `director` | `DirectorParams` overrides for `cutpoint 360 director`, for example `{"max_speed": 35, "min_hold_s": 3.0}`. `--param KEY=VALUE` overrides them. Unknown keys and bad values are errors |

## Styles

A style tells `analyze` how to score and how long clips and the final cut should be.

| Style | Clip length | Target total | Notes |
|---|---|---|---|
| `gaming` | 4-12 s | 90 s | Loudness and hype words, capped keywords |
| `action` | 3-12 s | 60 s | Loudness and scene cuts, plus laughter |
| `sport` | 5-20 s | 90 s | Crowd loudness, "goal", "what a ...", laughter |
| `funny` | 4-20 s | 60 s | Laughter-heavy |
| `commentary` | 20-120 s | 12.5 min | Speech continuity, sentence-snapped, chronological |
| `ski` | 2-10 s by moment type | 150 s | Action-cam telemetry, model tags and sound events; lifts and stops masked |

`ski` is the action-cam style. Speed, airtime and crash scoring need GoPro HERO 9-13 telemetry (or the Insta360 X3 IMU); other footage is scored from audio, speech and model features only.

Learned styles are JSON files in `<CUTPOINT_HOME>\styles\<name>.json`, written by `cutpoint
refmatch` and `cutpoint fit`. They appear as extra `--style` choices, hold the feature weights
and the fit report, and can be edited by hand. Style definitions live in `src/cutpoint/styles.py`.

## Activity profiles

Crash detection uses an activity profile per project (`<project>\activity.json`): built-in
profiles in `src/cutpoint/presets/activities/`, plus the ones Cutpoint builds with the local LLM,
saved as `<CUTPOINT_HOME>\profiles\<name>.json`. A file there with a built-in's name replaces
that built-in for profile selection. The files are readable JSON, but nothing has to be edited:
see [action-cam workflow](workflows/action-cam.md#activity-profiles). Profiles for the other
scoring layers and for games are planned.

## HUD presets

Game-HUD OCR regions come from a preset, looked up in this order:

1. `<project>\hud_presets\<game>.json`
2. `<CUTPOINT_HOME>\presets\<game>.json` (region discovery writes here)
3. bundled presets in `src/cutpoint/events/presets/` (`wt.json`, War Thunder)
4. built in: `r6` (Rainbow Six Siege)

See [game streams](workflows/game-streams.md).
