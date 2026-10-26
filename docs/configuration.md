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
| `CUTPOINT_LLAMA_BACKEND` | Backend `cutpoint setup` downloads for llama.cpp: `auto` (default), `cuda`, `vulkan` or `cpu`. Auto takes CUDA only when a usable NVIDIA driver is found, else Vulkan, else CPU; a named backend is never replaced by another one |
| `CUTPOINT_VIDEO_ENCODER` | Video encoder for exported clips, assembled videos and rendered 360 clips: `auto` (default), `nvenc`, `amf`, `vaapi` or `cpu`. Overrides `video_encoder` |
| `CUTPOINT_PANNS_URL` | URL for the PANNs checkpoint, tried before the built-in mirrors (see [models troubleshooting](troubleshooting/models.md#cutpoint-setup-fails-or-is-incomplete)) |
| `CUTPOINT_VISION_MODEL` | Ollama model for `analyze --vision` and for building activity profiles (default `qwen3-vl:8b`) |
| `CUTPOINT_INSTA360_SDK` | Full path to the SDK's `MediaSDKTest` executable (overrides `pano.sdk_exe`) |
| `OLLAMA_HOST` | Ollama URL; overrides `ollama_url` (a bare `host:port` gets `http://`) |

When `setup` finds an existing managed llama.cpp build for a different backend than the one
selected now, it replaces it through the normal staged installation, including in `auto` mode.
For example, an old CUDA build on an AMD-only machine is replaced with a Vulkan build.

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
  "ollama_url": "http://127.0.0.1:11434",
  "video_encoder": "auto",
  "vision": false,
  "compress_peaks": "off",
  "analysis": {"skip": []}
}
```

JSON needs doubled backslashes in Windows paths, or use forward slashes.

### projects_dir

Where project folders live. Default `<CUTPOINT_HOME>\projects`. A project argument that is a bare
name (`my-edit`) or starts with `projects/` is placed here. That applies to `analyze --out`,
`clips <project>`, `events <project>`, and, with a `projects/` prefix, to `clips --out`,
`clips --segments`, `assemble <clips_dir>` and `assemble --out`. Absolute paths, and relative
paths that exist in the current folder, are used as given. The UI's Settings (gear) dialog writes
this key.

### source_remap and source_remap_exclude

`{"old prefix": "new prefix"}`. A source under an old prefix is read from the new prefix when
the file exists there. Use it when footage moves to another drive or you work from a backup
copy. Projects keep the original paths in their records, so nothing has to be re-analysed;
only reads are redirected. `analyze` also accepts an input whose original path is gone when
the remap maps it to an existing location, and still records the original spelling. The analysis cache key uses the canonical (original) path, size and
modification time, so a copy that preserves the modification time keeps its cache.
`source_remap_exclude` is a list of file-name or path globs (a single string is rejected) that are never remapped (for example a
copy you have not verified).

Projects analyzed with a remap by cutpoint 0.1.0 recorded the copy paths. Run `analyze` again to
record the original paths, then run `crashes` and `events` again: their existing results still
name sources by the copy path, so event boosts and ski crash must-includes stop matching until
they are regenerated.

On Windows, a prefix on a mapped network drive matches in either form: `Z:\Footage` and
`\\server\share\Footage` are the same folder, also while `Z:` is disconnected. Projects name
sources on a mapped drive in UNC form (`settings.inputs` keeps your spelling).

A key spelled the Windows way (`D:\Videos`) also works on Linux and macOS: it matches
Windows-style sources case-insensitively with either slash (drive-letter forms), for example
`D:\Videos` -> `/mnt/d/Videos`. A network path (`\\server\share\...`) must be written with
backslashes in both the key and the source (in JSON, `"\\\\server\\share"`).

Prefix and key matching is case-sensitive on Linux (`source_remap` prefixes, `stage` keys), except
for Windows-form `source_remap` keys and the `stage` keys of Windows-form sources (above; when
reading a staged copy, case below the `source_remap` prefix must match the staged file, and files
there that differ only in case are staged separately), and case-insensitive on Windows.

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

### vision

`true` or `false` (default). With `true`, `analyze` and `edit` rate the top candidates with the
local Ollama vision model ([getting started](getting-started.md#optional-ollama-vision)) unless
`--no-vision` is given; with `false` or no key, only `--vision` turns it on for one run. The UI's
Settings (gear) dialog writes this key ("Rate frames with a local vision model (Ollama)"), and
every analysis the UI starts ("Analyze footage" or "Analyze again") passes its value at that
moment as `--vision` or `--no-vision`; the UI has no per-run choice. Other values are rejected.

### compress_peaks

`off` (default), `light` or `strong`: the [peak compression](commands.md#peak-compression)
that `clips`, `assemble` and `edit` use when `--compress-peaks` is not given (`light` and
`strong` include a makeup gain that keeps the overall loudness, at most 6 and 10 dB). Other
values are rejected. `cutpoint ui` starts its "Compress peaks" select on this value and passes the choice
to its exports and assemblies; it does not change the key.

### analysis

`{"skip": [...]}`. Analysis steps that `analyze` and `edit` leave out by default: any of
`reactions`, `keyframes` and `crashes`. `--skip` on the command overrides it for one run.
Default: none. Other step names are rejected.

### video_encoder

`auto` (default), `nvenc`, `amf`, `vaapi` or `cpu` (libx264). `CUTPOINT_VIDEO_ENCODER` overrides it. `auto` really encodes a short test clip with each candidate (NVENC, then AMF on Windows or VAAPI and AMF on Linux; on Linux every `/dev/dri/renderD*` node is tried) and uses the first one that works, else libx264. A named GPU encoder that does not work is an error, not a silent fallback (the error suggests `cpu`). With `auto`, a GPU encode that fails on real footage is redone once on the CPU with a WARNING, and the rest of the run uses the CPU. `cutpoint capabilities` shows what works on this machine.

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
3. the bundled presets folder `src/cutpoint/events/presets/` (no game presets are bundled)

The first preset whose `game` key (or file name, without `game`) matches wins: 1 overrides 2 and 3,
and 2 overrides 3. To make a preset for a game, write one
in folder 1 or 2, or let region discovery create it. The user folder (2) is trusted like the bundled one:
the `"builtin"` key is honoured there and in the bundled folder, so a preset with `"builtin": true` in
`<CUTPOINT_HOME>\presets` keeps its built-in interpretation and OCR cache keys. In a project
`hud_presets` folder (1) `"builtin"` is ignored with a WARNING (shown once) and the preset is read
as a generic JSON preset (generic kill-feed parsing, generic banners added, new OCR cache).

A builtin preset's `banner_phrases` are not part of the OCR cache key. After editing them, run
`cutpoint events <project> --force ocr` to read the footage again.

The OCR layer also keeps its raw read of each source (the text it read per frame) in the source's
cache folder, `ocr-raw-*`. When an update changes only how that text is interpreted, the next run
replays the raw read in seconds instead of reading the footage again; a change of the preset's
regions or `feed.newest`, or of the OCR frame rate, reads it again, and an interrupted read resumes
near where it stopped. `--force ocr` drops the raw read too. It holds OCR text: it is git-ignored and
never written inside a git checkout.

A preset may list `candidate_phrases`: banner texts mined from the discovery OCR. They are inactive
and not part of any cache key; to have OCR look for one, move it into `banner_phrases`. They are
texts of 2 to 5 words read in the banner and notice regions (and in the screen centre, text at
least 36 px tall at 1080p, as the OCR layer reads it there; smaller centre text is mined through
the notice regions) of at least 2 sampled frames that are at least 2 s apart (two frames
of one burst can count), in the casing the game shows ("Shield Saver" or "SHIELD SAVER"). They
may be part of a longer line or sentence: words OCR joined are split ("blockedShieldSaver" is
"blocked Shield Saver"), and numbers, timestamps, gamertags and your player tags are cut out.
Matching ignores case and spaces, so a phrase also finds text OCR read without spaces
("Hullbreached" for "Hull breached"), but only whole words: a phrase is not found inside a longer
word. Static hint lines (tutorial and control hints) are left out: a text found in at least 25%
of a recording's sampled gameplay frames whose sightings span at least half of the sampled time
(tested for recordings with at least 8 sampled bursts). An event banner shows for a few seconds
per event, far below that share; text in at least half of all gameplay frames is a static label
and left out too. A hint shown only briefly is left out when OCR read it a little differently
each time (words joined in one read): a phrase found only inside sentences of more than 5 words
(a number, gamertag or separator such as `|`, `/` or `:` ends a sentence; commas and brackets do
not), with the same text right around it each time, is a part of one sentence. A phrase in log lines with changing names or gamertags around it, one also
shown on its own, or one OCR reads in a box with a stats line after a separator or number, stays. Discovery leaves out menu and scoreboard frames (text across most of the screen centre, or
far more text than the recording usually shows), so a busy game HUD is not mistaken for a menu.
With a JSON preset, the OCR layer applies the same test to the text it detects in the preset's
regions other than feeds (a full kill feed is no menu) and drops what menu frames show. It also
treats a screen that holds still as a menu (hangar and crew screens show as little text as a busy
HUD): a 32x18 grey thumbnail of the HUD area within 3 grey levels of the first frame, more than 2 s
from any feed sighting or kill, for at least 5 s when new menu text shows (at least 2 text lines
that are no preset phrase, first read in the still stretch or up to 2 s before it, each on screen
for at least half of it) and at least 10 s otherwise, so
still gameplay (scoped, about 6 s) keeps its banners; it runs only when the bounding box of the
preset's regions is at least a quarter of the frame. Feed sightings and kills are never dropped,
by any menu test. A discovered kill feed takes in the rows above and below
it that show only when the feed is full (up to twice its height), counted from its outermost
rows seen in enough frames, so a stray box at its edge does not stop the growth. A row is added
when it shows in 5% of the feed's frames (at least 2) and in 2 bursts with changing text; as a
full feed is rare, a row seen in only 2 frames is also added when those frames are in 2 bursts,
it shows a different text in each burst (ignoring case, digits and punctuation, so a static label
is not added) and at least half of its frames show the full feed (the feed's row next to it and
all others but one, as OCR misses a box now and then; so a notice or chat line next to the feed
is not added).

### Event types and prompt keys

A preset may add these optional keys. Without them, scoring, clip tails and VLM windows behave as
before.

```json
{
  "title": "Example Game",
  "view": "first-person",
  "hype": "  hype: 0-1 ...\n",
  "events": [
    {"name": "objective",
     "when": [{"type": "banner", "prefix": ["OBJECTIVE CAPTURED"]}],
     "score": 6, "max": 2, "payoff": true, "tail_s": 2.5, "seed": 1.5}
  ]
}
```

- `title`, `view`, `hype`: the game name, the camera view and the hype rubric line in the VLM
  prompt. Unset `title`: a name built from the game key (`some-game` becomes "Some Game"); unset `view`: not mentioned; unset `hype`: the generic rubric. A `hype` value is trimmed, then indented and ended with a newline, so a plain one-liner works.
- `events`: custom event types. Each has `name` (the key in `signals.events`), `when` (matchers;
  any one is enough) and optional `score` (points per event, clamped to 0-12; unset: no boost),
  `max` (events counted per candidate, clamped to 1-3, default 1), `payoff` (true: the event ends
  a clip tail), `tail_s` (seconds kept after the payoff, clamped to 0-10) and `seed` (priority when choosing
  VLM windows, clamped to 0-3). Unset `payoff`, `tail_s` and `seed` keep the built-in rules.
  Two more fields: `reaction` (points added once per candidate when a reaction follows one of
  the type's events, clamped to 0-12; unset: none) and `styles` (a style name or a list of names;
  `score` and `reaction` count only in a project with one of these styles, and a learned style
  counts through its base; unset: every style). `styles` gates only the points: matching, tail
  payoffs and seeds ignore it. An unknown style name invalidates the entry. When `reaction` applies
  to the project's style, the evidence in `signals.events` gains `<name>_reaction` (0 or 1);
  otherwise the key is absent.
- A matcher is an OCR event `type` (such as `kill`, `banner` or `multikill`) plus at most one
  test: `prefix` or `contains` (lists of upper-case label text; the label must start with, or
  contain, one of them) or `flag` (one or more space-separated fields of the event that must be set
  (true or non-empty), such as `player`, `teamkill` or `player_kill`; a field written `!name` must be
  unset or false, for example `"flag": "player !verb"`). A lone `!` or `!!` is invalid: the type is
  skipped with a warning.
- Preset types are checked before the built-in ones (`teamkills`, `self_kills`, `player_deaths`,
  `player_kills`, `banners`); the first matching type wins, and a type with a built-in's name replaces it.
  `self_kills` matches a kill where the player is both killer and victim (`player_kill` and
  `player_death` set; a teamkill stays a teamkill) and adds 3 points each, half a kill, at most 3
  counted, in every style; it is not a player kill, and like a death it stays a tail payoff and a
  VLM seed.
  `player_deaths` matches the player's own deaths (not a self-kill) and adds no per-event boost, so a death
  does not count as a kill; it stays a tail payoff and a VLM seed. Its default is `reaction` 8 with
  `styles` `["funny"]`: with `--style funny`, a candidate where the player dies and someone
  reacts gets +8, once. A reaction is laughter, a scream or speech that starts within 6 s after
  the death; speech counts only after a 1.5 s pause before the death, so continuous banter and
  laughter already running do not count. It is rescored from the cache without re-running
  `events`. Other styles are unchanged. It needs the player's name in
  [`player_tags`](#player_tags) or `--player`; without it no event is a player event.
- Scored types add to the highlight boost, which stays capped, and show up as evidence in
  `signals.events`. The `prefix` phrases of `banner` matchers are also added to the OCR phrases of
  generic presets, after the generic banners.
- Names `laughter`, `scream`, `boost` and `vlm`, and names ending in `_reaction`, are reserved, and a `payoff` must be true or false.
  Such entries, and other invalid ones, are skipped with a WARNING shown once. Entries of `events`
  that are not objects are dropped without a warning.
- Changing scores, `max` or tails does not re-run OCR. A preset in the user or bundled folder
  never gets a separate VLM cache key. A project preset gets one only when its prompt (name, view,
  hype) differs from the user (or bundled) preset for the same game key, or from the default
  prompt if there is none; its windows are then described again once. Seeds do not affect the
  key. After adding a user-folder preset with `title`, `view` or `hype`, or editing those fields in one, run
  `cutpoint events <project> --force vlm` to describe the windows again. A preset moved from the
  bundled folder to the user folder keeps its OCR and VLM cache keys.

See [game streams](workflows/game-streams.md).
