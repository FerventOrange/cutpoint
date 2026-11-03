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
  models/          PANNs, CLIP, detector, aesthetic head, Silero VAD
  tools/           CLIProxyAPI binary (subscription models of the Models tab)
  stage/           local copies made by `cutpoint stage`; disposable
  styles/          learned styles from `scripts/refmatch.py` (JSON, editable)
  presets/         HUD presets written by region discovery
  profiles/        activity profiles built by `cutpoint profile` / `crashes` (JSON)
  projects/        project folders, unless `projects_dir` says otherwise
  sdk/insta360/    optional: your own Insta360 Media SDK (see docs/insta360-sdk.md)
  360-stitched/    optional: SDK stitch output (large)
  360-studio/      optional: Insta360 Studio equirect exports
  cliproxy/        `cutpoint proxy` state: auths/ (working files while it runs), run/,
                   keys.json, credential-recovery.enc; private to your user
```

`tools/cliproxy/` holds the CLIProxyAPI build `cutpoint proxy` downloads.

Folders are created when a command needs them.

## Environment variables

All optional.

| Variable | Purpose |
|---|---|
| `CUTPOINT_HOME` | Data root |
| `CUTPOINT_CONFIG` | Alternative path to `config.json` |
| `CUTPOINT_CACHE_DIR` | Cache location |
| `CUTPOINT_MODELS_DIR`, `CUTPOINT_TOOLS_DIR` | Model weights, the CLIProxyAPI binary |
| `CUTPOINT_PROJECTS_DIR` | Projects folder (overrides `projects_dir`) |
| `CUTPOINT_STAGE_DIR` | Folder for `cutpoint stage` copies (overrides `stage_dir`) |
| `CUTPOINT_STYLES_DIR` | Learned styles |
| `CUTPOINT_FFMPEG`, `CUTPOINT_FFPROBE` | Full paths to ffmpeg and ffprobe |
| `CUTPOINT_VIDEO_ENCODER` | Video encoder for exported clips, assembled videos and rendered 360 clips: `auto` (default), `nvenc`, `amf`, `vaapi` or `cpu`. Overrides `video_encoder` |
| `CUTPOINT_EXPORT_WORKERS` | Clips `clips` and `edit` export at once: `auto` (default) or a number from 1 to 3. Overrides `export_workers` |
| `CUTPOINT_VIDEO_DECODER` | `cpu` decodes every video on the CPU instead of NVDEC or VAAPI (analysis, HUD OCR, 360, UI proxies); unset or any other value detects the GPU decoder as usual (see [GPU troubleshooting](troubleshooting/ffmpeg-gpu.md#which-video-encoder-is-used-and-why-not-the-gpu)) |
| `CUTPOINT_ANALYZE_OVERLAP` | `0` turns off the background HUD OCR of `analyze` (on by default where it applies; see [analysis steps](commands.md#analysis-steps)). An escape hatch: the output is the same either way |
| `CUTPOINT_ANALYZE_SHARED_DECODE` | Experimental, opt-in: `split` makes the scene detection proxy of `analyze` in the same decode as the background HUD OCR, spooling up to about 2 GB to the cache folder. Same output either way (see [analysis steps](commands.md#analysis-steps)) |
| `CUTPOINT_FADE_COPY` | `0` re-encodes the whole video of `assemble --transition fade` instead of stream-copying between the fades (see [assemble](commands.md#assemble)). An escape hatch: the same number of frames at the same times, the same audio and the same timelines either way |
| `CUTPOINT_PANNS_URL` | URL for the PANNs checkpoint, tried before the built-in mirrors (see [models troubleshooting](troubleshooting/models.md#cutpoint-setup-fails-or-is-incomplete)) |
| `CUTPOINT_VISION_MODEL` | Ollama model for `analyze --vision`, the VLM layer of `cutpoint events`, `360 director --vl-check` and building activity profiles (default `qwen3.5:9b`; `qwen3-vl:8b`, the earlier default, still works); the AI Chat falls back to it when `qwen3.5:9b` is not pulled |
| `CUTPOINT_INSTA360_SDK` | Full path to the SDK's `MediaSDKTest` executable (overrides `pano.sdk_exe`) |
| `OLLAMA_HOST` | Ollama URL; overrides `ollama_url` (a bare `host:port` gets `http://`) |
| `CUTPOINT_CLIPROXY_PORT` | Loopback port of the [`cutpoint proxy`](commands.md#proxy) daemon (default 18317, or the port saved at the last start) |
| `CUTPOINT_CLIPROXY_BINARY` | CLIProxyAPI executable for `cutpoint proxy` to run instead of the pinned build it downloads |
| `CUTPOINT_PROGRESS` | `1` adds `PROGRESS {...}` lines per step on stderr (see [progress output](commands.md#progress-output)) |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, `CUTPOINT_ASSISTANT_API_KEY` | API keys of the hosted models in `cutpoint ui`, each sent only to its own host (see [assistant](#assistant)) |

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
  "export_workers": "auto",
  "vision": false,
  "compress_peaks": "off",
  "color": "off",
  "analysis": {"skip": []},
  "assistant": {"provider": "ollama", "model": "", "context_tokens": 8192}
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
name sources by the copy path, so event boosts and action-cam crash must-includes stop matching until
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

Ollama base URL for `analyze --vision`, the VLM layer of `events`, `360 director --vl-check`,
the AI Chat and new activity profiles. `events` also uses it to unload Ollama models before its
other model layers and to build crash profiles. `OLLAMA_HOST`
overrides it. Default `http://127.0.0.1:11434` (not `localhost`, which can hang on some Windows machines;
see [troubleshooting](troubleshooting/models.md)).

### vision

`true` or `false` (default). With `true`, `analyze` and `edit` rate the top candidates with the
configured vision model
([getting started](getting-started.md#optional-ollama-for-frame-rating-and-ai-chat)) unless
`--no-vision` is given; with `false` or no key, only `--vision` turns it on for one run. The UI's
Models tab writes this key ("Rate frames with vision model"), and
every analysis the UI starts ("Analyze footage" or "Analyze again") passes its value at that
moment as `--vision` or `--no-vision`; the UI has no per-run choice. Other values are rejected.

### vision_assistant

The model that rates frames and describes the windows of the `events` VLM layer, chosen
independently of the chat model. The same keys as
[`assistant`](#assistant) (`provider`, `model`, `url`, `consent`), but `provider` is `ollama`
(default), an API (`anthropic`, `openai` or `xai`, with your API key from the environment or the
keyring, as for the assistant; analyses started in `cutpoint ui` read it from the keyring only,
which the Vision role's key field in the Models tab sets) or a subscription (`claude-code`,
`codex` or `antigravity`). With `ollama`, `model` and `url` are the Ollama model and URL
(`--ollama-model`, `CUTPOINT_VISION_MODEL` and `--ollama-url` still win); an Ollama URL on
another machine (not `localhost` or a loopback address) is hosted and needs the Vision consent
below, else vision is skipped with a WARNING. With an API provider, choose a `model`
that accepts images; each request sends frames and is billed to your API key. Subscription
providers use only their official host; a `url` for them is rejected. An API provider may have a
`url` (an OpenAI-compatible or self-hosted server); one on this machine (`localhost` or a
loopback address) needs no Vision consent, because nothing leaves it.
The Models tab fetches the catalog from this configured Ollama URL, including custom servers.
Without this section the frame rater uses Ollama. The Models tab of `cutpoint ui` writes it.
Catalogs and frames go to a hosted provider only after you allow it there
(`vision_assistant.consent`, via "Allow vision requests"); choosing or saving a model never
grants it. Once allowed, the provider receives video frames, transcript excerpts, file names and
times, and detected on-screen text (such as player names) and telemetry (for frame ratings and
the `events` window descriptions), never the video file. That is more data, and more risk, than
the assistant gets, so this consent is separate from the assistant's: agreeing to one never
covers the other. Entries are `vision-v2:<provider>@<host>` (for example
`vision-v2:codex@chatgpt.com`, or `vision-v2:ollama@192.168.1.5:11434`: the host has its port
when that is not the scheme's default); an entry without the `vision-v2:` prefix, saved by an earlier
version under a narrower description, no longer counts, and the Models tab asks again.
**Use at your own risk** when selecting a hosted provider.
`--ollama-url` and `--ollama-model` have no effect for a provider other than `ollama`; ratings
from those providers have no logprob-based score. Activity detection for `profile`, `crashes` and the crash step of
`analyze` always uses Ollama, independently of this selection; an Ollama on another machine
needs this Vision consent for its URL there too, else the profile falls back without the model.

### compress_peaks

`off` (default), `light` or `strong`: the [peak compression](commands.md#peak-compression)
that `clips`, `assemble` and `edit` use when `--compress-peaks` is not given (`light` and
`strong` include a makeup gain that keeps the overall loudness, at most 6 and 10 dB). Other
values are rejected. `cutpoint ui` starts its "Compress Audio Peaks" slider on this value (Light when the key is unset) and passes the choice
to its exports and assemblies; it does not change the key.

### color

`off` (default), `auto` or `underwater`: the [automatic color correction](commands.md#clips)
that `clips` and `edit` use when `--color` is not given. It never applies to `gaming` footage
(only an explicit `--color` grades it) nor to a run with `--lut`. Other values are rejected.
`cutpoint ui` starts its "Color" slider on the choice you saved for the project, else on this
value (Off when the key is unset), and always at Off for gaming projects. It passes the choice to
its exports and presets; it does not change the key.

### analysis

`{"skip": [...]}`. Analysis steps that `analyze` and `edit` leave out by default: any of
`reactions`, `keyframes` and `crashes`. `--skip` on the command overrides it for one run.
Default: none. Other step names are rejected.

### video_encoder

`auto` (default), `nvenc`, `amf`, `vaapi` or `cpu` (libx264). `CUTPOINT_VIDEO_ENCODER` overrides it. `auto` really encodes a short test clip with each candidate (NVENC, then AMF on Windows or VAAPI and AMF on Linux; on Linux every `/dev/dri/renderD*` node is tried) and uses the first one that works, else libx264. A named GPU encoder that does not work is an error, not a silent fallback (the error suggests `cpu`). With `auto`, a GPU encode that fails on real footage is redone once on the CPU with a WARNING, and the rest of the run uses the CPU. `cutpoint capabilities` shows what works on this machine.

### export_workers

How many clips `clips` and `edit` export at once. `auto` (default) exports 2 at a time with a GPU encoder (`video_encoder`), which otherwise sits mostly idle on short clips, and 1 with libx264, which already uses every core. A number sets it; values are clamped to 1-3, as consumer NVIDIA cards allow only a few NVENC sessions at once. `CUTPOINT_EXPORT_WORKERS` overrides it. A run with a 360 clip exports one clip at a time.

The clips, their names and manifest.json are the same as with 1, and the per-clip log lines come in clip order; warnings from an export are logged when they happen. 

Failure handling: the first failure or Ctrl+C stops every export in progress and removes its partial file. With `video_encoder` `auto`, when a GPU encode fails and is redone on the CPU, a clip after it that was already encoding on the GPU at that moment keeps its GPU encode; clips that start later, during the redo too, are encoded on the CPU (with 1, every clip after the failed one is encoded on the CPU). It has the same frames and length; only the encoder differs. When NVENC has no free encode session (consumer NVIDIA drivers limit them, and ShadowPlay or OBS may hold one), the clip is not redone on the CPU: the rest of the run exports one clip at a time (one INFO line, "NVENC has no free encode session; exporting one clip at a time") and the clip is encoded on the GPU again once it runs alone; only if that fails too is it redone on the CPU as above.

### assistant

Settings for the `cutpoint ui` [AI Chat tab](commands.md#ai-chat), served by the UI's API
(`/api/projects/{id}/assistant`, token header only). It writes `ui/assistant/chat.json` and
`history.json` in the project. There is no API key setting (see below).

The UI's Models tab offers a Provider, so no base URL is typed: it saves `provider`
and `model`, with `url` empty (the preset's own). API providers and subscription logins are
separate entries with their own credentials and model lists:

| Preset | `provider` | Base URL | Model |
|---|---|---|---|
| None (local Ollama) | `ollama` | `ollama_url` (`OLLAMA_HOST`) | any pulled model; empty: the first pulled of `qwen3.5:9b`, the vision model and `qwen3-vl:8b`, else `qwen3.5:9b` |
| Anthropic API | `anthropic` | `https://api.anthropic.com` | empty: `claude-sonnet-5-5`; `claude-opus-5-5` offered; an API key |
| OpenAI API | `openai` | `https://api.openai.com/v1` | needed: an OpenAI model name; `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6.1-terra` and `gpt-6-luna` offered; an API key |
| Grok (xAI) | `xai` | `https://api.x.ai/v1` | needed: an xAI model name; `grok-4.7`, `grok-4.6`, `grok-4.5`, `grok-4.3` and `grok-build-0.1` offered |
| Claude Code subscription | `claude-code` | Anthropic via `cutpoint proxy` | models advertised by the proxy (`GET /v1/models`); local subscription session, no API key |
| Codex subscription | `codex` | OpenAI via `cutpoint proxy` | models advertised by the proxy (`/codex/models`); local subscription session, no API key |
| Antigravity subscription | `antigravity` | Google Cloud via `cutpoint proxy` | models advertised by the proxy (`fetchAvailableModels`); local subscription session, no API key |

A `url` set in `config.json` that is not the preset's (OpenRouter, LM Studio, a LAN Ollama) keeps
working and shows as "Custom (from config.json)": its base URL is read-only (edit the file to
change it), its model can be changed there, and choosing a preset and saving replaces it. The Model dropdown lists the models the provider offers (Refresh
asks again; see [AI Chat](commands.md#ai-chat)), and Other model… takes any other id.

- `provider`: `ollama` (default), `openai` (OpenAI API, or a server that speaks its API: LM Studio,
  vLLM, llama-server), `anthropic` (Anthropic API), `xai` (Grok, OpenAI-compatible), or a
  subscription login: `claude-code`, `codex` or `antigravity`. `openai` at
  `api.openai.com` uses the Responses API (`/v1/responses`) and asks OpenAI not to store the
  chat (`store: false`); any other `url` uses `/v1/chat/completions`.
- `model`: the model name. Empty (default): for Ollama the first that is pulled of
  `qwen3.5:9b`, the vision model (`CUTPOINT_VISION_MODEL`) and `qwen3-vl:8b`, else
  `qwen3.5:9b`; for Anthropic
  `claude-sonnet-5-5` (`claude-opus-5-5` is stronger, slower and costs more); for an
  OpenAI-compatible server its only model (`/v1/models`); OpenAI and xAI need a name.
- `url`: the base URL. Empty (default): `ollama_url`, `https://api.openai.com/v1`,
  `https://api.anthropic.com` or `https://api.x.ai/v1`. An OpenAI-compatible URL with no path gets `/v1`
  (`http://localhost:1234` is `http://localhost:1234/v1`). A URL with a backslash, a user name or
  password (`https://name@host`), or a host that reads differently to different URL parsers is
  refused, and redirects are not followed (by Ollama either) (set the
  address the server redirects to instead).
- `context_tokens`: the model's context size, a whole number of at least 2048. Default 8192 for a
  model on this machine (and any Ollama), 32768 for a hosted one; more lets the model see more
  moments and turns.
- `consent`: the hosted providers the user agreed to send project text to, as
  `assistant-v2:<provider>@<host>` (for example `assistant-v2:anthropic@api.anthropic.com`; the
  host has its port when that is not the scheme's default); the AI Chat tab adds them. An entry without the `assistant-v2:` prefix, saved by an earlier version
  under a shorter description, no longer counts, and AI Chat asks again.
  The assistant gets only analysis results (moment labels, scores and detection signals,
  transcript snippets, file names without paths, times) and your messages, never video or audio.
  This consent never covers the [Vision role](#vision_assistant), which has its own.
  Remove an entry to be asked again. A provider is local when its host is `localhost` or a
  loopback address; any other host, a LAN address too, is hosted.
- `frames_consent`: the projects whose video frames AI Chat may send to a hosted provider
  (`view_moment`, see [AI Chat](commands.md#ai-chat)): an object from a project folder to its
  `frames-v1:<provider>@<host>` entries, for example
  `{"D:\\Cutpoint\\projects\\dive": ["frames-v1:anthropic@api.anthropic.com"]}`; the allow box
  in AI Chat adds them. Each watched moment sends one image of 6 to 9 frames from its clip with
  their times, its transcript lines and detected sounds. Neither `consent` nor the Vision
  consent covers it. A provider on this machine needs none. Remove an entry to be asked again.

Other values are rejected.

Hosted API providers also display **Use at your own risk**. Ollama/Local has no risk warning;
the existing consent requirement still applies to a remote host.

API keys are never read from or written to `config.json`. A key belongs to a provider and the
host of its base URL. It is taken, in order, from:

| Source | Used for |
|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `XAI_API_KEY` | that provider's own host only: `api.anthropic.com` / `api.openai.com` / `api.x.ai`. An OpenAI key is never sent to OpenRouter, a LAN server or a server on this machine, and an xAI key never anywhere but `api.x.ai` (not to the `openai` provider with an `api.x.ai` URL either) |
| The system keyring (service `cutpoint`) | saved from the Models tab (Save key) for the provider and base URL saved there. The user is `anthropic`, `openai` or `xai` for the provider's own host, else `<provider>@<host>` (for example `openai@openrouter.ai`) |
| `CUTPOINT_ASSISTANT_API_KEY` | custom and self-hosted base URLs only (for example an OpenRouter key, or a key for your own server). Never sent to `api.anthropic.com`, `api.openai.com` or `api.x.ai`, not even when no other key is set: those need their own variable or a keyring key. The [Vision role](#vision_assistant) uses it only when its key goes to the same host as the assistant's (an API provider); for another host, save the Vision key in the Models tab |

A preset's key is for its own host, so it can be saved before the settings. For a custom base
URL, save the model settings first, then save the key for it. The key is sent
only to the host it belongs to, in a request header, and only over `https://` unless the server
is on this machine (a key with an `http://` base URL on another machine is refused). `cutpoint
ui` takes these variables out of its own environment when it starts, so jobs, ffmpeg and file
dialogs never see them; they are also redacted from log lines and error messages. An
OpenAI-compatible server on this machine needs no key. Ollama takes none.

Subscription providers (`claude-code`, `codex`, `antigravity`): **Use at your own risk**.
They use no API keys: they sign in through the local [`cutpoint proxy`](commands.md#proxy)
daemon (CLIProxyAPI on `127.0.0.1:18317`, HTTPS with a private per-start TLS identity), and
their credentials live in the native OS keyring. Model catalogs come from the proxy only after
consent for the selected role. How credentials, working files, recovery and process lifetime
are handled is described once, under [`proxy`](commands.md#proxy).

### pano

Settings for the optional 360 module. Every key is optional.

| Key | Meaning |
|---|---|
| `sdk_exe` | Path of the Insta360 Media SDK executable. `CUTPOINT_INSTA360_SDK` overrides it. Default: auto-detected under `<CUTPOINT_HOME>\sdk\insta360\`. See [insta360-sdk.md](insta360-sdk.md) |
| `stitched_dir` | Where `cutpoint 360 stitch` writes equirects. Default `<CUTPOINT_HOME>\360-stitched`. About 54 GB per hour of footage |
| `stitch_timeout_s` | Kill one clip's stitch after this many seconds. Default: no limit |
| `auto_stitch` | `true` (default) or `false`. When true, the SDK runs and an NVIDIA GPU is present, "Export clips" in `cutpoint ui` first stitches the recordings a 360 export needs (about 54 GB per hour of footage in `stitched_dir`). `false` exports from the dual-fisheye preview. The command-line `clips` never stitches. See [360 video](workflows/360.md#auto-stitch-on-export) |
| `stitch_args` | SDK arguments placed after the input and output. Replaces the defaults. See [insta360-sdk.md](insta360-sdk.md) |
| `studio_dirs` | Folders searched for Insta360 Studio equirect exports. Default `<CUTPOINT_HOME>\360-studio`; `_stitched\`, `export\` and `Export\` next to the camera files are always searched |
| `default_style` | `--style` of `cutpoint 360 analyze` when none is given. Default `action-cam` (formerly `ski`, no longer accepted) |
| `cmd_every` | Frames between virtual-camera updates when rendering. Default 1. 2 renders faster but leaves camera shake on every other frame. `--pano-cmd-every` overrides it |
| `level_strength` | Horizon lock of the stabilised view, 0 to 1. 1 keeps the horizon level; 0 lets the view lean with the camera's smoothed tilt. Default 0.7 |
| `tilt_follow_s` | Smoothing in seconds of the tilt the view follows when `level_strength` is below 1. Default 0.5 |
| `max_follow_deg` | Largest camera lean, in degrees from the clip's average attitude, that the view follows when `level_strength` is below 1. A larger lean (a fall, the camera being handled) is followed as this much. Default 30 |
| `imu_offset_s` | Time offset in seconds between the motion sensor and the video, if a clip's correction looks early or late. Default 0 |
| `equirect_yaw_offset` | Degrees added to the director's yaw for Studio exports whose yaw 0 is not the camera's forward direction. Default 0 |
| `director` | `DirectorParams` overrides for `cutpoint 360 director`, for example `{"max_speed": 35, "min_hold_s": 3.0}`. `--param KEY=VALUE` overrides them. Unknown keys and bad values are errors. `edge_rule` (`include_exclude`, the default, or `legacy`) chooses how people at the frame edge are fixed; its tuning keys are `w_shift`, `w_widen`, `include_margin`, `include_max_shift`, `include_min_subject_h`, `include_bridge_s`, `exclude_clear_deg`, `near_min_h_deg`, `near_min_score`, `near_box_scale`, `cut_min_gain_s` and `cut_min_gap_s`. See [360 director](workflows/360-director.md#director) |

## Styles

A style tells `analyze` how to score and how long clips and the final cut should be.

| Style | Clip length | Target total | Notes |
|---|---|---|---|
| `gaming` | 4-12 s | 90 s | Loudness and hype words, capped keywords |
| `action` | 3-12 s | 60 s | Loudness and scene cuts, plus laughter |
| `sport` | 5-20 s | 90 s | Crowd loudness, "goal", "what a ...", laughter |
| `funny` | 4-20 s | 60 s | Laughter-heavy |
| `commentary` | 20-120 s | 12.5 min | Speech continuity, sentence-snapped, chronological |
| `action-cam` | 2-10 s by moment type | 150 s | Action-cam telemetry, model tags and sound events; lifts and stops masked |

`action-cam` is the telemetry-driven action-cam style (formerly `ski`; `ski` is no longer accepted: a `--style ski`, a `config.json` value, a project or a learned preset that says `ski` stops with an error naming `action-cam`). Speed, airtime and crash scoring need GoPro HERO 9-13 telemetry (or the Insta360 X3 IMU); other footage is scored from audio, speech and model features only.

Learned styles are JSON files in `<CUTPOINT_HOME>\styles\<name>.json`, written by the
development script `scripts/refmatch.py`. They appear as extra `--style` choices, hold the feature weights
and the fit report, and can be edited by hand. Style definitions live in `src/cutpoint/styles.py`.
In an action-cam style, the weight of laughter heard over the wind noise is `voice_laughter`; older
learned styles call it `laughter`, which still works.

## Activity profiles

Crash detection, the action-cam ranking, `mix`, the visual layer's contact sheets and the 360
views, QA, vision check and director use an activity profile per project
(`<project>\activity.json`, written by the crash step): built-in
profiles in `src/cutpoint/presets/activities/`, plus the ones Cutpoint builds with the local LLM,
saved as `<CUTPOINT_HOME>\profiles\<name>.json`. A file there with a built-in's name replaces
that built-in for profile selection. The files are readable JSON, but nothing has to be edited:
see [action-cam workflow](workflows/action-cam.md#activity-profiles). When the crash layer runs
in an `events` run, the contact sheets are written after it and use the profile it chose; a run
without the crash layer (for example `--layers visual`) on a project with no `activity.json` uses
the default (ski) captions. Scorer weights from
profiles and profiles for games are planned.

## HUD presets

Game-HUD OCR regions come from a preset, looked up in this order:

1. `<project>\hud_presets\<game>.json`
2. `<CUTPOINT_HOME>\presets\<game>.json` (region discovery writes here)
3. the bundled presets folder `src/cutpoint/events/presets/` (no game presets are bundled)

The first preset whose `game` key (or file name, without `game`) matches wins: 1 overrides 2 and 3,
and 2 overrides 3. To make a preset for a game, write one
in folder 1 or 2, or let region discovery create it. The user folder (2) is trusted like the bundled one.
A `"builtin"` key, which older presets may have, is ignored.

The OCR layer also keeps its raw read of each source (the text it read per frame) in the source's
cache folder, `ocr-raw-*`. When an update changes only how that text is interpreted, the next run
replays the raw read in seconds instead of reading the footage again; a change of the preset's
regions or `feed.newest`, or of the OCR frame rate, reads it again, and an interrupted read resumes
near where it stopped. `--force ocr` drops the raw read too. It holds OCR text: it is git-ignored and
never written inside a git checkout.

Discovery leaves out menu and scoreboard frames (text across most of the screen centre, or
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

Discovery also proposes up to 2 extra `notice` regions (`extra_notice`, `extra_notice2`) where
text keeps showing outside every other region, including small text inside the big-text area
(see [game streams](workflows/game-streams.md)). Their entries carry `"origin": "outside"`; the
loader ignores this key, so it changes neither the region nor any cache key, and you can keep,
move or delete such a region like any other. For a game that already has a preset, they are
only written to `suggested_regions` in `<project>\discovery\<game>\proposal.json`: copy the
entries you want into the preset's `regions`. Regions in the preset that carry
`"origin": "outside"` count against the 2, so a preset with 2 of them gets no more
suggestions.

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
