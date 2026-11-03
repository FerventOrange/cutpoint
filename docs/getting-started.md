# Getting started

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.12 | `pyproject.toml` pins `>=3.12,<3.13`. uv installs it for you. |
| [uv](https://docs.astral.sh/uv/) | Manages the virtual environment and dependencies. |
| ffmpeg and ffprobe 5.1 or newer | Decoding, loudness, encoding. Working NVIDIA/AMD encoders are probed; libx264 is the software fallback. |
| NVIDIA GPU (recommended) | Whisper transcription uses CUDA when the `gpu` extra is installed and falls back to the CPU otherwise. The optional `events` layers pick the GPU automatically and run on the CPU otherwise, much more slowly. |
| Ollama (optional) | Only for `analyze --vision`, `edit --vision`, `360 director --vl-check`, new activity profiles and the local AI Chat ([setup](#optional-ollama-for-frame-rating-and-ai-chat)). |
| Disk space | Caches and models live under the data folder. Models for `cutpoint setup` are large (see [Download models](#download-models-only-for-cutpoint-events)). Clips and finals are re-encoded, so budget for them in the project folder. |

Windows and Linux are tested (CI runs the tests on both). Lite
transcribes on the CPU. For GPU Whisper on an NVIDIA GPU, use the full install, which includes
the `gpu` extra (the CUDA libraries cuBLAS and cuDNN, about 3.6 GB).

## Install

Run the installer. It installs uv and ffmpeg when they are missing, then Cutpoint as a global
`cutpoint` command with the torch build for your machine.

Linux and macOS (macOS gets CPU torch only):

```bash
curl -LsSf https://raw.githubusercontent.com/FerventOrange/cutpoint/main/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/FerventOrange/cutpoint/main/install.ps1 | iex
```

- **Pinned to a release.** Releases after 0.4.0 (0.4.0 and earlier have no such files)
  have `install.sh`, `install.ps1` and `SHA256SUMS` as assets, with that
  release's version stamped in as the default (`--source pypi` installs exactly it). Replace
  `X.Y.Z` with a release tag such as `0.5.0` (tags have no `v`):
  `curl -LsSf https://github.com/FerventOrange/cutpoint/releases/download/X.Y.Z/install.sh | sh`
  and `irm https://github.com/FerventOrange/cutpoint/releases/download/X.Y.Z/install.ps1 | iex`.
  For the newest release use
  `https://github.com/FerventOrange/cutpoint/releases/latest/download/install.sh` (and `install.ps1`).
  The `main` commands above install the version in `main`'s script, which can lag or lead the
  newest release. To verify a download first, fetch the script and `SHA256SUMS` from the same
  release, then run `sha256sum -c --ignore-missing SHA256SUMS` (Linux) or
  `shasum -a 256 -c --ignore-missing SHA256SUMS` (macOS). On Windows compare
  `(Get-FileHash install.ps1 -Algorithm SHA256).Hash` with the line for `install.ps1` in `SHA256SUMS`.

`sh install.sh --help` and `.\install.ps1 -Help` list every option and variable. `irm | iex`
cannot pass parameters, so on Windows each option also has a `CUTPOINT_*` variable; the variables
work in `install.sh` too. A flag wins over its variable, and an empty value or
`0` counts as off. So do `no` and `false`: on Windows in any case, in `install.sh` exactly as
`no`, `NO`, `false` or `FALSE`. Any other value counts as on.

### Full or lite

| Edition | What you get | How to ask for it |
|---|---|---|
| Full (default) | Every feature. With an NVIDIA GPU: CUDA torch and the GPU libraries (several GB). Without: CPU torch | No option |
| Lite | No torch, no `events` or `gpu` extras, no Ollama. `analyze`, `clips`, `assemble`, `edit` and `cutpoint ui` still run on the CPU | `sh -s -- --lite`, `curl ... \| CUTPOINT_LITE=1 sh`, `.\install.ps1 -Lite`, or `$env:CUTPOINT_LITE = 1` before `irm \| iex` |

```bash
curl -LsSf https://raw.githubusercontent.com/FerventOrange/cutpoint/main/install.sh | sh -s -- --lite
```

```powershell
$env:CUTPOINT_LITE = 1; irm https://raw.githubusercontent.com/FerventOrange/cutpoint/main/install.ps1 | iex
```

To go from lite to full, run the installer again without `--lite` (`-Lite`) and with
`CUTPOINT_LITE` unset (`Remove-Item Env:CUTPOINT_LITE` in PowerShell, `unset CUTPOINT_LITE` in a
shell). If the variable is still set, the installer installs lite again. `--lite` cannot be
combined with `--cpu` or `--cuda`. `cutpoint capabilities` shows the edition.

### Other options

- `--source SRC` (`-Source`, `CUTPOINT_SOURCE`; for example `sh -s -- --source git`, or
  `$env:CUTPOINT_SOURCE = 'git'` before `irm | iex`): `pypi` (the default, the pinned release), `git`
  or `git@REF` (a branch, tag or commit of the GitHub repository), or a local path (a checkout,
  wheel or sdist). `CUTPOINT_VERSION=X.Y.Z` picks the release for `pypi`.
- `--cpu` / `--cuda` (`CUTPOINT_TORCH=cpu|cuda`): force the CPU or the CUDA (cu126) torch build.
- `--no-ffmpeg`: do not install ffmpeg. `CUTPOINT_NO_MODIFY_PATH=1`: do not run
  `uv tool update-shell`.
- `--dry-run` (`-DryRun`): print every command and change nothing.

### Prompts

In full mode the installer asks up to three optional questions. The default answer is No. They
read from the terminal, so they also work under `curl | sh`; with no terminal the answer is No.

1. Install Ollama, when it is not found (`winget` on Windows, `brew` on macOS, Ollama's own
   script on Linux, which needs sudo). A declined or failed Ollama install never fails the
   Cutpoint install.
2. Run `cutpoint setup` (model weights, about 2.5 GB, 3.5 GB with an NVIDIA GPU).
3. Pull the `qwen3.5:9b` model (about 6.6 GB), when Ollama is running.

Lite asks none of them. To skip the prompts: `--yes` (`-Yes`, `CUTPOINT_YES=1`) answers yes to
all; `--ollama` / `--no-ollama` (`-Ollama` / `-NoOllama`, `CUTPOINT_OLLAMA=1|0`) decide only the
Ollama question. If you answer No, run `cutpoint setup` later.

### Upgrade

Run the installer again. Do not use `uv tool upgrade` or a plain `uv tool install`: they drop the
torch build chosen for your machine.

The installer remembers none of your earlier options. Give the same ones again: without `--lite`
a lite install becomes full, and without `--source` a git or path install goes back to PyPI.

### Other sources

The installers install from PyPI, from `github.com/FerventOrange/cutpoint` (`--source git[@REF]`)
or from a local path. For a fork, or a wheel or sdist URL, install the tool with uv and choose the
torch build yourself. Use `cpu`, or `cu126` with an NVIDIA GPU:

```bash
uv tool install "cutpoint[events,gpu] @ URL" --torch-backend cu126 --python 3.12 --reinstall --no-sources
```

Without `--torch-backend`, uv picks a torch build that may not match the machine. The `--lite`
equivalent is `"cutpoint[ui,keyring,ski] @ URL"` (no `--torch-backend`). The runtime hints print
this form with the extras in use.

To install into your own virtual environment instead, use `uv pip` or `pip`. For CUDA torch add
the PyTorch index; an already-installed CPU torch also needs a reinstall:

```bash
uv pip install "cutpoint[events,gpu]" --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match --reinstall-package torch --reinstall-package torchvision
pip install "cutpoint[events,gpu]" --extra-index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

Without an NVIDIA GPU, leave out the index and reinstall flags and `gpu`. A plain PyPI install of
torch is CPU-only on Windows. Check the result with `cutpoint capabilities`.

### From a clone

Install ffmpeg and uv. On Windows:

```powershell
winget install --id Gyan.FFmpeg -e
winget install --id astral-sh.uv -e
```

On Linux:

```bash
sudo apt install ffmpeg        # or dnf, pacman, zypper, ...
curl -LsSf https://astral.sh/uv/install.sh | sh
. "$HOME/.local/bin/env"
```

Get the code and install it:

```powershell
git clone https://github.com/FerventOrange/cutpoint.git
cd cutpoint
uv sync
uv run cutpoint --help
```

In a clone, `uv sync` installs everything, including the `events` and `gpu` extras (the `full`
dependency group, about 7 GB or more with torch). For a lite environment without them, use
`uv sync --no-group full` (and `uv run --no-group full ...`, or set `UV_NO_GROUP=full`). The lite
install is enough for `analyze`, `clips`, `extend`, `assemble`, `stage`, `cutpoint ui` and the
subscription proxy. The lite installer does not offer Ollama; hosted API providers and
subscription providers work without the full edition or a local model download.

### Browser workspace

```bash
uv run cutpoint ui
```

Open the complete URL printed in the terminal, including `#t=…`. Create a project from one or
more local videos or folders (one full path per line; "Choose video…" and "Choose folder…" add a
line, or type or paste the paths; the
buttons are hidden where the computer has no file dialog, such as Docker) and choose a highlight style (`action-cam` is best with GoPro or Insta360 telemetry and is preselected when an input has some). Run analysis, preview candidates, adjust their
start/end times, and add them
to the sequence. Reorder and save the sequence, export clips, then assemble a final video.
The final preview includes download links for the final video or, with a timeline output format,
the FCPXML or OTIO file, plus the `.srt` subtitles.

The workspace is local-only and supports standard video. Source files are never modified.
Insta360 X3 folders and files (`.insv`, `.lrv`) make 360 projects, analyzed and reframed as by
the CLI ([360 video](workflows/360.md#in-the-ui)); a folder mixing 360 and regular videos is refused.
Analysis can download a Whisper model when speech transcription first needs it, just as the
CLI does; the UI does not run `cutpoint setup` (`cutpoint setup --only whisper` fetches the
model ahead of time). Selecting a browser-incompatible source prepares
a cached preview conversion, which may take time and extra disk space.

Use `--port 8766` if the default port is occupied. Stop the server with Ctrl+C; this also stops
its running jobs. In a lite environment, keep `--no-group full` and every extra you use in the command.

The startup URL grants access to local footage: keep it private. The token is stored only in the
current browser tab's session storage. After restarting the server, open its newly printed URL.

### Hosted models and subscriptions

**Use at your own risk** when using a hosted API or subscription provider. In the Models tab,
choose Chat and Vision independently; saving a model does not grant permission. Hosted model
catalogs and requests load only after consent for that role. Ollama/Local shows no risk warning.
Activity detection always uses Ollama, not the hosted Vision selection; an Ollama on another
machine needs the Vision consent for it there too.

For Claude Code, Codex or Antigravity subscriptions, run `uv run cutpoint proxy` in a dedicated
terminal, then `uv run cutpoint login claude-code` (or `codex` / `antigravity`), or sign in under
Connected accounts. It works in lite too, and needs a native OS keyring; the first start
downloads a checksum-verified CLIProxyAPI binary (about 24 MB). How credentials are stored and
cleaned up: [`proxy`](commands.md#proxy) and [`login`](commands.md#login).

### Optional extras

| Extra | Adds |
|---|---|
| `ui`, `ski`, `keyring` | Aliases that add nothing: the browser workspace (`cutpoint ui`), action-cam telemetry (GoPro GPS/IMU, Insta360 X3 IMU, the 360 module) and saving an AI Chat API key in the system keyring are part of the core install. The names stay so `cutpoint[ui]` and `--extra ski` keep working. |
| `events` | `cutpoint events` and crash detection: torch and torchvision (CUDA 12.6 builds on Windows and Linux), PANNs, RapidOCR, open_clip. The venv grows to about 7 GB, including about 3.6 GB of CUDA wheels even without a GPU. |
| `gpu` | CUDA libraries (cuBLAS, cuDNN, about 3.6 GB) so Whisper transcribes on an NVIDIA GPU. Without it Whisper uses the CPU. |
| `all` | `events` and `gpu`. |

The development tools (pytest, ruff, pre-commit) are the `dev` dependency group, not an extra; a
clone's `uv sync` installs it.

With `--no-group full`, `uv sync` makes the environment match exactly the extras you name, so it
removes the packages of any extra you leave out. List every extra you use each time:

```powershell
uv sync --no-group full --extra events --extra gpu
```

A plain `uv sync` keeps everything (`events` and `gpu` through the `full` group, plus the `dev`
tools). It includes the large torch download of `events`.

With an NVIDIA GPU, `uv sync --no-group full --extra gpu` is the minimum for GPU Whisper.

To add a missing extra later, use the form that matches how you installed Cutpoint. In a clone a
plain `uv sync` already has every extra; in a lite environment use `uv sync --no-group full
--extra X` plus your other extras. For a tool install, run the installer again (without
`--lite`, see [Full or lite](#full-or-lite)). Do not use `uv tool install --reinstall`: it drops
the torch build for your GPU.

The installer picks the torch build. To check a tool install, run `& "$(uv tool dir)\cutpoint\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available())"`
(`True` means the GPU is used). On Linux:
`"$(uv tool dir)/cutpoint/bin/python" -c "import torch; print(torch.cuda.is_available())"`. See [ffmpeg and GPU](troubleshooting/ffmpeg-gpu.md#events-is-very-slow).

The repository also has wrappers that use `.venv` without activating it: `.\cutpoint <command>`
(PowerShell) and `cutpoint <command>` from cmd. If PowerShell blocks scripts, use
`.\cutpoint.cmd`. On Linux, `./cutpoint <command>` is a POSIX sh wrapper that runs
`.venv/bin/python -m cutpoint`.

### Docker (Linux, CPU only)

The repository `Dockerfile` builds an Ubuntu 24.04 image with ffmpeg and the browser workspace. uv is pinned to an exact
release and image digest, and the container runs as the non-root user `cutpoint` (uid 1000).
No model weights are downloaded at build or start-up. No GPU is forwarded, so inference and video
encoding run on the CPU. Create your footage folder first, then:

```bash
docker build -t cutpoint .
docker run --rm --network host \
  --mount source=cutpoint-data,target=/data \
  --mount type=bind,source="$HOME/Videos",target=/footage,readonly \
  cutpoint
```

Open the **complete** `http://127.0.0.1:8765/#t=…` URL printed in the log; the token is
required. The UI binds to loopback only, so Docker's bridge port publishing (`-p`) cannot reach it.
On Linux, `--network host` is required ([host networking](https://docs.docker.com/engine/network/drivers/host/)).

Files the container writes into a bind mount belong to uid 1000 on the host. If your host user is
not uid 1000, match it so projects stay yours, either at run time (the `/data` volume must then
be writable by that uid, so prefer a bind mount such as `--mount type=bind,source="$HOME/cutpoint-data",target=/data`):

```bash
mkdir -p "$HOME/cutpoint-data"
docker run --rm --network host --user "$(id -u):$(id -g)" \
  --mount type=bind,source="$HOME/cutpoint-data",target=/data \
  --mount type=bind,source="$HOME/Videos",target=/footage,readonly \
  cutpoint
```

or at build time, which also keeps the named volume writable:
`docker build --build-arg CUTPOINT_UID="$(id -u)" --build-arg CUTPOINT_GID="$(id -g)" -t cutpoint .`

This applies to Docker Engine on Linux. With Docker Desktop on Windows or macOS, bind-mounted files
already belong to your user, so skip `--user` (from a WSL shell `$(id -u)` works as written); with
rootless Docker, files are mapped to your subordinate uid range.

A `Permission denied` on `/data` with a volume from the 0.2.0 image: see
[storage troubleshooting](troubleshooting/storage.md#docker-volume-from-the-020-image).

Inside the UI, choose sources under `/footage`; the bind mount is read-only, so your videos are never
modified. The `cutpoint-data` volume is `CUTPOINT_HOME=/data` and keeps projects, models and the
Hugging Face cache (`HF_HOME=/data/huggingface`) between runs. The only automatic download is the
Whisper model on first transcription (`cutpoint setup --only whisper` fetches it ahead of
time).

For `cutpoint events`, build with `--build-arg INSTALL_EVENTS=true` (tag it `cutpoint:events`) and
download the models explicitly into the same volume:

```bash
docker build --build-arg INSTALL_EVENTS=true -t cutpoint:events .
docker run --rm --mount source=cutpoint-data,target=/data cutpoint:events setup
```

If you run with `--user` and a bind mount, pass the same `--user` and
`--mount type=bind,source="$HOME/cutpoint-data",target=/data` flags to `setup`, so the models land
in the directory you actually mount. The models are a large download (see [Download models](#download-models-only-for-cutpoint-events)), and so are the Python dependencies (they include CUDA libraries
even though the container runs on the CPU). Then run the UI as above with `cutpoint:events`.

## First run

```powershell
uv run cutpoint paths
```

This prints every folder Cutpoint uses. Nothing is created until a command needs it.

### Where data is stored: CUTPOINT_HOME

Models, caches, learned styles and (by default) projects live in one folder, `CUTPOINT_HOME`:

- Windows: `%LOCALAPPDATA%\cutpoint`
- Linux and macOS: `$XDG_DATA_HOME/cutpoint`, or `~/.local/share/cutpoint`

Set the `CUTPOINT_HOME` environment variable to move it, for example to a larger drive:

```powershell
setx CUTPOINT_HOME D:\cutpoint-data       # new shells only
```

```bash
export CUTPOINT_HOME=/data/cutpoint
```

Set the variable or edit `config.json` yourself. See
[Configuration](configuration.md) for the full layout.

Projects go to `<CUTPOINT_HOME>\projects` unless you set `projects_dir` in `config.json` or
`CUTPOINT_PROJECTS_DIR`. A bare name such as `--out my-edit` is placed there.

### Download models (only for `cutpoint events`)

`analyze`, `clips`, `extend` and `assemble` need no model download beyond the Whisper model,
which is fetched into the Hugging Face cache on first use. To fetch it ahead of time (in every
install, also Cutpoint lite): `uv run cutpoint setup --only whisper` (`--whisper-model NAME`
for another model). Once it is cached, runs load it without a Hugging Face Hub request.

For `cutpoint events`, install the `events` extra and run:

```powershell
uv run cutpoint setup
```

This downloads about 2.5 GB (3.5 GB with an NVIDIA GPU): the Whisper model into the Hugging
Face cache, and into `CUTPOINT_HOME` the PANNs sound-event checkpoint, CLIP ViT-L/14, a Faster
R-CNN person detector, an aesthetic head and Silero VAD. Downloads resume if
interrupted. Choose parts with `--only`:

```powershell
uv run cutpoint setup --only panns voice
```

Valid components: `pip`, `whisper`, `panns`, `visual`, `voice`. `setup` installs no Python
packages; the `pip` step only checks them and prints the install command. The VLM layer of
`cutpoint events` downloads nothing here: it uses the shared vision model (see
[Ollama](#optional-ollama-for-frame-rating-and-ai-chat)).

### Optional: Ollama for frame rating and AI Chat

Ollama runs local models. Cutpoint works without it. Only these features use it:

- `analyze --vision` and `edit --vision`: a vision model rates the top candidates. It is off
  unless you ask for it per run or turn it on by default with "Rate frames with vision
  model" in the UI's Models tab, which saves [`"vision": true`](configuration.md#vision)
  in `config.json` (`--no-vision` then skips it for one run). In `cutpoint ui`, that setting is
  the only switch: every analysis there follows it. The model is unloaded from GPU memory as soon
  as the rating is done.
- The VLM layer of `cutpoint events` (and `edit --events`): the same vision model describes
  candidate windows from frames, the transcript and the other layers' hints. Without Ollama, it
  can use a hosted (API or subscription) Vision model chosen in the Models tab once you allowed vision requests
  there ([`vision_assistant`](configuration.md#vision_assistant)); with neither, the layer is
  skipped with a WARNING and earlier descriptions are kept.
- `360 director --vl-check` and building a new activity profile (see
  [action-cam](workflows/action-cam.md#activity-profiles)).
- The [AI Chat](commands.md#ai-chat) tab in `cutpoint ui`, when its provider is `ollama` (the
  default). The hosted providers (OpenAI-compatible, Anthropic) do not need Ollama.

**Install Ollama.**

The one-step installers (`install.sh` on Linux and macOS, `install.ps1` on Windows) ask whether to
install Ollama in full mode (default No): `--ollama` / `-Ollama` installs without asking,
`--no-ollama` / `-NoOllama` never asks, `CUTPOINT_OLLAMA=1|0` does the same, and `--yes` / `-Yes`
answers yes. They install it with the official script (with sudo) on Linux, `brew` on macOS and
`winget install --id Ollama.Ollama -e` on Windows (with the download link when winget is missing
or fails), make sure the server is running and then offer the `qwen3.5:9b` pull. Lite mode
(`--lite` / `-Lite`) has no Ollama: it skips this step, and `--lite --ollama` / `-Lite -Ollama`
prints a warning and installs nothing. To do it by hand:

| System | Install |
|---|---|
| Windows | The installer from <https://ollama.com/download>, or `winget install Ollama.Ollama` |
| macOS | The app from <https://ollama.com/download>, or `brew install ollama` |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` (read the script first, or follow the [manual install](https://docs.ollama.com/linux)) |
| Docker | `docker run -d --gpus=all -v ollama:/root/.ollama -p 127.0.0.1:11434:11434 --name ollama ollama/ollama` (drop `--gpus=all` without the NVIDIA Container Toolkit) |

**Check that it runs.** `ollama --version` prints the version and `ollama list` lists the pulled
models. Opening `http://127.0.0.1:11434` (the default port) answers "Ollama is running". On Linux
the installer sets up a service; on Windows and macOS the app runs it in the background; `ollama
serve` starts it by hand.

**Pull the model.**

```powershell
ollama pull qwen3.5:9b         # vision model for --vision and the AI Chat model, about 6.6 GB
```

`qwen3.5:9b` is the default vision model and the AI Chat's model, so one download is enough.
`qwen3-vl:8b` (about 6 GB), the default vision model before, still works: pull it and pass
`analyze --ollama-model qwen3-vl:8b` or set `CUTPOINT_VISION_MODEL`. Models of this size want
about 8 GB of GPU memory to run fast; on the CPU they work but reply slowly. For another model,
set `CUTPOINT_VISION_MODEL` or pass `analyze --ollama-model`, and choose the chat model in the
Models tab under Chat role ([`assistant.model`](configuration.md#assistant)).

**Point Cutpoint at it.** The server is found through `OLLAMA_HOST`, else
[`ollama_url`](configuration.md#ollama_url) in `config.json`, else `http://127.0.0.1:11434`;
`analyze --ollama-url` overrides it for one run and `assistant.url` sets
it for the chat. For an Ollama on another machine use that machine's address. The AI Chat treats
it as hosted and asks for [consent](commands.md#ai-chat) before the first message, Vision sends
it frames only after "Allow vision requests" in the Models tab, and the Models tab
refuses a URL with a user name or password.

**Turn it on.** Run `analyze --vision` or `edit --vision`, or set `"vision": true`. For the chat,
open the AI Chat tab in `cutpoint ui` on an analyzed project. If it does not connect, see
[Ollama is not reachable](troubleshooting/models.md#ollama-is-not-reachable).

### ffmpeg lookup

Cutpoint finds ffmpeg through `CUTPOINT_FFMPEG` / `CUTPOINT_FFPROBE` (full paths), then
`PATH`, then (Windows) the WinGet package folder. A fresh winget install therefore works before your
shell's `PATH` refreshes.

## Your first edit

Use any video. Here one file becomes a highlights cut (with the installer, drop `uv run`):

```powershell
uv run cutpoint analyze D:\Videos\clip.mp4 --style gaming --out my-edit
uv run cutpoint clips my-edit --top 8
uv run cutpoint assemble projects/my-edit/clips --transition fade --loudnorm
```

On Linux: `uv run cutpoint analyze ~/Videos/clip.mp4 --style gaming --out my-edit`.

1. `analyze` writes `highlights.json` (ranked candidates) and `transcript.txt` to the project.
2. `clips` exports `clips\001_<label>.mp4`, `002_...` and a `manifest.json`.
3. Review the clips folder. Delete clips you do not want. Rename to reorder
   (`005_x.mp4` to `001b_x.mp4`, and its `005_x.srt` to `001b_x.srt`).
4. `assemble` joins what is left, in filename order, into `final.mp4` with `final.srt`
   subtitles (`--format fcpxml` or `otio` writes that timeline for an editor instead).

Styles are `gaming`, `action`, `sport`, `funny`, `commentary` and `action-cam` (formerly `ski`); see
[Configuration](configuration.md#styles). More recipes: [Workflows](workflows/README.md).

Next: [Using Cutpoint from an agent](agents.md), or [Troubleshooting](troubleshooting/README.md)
if something fails.
