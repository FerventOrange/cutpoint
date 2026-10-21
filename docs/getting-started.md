# Getting started

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.12 | `pyproject.toml` pins `>=3.12,<3.13`. uv installs it for you. |
| [uv](https://docs.astral.sh/uv/) | Manages the virtual environment and dependencies. |
| ffmpeg and ffprobe | Decoding, loudness, encoding. NVENC is used when available, libx264 otherwise. |
| NVIDIA GPU (recommended) | Whisper transcription uses CUDA when it can and falls back to the CPU. The optional `events` layers pick the GPU automatically and run on the CPU otherwise, much more slowly. |
| Ollama (optional) | Only for `analyze --vision` and `events` VLM features that use it. |
| Disk space | Caches and models live under the data folder. Models for `cutpoint setup` are about 8 GB. Clips and finals are re-encoded, so budget for them in the project folder. |

Windows is the main tested platform (CI runs the tests on Windows and Linux). The CUDA
libraries for GPU Whisper (cuBLAS, cuDNN) come with the base install on Windows and Linux.

## Install

Install ffmpeg and uv. On Windows:

```powershell
winget install --id Gyan.FFmpeg -e
winget install --id astral-sh.uv -e
```

On Linux, install ffmpeg with your package manager and uv from the uv documentation.

Get the code and install the core package:

```powershell
git clone https://github.com/FerventOrange/cutpoint.git
cd cutpoint
uv sync
uv run cutpoint --help
```

The core install is enough for `analyze`, `clips`, `extend`, `assemble` and `stage`.

### Optional extras

| Extra | Adds |
|---|---|
| `ski` | Action-cam telemetry (GoPro GPS/IMU, Insta360 X3 IMU). Also used by the 360 module. |
| `events` | `cutpoint events`, `fit`, `refmatch --taste` and crash detection: torch and torchvision (CUDA 12.6 builds on Windows and Linux), PANNs, RapidOCR, open_clip, scikit-learn. |
| `dev` | pytest, ruff, pre-commit. |
| `all` | `ski` and `events`. |
| `gpu` | Empty alias kept for old commands. |

`uv sync` makes the environment match exactly the extras you name, so it removes the
packages of any extra you leave out. List every extra you use each time:

```powershell
uv sync --extra ski --extra events
```

### Install as a tool (optional)

To get a global `cutpoint` command instead of `uv run cutpoint`:

```powershell
uv tool install .
```

Add extras with `uv tool install ".[ski,events]"`. This path is not exercised by the project's
tests; if it fails, use `uv sync` and `uv run`.

The repository also has wrappers that use `.venv` without activating it: `.\cutpoint <command>`
(PowerShell) and `cutpoint <command>` from cmd. If PowerShell blocks scripts, use
`.\cutpoint.cmd`.

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

`cutpoint init`, which will ask for these locations on first run, is planned and does not
exist yet. Until then, set the variable or edit `config.json` yourself. See
[Configuration](configuration.md) for the full layout.

Projects go to `<CUTPOINT_HOME>\projects` unless you set `projects_dir` in `config.json` or
`CUTPOINT_PROJECTS_DIR`. A bare name such as `--out my-edit` is placed there.

### Download models (only for `cutpoint events`)

`analyze`, `clips`, `extend` and `assemble` need no model download beyond the Whisper model,
which is fetched into the Hugging Face cache on first use.

For `cutpoint events`, install the `events` extra and run:

```powershell
uv run cutpoint setup
```

This downloads about 8 GB into `CUTPOINT_HOME`: the PANNs sound-event checkpoint, a
llama.cpp CUDA build, Qwen2.5-Omni-7B (Q4_K_M plus its mmproj), CLIP ViT-L/14, a Faster R-CNN
person detector, an aesthetic head and Silero VAD. Downloads resume if interrupted. Choose
parts with `--only`:

```powershell
uv run cutpoint setup --only panns voice
```

Valid components: `pip`, `panns`, `llama`, `omni`, `visual`, `voice`. `setup` installs no Python
packages; the `pip` step only checks them and prints the install command. The llama.cpp build it
downloads is a Windows CUDA build.

### Optional: Ollama vision

`analyze` can rate the top candidates with a vision model through Ollama. It is used
automatically when the server and model are reachable, and skipped quietly otherwise.

```powershell
winget install --id Ollama.Ollama -e
ollama pull qwen3-vl:8b        # about 6 GB
```

Override the model with `CUTPOINT_VISION_MODEL` or `analyze --ollama-model`, and the server
with `OLLAMA_HOST`, `ollama_url` or `analyze --ollama-url`.

### ffmpeg lookup

Cutpoint finds ffmpeg through `CUTPOINT_FFMPEG` / `CUTPOINT_FFPROBE` (full paths), then
`PATH`, then the WinGet package folder. A fresh winget install therefore works before your
shell's `PATH` refreshes.

## Your first edit

Use any video. Here one file becomes a highlights cut:

```powershell
uv run cutpoint analyze D:\Videos\clip.mp4 --style gaming --out my-edit
uv run cutpoint clips my-edit --top 8
uv run cutpoint assemble projects\my-edit\clips --transition fade --loudnorm
```

On Linux: `uv run cutpoint analyze ~/Videos/ride.mp4 --style action --out my-edit`.

1. `analyze` writes `highlights.json` (ranked candidates) and `transcript.txt` to the project.
2. `clips` exports `clips\001_<label>.mp4`, `002_...` and a `manifest.json`.
3. Review the clips folder. Delete clips you do not want. Rename to reorder
   (`005_x.mp4` to `001b_x.mp4`).
4. `assemble` joins what is left, in filename order, into `final.mp4` and writes
   `final.fcpxml` and `final.otio`.

Styles are `gaming`, `action`, `sport`, `funny`, `commentary` and `ski`; see
[Configuration](configuration.md#styles). More recipes: [Workflows](workflows/README.md).

Next: [Using Cutpoint from an agent](agents.md), or [Troubleshooting](troubleshooting/README.md)
if something fails.
