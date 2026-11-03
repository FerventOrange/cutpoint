<p align="center">
  <img src="https://raw.githubusercontent.com/FerventOrange/cutpoint/main/docs/assets/logo-256.png" alt="Cutpoint logo: a clapperboard with three characters peeking out" width="200">
</p>

# Cutpoint

Cutpoint is an AI video editor that your coding agent drives. Tell the agent to edit a video and
it finds the best moments, cuts them into clips and assembles the result. All analysis runs on
your own computer, so your agent does not spend API credits on video, and your original files
are never changed. Hosted models are optional and off by default.

[Are you an AI agent? Start here](https://github.com/FerventOrange/cutpoint/blob/main/AGENTS.md)

(Installed from PyPI? Run `cutpoint guide`.)

## What you get

- Trimmed, numbered clips of the best moments
- A transcript of the video with timestamps
- A finished video, `final.mp4`, or a timeline you can open in DaVinci Resolve
- Vertical (9:16) versions for shorts and reels
- Works on gameplay, action-cam, sport, funny moments, commentary and 360 video

## Use it

1. Install Cutpoint (below).
2. Open your agent (Claude Code, Codex, Grok CLI or any CLI agent) in the Cutpoint folder, or
   tell it where Cutpoint is.
3. Ask it to edit your video:

   > Edit D:\Videos\clip.mp4 into a 60 second highlights video with Cutpoint.

   > Find the funniest moments in ~/Videos/stream.mp4 and show me the clips before you
   > assemble them.

   > Make a vertical montage from ~/Videos/ride.mp4.

The agent runs Cutpoint, shows you the clips and writes the result into a project folder.
Delete or rename clips to change the cut, then ask the agent to assemble again.

Without an agent, `cutpoint edit D:\Videos\clip.mp4 --kind montage` does it in one command
(see [commands](https://github.com/FerventOrange/cutpoint/blob/main/docs/commands.md#edit)). The agent works from
[AGENTS.md](https://github.com/FerventOrange/cutpoint/blob/main/AGENTS.md); to let an agent find
Cutpoint on its own, see [Using Cutpoint from an agent](https://github.com/FerventOrange/cutpoint/blob/main/docs/agents.md).

### Local browser workspace

For standard video, you can review and edit highlights without an agent:

```bash
uv run cutpoint ui
```

Open the complete URL printed in the terminal (including its session token). Pick a source
file or folder, analyze it, trim and reorder moments, export clips, then assemble and preview
the final video. Footage stays on your
computer; each export gets a new folder so previous exports are retained. The UI also makes
360 projects from Insta360 X3 footage ([360 video](https://github.com/FerventOrange/cutpoint/blob/main/docs/workflows/360.md#in-the-ui)) but does not
expose the action-cam workflows.

With a tool install, run `cutpoint ui`.
See the [command reference](https://github.com/FerventOrange/cutpoint/blob/main/docs/commands.md#ui) for details.

## Requirements

- Windows, Linux or macOS (macOS: CPU only; CI tests Windows and Linux). The installer sets up
  uv, Python 3.12 and ffmpeg for you
- For a clone: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12)
- ffmpeg 5.1 or newer
- An NVIDIA GPU is recommended (the full install adds the CUDA libraries for GPU Whisper). Without one, everything
  still runs but slower.
- Optional: [Ollama](https://github.com/FerventOrange/cutpoint/blob/main/docs/getting-started.md#optional-ollama-for-frame-rating-and-ai-chat)
  for AI frame rating, the scene descriptions of `cutpoint events` and the local AI Chat
- Optional: `cutpoint setup` downloads (about 2.5 GB, 3.5 GB with an NVIDIA GPU) for sound,
  on-screen text and visual features, and the Whisper model (else fetched on first use)

## Install

Run the installer. It sets up uv, ffmpeg and Cutpoint as a global `cutpoint` command, with the
torch build for your machine.

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
- **Full or lite.** The default is full: every feature, with CUDA torch on an NVIDIA GPU (several
  GB). Lite has no torch and no Ollama; analyze, clips, assemble and edit still work. Ask for it
  with `curl ... | CUTPOINT_LITE=1 sh` (or `sh -s -- --lite`) on Linux and macOS, or
  `$env:CUTPOINT_LITE = 1` before the `irm` command on Windows. To go back to full, run the installer again
  without `--lite` and with `CUTPOINT_LITE` unset, or it installs lite again.
- **Source.** `--source` is `pypi` (the
  default), `git` or `git@REF` (a branch, tag or commit), or a local path (checkout, wheel or
  sdist). Example: `curl ... | sh -s -- --source git`. On Windows use `$env:CUTPOINT_SOURCE = 'git'`
  before the `irm` command.
- **Prompts.** In full mode the installer can ask to install Ollama, run `cutpoint setup`
  (about 2.5 GB, 3.5 GB with an NVIDIA GPU) and pull the `qwen3.5:9b` model (about 6.6 GB). The default
  answer is No. `--yes` (`-Yes`, `CUTPOINT_YES=1`) answers yes to all; without a terminal the
  answer is No. `--dry-run` prints what would run and changes nothing.
- **Upgrade.** Run the installer again, with the same options as before. It remembers none: without
  `--lite` a lite install becomes full, and without `--source` you go back to PyPI. Do not use
  `uv tool upgrade`: it drops the torch build.

`sh install.sh --help` and `.\install.ps1 -Help` list every option and variable. The details are
in [Getting started](https://github.com/FerventOrange/cutpoint/blob/main/docs/getting-started.md#install).

To work from a clone instead:

```powershell
winget install --id Gyan.FFmpeg -e
winget install --id astral-sh.uv -e
git clone https://github.com/FerventOrange/cutpoint.git
cd cutpoint
uv sync
uv run cutpoint paths
```

On Linux:

```bash
sudo apt install ffmpeg        # or dnf, pacman, zypper, ...
curl -LsSf https://astral.sh/uv/install.sh | sh
. "$HOME/.local/bin/env"
git clone https://github.com/FerventOrange/cutpoint.git
cd cutpoint
uv sync
uv run cutpoint paths
```

In a clone, `uv sync` installs everything, including torch and CUDA (about 7 GB or more). For a
lite environment use `uv sync --no-group full` and `uv run --no-group full ...` (or set
`UV_NO_GROUP=full`).

A clone, and the full installer on a machine with an NVIDIA GPU, include the `gpu` extra, so
Whisper uses the GPU. The optional extras and model downloads are explained in
[Getting started](https://github.com/FerventOrange/cutpoint/blob/main/docs/getting-started.md).

AMD GPUs encode video (VAAPI on Linux, AMF where FFmpeg and the driver support it) but
transcribe on the CPU. `uv run cutpoint capabilities` shows what works; see the
[FAQ](https://github.com/FerventOrange/cutpoint/blob/main/docs/faq.md) to override the encoder.

## Documentation

- [Getting started](https://github.com/FerventOrange/cutpoint/blob/main/docs/getting-started.md): install, first run, where data is stored, models
- [Using Cutpoint from an agent](https://github.com/FerventOrange/cutpoint/blob/main/docs/agents.md): set up Claude Code, Codex, Grok or another agent
- [Workflows](https://github.com/FerventOrange/cutpoint/blob/main/docs/workflows/README.md): montages, game streams, action-cam, 360 video, vertical output
- [Command reference](https://github.com/FerventOrange/cutpoint/blob/main/docs/commands.md)
- [Configuration](https://github.com/FerventOrange/cutpoint/blob/main/docs/configuration.md)
- [Troubleshooting](https://github.com/FerventOrange/cutpoint/blob/main/docs/troubleshooting/README.md)
- [FAQ](https://github.com/FerventOrange/cutpoint/blob/main/docs/faq.md)
- [Architecture](https://github.com/FerventOrange/cutpoint/blob/main/docs/architecture.md), [Contributing](https://github.com/FerventOrange/cutpoint/blob/main/CONTRIBUTING.md), [Security](https://github.com/FerventOrange/cutpoint/blob/main/SECURITY.md)
- [Pipeline](https://github.com/FerventOrange/cutpoint/blob/main/docs/pipeline.md): diagrams of the analyze, clips and assemble flow
- [Insta360 SDK](https://github.com/FerventOrange/cutpoint/blob/main/docs/insta360-sdk.md): optional, for high-quality 360 stitching

## License

Cutpoint is released under the [MIT License](https://github.com/FerventOrange/cutpoint/blob/main/LICENSE).
