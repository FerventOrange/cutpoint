<p align="center">
  <img src="https://raw.githubusercontent.com/FerventOrange/cutpoint/main/docs/assets/logo-256.png" alt="Cutpoint logo: a clapperboard with three characters peeking out" width="200">
</p>

# Cutpoint

Cutpoint finds the best moments in long recordings (gameplay, action-cam, sport, talk or 360
video), cuts them into clips and assembles a finished video. Drive it from your coding agent,
from the browser workspace or from the command line. All analysis runs on your own computer, so
no API credits are spent on video, and your original files are never changed. Hosted models are
optional and off by default.

Cutpoint aims to need nothing but your footage. It is meant to detect what is in the video
(the kind of footage, the activity and the moments) and score from that, so you do not have to
say what you recorded. Settings such as the style are overrides, not requirements. Today
`analyze` and `edit` detect the footage by default (`--style auto`); the vision rating is off by default, and the goal is to make it
automatic. What you ask for, such as funny moments or a length, steers the scoring.

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

`uv run cutpoint ui` (`cutpoint ui` with a tool install) opens a workspace to analyze, trim,
export and assemble standard video without an agent. Open the complete URL it prints, including
its session token. See the
[command reference](https://github.com/FerventOrange/cutpoint/blob/main/docs/commands.md#ui).

## Requirements

- Windows, Linux or macOS (macOS: CPU only; CI tests Windows and Linux). The installer sets up
  uv, Python 3.12 and ffmpeg for you
- ffmpeg 5.1 or newer; for a clone, [uv](https://docs.astral.sh/uv/)
- An NVIDIA GPU is recommended. Without one, everything still runs but slower.
- Optional: [Ollama](https://github.com/FerventOrange/cutpoint/blob/main/docs/getting-started.md#optional-ollama-for-frame-rating-and-ai-chat)
  for AI frame rating, scene descriptions and the local AI Chat

## Install

Run the installer. It sets up uv, ffmpeg and Cutpoint as a global `cutpoint` command, with the
torch build for your machine.

Linux and macOS:

```bash
curl -LsSf https://raw.githubusercontent.com/FerventOrange/cutpoint/main/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/FerventOrange/cutpoint/main/install.ps1 | iex
```

- For lite mode (no torch, no Ollama), end the Linux and macOS command with `| sh -s -- --lite`,
  or run `$env:CUTPOINT_LITE = 1` before `irm` on Windows.
- To upgrade, run the installer again. Do not use `uv tool upgrade`: it drops the torch build.
- Pinned releases, checksums, sources, prompts and installing from a clone are in
  [Getting started](https://github.com/FerventOrange/cutpoint/blob/main/docs/getting-started.md#install).

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
- [Roadmap](https://github.com/FerventOrange/cutpoint/blob/main/docs/roadmap.md): where Cutpoint has been and where it is going

## License

Cutpoint is released under the [MIT License](https://github.com/FerventOrange/cutpoint/blob/main/LICENSE).
