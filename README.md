<p align="center">
  <img src="https://raw.githubusercontent.com/FerventOrange/cutpoint/main/docs/assets/logo-256.png" alt="Cutpoint logo: a clapperboard with three characters peeking out" width="200">
</p>

# Cutpoint

Cutpoint is an AI video editor that your coding agent drives. Tell the agent to edit a video and
it finds the best moments, cuts them into clips and assembles the result. All analysis runs on
your own computer, so your agent does not spend API credits on video, and your original files
are never changed.

[Are you an AI agent? Start here](https://github.com/FerventOrange/cutpoint/blob/main/AGENTS.md)

(Installed from PyPI? Run `cutpoint guide`.)

## What you get

- Trimmed, numbered clips of the best moments
- A transcript of the video with timestamps
- A finished video, `final.mp4`, plus timelines you can open in DaVinci Resolve
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

One-command editing (`cutpoint edit`), a first-run setup wizard and an agent skill are
planned. Today the agent runs the steps itself using [AGENTS.md](https://github.com/FerventOrange/cutpoint/blob/main/AGENTS.md).

## Requirements

- Windows or Linux
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- ffmpeg
- An NVIDIA GPU is recommended. Without one, everything still runs but slower.
- Optional: Ollama for AI frame rating, extra downloads (about 8 GB) for sound, on-screen text
  and scene-description features

## Install

```powershell
winget install --id Gyan.FFmpeg -e
winget install --id astral-sh.uv -e
git clone https://github.com/FerventOrange/cutpoint.git
cd cutpoint
uv sync
uv run cutpoint paths
```

To get a global `cutpoint` command instead, run `uv tool install cutpoint` (from PyPI) after
installing ffmpeg and uv.

On Linux:

```bash
sudo apt install ffmpeg        # or dnf, pacman, zypper, ...
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/FerventOrange/cutpoint.git
cd cutpoint
uv sync
uv run cutpoint paths
```

The optional extras and model downloads are explained in
[Getting started](https://github.com/FerventOrange/cutpoint/blob/main/docs/getting-started.md).

## Documentation

- [Getting started](https://github.com/FerventOrange/cutpoint/blob/main/docs/getting-started.md): install, first run, where data is stored, models
- [Using Cutpoint from an agent](https://github.com/FerventOrange/cutpoint/blob/main/docs/agents.md): set up Claude Code, Codex, Grok or another agent
- [Workflows](https://github.com/FerventOrange/cutpoint/blob/main/docs/workflows/README.md): montages, game streams, action-cam, 360 video, vertical output
- [Command reference](https://github.com/FerventOrange/cutpoint/blob/main/docs/commands.md)
- [Configuration](https://github.com/FerventOrange/cutpoint/blob/main/docs/configuration.md)
- [Troubleshooting](https://github.com/FerventOrange/cutpoint/blob/main/docs/troubleshooting/README.md)
- [FAQ](https://github.com/FerventOrange/cutpoint/blob/main/docs/faq.md)
- [Architecture](https://github.com/FerventOrange/cutpoint/blob/main/docs/architecture.md), [Contributing](https://github.com/FerventOrange/cutpoint/blob/main/CONTRIBUTING.md), [Security](https://github.com/FerventOrange/cutpoint/blob/main/SECURITY.md)
- [Insta360 SDK](https://github.com/FerventOrange/cutpoint/blob/main/docs/insta360-sdk.md): optional, for high-quality 360 stitching

## License

Cutpoint is released under the [MIT License](https://github.com/FerventOrange/cutpoint/blob/main/LICENSE).
