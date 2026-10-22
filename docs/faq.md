# FAQ

**Does it change my original videos?**
No. Sources are only read. Everything is written to a project folder.

**Does it need an internet connection or API keys?**
No for editing. Whisper and the optional models download once. The analysis models run locally,
so the agent driving Cutpoint does not spend API credits on video.

**Which agents work with it?**
Any agent that can run shell commands (Claude Code, Codex, Grok CLI, others). See
[AGENTS.md](../AGENTS.md) and [Using Cutpoint from an agent](agents.md).

**Do I need a GPU?**
No, but transcription and the optional model layers are much faster on an NVIDIA GPU. Clip
export uses NVENC when available and libx264 otherwise.

**Which platforms are supported?**
Windows and Linux; the test suite runs on both in CI. `cutpoint setup` downloads a llama.cpp
build for each (on Linux: the oldest CUDA build when an NVIDIA driver is detected, otherwise Vulkan,
otherwise CPU). A self-built `llama-server` on `PATH` or in `CUTPOINT_LLAMA_SERVER` also works.

**What kinds of video work?**
Gameplay, action-cam, sport, funny moments and commentary or talk. Pick the matching `--style`
([Configuration](configuration.md#styles)). Scoring is heuristic and works best when there is
voice or mic audio.

**Can I choose the moments myself?**
Yes. Write a segments file and run `cutpoint clips <project> --segments FILE`
([command reference](commands.md#clips)).

**Can I edit the result in DaVinci Resolve?**
Yes. `assemble` writes `final.fcpxml` and `final.otio` next to `final.mp4`
([Workflows](workflows/README.md#importing-into-davinci-resolve-free)).

**Can it make vertical videos?**
Yes: `cutpoint clips <project> --vertical`, a fixed centre crop to 1080x1920.

**Where is my data?**
In `CUTPOINT_HOME` (run `cutpoint paths`). See [Configuration](configuration.md).

**Is there a single command that does the whole edit?**
Not yet. `cutpoint edit`, `cutpoint init`, JSON output,
SRT transcripts and an agent skill are planned. Today an agent runs `analyze`, `clips` and `assemble` in sequence.

**Do I need the Insta360 SDK?**
Only for high-quality stitching of Insta360 X3 `.insv` files. Everything else works without it,
and 360 clips render in preview quality without it. See [insta360-sdk.md](insta360-sdk.md).

**Is there automatic caching?**
Yes, per source file (path, size, modification time) in `<CUTPOINT_HOME>\cache`. Adding a file to
an analysed set only processes the new file.
