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
No, but transcription and the optional model layers are faster on an NVIDIA GPU (add the `gpu`
extra for GPU Whisper; AMD transcription remains on faster-whisper CPU). For video encoding,
Cutpoint probes usable hardware encoders (NVENC then AMF on Windows; NVENC then VAAPI then AMF
on Linux) and falls back to `libx264`. You can override this with the `video_encoder` config or
`CUTPOINT_VIDEO_ENCODER` (`auto`, `nvenc`, `amf`, `vaapi`, `cpu`). See
[Configuration](configuration.md#video_encoder) and [ffmpeg and GPU](troubleshooting/ffmpeg-gpu.md).

**Which platforms are supported?**
Windows and Linux; the test suite runs on both in CI. `cutpoint setup` automatically prefers CUDA
when a usable NVIDIA driver is detected, otherwise Vulkan, otherwise CPU when distribution
assets are available. You can override the backend with `CUTPOINT_LLAMA_BACKEND` (`auto`, `cuda`,
`vulkan`, `cpu`). A self-built `llama-server` on `PATH` or in `CUTPOINT_LLAMA_SERVER` also works.
See [Models and Ollama troubleshooting](troubleshooting/models.md#cutpoint-setup-fails-or-is-incomplete).
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
Yes: `cutpoint edit <video> --kind highlights|montage|clips|funny-moments` runs `analyze`,
`clips` and `assemble` in one go (see [commands](commands.md#edit)). `clips` and `assemble`
write SRT subtitles next to the clips and the final video. `cutpoint skill install` adds a
Claude Code skill so an agent finds Cutpoint on its own. `--json` writes the result of most
commands as one JSON object on stdout (see [JSON output](commands.md#json-output)).

**Do I need the Insta360 SDK?**
Only for high-quality stitching of Insta360 X3 `.insv` files. Everything else works without it,
and 360 clips render in preview quality without it. See [insta360-sdk.md](insta360-sdk.md).

**Is there automatic caching?**
Yes, per source file (path, size, modification time) in `<CUTPOINT_HOME>\cache`. Adding a file to
an analysed set only processes the new file.
