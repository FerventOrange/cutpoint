# Setting up an agent to use Cutpoint

Cutpoint is a command-line tool, so any agent that can run shell commands can drive it:
Claude Code, Codex, Grok CLI, or a script. Analysis runs locally (ffmpeg, Whisper, optional
Ollama and llama.cpp models), so the agent only issues commands and reads small JSON and text
files. No video is sent to an API, and local Qwen models are the default for the model features.

The instructions the agent itself should follow are in [AGENTS.md](../AGENTS.md) at the
repository root; `cutpoint guide` prints the same text, so an agent using a PyPI install
(`uv tool install cutpoint`) can read it too. This page is for the person setting it up.

## What exists today

| Feature | Status |
|---|---|
| CLI (`cutpoint`) | Available |
| Result files an agent can read (`highlights.json`, `manifest.json`, `events.json`, `transcript.txt`) | Available |
| Python library (`import cutpoint`) | Available, no stable API promise yet |
| `cutpoint edit` (video in, finished edit out) | Planned |
| `cutpoint init` (first-run questions about data and output locations) | Planned |
| JSON output on stdout for every command | Planned |
| Agent skill | Planned |
| Activity profiles for crash detection, detected or built from the footage (`cutpoint profile`, `--context` on `profile` and `crashes`) | Available |
| Profiles for the other scoring layers, and per-game profiles | Planned |

## Give the agent access

The agent needs a shell and the path to the repository. Add a note to its project instructions
file (`CLAUDE.md`, `AGENTS.md` or your tool's equivalent):

```markdown
Video editing: use the Cutpoint CLI (`uv run cutpoint ...` in D:\tools\cutpoint).
Read D:\tools\cutpoint\AGENTS.md first. Never edit source videos.
```

On Linux, for example `uv run cutpoint ...` in ~/tools/cutpoint, and read ~/tools/cutpoint/AGENTS.md.

With a tool install (`uv tool install cutpoint`) there is no repository path:

```markdown
Video editing: use the Cutpoint CLI (`cutpoint ...`). Run `cutpoint guide` and follow it first.
Never edit source videos.
```

Cutpoint is driven through the CLI. When the planned agent skill ships, this page will describe
how to install it for each agent.

## Example requests

These work today:

- "Using Cutpoint (D:\tools\cutpoint, see AGENTS.md), find the funniest moments in
  D:\Videos\stream.mp4 (Linux: ~/Videos/stream.mp4), export the top 6 clips and tell me their
  labels and lengths."
- "Make a 60 second vertical highlights video from ~/Videos/ride.mp4 with Cutpoint."
- "Read the transcript in the `stream-funny` project, pick the three best banter exchanges and
  export them as clips."
- "Find the crashes in the `bike-park` project. It's downhill mountain biking with a chest
  mount, and the lens gets muddy."

A bare "edit this video and make it a montage" with no pointer to Cutpoint depends on the planned
`cutpoint edit` and the skill.

## Limits to know

- Scoring is heuristic and works best with voice or mic audio. Review the clips before assembling.
- Free-text instructions are mostly applied by the agent when choosing moments. The exception
  is what the footage is: the agent passes the user's description to `cutpoint profile
  <project> --context "..."` (or `crashes --context`), and the local LLM turns it into crash
  prompts and slang. Tell your agent what the video is ("my first day mountain biking, chest
  mount"); no file needs editing.
- Optional modules need extras and models: see [Getting started](getting-started.md).
