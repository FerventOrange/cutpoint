# Setting up an agent to use Cutpoint

Cutpoint is a command-line tool, so any agent that can run shell commands can drive it:
Claude Code, Codex, Grok CLI, or a script. Analysis runs locally (ffmpeg, Whisper, optional
Ollama and llama.cpp models), so the agent only issues commands and reads small JSON and text
files. No video is sent to an API, and local models are the default for the model features. The AI
Chat in `cutpoint ui` can optionally use a hosted provider with your own API key; it then sends
text (moment labels, transcript snippets, file names, times), never video.

The instructions the agent itself should follow are in [AGENTS.md](../AGENTS.md) at the
repository root; `cutpoint guide` prints the same text, so an agent using a PyPI install
(`uv tool install cutpoint`) can read it too. This page is for the person setting it up.

## What exists today

The list of what is available and what is planned is the "Status" table in
[AGENTS.md](../AGENTS.md#status-what-you-can-rely-on). One item is not in that table: the Python
library (`import cutpoint`) is available, with no stable API promise yet.

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

## Install the skill (Claude Code)

Claude Code can find Cutpoint on its own, without a note in its instructions, once the
Cutpoint skill is installed:

```text
cutpoint skill install
```

This writes `~/.claude/skills/cutpoint/SKILL.md`, the Claude Code user skills folder, so every
project sees it. Use `--dir` for another folder: a project's `.claude/skills`
(`cutpoint skill install --dir .claude/skills`) or another agent's skills folder. If an
installed skill differs (an older version or your own edits), the command stops with an error
naming the file; add `--force` to replace it. Re-run it after upgrading Cutpoint. See
[commands](commands.md#skill).

The skill is short: it covers the first run, the edit loop and the rules, and tells the agent to
run `cutpoint guide` for everything else.

Other CLIs (Codex, Grok CLI, scripts) do not use this skill folder; they work from AGENTS.md,
through the repository file or `cutpoint guide`, as set up above. To give such an agent the
skill text anyway, `cutpoint skill print` prints it to stdout.

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

An agent pointed at Cutpoint can do "edit this video and make it a montage" in one step with
`cutpoint edit <video> --kind montage`. With the skill installed, Claude Code also picks up a
bare request such as "edit this video" with no pointer to Cutpoint; other agents need the
pointer above.

## Limits to know

- Scoring is heuristic and works best with voice or mic audio. Review the clips before assembling.
- Free-text instructions are mostly applied by the agent when choosing moments. The exception
  is what the footage is: the agent passes the user's description to `cutpoint profile
  <project> --context "..."` (or `crashes --context`), and the local LLM turns it into crash
  prompts and slang. Tell your agent what the video is ("my first day mountain biking, chest
  mount"); no file needs editing.
- Optional modules need extras and models: see [Getting started](getting-started.md).
