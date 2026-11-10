# Roadmap

Cutpoint is meant to be a universal AI footage editor: you give it footage and nothing else, it
detects what is in the video and scores the moments from that. Settings such as the style are
overrides, and what you ask for, such as funny moments or a length, steers the result. The
diagram shows what has shipped, what is on `main` for the next release and where Cutpoint is
going.

```mermaid
---
title: "Cutpoint: from inception to the universal AI footage editor"
---
timeline
    section Start
        2026-10-21 : 0.1.0 first public release
                   : analyze, clips, assemble pipeline
                   : AGENTS.md agent-driven CLI
                   : Game HUD OCR events
                   : Action-cam telemetry and crashes
                   : Insta360 360 module
    section Foundations
        2026-10-22 : 0.2.0 cutpoint ui browser workspace
                   : AMD and NVENC encoding, capabilities
                   : Linux llama.cpp setup and Docker
        2026-10-27 : 0.3.0 cutpoint edit, one command to a finished video
                   : --json envelope and Claude Code skill
                   : HDR tone mapping and LUTs
                   : Safer exports, locks, hardened UI
        2026-10-29 : 0.4.0 one-line installers, full and lite
                   : AI Chat assistant with BYO providers
                   : Automatic color and underwater grade
                   : 360 auto-stitch on export
    section Intelligence
        0.5.0 (next release) : Vision role, local or hosted with consent
                             : Subscription proxy for hosted models
                             : Visual overhaul, filmstrip trim, Cut mode
                             : Per-cut transitions
                             : Gaming analysis with event candidates
                             : Dialogue reader for funny and heated talk
                             : Visual index for AI Chat search
                             : Crash moments and custom styles
                             : --style auto, opt-in
                             : Timeline format choice, FCPXML or OTIO
    section Next
        Auto by default : Detection and scoring need no settings
                        : Auto becomes the default style
                        : Detected category shown, override in the UI
                        : Agents and lite on auto
        Requests steer scoring : Re-rank from a request
                               : Assistant-made styles
    section Goal
        Universal : Category per source
                  : Audio-only podcasts
                  : Coverage of more kinds of footage
        Goal reached : Footage in, finished edit out, no settings
                     : Same in CLI, agents and UI
```

This page is updated at each release. The [CHANGELOG](https://github.com/FerventOrange/cutpoint/blob/main/CHANGELOG.md) has the details.
