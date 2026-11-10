# Roadmap

Cutpoint is meant to be a universal AI footage editor: you give it footage and nothing else, it
detects what is in the video and scores the moments from that. Settings such as the style are
overrides, and what you ask for, such as funny moments or a length, steers the result. The
diagram shows what has shipped, what comes next and where Cutpoint is
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
        2026-11-10 : 0.5.0 Vision role, local or hosted with consent
                   : Subscription proxy for hosted models
                   : Visual overhaul, filmstrip trim, Cut mode
                   : Per-cut transitions
                   : Gaming analysis with event candidates
                   : Dialogue reader for funny and heated talk
                   : Visual index for AI Chat search
                   : Crash moments and custom styles
                   : --style auto, opt-in
                   : Activity profiles discovered from the footage
                   : Resume for interrupted runs
                   : Timeline format choice, FCPXML or OTIO
    section Next
        Auto by default : Detection and scoring need no settings
                        : Auto becomes the default style
                        : Detected category shown, override in the UI
                        : Agents and lite on auto
        Vision and speed : Vision on by default at lower cost
                         : Faster scene detection on game footage
                         : Intel QSV and Apple VideoToolbox encode and decode
                         : Re-run analysis with overrides
        Requests steer scoring : Re-rank from a request
                               : Assistant-made styles
        AI Chat and agents : Chat runs commands you approve
                           : Chat controls the UI
                           : One-click edit with a request and clip order chosen by a model
                           : llama.cpp provider, word-level trims, voice and attachments
                           : Editable transcript
        More footage : Gaming beyond kills, goal and score banners, round score, explosions
                     : 360 polish, GPU remap, manual framing, clean-up pass, mixed folders
                     : Color matching between clips
                     : Game profiles discovered from the footage
    section Goal
        Universal : Category per source
                  : Audio-only podcasts
                  : Coverage of more kinds of footage
        Goal reached : Footage in, finished edit out, no settings
                     : Same in CLI, agents and UI
```

Issues behind the Next bands:

- Auto by default: [#897](https://github.com/FerventOrange/cutpoint/issues/897) (detected category shown, one-click override), [#901](https://github.com/FerventOrange/cutpoint/issues/901) (agents and CLI on auto), [#904](https://github.com/FerventOrange/cutpoint/issues/904) (lite), [#899](https://github.com/FerventOrange/cutpoint/issues/899) (zero-config acceptance test), [#900](https://github.com/FerventOrange/cutpoint/issues/900) (cross-category benchmark).
- Vision and speed: [#909](https://github.com/FerventOrange/cutpoint/issues/909), [#907](https://github.com/FerventOrange/cutpoint/issues/907), [#893](https://github.com/FerventOrange/cutpoint/issues/893) (vision on by default at lower cost), [#730](https://github.com/FerventOrange/cutpoint/issues/730) (scene detection), [#448](https://github.com/FerventOrange/cutpoint/issues/448) (QSV and VideoToolbox), [#880](https://github.com/FerventOrange/cutpoint/issues/880) (re-run analysis with overrides).
- AI Chat and agents: [#706](https://github.com/FerventOrange/cutpoint/issues/706) (Chat runs commands, controls the UI), [#708](https://github.com/FerventOrange/cutpoint/issues/708) (llama.cpp provider, word-level trims, voice, attachments), [#260](https://github.com/FerventOrange/cutpoint/issues/260) (editable transcript), [#2](https://github.com/FerventOrange/cutpoint/issues/2) (one-click edit: `edit --context`, model-chosen clip order).
- More footage: [#800](https://github.com/FerventOrange/cutpoint/issues/800), [#872](https://github.com/FerventOrange/cutpoint/issues/872), [#874](https://github.com/FerventOrange/cutpoint/issues/874) (gaming), [#10](https://github.com/FerventOrange/cutpoint/issues/10), [#345](https://github.com/FerventOrange/cutpoint/issues/345), [#344](https://github.com/FerventOrange/cutpoint/issues/344), [#341](https://github.com/FerventOrange/cutpoint/issues/341) (360), [#290](https://github.com/FerventOrange/cutpoint/issues/290) (color matching), [#8](https://github.com/FerventOrange/cutpoint/issues/8) (game profiles discovered from the footage).

This page is updated at each release. The [CHANGELOG](https://github.com/FerventOrange/cutpoint/blob/main/CHANGELOG.md) has the details.
