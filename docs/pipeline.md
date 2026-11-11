# Pipeline

Cutpoint reads source videos (never modifying them), finds the best moments, exports them as
numbered clips and joins them into a final video. `cutpoint edit` runs the whole chain in one
process; the same stages are available as `analyze`, `events`, `clips` and `assemble`.
[Architecture](architecture.md) covers the modules behind each stage.

## Main flow

```mermaid
flowchart TD
    SRC[/"Source videos and folders"/]
    EDIT["cutpoint edit --kind K"]
    SRC --> EDIT
    SRC --> ANA

    subgraph ANALYZE["cutpoint analyze  (steps 1-7)"]
        direction TB
        ANA["Find inputs"]
        DET["--style auto: detect the category (tags, telemetry, 40 frames, HUD text and vision check when unsure)"]
        AUD["1 Transcribe: extract audio, read check, loudness"]
        TR["1 Transcribe: Whisper"]
        SC["1 Transcribe: scene cuts"]
        PL["Plan layers by style or detected category"]
        TALK["--style auto: talk decided from the transcript"]
        RE["2 Reactions: PANNs audio events"]
        KF["3 Keyframes: HUD OCR for game styles, CLIP for action"]
        DISC["HUD discovery, preset per game"]
        CR["4 Crashes: action-cam footage only"]
        DLG["5 Dialogue: text model marks funny or heated talk, gaming only"]
        RK["6 Ranking: signals plus events rescore"]
        VIS["7 Vision rating: Ollama, opt-in"]
        ANA --> DET --> AUD --> TR --> TALK --> SC --> PL --> RE --> KF --> CR --> DLG --> RK --> VIS
        DISC --> KF
    end

    CACHE[("Per-source cache folder: audio, transcript, loudness, scenes, OCR")]
    AUD -.-> CACHE
    TR -.-> CACHE
    SC -.-> CACHE
    KF -.-> CACHE

    subgraph BG["Background OCR overlap"]
        direction TB
        ELIG{{"Eligible if: game style, keyframes not skipped, events extra and OCR models present, clean read so far, Whisper on CUDA, HUD preset for every game, enough VRAM, CUTPOINT_ANALYZE_OVERLAP not 0"}}
        PF["Worker thread runs OCR per source"]
        ELIG -->|yes| PF
    end
    AUD -. "offers each source" .-> ELIG
    PF -. "result taken in the OCR turn" .-> KF
    ELIG -->|no| KF

    EVJ[/"events.json, events.txt, crashes.json"/]
    HL[/"highlights.json, transcript.txt"/]
    KF --> EVJ
    RE --> EVJ
    CR --> EVJ
    VIS --> HL
    RK --> HL

    EVS["cutpoint events: adds VLM layer, re-ranks"]
    HL -. "optional" .-> EVS
    EVS --> HL

    EDIT --> ANA
    HL --> CLIPS["cutpoint clips: choose --top, --max-total, --ids or --segments"]
    CLIPS --> CDIR[/"clips/001_label.mp4, manifest.json, .srt"/]
    CDIR -->|"review: delete or rename"| ASM["cutpoint assemble: transition, loudnorm, captions"]
    ASM --> FIN[/"final.mp4 or, with --format, final.fcpxml / final.otio; final.srt"/]
    EDIT --> CLIPS
    EDIT --> ASM
    FIN -->|edit only| COPY[/"name-final.mp4 and .srt"/]
    HL -->|"assemble PROJECT, edit --no-keep-clips"| TMP[/".assemble-tmp-id: temporary clips, removed"/]
    TMP --> ASM
    CLIPS -->|"--vertical"| VDIR[/"clips_vertical/"/]
    VDIR --> ASM

    SRC --> P360["cutpoint 360 analyze --scan"]
    P360 --> V360["views: proxy view scores"]
    V360 --> A360["analyze and events on proxies"]
    CLIPS -. "360 projects" .-> DIR["360 director: view path per chosen moment"]
    A360 -.-> HL
```

## Kind presets (`cutpoint edit --kind`)

```mermaid
flowchart LR
    K1["highlights (default)"] --> S1["style action, max 180 s, chronological, fade"]
    K2["montage"] --> S2["style action, max 60 s, best first, fade"]
    K3["clips"] --> S3["style action, top 10, best first, cut"]
    K4["funny-moments"] --> S4["style funny, top 8, chronological, cut, burned-in captions"]
```

All kinds also set loudnorm on. `--style`, `--top` and `--max-total` override the kind.
`--style auto` picks the style from the footage before the `--events` check (the funny style for
`funny-moments`); unclear footage gets `gaming` when its best guess is gameplay, else the General style (`action`; `funny` for `funny-moments`).

## Style auto (`--style auto`, `cutpoint detect`)

```mermaid
flowchart TD
    P["Per source: container tags, telemetry, 40 small frames (cached)"]
    P --> U{{"Unsure between gameplay and general, no camera tags, not a still?"}}
    U -->|yes| OCR["HUD text check: OCR of 8 frames, text that stays in place, less clocks, dates and subtitles (cached; skipped in lite)"]
    U -->|no| V
    OCR --> V{{"Still unsure, or General by camera tags alone (no telemetry)?"}}
    V -->|yes| VIS["Vision check: 6 frames to the vision model (cached; skipped in lite, without consent, or with --no-vision)"]
    V -->|no| T
    VIS --> T
    T{{"Telemetry in at least half of the footage?"}}
    T -->|yes| AC["Action cam: action-cam"]
    T -->|no| G{{"Gameplay (static HUD or HUD text) the largest share?"}}
    G -->|yes| GP["Gameplay: gaming"]
    G -->|no| S{{"After transcription: speech covers 55% of a long source?"}}
    S -->|yes| TK["Talk: commentary"]
    S -->|no| AX{{"Did the vision check answer action?"}}
    AX -->|yes| ACT["Action: action"]
    AX -->|no| GE["General: action"]
    AMB["Still unsure: gaming if the best guess is gameplay, else action; style_ambiguous warning"]
    GP -.-> AMB
    GE -.-> AMB
    TK -.-> AMB
```

One category per project, by the share of the footage duration (sources of another category
over 10% of it: `style_sources_disagree`). The HUD text check only lifts an unsure source to
sure gameplay; it does not make camera-tagged footage or a still picture gameplay. In `analyze`'s own steps the detected category, not
the style, plans the HUD OCR and CLIP layers, so `funny` on camera footage gets the CLIP layer,
not the HUD OCR. `edit --events` and `cutpoint events` still choose their layers by style.
`funny-moments` maps every category to `funny`. `highlights.json` records the result; `events`
and re-runs reuse it.

## Legend

- Parallelogram: input or output files and folders (project files).
- Cylinder: per-source cache, keyed by path, size and modification time.
- Hexagon: eligibility decision for the background OCR.
- Solid arrow: normal flow. Dotted arrow: cache writes, optional or background paths.
- Numbers 1-7 are the analyze progress steps: transcribe, reactions, keyframes, crashes, dialogue,
  ranking, vision.
