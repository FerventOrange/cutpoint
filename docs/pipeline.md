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

    subgraph ANALYZE["cutpoint analyze  (steps 1-6)"]
        direction TB
        ANA["Find inputs"]
        AUD["1 Transcribe: extract audio, read check, loudness"]
        TR["1 Transcribe: Whisper"]
        SC["1 Transcribe: scene cuts"]
        PL["Plan layers by style"]
        RE["2 Reactions: PANNs audio events"]
        KF["3 Keyframes: HUD OCR for game styles, CLIP for action"]
        DISC["HUD discovery, preset per game"]
        CR["4 Crashes: action-cam footage only"]
        RK["5 Ranking: signals plus events rescore"]
        VIS["6 Vision rating: Ollama, opt-in"]
        ANA --> AUD --> TR --> SC --> PL --> RE --> KF --> CR --> RK --> VIS
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

## Legend

- Parallelogram: input or output files and folders (project files).
- Cylinder: per-source cache, keyed by path, size and modification time.
- Hexagon: eligibility decision for the background OCR.
- Solid arrow: normal flow. Dotted arrow: cache writes, optional or background paths.
- Numbers 1-6 are the analyze progress steps: transcribe, reactions, keyframes, crashes, ranking,
  vision.
