# Workflows

All examples use `cutpoint` for `uv run cutpoint`. Paths are examples; use your own.

| Page | Use it for |
|---|---|
| This page | Montage and highlights from any video, vertical output, importing into Resolve |
| [Game streams](game-streams.md) | Gameplay and stream recordings, HUD OCR, boxed stream layouts |
| [Action-cam](action-cam.md) | GoPro-style footage with telemetry: jumps, crashes, a varied mix |
| [360 video](360.md) | Insta360 X3: mix, director, QA, repair, clips, assemble |
| [360 director](360-director.md) | How the 360 virtual-camera director works |

## Montage or highlights from one or more videos

```powershell
# 1. Find and rank moments. Writes projects/my-edit/highlights.json and transcript.txt
cutpoint analyze D:\Videos\clip.mp4 --style gaming --out my-edit

# Several files and folders at once, skipping some by glob:
cutpoint analyze D:\Videos\session1 D:\Videos\extra.mp4 --exclude "*DVR*" --style gaming --out my-edit

# 2. Optional: detect what happens (sounds, HUD text, model descriptions).
#    Needs the `events` extra and `cutpoint setup`. Re-ranks highlights.json.
cutpoint events my-edit

# 3. Export the best moments as 001_<label>.mp4, 002_<label>.mp4, ...
cutpoint clips my-edit --top 8

# 4. Review projects/my-edit/clips. Delete clips you do not want; rename to reorder.

# 5. Build the video from whatever is in the folder, in filename order.
cutpoint assemble projects/my-edit/clips --transition fade --loudnorm
```

Output: `projects/my-edit/final.mp4`, `final.fcpxml` and `final.otio`.

Choose the style by content: `funny` for banter and laughter, `gaming` or `action` for fast
footage, `sport` for crowd-driven events, `commentary` for long talk (clips of 20-120 s, in
chronological order).

### Pick moments yourself

Read `transcript.txt` (or `events.txt` after `events`), then write the ranges you want:

```json
[
  {"source": "D:/Videos/clip.mp4", "start": 312.4, "end": 330.0, "label": "intro"},
  {"source": "D:/Videos/clip.mp4", "start": 1204.0, "end": 1219.5, "label": "joke"}
]
```

```powershell
cutpoint clips my-edit --segments segments.json
```

You can mix hand-picked segments with auto-scored ones by exporting both and joining the clips.
Per-segment options (tails, crop, padding) are in the [command reference](../commands.md#clips).

### More room for trimming

If you will trim in an editor, export handles:

```powershell
cutpoint clips my-edit --top 8 --extend 1.5         # padding at export
cutpoint extend projects/my-edit/clips --seconds 2            # or re-export an existing folder
```

## Vertical (9:16) output

```powershell
cutpoint clips my-edit --vertical --max-total 60      # projects/my-edit/clips_vertical, 1080x1920
cutpoint assemble projects/my-edit/clips_vertical --captions   # projects/my-edit/final_vertical.mp4
```

`--vertical` is a fixed centre crop with no subject tracking. For 360 footage the renderer
frames the shot itself (see [360 video](360.md)).

## Importing into DaVinci Resolve (Free)

Resolve Free has no Python API, so the hand-off is through files.

1. Open your project and choose File > Import > Timeline (Ctrl+Shift+I).
2. Pick `final.fcpxml`. `final.otio` also works.
3. Leave "Automatically import source clips into media pool" checked and click OK.

The timeline places the clips back to back in filename order at their frame rate. It uses hard
cuts even if the MP4 was built with `--transition fade`, so you can add transitions in Resolve.
Audio fades are baked into the clip files. If you move the clips folder, relink the media in
Resolve or run `assemble` again.

## Reading slow or network sources once

Copy sources to a local disk first with `cutpoint stage`; see
[slow storage](../troubleshooting/storage.md).
