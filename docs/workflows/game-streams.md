# Game streams and gameplay recordings

Use style `gaming` for fast gameplay with commentary. `commentary` suits long stream VODs where
the talk matters more than the action. The optional `events` command adds sound events,
HUD text (kill feeds, round banners) and model descriptions, which re-rank the candidates and let
clips run on to the payoff after the banter.

```powershell
cutpoint analyze D:\Videos\Game --style gaming --out match-night
cutpoint events match-night --player YourGameName     # needs the `events` extra and `cutpoint setup`
cutpoint clips match-night --top 8
cutpoint assemble projects/match-night/clips --transition fade --loudnorm
```

Set `player_tags` in [`config.json`](../configuration.md#player_tags) once instead of passing
`--player` every time.

## What `events` adds

| Layer | Detects |
|---|---|
| `audio` | Laughter, screams, cheering, gunfire, explosions (PANNs) |
| `ocr` | Kill-feed rows (who killed whom, your kills and deaths), round banners, "N VS M", multi-kills |
| `vlm` | A local Qwen2.5-Omni model describes candidate windows from frames, audio and transcript |

Outputs: `events.json` and `events.txt`. `events.txt` interleaves the transcript with the
events by time, which is the easiest file to read when picking segments:

```text
[match.mp4 00:06:31.0] OCR kill: YourGameName > EnemyName. (player kill)
[match.mp4 00:06:36.8] AUDIO: Laughter (0.13, 1.6s)
```

`highlights.json` is re-ranked: sound events and kills near a candidate boost its score, and
the best model window is blended in at 35%. Re-running is idempotent.

With `events.json` present, `clips` also applies event tails: if a payoff starts within
`--event-window` seconds (default 10) after the last word, the clip ends on it. Use
`--no-event-tails` to turn that off.

## HUD presets

OCR regions come from a per-game preset (lookup order in
[Configuration](../configuration.md#hud-presets)). Rainbow Six Siege and War Thunder
ship as bundled presets (`r6.json`, `wt.json`) that you can copy and edit as a starting point. The game is picked from the file or folder name, or set with `--game KEY`.

For a game without a preset, `events` runs region discovery automatically. It samples bursts of
frames, finds text that changes quickly (kill feeds, banners, notices), ignores static HUD and
world text, and writes a proposal to `<project>\discovery\<game>\` (`proposal.json`, `overlay.jpg`)
and installs it as a preset. Check `overlay.jpg`, then edit the preset JSON to calibrate.
Force or skip it with `--discover-regions` and `--no-discover`.

A preset JSON has `regions` (`name`, `x`, `y`, `w`, `h`, `kind`: `feed`, `notice` or `banner`),
`banner_phrases`, `player_tags`, `match` hints and optional `feed` settings. Coordinates are in
1920x1080 reference pixels and are scaled per axis to the video. See
`src/cutpoint/events/presets/wt.json` for a complete example.

## Limits

- OCR samples 3 frames per second, so event times are accurate to about 0.3 s.
- Recordings with a different aspect ratio than 16:9 get a different HUD layout; OCR may find
  little.
- Text colours are unreliable on bright backgrounds, so colour-based teamkill detection is off
  for discovered presets.
- The Omni model gives fluent but generic descriptions; it is a soft signal.
- Stream layouts with the game in a box: there is no CLI flag to remap HUD regions into the box
  yet (a `--hud-box` option is a suggested feature). Crop the clips instead with
  `--crop auto` or a per-segment `"crop": "W:H:X:Y"`.
- `--avoid "GLOB@START-END"` marks ranges that are never decoded. Passing the complement of the
  windows you care about restricts OCR to them, which saves hours on long VODs. Each range is
  widened by 2 s on both sides.
