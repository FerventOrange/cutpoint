# Game streams and gameplay recordings

Use style `gaming` for fast gameplay with commentary. `commentary` suits long stream VODs where
the talk matters more than the action. `analyze` also runs the sound-event and HUD-text steps (kill feeds, round banners) when the `events`
extra and models are installed, which re-rank the candidates and let clips run on to the payoff
after the banter; otherwise those steps are skipped with an INFO line. The optional `events`
command adds the model descriptions (`vlm` layer) and can force layers.

```powershell
cutpoint analyze D:\Videos\Game --style gaming --out match-night
cutpoint events match-night --player YourGameName     # optional: VLM layer, forced layers; needs the `events` extra and `cutpoint setup`
cutpoint clips match-night --top 8
cutpoint assemble projects/match-night/clips --transition fade --loudnorm
```

Set `player_tags` in [`config.json`](../configuration.md#player_tags) once instead of passing
`--player` every time. Your own deaths are told apart from your kills only when your name is set
this way; without it no event counts as yours.

With `--style funny`, a moment where you die and someone reacts (laughter, a scream, or speech
after a pause, within 6 s) gets +8 points, so a friend's warning after the fact ranks higher.
This needs your name set, as above; other styles are unchanged.

## What `events` adds

| Layer | Detects |
|---|---|
| `audio` | Laughter, screams, cheering, gunfire, explosions (PANNs) |
| `ocr` | Kill-feed rows (who killed whom, your kills and deaths), round banners, "N VS M", multi-kills |
| `vlm` | A local Qwen2.5-Omni model describes candidate windows from frames, audio and transcript |

Outputs (written by `analyze` and by `events`): `events.json` and `events.txt`. `events.txt` interleaves the transcript with the
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
[Configuration](../configuration.md#hud-presets)). No game presets are bundled. Write one, or let region
discovery create it (below), in `<CUTPOINT_HOME>\presets` or in a project `hud_presets` folder. A
preset with `"builtin": true` in the user folder keeps the built-in interpretation; in a project `hud_presets`
folder it is read as a generic preset (see [Configuration](../configuration.md#hud-presets)). The game is
picked from the file or folder name, or set with `--game KEY`.

For a game without a preset, `analyze` (`gaming` and `funny` styles) and `events` run region discovery automatically; `analyze` logs `first run for <game>: discovering HUD regions`, which can take several minutes on CPU OCR. It samples bursts of
frames, finds text that changes quickly (kill feeds, banners, notices), ignores static HUD and
world text, and writes a proposal to `<project>\discovery\<game>\` (`proposal.json`, `overlay.jpg`)
and installs it as a preset. Check `overlay.jpg`, then edit the preset JSON to calibrate.
A discovered kill feed grows by the oldest rows that show only when the feed is full (up to
twice its height). With such a preset, menu and scoreboard frames produce no OCR events;
`stats.ocr.<source>.menu_frames` in `events.json` reports how many frames were left out.
Force or skip it with `--discover-regions` and `--no-discover`.

A preset JSON has `regions` (`name`, `x`, `y`, `w`, `h`, `kind`: `feed`, `notice` or `banner`; optional `gate`: `mask` (default, a HUD colour mask) or `det` (text detection, works for any game); optional `every`),
`banner_phrases`, `player_tags`, `match` hints and optional `feed` settings. Coordinates are in
1920x1080 reference pixels and are scaled per axis to the video. A small example, saved as
`my-game.json`:

```json
{
  "match": ["my game"],
  "regions": [
    {"name": "feed", "x": 1500, "y": 100, "w": 400, "h": 300, "kind": "feed", "gate": "det"},
    {"name": "banner", "x": 660, "y": 80, "w": 600, "h": 100, "kind": "banner", "gate": "det", "every": 3}
  ],
  "banner_phrases": ["VICTORY"],
  "player_tags": ["MyName"]
}
```

A preset can also define its own event types (what an event is worth, how clip tails end, which
moments the vision model describes first) and the game name used in the vision prompt; see
[Event types and prompt keys](../configuration.md#event-types-and-prompt-keys).

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
