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
and installs it as a preset. A stack of score popups (30% or more of its rows a "+" number alone,
such as "+10" over a label) gets no region, and no extra notice is proposed inside it;
`proposal.json` lists it as a `popup` proposal with `score_rows` evidence. Check `overlay.jpg`, then edit the preset JSON to calibrate.
A discovered kill feed grows by the oldest rows that show only when the feed is full (up to
twice its height). With such a preset, menu and scoreboard frames produce no OCR events outside the kill feed.
Menus with little text (hangar or crew screens) are found because they hold still: a frame is a
static screen when a small grey thumbnail of the HUD area stays nearly the same, with no feed
sighting or kill within 2 s, for at least 5 s when new menu text shows (at least 2 text lines
that are no preset phrase, first read in that stretch or just before it, each on screen for at
least half of it; one gameplay notice is not enough) or 10 s without, so still gameplay (scoped,
about 6 s) keeps its banners. Kills and other feed sightings are never dropped, even in a menu
frame. This test
runs only when the preset's regions span at least a quarter of the frame (`static_area`), since a
few small corner regions see too little of the screen.
`stats.ocr.<source>.menu_frames` in `events.json` reports how many frames were left out, and
`static_frames` how many of them held still.
With `cutpoint -v`, the OCR layer logs one line per frame that is a menu or has an OCR event:
the time, the box count of the regions other than feeds, the centre cover, the recording's
median box count, the thresholds (cover, list, jump), how long the screen held still, whether new
menu text showed (`menu text yes`, `no`, or `-` near feed activity) and the
rule that decided (`cover`, `jump`, `list`, `static` or `none`). Discovery logs how a kill feed grew (`Feed growth ...`: the row pitch, the frames with
a full feed, what a row needs, and above and below every row tried, with its frames (how many
with a full feed), bursts, label variety and sighting variety, ending in `ok: frames` (seen in
enough frames), `ok: varied` (fewer frames, but a different text per burst while the feed is
full) or `stop: <reason>` (`limit`, `screen edge`, `frames`, `bursts` or `variety`)). These lines hold no OCR text, so they can be shared when tuning a preset.
Force or skip it with `--discover-regions` and `--no-discover`.

OCR reads only inside the preset's regions, so discovery also looks for text that keeps showing
outside all of them (the discovered regions and the screen-centre big-text check, which reads
only text at least 36 px tall) but too rarely to be found as changing text, such as a brief
objective message at the top edge. From the frames it already read, it keeps text lines with
at least 4 letters (60% of the characters) that are not static labels, groups lines that stay
at one screen anchor (left edge, right edge or centre) and proposes up to 2 extra `notice`
regions, `extra_notice` and `extra_notice2`, for lines seen in at least 3 bursts. Left
out are a line that is, for 80% of its sightings, one or two texts (misreads and
cut-off reads count as the same text; the same phrase with another name does not) and shows in
more than 2 bursts per text and in at least 30% of the frames of those bursts, such as a map or
mode label or a lobby line (a notice that recurs but shows in only one of the 4 frames of each
burst stays; text in 2 of them is taken for a label, as a name plate that shows now and then
would otherwise become a region, so a two-frame notice that repeats one text in 3 or more
bursts is left out); a
line that is mostly short readouts (at most 8 letters and digits, with a digit or mixed case) of
at most 4 texts with digits left out, such as a gauge (a column of short player names has more
texts and stays); stacked lines that are mostly such labels; and a line seen mostly in frames
with far more text than usual (a partial scoreboard). When the preset already has regions with
`"origin": "outside"`, they count against the 2. These regions carry `"origin": "outside"`, which the preset loader ignores. A new
preset gets them with the other regions; the INFO line `Discovery: text shows outside the ...
regions` gives their rectangles, burst and frame counts, never the text. For a game that already
has a preset, `--discover-regions` never changes the preset file: the extra regions are listed under
`suggested_regions` in `proposal.json` (with sample texts in the `proposals` evidence) and an
INFO line says to copy them into the preset's `regions` to have OCR read them. `overlay.jpg`
draws them in magenta.

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
