# Game streams and gameplay recordings

Use style `gaming` for fast gameplay with commentary. `commentary` suits long stream VODs where
the talk matters more than the action. `analyze` also runs the sound-event and HUD-text steps (kill feeds, round banners) when the `events`
extra and models are installed, which re-rank the candidates and let clips run on to the payoff
after the banter; otherwise those steps are skipped with an INFO line. The optional `events`
command adds the model descriptions (`vlm` layer) and can force layers.

For `gaming`, `analyze` also runs a `dialogue` step: a text model reads the whole transcript and marks funny or heated talk, and those stretches become candidates that keep the whole stretch. A candidate such a stretch overlaps is widened to it, and candidates it joins become one, but only when the result stays within 90 s (a longer stretch is a candidate of its own), so an id seen in an earlier `highlights.json` can be gone. A widened candidate drops a silence of 8 s or more (no speech in the transcript) before or after its original moment. See [Dialogue reader](../commands.md#dialogue-reader); `--skip dialogue` leaves it out.

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

## Explosions

The audio layer also keeps PANNs "Explosion" sounds (Explosion, Boom, Eruption). Gameplay is full
of booms, so one counts only when it is `confirmed`: the loudness jumps at least 6 LU over the 3 s
before it, and those 3 s are not near-silent (at least -50 LUFS). A loud boom right after a black
screen is not confirmed: when the 3 s before it to 0.2 s after hold at least 0.4 s of near-black
(mean luma 0.03 or less, such as a loading screen ending; a single dark frame does not count), the
row gets `after_black: true`. Each row in
`events.json` has `loud_jump`, `backing` (a scene cut or OCR text nearby; recorded only, not
required) and `confirmed`. The thresholds are in `settings.explosions`.

Confirmed booms are the built-in event type `explosions`: 4 points each, at most 2 per candidate,
in the gaming, action and sport styles. In `gaming` a boom can nominate a moment of its own.

Limits: your own gunfire can pass, and quiet or microphone-heavy mixes get no explosions. The
first run after updating re-runs the audio model once per source (new class group).

## What `events` adds

| Layer | Detects |
|---|---|
| `audio` | Laughter, screams, cheering, explosions (PANNs) |
| `ocr` | Kill-feed rows (who killed whom, your kills and deaths), round banners, notices, multi-kills, learned announcements, on-screen score changes |
| `vlm` | The shared vision model (Ollama `qwen3.5:9b`, or a hosted (API or subscription) Vision model you allowed in the Models tab) describes candidate windows from frames, the transcript and the other layers' hints |

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
discovery create it (below), in `<CUTPOINT_HOME>\presets` or in a project `hud_presets` folder. The game is
picked from the file or folder name, or set with `--game KEY` (how the name becomes a key:
[Configuration](../configuration.md#hud-presets)). A source whose HUD text lands in other places on
screen than in the footage of that name's preset gets a key and a discovery of its own (the HUD
fit check, same page). To give an older preset the fit check, run
`cutpoint events <project> --discover-regions` on a folder
where that game is the majority; it adds only the check's reference to the preset, made from the
sources that fit.

For a game without a preset, `analyze` (`gaming` and `funny` styles) and `events` run region discovery automatically; `analyze` logs `first run for <game>: discovering HUD regions`, which can take several minutes on CPU OCR. It samples bursts of
frames, finds text that changes quickly (kill feeds, banners, notices), ignores static HUD and
world text, and writes a proposal to `<project>\discovery\<game>\` (`proposal.json`, `overlay.jpg`)
and installs it as a preset. A stack of score popups (30% or more of its rows a "+" number alone,
such as "+10" over a label) gets no region, and no extra notice is proposed inside it;
`proposal.json` lists it as a `popup` proposal with `score_rows` evidence. Check `overlay.jpg`, then edit the preset JSON to calibrate.
A discovered kill feed grows by the oldest rows that show only when the feed is full (up to
twice its height). Kill feeds without kill words: when at least 30% of the frames with text in a
region at a screen edge show two names split by an icon gap (in at least 3 bursts, with at least 4 different names, at least one of them on both sides of an icon gap somewhere in the region),
discovery proposes it as a `feed` even if names repeat or it shows one row at a time. A short clip
in which no player both kills and dies is therefore not found as a feed. The written
preset then gets `"feed": {"row_notices": true}`. The parser reads icon lines (exactly two names
split by an icon gap) as kills only with a preset whose `feed` has `row_notices` and no team
`colors`, that is a discovered icon feed; the feed's other rows (a name and text, such as
"X has found the bombs") become notices; such rows that stay longer than 20 s, like lobby text, are
dropped. With such a preset, menu and scoreboard frames produce no OCR events outside the kill feed,
and icon kills read on menu frames (a settings row such as "Subtitles | Enabled") are dropped too;
an icon kill first read as a still screen opens is not taken as feed activity, so a still menu
is still found as one.
A region proposed as a `feed` needs a score of at least 0.02, as a banner or notice does, and a
group of text whose rows change text height (stream chat over a battle log) is split into one
proposal per text size. Then the shared vision model (Ollama, or the hosted Vision model of the
Models tab of `cutpoint ui`) looks at up to four crops of each proposal and says what it is:
`kill_feed`, `chat`, `notice`, `banner`, `menu`, `readout` or `other`. It does not draw a box; it
names the rectangles discovery found. A `kill_feed` proposal becomes the feed region, a `feed`
the model calls anything but `kill_feed` or `chat` becomes a `notice`, and `chat` stays a `feed`
region, which the chat check below drops. The labels are saved in the preset (`discovery.labels`,
and `label` on the regions), so this runs once per game. The preset of a game with a `kill_feed` region also gets
`"feed": {"row_notices": true}`, so icon lines in it are read as kills. It runs whenever the vision model
answers: a local Ollama, or an Ollama on another machine or a hosted provider only with your
Vision consent (otherwise the `vision_consent_needed` warning). Without a model an INFO line says
so and the rules decide alone. An Ollama model is loaded first (up to 300 s); when Ollama runs it
on the CPU alone (no GPU memory in `ollama ps`) nothing is asked, an INFO line says so and the
rules decide. Each proposal then gets up to 60 s and all of them 120 s together, the likely kill
feeds first; a proposal that runs out of time ends the labelling and the labels so far are kept.
About 3.5 s per proposal on a GPU, at most 12 proposals.
A game that already has a preset does not pick this up: discovery does not run for it and the old
preset (no feed region, no `row_notices`) stays, so its kills stay at 0. Either move or rename the old
preset (`<presets>/<game>.json`, or the project's `hud_presets/<game>.json`) and run `analyze` or
`events` again, so discovery installs the new one, or run `events --discover-regions` (which leaves
the preset's regions alone, at most adds the HUD fit check's `discovery.fit_ref`, and writes
`projects/<name>/discovery/<game>/proposal.json`) and copy the
proposal's `feed` region and `"feed": {"row_notices": true}` into the preset by hand. Either way the new region needs new OCR reads (no cache replay).
Menus with little text (hangar or crew screens) are found because they hold still: a frame is a
static screen when a small grey thumbnail of the HUD area (the regions but `score` ones) stays nearly the same, with no feed
sighting or kill within 2 s, for at least 5 s when new menu text shows (at least 2 text lines
that are no preset phrase, first read in that stretch or just before it, each on screen for at
least half of it; one gameplay notice is not enough). Without new menu text the screen needs at
least 10 s plus another menu frame (box, list or text screen) within 15 s, or must hold still
for at least 45 s, so still gameplay (respawn and spectator waits, a preparation phase, scoped
moments of about 6 s) keeps its banners. Kills and other feed sightings are never dropped, even in a menu
frame. Menus that keep moving (lobbies, operator select, deploy maps, result screens with
animated backgrounds) are found by their text: a frame is a text screen when at least 5 lines
of new menu text are on screen at once, each staying 6 to 60 s, at least 3 of them first read
within 2 s of each other (a menu opens at once; gameplay labels arrive one by one), for at
least 3 s, with no feed sighting or kill within 2 s (a busy gameplay moment, with name tags,
labels and notices piling up, goes on beside the kill feed; a lobby does not). The rows a
discovered icon feed reads as no kill count as one line together, however many are on screen
at once, so a full feed whose rows the parser cannot read as kills is no text screen. A line on screen
for over 60 s, or from within 6 s of the start of the read to within 6 s of its end for at least half of it (also when a scoreboard hides it for a while), is part of
the HUD and does not count. A gap of up to 15 s between two menu or text screens, with no feed
sighting or kill in between (a loading or matchmaking screen), is menu time too; a still screen
does not count as an end of such a gap. These tests run only when the preset's regions (but `score` ones) span at least a quarter of the frame
(`static_area`), since a few small corner regions see too little of the screen.
`stats.ocr.<source>.menu_frames` in `events.json` reports how many frames were left out,
`static_frames` how many of them held still and `text_frames` how many were text screens;
`menu_spans` lists those frames as `[start, end]` spans in seconds.
Highlight candidates in menu time are left out too, by `analyze` and when `events` re-ranks
`highlights.json`: a candidate with half or more of its time in menu spans is dropped, and one
that starts or ends in a menu span is trimmed to its gameplay part when what is left is at least
the style's minimum clip length (menu time only in the middle is kept). An `INFO` line per
source counts the candidates dropped, trimmed and kept with menu time inside. `--ids` picks from
the candidates that are left; a `--segments` file is exported as given.
With `cutpoint -v`, the OCR layer logs one line per frame that is a menu or has an OCR event:
the time, the box count of the regions other than feeds, the centre cover, the recording's
median box count, the thresholds (cover, list, jump), how long the screen held still, whether new
menu text showed (`menu text yes`, `no`, or `-` near feed activity), how many staying lines
were on screen and how many of them were first read together (`staying lines`) and the
rule that decided (`cover`, `jump`, `list`, `static`, `text` or `none`). Discovery logs how a kill feed grew (`Feed growth ...`: the row pitch, the frames with
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
has a preset, `--discover-regions` never changes the preset's regions (it only adds the HUD fit
check's `discovery.fit_ref`): the extra regions are listed under
`suggested_regions` in `proposal.json` (with sample texts in the `proposals` evidence) and an
INFO line says to copy them into the preset's `regions` to have OCR read them. `overlay.jpg`
draws them in magenta.

A preset JSON has `regions` (`name`, `x`, `y`, `w`, `h`, `kind`: `feed`, `notice`, `banner` or `score` (see [Score readout](#score-readout)); optional `gate`: `mask` (default, a HUD colour mask), `det` (text detection, works for any game) or `read` (detection and recognition every time, for `score`); optional `every`; optional `upscale`, for `mask` and `read`: read the crop enlarged to this many times its reference size),
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

## Learned announcements

Pop-ups that are no preset phrase can still count. A big-text line that comes back as a short
pop-up all through one source is learned as an announcement: its event in `events.json`
gets `announce: true` and `announce_words` (the learned words) in `details`, and counts as the
`announcements` event type. Any line counts, a one-word line too. A line is learned by its words
of 4 letters or more; a short line with digits, a shorter word and no such word ("4 VS 5") by
its shape, digits as `#` ("# VS #"); the shape needs a word of 2 letters or more, so one letter beside digits is never learned. Such digit lines make no event otherwise. A bare number
("+100", "12:34") is never learned. A `score` region's reads never become text events or announcements, and a line another region reads with its box centre inside a `score` region is dropped (the region's other lines stay). A word must show at least 10 times,
in at least 4 different 10-minute stretches (so a source of at least 30 minutes), with a median
gap of at least 30 s, and at most a fifth of its appearances may stay longer than 30 s (labels
and place names stay). Reads within 8 s of each other are one appearance. Only big text is
learned: notice-sized lines (item and action prompts recur there), feed rows and banners never
are, and text that stays on a screen (the same word with the same name beside it
3 times in a row, each within 60 s) keeps only its first. Learning runs on the cached reads, so
it needs no new OCR.

An announcement is boost-only: it adds 3 points (at most 2 counted) to a candidate that already
exists, and it never makes a moment of its own or seeds a vision window, in `gaming` too.

## Score readout

A small number on the HUD that holds still and now and then steps by one, such as a round score,
a lap or a counter that counts down, can make events. Discovery looks for it in the frames it
already reads: a box that holds a number of 1 or 2 digits (one stray glyph beside it, such as an
icon stuck to the digit, is allowed; never a clock with `:`, a share with `%` or a pop-up
starting with `+`), at one screen spot in at least 6 bursts, one value all through 80% of those
bursts (a timer, ammo or a speed changes within the seconds of a burst), at least 2 values over
the bursts and none above 30. Up to 2 such spots become `score` regions (`score`, `score2`),
read once a second (every 3rd frame at the default 3 fps) and enlarged to 2.5 times their
reference size, as thin digits read far better that way. For a game that already has a preset,
`--discover-regions` lists them under `suggested_regions` in `proposal.json` (unless the preset
has a `score` region) and leaves the preset alone; copy them into its `regions` to read them.
Presets discovered before this have none, and are not discovered again for it. A new region
means new OCR reads (no cache replay).

A value counts once it is read twice in a row with a confidence of 0.7 or more. A change is an
event when the new value is one more or one less than the old, the old one stayed at least 15 s
and the HUD showed no number for at most 60 s in between. Each spot keeps the direction of its
first change: a counter that counts down is fine, and a step back afterwards is no event.
Anything else (a reset between matches, a jump) is taken as the new value without an event, a
value that comes back within 15 s is a misread, and reads on menu frames are left out. A spot
whose value changes more often than every 15 s (median) or that shows more than 12 values is no
score (a fast-changing clock or ammo) and makes no events. Nor does a spot whose value once
moves by two or more in its direction (the direction of its first change) to a value that then
stays 15 s, at most 60 s after the old one: a score moves one point at a time, a count of items
(gadgets placed, charges used) drops by several at once. A jump the other way is a reset and
changes nothing. A slow counter that only ever steps by one (lives, grenades) still makes events. Each change is an OCR event of `type` `score` in
`events.json`, labelled like `4 > 3`, at the last read of the old value, with `before`,
`after`, `seen_at` (the first read of the new value), `gap_s`, `region` and `slot` (the spot's
centre x in reference pixels) in `details`; `events.txt` shows it as `OCR score: 4 > 3` and the
vision model gets the hint `on-screen score: 4 > 3`. It counts as the `score_changes` event type
(see [Event types and prompt keys](../configuration.md#event-types-and-prompt-keys)).
`stats.ocr.<source>.score_slots` in `events.json` lists per spot its reads, values, changes
and whether it was disabled. A `score` region is left out of the HUD area the static-screen
test compares, so adding one does not change the other regions' events.

## Kill feed rows without a preset

Without a preset (or with one that sets no `kill_words`), these generic kill words read text rows:
`destroyed`, `shot down`, `killed`, `eliminated`, `assist`, `crashed` and `has crashed`. `wrecked`,
`set afire`, `critically damaged`, `severely damaged` and `has been wrecked` are no longer default
words, so `Kip wrecked Alder` is not read unless a preset's `kill_words` lists the word.

A clock-stamped log
(`2:20 Name destroyed` over a wrapped line) is joined into one row first; only the kill words read
such a log.

On any feed, a `:` or `;` (also full width) before `assist`, `crashed` or a word that reads as a kill
makes the row chat, so `Marlin: killed that corner` gives no event. This holds with a preset's own
`kill_words`. A clock that starts the row (`2:20`, `[2:20]`, `(2:20)`, `#2:20`) and a clan tag's own
colon (`[A:B] Kip destroyed Alder`) do not count; a colon elsewhere does, also in `Kip 2:20 destroyed
Alder`.

Player tags (`--player`) tolerate OCR variants: spaces and punctuation are ignored, look-alike
letters (`0`/`O`, `1`/`I`, `5`/`S`) are one, and a clan tag or weapon read with the name still
matches. A tag shorter than 5 letters and digits must match a whole word of the name. A longer tag
may also be read with a letter dropped or added (7 letters or more), or with two (12 or more) or
three (16 or more) letters wrong, and one read as another counts only from 12 letters: a 9-letter
`Emberfall` matches `Emberfal`, not `Emberwall`. A row cut at the wrong place between the names
(`Stonhar,QQ` and `uillmerefox7`) is credited to the side that holds most of the tag, and to
neither when the halves are equal; this is done for a tag of 7 letters or more only. A name that only has your tag inside it matches, as before; a name that needs an edit to hold a shorter tag (under 12 letters) matches only as a whole name or word (`Dunmorrow` is no `Dunmore`).
A recurring system line that was read as a kill row (a notice that comes back five times in the
feed region, 10 s or more apart) gives no kill; a notice that is a kill's own row does not count towards that.

Limits: a `name verb name` row with any other verb stays unread unless a preset's `kill_words` lists
it. A feed that writes a colon before its kill word (`Kip: killed Alder`) reads as chat and gives no
kill. On a feed not labelled chat (see
[Stream chat in a feed region](#stream-chat-in-a-feed-region)), chat without a colon still reads as a
kill, as `bro just killed that corner` does, and a row whose colon OCR lost on one frame can still
read as a kill. Clock-stamped chat without a colon can also be read as
a kill by the log join: with a later clock-stamped row such as `12:31 Alder left` below them,
`12:30 Kip he got destroyed` and `that was so bad` join into one row with a kill word. The two rows
alone are no log and give no kill. A log kill seen only inside a menu span is not kept, even if it is
real.

## Stream chat in a feed region

A stream overlay can put the chat where discovery finds a "feed". A chat shows far more different
text lines than a kill feed, so a discovered feed region is judged by the distinct lines it read per
hour over the whole OCR pass (text boxes read with low confidence or under 3 letters and digits are
not counted):

- 6000 lines per hour or more, on at least 10 minutes of footage: the region is chat and gives no
  events. An INFO line says so.
- 3000 to 6000 lines per hour, on at least 10 minutes of footage: it stays a feed, with a WARNING
  that it may be chat.
- Under 10 minutes of decoded footage at the chat rate: it stays a feed, with a WARNING that the
  footage is too short to tell. Under 10 minutes below the chat rate, it stays a feed with no
  warning.

A hand-made preset is never relabelled: its feed stays a feed, with a WARNING when it reads like
chat. A preset made by discovery has `"source": "discovered"`; if you edit it by hand it keeps that
key and is still relabelled, so remove the key to stop it.

A region that mixes chat and the game's log goes wholly to chat for now, so its log events are lost.

The OCR stats list each feed region under `feed_regions`: `label` (`chat` or `feed`),
`lines_per_hour`, `lines`, `hours`, and `dropped`, the events removed from a chat region by type.
A kept region can also have `unsure`, `too_short` or `chat_like` (hand-made preset) set to true.
The raw text stays in the raw OCR cache.

The label and the row rule of [Kill feed rows without a preset](#kill-feed-rows-without-a-preset) are
separate. The label drops a whole discovered feed at the chat rate. The row rule looks at one row, on
any feed: a `:` or `;` before the kill word makes that row chat, also on short clips, on feeds below
the rate and on hand-made presets, where the label removes nothing. Other rows of those feeds are read
as usual. Cached OCR interpretation is redone once after the update.

## Limits

- OCR samples 3 frames per second, so event times are accurate to about 0.3 s.
- Recordings with a different aspect ratio than 16:9 get a different HUD layout; OCR may find
  little.
- Text colours are unreliable on bright backgrounds, so colour-based teamkill detection is off
  for discovered presets.
- The vision model's window descriptions are a soft signal: they boost a candidate but never lower it.
- Stream layouts with the game in a box: HUD regions cannot be remapped into the box. Crop the
  clips instead with
  `--crop auto` or a per-segment `"crop": "W:H:X:Y"`.
- `--avoid "GLOB@START-END"` marks ranges that are never decoded. Passing the complement of the
  windows you care about restricts OCR to them, which saves hours on long VODs. Each range is
  widened by 2 s on both sides.
