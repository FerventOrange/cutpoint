// Cut mode of the center stage: the join of two neighbouring sequence clips, how the current cut
// moves, "Play across the cut", the roll and the whole-video transition.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

async function stage(t, { project = makeProject(), routes = {} } = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project, routes });
  t.after(() => page.close());
  const d = page.document;
  const w = page.window;
  const m = {
    page, d, w,
    $: (s) => d.querySelector(s),
    title: () => d.querySelector('#stageTitle').textContent,
    cutBtn: () => d.querySelector('#stageMode [data-mode=cut]'),
    clipBtn: () => d.querySelector('#stageMode [data-mode=clip]'),
    seqIds: () => [...d.querySelectorAll('#seqList > li')].map((li) => li.dataset.id),
    range: (id) => d.querySelector(`#seqList li[data-id=${id}] .seq-main`).title,
    strip: () => [...d.querySelectorAll('#cutStrip button[data-cut]')],
    joins: () => [...d.querySelectorAll('#cutStrip button[data-join]')],
    active: () => [...d.querySelectorAll('.screen video')].find((e) => e.classList.contains('active')),
    live: () => d.querySelector('#seqLive').textContent,
    async click(el) { (typeof el === 'string' ? m.$(el) : el).click(); await page.settle(); },
    async key(target, key, init = {}) {
      target.dispatchEvent(new w.KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init }));
      await page.settle();
    },
    async plan() { // /edges is asked 300 ms after a change
      t.mock.timers.tick(400);
      await page.settle();
    },
    /** A pointer on the join: the track is 600 px wide in the test, so 100 px are 1 s. */
    pointer(type, x, id = 1) {
      m.$('#cutJoin').dispatchEvent(Object.assign(new w.MouseEvent(type, { clientX: x, button: 0, bubbles: true, cancelable: true }), { pointerId: id, pointerType: 'mouse' }));
    },
    async drag(...xs) {
      m.pointer('pointerdown', xs[0]);
      for (const x of xs.slice(1)) {
        m.pointer('pointermove', x);
        t.mock.timers.tick(16);
        await page.settle();
      }
      m.pointer('pointerup', xs.at(-1));
      m.pointer('lostpointercapture', xs.at(-1));
      await page.settle();
    },
  };
  return m;
}
const withSegments = (...ids) => {
  const p = makeProject();
  p.review = { segments: ids.map((id) => {
    const c = p.highlights.candidates.find((x) => x.id === id);
    return { id, source: c.source, start: c.start, end: c.end, label: c.label };
  }) };
  return p;
};
/** ``ranges``: {id: [start, end]} over the candidates' own. */
const withRanges = (ranges) => {
  const p = withSegments(...Object.keys(ranges));
  for (const s of p.review.segments) [s.start, s.end] = ranges[s.id];
  return p;
};
/** As ``withSegments``, with ``moved`` in a second source file, b.mp4 (the others are in a.mp4). */
const withOtherFile = (moved, ...ids) => {
  const p = withSegments(...ids);
  p.sources.push({ id: 's2', path: '/videos/b.mp4', name: 'b.mp4', status: 'ready', url: '/api/projects/p1/sources/s2?t=x', duration: 600 });
  for (const x of [...p.highlights.candidates, ...p.review.segments]) if (x.id === moved) x.source = '/videos/b.mp4';
  return p;
};
const edgesRoute = (edges) => ({ 'POST /api/projects/p1/edges': () => ({ edges }) });
const inCut = async (t, opts) => {
  const m = await stage(t, opts);
  await m.click(m.cutBtn());
  return m;
};

// ---- the mode ----

test('Cut needs two clips: disabled with its reason until then, and Cut mode shows the join in place of the player', async (t) => {
  const m = await stage(t, { project: withSegments('h001') });
  assert.deepEqual([...m.$('#stageMode').children].map((b) => b.textContent), ['Clip', 'Cut', 'Sequence']);
  assert.equal(m.cutBtn().getAttribute('aria-disabled'), 'true');
  assert.equal(m.cutBtn().disabled, false, 'it can take focus and a tap');
  assert.match(m.cutBtn().title, /needs a sequence of at least two clips/);
  await m.click(m.cutBtn());
  assert.equal(m.$('#cutView').hidden, true);
  assert.match(m.$('#trimMsg').textContent, /needs a sequence of at least two clips/);
  assert.ok(!m.$('#trimMsg').classList.contains('vh'), 'on screen, for a tap');
  await m.click(m.$('#candList li[data-id=h002] [data-act=toggle]'));
  assert.equal(m.cutBtn().getAttribute('aria-disabled'), 'false');
  assert.equal(m.cutBtn().title, '');
  assert.doesNotMatch(m.$('#trimMsg').textContent, /needs a sequence/, 'the reason goes with the second clip');
  await m.click(m.cutBtn());
  assert.equal(m.cutBtn().getAttribute('aria-pressed'), 'true');
  assert.equal(m.clipBtn().getAttribute('aria-pressed'), 'false');
  assert.equal(m.$('#cutView').hidden, false);
  assert.ok(m.$('#stageBody').classList.contains('cut-mode'));
  assert.equal(m.title(), 'Cut 1 → 2');
  assert.equal(m.$('#cutOutName').textContent, '1 · last frame');
  assert.equal(m.$('#cutInName').textContent, '2 · first frame');
  assert.equal(m.$('#cutOutTime').textContent, 'out 0:18.0');
  assert.equal(m.$('#cutInTime').textContent, 'in 1:40.0');
  assert.ok(m.$('.stage-head').contains(m.$('#stageMode')), 'the switch stays reachable: the player bar is hidden');
});

test('Clip takes the stage back: the switch returns to the player bar and the clip title to the head', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') });
  await m.click(m.clipBtn());
  assert.equal(m.$('#cutView').hidden, true);
  assert.ok(!m.$('#stageBody').classList.contains('cut-mode'));
  assert.ok(m.$('.transport').contains(m.$('#stageMode')));
  assert.equal(m.clipBtn().getAttribute('aria-pressed'), 'true');
  assert.notEqual(m.title(), 'Cut 1 → 2');
  assert.equal(m.$('#stageTitle').getAttribute('aria-label'), null);
});

test('no typed time fields, nudge buttons or transcript box come with Cut mode', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') });
  assert.equal(m.$('#cutView').querySelectorAll('input, textarea, select').length, 0);
  assert.doesNotMatch(m.$('#cutView').textContent, /Use playhead|Transcript/);
});

test('Sequence leaves Cut mode and plays the whole sequence', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002', 'h003') });
  await m.click('#stageMode [data-mode=sequence]');
  assert.equal(m.$('#cutView').hidden, true);
  assert.match(m.live(), /Playing clip 1 of 3/);
  assert.equal(m.$('#stageMode [data-mode=sequence]').getAttribute('aria-pressed'), 'true');
});

// ---- the current cut ----

test('the cut follows the selected clip; the last clip shows the cut before it', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002', 'h003') });
  await m.click(m.$('#seqList li[data-id=h002] .seq-main'));
  assert.equal(m.title(), 'Cut 2 → 3');
  await m.click(m.$('#seqList li[data-id=h003] .seq-main'));
  assert.equal(m.title(), 'Cut 2 → 3');
  await m.click(m.$('#seqList li[data-id=h001] .seq-main'));
  assert.equal(m.title(), 'Cut 1 → 2');
});

test('Previous and Next cut step through the cuts and stop at the ends, keeping focus on a live button', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002', 'h003') });
  assert.equal(m.$('#prevCut').disabled, true);
  m.$('#nextCut').focus();
  await m.click('#nextCut');
  assert.equal(m.title(), 'Cut 2 → 3');
  assert.equal(m.$('#nextCut').disabled, true);
  assert.equal(m.d.activeElement, m.$('#prevCut'), 'a disabled button would drop the focus');
  await m.click('#prevCut');
  assert.equal(m.title(), 'Cut 1 → 2');
});

test('the Cuts strip has one button per cut, a single tab stop, arrow keys and Enter through a click', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002', 'h003') });
  assert.equal(m.$('#cutStrip').getAttribute('aria-label'), 'Cuts in sequence order');
  assert.equal(m.strip().length, 2);
  assert.match(m.strip()[0].getAttribute('aria-label'), /^Cut 1 of 2: Big laugh into Fall$/);
  assert.equal(m.strip()[0].getAttribute('aria-current'), 'true');
  assert.deepEqual(m.strip().map((b) => b.tabIndex), [0, -1]);
  m.strip()[0].focus();
  await m.key(m.strip()[0], 'ArrowRight');
  assert.equal(m.d.activeElement, m.strip()[1]);
  assert.equal(m.title(), 'Cut 1 → 2', 'arrows move the focus, a click opens');
  await m.click(m.strip()[1]);
  assert.equal(m.title(), 'Cut 2 → 3');
  assert.equal(m.strip()[1].getAttribute('aria-current'), 'true');
  await m.key(m.strip()[1], 'Home');
  assert.equal(m.d.activeElement, m.strip()[0]);
  await m.key(m.strip()[0], 'End');
  assert.equal(m.d.activeElement, m.strip()[1]);
});

test('a moment outside the sequence ends Cut mode; a removed clip leaves the cut that took its place', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') });
  await m.click(m.$('#candList li[data-id=h003] [data-act=focus]'));
  assert.equal(m.$('#cutView').hidden, true);
  assert.equal(m.clipBtn().getAttribute('aria-pressed'), 'true');

  await m.click(m.$('#candList li[data-id=h003] [data-act=toggle]')); // 3 clips
  await m.click(m.$('#seqList li[data-id=h002] .seq-main'));
  await m.click(m.cutBtn());
  assert.equal(m.title(), 'Cut 2 → 3');
  await m.click(m.$('#candList li[data-id=h002] [data-act=toggle]'));
  assert.deepEqual(m.seqIds(), ['h001', 'h003']);
  assert.equal(m.title(), 'Cut 1 → 2');
  assert.equal(m.$('#cutView').hidden, false);
});

test('an /edges reply leaves the reason for a disabled Cut on screen and still updates the length', async (t) => {
  let asked = 0;
  const m = await stage(t, {
    project: withSegments('h001'),
    routes: { 'POST /api/projects/p1/edges': () => { asked += 1; return { edges: { h001: { start: 10, end: 20, reason: 'smart tail' } } }; } },
  });
  await m.click(m.cutBtn());
  await m.plan();
  assert.equal(asked, 1, 'the reply came after the press');
  assert.match(m.$('#trimMsg').textContent, /needs a sequence of at least two clips/);
  assert.ok(!m.$('#trimMsg').classList.contains('vh'));
  assert.match(m.$('.cut-line').textContent, /10\.00 s/);
  await m.click(m.$('#candList li[data-id=h002] [data-act=toggle]'));
  assert.doesNotMatch(m.$('#trimMsg').textContent, /needs a sequence/, 'gone with the next change');
});

test('too few clips end Cut mode and say why', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') });
  await m.click(m.$('#candList li[data-id=h002] [data-act=toggle]'));
  assert.equal(m.$('#cutView').hidden, true);
  assert.equal(m.cutBtn().getAttribute('aria-disabled'), 'true');
  assert.match(m.$('#sideLive').textContent, /needs a sequence of at least two clips/);
});

test('Undo that leaves one clip moves the focus off the Cut view and keeps the reason', async (t) => {
  const m = await stage(t, { project: withSegments('h001') });
  await m.click(m.$('#candList li[data-id=h002] [data-act=toggle]'));
  await m.click(m.cutBtn());
  const join = m.$('#cutJoin');
  join.focus();
  await m.key(join, 'z', { ctrlKey: true });
  assert.equal(m.$('#cutView').hidden, true);
  assert.equal(m.d.activeElement, m.clipBtn());
  assert.match(m.$('#sideLive').textContent, /^Undone\. .*at least two clips/);
});

test('entering Cut mode pauses a clip playing in Clip mode', async (t) => {
  const m = await stage(t, { project: withSegments('h001', 'h002') });
  const v = m.active();
  v.__media.loaded(600);
  await m.page.settle();
  await m.key(m.d.body, ' ');
  assert.equal(v.__media.paused, false, 'it plays');
  await m.click(m.cutBtn());
  for (const x of m.d.querySelectorAll('.screen video')) assert.equal(x.__media.paused, true);
  assert.equal(m.$('#cutPlayBtn').textContent, 'Play across the cut');
});

test('the frames are the last frame of clip 1 and the first of clip 2, asked once a change rests', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') });
  t.mock.timers.tick(1000);
  await m.page.settle();
  assert.match(m.$('#cutOutFrame').getAttribute('src'), /\/frame\?t=x&at=17\.9\d*&w=640$/);
  assert.match(m.$('#cutInFrame').getAttribute('src'), /\/frame\?t=x&at=100&w=640$/);
  assert.match(m.$('#cutOutFrame').alt, /Last frame of clip 1: Big laugh/);
});

test('a still that is loading dims the old picture; a roll dims it again until the new one loads or fails', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') });
  t.mock.timers.tick(1000);
  await m.page.settle();
  const img = m.$('#cutOutFrame');
  const frame = img.closest('.cut-frame');
  const ev = (type) => img.dispatchEvent(new m.w.Event(type));
  assert.ok(frame.classList.contains('loading'), 'loading until the still arrives');
  ev('load');
  assert.ok(!frame.classList.contains('loading'));
  const url1 = img.dataset.url;
  await m.key(m.$('#cutJoin'), 'ArrowLeft'); // roll: the time changes, so does the src
  await m.plan();
  t.mock.timers.tick(1000);
  await m.page.settle();
  assert.notEqual(img.dataset.url, url1);
  assert.ok(frame.classList.contains('loading'), 'a new src dims again');
  ev('load'); // fires for the latest src
  assert.ok(!frame.classList.contains('loading'));
  ev('error');
  assert.ok(!frame.classList.contains('loading'));
});

// ---- play across the cut ----

test('Play across the cut plays the last 3 s of clip 1, then the first 3 s of clip 2 from the spare player', async (t) => {
  const m = await inCut(t, { project: withOtherFile('h002', 'h001', 'h002', 'h003') }); // a.mp4, then b.mp4
  await m.click('#cutPlayBtn');
  assert.equal(m.$('#cutPlayBtn').textContent, 'Stop');
  assert.ok(m.$('#stageBody').classList.contains('cut-playing'));
  const videos = [...m.d.querySelectorAll('.screen video')];
  for (const v of videos) v.__media.loaded(600);
  await m.page.settle();
  const first = m.active();
  assert.equal(first.__media.currentTime, 15, 'clip 1 from 3 s before its end');
  assert.ok(videos.some((v) => v.__media.currentTime === 100), 'the spare player waits at clip 2');

  first.__media.tick(18);
  t.mock.timers.tick(16);
  await m.page.settle();
  assert.match(m.live(), /Playing clip 2 of 2/);
  assert.notEqual(m.active(), first, 'the step is a swap');
  assert.equal(m.title(), 'Cut 1 → 2', 'the cut stays while it plays');

  m.active().__media.tick(103);
  t.mock.timers.tick(16);
  await m.page.settle();
  assert.equal(m.live(), 'Finished playing across the cut');
  assert.equal(m.$('#cutPlayBtn').textContent, 'Play across the cut');
  assert.ok(!m.$('#stageBody').classList.contains('cut-playing'));
  assert.equal(m.title(), 'Cut 1 → 2', 'clip 1 is selected again');
  assert.equal(m.$('#seqList li[aria-current=true]').dataset.id, 'h001');
});

test('Play across a Cut between two clips of one file seeks in the same player (#1020)', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') }); // both in a.mp4
  await m.click('#cutPlayBtn');
  const first = m.active();
  first.__media.loaded(600);
  await m.page.settle();
  const spare = [...m.d.querySelectorAll('.screen video')].find((e) => e !== first);
  assert.equal(spare.getAttribute('src'), null, 'the file is not opened a second time');
  first.__media.tick(18);
  t.mock.timers.tick(16);
  await m.page.settle();
  assert.match(m.live(), /Playing clip 2 of 2/);
  assert.equal(m.active(), first, 'no swap');
  assert.equal(first.__media.currentTime, 100);
  assert.equal(first.__media.paused, false);
});

test('Play across a Fade between two clips of one file opens it in the spare player too: both play at once', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') });
  await m.click('#cutTransition button[data-value=fade]');
  await m.click('#cutPlayBtn');
  const first = m.active();
  const spare = [...m.d.querySelectorAll('.screen video')].find((e) => e !== first);
  assert.equal(spare.dataset.url, first.dataset.url);
  for (const v of [first, spare]) v.__media.loaded(600);
  await m.page.settle();
  assert.equal(spare.__media.currentTime, 100, 'waiting at clip 2');
  first.__media.tick(17.8);
  t.mock.timers.tick(16);
  await m.page.settle();
  assert.match(m.live(), /Playing clip 2 of 2/);
  assert.equal(m.active(), spare, 'the Fade starts on the spare player');
  assert.equal(first.__media.paused, false, 'under the outgoing clip');
  await m.click('#cutPlayBtn'); // Stop
});

test('Play across the cut honours the tails: it ends in the planned out and starts at the planned in', async (t) => {
  const m = await stage(t, {
    project: withSegments('h001', 'h002'),
    routes: edgesRoute({ h001: { start: 10, end: 20, reason: 'smart tail' }, h002: { start: 98, end: 108, reason: 'speech-aware' } }),
  });
  await m.plan();
  await m.click(m.cutBtn());
  assert.equal(m.$('#cutOutTime').textContent, 'out 0:20.0 with tail');
  assert.equal(m.$('#cutInTime').textContent, 'in 1:38.0');
  await m.click('#cutPlayBtn');
  const videos = [...m.d.querySelectorAll('.screen video')];
  for (const v of videos) v.__media.loaded(600);
  await m.page.settle();
  assert.equal(m.active().__media.currentTime, 17);
  m.active().__media.tick(18.5); // past the trim, inside the tail: still clip 1
  t.mock.timers.tick(16);
  await m.page.settle();
  assert.match(m.live(), /Playing clip 1 of 2/);
  m.active().__media.tick(20);
  t.mock.timers.tick(16);
  await m.page.settle();
  assert.match(m.live(), /Playing clip 2 of 2/);
  assert.equal(m.active().__media.currentTime, 98, 'one file: sought to the planned in');
  assert.ok(!videos.some((v) => v.__media.currentTime === 100), 'not to the trim');
});

test('a clip shorter than 3 s plays whole', async (t) => {
  const m = await inCut(t, { project: withRanges({ h001: [10, 11], h002: [100, 108] }) });
  await m.click('#cutPlayBtn');
  for (const v of m.d.querySelectorAll('.screen video')) v.__media.loaded(600);
  await m.page.settle();
  assert.equal(m.active().__media.currentTime, 10);
});

test('Stop and Escape end it with no player state left, and the selected clip stays', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') });
  await m.click(m.$('#seqList li[data-id=h002] .seq-main')); // the last clip: the cut before it
  await m.click('#cutPlayBtn');
  for (const v of m.d.querySelectorAll('.screen video')) v.__media.loaded(600);
  await m.page.settle();
  const first = m.active();
  first.__media.tick(18);
  t.mock.timers.tick(16);
  await m.page.settle();
  await m.click('#cutPlayBtn'); // Stop, during clip 2
  assert.equal(m.$('#cutPlayBtn').textContent, 'Play across the cut');
  for (const v of m.d.querySelectorAll('.screen video')) assert.equal(v.__media.paused, true);
  assert.equal(m.$('#seqList li[aria-current=true]').dataset.id, 'h002');
  assert.equal(m.$('#seqList').querySelectorAll('.playing').length, 0);
  assert.equal(m.$('#seqPlayBtn').textContent, 'Play sequence');
  assert.equal(m.cutBtn().getAttribute('aria-pressed'), 'true', 'Cut mode stays');

  await m.key(m.d.body, ' ');
  assert.equal(m.$('#cutPlayBtn').textContent, 'Stop', 'Space plays over the cut');
  await m.key(m.d.body, 'Escape');
  assert.equal(m.$('#cutPlayBtn').textContent, 'Play across the cut');
});

test('Next cut and leaving Cut mode stop a playing cut', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002', 'h003') });
  await m.click('#cutPlayBtn');
  await m.click('#nextCut');
  assert.equal(m.$('#cutPlayBtn').textContent, 'Play across the cut');
  assert.equal(m.title(), 'Cut 2 → 3');
  await m.click('#cutPlayBtn');
  await m.click(m.clipBtn());
  assert.ok(!m.$('#stageBody').classList.contains('cut-playing'));
  assert.equal(m.$('#seqPlayBtn').textContent, 'Play sequence');
});

test('clips that stop being neighbours stop the play', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002', 'h003') });
  await m.click('#cutPlayBtn');
  await m.click(m.$('#candList li[data-id=h003] [data-act=toggle]')); // still neighbours
  assert.equal(m.$('#cutPlayBtn').textContent, 'Stop');
  await m.click(m.$('#candList li[data-id=h002] [data-act=toggle]'));
  assert.equal(m.$('#cutPlayBtn').textContent, 'Play across the cut');
});

test('removing clip 1 while clip 2 plays stops the play and Cut mode stays', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002', 'h003') });
  await m.click('#cutPlayBtn');
  for (const v of m.d.querySelectorAll('.screen video')) v.__media.loaded(600);
  await m.page.settle();
  m.active().__media.tick(18);
  t.mock.timers.tick(16);
  await m.page.settle();
  assert.match(m.live(), /Playing clip 2 of 2/);
  await m.click(m.$('#candList li[data-id=h001] [data-act=toggle]'));
  assert.equal(m.$('#cutPlayBtn').textContent, 'Play across the cut');
  assert.ok(!m.$('#stageBody').classList.contains('cut-playing'));
  assert.equal(m.$('#seqPlayBtn').textContent, 'Play sequence');
  assert.equal(m.$('#cutView').hidden, false);
  assert.equal(m.title(), 'Cut 1 → 2');
});

test('the player is scrolled into view once, when it opens', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') });
  const calls = [];
  m.$('.screen').scrollIntoView = (o) => calls.push(o);
  await m.click('#cutPlayBtn');
  for (const v of m.d.querySelectorAll('.screen video')) v.__media.loaded(600);
  await m.page.settle();
  assert.deepEqual(calls, [{ block: 'nearest' }]);
});

// ---- the roll ----

test('the roll is a slider: arrows move clip 1\'s out and clip 2\'s in together and the length stays', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') });
  const join = m.$('#cutJoin');
  assert.equal(join.getAttribute('role'), 'slider');
  assert.equal(join.getAttribute('aria-label'), 'Roll the cut');
  assert.equal(join.getAttribute('aria-valuenow'), '18');
  assert.equal(join.getAttribute('aria-valuetext'), 'Clip 1 out 0:18.00, clip 2 in 1:40.00');
  await m.key(join, 'ArrowRight');
  assert.equal(m.range('h001'), '0:10.00 to 0:18.10');
  assert.equal(m.range('h002'), '1:40.10 to 1:48.00');
  await m.key(join, 'ArrowLeft', { shiftKey: true });
  assert.equal(m.range('h001'), '0:10.00 to 0:17.10');
  assert.equal(m.range('h002'), '1:39.10 to 1:48.00');
  await m.key(join, 'PageUp');
  assert.equal(m.range('h001'), '0:10.00 to 0:22.10');
  assert.equal(m.$('#saveState').textContent, 'Unsaved changes');
  assert.equal(join.getAttribute('aria-valuetext'), 'Clip 1 out 0:22.10, clip 2 in 1:44.10');
  assert.equal(m.$('#cutOutTime').textContent, 'out 0:22.1');
});

test('the total stays exact after many small rolls', async (t) => {
  let saved = null;
  const m = await inCut(t, { project: withSegments('h001', 'h002'), routes: { 'PUT /api/projects/p1/review': (req) => { saved = req.body; return {}; } } });
  for (let i = 0; i < 7; i += 1) await m.key(m.$('#cutJoin'), 'ArrowRight');
  await m.click('#saveBtn');
  const [a, b] = saved.segments;
  assert.deepEqual([a.end, b.start], [18.7, 100.7]);
  assert.equal(Math.round(((a.end - a.start) + (b.end - b.start)) * 1000), 16000);
});

test('a drag rolls with the pointer and is one undo step, also with pauses', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') });
  Object.defineProperty(m.$('#cutTrack'), 'clientWidth', { value: 600, configurable: true });
  await m.drag(300, 350);
  t.mock.timers.tick(5000);
  await m.page.settle();
  assert.equal(m.range('h001'), '0:10.00 to 0:18.50');
  await m.click('#undoBtn');
  assert.equal(m.range('h001'), '0:10.00 to 0:18.00');
  assert.equal(m.range('h002'), '1:40.00 to 1:48.00');

  m.pointer('pointerdown', 300);
  m.pointer('pointermove', 350);
  t.mock.timers.tick(1500);
  await m.page.settle();
  m.pointer('pointermove', 400);
  t.mock.timers.tick(1500);
  await m.page.settle();
  assert.equal(m.$('#cutTrack').style.getPropertyValue('--join'), '66.67%', 'the join moves on the held track');
  m.pointer('pointerup', 400);
  m.pointer('lostpointercapture', 400);
  await m.page.settle();
  assert.equal(m.range('h001'), '0:10.00 to 0:19.00');
  assert.equal(m.$('#cutTrack').style.getPropertyValue('--join'), '50.00%', 'centred on the new cut');
  await m.click('#undoBtn');
  assert.equal(m.range('h001'), '0:10.00 to 0:18.00', 'pauses over 1 s do not split the step');
});

test('a press without a move changes nothing and keeps no undo step', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') });
  await m.drag(300, 301);
  assert.equal(m.range('h001'), '0:10.00 to 0:18.00');
  assert.equal(m.$('#saveState').textContent, 'Sequence saved');
  assert.equal(m.$('#undoBtn').disabled, true);
});

test('arrow presses in a row are one undo step, as on the trim handles', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') });
  await m.key(m.$('#cutJoin'), 'ArrowRight');
  await m.key(m.$('#cutJoin'), 'ArrowRight');
  await m.click('#undoBtn');
  assert.equal(m.range('h001'), '0:10.00 to 0:18.00');
  assert.equal(m.range('h002'), '1:40.00 to 1:48.00');
});

test('the roll stops at 0.1 s of either clip and says so', async (t) => {
  const m = await inCut(t, { project: withRanges({ h001: [10, 10.5], h002: [100, 100.5] }) });
  const join = m.$('#cutJoin');
  assert.equal(join.getAttribute('aria-valuemin'), '10.1');
  assert.equal(join.getAttribute('aria-valuemax'), '10.9');
  await m.key(join, 'PageDown');
  assert.equal(m.range('h001'), '0:10.00 to 0:10.10');
  assert.equal(m.range('h002'), '1:39.60 to 1:40.50');
  assert.match(m.$('#cutRollMsg').textContent, /Clip 1 is at its shortest, 0\.1 s/);
  await m.key(join, 'End');
  assert.equal(m.range('h001'), '0:10.00 to 0:10.90');
  assert.equal(m.range('h002'), '1:40.40 to 1:40.50');
  await m.key(join, 'ArrowRight');
  assert.match(m.$('#cutRollMsg').textContent, /Clip 2 is at its shortest, 0\.1 s/);
  await m.key(join, 'ArrowLeft');
  assert.equal(m.$('#cutRollMsg').textContent, '', 'a roll that fits clears the note');
});

test('the roll stops at the source bounds of either clip, also for clips from different places', async (t) => {
  const m = await inCut(t, { project: withRanges({ h001: [590, 598], h002: [1, 9] }) });
  const join = m.$('#cutJoin');
  await m.key(join, 'PageDown'); // clip 2 is 1 s from the start of its source
  assert.equal(m.range('h001'), '9:50.00 to 9:57.00');
  assert.equal(m.range('h002'), '0:00.00 to 0:09.00');
  assert.match(m.$('#cutRollMsg').textContent, /Clip 2 is at the start of its source/);
  await m.key(join, 'End'); // clip 1 can reach the end of the source
  assert.match(m.range('h001'), /^9:50\.00 to 10:00\.00$/);
  assert.equal(m.range('h002'), '0:03.00 to 0:09.00');
  await m.key(join, 'ArrowRight');
  assert.match(m.$('#cutRollMsg').textContent, /Clip 1 is at the end of its source/);
  assert.equal(m.$('#trimMsg').textContent.includes('inside the source'), false);
});

test('a rolled sequence saves ranges the server accepts: no start below 0, no end past the source', async (t) => {
  let saved = null;
  const p = withRanges({ h001: [590, 598], h002: [1, 9] });
  p.sources[0].duration = 600.126;
  const m = await inCut(t, { project: p, routes: { 'PUT /api/projects/p1/review': (req) => { saved = req.body; return {}; } } });
  await m.key(m.$('#cutJoin'), 'End');
  await m.click('#saveBtn');
  const [a, b] = saved.segments;
  assert.ok(a.end <= 600.126);
  assert.ok(b.start >= 0);
  assert.equal(Math.round((a.end - a.start + b.end - b.start) * 1000), 16000);
});

test('the limit note goes with Undo and with leaving Cut mode', async (t) => {
  const m = await inCut(t, { project: withRanges({ h001: [10, 10.5], h002: [100, 100.5] }) });
  const msg = m.$('#cutRollMsg');
  await m.key(m.$('#cutJoin'), 'PageDown');
  assert.match(msg.textContent, /shortest/);
  await m.click('#undoBtn');
  assert.equal(msg.textContent, '');
  await m.key(m.$('#cutJoin'), 'PageDown');
  assert.match(msg.textContent, /shortest/);
  await m.click(m.clipBtn());
  await m.click(m.cutBtn());
  assert.equal(msg.textContent, '');
});

test('a roll into a third clip of the same source is allowed and says so', async (t) => {
  const m = await inCut(t, { project: withRanges({ h001: [10, 18], h002: [100, 108], h003: [19, 25] }) });
  const msg = m.$('#cutRollMsg');
  await m.key(m.$('#cutJoin'), 'ArrowRight');
  assert.equal(msg.textContent, '', 'not there yet');
  await m.key(m.$('#cutJoin'), 'PageUp');
  assert.equal(m.range('h001'), '0:10.00 to 0:23.10');
  assert.equal(msg.textContent, 'Clip 1 repeats part of clip 3.');
  await m.key(m.$('#cutJoin'), 'PageDown');
  assert.equal(msg.textContent, '', 'clear of it again');
});

test('the repeat note names every clip it runs into', async (t) => {
  const p = withRanges({ h001: [10, 18], h002: [100, 108], h003: [19, 21] });
  p.review.segments.push({ id: 'x004', source: '/videos/a.mp4', start: 20.5, end: 22, label: 'Extra' });
  const m = await inCut(t, { project: p });
  await m.key(m.$('#cutJoin'), 'ArrowRight');
  await m.key(m.$('#cutJoin'), 'PageUp');
  assert.equal(m.$('#cutRollMsg').textContent, 'Clip 1 repeats part of clips 3 and 4.');
});

test('a drag step that rounds to no movement keeps the repeat note', async (t) => {
  const m = await inCut(t, { project: withRanges({ h001: [10, 18], h002: [100, 108], h003: [18.5, 25] }) });
  const msg = m.$('#cutRollMsg');
  m.pointer('pointerdown', 300);
  m.pointer('pointermove', 400); // 1 s on, into clip 3
  t.mock.timers.tick(16);
  await m.page.settle();
  assert.equal(msg.textContent, 'Clip 1 repeats part of clip 3.');
  m.pointer('pointermove', 400.2); // under a hundredth of a second: the same place
  t.mock.timers.tick(16);
  await m.page.settle();
  assert.equal(m.range('h001'), '0:10.00 to 0:19.00');
  assert.equal(msg.textContent, 'Clip 1 repeats part of clip 3.');
  m.pointer('pointerup', 400.2);
  m.pointer('lostpointercapture', 400.2);
  await m.page.settle();
  assert.equal(msg.textContent, 'Clip 1 repeats part of clip 3.');
});

test('the shown out and in move with a roll before the new plan comes in', async (t) => {
  const m = await stage(t, {
    project: withSegments('h001', 'h002'),
    routes: edgesRoute({ h001: { start: 10, end: 20, reason: 'smart tail' }, h002: { start: 100, end: 108 } }),
  });
  await m.plan();
  await m.click(m.cutBtn());
  await m.key(m.$('#cutJoin'), 'ArrowRight');
  assert.equal(m.$('#cutOutTime').textContent, 'out 0:20.1 with tail', 'the tail holds until /edges answers');
});

test('rolling while a cut plays stops the play first', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') });
  await m.click('#cutPlayBtn');
  await m.key(m.$('#cutJoin'), 'ArrowRight');
  assert.equal(m.$('#cutPlayBtn').textContent, 'Play across the cut');
  assert.equal(m.range('h001'), '0:10.00 to 0:18.10');
});

test('I and O do not trim in Cut mode, where no trim row shows', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') });
  await m.key(m.d.body, 'i');
  await m.key(m.d.body, 'o');
  assert.equal(m.range('h001'), '0:10.00 to 0:18.00');
  assert.equal(m.$('#saveState').textContent, 'Sequence saved');
});

// ---- the transition ----

test('the transition control is Export\'s one and the default of every join; no Dip', async (t) => {
  const m = await inCut(t, { project: withSegments('h001', 'h002') });
  const buttons = [...m.d.querySelectorAll('#cutTransition button')];
  assert.deepEqual(buttons.map((b) => b.textContent), ['Cut', 'Fade']);
  assert.match(m.$('#cutTransNote').textContent, /^Default for every join/);
  assert.equal(m.$('#optTransition').value, 'cut');
  await m.click(buttons[1]);
  assert.equal(m.$('#optTransition').value, 'fade');
  assert.equal(m.$('#optFadeDur').disabled, false, 'as if chosen in the Export panel');
  assert.equal(buttons[1].getAttribute('aria-pressed'), 'true');
  assert.doesNotMatch(m.$('#cutTransNote').textContent, /hard cut/, 'Play across shows the Fade now');

  m.$('#optTransition').value = 'cut';
  m.$('#optTransition').dispatchEvent(new m.w.Event('change', { bubbles: true }));
  await m.page.settle();
  assert.equal(buttons[0].getAttribute('aria-pressed'), 'true', 'the Export panel moves it too');
  assert.equal(m.$('#saveState').textContent, 'Sequence saved', 'not an edit of the sequence');
});

// ---- the transition of each join ----

const PUT_REVIEW = 'PUT /api/projects/p1/review';
/** A Cut mode page that records what Save sends. */
const saving = async (t, ids = ['h001', 'h002', 'h003'], extra = {}) => {
  const sent = [];
  const m = await inCut(t, { project: extra.project ?? withSegments(...ids), routes: { [PUT_REVIEW]: (req) => { sent.push(req.body); return {}; } } });
  m.sent = sent;
  m.save = async () => { await m.click('#saveBtn'); return sent.at(-1); };
  return m;
};
const letters = (m) => m.joins().map((b) => b.textContent);
const pair = (a, b, transition) => ({ a, b, transition });

test('a button sits between every two blocks, with the letter of what the join plays, and steps Default, Cut, Fade, Default', async (t) => {
  const m = await saving(t);
  assert.equal(m.joins().length, 2);
  assert.deepEqual(m.joins().map((b) => b.dataset.own), ['false', 'false']);
  assert.deepEqual(letters(m), ['C', 'C'], 'Default shows the Export panel\'s transition');
  assert.match(m.joins()[0].getAttribute('aria-label'), /^Transition of cut 1 of 2: Default, Cut\. Press for Cut$/);
  await m.click(m.joins()[0]);
  assert.equal(m.joins()[0].dataset.own, 'true');
  assert.match(m.joins()[0].getAttribute('aria-label'), /Cut, set for this cut\. Press for Fade$/);
  assert.equal(m.$('#sideLive').textContent, 'Transition of cut 1 set to Cut');
  await m.click(m.joins()[0]);
  assert.deepEqual(letters(m), ['F', 'C']);
  assert.equal(m.$('#sideLive').textContent, 'Transition of cut 1 set to Fade');
  assert.equal(m.$('#saveState').textContent, 'Unsaved changes');
  assert.deepEqual((await m.save()).joins, [pair('h001', 'h002', 'fade')]);
  assert.equal(m.$('#saveState').textContent, 'Sequence saved');
});

test('a join next to a word splice is always a Cut and its button cannot change it', async (t) => {
  const project = withSegments('h001', 'h002', 'h003');
  project.review.segments[1].exact = true; // h002: a word splice
  project.review.joins = [pair('h001', 'h002', 'fade')]; // a choice saved before: still a cut
  const m = await saving(t, undefined, { project });
  await m.click('#cutTransition button[data-value=fade]');
  assert.deepEqual(letters(m), ['C', 'C'], 'both joins touch the splice: cuts under a Fade default');
  const [first, second] = m.joins();
  assert.equal(first.getAttribute('aria-disabled'), 'true');
  assert.equal(first.getAttribute('aria-label'), 'Cut 1: always a cut next to a word splice');
  assert.equal(second.title, 'Cut 2: always a cut next to a word splice');
  await m.click(first);
  assert.deepEqual(letters(m), ['C', 'C'], 'a click changes nothing');
  assert.equal(m.$('#sideLive').textContent.includes('Transition of cut 1'), false);
});

test('a join between two clips that are no word splices keeps its button', async (t) => {
  const project = withSegments('h001', 'h002', 'h003');
  project.review.segments[0].exact = true;
  const m = await saving(t, undefined, { project });
  await m.click('#cutTransition button[data-value=fade]');
  assert.deepEqual(letters(m), ['C', 'F']);
  assert.equal(m.joins()[1].getAttribute('aria-disabled'), null);
});

test('Default removes the entry, so a join back on Default is no unsaved change', async (t) => {
  const m = await saving(t);
  await m.click(m.joins()[1]); // Cut
  await m.click(m.joins()[1]); // Fade
  await m.click(m.joins()[1]); // Default
  assert.equal(m.joins()[1].dataset.own, 'false');
  assert.equal(m.$('#sideLive').textContent, 'Transition of cut 2 set to Default (Cut)');
  assert.equal(m.$('#saveState').textContent, 'Sequence saved', 'nothing differs from what is saved');
  await m.click(m.joins()[1]);
  await m.click(m.joins()[1]);
  assert.deepEqual((await m.save()).joins, [pair('h002', 'h003', 'fade')]);
  await m.click(m.joins()[1]); // Default again
  assert.deepEqual((await m.save()).joins, [], 'the entry is gone, not stored as "default"');
});

test('a join on Default follows the Export panel\'s transition, and its own choice does not', async (t) => {
  const m = await saving(t);
  await m.click(m.joins()[0]); // its own Cut
  await m.click('#cutTransition button[data-value=fade]');
  assert.deepEqual(letters(m), ['C', 'F']);
  assert.match(m.$('#sideLive').textContent, /^Default transition set to Fade$/);
  assert.equal(m.$('#saveState').textContent, 'Unsaved changes', 'the choice is an edit');
  assert.deepEqual((await m.save()).joins, [pair('h001', 'h002', 'cut')], 'the Default one is not stored');
});

test('a choice is of its ordered pair: it rests while the clips are apart and counts again beside each other, but not the other way round', async (t) => {
  const m = await saving(t);
  await m.click(m.joins()[0]);
  await m.click(m.joins()[0]); // h001 -> h002: Fade
  await m.click(m.$('#seqList li[data-id=h003] [data-act=up]')); // h001, h003, h002
  assert.deepEqual(m.seqIds(), ['h001', 'h003', 'h002']);
  assert.deepEqual(letters(m), ['C', 'C'], 'the pair is apart: the global transition');
  assert.deepEqual(m.joins().map((b) => b.dataset.own), ['false', 'false'], 'h003 -> h002 is not h001 -> h002');
  await m.click(m.$('#seqList li[data-id=h003] [data-act=down]')); // h001, h002, h003
  assert.deepEqual(m.seqIds(), ['h001', 'h002', 'h003']);
  assert.deepEqual(letters(m), ['F', 'C'], 'adjacent again: the choice returns');
  // the other way round: h002 -> h001 is another join
  await m.click(m.$('#seqList li[data-id=h002] [data-act=up]'));
  assert.deepEqual(m.seqIds(), ['h002', 'h001', 'h003']);
  assert.deepEqual(letters(m), ['C', 'C']);
  assert.equal(m.joins()[0].dataset.own, 'false');
  await m.click(m.$('#seqList li[data-id=h001] [data-act=up]'));
  assert.deepEqual(letters(m), ['F', 'C']);
  assert.deepEqual((await m.save()).joins, [pair('h001', 'h002', 'fade')], 'the resting choice is saved with the sequence');
});

test('a saved join comes back with the project, and the page keeps it after a reload of the same review', async (t) => {
  const p = withSegments('h001', 'h002', 'h003');
  p.review.joins = [pair('h002', 'h003', 'fade'), pair('h001', 'h009', 'cut')];
  const m = await saving(t, undefined, { project: p });
  assert.deepEqual(letters(m), ['C', 'F']);
  assert.equal(m.$('#saveState').textContent, 'Sequence saved');
  await m.click(m.joins()[0]);
  assert.deepEqual((await m.save()).joins, [pair('h001', 'h002', 'cut'), pair('h001', 'h009', 'cut'), pair('h002', 'h003', 'fade')], 'the join of a clip not here is kept, in a fixed order');
});

test('Undo and Redo bring back the joins with the sequence', async (t) => {
  const m = await saving(t);
  await m.click(m.joins()[0]); // Cut
  await m.click(m.joins()[0]); // Fade
  assert.deepEqual(letters(m), ['F', 'C']);
  await m.click('#undoBtn');
  assert.deepEqual(letters(m), ['C', 'C']);
  assert.equal(m.joins()[0].dataset.own, 'true', 'back to its Cut');
  await m.click('#undoBtn');
  assert.equal(m.joins()[0].dataset.own, 'false');
  assert.equal(m.$('#saveState').textContent, 'Sequence saved');
  await m.click('#redoBtn');
  await m.click('#redoBtn');
  assert.deepEqual(letters(m), ['F', 'C']);
  // an edit that touches no join still undoes the join it follows
  await m.click(m.$('#candList li[data-id=h003] [data-act=toggle]'));
  await m.click('#undoBtn');
  assert.deepEqual(m.seqIds(), ['h001', 'h002', 'h003']);
  assert.deepEqual(letters(m), ['F', 'C']);
});

/** Play across the cut of ``m``: both players loaded, the first at its start. */
const playAcross = async (m) => {
  await m.click('#cutPlayBtn');
  const videos = [...m.d.querySelectorAll('.screen video')];
  for (const v of videos) v.__media.loaded(600);
  await m.page.settle();
  return videos;
};
const advance = async (m, el, to) => {
  el.__media.tick(to);
  m.t.mock.timers.tick(16);
  await m.page.settle();
};

test('Play across a join set to Fade overlaps the clips under a Cut default', async (t) => {
  const m = await saving(t, ['h001', 'h002']);
  m.t = t;
  await m.click(m.joins()[0]);
  await m.click(m.joins()[0]); // its own Fade, the default is Cut
  await playAcross(m);
  const first = m.active();
  await advance(m, first, 17.5); // before the last 0.25 s: no step yet
  assert.match(m.live(), /Playing clip 1 of 2/);
  await advance(m, first, 17.8); // inside the Fade's length: the next clip starts under it
  assert.match(m.live(), /Playing clip 2 of 2/);
  assert.notEqual(m.active(), first);
  assert.equal(first.__media.paused, false, 'the outgoing clip plays on under the Fade');
  const second = m.active();
  assert.deepEqual([first.__media.volume, second.__media.volume], [1, 0], 'the Fade starts on the outgoing clip\'s sound');
  await advance(m, second, second.__media.currentTime + 0.125); // half way
  assert.ok(Math.abs(first.__media.volume - 0.5) < 0.01 && Math.abs(second.__media.volume - 0.5) < 0.01, 'the sound ramps');
  await m.click('#cutPlayBtn'); // Stop
  assert.equal(first.__media.paused, true);
  assert.equal(first.style.opacity, '');
});

test('Play across a join set to Cut stays a hard cut under a Fade default', async (t) => {
  const n = await saving(t, ['h001', 'h002']);
  n.t = t;
  await n.click('#cutTransition button[data-value=fade]');
  await n.click(n.joins()[0]); // its own Cut
  await playAcross(n);
  const one = n.active();
  await advance(n, one, 17.8);
  assert.match(n.live(), /Playing clip 1 of 2/, 'a Cut waits for the end of the clip');
  await advance(n, one, 17.97);
  assert.match(n.live(), /Playing clip 2 of 2/);
  // one file: the same player seeks on to clip 2, so nothing overlaps (#1020)
  assert.equal(n.active(), one);
  assert.equal(one.__media.currentTime, 100);
  assert.ok([...n.d.querySelectorAll('.screen video')].every((e) => e === one || e.__media.paused), 'no overlap');
});

test('Assemble sends Fade length under a Cut default only while a join of the sequence is set to Fade', async (t) => {
  const jobs = [];
  const p = withSegments('h001', 'h002');
  p.clips = { dir: 'clips', clips: [{ file: '001_a.mp4' }, { file: '002_b.mp4' }] };
  const m = await inCut(t, { project: p, routes: {
    [PUT_REVIEW]: {},
    'POST /api/projects/p1/jobs': (req) => { jobs.push(req.body); return { id: `j${jobs.length}`, status: 'running' }; },
  } });
  const fadeLen = m.$('#optFadeDur');
  fadeLen.value = '0.5';
  fadeLen.dispatchEvent(new m.w.Event('change', { bubbles: true }));
  assert.equal(fadeLen.disabled, true, 'nothing fades: the CLI would refuse the length');
  await m.click(m.joins()[0]);
  await m.click(m.joins()[0]); // Fade
  assert.equal(fadeLen.disabled, false);
  await m.click('#assembleBtn');
  assert.equal(jobs.length, 1);
  assert.equal(jobs[0].transition, 'cut');
  assert.equal(jobs[0].fade_dur, 0.5);
  await m.click(m.joins()[0]); // Default
  assert.equal(fadeLen.disabled, true);
  await m.click('#assembleBtn');
  assert.equal(jobs.at(-1).transition, 'cut');
  assert.equal(jobs.at(-1).fade_dur, undefined);
});

test('Assemble saves the sequence with its joins first, so the server exports the same ones', async (t) => {
  const order = [];
  const p = withSegments('h001', 'h002');
  p.clips = { dir: 'clips', clips: [{ file: '001_a.mp4' }, { file: '002_b.mp4' }] };
  const m = await inCut(t, { project: p, routes: {
    [PUT_REVIEW]: (req) => { order.push(['save', req.body.joins]); return {}; },
    'POST /api/projects/p1/jobs': () => { order.push('job'); return { id: 'j1', status: 'running' }; },
  } });
  await m.click(m.joins()[0]); // Cut
  await m.click('#assembleBtn');
  assert.deepEqual(order, [['save', [pair('h001', 'h002', 'cut')]], 'job']);
});

test('Assembly audio fade follows the joins: on under a Cut default, off while every edge crossfades, on with a Cut join, off after Undo', async (t) => {
  const jobs = [];
  const p = withSegments('h001', 'h002');
  p.clips = { dir: 'clips', clips: [{ file: '001_a.mp4' }, { file: '002_b.mp4' }] };
  const m = await inCut(t, { project: p, routes: {
    [PUT_REVIEW]: {},
    'POST /api/projects/p1/jobs': (req) => { jobs.push(req.body); return { id: `j${jobs.length}`, status: 'running' }; },
  } });
  const field = m.$('#optFinalAudioFade');
  const steps = [
    ['global Cut', async () => {}, false],
    ['global Fade, Crossfade audio, every join Default', () => m.click('#cutTransition button[data-value=fade]'), true],
    ['one join set to Cut', () => m.click(m.joins()[0]), false],
    ['Undo', () => m.click('#undoBtn'), true],
  ];
  assert.equal(m.$('#optAudioTransition').value, 'cut');
  for (const [name, act, off] of steps) {
    await act();
    if (name === 'global Fade, Crossfade audio, every join Default') assert.equal(m.$('#optAudioTransition').value, 'crossfade');
    assert.equal(field.disabled, off, name);
    if (name === 'global Cut') {
      field.value = '0.1';
      field.dispatchEvent(new m.w.Event('change', { bubbles: true }));
    }
  }
  // the Cut join under a Fade default sends it
  await m.click(m.joins()[0]); // Cut
  assert.equal(field.disabled, false);
  await m.click('#assembleBtn');
  assert.equal(jobs.at(-1).transition, 'fade');
  assert.equal(jobs.at(-1).audio_transition, 'crossfade');
  assert.equal(jobs.at(-1).audio_fade, 0.1);
  await m.click('#undoBtn');
  await m.click('#assembleBtn');
  assert.equal(jobs.at(-1).audio_fade, undefined);
});
