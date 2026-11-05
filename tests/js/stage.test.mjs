// The center stage: the trim rail (handles, I/O keys, zoom levels, ruler, other moments), the
// storyboard under it and the Clip/Sequence mode switch.
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
    blocks: () => [...d.querySelectorAll('#sbList .sb-block')],
    block: (id) => d.querySelector(`#sbList .sb-block[data-id=${id}]`),
    cell: (id) => d.querySelector(`#sbList .sb-block[data-id=${id}]`).closest('.sb-cell'),
    seqIds: () => [...d.querySelectorAll('#seqList > li')].map((li) => li.dataset.id),
    range: (id) => d.querySelector(`#seqList li[data-id=${id}] .seq-main`).title,
    async click(el) { (typeof el === 'string' ? m.$(el) : el).click(); await page.settle(); },
    async key(target, key, init = {}) {
      target.dispatchEvent(new w.KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init }));
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
/** Move the playhead of the open clip to source time ``at``. */
const playheadAt = async (m, at) => {
  const v = m.$('#video');
  v.__media.loaded(600);
  v.__media.tick(at);
  v.dispatchEvent(new m.w.Event('seeked'));
  await m.page.settle();
};

// ---- trim ----

test('the trim handles are sliders with a time as their value text; arrows move the point and Save keeps it', async (t) => {
  let saved = null;
  const m = await stage(t, { project: withSegments('h001'), routes: { 'PUT /api/projects/p1/review': (req) => { saved = req.body; return {}; } } });
  const start = m.$('#startHandle');
  assert.equal(start.getAttribute('role'), 'slider');
  assert.equal(start.getAttribute('aria-valuetext'), 'Clip start, 0:10.00');
  assert.equal(m.$('#endHandle').getAttribute('aria-valuetext'), 'Clip end, 0:18.00');
  await m.key(start, 'ArrowRight');
  assert.equal(start.getAttribute('aria-valuetext'), 'Clip start, 0:10.10');
  await m.key(m.$('#endHandle'), 'ArrowLeft', { shiftKey: true });
  assert.equal(m.range('h001'), '0:10.10 to 0:17.00');
  await m.click('#saveBtn');
  assert.deepEqual([saved.segments[0].start, saved.segments[0].end], [10.1, 17]);
});

test('I and O set the trim in and out at the playhead; Shift+I and a text field do not', async (t) => {
  const m = await stage(t, { project: withSegments('h001') }); // 10 to 18
  await playheadAt(m, 12.5);
  await m.key(m.d.body, 'i');
  assert.equal(m.range('h001'), '0:12.50 to 0:18.00');
  await playheadAt(m, 16);
  await m.key(m.d.body, 'O');
  assert.equal(m.range('h001'), '0:12.50 to 0:16.00');
  await m.key(m.d.body, 'I', { shiftKey: true });
  await m.key(m.$('#filter'), 'i');
  assert.equal(m.range('h001'), '0:12.50 to 0:16.00');
  await m.click('#undoBtn');
  assert.equal(m.range('h001'), '0:12.50 to 0:18.00', 'an O is one undo step');
});

test('O at the very end of a source lands on its last time instead of rounding past it', async (t) => {
  const p = withSegments('h001'); // 10 to 18
  p.sources[0].duration = 600.126;
  for (const s of p.highlights.sources ?? []) s.duration = 600.126;
  const m = await stage(t, { project: p });
  await playheadAt(m, 600.126);
  await m.key(m.d.body, 'o');
  assert.doesNotMatch(m.$('#trimMsg').textContent, /inside the source/);
  assert.equal(m.$('#endHandle').getAttribute('aria-valuenow'), '600.12');
});

test('the page shortcuts do nothing while the filter popover is open, also with focus on its button', async (t) => {
  const m = await stage(t, { project: withSegments('h001') }); // 10 to 18
  await playheadAt(m, 12.5);
  await m.click('#filterBtn');
  m.$('#filterBtn').focus();
  for (const key of ['i', 'o', '[', ']', 'm']) await m.key(m.$('#filterBtn'), key);
  assert.equal(m.range('h001'), '0:10.00 to 0:18.00');
  assert.equal(m.$('#video').muted, false);
  await m.key(m.$('#filterBtn'), 'Escape');
  assert.ok(m.$('#filterPop').hidden);
  await m.key(m.$('#filterBtn'), 'i');
  assert.equal(m.range('h001'), '0:12.50 to 0:18.00', 'closed, the keys work again');
});

test('the zoom levels set the width of the view and show the pressed one', async (t) => {
  const m = await stage(t, { project: withSegments('h001') });
  const presets = () => [...m.d.querySelectorAll('#zoomPresets button')];
  const pressed = () => presets().filter((b) => b.getAttribute('aria-pressed') === 'true').map((b) => b.textContent);
  const secs = (s) => s.split(':').reduce((n, x) => n * 60 + Number(x), 0);
  const win = () => secs(m.$('#winHi').textContent) - secs(m.$('#winLo').textContent);
  assert.deepEqual(pressed(), [], 'the fit view is none of the levels');
  for (const [label, want] of [['30 s', 30], ['10 s', 10], ['2 min', 120]]) {
    await m.click(presets().find((b) => b.textContent === label));
    assert.ok(Math.abs(win() - want) < 0.05, `${label}: ${win()}`);
    assert.deepEqual(pressed(), [label]);
  }
  await m.click(presets().find((b) => b.textContent === '2 min')); // pressed again: back to the fit view
  assert.deepEqual(pressed(), []);
});

test('the ruler labels the view and the other moments of the source show under the rail', async (t) => {
  const m = await stage(t, { project: withSegments('h001') });
  await m.click(m.$('#zoomPresets [data-span="120"]'));
  assert.ok(m.$('#railRuler').children.length >= 2);
  assert.match(m.$('#railRuler .rail-tick').textContent, /^\d+:\d\d$/);
  const names = [...m.d.querySelectorAll('#railEvents .rail-event')].map((e) => e.textContent);
  assert.ok(names.includes('Fall'), 'the moment at 100 s starts inside the 2 min view');
  assert.ok(!names.includes('Big laugh'), 'not the open clip itself');
});

// ---- storyboard ----

test('the storyboard has one block per clip, sized by length, and follows a trim', async (t) => {
  const m = await stage(t, { project: withSegments('h001', 'h002') });
  assert.equal(m.$('#storyboard').hidden, false);
  assert.deepEqual(m.blocks().map((b) => b.dataset.id), ['h001', 'h002']);
  const grow = (id) => parseFloat(m.cell(id).style.flexGrow);
  assert.equal(grow('h001'), 8);
  await m.key(m.$('#endHandle'), 'PageDown');
  assert.equal(grow('h001'), 3);
  await m.click('#undoBtn');
  assert.equal(grow('h001'), 8);
});

test('the storyboard is hidden while the sequence is empty', async (t) => {
  const m = await stage(t);
  assert.equal(m.$('#storyboard').hidden, true);
});

test('clicking a block opens that clip for trimming', async (t) => {
  const m = await stage(t, { project: withSegments('h001', 'h002') });
  await m.click(m.block('h002'));
  assert.equal(m.block('h002').getAttribute('aria-current'), 'true');
  assert.equal(m.$('#startHandle').getAttribute('aria-valuetext'), 'Clip start, 1:40.00');
});

test('Alt+Arrow moves a block through the sequence reorder and keeps focus on it', async (t) => {
  const m = await stage(t, { project: withSegments('h001', 'h002', 'h003') });
  m.block('h001').focus();
  await m.key(m.block('h001'), 'ArrowRight', { altKey: true });
  assert.deepEqual(m.seqIds(), ['h002', 'h001', 'h003'], 'the sequence panel follows');
  assert.deepEqual(m.blocks().map((b) => b.dataset.id), ['h002', 'h001', 'h003']);
  assert.equal(m.d.activeElement, m.block('h001'));
  assert.match(m.$('#seqLive').textContent, /Big laugh moved to position 2 of 3/);
  await m.click('#undoBtn');
  assert.deepEqual(m.seqIds(), ['h001', 'h002', 'h003'], 'one undo step');
  await m.key(m.block('h001'), 'ArrowLeft', { altKey: true });
  assert.deepEqual(m.seqIds(), ['h001', 'h002', 'h003'], 'the first stays first');
});

test('plain arrows move between the blocks without reordering', async (t) => {
  const m = await stage(t, { project: withSegments('h001', 'h002', 'h003') });
  m.block('h001').focus();
  await m.key(m.block('h001'), 'ArrowRight');
  assert.equal(m.d.activeElement, m.block('h002'));
  assert.equal(m.block('h002').tabIndex, 0);
  assert.equal(m.block('h001').tabIndex, -1);
  await m.key(m.block('h002'), 'End');
  assert.equal(m.d.activeElement, m.block('h003'));
  assert.deepEqual(m.seqIds(), ['h001', 'h002', 'h003']);
});

test('dropping a block on another reorders the sequence', async (t) => {
  const m = await stage(t, { project: withSegments('h001', 'h002', 'h003') });
  const drag = (el, type) => {
    const e = new m.w.Event(type, { bubbles: true, cancelable: true });
    e.dataTransfer = { setData() {}, effectAllowed: '' };
    el.dispatchEvent(e);
  };
  drag(m.cell('h003'), 'dragstart');
  drag(m.cell('h001'), 'dragover');
  assert.ok(m.cell('h001').classList.contains('drop'));
  drag(m.cell('h001'), 'drop');
  assert.deepEqual(m.seqIds(), ['h003', 'h001', 'h002']);
});

test('double-clicking a block plays the sequence from that clip; Shift+Enter does too', async (t) => {
  const m = await stage(t, { project: withSegments('h001', 'h002', 'h003') });
  m.block('h002').dispatchEvent(new m.w.MouseEvent('dblclick', { bubbles: true }));
  await m.page.settle();
  assert.match(m.$('#seqLive').textContent, /Playing clip 2 of 3: Fall/);
  assert.ok(m.block('h002').classList.contains('playing'));
  await m.click('#seqPlayBtn'); // Stop
  assert.equal(m.$('#sbList .playing'), null);
  await m.key(m.block('h003'), 'Enter', { shiftKey: true });
  assert.match(m.$('#seqLive').textContent, /Playing clip 3 of 3: Intro joke/);
});

test('the storyboard keeps exactly one tab stop: the clicked block, or the first when the moment is not in the sequence', async (t) => {
  const m = await stage(t, { project: withSegments('h001', 'h002') });
  const stops = () => m.blocks().filter((b) => b.tabIndex === 0).map((b) => b.dataset.id);
  await m.click(m.block('h002'));
  assert.deepEqual(stops(), ['h002']);
  assert.equal(m.block('h001').tabIndex, -1);
  await m.click(m.$('#candList li[data-id=h003] [data-act=focus]')); // not in the sequence
  assert.deepEqual(stops(), ['h001']);
});

// ---- mode switch ----

test('the stage mode switch: Clip by default, Sequence plays the sequence and Clip stops it', async (t) => {
  const m = await stage(t, { project: withSegments('h001', 'h002') });
  const [clip, seq] = [m.$('#stageMode [data-mode=clip]'), m.$('#stageMode [data-mode=sequence]')];
  assert.equal(m.d.querySelectorAll('#stageMode button').length, 2, 'no Cut mode yet');
  assert.equal(clip.getAttribute('aria-pressed'), 'true');
  assert.equal(seq.getAttribute('aria-pressed'), 'false');
  await m.click(seq);
  assert.match(m.$('#seqLive').textContent, /Playing clip 1 of 2: Big laugh/);
  assert.equal(seq.getAttribute('aria-pressed'), 'true');
  assert.equal(clip.getAttribute('aria-pressed'), 'false');
  await m.click(clip);
  assert.equal(m.$('#seqPlayBtn').textContent, 'Play sequence');
  assert.equal(clip.getAttribute('aria-pressed'), 'true');
  assert.equal(seq.getAttribute('aria-pressed'), 'false');
  await m.click('#seqPlayBtn'); // the Play sequence button moves the switch too
  assert.equal(seq.getAttribute('aria-pressed'), 'true');
});

test('the Sequence mode is unavailable with an empty sequence', async (t) => {
  const m = await stage(t);
  assert.equal(m.$('#stageMode [data-mode=sequence]').disabled, true);
  await m.click(m.$('#candList li[data-id=h001] [data-act=toggle]'));
  assert.equal(m.$('#stageMode [data-mode=sequence]').disabled, false);
});

// ---- title row ----

test('the title row names a sequence clip by its place and a moment outside the sequence by its name', async (t) => {
  const m = await stage(t, { project: withSegments('h001', 'h002') });
  assert.equal(m.$('#stageTitle').textContent, 'Clip 1 · Big laugh');
  await m.click(m.block('h002'));
  assert.equal(m.$('#stageTitle').textContent, 'Clip 2 · Fall');
  assert.equal(m.$('#toggleBtn').textContent, 'Remove from sequence');
  await m.click(m.$('#candList li[data-id=h003] [data-act=focus]'));
  assert.equal(m.$('#stageTitle').textContent, 'Intro joke');
  assert.equal(m.$('#toggleBtn').textContent, 'Add to sequence');
});

test('previous and next clip buttons step through the sequence, are disabled at the ends and hidden off it', async (t) => {
  const m = await stage(t, { project: withSegments('h001', 'h002', 'h003') });
  const [prev, next] = [m.$('#prevClip'), m.$('#nextClip')];
  const title = () => m.$('#stageTitle').textContent;
  assert.equal(m.$('#clipNav').hidden, false);
  assert.equal(prev.disabled, true, 'nothing before the first clip');
  assert.equal(prev.getAttribute('aria-label'), 'Previous clip');
  assert.equal(next.getAttribute('aria-label'), 'Next clip, Clip 2');
  assert.equal(next.textContent, 'Clip 2 ›');
  await m.click(next);
  assert.equal(title(), 'Clip 2 · Fall');
  assert.equal(prev.getAttribute('aria-label'), 'Previous clip, Clip 1');
  assert.equal(next.getAttribute('aria-label'), 'Next clip, Clip 3');
  await m.click(next);
  assert.equal(title(), 'Clip 3 · Intro joke');
  assert.equal(next.disabled, true, 'nothing after the last clip');
  assert.equal(next.getAttribute('aria-label'), 'Next clip');
  await m.click(prev);
  assert.equal(title(), 'Clip 2 · Fall');
  await m.click(m.$('#candList li[data-id=h001] [data-act=toggle]')); // h001 leaves the sequence
  await m.click(m.$('#candList li[data-id=h001] [data-act=focus]'));
  assert.equal(m.$('#clipNav').hidden, true, 'a moment outside the sequence has no neighbours');
});

// ---- clock, cut line ----

test('the clock and the IN badge count from the clip, not from the source', async (t) => {
  const m = await stage(t, { project: withSegments('h001') }); // 10 to 18
  await playheadAt(m, 12.5);
  assert.equal(m.$('#clock').textContent, '0:02.5 / 0:08.0');
  assert.equal(m.$('#inBadge').textContent, 'IN +2.5 s');
  await playheadAt(m, 9);
  assert.equal(m.$('#inBadge').textContent, 'IN -1.0 s', 'before the in point');
});

test('the cut line shows the trim, its length and what the export makes of it', async (t) => {
  const m = await stage(t, { project: withSegments('h001') });
  const text = (id) => m.$(id).textContent;
  assert.equal(text('#cutRange'), '0:10.0 → 0:18.0');
  assert.equal(text('#cutLen'), '| 8.00 s');
  assert.equal(text('#cutOut'), '8.00 s');
  assert.equal(m.$('#cutTail').hidden, true, 'no tail until the export plans one');
  await m.key(m.$('#endHandle'), 'ArrowLeft', { shiftKey: true });
  assert.equal(text('#cutRange'), '0:10.0 → 0:17.0');
  assert.equal(text('#cutLen'), '| 7.00 s');
});

test('the cut line shows the planned tail as a pill, with its reason, and the length it exports as', async (t) => {
  const m = await stage(t, {
    project: withSegments('h001'),
    routes: { 'POST /api/projects/p1/edges': () => ({ edges: { h001: { start: 10, end: 20.5, reason: 'event tail' } } }) },
  });
  assert.equal(m.$('#cutTail').hidden, true);
  t.mock.timers.tick(400);
  await m.page.settle();
  assert.equal(m.$('#cutTail').hidden, false);
  assert.equal(m.$('#cutTail').textContent, '+2.50 s event tail');
  assert.equal(m.$('#cutLen').textContent, '| 8.00 s', 'the length stays the trim');
  assert.equal(m.$('#cutOut').textContent, '10.50 s');
});

// ---- Reset to moment ----

test('Reset to moment is off while the trim is the analyzed one, restores it otherwise and is one undo step', async (t) => {
  const m = await stage(t, { project: withSegments('h001') });
  const reset = m.$('#resetBtn');
  assert.equal(reset.disabled, true);
  await m.key(m.$('#startHandle'), 'ArrowRight', { shiftKey: true });
  await m.key(m.$('#endHandle'), 'ArrowLeft', { shiftKey: true });
  assert.equal(m.range('h001'), '0:11.00 to 0:17.00');
  assert.equal(reset.disabled, false);
  await m.click(reset);
  assert.equal(m.range('h001'), '0:10.00 to 0:18.00');
  assert.equal(reset.disabled, true);
  assert.equal(m.$('#cutRange').textContent, '0:10.0 → 0:18.0');
  await m.click('#undoBtn');
  assert.equal(m.range('h001'), '0:11.00 to 0:17.00', 'undo brings the trim back');
});

// ---- I and O refused ----

test('I and O that would make the clip invalid are refused with a visible message and change nothing', async (t) => {
  const m = await stage(t, { project: withSegments('h001') }); // 10 to 18
  const msg = m.$('#trimMsg');
  assert.equal(msg.classList.contains('vh'), true, 'the message is for screen readers while all is well');
  await playheadAt(m, 9);
  await m.key(m.d.body, 'o'); // an out before the in
  assert.equal(m.range('h001'), '0:10.00 to 0:18.00');
  assert.match(msg.textContent, /at least .* s long/);
  assert.equal(msg.classList.contains('error'), true);
  assert.equal(msg.classList.contains('vh'), false, 'and it shows on screen');
  await playheadAt(m, 590);
  await m.key(m.d.body, 'i'); // an in after the out
  assert.equal(m.range('h001'), '0:10.00 to 0:18.00');
  assert.match(msg.textContent, /at least .* s long/);
});

// ---- storyboard follows playback ----

test('the storyboard ruler runs from 0:00 to the sequence length and its playhead follows the clip playhead', async (t) => {
  const m = await stage(t, { project: withSegments('h001', 'h002') }); // 8 s + 8 s
  const labels = [...m.d.querySelectorAll('#sbRuler span')].map((s) => s.textContent);
  assert.equal(m.$('#sbRuler').hidden, false);
  assert.equal(labels[0], '0:00');
  assert.equal(labels.at(-1), '0:16', 'the total at the right end');
  assert.equal(m.$('#sbHead').hidden, false);
  // jsdom has no layout: give every block 400 px so the head has a place to be
  const proto = m.w.HTMLElement.prototype;
  Object.defineProperty(proto, 'offsetWidth', { configurable: true, get: () => 400 });
  Object.defineProperty(proto, 'clientWidth', { configurable: true, get: () => 800 });
  await playheadAt(m, 10);
  const at0 = m.$('#sbHead').style.left;
  await playheadAt(m, 14);
  assert.notEqual(m.$('#sbHead').style.left, at0, 'the head moves with the playhead');
  await m.click(m.$('#candList li[data-id=h003] [data-act=focus]'));
  assert.equal(m.$('#sbHead').hidden, true, 'no head for a clip outside the sequence');
});

test('a trim that shortens the sequence changes the end label of the storyboard ruler', async (t) => {
  const m = await stage(t, { project: withSegments('h001', 'h002') });
  await m.key(m.$('#endHandle'), 'PageDown'); // 8 s to 3 s
  const labels = [...m.d.querySelectorAll('#sbRuler span')].map((s) => s.textContent);
  assert.equal(labels.at(-1), '0:11');
});

// ---- zoom ----

test('pressing the pressed zoom level again fits the clip', async (t) => {
  const m = await stage(t, { project: withSegments('h001') });
  const lo = () => m.$('#winLo').textContent;
  const hi = () => m.$('#winHi').textContent;
  const fit = [lo(), hi()];
  const ten = m.$('#zoomPresets [data-span="10"]');
  await m.click(ten);
  assert.equal(ten.getAttribute('aria-pressed'), 'true');
  assert.notDeepEqual([lo(), hi()], fit, 'the level changed the view');
  await m.click(ten);
  assert.equal(ten.getAttribute('aria-pressed'), 'false');
  assert.deepEqual([lo(), hi()], fit, 'back to the fit view');
});
