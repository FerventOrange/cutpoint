// The review workspace: the moments list (search, Add/Remove), the sequence (reorder, rename,
// select, remove, undo/redo, save), the trim handles and "Play sequence".
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject, reply } from './harness.mjs';

async function review(t, { project = makeProject(), routes = {} } = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project, routes });
  t.after(() => page.close());
  const d = page.document;
  const w = page.window;
  const m = {
    page, d, w,
    $: (s) => d.querySelector(s),
    cards: () => [...d.querySelectorAll('#candList li.cand')].map((li) => li.dataset.id),
    seqIds: () => [...d.querySelectorAll('#seqList > li')].map((li) => li.dataset.id),
    seqTitles: () => [...d.querySelectorAll('#seqList .seq-title')].map((e) => e.textContent),
    async click(el) { (typeof el === 'string' ? m.$(el) : el).click(); await page.settle(); },
    card: (id) => m.$(`#candList li[data-id=${id}]`),
    row: (id) => m.$(`#seqList li[data-id=${id}]`),
    async add(id) { await m.click(m.card(id).querySelector('[data-act=toggle]')); },
    async key(target, key, init = {}) {
      target.dispatchEvent(new w.KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init }));
      await page.settle();
    },
    /** Answer the shared confirm dialog (the page's showModal stub only sets ``open``). */
    async answer(value) {
      const dlg = m.$('#confirmDialog');
      assert.ok(dlg.hasAttribute('open'), 'a confirmation is asked');
      dlg.returnValue = value;
      dlg.removeAttribute('open');
      dlg.dispatchEvent(new w.Event('close'));
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

// ---- moments list ----

test('search narrows the list by label, transcript or id, case-insensitively, and says when nothing matches', async (t) => {
  const m = await review(t);
  m.$('#filter').value = 'FALL';
  m.$('#filter').dispatchEvent(new m.w.Event('input'));
  assert.deepEqual(m.cards(), ['h002']);
  m.$('#filter').value = 'zzz';
  m.$('#filter').dispatchEvent(new m.w.Event('input'));
  assert.match(m.$('#candList').textContent, /No moments match this filter/);
});

test('Add puts a moment last in the sequence, Remove takes it out and Add back returns it to its place', async (t) => {
  const m = await review(t, { project: withSegments('h001', 'h003') });
  assert.deepEqual(m.seqIds(), ['h001', 'h003']);
  await m.add('h002');
  assert.deepEqual(m.seqIds(), ['h001', 'h003', 'h002']);
  assert.match(m.card('h002').querySelector('[data-act=toggle]').textContent, /Remove/);

  await m.click(m.card('h001').querySelector('[data-act=toggle]')); // Remove the first
  assert.deepEqual(m.seqIds(), ['h003', 'h002']);
  await m.add('h001');
  assert.deepEqual(m.seqIds(), ['h001', 'h003', 'h002'], 'back at the index it had');
});

test('Undo and Redo walk the edits (buttons and Ctrl+Z / Ctrl+Y); a new edit clears Redo', async (t) => {
  const m = await review(t);
  await m.add('h001');
  await m.add('h002');
  assert.equal(m.$('#undoBtn').disabled, false);
  assert.equal(m.$('#redoBtn').disabled, true);

  await m.key(m.d.body, 'z', { ctrlKey: true });
  assert.deepEqual(m.seqIds(), ['h001']);
  await m.key(m.d.body, 'z', { ctrlKey: true, shiftKey: true });
  assert.deepEqual(m.seqIds(), ['h001', 'h002'], 'Shift+Ctrl+Z redoes');
  await m.click('#undoBtn');
  await m.click('#undoBtn');
  assert.deepEqual(m.seqIds(), []);
  await m.key(m.d.body, 'y', { ctrlKey: true });
  assert.deepEqual(m.seqIds(), ['h001']);

  await m.add('h003'); // a new edit
  assert.equal(m.$('#redoBtn').disabled, true);
  assert.deepEqual(m.seqIds(), ['h001', 'h003']);
});

test('Ctrl+Z inside a text field keeps the field\'s own undo', async (t) => {
  const m = await review(t);
  await m.add('h001');
  await m.key(m.$('#filter'), 'z', { ctrlKey: true });
  assert.deepEqual(m.seqIds(), ['h001']);
});

test('Save sends the sequence in order; the button and state follow unsaved changes', async (t) => {
  let saved = null;
  const m = await review(t, { routes: { 'PUT /api/projects/p1/review': (req) => { saved = req.body; return {}; } } });
  assert.equal(m.$('#saveBtn').disabled, true);
  await m.add('h002');
  await m.add('h001');
  await m.click(m.row('h001').querySelector('[data-act=up]'));
  assert.equal(m.$('#saveState').textContent, 'Unsaved changes');
  await m.click('#saveBtn');
  assert.deepEqual(saved.segments.map((s) => s.id), ['h001', 'h002']);
  assert.deepEqual(Object.keys(saved.segments[0]).sort(), ['end', 'id', 'label', 'source', 'start']);
  assert.equal(m.$('#saveState').textContent, 'Sequence saved');
  assert.equal(m.$('#saveBtn').disabled, true);
});

test('a failed save keeps the changes unsaved and shows the error', async (t) => {
  const m = await review(t, { routes: { 'PUT /api/projects/p1/review': reply(500, { detail: 'disk full' }) } });
  await m.add('h001');
  await m.click('#saveBtn');
  assert.equal(m.$('#saveState').textContent, 'Unsaved changes');
  assert.equal(m.$('#banner').textContent, 'disk full');
  assert.equal(m.$('#saveBtn').disabled, false);
});

// ---- sequence rows ----

test('reordering with the chevrons moves a clip, and the end rows cannot move off the list', async (t) => {
  const m = await review(t, { project: withSegments('h001', 'h002', 'h003') });
  assert.equal(m.row('h001').querySelector('[data-act=up]').disabled, true);
  assert.equal(m.row('h003').querySelector('[data-act=down]').disabled, true);
  await m.click(m.row('h003').querySelector('[data-act=up]'));
  assert.deepEqual(m.seqIds(), ['h001', 'h003', 'h002']);
  await m.click('#undoBtn');
  assert.deepEqual(m.seqIds(), ['h001', 'h002', 'h003']);
});

test('rename: Enter saves (one undo step), Escape and an empty name keep the old one, an invalid name is refused', async (t) => {
  const m = await review(t, { project: withSegments('h001', 'h002') });
  const startRename = async (id) => {
    await m.click(m.row(id).querySelector('[data-act=rename]'));
    return m.row(id).querySelector('input[data-act=rename]');
  };
  const type = (input, text) => { input.value = text; input.dispatchEvent(new m.w.Event('input')); };

  let input = await startRename('h001');
  type(input, '  Best bit  ');
  await m.key(input, 'Enter');
  assert.deepEqual(m.seqTitles(), ['Best bit', 'Fall']);
  assert.equal(m.$('#saveState').textContent, 'Unsaved changes');

  input = await startRename('h001');
  type(input, 'Nope');
  await m.key(input, 'Escape');
  assert.deepEqual(m.seqTitles(), ['Best bit', 'Fall']);

  input = await startRename('h001');
  type(input, '   ');
  await m.key(input, 'Enter');
  assert.deepEqual(m.seqTitles(), ['Best bit', 'Fall'], 'empty keeps the old name');

  input = await startRename('h002');
  type(input, 'bad\u0007name');
  await m.key(input, 'Enter');
  assert.match(m.row('h002').querySelector('.seq-rename-err').textContent, /letters, digits, punctuation/);
  assert.ok(m.row('h002').querySelector('input[data-act=rename]'), 'still editing');
  type(input, 'x'.repeat(201));
  await m.key(input, 'Enter');
  assert.match(m.row('h002').querySelector('.seq-rename-err').textContent, /at most 200 characters/);

  await m.key(input, 'Escape');
  await m.click('#undoBtn');
  assert.deepEqual(m.seqTitles(), ['Big laugh', 'Fall'], 'the rename was one undo step');
});

test('leaving the field commits a rename', async (t) => {
  const m = await review(t, { project: withSegments('h001') });
  await m.click(m.row('h001').querySelector('[data-act=rename]'));
  const input = m.row('h001').querySelector('input[data-act=rename]');
  input.value = 'Renamed';
  input.dispatchEvent(new m.w.Event('input'));
  input.dispatchEvent(new m.w.Event('blur'));
  await m.page.settle();
  assert.deepEqual(m.seqTitles(), ['Renamed']);
});

test('selected clips are removed together after a confirmation; declining removes nothing', async (t) => {
  const m = await review(t, { project: withSegments('h001', 'h002', 'h003') });
  const tick = async (id) => {
    const box = m.row(id).querySelector('[data-act=select]');
    box.checked = true;
    box.dispatchEvent(new m.w.Event('change'));
    await m.page.settle();
  };
  assert.equal(m.$('#seqBatch').hidden, true);
  await tick('h001');
  await tick('h003');
  assert.equal(m.$('#seqSelCount').textContent, '2 selected');

  await m.click('#seqRemoveSel');
  await m.answer('cancel');
  assert.deepEqual(m.seqIds(), ['h001', 'h002', 'h003']);

  await m.click('#seqRemoveSel');
  await m.answer('ok');
  assert.deepEqual(m.seqIds(), ['h002']);
  assert.equal(m.$('#seqBatch').hidden, true);
  await m.click('#undoBtn');
  assert.deepEqual(m.seqIds(), ['h001', 'h002', 'h003']);
});

test('removing one clip asks first and leaves the moment in the list to add back', async (t) => {
  const m = await review(t, { project: withSegments('h001', 'h002') });
  await m.click(m.row('h001').querySelector('[data-act=remove]'));
  await m.answer('ok');
  assert.deepEqual(m.seqIds(), ['h002']);
  assert.deepEqual(m.cards(), ['h001', 'h002', 'h003']);
});

// ---- trim ----

test('trim handles: arrows nudge 0.1 s, Shift 1 s, End/Home stop at the limits, a run of nudges is one undo step', async (t) => {
  const m = await review(t, { project: withSegments('h001') }); // 10 to 18
  const start = m.$('#startHandle');
  const end = m.$('#endHandle');
  const range = () => m.row('h001').querySelector('.seq-main').title;
  assert.equal(range(), '0:10.00 to 0:18.00');

  await m.key(start, 'ArrowRight');
  await m.key(start, 'ArrowRight');
  await m.key(start, 'ArrowRight', { shiftKey: true });
  assert.equal(range(), '0:11.20 to 0:18.00');
  await m.click('#undoBtn');
  assert.equal(range(), '0:10.00 to 0:18.00', 'quick nudges of one point are one step');

  await m.key(start, 'End'); // the latest start: 0.1 s before the end
  assert.equal(range(), '0:17.90 to 0:18.00');
  await m.key(start, 'Home');
  assert.equal(range(), '0:00.00 to 0:18.00');
  await m.key(end, 'End'); // the end of the 600 s source
  assert.equal(range(), '0:00.00 to 10:00.00');
  await m.key(end, 'ArrowRight');
  assert.equal(range(), '0:00.00 to 10:00.00');
});

test('trim times typed in the fields are applied, or refused with the reason and the old time kept', async (t) => {
  const m = await review(t, { project: withSegments('h001') }); // 10 to 18 of a 600 s source
  const range = () => m.row('h001').querySelector('.seq-main').title;
  const type = async (which, text) => {
    const input = m.$(`#t-${which}`);
    input.value = text;
    input.dispatchEvent(new m.w.Event('change'));
    await m.page.settle();
  };
  await type('start', '0:12.5');
  assert.equal(range(), '0:12.50 to 0:18.00');
  assert.equal(m.$('#trimMsg').textContent.startsWith('Clip length 5.50 s'), true);

  const refused = [
    ['end', '12.55', /at least 0\.1 s/],
    ['end', '700', /inside the source/],
    ['start', 'abc', /Enter a time like/],
  ];
  for (const [which, text, why] of refused) {
    await type(which, text);
    assert.match(m.$('#trimMsg').textContent, why);
    assert.equal(range(), '0:12.50 to 0:18.00');
    assert.equal(m.$(`#t-${which}`).value, which === 'start' ? '0:12.50' : '0:18.00', 'the field shows the kept time again');
  }
});

test('dragging a trim handle moves the point with the pointer and ends as one undo step', async (t) => {
  const m = await review(t, { project: withSegments('h001') }); // 10 to 18
  const rail = m.$('#rail');
  // jsdom has no layout: give the rail a 1000 px width at x = 0
  Object.defineProperty(rail, 'clientWidth', { value: 1000, configurable: true });
  rail.getBoundingClientRect = () => ({ left: 0, right: 1000, top: 0, bottom: 20, width: 1000, height: 20 });
  m.w.PointerEvent ??= m.w.MouseEvent;
  const handle = m.$('#endHandle');
  const range = () => m.row('h001').querySelector('.seq-main').title;
  const ev = (type, x) => handle.dispatchEvent(Object.assign(new m.w.MouseEvent(type, { clientX: x, button: 0, bubbles: true, cancelable: true }), { pointerId: 1, pointerType: 'mouse' }));

  const secs = (text) => text.split(':').reduce((a, p) => a * 60 + Number(p), 0);
  const lo = secs(m.$('#winLo').textContent); // the rail's view, from its labels
  const hi = secs(m.$('#winHi').textContent);
  const px = (sec) => ((sec - lo) / (hi - lo)) * 1000;

  ev('pointerdown', px(18));
  ev('pointermove', px(20));
  t.mock.timers.tick(16);
  await m.page.settle();
  ev('pointerup', px(20));
  ev('lostpointercapture', px(20));
  await m.page.settle();
  assert.equal(range(), '0:10.00 to 0:20.00');
  await m.click('#undoBtn');
  assert.equal(range(), '0:10.00 to 0:18.00');
});

// ---- Play sequence ----

test('Play sequence runs the clips in order and stops after the last', async (t) => {
  const m = await review(t, { project: withSegments('h002', 'h001') }); // 100-108 then 10-18
  const live = () => m.$('#seqLive').textContent;
  const playBtn = m.$('#seqPlayBtn');
  assert.equal(playBtn.disabled, false);

  await m.click(playBtn);
  assert.match(live(), /Playing clip 1 of 2: Fall/);
  assert.equal(playBtn.textContent, 'Stop');
  assert.ok(m.row('h002').classList.contains('playing'));

  const v = m.$('#video');
  v.__media.loaded(600);
  await m.page.settle();
  assert.equal(v.__media.currentTime, 100);

  v.__media.tick(108);
  t.mock.timers.tick(16);
  await m.page.settle();
  assert.match(live(), /Playing clip 2 of 2: Big laugh/);
  assert.ok(m.row('h001').classList.contains('playing'));
  assert.ok(!m.row('h002').classList.contains('playing'));

  // the second clip plays in whichever player now has the stage
  const active = [m.$('#video'), m.$('#video2')].find((e) => e.classList.contains('active')) ?? m.$('#video');
  active.__media.loaded?.(600);
  await m.page.settle();
  active.__media.tick(18);
  t.mock.timers.tick(16);
  await m.page.settle();
  assert.equal(live(), 'Sequence finished');
  assert.equal(playBtn.textContent, 'Play sequence');
});

test('Stop ends the sequence early and pauses the player', async (t) => {
  const m = await review(t, { project: withSegments('h002', 'h001') });
  const playBtn = m.$('#seqPlayBtn');
  await m.click(playBtn);
  assert.equal(playBtn.textContent, 'Stop');
  assert.ok(m.row('h002').classList.contains('playing'));
  const v = m.$('#video');
  v.__media.loaded(600);
  await m.page.settle();

  await m.click(playBtn);
  assert.equal(playBtn.textContent, 'Play sequence');
  assert.equal(m.$('#seqList').querySelectorAll('.playing').length, 0);
  assert.equal(v.__media.paused, true);
});

test('Play sequence is unavailable with an empty sequence', async (t) => {
  const m = await review(t);
  assert.equal(m.$('#seqPlayBtn').disabled, true);
  await m.add('h001');
  assert.equal(m.$('#seqPlayBtn').disabled, false);
});
