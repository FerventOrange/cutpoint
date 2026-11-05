// Dragging a clip in the Sequence list and the storyboard (#749): the drop marker shows the side the
// clip lands on, a drag that ended unseen (re-render, window leave, foreign drop) is forgotten, and a
// held clip scrolls the list near its edges.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

const withSegments = (...ids) => {
  const p = makeProject();
  p.review = { segments: ids.map((id) => {
    const c = p.highlights.candidates.find((x) => x.id === id);
    return { id, source: c.source, start: c.start, end: c.end, label: c.label };
  }) };
  return p;
};

async function drag(t, ids = ['h001', 'h002', 'h003']) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project: withSegments(...ids) });
  t.after(() => page.close());
  const d = page.document;
  const w = page.window;
  const m = {
    page, d, w, t,
    $: (s) => d.querySelector(s),
    seqIds: () => [...d.querySelectorAll('#seqList > li')].map((li) => li.dataset.id),
    row: (id) => d.querySelector(`#seqList li[data-id=${id}]`),
    cell: (id) => d.querySelector(`#sbList .sb-block[data-id=${id}]`).closest('.sb-cell'),
    dt: null,
    /** Fire a drag event on ``el``; ``types`` defaults to what the drag of ours carries. */
    fire(el, type, { types = m.dt?.types ?? [], x = 10, y = 10 } = {}) {
      const e = new w.Event(type, { bubbles: true, cancelable: true });
      e.dataTransfer = type === 'dragstart'
        ? (m.dt = { types: [], effectAllowed: '', setData(k) { this.types.push(k); } })
        : { types, effectAllowed: '' };
      e.clientX = x;
      e.clientY = y;
      el.dispatchEvent(e);
      return e;
    },
    marked: (sel = '#seqList > li') => [...d.querySelectorAll(sel)].filter((el) => el.classList.contains('drop')).map((el) => el.dataset.id ?? el.querySelector('[data-id]').dataset.id),
    after: (el) => el.classList.contains('after'),
  };
  return m;
}

test('the Sequence list marks the side a clip lands on: below a target for a move down, above it for a move up', async (t) => {
  const m = await drag(t);
  m.fire(m.row('h001'), 'dragstart');
  m.fire(m.row('h003'), 'dragover');
  assert.deepEqual(m.marked(), ['h003']);
  assert.ok(m.after(m.row('h003')), 'a forward drag lands after the target');
  m.fire(m.row('h001'), 'dragend');
  m.fire(m.row('h003'), 'dragstart');
  m.fire(m.row('h001'), 'dragover');
  assert.deepEqual(m.marked(), ['h001']);
  assert.ok(!m.after(m.row('h001')), 'a backward drag lands before the target');
  m.fire(m.row('h001'), 'drop');
  assert.deepEqual(m.seqIds(), ['h003', 'h001', 'h002']);
  // and the forward marker told the truth
  await m.page.settle();
  m.fire(m.row('h003'), 'dragstart'); // now first; moved down onto the last
  m.fire(m.row('h002'), 'dragover');
  m.fire(m.row('h002'), 'drop');
  assert.deepEqual(m.seqIds(), ['h001', 'h002', 'h003'], 'it landed after the target, as marked');
});

test('the dragged clip has no marker; it lands after the target it was marked after', async (t) => {
  const m = await drag(t);
  m.fire(m.row('h001'), 'dragstart');
  m.fire(m.row('h001'), 'dragover');
  assert.deepEqual(m.marked(), []);
  m.fire(m.row('h002'), 'dragover');
  assert.ok(m.after(m.row('h002')));
  m.fire(m.row('h002'), 'drop');
  assert.deepEqual(m.seqIds(), ['h002', 'h001', 'h003'], 'dropped on h002, it sits right after it');
  assert.equal(m.d.querySelectorAll('.dragging, .drop').length, 0);
});

test('a drop and a dragend end the drag: no stale id, no markers', async (t) => {
  const m = await drag(t);
  m.fire(m.row('h001'), 'dragstart');
  m.fire(m.row('h003'), 'dragover');
  m.fire(m.row('h001'), 'dragend');
  assert.equal(m.d.querySelectorAll('.dragging, .drop').length, 0);
  // the id is gone: a drop with our type but no drag in progress moves nothing
  m.fire(m.row('h003'), 'drop', { types: ['application/x-cutpoint-clip'] });
  assert.deepEqual(m.seqIds(), ['h001', 'h002', 'h003']);
});

test('a background render during a drag waits for its end: the held row survives and its dragend still comes', async (t) => {
  const m = await drag(t);
  const held = m.row('h001');
  m.fire(held, 'dragstart');
  m.$('#seqList li[data-id=h002] button[data-act=rename]').click(); // a render with the same order (the poll, the edges reply)
  await m.page.settle();
  assert.equal(m.row('h001'), held, 'the rows were not replaced');
  assert.equal(m.$('#seqList input[data-act=rename]'), null, 'the render waits');
  m.fire(m.row('h003'), 'dragover');
  assert.deepEqual(m.marked(), ['h003'], 'the drag goes on');
  m.fire(held, 'dragend');
  await m.page.settle();
  assert.ok(m.$('#seqList input[data-act=rename]'), 'the render ran once the drag ended');
  assert.equal(m.d.querySelectorAll('.dragging, .drop').length, 0);
});

test('a render that changes the order during a drag ends it: nothing stale is dropped', async (t) => {
  const m = await drag(t);
  m.fire(m.row('h001'), 'dragstart');
  m.$('#seqList li[data-id=h002] [data-act=down]').click(); // h002 and h003 swap: the rows are new
  await m.page.settle();
  assert.deepEqual(m.seqIds(), ['h001', 'h003', 'h002']);
  const e = m.fire(m.row('h002'), 'dragover', { types: ['application/x-cutpoint-clip'] });
  assert.equal(e.defaultPrevented, false, 'no drop target for a drag that is gone');
  assert.deepEqual(m.marked(), []);
  m.fire(m.row('h002'), 'drop', { types: ['application/x-cutpoint-clip'] });
  assert.deepEqual(m.seqIds(), ['h001', 'h003', 'h002'], 'the stale drag moved nothing');
});

test('dragenter is accepted like dragover for our drag and refused for a file (Edge sends only dragenter as the pointer changes element)', async (t) => {
  const m = await drag(t);
  m.fire(m.row('h001'), 'dragstart');
  const ours = m.fire(m.row('h003'), 'dragenter');
  assert.equal(ours.defaultPrevented, true);
  assert.deepEqual(m.marked(), ['h003']);
  assert.equal(m.fire(m.row('h002'), 'dragenter', { types: ['Files'] }).defaultPrevented, false);
  const cell = m.fire(m.cell('h003'), 'dragenter');
  assert.equal(cell.defaultPrevented, true);
  assert.equal(m.fire(m.cell('h003'), 'dragenter', { types: ['Files'] }).defaultPrevented, false);
  assert.deepEqual(m.marked(), [], 'the marker moved to the storyboard');
  assert.ok(m.cell('h003').classList.contains('drop'));
});

test('a drop in the gap between rows or blocks goes to the nearest one', async (t) => {
  const m = await drag(t);
  const at = (list, centers, horiz) => [...list.children].forEach((k, i) => {
    k.getBoundingClientRect = () => (horiz ? { left: centers[i] - 5, right: centers[i] + 5, top: 0, bottom: 10 } : { top: centers[i] - 5, bottom: centers[i] + 5, left: 0, right: 10 });
  });
  at(m.$('#seqList'), [10, 60, 110], false);
  m.fire(m.row('h001'), 'dragstart');
  const over = m.fire(m.$('#seqList'), 'dragover', { y: 100 });
  assert.equal(over.defaultPrevented, true);
  assert.deepEqual(m.marked(), ['h003']);
  m.fire(m.$('#seqList'), 'drop', { y: 100 });
  assert.deepEqual(m.seqIds(), ['h002', 'h003', 'h001']);
  await m.page.settle();
  at(m.$('#sbList'), [5, 50, 95], true);
  m.fire(m.cell('h001'), 'dragstart');
  m.fire(m.$('#sbList'), 'dragenter', { x: 10 });
  assert.ok(m.cell('h002').classList.contains('drop') && !m.after(m.cell('h002')));
  m.fire(m.$('#sbList'), 'drop', { x: 10 });
  assert.deepEqual(m.seqIds(), ['h001', 'h002', 'h003']);
  // a drag from the other list is not routed
  m.fire(m.cell('h001'), 'dragstart');
  assert.equal(m.fire(m.$('#seqList'), 'dragover').defaultPrevented, false);
});

test('removing the last clip moves focus to the Sequence tab, not to the hidden heading', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project: withSegments('h001') });
  t.after(() => page.close());
  const d = page.document;
  d.querySelector('#seqList [data-act=remove]').click();
  await page.settle();
  const dlg = d.querySelector('#confirmDialog');
  dlg.returnValue = 'ok';
  dlg.removeAttribute('open');
  dlg.dispatchEvent(new page.window.Event('close'));
  await page.settle();
  assert.equal(d.querySelectorAll('#seqList > li').length, 0);
  assert.equal(d.activeElement, d.querySelector('#tabSequence'));
  assert.equal(d.querySelector('#seqTitle').hasAttribute('tabindex'), false, 'the heading is not a focus target');
});

test('a drop that is not ours (a file, text) moves nothing even while an id is held', async (t) => {
  const m = await drag(t);
  m.fire(m.row('h001'), 'dragstart');
  const over = m.fire(m.row('h003'), 'dragover', { types: ['Files'] });
  assert.equal(over.defaultPrevented, false, 'a foreign drag is not accepted');
  m.fire(m.row('h003'), 'drop', { types: ['Files'] });
  assert.deepEqual(m.seqIds(), ['h001', 'h002', 'h003']);
  assert.equal(m.d.querySelectorAll('.dragging, .drop').length, 0, 'and the drag is over');
});

test('leaving the window ends the drag; moving between elements does not', async (t) => {
  const m = await drag(t);
  m.fire(m.row('h001'), 'dragstart');
  m.fire(m.row('h003'), 'dragover');
  const between = new m.w.Event('dragleave', { bubbles: true });
  Object.assign(between, { relatedTarget: m.row('h002'), clientX: 50, clientY: 50 });
  m.row('h003').dispatchEvent(between);
  assert.deepEqual(m.marked(), ['h003'], 'still dragging');
  const out = new m.w.Event('dragleave', { bubbles: true });
  Object.assign(out, { relatedTarget: null, clientX: -1, clientY: 50 });
  m.d.body.dispatchEvent(out);
  assert.equal(m.d.querySelectorAll('.dragging, .drop').length, 0);
  m.fire(m.row('h003'), 'drop', { types: ['application/x-cutpoint-clip'] });
  assert.deepEqual(m.seqIds(), ['h001', 'h002', 'h003'], 'coming back in, the drop is ignored');
});

test('the storyboard marks the side a block lands on and ignores a foreign drop', async (t) => {
  const m = await drag(t);
  m.fire(m.cell('h001'), 'dragstart');
  m.fire(m.cell('h003'), 'dragover');
  assert.ok(m.cell('h003').classList.contains('drop') && m.after(m.cell('h003')));
  m.fire(m.cell('h003'), 'drop', { types: ['text/plain'] });
  assert.deepEqual(m.seqIds(), ['h001', 'h002', 'h003'], 'a foreign drop moves nothing');
  m.fire(m.cell('h003'), 'dragstart');
  m.fire(m.cell('h001'), 'dragover');
  assert.ok(m.cell('h001').classList.contains('drop') && !m.after(m.cell('h001')));
  m.fire(m.cell('h001'), 'drop');
  assert.deepEqual(m.seqIds(), ['h003', 'h001', 'h002']);
  assert.equal(m.d.querySelectorAll('.dragging, .drop').length, 0);
});

// ---- scrolling while a clip is held ----

/** Make #side the scrolling column: a 100..700 px box with a 40 px sticky footer; ``at`` is its scrollTop. */
function column(m) {
  const side = m.$('#side');
  side.style.overflowY = 'auto';
  side.style.setProperty('--seq-foot-h', '40px');
  side.getBoundingClientRect = () => ({ top: 100, bottom: 700, left: 0, right: 400, width: 400, height: 600 });
  const at = { top: 300 };
  Object.defineProperty(side, 'scrollTop', { configurable: true, get: () => at.top, set: (v) => { at.top = v; } });
  return at;
}
const frames = (m, n) => { for (let i = 0; i < n; i++) m.t.mock.timers.tick(16); };

test('a held clip scrolls the list near its top and bottom edges, faster nearer the edge, and stops when released', async (t) => {
  const m = await drag(t);
  const at = column(m);
  m.fire(m.row('h001'), 'dragstart');
  // the bottom zone sits above the sticky footer: 700 - 40 = 660
  m.fire(m.d, 'dragover', { x: 200, y: 560 });
  frames(m, 5);
  assert.equal(at.top, 300, 'mid-list: no scrolling');
  m.fire(m.d, 'dragover', { x: 200, y: 630 });
  frames(m, 5);
  const slow = at.top - 300;
  assert.ok(slow > 0, 'near the footer: scrolls down');
  at.top = 300;
  m.fire(m.d, 'dragover', { x: 200, y: 658 });
  frames(m, 5);
  assert.ok(at.top - 300 > slow, 'nearer the edge: faster');
  // the top zone starts under the sticky bars (a scroll padding of 0 here): 100
  at.top = 300;
  m.fire(m.d, 'dragover', { x: 200, y: 110 });
  frames(m, 5);
  assert.ok(at.top < 300, 'near the top: scrolls up');
  // a drop stops it
  m.fire(m.row('h002'), 'drop');
  const stopped = at.top;
  frames(m, 5);
  assert.equal(at.top, stopped, 'released: no more scrolling');
});

test('the scroll stops on dragend, when the pointer leaves the zone or the column, and at a re-render', async (t) => {
  const m = await drag(t);
  const at = column(m);
  const run = (y, x = 200) => { m.fire(m.row('h001'), 'dragstart'); m.fire(m.d, 'dragover', { x, y }); frames(m, 2); };
  run(650);
  assert.ok(at.top > 300);
  m.fire(m.d, 'dragover', { x: 200, y: 400 });
  let v = at.top;
  frames(m, 3);
  assert.equal(at.top, v, 'back in the middle: stopped');
  m.fire(m.row('h001'), 'dragend');
  at.top = 300;
  run(650, 900); // outside the column horizontally
  assert.equal(at.top, 300, 'over another column: no scroll');
  m.fire(m.row('h001'), 'dragend');
  run(650);
  v = at.top;
  m.fire(m.row('h001'), 'dragend');
  frames(m, 3);
  assert.equal(at.top, v, 'dragend stops it');
  run(650);
  v = at.top;
  m.$('#seqList li[data-id=h002] [data-act=down]').click(); // re-render
  await m.page.settle();
  frames(m, 3);
  assert.equal(at.top, v, 're-render stops it');
});

test('the storyboard scrolls sideways at its ends when its blocks overflow', async (t) => {
  const m = await drag(t);
  const sb = m.$('#sbList');
  const at = { left: 50 };
  Object.defineProperties(sb, {
    scrollWidth: { configurable: true, get: () => 900 },
    clientWidth: { configurable: true, get: () => 300 },
    scrollLeft: { configurable: true, get: () => at.left, set: (v) => { at.left = v; } },
  });
  sb.getBoundingClientRect = () => ({ left: 0, right: 300, top: 0, bottom: 44, width: 300, height: 44 });
  m.fire(m.cell('h001'), 'dragstart');
  m.fire(m.d, 'dragover', { x: 296, y: 20 });
  frames(m, 4);
  assert.ok(at.left > 50, 'near the right end: scrolls right');
  m.fire(m.d, 'dragover', { x: 4, y: 20 });
  const v = at.left;
  frames(m, 4);
  assert.ok(at.left < v, 'near the left end: scrolls left');
  m.fire(m.cell('h002'), 'dragend');
  const stopped = at.left;
  frames(m, 3);
  assert.equal(at.left, stopped);
});
