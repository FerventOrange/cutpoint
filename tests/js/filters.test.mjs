// Sort and filter: the quick filters and the order narrow the moments list, are remembered per
// browser (guarded against blocked or bad storage) and the source file only per open project.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

const FILTERS = 'cutpoint.momentFilters';
const ORDER = 'cutpoint.momentOrder';

async function list(t, { storage = {}, project = makeProject(), routes = {} } = {}) {
  t.mock.timers.reset();
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ storage, project, routes });
  t.after(() => page.close());
  const d = page.document;
  const m = {
    page, d,
    $: (s) => d.querySelector(s),
    cards: () => [...d.querySelectorAll('#candList li.cand')].map((li) => li.dataset.id),
    ranks: () => [...d.querySelectorAll('#candList .rank')].map((r) => r.textContent),
    async set(sel, value) {
      const el = d.querySelector(sel);
      if (el.type === 'checkbox' || el.type === 'radio') el.checked = value; else el.value = value;
      el.dispatchEvent(new page.window.Event('change', { bubbles: true }));
      await page.settle();
    },
    async radio(name, value) {
      const el = d.querySelector(`input[name=${name}][value=${value}]`);
      el.checked = true;
      el.dispatchEvent(new page.window.Event('change', { bubbles: true }));
      await page.settle();
    },
    stored: (k) => page.window.localStorage.getItem(k),
  };
  return m;
}

test('"In sequence" and "Not in sequence" split the list by what is picked, and Clear filters undoes them', async (t) => {
  const p = makeProject();
  p.review = { segments: [{ id: 'h002', source: '/videos/a.mp4', start: 100, end: 108, label: 'Fall' }] };
  const m = await list(t, { project: p });
  await m.radio('filterSeq', 'in');
  assert.deepEqual(m.cards(), ['h002']);
  await m.radio('filterSeq', 'out');
  assert.deepEqual(m.cards(), ['h001', 'h003']);
  assert.match(m.$('#filterBtn').getAttribute('aria-label'), /filters on/);
  m.$('#filterClear').click();
  await m.page.settle();
  assert.deepEqual(m.cards(), ['h001', 'h002', 'h003']);
  assert.equal(m.stored(FILTERS), null, 'nothing is stored once no filter is on');
});

test('Sequence and Ready-only filters survive a reload; a bad stored value is ignored', async (t) => {
  const first = await list(t);
  await first.radio('filterSeq', 'out');
  await first.set('#filterReady', true);
  const kept = JSON.parse(first.stored(FILTERS));
  assert.deepEqual(kept, { seq: 'out', ready: true });
  first.page.close();

  const again = await list(t, { storage: { [FILTERS]: JSON.stringify(kept) } });
  assert.equal(again.$('#filterReady').checked, true);
  assert.equal(again.d.querySelector('input[name=filterSeq][value=out]').checked, true);

  const bad = await list(t, { storage: { [FILTERS]: '{"seq":"sideways","ready":"yes"}' } });
  assert.equal(bad.d.querySelector('input[name=filterSeq][value=all]').checked, true);
  assert.equal(bad.$('#filterReady').checked, false);
  assert.equal(bad.cards().length, 3);
  const garbage = await list(t, { storage: { [FILTERS]: 'not json' } });
  assert.equal(garbage.cards().length, 3);
});

test('"Ready only" hides moments whose preview cannot play yet', async (t) => {
  const p = makeProject();
  p.sources[0] = { ...p.sources[0], status: 'needs-proxy', windows: [{ lo: 0, hi: 60, url: '/w' }], pending: [], failed: [] };
  const m = await list(t, { project: p, routes: { 'POST /api/projects/p1/sources/s1/prepare': {} } });
  await m.set('#filterReady', true);
  assert.deepEqual(m.cards(), ['h001'], 'only the moment inside the ready window (10-18 of 0-60)');
});

test('Position order sorts by video position while the badge keeps the score rank; it is remembered', async (t) => {
  const p = makeProject();
  p.highlights.candidates.reverse(); // ranked: h003 (#1), h002 (#2), h001 (#3)
  const m = await list(t, { project: p });
  assert.deepEqual(m.cards(), ['h003', 'h002', 'h001']);
  await m.radio('momentOrder', 'position');
  assert.deepEqual(m.cards(), ['h001', 'h002', 'h003']);
  assert.deepEqual(m.ranks(), ['#3', '#2', '#1']);
  assert.equal(m.stored(ORDER), 'position');
  await m.radio('momentOrder', 'score');
  assert.equal(m.stored(ORDER), null);
  m.page.close();

  const again = await list(t, { project: p, storage: { [ORDER]: 'position' } });
  assert.deepEqual(again.cards(), ['h001', 'h002', 'h003']);
  const odd = await list(t, { project: p, storage: { [ORDER]: 'bogus' } });
  assert.deepEqual(odd.cards(), ['h003', 'h002', 'h001'], 'an unknown order falls back to Score');
});

test('with several source files the Source file choice appears and filters the list', async (t) => {
  const p = makeProject();
  p.highlights.candidates[1].source = '/videos/b.mp4';
  p.sources.push({ ...p.sources[0], id: 's2', path: '/videos/b.mp4', name: 'b.mp4', url: '/api/projects/p1/sources/s2?t=x' });
  const m = await list(t, { project: p });
  assert.equal(m.$('#filterSourceWrap').hidden, false);
  await m.set('#filterSource', '/videos/b.mp4');
  assert.deepEqual(m.cards(), ['h002']);
});
