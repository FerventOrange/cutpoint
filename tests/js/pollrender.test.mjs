// A project poll (every 2 s while a proxy builds, every second while a 360 preview renders) that
// changes nothing the Moments tiles show keeps the tiles instead of rebuilding them (A753); a poll
// that changes what a tile shows still redraws it.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

const MOMENTS = 60;

/** A project whose source builds its proxy: ``state.src`` and ``state.cands`` are what the server lists now. */
function polled(t) {
  const base = makeProject();
  const state = {
    src: { ...base.sources[0], status: 'preparing', windows: [{ lo: 0, hi: 60, url: '/w' }], failed: [], progress: 0.1,
      pending: [{ lo: 90, hi: 120, state: 'encoding' }] },
    cands: Array.from({ length: MOMENTS }, (_, i) => ({
      id: `m${i}`, cli_id: `h${String(i + 1).padStart(3, '0')}`, source: '/videos/a.mp4', start: 10 + i * 2, end: 18 + i * 2,
      score: 90 - i / 10, label: `moment ${i}`, transcript: `line ${i}`,
    })),
  };
  // a new object on every request, as the server's reply is
  const project = () => structuredClone({ ...base, sources: [state.src], highlights: { ...base.highlights, candidates: state.cands } });
  return { state, project };
}

async function open(t) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { state, project } = polled(t);
  const page = await boot({ project: project(), routes: { 'GET /api/projects/p1': () => project(), 'POST /api/projects/p1/sources/s1/prepare': {} } });
  t.after(() => page.close());
  const list = page.document.querySelector('#candList');
  const changes = [];
  new page.window.MutationObserver((records) => changes.push(...records)).observe(list, { childList: true });
  const polls = () => page.requests.filter((r) => r.method === 'GET' && r.path === '/api/projects/p1').length;
  /** One poll of the project (the page asks every 2 s while the proxy builds). */
  const poll = async () => {
    const before = polls();
    await page.settle();
    t.mock.timers.tick(2000);
    await page.settle();
    assert.equal(polls(), before + 1, 'the page polled the project');
  };
  const tile = (id) => list.querySelector(`li.cand[data-id="${id}"]`);
  return { page, state, list, changes, poll, tile };
}

test('polls that change nothing a tile shows keep every tile, its focus and its badges', async (t) => {
  const { page, state, list, changes, poll, tile } = await open(t);
  assert.equal(list.querySelectorAll('li.cand').length, MOMENTS);
  const first = tile('m0');
  const btn = tile('m40').querySelector('[data-act=toggle]');
  btn.focus();
  changes.length = 0;
  for (let i = 0; i < 5; i++) {
    state.src = { ...state.src, progress: 0.2 + i / 10 }; // the whole proxy encodes on: no tile shows its share
    await poll();
  }
  assert.equal(changes.length, 0, 'no tile was replaced');
  assert.equal(tile('m0'), first, 'the same tile element');
  assert.equal(page.document.activeElement, btn, 'focus stays on the button');
  assert.equal(tile('m0').querySelector('.prep').dataset.state, 'ready');
  assert.equal(tile('m45').querySelector('.prep').dataset.state, 'encoding');
  page.assertNoUnhandled();
});

test('a poll that changes what a tile shows redraws the tiles', async (t) => {
  const { page, state, changes, poll, tile } = await open(t);
  changes.length = 0;
  // the window around m45 (100-108) is ready now: its badge turns to Ready
  state.src = { ...state.src, windows: [...state.src.windows, { lo: 90, hi: 120, url: '/w2' }], pending: [] };
  await poll();
  assert.ok(changes.length > 0, 'the tiles were redrawn');
  assert.equal(tile('m45').querySelector('.prep').dataset.state, 'ready');
  // a moment's label changes (an analysis rewrote highlights.json)
  changes.length = 0;
  state.cands = state.cands.map((c, i) => (i === 3 ? { ...c, label: 'renamed moment' } : c));
  await poll();
  assert.ok(changes.length > 0, 'the tiles were redrawn');
  assert.equal(tile('m3').querySelector('.cand-title').textContent, 'renamed moment');
  page.assertNoUnhandled();
});

test('adding a moment to the sequence still redraws its tile', async (t) => {
  const { page, changes, tile } = await open(t);
  changes.length = 0;
  tile('m7').querySelector('[data-act=toggle]').click();
  await page.settle();
  assert.ok(changes.length > 0, 'the tiles were redrawn');
  assert.ok(tile('m7').classList.contains('picked'));
  assert.equal(tile('m7').querySelector('[data-act=toggle]').getAttribute('aria-label'), 'Remove moment 7 from sequence');
  page.assertNoUnhandled();
});
