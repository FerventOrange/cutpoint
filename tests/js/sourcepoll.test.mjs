// The proxy polls ask for the sources alone (A754): GET /api/projects/<id>/sources every 2 s while a
// proxy builds, the whole project (every moment and its transcript) once a source's status changes.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

async function open(t, routes = {}, review = undefined) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const base = makeProject(review ? { review } : {});
  const state = { src: { ...base.sources[0], status: 'preparing', progress: 0.1, windows: [], pending: [], failed: [] } };
  const project = () => structuredClone({ ...base, sources: [state.src] });
  const page = await boot({
    project: project(),
    routes: {
      'GET /api/projects/p1': () => project(),
      'GET /api/projects/p1/sources': () => ({ sources: [structuredClone(state.src)] }),
      'POST /api/projects/p1/sources/s1/prepare': {},
      ...routes,
    },
  });
  t.after(() => page.close());
  const gets = (path) => page.requests.filter((r) => r.method === 'GET' && r.path === path).length;
  const count = () => ({ full: gets('/api/projects/p1'), light: gets('/api/projects/p1/sources') });
  const tick = async () => {
    await page.settle();
    t.mock.timers.tick(2000);
    await page.settle();
  };
  return { page, state, count, tick, prep: () => page.document.querySelector('#stagePrep').textContent };
}

test('while a proxy builds the page polls the sources, not the whole project, and shows the progress', async (t) => {
  const { page, state, count, tick, prep } = await open(t);
  assert.equal(prep(), 'Full preview 10%');
  const before = count();
  for (const progress of [0.3, 0.5, 0.5]) {
    state.src = { ...state.src, progress };
    await tick();
  }
  assert.deepEqual(count(), { full: before.full, light: before.light + 3 });
  assert.equal(prep(), 'Full preview 50%');
  page.assertNoUnhandled();
});

test('a proxy that finished refreshes the whole project once, then the polls stop', async (t) => {
  const { page, state, count, tick, prep } = await open(t);
  await tick();
  const before = count();
  state.src = { ...state.src, status: 'ready', progress: null, proxy: true };
  await tick();
  assert.deepEqual(count(), { full: before.full + 1, light: before.light + 1 });
  assert.equal(prep(), '');
  await tick();
  await tick();
  assert.deepEqual(count(), { full: before.full + 1, light: before.light + 1 }, 'nothing left to poll');
  page.assertNoUnhandled();
});

test('a light poll that answers after another project opened is dropped', async (t) => {
  const { page, count, tick } = await open(t, {
    'GET /api/projects': { projects: [{ id: 'p1', name: 'Project One' }, { id: 'p2', name: 'Two' }] },
    'GET /api/projects/p2': makeProject({ id: 'p2', name: 'Two' }),
    'GET /api/projects/p2/assistant': { provider: null, consent: null, turns: [], running: [] },
    'GET /api/projects/p2/assistant/history': { applied: [] },
    'GET /api/projects/p2/events': { events: [] },
    'GET /api/projects/p2/sources/s1/envelope': { step: 0.05, peaks: [] },
    'POST /api/projects/p2/edges': { edges: {} },
  });
  let answer;
  page.routes['GET /api/projects/p1/sources'] = () => new Promise((resolve) => { answer = resolve; });
  await tick(); // the poll is asked, not answered yet
  assert.ok(answer, 'the page polled the sources');
  const d = page.document;
  d.querySelector('#projectBtn').click();
  d.querySelector('#projectMenu [data-id=p2]').click();
  await page.settle();
  const full = count().full;
  answer({ sources: [{ ...makeProject().sources[0], status: 'failed' }] }); // p1's, would refresh it
  await page.settle();
  assert.equal(count().full, full, 'p1 is not read again');
  assert.equal(d.querySelector('#projectMenu [data-id=p2]').getAttribute('aria-checked'), 'true');
  page.assertNoUnhandled();
});

test('a sequence saved while a proxy builds stays as saved after a light poll', async (t) => {
  const seg = (c) => ({ id: c.id, source: c.source, start: c.start, end: c.end, label: c.label });
  const review = { segments: makeProject().highlights.candidates.map(seg) }; // h001, h002, h003
  let saved = review;
  const { page, tick } = await open(t, {
    'GET /api/projects/p1': () => ({ ...makeProject(), review: saved,
      sources: [{ ...makeProject().sources[0], status: 'preparing', progress: 0.1, windows: [], pending: [], failed: [] }] }),
    'PUT /api/projects/p1/review': (req) => { saved = req.body; return {}; },
  }, review);
  const d = page.document;
  const w = page.window;
  const ids = () => [...d.querySelectorAll('#seqList > li')].map((li) => li.dataset.id);
  assert.deepEqual(ids(), ['h001', 'h002', 'h003']);
  let dt = null;
  const fire = (el, type) => {
    const e = new w.Event(type, { bubbles: true, cancelable: true });
    e.dataTransfer = type === 'dragstart'
      ? (dt = { types: [], effectAllowed: '', setData(k) { this.types.push(k); } })
      : { types: dt.types, effectAllowed: '' };
    e.clientX = 10;
    e.clientY = 10;
    el.dispatchEvent(e);
  };
  const row = (id) => d.querySelector(`#seqList li[data-id=${id}]`);
  fire(row('h001'), 'dragstart');
  fire(row('h003'), 'dragover');
  fire(row('h003'), 'drop');
  await page.settle();
  const moved = ids();
  assert.notDeepEqual(moved, ['h001', 'h002', 'h003'], 'reordered');
  d.querySelector('#saveBtn').click();
  await page.settle();
  assert.deepEqual(saved.segments.map((s) => s.id), moved);
  await tick();
  assert.deepEqual(ids(), moved, 'the saved order stays');
  page.assertNoUnhandled();
});
