// Undo / Redo of the last ranking change (#902): the Moments link above the list and the Undo
// button of the In-use ranking card. Only the undo route writes; the page reloads the project after.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject, reply } from './harness.mjs';

const A = '/api/projects/p1/assistant';
const U = '/api/projects/p1/ranking/undo';
const provider = { name: 'ollama', label: 'Ollama · qwen', local: true };
const timed = new WeakSet();
const CARD = '#chatLog [data-rid="t1-2"]';
const funnyReady = { kind: 'ranking', n: 2, style: 'funny', base: 'k1', ready: true, can_analyze: true, missing: [], top: [], suggested_top: { before: 3, after: 3 }, notes: [], reason: '', new_ids_count: 0, dropped_count: 0 };

/** The project as the server answers it: ranked in ``style`` with ``undo`` as its ranking_undo. */
const proj = (style, undo, over = {}) => makeProject({
  ranking_key: 'k2', ranking_undo: undo, highlights: { ...makeProject().highlights, style }, ...over,
});

async function open(t, { project, routes = {} }) {
  if (!timed.has(t)) { timed.add(t); t.mock.timers.enable({ apis: ['setTimeout'] }); }
  const state = { project };
  const page = await boot({
    project,
    routes: {
      'GET /api/projects/p1': () => state.project,
      [`GET ${A}`]: { provider, consent: null, running: [], turns: [{ id: 't1', user: 'rank it funny', status: 'done', ranking_key: 'k1', events: [funnyReady] }] },
      ...routes,
    },
  });
  t.after(() => page.close());
  const $ = (s) => page.document.querySelector(s);
  return {
    page, $, state,
    link: () => $('#rankUndoBtn'),
    row: () => $('#rankUndoRow'),
    card: () => $(CARD),
    cardUndo: () => $(`${CARD} button[data-act=undo-ranking]`),
    posts: () => page.requests.filter((r) => r.method === 'POST' && r.path === U),
  };
}

test('the link says Undo ranking (was Gaming), Redo ranking (Funny), or is hidden without ranking_undo', async (t) => {
  const undo = await open(t, { project: proj('funny', { action: 'undo', style: 'gaming' }) });
  assert.equal(undo.row().hidden, false);
  assert.equal(undo.link().textContent, 'Undo ranking (was Gaming)');
  assert.equal(undo.link().disabled, false);
  undo.page.close();

  const redo = await open(t, { project: proj('gaming', { action: 'redo', style: 'funny' }) });
  assert.equal(redo.link().textContent, 'Redo ranking (Funny)');
  redo.page.close();

  const none = await open(t, { project: proj('funny', null) });
  assert.equal(none.row().hidden, true);
});

test('the link is disabled while a run holds the project', async (t) => {
  const m = await open(t, { project: proj('funny', { action: 'undo', style: 'gaming' }, { busy: true }) });
  assert.equal(m.link().disabled, true);
});

test('Undo posts {base}, shows Undoing ranking…, disabled, then reloads and announces the old ranking', async (t) => {
  let release;
  const gate = new Promise((r) => { release = r; });
  const m = await open(t, {
    project: proj('funny', { action: 'undo', style: 'gaming' }),
    routes: { [`POST ${U}`]: async () => { await gate; return { action: 'undo', style: 'gaming', from: 'funny', key: 'k3' }; } },
  });
  m.link().click();
  await m.page.settle();
  assert.equal(m.link().textContent, 'Undoing ranking…');
  assert.equal(m.link().disabled, true);
  assert.equal(m.$('#rankUndoLive').textContent, 'Undoing the ranking…');

  const loads = () => m.page.requests.filter((r) => r.method === 'GET' && r.path === '/api/projects/p1').length;
  const before = loads();
  m.state.project = proj('gaming', { action: 'redo', style: 'funny' }, { ranking_key: 'k3' });
  release();
  await m.page.settle();
  assert.deepEqual(m.posts().map((r) => r.body), [{ base: 'k2' }]);
  assert.equal(loads(), before + 1, 'the project is fetched again');
  assert.equal(m.$('#rankUndoLive').textContent, 'Back to the Gaming ranking.');
  assert.equal(m.link().textContent, 'Redo ranking (Funny)');
  assert.equal(m.link().disabled, false);
});

test('Redo posts the new key and announces the ranking in use again', async (t) => {
  const m = await open(t, {
    project: proj('gaming', { action: 'redo', style: 'funny' }),
    routes: { [`POST ${U}`]: { action: 'redo', style: 'funny', from: 'gaming', key: 'k3' } },
  });
  m.state.project = proj('funny', { action: 'undo', style: 'gaming' }, { ranking_key: 'k3' });
  m.link().click();
  await m.page.settle();
  assert.equal(m.$('#rankUndoLive').textContent, 'Using the Funny ranking again.');
  assert.equal(m.link().textContent, 'Undo ranking (was Gaming)');
});

test('409 stale: an error message, the page follows the server and the link is enabled again', async (t) => {
  const m = await open(t, {
    project: proj('funny', { action: 'undo', style: 'gaming' }),
    routes: { [`POST ${U}`]: () => reply(409, { detail: 'The moments changed since the page loaded', code: 'stale' }) },
  });
  m.state.project = proj('funny', null, { ranking_key: 'k9' }); // an analysis ran meanwhile
  m.link().click();
  await m.page.settle();
  assert.match(m.$('#banner').textContent, /The moments changed since the page loaded/);
  assert.equal(m.$('#banner').hidden, false);
  assert.equal(m.$('#rankUndoLive').textContent, '');
  assert.equal(m.row().hidden, true, 'no earlier ranking any more');

  // and when the Undo is still offered (a stale reply with the same offer), the link works again
  const again = await open(t, {
    project: proj('funny', { action: 'undo', style: 'gaming' }),
    routes: { [`POST ${U}`]: () => reply(409, { detail: 'moved on', code: 'stale' }) },
  });
  again.link().click();
  await again.page.settle();
  assert.equal(again.link().disabled, false);
  assert.equal(again.link().textContent, 'Undo ranking (was Gaming)');
});

test('409 folder_busy while a run holds the project: no error banner, link disabled', async (t) => {
  const m = await open(t, {
    project: proj('funny', { action: 'undo', style: 'gaming' }),
    routes: { [`POST ${U}`]: () => reply(409, { detail: 'busy', code: 'folder_busy' }) },
  });
  m.state.project = proj('funny', { action: 'undo', style: 'gaming' }, { busy: true });
  m.link().click();
  await m.page.settle();
  assert.equal(m.$('#banner').hidden, true, 'a busy project is a state, not an error');
  assert.equal(m.link().disabled, true);
});

test('the In-use card has an Undo button only while the action is undo, with its accessible name', async (t) => {
  const undo = await open(t, { project: proj('funny', { action: 'undo', style: 'gaming' }) });
  assert.equal(undo.card().dataset.state, 'in-use');
  assert.equal(undo.cardUndo().textContent, 'Undo');
  assert.equal(undo.cardUndo().getAttribute('aria-label'), 'Undo the Funny ranking, back to Gaming');
  assert.equal(undo.cardUndo().disabled, false);
  undo.page.close();

  const redo = await open(t, { project: proj('funny', { action: 'redo', style: 'gaming' }) });
  assert.equal(redo.card().dataset.state, 'in-use');
  assert.equal(redo.cardUndo(), null);
  redo.page.close();

  const none = await open(t, { project: proj('funny', null) });
  assert.equal(none.card().dataset.state, 'in-use');
  assert.equal(none.cardUndo(), null);
  none.page.close();

  // a card that is not In use (the moments are ranked in another style) has none either
  const other = await open(t, { project: proj('gaming', { action: 'undo', style: 'funny' }) });
  assert.notEqual(other.card().dataset.state, 'in-use');
  assert.equal(other.cardUndo(), null);
});

test('Undo from the card posts {base}, announces in the chat live region, and the card turns Stale', async (t) => {
  const m = await open(t, {
    project: proj('funny', { action: 'undo', style: 'gaming' }),
    routes: { [`POST ${U}`]: { action: 'undo', style: 'gaming', from: 'funny', key: 'k3' } },
  });
  m.state.project = proj('gaming', { action: 'redo', style: 'funny' }, { ranking_key: 'k3' });
  m.cardUndo().click();
  await m.page.settle();
  assert.deepEqual(m.posts().map((r) => r.body), [{ base: 'k2' }]);
  assert.equal(m.$('#chatLive').textContent, 'Back to the Gaming ranking.');
  assert.equal(m.card().dataset.state, 'stale');
  assert.equal(m.cardUndo(), null);
  assert.equal(m.link().textContent, 'Redo ranking (Funny)');
});

test('the card Undo shows Undoing… and disables the card while the route works', async (t) => {
  let release;
  const gate = new Promise((r) => { release = r; });
  const m = await open(t, {
    project: proj('funny', { action: 'undo', style: 'gaming' }),
    routes: { [`POST ${U}`]: async () => { await gate; return { action: 'undo', style: 'gaming', from: 'funny', key: 'k3' }; } },
  });
  m.cardUndo().click();
  await m.page.settle();
  assert.equal(m.cardUndo().textContent, 'Undoing…');
  assert.equal(m.cardUndo().disabled, true);
  assert.equal(m.$('#chatLive').textContent, 'Undoing the ranking…');
  assert.equal(m.link().disabled, true);
  m.state.project = proj('gaming', { action: 'redo', style: 'funny' }, { ranking_key: 'k3' });
  release();
  await m.page.settle();
});
