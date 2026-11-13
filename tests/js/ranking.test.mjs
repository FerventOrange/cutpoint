// AI Chat ranking card (#902): Ready / Applying / In use / Stale / Needs analyze / Busy, derived
// from the project's ranking_key so a reload shows the right state; only the apply route writes.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject, reply } from './harness.mjs';

const A = '/api/projects/p1/assistant';
const R = '/api/projects/p1/ranking';
const provider = { name: 'ollama', label: 'Ollama · qwen', local: true };
const timed = new WeakSet();
const CARD = '#chatLog [data-rid="t1-2"]';

/** Ten rows: h001-h003 exist in the fixture project, h004.. do not. */
function top() {
  const was = [3, 2, 1, null, null, null, null, null, null, null];
  return was.map((w, i) => ({
    id: `h00${i + 1}`, cli_id: `h00${i + 1}`, rank: i + 1, was: w, label: `Moment ${i + 1}`,
    file: 'a.mp4', start: 10 + i * 20, end: 18 + i * 20,
  }));
}
const ready = (over = {}) => ({
  kind: 'ranking', n: 2, style: 'funny', base: 'k1', ready: true, can_analyze: true, missing: [], top: top(),
  suggested_top: { before: 3, after: 5 }, notes: [], reason: 'You asked for jokes.', new_ids_count: 0, dropped_count: 0, ...over,
});
const proj = (over = {}) => makeProject({
  ranking_key: 'k1', highlights: { ...makeProject().highlights, style: 'gaming' }, ...over,
});

/**
 * ``m.state.project`` is what GET /api/projects/p1 answers (replace it to change what the server
 * says); ``m.state.jobs`` what GET /api/jobs answers.
 */
async function open(t, { project = proj(), events = [ready()], turnKey = 'k1', routes = {} } = {}) {
  if (!timed.has(t)) { timed.add(t); t.mock.timers.enable({ apis: ['setTimeout'] }); } // once per test, though a test may open several pages
  const state = { project, jobs: [] };
  const jobPosts = [];
  const page = await boot({
    project,
    routes: {
      'GET /api/projects/p1': () => state.project,
      'GET /api/jobs': () => ({ jobs: state.jobs }),
      'POST /api/projects/p1/jobs': (req) => { jobPosts.push(req.body); return { id: 'j1', status: 'running' }; },
      [`GET ${A}`]: { provider, consent: null, running: [], turns: [{ id: 't1', user: 'rank it funny', status: 'done', ranking_key: turnKey, events }] },
      ...routes,
    },
  });
  t.after(() => page.close());
  const $ = (s) => page.document.querySelector(s);
  const m = {
    page, $, state, jobPosts,
    card: () => $(CARD),
    btn: (act) => $(`${CARD} button[data-act=${act}]`),
    posts: (suffix) => page.requests.filter((r) => r.method === 'POST' && r.path === R + suffix),
    async click(act) { m.btn(act).click(); await page.settle(); },
    async tick(ms) { t.mock.timers.tick(ms); await page.settle(); },
  };
  return m;
}

test('Ready card: title, reason, chips, ten rows with their badges, and the two buttons', async (t) => {
  const m = await open(t);
  const c = m.card();
  assert.equal(c.dataset.state, 'ready');
  assert.equal(c.querySelector('.msg-title').textContent, 'Funny ranking (now: Gaming)');
  assert.match(c.textContent, /You asked for jokes\./);
  assert.match(c.querySelector('.chips').textContent, /7 new in top 10/);
  assert.match(c.querySelector('.chips').textContent, /suggested 5 \(was 3\)/);
  const rows = [...c.querySelectorAll('ol > li')];
  assert.equal(rows.length, 10);
  const badge = (i) => rows[i].querySelector('.prop-mark');
  assert.equal(badge(0).textContent, '↑2up 2 places');
  assert.ok(badge(0).classList.contains('rank-up'));
  assert.equal(badge(1).textContent, '=same place');
  assert.equal(badge(1).querySelector('[aria-hidden=true]').textContent, '=');
  assert.equal(badge(2).textContent, '↓2down 2 places');
  assert.ok(badge(2).classList.contains('rank-down'));
  assert.equal(badge(3).textContent, 'New');
  assert.equal(m.btn('use-ranking').textContent, 'Use this ranking');
  assert.equal(m.btn('use-ranking').disabled, false);
  assert.equal(m.btn('dismiss').textContent, 'Dismiss');
  assert.equal(m.btn('preview-again'), null);
});

test('a row focuses its moment when the page has it; a moment it lacks is plain text', async (t) => {
  const m = await open(t);
  const rows = [...m.card().querySelectorAll('ol > li')];
  assert.ok(rows[1].querySelector('button[data-act=focus]'));
  assert.equal(rows[5].querySelector('button'), null);
  rows[1].querySelector('button').click();
  await m.page.settle();
  assert.equal(m.$('#candList [data-id=h002]').getAttribute('aria-current'), 'true');
});

test('buttons have accessible names and the list is labelled', async (t) => {
  const m = await open(t);
  for (const b of m.card().querySelectorAll('button')) assert.ok(b.textContent.trim(), 'text name');
  assert.match(m.card().querySelector('ol').getAttribute('aria-label'), /first 10 moments in the Funny ranking/);
});

test('Use this ranking posts {style, base}, announces Applying, reloads the project and shows In use', async (t) => {
  let release;
  const gate = new Promise((r) => { release = r; });
  const m = await open(t, {
    routes: { [`POST ${R}/apply`]: async () => { await gate; return { new_ids: ['h009', 'h010'] }; } },
  });
  m.btn('use-ranking').click();
  await m.page.settle();
  assert.equal(m.$('#chatLive').textContent, 'Applying the Funny ranking…');
  assert.equal(m.card().dataset.state, 'applying');
  assert.equal(m.btn('use-ranking').textContent, 'Applying…');
  assert.equal(m.btn('use-ranking').disabled, true);
  assert.equal(m.btn('dismiss').disabled, true);

  const loads = () => m.page.requests.filter((r) => r.method === 'GET' && r.path === '/api/projects/p1').length;
  const before = loads();
  m.state.project = proj({ ranking_key: 'k2', highlights: { ...proj().highlights, style: 'funny' } });
  release();
  await m.page.settle();
  assert.deepEqual(m.posts('/apply').map((r) => r.body), [{ style: 'funny', base: 'k1' }]);
  assert.equal(loads(), before + 1, 'the project is fetched again');
  assert.equal(m.card().dataset.state, 'in-use');
  assert.match(m.card().textContent, /In use: the moments are ranked in the Funny style\./);
  assert.equal(m.btn('use-ranking'), null);
  assert.equal(m.$('#chatLive').textContent, 'Using the Funny ranking now; 2 new moments.');
});

test('409 stale: the card is Stale; Preview again posts to preview and the card is Ready with the new payload and the old reason', async (t) => {
  const fresh = ready({ base: 'k2', reason: 'a different reason', top: top().slice(0, 4) });
  const m = await open(t, {
    routes: {
      [`POST ${R}/apply`]: () => reply(409, { detail: 'moved on', code: 'stale' }),
      [`POST ${R}/preview`]: () => fresh,
    },
  });
  m.state.project = proj({ ranking_key: 'k2' }); // what the server holds by the time the apply is refused
  await m.click('use-ranking');
  assert.equal(m.card().dataset.state, 'stale');
  assert.match(m.card().textContent, /The moments changed since this preview\./);
  assert.equal(m.btn('use-ranking'), null);
  assert.equal(m.btn('preview-again').textContent, 'Preview again');

  await m.click('preview-again');
  assert.deepEqual(m.posts('/preview').map((r) => r.body), [{ style: 'funny' }]);
  const c = m.card();
  assert.equal(c.dataset.state, 'ready');
  assert.equal(c.querySelectorAll('ol > li').length, 4, 'the new payload');
  assert.match(c.textContent, /You asked for jokes\./, 'the model reason is kept');
  assert.doesNotMatch(c.textContent, /a different reason/);

  // and Use now sends the new base
  m.page.routes[`POST ${R}/apply`] = () => ({ new_ids: [] });
  await m.click('use-ranking');
  assert.deepEqual(m.posts('/apply').map((r) => r.body).at(-1), { style: 'funny', base: 'k2' });
});

test('409 needs_analyze: the card previews again and lists what is missing (3 and "and N more"); Analyze again as Funny starts a funny job', async (t) => {
  const missing = [1, 2, 3, 4, 5].map((n) => ({ file: `f${n}.mp4`, detail: `no laughs for f${n}.mp4` }));
  const m = await open(t, {
    routes: {
      [`POST ${R}/apply`]: () => reply(409, { detail: 'needs analysis', code: 'needs_analyze' }),
      [`POST ${R}/preview`]: () => ready({ ready: false, missing, top: [] }),
    },
  });
  await m.click('use-ranking');
  assert.equal(m.posts('/preview').length, 1);
  assert.equal(m.card().dataset.state, 'needs-analyze');
  const items = [...m.card().querySelectorAll('.rank-missing li')].map((li) => li.textContent);
  assert.deepEqual(items, ['no laughs for f1.mp4', 'no laughs for f2.mp4', 'no laughs for f3.mp4', 'and 2 more']);
  assert.equal(m.btn('use-ranking'), null);
  assert.equal(m.btn('analyze').textContent, 'Analyze again as Funny…');

  m.btn('analyze').click();
  const d = m.$('#analyzeDialog');
  assert.equal(m.$('#analyzeTitle').textContent, 'Analyze again as Funny?');
  // the harness's dialog has no close(value): set the return value and fire close, as the browser does
  d.close = (value) => { d.returnValue = value; d.removeAttribute('open'); d.dispatchEvent(new m.page.window.Event('close')); };
  d.close('ok');
  await m.page.settle();
  assert.equal(m.jobPosts.length, 1);
  assert.equal(m.jobPosts[0].action, 'analyze');
  assert.equal(m.jobPosts[0].style, 'funny');
});

test('footage offline only: Reconnect the footage, no Analyze button, Dismiss clears the card', async (t) => {
  const m = await open(t, { events: [ready({ ready: false, can_analyze: false, top: [], missing: [{ file: 'a.mp4', detail: 'a.mp4 is offline' }] })] });
  assert.equal(m.card().dataset.state, 'needs-analyze');
  assert.match(m.card().textContent, /Reconnect the footage/);
  assert.equal(m.btn('analyze'), null);
  assert.equal(m.btn('use-ranking'), null);
  const buttons = [...m.card().querySelectorAll('button')];
  assert.equal(buttons.length, 1, 'only Dismiss');
  assert.equal(buttons[0], m.btn('dismiss'));
  assert.equal((buttons[0].getAttribute('aria-label') || buttons[0].textContent).trim(), 'Dismiss');

  await m.click('dismiss');
  assert.equal(m.card().dataset.state, 'dismissed');
  assert.match(m.card().textContent, /Dismissed/);
  assert.equal(m.card().querySelectorAll('button').length, 0);
  assert.equal(m.$('#chatLive').textContent, 'Ranking dismissed');
  assert.equal(m.page.requests.filter((r) => r.method === 'POST').length, 0, 'nothing posted');
  assert.equal(m.jobPosts.length, 0);
});

test('folder_busy on apply: Busy text, buttons disabled and no error; back when the run ends', async (t) => {
  const m = await open(t, {
    routes: { [`POST ${R}/apply`]: () => reply(409, { detail: 'busy', code: 'folder_busy' }) },
  });
  m.state.project = proj({ busy: true });
  await m.click('use-ranking');
  assert.match(m.card().textContent, /Analysis is running on this project; wait for it to finish/);
  assert.equal(m.btn('use-ranking').disabled, true);
  assert.equal(m.$('#chatError').textContent, '', 'a busy project is a state, not an error');

  m.state.project = proj();
  await m.tick(5000); // the job poll re-reads a project held elsewhere
  await m.tick(5000);
  assert.doesNotMatch(m.card().textContent, /wait for it to finish/);
  assert.equal(m.btn('use-ranking').disabled, false);
});

test('a job of this page on the project disables the buttons and they return when it ends', async (t) => {
  const m = await open(t);
  assert.equal(m.btn('use-ranking').disabled, false);
  m.state.jobs = [{ id: 'j9', project_id: 'p1', action: 'analyze', status: 'running' }];
  await m.tick(5000);
  assert.equal(m.btn('use-ranking').disabled, true);
  assert.match(m.card().textContent, /wait for it to finish/);
  m.state.jobs = [{ id: 'j9', project_id: 'p1', action: 'analyze', status: 'done' }];
  await m.tick(1000);
  assert.equal(m.btn('use-ranking').disabled, false);
  assert.doesNotMatch(m.card().textContent, /wait for it to finish/);
});

test('Dismiss removes the buttons, keeps the card, announces it, and writes nothing', async (t) => {
  const m = await open(t);
  await m.click('dismiss');
  assert.equal(m.card().dataset.state, 'dismissed');
  assert.match(m.card().textContent, /Dismissed/);
  assert.equal(m.card().querySelectorAll('button:not([data-act=focus])').length, 0);
  assert.equal(m.$('#chatLive').textContent, 'Ranking dismissed');
  assert.equal(m.posts('/apply').length, 0);
});

test('state comes from the project on load: a ranking in use shows In use, an older base shows Stale', async (t) => {
  const inUse = await open(t, { project: proj({ ranking_key: 'k2', highlights: { ...proj().highlights, style: 'funny' } }) });
  assert.equal(inUse.card().dataset.state, 'in-use');
  assert.match(inUse.card().textContent, /In use/);
  inUse.page.close();

  const stale = await open(t, { project: proj({ ranking_key: 'k7' }) });
  assert.equal(stale.card().dataset.state, 'stale');
  assert.ok(stale.btn('preview-again'));
});

test('a reply from another ranking notes "(moment ids changed since)"; same key, no turn key or no project key say nothing', async (t) => {
  const events = [{ kind: 'reply', n: 1, text: 'See h002.' }];
  const note = (m) => m.$('#chatLog [data-act=ids-moved]');
  const moved = await open(t, { events, turnKey: 'k0' });
  assert.match(note(moved).textContent, /moment ids changed since/);
  moved.page.close();

  const same = await open(t, { events, turnKey: 'k1' });
  assert.equal(note(same), null);
  same.page.close();

  const none = await open(t, { events, turnKey: null });
  assert.equal(note(none), null);
  none.page.close();

  const noProject = await open(t, { events, turnKey: 'k0', project: proj({ ranking_key: undefined }) });
  assert.equal(note(noProject), null);
});

/** A page with no earlier turns whose one new turn polls in ``events`` with ``ranking_key`` ``key``. */
async function sent(t, events, key = 'k1', routes = {}) {
  const m = await open(t, {
    routes: {
      [`GET ${A}`]: { provider, consent: null, running: [], turns: [] },
      [`POST ${A}/turns`]: { id: 't1', status: 'running', provider },
      [`GET ${A}/turns/t1`]: () => ({ status: 'done', events, next: events.length, ranking_key: key }),
      ...routes,
    },
  });
  m.$('#chatInput').value = 'rank it funny';
  m.$('#chatForm').dispatchEvent(new m.page.window.Event('submit', { cancelable: true }));
  await m.page.settle();
  await m.tick(1000);
  return m;
}

test('a turn sent in this page takes its ranking_key from the poll: after Use this ranking its reply notes the moved ids, without a reload', async (t) => {
  const m = await sent(t, [{ kind: 'reply', n: 1, text: 'See h002.' }, ready()], 'k1', {
    [`POST ${R}/apply`]: () => ({ new_ids: [] }),
  });
  assert.equal(m.$('#chatLog [data-act=ids-moved]'), null, 'same key: no note');
  m.state.project = proj({ ranking_key: 'k2', highlights: { ...proj().highlights, style: 'funny' } });
  await m.click('use-ranking');
  assert.match(m.$('#chatLog [data-act=ids-moved]').textContent, /moment ids changed since/);
});

test('a preview_ranking tool error shows "Could not preview a ranking in the … style: …"', async (t) => {
  const events = [
    { kind: 'tool', n: 1, name: 'preview_ranking', style: 'funny', error: 'the project is busy' },
    { kind: 'reply', n: 2, text: 'Ask again later.' },
  ];
  const m = await sent(t, events);
  const log = [...m.page.document.querySelectorAll('#chatLog > li')].map((li) => li.textContent).join('\n');
  assert.match(log, /Could not preview a ranking in the Funny style: the project is busy/);
});
