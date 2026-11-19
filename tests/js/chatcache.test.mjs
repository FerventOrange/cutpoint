// The chat log keeps the bubbles of finished turns (A755): it is drawn once a second while a turn runs
// and on every render of the page, and a finished turn is drawn again (its replies parsed as Markdown
// again) only when it or what it shows changed.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { answerConfirms, boot, makeProject } from './harness.mjs';

const A = '/api/projects/p1/assistant';
const provider = { name: 'ollama', label: 'Ollama · qwen', local: true };
const DONE = {
  id: 't0', user: 'what is funny?', status: 'done',
  events: [
    { kind: 'reply', n: 0, text: 'The **fall** at 1:40.' },
    { kind: 'show', n: 1, ids: ['h002'], note: 'The fall' },
  ],
};

async function chatPage(t, routes = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({
    routes: {
      [`GET ${A}`]: { provider, consent: null, turns: [DONE], running: [] },
      [`POST ${A}/turns`]: { id: 't1', status: 'running', provider },
      ...routes,
    },
  });
  t.after(() => page.close());
  const d = page.document;
  return {
    page, d,
    row: (id) => d.querySelector(`#chatLog > li[data-id="${id}"]`),
    async send(text) {
      d.querySelector('#chatInput').value = text;
      d.querySelector('#chatForm').dispatchEvent(new page.window.Event('submit', { cancelable: true }));
      await page.settle();
    },
    async tick(ms) { t.mock.timers.tick(ms); await page.settle(); },
  };
}

test('a finished turn keeps its nodes while another turn runs; the running turn is drawn anew', async (t) => {
  let call = 0;
  const batches = [
    [{ kind: 'tool', n: 0, name: 'find_moments', query: 'jump', found: 1 }],
    [{ kind: 'reply', n: 1, text: 'One *jump*.' }],
  ];
  const m = await chatPage(t, {
    [`GET ${A}/turns/t1`]: () => ({ status: call === 0 ? 'running' : 'done', events: batches[call++] ?? [], next: call }),
  });
  const reply = m.row('t0-0');
  const md = reply.querySelector('.md');
  assert.equal(md.querySelector('strong').textContent, 'fall');
  await m.send('jumps?');
  const pending = m.row('t1-run');
  assert.ok(pending, 'the running turn shows Thinking');
  await m.tick(1000);
  assert.equal(m.row('t0-0'), reply, 'the finished reply is the same node');
  assert.equal(m.row('t0-0').querySelector('.md'), md, 'its Markdown was not parsed again');
  assert.ok(m.row('t1-s0'), 'the running turn shows its new step');
  assert.notEqual(m.row('t1-run'), pending, 'the running turn was drawn again');
  await m.tick(1000);
  assert.equal(m.row('t1-1').querySelector('em').textContent, 'jump');
  assert.equal(m.row('t1-run'), null, 'done');
  assert.equal(m.row('t0-0'), reply);
  // the turn just finished is kept from now on too
  const done = m.row('t1-1');
  m.page.routes[`POST ${A}/turns`] = { id: 't2', status: 'running', provider };
  m.page.routes[`GET ${A}/turns/t2`] = { status: 'running', events: [], next: 0 };
  await m.send('more?');
  await m.tick(1000);
  assert.ok(m.row('t2-run'));
  assert.equal(m.row('t1-1'), done);
  assert.equal(m.row('t0-0'), reply);
  m.page.assertNoUnhandled();
});

test('what a finished turn shows changing draws it again, with its handlers', async (t) => {
  const m = await chatPage(t);
  const show = () => m.row('t0-1').querySelector('[data-act=show]');
  const reply = m.row('t0-0');
  assert.equal(show().getAttribute('aria-pressed'), 'false');
  show().click(); // highlights the moments in the list: the card says so
  await m.page.settle();
  assert.equal(show().getAttribute('aria-pressed'), 'true');
  assert.notEqual(m.row('t0-0'), reply, 'drawn again');
  show().click();
  await m.page.settle();
  assert.equal(show().getAttribute('aria-pressed'), 'false', 'the redrawn button works');
  m.page.assertNoUnhandled();
});

test('opening another project draws its chat, not the kept turns of the one left', async (t) => {
  const P2 = makeProject({ id: 'p2', name: 'Two' });
  const m = await chatPage(t, {
    'GET /api/projects': { projects: [{ id: 'p1', name: 'Project One' }, { id: 'p2', name: 'Two' }] },
    'GET /api/projects/p2': P2,
    // the same turn id with other words: a cache by id alone would show p1's
    'GET /api/projects/p2/assistant': {
      provider, consent: null, running: [],
      turns: [{ ...DONE, events: [{ kind: 'reply', n: 0, text: 'The **jump** at 5:00.' }, DONE.events[1]] }],
    },
    'GET /api/projects/p2/assistant/history': { applied: [] },
    'GET /api/projects/p2/events': { events: [] },
    'GET /api/projects/p2/sources/s1/envelope': { step: 0.05, peaks: [] },
    'POST /api/projects/p2/edges': { edges: {} },
  });
  const reply = m.row('t0-0');
  assert.match(reply.textContent, /fall/);
  m.d.querySelector('#projectBtn').click();
  m.d.querySelector('#projectMenu [data-id=p2]').click();
  await m.page.settle();
  assert.notEqual(m.row('t0-0'), reply);
  assert.match(m.row('t0-0').textContent, /jump at 5:00/);
  m.page.assertNoUnhandled();
});

test('clearing the chat leaves no kept turn behind', async (t) => {
  const m = await chatPage(t, { [`DELETE ${A}/chat`]: {} });
  assert.ok(m.row('t0-0'));
  answerConfirms(m.page);
  m.d.querySelector('#chatClear').click();
  await m.page.settle();
  assert.equal(m.row('t0-0'), null);
  m.page.assertNoUnhandled();
});
