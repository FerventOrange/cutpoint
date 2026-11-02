// AI Chat: a turn is sent with the working sequence and polled; the assistant only proposes, and a
// proposal is previewed until Apply (which the server may refuse) or Discard.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, reply } from './harness.mjs';

const A = '/api/projects/p1/assistant';
const seg = (n, start, label) => ({ id: `h00${n}`, source: '/videos/a.mp4', start, end: start + 8, label });
const AFTER = [seg(1, 10, 'Big laugh'), seg(2, 100, 'Fall')];
const diff = (after, before = {}) => after.map((s, i) => ({
  id: s.id, label: s.label, changes: ['added'], before: before[s.id] ?? null, after: { pos: i + 1, start: s.start, end: s.end, label: s.label },
}));
const proposal = (over = {}) => ({ kind: 'proposal', n: 1, id: 'pr1', summary: 'Add the two best moments', after: AFTER, diff: diff(AFTER), base_key: 'bk', suggestions: [], ...over });
const provider = { name: 'ollama', label: 'Ollama · qwen', local: true };

async function chatPage(t, { assistant = {}, routes = {} } = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({
    routes: {
      [`GET ${A}`]: { provider, consent: null, turns: [], running: [], ...assistant },
      ...routes,
    },
  });
  t.after(() => page.close());
  const d = page.document;
  const m = {
    page, d,
    $: (s) => d.querySelector(s),
    posts: (suffix) => page.requests.filter((r) => r.method === 'POST' && r.path === A + suffix),
    seqIds: () => [...d.querySelectorAll('#seqList > li')].map((li) => li.dataset.id),
    log: () => [...d.querySelectorAll('#chatLog > li')].map((li) => li.textContent.trim()),
    async send(text) {
      m.$('#chatInput').value = text;
      m.$('#chatForm').dispatchEvent(new page.window.Event('submit', { cancelable: true }));
      await page.settle();
    },
    async click(sel) { m.$(sel).click(); await page.settle(); },
    async tick(ms) { t.mock.timers.tick(ms); await page.settle(); },
  };
  return m;
}

/** A server whose one turn yields ``events`` on the first poll. */
function turnRoutes(events, status = 'done', extra = {}) {
  return {
    [`POST ${A}/turns`]: { id: 't1', status: 'running', provider },
    [`GET ${A}/turns/t1`]: () => ({ status, events, next: events.length, ...extra }),
  };
}

test('a message is sent with the working sequence, the reply is polled in, and Send waits meanwhile', async (t) => {
  const m = await chatPage(t, { routes: turnRoutes([{ kind: 'reply', n: 1, text: 'Nothing funny after minute 5.' }]) });
  assert.equal(m.$('#chatSend').disabled, false);
  await m.send('  find jokes ');
  const [post] = m.posts('/turns');
  assert.deepEqual(post.body, { text: 'find jokes', segments: [] });
  assert.equal(m.$('#chatInput').value, '');
  assert.equal(m.$('#chatSend').disabled, true, 'one turn at a time');

  await m.send('and another'); // while it is answering
  assert.equal(m.posts('/turns').length, 1);
  assert.match(m.$('#chatError').textContent, /still answering/);

  await m.tick(1000);
  assert.ok(m.log().some((l) => l.includes('Nothing funny after minute 5.')));
  assert.equal(m.$('#chatSend').disabled, false);
  assert.equal(m.$('#chatError').textContent, '');
  m.page.assertNoUnhandled();
});

test('a turn the server no longer has ends as failed instead of polling forever', async (t) => {
  const m = await chatPage(t, {
    routes: { ...turnRoutes([]), [`GET ${A}/turns/t1`]: () => reply(404, { detail: 'gone' }) },
  });
  await m.send('hello');
  await m.tick(1000);
  assert.match(m.log().join('\n'), /server no longer has this reply/);
  const polls = () => m.page.requests.filter((r) => r.path === `${A}/turns/t1`).length;
  const n = polls();
  await m.tick(5000);
  assert.equal(polls(), n);
  assert.equal(m.$('#chatSend').disabled, false);
});

test('Cancel stops the running turn and frees Send', async (t) => {
  const m = await chatPage(t, {
    assistant: { running: [{ id: 't9', user: 'long question', status: 'running', events: [], next: 0 }] },
    routes: {
      [`POST ${A}/turns/t9/cancel`]: { id: 't9', user: 'long question', status: 'cancelled', events: [] },
      [`GET ${A}/turns/t9`]: { status: 'running', events: [], next: 0 },
    },
  });
  assert.equal(m.$('#chatSend').disabled, true, 'a turn already running when the page opens blocks Send');
  assert.ok(m.log().some((l) => l.includes('long question')));

  await m.click('#chatLog button[data-act=cancel]');
  assert.equal(m.posts('/turns/t9/cancel').length, 1);
  assert.equal(m.$('#chatSend').disabled, false);
  assert.equal(m.$('#chatLive').textContent, 'Reply cancelled.');
  const polls = () => m.page.requests.filter((r) => r.path === `${A}/turns/t9`).length;
  const n = polls();
  await m.tick(5000);
  assert.equal(polls(), n, 'no more polling');
});

test('hosted provider: no sending until the user agrees; a stale agreement reloads the provider in use', async (t) => {
  const consent = { needed: true, key: 'anthropic@api.anthropic.com', host: 'api.anthropic.com' };
  let n = 0;
  const m = await chatPage(t, {
    assistant: { provider: { name: 'anthropic', label: 'Anthropic', local: false }, consent },
    routes: { 'POST /api/assistant/consent': () => (++n === 1 ? reply(409, { detail: 'the provider changed' }) : { consent: { ...consent, needed: false } }) },
  });
  assert.equal(m.$('#chatConsent').hidden, false);
  assert.match(m.$('#chatConsentText').textContent, /Anthropic at api\.anthropic\.com/);
  assert.equal(m.$('#chatInput').disabled, true);
  assert.equal(m.$('#chatSend').disabled, true);

  const loads = () => m.page.requests.filter((r) => r.method === 'GET' && r.path === A).length;
  const before = loads();
  await m.click('#chatConsentOk'); // 409: the provider differs from the one shown
  assert.equal(loads(), before + 1);
  assert.equal(m.$('#chatError').textContent, 'the provider changed');

  await m.click('#chatConsentOk');
  assert.deepEqual(m.page.requests.filter((r) => r.path === '/api/assistant/consent')[1].body, { key: consent.key });
  assert.equal(m.$('#chatConsent').hidden, true);
  assert.equal(m.$('#chatInput').disabled, false);
});

test('a missing API key blocks the chat and offers no agreement', async (t) => {
  const m = await chatPage(t, {
    assistant: { provider: { name: 'openai', label: 'OpenAI', local: false }, consent: { needed: true, missing_key: 'Set an OpenAI key in the Models tab', key: 'k', host: 'h' } },
  });
  assert.equal(m.$('#chatConsent').hidden, false);
  assert.equal(m.$('#chatConsentTitle').textContent, 'API key needed');
  assert.equal(m.$('#chatConsentOk').hidden, true);
  assert.equal(m.$('#chatInput').disabled, true);
});

// ---- proposals ----

async function proposed(t, routes = {}) {
  const m = await chatPage(t, { routes: { ...turnRoutes([proposal()]), ...routes } });
  await m.send('add the best two');
  await m.tick(1000);
  return m;
}

test('a proposal is previewed without touching the sequence; Discard drops it', async (t) => {
  const m = await proposed(t);
  assert.equal(m.$('#propPanel').hidden, false);
  assert.equal(m.$('#propSummary').textContent, 'Add the two best moments');
  assert.equal(m.d.querySelectorAll('#propList > li').length, 2);
  assert.match(m.$('#propList').textContent, /Added/);
  assert.deepEqual(m.seqIds(), [], 'the working sequence is unchanged until Apply');
  assert.equal(m.$('#saveBtn').disabled, true);

  await m.click('#propDiscard');
  assert.equal(m.$('#propPanel').hidden, true);
  assert.deepEqual(m.seqIds(), []);
});

test('Apply sends the sequence and base, replaces the sequence, and is one undo step', async (t) => {
  const entry = { id: 'e1', proposal: 'pr1', summary: 'Add the two best moments', after: AFTER, diff: diff(AFTER), applied: '2026-11-01T10:00:00' };
  const m = await proposed(t, { [`POST ${A}/proposals/pr1/applied`]: entry });
  await m.click('#propApply');
  const [post] = m.posts('/proposals/pr1/applied');
  assert.deepEqual(post.body, { segments: [], base_key: 'bk' });
  assert.deepEqual(m.seqIds(), ['h001', 'h002']);
  assert.equal(m.$('#propPanel').hidden, true);
  assert.equal(m.$('#saveBtn').disabled, false, 'the applied change is unsaved work');
  assert.equal(m.$('#undoBtn').disabled, false);
  assert.match(m.$('#chatHistory summary').textContent, /Applied changes \(1\)/);

  await m.click('#undoBtn');
  assert.deepEqual(m.seqIds(), []);
});

test('Apply refused by the server keeps the proposal and the sequence, and says why', async (t) => {
  const cases = [
    [409, 'sequence moved on', /Can't apply: sequence moved on\. Discard it, or ask the assistant again\./],
    [404, 'unknown', /no longer has this proposal/],
  ];
  for (const [status, detail, expected] of cases) {
    const m = await proposed(t, { [`POST ${A}/proposals/pr1/applied`]: reply(status, { detail }) });
    await m.click('#propApply');
    assert.match(m.$('#propMsg').textContent, expected);
    assert.equal(m.$('#propPanel').hidden, false);
    assert.deepEqual(m.seqIds(), []);
    assert.equal(m.$('#propApply').disabled, false, 'can be retried or discarded');
    m.page.close();
    t.mock.timers.reset();
  }
});

test('Revert previews the change\'s before-state; applying a stale revert previews it again on the new sequence', async (t) => {
  const entry = { id: 'e1', proposal: 'pr1', summary: 'Add', after: AFTER, diff: diff(AFTER), applied: '2026-11-01T10:00:00' };
  const revertDiff = AFTER.map((s, i) => ({ id: s.id, label: s.label, changes: ['removed'], before: { pos: i + 1, start: s.start, end: s.end, label: s.label }, after: null }));
  const rev = (key) => ({ id: 'revert-e1', summary: 'Revert: Add', after: [], diff: revertDiff, base_key: key });
  let revertCalls = 0;
  const m = await chatPage(t, {
    assistant: { },
    routes: {
      [`GET ${A}/history`]: { applied: [entry] },
      [`POST ${A}/history/e1/revert`]: () => rev(`k${++revertCalls}`),
      [`POST ${A}/proposals/revert-e1/applied`]: reply(409, { detail: 'The sequence changed; preview it again' }),
    },
  });
  await m.click('#historyList button[data-act=revert]');
  assert.equal(m.$('#propPanel').hidden, false);
  assert.match(m.$('#propList').textContent, /Removed/);
  assert.equal(m.posts('/history/e1/revert')[0].body.segments.length, 0);

  await m.click('#propApply'); // the server says the sequence changed
  assert.equal(revertCalls, 2, 'previewed again from the server');
  assert.equal(m.$('#propPanel').hidden, false);
  assert.match(m.$('#sideLive').textContent, /previewed again/);
});
