// AI Chat: a turn is sent with the working sequence and polled; the assistant only proposes, and a
// proposal is previewed until Apply (which the server may refuse) or Discard.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject, reply } from './harness.mjs';

const A = '/api/projects/p1/assistant';
const seg = (n, start, label) => ({ id: `h00${n}`, source: '/videos/a.mp4', start, end: start + 8, label });
const AFTER = [seg(1, 10, 'Big laugh'), seg(2, 100, 'Fall')];
const diff = (after, before = {}) => after.map((s, i) => ({
  id: s.id, label: s.label, changes: ['added'], before: before[s.id] ?? null, after: { pos: i + 1, start: s.start, end: s.end, label: s.label },
}));
const proposal = (over = {}) => ({ kind: 'proposal', n: 1, id: 'pr1', summary: 'Add the two best moments', after: AFTER, diff: diff(AFTER), base_key: 'bk', suggestions: [], ...over });
const provider = { name: 'ollama', label: 'Ollama · qwen', local: true };

async function chatPage(t, { assistant = {}, routes = {}, project } = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({
    project,
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
  assert.deepEqual(post.body, { segments: [], base_key: 'bk', color: { mode: 'off', strength: 1 } });
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

// ---- color op (#529) ----

const COLOR_ENTRY = (over = {}) => ({
  id: 'e2', proposal: 'pr2', summary: 'Grade the dull footage', after: [], diff: [], applied: '2026-11-03T10:00:00',
  color: { before: { mode: 'off', strength: 1 }, after: { mode: 'auto', strength: 0.8 } }, ...over,
});
const colorProposal = (over = {}) => proposal({ id: 'pr2', summary: 'Grade the dull footage', after: [], diff: [], color: { mode: 'auto', strength: 0.8 }, ...over });
const sliders = (m) => [m.$('#optColor').value, m.$('#optColorStrength').value];

test('a color op is previewed with the Color it replaces, and the sliders stay until Apply', async (t) => {
  const m = await chatPage(t, { routes: turnRoutes([colorProposal()]) });
  await m.send('fix the dull colors');
  await m.tick(1000);
  assert.equal(m.$('#propPanel').hidden, false);
  assert.equal(m.$('#propColor').hidden, false);
  assert.equal(m.$('#propColor').textContent, 'Color: Auto, strength 80% (now Off)');
  assert.match(m.log().join('\n'), /Color auto, strength 80%/);
  assert.deepEqual(sliders(m), ['0', '100']);

  await m.click('#propDiscard');
  assert.equal(m.$('#propColor').hidden, true);
});

test('a proposal without a color op shows no Color line', async (t) => {
  const m = await proposed(t);
  assert.equal(m.$('#propColor').hidden, true);
});

test('Apply sets the Color sliders and is one undo step; redo sets them again', async (t) => {
  const m = await chatPage(t, {
    routes: { ...turnRoutes([colorProposal()]), [`POST ${A}/proposals/pr2/applied`]: COLOR_ENTRY() },
  });
  await m.send('fix the dull colors');
  await m.tick(1000);
  await m.click('#propApply');
  assert.deepEqual(m.posts('/proposals/pr2/applied')[0].body.color, { mode: 'off', strength: 1 });
  assert.deepEqual(sliders(m), ['1', '80']);
  assert.equal(m.$('#colorStrengthRow').hidden, false, 'Strength shows with a grade');
  assert.equal(m.$('#undoBtn').disabled, false);

  await m.click('#undoBtn');
  assert.deepEqual(sliders(m), ['0', '100']);
  await m.click('#redoBtn');
  assert.deepEqual(sliders(m), ['1', '80']);
  m.page.assertNoUnhandled();
});

test('undoing a change that did not set Color leaves a Color the user chose alone', async (t) => {
  const entry = { id: 'e1', proposal: 'pr1', summary: 'Add', after: AFTER, diff: diff(AFTER), applied: '2026-11-03T10:00:00' };
  const m = await proposed(t, { [`POST ${A}/proposals/pr1/applied`]: entry });
  await m.click('#propApply');
  m.$('#optColor').value = '2'; // the user picks Underwater by hand
  m.$('#optColor').dispatchEvent(new m.page.window.Event('input'));
  await m.click('#undoBtn');
  assert.deepEqual(m.seqIds(), []);
  assert.equal(m.$('#optColor').value, '2');
});

test('Revert of a color change previews the Color it puts back and applying sets it', async (t) => {
  const entry = COLOR_ENTRY();
  const rev = { id: 'revert-e2', summary: 'Revert: Grade the dull footage', after: [], diff: [], color: { mode: 'off', strength: 1 }, base_key: 'k1' };
  const m = await chatPage(t, {
    routes: {
      [`GET ${A}/history`]: { applied: [entry] },
      [`POST ${A}/history/e2/revert`]: rev,
      [`POST ${A}/proposals/revert-e2/applied`]: COLOR_ENTRY({ id: 'e3', proposal: 'revert-e2', color: { before: entry.color.after, after: { mode: 'off', strength: 1 } } }),
    },
  });
  assert.match(m.$('#historyList').textContent, /Color auto, strength 80%/);
  m.$('#optColor').value = '1';
  m.$('#optColor').dispatchEvent(new m.page.window.Event('input'));
  await m.click('#historyList button[data-act=revert]');
  assert.equal(m.$('#propColor').textContent, 'Color: Off (now Auto, strength 100%)');
  await m.click('#propApply');
  assert.equal(m.$('#optColor').value, '0');
});

test('look_for is a step, with the new moments it made', async (t) => {
  const m = await chatPage(t, {
    routes: turnRoutes([
      { kind: 'tool', n: 0, name: 'look_for', query: 'powder spray', found: 4, new: 2, weak: 1 },
      { kind: 'tool', n: 1, name: 'look_for', query: 'sunset', error: 'not available in Cutpoint lite' },
      { kind: 'reply', n: 2, text: 'Done.' },
    ]),
  });
  await m.send('find powder spray');
  await m.tick(1000);
  const log = m.log().join('\n');
  assert.match(log, /Looked through the footage for “powder spray”: 4 found, 2 new, 1 weak/);
  assert.match(log, /Could not look through the footage for “sunset”/);
});

test('look_for guesses are offered as possible windows; the user adds one, which Ctrl+Z undoes', async (t) => {
  const possible = [
    { id: 'lf1', source: '/videos/a.mp4', start: 50, end: 55, label: 'powder spray', confidence: 'strong' },
    { id: 'lf2', source: '/videos/a.mp4', start: 70, end: 75, label: 'powder spray', confidence: 'weak' },
  ];
  const m = await chatPage(t, {
    routes: turnRoutes([
      { kind: 'tool', n: 0, name: 'look_for', query: 'powder spray', found: 2, new: 0, weak: 1, unconfirmed: 2, possible },
      { kind: 'reply', n: 1, text: 'Two possible windows, not confirmed.' },
    ]),
  });
  await m.send('find powder spray');
  await m.tick(1000);
  assert.match(m.log().join('\n'), /2 found, 1 weak, 2 unconfirmed/);
  const rows = [...m.d.querySelectorAll('#chatLog .possible li')];
  assert.equal(rows.length, 2);
  assert.match(rows[0].textContent, /a\.mp4 .*\(strong\)/);
  assert.match(rows[1].textContent, /\(weak\)/);
  assert.match(m.$('#chatLog .possible').textContent, /not confirmed/);
  assert.deepEqual(m.seqIds(), [], 'nothing is added by itself');

  rows[1].querySelector('button').click(); // the user's click is the confirmation
  await m.page.settle();
  assert.deepEqual(m.seqIds(), ['lf2']);
  assert.equal(m.page.requests.filter((r) => r.method === 'POST' && r.path === `${A}/turns`).length, 1);
  const added = [...m.d.querySelectorAll('#chatLog .possible li')][1].querySelector('button');
  assert.equal(added.disabled, true);
  assert.equal(added.textContent, 'In the sequence');
  assert.equal(m.$('#saveBtn').disabled, false);

  await m.click('#undoBtn');
  assert.deepEqual(m.seqIds(), []);
  assert.equal([...m.d.querySelectorAll('#chatLog .possible li')][1].querySelector('button').disabled, false);
});

test('adding a possible window keeps keyboard focus in the list: the next add button, else the message box', async (t) => {
  const w = (n) => ({ id: `lf${n}`, source: '/videos/a.mp4', start: 10 * n, end: 10 * n + 5, label: 'powder spray', confidence: 'weak' });
  const m = await chatPage(t, {
    routes: turnRoutes([
      { kind: 'tool', n: 0, name: 'look_for', query: 'powder spray', found: 2, new: 0, weak: 2, unconfirmed: 2, possible: [w(1), w(2)] },
      { kind: 'reply', n: 1, text: 'Two possible windows.' },
    ]),
  });
  await m.send('find powder spray');
  await m.tick(1000);
  const buttons = () => [...m.d.querySelectorAll('#chatLog .possible-list button')];
  buttons()[0].focus();
  buttons()[0].click();
  await m.page.settle();
  assert.deepEqual(m.seqIds(), ['lf1']);
  assert.equal(m.d.activeElement, buttons()[1], 'the next window to add');
  buttons()[1].click();
  await m.page.settle();
  assert.deepEqual(m.seqIds(), ['lf1', 'lf2']);
  assert.equal(m.d.activeElement, m.$('#chatInput'), 'none left: the message box');
});

// ---- view_moment (#640) ----

test('the consent text says frames go out only once allowed for a project', async (t) => {
  const consent = { needed: true, key: 'assistant-v2:anthropic@api.anthropic.com', host: 'api.anthropic.com' };
  const m = await chatPage(t, { assistant: { provider: { name: 'anthropic', label: 'Anthropic', local: false }, consent } });
  assert.match(m.$('#chatConsentText').textContent, /never video frames unless you allow them for a project, and never audio/);
});

test('a watched moment is a step; one the limit stopped says so', async (t) => {
  const m = await chatPage(t, {
    routes: turnRoutes([
      { kind: 'tool', n: 0, name: 'view_moment', id: 'h001', frames: 6, cached: false },
      { kind: 'tool', n: 1, name: 'view_moment', id: 'h002', error: 'limit' },
      { kind: 'reply', n: 2, text: 'A fish in the first one.' },
    ]),
  });
  await m.send('show me the fish');
  await m.tick(1000);
  const log = m.log().join('\n');
  assert.match(log, /Watched Big laugh \(6 frames\)/);
  assert.match(log, /Did not watch Fall: the limit of moments watched per message was reached/);
});

test('view_moment without frame consent: Allow and ask again allows this project and sends the message again', async (t) => {
  const prov = { name: 'anthropic', label: 'Anthropic', local: false };
  const key = 'frames-v1:anthropic@api.anthropic.com';
  const frames = { needed: true, key, host: 'api.anthropic.com' };
  let turnsPosted = 0;
  let consentCalls = 0;
  let gets = 0;
  const askEvents = [{ kind: 'frames_consent', n: 0, id: 'h001', key, host: 'api.anthropic.com' }, { kind: 'reply', n: 1, text: 'Allow frames to let me look.' }];
  const m = await chatPage(t, {
    routes: {
      // after the first load, the server has the finished turn in chat.json
      [`GET ${A}`]: () => ({
        provider: prov, consent: { needed: false, key: 'a', host: 'api.anthropic.com' }, frames, running: [],
        turns: ++gets === 1 ? [] : [{ id: 't1', user: 'show me the fish', status: 'done', events: askEvents }],
      }),
      [`POST ${A}/turns`]: () => ({ id: `t${++turnsPosted}`, status: 'running', provider: prov }),
      [`GET ${A}/turns/t1`]: {
        status: 'done', next: 2, events: askEvents,
      },
      [`GET ${A}/turns/t2`]: { status: 'done', next: 1, events: [{ kind: 'reply', n: 0, text: 'A fish.' }] },
      [`POST ${A}/frames-consent`]: () => (++consentCalls === 1 ? reply(409, { detail: 'the provider changed' }) : { frames: { ...frames, needed: false } }),
    },
  });
  await m.send('show me the fish');
  await m.tick(1000);
  const box = m.$('#chatLog li.frames-consent');
  assert.ok(box, 'an allow box in the turn');
  assert.match(box.textContent, /Let the assistant watch Big laugh\?/);
  assert.match(box.textContent, /sends api\.anthropic\.com one image of 6 to 9 frames/);
  assert.match(box.textContent, /Nothing was sent/);

  const loads = () => m.page.requests.filter((r) => r.method === 'GET' && r.path === A).length;
  const before = loads();
  await m.click('#chatLog button[data-act=frames-ask]'); // 409: reloads, sends nothing
  assert.equal(loads(), before + 1);
  assert.equal(m.$('#chatError').textContent, 'the provider changed');
  assert.equal(turnsPosted, 1);

  await m.click('#chatLog button[data-act=frames-ask]');
  assert.deepEqual(m.posts('/frames-consent')[1].body, { key });
  assert.equal(turnsPosted, 2);
  assert.equal(m.posts('/turns')[1].body.text, 'show me the fish');
  await m.tick(1000);
  assert.match(m.log().join('\n'), /A fish\./);
  const done = m.$('#chatLog li.frames-consent');
  assert.match(done.textContent, /allowed for api\.anthropic\.com/);
  assert.equal(done.querySelector('button[data-act=frames-allow]'), null);
  m.page.assertNoUnhandled();
});

test('Allow for this project allows without sending again', async (t) => {
  const key = 'frames-v1:anthropic@api.anthropic.com';
  const prov = { name: 'anthropic', label: 'Anthropic', local: false };
  const m = await chatPage(t, {
    assistant: {
      provider: prov,
      frames: { needed: true, key, host: 'api.anthropic.com' },
      turns: [{ id: 't0', user: 'which has the best view?', status: 'done', events: [{ kind: 'frames_consent', n: 0, id: 'h002', key, host: 'api.anthropic.com' }] }],
    },
    routes: { [`POST ${A}/frames-consent`]: { frames: { needed: false, key, host: 'api.anthropic.com' } } },
  });
  await m.click('#chatLog button[data-act=frames-allow]');
  assert.equal(m.posts('/frames-consent').length, 1);
  assert.equal(m.posts('/turns').length, 0);
  assert.equal(m.$('#chatLive').textContent, 'Video frames allowed for this project.');
  assert.equal(m.$('#chatLog button[data-act=frames-ask]').textContent, 'Ask again');
});

test('a rejected attempt is a muted step; lengthened and shortened clips are counted and named', async (t) => {
  const rows = [
    { id: 'h001', label: 'A', changes: ['lengthened'], before: { pos: 1, start: 10, end: 18, label: 'A' }, after: { pos: 1, start: 10, end: 24, label: 'A' } },
    { id: 'h002', label: 'B', changes: ['shortened'], before: { pos: 2, start: 100, end: 108, label: 'B' }, after: { pos: 2, start: 100, end: 104, label: 'B' } },
  ];
  const m = await chatPage(t, {
    routes: turnRoutes([
      { kind: 'rejected', n: 0, text: 'This proposal makes 96 s of the 240 s asked' },
      proposal({ n: 1, diff: rows, after: AFTER }),
    ]),
  });
  await m.send('make it 4 min');
  await m.tick(1000);
  assert.match(m.log().join('\n'), /Not accepted, asked again: This proposal makes 96 s of the 240 s asked/);
  assert.ok(m.$('#chatLog li.msg.step.rejected'));
  assert.match(m.$('#propList').textContent, /Lengthened/);
  assert.match(m.$('#propList').textContent, /Shortened/);
  assert.match(m.log().join('\n'), /1 lengthened, 1 shortened/);
});

test('the sequence asks the server for its exported edges and totals and lists them', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const segs = [seg(1, 10, 'Big laugh'), seg(2, 100, 'Fall')];
  const asked = [];
  const page = await boot({
    project: makeProject({ review: { segments: segs } }),
    routes: {
      'POST /api/projects/p1/edges': (req) => {
        asked.push(req.body);
        return { edges: { h001: { start: 10, end: 20.5 }, h002: { start: 99, end: 108 } } };
      },
    },
  });
  t.after(() => page.close());
  assert.match(page.document.querySelector('#seqTotal').textContent, /^\d+:\d\d$/); // the footer: the total alone, no target set
  t.mock.timers.tick(400);
  await page.settle();
  assert.equal(asked.length, 1);
  assert.deepEqual(asked[0].segments.map((x) => x.id), ['h001', 'h002']);
  assert.equal(asked[0].smart_tail, true);
  assert.equal(asked[0].speech_aware, true);
  const lens = [...page.document.querySelectorAll('#seqList .seq-dur')].map((e) => e.firstChild.textContent);
  assert.deepEqual(lens, ['10.5 s', '9.0 s'], 'lengths as the export cuts them, not the raw 8 s');
});

test('rows the code filled in to reach the asked length are marked Auto-filled', async (t) => {
  const rows = diff(AFTER).map((r, i) => ({ ...r, auto: i === 1 }));
  const m = await chatPage(t, { routes: turnRoutes([proposal({ n: 0, diff: rows, summary: 'Add two. Auto-filled to reach ~4:00: added 1 from your searches; now 236 s.' })]) });
  await m.send('make it 4 min');
  await m.tick(1000);
  const marks = [...m.d.querySelectorAll('#propList .prop-mark.auto')].map((e) => e.textContent);
  assert.deepEqual(marks, ['Auto-filled']);
  assert.match(m.$('#propSummary').textContent, /Auto-filled to reach ~4:00/);
});

test('rows the code moved or added to spread the kinds are marked Auto-spread', async (t) => {
  const rows = diff(AFTER).map((r, i) => ({ ...r, auto: i === 0 ? 'spread' : i === 1 }));
  const m = await chatPage(t, { routes: turnRoutes([proposal({ n: 0, diff: rows, summary: 'Add two. Spread 2 funny clips between the skiing runs.' })]) });
  await m.send('mix in some funny moments');
  await m.tick(1000);
  const marks = [...m.d.querySelectorAll('#propList .prop-mark.auto')].map((e) => e.textContent);
  assert.deepEqual(marks, ['Auto-spread', 'Auto-filled']);
});

// ---- the restyled panel: grouped tool summary and proposal cards ----

const summary = (m) => [...m.d.querySelectorAll('#chatLog .tools .tool-lines > li')].map((li) => li.textContent.trim());

test('tool steps are grouped into counted lines; a failed tool keeps its own named line; attempts retried are counted', async (t) => {
  const tool = (n, name, extra) => ({ kind: 'tool', n, name, ...extra });
  const m = await chatPage(t, {
    routes: turnRoutes([
      tool(0, 'find_moments', { query: 'a', found: 10 }),
      tool(1, 'find_moments', { query: 'b', found: 10 }),
      tool(2, 'find_moments', { query: 'c', found: 10 }),
      tool(3, 'view_moment', { id: 'h001', frames: 6 }),
      tool(4, 'view_moment', { id: 'h002', frames: 6 }),
      tool(5, 'view_moment', { id: 'h003', frames: 6 }),
      tool(6, 'view_moment', { id: 'h001', frames: 6 }),
      tool(7, 'look_for', { query: 'sunset', error: 'not available in Cutpoint lite' }),
      { kind: 'rejected', n: 8, text: 'too short' },
      { kind: 'rejected', n: 9, text: 'too long' },
      { kind: 'reply', n: 10, text: 'Done.' },
    ]),
  });
  await m.send('go');
  await m.tick(1000);
  assert.deepEqual(summary(m), [
    'Searched 3 times · 30 moments found',
    'Watched 4 moments',
    'Could not look through the footage for “sunset”',
    '2 attempts retried',
  ]);
  assert.ok(m.$('#chatLog .tool-lines li.bad'), 'the failed tool is marked');
  assert.ok(m.$('#chatLog li.tools.rejected'));
  const full = [...m.d.querySelectorAll('#chatLog .tool-details li')].map((li) => li.textContent);
  assert.equal(full.length, 10, 'the folded details keep one line per step');
  assert.ok(full.includes('Searched for “b”: 10 found'));
  assert.ok(full.includes('Not accepted, asked again: too long'));
  assert.ok(full.some((l) => /Watched Big laugh \(6 frames\)/.test(l)));
});

test('a single search reads "once"; look_for possible matches keep their Add buttons and open details survive the redraw', async (t) => {
  const possible = [
    { id: 'lf1', source: '/videos/a.mp4', start: 50, end: 55, label: 'powder spray', confidence: 'strong' },
    { id: 'lf2', source: '/videos/a.mp4', start: 70, end: 75, label: 'powder spray', confidence: 'weak' },
  ];
  const m = await chatPage(t, {
    routes: turnRoutes([
      { kind: 'tool', n: 0, name: 'find_moments', query: 'x', found: 1 },
      { kind: 'tool', n: 1, name: 'look_for', query: 'powder spray', found: 2, possible },
      { kind: 'reply', n: 2, text: 'Two possible windows.' },
    ]),
  });
  await m.send('find powder spray');
  await m.tick(1000);
  assert.equal(summary(m)[0], 'Searched once · 1 moment found');
  assert.equal(m.d.querySelectorAll('#chatLog .possible button[data-act="add-possible"]').length, 2);

  const details = m.$('#chatLog details.tool-details');
  assert.equal(details.open, false);
  details.open = true;
  details.dispatchEvent(new m.page.window.Event('toggle'));
  m.$('#chatLog .possible button').click(); // redraws the chat
  await m.page.settle();
  assert.equal(m.$('#chatLog details.tool-details').open, true, 'still open after the redraw');
  assert.deepEqual(m.seqIds(), ['lf1']);
  assert.equal(m.d.querySelectorAll('#chatLog .possible button[data-act="add-possible"]:not(:disabled)').length, 1);
});

const mixedDiff = diff(AFTER).map((r, i) => (i ? { ...r, auto: 'spread' } : r));
const cardPage = (t, { routes = {}, events = [proposal()], assistant = {}, history = [] } = {}) => chatPage(t, {
  assistant: { turns: [{ id: 't0', user: 'earlier', status: 'done', events, next: events.length }], ...assistant },
  routes: { [`GET ${A}/history`]: { applied: history }, ...routes },
});
const card = (m, id = 'pr1') => m.$(`#chatLog [data-pid="${id}"]`);
const acts = (m, id) => [...card(m, id).querySelectorAll('.prop-acts button')].map((b) => b.dataset.act);
const ENTRY = { id: 'e1', proposal: 'pr1', summary: 'Add', after: AFTER, diff: diff(AFTER), applied: '2026-11-01T10:00:00' };

test('the proposal card shows a chip per change (dashed only for auto rows) and a strip of the sequence after it', async (t) => {
  const m = await cardPage(t, { events: [proposal({ diff: mixedDiff })] });
  const chips = [...card(m).querySelectorAll('.chip')];
  assert.deepEqual(chips.map((c) => c.textContent), ['2 added', '1 auto-spread']);
  assert.deepEqual(chips.map((c) => c.classList.contains('auto')), [false, true]);
  assert.equal(card(m).querySelectorAll('.prop-strip .prop-block').length, 2);
  assert.match(card(m).querySelector('.prop-strip').getAttribute('aria-label'), /2 clips/);
  assert.deepEqual(acts(m), ['apply', 'review', 'discard']);
  m.page.close();
  t.mock.timers.reset();

  const plain = await cardPage(t, { events: [proposal()] });
  assert.equal(card(plain).querySelectorAll('.chip.auto').length, 0);
});

test('card Apply on a proposal that is not previewed previews it and applies it; the card then has no Apply or Discard', async (t) => {
  const m = await cardPage(t, { routes: { [`POST ${A}/proposals/pr1/applied`]: ENTRY } });
  assert.equal(m.$('#propPanel').hidden, true, 'loaded from the saved chat: nothing previewed');
  await m.click('#chatLog [data-pid="pr1"] [data-act="apply"]');
  assert.equal(m.posts('/proposals/pr1/applied').length, 1);
  assert.deepEqual(m.seqIds(), ['h001', 'h002']);
  assert.deepEqual(acts(m), ['review']);
  assert.equal(card(m).querySelector('[data-act="review"]').textContent, 'Review again');
  assert.match(card(m).textContent, /Applied/);
});

test('card Apply on the previewed proposal applies it', async (t) => {
  const m = await proposed(t, { [`POST ${A}/proposals/pr1/applied`]: ENTRY });
  assert.equal(m.$('#propPanel').hidden, false);
  await m.click('#chatLog [data-pid="pr1"] [data-act="apply"]');
  assert.equal(m.posts('/proposals/pr1/applied').length, 1);
  assert.deepEqual(m.seqIds(), ['h001', 'h002']);
  assert.equal(m.$('#propPanel').hidden, true);
});

test('Apply failures from the card show in the chat tab', async (t) => {
  const cases = [
    [409, 'sequence moved on', /Can't apply: sequence moved on/],
    [404, 'unknown', /no longer has this proposal/],
  ];
  for (const [status, detail, expected] of cases) {
    const m = await cardPage(t, { routes: { [`POST ${A}/proposals/pr1/applied`]: reply(status, { detail }) } });
    m.$('#tabAssistant').click();
    await m.click('#chatLog [data-pid="pr1"] [data-act="apply"]');
    assert.equal(m.$('#panelAssistant').hidden, false);
    assert.match(m.$('#chatError').textContent, expected);
    assert.deepEqual(m.seqIds(), []);
    assert.ok(card(m).querySelector('[data-act="apply"]'), 'can be retried');
    m.page.close();
    t.mock.timers.reset();
  }
});

test('card Discard drops the previewed proposal; on another it only marks the card, and a new preview undoes that', async (t) => {
  const m = await proposed(t);
  await m.click('#chatLog [data-pid="pr1"] [data-act="discard"]');
  assert.equal(m.$('#propPanel').hidden, true);
  assert.deepEqual(acts(m), ['review'], 'a discarded card is dismissed whether or not it was previewed');
  m.page.close();
  t.mock.timers.reset();

  const o = await cardPage(t, { events: [proposal(), proposal({ n: 2, id: 'pr2', summary: 'Other' })] });
  await o.click('#chatLog [data-pid="pr2"] [data-act="review"]'); // previews pr2
  assert.equal(o.$('#propSummary').textContent, 'Other');
  await o.click('#chatLog [data-pid="pr1"] [data-act="discard"]');
  assert.equal(o.$('#propSummary').textContent, 'Other', 'the previewed one is untouched');
  assert.equal(o.$('#propPanel').hidden, false);
  assert.deepEqual(acts(o, 'pr1'), ['review'], 'dismissed card: Preview stays, no Apply or Discard');
  assert.match(card(o, 'pr1').textContent, /Discarded/);
  assert.equal(o.posts('/proposals/pr1/applied').length, 0);

  await o.click('#chatLog [data-pid="pr1"] [data-act="review"]'); // a new preview un-dismisses
  assert.deepEqual(acts(o, 'pr1'), ['apply', 'review', 'discard']);
  assert.doesNotMatch(card(o, 'pr1').textContent, /Discarded/);
});

test('a discarded card comes back when the chat is loaded again (project load)', async (t) => {
  const consent = { needed: true, key: 'k', host: 'h' };
  const m = await cardPage(t, {
    assistant: { provider: { name: 'anthropic', label: 'Anthropic', local: false }, consent },
    routes: { 'POST /api/assistant/consent': reply(409, { detail: 'changed' }) },
  });
  m.$('#chatLog [data-pid="pr1"] [data-act="discard"]').click();
  await m.page.settle();
  assert.deepEqual(acts(m), ['review']);
  await m.click('#chatConsentOk'); // 409: the assistant is loaded again, as on opening a project
  assert.deepEqual(acts(m), ['apply', 'review', 'discard']);
});

test('an applied proposal has no Apply or Discard and keeps Review again', async (t) => {
  const m = await cardPage(t, { history: [ENTRY] });
  assert.deepEqual(acts(m), ['review']);
  assert.equal(card(m).querySelector('[data-act="review"]').textContent, 'Review again');
});

test('card buttons are disabled while an Apply is in flight', async (t) => {
  let release;
  const gate = new Promise((r) => { release = r; });
  const m = await cardPage(t, { routes: { [`POST ${A}/proposals/pr1/applied`]: async () => { await gate; return ENTRY; } } });
  m.$('#chatLog [data-pid="pr1"] [data-act="apply"]').click();
  await m.page.settle();
  const btn = (a) => card(m).querySelector(`[data-act="${a}"]`);
  assert.equal(btn('apply').disabled, true);
  assert.equal(btn('apply').textContent, 'Applying…');
  assert.equal(btn('discard').disabled, true);
  btn('discard').click();
  await m.page.settle();
  assert.equal(m.$('#propPanel').hidden, false, 'Discard does nothing while applying');
  release();
  await m.page.settle();
  assert.deepEqual(m.seqIds(), ['h001', 'h002']);
});

test('card Apply on a stale base previews it again instead of applying (F1)', async (t) => {
  const m = await cardPage(t, {
    events: [proposal(), proposal({ n: 2, id: 'pr2', summary: 'Other' })],
    routes: { [`POST ${A}/proposals/pr1/applied`]: ENTRY },
  });
  m.$('#tabAssistant').click();
  await m.click('#chatLog [data-pid="pr1"] [data-act="apply"]');
  assert.deepEqual(m.seqIds(), ['h001', 'h002']);
  await m.click('#chatLog [data-pid="pr2"] [data-act="apply"]');
  assert.equal(m.posts('/proposals/pr2/applied').length, 0, 'nothing is sent');
  assert.equal(m.$('#propPanel').hidden, false);
  assert.equal(m.$('#propSummary').textContent, 'Other');
});

test('a refused card Apply on a proposal that was not previewed leaves the cards right, and the focus on Apply (F2, F9)', async (t) => {
  const m = await cardPage(t, {
    events: [proposal(), proposal({ n: 2, id: 'pr2', summary: 'Other' })],
    routes: { [`POST ${A}/proposals/pr1/applied`]: reply(409, { detail: 'nope' }) },
  });
  await m.click('#chatLog [data-pid="pr2"] [data-act="review"]');
  m.$('#tabAssistant').click();
  await m.page.settle();
  m.$('#chatLog [data-pid="pr1"] [data-act="apply"]').focus(); // as a real click does
  m.$('#chatLog [data-pid="pr1"] [data-act="apply"]').click();
  await m.page.settle();
  assert.equal(card(m, 'pr1').classList.contains('previewing'), true);
  assert.equal(card(m, 'pr2').classList.contains('previewing'), false);
  assert.match(m.$('#chatError').textContent, /Can't apply: nope/);
  assert.equal(m.d.activeElement, card(m, 'pr1').querySelector('[data-act="apply"]'));
});

test('Discard clears the failure alert (F3)', async (t) => {
  const m = await cardPage(t, { routes: { [`POST ${A}/proposals/pr1/applied`]: reply(409, { detail: 'nope' }) } });
  m.$('#tabAssistant').click();
  await m.click('#chatLog [data-pid="pr1"] [data-act="apply"]');
  assert.match(m.$('#chatError').textContent, /nope/);
  await m.click('#chatLog [data-pid="pr1"] [data-act="discard"]');
  assert.equal(m.$('#chatError').textContent, '');
});

test('a tool group that grows across polls keeps one stable item (F5)', async (t) => {
  let call = 0;
  const batches = [
    [{ kind: 'tool', n: 0, name: 'find_moments', query: 'a', found: 2 }],
    [{ kind: 'tool', n: 1, name: 'get_moment', id: 'h001' }],
  ];
  const m = await chatPage(t, { routes: {
    [`POST ${A}/turns`]: { id: 't1', status: 'running', provider },
    [`GET ${A}/turns/t1`]: () => ({ status: 'running', events: batches[call++] ?? [], next: call }),
  } });
  await m.send('go');
  await m.tick(1000);
  const ids = () => [...m.d.querySelectorAll('#chatLog > li.tools')].map((li) => li.dataset.id);
  const first = ids();
  const lines = () => m.d.querySelectorAll('#chatLog .tool-lines > li').length;
  assert.equal(lines(), 1);
  await m.tick(1000);
  assert.deepEqual(ids(), first);
  assert.equal(lines(), 2);
});

test('the status dot is not green when the provider cannot be used (F6)', async (t) => {
  const prov = { name: 'anthropic', label: 'Anthropic', local: false };
  const m = await chatPage(t, { assistant: { provider: prov, consent: { needed: false, missing_key: 'Add the key' } } });
  assert.equal(m.$('#chatDot').dataset.state, 'bad');
  m.page.close();
  t.mock.timers.reset();
  const o = await chatPage(t, { assistant: { provider: prov, consent: { needed: true, key: 'k', host: 'h' } } });
  assert.equal(o.$('#chatDot').dataset.state, 'wait');
});

test('Apply clears an earlier failure alert', async (t) => {
  let calls = 0;
  const m = await cardPage(t, {
    events: [proposal(), proposal({ n: 2, id: 'pr2', summary: 'Other' })],
    routes: { [`POST ${A}/proposals/pr1/applied`]: () => (calls++ ? ENTRY : reply(409, { detail: 'nope' })) },
  });
  m.$('#tabAssistant').click();
  await m.click('#chatLog [data-pid="pr1"] [data-act="apply"]');
  assert.match(m.$('#chatError').textContent, /nope/);
  await m.click('#propApply'); // the retry from the Sequence tab's button works
  assert.equal(m.$('#chatError').textContent, '');
  m.$('#chatError').textContent = 'an earlier failure';
  await m.click('#chatLog [data-pid="pr2"] [data-act="apply"]'); // stale now: previewed, not sent
  assert.equal(m.posts('/proposals/pr2/applied').length, 0);
  assert.equal(m.$('#chatError').textContent, '');
});

test('card Apply of a color-only proposal on a non-empty sequence posts directly (F15)', async (t) => {
  const rows = AFTER.map((s, i) => ({
    id: s.id, label: s.label, changes: [], before: { pos: i + 1, start: s.start, end: s.end, label: s.label }, after: { pos: i + 1, start: s.start, end: s.end, label: s.label },
  }));
  const events = [proposal({ diff: rows, color: { mode: 'auto', strength: 0.8 } })];
  const m = await chatPage(t, {
    project: makeProject({ review: { segments: AFTER } }),
    assistant: { turns: [{ id: 't0', user: 'grade it', status: 'done', events, next: 1 }] },
    routes: {
      [`GET ${A}/history`]: { applied: [] },
      [`POST ${A}/proposals/pr1/applied`]: { id: 'e1', proposal: 'pr1', summary: 'Grade', after: AFTER, diff: rows, applied: '2026-11-01T10:00:00', color: { before: { mode: 'off', strength: 1 }, after: { mode: 'auto', strength: 0.8 } } },
    },
  });
  await m.click('#chatLog [data-pid="pr1"] [data-act="apply"]');
  assert.equal(m.posts('/proposals/pr1/applied').length, 1);
});
