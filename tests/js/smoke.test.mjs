import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject, reply } from './harness.mjs';

test('app.js boots against a mocked project and renders its moments', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] }); // pollJobs re-arms a timer; keep it inert
  const page = await boot();
  t.after(() => page.close());

  const cards = [...page.document.querySelectorAll('#candList li.cand')];
  assert.equal(cards.length, 3);
  assert.deepEqual(cards.map((c) => c.dataset.id), ['h001', 'h002', 'h003']);
  assert.match(cards[0].textContent, /Big laugh/);
  assert.equal(page.document.querySelector('#momentsCount').textContent, '3');
  assert.equal(page.document.querySelector('#projectName').textContent, 'Project One');
  page.assertNoUnhandled();
});

test('no projects: boots to the empty workspace', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project: null });
  t.after(() => page.close());
  assert.equal(page.document.querySelectorAll('#candList li.cand').length, 0);
  page.assertNoUnhandled();
});

for (const hash of ['#%', '#%E0']) {
  test(`a malformed hash (${hash}) still opens the first project and polls jobs (#1085, A283)`, async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const page = await boot({ hash });
    t.after(() => page.close());
    assert.equal(page.document.querySelector('#projectName').textContent, 'Project One');
    assert.ok(page.requests.some((r) => r.method === 'GET' && r.path === '/api/jobs'), 'pollJobs started');
    page.assertNoUnhandled();
  });
}

test('an unmocked route is recorded and fails loudly', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ allowUnhandled: true });
  t.after(() => page.close());
  await page.window.fetch('/api/nope');
  assert.deepEqual(page.unhandled, ['GET /api/nope']);
  assert.throws(() => page.assertNoUnhandled(), /unhandled fetch routes/);
});

test('the New project dialog shows the telemetry hint only for the action-cam style', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ routes: { 'GET /api/session': () => ({ styles: ['funny', 'action-cam'], can_reveal: false, can_pick: false }) } });
  t.after(() => page.close());
  const doc = page.document;
  const hint = doc.querySelector('#newStyleHint');
  const style = doc.querySelector('#newStyle');
  doc.querySelector('#projectBtn').click();
  doc.querySelector('#newProjectItem').click();
  assert.deepEqual([...style.options].map((o) => o.value), ['auto', 'funny', 'action-cam']);
  assert.equal(style.value, 'auto'); // Auto is the default
  assert.equal(style.options[0].textContent, 'Auto (detect from footage)');
  assert.equal(hint.hidden, true);
  style.value = 'action-cam';
  style.dispatchEvent(new page.window.Event('change'));
  assert.equal(hint.hidden, false);
  assert.match(hint.textContent, /telemetry/);
  assert.match(hint.textContent, /lite/);
  page.assertNoUnhandled();
});

test('the New project dialog stays on Auto whatever the sources are, and asks the server nothing', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ routes: { 'GET /api/session': () => ({ styles: ['funny', 'action-cam'], can_reveal: false, can_pick: false }) } });
  t.after(() => page.close());
  const doc = page.document;
  const fire = async (el, v) => {
    if (v !== undefined) el.value = v;
    el.dispatchEvent(new page.window.Event('change'));
    await new Promise((r) => setImmediate(r));
  };
  doc.querySelector('#projectBtn').click();
  doc.querySelector('#newProjectItem').click();
  await fire(doc.querySelector('#newSource'), '/v/gopro.mp4');
  await fire(doc.querySelector('#newExclude'), '*DVR*');
  doc.querySelector('#newRecursive').checked = true;
  await fire(doc.querySelector('#newRecursive'));
  assert.equal(doc.querySelector('#newStyle').value, 'auto');
  assert.equal(doc.querySelector('#newStyleHint').hidden, true);
  page.assertNoUnhandled(); // no style suggestion request
});

const EXPIRED = reply(403, { detail: 'missing or wrong session token' }); // ui/app.py: not this run's token

test('each boot has its own module state: a session expired in one page does not reach the next', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const gone = await boot({ routes: { 'GET /api/session': EXPIRED } });
  assert.match(gone.document.querySelector('#banner').textContent, /server restarted or this link expired/);
  gone.close();
  const page = await boot(); // app.js, api.js, picker.js and util.js afresh: no expired switch, no old token
  t.after(() => page.close());
  assert.equal(page.document.querySelector('#banner').hidden, true);
  assert.equal(page.document.querySelector('#projectName').textContent, 'Project One');
});

test('an expired session stops every poll: the banner says so once and no timer asks the server again', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  // which pollers run, by name, once the mocked clock fires them
  const fired = [];
  const mocked = globalThis.setTimeout;
  globalThis.setTimeout = (fn, ms, ...args) => mocked((...a) => { fired.push(fn.name); fn(...a); }, ms, ...args);
  const base = makeProject();
  // the source builds its proxy: the page polls the project every 2 s; the job poll fails once, then expires
  const project = () => structuredClone({ ...base, sources: [{ ...base.sources[0], status: 'preparing', progress: 0.1, windows: [], pending: [] }] });
  let jobs = reply(500, { detail: 'busy' });
  const page = await boot({ project: project(), routes: {
    'GET /api/projects/p1': () => project(),
    'POST /api/projects/p1/sources/s1/prepare': {},
    'GET /api/jobs': () => jobs,
  } });
  t.after(() => page.close());
  const $ = (sel) => page.document.querySelector(sel);
  assert.equal($('#jobStatus').textContent, 'busy', 'the failed job poll is shown');
  const step = async () => { t.mock.timers.tick(1000); await page.settle(); };
  await step();
  await step();
  assert.ok(fired.includes('refreshProject'), 'the project is polled while the proxy builds');
  jobs = EXPIRED;
  for (let i = 0; i < 10 && $('#banner').hidden; i++) await step();
  assert.match($('#banner').textContent, /server restarted or this link expired/);
  assert.equal($('#jobStatus').textContent, '', 'the earlier failure gives way to the banner');
  const asked = page.requests.length;
  fired.length = 0;
  for (let i = 0; i < 60; i++) await step();
  assert.deepEqual(fired.filter((n) => n === 'refreshProject' || n === 'pollJobs'), [], 'no poll ran after the session ended');
  assert.equal(page.requests.length, asked);
  assert.match($('#banner').textContent, /server restarted or this link expired/, 'the message stays');
});

test('the system file dialog: Choose video asks the server near the last path, adds the pick as a line, and a cancel or failure adds nothing', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const answers = [];
  let release;
  const page = await boot({ routes: {
    'GET /api/session': { styles: ['funny'], can_reveal: false, can_pick: true },
    'POST /api/pick': () => new Promise((r) => { release = r; }).then(() => answers.shift()),
  } });
  t.after(() => page.close());
  const $ = (sel) => page.document.querySelector(sel);
  $('#projectBtn').click();
  $('#newProjectItem').click();
  const [file, folder] = [$('#newDialog [data-pick=file]'), $('#newDialog [data-pick=folder]')];
  assert.equal(file.hidden, false, 'shown when the server can open a dialog');
  $('#newSource').value = '/v/first.mp4\n  ';
  const pick = async (btn, answer) => {
    answers.push(answer);
    btn.click();
    await page.settle();
    const disabled = [file, folder].every((b) => b.disabled);
    folder.click(); // no second dialog while one is open
    release();
    await page.settle();
    return disabled;
  };
  assert.equal(await pick(file, { path: '/v/second.mp4' }), true, 'every pick button waits while the dialog is open');
  const asks = page.requests.filter((r) => r.path === '/api/pick');
  assert.deepEqual(asks.map((r) => r.query), [`?${new URLSearchParams({ kind: 'file', start: '/v/first.mp4' })}`]);
  assert.equal($('#newSource').value, '/v/first.mp4\n/v/second.mp4');
  assert.equal($('#newName').value, 'v', 'the name follows, as if typed: the folder both paths share');
  assert.ok([file, folder].every((b) => !b.disabled));

  await pick(folder, { path: null }); // cancelled
  assert.equal($('#newSource').value, '/v/first.mp4\n/v/second.mp4');
  answers.push(reply(500, { detail: 'No display to open a dialog on.' }));
  file.click();
  release();
  await page.settle();
  assert.equal($('#newError').textContent, 'No display to open a dialog on.');
  assert.equal($('#newSource').value, '/v/first.mp4\n/v/second.mp4');
  assert.equal(page.requests.filter((r) => r.path === '/api/pick').length, 3);
});
