import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot } from './harness.mjs';

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
