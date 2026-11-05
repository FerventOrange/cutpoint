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
  assert.deepEqual([...style.options].map((o) => o.value), ['funny', 'action-cam']);
  assert.equal(hint.hidden, true);
  style.value = 'action-cam';
  style.dispatchEvent(new page.window.Event('change'));
  assert.equal(hint.hidden, false);
  assert.match(hint.textContent, /telemetry/);
  assert.match(hint.textContent, /lite/);
  page.assertNoUnhandled();
});

test('sources with telemetry preselect action-cam unless the user already chose a style', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const asked = [];
  const page = await boot({
    routes: {
      'GET /api/session': () => ({ styles: ['funny', 'action-cam'], can_reveal: false, can_pick: false }),
      'POST /api/projects/suggest-style': (req) => (asked.push(req.body), { style: 'action-cam' }),
    },
  });
  t.after(() => page.close());
  const doc = page.document;
  const source = doc.querySelector('#newSource');
  const style = doc.querySelector('#newStyle');
  const typeSource = async (v) => {
    source.value = v;
    source.dispatchEvent(new page.window.Event('change'));
    await new Promise((r) => setImmediate(r));
  };
  doc.querySelector('#projectBtn').click();
  doc.querySelector('#newProjectItem').click();
  await typeSource('/v/gopro.mp4');
  assert.deepEqual(asked, [{ sources: ['/v/gopro.mp4'], recursive: false, exclude: [] }]);
  assert.match(doc.querySelector('#newStyleStatus').textContent, /action-cam/);
  assert.equal(style.getAttribute('aria-describedby'), 'newStyleHint');
  assert.equal(style.value, 'action-cam');
  assert.equal(doc.querySelector('#newStyleHint').hidden, false);

  doc.querySelector('#projectBtn').click();
  doc.querySelector('#newProjectItem').click(); // reopening starts over
  assert.equal(style.value, 'funny');
  style.value = 'funny';
  style.dispatchEvent(new page.window.Event('change')); // the user's choice
  await typeSource('/v/gopro.mp4');
  assert.equal(style.value, 'funny');
  assert.equal(asked.length, 1);
  page.assertNoUnhandled();
});

test('the preselect is undone when the sources change to footage without telemetry, and follows recursive and exclude', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const asked = [];
  const page = await boot({
    routes: {
      'GET /api/session': () => ({ styles: ['funny', 'action-cam'], can_reveal: false, can_pick: false }),
      'POST /api/projects/suggest-style': (req) => (asked.push(req.body), { style: req.body.sources[0].includes('gopro') ? 'action-cam' : null }),
    },
  });
  t.after(() => page.close());
  const doc = page.document;
  const style = doc.querySelector('#newStyle');
  const fire = async (el, v) => {
    if (v !== undefined) el.value = v;
    el.dispatchEvent(new page.window.Event('change'));
    await new Promise((r) => setImmediate(r));
  };
  doc.querySelector('#projectBtn').click();
  doc.querySelector('#newProjectItem').click();
  await fire(doc.querySelector('#newSource'), '/v/gopro.mp4');
  assert.equal(style.value, 'action-cam');
  await fire(doc.querySelector('#newSource'), '/v/plain.mp4');
  assert.equal(style.value, 'funny');
  assert.equal(doc.querySelector('#newStyleHint').hidden, true);
  assert.equal(doc.querySelector('#newStyleStatus').textContent, '');
  await fire(doc.querySelector('#newExclude'), '*DVR*');
  assert.deepEqual(asked.at(-1), { sources: ['/v/plain.mp4'], recursive: false, exclude: ['*DVR*'] });
  doc.querySelector('#newRecursive').checked = true;
  await fire(doc.querySelector('#newRecursive'));
  assert.equal(asked.at(-1).recursive, true);
  page.assertNoUnhandled();
});

test('a late reply for earlier sources does not change the style', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let release;
  const first = new Promise((r) => (release = r));
  let calls = 0;
  const page = await boot({
    routes: {
      'GET /api/session': () => ({ styles: ['funny', 'action-cam'], can_reveal: false, can_pick: false }),
      'POST /api/projects/suggest-style': () => (++calls === 1 ? first.then(() => ({ style: 'action-cam' })) : { style: null }),
    },
  });
  t.after(() => page.close());
  const doc = page.document;
  const source = doc.querySelector('#newSource');
  doc.querySelector('#projectBtn').click();
  doc.querySelector('#newProjectItem').click();
  source.value = '/v/gopro.mp4';
  source.dispatchEvent(new page.window.Event('change'));
  source.value = '/v/plain.mp4';
  source.dispatchEvent(new page.window.Event('change'));
  await new Promise((r) => setImmediate(r));
  release();
  await new Promise((r) => setImmediate(r));
  assert.equal(calls, 2);
  assert.equal(doc.querySelector('#newStyle').value, 'funny');
  page.assertNoUnhandled();
});

test('a busy reply is not "no telemetry": the preselect stays and the request is retried once', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const replies = [{ style: 'action-cam' }, { style: null, busy: true }, { style: 'action-cam' }];
  let calls = 0;
  const page = await boot({
    routes: {
      'GET /api/session': () => ({ styles: ['funny', 'action-cam'], can_reveal: false, can_pick: false }),
      'POST /api/projects/suggest-style': () => replies[calls++] ?? { style: null, busy: true },
    },
  });
  t.after(() => page.close());
  const doc = page.document;
  const source = doc.querySelector('#newSource');
  const style = doc.querySelector('#newStyle');
  const change = async (v) => {
    source.value = v;
    source.dispatchEvent(new page.window.Event('change'));
    await new Promise((r) => setImmediate(r));
  };
  doc.querySelector('#projectBtn').click();
  doc.querySelector('#newProjectItem').click();
  await change('/v/gopro.mp4');
  assert.equal(style.value, 'action-cam');
  await change('/v/gopro2.mp4'); // busy: no undo
  assert.equal(style.value, 'action-cam');
  assert.equal(calls, 2);
  t.mock.timers.tick(1000); // the one retry
  await new Promise((r) => setImmediate(r));
  assert.equal(calls, 3);
  assert.equal(style.value, 'action-cam');
  t.mock.timers.tick(5000); // a busy reply is retried only once
  await new Promise((r) => setImmediate(r));
  assert.equal(calls, 3);
  page.assertNoUnhandled();
});
