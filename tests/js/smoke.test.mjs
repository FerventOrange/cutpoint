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
  assert.match(page.document.querySelector('#momentsCount').textContent, /3 found, funny style/);
  assert.equal(page.document.querySelector('#projectSelect').value, 'p1');
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
