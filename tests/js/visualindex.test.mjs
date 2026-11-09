// Settings > Analysis > the frame index for AI Chat's visual search (#798): the box shows
// config.json's visual_index, a change is saved with PUT /api/settings, an unchanged one sends nothing.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

const settings = (over = {}) => ({ projects_dir: '/projects', projects_dir_locked: false, config_file: '/c.json', compress_peaks_set: false, visual_index: false, ...over });

async function open(t, over = {}, put = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({
    project: makeProject(),
    routes: { 'GET /api/settings': settings(over), 'PUT /api/settings': (req) => settings({ ...over, ...req.body }), ...put },
  });
  t.after(() => page.close());
  const d = page.document;
  d.querySelector('#settingsBtn').click();
  await page.settle();
  return { page, d, box: d.querySelector('#setVisualIndex'), puts: () => page.requests.filter((r) => r.method === 'PUT') };
}

test('off by default, with the cost and what it is for in the hint', async (t) => {
  const m = await open(t);
  assert.equal(m.box.checked, false);
  const hint = m.d.querySelector('#setVisualIndexHint').textContent;
  assert.match(hint, /about 373 s and 25 MB per hour/);
  assert.match(hint, /gaming, funny and commentary/);
  assert.match(hint, /changes no ranking/);
  assert.match(hint, /Needs a CUDA GPU, the full install and its model files \(cutpoint setup\); skipped otherwise/);
  assert.match(hint, /Action, sport and action-cam projects have it whenever a CUDA GPU is present/);
});

test('the box shows the saved visual_index', async (t) => {
  const m = await open(t, { visual_index: true });
  assert.equal(m.box.checked, true);
});

test('Save sends visual_index only when it changed', async (t) => {
  const m = await open(t);
  m.d.querySelector('#settingsSave').click();
  await m.page.settle();
  assert.equal(m.puts().length, 0, 'unchanged: nothing sent');

  m.d.querySelector('#settingsBtn').click();
  await m.page.settle();
  m.box.checked = true;
  m.d.querySelector('#settingsSave').click();
  await m.page.settle();
  assert.deepEqual(m.puts().map((r) => r.body), [{ visual_index: true }]);
});
