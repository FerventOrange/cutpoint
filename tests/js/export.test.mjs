// Export & Assembly Options: restored from this browser's storage, stored per change, and sent
// with the job; Export clips saves the sequence first.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

const KEY = 'cutpoint.exportOptions';

async function exportTab(t, { stored, raw, routes = {}, project = makeProject() } = {}) {
  t.mock.timers.reset(); // a test may boot twice
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const storage = raw !== undefined ? { [KEY]: raw } : stored ? { [KEY]: JSON.stringify(stored) } : {};
  const page = await boot({ storage, project, routes });
  t.after(() => page.close());
  const d = page.document;
  return {
    page, d,
    $: (s) => d.querySelector(s),
    stored: () => JSON.parse(page.window.localStorage.getItem(KEY) ?? '{}'),
    async change(sel, value) {
      const el = d.querySelector(sel);
      if (el.type === 'checkbox') el.checked = value; else el.value = value;
      el.dispatchEvent(new page.window.Event('change', { bubbles: true }));
      await page.settle();
    },
  };
}

test('stored options come back on a new visit; anything unreadable or unknown falls back to the default', async (t) => {
  const x = await exportTab(t, { stored: { vertical: true, loudnorm: false, transition: 'fade', size: '1280x720', order: 'bogus', speech_aware: 'yes' } });
  assert.equal(x.$('#optVertical').checked, true);
  assert.equal(x.$('#optLoudnorm').checked, false);
  assert.equal(x.$('#optTransition').value, 'fade');
  assert.equal(x.$('#optSize').value, '1280x720');
  assert.equal(x.$('#optOrder').value, '', 'a value the select does not offer is dropped');
  assert.equal(x.$('#optSpeech').checked, true, 'a non-boolean is ignored');
});

test('corrupt storage gives the defaults instead of breaking the page', async (t) => {
  const x = await exportTab(t, { raw: '{not json' });
  assert.equal(x.$('#optVertical').checked, false);
  assert.equal(x.$('#optTransition').value, 'cut');
  assert.equal(x.$('#optSpeech').checked, true);
  assert.equal(x.$('#candList').children.length, 3, 'the workspace still booted');
});

test('the old Smart edges box sets both Speech-aware cuts and Smart tails', async (t) => {
  const x = await exportTab(t, { stored: { smart_edges: false } });
  assert.equal(x.$('#optSpeech').checked, false);
  assert.equal(x.$('#optTail').checked, false);
  x.page.close();
  const y = await exportTab(t, { stored: { smart_edges: false, speech_aware: true } });
  assert.equal(y.$('#optSpeech').checked, true, 'the newer key wins');
  assert.equal(y.$('#optTail').checked, false);
});

test('a change is stored on its own: only changed options are kept, so the other defaults can follow', async (t) => {
  const x = await exportTab(t);
  assert.deepEqual(x.stored(), {});
  await x.change('#optVertical', true);
  await x.change('#optTransition', 'fade');
  assert.deepEqual(x.stored(), { vertical: true, transition: 'fade' });
});

test('Link Transitions: a video fade makes the audio transition a crossfade, a cut makes it a cut', async (t) => {
  const x = await exportTab(t);
  assert.ok(x.$('#optLink').checked);
  await x.change('#optTransition', 'fade');
  assert.equal(x.$('#optAudioTransition').value, 'crossfade');
  await x.change('#optTransition', 'cut');
  assert.equal(x.$('#optAudioTransition').value, 'cut');
});

test('Export clips saves the unsaved sequence first, then starts the job with the chosen options', async (t) => {
  const order = [];
  const x = await exportTab(t, {
    routes: {
      'PUT /api/projects/p1/review': () => { order.push('save'); return {}; },
      'POST /api/projects/p1/jobs': (req) => { order.push(['job', req.body]); return { id: 'j1', status: 'running' }; },
      'GET /api/jobs': { jobs: [] },
    },
  });
  x.$('#candList li[data-id=h001] [data-act=toggle]').click();
  await x.page.settle();
  await x.change('#optVertical', true);
  x.$('#clipsBtn').click();
  await x.page.settle();
  assert.equal(order[0], 'save');
  const [, body] = order[1];
  assert.equal(body.action, 'clips');
  assert.equal(body.picks, 'sequence');
  assert.equal(body.vertical, true);
  assert.equal(x.$('#saveState').textContent, 'Sequence saved');
});

test('Export clips is blocked with an invalid number, and no job is started', async (t) => {
  const x = await exportTab(t, {
    routes: {
      'PUT /api/projects/p1/review': {},
      'POST /api/projects/p1/jobs': { id: 'j1', status: 'running' },
    },
  });
  const jobPosts = () => x.page.requests.filter((r) => r.method === 'POST' && r.path === '/api/projects/p1/jobs').length;
  x.$('#candList li[data-id=h001] [data-act=toggle]').click();
  await x.page.settle();
  await x.change('#optTailMin', '-3');
  x.$('#clipsBtn').click();
  await x.page.settle();
  assert.equal(jobPosts(), 0, 'the invalid number blocks the export');
  // control: the same flow with a valid value does start the job
  await x.change('#optTailMin', '1');
  x.$('#clipsBtn').click();
  await x.page.settle();
  assert.equal(jobPosts(), 1);
  x.page.assertNoUnhandled();
});

test('One-Click Edit asks before replacing an earlier preset run and sends force only after Replace', async (t) => {
  const jobs = [];
  const project = { ...makeProject(), preset_taken: ['clips'] };
  const x = await exportTab(t, {
    project,
    routes: { 'POST /api/projects/p1/jobs': (req) => { jobs.push(req.body); return { id: 'j1', status: 'running' }; } },
  });
  const dlg = x.$('#confirmDialog');
  const answer = async (value) => {
    dlg.returnValue = value;
    dlg.removeAttribute('open');
    dlg.dispatchEvent(new x.page.window.Event('close'));
    await x.page.settle();
  };
  x.$('#makeBtn').click();
  await x.page.settle();
  assert.ok(dlg.hasAttribute('open'));
  await answer('cancel');
  assert.equal(jobs.length, 0);

  x.$('#makeBtn').click();
  await x.page.settle();
  await answer('ok');
  assert.equal(jobs.length, 1);
  assert.equal(jobs[0].action, 'edit');
  assert.equal(jobs[0].force, true);

  // nothing earlier in the folder: no question, no force
  x.page.close();
  const y = await exportTab(t, { routes: { 'POST /api/projects/p1/jobs': (req) => { jobs.push(req.body); return { id: 'j2', status: 'running' }; } } });
  y.$('#makeBtn').click();
  await y.page.settle();
  assert.equal(jobs.length, 2);
  assert.equal(jobs[1].force, undefined);
});

test('Color: sent with Export clips, remembered per project, Strength only when graded and below 100%', async (t) => {
  const jobs = [];
  const routes = {
    'PUT /api/projects/p1/review': {},
    'POST /api/projects/p1/jobs': (req) => { jobs.push(req.body); return { id: `j${jobs.length}`, status: 'running' }; },
  };
  const x = await exportTab(t, { routes });
  x.$('#candList li[data-id=h001] [data-act=toggle]').click();
  await x.page.settle();
  const exportClips = async () => { x.$('#clipsBtn').click(); await x.page.settle(); return jobs.at(-1); };
  const slide = async (sel, value) => {
    x.$(sel).value = String(value);
    x.$(sel).dispatchEvent(new x.page.window.Event('input', { bubbles: true }));
    await x.page.settle();
  };

  const off = await exportClips();
  assert.equal(off.color, 'off');
  assert.equal(off.color_strength, undefined);
  await slide('#optColor', 1); // Auto
  let body = await exportClips();
  assert.equal(body.color, 'auto');
  assert.equal(body.color_strength, undefined, 'full strength is the CLI default');
  await slide('#optColorStrength', 40);
  body = await exportClips();
  assert.equal(body.color_strength, 0.4);
  assert.deepEqual(x.stored().color_by_project, { p1: 'auto' });
});
