// Export & Assembly Options: restored from this browser's storage, stored per change, and sent
// with the job; Export clips saves the sequence first.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, importPure, makeProject } from './harness.mjs';

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

test('a stored smart_edges (the removed Smart edges box) is ignored', async (t) => {
  const x = await exportTab(t, { stored: { smart_edges: false } });
  assert.equal(x.$('#optSpeech').checked, true);
  assert.equal(x.$('#optTail').checked, true);
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

// ---------- moved from source-text pins in tests/test_ui.py (#1085 A594) ----------

const ADVANCED = ['#optTop', '#optIds', '#optSpeechExtend', '#optSpeechLead', '#optTailMin', '#optTailMax', '#optEventWindow', '#optExtendStart', '#optExtendEnd'];

async function setPicks(x, which) {
  const radio = x.$(which === 'auto' ? '#optPicksAuto' : '#optPicksSeq');
  radio.checked = true;
  radio.dispatchEvent(new x.page.window.Event('change', { bubbles: true }));
  await x.page.settle();
}

/** Type ``value`` into a field as a user does: input, then change. */
async function type(x, sel, value) {
  const el = x.$(sel);
  if (el.type === 'checkbox') el.checked = value; else el.value = value;
  for (const kind of ['input', 'change']) el.dispatchEvent(new x.page.window.Event(kind, { bubbles: true }));
  await x.page.settle();
}

test('advanced export options: hidden until the Settings box shows them, remembered, and Cancel puts the choice back', async (t) => {
  const x = await exportTab(t);
  const panel = x.$('#panelExport');
  for (const sel of ADVANCED) assert.ok(x.$(sel).classList.contains('adv'), sel); // app.css hides .adv off show-adv
  for (const sel of ['#optMaxTotal', '#optOrder', '#optEventTails']) assert.ok(!x.$(sel).classList.contains('adv'), sel);
  assert.ok(x.$('#extendBtn').classList.contains('adv'));
  assert.ok(!x.$('#revealExportBtn').classList.contains('adv'));
  assert.equal(x.$('#setAdvExport').checked, false);
  assert.ok(!panel.classList.contains('show-adv'));

  await x.change('#setAdvExport', true);
  assert.ok(panel.classList.contains('show-adv'), 'previews on change');
  assert.equal(x.page.window.localStorage.getItem('cutpoint.advancedExport'), '1');
  await x.change('#setAdvExport', false);

  x.$('#settingsBtn').click();
  await x.page.settle();
  await x.change('#setAdvExport', true);
  x.$('#settingsCancel').click();
  await x.page.settle();
  assert.ok(!panel.classList.contains('show-adv'), 'Cancel restores it');
  assert.equal(x.$('#setAdvExport').checked, false);

  x.page.close();
  t.mock.timers.reset();
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ storage: { 'cutpoint.advancedExport': '1' } });
  t.after(() => page.close());
  assert.ok(page.document.querySelector('#panelExport').classList.contains('show-adv'), 'a reload keeps it on');
});

test('a hidden advanced field is still sent, is named under the options, and one out of range shows the advanced options', async (t) => {
  const jobs = [];
  const x = await exportTab(t, { routes: { 'POST /api/projects/p1/jobs': (req) => { jobs.push(req.body); return { id: 'j1', status: 'running' }; } } });
  await setPicks(x, 'auto');
  await type(x, '#optTop', '5');
  await type(x, '#optMaxTotal', '60');
  assert.equal(x.$('#advInUse').hidden, false);
  assert.match(x.$('#advInUse').textContent, /^Advanced options in use: Top moments\./);
  await x.change('#setAdvExport', true);
  assert.equal(x.$('#advInUse').hidden, true, 'shown options need no note');
  assert.equal(x.$('#advInUse').textContent, '');
  await x.change('#setAdvExport', false);
  assert.equal(x.$('#advInUse').hidden, false, 'hidden again: the note is back');
  x.$('#clipsBtn').click();
  await x.page.settle();
  assert.equal(jobs.length, 1);
  assert.equal(jobs[0].picks, 'auto');
  assert.equal(jobs[0].top, 5, 'hidden, still sent');
  assert.equal(jobs[0].max_total, 60);
  await type(x, '#optTop', '');
  await type(x, '#optIds', 'h001, h003');
  x.$('#clipsBtn').click();
  await x.page.settle();
  assert.deepEqual(jobs[1].ids, ['h001', 'h003']);
  assert.ok(!('top' in jobs[1]));

  await type(x, '#optIds', '');
  await type(x, '#optTop', '99999');
  x.$('#clipsBtn').click();
  await x.page.settle();
  assert.equal(jobs.length, 2, 'out of range: no job');
  assert.ok(x.$('#panelExport').classList.contains('show-adv'), 'shown so the browser can point at it');
});

const withClips = () => makeProject({ clips: { dir: 'clips', clips: [{ file: '001_big-laugh.mp4', source: '/videos/a.mp4', start: 10, end: 18 }] } });

test('Extend export is off, with its reason, until there is something to extend; it sends the padding as start and end', async (t) => {
  const jobs = [];
  const x = await exportTab(t, { project: withClips(), routes: { 'POST /api/projects/p1/jobs': (req) => { jobs.push(req.body); return { id: 'j1', status: 'running' }; } } });
  const btn = x.$('#extendBtn');
  assert.equal(btn.getAttribute('aria-describedby').split(' ').includes('extendWhy'), true);
  await type(x, '#optSpeech', false);
  const reason = 'Set Extend start or Extend end above 0, or turn Speech-aware cuts on.';
  assert.equal(btn.disabled, true);
  assert.equal(btn.title, reason);
  assert.equal(x.$('#extendWhy').textContent, reason, 'screen readers get the reason too');
  x.$('#optExtendStart').value = '1.5';
  x.$('#optExtendStart').dispatchEvent(new x.page.window.Event('input', { bubbles: true }));
  await x.page.settle();
  assert.equal(btn.disabled, false, 'follows the field as it is typed');
  assert.equal(x.$('#extendWhy').textContent, '');
  btn.click();
  await x.page.settle();
  const body = jobs.at(-1);
  assert.equal(body.action, 'extend');
  assert.equal(body.extend_start, 1.5);
  assert.equal(body.speech_aware, false);
  assert.ok(!('extend' in body), 'no single padding');
  assert.ok(!('color' in body), 'Extend export sends no Color');

  x.page.close();
  const y = await exportTab(t);
  assert.equal(y.$('#extendBtn').title, 'Export clips first.');
});

test('Color: the slider names its stop, Strength shows only with a grade, and a project opens at config.json\'s Color unless it is gaming', async (t) => {
  const x = await exportTab(t, { routes: { 'GET /api/settings': { compress_peaks_set: false, color: 'auto' } } });
  assert.equal(x.$('#optColor').value, '1');
  assert.equal(x.$('#optColor').getAttribute('aria-valuetext'), 'Auto');
  assert.equal(x.$('#colorStrengthRow').hidden, false);
  const slide = async (sel, v) => {
    x.$(sel).value = String(v);
    x.$(sel).dispatchEvent(new x.page.window.Event('input', { bubbles: true }));
    await x.page.settle();
  };
  await slide('#optColorStrength', 40);
  assert.equal(x.$('#optColorStrength').getAttribute('aria-valuetext'), 'Strength 40%');
  await slide('#optColor', 0);
  assert.equal(x.$('#optColor').getAttribute('aria-valuetext'), 'Off');
  assert.equal(x.$('#colorStrengthRow').hidden, true);
  await slide('#optColor', 2);
  assert.equal(x.$('#optColor').getAttribute('aria-valuetext'), 'Underwater');

  x.page.close();
  const g = await exportTab(t, { project: makeProject({ gaming: true }), routes: { 'GET /api/settings': { compress_peaks_set: false, color: 'auto' } } });
  assert.equal(g.$('#optColor').value, '0', 'never a grade for gaming footage');
});

test('every "Latest export" sentence carries the Color note of the exported clips', async (t) => {
  const { colorNote } = await importPure('util.js');
  const clips = [{ file: '001_a.mp4', source: '/videos/a.mp4', start: 10, end: 18, color: { method: 'auto', strength: 1, skip: 'veto_clipped' } }];
  const note = colorNote(clips);
  assert.ok(note.trim(), 'the fixture gives a note');
  const x = await exportTab(t, { project: makeProject({ clips: { dir: 'clips', clips } }) });
  await setPicks(x, 'auto');
  assert.ok(x.$('#exportState').textContent.includes(note.trim()), x.$('#exportState').textContent);
  await setPicks(x, 'sequence');
  x.$('#candList li[data-id=h001] [data-act=toggle]').click(); // a sequence the export does not match
  await x.page.settle();
  assert.ok(x.$('#exportState').textContent.includes(note.trim()), x.$('#exportState').textContent);
});

test('a viewer who changed nothing exports with Speech-aware cuts, Smart tails, Light peaks and the sequence; loudness and Link on', async (t) => {
  const jobs = [];
  const x = await exportTab(t, { routes: { 'PUT /api/projects/p1/review': {}, 'POST /api/projects/p1/jobs': (req) => { jobs.push(req.body); return { id: 'j1', status: 'running' }; } } });
  const on = (sel) => x.$(sel).checked;
  assert.deepEqual(['#optSpeech', '#optTail', '#optLoudnorm', '#optLink', '#optPicksSeq'].map(on), [true, true, true, true, true]);
  assert.deepEqual(['#optVertical', '#optCaptions', '#optPicksAuto'].map(on), [false, false, false]);
  assert.equal(x.$('#optPeaks').value, '1', 'Light, the middle of None, Light, Max');
  assert.equal(x.$('#optEventTails').value, 'auto');
  assert.equal(x.$('#optAudioTransition').value, 'cut');
  x.$('#candList li[data-id=h001] [data-act=toggle]').click();
  await x.page.settle();
  x.$('#clipsBtn').click();
  await x.page.settle();
  const body = jobs.at(-1);
  assert.equal(body.speech_aware, true);
  assert.equal(body.smart_tail, true);
  assert.equal(body.compress_peaks, 'light');
  assert.equal(body.picks, 'sequence');
});
