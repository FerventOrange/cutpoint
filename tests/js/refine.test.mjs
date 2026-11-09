// "Describe the footage" (crashes --context) and "Look closer with vision" (events) in the Analyze
// dialog (#486): when each shows or is off, what it is filled with, and the one job each starts.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

const REFINE = { crash_step: true, context: null, unavailable: null };

async function open(t, over = {}, routes = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const jobs = [];
  const page = await boot({
    project: makeProject({ style: 'action', refine: REFINE, ...over }),
    routes: {
      'GET /api/session': { styles: ['funny', 'action'], can_reveal: false, can_pick: false },
      'POST /api/projects/p1/jobs': (req) => { jobs.push(req.body); return { id: `j${jobs.length}`, status: 'running' }; },
      ...routes,
    },
  });
  t.after(() => page.close());
  const $ = (sel) => page.document.querySelector(sel);
  const dialog = () => {
    $('#projectBtn').click();
    $('#analyzeAgainItem').click();
    return $('#analyzeDialog');
  };
  // the harness's dialog has no close(value): do what the browser does, set the return value and fire close
  const press = async (id) => {
    const d = $('#analyzeDialog');
    d.close = (value) => {
      d.returnValue = value ?? d.returnValue;
      d.removeAttribute('open');
      d.dispatchEvent(new page.window.Event('close'));
    };
    assert.equal($(id).disabled, false, `${id} is enabled`);
    if ($(id).type === 'submit') d.close($(id).value); // Analyze is a submit button of the dialog's form
    else $(id).click();
    await page.settle();
  };
  return { page, $, dialog, press, jobs };
}

test('Describe the footage and Look closer are real labelled controls with the agreed hints', async (t) => {
  const { $, dialog } = await open(t);
  dialog();
  assert.equal($('label[for=anDescribe]').textContent, 'Describe the footage');
  assert.equal($('#anDescribe').tagName, 'TEXTAREA');
  assert.equal($('#dAnDescribe').textContent, 'Picks the activity profile that tunes crash detection.');
  assert.equal($('#anDescribeBtn').tagName, 'BUTTON');
  assert.equal($('#anDescribeBtn').textContent, 'Apply');
  assert.equal($('#anVisionBtn').tagName, 'BUTTON');
  assert.equal($('#anVisionBtn').textContent, 'Look closer with vision');
  assert.equal($('#dAnVision').textContent, 'A vision model watches the top moments, adds what it sees and re-ranks.');
  assert.ok($('#anVisionBtn').getAttribute('aria-describedby').split(' ').includes('dAnVision'));
});

test('after an analysis of crash-step footage both are on; Apply waits for text and the saved description is prefilled', async (t) => {
  const { $, dialog } = await open(t, { refine: { ...REFINE, context: 'enduro race' } });
  dialog();
  assert.equal($('#describeBox').hidden, false);
  assert.equal($('#anDescribe').value, 'enduro race');
  assert.equal($('#anDescribe').disabled, false);
  assert.equal($('#anDescribeBtn').disabled, false);
  assert.equal($('#anVisionBtn').disabled, false);
  assert.equal($('#anDescribeWhy').textContent, '');
  $('#anDescribe').value = '  ';
  $('#anDescribe').dispatchEvent(new ($('#anDescribe').ownerDocument.defaultView.Event)('input'));
  assert.equal($('#anDescribeBtn').disabled, true, 'empty text keeps the profile: nothing to apply');
});

test('Apply starts one crashes job with the trimmed text; the dialog does not start an analysis', async (t) => {
  const { $, dialog, press, jobs } = await open(t);
  dialog();
  $('#anDescribe').value = ' enduro race, mud on the lens ';
  $('#anDescribe').dispatchEvent(new ($('#anDescribe').ownerDocument.defaultView.Event)('input'));
  await press('#anDescribeBtn');
  assert.deepEqual(jobs, [{ action: 'crashes', context: 'enduro race, mud on the lens' }]);
});

test('Look closer with vision starts one events job and nothing else', async (t) => {
  const { dialog, press, jobs } = await open(t);
  dialog();
  await press('#anVisionBtn');
  assert.deepEqual(jobs, [{ action: 'events' }]);
});

test('Enter in an option field starts no refining job: both buttons are type=button, Analyze is the default submit', async (t) => {
  const { $, page, dialog, jobs } = await open(t, { refine: { ...REFINE, context: 'mud' } });
  dialog();
  for (const id of ['#anDescribeBtn', '#anVisionBtn']) assert.equal($(id).type, 'button', id);
  const form = $('#analyzeDialog form');
  const submitters = [];
  form.addEventListener('submit', (e) => { e.preventDefault(); submitters.push(e.submitter?.id ?? null); });
  $('#anLanguage').focus();
  form.requestSubmit(); // what Enter in a field does: no submitter named, the form's first submit button
  assert.equal(submitters.length, 1);
  assert.ok(!['anDescribeBtn', 'anVisionBtn'].includes(submitters[0]));
  await page.settle();
  assert.deepEqual(jobs, []);
  assert.equal([...form.querySelectorAll('button')].find((b) => b.type === 'submit').id, 'againCancel');
});

test('the Analyze button still starts the analysis with the options only', async (t) => {
  const { dialog, press, jobs } = await open(t);
  dialog();
  await press('#againOk');
  assert.equal(jobs.length, 1);
  assert.equal(jobs[0].action, 'analyze');
  assert.ok(!('context' in jobs[0]));
});

test('before the first analysis a funny project has no Describe the footage box; Look closer is off', async (t) => {
  const { $ } = await open(t, { style: 'funny', highlights: null, analysis: null, refine: { ...REFINE, crash_step: false } });
  $('#analyzeOptsBtn').click();
  assert.equal($('#describeBox').hidden, true);
  assert.equal($('#anVisionBtn').disabled, true);
});

for (const [command, label] of [['crashes', 'Resume crash detection'], ['events', 'Resume closer look']]) {
  test(`a stopped ${command} run is resumed as "${label}"`, async (t) => {
    const { $ } = await open(t, { highlights: null, analysis: null, run: { command, status: 'interrupted', resumable: true } });
    assert.equal($('#resumeBig').textContent, label);
  });
}

test('before the first analysis both are off with "Available after the first analysis"', async (t) => {
  const { $ } = await open(t, { highlights: null, analysis: null, refine: REFINE });
  $('#analyzeOptsBtn').click();
  assert.equal($('#analyzeDialog').open, true);
  const why = 'Available after the first analysis';
  assert.equal($('#describeBox').hidden, false);
  assert.equal($('#anDescribe').disabled, true);
  assert.equal($('#anDescribeBtn').disabled, true);
  assert.equal($('#anDescribeWhy').textContent, why);
  assert.equal($('#anVisionBtn').disabled, true);
  assert.equal($('#anVisionWhy').textContent, why);
});

test('footage whose analysis ran no crash step has no Describe the footage; Look closer stays', async (t) => {
  const { $, dialog } = await open(t, { style: 'funny', refine: { ...REFINE, crash_step: false } });
  dialog();
  assert.equal($('#describeBox').hidden, true);
  assert.equal($('#anVisionBtn').disabled, false);
});

test('360 projects get a note instead of the description field, and no vision look', async (t) => {
  const { $, dialog } = await open(t, { kind: '360', refine: { crash_step: false, context: null, unavailable: null } });
  dialog();
  assert.equal($('#describeBox').hidden, false);
  assert.equal($('#anDescribe').hidden, true);
  assert.equal($('#anDescribeBtn').hidden, true);
  assert.equal($('#describeBox label').hidden, true);
  assert.equal($('#anDescribeWhy').textContent, "Describe the footage doesn't apply to 360 footage.");
  assert.equal($('#anVisionBtn').disabled, true);
});

test('in Cutpoint lite both are off with the server\'s message', async (t) => {
  const unavailable = "torch is not installed: this feature is not available in Cutpoint lite (it needs the 'events' extra).";
  const { $, dialog } = await open(t, { refine: { ...REFINE, context: 'mud', unavailable } });
  dialog();
  assert.equal($('#anDescribe').disabled, true);
  assert.equal($('#anDescribeBtn').disabled, true);
  assert.equal($('#anDescribeWhy').textContent, unavailable);
  assert.equal($('#anVisionBtn').disabled, true);
  assert.equal($('#anVisionWhy').textContent, unavailable);
});

test('a project payload without refine (an older server) does not fail; the server refuses what cannot run', async (t) => {
  const { $, dialog } = await open(t, { refine: undefined });
  dialog();
  assert.equal($('#anVisionBtn').disabled, false, 'the server decides: it refuses what cannot run');
  assert.equal($('#describeBox').hidden, true);
});

test('the jobs are named on the page', async (t) => {
  const jobs = { jobs: [{ id: 'j1', project_id: 'p1', action: 'crashes', status: 'running', logs: [], progress: { total: null, steps: [] }, started: '' }] };
  const { page } = await open(t, {}, { 'GET /api/jobs': jobs });
  assert.match(page.document.querySelector('#jobStatus').textContent, /^Crash detection running for/);
});
