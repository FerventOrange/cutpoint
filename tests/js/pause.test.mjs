// The compact empty state while the first analysis runs, and Pause analysis (#811): which parts
// hide, the Pause button cancelling the job, focus staying in the column, and the stopped state
// with Resume after the pause.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

const STEPS = [{ step: 'transcribe', label: 'Transcribe', state: 'running', pct: 40 }];
const RUNNING = { id: 'j1', project_id: 'p1', action: 'analyze', status: 'running', logs: [], progress: { total: 1, steps: STEPS }, started: '' };
const FRESH = () => makeProject({ highlights: null, analysis: null });

async function open(t, project, routes = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project, routes: { 'GET /api/session': { styles: ['funny'], can_reveal: true, can_pick: false }, ...routes } });
  t.after(() => page.close());
  const $ = (sel) => page.document.querySelector(sel);
  const answer = async (value) => {
    const dlg = $('#confirmDialog');
    dlg.returnValue = value;
    dlg.removeAttribute('open');
    dlg.dispatchEvent(new page.window.Event('close'));
    await page.settle();
  };
  const cancels = () => page.requests.filter((r) => r.method === 'POST' && r.path === '/api/jobs/j1/cancel').length;
  return { page, $, answer, cancels };
}

test('while the first analysis runs the art, options and Analyzing… button hide; Show folder, the steps and Pause stay', async (t) => {
  const jobs = { jobs: [] };
  const { page, $ } = await open(t, FRESH(), { 'GET /api/jobs': () => jobs });
  assert.equal($('#pauseBtn').hidden, true);
  assert.equal($('#analyzeBig').hidden, false);
  assert.equal($('#analyzeOptsBtn').hidden, false);
  assert.equal($('#momentsEmpty').classList.contains('running'), false);
  jobs.jobs = [RUNNING];
  t.mock.timers.tick(5000);
  await page.settle();
  assert.equal($('#momentsEmpty').classList.contains('running'), true, 'the CSS hides the illustration');
  assert.equal($('#analyzeBig').hidden, true);
  assert.equal($('#analyzeOptsBtn').hidden, true);
  assert.equal($('#revealBtn').hidden, false);
  assert.ok($('.fresh-count').classList.contains('fresh-lead'), 'the videos-found line hides with the lead (CSS)');
  assert.ok($('#pauseBtn').classList.contains('primary'), 'the amber button');
  assert.equal($('#freshSteps').hidden, false);
  assert.equal($('#pauseBtn').hidden, false);
  assert.equal($('#pauseBtn').tagName, 'BUTTON');
  assert.equal($('#pauseBtn').textContent, 'Pause analysis');
  assert.ok($('#pauseWhy').textContent.includes('Resume'));
  jobs.jobs = [{ ...RUNNING, status: 'failed', error: 'x' }];
  t.mock.timers.tick(1000);
  await page.settle();
  assert.equal($('#momentsEmpty').classList.contains('running'), false, 'the normal empty state is back');
  assert.equal($('#analyzeBig').hidden, false);
  assert.equal($('#analyzeOptsBtn').hidden, false);
  assert.equal($('#pauseBtn').hidden, true);
});

test('focus moves to Pause analysis when the parts hide, not to the page body', async (t) => {
  const jobs = { jobs: [] };
  const { page, $ } = await open(t, FRESH(), { 'GET /api/jobs': () => jobs });
  $('#analyzeBig').focus();
  jobs.jobs = [RUNNING];
  t.mock.timers.tick(5000);
  await page.settle();
  assert.equal(page.document.activeElement, $('#pauseBtn'));
});

test('focus on Show folder stays there across polls of a live run', async (t) => {
  const jobs = { jobs: [] };
  const { page, $ } = await open(t, FRESH(), { 'GET /api/jobs': () => jobs });
  jobs.jobs = [RUNNING];
  t.mock.timers.tick(5000);
  await page.settle();
  $('#revealBtn').focus();
  t.mock.timers.tick(1000);
  await page.settle();
  assert.equal(page.document.activeElement, $('#revealBtn'));
});

test('a resumed first analysis shows the compact panel and Pause, and hides Resume', async (t) => {
  const run = { command: 'analyze', status: 'interrupted', resumable: true };
  const project = makeProject({ highlights: null, analysis: null, run });
  const { page, $ } = await open(t, project, { 'GET /api/jobs': { jobs: [{ ...RUNNING, action: 'resume' }] } });
  t.mock.timers.tick(5000);
  await page.settle();
  assert.equal($('#momentsEmpty').classList.contains('running'), true);
  assert.equal($('#resumeBig').hidden, true);
  assert.equal($('#analyzeBig').hidden, true);
  assert.equal($('#pauseBtn').hidden, false);
});

test('Pause before any video is transcribed asks first; cancel is sent only after Stop', async (t) => {
  const { page, $, answer, cancels } = await open(t, FRESH(), { 'GET /api/jobs': { jobs: [RUNNING] }, 'POST /api/jobs/j1/cancel': {} });
  $('#pauseBtn').click();
  await page.settle();
  assert.ok($('#confirmDialog').hasAttribute('open'));
  assert.equal($('#confirmTitle').textContent, 'Stop the analysis?');
  assert.equal(cancels(), 0);
  await answer('cancel');
  assert.equal(cancels(), 0);
  assert.notEqual($('#pauseBtn').getAttribute('aria-disabled'), 'true');
  $('#pauseBtn').click();
  await page.settle();
  await answer('ok');
  assert.equal(cancels(), 1);
});

test('Pause cancels at once when the server already records the run, though the page has not refreshed', async (t) => {
  const { page, $, cancels } = await open(t, FRESH(), {
    'GET /api/jobs': { jobs: [RUNNING] },
    'GET /api/projects/p1': { ...FRESH(), run: { command: 'analyze', status: 'running', resumable: false } },
    'POST /api/jobs/j1/cancel': {},
  });
  $('#pauseBtn').click();
  await page.settle();
  assert.equal($('#confirmDialog').hasAttribute('open'), false);
  assert.equal(cancels(), 1);
});

test('Pause with a transcribed video cancels at once', async (t) => {
  const done = { ...RUNNING, progress: { total: 2, steps: [{ step: 'transcribe', label: 'Transcribe', state: 'done' }] } };
  const { page, $, cancels } = await open(t, FRESH(), { 'GET /api/jobs': { jobs: [done] }, 'POST /api/jobs/j1/cancel': {} });
  $('#pauseBtn').click();
  await page.settle();
  assert.equal(cancels(), 1);
});

test('Pause analysis cancels the job and the panel then offers Resume', async (t) => {
  const jobs = { jobs: [RUNNING] };
  let project = FRESH();
  const { page, $, answer } = await open(t, project, {
    'GET /api/jobs': () => jobs,
    'GET /api/projects/p1': () => project,
    'POST /api/jobs/j1/cancel': () => {
      jobs.jobs = [{ ...RUNNING, status: 'cancelled' }];
      project = { ...project, run: { command: 'analyze', status: 'interrupted', resumable: true } };
      return {};
    },
  });
  $('#pauseBtn').focus();
  $('#pauseBtn').click();
  await page.settle();
  await answer('ok'); // nothing transcribed yet: asks first
  await page.settle();
  assert.equal(page.requests.filter((r) => r.method === 'POST' && r.path === '/api/jobs/j1/cancel').length, 1);
  t.mock.timers.tick(1000);
  await page.settle();
  t.mock.timers.tick(5000);
  await page.settle();
  assert.match($('#momentsEmpty h3').textContent, /analysis stopped before it finished/);
  assert.equal($('#resumeBig').textContent, 'Resume analysis');
  assert.equal($('#momentsEmpty').classList.contains('running'), false);
  assert.equal($('#pauseBtn').hidden, true);
  assert.notEqual(page.document.activeElement, page.document.body);
});
