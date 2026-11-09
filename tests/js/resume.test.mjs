// Resume of a stopped run (#109): the Resume buttons, the confirm dialog fed by the server's
// `cutpoint resume --dry-run`, the "(interrupted)" mark in the project list and the plain
// folder_busy state of a job.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

const RUN = { command: 'analyze', status: 'interrupted', resumable: true };
const PREVIEW = {
  command: 'analyze', status: 'interrupted',
  steps: { transcribe: { state: 'failed', done: 1, total: 3 }, reactions: { state: null, done: null, total: null } },
  warnings: ['1 source(s) of the recorded run are not there now'],
};

/** A project that holds only progress.json: no ui/meta.json (no source), no highlights.json. */
const lone = (over = {}) => makeProject({ highlights: null, analysis: null, source: null, sources: [], run: RUN, ...over });

async function open(t, project, routes = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({
    project,
    routes: { 'GET /api/projects': { projects: [{ id: project.id, name: project.name, interrupted: !!project.run?.resumable }] }, ...routes },
  });
  t.after(() => page.close());
  const $ = (sel) => page.document.querySelector(sel);
  const answer = async (value) => {
    const dlg = $('#confirmDialog');
    dlg.returnValue = value;
    dlg.removeAttribute('open');
    dlg.dispatchEvent(new page.window.Event('close'));
    await page.settle();
  };
  return { page, $, answer };
}

test('a project with only progress.json is listed as interrupted and offers Resume, not Analyze', async (t) => {
  const { $ } = await open(t, lone());
  assert.equal($('#projectMenu [data-id=p1]').textContent, 'Project One (interrupted)');
  assert.equal($('#resumeBig').textContent, 'Resume analysis');
  assert.ok(!$('#resumeBig').disabled);
  assert.equal($('#analyzeBig').hidden, true, 'nothing to analyze without a source');
  assert.match($('#momentsEmpty h3').textContent, /analysis stopped before it finished/);
  assert.ok(!/No videos found/.test($('#momentsEmpty').textContent));
});

test('Resume shows what the dry run reports and starts the resume job only once confirmed', async (t) => {
  const posts = [];
  const { page, $, answer } = await open(t, lone(), {
    'GET /api/projects/p1/resume': PREVIEW,
    'POST /api/projects/p1/jobs': (req) => { posts.push(req.body); return { id: 'j1', status: 'running' }; },
  });
  $('#resumeBig').click();
  await page.settle();
  assert.ok(page.requests.some((r) => r.method === 'GET' && r.path === '/api/projects/p1/resume'));
  assert.ok($('#confirmDialog').hasAttribute('open'));
  assert.equal($('#confirmTitle').textContent, 'Resume analysis?');
  const text = $('#confirmText').textContent;
  assert.match(text, /The last analysis was stopped\./);
  assert.match(text, /Transcription: 1 of 3 done/);
  assert.match(text, /are not there now\./);
  assert.equal($('#confirmOk').textContent, 'Resume');
  assert.ok(!$('#confirmOk').classList.contains('danger'), 'resuming is not a destructive action');
  await answer('cancel');
  assert.deepEqual(posts, []);

  $('#resumeBig').click();
  await page.settle();
  await answer('ok');
  assert.deepEqual(posts, [{ action: 'resume' }]);
  assert.equal($('#momentsEmpty').dataset.refocus, '1', 'focus stays in the column once the empty state goes');
});

test('an analyzed project with a stopped export shows Resume export in the Moments header', async (t) => {
  const { $ } = await open(t, makeProject({ run: { command: 'clips', status: 'interrupted', resumable: true } }));
  assert.equal($('#resumeBtn').hidden, false);
  assert.equal($('#resumeBtn').textContent, 'Resume export');
});

test('a finished or missing run shows no Resume', async (t) => {
  const { $ } = await open(t, makeProject({ run: { command: 'analyze', status: 'done', resumable: false } }));
  assert.equal($('#resumeBtn').hidden, true);
  assert.equal($('#projectMenu [data-id=p1]').textContent, 'Project One');
});

test('a job stopped by folder_busy reads as a plain state', async (t) => {
  const busy = 'Analysis is running on this project; wait for it to finish';
  const job = { id: 'j1', project_id: 'p1', action: 'analyze', status: 'failed', error: busy, error_code: 'folder_busy', logs: [`12:00:00 ERROR ${busy}`], progress: { total: null, steps: [] }, started: '' };
  const { $ } = await open(t, makeProject(), { 'GET /api/jobs': { jobs: [job] } });
  assert.equal($('#jobStatus').textContent, `Analysis did not run for Project One: ${busy}`);
  assert.ok(!/delete/i.test($('#jobStatus').textContent));
});

test('a project another run holds says so before any job is tried, and Resume waits', async (t) => {
  const { $ } = await open(t, lone({ busy: true }));
  assert.equal($('#runBusy').hidden, false);
  assert.equal($('#runBusy').textContent, 'Analysis is running on this project; wait for it to finish');
  assert.ok($('#resumeBig').disabled);
});

test('the page own job on the project is not "another run"', async (t) => {
  const job = { id: 'j1', project_id: 'p1', action: 'analyze', status: 'running', logs: [], progress: { total: null, steps: [] }, started: '' };
  const { $ } = await open(t, makeProject({ busy: true }), { 'GET /api/jobs': { jobs: [job] } });
  assert.equal($('#runBusy').hidden, true);
});

const finished = (taken) => ({ id: 'j1', project_id: 'p1', action: 'clips', status: 'completed', logs: [], progress: { total: null, steps: [] }, started: '', ...(taken != null && { taken_over: taken }) });

for (const [taken, line] of [
  [3, 'Clip export finished for Project One. 3 clips taken over from the stopped export'],
  [1, 'Clip export finished for Project One. 1 clip taken over from the stopped export'],
  [null, 'Clip export finished for Project One'],
]) {
  test(`a finished clips job names the clips it took over from the stopped export (${taken})`, async (t) => {
    const { $ } = await open(t, makeProject(), { 'GET /api/jobs': { jobs: [finished(taken)] } });
    assert.equal($('#jobStatus').textContent, line);
  });
}

test('the dialog shows the command line Resume replays, and focus returns to the Resume button', async (t) => {
  let checks = 0;
  const { page, $, answer } = await open(t, lone(), {
    'GET /api/projects/p1/resume': () => { checks += 1; return { ...PREVIEW, argv: ['clips', 'p', '--force', '--out', 'D:/my clips'] }; },
  });
  $('#resumeBig').focus();
  $('#resumeBig').click();
  $('#resumeBig').click(); // a second click while the dry run is out starts no second one
  await page.settle();
  assert.equal(checks, 1);
  assert.match($('#confirmText').textContent, /Command: cutpoint clips p --force --out "D:\/my clips"$/);
  await answer('cancel');
  assert.equal(page.document.activeElement?.id, 'resumeBig');
  assert.equal($('#resumeBig').getAttribute('aria-disabled'), null);
});

test('the fresh-project empty state prints no "null" for the Resume button it does not have', async (t) => {
  const { $ } = await open(t, makeProject({ highlights: null, analysis: null, run: null }));
  const el = $('#momentsEmpty');
  assert.equal(el.hidden, false);
  assert.ok($('#analyzeBig'));
  assert.ok(!$('#resumeBig'));
  assert.ok(!/null|undefined|false/.test(el.textContent), el.textContent);
  assert.deepEqual([...el.childNodes].filter((n) => n.nodeType === 3), [], 'no stray text nodes');
});
