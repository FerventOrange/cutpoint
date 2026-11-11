// The header "cutpoint / <project>": the project menu (a menu button) that replaced the project select and
// the New project button, and the tab title.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, bootRoutes, makeProject, reply, WEB_DIR } from './harness.mjs';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const P2 = makeProject({ id: 'p2', name: 'Second trip' });

async function open(t, { project = makeProject(), two = true } = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const routes = two
    ? {
        ...bootRoutes(P2),
        'GET /api/projects': { projects: [{ id: 'p1', name: 'Project One' }, { id: 'p2', name: 'Second trip', interrupted: true }] },
      }
    : {};
  const page = await boot({ project, routes });
  t.after(() => page.close());
  const $ = (sel) => page.document.querySelector(sel);
  const key = (target, k) => target.dispatchEvent(new page.window.KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true }));
  return { page, $, key };
}

test('the header reads cutpoint / <project>, with a hidden separator and a menu button', async (t) => {
  const { $ } = await open(t);
  assert.equal($('.brand').textContent, 'cutpoint');
  assert.equal($('.brand-sep').textContent, '/');
  assert.equal($('.brand-sep').getAttribute('aria-hidden'), 'true');
  const btn = $('#projectBtn');
  assert.equal(btn.textContent.trim(), 'Project One');
  assert.equal(btn.getAttribute('aria-haspopup'), 'menu');
  assert.equal(btn.getAttribute('aria-expanded'), 'false');
  assert.equal(btn.title, 'Project One');
  assert.equal($('#newBtn'), null);
  assert.equal($('#projectSelect'), null);
});

test('the name truncates at 30 characters in CSS, never in the string', async (t) => {
  const css = readFileSync(join(WEB_DIR, 'app.css'), 'utf8');
  assert.match(css, /\.proj-name\s*\{[^}]*max-width: 30ch;[^}]*text-overflow: ellipsis/);
  const long = 'A very long project name that goes on and on and on';
  const { $ } = await open(t, { project: makeProject({ name: long }), two: false });
  assert.equal($('#projectName').textContent, long);
  assert.equal($('#projectBtn').title, long);
});

test('no project open: the trigger says Choose project (or No projects yet) and the tab title is cutpoint', async (t) => {
  const { page, $ } = await open(t, { project: null, two: false });
  assert.equal($('#projectName').textContent, 'No projects yet');
  assert.equal(page.document.title, 'cutpoint');
});

test('the tab title follows the open project', async (t) => {
  const { page, $ } = await open(t);
  assert.equal(page.document.title, 'cutpoint / Project One');
  $('#projectBtn').click();
  $('#projectMenu [data-id=p2]').click();
  await page.settle();
  assert.equal(page.document.title, 'cutpoint / Second trip');
  assert.equal($('#projectName').textContent, 'Second trip');
});

test('the menu lists the projects with the current one marked, a divider and New project…', async (t) => {
  const { page, $ } = await open(t);
  const pop = $('#projectMenu');
  assert.equal(pop.hidden, true);
  $('#projectBtn').click();
  assert.equal(pop.hidden, false);
  assert.equal($('#projectBtn').getAttribute('aria-expanded'), 'true');
  const kids = [...pop.children];
  assert.deepEqual(kids.map((k) => k.getAttribute('role')), ['menuitemradio', 'menuitemradio', 'separator', 'menuitem', 'separator', 'menuitem', 'none']);
  assert.deepEqual(kids.slice(0, 2).map((k) => k.getAttribute('aria-checked')), ['true', 'false']);
  assert.equal(kids[1].textContent, 'Second trip (interrupted)');
  assert.equal(kids[3].textContent, 'New project…');
  assert.equal(kids[5].textContent, 'Analyze again', 'the open project action; Show folder is absent where the server cannot open folders');
  assert.equal(page.document.activeElement, kids[0], 'focus starts on the current project');
});

test('arrow keys, Home and End move focus without acting; Escape closes and refocuses the trigger', async (t) => {
  const { page, $, key } = await open(t);
  const { requests } = page;
  const before = requests.length;
  $('#projectBtn').click();
  const items = [...$('#projectMenu').querySelectorAll('.proj-item')];
  const active = () => page.document.activeElement;
  key(active(), 'ArrowDown');
  assert.equal(active(), items[1]);
  key(active(), 'ArrowDown');
  assert.equal(active(), items[2], 'New project…');
  key(active(), 'ArrowDown');
  assert.equal(active(), items[3], 'Analyze again');
  key(active(), 'ArrowDown');
  assert.equal(active(), items[0], 'wraps');
  key(active(), 'ArrowUp');
  assert.equal(active(), items[3]);
  key(active(), 'Home');
  assert.equal(active(), items[0]);
  key(active(), 'End');
  assert.equal(active(), items[3]);
  assert.equal(page.document.title, 'cutpoint / Project One', 'nothing fired');
  assert.equal(requests.length, before, 'no request while moving');
  assert.equal($('#newDialog').open, false);
  key(active(), 'Escape');
  assert.equal($('#projectMenu').hidden, true);
  assert.equal($('#projectBtn').getAttribute('aria-expanded'), 'false');
  assert.equal(active(), $('#projectBtn'));
});

test('ArrowDown on the trigger opens the menu; ArrowUp opens it on the last item', async (t) => {
  const { page, $, key } = await open(t);
  key($('#projectBtn'), 'ArrowUp');
  assert.equal($('#projectMenu').hidden, false);
  assert.equal(page.document.activeElement, $('#analyzeAgainItem'));
});

test('a click outside closes the menu', async (t) => {
  const { page, $ } = await open(t);
  $('#projectBtn').click();
  page.document.body.dispatchEvent(new page.window.Event('pointerdown', { bubbles: true }));
  assert.equal($('#projectMenu').hidden, true);
});

test('choosing another project loads it by the existing path; the current one does nothing', async (t) => {
  const { page, $ } = await open(t);
  const loads = () => page.requests.filter((r) => r.url?.includes?.('/api/projects/p2') || r.path === '/api/projects/p2').length;
  $('#projectBtn').click();
  $('#projectMenu [data-id=p1]').click();
  await page.settle();
  assert.equal(loads(), 0);
  assert.equal(page.document.activeElement, $('#projectBtn'));
  $('#projectBtn').click();
  $('#projectMenu [data-id=p2]').click();
  await page.settle();
  assert.ok(loads() > 0, 'GET /api/projects/p2');
  assert.equal($('#projectMenu').hidden, true);
  assert.equal($('#projectMenu [data-id=p2]').getAttribute('aria-checked'), 'true');
});

test('New project… opens the new-project dialog', async (t) => {
  const { page, $ } = await open(t);
  $('#projectBtn').click();
  $('#newProjectItem').click();
  assert.equal($('#newDialog').hasAttribute('open'), true);
  assert.equal($('#projectMenu').hidden, true);
  assert.ok(page.document.getElementById('newStyle'));
});

test('a refused discard keeps the project open and loads nothing', async (t) => {
  const project = makeProject();
  project.review = { segments: project.highlights.candidates.slice(0, 2).map((c) => ({ id: c.id, source: c.source, start: c.start, end: c.end, label: c.label })) };
  const { page, $ } = await open(t, { project });
  page.window.confirm = () => false;
  $('#seqList .seq-chev[data-act=down]').click(); // reorder: unsaved changes
  $('#projectBtn').click();
  $('#projectMenu [data-id=p2]').click();
  await page.settle();
  assert.equal(page.requests.filter((r) => r.path === '/api/projects/p2').length, 0);
  assert.equal($('#projectName').textContent, 'Project One');
  assert.equal($('#projectMenu [data-id=p1]').getAttribute('aria-checked'), 'true');
});

test('a project list the server refuses shows Projects unavailable and disables the trigger', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project: null, routes: { 'GET /api/projects': reply(409, { detail: 'config.json is invalid' }) }, allowUnhandled: true });
  t.after(() => page.close());
  assert.equal(page.document.querySelector('#projectName').textContent, 'Projects unavailable');
  assert.equal(page.document.querySelector('#projectBtn').disabled, true);
});

test('with no projects the menu holds only New project… (no divider, no "null")', async (t) => {
  const { $ } = await open(t, { project: null, two: false });
  $('#projectBtn').click();
  assert.deepEqual([...$('#projectMenu').children], [$('#newProjectItem')]);
  assert.equal($('#projectMenu').textContent, 'New project…');
});

test('the page shortcuts (M, I, O, [ and ]) do nothing inside the menu', async (t) => {
  const { page, $, key } = await open(t);
  const muted = () => $('.vol button').getAttribute('aria-pressed');
  const range = () => [$('#startHandle').getAttribute('aria-valuenow'), $('#endHandle').getAttribute('aria-valuenow')];
  const [was, wasRange] = [muted(), range()];
  $('#projectBtn').click();
  for (const k of ['m', 'i', 'o', '[', ']']) key(page.document.activeElement, k);
  assert.equal(muted(), was);
  assert.deepEqual(range(), wasRange);
  // the same keys on the page do act (so the check above can fail)
  key(page.document.body, 'm');
  assert.notEqual(muted(), was);
});

// ---------- the Moments head and the project menu's Analyze again / Show folder / stats ----------

const SESSION = { styles: ['funny'], can_reveal: true, can_pick: false };
const RUNNING = { id: 'j1', project_id: 'p1', action: 'analyze', status: 'running', logs: [], progress: { total: null, steps: [] }, started: '' };

async function openRoutes(t, project, routes) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project, routes });
  t.after(() => page.close());
  const $ = (sel) => page.document.querySelector(sel);
  return { page, $ };
}

test('the head is "Moments" with the plain total beside it; the buttons and stats left it', async (t) => {
  const { $ } = await openRoutes(t, makeProject(), {});
  assert.equal($('#momentsTitle').textContent, 'Moments');
  assert.equal($('#momentsCount').textContent, '3');
  assert.equal($('#analyzeBtn'), null);
  assert.equal($('#revealHeadBtn'), null);
  assert.equal($('#analyzeStats'), null);
  assert.equal($('.moments .col-head').textContent.replace(/\s+/g, ' ').trim(), 'Moments 3 Resume analysis');
});

test('the count stays the total while a filter narrows the list', async (t) => {
  const { page, $ } = await openRoutes(t, makeProject(), {});
  const f = $('#filter');
  f.value = 'Fall';
  f.dispatchEvent(new page.window.Event('input', { bubbles: true }));
  await page.settle();
  assert.equal($('#candList').querySelectorAll('li.cand').length, 1);
  assert.equal($('#momentsCount').textContent, '3');
});

test('the menu shows the stats as text without a role, and Show folder where the server can open folders', async (t) => {
  const { $ } = await openRoutes(t, makeProject(), { 'GET /api/session': SESSION });
  $('#projectBtn').click();
  const kids = [...$('#projectMenu').children];
  assert.deepEqual(kids.map((k) => k.getAttribute('role')), ['menuitemradio', 'separator', 'menuitem', 'separator', 'menuitem', 'menuitem', 'none']);
  assert.deepEqual(kids.slice(4, 6).map((k) => k.textContent), ['Analyze again', 'Show folder']);
  const stats = $('.proj-stats');
  assert.equal(stats.querySelector('button, a, input'), null);
  assert.deepEqual([...stats.children].map((c) => c.textContent), ['3 found, funny style', 'Analyzed 10 min of video in 30 s']);
});

test('before analysis the menu has no project actions; the empty state keeps the big Analyze button', async (t) => {
  const { $ } = await openRoutes(t, makeProject({ highlights: null, analysis: null }), { 'GET /api/session': SESSION });
  $('#projectBtn').click();
  assert.equal($('#analyzeAgainItem'), null);
  assert.equal($('#revealItem'), null);
  assert.equal($('.proj-stats'), null);
  assert.equal($('#momentsCount').textContent, '');
  assert.equal($('#analyzeBig').hidden, false);
  assert.equal($('#revealBtn').hidden, false);
});

test('without a length or time the stats keep only the found line', async (t) => {
  const { $ } = await openRoutes(t, makeProject({ analysis: null }), {});
  $('#projectBtn').click();
  assert.deepEqual([...$('.proj-stats').children].map((c) => c.textContent), ['3 found, funny style']);
});

test('Analyze again from the menu closes it, asks first, and returns focus to the name', async (t) => {
  const { page, $ } = await openRoutes(t, makeProject(), {});
  const posts = () => page.requests.filter((r) => r.method === 'POST' && r.path === '/api/projects/p1/jobs');
  $('#projectBtn').click();
  $('#analyzeAgainItem').click();
  assert.equal($('#projectMenu').hidden, true);
  assert.equal($('#analyzeDialog').open, true);
  assert.equal($('#analyzeTitle').textContent, 'Analyze again?');
  assert.equal(posts().length, 0, 'nothing runs before the confirmation');
  const d = $('#analyzeDialog');
  d.returnValue = '';
  d.removeAttribute('open');
  d.dispatchEvent(new page.window.Event('close'));
  assert.equal(page.document.activeElement, $('#projectBtn'));
  assert.equal(posts().length, 0, 'cancelled');
});

test('while a job runs Analyze again is aria-disabled (still focusable) and does nothing', async (t) => {
  const { page, $ } = await openRoutes(t, makeProject(), { 'GET /api/jobs': { jobs: [RUNNING] } });
  $('#projectBtn').click();
  const item = $('#analyzeAgainItem');
  assert.equal(item.getAttribute('aria-disabled'), 'true');
  assert.equal(item.title, 'Another job is running.');
  assert.equal(item.disabled, false);
  item.click();
  assert.equal($('#analyzeDialog').open, false);
  assert.equal($('#projectMenu').hidden, false, 'the menu stays open');
  assert.equal(page.document.activeElement, $('#projectMenu [aria-checked=true]'));
});

test('while another run holds the project Analyze again is aria-disabled with the reason, as Resume is disabled', async (t) => {
  const { page, $ } = await openRoutes(t, makeProject({ busy: true }), { 'GET /api/jobs': { jobs: [] } });
  $('#projectBtn').click();
  const item = $('#analyzeAgainItem');
  const why = 'Analysis is running on this project; wait for it to finish';
  assert.equal(item.getAttribute('aria-disabled'), 'true');
  assert.equal(item.title, why);
  assert.equal(item.getAttribute('aria-describedby'), 'projBusy projStats');
  assert.equal($('#projBusy').textContent, why);
  assert.equal(item.disabled, false, 'still focusable');
  item.click();
  assert.equal($('#analyzeDialog').open, false);
  assert.equal($('#projectMenu').hidden, false, 'the menu stays open');
  assert.equal(page.document.activeElement, $('#projectMenu [aria-checked=true]'));
});

test('the big Analyze button and its options are disabled, with the reason, while another run holds the project', async (t) => {
  const { $ } = await openRoutes(t, makeProject({ highlights: null, analysis: null, busy: true }), { 'GET /api/jobs': { jobs: [] } });
  const why = 'Analysis is running on this project; wait for it to finish';
  for (const id of ['#analyzeBig', '#analyzeOptsBtn']) {
    assert.equal($(id).disabled, true, id);
    assert.equal($(id).title, why, id);
  }
});

test('an Analyze that fails with folder_busy leaves focus on a button that works, not on the disabled one or the page', async (t) => {
  const idle = makeProject({ highlights: null, analysis: null });
  let project = idle;
  const busy = 'Analysis is running on this project; wait for it to finish';
  const failed = { id: 'j1', project_id: 'p1', action: 'analyze', status: 'failed', error: busy, error_code: 'folder_busy', logs: [], progress: { total: null, steps: [] }, started: '' };
  let jobs = [];
  const { page, $ } = await openRoutes(t, idle, { 'GET /api/projects/p1': () => project, 'GET /api/jobs': () => ({ jobs }) });
  $('#analyzeBig').focus();
  $('#momentsEmpty').dataset.refocus = '1'; // what the button's click sets (keepEmptyFocus); the test starts the job itself via the jobs poll
  jobs = [{ ...failed, status: 'running', error: null, error_code: null }]; // the page's own Analyze starts
  t.mock.timers.tick(5000);
  await page.settle();
  const away = page.document.body.appendChild(page.document.createElement('button')); // a browser drops focus from a button that became disabled; jsdom does not
  away.focus();
  away.remove();
  assert.equal(page.document.activeElement, page.document.body);
  project = { ...idle, busy: true }; // it finds the command line holding the project and fails
  jobs = [failed];
  t.mock.timers.tick(1000);
  await page.settle();
  assert.equal($('#analyzeBig').disabled, true);
  const now = page.document.activeElement;
  assert.notEqual(now, page.document.body, 'focus did not fall to the page');
  assert.notEqual(now, $('#analyzeBig'), 'nor on the disabled button');
  assert.ok(!now.disabled);
});

test('Show folder asks for the sources folder, closes the menu and refocuses the name', async (t) => {
  const { page, $ } = await openRoutes(t, makeProject(), { 'GET /api/session': SESSION, 'POST /api/projects/p1/reveal': { ok: true } });
  $('#projectBtn').click();
  $('#revealItem').click();
  await page.settle();
  const post = page.requests.find((r) => r.method === 'POST' && r.path === '/api/projects/p1/reveal');
  assert.ok(post, 'asked the server');
  assert.equal(post.query, '?target=sources');
  assert.equal($('#projectMenu').hidden, true);
  assert.equal(page.document.activeElement, $('#projectBtn'));
  assert.equal($('#projectBtn').disabled, false);
});

test('an open menu keeps focus on Analyze again when a job starts and ends, and a tick that changes nothing does not rebuild it', async (t) => {
  const jobs = { jobs: [] };
  const { page, $ } = await openRoutes(t, makeProject(), { 'GET /api/jobs': () => jobs });
  $('#projectBtn').click();
  const item = $('#analyzeAgainItem');
  item.focus();
  t.mock.timers.tick(5000); // a poll that finds the same (no job)
  await page.settle();
  assert.equal($('#analyzeAgainItem'), item, 'not rebuilt');
  jobs.jobs = [RUNNING];
  t.mock.timers.tick(5000);
  await page.settle();
  assert.equal($('#analyzeAgainItem').getAttribute('aria-disabled'), 'true');
  assert.equal(page.document.activeElement, $('#analyzeAgainItem'));
  assert.equal($('#analyzeAgainItem').getAttribute('aria-describedby'), 'projBusy projStats');
  assert.equal($('#projBusy').textContent, 'Another job is running.');
  const busy = $('#analyzeAgainItem');
  t.mock.timers.tick(1000); // still running: same menu
  await page.settle();
  assert.equal($('#analyzeAgainItem'), busy, 'not rebuilt');
  jobs.jobs = [{ ...RUNNING, status: 'completed' }];
  t.mock.timers.tick(1000);
  await page.settle();
  assert.equal($('#analyzeAgainItem').getAttribute('aria-disabled'), null);
  assert.equal(page.document.activeElement, $('#analyzeAgainItem'));
  assert.equal($('#analyzeAgainItem').getAttribute('aria-describedby'), 'projStats');
});

test('the stats are described by Analyze again, and a press on them does not take focus', async (t) => {
  const { page, $ } = await openRoutes(t, makeProject(), {});
  $('#projectBtn').click();
  assert.equal($('#projStats').textContent, '3 found, funny styleAnalyzed 10 min of video in 30 s');
  assert.equal($('#analyzeAgainItem').getAttribute('aria-describedby'), 'projStats');
  const press = (el) => {
    const e = new page.window.MouseEvent('mousedown', { bubbles: true, cancelable: true });
    el.dispatchEvent(e);
    return e.defaultPrevented;
  };
  assert.equal(press($('#projStats')), true);
  assert.equal(press($('#projStats span')), true);
  assert.equal(press($('#projectMenu .proj-sep')), true, 'a divider too');
  assert.equal(press($('#analyzeAgainItem')), false, 'items take the press as usual');
});

test('the zero-moment text points at the project menu', async (t) => {
  const { $ } = await openRoutes(t, makeProject({ highlights: { style: 'funny', candidates: [] } }), {});
  assert.match($('#momentsEmpty').textContent, /choose Analyze again in the project menu/);
});

// ---------- the style button (#897) ----------

const CHOICES = [
  { category: 'gameplay', label: 'Gameplay', style: 'gaming', main: true },
  { category: 'action', label: 'Action', style: 'action', main: true },
  { category: 'action-cam', label: 'Action cam', style: 'action-cam', main: false },
  { category: 'talk', label: 'Talk', style: 'commentary', main: true },
];

/** A ``detected`` view as GET /api/projects/<id> returns it; ``over`` replaces fields. */
function detected(over = {}) {
  return {
    auto: true, chosen: 'auto', style: 'gaming', choices: CHOICES, category: 'gameplay', label: 'Gameplay',
    reason: 'static HUD 2%', ambiguous: false, fallback_label: null, used_style: 'gaming', suggest: null, stale: false, sources: [],
    ...over,
  };
}

const STYLES_SESSION = { styles: ['action', 'action-cam', 'commentary', 'funny', 'gaming', 'my-cut', 'sport'], can_reveal: false, can_pick: false };

async function openStyle(t, det, routes = {}) {
  const project = makeProject({ style: det?.chosen ?? 'gaming', highlights: { style: det?.style ?? 'gaming', candidates: makeProject().highlights.candidates }, detected: det });
  const { page, $ } = await openRoutes(t, project, { 'GET /api/session': STYLES_SESSION, 'POST /api/projects/p1/jobs': { id: 'j9', status: 'running' }, ...routes });
  const key = (target, k) => target.dispatchEvent(new page.window.KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true }));
  return { page, $, key };
}

const lines = ($) => [...($('#styleLines')?.children ?? [])].map((c) => c.textContent);
const radios = ($) => [...$('#styleMenu').querySelectorAll('.style-item')];

test('the style button is hidden before the first analysis and without a detected view (360 projects)', async (t) => {
  const { $ } = await openStyle(t, null);
  assert.equal($('#styleMenuWrap').hidden, true);
});

test('the style button reads the detected category, an unsure fallback, an override and an earlier detection', async (t) => {
  const cases = [
    [detected(), 'Detected: Gameplay'],
    [detected({ ambiguous: true, fallback_label: 'Gameplay' }), 'Not sure: Gameplay style'],
    [detected({ ambiguous: true, category: 'talk', label: 'Talk', fallback_label: 'General', used_style: 'action', style: 'action' }), 'Not sure: General style'],
    [detected({ auto: false, chosen: 'commentary', style: 'commentary', stale: true }), 'Style: Talk (your choice)'],
    [detected({ auto: false, chosen: 'my-cut', style: 'my-cut', category: null, label: null }), 'Style: my-cut (your choice)'],
    [detected({ stale: true, style: 'commentary' }), 'Detected earlier: Gameplay'],
    [detected({ category: null, label: null, reason: null }), 'Auto: analyze again to detect'],
    [detected({ category: 'action', label: 'Action', style: 'action', used_style: 'action' }), 'Detected: Action'],
    // an old (version 3) detection of General: a label only, not a menu choice
    [detected({ category: 'general', label: 'General', style: 'action', used_style: 'action' }), 'Detected: General'],
    [detected({ auto: false, chosen: 'action', style: 'action', stale: true }), 'Style: Action (your choice)'],
  ];
  for (const [det, text] of cases) {
    await t.test(text, async (st) => {
      const { $ } = await openStyle(st, det);
      assert.equal($('#styleMenuWrap').hidden, false);
      assert.equal($('#styleName').textContent, text);
      assert.equal($('#styleBtn').title, text, 'the full text where the chip ends in an ellipsis');
      assert.equal($('#styleBtn').getAttribute('aria-haspopup'), 'menu');
    });
  }
});

test('the chip truncates in CSS and sits in the header after the project menu', async (t) => {
  const css = readFileSync(join(WEB_DIR, 'app.css'), 'utf8');
  assert.match(css, /\.style-name\s*\{[^}]*text-overflow: ellipsis/);
  const html = readFileSync(join(WEB_DIR, 'index.html'), 'utf8');
  assert.ok(html.indexOf('id="styleMenuWrap"') > html.indexOf('id="projectMenuWrap"'));
  assert.ok(html.indexOf('id="styleMenuWrap"') < html.indexOf('id="exportBtn"'));
});

test('the menu gives the reason, the unsure fallback, the disagreeing sources and a suggestion as text lines', async (t) => {
  const sources = [{ name: 'b.mp4', path: '/v/b.mp4', category: 'talk', label: 'Talk', reason: 'speech 80% of 30 min', ambiguous: false }];
  const { $ } = await openStyle(t, detected({ ambiguous: true, fallback_label: 'Gameplay', sources, suggest: 'commentary' }));
  $('#styleBtn').click();
  assert.deepEqual(lines($), [
    'Leans Gameplay (static HUD 2%); used the Gameplay style (gaming)',
    'Disagree: b.mp4 looks like Talk (speech 80% of 30 min)',
    'Suggested: Talk (commentary)',
  ]);
  assert.equal($('#styleLines').getAttribute('role'), 'none');
  assert.equal($('#styleMenu').getAttribute('aria-describedby'), 'styleLines');
});

test('a sure detection gives its reason', async (t) => {
  const { $ } = await openStyle(t, detected());
  $('#styleBtn').click();
  assert.deepEqual(lines($), ['Detected: Gameplay (static HUD 2%)']);
});

test('the menu points at no lines when there are none', async (t) => {
  const { $ } = await openStyle(t, detected({ category: null, label: null, reason: null }));
  $('#styleBtn').click();
  assert.equal($('#styleLines'), null);
  assert.equal($('#styleMenu').hasAttribute('aria-describedby'), false);
});

test('after an override the menu shows the earlier detection', async (t) => {
  const { $ } = await openStyle(t, detected({ auto: false, chosen: 'commentary', style: 'commentary', stale: true }));
  $('#styleBtn').click();
  assert.deepEqual(lines($), ['Detected earlier: Gameplay (static HUD 2%)']);
  assert.equal($('#styleMenu [aria-checked=true]').dataset.style, 'commentary');
});

test('the items: Auto, the categories with their style, then Other styles with the rest (a custom style too)', async (t) => {
  const { $ } = await openStyle(t, detected());
  $('#styleBtn').click();
  const items = radios($);
  assert.deepEqual(items.map((i) => i.textContent), [
    'Auto (detect from footage)', 'Gameplay: gaming', 'Action: action', 'Talk: commentary',
    'Action cam', 'funny', 'my-cut', 'sport',
  ]);
  assert.ok(items.every((i) => i.getAttribute('role') === 'menuitemradio'));
  assert.deepEqual(items.map((i) => i.getAttribute('aria-checked')), ['true', 'false', 'false', 'false', 'false', 'false', 'false', 'false']);
  const group = $('#styleMenu [role=group]');
  assert.equal(group.getAttribute('aria-labelledby'), 'styleOtherLabel');
  assert.equal($('#styleOtherLabel').textContent, 'Other styles');
  assert.deepEqual([...group.children].map((i) => i.dataset.style), ['action-cam', 'funny', 'my-cut', 'sport']);
});

test('an Action detection and an old General one list each style once and offer no General', async (t) => {
  for (const det of [
    detected({ category: 'action', label: 'Action', style: 'action', used_style: 'action' }),
    detected({ category: 'general', label: 'General', style: 'action', used_style: 'action' }),
  ]) {
    await t.test(det.label, async (st) => {
      const { $ } = await openStyle(st, det);
      $('#styleBtn').click();
      const items = radios($);
      const styles = items.map((i) => i.dataset.style).filter(Boolean);
      assert.equal(new Set(styles).size, styles.length, 'no duplicate data-style items');
      assert.ok(!items.some((i) => /^General/.test(i.textContent)));
      assert.equal(items.filter((i) => /^Action:/.test(i.textContent)).length, 1);
    });
  }
});

test('Action cam is a main item when the footage has camera telemetry', async (t) => {
  const choices = CHOICES.map((c) => ({ ...c, main: true }));
  const { $ } = await openStyle(t, detected({ choices, category: 'action-cam', label: 'Action cam', style: 'action-cam' }));
  $('#styleBtn').click();
  assert.deepEqual(radios($).slice(0, 5).map((i) => i.textContent), ['Auto (detect from footage)', 'Gameplay: gaming', 'Action: action', 'Action cam: action-cam', 'Talk: commentary']);
  assert.ok(![...$('#styleMenu [role=group]').children].some((i) => i.dataset.style === 'action-cam'));
});

test('the style menu keys: focus starts on the chosen item; arrows, Home and End move; Escape closes and refocuses', async (t) => {
  const { page, $, key } = await openStyle(t, detected({ auto: false, chosen: 'commentary', style: 'commentary' }));
  const active = () => page.document.activeElement;
  key($('#styleBtn'), 'ArrowDown');
  assert.equal($('#styleMenu').hidden, false);
  assert.equal($('#styleBtn').getAttribute('aria-expanded'), 'true');
  const items = radios($);
  assert.equal(active(), items[3], 'the chosen style');
  key(active(), 'ArrowDown');
  assert.equal(active(), items[4]);
  key(active(), 'Home');
  assert.equal(active(), items[0]);
  key(active(), 'ArrowUp');
  assert.equal(active(), items.at(-1), 'wraps');
  key(active(), 'End');
  assert.equal(active(), items.at(-1));
  assert.equal($('#analyzeDialog').open, false, 'moving acts on nothing');
  key(active(), 'Escape');
  assert.equal($('#styleMenu').hidden, true);
  assert.equal($('#styleBtn').getAttribute('aria-expanded'), 'false');
  assert.equal(active(), $('#styleBtn'));
});

test('choosing a style asks in the Analyze dialog, filled in, and posts the analysis with the style', async (t) => {
  const { page, $ } = await openStyle(t, detected());
  $('#styleBtn').click();
  $('#styleMenu [data-style=commentary]').click();
  assert.equal($('#styleMenu').hidden, true);
  const d = $('#analyzeDialog');
  assert.equal(d.open, true);
  assert.equal($('#analyzeTitle').textContent, 'Analyze again as Talk?');
  assert.match($('#analyzeText').textContent, /in the commentary style/);
  assert.equal($('#refine').hidden, true, 'Describe and vision keep the old style: not offered here');
  const posts = () => page.requests.filter((r) => r.method === 'POST' && r.path === '/api/projects/p1/jobs');
  assert.equal(posts().length, 0, 'nothing runs before the confirmation');
  d.returnValue = 'ok';
  d.removeAttribute('open');
  d.dispatchEvent(new page.window.Event('close'));
  await page.settle();
  assert.equal(posts().length, 1);
  const { body } = posts()[0];
  assert.equal(body.action, 'analyze');
  assert.equal(body.style, 'commentary');
  assert.ok('skip' in body && 'model' in body, 'with the Analyze options');
  assert.equal(page.document.activeElement, $('#styleBtn'));
});

test('the dialog is filled with the analysis options, and Auto and Other styles get their own titles', async (t) => {
  const project = makeProject({ detected: detected({ auto: false, chosen: 'gaming' }), analyze_options: { model: 'large-v3', language: null, max_candidates: null, no_scenes: false, skip: ['dialogue'], must_include_crash: null, recover: false } });
  const { page, $ } = await openRoutes(t, project, { 'GET /api/session': STYLES_SESSION });
  $('#styleBtn').click();
  $('#styleMenu [data-style=auto]').click();
  assert.equal($('#analyzeTitle').textContent, 'Analyze again as Auto?');
  assert.match($('#analyzeText').textContent, /in the style detected from it/);
  assert.equal($('#anModel').value, 'large-v3');
  assert.equal(page.document.querySelector('input[name=anSkip][value=dialogue]').checked, true);
  const d = $('#analyzeDialog');
  d.returnValue = '';
  d.removeAttribute('open');
  d.dispatchEvent(new page.window.Event('close'));
  $('#styleBtn').click();
  $('#styleMenu [data-style=my-cut]').click();
  assert.equal($('#analyzeTitle').textContent, 'Analyze again as my-cut?');
  d.returnValue = '';
  d.removeAttribute('open');
  d.dispatchEvent(new page.window.Event('close'));
  $('#projectBtn').click();
  $('#analyzeAgainItem').click();
  assert.equal($('#analyzeTitle').textContent, 'Analyze again?', 'plain Analyze again keeps its title');
  assert.equal($('#refine').hidden, false);
});

test('choosing the current style only closes the menu', async (t) => {
  const { page, $ } = await openStyle(t, detected());
  $('#styleBtn').click();
  $('#styleMenu [data-style=auto]').click();
  assert.equal($('#styleMenu').hidden, true);
  assert.equal($('#analyzeDialog').open, false);
  assert.equal(page.document.activeElement, $('#styleBtn'));
});

test('while a job runs the style items are aria-disabled (still focusable) and do nothing', async (t) => {
  const { page, $ } = await openStyle(t, detected(), { 'GET /api/jobs': { jobs: [RUNNING] } });
  $('#styleBtn').click();
  const item = $('#styleMenu [data-style=commentary]');
  assert.equal(item.getAttribute('aria-disabled'), 'true');
  assert.equal(item.title, 'Another job is running.');
  assert.equal(item.getAttribute('aria-describedby'), 'styleBusy');
  assert.equal($('#styleBusy').textContent, 'Another job is running.');
  assert.equal(item.disabled, false);
  item.click();
  assert.equal($('#analyzeDialog').open, false);
  assert.equal($('#styleMenu').hidden, false, 'the menu stays open');
  assert.equal(page.requests.filter((r) => r.method === 'POST' && r.path === '/api/projects/p1/jobs').length, 0);
});

test('the project facts name the style the analysis ran in, not auto', async (t) => {
  const { $ } = await openStyle(t, detected());
  $('#projectBtn').click();
  assert.equal($('.proj-stats span').textContent, '3 found, gaming style');
});
