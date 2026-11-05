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
  assert.deepEqual(kids.map((k) => k.getAttribute('role')), ['menuitemradio', 'menuitemradio', 'separator', 'menuitem']);
  assert.deepEqual(kids.slice(0, 2).map((k) => k.getAttribute('aria-checked')), ['true', 'false']);
  assert.equal(kids[1].textContent, 'Second trip (interrupted)');
  assert.equal(kids[3].textContent, 'New project…');
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
  assert.equal(active(), items[0], 'wraps');
  key(active(), 'ArrowUp');
  assert.equal(active(), items[2]);
  key(active(), 'Home');
  assert.equal(active(), items[0]);
  key(active(), 'End');
  assert.equal(active(), items[2]);
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
  assert.equal(page.document.activeElement, $('#newProjectItem'));
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
