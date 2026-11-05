// Delete this project (Settings): the button, the confirm dialog fed by the delete plan, the
// Recycle Bin / Trash request and the second, explicit confirm before a permanent delete.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, importPure, makeProject, reply } from './harness.mjs';

const PLAN = {
  name: 'Project One', size: 1572864, files: 12, deleted: ['final.mp4', 'p-final.mp4'], deleted_count: 2,
  kept: ['notes.txt', 'staged/ (used as a source by project Other)'], kept_count: 2,
};
const SETTINGS = { projects_dir: '/projects', projects_dir_locked: false, config_file: '/c.json', compress_peaks_set: false };

async function open(t, { project = makeProject(), routes = {} } = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const deletes = [];
  const page = await boot({
    project,
    routes: {
      'GET /api/settings': SETTINGS,
      'GET /api/projects/p1/delete-plan': PLAN,
      'DELETE /api/projects/p1': (req) => { deletes.push(req.body); return { deleted: 'p1', kept: 2 }; },
      ...routes,
    },
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
  const start = async () => {
    $('#settingsBtn').click();
    await page.settle();
    $('#setDelete').click();
    await page.settle();
  };
  return { page, $, answer, start, deletes };
}

test('the button is there with a project open', async (t) => {
  const { page, $ } = await open(t);
  $('#settingsBtn').click();
  await page.settle();
  assert.equal($('#setDeletePanel').hidden, false);
  assert.equal($('#setDelete').textContent, 'Delete this project…');
});

test('the button is hidden with no project open', async (t) => {
  const { page, $ } = await open(t, { project: null });
  $('#settingsBtn').click();
  await page.settle();
  assert.equal($('#setDeletePanel').hidden, true);
});

test('the confirm names the project and its size, lists what stays and focuses Cancel; Cancel deletes nothing', async (t) => {
  const { page, $, answer, start, deletes } = await open(t);
  await start();
  assert.ok($('#confirmDialog').hasAttribute('open'));
  assert.equal($('#confirmTitle').textContent, 'Delete project Project One (1.5 MB)?');
  const text = $('#confirmText').textContent;
  assert.match(text, /12 files/);
  assert.match(text, /Recycle Bin or Trash/);
  assert.match(text, /including the final videos and exported clips/);
  assert.match(text, /Your source videos are not touched/);
  assert.match(text, /that go:\n {2}final\.mp4\n {2}p-final\.mp4/);
  assert.match(text, /These stay in the project folder:/);
  assert.match(text, /notes\.txt/);
  assert.match(text, /staged\/ \(used as a source by project Other\)/);
  assert.equal($('#confirmOk').textContent, 'Delete project');
  assert.ok($('#confirmOk').classList.contains('danger'));
  assert.equal(page.document.activeElement, $('#confirmCancel'), 'Cancel is focused');
  await answer('cancel');
  assert.deepEqual(deletes, []);
  assert.equal($('#projectName').textContent, 'Project One', 'the project stays open');
});

test('a long list of kept files is cut and says how many more', async (t) => {
  const kept = Array.from({ length: 20 }, (_, i) => `f${i}.txt`);
  const { $, start } = await open(t, { routes: { 'GET /api/projects/p1/delete-plan': { ...PLAN, kept, kept_count: 30 } } });
  await start();
  const text = $('#confirmText').textContent;
  assert.match(text, /f7\.txt/);
  assert.ok(!/f8\.txt/.test(text));
  assert.match(text, /and 22 more/);
});

test('Delete sends the trash request, closes the project and returns to the start screen', async (t) => {
  let gone = false;
  let busy = null;
  const { page, $, answer, start, deletes } = await open(t, {
    routes: {
      'GET /api/projects': () => ({ projects: gone ? [] : [{ id: 'p1', name: 'Project One' }] }),
      'DELETE /api/projects/p1': (req) => {
        busy = [page.document.querySelector('#setDelete').disabled, page.document.querySelector('#setDelete').textContent];
        gone = true; deletes.push(req.body); return { deleted: 'p1', kept: 2 };
      },
    },
  });
  await start();
  await answer('ok');
  assert.deepEqual(deletes, [{ permanent: false }]);
  assert.deepEqual(busy, [true, 'Deleting…'], 'the button shows the request in flight');
  assert.equal($('#setDelete').disabled, false);
  assert.equal($('#setDelete').textContent, 'Delete this project…');
  assert.equal($('#banner').textContent, 'Deleted project Project One. 2 files stayed in its folder.');
  assert.ok(!$('#settingsDialog').hasAttribute('open'), 'Settings is closed');
  assert.equal($('#projectName').textContent, 'No projects yet', 'no project is open');
  assert.equal(page.window.location.hash, '');
  assert.equal(page.document.title, 'cutpoint');
});

test('without a trash the page asks again, and only then deletes for good', async (t) => {
  let n = 0;
  const { $, answer, start, deletes } = await open(t, {
    routes: {
      'DELETE /api/projects/p1': (req) => {
        n += 1;
        deletes.push(req.body);
        return n === 1 ? reply(409, { detail: 'The Recycle Bin / Trash could not take the files (no trash).', code: 'no_trash' }) : { deleted: 'p1', kept: 0 };
      },
    },
  });
  await start();
  await answer('ok');
  assert.deepEqual(deletes, [{ permanent: false }]);
  assert.ok($('#confirmDialog').hasAttribute('open'));
  assert.equal($('#confirmTitle').textContent, 'Delete permanently?');
  assert.match($('#confirmText').textContent, /no trash/);
  assert.equal($('#confirmOk').textContent, 'Delete permanently');
  await answer('ok');
  assert.deepEqual(deletes, [{ permanent: false }, { permanent: true }]);
});

test('declining the permanent delete leaves the project open', async (t) => {
  const deletes = [];
  const { $, answer, start } = await open(t, {
    routes: { 'DELETE /api/projects/p1': (req) => { deletes.push(req.body); return reply(409, { detail: 'No trash here.', code: 'no_trash' }); } },
  });
  await start();
  await answer('ok');
  await answer('cancel');
  assert.deepEqual(deletes, [{ permanent: false }]);
  assert.equal($('#projectName').textContent, 'Project One');
});

test('a refusal (a job is running) shows in Settings and keeps the project', async (t) => {
  const { $, answer, start } = await open(t, {
    routes: { 'DELETE /api/projects/p1': reply(409, { detail: 'A job is running on this project; wait for it to finish or cancel it' }) },
  });
  await start();
  await answer('ok');
  assert.match($('#settingsError').textContent, /A job is running/);
  assert.ok($('#settingsDialog').hasAttribute('open'));
  assert.equal($('#projectName').textContent, 'Project One');
  assert.ok(!$('#confirmDialog').hasAttribute('open'), 'only no_trash asks again');
});

test('fmtBytes: sizes in 1024 steps', async () => {
  const u = await importPure('util.js');
  assert.equal(u.fmtBytes(0), '0 B');
  assert.equal(u.fmtBytes(1023), '1023 B');
  assert.equal(u.fmtBytes(1536), '1.5 KB');
  assert.equal(u.fmtBytes(1572864), '1.5 MB');
  assert.equal(u.fmtBytes(52428800), '50 MB');
  assert.equal(u.fmtBytes(3 * 1024 ** 3), '3 GB');
});
