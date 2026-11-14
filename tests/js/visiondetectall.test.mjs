// Analyze options > "Show every video to the vision model" (--vision-detect-all, #1069): the box
// is filled with what the analysis used, works only while the analysis runs in the Auto style,
// and a disabled box sends null so the project's own setting is kept.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

const OPTS = { model: null, language: null, max_candidates: null, no_scenes: false, skip: [], must_include_crash: null, recover: false };
const SESSION = { styles: ['action', 'funny', 'gaming', 'my-cut'], can_reveal: false, can_pick: false };

async function open(t, { style = 'funny', vision_detect_all = false, detected } = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const jobs = [];
  const page = await boot({
    project: makeProject({ style, analyze_options: { ...OPTS, vision_detect_all }, ...(detected && { detected }) }),
    routes: {
      'GET /api/session': SESSION,
      'POST /api/projects/p1/jobs': (req) => { jobs.push(req.body); return { id: `j${jobs.length}`, status: 'running' }; },
    },
  });
  t.after(() => page.close());
  const $ = (sel) => page.document.querySelector(sel);
  // the harness's dialog has no close(value): set the return value and fire close, as the browser does
  const confirm = async () => {
    const d = $('#analyzeDialog');
    d.returnValue = 'ok';
    d.removeAttribute('open');
    d.dispatchEvent(new page.window.Event('close'));
    await page.settle();
  };
  const again = () => {
    $('#projectBtn').click();
    $('#analyzeAgainItem').click();
  };
  return { page, $, jobs, confirm, again };
}

const AUTO = {
  auto: true, chosen: 'auto', style: 'gaming', choices: [], category: 'gameplay', label: 'Gameplay',
  reason: 'static HUD 2%', ambiguous: false, fallback_label: null, used_style: 'gaming', suggest: null, stale: false, sources: [],
};

test('the box is a labelled checkbox, disabled while the style is not Auto', async (t) => {
  const { $, again } = await open(t, { style: 'funny', vision_detect_all: true });
  again();
  const box = $('#anVisionDetectAll');
  assert.equal(box.type, 'checkbox');
  assert.match($('label:has(#anVisionDetectAll)').textContent, /Show every video to the vision model/);
  assert.ok(box.getAttribute('aria-describedby').split(' ').includes('dAnVisionDetectAll'));
  assert.equal(box.disabled, true);
});

for (const recorded of [true, false]) {
  test(`an Auto project fills the box with the analysis option (${recorded}) and enables it`, async (t) => {
    const { $, again } = await open(t, { style: 'auto', vision_detect_all: recorded, detected: AUTO });
    again();
    assert.equal($('#anVisionDetectAll').disabled, false);
    assert.equal($('#anVisionDetectAll').checked, recorded);
  });
}

for (const [style, recorded, ticked, sent] of [['auto', false, true, true], ['auto', true, false, false], ['funny', true, true, null]]) {
  test(`Analyze in ${style} with the box ${recorded} then ${ticked} sends ${sent}`, async (t) => {
    const { $, again, confirm, jobs } = await open(t, { style, vision_detect_all: recorded, ...(style === 'auto' && { detected: AUTO }) });
    again();
    $('#anVisionDetectAll').checked = ticked;
    await confirm();
    assert.equal(jobs.length, 1);
    assert.ok('vision_detect_all' in jobs[0], 'the key is always sent');
    assert.equal(jobs[0].vision_detect_all, sent, 'a disabled box sends null: the analysis keeps its own');
  });
}

test('choosing a style in the menu enables the box for Auto only', async (t) => {
  const { $, jobs, confirm } = await open(t, { style: 'gaming', vision_detect_all: true, detected: { ...AUTO, auto: false, chosen: 'gaming' } });
  $('#styleBtn').click();
  $('#styleMenu [data-style=auto]').click();
  assert.equal($('#anVisionDetectAll').disabled, false);
  assert.equal($('#anVisionDetectAll').checked, true);
  const d = $('#analyzeDialog');
  d.returnValue = '';
  d.removeAttribute('open');
  d.dispatchEvent(new d.ownerDocument.defaultView.Event('close'));
  $('#styleBtn').click();
  $('#styleMenu [data-style=my-cut]').click();
  assert.equal($('#anVisionDetectAll').disabled, true);
  await confirm();
  assert.equal(jobs.length, 1);
  assert.equal(jobs[0].style, 'my-cut');
  assert.equal(jobs[0].vision_detect_all, null);
});
