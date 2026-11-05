// Settings > Verbose moment labels: off by default, a moment tile then hides its source file, id
// and start-end time (CSS class on the list), keeps the length, and the filter still finds them.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

const KEY = 'cutpoint.verboseMoments';

function twoSources() {
  const p = makeProject();
  const [a, b, c] = p.highlights.candidates;
  p.highlights.candidates = [a, { ...b, source: '/videos/other.mp4' }, c];
  p.sources.push({ id: 's2', path: '/videos/other.mp4', name: 'other.mp4', status: 'ready', url: '/api/projects/p1/sources/s2?t=x', duration: 600 });
  return p;
}

async function open(t, { project = twoSources(), storage = {}, setup } = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project, storage, setup });
  t.after(() => page.close());
  return page;
}

const tile = (page, id) => page.document.querySelector(`#candList li.cand[data-id="${id}"]`);
const list = (page) => page.document.querySelector('#candList');
const box = (page) => page.document.querySelector('#setVerboseMoments');
async function tick(page, on) {
  box(page).checked = on;
  box(page).dispatchEvent(new page.window.Event('change', { bubbles: true }));
  await page.settle();
}
// jsdom has no stylesheet: the hiding is the .moments-compact class, which app.css pairs with these parts
const HIDDEN = ['.cand-id', '.cand-range', '.cand-src'];

test('off by default: the list is compact, the box is unticked and the parts stay in the markup for the filter', async (t) => {
  const page = await open(t);
  assert.equal(box(page).checked, false);
  assert.equal(list(page).classList.contains('moments-compact'), true);
  for (const sel of HIDDEN) assert.ok(tile(page, 'h002').querySelector(sel), sel);
  assert.equal(tile(page, 'h002').querySelector('.cand-src').textContent, 'in other.mp4');
  assert.equal(tile(page, 'h001').querySelector('.cand-id').textContent, 'Candidate h001');
  // what stays: title, kind and score, the length
  assert.ok(tile(page, 'h001').querySelector('.cand-title').textContent);
  assert.match(tile(page, 'h001').querySelector('.cand-meta').textContent, /score/);
  assert.doesNotMatch(tile(page, 'h001').querySelector('.cand-time').textContent, /8\.0 s/); // the length shows once, on the still
  assert.equal(tile(page, 'h001').querySelector('.cand-len').textContent, '8.0 s');
  page.assertNoUnhandled();
});

test('ticking the box shows the parts at once, unticking hides them again, with no reload', async (t) => {
  const page = await open(t);
  const before = tile(page, 'h001');
  await tick(page, true);
  assert.equal(list(page).classList.contains('moments-compact'), false);
  assert.equal(page.window.localStorage.getItem(KEY), '1');
  assert.equal(tile(page, 'h001'), before, 'the tiles are not rebuilt: the class does the work');
  await tick(page, false);
  assert.equal(list(page).classList.contains('moments-compact'), true);
  assert.equal(page.window.localStorage.getItem(KEY), null);
});

test('the choice persists across a boot', async (t) => {
  const page = await open(t, { storage: { [KEY]: '1' } });
  assert.equal(box(page).checked, true);
  assert.equal(list(page).classList.contains('moments-compact'), false);
});

test('Cancel puts the choice back', async (t) => {
  const page = await open(t);
  page.document.querySelector('#settingsBtn').click();
  await page.settle();
  await tick(page, true);
  assert.equal(list(page).classList.contains('moments-compact'), false);
  page.document.querySelector('#settingsCancel').click();
  await page.settle();
  assert.equal(list(page).classList.contains('moments-compact'), true);
  assert.equal(box(page).checked, false);
});

test('a localStorage that throws does not break the page; the choice lasts the session', async (t) => {
  const page = await open(t, {
    setup: (w) => {
      for (const m of ['getItem', 'setItem', 'removeItem']) {
        const was = w.Storage.prototype[m];
        w.Storage.prototype[m] = function (k, ...rest) {
          if (k === KEY) throw new Error('blocked');
          return was.call(this, k, ...rest);
        };
      }
    },
  });
  assert.equal(list(page).classList.contains('moments-compact'), true);
  assert.ok(tile(page, 'h001'));
  await tick(page, true);
  assert.equal(list(page).classList.contains('moments-compact'), false);
  assert.equal(box(page).checked, true);
});

test('the filter still finds a moment by its hidden source name and id', async (t) => {
  const page = await open(t);
  const filter = page.document.querySelector('#filter');
  for (const [text, want] of [['other.mp4', ['h002']], ['h003', ['h003']]]) {
    filter.value = text;
    filter.dispatchEvent(new page.window.Event('input', { bubbles: true }));
    await page.settle();
    assert.deepEqual([...page.document.querySelectorAll('#candList li.cand')].map((li) => li.dataset.id), want, text);
  }
  assert.equal(list(page).classList.contains('moments-compact'), true);
});
