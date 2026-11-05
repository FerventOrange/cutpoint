// The compact Moments tile: the length shows once, the preview readiness is an icon with the word as
// its hover title and screen reader text, Add/Remove is an icon button, and the transcript is the
// tile's tooltip (still found by the search).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

// h001 (10-18) plays from a ready window, h002 (100-108) is encoding, h003 (300-308) waits
function proxied() {
  const p = makeProject();
  p.sources[0] = {
    ...p.sources[0], status: 'needs-proxy', windows: [{ lo: 0, hi: 60, url: '/w' }], failed: [],
    pending: [{ lo: 90, hi: 120, state: 'encoding' }, { lo: 290, hi: 320, state: 'queued' }],
  };
  return p;
}

async function open(t, project = proxied()) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project, routes: { 'POST /api/projects/p1/sources/s1/prepare': {} } });
  t.after(() => page.close());
  return page;
}

const tile = (page, id) => page.document.querySelector(`#candList li.cand[data-id="${id}"]`);

test('the length shows once: on the still, not at the end of the time line, and a screen reader hears it once', async (t) => {
  const page = await open(t);
  const li = tile(page, 'h001');
  assert.equal(li.querySelector('.cand-len').textContent, '8.0 s');
  assert.equal(li.querySelector('.cand-len').getAttribute('aria-hidden'), null);
  assert.doesNotMatch(li.querySelector('.cand-time').textContent, /8\.0 s/);
  assert.equal(li.textContent.match(/8\.0 s/g).length, 1);
  page.assertNoUnhandled();
});

test('the preview status is an icon with a title and screen reader text, no visible word', async (t) => {
  const page = await open(t);
  for (const [id, state, word] of [['h001', 'ready', 'ready'], ['h002', 'encoding', 'encoding'], ['h003', 'queued', 'queued']]) {
    const badge = tile(page, id).querySelector('.cand-meta .prep');
    assert.ok(badge, id);
    assert.equal(badge.dataset.state, state);
    assert.equal(badge.title, `Preview ${word}`);
    assert.equal(badge.querySelector('.vh').textContent, `, preview ${word}`);
    assert.equal([...badge.childNodes].filter((n) => n.nodeType === 3).length, 0, 'no visible text');
    assert.equal(tile(page, id).querySelector('.cand-foot'), null);
  }
  assert.ok(tile(page, 'h001').querySelector('.prep .icon-check'));
  assert.ok(tile(page, 'h003').querySelector('.prep .icon-clock'));
  // the spinner ring (CSS-animated, stopped under prefers-reduced-motion) only while encoding
  assert.ok(tile(page, 'h002').querySelector('.prep .prep-spin'));
  assert.equal(tile(page, 'h001').querySelector('.prep-spin'), null);
  assert.equal(tile(page, 'h003').querySelector('.prep-spin'), null);
  page.assertNoUnhandled();
});

test('a source that plays as is has no status icon', async (t) => {
  const page = await open(t, makeProject());
  assert.equal(page.document.querySelector('#candList .prep'), null);
});

test('Add and Remove are icon buttons that keep their labels and toggle the moment', async (t) => {
  const page = await open(t, makeProject());
  const btn = () => tile(page, 'h001').querySelector('[data-act=toggle]');
  assert.equal(btn().getAttribute('aria-label'), 'Add Big laugh to sequence');
  assert.equal(btn().title, 'Add to sequence');
  assert.equal(btn().textContent, '');
  assert.ok(btn().querySelector('.icon-plus'));
  btn().click();
  await page.settle();
  assert.equal(tile(page, 'h001').classList.contains('picked'), true);
  assert.equal(btn().getAttribute('aria-label'), 'Remove Big laugh from sequence');
  assert.equal(btn().title, 'Remove from sequence');
  assert.ok(btn().querySelector('.icon-minus'));
  assert.deepEqual([...tile(page, 'h001').children].map((e) => e.dataset.act), ['focus', 'toggle']);
  btn().click();
  await page.settle();
  assert.equal(tile(page, 'h001').classList.contains('picked'), false);
  page.assertNoUnhandled();
});

test('the transcript is the tile tooltip, not tile text, and the search still finds it', async (t) => {
  const page = await open(t, makeProject());
  const d = page.document;
  const li = tile(page, 'h002');
  assert.equal(li.querySelector('.cand-main').title, 'Fall\nline 2');
  assert.equal(li.querySelector('.cand-text'), null);
  assert.doesNotMatch(li.textContent.replace(li.querySelector('.cand-main > .vh:last-child').textContent, ''), /line 2/); // only as screen reader text
  const filter = d.querySelector('#filter');
  filter.value = 'line 2';
  filter.dispatchEvent(new page.window.Event('input', { bubbles: true }));
  await page.settle();
  assert.deepEqual([...d.querySelectorAll('#candList li.cand')].map((x) => x.dataset.id), ['h002']);
  page.assertNoUnhandled();
});

test('a moment without a transcript has just its title as the tooltip', async (t) => {
  const p = makeProject();
  p.highlights.candidates[0].transcript = '';
  const page = await open(t, p);
  assert.equal(tile(page, 'h001').querySelector('.cand-main').title, 'Big laugh');
});

test('a 360 tile with camera path flags still gives screen readers the transcript', async (t) => {
  const p = makeProject();
  p.highlights.candidates[0].pano = { flags: ['shaky'], reasons: [{ text: 'the camera shakes' }] };
  const page = await open(t, p);
  const main = tile(page, 'h001').querySelector('.cand-main');
  assert.equal(main.getAttribute('aria-describedby'), 'tip-h001'); // overrides the title
  const last = main.lastElementChild;
  assert.equal(last.className, 'vh');
  assert.equal(last.textContent, ', line 1');
  page.assertNoUnhandled();
});
