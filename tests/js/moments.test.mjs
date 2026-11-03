// The moments list shows what analysis saw (#634): the label the server picked for a generic
// one (display_label) and a badge on visual-pool moments (#635).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

function project() {
  const p = makeProject();
  const [a, b, c] = p.highlights.candidates;
  p.highlights.candidates = [
    { ...a, label: 'action-moment', display_label: 'smooth skiing run' },
    { ...b, id: 'v001', cli_id: 'v001', label: 'view', pool: 'visual', kind: 'view' },
    { ...c, pool: 'visual' }, // no kind: the badge says "visual" alone
  ];
  return p;
}

async function open(t, p = project()) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project: p });
  t.after(() => page.close());
  return page;
}

test('a card shows display_label for its title and Add button, and the filter finds both labels', async (t) => {
  const page = await open(t);
  const d = page.document;
  const card = d.querySelector('#candList li.cand[data-id="h001"]');
  assert.equal(card.querySelector('.cand-title').textContent, 'smooth skiing run');
  assert.equal(card.querySelector('[data-act=toggle]').getAttribute('aria-label'), 'Add smooth skiing run to sequence');
  const filter = d.querySelector('#filter');
  for (const [text, want] of [['smooth skiing', ['h001']], ['action-moment', ['h001']]]) {
    filter.value = text;
    filter.dispatchEvent(new page.window.Event('input', { bubbles: true }));
    await page.settle();
    assert.deepEqual([...d.querySelectorAll('#candList li.cand')].map((li) => li.dataset.id), want, text);
  }
  page.assertNoUnhandled();
});

test('a visual-pool moment gets a "visual · kind" badge with screen-reader text; others none', async (t) => {
  const page = await open(t);
  const d = page.document;
  const badge = (id) => d.querySelector(`#candList li.cand[data-id="${id}"] .tag.pool`);
  assert.equal(badge('h001'), null);
  const view = badge('v001');
  assert.equal(view.textContent, 'visual · , view'); // the dot is hidden, ", " is read instead
  assert.equal(view.querySelector('[aria-hidden=true]').textContent, ' · ');
  assert.equal(view.querySelector('.vh').textContent, ', ');
  assert.equal(badge('h003').textContent, 'visual');
  page.assertNoUnhandled();
});

test('the preview title uses display_label; a renamed sequence row keeps its own name', async (t) => {
  const p = project();
  p.review = { segments: [{ id: 'h003', source: '/videos/a.mp4', start: 300, end: 308, label: 'my pick' }] };
  const page = await open(t, p);
  const d = page.document;
  const focus = async (id) => {
    d.querySelector(`#candList li.cand[data-id="${id}"] .cand-main`).click();
    await page.settle();
    return d.querySelector('#stageMeta').textContent;
  };
  assert.equal(await focus('h001'), 'smooth skiing run from a.mp4');
  assert.equal(await focus('h003'), 'my pick from a.mp4');
  page.assertNoUnhandled();
});

test('the filter finds a vision label; Add on the preview announces the display_label', async (t) => {
  const p = project();
  p.highlights.candidates[2] = { ...p.highlights.candidates[2], vision_label: 'static group shot' };
  const page = await open(t, p);
  const d = page.document;
  const filter = d.querySelector('#filter');
  filter.value = 'group shot';
  filter.dispatchEvent(new page.window.Event('input', { bubbles: true }));
  await page.settle();
  assert.deepEqual([...d.querySelectorAll('#candList li.cand')].map((li) => li.dataset.id), ['h003']);
  filter.value = '';
  filter.dispatchEvent(new page.window.Event('input', { bubbles: true }));
  await page.settle();
  d.querySelector('#candList li.cand[data-id="h001"] .cand-main').click();
  await page.settle();
  d.querySelector('#toggleBtn').click();
  await page.settle();
  assert.equal(d.querySelector('#seqLive').textContent, 'smooth skiing run added to the sequence');
  page.assertNoUnhandled();
});
