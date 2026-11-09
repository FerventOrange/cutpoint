// Settings > Single-key shortcuts (#748, WCAG 2.1.4): on by default; off, M [ ] I O do nothing and
// their hints go, while modifier shortcuts, Space and keys inside controls keep working.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

const KEY = 'cutpoint.singleKeys';

function withSegment() {
  const p = makeProject();
  const c = p.highlights.candidates.find((x) => x.id === 'h001');
  p.review = { segments: [{ id: 'h001', source: c.source, start: c.start, end: c.end, label: c.label }] };
  return p;
}

async function open(t, { storage = {}, setup } = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project: withSegment(), storage, setup });
  t.after(() => page.close());
  const d = page.document;
  const w = page.window;
  const m = {
    page, d, w,
    $: (s) => d.querySelector(s),
    range: () => d.querySelector('#seqList li[data-id=h001] .seq-main').title,
    async key(target, key, init = {}) {
      target.dispatchEvent(new w.KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init }));
      await page.settle();
    },
    async tick(on) {
      const box = d.querySelector('#setSingleKeys');
      box.checked = on;
      box.dispatchEvent(new w.Event('change', { bubbles: true }));
      await page.settle();
    },
    async playheadAt(at) {
      const v = d.querySelector('#video');
      v.__media.loaded(600);
      v.__media.tick(at);
      v.dispatchEvent(new w.Event('seeked'));
      await page.settle();
    },
  };
  return m;
}

const ALL = ['i', 'o', '[', ']', 'm'];

test('on by default: the box is a ticked checkbox with a label, I and M act, the hints show', async (t) => {
  const m = await open(t);
  const box = m.$('#setSingleKeys');
  assert.equal(box.type, 'checkbox');
  assert.equal(box.checked, true);
  assert.ok(box.closest('label').textContent.includes('Single-key shortcuts'));
  assert.ok(m.$(`#${box.getAttribute('aria-describedby')}`));
  assert.equal(m.$('#startHandle').getAttribute('aria-keyshortcuts'), 'I [');
  assert.match(m.$('#endHandle').title, /O sets it/);
  await m.playheadAt(12.5);
  await m.key(m.d.body, 'i');
  assert.equal(m.range(), '0:12.50 to 0:18.00');
  await m.key(m.d.body, 'm');
  assert.equal(m.$('#video').muted, true);
  m.page.assertNoUnhandled();
});

test('off: M [ ] I O do nothing and the hints go; the choice is stored', async (t) => {
  const m = await open(t);
  await m.tick(false);
  assert.equal(m.w.localStorage.getItem(KEY), '0');
  await m.playheadAt(12.5);
  for (const key of ALL) await m.key(m.d.body, key);
  await m.key(m.d.body, 'I'); // a capital I
  assert.equal(m.range(), '0:10.00 to 0:18.00');
  assert.equal(m.$('#video').muted, false);
  assert.equal(m.$('#startHandle').hasAttribute('aria-keyshortcuts'), false);
  assert.equal(m.$('#endHandle').hasAttribute('aria-keyshortcuts'), false);
  assert.doesNotMatch(m.$('#startHandle').title, /sets it/);
  assert.equal(m.d.querySelector('[aria-keyshortcuts="M"]'), null, 'the Mute button names no key');
  await m.tick(true);
  assert.equal(m.w.localStorage.getItem(KEY), null);
  await m.key(m.d.body, 'o');
  assert.equal(m.range(), '0:10.00 to 0:12.50');
});

test('off: Space and a modifier shortcut still work', async (t) => {
  const m = await open(t, { storage: { [KEY]: '0' } });
  assert.equal(m.$('#setSingleKeys').checked, false);
  await m.playheadAt(12.5);
  await m.key(m.d.body, 'ArrowRight');
  await m.key(m.d.body, 'i', { ctrlKey: true });
  assert.equal(m.range(), '0:10.00 to 0:18.00');
  // Ctrl+Z still undoes a trim made with the arrows
  m.$('#startHandle').focus();
  await m.key(m.$('#startHandle'), 'ArrowRight');
  const trimmed = m.range();
  assert.notEqual(trimmed, '0:10.00 to 0:18.00');
  await m.key(m.d.body, 'z', { ctrlKey: true });
  assert.equal(m.range(), '0:10.00 to 0:18.00');
});

test('the choice persists across a boot', async (t) => {
  const m = await open(t, { storage: { [KEY]: '0' } });
  assert.equal(m.$('#setSingleKeys').checked, false);
  assert.equal(m.$('#startHandle').hasAttribute('aria-keyshortcuts'), false);
  await m.playheadAt(12.5);
  await m.key(m.d.body, 'i');
  assert.equal(m.range(), '0:10.00 to 0:18.00');
});

test('Cancel puts the choice back', async (t) => {
  const m = await open(t);
  m.$('#settingsBtn').click();
  await m.page.settle();
  await m.tick(false);
  m.$('#settingsCancel').click();
  await m.page.settle();
  assert.equal(m.$('#setSingleKeys').checked, true);
  assert.equal(m.w.localStorage.getItem(KEY), null);
  assert.equal(m.$('#startHandle').getAttribute('aria-keyshortcuts'), 'I [');
});

test('a localStorage that throws does not break the page; the choice lasts the session', async (t) => {
  const m = await open(t, {
    setup: (w) => {
      for (const name of ['getItem', 'setItem', 'removeItem']) {
        const was = w.Storage.prototype[name];
        w.Storage.prototype[name] = function (k, ...rest) {
          if (k === KEY) throw new Error('blocked');
          return was.call(this, k, ...rest);
        };
      }
    },
  });
  assert.equal(m.$('#setSingleKeys').checked, true);
  await m.tick(false);
  await m.playheadAt(12.5);
  await m.key(m.d.body, 'i');
  assert.equal(m.range(), '0:10.00 to 0:18.00');
});
