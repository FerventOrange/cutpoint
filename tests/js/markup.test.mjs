// Accessibility facts of the booted page that tests/test_ui.py used to pin by regex over
// index.html (#1085 A594): read from the DOM, so a reformat of the markup cannot break them.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot } from './harness.mjs';

async function open(t) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot();
  t.after(() => page.close());
  return { page, d: page.document, $: (s) => page.document.querySelector(s) };
}

const labelOf = (d, el) => d.querySelector(`label[for="${el.id}"]`)?.textContent.trim() ?? el.closest('label')?.textContent.trim();

test('every export option names a description that exists and says something, so hover titles are not the only help', async (t) => {
  const { d } = await open(t);
  const controls = [...d.querySelectorAll('#exportPanel input, #exportPanel select')];
  assert.ok(controls.length >= 30, `${controls.length} controls`);
  for (const el of controls) {
    const ids = (el.getAttribute('aria-describedby') ?? '').split(/\s+/).filter(Boolean);
    assert.ok(ids.length, `${el.id} has no aria-describedby`);
    const text = d.getElementById(ids[0])?.textContent.trim() ?? '';
    assert.ok(text.length > 20, `${el.id}: description missing or empty`);
  }
});

test('Clips to Export reads "Sequence" and "Max video length", whose description says it caps the clips\' total', async (t) => {
  const { d, $ } = await open(t);
  assert.equal(labelOf(d, $('#optPicksSeq')), 'Sequence');
  assert.equal(labelOf(d, $('#optMaxTotal')), 'Max video length');
  const described = d.getElementById($('#optMaxTotal').getAttribute('aria-describedby')).textContent;
  assert.match(described, /^Caps the video's length: the clips' total in seconds/);
  assert.match(described, /\(--max-total\)/);
});

test('the AI Chat panel has one labelled input and one named send button; replies are announced from outside the panels', async (t) => {
  const { d, $ } = await open(t);
  const panel = $('#panelAssistant');
  const fields = [...panel.querySelectorAll('input:not([type=checkbox]), textarea')];
  assert.deepEqual(fields.map((f) => f.id), ['chatInput']);
  assert.ok(labelOf(d, fields[0]), 'labelled');
  const submit = [...panel.querySelectorAll('button[type=submit]')];
  assert.deepEqual(submit.map((b) => [b.id, b.getAttribute('aria-label')]), [['chatSend', 'Send']]);
  assert.doesNotMatch(panel.textContent, /\b(mic|microphone|attach|voice)\b/i);
  assert.equal($('#chatTitle').textContent.trim(), 'AI Assistant');
  const live = $('#chatLive');
  assert.equal(live.getAttribute('role'), 'status');
  assert.ok(live.closest('#side'), 'in the side column');
  assert.equal(live.closest('[role=tabpanel]'), null, 'a hidden tab\'s reply is still read out');
});

test('Sort by is native radios, Score first and checked; a status region announces the list', async (t) => {
  const { d, $ } = await open(t);
  const radios = [...d.querySelectorAll('#filterPop input[name=momentOrder]')];
  assert.deepEqual(radios.map((r) => [r.value, r.checked, labelOf(d, r)]), [['score', true, 'Score'], ['position', false, 'Position in video']]);
  assert.equal(radios[0].closest('fieldset').querySelector('legend').textContent.trim(), 'Sort by');
  assert.equal($('#orderLive').getAttribute('role'), 'status');
});

test('One-Click Edit carries a visible Experimental badge, and Make is described by the reason it is off', async (t) => {
  const { $ } = await open(t);
  const badge = $('#presetExp');
  assert.equal(badge.textContent, 'Experimental');
  assert.match(badge.closest('legend').textContent, /^One-Click Edit/);
  assert.ok($('#makeBtn').getAttribute('aria-describedby').split(' ').includes('presetWhy'));
});

test('the trim handles are named sliders in the tab order', async (t) => {
  const { $ } = await open(t);
  for (const [id, name] of [['#startHandle', 'Clip start'], ['#endHandle', 'Clip end']]) {
    const el = $(id);
    assert.equal(el.getAttribute('role'), 'slider');
    assert.equal(el.getAttribute('aria-label'), name);
    assert.equal(el.tabIndex, 0);
  }
});
