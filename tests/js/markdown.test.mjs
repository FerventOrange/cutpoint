// markdown(): a model's reply as DOM nodes. Only createElement and text nodes, so nothing in the
// text can run: no markup is parsed and there are no links.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { importPure } from './harness.mjs';

const dom = new JSDOM('<!doctype html><html><body></body></html>');
globalThis.document = dom.window.document;
const { markdown } = await importPure('util.js');

const ALLOWED = new Set(['DIV', 'P', 'BR', 'UL', 'OL', 'LI', 'STRONG', 'EM', 'CODE']);
const html = (text) => markdown(text).innerHTML;
const tags = (root) => [...root.querySelectorAll('*')].map((e) => e.tagName);

test('markdown: bold, both italic markers and inline code', () => {
  assert.equal(html('a **bold** b'), '<p>a <strong>bold</strong> b</p>');
  assert.equal(html('a *it* b'), '<p>a <em>it</em> b</p>');
  assert.equal(html('a _it_ b'), '<p>a <em>it</em> b</p>');
  assert.equal(html('run `cut --top 3` now'), '<p>run <code>cut --top 3</code> now</p>');
  assert.equal(html('**a *b* c**'), '<p><strong>a <em>b</em> c</strong></p>');
  assert.equal(html('`**not bold**`'), '<p><code>**not bold**</code></p>');
});

test('markdown: snake_case is not italic', () => {
  assert.equal(html('use some_file_name here'), '<p>use some_file_name here</p>');
});

test('markdown: -, * and 1. lists, with a paragraph before and after', () => {
  const root = markdown('Top picks:\n- one\n- **two**\n\nthen\n1. first\n2. second\n\n* star\n* bar');
  assert.equal(root.className, 'md');
  assert.deepEqual([...root.children].map((e) => e.tagName), ['P', 'UL', 'P', 'OL', 'UL']);
  assert.deepEqual([...root.children[1].children].map((e) => e.innerHTML), ['one', '<strong>two</strong>']);
  assert.deepEqual([...root.children[3].children].map((e) => e.textContent), ['first', 'second']);
  assert.deepEqual([...root.children[4].children].map((e) => e.textContent), ['star', 'bar']);
});

test('markdown: a loose numbered list keeps its numbers', () => {
  const root = markdown('1. a\n\n2. b\n\n7. c');
  assert.deepEqual([...root.children].map((e) => e.getAttribute('start')), ['1', '2', '7']);
  assert.equal(html('1. a\n2. b'), '<ol start="1"><li>a</li><li>b</li></ol>');
});

test('markdown: a list item is not made of a line that only starts with a marker', () => {
  assert.equal(html('-nothing'), '<p>-nothing</p>');
  assert.equal(html('2024. was a year'), '<ol start="2024"><li>was a year</li></ol>'); // any number starts an item
});

test('markdown: a blank line starts a paragraph, a single newline is a line break', () => {
  assert.equal(html('one\ntwo\r\nthree\n\nfour'), '<p>one<br>two<br>three</p><p>four</p>');
  assert.equal(html(''), '');
  assert.equal(html(null), '');
});

test('markdown: unmatched markers stay literal', () => {
  for (const text of ['2 * 3 = 6', 'a **b', 'a *b', 'a _b', 'a ` b', '** **', 'x * y * z']) {
    const root = markdown(text);
    assert.equal(root.textContent, text, text);
    assert.deepEqual(tags(root).filter((t) => !['P'].includes(t)), [], text);
  }
});

test('markdown: markup, an image with a handler and a javascript link are literal text', () => {
  for (const text of [
    '<b>x</b>',
    '<img src=x onerror=alert(1)>',
    '[a](javascript:x)',
    '<script>alert(1)</script>',
    '- <iframe src=x></iframe>',
    '**<b>x</b>**',
  ]) {
    const root = markdown(text);
    for (const el of root.querySelectorAll('*')) {
      assert.ok(ALLOWED.has(el.tagName), `${text}: ${el.tagName}`);
      assert.equal(el.attributes.length, 0, `${text}: ${el.tagName} has an attribute`);
    }
    assert.equal(root.querySelector('img, script, iframe, a, b'), null, text);
    assert.equal(root.textContent, text.replace(/^- /, '').replace(/\*\*/g, ''), text);
  }
  assert.equal(html('<b>x</b>'), '<p>&lt;b&gt;x&lt;/b&gt;</p>');
  assert.equal(html('[a](javascript:x)'), '<p>[a](javascript:x)</p>');
});

test('markdown: a moment id in bold is a strong element', () => {
  const root = markdown('Take **m9** first');
  const strong = root.querySelector('strong');
  assert.equal(strong.textContent, 'm9');
  assert.equal(strong.parentElement.tagName, 'P');
});
