// The Graphite refresh: the theme default and the old Dark theme's migration, the Moments grid
// (kind, score, kind filter), the header Export button and the sequence row's kind and tail line.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, bootRoutes, importPure, makeProject, reply } from './harness.mjs';

const u = await importPure('util.js');

async function open(t, { project = makeProject(), storage = {}, routes = {}, setup } = {}) {
  try {
    t.mock.timers.enable({ apis: ['setTimeout'] });
  } catch {
    // a second page in one test: the timers are already mocked
  }
  const page = await boot({ project, storage, routes, setup });
  t.after(() => page.close());
  return page;
}
const theme = (page) => page.document.documentElement.dataset.theme;

// ---- theme ----

test('themeFromStored: Graphite is the default; Light and Blue are kept; a stored "dark" is Graphite', () => {
  assert.deepEqual([null, undefined, '', 'graphite', 'dark', 'nope'].map(u.themeFromStored), Array(6).fill('graphite'));
  assert.equal(u.themeFromStored('light'), 'light');
  assert.equal(u.themeFromStored('blue'), 'blue');
  assert.deepEqual([...u.THEMES], ['graphite', 'light', 'blue']);
});

test('no saved theme boots as Graphite: no data-theme, the select says so and offers Graphite, Light, Blue', async (t) => {
  const page = await open(t);
  assert.equal(theme(page), undefined);
  const sel = page.document.querySelector('#setTheme');
  assert.equal(sel.value, 'graphite');
  assert.deepEqual([...sel.options].map((o) => o.value), ['graphite', 'light', 'blue']);
});

test('a saved "dark" becomes Graphite and its key is dropped', async (t) => {
  const page = await open(t, { storage: { 'cutpoint.theme': 'dark' } });
  assert.equal(theme(page), undefined);
  assert.equal(page.document.querySelector('#setTheme').value, 'graphite');
  assert.equal(page.window.localStorage.getItem('cutpoint.theme'), null);
});

for (const saved of ['light', 'blue']) {
  test(`a saved "${saved}" stays ${saved}`, async (t) => {
    const page = await open(t, { storage: { 'cutpoint.theme': saved } });
    assert.equal(theme(page), saved);
    assert.equal(page.document.querySelector('#setTheme').value, saved);
    assert.equal(page.window.localStorage.getItem('cutpoint.theme'), saved);
  });
}

test('picking a theme in Settings applies it at once; Graphite is stored as no key, Blue as "blue"', async (t) => {
  const page = await open(t);
  const sel = page.document.querySelector('#setTheme');
  const pick = async (v) => {
    sel.value = v;
    sel.dispatchEvent(new page.window.Event('change', { bubbles: true }));
    await page.settle();
  };
  await pick('blue');
  assert.equal(theme(page), 'blue');
  assert.equal(page.window.localStorage.getItem('cutpoint.theme'), 'blue');
  await pick('graphite');
  assert.equal(theme(page), undefined);
  assert.equal(page.window.localStorage.getItem('cutpoint.theme'), null);
});

// ---- kind ----

test('kindOf: scenic views, laughter, a few spoken words, else action; scoreText fits both scales', () => {
  assert.equal(u.kindOf({ pool: 'visual', kind: 'view' }), 'scenic');
  assert.equal(u.kindOf({ pool: 'visual', kind: 'cruise', transcript: '' }), 'action');
  assert.equal(u.kindOf({ signals: { laughter: 0.9 }, transcript: 'one two three four five' }), 'funny');
  assert.equal(u.kindOf({ label: 'laugh', transcript: '' }), 'funny');
  assert.equal(u.kindOf({ transcript: 'we are going down the hill' }), 'dialogue');
  assert.equal(u.kindOf({ transcript: 'wow' }), 'action');
  assert.equal(u.kindOf(undefined), null);
  assert.deepEqual([88.4, 9.06, 100].map((n) => u.scoreText(n, n < 10)), ['88', '9.1', '100']);
  assert.equal(u.scoreText(9.6, false), '10');
  assert.equal(u.scoreText(10, true), '10.0');
  // pacing.moment_kind's order: scenic word, funny word, action word, talk word, then laughter, then the transcript
  assert.equal(u.kindOf({ label: 'Beautiful vista', transcript: 'a b c d e' }), 'scenic');
  assert.equal(u.kindOf({ label: 'Fast joke' }), 'funny');
  assert.equal(u.kindOf({ label: 'Banter at speed' }), 'action');
  assert.equal(u.kindOf({ label: 'Banter', signals: { laughter: 0.9 } }), 'dialogue');
  assert.equal(u.kindOf({ label: 'ski-moment', vision_label: 'two friends talking' }), 'dialogue');
  assert.equal(u.kindOf({ label: 'ski-moment', transcript: '' }), 'action'); // a generic label names nothing
  // then the strongest signal cue of 5+ of 10 (0-1 signals scaled, rounded half to even)
  assert.equal(u.kindOf({ label: 'ski-moment', signals: { beauty: 0.8 }, transcript: 'a b c d e' }), 'scenic');
  assert.equal(u.kindOf({ label: 'ski-moment', signals: { laughter: 0.6, crash: 0.9 } }), 'action');
  assert.equal(u.kindOf({ label: 'ski-moment', signals: { laughter: 0.45 }, transcript: 'a b c d e' }), 'dialogue'); // 4.5 rounds to 4
  assert.equal(u.kindOf({ label: 'ski-moment', signals: { laughter: 0.55 } }), 'funny'); // 5.5 rounds to 6
  assert.equal(u.kindOf({ label: 'ski-moment', signals: { laughter: 0.5 } }), 'funny'); // 5
  assert.equal(u.kindOf({ label: 'ski-moment', signals: { loudness: 0.9, laughter: 0.6 } }), 'funny'); // loudness has no kind
});

function kinds() {
  const p = makeProject();
  const [a, b, c] = p.highlights.candidates;
  p.highlights.candidates = [
    { ...a, label: 'Fast run', score: 88.4, transcript: '' },
    { ...b, id: 'v001', cli_id: 'v001', label: 'view', pool: 'visual', kind: 'view', score: 9.06, transcript: '' },
    { ...c, signals: { laughter: 0.8 } },
  ];
  return p;
}

test('a Moments tile has its kind, the kind and score line, the length and a still box; the buttons keep their order', async (t) => {
  const page = await open(t, { project: kinds() });
  const d = page.document;
  const tile = (id) => d.querySelector(`#candList li.cand[data-id="${id}"]`);
  assert.equal(tile('h001').dataset.kind, 'action');
  assert.equal(tile('v001').dataset.kind, 'scenic');
  assert.equal(tile('h003').dataset.kind, 'funny');
  const meta = (id) => tile(id).querySelector('.cand-meta').textContent;
  assert.equal(meta('h001'), 'Action · , score 88');
  assert.equal(meta('v001'), 'Scenic · , score 9.1');
  assert.equal(tile('h001').querySelector('.cand-len').textContent, '8.0 s');
  assert.equal(tile('h001').querySelector('.cand-thumb .cand-dot')?.getAttribute('aria-hidden'), 'true');
  // the sequence row's URL, so the two share one cached still: a box that shows the kind tint until it loads
  assert.ok(tile('h001').querySelector('.cand-thumb'));
  assert.deepEqual([...tile('h001').children].map((e) => e.dataset.act), ['focus', 'toggle']);
});

test('a tile asks for its still only once it scrolls into view, through one shared observer', async (t) => {
  const made = [];
  const seen = new Set();
  class Watch {
    constructor(cb) { this.cb = cb; made.push(this); }
    observe(el) { seen.add(el); }
    unobserve(el) { seen.delete(el); }
    disconnect() {}
  }
  const page = await open(t, { project: kinds(), setup: (w) => { w.IntersectionObserver = Watch; } });
  const d = page.document;
  const img = (id) => d.querySelector(`#candList li.cand[data-id="${id}"] .cand-thumb img`);
  assert.equal(made.length, 1);
  assert.equal(img('h001').getAttribute('src'), null);
  assert.equal(page.requests.filter((r) => /\/frame\?/.test(String(r.url ?? r))).length, 0);
  const box = img('h001').parentElement;
  assert.ok(seen.has(box));
  made[0].cb([{ target: box, isIntersecting: false }]);
  await page.settle();
  assert.equal(img('h001').getAttribute('src'), null);
  made[0].cb([{ target: box, isIntersecting: true }]);
  await page.settle();
  assert.match(img('h001').getAttribute('src'), /\/frame\?/);
  assert.equal(seen.has(box), false); // not watched any more
  assert.equal(img('h003').getAttribute('src'), null); // its tile has not scrolled in
});

test('the kind filter shows one kind, All brings the rest back, and the pressed segment says which', async (t) => {
  const page = await open(t, { project: kinds() });
  const d = page.document;
  const ids = () => [...d.querySelectorAll('#candList li.cand')].map((li) => li.dataset.id);
  const seg = (k) => d.querySelector(`#kindSeg [data-kind="${k}"]`);
  assert.equal(d.querySelector('#kindSeg').hidden, false);
  assert.equal(seg('').getAttribute('aria-pressed'), 'true');
  seg('scenic').click();
  await page.settle();
  assert.deepEqual(ids(), ['v001']);
  assert.equal(seg('scenic').getAttribute('aria-pressed'), 'true');
  assert.equal(seg('').getAttribute('aria-pressed'), 'false');
  seg('dialogue').click();
  await page.settle();
  assert.deepEqual(ids(), []);
  assert.match(d.querySelector('#candList').textContent, /No moments match these filters/);
  seg('').click();
  await page.settle();
  assert.deepEqual(ids(), ['h001', 'v001', 'h003']);
});

// ---- header Export ----

test('the header Export button shows the Export panel in the side column with no tab selected; Back, Escape and a second click return', async (t) => {
  const page = await open(t);
  const d = page.document;
  const btn = d.querySelector('.top #exportBtn');
  assert.ok(btn, 'Export sits in the header');
  assert.equal(btn.nextElementSibling.id, 'saveBtn', 'next to Save sequence');
  assert.equal(d.querySelector('#panelExport').hidden, true);
  btn.click();
  await page.settle();
  assert.equal(d.querySelector('#panelExport').hidden, false);
  assert.equal(btn.getAttribute('aria-expanded'), 'true');
  assert.equal(d.querySelector('#tabExport'), null, 'Export is no tab');
  assert.deepEqual([...d.querySelectorAll('#side [role=tab]')].map((x) => x.getAttribute('aria-selected')), ['false', 'false', 'false']);
  assert.equal(d.querySelector('#panelSequence').hidden, true);
  d.querySelector('#exportBack').click(); // Back returns to the tab it came from
  assert.equal(d.querySelector('#panelExport').hidden, true);
  assert.equal(d.querySelector('#panelSequence').hidden, false);
  assert.equal(btn.getAttribute('aria-expanded'), 'false');
  btn.click();
  btn.click(); // a second click returns too
  assert.equal(d.querySelector('#panelExport').hidden, true);
  btn.click();
  const esc = (el) => el.dispatchEvent(new page.window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
  esc(d.querySelector('#candList')); // Escape in another column keeps its own meaning
  assert.equal(d.querySelector('#panelExport').hidden, false);
  esc(d.querySelector('#exportBack'));
  assert.equal(d.querySelector('#panelExport').hidden, true);
  assert.equal(d.querySelector('#tabSequence').getAttribute('aria-selected'), 'true');
  d.querySelector('#tabAssistant').click(); // from another tab, Back returns there
  btn.click();
  d.querySelector('#exportBack').click();
  assert.equal(d.querySelector('#panelAssistant').hidden, false);
});

// ---- sequence rows ----

function withSegment() {
  const p = kinds();
  const c = p.highlights.candidates[0];
  p.review = { segments: [{ id: c.id, source: c.source, start: c.start, end: c.end, label: c.label }] };
  return p;
}

test('a sequence row has its kind and name, and "cut N s + tail" only when the export runs past the cut', async (t) => {
  const page = await open(t, {
    project: withSegment(),
    routes: { 'POST /api/projects/p1/edges': () => ({ edges: { h001: { start: 10, end: 26.22, reason: 'event tail' } } }) },
  });
  const d = page.document;
  const row = () => d.querySelector('#seqList li[data-id="h001"]');
  assert.equal(row().dataset.kind, 'action');
  assert.equal(row().querySelector('.seq-kind').textContent, 'Action');
  assert.equal(row().querySelector('.seq-tail'), null, 'no tail until the server plans one');
  assert.equal(row().querySelector('.seq-dur').firstChild.textContent, '8.0 s');
  t.mock.timers.tick(400);
  await page.settle();
  assert.equal(row().querySelector('.seq-dur').firstChild.textContent, '16.2 s');
  assert.equal(row().querySelector('.seq-tail').textContent, 'cut 8.0 s + tail');
});

const OPTS = 'cutpoint.exportOptions';

const putTarget = { 'PUT /api/projects/p1/target': (req) => ({ target: req.body.seconds }) };

test('the footer total is a button: it opens a target field; set, over the target and clear', async (t) => {
  const page = await open(t, { project: withSegment(), routes: putTarget });
  const d = page.document;
  const total = d.querySelector('#seqTotal');
  const input = d.querySelector('#seqTargetInput');
  const submit = () => d.querySelector('#seqTargetForm').dispatchEvent(new page.window.Event('submit', { bubbles: true, cancelable: true }));
  assert.equal(total.tagName, 'BUTTON');
  assert.equal(total.textContent, '0:08');
  assert.equal(d.querySelector('#seqTarget').textContent, '', 'no target: the total alone');
  assert.equal(d.querySelector('#seqBar').hidden, true, 'no target, no bar');
  assert.match(total.title, /Set a target length/);
  total.click();
  assert.equal(d.querySelector('#seqTargetForm').hidden, false);
  assert.equal(total.getAttribute('aria-expanded'), 'true');
  assert.equal(d.activeElement, input, 'the field takes the focus');
  input.value = '0:16';
  submit();
  await page.settle();
  assert.equal(d.querySelector('#seqTargetForm').hidden, true);
  assert.equal(d.activeElement, total, 'focus returns to the total');
  assert.equal(d.querySelector('#seqTarget').textContent, 'of ~0:16');
  assert.equal(d.querySelector('#seqBar').hidden, false);
  assert.equal(d.querySelector('#seqBarFill').style.width, '50%');
  input.value = '4'; // seconds: shorter than the sequence
  total.click();
  input.value = '4';
  submit();
  await page.settle();
  assert.equal(d.querySelector('#seqBarFill').style.width, '100%', 'over the target: a full bar, never wider');
  assert.match(d.querySelector('#seqTarget').textContent, /^of ~0:04, 0:04 over$/);
  total.click();
  input.value = '';
  submit();
  await page.settle();
  assert.equal(d.querySelector('#seqTarget').textContent, '');
  assert.equal(d.querySelector('#seqBar').hidden, true, 'clearing removes the target');
});

test('the storyboard total carries the target in the footer words and follows it; it cuts nothing', async (t) => {
  const page = await open(t, { project: withSegment(), routes: putTarget });
  const d = page.document;
  const sb = d.querySelector('#sbTotal');
  const submit = () => d.querySelector('#seqTargetForm').dispatchEvent(new page.window.Event('submit', { bubbles: true, cancelable: true }));
  const setTarget = async (v) => {
    d.querySelector('#seqTotal').click();
    d.querySelector('#seqTargetInput').value = v;
    submit();
    await page.settle();
  };
  assert.equal(sb.textContent, '1 clip · 0:08', 'no target: plain');
  await setTarget('0:16');
  assert.equal(sb.textContent, '1 clip · 0:08 of ~0:16');
  await setTarget('4');
  assert.equal(sb.textContent, '1 clip · 0:08 of ~0:04, 0:04 over');
  assert.equal(d.querySelectorAll('#seqList li').length, 1, 'the sequence is untouched');
  await setTarget('');
  assert.equal(sb.textContent, '1 clip · 0:08');
});

test('a bad target is refused with a message and changes nothing; Escape closes the field', async (t) => {
  const page = await open(t, { project: withSegment() });
  const d = page.document;
  const input = d.querySelector('#seqTargetInput');
  const submit = () => d.querySelector('#seqTargetForm').dispatchEvent(new page.window.Event('submit', { bubbles: true, cancelable: true }));
  for (const bad of ['soon', '-5', '0', '1:2:3:4', '99999']) {
    if (d.querySelector('#seqTargetForm').hidden) d.querySelector('#seqTotal').click();
    input.value = bad;
    submit();
    assert.match(d.querySelector('#seqTargetErr').textContent, /Enter a length/, bad);
    assert.equal(input.getAttribute('aria-invalid'), 'true');
    assert.equal(d.querySelector('#seqTargetForm').hidden, false, 'the field stays open');
    assert.equal(d.querySelector('#seqBar').hidden, true);
    assert.equal(page.window.localStorage.getItem(OPTS), null, 'nothing stored');
  }
  input.dispatchEvent(new page.window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
  assert.equal(d.querySelector('#seqTargetForm').hidden, true);
  assert.equal(d.activeElement, d.querySelector('#seqTotal'));
});

test('the target is saved with the project, is not exported, does not mark the sequence unsaved and is not kept in the browser', async (t) => {
  const posted = [];
  const puts = [];
  const routes = {
    'POST /api/projects/p1/jobs': (req) => { posted.push(req.body); return { id: 'j1', status: 'running' }; },
    'PUT /api/projects/p1/target': (req) => { puts.push(req.body); return { target: req.body.seconds }; },
  };
  const page = await open(t, { project: withSegment(), routes });
  const d = page.document;
  d.querySelector('#seqTotal').click();
  d.querySelector('#seqTargetInput').value = '4:00';
  d.querySelector('#seqTargetForm').dispatchEvent(new page.window.Event('submit', { bubbles: true, cancelable: true }));
  await page.settle();
  assert.deepEqual(puts, [{ seconds: 240 }]);
  assert.equal(d.querySelector('#seqTarget').textContent, 'of ~4:00');
  assert.equal(page.window.localStorage.getItem(OPTS), null, 'nothing about it in this browser');
  assert.equal(d.querySelector('#saveBtn').disabled, true, 'the sequence is not dirty');
  assert.equal(d.querySelector('#optMaxTotal').value, '', 'Max video length is not touched');
  d.querySelector('#clipsBtn').click();
  await page.settle();
  for (const body of posted) assert.ok(!JSON.stringify(body).includes('240') && !('target' in body) && !('max_total' in body));
  page.close();

  const saved = withSegment();
  saved.review.target = 240;
  const again = await open(t, { project: saved });
  assert.equal(again.document.querySelector('#seqTarget').textContent, 'of ~4:00', 'read from the project, whatever the browser');
  const other = await open(t, { project: withSegment() });
  assert.equal(other.document.querySelector('#seqTarget').textContent, '', 'a project without one has none');
  const junk = withSegment();
  junk.review.target = 'x';
  const bad = await open(t, { project: junk });
  assert.equal(bad.document.querySelector('#seqBar').hidden, true);
});

test('a failed save of the target keeps the old value and says so', async (t) => {
  const routes = { 'PUT /api/projects/p1/target': reply(400, { detail: 'disk full' }) };
  const project = withSegment();
  project.review.target = 240;
  const page = await open(t, { project, routes });
  const d = page.document;
  d.querySelector('#seqTotal').click();
  d.querySelector('#seqTargetInput').value = '0:16';
  d.querySelector('#seqTargetForm').dispatchEvent(new page.window.Event('submit', { bubbles: true, cancelable: true }));
  await page.settle();
  assert.equal(d.querySelector('#seqTarget').hidden, true, 'the field is still open');
  assert.equal(d.querySelector('#seqTargetForm').hidden, false);
  assert.match(d.querySelector('#seqTargetErr').textContent, /Could not save the target length: disk full/);
  d.querySelector('#seqTargetCancel').click();
  assert.equal(d.querySelector('#seqTarget').textContent, 'of ~4:00', 'the old target stays');
  assert.equal(d.querySelector('#seqTargetErr').textContent, '');
});

for (const fails of [false, true]) {
  test(`a target save answered after another project opened ${fails ? 'with an error ' : ''}leaves that project alone`, async (t) => {
    let release;
    const gate = new Promise((r) => { release = r; });
    const p2 = makeProject({ id: 'p2', name: 'Second trip', review: { segments: [], target: 600 } });
    const routes = {
      ...bootRoutes(p2),
      'GET /api/projects': { projects: [{ id: 'p1', name: 'Project One' }, { id: 'p2', name: 'Second trip' }] },
      'PUT /api/projects/p1/target': async (req) => {
        await gate;
        return fails ? reply(500, { detail: 'disk full' }) : { target: req.body.seconds };
      },
    };
    const page = await open(t, { project: withSegment(), routes });
    const d = page.document;
    d.querySelector('#seqTotal').click();
    d.querySelector('#seqTargetInput').value = '0:16';
    d.querySelector('#seqTargetForm').dispatchEvent(new page.window.Event('submit', { bubbles: true, cancelable: true }));
    await page.settle(); // the PUT is in flight
    d.querySelector('#projectBtn').click();
    d.querySelector('#projectMenu [data-id=p2]').click();
    await page.settle();
    assert.equal(d.querySelector('#seqTarget').textContent, 'of ~10:00', 'p2 is open with its own target');
    release();
    await page.settle();
    assert.equal(d.querySelector('#seqTarget').textContent, 'of ~10:00', "p2's target stays");
    assert.equal(d.querySelector('#seqTargetErr').textContent, '');
    assert.equal(d.querySelector('#seqTargetForm').hidden, true, 'the form stays closed');
  });
}

test('Cancel while the target is being saved does not steal the focus when the save lands', async (t) => {
  let release;
  const gate = new Promise((r) => { release = r; });
  const routes = { 'PUT /api/projects/p1/target': async (req) => { await gate; return { target: req.body.seconds }; } };
  const page = await open(t, { project: withSegment(), routes });
  const d = page.document;
  d.querySelector('#seqTotal').click();
  d.querySelector('#seqTargetInput').value = '0:16';
  d.querySelector('#seqTargetForm').dispatchEvent(new page.window.Event('submit', { bubbles: true, cancelable: true }));
  await page.settle();
  d.querySelector('#seqTargetCancel').click();
  d.querySelector('#projectBtn').focus();
  release();
  await page.settle();
  assert.equal(d.activeElement, d.querySelector('#projectBtn'), 'the focus stays where the user moved it');
  assert.equal(d.querySelector('#seqTargetForm').hidden, true);
  assert.equal(d.querySelector('#seqTarget').textContent, 'of ~0:16', 'the saved target shows');
});

test('Escape on Set or Cancel closes the target field and does not reach the page', async (t) => {
  const page = await open(t, { project: withSegment() });
  const d = page.document;
  let reached = false;
  d.addEventListener('keydown', () => { reached = true; });
  d.querySelector('#seqTotal').click();
  d.querySelector('#seqTargetInput').value = 'soon';
  d.querySelector('#seqTargetForm').dispatchEvent(new page.window.Event('submit', { bubbles: true, cancelable: true }));
  assert.equal(d.querySelector('#seqTargetInput').getAttribute('aria-invalid'), 'true');
  d.querySelector('#seqTargetCancel').dispatchEvent(new page.window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
  assert.equal(d.querySelector('#seqTargetForm').hidden, true);
  assert.equal(reached, false);
  d.querySelector('#seqTotal').click();
  assert.equal(d.querySelector('#seqTargetInput').hasAttribute('aria-invalid'), false, 'no stale invalid mark');
});

test('a row drags as a whole except from its checkbox and buttons, and shows no status when its clip is ready', async (t) => {
  const page = await open(t, { project: withSegment() });
  const d = page.document;
  const row = d.querySelector('#seqList li[data-id="h001"]');
  const press = (el) => el.dispatchEvent(new page.window.Event('pointerdown', { bubbles: true }));
  assert.equal(row.querySelector('.seq-handle'), null, 'no drag handle');
  press(row.querySelector('.seq-title'));
  assert.equal(row.draggable, true);
  row.dispatchEvent(new page.window.Event('pointerup', { bubbles: true }));
  assert.equal(row.draggable, false);
  press(row.querySelector('[data-act=select]'));
  assert.equal(row.draggable, false, 'the checkbox does not start a drag');
  press(row.querySelector('[data-act=remove]'));
  assert.equal(row.draggable, false, 'nor do the row buttons');
  assert.equal(row.querySelector('.prep'), null, 'no "Ready" text or badge unless a clip is not ready');
});

test('a stored Export tab from before Export left the tabs starts on Sequence and is forgotten', async (t) => {
  const page = await open(t, { storage: { 'cutpoint.sideTab': 'export' } });
  const d = page.document;
  assert.equal(d.querySelector('#tabSequence').getAttribute('aria-selected'), 'true');
  assert.equal(d.querySelector('#panelExport').hidden, true);
  assert.equal(page.window.localStorage.getItem('cutpoint.sideTab'), null);
});

test('revealRow keeps the playing row above the chat input and the Sequence footer (--seq-foot-h)', async (t) => {
  const page = await open(t, { project: withSegment() });
  const d = page.document;
  const side = d.querySelector('#side');
  const rect = (top, bottom) => ({ top, bottom, left: 0, right: 0, width: 0, height: bottom - top });
  const row = d.querySelector('#seqList').firstElementChild;
  const win = page.window;
  const real = win.getComputedStyle.bind(win);
  win.getComputedStyle = (el, ...a) => {
    if (el !== side) return real(el, ...a);
    return { overflowY: 'auto', scrollPaddingTop: '10px', scrollPaddingBottom: '50px', getPropertyValue: (n) => (n === '--seq-foot-h' ? '100px' : '') };
  };
  side.getBoundingClientRect = () => rect(0, 500); // usable bottom: 500 - 50 - 100 = 350
  side.scrollTop = 0;
  const reveal = (bottom) => {
    row.getBoundingClientRect = () => rect(bottom - 40, bottom);
    side.scrollTop = 0;
    d.querySelector('#seqPlayBtn').click();
    page.window.document.querySelector('#seqPlayBtn').click(); // Stop
  };
  reveal(400); // under the footer's 350 line
  assert.equal(side.scrollTop, 50);
  reveal(340); // clear of it
  assert.equal(side.scrollTop, 0);
  // hidden (the Export panel covers the list): it measures as nothing, and the column stays where it is
  row.getBoundingClientRect = () => rect(0, 0);
  side.scrollTop = 200;
  d.querySelector('#seqPlayBtn').click();
  page.window.document.querySelector('#seqPlayBtn').click(); // Stop
  assert.equal(side.scrollTop, 200);
});
