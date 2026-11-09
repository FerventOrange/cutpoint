// The Export panel against the latest export: whether it still matches the sequence (smart tails
// included) and the "Assemble again" note when the joins differ from those of the final.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

const NOTE = 'Assemble again to apply the joins.';
const PUT_REVIEW = 'PUT /api/projects/p1/review';

/** A project with the sequence ``ids`` and an export of exactly those clips; ``clip`` edits a manifest entry. */
function exported(ids, { clip = () => ({}), over = {} } = {}) {
  const p = makeProject();
  const cands = ids.map((id) => p.highlights.candidates.find((c) => c.id === id));
  p.review = { segments: cands.map((c) => ({ id: c.id, source: c.source, start: c.start, end: c.end, label: c.label })) };
  p.clips = {
    dir: 'clips',
    clips: cands.map((c, i) => ({ file: `00${i + 1}_${c.id}.mp4`, id: c.id, start: c.start, end: c.end, ...clip(c, i) })),
  };
  return Object.assign(p, over);
}

async function open(t, project, routes = {}) {
  t.mock.timers.reset();
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project, routes });
  t.after(() => page.close());
  const d = page.document;
  const m = {
    page, d, project,
    $: (s) => d.querySelector(s),
    state: () => d.querySelector('#exportState').textContent,
    note: () => d.querySelector('#joinsNote'),
    live: () => d.querySelector('#sideLive').textContent,
    joins: () => [...d.querySelectorAll('#cutStrip button[data-join]')],
    async click(el) { (typeof el === 'string' ? d.querySelector(el) : el).click(); await page.settle(); },
  };
  d.querySelector('#optPicksSeq')?.click();
  await page.settle();
  return m;
}

// ---- exportMatches ----

test('an export with smart tails still matches the sequence: the manifest\'s original_end is compared, not the end with the tail', async (t) => {
  const m = await open(t, exported(['h001', 'h002'], { clip: (c) => ({ end: c.end + 1.5, pick: { start: c.start, end: c.end + 1.5 }, original_end: c.end }) }));
  assert.match(m.state(), /matches this sequence/);
  assert.doesNotMatch(m.state(), /differs/);
});

test('the Forza case: pick.end 1852.14 with a smart tail still matches a sequence end of 1849.38 (#775)', async (t) => {
  const m = await open(t, exported(['h001', 'h002'], { clip: (c, i) => (i ? { end: 1852.14, original_end: 1849.38, pick: { start: c.start, end: 1852.14 }, tail: { end: 1852.14, extension: 2.76, reason: 'settled' } } : {}) }));
  m.project.review.segments[1].end = 1849.38;
  m.page.close();
  const n = await open(t, m.project);
  assert.match(n.state(), /matches this sequence/);
  n.page.close();
  m.project.review.segments[1].end = 1848; // a real trim after the export
  const o = await open(t, m.project);
  assert.match(o.state(), /differs from this sequence/);
});

test('without original_end, pick.end is the sequence\'s end even when the clip\'s own start and end are wider', async (t) => {
  const m = await open(t, exported(['h001', 'h002'], { clip: (c) => ({ start: c.start - 1, end: c.end + 1, pick: { start: c.start, end: c.end } }) }));
  assert.match(m.state(), /matches this sequence/);
});

test('a clip without original_end is compared by its own end, as before', async (t) => {
  const same = await open(t, exported(['h001', 'h002']));
  assert.match(same.state(), /matches this sequence/);
  const longer = await open(t, exported(['h001', 'h002'], { clip: (c, i) => (i ? { end: c.end + 1.5 } : {}) }));
  assert.match(longer.state(), /differs from this sequence/);
});

test('a changed start or end, or another order, is still a difference', async (t) => {
  const cases = {
    'start': exported(['h001', 'h002'], { clip: (c, i) => (i ? { start: c.start + 1 } : {}) }),
    'end': exported(['h001', 'h002'], { clip: (c, i) => (i ? { original_end: c.end - 1, end: c.end + 0.5 } : {}) }),
    'order': (() => { const p = exported(['h001', 'h002']); p.clips.clips.reverse(); return p; })(),
  };
  for (const [name, p] of Object.entries(cases)) {
    const m = await open(t, p);
    assert.match(m.state(), /differs from this sequence/, name);
    m.page.close();
  }
});

// ---- the joins note ----

const BOTH = (transition) => [{ a: 'h001', b: 'h002', transition }, { a: 'h002', b: 'h003', transition }];
/** A project whose final was assembled with every cut coming out as ``transition`` (the Video transition was that). */
const withFinal = (finalJoins = BOTH('cut')) =>
  exported(['h001', 'h002', 'h003'], { over: { final_current: true, final_joins: finalJoins } });

/** A page whose server keeps what Save sends and answers the project with it. */
async function serving(t, project) {
  let status = null;
  const server = structuredClone(project);
  const m = await open(t, project, {
    'GET /api/projects/p1': () => server,
    'GET /api/jobs': () => ({ jobs: status ? [{ id: 'j1', project_id: 'p1', action: 'assemble', status, logs: [], progress: { total: null, steps: [] }, started: '' }] : [] }),
    [PUT_REVIEW]: (req) => { server.review = { ...server.review, ...req.body }; return {}; },
    'POST /api/projects/p1/jobs': (req) => { m.sent = req.body; status = 'running'; return { id: 'j1', status: 'running' }; },
  });
  m.server = server;
  m.hidden = () => m.note().hidden;
  m.finish = async () => { // the assemble job ends; the server has recorded the transitions it used
    status = 'completed';
    server.final_joins = server.final_joins.map((f) => ({ ...f, transition: (server.review.joins ?? []).find((j) => j.a === f.a && j.b === f.b)?.transition ?? m.sent.transition }));
    t.mock.timers.tick(5000);
    await m.page.settle();
  };
  m.global = (value) => m.click(`#cutTransition button[data-value=${value}]`);
  await m.click('#stageMode [data-mode=cut]');
  return m;
}

test('final under Cut: a join set to Cut shows no note, Fade shows it, Default clears it; the live region follows', async (t) => {
  const m = await serving(t, withFinal());
  assert.equal(m.hidden(), true);
  assert.equal(m.note().textContent, '');
  await m.click(m.joins()[0]); // Cut: what the final has
  assert.equal(m.hidden(), true);
  assert.equal(m.live(), 'Transition of cut 1 set to Cut');
  await m.click(m.joins()[0]); // Fade
  assert.equal(m.hidden(), false);
  assert.equal(m.note().textContent, NOTE);
  assert.equal(m.live(), `Transition of cut 1 set to Fade. ${NOTE}`);
  await m.click(m.joins()[0]); // Default (Cut)
  assert.equal(m.hidden(), true);
  assert.equal(m.note().textContent, '');
  assert.equal(m.live(), 'Transition of cut 1 set to Default (Cut)');
});

test('final under Fade: a join set to Fade, or reset to Default under Fade, shows no note', async (t) => {
  const m = await serving(t, withFinal(BOTH('fade')));
  assert.equal(m.hidden(), false, 'the Video transition is still Cut');
  await m.global('fade');
  assert.equal(m.hidden(), true);
  await m.click(m.joins()[0]); // Cut
  assert.equal(m.hidden(), false);
  await m.click(m.joins()[0]); // its own Fade
  assert.equal(m.hidden(), true);
  await m.click(m.joins()[0]); // Default, which is Fade
  assert.equal(m.hidden(), true);
  assert.equal(m.live(), 'Transition of cut 1 set to Default (Fade)');
});

test('final under Fade: Cut in the whole-video control shows the note and says it; Fade again clears it', async (t) => {
  const m = await serving(t, withFinal(BOTH('fade')));
  await m.global('fade');
  assert.equal(m.hidden(), true);
  await m.global('cut');
  assert.equal(m.hidden(), false);
  assert.equal(m.live(), `Default transition set to Cut. ${NOTE}`);
  await m.global('fade');
  assert.equal(m.hidden(), true);
  assert.equal(m.live(), 'Default transition set to Fade');
});

test('a change of the Video transition field re-renders the note', async (t) => {
  const m = await serving(t, withFinal());
  assert.equal(m.hidden(), true);
  const field = m.$('#optTransition');
  field.value = 'fade';
  field.dispatchEvent(new m.page.window.Event('change', { bubbles: true }));
  await m.page.settle();
  assert.equal(m.hidden(), false);
  field.value = 'cut';
  field.dispatchEvent(new m.page.window.Event('change', { bubbles: true }));
  await m.page.settle();
  assert.equal(m.hidden(), true);
});

test('a join on a pair the final does not have shows no note', async (t) => {
  const m = await serving(t, withFinal([{ a: 'h001', b: 'h002', transition: 'cut' }]));
  await m.click(m.joins()[1]); // h002 -> h003
  await m.click(m.joins()[1]); // Fade
  assert.equal(m.hidden(), true);
  assert.equal(m.live(), 'Transition of cut 2 set to Fade');
});

test('the live region carries the note only while the Export panel is closed', async (t) => {
  const m = await serving(t, withFinal());
  await m.click('#exportBtn');
  assert.equal(m.$('#panelExport').hidden, false);
  await m.click(m.joins()[0]);
  await m.click(m.joins()[0]); // Fade
  assert.equal(m.hidden(), false, 'the open panel shows it');
  assert.equal(m.live(), 'Transition of cut 1 set to Fade');
  await m.click('#exportBtn'); // closed
  assert.equal(m.$('#panelExport').hidden, true);
  await m.click(m.joins()[0]); // Default
  await m.click(m.joins()[0]); // Cut
  await m.click(m.joins()[0]); // Fade
  assert.equal(m.live(), `Transition of cut 1 set to Fade. ${NOTE}`);
});

test('Undo back to the exported joins clears the note', async (t) => {
  const m = await serving(t, withFinal());
  await m.click(m.joins()[0]);
  await m.click(m.joins()[0]); // Fade
  assert.equal(m.hidden(), false);
  await m.click('#undoBtn');
  await m.click('#undoBtn');
  assert.equal(m.hidden(), true);
  assert.equal(m.note().textContent, '');
});

test('Assemble clears the note once the final has the new joins', async (t) => {
  const m = await serving(t, withFinal());
  await m.click(m.joins()[0]);
  await m.click(m.joins()[0]); // Fade
  assert.equal(m.hidden(), false);
  await m.click('#assembleBtn');
  await m.finish();
  assert.deepEqual(m.server.final_joins[0], { a: 'h001', b: 'h002', transition: 'fade' });
  assert.equal(m.hidden(), true);
});

test('with the final\'s joins unknown (final_joins null) the note never shows', async (t) => {
  const m = await serving(t, withFinal(null));
  await m.click(m.joins()[0]);
  await m.click(m.joins()[0]); // Fade
  assert.equal(m.hidden(), true);
  assert.equal(m.live(), 'Transition of cut 1 set to Fade');
  await m.global('fade');
  assert.equal(m.hidden(), true);
});

test('the note does not show for a final that is not the current export\'s', async (t) => {
  const p = withFinal();
  p.final_current = false;
  const m = await serving(t, p);
  await m.click(m.joins()[0]);
  await m.click(m.joins()[0]); // Fade
  assert.equal(m.hidden(), true);
});
