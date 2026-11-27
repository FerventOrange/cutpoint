// cutpoint editing workspace: the style button and its menu (#897).
import { $, h } from './util.js';
import { analyzeAgain, BUSY_TEXT, heldElsewhere, icon, S } from './app.js';

/** The name of ``style`` in the style menu: its category's (Talk for commentary), else the style's own. */
const styleName = (d, style) => d.choices.find((c) => c.style === style)?.label ?? style;

/** The style button's text for the project's ``detected`` view. */
function styleChipText(d) {
  if (!d.auto) return `Style: ${styleName(d, d.chosen)} (your choice)`;
  if (d.category && !d.stale) return d.ambiguous ? `Not sure: ${d.fallback_label} style` : `Detected: ${d.label}`;
  if (d.category) return `Detected earlier: ${d.label}`;
  return 'Auto: analyze again to detect';
}

/** The text lines of the style menu: why (the reason as recorded), the sources that disagree, an earlier detection, a suggestion. */
function styleLines(d) {
  if (!d.category) return [];
  const why = d.reason ? ` (${d.reason})` : '';
  const lines = [];
  if (d.stale) lines.push(d.ambiguous ? `Detected earlier: not sure, leans ${d.label}${why}` : `Detected earlier: ${d.label}${why}`);
  else if (d.ambiguous) lines.push(`Leans ${d.label}${why}; used the ${d.fallback_label} style${d.used_style ? ` (${d.used_style})` : ''}`);
  else lines.push(`Detected: ${d.label}${why}`);
  for (const s of d.sources) lines.push(`Disagree: ${s.name} looks like ${s.label ?? s.category}${s.reason ? ` (${s.reason})` : ''}`);
  if (d.suggest && d.suggest !== d.chosen) lines.push(`Suggested: ${styleName(d, d.suggest)} (${d.suggest})`);
  return lines;
}

/**
 * The header's style button: hidden before the first analysis and for 360 projects (``detected`` null).
 * Its menu: the lines of ``styleLines``, then radio items Auto, the categories with their style (Gameplay, Action, Action cam, Talk; General is only a detection label)
 * (Action cam only with camera telemetry or when chosen, else under Other styles) and Other styles
 * (the rest of the session's styles). Choosing one asks first in the Analyze dialog; while a job runs
 * or another run holds the project the items are aria-disabled (still focusable), as Analyze again.
 */
export function renderStyleMenu() {
  const wrap = $('#styleMenuWrap');
  const p = S.id && S.project?.id === S.id ? S.project : null;
  const d = p?.detected ?? null;
  wrap.hidden = !d;
  const pop = $('#styleMenu');
  if (!d) {
    if (!pop.hidden) closeStyleMenu();
    pop.dataset.sig = '';
    return;
  }
  const text = styleChipText(d);
  $('#styleName').textContent = text;
  $('#styleBtn').title = text;
  const busy = !!S.active || heldElsewhere(p);
  const why = S.active ? 'Another job is running.' : BUSY_TEXT;
  const main = d.choices.filter((c) => c.main);
  const others = S.styles.filter((s) => s !== 'auto' && !main.some((c) => c.style === s));
  const lines = styleLines(d);
  const sig = JSON.stringify([p.id, d, others, busy, why]);
  if (pop.dataset.sig === sig) return;
  pop.dataset.sig = sig;
  const had = pop.contains(document.activeElement) ? document.activeElement.dataset.style : null;
  const item = (style, label, name = label) => h('button', {
    type: 'button', role: 'menuitemradio', class: 'proj-item style-item', tabindex: '-1', 'data-style': style,
    'aria-checked': String(style === d.chosen), 'aria-disabled': busy ? 'true' : null,
    'aria-describedby': busy ? 'styleBusy' : null, title: busy ? why : null,
    onclick: () => pickStyle(style, name),
  }, icon('check'), h('span', { text: label }));
  pop.replaceChildren(
    ...(lines.length ? [h('div', { class: 'proj-stats', id: 'styleLines', role: 'none' }, ...lines.map((l) => h('span', { text: l }))),
      h('div', { class: 'proj-sep', role: 'separator' })] : []),
    item('auto', 'Auto (detect from footage)', 'Auto'),
    ...main.map((c) => item(c.style, `${c.label}: ${c.style}`, c.label)),
    ...(others.length ? [h('div', { class: 'proj-sep', role: 'separator' }),
      h('div', { class: 'style-group-label', id: 'styleOtherLabel', role: 'none', text: 'Other styles' }),
      h('div', { role: 'group', 'aria-labelledby': 'styleOtherLabel', class: 'style-group' },
        ...others.map((s) => item(s, styleName(d, s))))] : []),
    ...(busy ? [h('div', { class: 'vh', id: 'styleBusy', role: 'none', text: why })] : []),
  );
  // the lines sit outside the menu's items, so the menu itself carries them to a screen reader
  if (lines.length) pop.setAttribute('aria-describedby', 'styleLines'); else pop.removeAttribute('aria-describedby');
  if (had != null) (pop.querySelector(`[data-style="${CSS.escape(had)}"]`) ?? pop.querySelector('.style-item'))?.focus();
}

/** Close the style menu; ``refocus`` returns focus to its button. */
function closeStyleMenu(refocus = false) {
  $('#styleMenu').hidden = true;
  $('#styleBtn').setAttribute('aria-expanded', 'false');
  if (refocus) $('#styleBtn').focus();
}

/** A style item (``name``: what the dialog calls it): the current one only closes the menu; another asks in the Analyze dialog, then analyzes in it. */
function pickStyle(style, name) {
  const item = $('#styleMenu').querySelector(`[data-style="${CSS.escape(style)}"]`);
  if (item?.getAttribute('aria-disabled') === 'true') return; // a job runs: the menu stays open
  closeStyleMenu(true);
  if (style === S.project?.detected?.chosen) return;
  analyzeAgain(false, { style, name });
}

/** The style menu: a menu button that works like the project menu (arrows, Home, End, Escape, Tab, a click outside). */
export function initStyleMenu() {
  const btn = $('#styleBtn');
  const pop = $('#styleMenu');
  const wrap = $('#styleMenuWrap');
  const items = () => [...pop.querySelectorAll('.style-item')];
  const show = (last = false) => {
    pop.hidden = false;
    btn.setAttribute('aria-expanded', 'true');
    (last ? items().at(-1) : pop.querySelector('[aria-checked=true]') ?? items()[0])?.focus();
  };
  btn.addEventListener('click', () => (pop.hidden ? show() : closeStyleMenu()));
  btn.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault();
    if (pop.hidden) show(e.key === 'ArrowUp');
  });
  pop.addEventListener('mousedown', (e) => !e.target.closest('.style-item') && e.preventDefault());
  pop.addEventListener('keydown', (e) => {
    const all = items();
    const at = all.indexOf(document.activeElement);
    const to = { ArrowDown: (at + 1) % all.length, ArrowUp: (at - 1 + all.length) % all.length, Home: 0, End: all.length - 1 }[e.key];
    if (to != null) {
      e.preventDefault();
      all[to].focus();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      closeStyleMenu(true);
    } else if (e.key === 'Tab') closeStyleMenu();
  });
  document.addEventListener('pointerdown', (e) => !pop.hidden && !wrap.contains(e.target) && closeStyleMenu());
}
