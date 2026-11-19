// cutpoint editing workspace: preferences kept in this browser's storage (Settings toggles, theme,
// column widths, volume, the side tab and the moments list's order and filters).
import { $, baseName, filtersOn, h, naturalCompare, NO_FILTERS, readStore, themeFromStored, writeStore } from './util.js';
import { blend, cands, dip, els, icon, OPT_FIELDS, OPT_SELECTS, popOpen, renderMoments, renderNerdStats, S } from './app.js';
import { openModelsTab } from './models.js';

const SIDE_KEY = 'cutpoint.sideWidth'; // localStorage: the sequence column width the user dragged to
const SIDE_MIN = 290; // pixels; the column's CSS default is 290-360
const SIDE_MAX = 640;
const MOM_KEY = 'cutpoint.momentsWidth'; // localStorage: the moments column width the user dragged to
const MOM_MIN = 260; // pixels; the column's CSS default is 260-300
const MOM_MAX = 640;
const STAGE_MIN = 360; // pixels the preview keeps when either column widens (app.css: the stage track)
const SPLIT_STEP = 16; // pixels per arrow key on a column splitter; Shift moves four steps
const VOL_KEY = 'cutpoint.volume'; // localStorage: {volume: 0-1, muted} for every preview video
const NERD_KEY = 'cutpoint.nerdStats'; // localStorage: "1" while Stats for Nerds is on (off by default)
const ADV_KEY = 'cutpoint.advancedExport'; // localStorage: "1" while Show advanced export options is on (off by default)
const THEME_KEY = 'cutpoint.theme'; // localStorage: "light" or "blue"; absent for Graphite, the default (a stored "dark" is dropped). index.html reads it too
const VERBOSE_KEY = 'cutpoint.verboseMoments'; // localStorage: "1" while Verbose moment labels is on (off by default)
const KEYS_KEY = 'cutpoint.singleKeys'; // localStorage: "0" while Single-key shortcuts is off (WCAG 2.1.4); absent for on, the default
const TAB_KEY = 'cutpoint.sideTab'; // localStorage: the sequence column's tab; absent for Sequence, the default
const ORDER_KEY = 'cutpoint.momentOrder'; // localStorage: "position" for video order; absent for Score, the default
const ORDER_NAMES = { score: 'score', position: 'position in video' }; // the moments list's orders (#469)
// localStorage: the moments list's quick filters the user set ({seq, ready, must}); absent for none. The source file is per project
const FILTERS_KEY = 'cutpoint.momentFilters';

export let previewVolume = () => 1; // the shared volume of the preview videos (initVolume)

/**
 * A Settings check box kept in this browser's storage, not in config.json (A424): ``key`` holds
 * '1' while the box is on and is absent for off, or with ``byDefault`` on, '0' while it is off and
 * absent for on. ``set(on)`` sets ``S[prop]``, stores it, ticks ``box`` and calls ``apply(on)``;
 * ``load()`` reads ``S[prop]`` from storage (blocked storage: the default), wires the box to
 * ``set`` and returns it. The box previews on change; Save keeps it and Cancel or Escape puts back
 * the state the dialog opened with (bindSettings). Without storage it lasts the session.
 */
function storedToggle({ key, box, prop, byDefault = false, apply }) {
  const set = (on) => {
    S[prop] = on;
    writeStore(key, on === byDefault ? null : on ? '1' : '0');
    $(box).checked = on;
    apply(on);
  };
  const load = () => {
    const v = readStore(key);
    S[prop] = byDefault ? v !== '0' : v === '1';
    $(box).addEventListener('change', (e) => set(e.target.checked));
    return S[prop];
  };
  return { set, load };
}

/** Stats for Nerds (off by default): the scoring overlay on the preview. */
export const NERD = storedToggle({ key: NERD_KEY, box: '#setNerdStats', prop: 'nerdStats', apply: () => renderNerdStats() });

/** Show or hide the Export panel's advanced controls (.adv) and say which hidden ones are in use. */
export function showAdvExport(on) {
  $('#panelExport').classList.toggle('show-adv', on);
  renderAdvInUse();
}

/**
 * Hidden advanced fields keep their values and are still sent, so a filled one that a job sends
 * (enabled: the pick mode and boxes use it) is named under the options instead of acting unseen.
 */
export function renderAdvInUse() {
  const out = $('#advInUse');
  const shown = $('#panelExport').classList.contains('show-adv');
  const used = shown ? [] : [...Object.values(OPT_FIELDS), ...Object.values(OPT_SELECTS)].map((id) => $(id))
    .filter((el) => el.classList.contains('adv') && !el.disabled && el.value.trim())
    .map((el) => document.querySelector(`label[for=${el.id}]`).textContent.replace(/ \(s\)$/, ''));
  out.hidden = !used.length;
  out.textContent = used.length ? `Advanced options in use: ${used.join(', ')}. Settings > Show advanced export options shows them.` : '';
}

/** Show advanced export options (off by default): the Export panel's .adv controls. */
export const ADV = storedToggle({ key: ADV_KEY, box: '#setAdvExport', prop: 'advExport', apply: showAdvExport });

/** Verbose moment labels (off by default): the tiles' file, id and times. */
export const VERBOSE = storedToggle({
  key: VERBOSE_KEY, box: '#setVerboseMoments', prop: 'verboseMoments',
  apply: (on) => $('#candList').classList.toggle('moments-compact', !on), // CSS hides the parts; the filter still searches them
});

/** Show the keys the page answers to in the accessible names and tooltips; with Single-key shortcuts off, no character key is promised. */
function renderKeyHints() {
  const on = S.singleKeys;
  for (const el of document.querySelectorAll('[data-keys]')) {
    if (on) el.setAttribute('aria-keyshortcuts', el.dataset.keys);
    else el.removeAttribute('aria-keyshortcuts');
  }
  for (const [id, key] of [['startHandle', 'I'], ['endHandle', 'O']]) {
    $(`#${id}`).title = `Drag, or use the arrow keys.${on ? ` ${key} sets it at the playhead.` : ''}`;
  }
}

/** Single-key shortcuts (M, [, ], I, O; on by default, #748) can be turned off to meet WCAG 2.1.4; on is stored as no key. */
export const KEYS = storedToggle({ key: KEYS_KEY, box: '#setSingleKeys', prop: 'singleKeys', byDefault: true, apply: renderKeyHints });

/** The theme in use: index.html set data-theme on <html> before first paint; none is Graphite. */
export function theme() {
  return document.documentElement.dataset.theme ?? 'graphite';
}

/** Switch the theme (#445): recolour the page at once and store it in this browser; Graphite is stored as no key. */
export function setTheme(name) {
  name = themeFromStored(name);
  if (name === 'graphite') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = name;
  writeStore(THEME_KEY, name === 'graphite' ? null : name); // blocked or full: it still applies until reload
  $('#setTheme').value = name;
}

/** Boot: the stored choice, if index.html's inline script did not apply it (it also drops a stored "dark": Graphite replaced that theme). */
export function initTheme() {
  setTheme(readStore(THEME_KEY)); // storage blocked: Graphite
}


/**
 * The separators beside the moments column and the sequence column: drag, arrow keys (Home/End for
 * the limits) or double-click to reset. Each width is remembered per browser; without storage it
 * lasts the session. A column never widens past half the window, nor so far that the preview and
 * the other column together get less than STAGE_MIN.
 */
export function initSplitters() {
  const ws = $('#workspace');
  const applies = [];
  // Each column's limit depends on the other's width: apply ``first``, then the other (its limit
  // and its CSS default follow), then ``first`` once more, as its limit follows the other's width.
  const refresh = (first) => [first, ...applies.filter((a) => a !== first), first].forEach((a) => a());
  const splitter = ({ bar, col, other, prop, key, min, maxW, dir }) => {
    let pref = Number(readStore(key)) || null; // dragged width in pixels, or null for the CSS default
    const store = () => writeStore(key, pref == null ? null : String(pref)); // blocked or full: it still applies until reload
    const max = () => Math.max(min, Math.min(maxW, Math.floor(innerWidth / 2),
      ws.clientWidth - other.offsetWidth - STAGE_MIN));
    const clamp = (w, hi = max()) => Math.round(Math.min(Math.max(w, min), hi));
    // pref stays as chosen; only what is applied is clamped, so a wider window gets it back.
    // Each apply reads layout before its own write; refresh() chains three, so a change costs up to
    // two forced relayouts (at most one change per frame while dragging).
    const apply = () => {
      const hi = max();
      const w = pref == null ? null : clamp(pref, hi);
      if (w == null) ws.style.removeProperty(prop);
      else ws.style.setProperty(prop, `${w}px`);
      bar.setAttribute('aria-valuemin', min);
      bar.setAttribute('aria-valuemax', hi);
      bar.setAttribute('aria-valuenow', w ?? col.offsetWidth);
    };
    const set = (w) => {
      pref = w == null ? null : clamp(w);
      refresh(apply);
    };

    bar.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      bar.setPointerCapture(e.pointerId);
      bar.classList.add('dragging');
      const x0 = e.clientX;
      const w0 = col.offsetWidth;
      // At most one resize per frame, however fast the pointer reports.
      let next = null;
      let frame = 0;
      const flush = () => {
        frame = 0;
        if (next != null) set(next);
        next = null;
      };
      const move = (ev) => {
        next = w0 + dir * (ev.clientX - x0); // dragging away from the column widens it
        if (!frame) frame = requestAnimationFrame(flush);
      };
      const end = () => {
        cancelAnimationFrame(frame);
        flush();
        bar.classList.remove('dragging');
        bar.removeEventListener('pointermove', move);
        bar.removeEventListener('lostpointercapture', end);
        store();
      };
      bar.addEventListener('pointermove', move);
      bar.addEventListener('lostpointercapture', end);
    });
    bar.addEventListener('keydown', (e) => {
      const step = (e.shiftKey ? SPLIT_STEP * 4 : SPLIT_STEP) * dir;
      const w = col.offsetWidth;
      const next = { ArrowLeft: w - step, ArrowRight: w + step, Home: min, End: max() }[e.key];
      if (next == null) return;
      e.preventDefault();
      set(next);
      store();
    });
    bar.addEventListener('dblclick', () => {
      set(null);
      store();
    });
    return apply;
  };
  const side = $('#side');
  const moments = $('.moments');
  // dir: +1 when the column is left of its bar (dragging right widens it), -1 when right of it
  applies.push(
    splitter({ bar: $('#sideSplit'), col: side, other: moments, prop: '--side-w', key: SIDE_KEY,
      min: SIDE_MIN, maxW: SIDE_MAX, dir: -1 }),
    splitter({ bar: $('#momSplit'), col: moments, other: side, prop: '--mom-w', key: MOM_KEY,
      min: MOM_MIN, maxW: MOM_MAX, dir: 1 }),
  );
  addEventListener('resize', () => refresh(applies[0]));
  refresh(applies[0]);
}

/**
 * Mute and volume for every preview video (the stage and the final video), in the player's bar
 * on the stage. The setting is remembered per browser; without storage it lasts the session. The
 * final video's own controls change it too, so each video's volumechange feeds back into the shared
 * setting.
 */
export function initVolume() {
  const vids = [...els, $('#finalVideo')];
  const DEFAULT_VOL = { volume: 1, muted: false }; // the browser default when nothing is stored
  let vol = { ...DEFAULT_VOL };
  const saved = readStore(VOL_KEY, null, { json: true }); // blocked or unreadable: the default
  const n = Number(saved?.volume);
  if (Number.isFinite(n) && n >= 0 && n <= 1) vol.volume = n;
  if (typeof saved?.muted === 'boolean') vol.muted = saved.muted;
  // blocked or full: the setting still applies until reload
  const store = () => writeStore(VOL_KEY, vol.volume === DEFAULT_VOL.volume && vol.muted === DEFAULT_VOL.muted ? null : vol);
  const btn = h('button', { type: 'button', class: 'btn icon-btn', 'aria-label': 'Mute', title: 'Mute', 'aria-keyshortcuts': S.singleKeys ? 'M' : null, 'data-keys': 'M' });
  const range = h('input', { type: 'range', min: 0, max: 100, step: 1, 'aria-label': 'Volume' });
  const apply = () => {
    for (const el of vids) {
      if (el.volume !== vol.volume) el.volume = vol.volume;
      if (el.muted !== vol.muted) el.muted = vol.muted;
    }
    const pct = Math.round(vol.volume * 100);
    range.value = pct;
    range.setAttribute('aria-valuetext', `${pct}%`);
    btn.setAttribute('aria-pressed', String(vol.muted));
    btn.replaceChildren(icon(vol.muted ? 'muted' : 'volume'));
  };
  const set = (next) => {
    if (next.volume === vol.volume && next.muted === vol.muted) return;
    vol = next;
    apply();
    store();
  };
  // unmuting at zero would stay silent, so it brings the volume back up as native controls do
  const toggleMute = () => set({ volume: vol.muted && vol.volume === 0 ? 1 : vol.volume, muted: !vol.muted });
  btn.addEventListener('click', toggleMute);
  // Space stays play/pause here too: the global keydown handler plays and cancels the keydown, and the
  // button's own Space click (fired on keyup) is cancelled here, so Space never mutes
  btn.addEventListener('keyup', (e) => {
    if (e.key === ' ') e.preventDefault();
  });
  range.addEventListener('input', () => {
    // moving the slider unmutes, as native controls do, unless it is at zero
    const volume = Number(range.value) / 100;
    set({ volume, muted: vol.muted && volume === 0 });
  });
  // a Fade (or its dip to black) ramps the stage players itself: those volumes are not the setting
  for (const el of vids) el.addEventListener('volumechange', () => ((blend || dip) && el !== vids.at(-1) ? null : set({ volume: el.volume, muted: el.muted })));
  previewVolume = () => vol.volume;
  $('.transport-end').append(h('div', { class: 'vol', role: 'group', 'aria-label': 'Preview sound' }, btn, range));
  apply();
  return toggleMute;
}

/**
 * The sequence column's tabs (Sequence, AI Chat, Models; more can be added as a button and a panel):
 * click, or arrow keys and Home/End on the focused tab, which select as they move (automatic
 * activation). The choice is remembered per browser; without storage it lasts the session. The
 * header's Export shows the Export panel in this column with no tab selected, under its own heading
 * and a Back button; Back, Escape, a second click on Export or any tab returns to the tab it came
 * from, and the preview and running jobs are not touched. Returns showTab(name, focus) for the tabs
 * (Export has no tab name: only the header's button opens it).
 */
export function initSideTabs() {
  const tabs = [...document.querySelectorAll('#side [role=tab]')];
  const exportBtn = $('#exportBtn');
  const exportPanel = $('#panelExport');
  let currentTab = null; // the selected tab; while Export is open, the one Back returns to
  let exportOpen = false;
  const closeExport = () => {
    exportOpen = false;
    exportPanel.hidden = true;
    for (const vid of exportPanel.querySelectorAll('video')) vid.pause(); // the final video: not heard unseen
    exportBtn.setAttribute('aria-expanded', 'false');
  };
  const select = (tab, focus = false) => {
    if (tab === currentTab && !exportOpen) {
      if (focus) tab.focus();
      return;
    }
    const prev = currentTab;
    if (exportOpen) closeExport();
    currentTab = tab;
    for (const t of tabs) {
      const on = t === tab;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      const panel = $(`#${t.getAttribute('aria-controls')}`);
      panel.hidden = !on;
    }
    if (tab.dataset.tab === 'models' && prev?.dataset.tab !== 'models') openModelsTab();
    if (focus) tab.focus();
    writeStore(TAB_KEY, tab === tabs[0] ? null : tab.dataset.tab); // blocked or full: it still shows until reload
  };
  // Export is not a tab and is not remembered: the tab it hides stays the stored choice and keeps the tab stop
  const openExport = () => {
    if (!exportOpen) {
      exportOpen = true;
      for (const t of tabs) {
        t.setAttribute('aria-selected', 'false');
        $(`#${t.getAttribute('aria-controls')}`).hidden = true;
      }
      exportPanel.hidden = false;
      exportBtn.setAttribute('aria-expanded', 'true');
    }
    $('#exportTitle').focus();
  };
  const backFromExport = () => {
    if (exportOpen) select(currentTab);
    exportBtn.focus();
  };
  const saved = readStore(TAB_KEY); // storage blocked: the first tab
  const initial = tabs.find((t) => t.dataset.tab === saved) ?? tabs[0]; // an unknown stored tab starts the first
  select(initial);
  for (const t of tabs) t.addEventListener('click', () => select(t));
  exportBtn.addEventListener('click', () => {
    if (exportOpen) {
      backFromExport();
    } else {
      openExport();
      $('#side').scrollIntoView({ block: 'start' }); // when the column is below the preview, the page scrolls to it
    }
  });
  $('#exportBack').addEventListener('click', backFromExport);
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || !exportOpen || e.defaultPrevented || document.querySelector('dialog[open]') || popOpen()) return;
    // from the panel or the button, or with nothing focused; Escape in another column keeps its own meaning
    if (!exportPanel.contains(e.target) && e.target !== exportBtn && e.target !== document.body) return;
    e.preventDefault(); // the page's own Escape (stop Play sequence) waits
    backFromExport();
  });
  // rows brought into view (focus, revealRow) land between the sticky tab bar and panel header and
  // the sticky chat input; their heights follow the column's width and the selected tab (a hidden
  // panel's measure 0)
  const bars = [$('#side .side-bar'), $('#panelSequence .col-head'), $('#panelAssistant .chat-head')].filter(Boolean);
  const pad = new ResizeObserver(() => {
    const side = $('#side');
    const px = `${bars.reduce((n, el) => n + el.offsetHeight, 0)}px`;
    side.style.scrollPaddingTop = px;
    side.style.scrollPaddingBottom = `${$('#chatForm').offsetHeight}px`;
    // the Sequence footer is not in the padding: focusing inside it would scroll the column; the rows keep clear of it by scroll-margin
    side.style.setProperty('--seq-foot-h', `${$('#seqFoot').offsetHeight}px`);
  });
  for (const el of [...bars, $('#chatForm'), $('#seqFoot')]) pad.observe(el);
  $('#side [role=tablist]').addEventListener('keydown', (e) => {
    const i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    const next = { ArrowLeft: i - 1, ArrowRight: i + 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (next == null) return;
    e.preventDefault();
    select(tabs[(next + tabs.length) % tabs.length], true);
  });
  return (name, focus = false) => {
    const tab = tabs.find((t) => t.dataset.tab === name);
    if (tab) select(tab, focus); // selecting the selected tab only moves focus
  };
}

/**
 * The kind filter above the Moments grid: All or one of util.js KINDS, buttons that press in turn
 * (aria-pressed); renderMoments hides a kind no moment has. A view of the list only: it is not
 * stored and a project switch clears it.
 */
export function initKindFilter() {
  $('#kindSeg').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b || b.dataset.kind === S.kind) return;
    S.kind = b.dataset.kind;
    renderMoments(); // also presses the button
    const n = $('#candList').querySelectorAll('.cand').length;
    $('#orderLive').textContent = `${n} of ${cands().length} moments shown`;
  });
}

/**
 * The moments list's order (#469): Score (rank #1 first, the default) or Position in video, the
 * Sort by radios of Sort and filter. The choice is remembered per browser; without storage it lasts
 * the session.
 */
export function initMomentOrder() {
  const radios = [...document.querySelectorAll('#filterPop input[name=momentOrder]')];
  const v = readStore(ORDER_KEY); // storage blocked: Score
  if (Object.hasOwn(ORDER_NAMES, v)) S.order = v;
  for (const r of radios) {
    r.checked = r.value === S.order;
    r.addEventListener('change', () => {
      if (!r.checked) return;
      S.order = Object.hasOwn(ORDER_NAMES, r.value) ? r.value : 'score';
      writeStore(ORDER_KEY, S.order === 'score' ? null : S.order); // blocked or full: it still applies until reload
      $('#orderLive').textContent = `Moments sorted by ${ORDER_NAMES[S.order]}`;
      renderMoments();
    });
  }
}

/**
 * Sort and filter: the funnel button beside the search opens the sort order and the quick filters
 * (a disclosure: Escape closes it and returns focus to the button; a click or focus outside closes
 * it). The filters combine with the search text (util.js momentMatches). Sequence and Ready only
 * are remembered per browser like the order; the source file only while its project stays open
 * (resetWorkspace), as paths differ per project. Without storage they last the session.
 */
export function initMomentFilters() {
  const btn = $('#filterBtn');
  const pop = $('#filterPop');
  const menu = btn.parentElement;
  btn.prepend(icon('funnel'));
  const saved = readStore(FILTERS_KEY, null, { json: true }); // blocked or not JSON: no filters
  if (['all', 'in', 'out'].includes(saved?.seq)) S.filters.seq = saved.seq;
  S.filters.ready = saved?.ready === true;
  const seqRadios = [...pop.querySelectorAll('input[name=filterSeq]')];
  const showFilters = () => {
    for (const r of seqRadios) r.checked = r.value === S.filters.seq;
    $('#filterReady').checked = S.filters.ready;
    $('#filterBits').checked = S.filters.bits;
  };
  showFilters();
  const open = (on, refocus = false) => {
    pop.hidden = !on;
    btn.setAttribute('aria-expanded', String(on));
    if (refocus) btn.focus();
  };
  btn.addEventListener('click', () => open(pop.hidden));
  menu.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || pop.hidden) return;
    e.preventDefault(); // the page's own Escape (stop Play sequence) waits
    open(false, true);
  });
  document.addEventListener('pointerdown', (e) => !pop.hidden && !menu.contains(e.target) && open(false));
  menu.addEventListener('focusout', (e) => !pop.hidden && e.relatedTarget && !menu.contains(e.relatedTarget) && open(false));
  const changed = () => {
    const { seq, ready } = S.filters;
    writeStore(FILTERS_KEY, filtersOn({ seq, ready }) ? { seq, ready } : null); // blocked or full: they still apply until reload
    renderMoments();
    const n = $('#candList').querySelectorAll('.cand').length;
    $('#orderLive').textContent = `${n} of ${cands().length} moments shown`;
  };
  for (const r of seqRadios)
    r.addEventListener('change', () => {
      if (!r.checked) return;
      S.filters.seq = r.value;
      changed();
    });
  $('#filterReady').addEventListener('change', (e) => {
    S.filters.ready = e.target.checked;
    changed();
  });
  $('#filterBits').addEventListener('change', (e) => {
    S.filters.bits = e.target.checked;
    changed();
  });
  $('#filterSource').addEventListener('change', (e) => {
    S.filters.source = e.target.value;
    changed();
  });
  $('#filterClear').addEventListener('click', () => {
    S.filters = { ...NO_FILTERS };
    showFilters();
    changed();
  });
}

/**
 * Bring Sort and filter up to date for the list: the source file choice (only with several
 * ``sources``; ``f`` holds the one in use), and the funnel's dot and name while any filter or a
 * sort other than Score is on. Closed while the list has no moments (``shown`` false).
 */
export function syncFilterMenu(sources, f, shown, anyBit) {
  const sel = $('#filterSource');
  const names = sources.slice().sort((a, b) => naturalCompare(baseName(a), baseName(b)) || naturalCompare(a, b));
  const tag = JSON.stringify(names);
  if (sel.dataset.sources !== tag) {
    sel.dataset.sources = tag;
    sel.replaceChildren(h('option', { value: '', text: 'All files' }), ...names.map((s) => h('option', { value: s, title: s, text: baseName(s) })));
  }
  sel.value = f.source;
  $('#filterSourceWrap').hidden = sources.length < 2;
  $('#filterBits').checked = f.bits;
  $('#filterBitsWrap').hidden = !anyBit && !f.bits; // only where the project has a long dialogue bit
  const on = [filtersOn(f) && 'filters on', S.order !== 'score' && `sorted by ${ORDER_NAMES[S.order]}`].filter(Boolean);
  const btn = $('#filterBtn');
  btn.classList.toggle('on', !!on.length);
  btn.setAttribute('aria-label', ['Sort and filter moments', ...on].join(', '));
  if (!shown && !$('#filterPop').hidden) {
    $('#filterPop').hidden = true;
    btn.setAttribute('aria-expanded', 'false');
  }
}
