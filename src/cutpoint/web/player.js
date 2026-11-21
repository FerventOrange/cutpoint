// cutpoint editing workspace: the stage players, the source proxies they play and the seek slider.
import { api, startSession } from './api.js';
import { $, baseName, fmt, round2 } from './util.js';
import { askDirector, blend, cued, current, cutPlay, directed, directorBad, drawRail, drawSbHead, durOf, exportedLen, fail, follow, FRAME, holdsClip, icon, inLoaded, loaded, offsetOf, ofSource, planned, playable, playhead, pollSources, precueCut, readyFor, refreshProject, renderPanoNote, S, seqCheck, seqClock, seqGo, seqNow, seqSeg, seqStop, srcOf, tenths } from './app.js';

// Two stage players: the visible one (``v``) plays, the other preloads the next clip of "Play sequence".
export const els = [$('#video'), $('#video2')];
export let v = els[0];
export const spare = () => (v === els[0] ? els[1] : els[0]);
export const seek = $('#seek');
export let pendingSeek = null;
export let seekWant = null; // {t, fast}: the latest scrub target while a seek is in flight
let wasIn = true; // the playhead was on the rail at the last frame or view change (tick follows it off the edge)

// app.js changes these through the setters (an imported binding is read-only)
export const setV = (el) => {
  v = el;
};
export const setPendingSeek = (t) => {
  pendingSeek = t;
};
export const setSeekWant = (w) => {
  seekWant = w;
};
export const setWasIn = (on) => {
  wasIn = on;
};

function screenMsg(text) {
  const el = $('#screenMsg');
  if (el.textContent !== text) el.textContent = text; // the same text again (a poll): not re-announced
  el.hidden = !text;
}

// A player that waits for data this long says why it may be slow (#1014), until it plays.
const STALL_NOTE_MS = 5000;
let stallTimer = null;

/**
 * The likely reasons the stage player opens slowly, from the project listing: its folder on a network
 * or external drive (``network``), and a fragmented MP4 played as is (``fragmented``: the browser reads it in
 * hundreds of small requests).
 */
function stallText() {
  const src = stallSource();
  const why = [];
  if (S.project?.network) why.push('the project folder is on a network or external drive');
  if (src?.fragmented && !src.proxy) why.push('this video is a fragmented MP4, which the browser opens in many small reads');
  if (!why.length) return 'Still loading. The drive or the video may be slow to read.';
  return `Still loading, likely because ${why.join(' and ')}.`;
}

/** The source the visible player plays as listed, if it plays one as is or from its whole proxy. */
const stallSource = () => S.project?.sources?.find((s) => s.url === v.dataset.url);

/**
 * The visible player waits for data: name the likely reason if it still waits after
 * ``STALL_NOTE_MS``. A source not probed when the project was listed has no ``fragmented``
 * yet; serving it probed it, so one refresh fills the reason in.
 */
function watchStall() {
  if (stallTimer != null) return;
  stallTimer = setTimeout(() => {
    stallTimer = null;
    // a paused element never fires `waiting`: Play sequence opening a clip waits while paused
    if (v.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA || (v.paused && !S.seqPlay?.wait)) return;
    const el = $('#stallNote');
    el.textContent = stallText();
    el.hidden = false;
    const src = stallSource();
    if (!src || src.proxy || src.fragmented != null) return;
    const url = v.dataset.url;
    refreshProject().then(() => {
      if (!el.hidden && v.dataset.url === url) el.textContent = stallText(); // still the same wait
    });
  }, STALL_NOTE_MS);
}

/** Playback started (or stopped): no stall note. */
export function clearStall() {
  if (stallTimer != null) clearTimeout(stallTimer);
  stallTimer = null;
  const el = $('#stallNote');
  el.hidden = true;
  el.textContent = '';
}

// Source proxies are made on demand: only the focused source is prepared, and only it is polled.
// Keys "<project>:<path>": whole proxies asked for. Window requests dedupe only against the last
// one (``lastPart``, the key plus "@<lo>-<hi>"): the server may drop a window (superseded, stopped,
// evicted, failed), so asking for an earlier range again must reach it.
const prepareRequested = new Set();
let lastPart = null;
let nextFor = null; // the window proxy url whose successor ``requestNext`` asked for
// Playback paused itself at a window proxy's end (``partEnd``): {source, t, url}, the source path
// and time it waits at and the file it stopped in. A file covering ``t`` that loads in its place
// plays on by itself (``canplay``); a pause, a scrub, a play or any other wait drops it.
let resumeAt = null;
// The wait ``awaitPart`` holds: {source, t, lo, hi}, the source time it waits at and the range it
// asked a window for. A refresh re-running ``ensureVideo`` during it asks for that range again
// (the ``lastPart`` dedupe then drops it), never for the moment's: playback waiting at a window's
// end inside a long moment needs the window there, not one spanning the moment. ``retried``: it
// asked once more after the server listed no window for it; ``failed``: that failed too, so the
// stage shows the error and Retry.
let waitFor = null;
/** Drop a pending resume; a paused video then shows Play (a pause on a paused video fires no event). */
export function cancelResume() {
  if (!resumeAt) return;
  resumeAt = null;
  if (v.paused) showPlaying(false);
}
// seconds before a playing window proxy's end the next one is asked for: a window encodes in a few
// seconds, about twice that while the whole proxy builds, and the page polls every 2 s
const EARLY = 20;
const PART_MSG = 'Preparing this part of the source…';
const MOMENT_MSG = 'Preparing this moment\'s preview…';

/** Forget the prepare requests of ``src`` (whole and window), so they are made again. */
function forgetPrepared(src) {
  const k = `${S.id}:${src.path}`;
  prepareRequested.delete(k);
  if (lastPart?.startsWith(`${k}@`)) lastPart = null;
}

/**
 * Queue the proxy of ``src``; with ``range`` ({lo, hi}), a window proxy around it first (the server
 * queues both). ``queue``: the window waits behind the queued ones instead of superseding them
 * ("Play sequence" asks for its next clips so; its poll refreshes the project).
 */
export async function prepareSource(src, range, queue = false) {
  const whole = `${S.id}:${src.path}`;
  const k = range && !queue ? `${whole}@${range.lo}-${range.hi}` : null;
  const had = prepareRequested.has(whole);
  if (had && (!range || (k && k === lastPart))) return;
  prepareRequested.add(whole); // a window request queues the whole proxy too
  if (k) lastPart = k;
  const sid = src.id ?? decodeURIComponent(src.url.split('?')[0].split('/').pop());
  try {
    await api(`/api/projects/${encodeURIComponent(S.id)}/sources/${encodeURIComponent(sid)}/prepare`, { method: 'POST', body: queue ? { ...range, queue } : range });
    if (k) S.partAck = k; // the server has this window request: listings from now on reflect it
    if (!queue) refreshProject();
  } catch (e) {
    if (!had) prepareRequested.delete(whole);
    if (k && lastPart === k) lastPart = null;
    fail(e);
  }
}

/**
 * Ask for a window proxy around [lo, hi] of ``src`` unless the one encoding covers it; ``delay`` ms
 * debounces scrubbing. A covering window still queued (behind "Play sequence"'s) is asked for again:
 * a focus moves it to the front. The ``lastPart`` dedupe still holds: the page's last focus is already
 * first in the server's queue, and queued requests never set it.
 */
function requestPart(src, lo, hi, delay = 0) {
  clearTimeout(S.partTimer);
  const range = { lo: round2(Math.max(0, lo)), hi: round2(Math.max(0, hi)) };
  const pend = (src.pending ?? []).filter((p) => p.lo <= range.lo && p.hi >= range.hi);
  if (pend.some((p) => p.state === 'encoding')) return;
  const id = S.id;
  if (delay) S.partTimer = setTimeout(() => S.id === id && prepareSource(src, range), delay);
  else prepareSource(src, range);
}

/** Load ``p`` (from ``playable``) into the stage player, to show source time ``t``. */
function loadFile(src, p, t) {
  v.dataset.url = p.url;
  v.dataset.offset = p.offset;
  if (p.url === src.url) delete v.dataset.hi;
  else v.dataset.hi = p.hi;
  if (p.director) v.dataset.director = p.director;
  else delete v.dataset.director;
  seekWant = null;
  pendingSeek = t;
  screenMsg('Loading video…');
  clearStall(); // another file: its own wait counts afresh
  v.src = p.url;
  waitFor = null;
  if (resumeAt && src.path !== resumeAt.source) resumeAt = null; // another source: no resume
  showPlaying(!!resumeAt); // a new src stops playback without a pause event; a resume plays on
  renderPanoNote(current()); // director view or raw 360 preview (#517)
}

/** Nothing ready covers source time ``t``: hold the playhead there, paused, until a window proxy around [lo, hi] is built. */
function awaitPart(src, t, delay = 0, lo = t, hi = t) {
  // A wait elsewhere (a scrub, another moment) cancels a resume; the same wait again (a refresh
  // re-running ensureVideo) keeps it. ``partEnd`` arms it after this.
  if (resumeAt && (src.path !== resumeAt.source || Math.abs(t - resumeAt.t) > FRAME)) cancelResume();
  v.pause();
  seekWant = null;
  pendingSeek = t;
  screenMsg(PART_MSG);
  waitFor = { source: src.path, t, lo, hi };
  requestPart(src, lo, hi, delay);
  updatePlayhead();
}

export function ensureVideo(seekC) {
  clearTimeout(S.prepTimer);
  if (S.expired) return;
  // shown only when this pass ends in a failure; set once, so a focused Retry keeps its focus
  const hide = !stageVideo(seekC);
  if (hide && document.activeElement === $('#retryBtn')) $('#playBtn').focus(); // not lost to <body>
  $('#retryBtn').hidden = hide;
}

/** ``ensureVideo``'s work; true when it ends in a failure Retry can fix. */
function stageVideo(seekC) {
  const c = current();
  const src = c && srcOf(c.source);
  // drop() forgets pending seeks; a moment waiting for its video keeps its start or the user's scrub
  // Play sequence starts a clip where its export starts
  const from = seekC && (S.seqPlay && seqNow() === seekC.id ? seqSeg(S.seqPlay.i).start : seekC.start);
  const want = seekC ? from : pendingSeek;
  // over a cut, only its part of the clip has to be playable: it can lie in a tail, past the trim
  const part = (S.seqPlay?.across && seqNow() === c?.id && seqSeg(S.seqPlay.i)) || c;
  const wait = (msg) => {
    drop(msg);
    pendingSeek = want;
  };
  if (!src) {
    wait('This source file is not part of the project, so it cannot be previewed. You can still trim it on the rail.');
    return;
  }
  askDirector(c);
  // A 360 moment's rendered director preview (#517) plays in place of the raw fisheye; a raw file
  // playing on switches at the next pause (as to a whole proxy)
  const dAt = want ?? (ofSource(src) ? playhead(c) : c.start);
  const d = directed(src, dAt, c);
  if (d) {
    // the raw preview is still prepared, for the source outside the director preview's span
    if (src.status !== 'ready' && !prepareRequested.has(`${S.id}:${src.path}`)) prepareSource(src);
    const rawOn = ofSource(src) && !v.paused && !v.error && inLoaded(dAt, durOf(c));
    if (v.dataset.url !== d.url && !rawOn) loadFile(src, d, dAt);
    else if (seekC) seekTo(from);
    return;
  }
  if (src.status === 'unknown') {
    // not probed yet (the project view never probes): preparing probes it and refreshes. Still
    // unknown after an earlier prepare (the file changed, or the server restarted): prepare again.
    // A window around the moment comes with it, in case the source needs a proxy.
    const k = `${S.id}:${src.path}`;
    const range = { lo: round2(c.start), hi: round2(c.end) };
    if (!prepareRequested.has(k)) prepareSource(src, range);
    else
      S.prepTimer = setTimeout(() => {
        forgetPrepared(src);
        prepareSource(src, range);
      }, 2000);
    return wait('Loading video…');
  }
  if (src.status === 'needs-proxy' || src.status === 'preparing') {
    // The whole proxy builds; a window proxy around the moment (or the playhead) plays meanwhile.
    const k = `${S.id}:${src.path}`;
    if (src.status === 'needs-proxy' && prepareRequested.has(k))
      // requested before, but not building (the server restarted): prepare again
      S.prepTimer = setTimeout(() => {
        forgetPrepared(src);
        prepareSource(src);
      }, 2000);
    else S.prepTimer = setTimeout(pollSources, 2000);
    const mine = ofSource(src);
    const at = want ?? (mine ? playhead(c) : c.start);
    const p = playable(src, at);
    // "Play sequence" plays a clip only from a file covering all of it: a window ending inside it would cut it short
    const fits = !S.seqPlay || readyFor(src, part);
    // and keeps the loaded file only when it covers all of the clip too (``fileFor`` picks the same)
    const keep = mine && inLoaded(at, durOf(c)) && (!S.seqPlay || holdsClip(part));
    if (fits && (keep || p)) {
      if (src.status === 'needs-proxy' && !prepareRequested.has(k)) prepareSource(src); // no window request brings it
      if (!keep) loadFile(src, p, at);
      else if (seekC) seekTo(from);
      return;
    }
    const moment = !!S.seqPlay || (at >= c.start && at <= c.end);
    // the same wait again (a refresh, not a moment click): its own range, e.g. [at, at] at a
    // window's end or a scrub, also while no file is loaded yet
    const again = !seekC && !S.seqPlay && waitFor?.source === src.path && Math.abs(at - waitFor.t) <= FRAME;
    const [lo, hi] = again ? [waitFor.lo, waitFor.hi] : moment ? [part.start, part.end] : [at, at];
    const kept = again ? waitFor : null;
    const wasRetried = !!kept?.retried;
    if (kept && !kept.retried && lastPart === `${k}@${round2(Math.max(0, lo))}-${round2(Math.max(0, hi))}`
      && !(src.pending ?? []).some((w) => w.lo <= round2(Math.max(0, lo)) && w.hi >= round2(Math.max(0, hi)))) {
      // the server listed no window for the range this wait asked for (it failed or was superseded):
      // ask once more, past the lastPart dedupe; once per wait, so a failing encode is not retried forever
      kept.retried = true;
      lastPart = null;
      S.partAck = null;
    }
    const range = { lo: round2(Math.max(0, lo)), hi: round2(Math.max(0, hi)) };
    // failed only once the server took the re-ask: a listing fetched before it still shows the old failure
    const failed = wasRetried && S.partAck === `${k}@${range.lo}-${range.hi}`
      && !(src.pending ?? []).some((w) => w.lo <= range.lo && w.hi >= range.hi)
      && (src.failed ?? []).find((w) => w.lo <= range.lo && w.hi >= range.hi);
    if (failed) {
      // the window asked for once more failed too: stop waiting, say so, offer Retry (``retryBtn``)
      kept.failed = true;
      cancelResume(); // playback stopped for good: Play, not Pause
      screenMsg(`This part of the source could not be prepared for preview${failed.reason ? ` (${failed.reason})` : ''}. Retry it, or trim on the rail.`);
      return true;
    }
    if (kept) kept.failed = false;
    if (mine && !S.seqPlay) {
      awaitPart(src, at, 0, lo, hi); // keeps the last frame
      if (kept) waitFor = kept;
      return;
    }
    requestPart(src, lo, hi);
    wait(S.seqPlay ? `Preparing preview of ${baseName(c.source)}…` : moment && !again ? MOMENT_MSG : PART_MSG);
    if (kept) waitFor = kept; // drop() forgot it; the wait goes on
    return;
  }
  if (src.status === 'failed') {
    wait('The browser-playable preview could not be built for this source. Retry it, or trim on the rail.');
    return true;
  }
  forgetPrepared(src); // ready: a later change to the file prepares again
  const p = playable(src, 0);
  // A window proxy of this source playing on switches to the whole proxy at the next pause or
  // seek; during "Play sequence" only while it covers all of the clip.
  const playingOn = v.dataset.hi && ofSource(src) && !v.paused && !v.error && inLoaded(want ?? playhead(c), durOf(c))
    && (!S.seqPlay || holdsClip(part));
  if (v.dataset.url !== p.url && !playingOn) {
    loadFile(src, p, want ?? (ofSource(src) ? playhead(c) : c.start));
  } else if (seekC) seekTo(from);
}

/** Stop a video element and release its file. */
export function unload(el) {
  el.pause();
  el.removeAttribute('src');
  delete el.dataset.url;
  delete el.dataset.offset;
  delete el.dataset.hi;
  delete el.dataset.director;
  el.load();
}

export function drop(msg) {
  resumeAt = null;
  waitFor = null;
  seekWant = null;
  pendingSeek = null;
  unload(v);
  clearStall(); // load() fires no pause: no note over an empty stage
  showPlaying(false); // load() drops the queued pause event, so a playing video would keep the Pause icon
  screenMsg(msg);
}

/**
 * Seek to source time ``t`` (kept inside the source) and bring it onto the rail. One seek is in
 * flight at a time: while the video seeks, only the latest target is kept and applied on
 * ``seeked``, so dragging stays smooth. ``fast``: a keyframe seek (fastSeek) while dragging.
 */
export function seekTo(t, fast = false) {
  const c = current();
  if (!c) return;
  const dur = durOf(c);
  t = Math.max(0, dur ? Math.min(t, dur) : t);
  if (resumeAt && Math.abs(t - resumeAt.t) > FRAME) cancelResume(); // a scrub, not the resume's own seek
  if (!S.seqPlay && !stays(c, t, dur)) {
    wasIn = true;
    follow(t);
    updatePlayhead();
    return;
  }
  if (waitFor) { // the target is on playable media: a wait (e.g. at a window's end) is over
    if (waitFor.failed) {
      if (document.activeElement === $('#retryBtn')) $('#playBtn').focus(); // not lost to <body>
      $('#retryBtn').hidden = true;
      screenMsg('');
    }
    waitFor = null;
  }
  if (v.readyState < HTMLMediaElement.HAVE_METADATA) {
    pendingSeek = t; // applied on loadedmetadata
    wasIn = true;
    follow(t);
    updatePlayhead();
    return;
  }
  wasIn = true;
  pendingSeek = null; // a target held while no file covered it
  if ($('#screenMsg').textContent === PART_MSG) screenMsg('');
  if (v.seeking) seekWant = { t, fast };
  else {
    seekWant = null;
    const vt = t - offsetOf();
    if (fast && v.fastSeek) v.fastSeek(vt);
    else v.currentTime = vt;
  }
  follow(t);
  updatePlayhead();
}

/**
 * Before a seek to ``t``: when the stage file does not cover it (a window proxy, #323) or the
 * whole source became playable, load the ready file covering it, else hold the playhead there
 * and ask for a window around it. True when the loaded file seeks there itself.
 */
function stays(c, t, dur) {
  const src = srcOf(c.source);
  const d = src && directed(src, t, c);
  if (d) {
    if (v.dataset.url === d.url) return true;
    loadFile(src, d, t); // back on the moment's director preview (#517)
    return false;
  }
  if (src && v.dataset.director) { // off the director preview's span: the raw preview there
    const p = playable(src, t);
    if (p) loadFile(src, p, t);
    else awaitPart(src, t, 300);
    return false;
  }
  if (!src || !['ready', 'needs-proxy', 'preparing'].includes(src.status)) return true;
  if (!v.dataset.url && src.status === 'ready') return true; // ensureVideo is loading it
  if (v.dataset.url && inLoaded(t, dur) && (src.status !== 'ready' || !v.dataset.hi)) return true;
  const p = playable(src, t);
  if (p) loadFile(src, p, t);
  else awaitPart(src, t, 300);
  return false;
}

/**
 * Playback reached the end of the window proxy it plays: go on in a ready file covering what
 * follows, else pause there until one is built, and play on once it loads (``resumeAt``).
 */
function partEnd() {
  const c = current();
  const src = c && srcOf(c.source);
  const r = loaded();
  const dur = c && durOf(c);
  if (!src || !r || r.hi == null || (dur && r.hi >= dur - FRAME)) return false;
  const t = Math.min(playhead(c), r.hi);
  if (t < r.hi - FRAME && !v.ended) return false;
  const p = playable(src, t);
  if (p && p.url !== v.dataset.url) {
    loadFile(src, p, t);
    v.play().catch(() => {});
  } else {
    awaitPart(src, t);
    resumeAt = { source: src.path, t, url: v.dataset.url };
    showPlaying(true); // still playing, waiting: a click pauses (cancels the resume)
  }
  return true;
}

/**
 * A window proxy plays within EARLY s of its end (not the source's): ask for the window after it
 * now, as a focus on its end, so playback goes on there without a stall. Once per window file.
 */
function requestNext() {
  const r = loaded();
  const c = current();
  const src = c && srcOf(c.source);
  if (!src || !r || r.hi == null || src.status === 'ready' || nextFor === v.dataset.url) return;
  const dur = durOf(c);
  if ((dur && r.hi >= dur - FRAME) || r.hi - playhead(c) > EARLY) return;
  nextFor = v.dataset.url;
  if (!playable(src, r.hi)) requestPart(src, r.hi, r.hi);
}

/** ``playing``: called every frame during playback, so the slider's spoken value waits for a pause. */
export function updatePlayhead(playing = false) {
  const c = current();
  const t = playhead(c);
  seek.value = t;
  seek.classList.toggle('off-view', !!S.view && (t < S.view.lo || t > S.view.hi)); // no thumb parked at an edge
  if (!playing) seek.setAttribute('aria-valuetext', fmt(t));
  $('#clock').textContent = S.seqPlay ? seqClock(t) : c ? `${tenths(t - planned(c).start)} / ${tenths(exportedLen(c))}` : '';
  const off = c ? Math.round((t - c.start) * 10) / 10 : 0;
  $('#inBadge').textContent = c ? `IN ${off < 0 ? '-' : '+'}${Math.abs(off).toFixed(1)} s` : '';
  drawSbHead(c, t);
}

let tickId = 0;
const loop = () => {
  if (!tickId) tickId = requestAnimationFrame(tick); // one loop, also across player swaps
};

function tick() {
  tickId = 0;
  if (v.paused || !seqCheck() || (!S.seqPlay && partEnd())) return;
  if (!S.seqPlay) requestNext();
  // playback runs on past the clip end; the rail jumps along when the playhead runs off it, not
  // when the user panned away from it
  const t = playhead();
  const w = S.view;
  const inV = !w || (t >= w.lo && t <= w.hi);
  if (!inV && wasIn) follow(t);
  else wasIn = inV;
  updatePlayhead(true);
  loop();
}

export function togglePlay() {
  if (S.cutMode && !S.seqPlay) return cutPlay(); // no player shows: Space plays over the cut
  const c = current();
  if (!c || !v.src) return;
  if (!v.paused) return v.pause();
  if (resumeAt) { // paused at a window's end, about to play on: this is a pause
    resumeAt = null;
    return showPlaying(false);
  }
  if (pendingSeek != null && !inLoaded(pendingSeek, durOf(c))) return; // waiting for that part's window proxy
  const dur = durOf(c);
  if (!S.seqPlay && (v.ended || (dur && playhead(c) >= dur - 0.05))) seekTo(c.start); // at the end of the source: replay the clip
  v.play().catch(() => {});
}

/** Listen on both players; only the visible one (``v``) drives the stage. */
const onStage = (type, fn) => els.forEach((el) => el.addEventListener(type, (e) => el === v && fn(e)));

/** The play button is an icon; its name and tooltip say what a click does. */
export function showPlaying(playing) {
  const name = playing ? 'Pause' : 'Play';
  const b = $('#playBtn');
  b.setAttribute('aria-label', name);
  b.title = `${name} (Space)`;
  b.replaceChildren(icon(playing ? 'pause' : 'play'));
}

// One preview sounds at a time: the stage (a moment or "Play sequence") and the Export panel's final
// video pause each other.
const finalVideo = $('#finalVideo');

/** The stage players' and the seek slider's listeners; app.js registers them where they always were. */
export function bindPlayer() {
  onStage('loadedmetadata', () => {
    drawRail(); // the source length may only be known now
    const t = pendingSeek;
    pendingSeek = null;
    if (t != null) seekTo(t);
  });
  onStage('canplay', () => {
    screenMsg('');
    if (S.seqPlay?.wait) { // the clip's video the sequence waited for is ready
      S.seqPlay.wait = false;
      v.play().catch(() => {});
    } else if (resumeAt && v.dataset.url !== resumeAt.url && current()?.source === resumeAt.source && pendingSeek == null && inLoaded(resumeAt.t, durOf(current()))) {
      resumeAt = null; // the file after the window playback stopped at: play on
      v.play().catch(() => showPlaying(false));
    }
    if (S.cutMode && !S.seqPlay) precueCut(); // the stage player settled: a Fade cut may open its clip 2 now
  });
  onStage('playing', () => {
    screenMsg('');
    clearStall();
  });
  onStage('timeupdate', () => document.hidden && seqCheck()); // frames stop in a hidden tab; timeupdate still fires
  onStage('waiting', () => {
    screenMsg('Buffering…');
    watchStall();
  });
  onStage('pause', clearStall); // a pause while it waits: the user is not waiting for it
  // another file loads (a new src, or load() emptied it); a sequence clip that waits is watched
  onStage('loadstart', () => { clearStall(); if (S.seqPlay?.wait) watchStall(); });
  onStage('ended', () => {
    // a source shorter than its clip; not once the next clip of the same file was sought to (#1020)
    if (S.seqPlay) !S.seqPlay.wait && v.ended && seqGo(S.seqPlay.i + 1);
    else partEnd(); // a window proxy's end
  });
  onStage('error', () => {
    seekWant = null;
    if (!v.dataset.url) return;
    if (v.dataset.director) {
      // A director preview that fails (its path was re-planned since the listing) plays raw meanwhile;
      // one refresh per url picks up the new preview, and a url that failed is not tried again.
      const bad = v.dataset.url;
      const first = !directorBad.has(bad);
      directorBad.add(bad);
      delete v.dataset.director;
      delete v.dataset.url;
      ensureVideo(); // Retry shows if the raw preview fails too
      if (first) refreshProject();
      return;
    }
    const url = v.dataset.url;
    const msg = `The browser could not play this video${v.error?.message ? `: ${v.error.message}` : ''}.`;
    // The element hides the status; one API call tells an expired token (banner only) from a bad
    // file, so the message is shown, and announced, only when the session is still valid.
    const show = () => {
      if (!S.expired && v.dataset.url === url) screenMsg(msg);
    };
    // A window proxy goes once the whole proxy is built: the refresh switches to that instead.
    (v.dataset.hi ? refreshProject() : startSession()).then(show, show);
  });
  finalVideo.addEventListener('play', () => {
    if (S.seqPlay) seqStop();
    else {
      cancelResume(); // a resume at a window's end would start the stage again
      v.pause();
    }
  });
  onStage('play', () => {
    if (blend && blend.old.currentTime < blend.end) blend.old.play().catch(() => {});
    finalVideo.pause();
    resumeAt = null;
    showPlaying(true);
    follow(playhead()); // Play shows the playhead, also after a pan away from it
    loop();
  });
  onStage('pause', () => {
    blend?.old.pause();
    showPlaying(!!resumeAt); // waiting at a window's end to play on
    updatePlayhead();
    const c = current();
    const src = c && srcOf(c.source);
    const d = src && directed(src, playhead(c), c);
    if (S.seqPlay) return;
    if ((v.dataset.hi && src?.status === 'ready') || (d && d.url !== v.dataset.url)) ensureVideo(null); // the whole proxy or a director preview is ready: switch to it now
  });
  onStage('seeked', () => {
    const w = seekWant;
    if (w) seekTo(w.t, w.fast); // the latest scrub target that waited for this seek
    else updatePlayhead();
    if (S.cutMode && !S.seqPlay) precueCut(); // a seek that kept readyState high ends with only seeked
  });
  onStage('click', togglePlay);
  // the spare player seeks to the clip it preloads as soon as it knows the source
  els.forEach((el) => el.addEventListener('loadedmetadata', () => {
    if (el !== v && cued) el.currentTime = cued.vt;
  }));
  $('#playBtn').addEventListener('click', togglePlay);
  $('#retryBtn').addEventListener('click', () => {
    const c = current();
    const src = c && srcOf(c.source);
    if (!src) return;
    if (waitFor?.failed && waitFor.source === src.path) { // a window proxy failed twice: ask again
      const w = waitFor;
      w.failed = false;
      w.retried = false;
      lastPart = null;
      S.partAck = null;
      if (document.activeElement === $('#retryBtn')) $('#playBtn').focus(); // not lost to <body>
      $('#retryBtn').hidden = true;
      screenMsg(PART_MSG);
      requestPart(src, w.lo, w.hi);
      return;
    }
    prepareRequested.delete(`${S.id}:${src.path}`);
    prepareSource(src);
  });
  seek.addEventListener('input', () => seekTo(Number(seek.value), true));
  seek.addEventListener('change', () => seekTo(Number(seek.value))); // released: an exact seek
  seek.addEventListener('keydown', (e) => {
    const d = { ArrowLeft: -0.005, ArrowDown: -0.005, ArrowRight: 0.005, ArrowUp: 0.005, PageDown: -0.1, PageUp: 0.1 }[e.key];
    if (d == null || !S.view) return;
    // arrows: 0.5% of the rail (at least 0.05 s), so a zoomed-out long source moves too; PageUp/PageDown: 10%
    e.preventDefault();
    const span = S.view.hi - S.view.lo;
    seekTo(playhead() + Math.sign(d) * Math.max(0.05, Math.abs(d) * span));
  });
}
