// cutpoint editing workspace: AI Chat, its proposals and ranking cards (#346, #902).
import { api } from './api.js';
import { $, baseName, filtersOn, fmt, fmtLen, h, kindOf, markdown, momentMatches, namedFolders, NO_FILTERS, svg } from './util.js';
import { addSeg, analyzeAgain, banner, BUSY_TEXT, cands, chooseColor, COLOR_NAMES, colorChoice, confirmAction, drawGhost, filterLabel, focusMoment, icon, keepDraft, key, preserveFocus, refreshProject, remember, renderAfterSeqChange, renderMoments, revealFolder, revealRow, S, sentence, seqHome, shownName, showTab } from './app.js';

// ---------- AI Chat (#346) ----------
// A turn is posted with the working sequence and polled like a job. The assistant only proposes: a
// proposal is previewed (S.proposal: the Sequence tab, the moments list and ghost trim handles on
// the rail) until Apply replaces the working sequence, which Save and Export use as before. Text
// from the model or the project is only ever set as text (h()'s text, textContent), never as markup.

const CHAT_POLL_MS = 1000;
const MAX_SUGGESTIONS = 3; // suggestion buttons under a proposal
const MAX_HISTORY_SHOWN = 20; // newest applied changes listed for Revert
const PROVIDER_NAMES = { ollama: 'Ollama', openai: 'OpenAI-compatible', anthropic: 'Anthropic' };
// What a hosted provider gets: analysis results only (context.py sends only these; file names,
// never paths); video frames only once allowed for a project (view_moment, #640), never audio.
const CONSENT_SENDS = "this project's analysis results (moment labels, scores and detection signals, vision descriptions, transcript snippets, file names (not full paths) and times) and your messages; never video frames unless you allow them for a project, and never audio";
// What analyze's dialogue reader (cutpoint.events.dialogue) sends a hosted chat model: whole transcripts
const ANALYSIS_SENDS = 'the whole transcripts of the footage (everything said in it), in pieces of about 8 minutes, to find funny, heated or story-telling talk';
// What view_moment sends a hosted model once allowed (watch.py): one image per watched moment
const FRAMES_SENDS = (host) => `To answer questions about what can be seen, the assistant can watch a few moments. Each one sends ${host} one image of 6 to 9 frames from the moment's clip, with its times, transcript and detected sounds. Nothing was sent. You allow it once for this project and host; Cutpoint saves it in config.json.`;
const CHANGE_NAMES = { added: 'added', removed: 'removed', moved: 'moved', lengthened: 'lengthened', shortened: 'shortened', trimmed: 'trimmed', relabelled: 'renamed' };
const SMALL_MODEL_NOTE = 'A small local model can miss moments or misread a request. Check each change in the preview before you apply it.';
/** The one line under the chat header when its visual search cannot run: why, and what adds it (GET assistant's ``frame_index``). */
const indexNote = (idx) => `${sentence(idx.why || 'Visual search isn’t available for this project')} To add it: ${idx.fix || 'analyze the project again'}.`;
const aiUrl = (path = '') => `/api/projects/${encodeURIComponent(S.id)}/assistant${path}`;
// a moment's name as the Moments list shows it (shownName; #634)
const momentName = (id) => {
  const c = cands().find((m) => m.id === id);
  return c?.display_label || (c ?? S.segments.find((s) => s.id === id))?.label || id;
};
const REVERT = 'revert-';
const isRevert = (p) => p.id.startsWith(REVERT);
/** The sequence a proposal was made on, from its diff's "before" rows, as [id, start, end] in order. */
const baseOf = (diff) => JSON.stringify(diff.filter((r) => r.before).sort((a, b) => a.before.pos - b.before.pos)
  .map((r) => [r.id, r.before.start, r.before.end]));
/** The working sequence in the same shape. */
const seqShape = () => JSON.stringify(S.segments.map((s) => [s.id, s.start, s.end]));
/** A saved turn (chat.json) or a polled one, in one shape. */
const chatTurn = (t) => ({ id: t.id, user: t.user ?? t.text ?? '', status: t.status, error: t.error, events: [...(t.events ?? [])], ranking_key: t.ranking_key ?? null });
const arrow = () => [h('span', { 'aria-hidden': 'true', text: ' → ' }), h('span', { class: 'vh', text: ' to ' })];

/** "Auto, strength 80%" for a Color ({mode, strength}); Off has no strength, a strength the op left out is not named. */
function colorText(c) {
  const pct = c.mode !== 'off' && Number.isFinite(c.strength) ? `, strength ${Math.round(c.strength * 100)}%` : '';
  return `${COLOR_NAMES[c.mode] ?? c.mode}${pct}`;
}

/** "2 added, 1 trimmed, color auto" for a proposal's diff and its color op. */
function changeCounts(diff, color = null) {
  return changeParts(diff, color).join(', ') || 'no change';
}

/** The parts of changeCounts, one per change: ["2 added", "1 trimmed", "Color auto"]. */
function changeParts(diff, color = null) {
  const n = {};
  for (const r of diff ?? []) for (const c of r.changes) n[c] = (n[c] ?? 0) + 1;
  const parts = Object.keys(CHANGE_NAMES).filter((c) => n[c]).map((c) => `${n[c]} ${CHANGE_NAMES[c]}`);
  if (color) parts.push(`Color ${colorText(color).toLowerCase()}`);
  return parts;
}

/** The open project's chat, the turn still running (polled on) and the applied changes. */
export async function loadAssistant() {
  const pid = S.id;
  clearTimeout(S.chatTimer);
  S.turn = null;
  const chat = S.chat = pid ? { pid, provider: null, consent: null, frames: null, frameIndex: null, turns: [], files: [], error: '', loaded: false } : null;
  S.history = [];
  S.dismissed.clear();
  S.rankings.clear();
  chatError('');
  renderChat();
  if (!chat) return;
  try {
    const [a, hist] = await Promise.all([api(aiUrl()), api(aiUrl('/history'))]);
    if (S.chat !== chat) return; // another project opened meanwhile
    chat.provider = a.provider;
    chat.consent = a.consent ?? null;
    chat.frames = a.frames ?? null;
    chat.frameIndex = a.frame_index ?? null;
    chat.files = a.files ?? [];
    chat.turns = (a.turns ?? []).map(chatTurn);
    S.history = hist.applied ?? [];
    const run = a.running?.[0];
    if (run) {
      chat.turns.push(chatTurn(run));
      S.turn = { id: run.id, next: run.next };
      S.chatTimer = setTimeout(pollTurn, CHAT_POLL_MS);
    }
  } catch (e) {
    if (S.chat !== chat) return;
    chat.error = e.message;
  }
  chat.loaded = true; // Send waits for this: the saved turns would replace one sent meanwhile
  renderChat();
}

/** Ask again whether the chat's visual search can run (an analysis may have written the frame index since the chat loaded) and update its one-line note. */
export async function refreshFrameIndex() {
  const chat = S.chat;
  if (!chat?.loaded) return;
  try {
    const a = await api(aiUrl());
    if (S.chat !== chat) return; // another project opened meanwhile
    chat.frameIndex = a.frame_index ?? null;
    renderChat();
  } catch { /* the note stays as it was */ }
}

function chatError(text) {
  $('#chatError').textContent = text;
}

async function sendChat(e) {
  e.preventDefault();
  const input = $('#chatInput');
  const text = input.value.trim();
  const chat = S.chat;
  if (S.turn) chatError('The assistant is still answering: wait for it, or cancel it.');
  if (!text || !chat?.loaded || S.turn || S.sending) return;
  S.sending = true;
  chatError('');
  renderChatForm();
  try {
    const t = await api(aiUrl('/turns'), { method: 'POST', body: { text, segments: JSON.parse(key()) } });
    if (S.chat !== chat) return;
    input.value = '';
    // its events, even those already in this answer, come in through pollTurn, which acts on them
    chat.turns.push(chatTurn({ ...t, status: 'running', events: [] }));
    if (t.provider) chat.provider = t.provider;
    if (t.files) chat.files = t.files;
    S.turn = { id: t.id, next: 0 };
    $('#chatLive').textContent = 'Sent. The assistant is answering.';
    S.chatTimer = setTimeout(pollTurn, t.status === 'running' ? CHAT_POLL_MS : 0);
  } catch (err) {
    if (S.chat === chat) chatError(err.message);
  } finally {
    S.sending = false;
    renderChat();
    if (document.activeElement === document.body) input.focus(); // the Send button was disabled under it
  }
}

async function pollTurn() {
  clearTimeout(S.chatTimer);
  const run = S.turn;
  const chat = S.chat;
  if (!run || !chat || S.expired) return;
  const t = chat.turns.find((x) => x.id === run.id);
  try {
    const v = await api(aiUrl(`/turns/${encodeURIComponent(run.id)}?after=${run.next}`));
    if (S.turn !== run) return; // cancelled, cleared or another project meanwhile
    const news = v.events.length > 0 || v.status !== t.status;
    t.events.push(...v.events);
    t.status = v.status;
    t.error = v.error;
    t.ranking_key = v.ranking_key ?? t.ranking_key;
    run.next = v.next;
    if (v.provider) chat.provider = v.provider;
    if (v.files) chat.files = v.files; // the files this turn's model was told of
    for (const ev of v.events) onChatEvent(ev);
    if (v.status !== 'running') {
      S.turn = null;
      chatError(''); // a "still answering" or a failed poll no longer applies
      $('#chatLive').textContent = turnNews(t);
    }
    if (news) renderChat();
  } catch (e) {
    if (S.turn !== run) return;
    if (e.status === 404) { // the server restarted without it
      S.turn = null;
      t.status = 'failed';
      t.error = 'The server no longer has this reply. Ask again.';
      $('#chatLive').textContent = t.error;
      renderChat();
    } else chatError(e.message); // polling goes on
  }
  if (S.turn === run) S.chatTimer = setTimeout(pollTurn, CHAT_POLL_MS);
}

/** A new event of the polled turn: shown moments are highlighted, a proposal is previewed at once. */
function onChatEvent(ev) {
  if (ev.kind === 'show') {
    S.shown = new Set(ev.ids);
    markMoments();
  } else if (ev.kind === 'proposal') previewProposal(ev);
}

/** What the live region says when a turn ends. */
function turnNews(t) {
  const last = [...t.events].reverse().find((e) => ['reply', 'proposal', 'ranking', 'error'].includes(e.kind));
  if (t.status === 'cancelled') return 'Reply cancelled.';
  if (last?.kind === 'proposal') return `Proposed: ${last.summary}. It is previewed in the Sequence tab.`;
  if (last?.kind === 'ranking') return `Previewed a ${styleTitle(last.style)} ranking. Use it or dismiss it from its card in the chat.`;
  if (last) return last.text;
  return t.error || 'The assistant gave no answer.';
}

async function cancelTurn() {
  const run = S.turn;
  if (!run) return;
  try {
    const v = await api(aiUrl(`/turns/${encodeURIComponent(run.id)}/cancel`), { method: 'POST' });
    if (S.turn !== run) return;
    S.turn = null;
    clearTimeout(S.chatTimer);
    const t = S.chat.turns.find((x) => x.id === run.id);
    Object.assign(t, chatTurn(v));
    $('#chatLive').textContent = turnNews(t);
    renderChat();
    $('#chatInput').focus(); // the Cancel button is gone
  } catch (e) {
    chatError(e.message);
  }
}

async function clearChat() {
  const chat = S.chat;
  if (!chat) return;
  if (!(await confirmAction('Clear the chat?', 'The conversation is deleted. Changes you applied stay in the sequence, and Applied changes keeps them for reverting.', 'Clear chat'))) return;
  try {
    await api(aiUrl('/chat'), { method: 'DELETE' });
    if (S.chat !== chat) return;
    clearTimeout(S.chatTimer);
    S.turn = null;
    chat.turns = [];
    S.rankings.clear();
    S.shown = null;
    if (S.proposal && !isRevert(S.proposal)) S.proposal = null; // its turn is gone: it can no longer be applied; no edit is undone
    if (S.pendingProposal && !isRevert(S.pendingProposal)) S.pendingProposal = null;
    chatError('');
    renderProposal();
    renderChat();
    $('#chatLive').textContent = 'Chat cleared. Applied changes stay.';
    $('#chatInput').focus();
  } catch (e) {
    chatError(e.message);
  }
}

/** Preview proposal ``p`` ({id, summary, after, diff}), without changing the working sequence. */
function previewProposal(p) {
  if (S.applying) { // the page is waiting for Apply's answer: this one comes next
    S.pendingProposal = p;
    return;
  }
  S.dismissed.delete(p.id);
  S.proposal = { id: p.id, summary: p.summary, after: p.after, diff: p.diff, color: p.color ?? null, base_key: p.base_key, seqKey: baseOf(p.diff) };
  $('#propMsg').textContent = '';
  renderProposal();
}

/** A button's preview: in the Sequence tab, which gets the focus. */
function reviewProposal(p) {
  previewProposal(p);
  renderChat();
  showTab('sequence');
  $('#propTitle').focus();
}

/** The Sequence tab's preview of S.proposal, the marks in the moments list and the rail's ghost handles. */
export function renderProposal() {
  const p = S.proposal;
  $('#panelSequence').classList.toggle('previewing', !!p);
  $('#propPanel').hidden = !p;
  if (p) {
    $('#propSummary').textContent = p.summary;
    const note = $('#propNote');
    note.hidden = seqShape() === p.seqKey;
    note.textContent = isRevert(p)
      ? 'The sequence changed since this preview. Apply previews the revert again on the sequence as it is now.'
      : 'The sequence changed since this was proposed. Apply makes the same edits to the sequence as it is now, if they still fit.';
    preserveFocus(() => $('#propList').replaceChildren(...p.diff.map(propRow)));
    if (p.color) $('#propColor').textContent = `Color: ${colorText(p.color)} (now ${colorText(colorChoice())})`;
  }
  $('#propColor').hidden = !p?.color;
  renderPropBar();
  markMoments();
  drawGhost();
}

function renderPropBar() {
  $('#propApply').disabled = !S.proposal || S.applying;
  $('#propApply').textContent = S.applying ? 'Applying…' : 'Apply';
  $('#propDiscard').disabled = !S.proposal || S.applying;
  for (const b of $('#chatLog').querySelectorAll('button[data-act="apply"]')) {
    b.disabled = !!S.applying;
    b.textContent = S.applying && S.proposal?.id === b.closest('[data-pid]')?.dataset.pid ? 'Applying…' : 'Apply';
  }
  for (const b of $('#chatLog').querySelectorAll('button[data-act="discard"]')) b.disabled = !!S.applying;
}

/** Say why Apply failed where the user is: the Sequence tab's line, and the chat's alert when the chat tab is the one in view. */
function applyMsg(text) {
  $('#propMsg').textContent = text;
  if (text && !$('#panelAssistant').hidden) chatError(text);
}

/** After a card's Apply or Discard the buttons are gone: focus goes to the card's Review button, else the box, in the tab in view. */
function focusAfterChange(pid) {
  if (!$('#panelSequence').hidden) return seqHome().focus();
  ($(`#chatLog [data-pid="${CSS.escape(pid)}"] [data-act="review"]`) ?? $('#chatInput')).focus();
}

/** One row of the previewed sequence, with its change in words: Added, Removed, Moved 3 → 1, Trimmed 0:10.00–0:18.00 → .... */
function propRow(r) {
  const name = shownName(cands().find((c) => c.id === r.id), r.label) || momentName(r.id);
  const at = r.after ?? r.before;
  const span = (s) => `${fmt(s.start)}–${fmt(s.end)}`;
  const marks = r.changes.map((c) => h('span', { class: `prop-mark ${c}` },
    c === 'moved' ? ['Moved ', String(r.before.pos), ...arrow(), String(r.after.pos)]
      : c === 'trimmed' || c === 'lengthened' || c === 'shortened' ? [c === 'trimmed' ? 'Trimmed ' : c === 'lengthened' ? 'Lengthened ' : 'Shortened ', span(r.before), ...arrow(), span(r.after)]
        : c === 'relabelled' ? ['Renamed “', r.before.label, '”', ...arrow(), '“', r.after.label, '”']
          : c === 'added' ? 'Added' : 'Removed'));
  if (r.auto === 'spread') marks.push(h('span', { class: 'prop-mark auto', title: 'Moved or added by Cutpoint to spread the clip kinds you asked to mix' }, 'Auto-spread'));
  else if (r.auto) marks.push(h('span', { class: 'prop-mark auto', title: 'Added by Cutpoint to reach the length you asked for' }, 'Auto-filled'));
  return h('li', { class: ['prop', ...r.changes].join(' '), 'data-id': r.id },
    h('button', { type: 'button', class: 'prop-main', 'data-act': 'focus', onclick: () => focusMoment(r.id) },
      h('span', { class: 'rank tc', text: r.after ? String(r.after.pos) : '–' }),
      h('span', { class: 'prop-text' },
        h('span', { class: 'prop-title', text: name }),
        h('span', { class: 'prop-time tc', text: `${fmt(at.start)} to ${fmt(at.end)}, ${(at.end - at.start).toFixed(1)} s` }),
        marks.length ? h('span', { class: 'prop-marks' }, marks) : null)));
}

/** Mark the moments the assistant showed or its proposal changes, by class only: the list itself is renderMoments'. */
export function markMoments() {
  const rows = new Map((S.proposal?.diff ?? []).map((r) => [r.id, r.changes]));
  for (const li of $('#candList').children) {
    const id = li.dataset.id;
    const ch = rows.get(id) ?? [];
    li.classList.toggle('ai-shown', !!id && !!S.shown?.has(id));
    li.classList.toggle('ai-added', ch.includes('added'));
    li.classList.toggle('ai-removed', ch.includes('removed'));
    li.classList.toggle('ai-changed', ch.some((c) => ['moved', 'trimmed', 'lengthened', 'shortened', 'relabelled'].includes(c)));
  }
}

/** Swap the working sequence for ``after``, keeping drafts of the clips it drops, as removeSeg does. */
function replaceSequence(after) {
  const ids = new Set(after.map((s) => s.id));
  S.segments.forEach((s, i) => !ids.has(s.id) && keepDraft(s, i));
  for (const s of after) delete S.drafts[s.id];
  S.segments = structuredClone(after);
}

async function applyProposal() {
  const p = S.proposal;
  if (!p || S.applying) return;
  const pid = S.id;
  S.applying = true;
  $('#propMsg').textContent = '';
  chatError('');
  renderPropBar();
  try {
    // The server checks it against the sequence as it is now; a changed one gets the same edits again.
    const entry = await api(aiUrl(`/proposals/${encodeURIComponent(p.id)}/applied`), { method: 'POST', body: { segments: JSON.parse(key()), base_key: p.base_key, color: colorChoice() } });
    if (S.id !== pid || S.proposal !== p) return;
    remember(null, !!entry.color);
    replaceSequence(entry.after);
    if (entry.color) chooseColor(entry.color.after);
    S.proposal = null;
    S.history.push(entry);
    renderAfterSeqChange();
    renderChat();
    $('#sideLive').textContent = `${entry.rebased ? 'Applied to the sequence as it is now' : 'Applied'}: ${p.summary}. Ctrl+Z undoes it.`;
    focusAfterChange(p.id); // the preview and its buttons are gone
  } catch (e) {
    if (S.proposal !== p) return;
    if (isRevert(p) && e.status === 409 && e.message.includes('preview it again')) {
      S.applying = false; // revertChange previews the new one at once
      const err = await revertChange({ id: p.id.slice(REVERT.length) });
      if (!err) S.pendingProposal = null; // the revert just previewed again stays in view
      if (err && S.proposal === p) applyMsg(`Can't apply: ${err}. Discard it.`);
      else if (!err) $('#sideLive').textContent = 'The sequence changed since that preview: the revert is previewed again on it. Check it, then Apply.';
      return;
    }
    applyMsg(e.status === 409 ? `Can't apply: ${e.message}. Discard it, or ask the assistant again.`
      : e.status === 404 ? 'The assistant no longer has this proposal (the chat was cleared or the server restarted). Discard it and ask again.'
        : e.message);
  } finally {
    S.applying = false;
    const next = S.pendingProposal;
    S.pendingProposal = null;
    if (next && S.id === pid) previewProposal(next);
    else renderPropBar();
    renderChat(); // the cards show which proposal is previewed and whether Apply is busy
  }
}

function discardProposal() {
  if (!S.proposal || S.applying) return;
  const id = S.proposal.id;
  S.proposal = null;
  chatError('');
  renderProposal();
  renderChat();
  $('#sideLive').textContent = 'Proposed change discarded';
  focusAfterChange(id);
}

/** A card's Apply: preview the proposal if it is not the previewed one, then the usual apply (its checks and messages). */
async function applyFromCard(ev) {
  if (S.applying) return;
  chatError('');
  if (seqShape() !== baseOf(ev.diff)) return reviewProposal(ev); // the sequence moved on since this was proposed: look at it again before applying
  if (S.proposal?.id !== ev.id) previewProposal(ev);
  await applyProposal();
  if (document.activeElement === document.body) $(`#chatLog [data-pid="${CSS.escape(ev.id)}"] [data-act="apply"]`)?.focus(); // a refused Apply: its button is back
}

/** A card's Discard: the previewed proposal is discarded as in the Sequence tab; any other is only marked dismissed on its card. */
function discardFromCard(ev) {
  if (S.applying) return;
  chatError('');
  S.dismissed.add(ev.id);
  if (S.proposal?.id === ev.id) return discardProposal();
  renderChat();
  $('#chatLive').textContent = 'Proposed change discarded';
  focusAfterChange(ev.id);
}

/** History > Revert this change: a proposal back to the change's "before", previewed like any other.
 * Returns the error's message, or '' once previewed. */
async function revertChange(entry) {
  const chat = S.chat;
  chatError('');
  try {
    const p = await api(aiUrl(`/history/${encodeURIComponent(entry.id)}/revert`), { method: 'POST', body: { segments: JSON.parse(key()) } });
    if (S.chat === chat) reviewProposal(p);
    return '';
  } catch (e) {
    if (S.chat === chat) chatError(e.message);
    return e.message;
  }
}

const TOOL_TEXT = {
  find_moments: (ev) => `Searched for “${ev.query}”: ${ev.found} found`,
  find_words: (ev) => `Looked for the words “${ev.query}”: said ${times(Number(ev.found) || 0)}`,
  get_moment: (ev) => `Looked at ${momentName(ev.id)}`,
  get_transcript: (ev) => `Read the transcript around ${momentName(ev.id)}`,
  look_for: (ev) => ev.error ? (ev.reason || `Could not look through the footage for “${ev.query}”`)
    : `Looked through the footage for “${ev.query}”: ${ev.found} found${ev.new ? `, ${ev.new} new` : ''}${ev.weak ? `, ${ev.weak} weak` : ''}${ev.unconfirmed ? `, ${ev.unconfirmed} unconfirmed` : ''}`,
  view_moment: (ev) => ev.error === 'limit' ? `Did not watch ${momentName(ev.id)}: the limit of moments watched per message was reached`
    : ev.error ? `Could not watch ${momentName(ev.id)}: ${ev.error}`
      : `Watched ${momentName(ev.id)} (${ev.frames} frames)`,
  preview_ranking: (ev) => ev.error ? `Could not preview a ranking${ev.style ? ` in the ${styleTitle(ev.style)} style` : ''}: ${ev.error}`
    : `Previewed a ranking${ev.style ? ` in the ${styleTitle(ev.style)} style` : ''}`,
};

/** 13 px stroke icons of the tool summary (decorative: the line says it in words). */
const STEP_ICONS = {
  search: [['circle', { cx: 11, cy: 11, r: 7 }], ['path', { d: 'm20 20-3.5-3.5' }]],
  scan: [['path', { d: 'M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3' }], ['circle', { cx: 12, cy: 12, r: 3 }]],
  eye: [['path', { d: 'M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z' }], ['circle', { cx: 12, cy: 12, r: 3 }]],
  clip: [['rect', { x: 3, y: 5, width: 18, height: 14, rx: 2 }], ['path', { d: 'm10 9.5 4 2.5-4 2.5z' }]],
  text: [['path', { d: 'M6 3h9l4 4v14H6zM9 12h7M9 16h7' }]],
  retry: [['path', { d: 'M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7' }]],
  alert: [['path', { d: 'M12 3 2 20h20zM12 10v4M12 17.5v.01' }]],
};
/** One tool summary line from a tally ({n, found}): "Searched 3 times · 30 moments found". */
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const times = (n) => (n === 1 ? 'once' : `${n} times`);
const STEP_SUMMARY = {
  find_moments: { icon: 'search', text: (l) => `Searched ${times(l.n)} · ${plural(l.found, 'moment')} found` },
  look_for: { icon: 'scan', text: (l) => `Looked through the footage ${times(l.n)} · ${l.found} found` },
  find_words: { icon: 'text', text: (l) => `Looked for words ${times(l.n)} · ${plural(l.found, 'place')} found` },
  view_moment: { icon: 'eye', text: (l) => `Watched ${plural(l.n, 'moment')}` },
  get_moment: { icon: 'clip', text: (l) => `Looked at ${plural(l.n, 'moment')}` },
  get_transcript: { icon: 'text', text: (l) => `Read ${plural(l.n, 'transcript')}` },
  preview_ranking: { icon: 'clip', text: (l) => `Previewed ${plural(l.n, 'ranking')}` },
  rejected: { icon: 'retry', text: (l) => `${plural(l.n, 'attempt')} retried` },
};
const openSteps = new Set(); // ids of the tool details the user opened: they stay open when the chat is drawn again

/** The tool steps of one stretch of a turn as a few lines (a failed one stays on its own line, by name) and, folded away, every step in words. */
function stepsItem(steps, id) {
  const lines = [];
  const tally = new Map();
  for (const ev of steps) {
    if (ev.kind === 'tool' && ev.error) {
      lines.push({ icon: 'alert', text: stepText(ev), bad: true });
      continue;
    }
    const key = ev.kind === 'rejected' ? 'rejected' : ev.name;
    let l = tally.get(key);
    if (!l) lines.push(l = { key, n: 0, found: 0 }), tally.set(key, l);
    l.n++;
    l.found += Number(ev.found) || 0;
  }
  for (const l of lines) {
    if (!l.key) continue;
    const def = STEP_SUMMARY[l.key];
    l.icon = def?.icon ?? 'clip';
    l.text = def ? def.text(l) : `Used ${l.key}${l.n > 1 ? ` ${times(l.n)}` : ''}`;
  }
  return h('li', { class: `msg step tools${steps.some((e) => e.kind === 'rejected') ? ' rejected' : ''}`, 'data-id': id },
    h('ul', { class: 'tool-lines' }, lines.map((l) => h('li', { class: l.bad ? 'bad' : null }, svg('chat-ic', STEP_ICONS[l.icon]), h('span', { text: l.text })))),
    h('details', { class: 'tool-details', open: openSteps.has(id), ontoggle: (e) => (e.target.open ? openSteps.add(id) : openSteps.delete(id)) },
      h('summary', { 'data-act': 'steps', text: 'Details' }),
      h('ul', {}, steps.map((ev) => h('li', { text: stepText(ev) })))));
}

const stepText = (ev) => (ev.kind === 'rejected' ? `Not accepted, asked again: ${ev.text}` : (TOOL_TEXT[ev.name] ?? (() => ev.name))(ev));

/** Chat bubbles of turn ``t``: the user's message, then the model's steps, its answer and its proposals. */
function turnItems(t) {
  const items = [h('li', { class: 'msg user', 'data-id': `${t.id}-u` }, h('span', { class: 'vh', text: 'You: ' }), h('p', { text: t.user }))];
  let query = '';
  let steps = []; // consecutive tool steps (and rejected attempts), shown as one summary
  const flush = () => {
    if (steps.length) items.push(stepsItem(steps, `${t.id}-s${steps[0].n}`));
    steps = [];
  };
  for (const ev of t.events) {
    const id = `${t.id}-${ev.n}`;
    if (ev.kind === 'tool' || ev.kind === 'rejected') {
      if (ev.name === 'find_moments') query = ev.query;
      steps.push(ev);
      if (ev.name === 'look_for' && ev.possible?.length) {
        flush();
        items.push(possibleItem(ev, id));
      }
      continue;
    }
    flush();
    if (ev.kind === 'show') items.push(showItem(ev, id, query));
    else if (ev.kind === 'reply') items.push(h('li', { class: 'msg ai plain', 'data-id': id }, h('span', { class: 'vh', text: 'Assistant: ' }), markdown(ev.text),
      idsMoved(t) ? h('p', { class: 'msg-meta', 'data-act': 'ids-moved', text: '(moment ids changed since)' }) : null, folderActs(ev.text)));
    else if (ev.kind === 'proposal') items.push(proposalItem(ev, id));
    else if (ev.kind === 'ranking') items.push(rankingItem(ev, id));
    else if (ev.kind === 'frames_consent') items.push(framesItem(ev, id, t));
    else if (ev.kind === 'error') items.push(h('li', { class: 'msg ai bad', 'data-id': id, text: ev.text }));
  }
  flush();
  if (t.status === 'running')
    items.push(h('li', { class: 'msg ai plain pending', 'data-id': `${t.id}-run` },
      h('span', { class: 'spin', 'aria-hidden': 'true' }), h('span', { text: 'Thinking…' }),
      h('button', { type: 'button', class: 'btn small', 'data-act': 'cancel', onclick: cancelTurn, text: 'Cancel' })));
  else if (t.status === 'cancelled') items.push(h('li', { class: 'msg step', 'data-id': `${t.id}-x`, text: 'Cancelled.' }));
  else if (t.error && !t.events.some((e) => e.kind === 'error')) items.push(h('li', { class: 'msg ai bad', 'data-id': `${t.id}-x`, text: t.error }));
  return items;
}

const FOLDER_ACTS = { project: 'Open project folder', clips: 'Open clips folder' };

/** Open folder buttons for a reply that names a project file (#1010): the page knows where the files are, the model only their names. None where folders cannot be opened (as Show folder). */
function folderActs(text) {
  const targets = S.canReveal ? namedFolders(text, S.chat?.files) : [];
  if (!targets.length) return null;
  return h('span', { class: 'msg-acts' }, targets.map((target) => h('button', {
    type: 'button', class: 'btn small', 'data-act': 'open-folder', 'data-target': target,
    onclick: (e) => revealFolder(e.currentTarget, target),
  }, icon('folder'), FOLDER_ACTS[target])));
}

/** Turn ``t`` was answered on another ranking than the project's now (Q7, #902): the short moment ids its replies name may point elsewhere. Nothing without both keys. */
const idsMoved = (t) => !!t.ranking_key && !!S.project?.ranking_key && t.ranking_key !== S.project.ranking_key;

/** show_moments: the moments, highlighted in the list (a toggle) or, by the words the model searched for, filtered through the list's own filter. */
function showItem(ev, id, query) {
  // Filter only when the list's filter finds one of these moments by the model's words
  const found = query && ev.ids.some((x) => {
    const c = cands().find((m) => m.id === x);
    return c && momentMatches({ ...c, label: filterLabel(c) }, query, NO_FILTERS);
  });
  const on = !!S.shown && S.shown.size === ev.ids.length && ev.ids.every((x) => S.shown.has(x));
  const n = ev.ids.length;
  return h('li', { class: 'msg ai plain', 'data-id': id },
    h('span', { class: 'vh', text: 'Assistant: ' }),
    h('p', { text: ev.note || `${n} moment${n > 1 ? 's' : ''} found` }),
    h('p', { class: 'msg-meta', text: ev.ids.map(momentName).join(', ') }),
    h('span', { class: 'msg-acts' },
      h('button', {
        type: 'button', class: 'btn small', 'data-act': 'show', 'aria-pressed': String(on),
        onclick: () => {
          S.shown = on ? null : new Set(ev.ids);
          markMoments();
          renderChat();
        },
      }, 'Highlight in list'),
      found ? h('button', {
        type: 'button', class: 'btn small', 'data-act': 'filter',
        onclick: () => {
          S.filter = query;
          $('#filter').value = query;
          renderMoments();
          const n = $('#candList').querySelectorAll('.cand').length;
          $('#chatLive').textContent = `Moments filtered by ${query}: ${n} shown${filtersOn(S.filters) || S.kind ? '; Sort and filter or the kind filter is also on' : ''}`;
        },
      }, `Filter list by “${query}”`) : null));
}

/** look_for's guesses: windows the assistant cannot add by itself. Each is a button that adds it to the sequence, which is the user's confirmation (the same add as the Moments list). */
function possibleItem(ev, id) {
  return h('li', { class: 'msg ai possible', 'data-id': `${id}-possible` },
    h('span', { class: 'vh', text: 'Assistant: ' }),
    h('p', { text: `Possible matches for “${ev.query}”, not confirmed. Check each in the video before you add it.` }),
    h('ul', { class: 'possible-list', 'data-ev': id }, ev.possible.map((w) => {
      const inSeq = S.segments.some((x) => x.id === w.id);
      return h('li', { 'data-id': w.id },
        h('span', { class: 'tc', text: `${baseName(w.source)} ${fmt(w.start)}–${fmt(w.end)} (${w.confidence}) ` }),
        h('button', {
          type: 'button', class: 'btn small', 'data-act': 'add-possible', disabled: inSeq,
          onclick: () => addPossible(w),
        }, inSeq ? 'In the sequence' : 'Add to sequence'));
    })));
}

/** The user adds one of look_for's possible windows: the existing add path, with the window kept as a draft like a removed clip that is no moment. */
function addPossible(w) {
  if (S.segments.some((x) => x.id === w.id)) return;
  const list = document.activeElement?.closest?.('.possible-list');
  S.drafts[w.id] = { source: w.source, start: w.start, end: w.end, label: w.label };
  addSeg(w.id); // re-renders the chat, which shows it as in the sequence
  // the pressed button is gone with the re-render: the next one to add in the same list, else the message box
  const next = list && [...$('#chatLog').querySelectorAll('.possible-list')]
    .find((l) => l.dataset.ev === list.dataset.ev)?.querySelector('[data-act="add-possible"]:not(:disabled)');
  (next ?? $('#chatInput')).focus();
  $('#sideLive').textContent = `Added to the sequence: ${w.label}. Ctrl+Z undoes it.`;
}

/** A proposal as a card: its summary, one chip per change, a strip of the sequence after it, then Apply, Preview and Discard (Apply and Discard are gone once applied or discarded) and up to three suggestions, previewed the same way. */
function proposalItem(ev, id) {
  const applied = S.history.some((e) => e.proposal === ev.id);
  const previewing = S.proposal?.id === ev.id;
  const dismissed = !applied && !previewing && S.dismissed.has(ev.id);
  const sugs = (ev.suggestions ?? []).slice(0, MAX_SUGGESTIONS);
  const note = applied ? 'Applied' : previewing ? 'Previewed in the Sequence tab' : dismissed ? 'Discarded' : '';
  return h('li', { class: `msg ai proposal prop-card${previewing ? ' previewing' : ''}`, 'data-id': id, 'data-pid': ev.id },
    h('span', { class: 'prop-label', text: 'Proposal' }),
    h('p', { text: ev.summary }),
    proposalChips(ev),
    proposalStrip(ev),
    note ? h('p', { class: 'msg-meta', text: note }) : null,
    h('span', { class: 'prop-acts' },
      applied || dismissed ? null : h('button', { type: 'button', class: 'btn primary', 'data-act': 'apply', disabled: !!S.applying, onclick: () => applyFromCard(ev) }, S.applying && previewing ? 'Applying…' : 'Apply'),
      h('button', { type: 'button', class: 'btn', 'data-act': 'review', onclick: () => reviewProposal(ev) }, applied ? 'Review again' : 'Preview'),
      applied || dismissed ? null : h('button', { type: 'button', class: 'btn', 'data-act': 'discard', disabled: !!S.applying, onclick: () => discardFromCard(ev) }, 'Discard')),
    sugs.length ? h('span', { class: 'msg-acts', role: 'group', 'aria-label': 'Suggestions' },
      sugs.map((s, i) => h('button', {
        type: 'button', class: `btn small outline${S.proposal?.id === s.id ? ' on' : ''}`, 'data-act': `suggest${i}`,
        title: changeCounts(s.diff, s.color), onclick: () => reviewProposal(s),
      }, s.summary))) : null);
}

/** One chip per change count (the words of changeCounts, the screen reader hears them as a list); a dashed chip for each automatic fact the diff carries. */
function proposalChips(ev) {
  const parts = changeParts(ev.diff, ev.color);
  const spread = (ev.diff ?? []).filter((r) => r.auto === 'spread').length;
  const filled = (ev.diff ?? []).filter((r) => r.auto && r.auto !== 'spread').length;
  const kids = (parts.length ? parts : ['no change']).flatMap((x, i) => [i ? h('span', { class: 'vh', text: ', ' }) : null, h('span', { class: 'chip', text: x })]);
  if (spread) kids.push(h('span', { class: 'vh', text: ', ' }), h('span', { class: 'chip auto', title: 'Moved or added by Cutpoint to spread the clip kinds you asked to mix', text: `${spread} auto-spread` }));
  if (filled) kids.push(h('span', { class: 'vh', text: ', ' }), h('span', { class: 'chip auto', title: 'Added by Cutpoint to reach the length you asked for', text: `${filled} auto-filled` }));
  return h('span', { class: 'chips' }, kids);
}

/** The sequence after the change as a thin strip: a block per clip, as wide as the clip is long, in its kind's colour. */
function proposalStrip(ev) {
  const segs = (ev.after ?? []).filter((x) => x.end > x.start);
  if (!segs.length) return null;
  const total = segs.reduce((n, x) => n + x.end - x.start, 0);
  return h('span', { class: 'prop-strip', role: 'img', 'aria-label': `The sequence after the change: ${plural(segs.length, 'clip')}, ${fmtLen(total)}` },
    segs.map((x) => h('span', { class: 'prop-block', 'data-kind': kindOf(cands().find((c) => c.id === x.id)), style: `flex-grow: ${x.end - x.start}` })));
}

// ---------- ranking cards (#902) ----------
// preview_ranking's card: the project ranked in another library style, before/after. Its state is
// derived from the project's ranking_key and the style its highlights.json is ranked in, so it
// survives a reload; only the apply route (Use this ranking) writes anything.

/** A style name as a card says it: "funny" → "Funny". */
const styleTitle = (s) => (s ? `${String(s)[0].toUpperCase()}${String(s).slice(1)}` : '');
/** A run holds the open project (a job of this page on it, the command line, another server): rankings wait. */
const rankBusy = () => !!S.project?.busy || (!!S.active && S.active.project_id === S.id);
const rankUrl = (what) => `/api/projects/${encodeURIComponent(S.id)}/ranking/${what}`;
const MISSING_SHOWN = 3; // details of what is missing on a Needs analyze card

/** The card's preview (a newer one from Preview again wins; the model's reason stays) and its state:
 * applying, in-use, stale, needs-analyze or ready. */
function rankView(ev, id) {
  const local = S.rankings.get(id) ?? {};
  const v = { ...ev, ...(local.payload ?? {}), reason: ev.reason };
  const key = S.project?.ranking_key ?? null;
  const now = S.project?.highlights?.style ?? null;
  const state = S.rankApplying === id ? 'applying'
    : key !== v.base && now === v.style ? 'in-use'
      : key !== v.base || local.stale === v.base ? 'stale'
        : !v.ready ? 'needs-analyze' : 'ready';
  return { v, state, now, previewing: !!local.previewing };
}

/** A row's move: New, ↑n, ↓n or = (the arrows read as words). */
function rankBadge(r) {
  if (r.was == null) return h('span', { class: 'prop-mark added rank-new', text: 'New' });
  const d = r.was - r.rank;
  if (!d) return h('span', { class: 'prop-mark' }, h('span', { 'aria-hidden': 'true', text: '=' }), h('span', { class: 'vh', text: 'same place' }));
  return h('span', { class: `prop-mark ${d > 0 ? 'rank-up' : 'rank-down'}` },
    h('span', { 'aria-hidden': 'true', text: `${d > 0 ? '↑' : '↓'}${Math.abs(d)}` }),
    h('span', { class: 'vh', text: `${d > 0 ? 'up' : 'down'} ${plural(Math.abs(d), 'place')}` }));
}

/** One moment of the new top: rank, label, file and time, its move; a moment the page has focuses on click. */
function rankRow(r) {
  const c = cands().find((m) => m.id === r.id);
  const text = h('span', { class: 'prop-text' },
    h('span', { class: 'prop-title', text: shownName(c, r.label) || r.label || r.cli_id }),
    h('span', { class: 'prop-time tc', text: `${r.file} ${fmt(r.start)}–${fmt(r.end)}` }),
    h('span', { class: 'prop-marks' }, rankBadge(r)));
  const rank = h('span', { class: 'rank tc', text: String(r.rank) });
  return h('li', { class: `prop${r.was == null ? ' added' : ''}`, 'data-id': r.id },
    c ? h('button', { type: 'button', class: 'prop-main', 'data-act': 'focus', onclick: () => focusMoment(r.id) }, rank, text)
      : h('span', { class: 'prop-main' }, rank, text));
}

/** "7 new in top 10", "suggested 8 (was 12)" and, when the moments themselves changed, the new and dropped counts. */
function rankChips(v) {
  const top = v.top ?? [];
  const fresh = top.filter((r) => r.was == null || r.was > top.length).length;
  const parts = [[`${fresh} new in top ${top.length}`, 'chip']];
  const st = v.suggested_top ?? {};
  if (Number.isFinite(st.after)) parts.push([`suggested ${st.after}${Number.isFinite(st.before) && st.before !== st.after ? ` (was ${st.before})` : ''}`, 'chip']);
  if (v.new_ids_count) parts.push([`${plural(v.new_ids_count, 'new moment')}`, 'chip auto']);
  if (v.dropped_count) parts.push([`${v.dropped_count} dropped`, 'chip auto']);
  return h('span', { class: 'chips' }, parts.flatMap(([text, cls], i) => [i ? h('span', { class: 'vh', text: ', ' }) : null, h('span', { class: cls, text })]));
}

/** A ranking card: Ready (the new top 10, Use this ranking, Dismiss), Applying, In use, Stale (Preview
 * again), Needs analyze (Analyze again as X…, or reconnect the footage); Busy while a run holds the project. */
function rankingItem(ev, id) {
  const { v, state, now, previewing } = rankView(ev, id);
  const name = styleTitle(v.style);
  const busy = rankBusy();
  const dismissed = S.dismissed.has(id) && state !== 'in-use';
  const off = busy || previewing || S.rankUndoing || (!!S.rankApplying && S.rankApplying !== id);
  const btn = (act, label, onclick, primary = false, disabled = off) => h('button', { type: 'button', class: `btn${primary ? ' primary' : ''}`, 'data-act': act, disabled, onclick }, label);
  const kids = [
    h('span', { class: 'prop-label', text: 'Ranking' }),
    h('p', { class: 'msg-title', text: `${name} ranking${now && now !== v.style ? ` (now: ${styleTitle(now)})` : ''}` }),
    v.reason ? h('p', { text: v.reason }) : null,
  ];
  let acts = [];
  const notes = [];
  if (state === 'in-use') {
    notes.push(`In use: the moments are ranked in the ${name} style.`);
    const u = S.project?.ranking_undo;
    if (u?.action === 'undo') { // this ranking is the change Undo takes back
      acts = [h('button', { type: 'button', class: 'btn', 'data-act': 'undo-ranking', disabled: off || !!S.rankApplying,
        'aria-label': `Undo the ${name} ranking, back to ${styleTitle(u.style)}`, onclick: () => undoRanking(id) }, S.rankUndoing ? 'Undoing…' : 'Undo')];
    }
  }
  else if (dismissed) notes.push('Dismissed');
  else if (state === 'stale') {
    notes.push('The moments changed since this preview.');
    acts = [btn('preview-again', previewing ? 'Previewing…' : 'Preview again', () => previewRanking(ev, id), true), btn('dismiss', 'Dismiss', () => dismissRanking(id), false, false)];
  } else if (state === 'needs-analyze') {
    const missing = v.missing ?? [];
    const more = missing.length - MISSING_SHOWN;
    kids.push(h('p', { text: v.can_analyze ? `Ranking in the ${name} style needs another analysis first.` : 'Reconnect the footage to rank it in this style.' }),
      h('ul', { class: 'rank-missing msg-meta' }, missing.slice(0, MISSING_SHOWN).map((m) => h('li', { text: `${m.detail}${m.file && !String(m.detail).includes(m.file) ? ` (${m.file})` : ''}` })),
        more > 0 ? h('li', { text: `and ${more} more` }) : null));
    acts = [...(v.can_analyze ? [btn('analyze', `Analyze again as ${name}…`, () => analyzeAgain(false, { style: v.style, name }), true, off || !!S.active)] : []), btn('dismiss', 'Dismiss', () => dismissRanking(id), false, false)];
  } else {
    kids.push(rankChips(v),
      ...(v.notes ?? []).map((n) => h('p', { class: 'msg-meta', text: n })),
      h('ol', { class: 'prop-list', 'aria-label': `The first ${plural((v.top ?? []).length, 'moment')} in the ${name} ranking` }, (v.top ?? []).map(rankRow)));
    acts = [btn('use-ranking', state === 'applying' ? 'Applying…' : 'Use this ranking', () => useRanking(ev, id), true, off || state === 'applying'),
      btn('dismiss', 'Dismiss', () => dismissRanking(id), false, state === 'applying')];
  }
  if (busy && acts.length) notes.push(BUSY_TEXT);
  return h('li', { class: `msg ai proposal prop-card rank-card${state === 'in-use' ? ' previewing' : ''}`, 'data-id': id, 'data-rid': id, 'data-state': dismissed ? 'dismissed' : state },
    ...kids,
    ...notes.map((n) => h('p', { class: 'msg-meta rank-note', text: n })),
    acts.length ? h('span', { class: 'prop-acts' }, acts) : null);
}

/** Dismiss a ranking card: it stays in the chat without its buttons. */
function dismissRanking(id) {
  if (S.rankApplying === id) return;
  chatError('');
  S.dismissed.add(id);
  renderChat();
  $('#chatLive').textContent = 'Ranking dismissed';
  $('#chatInput').focus(); // the card's buttons are gone
}

/** After a card's request its pressed button may be gone or disabled: focus the card's first enabled button, else the message box. */
function focusRankCard(id) {
  if (document.activeElement !== document.body && document.activeElement?.isConnected) return;
  ($(`#chatLog [data-rid="${CSS.escape(id)}"] button:not(:disabled)`) ?? $('#chatInput')).focus();
}

/** Preview again: the preview route straight away (no model turn); the answer replaces the card's preview. */
async function previewRanking(ev, id) {
  const pid = S.id;
  const { v } = rankView(ev, id);
  if (!pid || S.rankApplying || S.rankUndoing || S.rankings.get(id)?.previewing || rankBusy()) return;
  chatError('');
  S.rankings.set(id, { ...S.rankings.get(id), previewing: true });
  renderChat();
  let said = '';
  try {
    const r = await api(rankUrl('preview'), { method: 'POST', body: { style: v.style } });
    if (S.id !== pid) return;
    S.rankings.set(id, { payload: r });
    S.dismissed.delete(id);
    if (r.base !== S.project?.ranking_key) await refreshProject(); // the page's key is older than the server's
    said = r.ready ? `Previewed the ${styleTitle(r.style)} ranking again.` : `The ${styleTitle(r.style)} ranking needs another analysis first.`;
  } catch (e) {
    if (S.id !== pid) return;
    S.rankings.set(id, { ...S.rankings.get(id), previewing: false });
    if (e.code === 'folder_busy') await refreshProject();
    if (e.code !== 'folder_busy' || !rankBusy()) chatError(e.message);
  } finally {
    if (S.id === pid) {
      const local = S.rankings.get(id);
      if (local) local.previewing = false;
      renderChat();
      if (said) $('#chatLive').textContent = said;
      focusRankCard(id);
    }
  }
}

/** Use this ranking: the apply route redoes the ranking against the previewed key, then the page reloads the project. */
async function useRanking(ev, id) {
  const pid = S.id;
  const { v, state } = rankView(ev, id);
  if (!pid || S.rankApplying || S.rankUndoing || state !== 'ready' || rankBusy()) return;
  const name = styleTitle(v.style);
  S.rankApplying = id;
  chatError('');
  renderChat();
  $('#chatLive').textContent = `Applying the ${name} ranking…`;
  let said = '';
  try {
    const r = await api(rankUrl('apply'), { method: 'POST', body: { style: v.style, base: v.base } });
    if (S.id !== pid) return;
    S.shown = null; // the assistant's highlights were of the ranking before
    S.rankApplying = null;
    await refreshProject();
    said = `Using the ${name} ranking now${r.new_ids?.length ? `; ${plural(r.new_ids.length, 'new moment')}` : ''}.`;
  } catch (e) {
    if (S.id !== pid) return;
    S.rankApplying = null;
    if (e.code === 'stale') {
      S.rankings.set(id, { ...S.rankings.get(id), stale: v.base });
      await refreshProject();
      said = 'The moments changed since this preview. Preview it again.';
    } else if (e.code === 'needs_analyze') {
      return await previewRanking(ev, id); // the card shows what is missing now
    } else if (e.code === 'folder_busy') {
      await refreshProject();
      if (!rankBusy()) chatError(e.message);
    } else chatError(e.message);
  } finally {
    if (S.rankApplying === id) S.rankApplying = null;
    if (S.id === pid) {
      renderChat();
      if (said) $('#chatLive').textContent = said;
      focusRankCard(id);
    }
  }
}

/** The Moments column's "Undo ranking (was Gaming)" or, after an Undo, "Redo ranking (Funny)": shown
 * while the backup matches the analysis (ranking_undo), disabled while a run or a ranking change holds the project. */
export function renderRankUndo() {
  const p = S.project;
  const u = p?.highlights ? p.ranking_undo : null;
  const row = $('#rankUndoRow');
  const b = $('#rankUndoBtn');
  if (!u && b === document.activeElement) $('#momentsTitle').focus(); // it is going: keep focus in the column
  row.hidden = !u;
  if (!u) return;
  const name = styleTitle(u.style);
  const redo = u.action === 'redo';
  const label = S.rankUndoing ? (redo ? 'Redoing ranking…' : 'Undoing ranking…') : redo ? `Redo ranking (${name})` : `Undo ranking (was ${name})`;
  if (b.textContent !== label) b.textContent = label;
  const busy = rankBusy();
  b.disabled = busy || S.rankUndoing || !!S.rankApplying;
  b.title = busy ? BUSY_TEXT : redo ? `Use the ${name} ranking again` : `Go back to the ${name} ranking`;
}

/** Undo (or Redo) the last ranking change: the undo route swaps highlights.json and its backup against the
 * page's ranking_key, then the page reloads the project. ``id``: the ranking card it was asked from (else the Moments link). */
export async function undoRanking(id = null) {
  const pid = S.id;
  const u = S.project?.ranking_undo;
  if (!pid || !u || S.rankUndoing || S.rankApplying || rankBusy()) return;
  const name = styleTitle(u.style);
  const redo = u.action === 'redo';
  const live = id ? $('#chatLive') : $('#rankUndoLive');
  const error = (text) => (id ? chatError(text) : banner(text));
  S.rankUndoing = true;
  error('');
  renderRankUndo();
  renderChat();
  live.textContent = redo ? 'Redoing the ranking…' : 'Undoing the ranking…';
  let said = '';
  try {
    await api(rankUrl('undo'), { method: 'POST', body: { base: S.project.ranking_key } });
    if (S.id !== pid) return;
    S.shown = null; // the assistant's highlights were of the ranking before
    S.rankUndoing = false;
    await refreshProject();
    said = redo ? `Using the ${name} ranking again.` : `Back to the ${name} ranking.`;
  } catch (e) {
    if (S.id !== pid) return;
    S.rankUndoing = false;
    await refreshProject(); // stale, busy or nothing to undo: the page follows the server
    if (e.code !== 'folder_busy' || !rankBusy()) error(e.message);
  } finally {
    S.rankUndoing = false;
    if (S.id === pid) {
      renderRankUndo();
      renderChat();
      live.textContent = said;
      if (id) focusRankCard(id);
      else if (!$('#rankUndoRow').hidden) $('#rankUndoBtn').focus();
    }
  }
}

/** Draw the ranking cards again when a run starts or stops holding the project (their buttons follow it). */
let rankBusySeen = false;
export function syncRankingBusy() {
  renderRankUndo();
  const busy = rankBusy();
  if (busy === rankBusySeen) return;
  rankBusySeen = busy;
  if (S.chat?.turns.some((t) => t.events.some((e) => e.kind === 'ranking'))) renderChat();
}

/** view_moment without frame consent: allow video frames of this project to go to the host (and ask again). */
function framesItem(ev, id, t) {
  const f = S.chat?.frames;
  const allowed = !!f && !f.needed && f.key === ev.key;
  const busy = !!S.turn || S.sending;
  return h('li', { class: 'msg ai frames-consent', 'data-id': id },
    h('div', { role: 'group', 'aria-label': 'Video frames' },
      h('p', { class: 'msg-title', text: allowed ? `Video frames of this project are allowed for ${ev.host}.` : `Let the assistant watch ${momentName(ev.id)}?` }),
      allowed ? null : h('p', { class: 'msg-meta', text: FRAMES_SENDS(ev.host) }),
      h('span', { class: 'msg-acts' },
        allowed ? null : h('button', { type: 'button', class: 'btn small primary', 'data-act': 'frames-allow', onclick: () => giveFramesConsent(ev.key, '') }, 'Allow for this project'),
        h('button', {
          type: 'button', class: 'btn small', 'data-act': 'frames-ask', disabled: busy,
          onclick: () => (allowed ? askAgain(t.user) : giveFramesConsent(ev.key, t.user)),
        }, allowed ? 'Ask again' : 'Allow and ask again'))));
}

/** Allow frames of this project for the provider in use (the server checks it is still the one asked about), then send ``again`` if given. */
async function giveFramesConsent(key, again) {
  const chat = S.chat;
  if (!chat || !key) return;
  chatError('');
  try {
    const r = await api(aiUrl('/frames-consent'), { method: 'POST', body: { key } });
    if (S.chat !== chat) return;
    chat.frames = r.frames;
  } catch (e) {
    if (S.chat !== chat) return;
    if (e.status === 409) await loadAssistant(); // the provider changed: show the one in use
    return chatError(e.message);
  }
  $('#chatLive').textContent = 'Video frames allowed for this project.';
  renderChat();
  if (again) askAgain(again);
  else $('#chatInput').focus(); // the button is gone
}

/** Send ``text`` again as a new message. */
function askAgain(text) {
  if (S.turn || S.sending) return;
  $('#chatInput').value = text;
  $('#chatForm').requestSubmit();
}

function historyRow(entry) {
  const when = String(entry.applied ?? '').replace('T', ' ').slice(0, 16);
  return h('li', { class: 'history-row', 'data-id': entry.id },
    h('span', { class: 'history-text' },
      h('span', { text: entry.summary || 'Change' }),
      h('span', { class: 'msg-meta tc', text: `${when}, ${changeCounts(entry.diff, entry.color?.after)}` })),
    h('button', { type: 'button', class: 'btn small', 'data-act': 'revert', onclick: () => revertChange(entry) }, 'Revert this change'));
}

function renderChatForm() {
  const ready = !!S.chat?.loaded && !!S.project?.highlights && !S.chat.consent?.needed && !S.chat.consent?.missing_key;
  const send = $('#chatSend');
  const had = document.activeElement;
  $('#chatInput').disabled = !ready;
  send.disabled = !ready || !!S.turn || S.sending;
  if (had === send && send.disabled) $('#chatInput').focus();
}

// The bubbles of each finished turn as last drawn (A755): turn id -> {key, items}. The chat is
// drawn once a second while a turn runs and on every renderAll; a finished turn whose events and
// the page state its bubbles show (chatDeps) are as they were keeps its nodes, so no reply is
// parsed as Markdown again. A running turn is always drawn again.
let turnCache = new Map();
let turnCacheDeps = '';
const objIds = new WeakMap();
let objN = 0;
/** A number for object ``o`` (the same object, the same number), or null. */
function objId(o) {
  if (!o || typeof o !== 'object') return null;
  if (!objIds.has(o)) objIds.set(o, ++objN);
  return objIds.get(o);
}

/** What turnItems reads besides the turn: a change draws every turn again. Whether a turn is
 * being sent or answered matters only to a frames consent card (its Ask again). */
const chatDeps = (turns) => JSON.stringify([
  S.id, objId(S.chat), objId(S.project?.highlights), S.project?.highlights?.style ?? null,
  S.project?.ranking_key ?? null, S.project?.ranking_undo ?? null, rankBusy(), !!S.active,
  S.segments.map((s) => [s.id, s.label]), S.shown && [...S.shown], [...S.dismissed],
  S.history.map((e) => e.proposal), S.proposal?.id ?? null, !!S.applying,
  S.rankApplying ?? null, !!S.rankUndoing,
  [...S.rankings].map(([id, r]) => [id, objId(r.payload), !!r.previewing, r.stale ?? null]),
  S.chat?.frames ?? null, S.chat?.files ?? null, !!S.canReveal,
  turns.some((t) => t.events.some((e) => e.kind === 'frames_consent')) && (!!S.turn || !!S.sending),
]);

/** The chat log's bubbles of ``turns``: a finished turn's kept ones when nothing it shows changed. */
function chatItems(turns) {
  const deps = chatDeps(turns);
  const old = deps === turnCacheDeps ? turnCache : new Map();
  turnCache = new Map(); // only the turns shown now
  turnCacheDeps = deps;
  return turns.flatMap((t) => {
    if (t.status === 'running') return turnItems(t);
    const key = JSON.stringify([t.events.length, t.status, t.error ?? null, t.ranking_key ?? null,
      t.user]);
    const hit = old.get(t.id);
    const items = hit?.key === key ? hit.items : turnItems(t);
    turnCache.set(t.id, { key, items });
    return items;
  });
}

export function renderChat() {
  const c = S.chat;
  const prov = c?.provider;
  $('#chatProvider').textContent = !c ? ''
    : prov ? prov.label // "Anthropic · claude-sonnet-5-5 · api.anthropic.com"
      : c.error ? `Unavailable: ${c.error}` : 'Connecting…';
  $('#chatDot').dataset.state = !c ? '' : prov ? (c.consent?.missing_key ? 'bad' : c.consent?.needed ? 'wait' : 'ok') : c.error ? 'bad' : 'wait';
  $('#chatProvider').title = $('#chatProvider').textContent;
  const note = $('#chatNote');
  note.hidden = !prov?.local;
  note.textContent = prov?.local ? SMALL_MODEL_NOTE : '';
  const idx = c?.frameIndex;
  $('#chatIndexNote').hidden = !idx || idx.ok;
  $('#chatIndexNote').textContent = idx && !idx.ok ? indexNote(idx) : '';
  // A missing key comes first: no agreeing to send to a provider that cannot be asked yet.
  const noKey = c?.consent?.missing_key || '';
  const consent = !noKey && c?.consent?.needed ? c.consent : null;
  $('#chatConsent').hidden = !consent && !noKey;
  $('#chatConsentTitle').textContent = !noKey ? 'Send project data to a hosted model?'
    : c.consent.missing_kind === 'login' ? 'Login needed' : 'API key needed';
  $('#chatConsentOk').hidden = !consent;
  $('#chatConsentText').textContent = noKey ? sentence(noKey)
    : consent
    ? `Each message sends ${CONSENT_SENDS} to ${PROVIDER_NAMES[prov?.name] ?? prov?.name ?? 'the provider'} at ${consent.host}. Analyzing footage with this model chosen also sends ${ANALYSIS_SENDS}. Nothing is sent until you agree. You agree once for this provider and host; Cutpoint saves it in config.json. Change the provider in the Models tab.`
    : '';
  $('#chatClear').disabled = !c?.turns.length;
  const log = $('#chatLog');
  const items = c ? chatItems(c.turns) : [];
  const col = log.closest('.col');
  const atEnd = !col || col.scrollHeight - col.scrollTop - col.clientHeight < 24;
  const count = log.children.length;
  const height = log.scrollHeight;
  preserveFocus(() => log.replaceChildren(...(items.length ? items : [h('li', { class: 'msg step chat-empty' },
    !S.project ? 'Open a project to chat about its moments.'
      : !S.project.highlights ? 'Analyze the footage first; the assistant works with its moments.'
        : 'Ask for moments to find, add, remove, reorder or trim. Each change is proposed first: apply it from its card, or preview it in the Sequence tab.')])));
  if (items.length && (items.length !== count || log.scrollHeight !== height) && atEnd && !$('#panelAssistant').hidden) revealRow(log.lastElementChild);
  const shown = S.history.slice(-MAX_HISTORY_SHOWN).reverse();
  $('#chatHistory').hidden = !shown.length;
  $('#chatHistory summary').textContent = `Applied changes (${S.history.length})`;
  preserveFocus(() => $('#historyList').replaceChildren(...shown.map(historyRow)));
  renderChatForm();
}

/** Agree once to send project data to the hosted provider in use; the server checks it is still the one shown. */
async function giveConsent() {
  const chat = S.chat;
  const key = chat?.consent?.key;
  if (!key) return;
  chatError('');
  try {
    const r = await api('/api/assistant/consent', { method: 'POST', body: { key } });
    if (S.chat !== chat) return;
    chat.consent = r.consent;
  } catch (e) {
    if (S.chat !== chat) return;
    if (e.status === 409) await loadAssistant(); // the provider changed: show the one in use
    return chatError(e.message);
  }
  renderChat();
  if (!chat.consent?.needed) $('#chatInput').focus(); // the button is gone
}

export function bindAssistant() {
  $('#chatForm').addEventListener('submit', sendChat);
  $('#chatInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $('#chatForm').requestSubmit();
    }
  });
  $('#chatClear').addEventListener('click', clearChat);
  $('#chatConsentOk').addEventListener('click', giveConsent);
  $('#propApply').addEventListener('click', applyProposal);
  $('#propDiscard').addEventListener('click', discardProposal);
}
