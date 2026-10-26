// "New project" dialog, and the machine's own open dialog for choosing a path (the server opens it;
// nothing uploads). Without one (headless, Docker) the path is typed or pasted.
import { api } from './api.js';
import { $, baseName, h } from './util.js';

const SOURCE_HINT = 'Choose a video or a folder of videos, or type or paste its full path.';
const SOURCE_HINT_TYPED = 'Type or paste the full path of a video, or of a folder of videos, on the computer running cutpoint.';
// Added for an Insta360 file (a 360 folder cannot be told apart here); the server sets the style.
const PANO_HINT = 'Insta360 360 footage uses the 360 style (ski unless pano.default_style in config.json says otherwise), not the one chosen below.';

let picking = false; // the server allows one open dialog at a time

/**
 * Ask for a video (`kind` 'file') or a folder in the native dialog, opened near `start`. Every
 * [data-pick] button is disabled while it is open. Resolves the chosen path, or null on cancel.
 */
export async function pickPath(kind, start = '') {
  if (picking) return null;
  picking = true;
  const back = document.activeElement;
  for (const b of document.querySelectorAll('[data-pick]')) b.disabled = true;
  try {
    const q = new URLSearchParams({ kind });
    if (start.trim()) q.set('start', start.trim());
    return (await api(`/api/pick?${q}`, { method: 'POST' })).path;
  } finally {
    picking = false;
    for (const b of document.querySelectorAll('[data-pick]')) b.disabled = false;
    if (back?.isConnected && document.activeElement === document.body) back.focus(); // disabling dropped it
  }
}

export function initNewProject(onCreated, canPick) {
  const dlg = $('#newDialog');
  const source = $('#newSource');
  const name = $('#newName');
  const style = $('#newStyle');
  const error = $('#newError');
  let nameTouched = false;

  const setError = (msg) => (error.textContent = msg || '');
  const stem = (p) => baseName(p.replace(/[\\/]+$/, '')).replace(/\.[^.]+$/, '');

  for (const b of dlg.querySelectorAll('[data-pick]')) {
    b.hidden = !canPick;
    b.addEventListener('click', async () => {
      setError('');
      try {
        const path = await pickPath(b.dataset.pick, source.value);
        if (!path) return;
        source.value = path;
        source.dispatchEvent(new Event('input')); // as if typed: the name follows
        source.focus();
      } catch (e) {
        setError(e.message);
      }
    });
  }
  const hint = canPick ? SOURCE_HINT : SOURCE_HINT_TYPED;
  $('#newSourceHint').textContent = hint;
  name.addEventListener('input', () => (nameTouched = true));
  source.addEventListener('input', () => {
    if (!nameTouched) name.value = stem(source.value.trim());
    $('#newSourceHint').textContent = /\.(insv|lrv)$/i.test(source.value.trim()) ? `${hint} ${PANO_HINT}` : hint;
  });
  $('#newCancel').addEventListener('click', () => dlg.close());

  $('#newForm').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const body = { name: name.value.trim(), source: source.value.trim(), style: style.value };
    if (!body.source) return setError('Choose a video file or a folder of footage.');
    if (!body.name) return setError('Give the project a name.');
    if (!body.style) return setError('No highlight style is available from the server.');
    setError('');
    $('#newCreate').disabled = true;
    try {
      const created = await api('/api/projects', { method: 'POST', body });
      dlg.close();
      onCreated(created);
    } catch (e) {
      setError(e.message);
    } finally {
      $('#newCreate').disabled = false;
    }
  });

  return function open(styles) {
    nameTouched = false;
    source.value = name.value = '';
    $('#newSourceHint').textContent = hint;
    setError('');
    style.replaceChildren(...styles.map((s) => h('option', { value: s, text: s })));
    dlg.showModal();
    source.focus();
  };
}
