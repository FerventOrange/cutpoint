// "New project" dialog, and the machine's own open dialog for choosing a path (the server opens it;
// nothing uploads). Without one (headless, Docker) the path is typed or pasted.
import { api } from './api.js';
import { $, baseName, h } from './util.js';

const SOURCE_HINT = 'Add videos or folders of videos, or type or paste their full paths, one per line.';
const SOURCE_HINT_TYPED = 'Type or paste the full paths of videos, or of folders of videos, on the computer running cutpoint, one per line.';
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
  const recursive = $('#newRecursive');
  const exclude = $('#newExclude');
  const error = $('#newError');
  let nameTouched = false;

  const setError = (msg) => (error.textContent = msg || '');
  const stem = (p) => baseName(p.replace(/[\\/]+$/, '')).replace(/\.[^.]+$/, '');
  const lines = (v) => v.split(/\r?\n/).map((x) => x.trim()).filter(Boolean);
  // as the server reads them: Windows' Copy as path puts each path in double quotes
  const sourceLines = (v) => lines(v).map((x) => x.replace(/^"(.+)"$/, '$1'));
  // As analyze names a project: its one input's name, else that of the folder its inputs share
  const defaultName = (paths) => {
    if (paths.length < 2) return stem(paths[0] ?? '');
    const parts = paths.map((p) => p.split(/[\\/]+/).filter(Boolean));
    let n = 0;
    while (parts.every((x) => n < x.length - 1 && x[n] === parts[0][n])) n++;
    const shared = n ? parts[0][n - 1] : '';
    return shared && !/:$/.test(shared) ? shared : stem(paths[0]); // not a bare drive (D:)
  };

  for (const b of dlg.querySelectorAll('[data-pick]')) {
    b.hidden = !canPick;
    b.addEventListener('click', async () => {
      setError('');
      try {
        const have = sourceLines(source.value);
        const path = await pickPath(b.dataset.pick, have.at(-1) ?? '');
        if (!path) return;
        if (!have.includes(path)) source.value = [...have, path].join('\n'); // a line more
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
    const paths = sourceLines(source.value);
    if (!nameTouched) name.value = defaultName(paths);
    $('#newSourceHint').textContent = paths.some((p) => /\.(insv|lrv)$/i.test(p)) ? `${hint} ${PANO_HINT}` : hint;
  });
  $('#newCancel').addEventListener('click', () => dlg.close());

  $('#newForm').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    // each field is an analyze option: its inputs, --recursive and --exclude
    const body = {
      name: name.value.trim(),
      sources: sourceLines(source.value),
      style: style.value,
      recursive: recursive.checked,
      exclude: lines(exclude.value),
    };
    if (!body.sources.length) return setError('Choose a video file or a folder of footage.');
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
    source.value = name.value = exclude.value = '';
    recursive.checked = false;
    $('#newSourceHint').textContent = hint;
    setError('');
    style.replaceChildren(...styles.map((s) => h('option', { value: s, text: s })));
    dlg.showModal();
    source.focus();
  };
}
