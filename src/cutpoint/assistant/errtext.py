"""Error text a chat turn may keep: no folders of this computer (#1017).

A turn's error is saved as its reply in ``chat.json`` and sent to the model on later turns, and
that model can be a hosted one, which may get file names but never where they are. So every
absolute path in such a text (``C:\\...``, ``\\\\server\\...``, ``/...``, ``~/...``, quoted or
not) becomes its bare file name, or ``<folder>`` when it ends in a folder (a name without an
extension); the full error stays in the log.

Known limit: a path that is not quoted ends at a space unless the next word goes on with a
separator, so one whose last folder has a space (``C:\\Users\\Travis Smith now``) keeps the
words after that space (``<folder> Smith now``). Python quotes the paths of its own errors, so
those are replaced whole.
"""

from __future__ import annotations

import re

# a path's start: a file:// URL, a drive, a UNC (or repr'd UNC) prefix, a forward-slash UNC,
# a drive-less Windows root (two parts at least), a POSIX root (also right after a colon, as in
# "error:/home/..."; not the path of an http(s) URL, nor the "url: /api/chat" of a requests
# error) or ~
_START = (
    r"(?:(?<![\w])(?i:file):/+"
    r"|(?<![\w])[A-Za-z]:[\\/]"
    r"|(?<![\w\\])\\\\+(?=[^\s\\'\"])"
    r"|(?<![\w:/\\])//(?=[^\s/])"
    r"|(?<![\w\\:])\\(?=[^\s\\'\"]+\\)"
    r"|(?<![\w/.~\\-])(?<!url: )/(?=[^\s/])"
    r"|(?<![\w])~[\\/])"
)
# quoted: everything up to the closing quote of the same kind (spaces too; Python puts a path
# with a ' in double quotes). In single quotes, a ' before a letter is part of the path
# ('C:\Users\Sean O'Brien\a.mp4'): the closing one is followed by the end, a space or
# punctuation. Its parts and apostrophes are bounded, so a text of many ' with no closing one
# costs a bounded try at each quote, not the rest of the text (a longer path is left to _BARE)
_QUOTED = re.compile(
    rf"\"({_START}[^\"]*)\""
    rf"|'({_START}[^'\n]{{0,1024}}(?:'(?=\w)[^'\n]{{0,1024}}){{0,4}})'(?=[\s,.;:)\]}}!?]|$)"
)
# not quoted: up to whitespace, or over a space when the next word goes on with a separator
# ("C:\Users\Me\My Videos\a.mp4"); a ' is part of it ("C:\Users\Sean O'Brien\...")
_BARE = re.compile(rf"{_START}(?:[^\s\"<>|]|[ ](?=[^\s\"<>|]*[\\/]))*")
_TRAIL = ".,:;)]}"
_FILE = re.compile(r".*[^.]\.[^.\s]+")  # a name with an extension: a file's
FOLDER = "<folder>"  # what a path that ends in a folder becomes: its name is host info too
_ERRNO = re.compile(r"^\[(?:Errno|WinError) -?\d+\]\s*")


def _name(path: str) -> str:
    """``path``'s last part, with the punctuation that ended the sentence kept."""
    body = path.rstrip(_TRAIL)
    tail = path[len(body) :]
    parts = [p for p in re.split(r"[\\/]+", body) if p]
    name = parts[-1] if parts else ""
    if not _FILE.fullmatch(name):  # a folder (a user's name, say), a drive or home alone
        name = FOLDER
    return name + tail


def no_paths(text: str) -> str:
    """``text`` with every absolute path replaced by its bare name."""
    text = _QUOTED.sub(lambda m: _name(m.group(1) or m.group(2)), text)
    return _BARE.sub(lambda m: _name(m.group(0)), text)


def error_text(err: BaseException) -> str:
    """``err`` for a chat turn: its kind and message, without paths or an errno prefix
    (``PermissionError: Permission denied: highlights.json``)."""
    message = _ERRNO.sub("", no_paths(str(err)))
    kind = type(err).__name__
    return f"{kind}: {message}" if message else kind
