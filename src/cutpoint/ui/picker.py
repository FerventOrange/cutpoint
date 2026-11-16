"""The native file and folder pickers of ``cutpoint ui``."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from cutpoint.assistant import keys as ai_keys
from cutpoint.ui.common import bad, logger
from cutpoint.utils import PANO_EXTS, VIDEO_EXTS


def folder_opener() -> Callable[[Path], None] | None:
    """How this machine shows a folder in its file manager (no shell, no waiting), else None.

    None where nobody could see the window: no ``xdg-open`` or no display (servers, Docker).
    """
    if sys.platform == "win32":
        start = getattr(os, "startfile", None)
        return None if start is None else lambda folder: start(str(folder))
    if sys.platform == "darwin":
        exe = shutil.which("open")
    elif os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        exe = shutil.which("xdg-open")
    else:
        exe = None
    if exe is None:
        return None

    def run(folder: Path) -> None:
        proc = subprocess.Popen(
            [exe, str(folder)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=ai_keys.child_env(),
        )
        threading.Thread(target=proc.wait, daemon=True).start()  # reap it; never block the request

    return run


# Windows: Tk 8.6's dialogs are the system's IFileDialog (breadcrumbs, search, quick access; the
# folder one too, unlike WinForms' tree FolderBrowserDialog in Windows PowerShell). A hidden topmost
# owner keeps them above the browser. Runs as ``python -I -c`` (no current folder on sys.path).
TK_PICK = """\
import sys
kind, start, exts, pano = sys.argv[1], sys.argv[2] or None, sys.argv[3], sys.argv[4]
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # sharp text on scaled displays
except Exception:
    pass
try:
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    if kind == "folder":
        out = filedialog.askdirectory(
            parent=root, initialdir=start, mustexist=True, title="Choose a folder"
        )
    else:
        out = filedialog.askopenfilename(
            parent=root, initialdir=start, title="Choose a video",
            filetypes=[("Videos", exts), ("Insta360 360 video", pano), ("All files", "*.*")],
        )
    root.destroy()
except Exception as err:
    sys.stderr.write(f"{err}\\n")
    sys.exit(3)
sys.stdout.buffer.write((out or "").encode("utf-8"))
"""

# macOS: ``choose file`` has no "All files" switch, so it lists videos only. A cancel prints nothing.
OSA_PICK = """\
on run argv
  activate
  try
    if item 1 of argv is "folder" then
      if (count of argv) > 1 then
        set f to choose folder with prompt "Choose a folder" default location (POSIX file (item 2 of argv))
      else
        set f to choose folder with prompt "Choose a folder"
      end if
    else
      if (count of argv) > 1 then
        set f to choose file with prompt "Choose a video" of type {%s} default location (POSIX file (item 2 of argv))
      else
        set f to choose file with prompt "Choose a video" of type {%s}
      end if
    end if
  on error number -128
    return ""
  end try
  return POSIX path of f
end run
"""


def path_picker() -> Callable[[str, Path | None], tuple[list[str], int | None]] | None:
    """How this machine asks the user for a video (``kind`` "file") or a folder, else None.

    The result maps ``(kind, start folder)`` to the dialog's argv (no shell) and the exit code that
    means "cancelled" (None: a cancel exits 0 and prints nothing). None where nobody could see the
    window: no display, or neither zenity nor kdialog (Linux), no ``osascript`` (macOS), no
    tkinter (Windows), or an SSH session (Windows, macOS: the dialog would open on the remote
    desktop, if any, and keep the one-dialog lock until someone closes it there).
    """
    exts = sorted(e.lstrip(".") for e in VIDEO_EXTS)
    pano = sorted(e.lstrip(".") for e in PANO_EXTS)  # a second filter: VIDEO_EXTS stays flat-only
    if sys.platform in ("win32", "darwin") and (
        os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY")
    ):
        return None
    if sys.platform == "win32":
        if importlib.util.find_spec("_tkinter") is None:
            return None
        patterns, pano_patterns = (" ".join(f"*.{e}" for e in x) for x in (exts, pano))
        return lambda kind, start: (
            [sys.executable, "-I", "-c", TK_PICK, kind, str(start or ""), patterns, pano_patterns],
            None,
        )
    if sys.platform == "darwin":
        osa = shutil.which("osascript")
        if osa is None:
            return None
        script = OSA_PICK % ((", ".join(f'"{e}"' for e in exts + pano),) * 2)
        return lambda kind, start: (
            [osa, "-e", script, kind, *([str(start)] if start else [])],
            None,
        )
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    if zenity := shutil.which("zenity"):
        globs, pano_globs = (
            " ".join(f"*.{x}" for e in g for x in (e, e.upper())) for g in (exts, pano)
        )

        def zenity_args(kind: str, start: Path | None) -> tuple[list[str], int | None]:
            args = [zenity, "--file-selection"]
            args += (
                ["--directory", "--title=Choose a folder"]
                if kind == "folder"
                else [
                    "--title=Choose a video",
                    f"--file-filter=Videos | {globs}",
                    f"--file-filter=Insta360 360 video | {pano_globs}",
                    "--file-filter=All files | *",
                ]
            )
            if start:
                args.append(f"--filename={start}{os.sep}")
            return args, 1

        return zenity_args
    if kdialog := shutil.which("kdialog"):
        kfilter = "\n".join(
            f"{' '.join(f'*.{e}' for e in x)}|{label}"
            for x, label in ((exts, "Videos"), (pano, "Insta360 360 video"))
        )
        kfilter += "\n*|All files"

        def kdialog_args(kind: str, start: Path | None) -> tuple[list[str], int | None]:
            where = str(start or Path.home())
            if kind == "folder":
                return [kdialog, "--title", "Choose a folder", "--getexistingdirectory", where], 1
            return [kdialog, "--title", "Choose a video", "--getopenfilename", where, kfilter], 1

        return kdialog_args
    return None


class Picker:
    """The one native open dialog allowed at a time."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen[str] | None = None

    def ask(self, kind: str, start: Path | None) -> str | None:
        """The path the user chose, or None if they cancelled; waits as long as the dialog is open.

        Raises:
            HTTPException: 501 without a dialog here, 409 while one is open, 500 if it failed.
        """
        build = path_picker()
        if build is None:
            raise bad("File dialogs aren't available on this machine; type or paste the path", 501)
        if not self.lock.acquire(blocking=False):
            raise bad("a file dialog is already open", 409)
        try:
            args, cancel_code = build(kind, start)
            try:
                self.proc = subprocess.Popen(
                    args,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    env=ai_keys.child_env(),
                )
                out, err = self.proc.communicate()
            except OSError as e:
                raise bad(f"could not open the file dialog: {e.strerror or e}", 500) from e
            code = self.proc.returncode
            if cancel_code is not None and code == cancel_code:
                return None
            if code != 0:
                logger.warning("file dialog exited with %s: %s", code, err.strip()[-500:])
                raise bad("the file dialog failed", 500)
            path = out.rstrip("\r\n")
            if not path:
                return None
            if not Path(path).is_absolute():
                raise bad("the file dialog returned no usable path", 500)
            return os.path.normpath(path)
        finally:
            self.proc = None
            self.lock.release()

    def shutdown(self) -> None:
        if (proc := self.proc) is not None:
            with contextlib.suppress(OSError):
                proc.kill()


def pick_start(start: str) -> Path | None:
    """Where a dialog opens: ``start``'s folder, or its nearest existing ancestor (None if none)."""
    if not start:
        return None
    try:
        p = Path(start)
        if not p.is_absolute():
            return None
        while not p.is_dir():
            if p.parent == p:
                return None
            p = p.parent
    except (OSError, ValueError):  # ValueError: NUL byte in the path
        return None
    return p
