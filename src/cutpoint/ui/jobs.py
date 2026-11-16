"""Child ``cutpoint`` jobs of ``cutpoint ui`` and their progress."""

from __future__ import annotations

import contextlib
import json
import math
import re
import secrets
import subprocess
import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from cutpoint import jsonout, procs
from cutpoint.ui.common import (
    BUSY_TEXT,
    MAX_JOBS,
    MAX_LOG_LINES,
    PROGRESS_LINE,
    STEP_STATES,
    bad,
    logger,
    spawn,
)


# --------------------------------------------------------------------------- jobs
def parse_progress(line: str) -> dict[str, Any] | None:
    """The step update a ``PROGRESS {...}`` output line carries, or None for any other line.

    A line that starts like one but is not a valid update (bad JSON, missing or wrong fields) is
    None as well, so it stays in the log as text.
    """
    m = PROGRESS_LINE.fullmatch(line.strip())
    if not m:
        return None
    try:
        raw = json.loads(m.group(1))
    except (ValueError, RecursionError):  # RecursionError: nested too deeply
        return None
    if not isinstance(raw, dict):
        return None
    step, index, label, state = (raw.get(k) for k in ("step", "index", "label", "state"))
    total, pct, detail = raw.get("total"), raw.get("pct"), raw.get("detail")
    if (
        not isinstance(step, str)
        or not step
        or not isinstance(index, int)
        or isinstance(index, bool)
        or state not in STEP_STATES
    ):
        return None
    number = isinstance(pct, (int, float)) and not isinstance(pct, bool) and math.isfinite(pct)
    return {
        "step": step,
        "index": index,
        "total": total if isinstance(total, int) and not isinstance(total, bool) else None,
        "label": label if isinstance(label, str) and label else step,
        "state": state,
        "pct": min(100.0, max(0.0, float(pct))) if number else None,
        "detail": detail if isinstance(detail, str) and detail else None,
    }


def parse_result(line: str) -> dict[str, Any] | None:
    """The ``--json`` envelope (:mod:`cutpoint.jsonout`) an output line carries, else None.

    The envelope is one JSON object on stdout; jobs merge stdout into their output."""
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        raw = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != jsonout.SCHEMA
        or not isinstance(raw.get("ok"), bool)
        or not isinstance(raw.get("outputs"), dict)
    ):
        return None
    return raw


class Job:
    """One child process and what it said."""

    def __init__(
        self,
        project_id: str,
        action: str,
        plan: list[dict[str, str]] | None = None,
        cwd: Path | None = None,
    ) -> None:
        self.id = secrets.token_hex(6)
        self.project_id = project_id
        self.action = action
        self.plan = plan  # the steps the command will report (``{step, label}``), if known
        self.cwd = cwd  # the folder its commands run in (the projects folder)
        self.result: dict[str, Any] | None = None  # its command's ``--json`` envelope
        self.error_code: str | None = None  # the envelope's ``error.code`` of a failed job
        self.status = "running"
        self.logs: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self.logged = 0  # lines ever added to ``logs`` (it drops the oldest past its size)
        # Latest update per step (``parse_progress``), keyed by step: the CLI's PROGRESS lines.
        self.steps: dict[str, dict[str, Any]] = {}
        self.steps_total: int | None = None
        self.error: str | None = None
        self.started = datetime.now().isoformat(timespec="seconds")
        self.proc: subprocess.Popen[str] | None = None
        self.cancelled = False
        self.done = threading.Event()

    def add_output(self, line: str) -> None:
        """One output line: a step update goes to the steps, the ``--json`` envelope to
        ``result``, any other text to the log."""
        step = parse_progress(line)
        if step is None and (result := parse_result(line)) is not None:
            self.result = result
            return
        if step is None:
            if line.strip():
                self.logs.append(line.rstrip())
                self.logged += 1
            return
        total = step.pop("total")
        if total is not None:
            self.steps_total = total
        self.steps[step["step"]] = step  # replaced, never changed in place: ``view`` copies it

    def note(self, level: str, message: str) -> None:
        """A log line of the server's own, in the CLI's ``HH:MM:SS LEVEL message`` form."""
        self.logs.append(f"{datetime.now():%H:%M:%S} {level} {message}")
        self.logged += 1

    def drop_lines(self, text: str) -> None:
        """Take the log lines holding ``text`` out of the log."""
        self.logs = deque((ln for ln in self.logs if text not in ln), maxlen=MAX_LOG_LINES)

    def interrupt_steps(self) -> None:
        """The job failed or was stopped: a step still running will not finish."""
        for name, step in list(self.steps.items()):
            if step["state"] == "running":
                self.steps[name] = {**step, "state": "interrupted"}

    def view(self, logs: bool = True) -> dict[str, Any]:
        """The job as the page reads it; ``logs`` false leaves its log out."""
        # Ordered by the CLI's step index; a step sent again later keeps its place.
        steps = sorted(self.steps.values(), key=lambda s: s["index"])
        out: dict[str, Any] = {
            "id": self.id,
            "project_id": self.project_id,
            "action": self.action,
            # Cancelled but the watcher has not seen the exit yet (Windows can lag a poll or two).
            "status": "cancelling" if self.cancelled and self.status == "running" else self.status,
            **({"logs": list(self.logs)} if logs else {}),
            "progress": {"total": self.steps_total, "steps": steps},
            "started": self.started,
        }
        if self.plan:
            out["progress"]["plan"] = self.plan
        if self.error:
            out["error"] = self.error
        if self.error_code:
            out["error_code"] = self.error_code
        # clips, extend or a resumed export (#109): finished clips of a stopped export it kept
        taken = (self.result or {}).get("outputs", {}).get("taken_over")
        if isinstance(taken, int) and not isinstance(taken, bool) and taken > 0:
            out["taken_over"] = taken
        return out


class Jobs:
    """At most one running job serverwide."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}

    def start(
        self,
        project_id: str,
        action: str,
        args: list[str],
        on_success: Callable[[], None] | None = None,
        check: Callable[[], None] | None = None,
        plan: list[dict[str, str]] | None = None,
        before: Callable[[Job], list[str] | None] | None = None,
        *,
        cwd: Path | None = None,
        on_result: Callable[[Job], None] | None = None,
        on_end: Callable[[], None] | None = None,
        on_refused: Callable[[], None] | None = None,
    ) -> Job:
        """Start ``args`` as a job; ``check`` runs under the lock first and may refuse it.
        ``plan``: the steps it will report, shown as waiting before their first update.
        ``before``: runs in the job's thread first and returns a command to run ahead of
        ``args``, or None; that command failing (or ``before`` raising) only warns, then
        ``args`` runs. Cancel stops both. ``cwd``: the folder the commands run in.
        ``on_result``: runs after ``on_success``, given the job (its ``--json`` envelope in
        ``result``). ``on_end``: runs when the job ended, however it did (a failure is only
        logged).
        ``on_refused``: runs under the lock when the command could not be started, to put back
        what ``check`` changed."""
        with self.lock:
            if any(j.status == "running" for j in self.jobs.values()):
                raise bad("another job is already running", 409)
            if check:
                check()
            job = Job(project_id, action, plan, cwd)
            if before is None:
                try:
                    job.proc = self._popen(job, args)
                except OSError as err:
                    if on_refused:
                        with contextlib.suppress(OSError):
                            on_refused()
                    raise bad(f"could not start job: {err}", 500) from err
            for old in [k for k, j in self.jobs.items() if j.status != "running"][
                : max(0, len(self.jobs) + 1 - MAX_JOBS)
            ]:
                del self.jobs[old]
            self.jobs[job.id] = job
        threading.Thread(
            target=self._watch, args=(job, args, on_success, before, on_result, on_end), daemon=True
        ).start()
        return job

    @staticmethod
    def _popen(job: Job, args: list[str]) -> subprocess.Popen[str]:
        return spawn(args) if job.cwd is None else spawn(args, cwd=job.cwd)

    def _spawn(self, job: Job, args: list[str]) -> bool:
        """Start ``args`` as the job's process, unless it was cancelled; False if it was."""
        with self.lock:  # a Cancel either sees this process or stops the job before it starts
            if job.cancelled:
                return False
            job.proc = self._popen(job, args)
        return True

    @staticmethod
    def _follow(job: Job) -> int:
        """Read the job's process output into the job until it exits; its exit code."""
        proc = job.proc
        assert proc is not None and proc.stdout is not None
        for line in proc.stdout:
            try:
                job.add_output(line)
            except Exception:  # one odd line must not end the watcher: keep it as text
                logger.exception("job %s could not read an output line", job.id)
                job.logs.append(line.rstrip())
                job.logged += 1
        return proc.wait()

    def _first(self, job: Job, before: Callable[[Job], list[str] | None]) -> None:
        """Run the command ``before`` asks for ahead of the job's own; a failure only warns."""
        try:
            first = before(job)
        except Exception as err:
            logger.exception("job %s: the step before it failed", job.id)
            job.note("WARNING", f"skipped the step before {job.action}: {err}")
            return
        if first is None:
            return
        try:
            if not self._spawn(job, first):
                return
        except OSError as err:
            job.note("WARNING", f"could not start the step before {job.action}: {err}")
            return
        start = job.logged
        code = self._follow(job)
        with self.lock:
            job.proc = None  # exited: a Cancel from now on has nothing of it to stop
        if code != 0 and not job.cancelled:
            job.interrupt_steps()
            job.note("WARNING", f"the step before {job.action} failed (exit code {code})")
        elif code == 0 and job.logged > start:
            # e.g. 360 stitch exits 0 when the SDK failed on a recording: it logs an ERROR
            said = list(job.logs)[-(job.logged - start) :]
            if any(re.search(r"\bERROR\s", ln) for ln in said):
                job.note(
                    "WARNING",
                    f"the step before {job.action} reported errors: those recordings export "
                    "from the 360 preview",
                )

    def _watch(
        self,
        job: Job,
        args: list[str],
        on_success: Callable[[], None] | None,
        before: Callable[[Job], list[str] | None] | None = None,
        on_result: Callable[[Job], None] | None = None,
        on_end: Callable[[], None] | None = None,
    ) -> None:
        try:
            mark = 0  # the job's own error is an ERROR line of its own command
            if before is not None:
                self._first(job, before)
                mark = job.logged
                try:
                    started = self._spawn(job, args)
                except OSError as err:
                    job.error, job.status = f"could not start job: {err}", "failed"
                    return
                if not started:
                    job.interrupt_steps()
                    job.status = "cancelled"
                    return
            code = self._follow(job)
            if job.cancelled or code != 0:
                job.interrupt_steps()
            if job.cancelled:
                job.status = "cancelled"
            elif code != 0 and self._busy(job):
                job.status = "failed"
            elif code != 0:
                own = list(job.logs)[-(job.logged - mark) :] if job.logged > mark else []
                last = next(
                    (m.group(1) for ln in reversed(own) if (m := re.search(r"ERROR\s+(.*)", ln))),
                    None,
                )
                job.error, job.status = last or f"exit code {code}", "failed"
            else:
                try:
                    if on_success:
                        on_success()
                    if on_result:
                        on_result(job)
                    job.status = "completed"
                except HTTPException as err:  # a refused write: its message is the error
                    job.error, job.status = err.detail, "failed"
                except Exception as err:  # surfaced to the user as the job error
                    logger.exception("job %s post-processing failed", job.id)
                    job.error, job.status = str(err), "failed"
        except Exception as err:  # the watcher itself broke: never leave the job "running"
            logger.exception("job %s: watching it failed", job.id)
            job.error = job.error or f"the server stopped following the job: {err}"
            with self.lock:
                proc, job.proc = job.proc, None
            if proc is not None and proc.poll() is None:  # unread, it would block on its output
                with contextlib.suppress(Exception):
                    procs.kill_tree(proc)
        finally:
            if job.status == "running":  # also after a BaseException: the next job may start
                job.interrupt_steps()
                job.error = job.error or "the server stopped following the job"
                job.status = "failed"
            if on_end:
                try:
                    on_end()
                except Exception:
                    logger.exception("job %s: the cleanup after it failed", job.id)
            job.done.set()

    @staticmethod
    def _busy(job: Job) -> bool:
        """The job's command stopped with ``folder_busy`` (another run holds the project's run
        lock or a clips folder's): the error is :data:`BUSY_TEXT`, and the CLI's ERROR line,
        which tells the user to delete the lock file, leaves the log."""
        err = (job.result or {}).get("error")
        code = err.get("code") if isinstance(err, dict) else None
        if isinstance(code, str):
            job.error_code = code
        if code != jsonout.FOLDER_BUSY:
            return False
        if isinstance(err.get("message"), str) and err["message"]:
            job.drop_lines(err["message"])
        job.note("ERROR", BUSY_TEXT)
        job.error = BUSY_TEXT
        return True

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise bad("unknown job", 404)
        return job

    def cancel(self, job_id: str) -> Job:
        """Stop the job in the background and return at once; it reads "cancelling" until it exits."""
        job = self.get(job_id)
        with self.lock:  # one killer per job, however often Cancel is pressed
            start = job.status == "running" and not job.cancelled
            if start:
                job.cancelled = True  # also between two steps: the next one never starts
            proc = job.proc
        if start and proc is not None:
            threading.Thread(target=procs.kill_tree, args=(proc,), daemon=True).start()
        return job

    def shutdown(self) -> None:
        # One by one: at most one job runs at a time (``start``), so this waits JOB_STOP_S at most.
        for job in list(self.jobs.values()):
            if job.status != "running":
                continue
            with self.lock:  # as in ``cancel``: did a killer already send the TERM?
                stopping, job.cancelled = job.cancelled, True
                proc = job.proc
            if proc is None:  # between two steps: the next one never starts
                continue
            if not stopping:
                procs.kill_tree(proc)
            elif not job.done.wait(procs.JOB_STOP_S):  # a second TERM could cut short its cleanup
                procs.kill_tree(proc, term=False)
