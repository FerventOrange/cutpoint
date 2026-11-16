"""The ``cutpoint.ui`` package: its page files and its one logger."""

from __future__ import annotations

import sys
from pathlib import Path

import cutpoint
from cutpoint import ui


def test_web_dir_is_the_web_folder_beside_the_ui_package():
    web = Path(cutpoint.__file__).parent / "web"
    assert web == ui.WEB_DIR
    assert (ui.WEB_DIR / "index.html").exists()


def test_every_ui_module_logs_through_the_one_cutpoint_ui_logger():
    loggers = {
        name: mod.logger
        for name, mod in list(sys.modules.items())
        if name.startswith("cutpoint.ui.") and hasattr(mod, "logger")
    }
    assert loggers, "no cutpoint.ui submodule loaded"
    assert ui.logger.name == "cutpoint.ui"
    assert all(logger is ui.logger for logger in loggers.values()), loggers
