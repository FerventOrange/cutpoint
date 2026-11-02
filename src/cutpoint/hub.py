"""Hugging Face Hub logging: keep the Hub's warnings to one copy on stderr.

huggingface_hub is imported only when a model loads (whisper or open_clip), so every
``cutpoint`` start stays fast.
"""

from __future__ import annotations

import logging

_hub_handler_removed = False


def quiet_hub_handler() -> None:
    """Remove huggingface_hub's own stderr handler (once), which model downloads
    (faster-whisper, open_clip's ``hf_hub_download``) log through.

    It also propagates to Cutpoint's handler, so the Hub's warnings (e.g. "unauthenticated
    requests") came out twice: keep only ours. The handler is added on import, so import it
    first.
    """
    global _hub_handler_removed
    if _hub_handler_removed:
        return
    _hub_handler_removed = True
    try:
        import huggingface_hub.utils.logging  # noqa: F401, PLC0415 - only when a model loads
    except ImportError:
        return
    hf_logger = logging.getLogger("huggingface_hub")
    for handler in list(hf_logger.handlers):
        hf_logger.removeHandler(handler)
