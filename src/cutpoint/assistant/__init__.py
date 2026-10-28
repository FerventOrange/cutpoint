"""The ``cutpoint ui`` assistant: a chat model that proposes sequence edits it cannot apply itself.

``providers`` talks to the model, ``context`` builds what it sees, ``ops`` checks and applies the
edits it proposes, ``session`` runs one chat turn in a thread and ``store`` keeps the chat and the
history of applied proposals under ``<project>/ui/assistant``.
"""
