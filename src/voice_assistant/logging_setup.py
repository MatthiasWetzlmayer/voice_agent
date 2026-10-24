"""Lesbares Logging: zeigt, wer gerade mit wem spricht."""

from __future__ import annotations

import logging
import sys

# Logger-Name -> (Anzeige-Label, ANSI-Farbe)
ROLES: dict[str, tuple[str, str]] = {
    "va.user": ("NUTZER", "\033[1;32m"),
    "va.voice": ("VOICE-AGENT", "\033[1;36m"),
    "va.task": ("TASK-AGENT", "\033[1;35m"),
    "va.notify": ("BENACHRICHTIGUNG", "\033[1;33m"),
    "va.audio": ("AUDIO", "\033[0;34m"),
    "va.system": ("SYSTEM", "\033[0;37m"),
}
_RESET = "\033[0m"


class RoleFormatter(logging.Formatter):
    def __init__(self, color: bool) -> None:
        super().__init__()
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        label, color = ROLES.get(record.name, (record.name, ""))
        time = self.formatTime(record, "%H:%M:%S")
        text = record.getMessage()
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        if self.color and color:
            return f"{time} {color}{label:<16}{_RESET} {text}"
        return f"{time} {label:<16} {text}"


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(RoleFormatter(color=sys.stdout.isatty()))
    root = logging.getLogger("va")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False
    # Bibliotheken leiser stellen.
    logging.getLogger("httpx").setLevel(logging.WARNING)
