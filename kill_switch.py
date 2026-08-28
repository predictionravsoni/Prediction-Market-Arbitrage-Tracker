"""
Emergency kill switch.

Two independent triggers, either of which halts the bot and cancels all resting orders:

1. File-based: create a file (default name "KILL_SWITCH") in the working directory.
   This lets you halt the bot from another terminal / script without touching the
   running process — e.g. `touch KILL_SWITCH`.
2. Signal-based: Ctrl+C (SIGINT) or SIGTERM (e.g. `kill <pid>`, or systemd stop).

Both set the same threading.Event, which the main loop checks every iteration and
before placing any order.
"""

import logging
import os
import signal
import threading

logger = logging.getLogger("kill_switch")


class KillSwitch:
    def __init__(self, sentinel_file: str):
        self.sentinel_file = sentinel_file
        self._event = threading.Event()
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, signum, _frame):
        logger.warning("Received signal %s — triggering kill switch.", signum)
        self._event.set()

    def is_triggered(self) -> bool:
        if self._event.is_set():
            return True
        if os.path.exists(self.sentinel_file):
            logger.warning("Kill switch file '%s' detected.", self.sentinel_file)
            self._event.set()
            return True
        return False

    def trigger(self, reason: str = "manual"):
        logger.warning("Kill switch triggered programmatically: %s", reason)
        self._event.set()
