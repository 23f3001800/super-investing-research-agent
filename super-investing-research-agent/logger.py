"""Central logging module for the Super Investing Research Agent.

Provides console formatting, rotating file logging, and an in-memory queue
for streaming events to web interfaces (SSE / WebSockets).
"""

import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

# Global in-memory broadcast subscribers for SSE
_subscribers: List[Callable[[dict], None]] = []
_subscribers_lock = threading.Lock()


def add_log_subscriber(callback: Callable[[dict], None]) -> None:
    """Register a callback for streaming log events."""
    with _subscribers_lock:
        if callback not in _subscribers:
            _subscribers.append(callback)


def remove_log_subscriber(callback: Callable[[dict], None]) -> None:
    """Unregister a log subscriber."""
    with _subscribers_lock:
        if callback in _subscribers:
            _subscribers.remove(callback)


class SSEBroadcastHandler(logging.Handler):
    """Custom logging handler that dispatches events to registered SSE subscribers."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            event = {
                "timestamp": datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3],
                "level": record.levelname,
                "name": record.name,
                "node": getattr(record, "node", None),
                "message": record.getMessage(),
                "formatted": msg,
            }
            with _subscribers_lock:
                for sub in list(_subscribers):
                    try:
                        sub(event)
                    except Exception:
                        pass
        except Exception:
            self.handleError(record)


def setup_logger(
    name: str = "research_agent",
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return the root package logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    # 1. Console handler with clean formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # 2. File handler
    if log_file is None:
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "research_agent.log"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # 3. SSE broadcast handler
    sse_handler = SSEBroadcastHandler()
    sse_handler.setLevel(logging.INFO)
    logger.addHandler(sse_handler)

    logger.propagate = False
    return logger


# Default logger instance
agent_logger = setup_logger()
