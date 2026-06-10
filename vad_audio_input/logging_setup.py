"""Logging configuration: Bangkok-timezone timestamps and the required formatter."""

import logging
import sys
from datetime import (datetime, timezone, timedelta)
from pathlib import Path
from typing import (Final, Optional)

_BANGKOK_TZ: Final[timezone] = timezone(timedelta(hours=7))
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
)


class _BangkokFormatter(logging.Formatter):
    """Formatter that renders timestamps in the Bangkok timezone."""

    def formatTime(  # noqa: N802 (overriding stdlib signature)
        self,
        record: logging.LogRecord,
        datefmt: Optional[str] = None,
    ) -> str:
        """Render the record's creation time in the Bangkok timezone."""
        dt: datetime = datetime.fromtimestamp(record.created, tz=_BANGKOK_TZ)
        return dt.strftime(datefmt or _DATE_FORMAT)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure and return the package root logger.

    Installs a stderr handler and, when ``log_file`` is given, an additional file
    handler (creating parent directories as needed). Existing handlers on the
    package logger are cleared, so this is safe to call once at startup.

    Args:
        level: Logging level for the package logger.
        log_file: Optional path to also write logs to.

    Returns:
        The configured ``vad_audio_input`` logger.
    """
    logger: logging.Logger = logging.getLogger("vad_audio_input")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter: _BangkokFormatter = _BangkokFormatter(_LOG_FORMAT, _DATE_FORMAT)

    stream_handler: logging.StreamHandler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler: logging.FileHandler = logging.FileHandler(
            log_file, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
