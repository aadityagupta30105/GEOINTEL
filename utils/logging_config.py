"""
Logging Configuration
=====================
Single source of truth for console output across the GeoIntel platform.

The platform targets operator consoles that may not support UTF-8 (Windows
``cp1252`` code pages, redirected pipes, CI log collectors). All emitted
records are therefore restricted to ASCII and use bracketed status markers
rather than pictographic characters.

Status marker convention
------------------------
``[OK]``     Operation completed as expected.
``[WARN]``   Degraded but recoverable; a fallback path was taken.
``[ERROR]``  Operation failed; caller must handle the outcome.
``[INFO]``   Progress reporting.
"""

from __future__ import annotations

import logging
import sys
from typing import Final, TextIO

__all__ = [
    "configure_logging",
    "get_logger",
    "OK",
    "WARN",
    "ERROR",
    "ARROW",
    "RULE_WIDTH",
    "rule",
    "section",
]

OK: Final[str] = "[OK]"
WARN: Final[str] = "[WARN]"
ERROR: Final[str] = "[ERROR]"
ARROW: Final[str] = "->"

RULE_WIDTH: Final[int] = 68

_LOG_FORMAT: Final[str] = "%(asctime)s  %(levelname)-7s  %(name)-24s  %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S"
_ROOT_LOGGER_NAME: Final[str] = "geointel"

_configured: bool = False


class _AsciiFormatter(logging.Formatter):
    """Formatter that guarantees ASCII-safe output on every console encoding.

    Non-ASCII characters that reach the logger from third-party libraries or
    upstream data are replaced with their backslash escapes rather than
    raising ``UnicodeEncodeError`` on a legacy code page.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` and strip any non-ASCII code points.

        Parameters
        ----------
        record : logging.LogRecord
            The record to render.

        Returns
        -------
        str
            An ASCII-only rendering of the record.
        """
        rendered = super().format(record)
        return rendered.encode("ascii", errors="backslashreplace").decode("ascii")


def configure_logging(
    level: int = logging.INFO,
    stream: TextIO | None = None,
    force: bool = False,
) -> logging.Logger:
    """Install the platform console handler exactly once.

    Repeated calls are no-ops unless ``force`` is set, which keeps the
    Streamlit execution model (module re-import on every rerun) from stacking
    duplicate handlers.

    Parameters
    ----------
    level : int, optional
        Minimum severity emitted to the console. Default ``logging.INFO``.
    stream : TextIO or None, optional
        Destination stream. Defaults to ``sys.stdout``.
    force : bool, optional
        Reinstall the handler even if logging was already configured.

    Returns
    -------
    logging.Logger
        The configured ``geointel`` root logger.
    """
    global _configured

    logger = logging.getLogger(_ROOT_LOGGER_NAME)

    if _configured and not force:
        return logger

    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        existing.close()

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(_AsciiFormatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    handler.setLevel(level)

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    _configured = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child of the platform logger.

    Parameters
    ----------
    name : str
        Module name, conventionally ``__name__``. A leading ``geointel.``
        prefix is added when absent.

    Returns
    -------
    logging.Logger
        Logger bound to the platform handler hierarchy.
    """
    configure_logging()
    if name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def rule(char: str = "-", width: int = RULE_WIDTH) -> str:
    """Build a horizontal console rule.

    Parameters
    ----------
    char : str, optional
        Single ASCII character used to draw the rule.
    width : int, optional
        Total rule width in characters.

    Returns
    -------
    str
        The rendered rule.
    """
    return char * width


def section(title: str, char: str = "-", width: int = RULE_WIDTH) -> str:
    """Build a titled console section banner.

    Parameters
    ----------
    title : str
        Section heading, rendered upper case.
    char : str, optional
        Rule character drawn above and below the heading.
    width : int, optional
        Total banner width in characters.

    Returns
    -------
    str
        A three-line banner terminated without a trailing newline.
    """
    bar = rule(char, width)
    return f"{bar}\n{title.upper()}\n{bar}"
