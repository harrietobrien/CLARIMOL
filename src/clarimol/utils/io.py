"""I/O + Logging Utilities"""

from __future__ import annotations
import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a clean format"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )