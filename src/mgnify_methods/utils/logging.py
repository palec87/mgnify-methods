"""Logging utilities for mgnify_methods.

Use Logger.get_logger(__name__) to get a configured logger.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union


class Logger:
    """Configurable logging helper with safe defaults."""

    _configured: bool = False
    _default_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    _default_datefmt: str = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def configure(
        cls,
        level: Union[int, str] = "INFO",
        log_file: Optional[str] = None,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
    ) -> None:
        fmt = fmt or cls._default_format
        datefmt = datefmt or cls._default_datefmt

        root = logging.getLogger()
        root.setLevel(cls._parse_level(level))

        if not root.handlers:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
            root.addHandler(stream_handler)

        if log_file:
            cls._add_file_handler(log_file, fmt, datefmt)

        cls._configured = True

    @classmethod
    def get_logger(
        cls,
        name: str,
        level: Optional[Union[int, str]] = None,
        log_file: Optional[str] = None,
    ) -> logging.Logger:
        if not cls._configured:
            cls.configure(level=level or "INFO", log_file=log_file)
        elif log_file:
            cls._add_file_handler(log_file, cls._default_format, cls._default_datefmt)

        logger = logging.getLogger(name)
        if level is not None:
            logger.setLevel(cls._parse_level(level))
        return logger

    @classmethod
    def _add_file_handler(cls, log_file: str, fmt: str, datefmt: str) -> None:
        root = logging.getLogger()
        log_path = Path(log_file).resolve()

        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler):
                if Path(handler.baseFilename).resolve() == log_path:
                    return

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(file_handler)

    @staticmethod
    def _parse_level(level: Union[int, str]) -> int:
        if isinstance(level, int):
            return level

        level_name = str(level).upper()
        if not hasattr(logging, level_name):
            raise ValueError(f"Unknown log level: {level}")
        return getattr(logging, level_name)


def get_logger(
    name: str,
    level: Optional[Union[int, str]] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Convenience wrapper for Logger.get_logger."""
    return Logger.get_logger(name=name, level=level, log_file=log_file)
