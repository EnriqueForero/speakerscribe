"""Centralized logging setup with loguru.

Design:
    - Console sink with colors and human-readable format (default INFO)
    - Persistent file sink with automatic rotation (always DEBUG)
    - Optional JSON format for machine-parseable logs
    - Idempotent: safe to call multiple times (replaces handlers, no duplication)

Usage:
    >>> from speakerscribe.logging_config import configure_logging, logger
    >>> configure_logging(workspace=Path("/path"), console_level="INFO")
    >>> logger.info("Starting")
    >>> logger.bind(audio_file="x.mp4").success("Processed OK")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from loguru import logger

LogLevel = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

_CONFIGURED = False


def configure_logging(
    workspace: Path | None = None,
    console_level: LogLevel = "INFO",
    file_level: LogLevel = "DEBUG",
    json_format: bool = False,
    rotation_mb: int = 100,
    retention_days: int = 30,
) -> None:
    """Configure loguru with console and (optional) file sinks.

    Idempotent: calling this multiple times reconfigures, never duplicates.

    Args:
        workspace: Path under which `_logs/` is created. If None, no file sink.
        console_level: Minimum level for the console sink.
        file_level: Minimum level for the file sink (typically more verbose).
        json_format: If True, file sink emits one JSON object per line.
        rotation_mb: Rotate the log file every N megabytes.
        retention_days: Keep log files for N days, then delete.
    """
    global _CONFIGURED

    # Clean previous handlers (idempotency)
    logger.remove()

    # ── Console sink ───────────────────────────────────────────────
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=console_format,
        level=console_level,
        colorize=True,
        backtrace=True,
        diagnose=False,  # diagnose=True leaks variable values; off for safety
    )

    # ── Persistent file sink ───────────────────────────────────────
    if workspace is not None:
        log_dir = Path(workspace) / "_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "run_{time:YYYY-MM-DD}.log"

        if json_format:
            logger.add(
                log_file,
                serialize=True,
                level=file_level,
                rotation=f"{rotation_mb} MB",
                retention=f"{retention_days} days",
                encoding="utf-8",
            )
        else:
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            )
            logger.add(
                log_file,
                format=file_format,
                level=file_level,
                rotation=f"{rotation_mb} MB",
                retention=f"{retention_days} days",
                encoding="utf-8",
                enqueue=True,  # thread-safe + non-blocking
            )

    _CONFIGURED = True


def set_log_level(level: LogLevel) -> None:
    """Change the console log level at runtime.

    Useful for enabling DEBUG temporarily without restarting the process.

    Args:
        level: New minimum level for the console sink.
    """
    if not _CONFIGURED:
        configure_logging(console_level=level)
        return

    # loguru does not support changing the level of an existing sink; recreate
    configure_logging(console_level=level)


__all__ = ["configure_logging", "logger", "set_log_level"]
