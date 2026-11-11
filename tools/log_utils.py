from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str,
    log_file: Optional[str | Path] = None,
    console_level: int = logging.INFO,
    quiet: bool = False,
) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    if quiet:
        console_level = logging.ERROR

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    return logger

def important(msg: str) -> None:
    print(msg, file=sys.stdout)

def important_lines(lines: list[str]) -> None:
    for line in lines:
        print(line, file=sys.stdout)
