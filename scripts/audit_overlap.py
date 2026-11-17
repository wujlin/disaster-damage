#!/usr/bin/env python3
"""Overlap audit. Paths come from the config, not from this script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvbridge.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["audit", *sys.argv[1:]]))
