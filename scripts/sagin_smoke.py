#!/usr/bin/env python3
"""SAGIN multi-view loader smoke test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvbridge.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["sagin-smoke", *sys.argv[1:]]))
