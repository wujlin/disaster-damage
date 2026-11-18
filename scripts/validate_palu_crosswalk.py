#!/usr/bin/env python3
"""Validate the Palu mapping ledger and write the Jlin decision packet."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mvbridge.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["palu-crosswalk", *sys.argv[1:]]))
