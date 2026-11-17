#!/usr/bin/env python3
"""DINOv2 ViT-B xBD baseline evaluation on an official split."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvbridge.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["eval-xbd", *sys.argv[1:]]))
