"""CPU entry point for the synthetic Group-3 fixture generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mvbridge.priors.features import compute_group3_features
from mvbridge.priors.fixtures import build_default_fixture, records_to_dicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write synthetic Group-3 fixtures. Never used for official scoring."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON path. Prints to stdout when omitted.",
    )
    args = parser.parse_args(argv)
    fixture = build_default_fixture()
    features = compute_group3_features(
        [row for row in fixture.records if row.event_id == fixture.records[0].event_id],
        elevation=fixture.elevation,
        water=fixture.water,
        coverage=fixture.coverage,
        frame=fixture.frame,
    )
    payload = {
        "synthetic": True,
        "fixture": fixture.name,
        "notes": fixture.notes,
        "records": records_to_dicts(fixture.records),
        "features": [row.as_dict() for row in features],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is None:
        print(text)
    else:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
