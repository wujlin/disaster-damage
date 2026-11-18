"""Synthetic Palu-like ground fixtures. Not real Palu microdata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FIXTURE_NOTE = (
    "Synthetic fixture for Palu crosswalk tests. Coordinates are in a local "
    "metric plane and grades are invented. This is not Paulik 2019 microdata "
    "and not a secondary compilation of Palu field records."
)


def build_synthetic_ground_fixture() -> dict[str, Any]:
    """Build a small deterministic fixture covering the check cases."""
    # Four 20 m x 20 m footprints along x, plus one overlapping pair for IoU ambiguity.
    footprints = [
        {"id": "fp-a", "box": [0.0, 0.0, 20.0, 20.0]},
        {"id": "fp-b", "box": [40.0, 0.0, 60.0, 20.0]},
        {"id": "fp-c", "box": [80.0, 0.0, 100.0, 20.0]},
        {"id": "fp-d", "box": [120.0, 0.0, 140.0, 20.0]},
        # fp-overlap-1 and fp-overlap-2 share most of their area (IoU > 0.3).
        {"id": "fp-overlap-1", "box": [200.0, 0.0, 230.0, 30.0]},
        {"id": "fp-overlap-2", "box": [210.0, 0.0, 240.0, 30.0]},
        {"id": "fp-far", "box": [400.0, 400.0, 420.0, 420.0]},
    ]
    records = [
        {
            "id": "g-ds0",
            "source_class": "DS0",
            "x": 10.0,
            "y": 10.0,
            "survey_date": "2018-10-05",
            "imagery_date": "2018-09-30",
            "phase": "field",
        },
        {
            "id": "g-ds1",
            "source_class": "DS1",
            "x": 50.0,
            "y": 10.0,
            "survey_date": "2018-10-05",
            "imagery_date": "2018-09-30",
            "phase": "field",
        },
        {
            "id": "g-ds2",
            "source_class": "DS2",
            "x": 90.0,
            "y": 10.0,
            "survey_date": "2018-10-05",
            "imagery_date": "2018-09-30",
            "phase": "rapid",
        },
        {
            "id": "g-ds3",
            "source_class": "DS3",
            "x": 130.0,
            "y": 10.0,
            "survey_date": None,
            "imagery_date": None,
            "phase": None,
        },
        # Second point on fp-a with a different class: multipoint inconsistency.
        {
            "id": "g-ds1-on-a",
            "source_class": "DS1",
            "x": 12.0,
            "y": 8.0,
            "survey_date": "2018-09-10",
            "imagery_date": "2018-09-20",
            "phase": "field",
        },
        # Point more than 30 m from every footprint.
        {
            "id": "g-far",
            "source_class": "DS2",
            "x": 300.0,
            "y": 10.0,
            "survey_date": "2018-10-05",
            "imagery_date": "2018-09-30",
            "phase": "field",
        },
        # Polygon that overlaps two footprints at IoU >= 0.3.
        {
            "id": "g-ambiguous-poly",
            "source_class": "DS3",
            "box": [205.0, 0.0, 235.0, 30.0],
            "survey_date": "2018-10-05",
            "imagery_date": "2018-09-30",
            "phase": "field",
        },
        # Unique polygon match onto fp-b (40,0,60,20): identical box, IoU = 1.
        # Class DS0 disagrees with point g-ds1 (DS1) on the same footprint.
        {
            "id": "g-poly-unique",
            "source_class": "DS0",
            "box": [40.0, 0.0, 60.0, 20.0],
            "survey_date": "2018-10-05",
            "imagery_date": "2018-09-30",
            "phase": "field",
        },
    ]
    return {
        "name": "synthetic-palu-crosswalk",
        "crs": "local-meters",
        "note": FIXTURE_NOTE,
        "event_id": "synthetic-event",
        "ground_records": records,
        "footprints": footprints,
    }


def write_synthetic_ground_fixture(path: str | Path) -> Path:
    """Write the synthetic fixture as UTF-8 JSON. No real Palu records."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = build_synthetic_ground_fixture()
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest
