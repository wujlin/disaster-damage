"""Multi-view bridge between SAGIN place groups and xBD building damage.

Week-1 code separates two pathways:

* identity/geometry, for DINO-style multi-view positives
* damage state, for labeled pre/post building damage

Simulated UAV imagery is a viewpoint bridge. It is not damage evidence.
"""

__version__ = "0.1.0"
