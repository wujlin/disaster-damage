"""Group-3 wide-area neighborhood priors and public geographic layers.

This package builds leave-one-building-out neighborhood features from
satellite-side alerts and samples elevation and distance-to-water layers.
It does not run official scoring or retraining, and it does not read the
current disaster's ground labels into neighborhood statistics.

Michael's real satellite alert column is an official-export placeholder
until that export exists. Palu geographic layers may be prepared, but that
does not announce that cross-hazard transfer has been established.
"""

from mvbridge.priors.coordinates import MetricFrame
from mvbridge.priors.coverage import AxisAlignedBox, CoverageMap
from mvbridge.priors.features import (
    ANNULUS_INNER_M,
    ANNULUS_OUTER_M,
    NEAR_RADIUS_M,
    compute_group3_features,
)
from mvbridge.priors.layers import ElevationGrid, WaterGeometries
from mvbridge.priors.neighborhood import NeighborhoodIndex
from mvbridge.priors.provenance import LayerLedger
from mvbridge.priors.schema import (
    OFFICIAL_EXPORT_PLACEHOLDER,
    BuildingRecord,
    Group3FeatureRow,
)

__all__ = [
    "ANNULUS_INNER_M",
    "ANNULUS_OUTER_M",
    "AxisAlignedBox",
    "BuildingRecord",
    "CoverageMap",
    "ElevationGrid",
    "Group3FeatureRow",
    "LayerLedger",
    "MetricFrame",
    "NEAR_RADIUS_M",
    "NeighborhoodIndex",
    "OFFICIAL_EXPORT_PLACEHOLDER",
    "WaterGeometries",
    "compute_group3_features",
]
