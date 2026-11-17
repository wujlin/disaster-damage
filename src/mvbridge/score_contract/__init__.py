"""Score contract: protocol reading, split inventory, and a decoupled score adapter.

This package implements the feat/score-contract worktree. It depends on the
shipped protocol and fold list. It does not run official Michael scoring, does
not load official checkpoints, and does not change the official training recipe.
"""

from mvbridge.score_contract.adapter import (
    ALIGNMENT_KEY,
    InferenceRow,
    InferenceTable,
    align_inference_and_labels,
    build_inference_table,
    class_index,
    entropy,
    normalize_class_name,
    one_minus_max_p,
    predict_satellite_class,
    score_probabilities,
)
from mvbridge.score_contract.eligibility import (
    COVERAGE_CLASSES,
    Eligibility,
    assign_coverage_class,
    four_class_eligibility,
    is_four_class_label,
    sufficient_satellite_evidence,
)
from mvbridge.score_contract.inventory import (
    FoldHistoricalInventory,
    HistoricalPair,
    inventory_historical_pairs_by_fold,
    reporting_model_defines_u,
    reporting_model_version,
)
from mvbridge.score_contract.protocol import (
    OFFICIAL_CLASS_ORDER,
    P4_BASELINES,
    P5_BASELINES,
    ProtocolContract,
    authorized_artifact_inventory,
    load_protocol_contract,
    p4_comparison_baselines,
    p5_comparison_baselines,
)

__all__ = [
    "ALIGNMENT_KEY",
    "COVERAGE_CLASSES",
    "Eligibility",
    "FoldHistoricalInventory",
    "HistoricalPair",
    "InferenceRow",
    "InferenceTable",
    "OFFICIAL_CLASS_ORDER",
    "P4_BASELINES",
    "P5_BASELINES",
    "ProtocolContract",
    "align_inference_and_labels",
    "assign_coverage_class",
    "authorized_artifact_inventory",
    "build_inference_table",
    "class_index",
    "entropy",
    "four_class_eligibility",
    "inventory_historical_pairs_by_fold",
    "is_four_class_label",
    "load_protocol_contract",
    "normalize_class_name",
    "one_minus_max_p",
    "p4_comparison_baselines",
    "p5_comparison_baselines",
    "predict_satellite_class",
    "reporting_model_defines_u",
    "reporting_model_version",
    "score_probabilities",
    "sufficient_satellite_evidence",
]
