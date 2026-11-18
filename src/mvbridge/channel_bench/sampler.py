"""Lock scores first, then sample a blinded review pack, then attach images."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from mvbridge.channel_bench.locks import ANALYSIS_LOCK, AnalysisLock
from mvbridge.channel_bench.schema import BuildingAnalysisRow, imprint_bin


def canonical_score_payload(
    rows: Sequence[BuildingAnalysisRow],
    scores: Mapping[str, float],
) -> bytes:
    """Byte-stable lock payload: uids, scores, grades, and stratum features."""
    records = []
    for row in sorted(rows, key=lambda item: item.building_uid):
        if row.building_uid not in scores:
            raise ValueError(f"missing locked score for {row.building_uid}")
        records.append(
            {
                "building_uid": row.building_uid,
                "event_id": row.event_id,
                "seed_id": row.seed_id,
                "block_id": row.block_id,
                "y": row.y,
                "s": row.s,
                "mismatch": row.s != row.y,
                "score": float(scores[row.building_uid]),
                "visibility_bin": row.visibility_bin,
                "imprint_bin": int(imprint_bin(row.imprint_score)),
                "imprint_score": float(row.imprint_score),
            }
        )
    return json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class ScoreLock:
    sha256: str
    n_rows: int
    design_version: str
    payload: bytes


@dataclass(frozen=True)
class ReviewCase:
    building_uid: str
    stratum: str
    score: float
    visibility_bin: str
    imprint_bin: int
    mismatch: bool
    matched_to: str | None = None
    image_ref: str | None = None


@dataclass
class ReviewManifest:
    lock_sha256: str
    design_version: str
    cases: list[ReviewCase] = field(default_factory=list)
    shortfalls: dict[str, int] = field(default_factory=dict)
    images_attached: bool = False
    notes: list[str] = field(default_factory=list)

    def uids(self) -> list[str]:
        return [case.building_uid for case in self.cases]


class BlindedReviewSampler:
    """Select review cases from a locked score table without opening images."""

    def __init__(self, lock: AnalysisLock = ANALYSIS_LOCK) -> None:
        self.lock_spec = lock
        self._lock: ScoreLock | None = None
        self._rows: list[BuildingAnalysisRow] | None = None
        self._scores: dict[str, float] | None = None

    def lock_scores(
        self,
        rows: Sequence[BuildingAnalysisRow],
        scores: Mapping[str, float],
    ) -> ScoreLock:
        eligible = [row for row in rows if row.eligible]
        if not eligible:
            raise ValueError("no eligible rows to lock")
        missing = [row.building_uid for row in eligible if row.building_uid not in scores]
        if missing:
            raise ValueError(f"scores missing for {len(missing)} buildings")
        extra = set(scores) - {row.building_uid for row in eligible}
        if extra:
            raise ValueError("scores include buildings outside the eligible locked table")
        if any(not np.isfinite(float(scores[row.building_uid])) for row in eligible):
            raise ValueError("locked scores must be finite")
        payload = canonical_score_payload(eligible, scores)
        digest = hashlib.sha256(payload).hexdigest()
        self._lock = ScoreLock(
            sha256=digest,
            n_rows=len(eligible),
            design_version=self.lock_spec.design_version,
            payload=payload,
        )
        self._rows = list(eligible)
        self._scores = {row.building_uid: float(scores[row.building_uid]) for row in eligible}
        return self._lock

    def require_lock(self) -> ScoreLock:
        if self._lock is None or self._rows is None or self._scores is None:
            raise RuntimeError("scores must be locked before sampling or viewing images")
        return self._lock

    def sample(
        self,
        *,
        rng_seed: int = 0,
        expected_lock_sha256: str | None = None,
        quotas: Mapping[str, int] | None = None,
    ) -> ReviewManifest:
        locked = self.require_lock()
        if expected_lock_sha256 is not None and expected_lock_sha256 != locked.sha256:
            raise ValueError("lock hash does not match the recorded score table")
        rows = self._rows or []
        scores = self._scores or {}
        quota = int(self.lock_spec.review_quota_per_stratum)
        planned = {
            "high_score_error": quota,
            "missed_error": quota,
            "matched_condition_correct": quota,
        }
        if quotas is not None:
            planned.update({key: int(value) for key, value in quotas.items()})
        score_values = np.array([scores[row.building_uid] for row in rows], dtype=np.float64)
        high_cut = float(np.quantile(score_values, 1.0 - self.lock_spec.high_score_tail))
        low_cut = float(np.quantile(score_values, self.lock_spec.missed_error_tail))
        high_errors = [
            row
            for row in rows
            if row.s != row.y and scores[row.building_uid] >= high_cut
        ]
        missed_errors = [
            row
            for row in rows
            if row.s != row.y and scores[row.building_uid] <= low_cut
        ]
        correct = [row for row in rows if row.s == row.y]
        rng = np.random.default_rng(int(rng_seed))
        cases: list[ReviewCase] = []
        shortfalls: dict[str, int] = {}
        used: set[str] = set()

        def take(pool: list[BuildingAnalysisRow], k: int) -> list[BuildingAnalysisRow]:
            available = [row for row in pool if row.building_uid not in used]
            if k <= 0 or not available:
                return []
            order = rng.permutation(len(available))
            chosen = [available[i] for i in order[: min(k, len(available))]]
            used.update(row.building_uid for row in chosen)
            return chosen

        high_chosen = take(high_errors, planned["high_score_error"])
        missed_chosen = take(missed_errors, planned["missed_error"])
        shortfalls["high_score_error"] = max(0, planned["high_score_error"] - len(high_chosen))
        shortfalls["missed_error"] = max(0, planned["missed_error"] - len(missed_chosen))

        for row in high_chosen:
            cases.append(self._case(row, "high_score_error", scores[row.building_uid]))
        for row in missed_chosen:
            cases.append(self._case(row, "missed_error", scores[row.building_uid]))

        anchors = high_chosen + missed_chosen
        matched: list[ReviewCase] = []
        for anchor in anchors:
            if len(matched) >= planned["matched_condition_correct"]:
                break
            partners = [
                row
                for row in correct
                if row.building_uid not in used
                and row.visibility_bin == anchor.visibility_bin
                and imprint_bin(row.imprint_score) == imprint_bin(anchor.imprint_score)
            ]
            if not partners:
                continue
            pick = partners[int(rng.integers(0, len(partners)))]
            used.add(pick.building_uid)
            matched.append(
                self._case(
                    pick,
                    "matched_condition_correct",
                    scores[pick.building_uid],
                    matched_to=anchor.building_uid,
                )
            )
        shortfalls["matched_condition_correct"] = max(
            0, planned["matched_condition_correct"] - len(matched)
        )
        cases.extend(matched)
        notes = [
            "Images are not attached at sampling time.",
            "Selection used only the locked score table and pre-specified strata.",
        ]
        return ReviewManifest(
            lock_sha256=locked.sha256,
            design_version=locked.design_version,
            cases=cases,
            shortfalls=shortfalls,
            images_attached=False,
            notes=notes,
        )

    def attach_images(
        self,
        manifest: ReviewManifest,
        image_index: Mapping[str, str],
        *,
        expected_lock_sha256: str,
    ) -> ReviewManifest:
        locked = self.require_lock()
        if expected_lock_sha256 != locked.sha256 or manifest.lock_sha256 != locked.sha256:
            raise ValueError("refusing to attach images: lock hash mismatch")
        if manifest.images_attached:
            raise ValueError("images were already attached to this manifest")
        attached: list[ReviewCase] = []
        missing: list[str] = []
        for case in manifest.cases:
            ref = image_index.get(case.building_uid)
            if ref is None:
                missing.append(case.building_uid)
                continue
            attached.append(
                ReviewCase(
                    building_uid=case.building_uid,
                    stratum=case.stratum,
                    score=case.score,
                    visibility_bin=case.visibility_bin,
                    imprint_bin=case.imprint_bin,
                    mismatch=case.mismatch,
                    matched_to=case.matched_to,
                    image_ref=ref,
                )
            )
        if missing:
            raise ValueError(f"image refs missing for {len(missing)} locked review cases")
        return ReviewManifest(
            lock_sha256=manifest.lock_sha256,
            design_version=manifest.design_version,
            cases=attached,
            shortfalls=dict(manifest.shortfalls),
            images_attached=True,
            notes=list(manifest.notes) + ["Images attached after lock verification."],
        )

    def _case(
        self,
        row: BuildingAnalysisRow,
        stratum: str,
        score: float,
        matched_to: str | None = None,
    ) -> ReviewCase:
        return ReviewCase(
            building_uid=row.building_uid,
            stratum=stratum,
            score=float(score),
            visibility_bin=row.visibility_bin,
            imprint_bin=int(imprint_bin(row.imprint_score)),
            mismatch=row.s != row.y,
            matched_to=matched_to,
            image_ref=None,
        )
