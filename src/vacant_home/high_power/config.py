from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


DECISION = {
    "relative_high_power_percentile": 0.75,
    "strict_high_power_percentile": 0.90,
    "persistent_sample_median_kw": 1.0,
    "persistent_gt_1kw_fraction": 0.50,
    "persistent_day_count": 21,
    "persistent_daily_q90_kw": 2.0,
    "family_top_fraction": 0.30,
    "human_opposition_top_fraction": 0.35,
    "automatic_or_subtype_top_fraction": 0.40,
    "tier_a_rank_median": 0.85,
    "tier_a_rank_p10": 0.70,
    "tier_b_family_support": 0.65,
    "tier_b_rank_median": 0.70,
    "behavior_cosine_median": 0.95,
    "behavior_cosine_p10": 0.85,
    "branch_x10_median_decline_max": 0.10,
}

SEEDS = (20260926, 20260927, 20260928)
WINDOWS = (14, 21, 28, 42)
PRIORS = (0.06, 0.10, 0.16)
MEDIUM_WEIGHTS = (0.25, 0.50, 0.75)
SCORE_VARIANTS = ("full", "no_pu", "no_subtype", "no_tcn")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def bool_col(series):
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


@dataclass(frozen=True)
class BranchPaths:
    root: Path
    upstream: Path
    source_legacy: Path
    source_word: Path

    @classmethod
    def discover(cls) -> "BranchPaths":
        from ..runtime import workspace
        main = workspace("recognition")
        return cls(
            root=workspace("high_power"),
            upstream=main,
            source_legacy=main / "inputs" / "legacy_labels.xlsx",
            source_word=main / "inputs" / "reference_method.docx",
        )


def load_decision_spec(paths: BranchPaths | None = None) -> str:
    paths = paths or BranchPaths.discover()
    return (Path(__file__).with_name("decision_spec.md")).read_text(encoding="utf-8")


def decision_freeze_payload(paths: BranchPaths | None = None) -> dict:
    paths = paths or BranchPaths.discover()
    spec = Path(__file__).with_name("decision_spec.md")
    return {
        "stage": "HIGH_POWER_VACANCY_DISCOVERY",
        "decision_spec_sha256": sha256(spec),
        "decision": DECISION,
        "seeds": list(SEEDS),
        "windows": list(WINDOWS),
        "priors": list(PRIORS),
        "medium_weights": list(MEDIUM_WEIGHTS),
        "score_variants": list(SCORE_VARIANTS),
        "threshold_interpretation": "pre-registered no-Gold engineering gates",
        "gold_labels_opened": False,
        "point_semantics": "15-minute periodic frozen active-power samples",
    }


def write_decision_freeze(paths: BranchPaths | None = None) -> Path:
    paths = paths or BranchPaths.discover()
    target = paths.root / "high_power_branch_decision_freeze.json"
    target.write_text(json.dumps(decision_freeze_payload(paths), ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def manifest_tree(root: Path) -> list[dict]:
    records = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        records.append({
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return records

