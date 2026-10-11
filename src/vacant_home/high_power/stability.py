from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .config import MEDIUM_WEIGHTS, PRIORS, SCORE_VARIANTS, SEEDS, WINDOWS
from .data import empirical_cdf


def _row_nanmedian(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, float)
    result = np.full(matrix.shape[0], np.nan)
    valid = np.isfinite(matrix).any(axis=1)
    if valid.any():
        result[valid] = np.nanmedian(matrix[valid], axis=1)
    return result


def expected_stability_grid() -> pd.DataFrame:
    rows = [
        {"seed": seed, "window_days": window, "prior": prior, "medium_weight": weight, "score_variant": variant}
        for seed, window, prior, weight, variant in itertools.product(SEEDS, WINDOWS, PRIORS, MEDIUM_WEIGHTS, SCORE_VARIANTS)
    ]
    return pd.DataFrame(rows)


def score_variant(frame: pd.DataFrame, variant: str) -> np.ndarray:
    human = pd.to_numeric(frame.family_human_absence_support, errors="coerce").to_numpy(float)
    automatic = pd.to_numeric(frame.family_automatic_structure_support, errors="coerce").to_numpy(float)
    representation = pd.to_numeric(frame.family_learned_representation_support, errors="coerce").to_numpy(float)
    if variant == "no_pu":
        representation = frame[["equipment_only_similarity_percentile", "equipment_intermit_similarity_percentile"]].median(axis=1, skipna=True).to_numpy(float)
    elif variant == "no_subtype":
        representation = frame[["behavior_only_pu_hp_percentile", "equipment_only_similarity_percentile"]].median(axis=1, skipna=True).to_numpy(float)
    elif variant == "no_tcn":
        return _row_nanmedian(np.column_stack([human, automatic]))
    return _row_nanmedian(np.column_stack([human, automatic, representation]))


def summarize_stability(long_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        long_frame.groupby("meter_id", as_index=False)
        .agg(
            stability_rank_median=("hp_conditional_rank", "median"),
            stability_rank_p10=("hp_conditional_rank", lambda values: float(np.nanquantile(values, .10))),
            stability_rank_p90=("hp_conditional_rank", lambda values: float(np.nanquantile(values, .90))),
            stability_rank_std=("hp_conditional_rank", "std"),
            stability_scenario_count=("hp_conditional_rank", "count"),
            stability_seed_min=("seed", "nunique"),
            stability_window_min=("window_days", "nunique"),
            stability_prior_min=("prior", "nunique"),
            stability_medium_weight_min=("medium_weight", "nunique"),
            stability_variant_count=("score_variant", "nunique"),
        )
    )


def build_stability_from_variants(variant_frames: list[pd.DataFrame], reference_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for variant_frame in variant_frames:
        reference = variant_frame.meter_id.astype(str).isin(reference_ids) & variant_frame.relative_high_power_flag.fillna(False) & variant_frame.persistent_high_load.fillna(False)
        if reference.sum() < 20:
            reference = variant_frame.meter_id.astype(str).isin(reference_ids) & variant_frame.relative_high_power_flag.fillna(False)
        for score_name in SCORE_VARIANTS:
            score = score_variant(variant_frame, score_name)
            rank = empirical_cdf(score, score[reference.to_numpy()])
            part = pd.DataFrame({
                "meter_id": variant_frame.meter_id.astype(str),
                "seed": int(variant_frame.attrs["seed"]),
                "window_days": int(variant_frame.attrs["window_days"]),
                "prior": float(variant_frame.attrs["prior"]),
                "medium_weight": float(variant_frame.attrs["medium_weight"]),
                "score_variant": score_name,
                "hp_branch_score": score,
                "hp_conditional_rank": rank,
            })
            rows.append(part)
    long = pd.concat(rows, ignore_index=True)
    return long, summarize_stability(long)
