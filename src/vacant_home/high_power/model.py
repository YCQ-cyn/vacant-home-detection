from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from .config import BranchPaths, DECISION, SEEDS, key
from .data import empirical_cdf, fit_reference_mask, load_holdout_ids, load_scope, load_unique_legacy


BEHAVIOR_ROBUST_FEATURES = [
    "cross_day_normalized_shape_similarity",
    "daily_trajectory_similarity_sample",
    "sample_state_transition_rate",
    "sample_run_length_median",
    "sample_run_length_q90",
    "7day_autocorrelation_sampled_load",
    "7day_spectral_structure",
    "background_recurrence",
    "interaction_complexity",
    "unexplained_activity",
    "relative_event_strength",
    "soft_active_fraction",
    "activity_episode_count",
    "longest_silent_run",
    "temporal_stability",
    "raw_adjacent_day_shape_distance",
    "raw_lag7_shape_similarity",
    "raw_lag14_shape_similarity",
    "raw_early_late_shape_distance",
    "raw_weekday_weekend_shape_difference",
    "cross_weekend_consistency",
]
RAW_MAGNITUDE_FEATURES = {"raw_q10_power", "raw_median_power", "raw_q90_power", "baseline_floor", "raw_max_power"}
POWER_CONDITION_COLUMNS = ["raw_median_power", "raw_q90_power", "baseline_floor", "frozen_sample_gt_1kw_fraction"]


def _cosine_rows(matrix: np.ndarray, reference: np.ndarray, exclude_index: int | None = None, top_k: int = 5) -> float:
    matrix = np.asarray(matrix, float)
    reference = np.asarray(reference, float)
    if not np.isfinite(matrix).all() or not len(reference):
        return np.nan
    norm = np.linalg.norm(matrix)
    ref_norm = np.linalg.norm(reference, axis=1)
    valid = ref_norm > 1e-9
    if not valid.any() or norm <= 1e-9:
        return np.nan
    similarities = reference[valid] @ matrix / (ref_norm[valid] * norm)
    if exclude_index is not None and 0 <= exclude_index < len(similarities):
        similarities = np.delete(similarities, exclude_index)
    if not len(similarities):
        return np.nan
    k = min(top_k, len(similarities))
    return float(np.mean(np.partition(similarities, len(similarities) - k)[-k:]))


def load_behavior_embeddings(paths: BranchPaths | None = None, window: int = 42) -> pd.DataFrame:
    paths = paths or BranchPaths.discover()
    frames = []
    for seed in SEEDS:
        folder = paths.upstream / "models" / "representations" / "final_plain" / f"daily_seed_{seed}"
        index = pd.read_csv(folder / "daily_embedding_index.csv", dtype={"meter_id": str})
        index["meter_id"] = index.meter_id.map(key)
        index["date"] = pd.to_datetime(index.date)
        embedding = np.load(folder / "daily_embeddings.npy")[:, :32]
        rows = []
        for meter_id, group in index.groupby("meter_id", sort=False):
            positions = group.sort_values("date").tail(window).index.to_numpy()
            vector = np.nanmean(embedding[positions], axis=0)
            rows.append([meter_id, *vector])
        columns = ["meter_id"] + [f"behavior_e{i:02d}_seed{seed}" for i in range(32)]
        frames.append(pd.DataFrame(rows, columns=columns))
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="meter_id", how="outer")
    for index in range(32):
        result[f"behavior_mean_e{index:02d}"] = result[[f"behavior_e{index:02d}_seed{seed}" for seed in SEEDS]].mean(axis=1)
    return result


class PowerConditioner:
    def __init__(self, columns=POWER_CONDITION_COLUMNS, neighbors: int = 120):
        self.columns = list(columns)
        self.neighbors = int(neighbors)

    def fit(self, frame: pd.DataFrame):
        raw = np.log1p(frame[self.columns].clip(lower=0).to_numpy(float))
        self.imputer = SimpleImputer(strategy="median").fit(raw)
        raw = self.imputer.transform(raw)
        self.scaler = RobustScaler().fit(raw)
        self.reference_power = self.scaler.transform(raw)
        self.reference_meter_ids = frame.meter_id.astype(str).tolist()
        return self

    def conditional_percentile(self, frame: pd.DataFrame, values, reference_values) -> np.ndarray:
        raw = np.log1p(frame[self.columns].clip(lower=0).to_numpy(float))
        current = self.scaler.transform(self.imputer.transform(raw))
        reference_values = np.asarray(reference_values, float)
        values = np.asarray(values, float)
        result = np.full(len(frame), np.nan)
        for index, vector in enumerate(current):
            if not np.isfinite(values[index]):
                continue
            distance = np.linalg.norm(self.reference_power - vector, axis=1)
            nearest = np.argsort(distance)[: min(self.neighbors, len(distance))]
            comparison = reference_values[nearest]
            comparison = comparison[np.isfinite(comparison)]
            if len(comparison):
                result[index] = np.searchsorted(np.sort(comparison), values[index], side="right") / len(comparison)
        return result


class BehaviorPUEnsemble:
    def __init__(self, base_columns, embedding_columns, seed=20260926):
        self.base_columns = list(base_columns)
        self.embedding_columns = list(embedding_columns)
        self.seed = int(seed)
        self.models = []

    def _pipeline(self, algorithm: str, seed: int):
        preprocessor = ColumnTransformer([
            ("base", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", RobustScaler())]), self.base_columns),
            ("embedding", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=12, random_state=seed)),
            ]), self.embedding_columns),
        ])
        if algorithm == "logistic":
            classifier = LogisticRegression(C=.25, max_iter=2000, solver="liblinear", random_state=seed)
        else:
            classifier = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=.045, l2_regularization=5, min_samples_leaf=20, random_state=seed)
        return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    def fit(self, frame, positive_rows, u_rows, positive_weights, u_weight=.12, bags=12):
        rng = np.random.default_rng(self.seed)
        self.models = []
        for bag in range(bags):
            seed = self.seed + bag * 19
            sampled_u = rng.choice(u_rows, min(len(u_rows), max(500, len(positive_rows) * 5)), replace=True)
            rows = np.r_[positive_rows, sampled_u]
            target = np.r_[np.ones(len(positive_rows)), np.zeros(len(sampled_u))]
            weights = np.r_[positive_weights, np.full(len(sampled_u), u_weight)]
            algorithm = "logistic" if bag % 3 == 0 else "hgb"
            model = self._pipeline(algorithm, seed)
            model.fit(frame.iloc[rows], target, classifier__sample_weight=weights)
            self.models.append(model)
        return self

    def predict(self, frame):
        return np.mean([model.predict_proba(frame)[:, 1] for model in self.models], axis=0)


def _reference_percentiles(frame, columns, reference_mask):
    output = {}
    for column in columns:
        output[column] = empirical_cdf(frame[column], frame.loc[reference_mask, column])
    return output


def _subtype_balanced_positive_weights(positive_ids, legacy, scope, medium_weight=.50):
    subtype = legacy.set_index("meter_id").subtype.to_dict()
    count = pd.Series([subtype[meter] for meter in positive_ids]).value_counts()
    high = set(scope.loc[scope.legacy_high_hard, "meter_id"])
    medium = set(scope.loc[scope.legacy_medium_soft, "meter_id"])
    weights = []
    for meter in positive_ids:
        support = 1.0 if meter in high else medium_weight if meter in medium else .10
        weights.append(support / count[subtype[meter]])
    weights = np.asarray(weights, float)
    return weights * len(weights) / weights.sum()


def assign_tiers(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    valid_families = out[["family_human_absence_support", "family_automatic_structure_support", "family_learned_representation_support"]].notna()
    family_positive = out[["family_human_absence_support", "family_automatic_structure_support", "family_learned_representation_support"]].ge(1 - DECISION["family_top_fraction"])
    base = (
        out.data_status.eq("valid")
        & out.relative_high_power_flag.fillna(False)
        & out.persistent_high_load.fillna(False)
        & ~out.severe_quality_issue.fillna(True)
        & ~out.occasional_high_peak_only.fillna(False)
    )
    tier_a = (
        base
        & valid_families.all(axis=1)
        & family_positive.all(axis=1)
        & (out.family_human_absence_support >= 1 - DECISION["human_opposition_top_fraction"])
        & (out.equipment_only_similarity_percentile >= 1 - DECISION["family_top_fraction"])
        & (out.equipment_support_or_automatic_percentile >= 1 - DECISION["automatic_or_subtype_top_fraction"])
        & (out.stability_rank_median >= DECISION["tier_a_rank_median"])
        & (out.stability_rank_p10 >= DECISION["tier_a_rank_p10"])
    )
    tier_b = (
        base & ~tier_a
        & (valid_families.sum(axis=1) >= 2)
        & (out[["family_human_absence_support", "family_automatic_structure_support", "family_learned_representation_support"]] >= DECISION["tier_b_family_support"]).sum(axis=1).ge(2)
        & (out.family_human_absence_support >= .50)
        & (out.stability_rank_median >= DECISION["tier_b_rank_median"])
    )
    strict = out["strict_high_power_flag"].fillna(False) if "strict_high_power_flag" in out else pd.Series(False, index=out.index)
    audited = out.relative_high_power_flag.fillna(False) | strict | out.persistent_high_load.fillna(False)
    out["hp_candidate_tier"] = np.select([tier_a, tier_b, audited], ["A", "B", "C"], default="NOT_HIGH_POWER")
    out["hp_candidate_flag"] = out.hp_candidate_tier.isin(["A", "B"])
    out["hp_core_candidate_flag"] = out.hp_candidate_tier.eq("A")
    return out


def validate_fit_audit(audit: dict):
    if audit.get("future_holdout_overlap_count") != 0:
        raise RuntimeError("Future holdout leaked into high-power branch fit")
    if audit.get("equipment_reference_count") != 29 or audit.get("equipment_intermittent_reference_count") != 21:
        raise RuntimeError("Legacy equipment reference counts do not match 29/21 freeze")
    if RAW_MAGNITUDE_FEATURES & set(audit.get("behavior_pu_features", [])):
        raise RuntimeError("Raw magnitude feature leaked into behavior-only PU")


def fit_high_power_branch(power_frame: pd.DataFrame, paths: BranchPaths | None = None, *, seed=20260926, medium_weight=.50, prior=.10, window=42, embedding_seed: int | None = None, bags: int = 9):
    paths = paths or BranchPaths.discover()
    legacy, legacy_audit = load_unique_legacy(paths)
    holdout = load_holdout_ids(paths)
    scope = load_scope(paths)
    features = pd.read_parquet(paths.upstream / "final_fit_feature_table.parquet")
    features["meter_id"] = features.meter_id.map(key)
    embeddings = load_behavior_embeddings(paths, window=window)
    if embedding_seed is not None:
        for index in range(32):
            embeddings[f"behavior_mean_e{index:02d}"] = embeddings[f"behavior_e{index:02d}_seed{embedding_seed}"]
    frame = power_frame.merge(features, on="meter_id", how="left", suffixes=("", "_feature"))
    frame = frame.merge(embeddings, on="meter_id", how="left")
    reference = fit_reference_mask(frame)
    reference_ids = set(frame.loc[reference, "meter_id"])
    if reference_ids & holdout:
        raise RuntimeError("Future holdout leaked into fit reference")
    high_reference = reference & frame.relative_high_power_flag.fillna(False) & frame.persistent_high_load.fillna(False)
    if high_reference.sum() < 30:
        high_reference = reference & frame.relative_high_power_flag.fillna(False)

    conditioner = PowerConditioner().fit(frame.loc[reference, ["meter_id"] + POWER_CONDITION_COLUMNS])
    human = pd.to_numeric(frame.human_interaction_evidence, errors="coerce").to_numpy(float)
    active = pd.to_numeric(frame.active_evidence_score, errors="coerce").to_numpy(float)
    reference_human = human[reference.to_numpy()]
    reference_active = active[reference.to_numpy()]
    frame["human_power_conditioned_percentile"] = conditioner.conditional_percentile(frame, human, reference_human)
    frame["active_power_conditioned_percentile"] = conditioner.conditional_percentile(frame, active, reference_active)
    frame["human_interaction_opposition"] = 1 - frame.human_power_conditioned_percentile
    frame["active_evidence_opposition"] = 1 - frame.active_power_conditioned_percentile

    auto_columns = [
        "automatic_control_compatibility", "cross_day_normalized_shape_similarity",
        "daily_trajectory_similarity_sample", "background_recurrence", "temporal_stability",
        "raw_lag7_shape_similarity",
    ]
    auto_percentiles = _reference_percentiles(frame, auto_columns, high_reference)
    for column, values in auto_percentiles.items():
        frame[f"hp_pct_{column}"] = values
    frame["transition_automatic_support"] = 1 - empirical_cdf(frame.sample_state_transition_rate, frame.loc[high_reference, "sample_state_transition_rate"])

    embedding_columns = [f"behavior_mean_e{i:02d}" for i in range(32)]
    vectors = frame[embedding_columns].to_numpy(float)
    equipment_ids = legacy.loc[legacy.subtype.eq("equipment"), "meter_id"].tolist()
    intermit_ids = legacy.loc[legacy.subtype.eq("equipment_intermittent"), "meter_id"].tolist()
    lookup = {meter: index for index, meter in enumerate(frame.meter_id)}
    equipment_vectors = np.asarray([vectors[lookup[meter]] for meter in equipment_ids if meter in lookup])
    intermit_vectors = np.asarray([vectors[lookup[meter]] for meter in intermit_ids if meter in lookup])
    frame["equipment_only_similarity"] = [_cosine_rows(vector, equipment_vectors) for vector in vectors]
    frame["equipment_intermit_similarity"] = [_cosine_rows(vector, intermit_vectors) for vector in vectors]

    pu_base = [column for column in BEHAVIOR_ROBUST_FEATURES if column in frame]
    pu_embedding = embedding_columns
    eligible_positive_ids = [meter for meter in legacy.meter_id if meter in reference_ids]
    positive_rows = np.array([lookup[meter] for meter in eligible_positive_ids], int)
    u_ids = sorted(reference_ids - set(legacy.meter_id))
    u_rows = np.array([lookup[meter] for meter in u_ids], int)
    positive_weight = _subtype_balanced_positive_weights(eligible_positive_ids, legacy, scope, medium_weight)
    pu = BehaviorPUEnsemble(pu_base, pu_embedding, seed=seed).fit(
        frame, positive_rows, u_rows, positive_weight,
        u_weight=float(np.clip(.12 * .10 / prior, .05, .30)), bags=bags,
    )
    frame["behavior_only_pu_score"] = pu.predict(frame)
    frame["behavior_only_pu_rank"] = empirical_cdf(frame.behavior_only_pu_score, frame.loc[reference, "behavior_only_pu_score"])

    for source, target in [
        ("equipment_only_similarity", "equipment_only_similarity_percentile"),
        ("equipment_intermit_similarity", "equipment_intermit_similarity_percentile"),
        ("behavior_only_pu_score", "behavior_only_pu_hp_percentile"),
    ]:
        frame[target] = empirical_cdf(frame[source], frame.loc[high_reference, source])
    frame["family_human_absence_support"] = frame[["human_interaction_opposition", "active_evidence_opposition"]].median(axis=1, skipna=True)
    frame["family_automatic_structure_support"] = frame[[f"hp_pct_{c}" for c in auto_columns] + ["transition_automatic_support"]].median(axis=1, skipna=True)
    frame["family_learned_representation_support"] = frame[["equipment_only_similarity_percentile", "equipment_intermit_similarity_percentile", "behavior_only_pu_hp_percentile"]].median(axis=1, skipna=True)
    subtype_columns = [c for c in ["equipment_support", "equipment_intermit_support"] if c in frame]
    subtype_raw = frame[subtype_columns].max(axis=1) if subtype_columns else pd.Series(np.nan, index=frame.index)
    frame["equipment_subtype_support_percentile"] = empirical_cdf(subtype_raw, subtype_raw[high_reference])
    frame["equipment_support_or_automatic_percentile"] = frame[["equipment_subtype_support_percentile", "family_automatic_structure_support"]].max(axis=1)
    frame["hp_branch_score"] = frame[["family_human_absence_support", "family_automatic_structure_support", "family_learned_representation_support"]].median(axis=1, skipna=True)
    frame["hp_conditional_rank"] = empirical_cdf(frame.hp_branch_score, frame.loc[high_reference, "hp_branch_score"])
    definition_masks = {
        "relative": reference & frame.relative_high_power_flag.fillna(False),
        "strict": reference & frame.strict_high_power_flag.fillna(False),
        "persistent": reference & frame.persistent_high_load.fillna(False),
    }
    for name, mask in definition_masks.items():
        frame[f"hp_rank_within_{name}_definition"] = empirical_cdf(frame.hp_branch_score, frame.loc[mask, "hp_branch_score"])
    frame["stability_rank_median"] = frame.hp_conditional_rank
    frame["stability_rank_p10"] = frame.hp_conditional_rank
    frame["stability_rank_p90"] = frame.hp_conditional_rank
    frame["stability_rank_std"] = 0.0
    frame = assign_tiers(frame)

    audit = {
        "legacy_source_rows": legacy_audit["source_rows"],
        "legacy_unique_meter_ids": legacy_audit["unique_meter_ids"],
        "equipment_reference_count": len(equipment_ids),
        "equipment_intermittent_reference_count": len(intermit_ids),
        "fit_reference_count": int(reference.sum()),
        "high_power_reference_count": int(high_reference.sum()),
        "future_holdout_count": len(holdout),
        "future_holdout_overlap_count": len(reference_ids & holdout),
        "behavior_pu_positive_count": len(eligible_positive_ids),
        "legacy_used_in_power_coverage_and_selection_bias_audit": int(len(legacy)),
        "legacy_excluded_from_behavior_pu_count": int(len(legacy) - len(eligible_positive_ids)),
        "legacy_behavior_pu_exclusion_reason": "strict-zero or outside frozen final-fit behavioral scope; retained in router/coverage/subtype audits",
        "behavior_pu_u_comparison_count": len(u_ids),
        "behavior_pu_features": pu_base + pu_embedding,
        "raw_magnitude_distance_used_as_veto": False,
        "u_interpretation": "unlabeled comparison pool, not known negative",
        "medium_weight": medium_weight,
        "prior_regularization": prior,
        "seed": seed,
        "window_days": window,
        "embedding_seed": embedding_seed,
        "bags": bags,
    }
    validate_fit_audit(audit)
    artifact = {
        "conditioner": conditioner,
        "conditioner_reference_human": reference_human,
        "conditioner_reference_active": reference_active,
        "behavior_pu": pu,
        "equipment_reference_vectors": equipment_vectors,
        "equipment_intermittent_reference_vectors": intermit_vectors,
        "equipment_reference_ids": equipment_ids,
        "equipment_intermittent_reference_ids": intermit_ids,
        "fit_reference_ids": sorted(reference_ids),
        "high_power_reference_ids": sorted(frame.loc[high_reference, "meter_id"]),
        "high_power_reference_scores": np.sort(frame.loc[high_reference, "hp_branch_score"].dropna().to_numpy(float)),
        "high_power_reference_distributions": {
            "equipment_only_similarity": np.sort(frame.loc[high_reference, "equipment_only_similarity"].dropna().to_numpy(float)),
            "equipment_intermit_similarity": np.sort(frame.loc[high_reference, "equipment_intermit_similarity"].dropna().to_numpy(float)),
            "behavior_only_pu_score": np.sort(frame.loc[high_reference, "behavior_only_pu_score"].dropna().to_numpy(float)),
        },
        "behavior_embedding_columns": embedding_columns,
        "behavior_pu_base_columns": pu_base,
        "decision": DECISION,
        "audit": audit,
    }
    return frame, artifact, audit


def save_branch_model(artifact, audit, paths: BranchPaths | None = None):
    paths = paths or BranchPaths.discover()
    (paths.root / "models").mkdir(exist_ok=True)
    joblib.dump(artifact, paths.root / "models" / "high_power_branch_model.joblib")
    (paths.root / "high_power_branch_fit_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
