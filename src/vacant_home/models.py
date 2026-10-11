from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

from .data import ROOT, SEED, TIME_COLS, bool_col, key, load_daily, write_json
from .representation import SEEDS


MEDIUM_WEIGHTS = (.25, .50, .75)
PRIOR_SETTINGS = (.06, .10, .16)
CENTRAL_WEIGHT = .50
CENTRAL_PRIOR = .10

SUBTYPE_MAP = {
    "完全空置": "complete",
    "设备维持型空置": "equipment",
    "设备维持间歇型": "equipment_intermittent",
    "间歇使用型空置": "intermittent_use",
}

ROBUST_FEATURES = [
    "raw_q10_power", "raw_median_power", "raw_q90_power", "baseline_floor", "raw_relative_amplitude",
    "sample_zero_fraction", "sample_near_zero_fraction", "cross_day_normalized_shape_similarity",
    "daily_trajectory_similarity_sample", "sample_state_transition_rate", "sample_run_length_median",
    "sample_run_length_q90", "clock_time_recurrence_dispersion", "7day_autocorrelation_sampled_load",
    "7day_spectral_structure", "change_point_structure", "daily_nonzero_fraction_mean",
    "background_recurrence", "device_likeness", "interaction_complexity", "unexplained_activity",
    "relative_event_strength", "soft_active_fraction", "activity_episode_count", "longest_silent_run",
    "temporal_stability", "raw_adjacent_day_shape_distance", "raw_lag7_shape_similarity",
    "raw_lag14_shape_similarity", "raw_early_late_shape_distance", "raw_weekday_weekend_shape_difference",
    "weekend_sampled_load_index_ratio", "weekend_nonzero_sample_fraction_difference",
    "weekend_q90_difference", "weekend_episode_fraction", "cross_weekend_consistency",
]

AUTOMATIC_COMPONENTS = [
    "background_recurrence", "device_likeness", "daily_trajectory_similarity_sample",
    "cross_day_normalized_shape_similarity", "temporal_stability", "raw_lag7_shape_similarity",
]
HUMAN_COMPONENTS = [
    "interaction_complexity", "unexplained_activity", "relative_event_strength",
    "soft_active_fraction", "activity_episode_count", "sample_state_transition_rate",
    "change_point_structure", "raw_weekday_weekend_shape_difference",
]
WEEKEND_COMPONENTS = [
    "weekend_sampled_load_index_ratio", "weekend_episode_fraction", "raw_lag7_shape_similarity",
]


def percentile(values):
    values = np.asarray(values, float)
    out = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    if finite.sum() > 1:
        out[finite] = (rankdata(values[finite], method="average") - 1) / (finite.sum() - 1)
    elif finite.sum() == 1:
        out[finite] = .5
    return out


def u_comparison_weight(prior):
    """Prior sensitivity weight; prior is not interpreted as population prevalence."""
    return float(np.clip(.12 * CENTRAL_PRIOR / float(prior), .05, .30))


class ReferenceRanker:
    def __init__(self, columns):
        self.columns = list(columns)
        self.reference = {}

    def fit(self, frame):
        for column in self.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            self.reference[column] = np.sort(values[np.isfinite(values)])
        return self

    def transform_column(self, frame, column):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        reference = self.reference[column]
        out = np.full(len(frame), np.nan)
        ok = np.isfinite(values)
        out[ok] = np.searchsorted(reference, values[ok], side="right") / max(1, len(reference))
        return out


def load_scope():
    scope = pd.read_csv(ROOT / "fit_scope.csv", dtype={"meter_id": str})
    scope["meter_id"] = scope.meter_id.map(key)
    bools = [
        "data_invalid", "strict_all_zero", "legacy_high_hard", "legacy_medium_soft",
        "legacy_low_diagnostic_only", "future_holdout", "research_fit", "final_fit", "ssl_fit",
    ]
    for column in bools:
        scope[column] = bool_col(scope[column])
    return scope


def load_legacy():
    legacy = pd.read_excel(ROOT / "inputs" / "legacy_labels.xlsx")
    legacy = legacy.rename(columns={legacy.columns[0]: "meter_id", legacy.columns[1]: "raw_subtype"})
    legacy["meter_id"] = legacy.meter_id.map(key)
    legacy = legacy.drop_duplicates("meter_id", keep="first")
    legacy["subtype"] = legacy.raw_subtype.map(SUBTYPE_MAP)
    if legacy.subtype.isna().any():
        raise RuntimeError("Unknown Legacy subtype")
    return legacy


def _run_lengths(mask):
    lengths, current = [], 0
    for value in np.asarray(mask, bool):
        if value:
            current += 1
        elif current:
            lengths.append(current); current = 0
    if current:
        lengths.append(current)
    return lengths


def build_direct_raw_temporal_features():
    """Fit-free per-meter features; no cohort statistics or learned representation."""
    daily = load_daily().sort_values(["meter_id", "date"])
    rows = []
    for meter_id, group in daily.groupby("meter_id", sort=False):
        values = group[TIME_COLS].to_numpy(float)
        dates = pd.to_datetime(group.date)
        finite = values[np.isfinite(values)]
        if not len(finite):
            rows.append({"meter_id": meter_id}); continue
        median = np.nanmedian(values, axis=1, keepdims=True)
        spread = np.nanquantile(values, .9, axis=1, keepdims=True) - np.nanquantile(values, .1, axis=1, keepdims=True)
        normalized = np.nan_to_num((values - median) / np.maximum(spread, .02))
        adjacent = np.mean(np.abs(np.diff(normalized, axis=0)), axis=1) if len(values) > 1 else np.array([0.0])

        def lag_similarity(lag):
            scores = []
            for index in range(lag, len(normalized)):
                left, right = normalized[index], normalized[index - lag]
                denominator = np.linalg.norm(left) * np.linalg.norm(right)
                if denominator > 1e-9:
                    scores.append(float(left @ right / denominator))
            return float(np.nanmean(scores)) if scores else np.nan

        half = max(1, len(values) // 2)
        first = np.nanmean(normalized[:half], axis=0)
        second = np.nanmean(normalized[half:], axis=0) if len(values) > half else first
        weekend = dates.dt.dayofweek.to_numpy() >= 5
        weekday = ~weekend
        weekend_profile = np.nanmean(normalized[weekend], axis=0) if weekend.any() else np.full(96, np.nan)
        weekday_profile = np.nanmean(normalized[weekday], axis=0) if weekday.any() else np.full(96, np.nan)
        daily_mean = np.nanmean(values, axis=1)
        daily_nonzero = np.nanmean(np.nan_to_num(values) > 1e-8, axis=1)
        daily_q90 = np.nanquantile(values, .9, axis=1)
        threshold = np.nanquantile(finite, .75)
        episodes = np.array([len(_run_lengths(np.nan_to_num(row) > threshold)) for row in values], float)

        def mean_where(vector, mask):
            return float(np.nanmean(vector[mask])) if mask.any() else np.nan

        weekday_load = mean_where(daily_mean, weekday)
        weekend_load = mean_where(daily_mean, weekend)
        ratio = weekend_load / (weekday_load + 1e-8) if np.isfinite(weekend_load) and np.isfinite(weekday_load) else np.nan
        weekend_days = normalized[weekend]
        weekend_consistency = 1 / (1 + np.nanmean(np.abs(np.diff(weekend_days, axis=0)))) if len(weekend_days) > 1 else np.nan
        rows.append({
            "meter_id": meter_id,
            "raw_adjacent_day_shape_distance": float(np.nanmean(adjacent)),
            "raw_lag7_shape_similarity": lag_similarity(7),
            "raw_lag14_shape_similarity": lag_similarity(14),
            "raw_early_late_shape_distance": float(np.nanmean(np.abs(second - first))),
            "raw_weekday_weekend_shape_difference": float(np.nanmean(np.abs(weekend_profile - weekday_profile))),
            "weekend_sampled_load_index_ratio": ratio,
            "weekend_nonzero_sample_fraction_difference": mean_where(daily_nonzero, weekend) - mean_where(daily_nonzero, weekday),
            "weekend_q90_difference": mean_where(daily_q90, weekend) - mean_where(daily_q90, weekday),
            "weekend_episode_fraction": float(np.nansum(episodes[weekend]) / (np.nansum(episodes) + 1e-8)),
            "cross_weekend_consistency": float(weekend_consistency),
        })
    output = pd.DataFrame(rows)
    output.to_csv(ROOT / "direct_raw_temporal_features.csv", index=False)
    return output


def build_regional_features(fit_ids):
    daily = load_daily()
    holdout = set(load_scope().loc[lambda d: d.future_holdout, "meter_id"])
    if set(fit_ids) & holdout:
        raise RuntimeError("Future holdout leaked into regional common factor")
    values = daily[TIME_COLS].to_numpy(float)
    median = np.nanmedian(values, axis=1, keepdims=True)
    spread = np.nanquantile(values, .9, axis=1, keepdims=True) - np.nanquantile(values, .1, axis=1, keepdims=True)
    normalized = np.clip((values - median) / np.maximum(spread, .02), -8, 8)
    raw_factors = {}
    fit_mask = daily.meter_id.isin(fit_ids).to_numpy() & (np.nanmax(np.abs(values), axis=1) > .005)
    for (utility, date), positions in daily[fit_mask].groupby(["utility", "date"]).groups.items():
        matrix = normalized[np.asarray(list(positions), int)]
        raw_factors[(utility, pd.Timestamp(date))] = np.nanmedian(matrix, axis=0)
    calendar_templates = {}
    template_groups = {}
    for (utility, date), factor in raw_factors.items():
        template_groups.setdefault((utility, int(date.dayofweek >= 5)), []).append(factor)
    for group, group_factors in template_groups.items():
        calendar_templates[group] = np.nanmedian(np.stack(group_factors), axis=0)
    factors = {
        (utility, date): factor - calendar_templates[(utility, int(date.dayofweek >= 5))]
        for (utility, date), factor in raw_factors.items()
    }
    rows = []
    for meter_id, group in daily.groupby("meter_id", sort=False):
        correlations = []
        residuals = []
        for position in group.index:
            factor = factors.get((daily.at[position, "utility"], pd.Timestamp(daily.at[position, "date"])))
            if factor is None:
                continue
            curve = normalized[position]
            valid = np.isfinite(curve) & np.isfinite(factor)
            if valid.sum() < 24:
                continue
            centered_curve = curve[valid] - np.mean(curve[valid])
            centered_factor = factor[valid] - np.mean(factor[valid])
            denominator = np.linalg.norm(centered_curve) * np.linalg.norm(centered_factor)
            if denominator > 1e-9:
                correlation = float(centered_curve @ centered_factor / denominator)
                correlations.append(correlation)
                beta = float(centered_curve @ centered_factor / (centered_factor @ centered_factor + 1e-9))
                residuals.extend((centered_curve - beta * centered_factor).tolist())
        rows.append({
            "meter_id": meter_id,
            "regional_common_factor_coupling": float(np.nanmedian(correlations)) if correlations else np.nan,
            "regional_common_factor_residual": float(np.nanmedian([1 - x for x in correlations])) if correlations else np.nan,
            "regional_idiosyncratic_variability": float(np.nanstd(residuals)) if residuals else np.nan,
        })
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "regional_common_factor_main.csv", index=False)
    write_json(ROOT / "regional_common_factor_fit_audit.json", {
        "fit_meter_count": len(fit_ids), "future_holdout_overlap": len(set(fit_ids) & holdout),
        "interpretation": "regional common exogenous-pattern compatibility after utility-specific weekday/weekend clock-pattern removal; not weather or occupancy",
        "calendar_control": "utility x weekday/weekend 96-position population template removed before coupling",
    })
    return result


def load_representations(variant, kind="tcn", window=42):
    frames = []
    for seed in SEEDS:
        if kind == "daily":
            path = ROOT / "models" / "representations" / variant / f"daily_seed_{seed}"
            index = pd.read_csv(path / "daily_embedding_index.csv", dtype={"meter_id": str})
            index["date"] = pd.to_datetime(index.date)
            embedding = np.load(path / "daily_embeddings.npy")
            rows = []
            for meter_id, group in index.groupby("meter_id", sort=False):
                positions = group.sort_values("date").tail(window).index.to_numpy()
                rows.append([meter_id, *embedding[positions].mean(axis=0)])
            frame = pd.DataFrame(rows, columns=["meter_id"] + [f"daily_e{i:02d}" for i in range(64)])
        else:
            path = ROOT / "models" / "representations" / variant / f"long_{kind}_seed_{seed}"
            index = pd.read_csv(path / "user_embedding_index.csv", dtype={"meter_id": str})
            embedding = np.load(path / f"user_embeddings_window_{window}.npy")
            frame = pd.DataFrame(embedding, columns=[f"{kind}_e{i:02d}" for i in range(64)])
            frame.insert(0, "meter_id", index.meter_id.map(key))
        frame = frame.rename(columns={c: f"{c}_seed{seed}" for c in frame if c != "meter_id"})
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="meter_id", how="inner")
    prefix = "daily" if kind == "daily" else kind
    for index in range(64):
        source = [f"{prefix}_e{index:02d}_seed{seed}" for seed in SEEDS]
        merged[f"{prefix}_mean_e{index:02d}"] = merged[source].mean(axis=1)
    return merged


def build_feature_table():
    scope = load_scope()
    core = pd.read_csv(ROOT / "inputs" / "semantics_robust_core.csv", dtype={"meter_id": str})
    core["meter_id"] = core.meter_id.map(key)
    direct = build_direct_raw_temporal_features()
    core = core.merge(direct, on="meter_id", how="left")
    robust = [column for column in ROBUST_FEATURES if column in core]
    frame = core[["meter_id"] + robust].copy()
    reference_ids = set(scope.loc[scope.research_fit & ~scope.strict_all_zero, "meter_id"])
    ranker = ReferenceRanker(sorted(set(AUTOMATIC_COMPONENTS + HUMAN_COMPONENTS + WEEKEND_COMPONENTS))).fit(
        frame[frame.meter_id.isin(reference_ids)]
    )
    auto = np.column_stack([ranker.transform_column(frame, column) for column in AUTOMATIC_COMPONENTS])
    human = np.column_stack([ranker.transform_column(frame, column) for column in HUMAN_COMPONENTS])
    frame["automatic_control_compatibility"] = np.nanmean(auto, axis=1)
    frame["human_interaction_evidence"] = np.nanmean(human, axis=1)
    frame["weekend_tendency"] = np.nanmean(np.column_stack([
        ranker.transform_column(frame, "weekend_sampled_load_index_ratio"),
        ranker.transform_column(frame, "weekend_episode_fraction"),
        ranker.transform_column(frame, "raw_lag7_shape_similarity"),
    ]), axis=1)
    plain_daily = load_representations("research_plain", "daily")
    plain_tcn = load_representations("research_plain", "tcn")
    contrastive_tcn = load_representations("research_contrastive", "tcn")
    plain_tcn = plain_tcn.rename(columns={
        column: column.replace("tcn_", "tcn_plain_", 1)
        for column in plain_tcn.columns if column != "meter_id"
    })
    contrastive_tcn = contrastive_tcn.rename(columns={
        column: column.replace("tcn_", "tcn_contrast_", 1)
        for column in contrastive_tcn.columns if column != "meter_id"
    })
    gru = load_representations("research_gru", "gru")
    regional = build_regional_features(reference_ids)
    for extra in (plain_daily, plain_tcn, contrastive_tcn, gru, regional):
        frame = frame.merge(extra, on="meter_id", how="left")
    joblib.dump(ranker, ROOT / "models" / "automatic_human_reference_ranker.joblib")
    families = {
        "POWER_ONLY": {"base": ["raw_q10_power", "raw_median_power", "raw_q90_power", "baseline_floor"], "embedding": []},
        "ROBUST_FEATURES": {"base": robust + ["automatic_control_compatibility", "human_interaction_evidence", "weekend_tendency"], "embedding": []},
        "DAILY_TCN_FUSION": {"base": robust + ["automatic_control_compatibility", "human_interaction_evidence", "weekend_tendency"], "embedding": [f"daily_mean_e{i:02d}" for i in range(64)]},
        "LONG_HORIZON_TCN": {"base": robust + ["automatic_control_compatibility", "human_interaction_evidence", "weekend_tendency"], "embedding": [f"tcn_plain_mean_e{i:02d}" for i in range(64)]},
        "CONTEXTUAL_TCN": {"base": robust + ["automatic_control_compatibility", "human_interaction_evidence", "weekend_tendency"], "embedding": [f"tcn_contrast_mean_e{i:02d}" for i in range(64)]},
        "LONG_HORIZON_GRU": {"base": robust + ["automatic_control_compatibility", "human_interaction_evidence", "weekend_tendency"], "embedding": [f"gru_mean_e{i:02d}" for i in range(64)]},
    }
    frame.to_parquet(ROOT / "feature_table.parquet", index=False)
    regional_cols = ["regional_common_factor_coupling", "regional_common_factor_residual", "regional_idiosyncratic_variability"]
    write_json(ROOT / "feature_families.json", {"families": families, "regional": regional_cols})
    return frame, families, regional_cols


class MixedEstimator:
    def __init__(self, base_columns, embedding_columns, algorithm="hgb", seed=SEED, pca_components=12):
        self.base_columns = list(base_columns)
        self.embedding_columns = list(embedding_columns)
        self.algorithm = algorithm
        self.seed = seed
        self.pca_components = pca_components

    def fit(self, frame, y, sample_weight):
        self.base_imputer = SimpleImputer(strategy="median").fit(frame[self.base_columns])
        base = self.base_imputer.transform(frame[self.base_columns])
        self.base_scaler = RobustScaler().fit(base)
        base = self.base_scaler.transform(base)
        if self.embedding_columns:
            self.embedding_imputer = SimpleImputer(strategy="median").fit(frame[self.embedding_columns])
            embedding = self.embedding_imputer.transform(frame[self.embedding_columns])
            self.embedding_scaler = StandardScaler().fit(embedding)
            embedding = self.embedding_scaler.transform(embedding)
            n_components = min(self.pca_components, embedding.shape[1], max(2, embedding.shape[0] - 1))
            self.pca = PCA(n_components=n_components, random_state=self.seed).fit(embedding)
            base = np.c_[base, self.pca.transform(embedding)]
        if self.algorithm == "logistic":
            self.model = LogisticRegression(C=.25, max_iter=2000, solver="liblinear", random_state=self.seed)
        else:
            self.model = HistGradientBoostingClassifier(
                max_depth=3, max_iter=110, learning_rate=.045, l2_regularization=4,
                min_samples_leaf=20, random_state=self.seed,
            )
        self.model.fit(base, y, sample_weight=sample_weight)
        return self

    def transform(self, frame):
        base = self.base_scaler.transform(self.base_imputer.transform(frame[self.base_columns]))
        if self.embedding_columns:
            embedding = self.embedding_scaler.transform(self.embedding_imputer.transform(frame[self.embedding_columns]))
            base = np.c_[base, self.pca.transform(embedding)]
        return base

    def predict_proba(self, frame):
        return self.model.predict_proba(self.transform(frame))[:, 1]


PROPENSITY_FEATURES = {
    "raw_median_power": "magnitude affects historical visual screening and shortlist construction",
    "raw_q90_power": "large sampled peaks can make historical vacancy selection less likely",
    "baseline_floor": "persistent sampled baseload affects historical visual salience",
    "sample_zero_fraction": "strict and near silence were easy for reviewers to identify",
    "sample_near_zero_fraction": "persistent near-zero curves were historically salient",
    "longest_silent_run": "long silent sample runs are visually obvious",
    "activity_episode_count": "obvious intermittent return episodes affect selection",
    "clock_time_recurrence_dispersion": "simple recurring shapes can be easier to categorize",
    "cross_day_normalized_shape_similarity": "stable repeated motifs affect historical typicality",
}


def fit_propensity(frame, scope, excluded_positive_ids):
    holdout = set(scope.loc[scope.future_holdout, "meter_id"])
    strict_zero = set(scope.loc[scope.strict_all_zero, "meter_id"])
    fit_ids = set(scope.loc[scope.research_fit, "meter_id"]) - set(excluded_positive_ids) - holdout - strict_zero
    selected = set(scope.loc[scope.legacy_high_hard | scope.legacy_medium_soft, "meter_id"]) - set(excluded_positive_ids)
    rows = frame.meter_id.isin(fit_ids)
    y = frame.meter_id.isin(selected).astype(int)
    columns = [c for c in PROPENSITY_FEATURES if c in frame]
    model = MixedEstimator(columns, [], "logistic", SEED).fit(frame.loc[rows], y[rows], np.ones(rows.sum()))
    propensity = np.clip(model.predict_proba(frame), .03, .95)
    return propensity, model


def positive_weights(ids, scope, subtype, medium_weight):
    labels = [subtype[x] for x in ids]
    counts = pd.Series(labels).value_counts()
    weights = np.array([1 / counts[label] for label in labels], float)
    weights *= len(weights) / weights.sum()
    hard = set(scope.loc[scope.legacy_high_hard, "meter_id"])
    weights *= np.array([1.0 if meter in hard else medium_weight for meter in ids])
    return weights


def crossfit(frame, family_name, family, scope, medium_weight=CENTRAL_WEIGHT, prior=CENTRAL_PRIOR,
             propensity=False, bags=10, algorithm="hgb"):
    legacy = load_legacy()
    subtype = legacy.set_index("meter_id").subtype.to_dict()
    hard_medium = set(scope.loc[scope.legacy_high_hard | scope.legacy_medium_soft, "meter_id"])
    research = set(scope.loc[scope.research_fit, "meter_id"])
    strict_zero = set(scope.loc[scope.strict_all_zero, "meter_id"])
    positives = sorted((hard_medium & research & set(frame.meter_id)) - strict_zero)
    u_pool = sorted(research - set(legacy.meter_id) - set(scope.loc[scope.future_holdout, "meter_id"]) - strict_zero)
    lookup = {meter: index for index, meter in enumerate(frame.meter_id)}
    labels = np.array([subtype[meter] for meter in positives])
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    positive_rows = []
    u_rows_output = []
    for fold, (train_indices, held_indices) in enumerate(folds.split(positives, labels)):
        train_positive = [positives[i] for i in train_indices]
        held_positive = [positives[i] for i in held_indices]
        train_positive_rows = np.array([lookup[meter] for meter in train_positive])
        held_positive_rows = np.array([lookup[meter] for meter in held_positive])
        u_candidate_rows = np.array([lookup[meter] for meter in u_pool])
        propensity_values = None
        if propensity:
            propensity_values, _ = fit_propensity(frame, scope, held_positive)
        held_scores = []
        u_scores = []
        for bag in range(bags):
            seed = SEED + fold * 101 + bag * 17
            rng = np.random.default_rng(seed)
            sampled_u = rng.choice(u_candidate_rows, min(len(u_candidate_rows), max(420, len(train_positive_rows) * 5)), replace=True)
            train_rows = np.r_[train_positive_rows, sampled_u]
            target = np.r_[np.ones(len(train_positive_rows)), np.zeros(len(sampled_u))]
            p_weight = positive_weights(train_positive, scope, subtype, medium_weight)
            if propensity_values is not None:
                adjusted = p_weight / propensity_values[train_positive_rows]
                p_weight = np.clip(adjusted, 0, np.quantile(adjusted, .95))
            weights = np.r_[p_weight, np.full(len(sampled_u), u_comparison_weight(prior))]
            estimator = MixedEstimator(family["base"], family["embedding"], algorithm, seed)
            estimator.fit(frame.iloc[train_rows], target, weights)
            held_scores.append(estimator.predict_proba(frame.iloc[held_positive_rows]))
            u_scores.append(estimator.predict_proba(frame.iloc[u_candidate_rows]))
        held_mean = np.mean(held_scores, axis=0)
        u_mean = np.mean(u_scores, axis=0)
        reference = np.sort(u_mean)
        for meter, score in zip(held_positive, held_mean):
            positive_rows.append({
                "meter_id": meter, "family": family_name, "fold": fold,
                "medium_weight": medium_weight, "prior": prior, "propensity": propensity,
                "score": score, "u_percentile_rank": np.searchsorted(reference, score, side="right") / len(reference),
                "subtype": subtype[meter],
                "support": "HIGH" if meter in set(scope.loc[scope.legacy_high_hard, "meter_id"]) else "MEDIUM",
            })
        u_rows_output.append(pd.DataFrame({
            "meter_id": u_pool, "family": family_name, "fold": fold,
            "medium_weight": medium_weight, "prior": prior, "propensity": propensity,
            "score": u_mean, "u_percentile_rank": percentile(u_mean),
        }))
    return pd.DataFrame(positive_rows), pd.concat(u_rows_output, ignore_index=True)


def power_matched(frame, positives, u_scores, reference_ids):
    power_columns = ["raw_median_power", "raw_q10_power", "raw_q90_power", "baseline_floor"]
    available = [c for c in power_columns if c in frame]
    power = frame.set_index("meter_id")[available].replace([np.inf, -np.inf], np.nan)
    reference = power.loc[power.index.intersection(reference_ids)]
    medians = reference.median()
    power = power.fillna(medians)
    scaler = RobustScaler().fit(reference.fillna(medians))
    scaled = pd.DataFrame(scaler.transform(power), index=power.index, columns=available)
    rows = []
    for _, positive in positives.iterrows():
        u = u_scores[u_scores.fold.eq(positive.fold)].copy()
        u = u[u.meter_id.isin(scaled.index)]
        if positive.meter_id not in scaled.index or len(u) < 20:
            continue
        distances = np.linalg.norm(scaled.loc[u.meter_id].to_numpy() - scaled.loc[positive.meter_id].to_numpy(), axis=1)
        nearest = np.argsort(distances)[:20]
        matched = u.iloc[nearest]
        local_win_rate = float(np.mean(positive.score > matched.score))
        matched_rank = float(np.median(matched.u_percentile_rank))
        rows.append({
            **positive.to_dict(), "matched_u_rank_median": matched_rank,
            "power_matched_local_rank": local_win_rate,
            "power_matched_advantage": local_win_rate - .5,
            "positive_minus_matched_score": float(positive.score - matched.score.median()),
            "matched_power_distance": float(np.median(distances[nearest])),
        })
    return pd.DataFrame(rows)


def bootstrap_ci(values, seed=SEED, repetitions=2000):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    distribution = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(repetitions)])
    return float(np.quantile(distribution, .025)), float(np.quantile(distribution, .975))


def evaluate_configuration(frame, positive, u_scores):
    scope = load_scope()
    reference_ids = set(scope.loc[scope.research_fit & ~scope.strict_all_zero, "meter_id"])
    matched = power_matched(frame, positive, u_scores, set(u_scores.meter_id))
    subtype = positive.groupby("subtype").u_percentile_rank.mean()
    advantage = matched.power_matched_advantage.to_numpy()
    ci_low, ci_high = bootstrap_ci(advantage)
    reference = frame[frame.meter_id.isin(reference_ids)].copy()
    power_views = [column for column in ["raw_median_power", "raw_q90_power", "baseline_floor"] if column in frame]
    power_percentiles = pd.DataFrame(index=frame.index)
    for column in power_views:
        values = pd.to_numeric(reference[column], errors="coerce").dropna().sort_values().to_numpy()
        current = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        power_percentiles[column] = np.searchsorted(values, current, side="right") / max(1, len(values))
    # A known positive is an absolute high-power case when any audited power
    # view is in the research-universe top quartile. Averaging the views first
    # was too restrictive and could make the cohort empty even when q90 or a
    # sustained baseline was genuinely high.
    high_power_mask = (power_percentiles >= .75).any(axis=1)
    high_ids = set(frame.loc[high_power_mask, "meter_id"])
    high_power = positive[positive.meter_id.isin(high_ids)].u_percentile_rank
    central_u = u_scores.groupby("meter_id", as_index=False).agg(
        rank=("u_percentile_rank", "mean")
    ).merge(frame[["meter_id", "raw_median_power", "raw_q90_power"]], on="meter_id", how="left")
    power_index = central_u[["raw_median_power", "raw_q90_power"]].rank(pct=True).mean(axis=1)
    score_power_spearman = spearmanr(central_u["rank"], power_index, nan_policy="omit").statistic
    return {
        "legacy_oof_mean_rank": float(positive.u_percentile_rank.mean()),
        "legacy_oof_median_rank": float(positive.u_percentile_rank.median()),
        "top10_retrieval": float((positive.u_percentile_rank >= .90).mean()),
        "top20_retrieval": float((positive.u_percentile_rank >= .80).mean()),
        "minimum_subtype_mean_rank": float(subtype.min()),
        **{f"subtype_{name}_mean_rank": float(value) for name, value in subtype.items()},
        "power_matched_advantage": float(np.mean(advantage)),
        "power_matched_ci_low": ci_low,
        "power_matched_ci_high": ci_high,
        "absolute_high_power_legacy_count": int(len(high_power)),
        "absolute_high_power_legacy_mean_rank": float(high_power.mean()) if len(high_power) else np.nan,
        "score_power_spearman": float(score_power_spearman),
    }, matched


def fit_full(frame, family_name, family, scope, fit_column, medium_weight, prior, propensity=False,
             bags=18, algorithm="hgb"):
    legacy = load_legacy()
    subtype = legacy.set_index("meter_id").subtype.to_dict()
    fit_ids = set(scope.loc[scope[fit_column], "meter_id"])
    strict_zero = set(scope.loc[scope.strict_all_zero, "meter_id"])
    holdout = set(scope.loc[scope.future_holdout, "meter_id"])
    if fit_ids & holdout:
        raise RuntimeError("Future holdout leaked into final PU fit")
    positives = sorted((set(scope.loc[scope.legacy_high_hard | scope.legacy_medium_soft, "meter_id"]) & fit_ids) - strict_zero)
    u_pool = sorted(fit_ids - set(legacy.meter_id) - strict_zero)
    lookup = {meter: index for index, meter in enumerate(frame.meter_id)}
    p_rows = np.array([lookup[meter] for meter in positives])
    u_rows = np.array([lookup[meter] for meter in u_pool])
    propensity_values = None
    propensity_model = None
    if propensity:
        modified = scope.copy()
        modified["research_fit"] = modified[fit_column]
        propensity_values, propensity_model = fit_propensity(frame, modified, [])
    all_scores = []
    models = []
    for bag in range(bags):
        seed = SEED + bag * 19
        rng = np.random.default_rng(seed)
        sampled_u = rng.choice(u_rows, min(len(u_rows), max(500, len(p_rows) * 5)), replace=True)
        rows = np.r_[p_rows, sampled_u]
        target = np.r_[np.ones(len(p_rows)), np.zeros(len(sampled_u))]
        p_weight = positive_weights(positives, scope, subtype, medium_weight)
        if propensity_values is not None:
            adjusted = p_weight / propensity_values[p_rows]
            p_weight = np.clip(adjusted, 0, np.quantile(adjusted, .95))
        weights = np.r_[p_weight, np.full(len(sampled_u), u_comparison_weight(prior))]
        model = MixedEstimator(family["base"], family["embedding"], algorithm, seed)
        model.fit(frame.iloc[rows], target, weights)
        all_scores.append(model.predict_proba(frame))
        models.append(model)
    scores = np.stack(all_scores)
    mean_score = scores.mean(axis=0)
    fit_mask = frame.meter_id.isin(fit_ids - strict_zero).to_numpy()
    rank_reference = np.sort(mean_score[fit_mask & np.isfinite(mean_score)])
    ranks = np.searchsorted(rank_reference, mean_score, side="right") / max(1, len(rank_reference))
    output = pd.DataFrame({
        "meter_id": frame.meter_id, "score_mean": mean_score, "score_std": scores.std(axis=0),
        "rank": ranks, "family": family_name,
    })
    return output, models, propensity_model


def run_ablation():
    frame, families, regional = build_feature_table()
    scope = load_scope()
    configurations = [
        ("POWER_ONLY", False, "logistic"),
        ("ROBUST_FEATURES", False, "hgb"), ("DAILY_TCN_FUSION", False, "hgb"), ("LONG_HORIZON_TCN", False, "hgb"),
        ("CONTEXTUAL_TCN", False, "hgb"), ("LONG_HORIZON_GRU", False, "hgb"),
    ]
    metrics = []
    all_positive = []
    all_u = []
    all_matched = []
    for family_name, propensity, algorithm in configurations:
        positive, u_scores = crossfit(frame, family_name, families[family_name], scope, propensity=propensity, algorithm=algorithm)
        row, matched = evaluate_configuration(frame, positive, u_scores)
        metrics.append({"family": family_name, "head": f"bagging_{algorithm}", **row})
        all_positive.append(positive)
        all_u.append(u_scores)
        matched["family"] = family_name
        all_matched.append(matched)
        print(json.dumps({"completed": family_name, **row}), flush=True)
    base_metrics = pd.DataFrame(metrics)
    viable = base_metrics[
        ~base_metrics.family.eq("POWER_ONLY") &
        (base_metrics.power_matched_ci_low > 0) &
        (base_metrics.minimum_subtype_mean_rank >= base_metrics.loc[base_metrics.family.eq("ROBUST_FEATURES"), "minimum_subtype_mean_rank"].iloc[0] - .05) &
        (
            base_metrics.absolute_high_power_legacy_count.eq(0)
            | (base_metrics.absolute_high_power_legacy_mean_rank >= .60)
        )
    ].copy()
    if viable.empty:
        representation_winner = "ROBUST_FEATURES"
    else:
        representation_winner = viable.sort_values([
            "power_matched_advantage", "minimum_subtype_mean_rank", "legacy_oof_mean_rank"
        ], ascending=False).iloc[0].family
    propensity_family = {"base": families[representation_winner]["base"], "embedding": families[representation_winner]["embedding"]}
    families["PROPENSITY_PU"] = propensity_family
    positive, u_scores = crossfit(frame, "PROPENSITY_PU", propensity_family, scope, propensity=True)
    row, matched = evaluate_configuration(frame, positive, u_scores)
    metrics.append({"family": "PROPENSITY_PU", "head": "propensity_bagging_hgb", "representation_source": representation_winner, **row})
    all_positive.append(positive); all_u.append(u_scores); matched["family"] = "PROPENSITY_PU"; all_matched.append(matched)
    regional_family = {"base": propensity_family["base"] + regional, "embedding": propensity_family["embedding"]}
    families["REGIONAL_FACTOR"] = regional_family
    positive, u_scores = crossfit(frame, "REGIONAL_FACTOR", regional_family, scope, propensity=False)
    row, matched = evaluate_configuration(frame, positive, u_scores)
    metrics.append({"family": "REGIONAL_FACTOR", "head": "bagging_hgb", "representation_source": representation_winner, **row})
    all_positive.append(positive); all_u.append(u_scores); matched["family"] = "REGIONAL_FACTOR"; all_matched.append(matched)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(ROOT / "ablation_metrics.csv", index=False)
    pd.concat(all_positive).to_csv(ROOT / "legacy_oof_scores.csv", index=False)
    pd.concat(all_u).to_csv(ROOT / "u_oof_reference_scores.csv", index=False)
    pd.concat(all_matched).to_csv(ROOT / "power_matched_retrieval.csv", index=False)
    write_json(ROOT / "ablation_context.json", {
        "representation_winner_for_auxiliary_branches": representation_winner,
        "selection_note": "propensity/regional branches use the strongest representation passing preliminary hard gates; final selection occurs after full scorecard",
    })
    write_json(ROOT / "feature_families.json", {"families": families, "regional": regional})
    return frame, families, metrics_frame


if __name__ == "__main__":
    run_ablation()
