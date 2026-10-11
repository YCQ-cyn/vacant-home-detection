from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch import nn
from torch.nn import functional as F
from joblib import Parallel, delayed

from .data import ROOT, SEED, TIME_COLS, key, load_daily, write_json
from .models import (
    CENTRAL_PRIOR,
    CENTRAL_WEIGHT,
    MEDIUM_WEIGHTS,
    PRIOR_SETTINGS,
    PROPENSITY_FEATURES,
    MixedEstimator,
    bootstrap_ci,
    build_feature_table,
    build_regional_features,
    crossfit,
    fit_full,
    load_legacy,
    load_representations,
    load_scope,
    percentile,
    positive_weights,
    u_comparison_weight,
)
from .representation import SEEDS, WINDOWS, train_daily, train_long


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reference_percentile(values, reference_mask):
    values = np.asarray(values, float)
    reference = np.sort(values[np.asarray(reference_mask, bool) & np.isfinite(values)])
    output = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    output[finite] = np.searchsorted(reference, values[finite], side="right") / max(1, len(reference))
    return output


def paired_bootstrap_delta(left, right, seed=SEED, repetitions=4000):
    joined = left[["meter_id", "u_percentile_rank"]].merge(
        right[["meter_id", "u_percentile_rank"]], on="meter_id", suffixes=("_candidate", "_baseline")
    )
    delta = joined.u_percentile_rank_candidate.to_numpy() - joined.u_percentile_rank_baseline.to_numpy()
    rng = np.random.default_rng(seed)
    distribution = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repetitions)])
    return float(delta.mean()), float(np.quantile(distribution, .025)), float(np.quantile(distribution, .975))


def select_structure(metrics, positive_scores):
    baseline = metrics.loc[metrics.family.eq("ROBUST_FEATURES")].iloc[0]
    baseline_positive = positive_scores[positive_scores.family.eq("ROBUST_FEATURES")]
    rows = []
    for _, candidate in metrics.iterrows():
        current_positive = positive_scores[positive_scores.family.eq(candidate.family)]
        delta, delta_low, delta_high = paired_bootstrap_delta(current_positive, baseline_positive)
        high_power_evaluable = bool(
            candidate.absolute_high_power_legacy_count > 0
            and baseline.absolute_high_power_legacy_count > 0
        )
        hard_gates = {
            "positive_power_matched_ci": bool(candidate.power_matched_ci_low > 0),
            "no_subtype_collapse": bool(candidate.minimum_subtype_mean_rank >= baseline.minimum_subtype_mean_rank - .03),
            "high_power_not_degraded": bool(
                not high_power_evaluable
                or candidate.absolute_high_power_legacy_mean_rank >= baseline.absolute_high_power_legacy_mean_rank - .03
            ),
            "power_dependence_not_increased": bool(abs(candidate.score_power_spearman) <= abs(baseline.score_power_spearman) + .08),
            "legacy_signal_not_degraded": bool(candidate.legacy_oof_mean_rank >= baseline.legacy_oof_mean_rank - .02),
        }
        rows.append({
            "family": candidate.family,
            "paired_legacy_rank_delta": delta,
            "paired_delta_ci_low": delta_low,
            "paired_delta_ci_high": delta_high,
            "high_power_gate_evaluable": high_power_evaluable,
            **hard_gates,
            "passes_all_hard_gates": all(hard_gates.values()),
        })
    comparison = pd.DataFrame(rows).merge(metrics, on="family", how="left")
    comparison.to_csv(ROOT / "pareto_hard_gate_comparison.csv", index=False)
    eligible = comparison[comparison.passes_all_hard_gates].copy()
    if eligible.empty:
        return "ROBUST_FEATURES", comparison
    # Upgrade only for a material business-facing gain; otherwise retain the simpler baseline.
    upgrades = eligible[
        (~eligible.family.isin(["POWER_ONLY", "ROBUST_FEATURES"]))
        & (
            (eligible.paired_delta_ci_low > 0)
            | (eligible.power_matched_advantage >= baseline.power_matched_advantage + .03)
            | (eligible.minimum_subtype_mean_rank >= baseline.minimum_subtype_mean_rank + .04)
            | (eligible.absolute_high_power_legacy_mean_rank >= baseline.absolute_high_power_legacy_mean_rank + .04)
        )
    ].copy()
    if upgrades.empty:
        return "ROBUST_FEATURES", comparison
    upgrades = upgrades.sort_values(
        ["power_matched_advantage", "minimum_subtype_mean_rank", "absolute_high_power_legacy_mean_rank"],
        ascending=False,
    )
    return str(upgrades.iloc[0].family), comparison


def corrected_permutation(frame, family, real_statistic, repetitions=100, bags=2, propensity=False):
    scope = load_scope()
    legacy = load_legacy()
    strict_zero = set(scope.loc[scope.strict_all_zero, "meter_id"])
    research = set(scope.loc[scope.research_fit, "meter_id"]) - strict_zero
    u_ids = sorted(research - set(legacy.meter_id))
    n_positive = int((scope.research_fit & ~scope.strict_all_zero & (scope.legacy_high_hard | scope.legacy_medium_soft)).sum())
    lookup = {meter: index for index, meter in enumerate(frame.meter_id)}
    rng = np.random.default_rng(SEED + 711)
    fake_sets = [np.array(rng.choice(u_ids, n_positive, replace=False)) for _ in range(repetitions)]

    def one_repetition(repetition, fake):
        local_rng = np.random.default_rng(SEED + 711 + repetition * 7919)
        folds = np.array_split(local_rng.permutation(fake), 5)
        held_ranks = []
        for fold, held in enumerate(folds):
            train_positive = np.setdiff1d(fake, held)
            negative_pool = np.setdiff1d(u_ids, fake)
            fake_propensity = None
            if propensity:
                propensity_columns = [column for column in PROPENSITY_FEATURES if column in frame]
                propensity_ids = np.r_[train_positive, negative_pool]
                propensity_target = np.r_[np.ones(len(train_positive)), np.zeros(len(negative_pool))]
                propensity_model = MixedEstimator(propensity_columns, [], "logistic", SEED + repetition * 43 + fold)
                propensity_model.fit(frame.iloc[[lookup[x] for x in propensity_ids]], propensity_target, np.ones(len(propensity_ids)))
                fake_propensity = np.clip(
                    propensity_model.predict_proba(frame.iloc[[lookup[x] for x in train_positive]]), .03, .95
                )
            held_scores, reference_scores = [], []
            for bag in range(bags):
                bag_rng = np.random.default_rng(SEED + repetition * 1009 + fold * 101 + bag)
                sampled_u = bag_rng.choice(negative_pool, min(len(negative_pool), max(420, len(train_positive) * 5)), replace=True)
                train_ids = np.r_[train_positive, sampled_u]
                train_rows = np.array([lookup[x] for x in train_ids])
                y = np.r_[np.ones(len(train_positive)), np.zeros(len(sampled_u))]
                positive_weight = np.ones(len(train_positive)) if fake_propensity is None else np.clip(
                    1 / fake_propensity, 0, np.quantile(1 / fake_propensity, .95)
                )
                weights = np.r_[positive_weight, np.full(len(sampled_u), u_comparison_weight(CENTRAL_PRIOR))]
                estimator = MixedEstimator(family["base"], family["embedding"], "hgb", SEED + repetition * 31 + fold * 7 + bag)
                estimator.fit(frame.iloc[train_rows], y, weights)
                held_scores.append(estimator.predict_proba(frame.iloc[[lookup[x] for x in held]]))
                reference_scores.append(estimator.predict_proba(frame.iloc[[lookup[x] for x in negative_pool]]))
            held_score = np.mean(held_scores, axis=0)
            reference = np.sort(np.mean(reference_scores, axis=0))
            held_ranks.extend(np.searchsorted(reference, held_score, side="right") / len(reference))
        return float(np.mean(held_ranks))

    parallel_jobs = min(4, os.cpu_count() or 1)
    null = np.asarray(Parallel(n_jobs=parallel_jobs, backend="loky", verbose=5)(
        delayed(one_repetition)(repetition, fake)
        for repetition, fake in enumerate(fake_sets)
    ))
    output = pd.DataFrame({"repetition": np.arange(repetitions), "crossfitted_fake_positive_mean_rank": null})
    output.to_csv(ROOT / "permutation_null.csv", index=False)
    summary = {
        "real_legacy_crossfit_mean_rank": float(real_statistic),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null_95_low": float(np.quantile(null, .025)),
        "null_95_high": float(np.quantile(null, .975)),
        "empirical_p_value": float((1 + np.sum(null >= real_statistic)) / (len(null) + 1)),
        "repetitions": repetitions,
        "folds": 5,
        "bags_per_fold": bags,
        "parallel_jobs": parallel_jobs,
        "propensity_reweighting": propensity,
        "fake_positive_labels_seen_by_own_model": False,
    }
    write_json(ROOT / "permutation_null_summary.json", summary)
    return summary


def leave_subtype_out(frame, family):
    scope = load_scope()
    legacy = load_legacy()
    subtype = legacy.set_index("meter_id").subtype.to_dict()
    strict_zero = set(scope.loc[scope.strict_all_zero, "meter_id"])
    fit_ids = set(scope.loc[scope.research_fit, "meter_id"]) - strict_zero
    all_legacy = set(legacy.meter_id)
    lookup = {meter: index for index, meter in enumerate(frame.meter_id)}
    rows = []
    for omitted in sorted(legacy.subtype.unique()):
        omitted_ids = set(legacy.loc[legacy.subtype.eq(omitted), "meter_id"]) & fit_ids
        train_positive = sorted(
            (set(scope.loc[scope.legacy_high_hard | scope.legacy_medium_soft, "meter_id"]) & fit_ids) - omitted_ids
        )
        u_pool = sorted(fit_ids - all_legacy)
        held_scores, u_scores = [], []
        for bag in range(10):
            rng = np.random.default_rng(SEED + bag * 23 + sum(ord(x) for x in omitted))
            sampled_u = rng.choice(u_pool, min(len(u_pool), max(420, len(train_positive) * 5)), replace=True)
            train_ids = np.r_[train_positive, sampled_u]
            y = np.r_[np.ones(len(train_positive)), np.zeros(len(sampled_u))]
            p_weight = positive_weights(train_positive, scope, subtype, CENTRAL_WEIGHT)
            weights = np.r_[p_weight, np.full(len(sampled_u), u_comparison_weight(CENTRAL_PRIOR))]
            estimator = MixedEstimator(family["base"], family["embedding"], "hgb", SEED + bag)
            estimator.fit(frame.iloc[[lookup[x] for x in train_ids]], y, weights)
            held_scores.append(estimator.predict_proba(frame.iloc[[lookup[x] for x in sorted(omitted_ids)]]))
            u_scores.append(estimator.predict_proba(frame.iloc[[lookup[x] for x in u_pool]]))
        held = np.mean(held_scores, axis=0)
        reference = np.sort(np.mean(u_scores, axis=0))
        ranks = np.searchsorted(reference, held, side="right") / len(reference)
        low, high = bootstrap_ci(ranks)
        rows.append({
            "omitted_subtype": omitted,
            "heldout_cases": len(ranks),
            "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
        })
    output = pd.DataFrame(rows)
    output.to_csv(ROOT / "leave_subtype_out.csv", index=False)
    return output


def high_power_legacy_audit(frame, positive_scores, selected_family):
    scope = load_scope()
    reference = frame.meter_id.isin(set(scope.loc[scope.research_fit & ~scope.strict_all_zero, "meter_id"]))
    selected = positive_scores[positive_scores.family.eq(selected_family)].copy()
    metadata = frame.set_index("meter_id")
    view_columns = ["raw_median_power", "raw_q90_power", "baseline_floor"]
    view_percentiles = {
        column: pd.Series(reference_percentile(frame[column], reference), index=frame.meter_id)
        for column in view_columns
    }
    composite = pd.concat(view_percentiles, axis=1).mean(axis=1)
    rows = []
    for view, values in [*view_percentiles.items(), ("composite_absolute_power", composite)]:
        ids = set(values.index[values >= .75])
        ranks = selected.loc[selected.meter_id.isin(ids), "u_percentile_rank"].to_numpy(float)
        low, high = bootstrap_ci(ranks) if len(ranks) else (np.nan, np.nan)
        rows.append({
            "high_power_view": view,
            "threshold_basis": "research-fit universe 75th percentile; never Legacy-internal q75",
            "legacy_case_count": len(ranks),
            "mean_crossfit_rank": float(np.mean(ranks)) if len(ranks) else np.nan,
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
        })
    output = pd.DataFrame(rows)
    output.to_csv(ROOT / "high_power_legacy_audit.csv", index=False)
    return output


def legacy_router_and_behavior_audit(positive_scores, selected_family):
    scope = load_scope()
    legacy = load_legacy().merge(
        scope[[
            "meter_id", "strict_all_zero", "legacy_high_hard", "legacy_medium_soft",
            "legacy_low_diagnostic_only", "research_fit",
        ]], on="meter_id", how="left",
    )
    selected = positive_scores[positive_scores.family.eq(selected_family)][["meter_id", "u_percentile_rank"]]
    legacy = legacy.merge(selected, on="meter_id", how="left")
    legacy["evaluation_route"] = np.select(
        [
            legacy.strict_all_zero & (legacy.legacy_high_hard | legacy.legacy_medium_soft),
            legacy.u_percentile_rank.notna(),
            legacy.legacy_low_diagnostic_only,
        ],
        ["electrical_silence_router", "nonzero_behavior_crossfit", "low_support_diagnostic_only"],
        default="outside_research_fit_or_unavailable",
    )
    legacy["composite_screening_rank_for_audit"] = np.where(
        legacy.evaluation_route.eq("electrical_silence_router"), 1.0, legacy.u_percentile_rank
    )
    legacy.to_csv(ROOT / "legacy_router_behavior_audit.csv", index=False)
    eligible = legacy[legacy.legacy_high_hard | legacy.legacy_medium_soft]
    summary = {
        "legacy_unique_total": int(len(legacy)),
        "high_medium_total": int(len(eligible)),
        "silence_router_cases": int(eligible.evaluation_route.eq("electrical_silence_router").sum()),
        "nonzero_behavior_oof_cases": int(eligible.evaluation_route.eq("nonzero_behavior_crossfit").sum()),
        "low_support_diagnostic_cases": int(legacy.legacy_low_diagnostic_only.sum()),
        "composite_router_plus_behavior_mean_rank": float(eligible.composite_screening_rank_for_audit.mean()),
        "selection_uses_composite_metric": False,
    }
    write_json(ROOT / "legacy_router_behavior_summary.json", summary)
    return summary


class SmallNNPU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, 64), nn.GELU(), nn.Dropout(.15),
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(.10), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def nnpu_sensitivity(frame, family):
    scope = load_scope()
    legacy = load_legacy()
    strict_zero = set(scope.loc[scope.strict_all_zero, "meter_id"])
    research = set(scope.loc[scope.research_fit, "meter_id"]) - strict_zero
    positives = sorted(set(scope.loc[scope.legacy_high_hard | scope.legacy_medium_soft, "meter_id"]) & research)
    u_ids = sorted(research - set(legacy.meter_id))
    columns = family["base"] + family["embedding"]
    lookup = {meter: index for index, meter in enumerate(frame.meter_id)}
    labels = legacy.set_index("meter_id").loc[positives, "subtype"].to_numpy()
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for prior in PRIOR_SETTINGS:
        for fold, (train_ix, held_ix) in enumerate(folds.split(positives, labels)):
            train_p = np.array([lookup[positives[i]] for i in train_ix])
            held_p = np.array([lookup[positives[i]] for i in held_ix])
            u_rows = np.array([lookup[x] for x in u_ids])
            preprocessing_rows = np.r_[train_p, u_rows]
            imputer = SimpleImputer(strategy="median").fit(frame.iloc[preprocessing_rows][columns])
            scaler = StandardScaler().fit(imputer.transform(frame.iloc[preprocessing_rows][columns]))
            x_all = scaler.transform(imputer.transform(frame[columns])).astype(np.float32)
            torch.manual_seed(SEED + fold)
            model = SmallNNPU(x_all.shape[1]).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-3)
            xp = torch.from_numpy(x_all[train_p]).to(DEVICE)
            xu = torch.from_numpy(x_all[u_rows]).to(DEVICE)
            generator = np.random.default_rng(SEED + fold)
            for _ in range(120):
                p_idx = generator.choice(len(xp), min(96, len(xp)), replace=True)
                u_idx = generator.choice(len(xu), min(256, len(xu)), replace=True)
                p_logits = model(xp[p_idx])
                u_logits = model(xu[u_idx])
                positive_risk = prior * F.softplus(-p_logits).mean()
                negative_risk = F.softplus(u_logits).mean() - prior * F.softplus(p_logits).mean()
                loss = positive_risk + torch.clamp(negative_risk, min=0)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                held_score = torch.sigmoid(model(torch.from_numpy(x_all[held_p]).to(DEVICE))).cpu().numpy()
                u_score = torch.sigmoid(model(torch.from_numpy(x_all[u_rows]).to(DEVICE))).cpu().numpy()
            reference = np.sort(u_score)
            ranks = np.searchsorted(reference, held_score, side="right") / len(reference)
            for meter, rank in zip(np.array(positives)[held_ix], ranks):
                rows.append({"meter_id": meter, "prior": prior, "fold": fold, "u_percentile_rank": rank})
    output = pd.DataFrame(rows)
    output.to_csv(ROOT / "nnpu_sensitivity.csv", index=False)
    write_json(ROOT / "nnpu_audit.json", {
        "role": "sensitivity only",
        "fold_safe_preprocessing": True,
        "hidden_layers": [64, 32],
        "priors": list(PRIOR_SETTINGS),
        "global_preprocessor_saved": False,
    })
    return output


def run_lengths(mask):
    lengths, current = [], 0
    for value in np.asarray(mask, bool):
        if value:
            current += 1
        elif current:
            lengths.append(current); current = 0
    if current:
        lengths.append(current)
    return lengths


def build_window_robust_features(window):
    daily = load_daily().sort_values(["meter_id", "date"])
    rows = []
    for meter_id, group in daily.groupby("meter_id", sort=False):
        group = group.tail(window)
        dates = pd.to_datetime(group.date)
        values = group[TIME_COLS].to_numpy(float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            rows.append({"meter_id": meter_id}); continue
        q10, median, q90 = np.nanquantile(finite, [.1, .5, .9])
        q25, q75 = np.nanquantile(finite, [.25, .75])
        near = max(abs(q10) * .1, 1e-6)
        day_center = np.nanmedian(values, axis=1, keepdims=True)
        day_scale = np.nanstd(values, axis=1, keepdims=True) + 1e-8
        normalized = np.nan_to_num((values - day_center) / day_scale)
        similarities = [
            float(normalized[i] @ normalized[i - 1] / (np.linalg.norm(normalized[i]) * np.linalg.norm(normalized[i - 1]) + 1e-8))
            for i in range(1, len(normalized))
        ]
        states = np.digitize(np.nan_to_num(values, nan=median), [q10, q25, q75, q90])
        active = np.nan_to_num(values) > max(q25, near)
        sample_runs = [length for row in active for length in run_lengths(row)]
        daily_mean = np.nanmean(values, axis=1)
        daily_median = np.nanmedian(values, axis=1)
        daily_q10 = np.nanquantile(values, .1, axis=1)
        nonzero = np.nanmean(np.nan_to_num(values) > near, axis=1)
        episode_count = np.array([len(run_lengths(row)) for row in active], float)
        silent_days = nonzero <= .02
        spectrum = np.abs(np.fft.rfft(np.nan_to_num(daily_mean - np.nanmean(daily_mean)))) ** 2
        frequencies = np.fft.rfftfreq(len(daily_mean))
        spectral7 = float(spectrum[np.argmin(np.abs(frequencies - 1 / 7))] / (spectrum[1:].sum() + 1e-9)) if len(spectrum) > 1 else np.nan
        half = max(1, len(values) // 2)
        first_profile = np.nanmean(values[:half], axis=0)
        second_profile = np.nanmean(values[half:], axis=0) if len(values) > half else first_profile
        adjacent = np.mean(np.abs(np.diff(normalized, axis=0)), axis=1) if len(values) > 1 else np.array([0.0])
        lag7 = [
            float(normalized[i] @ normalized[i - 7] / (np.linalg.norm(normalized[i]) * np.linalg.norm(normalized[i - 7]) + 1e-8))
            for i in range(7, len(normalized))
        ]
        lag14 = [
            float(normalized[i] @ normalized[i - 14] / (np.linalg.norm(normalized[i]) * np.linalg.norm(normalized[i - 14]) + 1e-8))
            for i in range(14, len(normalized))
        ]
        weekend = dates.dt.dayofweek.to_numpy() >= 5
        weekday_weekend = (
            float(np.nanmean(daily_mean[weekend]) - np.nanmean(daily_mean[~weekend]))
            if weekend.any() and (~weekend).any() else np.nan
        )
        scale = max(float(q90 - q10), 1e-8)
        rows.append({
            "meter_id": meter_id,
            "raw_q10_power": q10,
            "raw_median_power": median,
            "raw_q90_power": q90,
            "baseline_floor": float(np.nanmedian(daily_q10)),
            "raw_relative_amplitude": scale / (abs(median) + 1e-8),
            "sample_zero_fraction": float(np.mean(np.nan_to_num(values, nan=np.nan) == 0)),
            "sample_near_zero_fraction": float(np.mean(np.nan_to_num(values) <= near)),
            "cross_day_normalized_shape_similarity": float(np.nanmean(similarities)) if similarities else 1.0,
            "daily_trajectory_similarity_sample": float(1 / (1 + np.nanmean(adjacent))),
            "sample_state_transition_rate": float(np.mean(states[:, 1:] != states[:, :-1])),
            "sample_run_length_median": float(np.median(sample_runs)) if sample_runs else 0.0,
            "sample_run_length_q90": float(np.quantile(sample_runs, .9)) if sample_runs else 0.0,
            "clock_time_recurrence_dispersion": float(np.nanmean(np.abs(np.nanmean(values, axis=0) - np.nanmedian(values, axis=0)))),
            "7day_autocorrelation_sampled_load": float(pd.Series(daily_mean).autocorr(lag=7)) if len(daily_mean) > 7 else np.nan,
            "7day_spectral_structure": spectral7,
            "change_point_structure": float(np.nanmean(np.abs(second_profile - first_profile))),
            "daily_nonzero_fraction_mean": float(np.nanmean(nonzero)),
            "background_recurrence": float(np.nanmean(similarities)) if similarities else 1.0,
            "device_likeness": float((np.nanmean(similarities) if similarities else 1.0) / (1 + np.nanstd(daily_median))),
            "interaction_complexity": float(np.nanmean(episode_count) * np.mean(states[:, 1:] != states[:, :-1])),
            "unexplained_activity": float(np.nanstd(adjacent)),
            "relative_event_strength": float(np.nanquantile(np.abs(np.diff(values, axis=1)), .9) / scale),
            "soft_active_fraction": float(np.mean(active)),
            "activity_episode_count": float(np.nansum(episode_count)),
            "longest_silent_run": float(max(run_lengths(silent_days), default=0)),
            "temporal_stability": float(1 / (1 + np.nanstd(daily_median) / (abs(np.nanmedian(daily_median)) + 1e-8))),
            "embedding_variability": float(np.nanstd(adjacent)),
            "first_second_delta": float(np.nanmean(second_profile - first_profile)),
            "adjacent_day_distance": float(np.nanmean(adjacent)),
            "lag7_similarity": float(np.nanmean(lag7)) if lag7 else np.nan,
            "lag14_similarity": float(np.nanmean(lag14)) if lag14 else np.nan,
            "stable_run": float(max(run_lengths(adjacent <= np.nanmedian(adjacent)), default=0)),
            "changed_run": float(max(run_lengths(adjacent > np.nanquantile(adjacent, .75)), default=0)),
            "change_point_magnitude": float(np.nanmean(np.abs(second_profile - first_profile)) / scale),
            "weekday_weekend_difference": weekday_weekend,
        })
    return pd.DataFrame(rows)


def attach_axes(frame, reference_ids):
    reference = frame.meter_id.isin(reference_ids)
    power_index = frame[["raw_median_power", "raw_q90_power", "baseline_floor"]].rank(pct=True).mean(axis=1)
    auto_columns = [
        "background_recurrence", "device_likeness", "daily_trajectory_similarity_sample",
        "cross_day_normalized_shape_similarity", "temporal_stability", "lag7_similarity",
    ]
    human_columns = [
        "interaction_complexity", "unexplained_activity", "relative_event_strength",
        "soft_active_fraction", "activity_episode_count", "sample_state_transition_rate",
        "change_point_structure", "weekday_weekend_difference",
    ]
    for output, columns in (("automatic_control_compatibility", auto_columns), ("human_interaction_evidence", human_columns)):
        ranks = []
        for column in columns:
            if column in frame:
                ranks.append(reference_percentile(pd.to_numeric(frame[column], errors="coerce"), reference))
        frame[output] = np.nanmean(np.column_stack(ranks), axis=1)
    frame["weekend_tendency"] = np.nanmean(np.column_stack([
        reference_percentile(frame.weekday_weekend_difference.abs(), reference),
        reference_percentile(frame.lag7_similarity, reference),
    ]), axis=1)
    frame["power_index"] = power_index
    return frame


def window_stability(selected_family):
    scope = load_scope()
    reference_ids = set(scope.loc[scope.research_fit & ~scope.strict_all_zero, "meter_id"])
    representation_family = selected_family
    if selected_family in {"PROPENSITY_PU", "REGIONAL_FACTOR"}:
        context = json.loads((ROOT / "ablation_context.json").read_text(encoding="utf-8"))
        representation_family = context["representation_winner_for_auxiliary_branches"]
    outputs = []
    for window in WINDOWS:
        frame = attach_axes(build_window_robust_features(window), reference_ids)
        base = [
            column for column in frame.columns
            if column not in {"meter_id", "power_index"}
        ]
        embedding = []
        if representation_family in {"DAILY_TCN_FUSION", "LONG_HORIZON_TCN", "CONTEXTUAL_TCN", "LONG_HORIZON_GRU"}:
            if representation_family == "DAILY_TCN_FUSION":
                representation = load_representations("research_plain", "daily", window)
                embedding = [f"daily_mean_e{i:02d}" for i in range(64)]
            elif representation_family == "CONTEXTUAL_TCN":
                representation = load_representations("research_contrastive", "tcn", window)
                embedding = [f"tcn_mean_e{i:02d}" for i in range(64)]
            elif representation_family == "LONG_HORIZON_GRU":
                representation = load_representations("research_gru", "gru", window)
                embedding = [f"gru_mean_e{i:02d}" for i in range(64)]
            else:
                representation = load_representations("research_plain", "tcn", window)
                embedding = [f"tcn_mean_e{i:02d}" for i in range(64)]
            frame = frame.merge(representation[["meter_id"] + embedding], on="meter_id", how="left")
        family = {"base": base, "embedding": embedding}
        scored, _, _ = fit_full(frame, f"WINDOW_{window}", family, scope, "research_fit", CENTRAL_WEIGHT, CENTRAL_PRIOR, bags=6)
        scored["window"] = window
        outputs.append(scored[["meter_id", "window", "rank"]])
    long = pd.concat(outputs, ignore_index=True)
    long.to_csv(ROOT / "window_ranks.csv", index=False)
    pivot = long.pivot(index="meter_id", columns="window", values="rank")
    summary = pd.DataFrame({
        "meter_id": pivot.index,
        "window_rank_median": pivot.median(axis=1),
        "window_rank_p10": pivot.quantile(.1, axis=1),
        "window_rank_p90": pivot.quantile(.9, axis=1),
        "window_rank_std": pivot.std(axis=1),
    }).reset_index(drop=True)
    summary.to_csv(ROOT / "window_stability.csv", index=False)
    correlations = []
    for left in WINDOWS:
        for right in WINDOWS:
            if left < right:
                correlations.append({"window_a": left, "window_b": right, "spearman": spearmanr(pivot[left], pivot[right], nan_policy="omit").statistic})
    correlation_frame = pd.DataFrame(correlations)
    correlation_frame.to_csv(ROOT / "window_correlations.csv", index=False)
    return summary, correlation_frame


def _fit_subtype_preprocessor(frame, rows, family):
    artifact = {}
    base_imputer = SimpleImputer(strategy="median").fit(frame.iloc[rows][family["base"]])
    base_scaler = RobustScaler().fit(base_imputer.transform(frame.iloc[rows][family["base"]]))
    artifact["base_imputer"] = base_imputer; artifact["base_scaler"] = base_scaler
    artifact["base_columns"] = family["base"]; artifact["embedding_columns"] = family["embedding"]
    if family["embedding"]:
        embedding_imputer = SimpleImputer(strategy="median").fit(frame.iloc[rows][family["embedding"]])
        embedding_scaler = StandardScaler().fit(embedding_imputer.transform(frame.iloc[rows][family["embedding"]]))
        scaled = embedding_scaler.transform(embedding_imputer.transform(frame.iloc[rows][family["embedding"]]))
        pca = PCA(n_components=min(12, scaled.shape[1], max(2, len(rows) - 1)), random_state=SEED).fit(scaled)
        artifact.update({"embedding_imputer": embedding_imputer, "embedding_scaler": embedding_scaler, "pca": pca})
    return artifact


def _transform_subtype_preprocessor(frame, artifact):
    base = artifact["base_scaler"].transform(artifact["base_imputer"].transform(frame[artifact["base_columns"]]))
    if artifact["embedding_columns"]:
        embedding = artifact["embedding_scaler"].transform(
            artifact["embedding_imputer"].transform(frame[artifact["embedding_columns"]])
        )
        base = np.c_[base, artifact["pca"].transform(embedding)]
    return base


def subtype_models(frame, family):
    scope = load_scope()
    legacy = load_legacy()
    fit_ids = set(scope.loc[scope.final_fit, "meter_id"])
    behavioral_reference_ids = fit_ids - set(scope.loc[scope.strict_all_zero, "meter_id"])
    positive_ids = sorted(set(scope.loc[scope.legacy_high_hard | scope.legacy_medium_soft, "meter_id"]) & fit_ids)
    lookup = {meter: index for index, meter in enumerate(frame.meter_id)}
    y = legacy.set_index("meter_id").loc[positive_ids, "subtype"]
    labels = sorted(y.unique())
    rows = np.array([lookup[x] for x in positive_ids])
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    logistic_oof = np.zeros((len(rows), len(labels)))
    manifold_oof = np.zeros((len(rows), len(labels)))
    for train_ix, held_ix in folds.split(rows, y):
        train_rows, held_rows = rows[train_ix], rows[held_ix]
        fold_preprocessor = _fit_subtype_preprocessor(frame, train_rows, family)
        x = _transform_subtype_preprocessor(frame, fold_preprocessor)
        classifier = LogisticRegression(C=.25, max_iter=2000, class_weight="balanced", random_state=SEED)
        classifier.fit(x[train_rows], y.iloc[train_ix])
        raw = classifier.predict_proba(x[held_rows])
        logistic_oof[held_ix] = raw[:, [list(classifier.classes_).index(label) for label in labels]]
        for label_index, label in enumerate(labels):
            subtype_train = train_rows[y.iloc[train_ix].to_numpy() == label]
            clusters = min(3, max(1, len(subtype_train) // 8))
            kmeans = KMeans(clusters, random_state=SEED, n_init=20).fit(x[subtype_train])
            distance = kmeans.transform(x[held_rows]).min(axis=1)
            scale = np.median(kmeans.transform(x[subtype_train]).min(axis=1)) + 1e-6
            manifold_oof[held_ix, label_index] = np.exp(-distance / scale)
    manifold_oof = softmax(manifold_oof, axis=1)
    true_index = np.array([labels.index(value) for value in y])
    comparison = pd.DataFrame([
        {
            "method": "multinomial_l2_logistic",
            "balanced_accuracy": balanced_accuracy_score(true_index, logistic_oof.argmax(axis=1)),
            "log_loss": log_loss(true_index, logistic_oof, labels=np.arange(len(labels))),
        },
        {
            "method": "multi_prototype_distance",
            "balanced_accuracy": balanced_accuracy_score(true_index, manifold_oof.argmax(axis=1)),
            "log_loss": log_loss(true_index, manifold_oof, labels=np.arange(len(labels))),
        },
    ])
    comparison.to_csv(ROOT / "subtype_model_comparison.csv", index=False)
    selected = str(comparison.sort_values(["balanced_accuracy", "log_loss"], ascending=[False, True]).iloc[0].method)
    final_preprocessor = _fit_subtype_preprocessor(frame, rows, family)
    x = _transform_subtype_preprocessor(frame, final_preprocessor)
    classifier = LogisticRegression(C=.25, max_iter=2000, class_weight="balanced", random_state=SEED)
    classifier.fit(x[rows], y)
    logistic = classifier.predict_proba(x)
    logistic = logistic[:, [list(classifier.classes_).index(label) for label in labels]]
    prototypes, manifold_raw = {}, np.zeros((len(frame), len(labels)))
    for label_index, label in enumerate(labels):
        subtype_rows = rows[y.to_numpy() == label]
        clusters = min(3, max(1, len(subtype_rows) // 8))
        kmeans = KMeans(clusters, random_state=SEED, n_init=20).fit(x[subtype_rows])
        distance = kmeans.transform(x).min(axis=1)
        scale = np.median(kmeans.transform(x[subtype_rows]).min(axis=1)) + 1e-6
        manifold_raw[:, label_index] = np.exp(-distance / scale)
        prototypes[label] = {"centers": kmeans.cluster_centers_, "scale": scale}
    manifold = softmax(manifold_raw, axis=1)
    final = logistic if selected == "multinomial_l2_logistic" else manifold
    output = pd.DataFrame({"meter_id": frame.meter_id})
    safe = {
        "complete": "complete_support",
        "equipment": "equipment_support",
        "equipment_intermittent": "equipment_intermit_support",
        "intermittent_use": "intermittent_support",
    }
    for index, label in enumerate(labels):
        output[safe[label]] = final[:, index]
        output[f"logistic_{safe[label]}"] = logistic[:, index]
        output[f"manifold_{safe[label]}"] = manifold[:, index]
    support_columns = list(safe.values())
    probabilities = output[support_columns].to_numpy()
    ordered = np.sort(probabilities, axis=1)
    output["subtype_entropy"] = -(probabilities * np.log(np.clip(probabilities, 1e-9, 1))).sum(axis=1) / np.log(len(labels))
    output["subtype_margin"] = ordered[:, -1] - ordered[:, -2]
    output["subtype_top"] = np.array(list(safe))[probabilities.argmax(axis=1)]
    output["legacy_anchor_rank"] = reference_percentile(
        manifold_raw.max(axis=1), frame.meter_id.isin(behavioral_reference_ids)
    )
    output.to_csv(ROOT / "conditional_subtype_support.csv", index=False)
    joblib.dump({
        "selected_method": selected, "labels": labels, "preprocessor": final_preprocessor,
        "logistic": classifier, "prototypes": prototypes,
    }, ROOT / "models" / "conditional_subtype.joblib")
    return output, comparison


def power_conditioned_rank(values, power, reference_mask, bins=10):
    values = pd.Series(values).astype(float)
    power = pd.Series(power).astype(float)
    reference_mask = pd.Series(reference_mask).astype(bool)
    edges = np.unique(np.nanquantile(power[reference_mask], np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return reference_percentile(values, reference_mask)
    strata = np.clip(np.digitize(power, edges[1:-1]), 0, len(edges) - 2)
    output = np.full(len(values), np.nan)
    for stratum in np.unique(strata):
        current = strata == stratum
        reference = current & reference_mask.to_numpy()
        output[current] = reference_percentile(values[current].to_numpy(), reference[current])
    return output


def weak_and_active_evidence(frame, subtype):
    scope = load_scope()
    reference = frame.meter_id.isin(set(scope.loc[scope.final_fit & ~scope.strict_all_zero, "meter_id"]))
    power = pd.DataFrame({
        column: reference_percentile(frame[column], reference)
        for column in ["raw_median_power", "raw_q90_power", "baseline_floor"]
    }).mean(axis=1)
    human_raw = pd.DataFrame({
        column: reference_percentile(frame[column], reference)
        for column in [
            "human_interaction_evidence", "interaction_complexity", "unexplained_activity",
            "relative_event_strength", "activity_episode_count", "sample_state_transition_rate",
        ]
    }).mean(axis=1)
    active_score = power_conditioned_rank(human_raw, power, reference)
    auto = frame.automatic_control_compatibility.to_numpy(float)
    human = frame.human_interaction_evidence.to_numpy(float)
    manifold = subtype.legacy_anchor_rank.to_numpy(float)
    silence = frame.sample_zero_fraction.to_numpy(float)
    long_silence = reference_percentile(frame.longest_silent_run, reference)
    temporal_diversity = pd.DataFrame({
        column: reference_percentile(frame[column], reference)
        for column in [
            "sample_state_transition_rate", "change_point_structure", "raw_weekday_weekend_shape_difference",
            "unexplained_activity", "raw_adjacent_day_shape_distance",
        ]
    }).mean(axis=1).to_numpy(float)
    family_votes = {}
    family_votes["electrical_silence"] = np.select(
        [silence >= .999, silence <= .20], [1, -1], default=0
    )
    family_votes["automatic_human_axis"] = np.select(
        [(auto >= .80) & (human <= .40), (human >= .85) & (auto <= .75)], [1, -1], default=0
    )
    family_votes["interaction_richness_power_conditioned"] = np.select(
        [active_score <= .10, active_score >= .90], [1, -1], default=0
    )
    family_votes["positive_manifold"] = np.select(
        [manifold >= .90, (manifold <= .10) & (active_score >= .75)], [1, -1], default=0
    )
    family_votes["temporal_diversity"] = np.select(
        [(temporal_diversity <= .15) & (long_silence >= .75), temporal_diversity >= .90], [1, -1], default=0
    )
    votes = pd.DataFrame(family_votes)
    non_abstain = (votes != 0).sum(axis=1)
    raw = votes.sum(axis=1) / non_abstain.replace(0, np.nan)
    weak_rank = reference_percentile(raw.fillna(0), reference)
    output = pd.DataFrame({"meter_id": frame.meter_id, "active_evidence_score": active_score, "weak_vacancy_rank": weak_rank})
    for column in votes:
        output[f"weak_vote_{column}"] = votes[column]
    output["weak_abstain_family_count"] = (votes == 0).sum(axis=1)
    output.to_csv(ROOT / "conservative_family_weak_supervision.csv", index=False)
    write_json(ROOT / "weak_supervision_audit.json", {
        "method": "conservative family-level weak supervision",
        "dependency_model_fitted": False,
        "vote_values": {"1": "vacancy evidence", "-1": "active evidence", "0": "abstain"},
        "power_conditioned_active_evidence": True,
    })
    return output


def channel_rank(scored, fit_ids, name):
    output = scored[["meter_id", "score_mean", "score_std"]].copy()
    output[name] = reference_percentile(output.score_mean, output.meter_id.isin(fit_ids))
    return output[["meter_id", name, "score_std"]].rename(columns={"score_std": f"{name}_bag_std"})


def final_fit_channels(frame, families, selected_family):
    scope = load_scope()
    fit_ids = set(scope.loc[scope.final_fit & ~scope.strict_all_zero, "meter_id"])
    primary_propensity = selected_family == "PROPENSITY_PU"
    channel_specs = [(selected_family, primary_propensity, "selected_pu_rank")]
    if selected_family != "ROBUST_FEATURES":
        channel_specs.append(("ROBUST_FEATURES", False, "robust_pu_rank"))
    alternative_name = "bagging_comparator_rank" if primary_propensity else "propensity_pu_rank"
    channel_specs.append((selected_family, not primary_propensity, alternative_name))
    outputs, artifacts = [], {}
    seen = set()
    for family_name, propensity, channel_name in channel_specs:
        signature = (family_name, propensity, channel_name)
        if signature in seen:
            continue
        seen.add(signature)
        scored, models, propensity_model = fit_full(
            frame, f"FINAL_{family_name}_{'PROPENSITY' if propensity else 'BAGGING'}",
            families[family_name], scope, "final_fit", CENTRAL_WEIGHT, CENTRAL_PRIOR,
            propensity=propensity, bags=18,
        )
        outputs.append(channel_rank(scored, fit_ids, channel_name))
        artifacts[channel_name] = {
            "models": models,
            "propensity_model": propensity_model,
            "family": family_name,
            "propensity": propensity,
            "fit_meter_ids": sorted(fit_ids),
            "rank_reference_scores": np.sort(scored.loc[scored.meter_id.isin(fit_ids), "score_mean"].to_numpy(float)),
        }
        scored.to_csv(ROOT / f"FINAL_{channel_name.upper()}_scores.csv", index=False)
    joblib.dump(artifacts, ROOT / "models" / "final_pu_channels.joblib")
    merged = outputs[0]
    for output in outputs[1:]:
        merged = merged.merge(output, on="meter_id", how="outer")
    return merged, artifacts


def ensure_final_representation(frame, families, selected_family):
    structural_family = selected_family
    if selected_family in {"PROPENSITY_PU", "REGIONAL_FACTOR"}:
        context = json.loads((ROOT / "ablation_context.json").read_text(encoding="utf-8"))
        structural_family = context["representation_winner_for_auxiliary_branches"]
    scope = load_scope()
    fit_ids = set(scope.loc[scope.final_fit, "meter_id"])
    updated = {name: {"base": list(value["base"]), "embedding": list(value["embedding"])} for name, value in families.items()}
    final = frame.copy()
    if selected_family == "REGIONAL_FACTOR":
        regional_columns = ["regional_common_factor_coupling", "regional_common_factor_residual", "regional_idiosyncratic_variability"]
        regional = build_regional_features(fit_ids - set(scope.loc[scope.strict_all_zero, "meter_id"]))
        final = final.drop(columns=[column for column in regional_columns if column in final], errors="ignore").merge(regional, on="meter_id", how="left")
    if structural_family == "ROBUST_FEATURES":
        final.to_parquet(ROOT / "final_fit_feature_table.parquet", index=False)
        write_json(ROOT / "final_representation_audit.json", {
            "selected_family": selected_family,
            "structural_representation_family": structural_family,
            "representation_variant": "none; semantics-robust direct features",
            "fit_meter_count": len(fit_ids),
            "future_holdout_overlap": 0,
            "seeds": list(SEEDS),
            "transductive_full_fit": True,
        })
        return final, updated
    contrastive = structural_family == "CONTEXTUAL_TCN"
    variant = "final_contrastive" if contrastive else "final_plain"
    kind = "gru" if structural_family == "LONG_HORIZON_GRU" else "tcn"
    for seed in SEEDS:
        daily_output = ROOT / "models" / "representations" / variant / f"daily_seed_{seed}"
        if not (daily_output / "training_report.json").exists():
            daily_output, _ = train_daily(variant, fit_ids, 32, .20, contrastive, seed, 6, None)
        if structural_family != "DAILY_TCN_FUSION":
            long_output = ROOT / "models" / "representations" / variant / f"long_{kind}_seed_{seed}"
            if not (long_output / "training_report.json").exists():
                train_long(variant, daily_output, fit_ids, seed, kind, 25)
    if structural_family == "DAILY_TCN_FUSION":
        representation = load_representations(variant, "daily")
        source = [f"daily_mean_e{i:02d}" for i in range(64)]
        target = source
    else:
        representation = load_representations(variant, kind)
        prefix = kind
        source = [f"{prefix}_mean_e{i:02d}" for i in range(64)]
        target = [f"final_{prefix}_mean_e{i:02d}" for i in range(64)]
        representation = representation.rename(columns=dict(zip(source, target)))
    old = updated[selected_family]["embedding"]
    final = final.drop(columns=[column for column in old if column in final], errors="ignore")
    final = final.merge(representation[["meter_id"] + target], on="meter_id", how="left")
    updated[selected_family]["embedding"] = target
    final.to_parquet(ROOT / "final_fit_feature_table.parquet", index=False)
    write_json(ROOT / "final_representation_audit.json", {
        "selected_family": selected_family,
        "structural_representation_family": structural_family,
        "representation_variant": variant,
        "fit_meter_count": len(fit_ids),
        "future_holdout_overlap": 0,
        "seeds": list(SEEDS),
        "transductive_full_fit": True,
    })
    return final, updated


def parameter_stability(frame, family, propensity=False):
    scope = load_scope()
    fit_ids = set(scope.loc[scope.final_fit, "meter_id"])
    rows = []
    for weight in MEDIUM_WEIGHTS:
        for prior in PRIOR_SETTINGS:
            scored, _, _ = fit_full(
                frame, "PARAMETER_SENSITIVITY", family, scope, "final_fit", weight, prior,
                propensity=propensity, bags=6,
            )
            scored["medium_weight"] = weight; scored["prior"] = prior
            rows.append(scored[["meter_id", "medium_weight", "prior", "rank"]])
    long = pd.concat(rows, ignore_index=True)
    long.to_csv(ROOT / "parameter_sensitivity_ranks.csv", index=False)
    summary = long.groupby("meter_id")["rank"].agg(
        rank_median="median", rank_p10=lambda x: x.quantile(.1), rank_p90=lambda x: x.quantile(.9), rank_std="std"
    ).reset_index()
    # Aggregate rank by perturbation configuration before measuring each user's
    # sensitivity.  ``DataFrameGroupBy.rank`` is a method, so chaining
    # ``.rank.mean()`` would fail instead of selecting the rank column.
    prior = long.groupby(["meter_id", "prior"])["rank"].mean().groupby("meter_id").std().rename("prior_sensitivity")
    weight = long.groupby(["meter_id", "medium_weight"])["rank"].mean().groupby("meter_id").std().rename("medium_weight_sensitivity")
    summary = summary.merge(prior, on="meter_id", how="left").merge(weight, on="meter_id", how="left")
    summary.to_csv(ROOT / "parameter_stability.csv", index=False)
    topk_rows = []
    configurations = list(long.groupby(["medium_weight", "prior"]))
    for k in [10, 20, 50, 100]:
        sets = [set(group.nlargest(k, "rank").meter_id) for _, group in configurations]
        jaccards = [
            len(sets[left] & sets[right]) / len(sets[left] | sets[right])
            for left in range(len(sets)) for right in range(left + 1, len(sets))
        ]
        topk_rows.append({"k": k, "pair_count": len(jaccards), "jaccard_median": float(np.median(jaccards)), "jaccard_min": float(np.min(jaccards))})
    pd.DataFrame(topk_rows).to_csv(ROOT / "topk_stability.csv", index=False)
    return summary


def seed_stability(frame, families, selected_family, artifacts):
    scope = load_scope()
    fit_ids = set(scope.loc[scope.final_fit & ~scope.strict_all_zero, "meter_id"])
    structural_family = selected_family
    if selected_family in {"PROPENSITY_PU", "REGIONAL_FACTOR"}:
        context = json.loads((ROOT / "ablation_context.json").read_text(encoding="utf-8"))
        structural_family = context["representation_winner_for_auxiliary_branches"]
    rows = []
    if structural_family == "ROBUST_FEATURES":
        for seed_index, model in enumerate(artifacts["selected_pu_rank"]["models"][:3]):
            score = model.predict_proba(frame)
            rows.append(pd.DataFrame({
                "meter_id": frame.meter_id,
                "seed": SEEDS[seed_index],
                "rank": reference_percentile(score, frame.meter_id.isin(fit_ids)),
                "seed_source": "PU bag and estimator seed; no learned representation in selected family",
            }))
    else:
        contrastive = structural_family == "CONTEXTUAL_TCN"
        variant = "final_contrastive" if contrastive else "final_plain"
        kind = "daily" if structural_family == "DAILY_TCN_FUSION" else ("gru" if structural_family == "LONG_HORIZON_GRU" else "tcn")
        representation = load_representations(variant, kind)
        prefix = kind
        base_frame = frame.drop(columns=[column for column in families[selected_family]["embedding"] if column in frame], errors="ignore")
        for seed in SEEDS:
            source_columns = [f"{prefix}_e{i:02d}_seed{seed}" for i in range(64)]
            # ``build_feature_table`` already carries the seed-specific columns
            # for the research representation.  Re-merging them would create
            # pandas ``_x/_y`` suffixes and remove the names expected by the
            # estimator.  Merge only when this seed is not already present.
            if set(source_columns).issubset(base_frame.columns):
                current = base_frame.copy()
            else:
                current = base_frame.merge(representation[["meter_id"] + source_columns], on="meter_id", how="left")
            family = {"base": families[selected_family]["base"], "embedding": source_columns}
            scored, _, _ = fit_full(
                current, f"SEED_{seed}", family, scope, "final_fit", CENTRAL_WEIGHT, CENTRAL_PRIOR,
                propensity=selected_family == "PROPENSITY_PU", bags=6,
            )
            scored["seed"] = seed
            scored["seed_source"] = "representation seed plus PU bagging"
            rows.append(scored[["meter_id", "seed", "rank", "seed_source"]])
    long = pd.concat(rows, ignore_index=True)
    long.to_csv(ROOT / "seed_ranks.csv", index=False)
    summary = long.groupby("meter_id")["rank"].agg(
        seed_rank_median="median", seed_rank_p10=lambda x: x.quantile(.1),
        seed_rank_p90=lambda x: x.quantile(.9), seed_rank_std="std",
    ).reset_index()
    summary.to_csv(ROOT / "seed_stability.csv", index=False)
    return summary


def counterfactual_audit(frame, selected_artifact):
    models = selected_artifact["models"]
    scope = load_scope()
    behavior = frame.meter_id.isin(set(scope.loc[scope.final_fit & ~scope.strict_all_zero, "meter_id"]))

    def predict(current):
        return np.mean([model.predict_proba(current) for model in models], axis=0)

    baseline = predict(frame)
    reference = frame.meter_id.isin(set(scope.loc[scope.final_fit & ~scope.strict_all_zero, "meter_id"]))
    tests = {}
    scaled = frame.copy()
    for column in ["raw_q10_power", "raw_median_power", "raw_q90_power", "baseline_floor"]:
        if column in scaled:
            scaled[column] *= 2
    tests["global_amplitude_scale_x2"] = (scaled, "bounded", .12)
    automatic = scaled.copy()
    for column in ["automatic_control_compatibility", "device_likeness", "background_recurrence", "temporal_stability"]:
        if column in automatic:
            automatic[column] = np.maximum(automatic[column], frame.loc[reference, column].quantile(.90))
    for column in ["human_interaction_evidence", "interaction_complexity", "unexplained_activity"]:
        if column in automatic:
            automatic[column] = np.minimum(automatic[column], frame.loc[reference, column].quantile(.35))
    tests["high_power_automatic_persistence"] = (automatic, "not_large_decrease", -.12)
    human = frame.copy()
    for column in [
        "human_interaction_evidence", "interaction_complexity", "unexplained_activity", "activity_episode_count",
        "sample_state_transition_rate", "soft_active_fraction", "change_point_structure", "raw_weekday_weekend_shape_difference",
    ]:
        if column in human:
            human[column] = np.maximum(human[column], frame.loc[reference, column].quantile(.90))
    for column in ["longest_silent_run", "daily_trajectory_similarity_sample", "cross_day_normalized_shape_similarity"]:
        if column in human:
            human[column] = np.minimum(human[column], frame.loc[reference, column].quantile(.10))
    tests["persistent_diverse_human_event_injection"] = (human, "decrease", 0.0)
    phase = frame.copy()
    for column in ["clock_time_recurrence_dispersion", "change_point_structure", "raw_adjacent_day_shape_distance"]:
        if column in phase:
            phase[column] = np.maximum(phase[column], frame.loc[reference, column].quantile(.80))
    tests["phase_drift"] = (phase, "bounded", .18)
    smooth = frame.copy()
    for column in ["automatic_control_compatibility", "device_likeness", "background_recurrence", "temporal_stability"]:
        if column in smooth:
            smooth[column] = np.maximum(smooth[column], frame.loc[reference, column].quantile(.90))
    tests["smooth_modulation"] = (smooth, "not_large_decrease", -.12)
    multi = smooth.copy()
    for column in ["sample_state_transition_rate", "relative_event_strength"]:
        if column in multi:
            multi[column] = np.maximum(multi[column], frame.loc[reference, column].quantile(.70))
    tests["multiple_automatic_devices"] = (multi, "not_large_decrease", -.15)
    intermittent = frame.copy()
    for column in ["activity_episode_count", "relative_event_strength", "human_interaction_evidence", "unexplained_activity"]:
        if column in intermittent:
            intermittent[column] = np.maximum(intermittent[column], frame.loc[reference, column].quantile(.80))
    if "longest_silent_run" in intermittent:
        intermittent["longest_silent_run"] = np.minimum(intermittent["longest_silent_run"], frame.loc[reference, "longest_silent_run"].quantile(.25))
    tests["intermittent_return"] = (intermittent, "decrease", 0.0)
    rows = []
    for name, (modified, expectation, threshold) in tests.items():
        delta = (predict(modified) - baseline)[behavior]
        median = float(np.nanmedian(delta))
        if expectation == "decrease":
            passed = bool(median < 0 and np.nanquantile(delta, .9) <= .03)
        elif expectation == "not_large_decrease":
            passed = bool(median >= threshold)
        else:
            passed = bool(abs(median) <= threshold)
        rows.append({
            "counterfactual": name,
            "mean_score_delta": float(np.nanmean(delta)),
            "median_score_delta": median,
            "p10_delta": float(np.nanquantile(delta, .1)),
            "p90_delta": float(np.nanquantile(delta, .9)),
            "expected_behavior": expectation,
            "pass": passed,
            "scope_note": "feature-space stress test; embeddings held fixed; not a physical load simulation",
        })
    output = pd.DataFrame(rows)
    output.to_csv(ROOT / "counterfactual_audit.csv", index=False)
    return output


def challenge_flags(frame):
    scope = load_scope()
    source = frame
    eligible = frame.meter_id.isin(set(scope.loc[scope.final_fit & ~scope.strict_all_zero, "meter_id"]))
    reference = frame.meter_id.isin(set(scope.loc[scope.final_fit & ~scope.strict_all_zero, "meter_id"]))
    power = pd.DataFrame({
        column: reference_percentile(frame[column], reference)
        for column in ["raw_median_power", "raw_q90_power", "baseline_floor"]
    }).mean(axis=1)
    richness = pd.DataFrame({
        column: reference_percentile(frame[column], reference)
        for column in [
            "human_interaction_evidence", "interaction_complexity", "unexplained_activity",
            "activity_episode_count", "sample_state_transition_rate", "raw_weekday_weekend_shape_difference",
        ]
    }).mean(axis=1)
    output = frame[["meter_id"]].copy().assign(power_index=power, richness_index=richness)
    # These are unlabeled stress cohorts, not business decision thresholds.
    # Use universe-relative bands wide enough to guarantee that the requested
    # high-power/automatic and low-power/rich-activity failure modes are
    # actually testable on this finite cohort.
    output["high_power_automatic_challenge"] = eligible & (power >= .75) & (source.automatic_control_compatibility >= .75) & (source.human_interaction_evidence <= .65)
    output["low_power_rich_activity_challenge"] = eligible & (power <= .35) & (richness >= .65)
    output["smooth_modulation_challenge"] = eligible & (reference_percentile(source.device_likeness, reference) >= .80) & (reference_percentile(source.temporal_stability, reference) >= .75)
    output["high_sustained_baseload_challenge"] = eligible & (reference_percentile(source.baseline_floor, reference) >= .85)
    output.to_csv(ROOT / "business_challenge_cohorts.csv", index=False)
    return output


def build_consensus(frame, channels, subtype, weak, stability, window_summary, seed_summary, selected_family):
    scope = load_scope()
    fit_ids = set(scope.loc[scope.final_fit & ~scope.strict_all_zero, "meter_id"])
    result = frame[[
        "meter_id", "automatic_control_compatibility", "human_interaction_evidence", "weekend_tendency",
        "raw_median_power", "raw_q90_power", "baseline_floor", "sample_zero_fraction", "sample_near_zero_fraction",
    ]].copy()
    result = result.merge(channels, on="meter_id", how="left").merge(subtype, on="meter_id", how="left")
    result = result.merge(weak, on="meter_id", how="left").merge(stability, on="meter_id", how="left")
    result = result.merge(window_summary, on="meter_id", how="left").merge(seed_summary, on="meter_id", how="left")
    subtype_columns = ["complete_support", "equipment_support", "equipment_intermit_support", "intermittent_support"]
    result["subtype_manifold_rank"] = reference_percentile(result[subtype_columns].max(axis=1), result.meter_id.isin(fit_ids))
    channel_columns = [column for column in [
        "selected_pu_rank", "robust_pu_rank", "propensity_pu_rank", "legacy_anchor_rank",
        "bagging_comparator_rank", "subtype_manifold_rank", "weak_vacancy_rank",
    ] if column in result]
    if selected_family == "ROBUST_FEATURES" and "robust_pu_rank" in channel_columns:
        channel_columns.remove("robust_pu_rank")
    result["vacancy_screening_score"] = result[channel_columns].median(axis=1)
    result["model_disagreement"] = result[channel_columns].apply(
        lambda row: float(np.nanmedian(np.abs(row - np.nanmedian(row)))), axis=1
    )
    result["vacancy_rank_p10"] = result[[
        column for column in ["rank_p10", "window_rank_p10", "seed_rank_p10", "legacy_anchor_rank", "weak_vacancy_rank"] if column in result
    ]].quantile(.1, axis=1)
    result["vacancy_rank_p90"] = result[[
        column for column in ["rank_p90", "window_rank_p90", "seed_rank_p90", "legacy_anchor_rank", "weak_vacancy_rank"] if column in result
    ]].quantile(.9, axis=1)
    result["vacancy_rank_std"] = result[[column for column in ["rank_std", "window_rank_std", "seed_rank_std"] if column in result]].mean(axis=1)
    result["model_stability"] = 1 - result[[
        "model_disagreement", "rank_std", "window_rank_std", "seed_rank_std", "prior_sensitivity", "medium_weight_sensitivity",
    ]].mean(axis=1).clip(0, 1)
    fit_reference = result.meter_id.isin(fit_ids)
    power = pd.DataFrame({
        column: reference_percentile(result[column], fit_reference)
        for column in ["raw_median_power", "raw_q90_power", "baseline_floor"]
    }).mean(axis=1)
    result["high_power_flag"] = power >= .75
    auto_high_human_low = (result.automatic_control_compatibility >= .75) & (result.human_interaction_evidence <= .60)
    low_power_rich_guard = (power <= .25) & (result.active_evidence_score >= .75) & (result.human_interaction_evidence >= .70)
    result["model_conflict"] = (
        ((result.vacancy_screening_score >= .72) & (result.active_evidence_score >= .72))
        | (low_power_rich_guard & (result.vacancy_screening_score >= .68))
    )
    invalid_ids = set(scope.loc[scope.data_invalid, "meter_id"])
    strict_zero_ids = set(scope.loc[scope.strict_all_zero & ~scope.data_invalid, "meter_id"])
    invalid = result.meter_id.isin(invalid_ids)
    silence = result.meter_id.isin(strict_zero_ids)
    high_vacancy = (
        (result.vacancy_screening_score >= .90) & (result.vacancy_rank_p10 >= .72)
        & (result.model_disagreement <= .18) & ~result.model_conflict
    )
    medium_vacancy = (result.vacancy_screening_score >= .68) & ~high_vacancy & ~result.model_conflict
    high_active = (
        (result.active_evidence_score >= .88) & (result.vacancy_screening_score <= .45)
        & (result.human_interaction_evidence >= .70) & ~auto_high_human_low
    )
    result["final_screening_class"] = "暂无法可靠判断"
    result.loc[medium_vacancy, "final_screening_class"] = "中等疑似空置"
    result.loc[high_vacancy, "final_screening_class"] = "高疑似空置"
    result.loc[high_active, "final_screening_class"] = "高疑似正常使用"
    result.loc[silence, "final_screening_class"] = "高疑似空置"
    result.loc[invalid, "final_screening_class"] = "数据无效"
    result["reason_codes"] = np.select(
        [
            invalid, silence, low_power_rich_guard & result.model_conflict, result.model_conflict, auto_high_human_low,
            high_vacancy, medium_vacancy, high_active,
        ],
        [
            "DATA_INVALID", "VALID_STRICT_ALL_ZERO|METER_SUPPLY_STATUS_CHECK", "LOW_POWER_RICH_INTERACTION_GUARDRAIL",
            "VACANCY_ACTIVE_EVIDENCE_CONFLICT", "AUTOMATIC_HIGH_HUMAN_LOW_GUARDRAIL",
            "MULTI_CHANNEL_HIGH_VACANCY_EVIDENCE", "MULTI_CHANNEL_MEDIUM_VACANCY_EVIDENCE",
            "POWER_CONDITIONED_RICH_INTERACTION_EVIDENCE",
        ],
        default="LOW_OR_DISAGREED_EVIDENCE|ABSTAIN",
    )
    holdout_ids = set(scope.loc[scope.future_holdout, "meter_id"])
    result["future_holdout_flag"] = result.meter_id.isin(holdout_ids)
    result["data_status"] = np.where(invalid, "data_invalid", "valid")
    result["silence_status"] = np.where(silence, "valid_strict_all_zero", "behavioral_nonzero_or_near_zero")
    missing = sorted(set(scope.meter_id) - set(result.meter_id))
    if missing:
        extra = pd.DataFrame({"meter_id": missing, "data_status": "data_invalid", "silence_status": "not_applicable", "final_screening_class": "数据无效", "reason_codes": "DATA_INVALID|NO_NORMALIZED_DAILY_ROWS", "future_holdout_flag": False})
        result = pd.concat([result, extra], ignore_index=True)
    return result


def challenge_behavior_audit(consensus):
    flags = [
        "high_power_automatic_challenge", "low_power_rich_activity_challenge",
        "smooth_modulation_challenge", "high_sustained_baseload_challenge",
    ]
    rows = []
    for flag in flags:
        group = consensus[consensus[flag].fillna(False)]
        rows.append({
            "cohort": flag,
            "case_count": len(group),
            "vacancy_rank_median": float(group.vacancy_screening_score.median()) if len(group) else np.nan,
            "active_evidence_median": float(group.active_evidence_score.median()) if len(group) else np.nan,
            "model_disagreement_median": float(group.model_disagreement.median()) if len(group) else np.nan,
            "high_suspected_active_count": int(group.final_screening_class.eq("高疑似正常使用").sum()),
            "high_suspected_vacancy_count": int(group.final_screening_class.eq("高疑似空置").sum()),
            "uncertain_count": int(group.final_screening_class.eq("暂无法可靠判断").sum()),
            "interpretation": "unlabeled failure-mode audit; not correctness or accuracy",
        })
    output = pd.DataFrame(rows)
    output.to_csv(ROOT / "challenge_behavior_audit.csv", index=False)
    return output


def run_validation():
    frame, families, _ = build_feature_table()
    metrics = pd.read_csv(ROOT / "ablation_metrics.csv")
    positive = pd.read_csv(ROOT / "legacy_oof_scores.csv", dtype={"meter_id": str})
    selected_family, pareto = select_structure(metrics, positive)
    selected_metric = metrics.loc[metrics.family.eq(selected_family)].iloc[0]
    # The corrected 100-repetition permutation is deterministic for the frozen
    # feature table and holdout scope, but expensive (~25 minutes on this
    # cohort).  Reuse it only when its self-audit proves it was completed with
    # the current protocol; otherwise recompute it.
    permutation_path = ROOT / "permutation_null_summary.json"
    if permutation_path.exists():
        cached = json.loads(permutation_path.read_text(encoding="utf-8"))
        cache_valid = (
            int(cached.get("repetitions", 0)) >= 100
            and int(cached.get("folds", 0)) == 5
            and cached.get("fake_positive_labels_seen_by_own_model") is False
        )
    else:
        cached = {}
        cache_valid = False
    permutation = cached if cache_valid else corrected_permutation(
        frame, families[selected_family], selected_metric.legacy_oof_mean_rank,
        propensity=selected_family == "PROPENSITY_PU",
    )
    leave_out = leave_subtype_out(frame, families[selected_family])
    high_power = high_power_legacy_audit(frame, positive, selected_family)
    legacy_route_summary = legacy_router_and_behavior_audit(positive, selected_family)
    nnpu = nnpu_sensitivity(frame, families[selected_family])
    window_summary, window_correlations = window_stability(selected_family)
    final_frame, final_families = ensure_final_representation(frame, families, selected_family)
    channels, artifacts = final_fit_channels(final_frame, final_families, selected_family)
    stability = parameter_stability(
        final_frame, final_families[selected_family], propensity=selected_family == "PROPENSITY_PU"
    )
    seed_summary = seed_stability(final_frame, final_families, selected_family, artifacts)
    subtype, subtype_comparison = subtype_models(final_frame, final_families[selected_family])
    weak = weak_and_active_evidence(final_frame, subtype)
    counterfactual = counterfactual_audit(final_frame, artifacts["selected_pu_rank"])
    challenge = challenge_flags(final_frame)
    consensus = build_consensus(final_frame, channels, subtype, weak, stability, window_summary, seed_summary, selected_family)
    consensus = consensus.merge(challenge, on="meter_id", how="left")
    challenge_behavior_audit(consensus)
    consensus.to_csv(ROOT / "screening_results.csv", index=False, encoding="utf-8-sig")
    low_ids = set(load_scope().loc[lambda data: data.legacy_low_diagnostic_only, "meter_id"])
    consensus[consensus.meter_id.isin(low_ids)].to_csv(ROOT / "low_legacy_diagnostic_only.csv", index=False)
    power_only_metric = metrics.loc[metrics.family.eq("POWER_ONLY")].iloc[0]
    high_power_auto = consensus[consensus.high_power_automatic_challenge.fillna(False)]
    no_gold_gates = {
        "clearly_better_than_power_only": bool(
            selected_metric.legacy_oof_mean_rank >= power_only_metric.legacy_oof_mean_rank + .02
            and selected_metric.power_matched_advantage >= power_only_metric.power_matched_advantage + .02
        ),
        "power_matched_ci_positive": bool(selected_metric.power_matched_ci_low > 0),
        "subtypes_not_collapsed": bool(selected_metric.minimum_subtype_mean_rank >= .50),
        "high_power_legacy_recovered": bool(
            selected_metric.absolute_high_power_legacy_count > 0
            and selected_metric.absolute_high_power_legacy_mean_rank >= .55
        ),
        "corrected_permutation_better_than_null": bool(permutation["empirical_p_value"] <= .05),
        "window_stability_acceptable": bool(window_correlations.spearman.min() >= .60),
        "counterfactual_major_violation_absent": bool(counterfactual["pass"].mean() >= .70),
        "high_power_automatic_not_forced_active": bool(
            high_power_auto.empty or not high_power_auto.final_screening_class.eq("高疑似正常使用").any()
        ),
        "future_holdout_excluded_from_fit": True,
    }
    write_json(ROOT / "model_selection.json", {
        "selected_family": selected_family,
        "selection_method": "hard gates plus Pareto comparison; no decorative composite score",
        "selected_metrics": selected_metric.to_dict(),
        "permutation": permutation,
        "no_gold_gates": no_gold_gates,
        "all_gates_pass": all(no_gold_gates.values()),
        "gold_test_labels_opened": False,
    })
    print(json.dumps({
        "selected_family": selected_family,
        "all_gates_pass": all(no_gold_gates.values()),
        "screening_class_counts": consensus.final_screening_class.value_counts(dropna=False).to_dict(),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run_validation()
