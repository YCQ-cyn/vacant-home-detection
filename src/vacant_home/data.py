from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


from .runtime import workspace

ROOT = workspace('recognition')
INPUTS = ROOT / "inputs"
SEED = 20260926
TIME_COLS = [f"{h}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]


def key(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def bool_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_scope() -> pd.DataFrame:
    scope = pd.read_csv(INPUTS / "training_scope.csv", dtype={"meter_id": str})
    scope["meter_id"] = scope.meter_id.map(key)
    for column in [
        "data_invalid", "strict_all_zero", "model_a_fit",
        "shadow_fit_excluding_gold_test", "literal_transductive_full_fit",
        "legacy_high_hard", "legacy_medium_soft", "legacy_low_diagnostic_only",
    ]:
        scope[column] = bool_col(scope[column])
    scope["protected_cohort"] = scope.protected_cohort.fillna("").astype(str)
    return scope


def load_daily() -> pd.DataFrame:
    # Downstream NumPy arrays use row positions. The frozen pickle retains the
    # pre-cleaning row labels, so normalize them once at the loading boundary.
    daily = pd.read_pickle(INPUTS / "data_snapshot" / "daily_curves.pkl").copy().reset_index(drop=True)
    daily["meter_id"] = daily.meter_id.map(key)
    daily["date"] = pd.to_datetime(daily.date)
    return daily


def load_features() -> pd.DataFrame:
    core = pd.read_csv(INPUTS / "semantics_robust_core.csv", dtype={"meter_id": str})
    core["meter_id"] = core.meter_id.map(key)
    axes = pd.read_csv(INPUTS / "automatic_human_axes.csv", dtype={"meter_id": str})
    axes["meter_id"] = axes.meter_id.map(key)
    axes = axes.rename(columns={"automatic_control_explainability": "automatic_control_compatibility"})
    baseline = pd.read_csv(INPUTS / "scout_screening.csv", dtype={"meter_id": str})
    baseline["meter_id"] = baseline.meter_id.map(key)
    keep_axes = [c for c in ["meter_id", "automatic_control_compatibility", "human_interaction_evidence"] if c in axes]
    keep_base = [c for c in [
        "meter_id", "vacancy_rank_median", "active_evidence_score", "model_disagreement",
        "legacy_anchor_rank", "reason_codes",
    ] if c in baseline]
    return core.merge(axes[keep_axes], on="meter_id", how="left").merge(
        baseline[keep_base], on="meter_id", how="left"
    )


def build_point_semantics_inventory(raw: pd.DataFrame) -> pd.DataFrame:
    fields = ["电表厂家", "终端厂家", "电表版本", "终端规约", "数据项", "供电单位"]
    inventory = (
        raw.groupby(fields, dropna=False)
        .agg(
            户数=("电表局编号", "nunique"),
            终端资产号数量=("终端资产号", "nunique"),
            日记录数=("电表局编号", "size"),
        )
        .reset_index()
    )
    inventory["point_semantics"] = "unknown_periodic_frozen_active_power_sample"
    inventory["interval_average_confirmed"] = False
    inventory["evidence_note"] = "缺少具体型号、DI/OAD、负荷记录任务配置和时间戳定义"
    inventory.to_csv(ROOT / "point_semantics_inventory.csv", index=False, encoding="utf-8-sig")
    return inventory


def build_profile() -> dict:
    raw_path = INPUTS / "power_samples.xlsx"
    raw = pd.read_excel(raw_path, sheet_name="Sheet1")
    raw["电表局编号"] = raw["电表局编号"].map(key)
    raw["数据日期解析"] = pd.to_datetime(raw["数据日期"], unit="ms", errors="coerce")
    values = raw[TIME_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    inventory = build_point_semantics_inventory(raw)
    legacy = pd.read_excel(INPUTS / "legacy_labels.xlsx")
    legacy["meter_id"] = legacy.iloc[:, 0].map(key)
    duplicate_legacy = legacy[legacy.meter_id.duplicated(keep=False)].sort_values("meter_id")
    duplicate_legacy.to_csv(ROOT / "legacy_duplicate_audit.csv", index=False, encoding="utf-8-sig")
    profile = {
        "raw_rows": int(len(raw)),
        "raw_unique_meters": int(raw["电表局编号"].nunique()),
        "raw_unique_dates": int(raw["数据日期解析"].nunique()),
        "raw_date_min": str(raw["数据日期解析"].min().date()),
        "raw_date_max": str(raw["数据日期解析"].max().date()),
        "expected_1827x42_rows": 1827 * 42,
        "normalized_daily_rows": int(len(load_daily())),
        "normalized_daily_meters": int(load_daily().meter_id.nunique()),
        "finite_sample_fraction": float(np.isfinite(values).mean()),
        "zero_sample_fraction": float(np.nanmean(values == 0)),
        "negative_sample_fraction": float(np.nanmean(values < 0)),
        "raw_complete_96_point_rows": int(np.isfinite(values).all(axis=1).sum()),
        "legacy_source_rows": int(len(legacy)),
        "legacy_unique_meters": int(legacy.meter_id.nunique()),
        "legacy_duplicate_rows": int(legacy.meter_id.duplicated(keep=False).sum()),
        "legacy_subtypes": {str(k): int(v) for k, v in legacy.iloc[:, 1].value_counts().items()},
        "utility_meter_counts": {
            str(k): int(v) for k, v in raw.groupby("供电单位")["电表局编号"].nunique().sort_values(ascending=False).items()
        },
        "meter_version_counts": {
            str(k): int(v) for k, v in raw.groupby("电表版本", dropna=False)["电表局编号"].nunique().items()
        },
        "data_item_counts": {
            str(k): int(v) for k, v in raw.groupby("数据项", dropna=False)["电表局编号"].nunique().items()
        },
        "terminal_protocol_nonempty_meters": int(
            raw.loc[raw["终端规约"].notna(), "电表局编号"].nunique()
        ),
        "point_semantics_inventory_groups": int(len(inventory)),
        "raw_excel_sha256": sha256(raw_path),
        "legacy_excel_sha256": sha256(INPUTS / "legacy_labels.xlsx"),
    }
    write_json(ROOT / "full_cohort_data_profile.json", profile)
    per_meter = raw.groupby("电表局编号").agg(
        utility=("供电单位", "first"),
        terminal_asset_count=("终端资产号", "nunique"),
        day_rows=("数据日期解析", "nunique"),
        meter_vendor=("电表厂家", "first"),
        terminal_vendor=("终端厂家", "first"),
        meter_version=("电表版本", "first"),
        terminal_protocol=("终端规约", "first"),
        data_item=("数据项", "first"),
    ).reset_index().rename(columns={"电表局编号": "meter_id"})
    per_meter.to_csv(ROOT / "full_cohort_meter_inventory.csv", index=False, encoding="utf-8-sig")
    return profile


def _power_tier(series: pd.Series, eligible: pd.Series) -> pd.Series:
    values = np.log1p(series.clip(lower=0))
    quantiles = values[eligible & values.notna()].quantile([.25, .5, .75]).to_numpy()
    return pd.Series(np.digitize(values.fillna(values[eligible].median()), quantiles), index=series.index).map(
        {0: "P1", 1: "P2", 2: "P3", 3: "P4"}
    )


def _select_random_blind(candidates: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    frame = candidates.copy()
    frame["random_key"] = rng.random(len(frame))
    frame["selection_stratum"] = (
        frame.zero_band.astype(str) + "|" + frame.power_tier.astype(str) + "|" + frame.quality_band.astype(str)
    )
    target_zero = {"strict_zero": 2, "near_zero": 2, "active_nonzero": 6}
    selected = []
    utilities = set()
    for band, target in target_zero.items():
        pool = frame[frame.zero_band.eq(band)].sort_values("random_key")
        for _, row in pool.iterrows():
            if len([x for x in selected if x.zero_band == band]) >= target:
                break
            utility_bonus = row.utility not in utilities
            tier_bonus = row.power_tier not in {x.power_tier for x in selected}
            if utility_bonus or tier_bonus or len(pool) < target * 3:
                selected.append(row)
                utilities.add(row.utility)
    if len(selected) < n:
        used = {x.meter_id for x in selected}
        for _, row in frame[~frame.meter_id.isin(used)].sort_values("random_key").iterrows():
            selected.append(row)
            if len(selected) == n:
                break
    return pd.DataFrame(selected[:n])


def _target_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    active = ~out.zero_band.eq("strict_zero")
    q = lambda column, value: out.loc[active, column].quantile(value)
    power_index = pd.Series(np.nan, index=out.index, dtype=float)
    power_index.loc[active] = out.loc[active, [
        "raw_median_power", "raw_q90_power", "baseline_floor"
    ]].rank(pct=True).mean(axis=1)
    out["absolute_power_percentile"] = power_index
    out["target_nonzero_high_vacancy"] = active & (out.vacancy_rank_median >= q("vacancy_rank_median", .90))
    out["target_vacancy_active_conflict"] = active & (out.vacancy_rank_median >= .68) & (out.active_evidence_score >= .68)
    out["target_high_power_automatic"] = active & (power_index >= .90) & (
        out.automatic_control_compatibility >= q("automatic_control_compatibility", .75)
    ) & (out.human_interaction_evidence <= q("human_interaction_evidence", .60))
    richness = pd.Series(np.nan, index=out.index, dtype=float)
    richness.loc[active] = out.loc[active, [
        "interaction_complexity", "unexplained_activity", "activity_episode_count",
        "sample_state_transition_rate"
    ]].rank(pct=True).mean(axis=1)
    out["target_low_power_rich_activity"] = active & (power_index <= .25) & (richness >= .75)
    smooth = pd.Series(np.nan, index=out.index, dtype=float)
    smooth.loc[active] = out.loc[active, [
        "automatic_control_compatibility", "device_likeness", "background_recurrence",
        "daily_trajectory_similarity_sample"
    ]].rank(pct=True).mean(axis=1)
    out["target_smooth_modulation"] = active & (smooth >= .80) & (out.human_interaction_evidence.rank(pct=True) <= .65)
    out["target_high_disagreement"] = active & (out.model_disagreement >= q("model_disagreement", .90))
    out["target_legacy_manifold_far"] = active & (out.legacy_anchor_rank <= q("legacy_anchor_rank", .15))
    temporal = out[["change_point_structure", "weekday_weekend_difference", "lag7_similarity",
                    "clock_time_recurrence_dispersion"]].rank(pct=True).std(axis=1)
    out["target_distinct_temporal_structure"] = active & (temporal >= temporal[active].quantile(.80))
    flag_cols = [c for c in out if c.startswith("target_")]
    out["target_category_count"] = out[flag_cols].sum(axis=1)
    out["target_categories"] = out[flag_cols].apply(
        lambda row: "|".join(c.removeprefix("target_") for c, value in row.items() if bool(value)), axis=1
    )
    return out


def _select_targeted(candidates: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    frame = _target_flags(candidates)
    flag_cols = [
        c for c in frame
        if c.startswith("target_") and c not in {"target_category_count", "target_categories"}
        and pd.api.types.is_bool_dtype(frame[c])
    ]
    feature_cols = [
        "raw_median_power", "raw_q90_power", "baseline_floor", "sample_zero_fraction",
        "automatic_control_compatibility", "human_interaction_evidence", "interaction_complexity",
        "change_point_structure", "lag7_similarity", "vacancy_rank_median", "active_evidence_score",
        "model_disagreement", "legacy_anchor_rank",
    ]
    pool = frame[(frame.target_category_count > 0) & frame[feature_cols].notna().any(axis=1)].copy()
    values = pool[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(pool[feature_cols].median())
    z = RobustScaler().fit_transform(values)
    chosen: list[int] = []
    covered: dict[str, int] = {c: 0 for c in flag_cols}
    utility_count: dict[str, int] = {}
    rng = np.random.default_rng(SEED + 1)
    jitter = rng.random(len(pool)) * 1e-6
    # First guarantee that every business challenge with an available candidate is represented.
    for category in flag_cols:
        category_pool = pool[pool[category] & ~pool.index.isin(chosen)]
        if category_pool.empty:
            continue
        if not chosen:
            best_index = category_pool.target_category_count.idxmax()
        else:
            chosen_positions = [pool.index.get_loc(x) for x in chosen]
            candidate_positions = [pool.index.get_loc(x) for x in category_pool.index]
            distances = [float(np.min(np.linalg.norm(z[pos] - z[chosen_positions], axis=1))) for pos in candidate_positions]
            best_index = category_pool.index[int(np.argmax(distances))]
        chosen.append(best_index)
        selected = pool.loc[best_index]
        utility_count[str(selected.utility)] = utility_count.get(str(selected.utility), 0) + 1
        for column in flag_cols:
            covered[column] += int(bool(selected[column]))
    while len(chosen) < n:
        best_index = None
        best_value = -np.inf
        for pos, (idx, row) in enumerate(pool.iterrows()):
            if idx in chosen:
                continue
            novelty = 1.0 if not chosen else float(np.min(np.linalg.norm(z[pos] - z[[pool.index.get_loc(x) for x in chosen]], axis=1)))
            category_gain = sum(1 / (1 + covered[c]) for c in flag_cols if bool(row[c]))
            utility_gain = 1 / (1 + utility_count.get(str(row.utility), 0))
            tier_gain = 1 / (1 + sum(pool.loc[x, "power_tier"] == row.power_tier for x in chosen))
            value = 1.6 * category_gain + .7 * novelty + .4 * utility_gain + .3 * tier_gain + jitter[pos]
            if value > best_value:
                best_value, best_index = value, idx
        if best_index is None:
            break
        chosen.append(best_index)
        selected = pool.loc[best_index]
        utility_count[str(selected.utility)] = utility_count.get(str(selected.utility), 0) + 1
        for column in flag_cols:
            covered[column] += int(bool(selected[column]))
    return pool.loc[chosen].copy()


def build_holdout() -> pd.DataFrame:
    scope = load_scope()
    daily = load_daily()
    features = load_features()
    metadata = daily.groupby("meter_id").agg(
        utility=("utility", "first"),
        day_count=("date", "nunique"),
        unresolved_fraction=("unresolved_missing_points", lambda x: float(np.nansum(x) / (96 * max(1, len(x))))),
    ).reset_index()
    frame = scope.merge(features, on="meter_id", how="left").merge(metadata, on="meter_id", how="left")
    legacy = frame.legacy_high_hard | frame.legacy_medium_soft | frame.legacy_low_diagnostic_only
    candidate = (~frame.data_invalid) & (~legacy) & frame.protected_cohort.eq("") & frame.day_count.notna()
    frame["power_tier"] = _power_tier(frame.raw_median_power, candidate)
    frame["zero_band"] = np.select(
        [frame.strict_all_zero, frame.sample_near_zero_fraction >= .90],
        ["strict_zero", "near_zero"], default="active_nonzero",
    )
    frame["quality_band"] = np.where(frame.unresolved_fraction.fillna(1) <= .002, "complete", "quality_issue")
    candidates = frame[candidate].copy()
    random_blind = _select_random_blind(candidates, 10)
    remaining = candidates[~candidates.meter_id.isin(random_blind.meter_id)].copy()
    targeted = _select_targeted(remaining, 20)
    random_blind["holdout_group"] = "RANDOM_BLIND"
    random_blind["selection_basis"] = "fixed-seed stratified random before Recognition model fitting"
    random_blind["target_categories"] = "not_score_selected"
    targeted["holdout_group"] = "TARGETED_CHALLENGE"
    targeted["selection_basis"] = "Recognition frozen scout plus diversity selection before Recognition fitting"
    holdout = pd.concat([random_blind, targeted], ignore_index=True)
    holdout["future_holdout_flag"] = True
    holdout["selected_before_main_fit"] = True
    holdout["permitted_future_use"] = np.where(
        holdout.holdout_group.eq("RANDOM_BLIND"),
        "future protected human validation; more natural sample but too small for population accuracy",
        "future targeted failure discovery; never population-unbiased accuracy",
    )
    columns = [
        "meter_id", "holdout_group", "selection_basis", "selection_stratum", "target_categories",
        "utility", "power_tier", "zero_band", "quality_band", "day_count", "unresolved_fraction",
        "raw_median_power", "raw_q90_power", "baseline_floor", "sample_zero_fraction",
        "automatic_control_compatibility", "human_interaction_evidence", "vacancy_rank_median",
        "active_evidence_score", "model_disagreement", "legacy_anchor_rank", "future_holdout_flag",
        "selected_before_main_fit", "permitted_future_use",
    ]
    for column in columns:
        if column not in holdout:
            holdout[column] = np.nan
    holdout[columns].to_csv(ROOT / "future_human_holdout_manifest.csv", index=False, encoding="utf-8-sig")
    targeted_priority = holdout[holdout.holdout_group.eq("TARGETED_CHALLENGE")].copy()
    targeted_priority["review_priority"] = np.arange(1, len(targeted_priority) + 1)
    targeted_priority["recommended_first_20"] = targeted_priority.review_priority <= 10
    targeted_priority[columns + ["review_priority", "recommended_first_20"]].to_csv(
        ROOT / "targeted_human_review_priority.csv", index=False, encoding="utf-8-sig"
    )
    holdout_ids = set(holdout.meter_id)
    fit_scope = scope.copy()
    fit_scope["future_holdout"] = fit_scope.meter_id.isin(holdout_ids)
    fit_scope["research_fit"] = fit_scope.model_a_fit & ~fit_scope.future_holdout & ~fit_scope.data_invalid
    fit_scope["final_fit"] = fit_scope.literal_transductive_full_fit & ~fit_scope.future_holdout & ~fit_scope.data_invalid
    fit_scope["ssl_fit"] = fit_scope.final_fit
    fit_scope["inference_only"] = fit_scope.future_holdout | fit_scope.data_invalid
    fit_scope.to_csv(ROOT / "fit_scope.csv", index=False, encoding="utf-8-sig")
    assert len(holdout_ids) == 30
    assert not (fit_scope.future_holdout & (fit_scope.research_fit | fit_scope.final_fit | fit_scope.ssl_fit)).any()
    assert not holdout_ids & set(scope.loc[legacy, "meter_id"])
    write_json(ROOT / "future_human_holdout_freeze.json", {
        "seed": SEED,
        "count": 30,
        "random_blind_count": int((holdout.holdout_group == "RANDOM_BLIND").sum()),
        "targeted_challenge_count": int((holdout.holdout_group == "TARGETED_CHALLENGE").sum()),
        "meter_ids": sorted(holdout_ids),
        "manifest_sha256": sha256(ROOT / "future_human_holdout_manifest.csv"),
        "forbidden_fit_components": [
            "SSL pretraining", "scaler", "PCA", "feature selection", "regional factor",
            "prototype", "PU", "propensity", "threshold", "model selection", "hyperparameter tuning",
        ],
    })
    return holdout[columns]


def assert_holdout_isolation() -> None:
    scope = pd.read_csv(ROOT / "fit_scope.csv", dtype={"meter_id": str})
    for column in ["future_holdout", "research_fit", "final_fit", "ssl_fit", "data_invalid"]:
        scope[column] = bool_col(scope[column])
    overlap = scope.future_holdout & (scope.research_fit | scope.final_fit | scope.ssl_fit)
    if overlap.any():
        raise RuntimeError(f"Future holdout leaked into fit scope: {scope.loc[overlap, 'meter_id'].tolist()}")
    if (scope.data_invalid & (scope.research_fit | scope.final_fit | scope.ssl_fit)).any():
        raise RuntimeError("Data-invalid meter leaked into a fitting scope")


def main() -> None:
    profile = build_profile()
    holdout = build_holdout()
    assert_holdout_isolation()
    print(json.dumps({
        "profile": {k: profile[k] for k in ["raw_rows", "raw_unique_meters", "raw_unique_dates", "legacy_unique_meters"]},
        "holdout_counts": holdout.holdout_group.value_counts().to_dict(),
        "status": "HOLDOUT_FROZEN_BEFORE_ANY_MODEL_FIT",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
