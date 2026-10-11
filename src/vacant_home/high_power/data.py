from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import BranchPaths, DECISION, bool_col, key


SUBTYPE_MAP = {
    "完全空置": "complete",
    "设备维持型空置": "equipment",
    "设备维持间歇型": "equipment_intermittent",
    "设备维持-间歇型": "equipment_intermittent",
    "间歇使用型空置": "intermittent_use",
}
EXPECTED_SUBTYPE_COUNTS = {
    "intermittent_use": 62,
    "equipment": 29,
    "complete": 24,
    "equipment_intermittent": 21,
}
TIME_COLS = [f"{hour}:{minute:02d}" for hour in range(24) for minute in (0, 15, 30, 45)]


def empirical_cdf(values, reference) -> np.ndarray:
    values = np.asarray(values, float)
    reference = np.sort(np.asarray(reference, float))
    reference = reference[np.isfinite(reference)]
    result = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    if len(reference):
        result[finite] = np.searchsorted(reference, values[finite], side="right") / len(reference)
    return result


def load_unique_legacy(paths: BranchPaths | None = None):
    paths = paths or BranchPaths.discover()
    source = paths.source_legacy if paths.source_legacy.exists() else paths.upstream / "inputs" / "legacy_labels.xlsx"
    raw = pd.read_excel(source)
    legacy = raw.rename(columns={raw.columns[0]: "meter_id", raw.columns[1]: "raw_subtype"}).copy()
    legacy["meter_id"] = legacy.meter_id.map(key)
    duplicate = sorted(legacy.loc[legacy.meter_id.duplicated(keep=False), "meter_id"].unique())
    conflict = legacy.groupby("meter_id").raw_subtype.nunique().gt(1)
    if conflict.any():
        raise RuntimeError(f"Conflicting duplicate Legacy subtype: {conflict[conflict].index.tolist()}")
    legacy = legacy.drop_duplicates("meter_id", keep="first").copy()
    legacy["subtype"] = legacy.raw_subtype.map(SUBTYPE_MAP)
    if legacy.subtype.isna().any():
        unknown = legacy.loc[legacy.subtype.isna(), "raw_subtype"].unique().tolist()
        raise RuntimeError(f"Unknown Legacy subtype labels: {unknown}")
    counts = legacy.subtype.value_counts().to_dict()
    audit = {
        "source_rows": int(len(raw)),
        "unique_meter_ids": int(legacy.meter_id.nunique()),
        "duplicate_meter_ids": duplicate,
        "subtype_counts": counts,
    }
    if audit["source_rows"] != 137 or audit["unique_meter_ids"] != 136:
        raise RuntimeError(f"Unexpected Legacy identity counts: {audit}")
    if len(duplicate) != 1 or counts != EXPECTED_SUBTYPE_COUNTS:
        raise RuntimeError(f"Legacy subtype freeze mismatch: {audit}")
    return legacy[["meter_id", "raw_subtype", "subtype"]].reset_index(drop=True), audit


def load_scope(paths: BranchPaths | None = None) -> pd.DataFrame:
    paths = paths or BranchPaths.discover()
    scope = pd.read_csv(paths.upstream / "fit_scope.csv", dtype={"meter_id": str})
    scope["meter_id"] = scope.meter_id.map(key)
    for column in ["data_invalid", "strict_all_zero", "future_holdout", "final_fit", "legacy_high_hard", "legacy_medium_soft", "legacy_low_diagnostic_only"]:
        if column in scope:
            scope[column] = bool_col(scope[column])
    return scope


def load_holdout_ids(paths: BranchPaths | None = None) -> set[str]:
    paths = paths or BranchPaths.discover()
    frame = pd.read_csv(paths.upstream / "future_human_holdout_manifest.csv", dtype={"meter_id": str})
    result = set(frame.meter_id.map(key))
    if len(result) != 30:
        raise RuntimeError(f"Expected 30 future holdout meters, found {len(result)}")
    return result


def load_daily(paths: BranchPaths | None = None) -> pd.DataFrame:
    paths = paths or BranchPaths.discover()
    daily = pd.read_pickle(paths.upstream / "inputs" / "data_snapshot" / "daily_curves.pkl").copy().reset_index(drop=True)
    daily["meter_id"] = daily.meter_id.map(key)
    daily["date"] = pd.to_datetime(daily.date)
    return daily


def sample_power_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for meter_id, group in daily.sort_values(["meter_id", "date"]).groupby("meter_id", sort=False):
        values = group[TIME_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            rows.append({"meter_id": meter_id, "raw_curve_available": False})
            continue
        daily_median = np.nanmedian(values, axis=1)
        daily_q90 = np.nanquantile(values, .90, axis=1)
        daily_q10 = np.nanquantile(values, .10, axis=1)
        day_peak = np.nanmax(values, axis=1)
        peak_threshold = np.nanquantile(finite, .995)
        peak_count = int(np.sum(finite >= peak_threshold))
        rows.append({
            "meter_id": meter_id,
            "raw_curve_available": True,
            "raw_sample_count": int(values.size),
            "finite_sample_count": int(np.isfinite(values).sum()),
            "missing_sample_fraction": float(np.isnan(values).mean()),
            "negative_sample_fraction": float(np.nanmean(values < 0)),
            "raw_q10_power": float(np.nanquantile(finite, .10)),
            "raw_median_power": float(np.nanmedian(finite)),
            "raw_q90_power": float(np.nanquantile(finite, .90)),
            "baseline_floor": float(np.nanmedian(daily_q10)),
            "raw_max_power": float(np.nanmax(finite)),
            "frozen_sample_gt_0_5kw_fraction": float(np.mean(finite > .5)),
            "frozen_sample_gt_1kw_fraction": float(np.mean(finite > 1.0)),
            "frozen_sample_gt_2kw_fraction": float(np.mean(finite > 2.0)),
            "frozen_sample_gt_5kw_fraction": float(np.mean(finite > 5.0)),
            "daily_sample_median_gt_1kw_count": int(np.sum(daily_median > 1.0)),
            "daily_sample_q90_gt_2kw_count": int(np.sum(daily_q90 > 2.0)),
            "daily_curve_count": int(len(group)),
            "peak_sample_concentration": float(peak_count / max(1, len(finite))),
            "peak_day_concentration": float(np.sum(day_peak >= peak_threshold) / max(1, len(day_peak))),
            "utility": str(group.utility.mode().iloc[0]) if "utility" in group and not group.utility.mode().empty else "unknown",
            "unresolved_missing_points": int(pd.to_numeric(group.get("unresolved_missing_points", 0), errors="coerce").fillna(0).sum()) if "unresolved_missing_points" in group else 0,
            "original_missing_points": int(pd.to_numeric(group.get("original_missing_points", 0), errors="coerce").fillna(0).sum()) if "original_missing_points" in group else 0,
        })
    return pd.DataFrame(rows)


def persistent_high_load_flags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    route_one = (
        (result.raw_median_power >= DECISION["persistent_sample_median_kw"])
        & (result.frozen_sample_gt_1kw_fraction >= DECISION["persistent_gt_1kw_fraction"])
        & (result.daily_sample_median_gt_1kw_count >= DECISION["persistent_day_count"])
    )
    route_two = (
        (result.baseline_floor_percentile >= DECISION["strict_high_power_percentile"])
        & (result.gt_1kw_fraction_percentile >= DECISION["relative_high_power_percentile"])
        & (result.daily_sample_q90_gt_2kw_count >= DECISION["persistent_day_count"])
    )
    result["persistent_high_load_route"] = np.select([route_one, route_two], ["sample_median_and_fraction", "baseline_and_daily_q90"], default="none")
    result["persistent_high_load"] = route_one | route_two
    result["occasional_high_peak_only"] = (result.raw_q90_power >= 2.0) & ~result.persistent_high_load
    return result


def fit_reference_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.final_fit.fillna(False).astype(bool)
        & ~frame.future_holdout.fillna(False).astype(bool)
        & ~frame.data_invalid.fillna(False).astype(bool)
        & ~frame.strict_all_zero.fillna(False).astype(bool)
        & frame.raw_curve_available.fillna(False).astype(bool)
    )


def build_power_audit(paths: BranchPaths | None = None):
    paths = paths or BranchPaths.discover()
    upstream = pd.read_csv(paths.upstream / "screening_results.csv", dtype={"meter_id": str})
    upstream["meter_id"] = upstream.meter_id.map(key)
    scope = load_scope(paths)
    daily_metrics = sample_power_metrics(load_daily(paths))
    frame = upstream.merge(scope[["meter_id", "data_invalid", "strict_all_zero", "future_holdout", "final_fit"]], on="meter_id", how="left", suffixes=("", "_scope"))
    frame = frame.merge(daily_metrics, on="meter_id", how="left", suffixes=("_main", ""))
    frame["raw_curve_available"] = frame.raw_curve_available.eq(True)
    frame["data_status"] = frame.data_status.fillna("unknown")
    reference = fit_reference_mask(frame)
    power_views = ["raw_median_power", "raw_q90_power", "baseline_floor", "frozen_sample_gt_1kw_fraction"]
    for column in power_views:
        frame[f"{column}_percentile"] = empirical_cdf(frame[column], frame.loc[reference, column])
    frame["baseline_floor_percentile"] = frame["baseline_floor_percentile"]
    frame["gt_1kw_fraction_percentile"] = frame["frozen_sample_gt_1kw_fraction_percentile"]
    frame["main_power_index"] = pd.to_numeric(frame["power_index"], errors="coerce")
    frame["hp_power_index"] = frame[[f"{c}_percentile" for c in power_views]].median(axis=1, skipna=True)
    frame["power_index"] = frame.hp_power_index
    frame["relative_high_power_flag"] = frame.power_index >= DECISION["relative_high_power_percentile"]
    frame["strict_high_power_flag"] = frame.power_index >= DECISION["strict_high_power_percentile"]
    frame = persistent_high_load_flags(frame)
    frame["severe_quality_issue"] = (
        ~frame.raw_curve_available
        | (frame.missing_sample_fraction.fillna(1) > .10)
        | (frame.unresolved_missing_points.fillna(0) > 96)
        | frame.data_status.eq("data_invalid")
    )
    frame["high_power_definition"] = np.select(
        [frame.strict_high_power_flag & frame.persistent_high_load, frame.relative_high_power_flag & frame.persistent_high_load, frame.relative_high_power_flag],
        ["strict_and_persistent", "relative_and_persistent", "relative_only"], default="not_high_power",
    )
    if (reference & frame.future_holdout.fillna(False).astype(bool)).any():
        raise RuntimeError("Future holdout leaked into high-power reference")
    return frame, reference


def build_legacy_power_coverage(frame: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    merged = legacy.merge(frame, on="meter_id", how="left")
    columns = ["raw_median_power", "raw_q90_power", "baseline_floor", "raw_max_power", "frozen_sample_gt_1kw_fraction", "daily_sample_median_gt_1kw_count", "daily_sample_q90_gt_2kw_count"]
    rows = []
    for subtype, group in merged.groupby("subtype"):
        row = {"subtype": subtype, "unique_meter_count": int(group.meter_id.nunique())}
        for column in columns:
            row[f"{column}_median"] = float(group[column].median())
            row[f"{column}_max"] = float(group[column].max())
        row["relative_high_power_count"] = int(group.relative_high_power_flag.fillna(False).sum())
        row["strict_high_power_count"] = int(group.strict_high_power_flag.fillna(False).sum())
        row["persistent_high_load_count"] = int(group.persistent_high_load.fillna(False).sum())
        row["any_kw_level_sample_count"] = int((group.raw_max_power >= 1.0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def channel_power_bias(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "main_global_vacancy": "vacancy_screening_score",
        "selected_pu": "selected_pu_rank",
        "robust_pu": "robust_pu_rank",
        "propensity_pu": "propensity_pu_rank",
        "legacy_anchor": "legacy_anchor_rank",
        "weak_vacancy": "weak_vacancy_rank",
        "active_evidence": "active_evidence_score",
        "automatic_compatibility": "automatic_control_compatibility",
        "human_interaction": "human_interaction_evidence",
        "subtype_manifold": "subtype_manifold_rank",
    }
    eligible = frame.data_status.eq("valid") & ~frame.strict_all_zero.fillna(False)
    rows = []
    for name, column in candidates.items():
        if column not in frame:
            continue
        values = pd.to_numeric(frame.loc[eligible, column], errors="coerce")
        power = pd.to_numeric(frame.loc[eligible, "power_index"], errors="coerce")
        valid = values.notna() & power.notna()
        statistic = spearmanr(values[valid], power[valid]).statistic if valid.sum() > 2 else np.nan
        rows.append({
            "channel": name, "source_column": column, "eligible_count": int(valid.sum()),
            "spearman_vs_power_index": float(statistic),
            "power_index_definition": "HP branch median percentile of raw median, q90, baseline, and >1kW frozen-sample fraction",
        })
    return pd.DataFrame(rows)
